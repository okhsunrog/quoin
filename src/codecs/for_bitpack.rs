//! FOR_BITPACK: frame-of-reference + FastLanes bit-packing, **patched**.
//!
//! Splits the block into 1024-value sub-blocks (the FastLanes unit). Per
//! sub-block: subtract the local minimum (FoR) and bit-pack the residuals via
//! [`crate::bitpack`] — the `u32` kernel for widths up to 32 (always, on a
//! 32-bit lane), the `u64` kernel for 33..=64 (wide 64-bit integer columns).
//!
//! The packed width is chosen by **total cost**, not by the maximum residual:
//! a residual too wide for the chosen width becomes an **exception** — its low
//! `width` bits stay in the packed stream and `(position, high bits)` is
//! appended after it (PFOR / "patched" bit-packing). One outlier in a
//! sub-block of 10-bit values therefore costs a few bytes instead of widening
//! all 1024 values to its width. When the widest residual is the cheapest
//! choice the sub-block is stored exactly as before (no flag, no exceptions).
//!
//! This is the bread-and-butter integer-column codec, and the packing
//! substrate for delta, RLE lengths, dictionary codes, ALP-RD and FLOAT_MULT.
//!
//! Payload: `varint(n)` then per sub-block `min:lane ++ residuals`, where the
//! residuals are `width:u8 ++ packed` — or, with [`PATCHED`] set in the width
//! byte, `width|PATCHED:u8 ++ n_exc:u16 ++ packed ++ (pos:u16 ++ high:varint)*`.

use crate::bitpack::{self, BLOCK};
use crate::error::Error;
use crate::lane::Lane;
use crate::varint;

/// Lanes in the `u64` packing kernel (16 lanes × 64 bits); a width-`w` sub-block
/// packs into `16 * w` u64 words. Mirrors the `32 * w` u32 layout.
const LANES64: usize = 16;
/// Flag in the width byte: exceptions follow the packed stream.
const PATCHED: u8 = 0x80;

/// Choose the packed width for a sub-block from the histogram of residual bit
/// widths: minimise `packed bytes + exception bytes`, where a residual wider
/// than the chosen width costs a 2-byte position plus a LEB128 of its high
/// bits. Returns `(width, exception count)`; the widest residual (no
/// exceptions) is always a candidate, so this never loses to plain packing.
fn choose_width(hist: &[u32; 65], max_bw: u32) -> (u32, u32) {
    let mut best = (max_bw, 0u32);
    let mut best_cost = 128 * max_bw as usize;
    // Walking `w` downwards, every residual wider than `w` is an exception; its
    // LEB128 grows by one byte each time `bw - w` crosses a multiple of 7.
    let mut exc = 0u32;
    let mut exc_bytes = 0usize;
    for w in (0..max_bw).rev() {
        let c = hist[(w + 1) as usize];
        exc += c;
        exc_bytes += c as usize * 3; // 2-byte position + a 1-byte LEB128
        // Residuals whose high part just grew past 7·k bits need another byte.
        let mut bw = w + 8;
        while bw <= max_bw {
            exc_bytes += hist[bw as usize] as usize;
            bw += 7;
        }
        let cost = 128 * w as usize + exc_bytes;
        if cost < best_cost {
            best_cost = cost;
            best = (w, exc);
        }
    }
    best
}

/// Encode `vals` as FoR + bit-packing. When `signed`, the lane is interpreted as
/// a two's-complement integer so the frame-of-reference uses the signed minimum
/// (a mixed-sign column references its true minimum instead of treating
/// negatives as huge unsigned values). `delta_bitpack` passes `false` — its
/// deltas are already zigzagged.
pub(crate) fn encode<L: Lane>(vals: &[L], signed: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2 + 16);
    varint::write_u64(&mut out, vals.len() as u64);
    let mut i = 0;
    while i < vals.len() {
        let end = (i + BLOCK).min(vals.len());
        encode_subblock(&vals[i..end], signed, &mut out);
        i = end;
    }
    out
}

/// Subtract the (signed-aware) minimum; residuals are non-negative and `<= range`.
#[inline]
fn residual<L: Lane>(v: L, min: L, signed: bool) -> u64 {
    if signed {
        v.as_signed().wrapping_sub(min.as_signed()) as u64
    } else {
        v.wrapping_sub(min).to_u64()
    }
}

fn encode_subblock<L: Lane>(sub: &[L], signed: bool, out: &mut Vec<u8>) {
    let min = if signed {
        L::from_signed(sub.iter().map(|v| v.as_signed()).min().unwrap())
    } else {
        *sub.iter().min().unwrap()
    };
    let mut residuals = [0u64; BLOCK];
    for (r, &v) in residuals.iter_mut().zip(sub) {
        *r = residual(v, min, signed);
    }
    min.write_le(out);
    pack_residuals(&residuals, sub.len(), out);
}

/// Pack `count` non-negative residuals (the rest of the array must be zero)
/// with the cost-chosen width and out-of-line exceptions:
/// `width[|PATCHED]:u8 ++ [n_exc:u16] ++ packed ++ (pos:u16 ++ high:varint)*`.
/// Shared by the FoR codec and ALP's digit streams.
pub(crate) fn pack_residuals(residuals: &[u64; BLOCK], count: usize, out: &mut Vec<u8>) {
    // Histogram of residual bit widths for the width choice. Four interleaved
    // histograms: a single one serialises on the store-to-load dependency of
    // `hist[bw] += 1` when neighbouring values share a width (the common
    // case), which measured ~2× on the whole packer.
    let mut hists = [[0u32; 65]; 4];
    let mut acc = 0u64;
    for (k, &r) in residuals.iter().enumerate().take(count) {
        acc |= r;
        hists[k & 3][(64 - r.leading_zeros()) as usize] += 1;
    }
    let max_bw = 64 - acc.leading_zeros();
    let mut hist = [0u32; 65];
    for h in &hists {
        for (a, b) in hist.iter_mut().zip(h) {
            *a += b;
        }
    }
    let (width, n_exc) = choose_width(&hist, max_bw);

    out.push(width as u8 | if n_exc > 0 { PATCHED } else { 0 });
    if n_exc > 0 {
        out.extend_from_slice(&(n_exc as u16).to_le_bytes());
    }
    if width > 0 {
        // Pad to a full 1024 block (padding residuals = 0); the decoder only
        // takes the real count back. Exceptions keep their low bits in the
        // stream and carry the high bits out of line.
        if width <= 32 {
            let mask = if width == 32 { u32::MAX } else { (1u32 << width) - 1 };
            let mut lows = [0u32; BLOCK];
            for (l, &r) in lows.iter_mut().zip(residuals.iter()) {
                *l = r as u32 & mask;
            }
            let mut packed = vec![0u32; 32 * width as usize];
            bitpack::pack(&lows, width, &mut packed);
            for w in &packed {
                out.extend_from_slice(&w.to_le_bytes());
            }
        } else {
            let mask = if width == 64 { u64::MAX } else { (1u64 << width) - 1 };
            let mut lows = [0u64; BLOCK];
            for (l, &r) in lows.iter_mut().zip(residuals.iter()) {
                *l = r & mask;
            }
            let mut packed = vec![0u64; LANES64 * width as usize];
            bitpack::pack64(&lows, width, &mut packed);
            for w in &packed {
                out.extend_from_slice(&w.to_le_bytes());
            }
        }
    }
    if n_exc > 0 {
        for (k, &r) in residuals.iter().enumerate().take(count) {
            if 64 - r.leading_zeros() > width {
                out.extend_from_slice(&(k as u16).to_le_bytes());
                varint::write_u64(out, r >> width);
            }
        }
    }
}

/// Inverse of [`pack_residuals`]: read one packed sub-block at `*pos` into
/// `residuals[..count]` (the rest is left zero). `max_bits` bounds the width
/// (the lane width, or the digit width for ALP).
pub(crate) fn unpack_residuals(
    payload: &[u8],
    pos: &mut usize,
    count: usize,
    max_bits: u32,
    residuals: &mut [u64; BLOCK],
) -> Result<(), Error> {
    let wbyte = *payload.get(*pos).ok_or(Error::Truncated)?;
    *pos += 1;
    let patched = wbyte & PATCHED != 0;
    let width = u32::from(wbyte & !PATCHED);
    if width > max_bits {
        return Err(Error::CorruptPayload("for_bitpack bad width"));
    }
    let n_exc = if patched {
        let eb = payload.get(*pos..*pos + 2).ok_or(Error::Truncated)?;
        *pos += 2;
        let e = usize::from(u16::from_le_bytes(eb.try_into().unwrap()));
        if e == 0 || e > count {
            return Err(Error::CorruptPayload("for_bitpack exception count"));
        }
        e
    } else {
        0
    };
    if width == 0 {
        residuals[..count].fill(0);
    } else if width <= 32 {
        let nwords = 32 * width as usize;
        let pb = payload.get(*pos..*pos + nwords * 4).ok_or(Error::Truncated)?;
        *pos += nwords * 4;
        let mut packed = vec![0u32; nwords];
        for (k, c) in pb.chunks_exact(4).enumerate() {
            packed[k] = u32::from_le_bytes(c.try_into().unwrap());
        }
        let mut lows = [0u32; BLOCK];
        bitpack::unpack(&packed, width, &mut lows);
        for (r, &l) in residuals.iter_mut().zip(lows.iter()) {
            *r = u64::from(l);
        }
    } else {
        let nwords = LANES64 * width as usize;
        let pb = payload.get(*pos..*pos + nwords * 8).ok_or(Error::Truncated)?;
        *pos += nwords * 8;
        let mut packed = vec![0u64; nwords];
        for (k, c) in pb.chunks_exact(8).enumerate() {
            packed[k] = u64::from_le_bytes(c.try_into().unwrap());
        }
        bitpack::unpack64(&packed, width, residuals);
    }
    // Patch the exceptions: the stream holds their low `width` bits, the
    // high bits are here.
    for _ in 0..n_exc {
        let pb = payload.get(*pos..*pos + 2).ok_or(Error::Truncated)?;
        *pos += 2;
        let p = usize::from(u16::from_le_bytes(pb.try_into().unwrap()));
        if p >= count {
            return Err(Error::CorruptPayload("for_bitpack exception position"));
        }
        let high = varint::read_u64(payload, pos)?;
        if width == 64 || (high << width) >> width != high {
            return Err(Error::CorruptPayload("for_bitpack exception overflow"));
        }
        residuals[p] |= high << width;
    }
    Ok(())
}

/// Reconstruct a value from its FoR residual (inverse of [`residual`]).
#[inline]
fn unresidual<L: Lane>(r: u64, min: L, signed: bool) -> L {
    if signed {
        L::from_signed(min.as_signed().wrapping_add(r as i64))
    } else {
        min.wrapping_add(L::from_u64(r))
    }
}

pub(crate) fn decode<L: Lane>(
    payload: &[u8],
    n_values: usize,
    signed: bool,
) -> Result<Vec<L>, Error> {
    let mut pos = 0;
    let n = varint::read_u64(payload, &mut pos)? as usize;
    if n != n_values {
        return Err(Error::CorruptPayload("for_bitpack length mismatch"));
    }
    let mut out = Vec::with_capacity(n);
    let mut residuals = [0u64; BLOCK];
    while out.len() < n {
        let count = (n - out.len()).min(BLOCK);
        let mb = payload.get(pos..pos + L::BYTES).ok_or(Error::Truncated)?;
        let min = L::read_le(mb);
        pos += L::BYTES;
        unpack_residuals(payload, &mut pos, count, L::BITS, &mut residuals)?;
        for &r in &residuals[..count] {
            out.push(unresidual(r, min, signed));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(vals: &[u64]) -> usize {
        let enc = encode(vals, false);
        let dec = decode::<u64>(&enc, vals.len(), false).unwrap();
        assert_eq!(dec, vals);
        enc.len()
    }

    fn roundtrip_signed(vals: &[i64]) -> usize {
        let lane: Vec<u64> = vals.iter().map(|&v| v as u64).collect();
        let enc = encode(&lane, true);
        let dec = decode::<u64>(&enc, lane.len(), true).unwrap();
        let got: Vec<i64> = dec.iter().map(|&w| w as i64).collect();
        assert_eq!(got, vals);
        enc.len()
    }

    fn roundtrip32(vals: &[u32], signed: bool) -> usize {
        let enc = encode(vals, signed);
        assert_eq!(decode::<u32>(&enc, vals.len(), signed).unwrap(), vals);
        enc.len()
    }

    #[test]
    fn signed_mixed_sign_packs() {
        // Mixed small +/- values: unsigned FoR would see a ~2^64 range and bail
        // to 64-bit; signed FoR references the true minimum and packs tight.
        let vals: Vec<i64> = (0..4096).map(|i| (i % 200) as i64 - 100).collect();
        let size = roundtrip_signed(&vals);
        assert!(
            size < vals.len() * 2,
            "mixed-sign column should pack to <2 B/value, got {}",
            size as f64 / vals.len() as f64
        );
        // edges
        roundtrip_signed(&[]);
        roundtrip_signed(&[-1, 0, 1]);
        roundtrip_signed(&[i64::MIN, 0, i64::MAX]);
        roundtrip_signed(&(-5000..5000i64).collect::<Vec<_>>());
    }

    #[test]
    fn integer_column_shapes() {
        // Narrow-range integer column (the target use case): a base + small deltas.
        let narrow: Vec<u64> = (0..5000u64).map(|i| 1_000_000 + (i % 250)).collect();
        let size = roundtrip(&narrow);
        assert!(
            size < narrow.len() * 8 / 3,
            "narrow column should pack small"
        );

        roundtrip(&[]);
        roundtrip(&[42]);
        roundtrip(&vec![7u64; 3000]); // constant -> width 0
        roundtrip(&(0..3000u64).collect::<Vec<_>>()); // ramp
        // full 64-bit-range values pack at width ~64 (was the raw fallback).
        let mut s = 1u64;
        let wide: Vec<u64> = (0..3000)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                s
            })
            .collect();
        roundtrip(&wide);
    }

    #[test]
    fn wide_bounded_range_packs() {
        // A >32-bit but bounded range: base near 2^50, spread ~2^40 (width ~41).
        // Previously this fell back to raw u64 (8 B/value); now the u64 kernel
        // packs it to ~41 bits/value.
        let base = 1u64 << 50;
        let vals: Vec<u64> = (0..4096u64)
            .map(|i| base + (i.wrapping_mul(2_500_003) & ((1 << 40) - 1)))
            .collect();
        let size = roundtrip(&vals);
        assert!(
            size < vals.len() * 6,
            "wide-but-bounded column should pack under 6 B/value, got {} B/value",
            size as f64 / vals.len() as f64
        );
    }

    #[test]
    fn outliers_are_patched_not_widened() {
        // 10-bit residuals with one 40-bit outlier per sub-block: patched
        // packing keeps the stream at ~10 bits/value + a few exception bytes.
        let vals: Vec<u64> = (0..4096u64)
            .map(|i| if i % 1024 == 500 { 1 << 40 } else { i.wrapping_mul(7919) & 1023 })
            .collect();
        let enc = encode(&vals, false);
        assert_eq!(decode::<u64>(&enc, vals.len(), false).unwrap(), vals);
        assert!(
            enc.len() < 4 * (128 * 10 + 9 + 16) + 8,
            "outliers must not widen the sub-block: {}",
            enc.len()
        );
        // Signed lane with a negative outlier, 32-bit lane, and 64-bit-width
        // residuals (an exception's high bits above width 32).
        let signed: Vec<u64> = (0..2048i64)
            .map(|i| if i == 77 { -(1 << 50) } else { (i % 100) - 50 } as u64)
            .collect();
        let enc = encode(&signed, true);
        assert_eq!(decode::<u64>(&enc, signed.len(), true).unwrap(), signed);
        let v32: Vec<u32> = (0..3000u32).map(|i| if i % 700 == 3 { u32::MAX - i } else { i & 255 }).collect();
        let enc = encode(&v32, false);
        assert_eq!(decode::<u32>(&enc, v32.len(), false).unwrap(), v32);
        assert!(enc.len() < 3 * (128 * 8 + 9 + 40) + 8);
        let wide: Vec<u64> = (0..2000u64).map(|i| if i == 9 { u64::MAX } else { i << 30 }).collect();
        let enc = encode(&wide, false);
        assert_eq!(decode::<u64>(&enc, wide.len(), false).unwrap(), wide);
        // Truncating the exception list is an error, not a panic.
        assert!(decode::<u64>(&enc[..enc.len() - 3], wide.len(), false).is_err());
    }

    #[test]
    fn lane32_shapes() {
        // 32-bit lane: the min is 4 bytes and the width never exceeds 32.
        roundtrip32(&[], false);
        roundtrip32(&[42], false);
        roundtrip32(&[0, u32::MAX], false);
        roundtrip32(&vec![7u32; 3000], false);
        let mut s = 1u32;
        let noise: Vec<u32> = (0..3000)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                s
            })
            .collect();
        let size = roundtrip32(&noise, false);
        // Width 32 on every (1024-padded) sub-block: 4 KiB packed + 5 B header.
        let padded = noise.len().div_ceil(BLOCK) * (32 * 32 * 4 + 5) + 2;
        assert!(size <= padded, "u32 noise packs at ~4 B: {size} > {padded}");
        // Signed: mixed-sign i32 references the signed minimum.
        let signed: Vec<u32> = (0..4096i32).map(|i| ((i % 200) - 100) as u32).collect();
        let size = roundtrip32(&signed, true);
        assert!(size < signed.len() * 2, "mixed-sign i32 packs <2 B/value");
        roundtrip32(&[i32::MIN as u32, 0, i32::MAX as u32], true);
        roundtrip32(&[u32::MAX, 0, 1], true);
        // A 64-bit width tag is corrupt on the 32-bit lane.
        let mut enc = encode(&[1u32, 2, 3], false);
        enc[1 + 4] = 40; // width byte follows the 4-byte min
        assert!(decode::<u32>(&enc, 3, false).is_err());
    }
}
