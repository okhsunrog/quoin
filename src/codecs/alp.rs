//! ALP — Adaptive Lossless floating-Point (the main scheme), ported from the
//! CWI reference (SIGMOD'24). For decimal-like floats it represents each value
//! as a scaled integer `digit = round(v · 10^e · 10^-f)` and stores the digits
//! via frame-of-reference + FastLanes bit-packing. Values that don't round-trip
//! (and NaN/inf) are stored verbatim as **exceptions**, so a few outliers don't
//! force the whole block to raw — the key robustness win over [`super::float_mult`].
//!
//! Decoding is exact: encode verifies `decode(encode(v)) == v` bit-for-bit per
//! value (else it's an exception), and the decoder recomputes the same product.
//! The arithmetic runs in the lane's float (`f32` on a 32-bit lane, with the
//! reference's float constants: exponents to 10, magic `1.5·2^23`), so an `f32`
//! column is encoded as the `f32` decimal it is — not as a widened double.
//!
//! The `(e, f)` search is two-stage, as in the reference: every pair is scored
//! once on a sample spread over the whole block ([`stage1`]), and each
//! 1024-value sub-block then picks among the top [`TOP_K`] of those on its own
//! 32-value sample — falling back to the full search only when none of them
//! fits. A sub-block that isn't decimal at all is stored verbatim
//! ([`RAW_SUB`]) instead of failing the whole block, and digit ranges wider than
//! 32 bits use the `u64` packing kernel on the 64-bit lane.
//!
//! Non-decimal reals are handled by [`super::alp_rd`], the ALP-RD
//! split-dictionary scheme.
//!
//! Payload: `varint(n)` then per 1024-value sub-block either
//! `e:u8 ++ f:u8 ++ n_exc:u16 ++ min:lane ++ width:u8 ++ packed ++ (pos:u16 ++ bits:lane)*`
//! or `RAW_SUB:u8 ++ count lane words`.

use crate::bitpack::{self, BLOCK};
use crate::error::Error;
use crate::lane::{Lane, LaneFloat};
use crate::varint;

/// `e` byte marking a sub-block stored as verbatim lane words.
const RAW_SUB: u8 = 0xFF;
/// `(e, f)` candidates carried from the block-level sample to each sub-block.
const TOP_K: usize = 5;
/// Values sampled across the block for the stage-1 search.
const STAGE1_SAMPLES: usize = 1024;
/// Lanes in the `u64` packing kernel (mirrors `for_bitpack`).
const LANES64: usize = 16;

#[inline]
fn encode_value<F: LaneFloat>(v: F, e: usize, f: usize) -> Option<i64> {
    let tmp = v * F::exp10(e) * F::frac10(f);
    if !tmp.is_finite() || tmp >= F::ALP_UPPER || tmp <= -F::ALP_UPPER {
        return None;
    }
    Some((tmp + F::ALP_MAGIC - F::ALP_MAGIC).to_i64())
}

#[inline]
fn decode_value<F: LaneFloat>(digit: i64, e: usize, f: usize) -> F {
    F::from_i64(digit) * F::exp10(f) * F::frac10(e)
}

/// Estimated cost (bits) of coding `sample` with `(e, f)`: packed digit width
/// times the count plus the exception bytes; also returns the exception count.
fn cost<L: Lane>(sample: &[L::Float], e: usize, f: usize) -> (usize, usize) {
    let mut exc = 0usize;
    let (mut lo, mut hi) = (i64::MAX, i64::MIN);
    for &v in sample {
        match encode_value(v, e, f) {
            Some(d) if decode_value::<L::Float>(d, e, f).to_bits() == v.to_bits() => {
                lo = lo.min(d);
                hi = hi.max(d);
            }
            _ => exc += 1,
        }
    }
    let width = if exc == sample.len() {
        L::BITS as usize
    } else if hi <= lo {
        0
    } else {
        (64 - (hi.wrapping_sub(lo) as u64).leading_zeros()) as usize
    };
    (width * sample.len() + exc * (16 + L::BITS as usize), exc)
}

/// Every `(e, f)` pair, `0 <= f <= e <= ALP_MAX_EXP`.
fn all_pairs<F: LaneFloat>() -> impl Iterator<Item = (usize, usize)> {
    (0..=F::ALP_MAX_EXP).flat_map(|e| (0..=e).map(move |f| (e, f)))
}

/// Stride-sample `count` values of `vals` as floats.
fn sample_floats<L: Lane>(vals: &[L], count: usize) -> Vec<L::Float> {
    let n = vals.len();
    let stride = (n / count).max(1);
    (0..n)
        .step_by(stride)
        .take(count)
        .map(|i| vals[i].to_float())
        .collect()
}

/// Stage 1: score every pair on a block-wide sample; return the best
/// [`TOP_K`] pairs (best first), or `None` when even the best leaves more than
/// half the sample as exceptions — the block isn't decimal, don't bother.
fn stage1<L: Lane>(vals: &[L]) -> Option<Vec<(usize, usize)>> {
    let sample = sample_floats(vals, STAGE1_SAMPLES);
    let mut scored: Vec<(usize, usize, (usize, usize))> = all_pairs::<L::Float>()
        .map(|(e, f)| {
            let (c, exc) = cost::<L>(&sample, e, f);
            (c, exc, (e, f))
        })
        .collect();
    scored.sort_unstable_by_key(|&(c, _, ef)| (c, ef));
    if scored[0].1 * 2 > sample.len() {
        return None;
    }
    Some(scored.iter().take(TOP_K).map(|&(_, _, ef)| ef).collect())
}

/// Pick the best of `cands` on a 32-value sample of the sub-block, or `None`
/// when the best still leaves more than half the sample as exceptions.
fn pick_ef<L: Lane>(
    sub: &[L],
    cands: impl Iterator<Item = (usize, usize)>,
) -> Option<(usize, usize)> {
    let sample = sample_floats(sub, 32);
    let mut best = None;
    let mut best_cost = usize::MAX;
    let mut best_exc = sample.len();
    for (e, f) in cands {
        let (c, exc) = cost::<L>(&sample, e, f);
        if c < best_cost {
            best_cost = c;
            best = Some((e, f));
            best_exc = exc;
        }
    }
    if best_exc * 2 > sample.len() {
        return None;
    }
    best
}

pub(crate) fn encode<L: Lane>(vals: &[L]) -> Option<Vec<u8>> {
    if vals.is_empty() {
        return None;
    }
    let cands = stage1(vals)?;
    let mut out = Vec::with_capacity(vals.len());
    varint::write_u64(&mut out, vals.len() as u64);
    let mut i = 0;
    while i < vals.len() {
        let end = (i + BLOCK).min(vals.len());
        let sub = &vals[i..end];
        // Stage 2: the block's top candidates first; the full search only when
        // none of them fits this sub-block; verbatim when nothing does.
        let ef = pick_ef(sub, cands.iter().copied())
            .or_else(|| pick_ef(sub, all_pairs::<L::Float>()));
        match ef {
            Some((e, f)) => encode_subblock(sub, e, f, &mut out),
            None => {
                out.push(RAW_SUB);
                out.extend_from_slice(&L::le_bytes(sub));
            }
        }
        i = end;
    }
    Some(out)
}

fn encode_subblock<L: Lane>(sub: &[L], e: usize, f: usize, out: &mut Vec<u8>) {
    let mut digits = [0i64; BLOCK];
    let mut exceptions: Vec<(u16, L)> = Vec::new();
    let mut first_valid: Option<i64> = None;
    for (k, &bits) in sub.iter().enumerate() {
        let v = bits.to_float();
        match encode_value(v, e, f) {
            Some(d) if decode_value::<L::Float>(d, e, f).to_bits() == bits => {
                digits[k] = d;
                first_valid.get_or_insert(d);
            }
            _ => exceptions.push((k as u16, bits)),
        }
    }
    // Exceptions (and padding) get a filler digit so they don't widen the range.
    let filler = first_valid.unwrap_or(0);
    for &(pos, _) in &exceptions {
        digits[pos as usize] = filler;
    }
    for d in digits.iter_mut().take(BLOCK).skip(sub.len()) {
        *d = filler;
    }

    let min = *digits.iter().min().unwrap();
    let max = *digits.iter().max().unwrap();
    let range = max.wrapping_sub(min) as u64;
    let width = if range == 0 {
        0
    } else {
        64 - range.leading_zeros()
    };

    out.push(e as u8);
    out.push(f as u8);
    out.extend_from_slice(&(exceptions.len() as u16).to_le_bytes());
    // The digit range is bounded by `ALP_UPPER`, which fits the lane as a
    // signed value (i32 on the 32-bit lane).
    L::from_signed(min).write_le(out);
    out.push(width as u8);
    if width == 0 {
        // constant digits
    } else if width <= 32 {
        let mut residuals = [0u32; BLOCK];
        for (r, &d) in residuals.iter_mut().zip(digits.iter()) {
            *r = d.wrapping_sub(min) as u32;
        }
        let mut packed = vec![0u32; 32 * width as usize];
        bitpack::pack(&residuals, width, &mut packed);
        for w in &packed {
            out.extend_from_slice(&w.to_le_bytes());
        }
    } else {
        // Wide digit range (64-bit lane only: the 32-bit lane's digits are
        // bounded by 2^22): the u64 FastLanes kernel.
        let mut residuals = [0u64; BLOCK];
        for (r, &d) in residuals.iter_mut().zip(digits.iter()) {
            *r = d.wrapping_sub(min) as u64;
        }
        let mut packed = vec![0u64; LANES64 * width as usize];
        bitpack::pack64(&residuals, width, &mut packed);
        for w in &packed {
            out.extend_from_slice(&w.to_le_bytes());
        }
    }
    for &(pos, bits) in &exceptions {
        out.extend_from_slice(&pos.to_le_bytes());
        bits.write_le(out);
    }
}

pub(crate) fn decode<L: Lane>(payload: &[u8], n_values: usize) -> Result<Vec<L>, Error> {
    let mut pos = 0usize;
    let n = varint::read_u64(payload, &mut pos)? as usize;
    if n != n_values {
        return Err(Error::CorruptPayload("alp length mismatch"));
    }
    let head_len = 5 + L::BYTES;
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let count = (n - out.len()).min(BLOCK);
        if payload.get(pos) == Some(&RAW_SUB) {
            pos += 1;
            let raw = payload
                .get(pos..pos + count * L::BYTES)
                .ok_or(Error::Truncated)?;
            pos += count * L::BYTES;
            out.extend(L::from_le_bytes(raw));
            continue;
        }
        let head = payload.get(pos..pos + head_len).ok_or(Error::Truncated)?;
        let e = head[0] as usize;
        let f = head[1] as usize;
        if e > L::Float::ALP_MAX_EXP || f > e {
            return Err(Error::CorruptPayload("alp exponent/factor"));
        }
        let exc_count = u16::from_le_bytes(head[2..4].try_into().unwrap()) as usize;
        let min = L::read_le(&head[4..4 + L::BYTES]).as_signed();
        let width = head[4 + L::BYTES];
        pos += head_len;
        if u32::from(width) > L::BITS {
            return Err(Error::CorruptPayload("alp width"));
        }

        let mut digits = [0i64; BLOCK];
        if width == 0 {
            digits.fill(min);
        } else if width <= 32 {
            let nwords = 32 * width as usize;
            let pb = payload.get(pos..pos + nwords * 4).ok_or(Error::Truncated)?;
            pos += nwords * 4;
            let mut packed = vec![0u32; nwords];
            for (k, c) in pb.chunks_exact(4).enumerate() {
                packed[k] = u32::from_le_bytes(c.try_into().unwrap());
            }
            let mut residuals = [0u32; BLOCK];
            bitpack::unpack(&packed, u32::from(width), &mut residuals);
            for (d, &r) in digits.iter_mut().zip(residuals.iter()) {
                *d = min.wrapping_add(i64::from(r));
            }
        } else {
            let nwords = LANES64 * width as usize;
            let pb = payload.get(pos..pos + nwords * 8).ok_or(Error::Truncated)?;
            pos += nwords * 8;
            let mut packed = vec![0u64; nwords];
            for (k, c) in pb.chunks_exact(8).enumerate() {
                packed[k] = u64::from_le_bytes(c.try_into().unwrap());
            }
            let mut residuals = [0u64; BLOCK];
            bitpack::unpack64(&packed, u32::from(width), &mut residuals);
            for (d, &r) in digits.iter_mut().zip(residuals.iter()) {
                *d = min.wrapping_add(r as i64);
            }
        }

        let start = out.len();
        for &d in digits.iter().take(count) {
            out.push(decode_value::<L::Float>(d, e, f).to_bits());
        }
        // Patch exceptions over the decoded digits.
        let exc_len = 2 + L::BYTES;
        for _ in 0..exc_count {
            let ex = payload.get(pos..pos + exc_len).ok_or(Error::Truncated)?;
            let p = u16::from_le_bytes(ex[..2].try_into().unwrap()) as usize;
            let bits = L::read_le(&ex[2..]);
            pos += exc_len;
            if p >= count {
                return Err(Error::CorruptPayload("alp exception position"));
            }
            out[start + p] = bits;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip<L: Lane>(vals: &[L]) -> Option<usize> {
        let enc = encode(vals)?;
        let dec = decode::<L>(&enc, vals.len()).unwrap();
        assert_eq!(dec, vals);
        Some(enc.len())
    }

    #[test]
    fn decimal_with_exceptions() {
        // cent-rounded prices with a few NaN/huge outliers → exceptions.
        let mut v: Vec<u64> = (0..5000)
            .map(|i| (100.0_f64 + (i % 700) as f64 / 100.0).to_bits())
            .collect();
        v[10] = f64::NAN.to_bits();
        v[2000] = 1e300_f64.to_bits();
        v[4999] = f64::INFINITY.to_bits();
        roundtrip(&v);

        roundtrip(
            &(0..3000)
                .map(|i| ((i % 100) as f64 * 0.25).to_bits())
                .collect::<Vec<_>>(),
        );
        roundtrip::<u64>(&[]);
        roundtrip(&[1.25_f64.to_bits()]);
        roundtrip(&vec![0.0f64.to_bits(); 2000]);
    }

    #[test]
    fn f32_decimals_encode_natively() {
        // f32 prices in cents: decimal in f32 arithmetic, 4-byte exceptions.
        let mut v: Vec<u32> = (0..5000)
            .map(|i| (100.0_f32 + (i % 700) as f32 / 100.0).to_bits())
            .collect();
        v[10] = f32::NAN.to_bits();
        v[11] = 0x7F80_0001; // signaling NaN: exception, exact
        v[12] = (-0.0f32).to_bits(); // ±0 differ in bits → exception, exact
        v[2000] = 3e38f32.to_bits();
        v[4999] = f32::INFINITY.to_bits();
        let size = roundtrip(&v).expect("f32 decimals should ALP-encode");
        assert!(size < v.len() * 2, "cent prices pack under 2 B/value: {size}");

        // Stylus-like coordinates with one fractional decimal digit.
        let xy: Vec<u32> = (0..4096)
            .map(|i| ((i as f32) * 0.3 + 1234.5).to_bits())
            .collect();
        let size = roundtrip(&xy).expect("one-decimal f32 should ALP-encode");
        assert!(size < xy.len() * 4);
        roundtrip::<u32>(&[]);
        roundtrip(&[1.25_f32.to_bits()]);
        roundtrip(&vec![0.0f32.to_bits(); 2000]);
    }

    #[test]
    fn non_decimal_bails() {
        let mut s = 1u64;
        let noise: Vec<u32> = (0..2048)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                (f32::from_bits((s >> 40) as u32 & 0x3FFF_FFFF)).to_bits()
            })
            .collect();
        assert!(encode(&noise).is_none());
    }

    #[test]
    fn wide_digits_use_the_u64_kernel() {
        // Cents spread over ~2^40: the digit range needs >32 bits. Previously
        // the whole block bailed; now the u64 kernel packs it at ~41 bits.
        let v: Vec<u64> = (0..3000u64)
            .map(|i| ((i.wrapping_mul(2_654_435_761) & ((1 << 40) - 1)) as f64 * 0.01).to_bits())
            .collect();
        let size = roundtrip(&v).expect("wide decimals should ALP-encode");
        assert!(size < v.len() * 6, "wide digits pack under 6 B/value: {size}");
    }

    #[test]
    fn mixed_block_falls_back_per_subblock() {
        // Decimal data with one non-decimal (random real) sub-block in the
        // middle: that sub-block is stored verbatim, the rest is ALP'd.
        let mut s = 7u64;
        let mut v: Vec<u64> = (0..6000)
            .map(|i| (1000.0 + (i % 900) as f64 * 0.01).to_bits())
            .collect();
        for x in v[2048..3072].iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            *x = f64::from_bits(0x3FF0_0000_0000_0000 | (s >> 12)).to_bits();
        }
        let size = roundtrip(&v).expect("mixed block should still ALP-encode");
        // 5 decimal sub-blocks at ~2 B + 1 raw sub-block at 8 B.
        assert!(size < 5 * 1024 * 3 + 1024 * 8 + 64, "mixed block size {size}");
        // Sub-block-level corruption is caught.
        let enc = encode(&v).unwrap();
        assert!(decode::<u64>(&enc[..enc.len() - 100], v.len()).is_err());
    }
}
