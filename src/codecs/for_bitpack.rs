//! FOR_BITPACK: frame-of-reference + FastLanes bit-packing.
//!
//! Splits the block into 1024-value sub-blocks (the FastLanes unit). Per
//! sub-block: subtract the local minimum (FoR) and bit-pack the residuals at
//! `ceil(log2(max-min+1))` bits via [`crate::bitpack`] — the `u32` kernel for
//! widths up to 32, the `u64` kernel for 33..=64 (wide integer columns).
//!
//! This is the bread-and-butter integer-column codec. On `f64` *bit patterns*
//! it rarely wins (float bit patterns aren't frame-of-reference-friendly except
//! within one exponent band) — its real payoff is genuine integer columns, the
//! foundation for the typed-columnar work.

use crate::bitpack::{self, BLOCK};
use crate::error::Error;
use crate::varint;

/// Lanes in the `u64` packing kernel (16 lanes × 64 bits); a width-`w` sub-block
/// packs into `16 * w` u64 words. Mirrors the `32 * w` u32 layout.
const LANES64: usize = 16;

/// Encode `vals` as FoR + bit-packing. When `signed`, the lane is interpreted as
/// `i64` so the frame-of-reference uses the signed minimum (a mixed-sign column
/// references its true minimum instead of treating negatives as huge unsigned
/// values). `delta_bitpack` passes `false` — its deltas are already zigzagged.
pub(crate) fn encode(vals: &[u64], signed: bool) -> Vec<u8> {
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
fn residual(v: u64, min: u64, signed: bool) -> u64 {
    if signed {
        (v as i64).wrapping_sub(min as i64) as u64
    } else {
        v.wrapping_sub(min)
    }
}

fn encode_subblock(sub: &[u64], signed: bool, out: &mut Vec<u8>) {
    let (min, range) = if signed {
        let mut mn = i64::MAX;
        let mut mx = i64::MIN;
        for &v in sub {
            let s = v as i64;
            mn = mn.min(s);
            mx = mx.max(s);
        }
        (mn as u64, mx.wrapping_sub(mn) as u64)
    } else {
        let mn = *sub.iter().min().unwrap();
        let mx = *sub.iter().max().unwrap();
        (mn, mx - mn)
    };
    let width = if range == 0 {
        0
    } else {
        64 - range.leading_zeros()
    };

    out.push(width as u8);
    out.extend_from_slice(&min.to_le_bytes());
    if width == 0 {
        return;
    }
    // Pad to a full 1024 block (padding residuals = 0); the decoder only takes
    // the real count back.
    if width <= 32 {
        let mut residuals = [0u32; BLOCK];
        for (k, &v) in sub.iter().enumerate() {
            residuals[k] = residual(v, min, signed) as u32;
        }
        let mut packed = vec![0u32; 32 * width as usize];
        bitpack::pack(&residuals, width, &mut packed);
        for w in &packed {
            out.extend_from_slice(&w.to_le_bytes());
        }
    } else {
        let mut residuals = [0u64; BLOCK];
        for (k, &v) in sub.iter().enumerate() {
            residuals[k] = residual(v, min, signed);
        }
        let mut packed = vec![0u64; LANES64 * width as usize];
        bitpack::pack64(&residuals, width, &mut packed);
        for w in &packed {
            out.extend_from_slice(&w.to_le_bytes());
        }
    }
}

/// Reconstruct a value from its FoR residual (inverse of [`residual`]).
#[inline]
fn unresidual(r: u64, min: u64, signed: bool) -> u64 {
    if signed {
        (min as i64).wrapping_add(r as i64) as u64
    } else {
        min.wrapping_add(r)
    }
}

pub(crate) fn decode(payload: &[u8], n_values: usize, signed: bool) -> Result<Vec<u64>, Error> {
    let mut pos = 0;
    let n = varint::read_u64(payload, &mut pos)? as usize;
    if n != n_values {
        return Err(Error::CorruptPayload("for_bitpack length mismatch"));
    }
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let width = *payload.get(pos).ok_or(Error::Truncated)?;
        pos += 1;
        if width > 64 {
            return Err(Error::CorruptPayload("for_bitpack bad width"));
        }
        let count = (n - out.len()).min(BLOCK);

        let mb = payload.get(pos..pos + 8).ok_or(Error::Truncated)?;
        let min = u64::from_le_bytes(mb.try_into().unwrap());
        pos += 8;
        if width == 0 {
            out.extend(std::iter::repeat_n(min, count));
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
            for &r in &residuals[..count] {
                out.push(unresidual(u64::from(r), min, signed));
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
            for &r in &residuals[..count] {
                out.push(unresidual(r, min, signed));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(vals: &[u64]) -> usize {
        let enc = encode(vals, false);
        let dec = decode(&enc, vals.len(), false).unwrap();
        assert_eq!(dec, vals);
        enc.len()
    }

    fn roundtrip_signed(vals: &[i64]) -> usize {
        let lane: Vec<u64> = vals.iter().map(|&v| v as u64).collect();
        let enc = encode(&lane, true);
        let dec = decode(&enc, lane.len(), true).unwrap();
        let got: Vec<i64> = dec.iter().map(|&w| w as i64).collect();
        assert_eq!(got, vals);
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
}
