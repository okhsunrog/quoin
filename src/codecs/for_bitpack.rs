//! FOR_BITPACK: frame-of-reference + FastLanes bit-packing.
//!
//! Splits the block into 1024-value sub-blocks (the FastLanes unit). Per
//! sub-block: subtract the local minimum (FoR) and bit-pack the residuals at
//! `ceil(log2(max-min+1))` bits via [`crate::bitpack`]. Sub-blocks whose range
//! needs more than 32 bits fall back to raw u64.
//!
//! This is the bread-and-butter integer-column codec. On `f64` *bit patterns*
//! it rarely wins (float bit patterns aren't frame-of-reference-friendly except
//! within one exponent band) — its real payoff is genuine integer columns, the
//! foundation for the typed-columnar work.

use crate::bitpack::{self, BLOCK};
use crate::error::Error;
use crate::varint;

/// Sentinel width: the sub-block is stored raw (range needed > 32 bits).
const RAW_WIDTH: u8 = 64;

pub(crate) fn encode(vals: &[u64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2 + 16);
    varint::write_u64(&mut out, vals.len() as u64);
    let mut i = 0;
    while i < vals.len() {
        let end = (i + BLOCK).min(vals.len());
        encode_subblock(&vals[i..end], &mut out);
        i = end;
    }
    out
}

fn encode_subblock(sub: &[u64], out: &mut Vec<u8>) {
    let min = *sub.iter().min().unwrap();
    let max = *sub.iter().max().unwrap();
    let range = max - min;
    let width = if range == 0 { 0 } else { 64 - range.leading_zeros() };

    if width <= 32 {
        out.push(width as u8);
        out.extend_from_slice(&min.to_le_bytes());
        if width > 0 {
            // Pad to a full 1024 block (padding residuals = 0); the decoder
            // only takes the real count back.
            let mut residuals = [0u32; BLOCK];
            for (k, &v) in sub.iter().enumerate() {
                residuals[k] = (v - min) as u32;
            }
            let mut packed = vec![0u32; 32 * width as usize];
            bitpack::pack(&residuals, width, &mut packed);
            for w in &packed {
                out.extend_from_slice(&w.to_le_bytes());
            }
        }
    } else {
        out.push(RAW_WIDTH);
        for &v in sub {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
}

pub(crate) fn decode(payload: &[u8], n_values: usize) -> Result<Vec<u64>, Error> {
    let mut pos = 0;
    let n = varint::read_u64(payload, &mut pos)? as usize;
    if n != n_values {
        return Err(Error::CorruptPayload("for_bitpack length mismatch"));
    }
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let width = *payload.get(pos).ok_or(Error::Truncated)?;
        pos += 1;
        let count = (n - out.len()).min(BLOCK);

        if width == RAW_WIDTH {
            let bytes = payload.get(pos..pos + count * 8).ok_or(Error::Truncated)?;
            for c in bytes.chunks_exact(8) {
                out.push(u64::from_le_bytes(c.try_into().unwrap()));
            }
            pos += count * 8;
        } else if width > 32 {
            return Err(Error::CorruptPayload("for_bitpack bad width"));
        } else {
            let mb = payload.get(pos..pos + 8).ok_or(Error::Truncated)?;
            let min = u64::from_le_bytes(mb.try_into().unwrap());
            pos += 8;
            if width == 0 {
                out.extend(std::iter::repeat_n(min, count));
            } else {
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
                    out.push(min + u64::from(r));
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(vals: &[u64]) -> usize {
        let enc = encode(vals);
        let dec = decode(&enc, vals.len()).unwrap();
        assert_eq!(dec, vals);
        enc.len()
    }

    #[test]
    fn integer_column_shapes() {
        // Narrow-range integer column (the target use case): a base + small deltas.
        let narrow: Vec<u64> = (0..5000u64).map(|i| 1_000_000 + (i % 250)).collect();
        let size = roundtrip(&narrow);
        assert!(size < narrow.len() * 8 / 3, "narrow column should pack small");

        roundtrip(&[]);
        roundtrip(&[42]);
        roundtrip(&vec![7u64; 3000]); // constant -> width 0
        roundtrip(&(0..3000u64).collect::<Vec<_>>()); // ramp
        // wide-range values force the raw fallback for some sub-blocks.
        let mut s = 1u64;
        let wide: Vec<u64> = (0..3000)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                s
            })
            .collect();
        roundtrip(&wide);
    }
}
