//! DELTA_BITPACK: first-order delta of the bit patterns (zigzag) feeding the
//! FoR + FastLanes bit-pack codec. This is Parquet's `DELTA_BINARY_PACKED`, and
//! our first **cascade** — a delta *transform* composed with the bit-pack
//! *encoding* ([`super::for_bitpack`]), rather than a monolithic mode.
//!
//! Wins on monotonic / regularly-stepped integer columns (timestamps, ids):
//! the deltas are small and clustered, so FoR+bitpack squeezes them to a few
//! bits. (Float bit patterns aren't delta-friendly across exponent bands, so it
//! rarely wins on `f64` — like the other integer codecs.)
//!
//! Decode is a scalar prefix sum. A lane-parallel (FastLanes) prefix sum was
//! tried and reverted: as a *separate* layer over [`super::for_bitpack`] it
//! didn't beat the scalar version — the required untranspose / strided writes
//! cancel the ILP gain (the unpack, not the add chain, is the larger cost). A
//! real speedup needs the prefix sum *fused into* the bit-unpack kernel
//! (FastLanes' `undelta_pack`), which is a `for_bitpack` rewrite, not done here;
//! and delta decode (~1.5 GB/s) isn't the dominant bottleneck anyway.

use crate::codecs::for_bitpack;
use crate::error::Error;

#[inline]
fn zigzag(n: u64) -> u64 {
    (n << 1) ^ ((n as i64 >> 63) as u64)
}

#[inline]
fn unzigzag(z: u64) -> u64 {
    (z >> 1) ^ 0u64.wrapping_sub(z & 1)
}

pub(crate) fn encode(vals: &[u64]) -> Vec<u8> {
    // Store the first value as the base so delta[0] = 0 — otherwise the absolute
    // first value (potentially huge) would blow the first sub-block's bit width.
    let base = vals.first().copied().unwrap_or(0);
    let mut out = base.to_le_bytes().to_vec();
    let mut deltas = Vec::with_capacity(vals.len());
    let mut prev = base;
    for &v in vals {
        deltas.push(zigzag(v.wrapping_sub(prev)));
        prev = v;
    }
    // Deltas are already zigzagged (unsigned magnitude), so FoR is unsigned.
    out.extend_from_slice(&for_bitpack::encode(&deltas, false));
    out
}

pub(crate) fn decode(payload: &[u8], n: usize) -> Result<Vec<u64>, Error> {
    let base = u64::from_le_bytes(
        payload
            .get(0..8)
            .ok_or(Error::Truncated)?
            .try_into()
            .unwrap(),
    );
    let deltas = for_bitpack::decode(&payload[8..], n, false)?;
    let mut out = Vec::with_capacity(n);
    let mut prev = base;
    for z in deltas {
        prev = prev.wrapping_add(unzigzag(z));
        out.push(prev);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monotonic_roundtrip() {
        // timestamp-like column: regular step + small noise.
        let mut s = 1u64;
        let mut t = 1_700_000_000_000u64;
        let vals: Vec<u64> = (0..5000)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                t = t.wrapping_add(1000 + (s >> 40) % 100);
                t
            })
            .collect();
        let enc = encode(&vals);
        assert!(
            enc.len() < vals.len() * 8 / 4,
            "monotonic ids should pack small"
        );
        assert_eq!(decode(&enc, vals.len()).unwrap(), vals);

        assert_eq!(decode(&encode(&[]), 0).unwrap(), Vec::<u64>::new());
        let one = [12345u64];
        assert_eq!(decode(&encode(&one), 1).unwrap(), one);
    }
}
