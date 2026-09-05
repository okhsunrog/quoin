//! DELTA_BITPACK: first-order delta of the bit patterns (zigzag) feeding the
//! FoR + FastLanes bit-pack codec. This is Parquet's `DELTA_BINARY_PACKED`, and
//! our first **cascade** — a delta *transform* composed with the bit-pack
//! *encoding* ([`super::for_bitpack`]), rather than a monolithic mode.
//!
//! Wins on monotonic / regularly-stepped integer columns (timestamps, ids):
//! the deltas are small and clustered, so FoR+bitpack squeezes them to a few
//! bits. (Float bit patterns aren't delta-friendly across exponent bands, so it
//! rarely wins on floats — like the other integer codecs.)
//!
//! Decode is a scalar prefix sum. A lane-parallel (FastLanes) prefix sum was
//! tried and reverted: as a *separate* layer over [`super::for_bitpack`] it
//! didn't beat the scalar version — the required untranspose / strided writes
//! cancel the ILP gain (the unpack, not the add chain, is the larger cost). A
//! real speedup needs the prefix sum *fused into* the bit-unpack kernel
//! (FastLanes' `undelta_pack`), which is a `for_bitpack` rewrite, not done here;
//! and delta decode (~1.5 GB/s) isn't the dominant bottleneck anyway.
//!
//! Payload: `base:lane ++ for_bitpack(zigzag deltas)`.

use crate::codecs::for_bitpack;
use crate::error::Error;
use crate::lane::Lane;

pub(crate) fn encode<L: Lane>(vals: &[L]) -> Vec<u8> {
    // Store the first value as the base so delta[0] = 0 — otherwise the absolute
    // first value (potentially huge) would blow the first sub-block's bit width.
    let base = vals.first().copied().unwrap_or(L::ZERO);
    let mut out = Vec::with_capacity(L::BYTES + vals.len() * 2);
    base.write_le(&mut out);
    let mut deltas = Vec::with_capacity(vals.len());
    let mut prev = base;
    for &v in vals {
        deltas.push(v.wrapping_sub(prev).zigzag());
        prev = v;
    }
    // Deltas are already zigzagged (unsigned magnitude), so FoR is unsigned.
    out.extend_from_slice(&for_bitpack::encode(&deltas, false));
    out
}

pub(crate) fn decode<L: Lane>(payload: &[u8], n: usize) -> Result<Vec<L>, Error> {
    let base = L::read_le(payload.get(0..L::BYTES).ok_or(Error::Truncated)?);
    let deltas = for_bitpack::decode::<L>(&payload[L::BYTES..], n, false)?;
    let mut out = Vec::with_capacity(n);
    let mut prev = base;
    for z in deltas {
        prev = prev.wrapping_add(z.unzigzag());
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
        assert_eq!(decode::<u64>(&enc, vals.len()).unwrap(), vals);

        assert_eq!(
            decode::<u64>(&encode::<u64>(&[]), 0).unwrap(),
            Vec::<u64>::new()
        );
        let one = [12345u64];
        assert_eq!(decode::<u64>(&encode(&one), 1).unwrap(), one);
    }

    #[test]
    fn lane32_roundtrip() {
        // Relative-time-like i32 column with a negative excursion and a wrap.
        let vals: Vec<u32> = (0..5000i32)
            .map(|i| (i * 2 - 300 + (i % 7)) as u32)
            .chain([u32::MAX, 0, 5])
            .collect();
        let enc = encode(&vals);
        assert!(
            enc.len() < vals.len(),
            "smooth i32 deltas pack under 1 B/value"
        );
        assert_eq!(decode::<u32>(&enc, vals.len()).unwrap(), vals);
        assert_eq!(
            decode::<u32>(&encode::<u32>(&[]), 0).unwrap(),
            Vec::<u32>::new()
        );
        assert_eq!(decode::<u32>(&encode(&[7u32]), 1).unwrap(), [7]);
    }
}
