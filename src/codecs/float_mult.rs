//! FLOAT_MULT: when every value is an integer multiple of `1/scale` (cent-
//! rounded prices, fixed-decimal data), store the integers `k = round(v*scale)`
//! instead of the float bit patterns. The integers are small and smooth where
//! the floats look like high-entropy mantissas.
//!
//! Float `*`/`/` only round-trips for the right values, so the encoder tries a
//! set of decimal scales and **verifies** `k/scale == v` bit-for-bit for every
//! value — in the lane's own float type, so an `f32` column is verified and
//! decoded as `f32` — and returns `None` if no scale works (another mode wins).
//! The decoder recomputes `k/scale`, which is exact by construction.
//!
//! The `k` stream is then coded one of two ways (the payload's first byte tags
//! which): **bit-packed** (signed frame-of-reference on the lane — random-
//! access, available at every level) or **entropy-coded** (zig-zag delta through
//! the residual coder — wins on smooth/monotone `k`, only at the entropy
//! levels). Keeping the bit-pack path lets FLOAT_MULT compete at the fast
//! levels, where it used to be gated out for relying on the entropy coder.

use crate::codecs::for_bitpack;
use crate::entropy::{code_residuals, decode_residuals};
use crate::error::Error;
use crate::lane::{Lane, LaneFloat};
use crate::varint;

/// Payload tag: the `k` values are signed-FoR bit-packed (random-access).
const FM_BITPACK: u8 = 0;
/// Payload tag: the `k` values are zig-zag delta + residual-coded.
const FM_ENTROPY: u8 = 1;

#[inline]
fn zigzag(n: i64) -> u64 {
    ((n << 1) ^ (n >> 63)) as u64
}

#[inline]
fn unzigzag(z: u64) -> i64 {
    ((z >> 1) as i64) ^ -((z & 1) as i64)
}

/// Find the first scale for which every value is exactly `k/scale`, returning the
/// scale index and the integer `k` values (as lane words, two's-complement).
fn find_scale<L: Lane>(vals: &[L]) -> Option<(u8, Vec<L>)> {
    'scales: for (idx, &scale) in L::Float::FM_SCALES.iter().enumerate() {
        let mut ks = Vec::with_capacity(vals.len());
        for &bits in vals {
            let x = bits.to_float();
            let k = (x * scale).round();
            // Must be a finite integer that fits the lane and reconstructs v exactly.
            if !k.is_finite() || k.abs() >= L::Float::FM_LIMIT {
                continue 'scales;
            }
            let ki = k.to_i64();
            if (L::Float::from_i64(ki) / scale).to_bits() != bits {
                continue 'scales;
            }
            ks.push(L::from_signed(ki));
        }
        return Some((idx as u8, ks));
    }
    None
}

pub(crate) fn encode<L: Lane>(
    vals: &[L],
    entropy: bool,
    lambda: u64,
    allow_lz: bool,
) -> Option<Vec<u8>> {
    if vals.is_empty() {
        return None;
    }
    let (scale_idx, ks) = find_scale(vals)?;

    // Bit-pack path: always available, so FLOAT_MULT survives the fast levels.
    let mut best_tag = FM_BITPACK;
    let mut best_blob = for_bitpack::encode(&ks, true);

    // Entropy path: zig-zag delta through the residual coder. Wins on smooth or
    // monotone `k` (tiny, skewed deltas); only at the entropy levels.
    if entropy {
        let mut delta = Vec::with_capacity(ks.len());
        let mut prev = 0i64;
        for &k in &ks {
            let ki = k.as_signed();
            varint::write_u64(&mut delta, zigzag(ki.wrapping_sub(prev)));
            prev = ki;
        }
        let coded = code_residuals(&delta, lambda, allow_lz);
        if coded.len() < best_blob.len() {
            best_tag = FM_ENTROPY;
            best_blob = coded;
        }
    }

    let mut out = Vec::with_capacity(best_blob.len() + 2);
    out.push(best_tag);
    out.push(scale_idx);
    out.extend_from_slice(&best_blob);
    Some(out)
}

pub(crate) fn decode<L: Lane>(payload: &[u8], n: usize) -> Result<Vec<L>, Error> {
    let tag = *payload.first().ok_or(Error::Truncated)?;
    let scale_idx = *payload.get(1).ok_or(Error::Truncated)?;
    let scale = *L::Float::FM_SCALES
        .get(scale_idx as usize)
        .ok_or(Error::CorruptPayload("float_mult scale index"))?;
    let rest = &payload[2..];

    let ks: Vec<i64> = match tag {
        FM_BITPACK => for_bitpack::decode::<L>(rest, n, true)?
            .into_iter()
            .map(|k| k.as_signed())
            .collect(),
        FM_ENTROPY => {
            let delta = decode_residuals(rest, n.saturating_mul(10) + 16)?;
            let mut ks = Vec::with_capacity(n);
            let mut prev = 0i64;
            let mut pos = 0usize;
            for _ in 0..n {
                let d = unzigzag(varint::read_u64(&delta, &mut pos)?);
                let ki = prev.wrapping_add(d);
                ks.push(ki);
                prev = ki;
            }
            if pos != delta.len() {
                return Err(Error::CorruptPayload("float_mult trailing bytes"));
            }
            ks
        }
        _ => return Err(Error::CorruptPayload("float_mult tag")),
    };

    Ok(ks
        .iter()
        .map(|&ki| (L::Float::from_i64(ki) / scale).to_bits())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(vals: &[f64], entropy: bool) {
        let lane: Vec<u64> = vals.iter().map(|v| v.to_bits()).collect();
        let payload = encode(&lane, entropy, 0, true).expect("decimal data must encode");
        let back = decode::<u64>(&payload, lane.len()).unwrap();
        assert_eq!(back, lane);
    }

    fn roundtrip32(vals: &[f32], entropy: bool) -> usize {
        let lane: Vec<u32> = vals.iter().map(|v| v.to_bits()).collect();
        let payload = encode(&lane, entropy, 0, true).expect("decimal f32 data must encode");
        let back = decode::<u32>(&payload, lane.len()).unwrap();
        assert_eq!(back, lane);
        payload.len()
    }

    #[test]
    fn both_paths_roundtrip() {
        // Cent-rounded prices: exact at scale 100.
        let scattered: Vec<f64> = (0..2000)
            .map(|i| ((i * 7) % 100_000) as f64 / 100.0)
            .collect();
        // Smooth/monotone k (favours the entropy delta path).
        let smooth: Vec<f64> = (0..2000).map(|i| 1000.0 + i as f64 / 1000.0).collect();
        for data in [&scattered, &smooth] {
            roundtrip(data, false); // bit-pack path (fast levels)
            roundtrip(data, true); // both paths considered (entropy levels)
        }
        // Negative values and zero through the signed FoR.
        roundtrip(&[-12.34, 0.0, 56.78, -0.01, 99.99], false);
        roundtrip(&[-12.34, 0.0, 56.78, -0.01, 99.99], true);
    }

    #[test]
    fn f32_paths_roundtrip() {
        let scattered: Vec<f32> = (0..2000)
            .map(|i| ((i * 7) % 100_000) as f32 / 100.0)
            .collect();
        let smooth: Vec<f32> = (0..2000).map(|i| 1000.0 + i as f32 / 100.0).collect();
        for data in [&scattered, &smooth] {
            let bp = roundtrip32(data, false);
            assert!(bp < data.len() * 4, "f32 decimals pack under 4 B/value");
            roundtrip32(data, true);
        }
        roundtrip32(&[-12.34, 0.0, 56.78, -0.01, 99.99], false);
        roundtrip32(&[-12.34, 0.0, 56.78, -0.01, 99.99], true);
        // Values whose k does not fit an i32 bail (no silent truncation).
        let huge: Vec<u32> = vec![3.0e8f32.to_bits(); 4];
        assert!(encode(&huge, false, 0, false).is_none());
    }

    #[test]
    fn non_decimal_bails() {
        let pi = [
            std::f64::consts::PI.to_bits(),
            std::f64::consts::E.to_bits(),
        ];
        assert!(encode(&pi, true, 0, true).is_none());
        // f32 has ~7 significant digits, so moderate-magnitude f32 values *are*
        // exact multiples of 1e-7 at the k/scale precision — FLOAT_MULT may
        // legitimately encode π; it just has to be exact when it does.
        let pi32 = [
            std::f32::consts::PI.to_bits(),
            std::f32::consts::E.to_bits(),
        ];
        if let Some(p) = encode(&pi32, true, 0, true) {
            assert_eq!(decode::<u32>(&p, 2).unwrap(), pi32);
        }
        // Magnitudes whose `k` would exceed the i32 lane at every scale: bails
        // rather than truncating.
        let big = [(std::f32::consts::PI * 1e8).to_bits(), 5.0e8f32.to_bits()];
        assert!(encode(&big, true, 0, true).is_none());
        // -0.0 has no integer k (k/scale would be +0.0): the block bails.
        assert!(encode(&[(-0.0f32).to_bits(), 1.5f32.to_bits()], false, 0, false).is_none());
    }
}
