//! FLOAT_MULT: when every value is an integer multiple of `1/scale` (cent-
//! rounded prices, fixed-decimal data), store the integers `k = round(v*scale)`
//! instead of the float bit patterns. The integers are small and smooth where
//! the floats look like high-entropy mantissas.
//!
//! Float `*`/`/` only round-trips for the right values, so the encoder tries a
//! set of decimal scales and **verifies** `k/scale == v` bit-for-bit for every
//! value; it returns `None` if no scale works (another mode wins). The decoder
//! recomputes `k/scale`, which is exact by construction.
//!
//! The `k` stream is then coded one of two ways (the payload's first byte tags
//! which): **bit-packed** (signed frame-of-reference — random-access, available
//! at every level) or **entropy-coded** (zig-zag delta through the residual
//! coder — wins on smooth/monotone `k`, only at the entropy levels). Keeping the
//! bit-pack path lets FLOAT_MULT compete at the fast levels, where it used to be
//! gated out for relying on the entropy coder.

use crate::codecs::for_bitpack;
use crate::entropy::{code_residuals, decode_residuals};
use crate::error::Error;
use crate::varint;

/// Candidate scales (decimal). Index stored in the payload header.
const SCALES: [f64; 7] = [
    10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0,
];

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
/// scale index and the integer `k` values.
fn find_scale(vals: &[u64]) -> Option<(u8, Vec<i64>)> {
    'scales: for (idx, &scale) in SCALES.iter().enumerate() {
        let mut ks = Vec::with_capacity(vals.len());
        for &bits in vals {
            let x = f64::from_bits(bits);
            let k = (x * scale).round();
            // Must be a finite integer in i64 range that reconstructs v exactly.
            if !k.is_finite() || k.abs() >= 9.0e18 {
                continue 'scales;
            }
            let ki = k as i64;
            if (ki as f64 / scale).to_bits() != bits {
                continue 'scales;
            }
            ks.push(ki);
        }
        return Some((idx as u8, ks));
    }
    None
}

pub(crate) fn encode(vals: &[u64], entropy: bool, lambda: u64, allow_lz: bool) -> Option<Vec<u8>> {
    if vals.is_empty() {
        return None;
    }
    let (scale_idx, ks) = find_scale(vals)?;

    // Bit-pack path: always available, so FLOAT_MULT survives the fast levels.
    let k_u64: Vec<u64> = ks.iter().map(|&k| k as u64).collect();
    let mut best_tag = FM_BITPACK;
    let mut best_blob = for_bitpack::encode(&k_u64, true);

    // Entropy path: zig-zag delta through the residual coder. Wins on smooth or
    // monotone `k` (tiny, skewed deltas); only at the entropy levels.
    if entropy {
        let mut delta = Vec::with_capacity(ks.len());
        let mut prev = 0i64;
        for &ki in &ks {
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

pub(crate) fn decode(payload: &[u8], n: usize) -> Result<Vec<u64>, Error> {
    let tag = *payload.first().ok_or(Error::Truncated)?;
    let scale_idx = *payload.get(1).ok_or(Error::Truncated)?;
    let scale = *SCALES
        .get(scale_idx as usize)
        .ok_or(Error::CorruptPayload("float_mult scale index"))?;
    let rest = &payload[2..];

    let ks: Vec<i64> = match tag {
        FM_BITPACK => for_bitpack::decode(rest, n, true)?
            .into_iter()
            .map(|u| u as i64)
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

    Ok(ks.iter().map(|&ki| (ki as f64 / scale).to_bits()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(vals: &[f64], entropy: bool) {
        let lane: Vec<u64> = vals.iter().map(|v| v.to_bits()).collect();
        let payload = encode(&lane, entropy, 0, true).expect("decimal data must encode");
        let back = decode(&payload, lane.len()).unwrap();
        assert_eq!(back, lane);
    }

    #[test]
    fn both_paths_roundtrip() {
        // Cent-rounded prices: exact at scale 100.
        let scattered: Vec<f64> = (0..2000).map(|i| ((i * 7) % 100_000) as f64 / 100.0).collect();
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
    fn non_decimal_bails() {
        let pi = [std::f64::consts::PI.to_bits(), std::f64::consts::E.to_bits()];
        assert!(encode(&pi, true, 0, true).is_none());
    }
}
