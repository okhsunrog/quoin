//! FLOAT_MULT: when every value is an integer multiple of `1/scale` (cent-
//! rounded prices, fixed-decimal data), store the integers `k = round(v*scale)`
//! delta-coded instead of the float bit patterns. The integers are small and
//! smooth where the floats look like high-entropy mantissas.
//!
//! Float `*`/`/` only round-trips for the right values, so the encoder tries a
//! set of decimal scales and **verifies** `k/scale == v` bit-for-bit for every
//! value; it returns `None` if no scale works (another mode wins). The decoder
//! recomputes `k/scale`, which is exact by construction.

use crate::error::Error;
use crate::varint;

/// Candidate scales (decimal). Index stored in the payload header.
const SCALES: [f64; 7] = [
    10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0,
];

#[inline]
fn zigzag(n: i64) -> u64 {
    ((n << 1) ^ (n >> 63)) as u64
}

#[inline]
fn unzigzag(z: u64) -> i64 {
    ((z >> 1) as i64) ^ -((z & 1) as i64)
}

pub(crate) fn encode(vals: &[u64]) -> Option<Vec<u8>> {
    if vals.is_empty() {
        return None;
    }
    for (idx, &scale) in SCALES.iter().enumerate() {
        if let Some(bytes) = try_scale(vals, scale, idx as u8) {
            return Some(bytes);
        }
    }
    None
}

fn try_scale(vals: &[u64], scale: f64, scale_idx: u8) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(vals.len() + 1);
    out.push(scale_idx);
    let mut prev: i64 = 0;
    for &bits in vals {
        let x = f64::from_bits(bits);
        let k = (x * scale).round();
        // Must be a finite integer in i64 range that reconstructs v exactly.
        if !k.is_finite() || k.abs() >= 9.0e18 {
            return None;
        }
        let ki = k as i64;
        if (ki as f64 / scale).to_bits() != bits {
            return None;
        }
        varint::write_u64(&mut out, zigzag(ki.wrapping_sub(prev)));
        prev = ki;
    }
    Some(out)
}

pub(crate) fn decode(payload: &[u8], n: usize) -> Result<Vec<u64>, Error> {
    let (&scale_idx, rest) = payload.split_first().ok_or(Error::Truncated)?;
    let scale = *SCALES
        .get(scale_idx as usize)
        .ok_or(Error::CorruptPayload("float_mult scale index"))?;
    let mut out = Vec::with_capacity(n);
    let mut prev: i64 = 0;
    let mut pos = 0usize;
    for _ in 0..n {
        let d = unzigzag(varint::read_u64(rest, &mut pos)?);
        let ki = prev.wrapping_add(d);
        out.push((ki as f64 / scale).to_bits());
        prev = ki;
    }
    if pos != rest.len() {
        return Err(Error::CorruptPayload("float_mult trailing bytes"));
    }
    Ok(out)
}
