//! DELTA2: second-order linear extrapolation in *floating-point* space.
//!
//! For a locally-smooth signal, `v[i]` is well approximated by the line through
//! its two predecessors: `pred = 2*v[i-1] - v[i-2]`. We XOR the actual bit
//! pattern with the prediction's bit pattern; when the prediction is close, the
//! operands share exponent and high-mantissa bits, so the XOR has many leading
//! zero bits and LEB128 + entropy coding shrink it.
//!
//! The float arithmetic is deterministic and reproduced exactly on decode, so
//! storing `bits(v) ^ bits(pred)` is lossless regardless of rounding — this is
//! where the FCM/DFCM predictors (which work on raw integers) fall down on
//! oscillating signals that cross zero.

use crate::error::Error;
use crate::varint;

#[inline]
fn zigzag(n: u64) -> u64 {
    (n << 1) ^ ((n as i64 >> 63) as u64)
}

#[inline]
fn unzigzag(z: u64) -> u64 {
    (z >> 1) ^ 0u64.wrapping_sub(z & 1)
}

/// IDELTA2: second-order delta of the raw `u64` bit patterns (subtractive,
/// wrapping), zigzag + LEB128. For monotone-ish data (ramps, `0.5*i*i`) the
/// integer second difference is constant within each exponent band and spikes
/// only at band boundaries — far more compressible than the float-XOR variant.
pub(crate) fn idelta2_encode(vals: &[u64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len());
    let (mut p1, mut p2) = (0u64, 0u64);
    for (i, &v) in vals.iter().enumerate() {
        let pred = match i {
            0 => 0,
            1 => p1,
            _ => p1.wrapping_mul(2).wrapping_sub(p2),
        };
        varint::write_u64(&mut out, zigzag(v.wrapping_sub(pred)));
        p2 = p1;
        p1 = v;
    }
    out
}

pub(crate) fn idelta2_decode(payload: &[u8], n: usize) -> Result<Vec<u64>, Error> {
    let mut out = Vec::with_capacity(n);
    let (mut p1, mut p2) = (0u64, 0u64);
    let mut pos = 0usize;
    for i in 0..n {
        let pred = match i {
            0 => 0,
            1 => p1,
            _ => p1.wrapping_mul(2).wrapping_sub(p2),
        };
        let v = pred.wrapping_add(unzigzag(varint::read_u64(payload, &mut pos)?));
        out.push(v);
        p2 = p1;
        p1 = v;
    }
    if pos != payload.len() {
        return Err(Error::CorruptPayload("idelta2 trailing bytes"));
    }
    Ok(out)
}

#[inline]
fn predict(i: usize, prev1: f64, prev2: f64) -> u64 {
    let pred = match i {
        0 => 0.0,
        1 => prev1,
        _ => 2.0 * prev1 - prev2,
    };
    pred.to_bits()
}

pub(crate) fn encode(vals: &[u64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len());
    let (mut prev1, mut prev2) = (0.0f64, 0.0f64);
    for (i, &bits) in vals.iter().enumerate() {
        let pred = predict(i, prev1, prev2);
        varint::write_u64(&mut out, bits ^ pred);
        prev2 = prev1;
        prev1 = f64::from_bits(bits);
    }
    out
}

pub(crate) fn decode(payload: &[u8], n: usize) -> Result<Vec<u64>, Error> {
    let mut out = Vec::with_capacity(n);
    let (mut prev1, mut prev2) = (0.0f64, 0.0f64);
    let mut pos = 0usize;
    for i in 0..n {
        let pred = predict(i, prev1, prev2);
        let resid = varint::read_u64(payload, &mut pos)?;
        let bits = resid ^ pred;
        out.push(bits);
        prev2 = prev1;
        prev1 = f64::from_bits(bits);
    }
    if pos != payload.len() {
        return Err(Error::CorruptPayload("delta2 trailing bytes"));
    }
    Ok(out)
}

/// DELTA_DP: like [`encode`] but stores the *floating-point* residual
/// `r = v - pred` (bit pattern, delta-coded) instead of the XOR. For smooth
/// data the subtraction is exact (Sterbenz) and the residual is tiny and often
/// constant — e.g. a parabola's second difference is exactly `1.0`.
///
/// Float subtract/add is only invertible when the subtraction is exact, so the
/// encoder **verifies** `pred + r == v` bit-for-bit and returns `None` if any
/// value fails (another mode then wins). The decoder can therefore trust that
/// reconstruction is exact.
pub(crate) fn dp_encode(vals: &[u64]) -> Option<Vec<u8>> {
    if vals.is_empty() {
        return None;
    }
    let mut out = Vec::with_capacity(vals.len());
    let (mut prev1, mut prev2) = (0.0f64, 0.0f64);
    let mut prev_rbits = 0u64;
    for (i, &bits) in vals.iter().enumerate() {
        let v = f64::from_bits(bits);
        let pred = match i {
            0 => 0.0,
            1 => prev1,
            _ => 2.0 * prev1 - prev2,
        };
        let r = v - pred;
        if (pred + r).to_bits() != bits {
            return None; // not exactly invertible for this block
        }
        let rbits = r.to_bits();
        varint::write_u64(&mut out, rbits ^ prev_rbits);
        prev_rbits = rbits;
        prev2 = prev1;
        prev1 = v;
    }
    Some(out)
}

pub(crate) fn dp_decode(payload: &[u8], n: usize) -> Result<Vec<u64>, Error> {
    let mut out = Vec::with_capacity(n);
    let (mut prev1, mut prev2) = (0.0f64, 0.0f64);
    let mut prev_rbits = 0u64;
    let mut pos = 0usize;
    for i in 0..n {
        let pred = match i {
            0 => 0.0,
            1 => prev1,
            _ => 2.0 * prev1 - prev2,
        };
        let rbits = varint::read_u64(payload, &mut pos)? ^ prev_rbits;
        let v = pred + f64::from_bits(rbits);
        out.push(v.to_bits());
        prev_rbits = rbits;
        prev2 = prev1;
        prev1 = v;
    }
    if pos != payload.len() {
        return Err(Error::CorruptPayload("delta_dp trailing bytes"));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zigzag_roundtrips_including_negatives() {
        for &v in &[0u64, 1, 2, u64::MAX, u64::MAX - 1, 1u64 << 63, 12345] {
            assert_eq!(unzigzag(zigzag(v)), v);
        }
    }

    #[test]
    fn idelta2_roundtrips() {
        // Includes a monotone ramp and a wrap-around to exercise signed deltas.
        let vals: Vec<u64> = (0..1000u64).map(|i| i.wrapping_mul(3).wrapping_sub(7)).collect();
        let enc = idelta2_encode(&vals);
        assert_eq!(idelta2_decode(&enc, vals.len()).unwrap(), vals);
    }

    #[test]
    fn float_delta2_roundtrips() {
        let vals: Vec<u64> = (0..1000).map(|i| ((i as f64) * 0.5).to_bits()).collect();
        let enc = encode(&vals);
        assert_eq!(decode(&enc, vals.len()).unwrap(), vals);
    }
}
