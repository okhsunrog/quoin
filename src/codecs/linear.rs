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
