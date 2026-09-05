//! DELTA2: second-order linear extrapolation in *floating-point* space.
//!
//! For a locally-smooth signal, `v[i]` is well approximated by the line through
//! its two predecessors: `pred = 2*v[i-1] - v[i-2]`. We XOR the actual bit
//! pattern with the prediction's bit pattern; when the prediction is close, the
//! operands share exponent and high-mantissa bits, so the XOR has many leading
//! zero bits and LEB128 + entropy coding shrink it.
//!
//! The arithmetic runs in the lane's own float type (`f32` on a 32-bit lane,
//! `f64` on a 64-bit one): plain IEEE-754 add/mul, no fused ops, so encode and
//! decode reproduce the prediction bit-for-bit on every platform — *provided
//! every input is finite*. NaN-payload propagation and the sign of a default
//! NaN (`inf - inf`) are implementation-defined, so the encoder skips these
//! modes for a block containing any inf/NaN (see `encoder.rs`); nothing here
//! is allowed to depend on such a value.
//!
//! This is where the FCM/DFCM predictors (which work on raw integers) fall down
//! on oscillating signals that cross zero.

use crate::error::Error;
use crate::lane::{Lane, LaneFloat};
use crate::varint;

/// IDELTA2: second-order delta of the raw lane bit patterns (subtractive,
/// wrapping), zigzag + LEB128. For monotone-ish data (ramps, `0.5*i*i`) the
/// integer second difference is constant within each exponent band and spikes
/// only at band boundaries — far more compressible than the float-XOR variant.
pub(crate) fn idelta2_encode<L: Lane>(vals: &[L]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len());
    let (mut p1, mut p2) = (L::ZERO, L::ZERO);
    for (i, &v) in vals.iter().enumerate() {
        let pred = match i {
            0 => L::ZERO,
            1 => p1,
            _ => p1.wrapping_add(p1).wrapping_sub(p2),
        };
        varint::write_u64(&mut out, v.wrapping_sub(pred).zigzag().to_u64());
        p2 = p1;
        p1 = v;
    }
    out
}

pub(crate) fn idelta2_decode<L: Lane>(payload: &[u8], n: usize) -> Result<Vec<L>, Error> {
    let mut out = Vec::with_capacity(n);
    let (mut p1, mut p2) = (L::ZERO, L::ZERO);
    let mut pos = 0usize;
    for i in 0..n {
        let pred = match i {
            0 => L::ZERO,
            1 => p1,
            _ => p1.wrapping_add(p1).wrapping_sub(p2),
        };
        let z = L::try_from_u64(varint::read_u64(payload, &mut pos)?)?;
        let v = pred.wrapping_add(z.unzigzag());
        out.push(v);
        p2 = p1;
        p1 = v;
    }
    if pos != payload.len() {
        return Err(Error::CorruptPayload("idelta2 trailing bytes"));
    }
    Ok(out)
}

/// Forward-difference (Newton) extrapolation coefficients by order, most-recent
/// first: order 1 → `[1]` (hold), 2 → `[2,-1]` (linear), 3 → `[3,-3,1]`
/// (quadratic), 4 → `[4,-6,4,-1]` (cubic). The order-`d` predictor's residual is
/// exactly the `d`-th finite difference `Δ^d`, so it vanishes for degree-`<d`
/// polynomials and *shrinks on any smooth signal* as `d` rises — while *growing*
/// on noise (each differencing amplifies it), which is what [`select_order`]
/// exploits to back off. Order 2 is the original DELTA2 behaviour.
const COEFFS: [&[i32]; 5] = [&[], &[1], &[2, -1], &[3, -3, 1], &[4, -6, 4, -1]];
const MAX_ORDER: usize = 4;

/// Predict `v[i]` from the history `h` (most-recent first; `avail` valid entries)
/// by extrapolating a degree-`order-1` polynomial. During warm-up (`avail <
/// order`) the effective order drops to what's available, so encode and decode
/// stay in lock-step.
#[inline]
fn predict<F: LaneFloat>(h: &[F; MAX_ORDER], avail: usize, order: usize) -> F {
    let eff = order.min(avail);
    if eff == 0 {
        return F::ZERO;
    }
    let c = COEFFS[eff];
    let mut pred = F::ZERO;
    for (k, &coef) in c.iter().enumerate() {
        pred = pred + F::from_i32(coef) * h[k];
    }
    pred
}

#[inline]
fn push_hist<F: LaneFloat>(h: &mut [F; MAX_ORDER], v: F) {
    h[3] = h[2];
    h[2] = h[1];
    h[1] = h[0];
    h[0] = v;
}

/// Difference a contiguous sample in place (`d[i] ← d[i+1] − d[i]`, length − 1).
fn diff_in_place(d: &mut Vec<f64>) {
    let len = d.len();
    for i in 0..len.saturating_sub(1) {
        d[i] = d[i + 1] - d[i];
    }
    d.pop();
}

fn mean_abs(d: &[f64]) -> f64 {
    if d.is_empty() {
        return f64::INFINITY;
    }
    let s: f64 = d.iter().map(|x| x.abs()).sum();
    s / d.len() as f64
}

/// Choose the predictor order (1..=`MAX_ORDER`) whose residual — the order-th
/// finite difference — is smallest on a contiguous sample. Smooth data drives
/// this up (higher differences shrink); noisy/random data keeps it low (they
/// grow), so the higher orders never get a chance to amplify noise. An
/// encode-side heuristic only (the order is stored), so it evaluates in `f64`
/// for either lane.
pub(crate) fn select_order<L: Lane>(vals: &[L]) -> usize {
    let n = vals.len();
    if n < 8 {
        return 2;
    }
    let win = 1024.min(n);
    let start = (n - win) / 2; // skip the warm-up edge
    let mut d: Vec<f64> = vals[start..start + win]
        .iter()
        .map(|&b| b.to_float().to_f64())
        .collect();
    let (mut best_order, mut best_mag) = (1usize, f64::INFINITY);
    for order in 1..=MAX_ORDER {
        diff_in_place(&mut d);
        let mag = mean_abs(&d);
        if mag.is_finite() && mag < best_mag {
            best_mag = mag;
            best_order = order;
        }
    }
    best_order
}

/// DELTA2 (now order-parameterised): extrapolate a low-degree polynomial through
/// the previous values and XOR the actual bit pattern with the prediction's. The
/// float arithmetic is deterministic and reproduced on decode, so this is
/// lossless regardless of rounding. The payload is `[order] ++ xor-residuals`.
pub(crate) fn encode<L: Lane>(vals: &[L], order: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() + 1);
    out.push(order as u8);
    let mut h = [L::Float::ZERO; MAX_ORDER];
    for (i, &bits) in vals.iter().enumerate() {
        let pred = predict(&h, i.min(MAX_ORDER), order).to_bits();
        varint::write_u64(&mut out, (bits ^ pred).to_u64());
        push_hist(&mut h, bits.to_float());
    }
    out
}

pub(crate) fn decode<L: Lane>(payload: &[u8], n: usize) -> Result<Vec<L>, Error> {
    let (&order_b, rest) = payload.split_first().ok_or(Error::Truncated)?;
    let order = order_b as usize;
    if !(1..=MAX_ORDER).contains(&order) {
        return Err(Error::CorruptPayload("delta2 order"));
    }
    let mut out = Vec::with_capacity(n);
    let mut h = [L::Float::ZERO; MAX_ORDER];
    let mut pos = 0usize;
    for i in 0..n {
        let pred = predict(&h, i.min(MAX_ORDER), order).to_bits();
        let bits = L::try_from_u64(varint::read_u64(rest, &mut pos)?)? ^ pred;
        out.push(bits);
        push_hist(&mut h, bits.to_float());
    }
    if pos != rest.len() {
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
/// reconstruction is exact. Payload is `[order] ++ residuals`.
pub(crate) fn dp_encode<L: Lane>(vals: &[L], order: usize) -> Option<Vec<u8>> {
    if vals.is_empty() {
        return None;
    }
    let mut out = Vec::with_capacity(vals.len() + 1);
    out.push(order as u8);
    let mut h = [L::Float::ZERO; MAX_ORDER];
    let mut prev_rbits = L::ZERO;
    for (i, &bits) in vals.iter().enumerate() {
        let v = bits.to_float();
        let pred = predict(&h, i.min(MAX_ORDER), order);
        let r = v - pred;
        if (pred + r).to_bits() != bits {
            return None; // not exactly invertible for this block
        }
        let rbits = r.to_bits();
        varint::write_u64(&mut out, (rbits ^ prev_rbits).to_u64());
        prev_rbits = rbits;
        push_hist(&mut h, v);
    }
    Some(out)
}

pub(crate) fn dp_decode<L: Lane>(payload: &[u8], n: usize) -> Result<Vec<L>, Error> {
    let (&order_b, rest) = payload.split_first().ok_or(Error::Truncated)?;
    let order = order_b as usize;
    if !(1..=MAX_ORDER).contains(&order) {
        return Err(Error::CorruptPayload("delta_dp order"));
    }
    let mut out = Vec::with_capacity(n);
    let mut h = [L::Float::ZERO; MAX_ORDER];
    let mut prev_rbits = L::ZERO;
    let mut pos = 0usize;
    for i in 0..n {
        let pred = predict(&h, i.min(MAX_ORDER), order);
        let rbits = L::try_from_u64(varint::read_u64(rest, &mut pos)?)? ^ prev_rbits;
        let v = pred + rbits.to_float();
        out.push(v.to_bits());
        prev_rbits = rbits;
        push_hist(&mut h, v);
    }
    if pos != rest.len() {
        return Err(Error::CorruptPayload("delta_dp trailing bytes"));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idelta2_roundtrips() {
        // Includes a monotone ramp and a wrap-around to exercise signed deltas.
        let vals: Vec<u64> = (0..1000u64)
            .map(|i| i.wrapping_mul(3).wrapping_sub(7))
            .collect();
        let enc = idelta2_encode(&vals);
        assert_eq!(idelta2_decode::<u64>(&enc, vals.len()).unwrap(), vals);
        let vals32: Vec<u32> = (0..1000u32)
            .map(|i| i.wrapping_mul(3).wrapping_sub(7))
            .collect();
        let enc = idelta2_encode(&vals32);
        assert!(
            enc.len() <= vals32.len() + 8,
            "constant 2nd difference → 1 byte/value"
        );
        assert_eq!(idelta2_decode::<u32>(&enc, vals32.len()).unwrap(), vals32);
    }

    #[test]
    fn float_delta_roundtrips_all_orders() {
        let vals: Vec<u64> = (0..1000).map(|i| ((i as f64) * 0.5).to_bits()).collect();
        for order in 1..=MAX_ORDER {
            let enc = encode(&vals, order);
            assert_eq!(
                decode::<u64>(&enc, vals.len()).unwrap(),
                vals,
                "xor order {order}"
            );
            if let Some(dp) = dp_encode(&vals, order) {
                assert_eq!(
                    dp_decode::<u64>(&dp, vals.len()).unwrap(),
                    vals,
                    "dp order {order}"
                );
            }
        }
    }

    #[test]
    fn f32_lane_predicts_in_f32() {
        // A parabola in f32: the 2nd difference is exactly 1.0 → DELTA_DP is
        // exact and tiny; DELTA2 XOR residuals are small.
        let vals: Vec<u32> = (0..2000)
            .map(|i| ((i * i) as f32 * 0.5).to_bits())
            .collect();
        for order in 1..=MAX_ORDER {
            let enc = encode(&vals, order);
            assert_eq!(
                decode::<u32>(&enc, vals.len()).unwrap(),
                vals,
                "xor order {order}"
            );
            if let Some(dp) = dp_encode(&vals, order) {
                assert_eq!(
                    dp_decode::<u32>(&dp, vals.len()).unwrap(),
                    vals,
                    "dp order {order}"
                );
                if order == 3 {
                    assert!(
                        dp.len() < vals.len() + 16,
                        "exact cubic residuals are ~1 byte"
                    );
                }
            }
        }
        // Signed zero and subnormals are ordinary finite values: bit-exact.
        let edge: Vec<u32> = [0.0f32, -0.0, f32::from_bits(1), -1.5, 3e-39, 1e30, -1e-30]
            .iter()
            .map(|f| f.to_bits())
            .collect();
        for order in 1..=MAX_ORDER {
            assert_eq!(
                decode::<u32>(&encode(&edge, order), edge.len()).unwrap(),
                edge
            );
            if let Some(dp) = dp_encode(&edge, order) {
                assert_eq!(dp_decode::<u32>(&dp, edge.len()).unwrap(), edge);
            }
        }
    }

    #[test]
    fn select_order_prefers_high_on_smooth_low_on_noise() {
        // A finely-sampled sine: higher differences shrink → high order.
        let smooth: Vec<u64> = (0..2000)
            .map(|i| ((i as f64 * 0.002).sin() * 100.0 + i as f64 * 0.01).to_bits())
            .collect();
        assert!(
            select_order(&smooth) >= 3,
            "smooth signal should pick a high order"
        );
        let smooth32: Vec<u32> = (0..2000)
            .map(|i| ((i as f32 * 0.002).sin() * 100.0 + i as f32 * 0.01).to_bits())
            .collect();
        assert!(
            select_order(&smooth32) >= 2,
            "smooth f32 signal picks a high order"
        );
        // Random bit patterns: differences explode → back off to a low order.
        let mut s = 1u64;
        let noise: Vec<u64> = (0..2000)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                f64::from_bits(s >> 2).to_bits()
            })
            .collect();
        assert!(
            select_order(&noise) <= 2,
            "noise should not pick a high order"
        );
    }
}
