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
fn predict_f64(h: &[f64; MAX_ORDER], avail: usize, order: usize) -> f64 {
    let eff = order.min(avail);
    if eff == 0 {
        return 0.0;
    }
    let c = COEFFS[eff];
    let mut pred = 0.0f64;
    for (k, &coef) in c.iter().enumerate() {
        pred += f64::from(coef) * h[k];
    }
    pred
}

#[inline]
fn push_hist(h: &mut [f64; MAX_ORDER], v: f64) {
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
/// grow), so the higher orders never get a chance to amplify noise.
pub(crate) fn select_order(vals: &[u64]) -> usize {
    let n = vals.len();
    if n < 8 {
        return 2;
    }
    let win = 1024.min(n);
    let start = (n - win) / 2; // skip the warm-up edge
    let mut d: Vec<f64> = vals[start..start + win].iter().map(|&b| f64::from_bits(b)).collect();
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
pub(crate) fn encode(vals: &[u64], order: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() + 1);
    out.push(order as u8);
    let mut h = [0.0f64; MAX_ORDER];
    for (i, &bits) in vals.iter().enumerate() {
        let pred = predict_f64(&h, i.min(MAX_ORDER), order).to_bits();
        varint::write_u64(&mut out, bits ^ pred);
        push_hist(&mut h, f64::from_bits(bits));
    }
    out
}

pub(crate) fn decode(payload: &[u8], n: usize) -> Result<Vec<u64>, Error> {
    let (&order_b, rest) = payload.split_first().ok_or(Error::Truncated)?;
    let order = order_b as usize;
    if !(1..=MAX_ORDER).contains(&order) {
        return Err(Error::CorruptPayload("delta2 order"));
    }
    let mut out = Vec::with_capacity(n);
    let mut h = [0.0f64; MAX_ORDER];
    let mut pos = 0usize;
    for i in 0..n {
        let pred = predict_f64(&h, i.min(MAX_ORDER), order).to_bits();
        let bits = varint::read_u64(rest, &mut pos)? ^ pred;
        out.push(bits);
        push_hist(&mut h, f64::from_bits(bits));
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
pub(crate) fn dp_encode(vals: &[u64], order: usize) -> Option<Vec<u8>> {
    if vals.is_empty() {
        return None;
    }
    let mut out = Vec::with_capacity(vals.len() + 1);
    out.push(order as u8);
    let mut h = [0.0f64; MAX_ORDER];
    let mut prev_rbits = 0u64;
    for (i, &bits) in vals.iter().enumerate() {
        let v = f64::from_bits(bits);
        let pred = predict_f64(&h, i.min(MAX_ORDER), order);
        let r = v - pred;
        if (pred + r).to_bits() != bits {
            return None; // not exactly invertible for this block
        }
        let rbits = r.to_bits();
        varint::write_u64(&mut out, rbits ^ prev_rbits);
        prev_rbits = rbits;
        push_hist(&mut h, v);
    }
    Some(out)
}

pub(crate) fn dp_decode(payload: &[u8], n: usize) -> Result<Vec<u64>, Error> {
    let (&order_b, rest) = payload.split_first().ok_or(Error::Truncated)?;
    let order = order_b as usize;
    if !(1..=MAX_ORDER).contains(&order) {
        return Err(Error::CorruptPayload("delta_dp order"));
    }
    let mut out = Vec::with_capacity(n);
    let mut h = [0.0f64; MAX_ORDER];
    let mut prev_rbits = 0u64;
    let mut pos = 0usize;
    for i in 0..n {
        let pred = predict_f64(&h, i.min(MAX_ORDER), order);
        let rbits = varint::read_u64(rest, &mut pos)? ^ prev_rbits;
        let v = pred + f64::from_bits(rbits);
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
    fn zigzag_roundtrips_including_negatives() {
        for &v in &[0u64, 1, 2, u64::MAX, u64::MAX - 1, 1u64 << 63, 12345] {
            assert_eq!(unzigzag(zigzag(v)), v);
        }
    }

    #[test]
    fn idelta2_roundtrips() {
        // Includes a monotone ramp and a wrap-around to exercise signed deltas.
        let vals: Vec<u64> = (0..1000u64)
            .map(|i| i.wrapping_mul(3).wrapping_sub(7))
            .collect();
        let enc = idelta2_encode(&vals);
        assert_eq!(idelta2_decode(&enc, vals.len()).unwrap(), vals);
    }

    #[test]
    fn float_delta_roundtrips_all_orders() {
        let vals: Vec<u64> = (0..1000).map(|i| ((i as f64) * 0.5).to_bits()).collect();
        for order in 1..=MAX_ORDER {
            let enc = encode(&vals, order);
            assert_eq!(decode(&enc, vals.len()).unwrap(), vals, "xor order {order}");
            if let Some(dp) = dp_encode(&vals, order) {
                assert_eq!(dp_decode(&dp, vals.len()).unwrap(), vals, "dp order {order}");
            }
        }
    }

    #[test]
    fn select_order_prefers_high_on_smooth_low_on_noise() {
        // A finely-sampled sine: higher differences shrink → high order.
        let smooth: Vec<u64> = (0..2000)
            .map(|i| ((i as f64 * 0.002).sin() * 100.0 + i as f64 * 0.01).to_bits())
            .collect();
        assert!(select_order(&smooth) >= 3, "smooth signal should pick a high order");
        // Random bit patterns: differences explode → back off to a low order.
        let mut s = 1u64;
        let noise: Vec<u64> = (0..2000)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                f64::from_bits(s >> 2).to_bits()
            })
            .collect();
        assert!(select_order(&noise) <= 2, "noise should not pick a high order");
    }
}
