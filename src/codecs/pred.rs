//! PRED: finite-context-method (FCM) predictor.
//!
//! A hash of the most recent value indexes a table holding the last value seen
//! in that context; that becomes the prediction. We emit the XOR residual
//! (value ^ prediction) as a LEB128 varint — small when the prediction is
//! close, which is the common case for structured floating-point streams.
//!
//! This is the heart of the original `fc` and the first real consumer of the
//! [`crate::hash`] CRC32C kernel. Encode and decode evolve the table
//! identically, so the residual stream is all that needs storing. The table
//! holds lane words, so a 32-bit column's table is half the size of a 64-bit
//! one at the same `predictor_log2` (better cache residency).
//!
//! Two predictor flavors live here:
//! * [`encode`]/[`decode`] — **FCM**: predicts the last value seen in the
//!   current context. Great for repeats, poor for never-repeating ramps.
//! * [`dfcm_encode`]/[`dfcm_decode`] — **DFCM** (differential FCM): predicts
//!   `last_value + last_delta_seen_for_this_delta_context`. Nails smooth and
//!   linear data where every value is new but the *deltas* repeat.

use crate::error::Error;
use crate::hash::best_hash_fn;
use crate::lane::Lane;
use crate::varint;

pub(crate) fn encode<L: Lane>(vals: &[L], predictor_log2: u8) -> Vec<u8> {
    let mask = (1usize << predictor_log2) - 1;
    let mut table = vec![L::ZERO; mask + 1];
    let hash = best_hash_fn();
    let mut out = Vec::with_capacity(vals.len());
    let mut ctx = 0usize;
    for &v in vals {
        let pred = table[ctx];
        varint::write_u64(&mut out, (v ^ pred).to_u64());
        table[ctx] = v;
        ctx = (v.hash_step(hash) as usize) & mask;
    }
    out
}

pub(crate) fn decode<L: Lane>(
    payload: &[u8],
    n: usize,
    predictor_log2: u8,
) -> Result<Vec<L>, Error> {
    let mask = (1usize << predictor_log2) - 1;
    let mut table = vec![L::ZERO; mask + 1];
    let hash = best_hash_fn();
    let mut out = Vec::with_capacity(n);
    let mut ctx = 0usize;
    let mut pos = 0usize;
    for _ in 0..n {
        let resid = L::try_from_u64(varint::read_u64(payload, &mut pos)?)?;
        let v = resid ^ table[ctx];
        out.push(v);
        table[ctx] = v;
        ctx = (v.hash_step(hash) as usize) & mask;
    }
    if pos != payload.len() {
        return Err(Error::CorruptPayload("pred trailing bytes"));
    }
    Ok(out)
}

pub(crate) fn dfcm_encode<L: Lane>(vals: &[L], predictor_log2: u8) -> Vec<u8> {
    let mask = (1usize << predictor_log2) - 1;
    let mut table = vec![L::ZERO; mask + 1];
    let hash = best_hash_fn();
    let mut out = Vec::with_capacity(vals.len());
    let mut last = L::ZERO;
    let mut dctx = 0usize;
    for &v in vals {
        let pred = last.wrapping_add(table[dctx]);
        varint::write_u64(&mut out, (v ^ pred).to_u64());
        let delta = v.wrapping_sub(last);
        table[dctx] = delta;
        dctx = (delta.hash_step(hash) as usize) & mask;
        last = v;
    }
    out
}

pub(crate) fn dfcm_decode<L: Lane>(
    payload: &[u8],
    n: usize,
    predictor_log2: u8,
) -> Result<Vec<L>, Error> {
    let mask = (1usize << predictor_log2) - 1;
    let mut table = vec![L::ZERO; mask + 1];
    let hash = best_hash_fn();
    let mut out = Vec::with_capacity(n);
    let mut last = L::ZERO;
    let mut dctx = 0usize;
    let mut pos = 0usize;
    for _ in 0..n {
        let pred = last.wrapping_add(table[dctx]);
        let resid = L::try_from_u64(varint::read_u64(payload, &mut pos)?)?;
        let v = resid ^ pred;
        out.push(v);
        let delta = v.wrapping_sub(last);
        table[dctx] = delta;
        dctx = (delta.hash_step(hash) as usize) & mask;
        last = v;
    }
    if pos != payload.len() {
        return Err(Error::CorruptPayload("dfcm trailing bytes"));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_lanes_roundtrip() {
        let v64: Vec<u64> = (0..3000)
            .map(|i| ((i % 50) as f64 * 0.5).to_bits())
            .collect();
        assert_eq!(
            decode::<u64>(&encode(&v64, 12), v64.len(), 12).unwrap(),
            v64
        );
        assert_eq!(
            dfcm_decode::<u64>(&dfcm_encode(&v64, 12), v64.len(), 12).unwrap(),
            v64
        );
        let v32: Vec<u32> = (0..3000)
            .map(|i| ((i % 50) as f32 * 0.5).to_bits())
            .collect();
        let fcm = encode(&v32, 12);
        assert!(fcm.len() < v32.len() * 2, "repeats predict well");
        assert_eq!(decode::<u32>(&fcm, v32.len(), 12).unwrap(), v32);
        let dfcm = dfcm_encode(&v32, 12);
        assert_eq!(dfcm_decode::<u32>(&dfcm, v32.len(), 12).unwrap(), v32);
        assert_eq!(
            decode::<u32>(&encode::<u32>(&[], 10), 0, 10).unwrap(),
            Vec::<u32>::new()
        );
    }
}
