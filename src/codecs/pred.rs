//! PRED: finite-context-method (FCM) predictor.
//!
//! A hash of the most recent value indexes a table holding the last value seen
//! in that context; that becomes the prediction. We emit the XOR residual
//! (value ^ prediction) as a LEB128 varint — small when the prediction is
//! close, which is the common case for structured floating-point streams.
//!
//! This is the heart of the original `fc` and the first real consumer of the
//! [`crate::hash`] CRC32C kernel. Encode and decode evolve the table
//! identically, so the residual stream is all that needs storing.

use crate::error::Error;
use crate::hash::{HASH_SEED, best_hash_fn};
use crate::varint;

pub(crate) fn encode(vals: &[u64], predictor_log2: u8) -> Vec<u8> {
    let mask = (1usize << predictor_log2) - 1;
    let mut table = vec![0u64; mask + 1];
    let hash = best_hash_fn();
    let mut out = Vec::with_capacity(vals.len());
    let mut ctx = 0usize;
    for &v in vals {
        let pred = table[ctx];
        varint::write_u64(&mut out, v ^ pred);
        table[ctx] = v;
        ctx = (hash(HASH_SEED, v) as usize) & mask;
    }
    out
}

pub(crate) fn decode(payload: &[u8], n: usize, predictor_log2: u8) -> Result<Vec<u64>, Error> {
    let mask = (1usize << predictor_log2) - 1;
    let mut table = vec![0u64; mask + 1];
    let hash = best_hash_fn();
    let mut out = Vec::with_capacity(n);
    let mut ctx = 0usize;
    let mut pos = 0usize;
    for _ in 0..n {
        let resid = varint::read_u64(payload, &mut pos)?;
        let v = resid ^ table[ctx];
        out.push(v);
        table[ctx] = v;
        ctx = (hash(HASH_SEED, v) as usize) & mask;
    }
    if pos != payload.len() {
        return Err(Error::CorruptPayload("pred trailing bytes"));
    }
    Ok(out)
}
