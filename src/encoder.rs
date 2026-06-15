//! Block planning and the mode competition.
//!
//! For now blocks are a fixed quantum of values (the original `fc` uses an
//! adaptive 256 KiB–1 MiB quantum; adaptive sizing is a roadmap item). Each
//! block is encoded by every applicable mode and the smallest output wins —
//! the core idea that buys `fc` its ratio. Threading is also a roadmap item;
//! this pass is single-threaded.

use crate::Config;
use crate::codecs::{const_block, linear, pred, raw, stride, xorz};
use crate::entropy::code_residuals;
use crate::format::{Header, FRAME_HEADER_LEN};
use crate::mode::Mode;

/// Values per block. 32768 * 8 B = 256 KiB, matching `fc`'s default quantum.
const QUANTUM_VALUES: usize = 32 * 1024;

pub(crate) fn compress(src: &[f64], cfg: Config) -> Vec<u8> {
    let predictor_log2 = cfg.clamped_predictor_log2();
    let vals: Vec<u64> = src.iter().map(|f| f.to_bits()).collect();

    let mut out = Vec::with_capacity(src.len() * 8 / 2 + 64);
    Header { predictor_log2, n_values: vals.len() as u64 }.write(&mut out);

    for block in vals.chunks(QUANTUM_VALUES) {
        encode_block(block, predictor_log2, &mut out);
    }
    out
}

fn encode_block(block: &[u64], predictor_log2: u8, out: &mut Vec<u8>) {
    // RAW is the always-available baseline; every other mode must beat it.
    let mut best_mode = Mode::Raw;
    let mut best_payload = raw::encode(block);

    let consider = |mode: Mode, payload: Vec<u8>, best_mode: &mut Mode, best: &mut Vec<u8>| {
        if payload.len() < best.len() {
            *best_mode = mode;
            *best = payload;
        }
    };

    if let Some(p) = const_block::encode(block) {
        consider(Mode::Const, p, &mut best_mode, &mut best_payload);
    }
    if let Some(p) = stride::encode(block) {
        consider(Mode::Stride, p, &mut best_mode, &mut best_payload);
    }
    consider(Mode::Xorz, xorz::encode(block), &mut best_mode, &mut best_payload);

    let raw_bytes = block.len() * 8;

    // FCM predictor: store residuals verbatim (PRED, cheap) and, when the
    // residual stream looks compressible, also range-code them (PRED_RC).
    let fcm_res = pred::encode(block, predictor_log2);
    if looks_compressible(fcm_res.len(), raw_bytes) {
        consider(Mode::PredRc, code_residuals(&fcm_res), &mut best_mode, &mut best_payload);
    }
    consider(Mode::Pred, fcm_res, &mut best_mode, &mut best_payload);

    // DFCM predictor (entropy-coded only): wins on smooth/ramp data where FCM's
    // exact-repeat prediction fails but the deltas are predictable.
    let dfcm_res = pred::dfcm_encode(block, predictor_log2);
    if looks_compressible(dfcm_res.len(), raw_bytes) {
        consider(Mode::Pred2, code_residuals(&dfcm_res), &mut best_mode, &mut best_payload);
    }

    // Second-order float-linear predictor: wins on smooth oscillating signals
    // (sine, audio, climate) where integer-delta predictors fail at zero crossings.
    let lin2_res = linear::encode(block);
    if looks_compressible(lin2_res.len(), raw_bytes) {
        consider(Mode::Delta2, code_residuals(&lin2_res), &mut best_mode, &mut best_payload);
    }

    write_frame(best_mode, block.len(), &best_payload, out);
}

/// Cheap gate for the expensive range-coded modes: only bother when the
/// predictor already shrank the stream below ~95% of raw. Skips the slow
/// arithmetic coder on essentially-incompressible blocks (e.g. random data),
/// where it can't help anyway.
fn looks_compressible(residual_bytes: usize, raw_bytes: usize) -> bool {
    residual_bytes.saturating_mul(20) < raw_bytes.saturating_mul(19)
}

fn write_frame(mode: Mode, n_values: usize, payload: &[u8], out: &mut Vec<u8>) {
    out.reserve(FRAME_HEADER_LEN + payload.len());
    out.push(mode.id());
    out.extend_from_slice(&(n_values as u32).to_le_bytes());
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
}
