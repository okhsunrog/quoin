//! Block planning and the mode competition.
//!
//! For now blocks are a fixed quantum of values (the original `fc` uses an
//! adaptive 256 KiB–1 MiB quantum; adaptive sizing is a roadmap item). Each
//! block is encoded by every applicable mode and the smallest output wins —
//! the core idea that buys `fc` its ratio. Threading is also a roadmap item;
//! this pass is single-threaded.

use crate::Config;
use crate::codecs::{const_block, pred, raw, stride, xorz};
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
    consider(
        Mode::Pred,
        pred::encode(block, predictor_log2),
        &mut best_mode,
        &mut best_payload,
    );

    write_frame(best_mode, block.len(), &best_payload, out);
}

fn write_frame(mode: Mode, n_values: usize, payload: &[u8], out: &mut Vec<u8>) {
    out.reserve(FRAME_HEADER_LEN + payload.len());
    out.push(mode.id());
    out.extend_from_slice(&(n_values as u32).to_le_bytes());
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
}
