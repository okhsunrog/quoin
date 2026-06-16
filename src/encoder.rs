//! Block planning and the mode competition.
//!
//! Blocks are a fixed quantum of values (the original `fc` uses an adaptive
//! 256 KiB–1 MiB quantum; adaptive sizing is a roadmap item). Each block is
//! encoded by every applicable mode and the smallest output wins — the core
//! idea that buys `fc` its ratio. Blocks are independent, so with the
//! `parallel` feature they are encoded across a rayon pool.

use crate::Config;
use crate::codecs::{const_block, linear, lz, pred, raw, stride, transpose, xorz};
use crate::entropy::code_residuals;
use crate::format::{FRAME_HEADER_LEN, Header};
use crate::mode::Mode;

/// Values per block. 32768 * 8 B = 256 KiB, matching `fc`'s default quantum.
/// Must not exceed the decoder's `MAX_BLOCK_VALUES`, which is defined as this.
const QUANTUM_VALUES: usize = crate::format::MAX_BLOCK_VALUES;

pub(crate) fn compress(src: &[f64], cfg: Config) -> Vec<u8> {
    let predictor_log2 = cfg.clamped_predictor_log2();
    let vals: Vec<u64> = src.iter().map(|f| f.to_bits()).collect();

    let frames = build_frames(&vals, predictor_log2, cfg.threads);

    let total: usize = frames.iter().map(Vec::len).sum();
    let mut out = Vec::with_capacity(crate::format::HEADER_LEN + total);
    Header {
        predictor_log2,
        n_values: vals.len() as u64,
    }
    .write(&mut out);
    for f in &frames {
        out.extend_from_slice(f);
    }
    out
}

#[cfg(feature = "parallel")]
fn build_frames(vals: &[u64], predictor_log2: u8, threads: Option<usize>) -> Vec<Vec<u8>> {
    use rayon::prelude::*;
    let run = || {
        vals.par_chunks(QUANTUM_VALUES)
            .map(|block| encode_block(block, predictor_log2))
            .collect::<Vec<_>>()
    };
    match threads {
        Some(n) if n > 0 => match rayon::ThreadPoolBuilder::new().num_threads(n).build() {
            Ok(pool) => pool.install(run),
            Err(_) => run(),
        },
        _ => run(), // None or Some(0): use rayon's global pool (all cores)
    }
}

#[cfg(not(feature = "parallel"))]
fn build_frames(vals: &[u64], predictor_log2: u8, _threads: Option<usize>) -> Vec<Vec<u8>> {
    vals.chunks(QUANTUM_VALUES)
        .map(|block| encode_block(block, predictor_log2))
        .collect()
}

fn encode_block(block: &[u64], predictor_log2: u8) -> Vec<u8> {
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
    consider(
        Mode::Xorz,
        xorz::encode(block),
        &mut best_mode,
        &mut best_payload,
    );

    let raw_bytes = block.len() * 8;
    let feats = probe_block_features(block);

    // FCM predictor: store residuals verbatim (PRED, cheap) and, when the
    // residual stream looks compressible, also range-code them (PRED_RC).
    let fcm_res = pred::encode(block, predictor_log2);
    if looks_compressible(fcm_res.len(), raw_bytes) {
        consider(
            Mode::PredRc,
            code_residuals(&fcm_res),
            &mut best_mode,
            &mut best_payload,
        );
    }
    consider(Mode::Pred, fcm_res, &mut best_mode, &mut best_payload);

    // DFCM predictor (entropy-coded only): wins on smooth/ramp data where FCM's
    // exact-repeat prediction fails but the deltas are predictable.
    let dfcm_res = pred::dfcm_encode(block, predictor_log2);
    if looks_compressible(dfcm_res.len(), raw_bytes) {
        consider(
            Mode::Pred2,
            code_residuals(&dfcm_res),
            &mut best_mode,
            &mut best_payload,
        );
    }

    // Second-order float-linear predictor: wins on smooth oscillating signals
    // (sine, audio, climate) where integer-delta predictors fail at zero crossings.
    let lin2_res = linear::encode(block);
    if looks_compressible(lin2_res.len(), raw_bytes) {
        consider(
            Mode::Delta2,
            code_residuals(&lin2_res),
            &mut best_mode,
            &mut best_payload,
        );

        // DELTA_DP: exact float residual of the same predictor. Wins big on
        // polynomial/smooth data (constant second difference); self-bails via
        // `None` when float subtraction isn't exactly invertible.
        if let Some(dp_res) = linear::dp_encode(block) {
            consider(
                Mode::DeltaDp,
                code_residuals(&dp_res),
                &mut best_mode,
                &mut best_payload,
            );
        }
    }

    // Second-order integer delta: wins on monotone/polynomial data (ramps,
    // parabola) where the bit-pattern second difference is near-constant.
    let idelta2_res = linear::idelta2_encode(block);
    if looks_compressible(idelta2_res.len(), raw_bytes) {
        consider(
            Mode::OrderedDelta,
            code_residuals(&idelta2_res),
            &mut best_mode,
            &mut best_payload,
        );
    }

    let block_compressible = looks_compressible(best_payload.len(), raw_bytes);

    // LZ: only worth its match finder + entropy pass on low-distinct or
    // repetitive data (dictionaries, quantized levels, cent-rounded prices).
    // Skipping it on high-distinct noisy floats is where most of the encode
    // speedup comes from — LZ finds no matches there and always loses.
    if block_compressible && (feats.distinct_low || feats.looks_like_repeats) {
        let lz_res = lz::encode(block);
        consider(
            Mode::Lz,
            code_residuals(&lz_res),
            &mut best_mode,
            &mut best_payload,
        );
    }

    // Byte-plane transpose: helps when a byte position is low-entropy across
    // values (similar-magnitude floats share sign/exponent bytes). Skip on
    // full-exponent-range data (random, polynomial) where it can't win and the
    // entropy pass over the transposed block would be wasted.
    if block_compressible && (feats.exp_range <= TRANSPOSE_EXP_LIMIT || feats.looks_like_repeats) {
        let soa = transpose::encode(block);
        consider(
            Mode::ByteTranspose,
            code_residuals(&soa),
            &mut best_mode,
            &mut best_payload,
        );
    }

    crate::diag::record_win(best_mode.id());
    frame_bytes(best_mode, block.len(), &best_payload)
}

/// Cheap gate for the expensive range-coded modes: only bother when the
/// predictor already shrank the stream below ~95% of raw. Skips the slow
/// arithmetic coder on essentially-incompressible blocks (e.g. random data),
/// where it can't help anyway.
fn looks_compressible(residual_bytes: usize, raw_bytes: usize) -> bool {
    residual_bytes.saturating_mul(20) < raw_bytes.saturating_mul(19)
}

/// Block below this exponent-field spread is "similar magnitude" — byte
/// transpose can find low-entropy planes. Above it (random / polynomial data
/// spanning many exponents) the transpose can't win.
const TRANSPOSE_EXP_LIMIT: u32 = 64;
/// Estimated distinct-value count below which LZ / dictionary modes are worth
/// trying.
const DISTINCT_LOW: u32 = 2048;

/// Cheap per-block features (one+ passes over the block) used to decide which
/// expensive mode families to try, mirroring `fc`'s `exp_range` /
/// `distinct_count` / repetition gates.
struct BlockFeatures {
    /// Spread of the IEEE-754 exponent field (max − min) across the block.
    exp_range: u32,
    /// Estimated distinct values (sampled) below [`DISTINCT_LOW`].
    distinct_low: bool,
    /// Most consecutive pairs are equal (run-heavy / RLE-friendly).
    looks_like_repeats: bool,
}

fn probe_block_features(block: &[u64]) -> BlockFeatures {
    let n = block.len();
    if n == 0 {
        return BlockFeatures {
            exp_range: 0,
            distinct_low: true,
            looks_like_repeats: false,
        };
    }

    let mut min_exp = u32::MAX;
    let mut max_exp = 0u32;
    let mut consec_eq = 0u32;
    let mut prev = block[0];
    for &v in block {
        let exp = ((v >> 52) & 0x7FF) as u32;
        min_exp = min_exp.min(exp);
        max_exp = max_exp.max(exp);
        consec_eq += u32::from(v == prev);
        prev = v;
    }
    // `consec_eq` counts the first element against itself; harmless for the ratio.

    // Distinct estimate over a sample, via a 14-bit bucket bitset (2 KiB stack).
    let mut seen = [0u64; 256];
    let mut distinct = 0u32;
    let stride = (n / 4096).max(1);
    let mut k = 0;
    while k < n {
        let h = block[k].wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let bucket = ((h >> 50) & 0x3FFF) as usize;
        let word = bucket >> 6;
        let bit = 1u64 << (bucket & 63);
        if seen[word] & bit == 0 {
            seen[word] |= bit;
            distinct += 1;
        }
        k += stride;
    }

    BlockFeatures {
        exp_range: max_exp - min_exp,
        distinct_low: distinct < DISTINCT_LOW,
        looks_like_repeats: consec_eq.saturating_mul(2) > n as u32,
    }
}

fn frame_bytes(mode: Mode, n_values: usize, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(FRAME_HEADER_LEN + payload.len());
    out.push(mode.id());
    out.extend_from_slice(&(n_values as u32).to_le_bytes());
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    out
}
