//! Block planning and the mode competition.
//!
//! Blocks are adaptively sized (256 KiB base, grown to 1 MiB for low-entropy
//! data — see [`plan_blocks`]), matching `fc`'s quantum range. Cheap per-block
//! features ([`probe_block_features`]) then gate which mode families are worth
//! trying; each applicable mode encodes the block and the smallest output wins.
//! Blocks are independent, so with the `parallel` feature they are encoded
//! across a rayon pool.

use crate::codecs::{
    alp, const_block, float_mult, for_bitpack, linear, lz, pred, raw, stride, transpose, xorz,
};
use crate::entropy::code_residuals;
use crate::format::{FRAME_HEADER_LEN, Header};
use crate::mode::Mode;
use crate::{Config, Selection};

/// Default block: 32768 * 8 B = 256 KiB, matching `fc`'s base quantum. Kept
/// small so noisy/incompressible data parallelizes well.
const BASE_QUANTUM: usize = 32 * 1024;
/// Grown block for low-entropy data: 128 Ki * 8 B = 1 MiB (== `MAX_BLOCK_VALUES`).
/// Bigger blocks give LZ a larger window and entropy models more data to adapt,
/// at no parallelism cost since such blocks compress to almost nothing.
const MAX_QUANTUM: usize = crate::format::MAX_BLOCK_VALUES;

/// Plan block boundaries: probe each base-quantum region and grow it to
/// `MAX_QUANTUM` when it looks low-entropy (dictionary / constant / run-heavy),
/// otherwise keep it at `BASE_QUANTUM`.
fn plan_blocks(vals: &[u64]) -> Vec<(usize, usize)> {
    let n = vals.len();
    let mut ranges = Vec::new();
    let mut start = 0;
    while start < n {
        let base_end = (start + BASE_QUANTUM).min(n);
        let feats = probe_block_features(&vals[start..base_end]);
        let end = if feats.distinct_low || feats.looks_like_repeats {
            (start + MAX_QUANTUM).min(n)
        } else {
            base_end
        };
        ranges.push((start, end));
        start = end;
    }
    ranges
}

pub(crate) fn compress(src: &[f64], cfg: Config) -> Vec<u8> {
    let predictor_log2 = cfg.clamped_predictor_log2();
    let vals: Vec<u64> = src.iter().map(|f| f.to_bits()).collect();

    let frames = build_frames(&vals, predictor_log2, cfg.threads, cfg.selection);

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
fn build_frames(
    vals: &[u64],
    predictor_log2: u8,
    threads: Option<usize>,
    sel: Selection,
) -> Vec<Vec<u8>> {
    use rayon::prelude::*;
    let ranges = plan_blocks(vals);
    let run = || {
        ranges
            .par_iter()
            .map(|&(s, e)| encode_block(&vals[s..e], predictor_log2, sel))
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
fn build_frames(
    vals: &[u64],
    predictor_log2: u8,
    _threads: Option<usize>,
    sel: Selection,
) -> Vec<Vec<u8>> {
    plan_blocks(vals)
        .iter()
        .map(|&(s, e)| encode_block(&vals[s..e], predictor_log2, sel))
        .collect()
}

fn encode_block(block: &[u64], predictor_log2: u8, sel: Selection) -> Vec<u8> {
    match sel {
        Selection::Full => encode_block_full(block, predictor_log2),
        Selection::Sample => encode_block_sampled(block, predictor_log2),
    }
}

fn encode_block_full(block: &[u64], predictor_log2: u8) -> Vec<u8> {
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

    // FLOAT_MULT / ALP compress via decimal *value*, not predictability, so they
    // are tried regardless of `block_compressible` (random decimals defeat the
    // predictors but pack fine as scaled integers). Both self-bail cheaply on
    // non-decimal data.
    if let Some(fm_res) = float_mult::encode(block) {
        consider(
            Mode::FloatMult,
            code_residuals(&fm_res),
            &mut best_mode,
            &mut best_payload,
        );
    }
    if let Some(p) = alp::encode(block) {
        consider(Mode::Alp, p, &mut best_mode, &mut best_payload);
    }

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

    // FOR + bit-packing: the integer-column codec. Rarely wins on f64 bit
    // patterns (not frame-of-reference-friendly) but cheap to try, and the
    // substrate the typed-columnar path is built on.
    if block_compressible {
        consider(
            Mode::ForBitpack,
            for_bitpack::encode(block),
            &mut best_mode,
            &mut best_payload,
        );
    }

    crate::diag::record_win(best_mode.id());
    frame_bytes(best_mode, block.len(), &best_payload)
}

// ---------------------------------------------------------------------------
// Sampling-based selection (Selection::Sample): estimate every mode on a small
// stratified sample, then encode only the winner in full. The BtrBlocks/Vortex
// approach — much cheaper than encoding every mode in full.
// ---------------------------------------------------------------------------

/// Modes ranked by sample estimate. CONST/STRIDE/RAW (global structure) and LZ
/// (long-range repeats) are handled on the full block instead — a non-contiguous
/// sample can't see that structure. The rest estimate well on a sample.
const SAMPLE_MODES: [Mode; 11] = [
    Mode::Xorz,
    Mode::Pred,
    Mode::PredRc,
    Mode::Pred2,
    Mode::Delta2,
    Mode::DeltaDp,
    Mode::OrderedDelta,
    Mode::FloatMult,
    Mode::ByteTranspose,
    Mode::ForBitpack,
    Mode::Alp,
];

const SAMPLE_RUNS: usize = 8;
const SAMPLE_RUN_LEN: usize = 64;
/// Small predictor table for sample estimates — the sample is tiny, so a 1 MiB
/// table would dominate the cost. The sample is never decoded, so the table
/// size needn't match the full encode.
const SAMPLE_PLOG2: u8 = 10;

/// Stratified sample: `SAMPLE_RUNS` contiguous runs spread across the block, so
/// local structure (deltas, repeats) survives within each run.
fn build_sample(block: &[u64]) -> Vec<u64> {
    let n = block.len();
    let total = SAMPLE_RUNS * SAMPLE_RUN_LEN;
    if n <= total {
        return block.to_vec();
    }
    let mut s = Vec::with_capacity(total);
    for r in 0..SAMPLE_RUNS {
        let start = r * (n - SAMPLE_RUN_LEN) / (SAMPLE_RUNS - 1);
        s.extend_from_slice(&block[start..start + SAMPLE_RUN_LEN]);
    }
    s
}

/// Produce a mode's payload for `block`, or `None` if the mode doesn't apply.
/// Centralized encode dispatch used by the sample path (and to encode the
/// winner in full).
fn encode_mode(mode: Mode, block: &[u64], predictor_log2: u8) -> Option<Vec<u8>> {
    match mode {
        Mode::Raw => Some(raw::encode(block)),
        Mode::Const => const_block::encode(block),
        Mode::Stride => stride::encode(block),
        Mode::Xorz => Some(xorz::encode(block)),
        Mode::Pred => Some(pred::encode(block, predictor_log2)),
        Mode::PredRc => Some(code_residuals(&pred::encode(block, predictor_log2))),
        Mode::Pred2 => Some(code_residuals(&pred::dfcm_encode(block, predictor_log2))),
        Mode::Delta2 => Some(code_residuals(&linear::encode(block))),
        Mode::DeltaDp => linear::dp_encode(block).map(|r| code_residuals(&r)),
        Mode::OrderedDelta => Some(code_residuals(&linear::idelta2_encode(block))),
        Mode::FloatMult => float_mult::encode(block).map(|r| code_residuals(&r)),
        Mode::Lz => Some(code_residuals(&lz::encode(block))),
        Mode::ByteTranspose => Some(code_residuals(&transpose::encode(block))),
        Mode::ForBitpack => Some(for_bitpack::encode(block)),
        Mode::Alp => alp::encode(block),
    }
}

fn encode_block_sampled(block: &[u64], predictor_log2: u8) -> Vec<u8> {
    let feats = probe_block_features(block);
    let mut best_mode = Mode::Raw;
    let mut best_payload = raw::encode(block);

    let consider_full = |m: Mode, best_mode: &mut Mode, best: &mut Vec<u8>| {
        if let Some(p) = encode_mode(m, block, predictor_log2)
            && p.len() < best.len()
        {
            *best_mode = m;
            *best = p;
        }
    };
    // Exact O(n) global-structure modes — always on the full block.
    consider_full(Mode::Const, &mut best_mode, &mut best_payload);
    consider_full(Mode::Stride, &mut best_mode, &mut best_payload);
    // LZ's long-range repeats are invisible to a sample, so run it on the full
    // block when the cheap features say the data is dictionary-like.
    if feats.distinct_low || feats.looks_like_repeats {
        consider_full(Mode::Lz, &mut best_mode, &mut best_payload);
    }

    // Rank the remaining modes by their estimate on a small sample, then encode
    // only the winner in full and let it challenge the structural best.
    let sample = build_sample(block);
    let mut win = None;
    let mut win_est = usize::MAX;
    for &m in &SAMPLE_MODES {
        if let Some(p) = encode_mode(m, &sample, SAMPLE_PLOG2)
            && p.len() < win_est
        {
            win_est = p.len();
            win = Some(m);
        }
    }
    if let Some(m) = win
        && let Some(p) = encode_mode(m, block, predictor_log2)
        && p.len() < best_payload.len()
    {
        best_mode = m;
        best_payload = p;
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
