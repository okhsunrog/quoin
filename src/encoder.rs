//! Block planning and the mode competition.
//!
//! Blocks are adaptively sized (256 KiB base, grown to 1 MiB for low-entropy
//! data — see [`plan_blocks`]), matching `fc`'s quantum range. Cheap per-block
//! features ([`probe_block_features`]) then gate which mode families are worth
//! trying; each applicable mode encodes the block and the smallest output wins.
//! Blocks are independent, so with the `parallel` feature they are encoded
//! across a rayon pool.

use crate::codecs::{
    alp, alp_rd, const_block, delta_bitpack, dict, float_mult, for_bitpack, linear, lz, pcodec,
    pred, raw, rle, stride, transpose, xorz,
};
use crate::dtype::{DType, Family};
use crate::entropy::code_residuals;
use crate::format::{FRAME_HEADER_LEN, Header};
use crate::mode::Mode;
use crate::{Config, Level, Selection};

/// Float-value codecs: valid only when the column is a float family. On an
/// integer column they'd run float arithmetic on the lane bits — meaningless,
/// so they're gated out. Every other mode is type-agnostic (operates on the
/// raw `u64` lane) and applies to both families.
fn mode_applies(mode: Mode, family: Family) -> bool {
    match mode {
        Mode::Delta2 | Mode::DeltaDp | Mode::FloatMult | Mode::Alp | Mode::AlpRd => {
            family == Family::Float
        }
        _ => true,
    }
}

/// Whether a mode decodes through the entropy coder (range/rANS).
/// Fast levels gate these out to keep decode vectorized / random-access.
fn is_entropy_mode(mode: Mode) -> bool {
    matches!(
        mode,
        Mode::PredRc
            | Mode::Pred2
            | Mode::Delta2
            | Mode::DeltaDp
            | Mode::OrderedDelta
            | Mode::FloatMult
            | Mode::Lz
            | Mode::ByteTranspose
    )
}

/// Whether `mode` may compete for a column of `family` at speed `level`.
fn mode_runs(mode: Mode, family: Family, level: Level) -> bool {
    if !mode_applies(mode, family) {
        return false;
    }
    // Sequential predictors are gated tighter (High/Max) than the vectorizable
    // entropy modes (Balanced+).
    if is_predictor_mode(mode) {
        return level.allows_predictors();
    }
    level.allows_entropy() || !is_entropy_mode(mode)
}

/// The sequential, non-vectorizable predictor modes — gated to `High`/`Max`.
fn is_predictor_mode(mode: Mode) -> bool {
    matches!(
        mode,
        Mode::Pred
            | Mode::PredRc
            | Mode::Pred2
            | Mode::Delta2
            | Mode::DeltaDp
            | Mode::OrderedDelta
    )
}

/// Relative decode-cost class per mode (higher = slower). Used by the cost-aware
/// selection so a faster level can prefer a slightly larger but cheaper-to-decode
/// codec. Rough calibration vs `benches/kernels.rs`; the dominant axis is whether
/// the mode runs the sequential entropy coder.
fn decode_weight(mode: Mode) -> u64 {
    match mode {
        Mode::Raw | Mode::Const | Mode::Stride => 0,
        Mode::Xorz | Mode::ForBitpack | Mode::DeltaBitpack | Mode::Rle => 1,
        Mode::Alp | Mode::AlpRd | Mode::Dict => 2,
        Mode::Pred => 3,
        Mode::ByteTranspose | Mode::FloatMult => 6,
        Mode::OrderedDelta | Mode::Delta2 | Mode::DeltaDp | Mode::Lz | Mode::Pco => 7,
        Mode::PredRc | Mode::Pred2 => 10,
    }
}

/// Decode-cost penalty (in bytes-equivalent) added to a candidate's size:
/// `λ · weight · decoded_bytes`, scaled. `λ = 0` (level `Max`) → no penalty, so
/// selection is pure size and reproduces the historical behavior.
fn penalty(mode: Mode, lambda: u64, decoded_bytes: usize) -> usize {
    ((lambda.saturating_mul(decode_weight(mode))).saturating_mul(decoded_bytes as u64) >> 8)
        as usize
}

/// Tracks the smallest-scoring candidate (`size + decode penalty`) for a block.
struct Best {
    mode: Mode,
    payload: Vec<u8>,
    score: usize,
    lambda: u64,
    decoded_bytes: usize,
}

impl Best {
    fn new(mode: Mode, payload: Vec<u8>, lambda: u64, decoded_bytes: usize) -> Self {
        let score = payload.len() + penalty(mode, lambda, decoded_bytes);
        Best {
            mode,
            payload,
            score,
            lambda,
            decoded_bytes,
        }
    }

    fn consider(&mut self, mode: Mode, payload: Vec<u8>) {
        let score = payload.len() + penalty(mode, self.lambda, self.decoded_bytes);
        if score < self.score {
            self.mode = mode;
            self.payload = payload;
            self.score = score;
        }
    }
}

/// Default block: 32768 * 8 B = 256 KiB, matching `fc`'s base quantum. Kept
/// small so noisy/incompressible data parallelizes well.
const BASE_QUANTUM: usize = 32 * 1024;
/// Grown block for low-entropy data: 128 Ki * 8 B = 1 MiB (== `MAX_BLOCK_VALUES`).
/// Bigger blocks give LZ a larger window and entropy models more data to adapt,
/// at no parallelism cost since such blocks compress to almost nothing.
const MAX_QUANTUM: usize = crate::format::MAX_BLOCK_VALUES;

/// Plan block boundaries. With a fixed `block_size` (from [`Config::block_size`])
/// every block is exactly that many values (the last may be shorter). Otherwise
/// probe each base-quantum region and grow it to `MAX_QUANTUM` when it looks
/// low-entropy (dictionary / constant / run-heavy), else keep it at `BASE_QUANTUM`.
fn plan_blocks(vals: &[u64], block_size: Option<usize>) -> Vec<(usize, usize)> {
    let n = vals.len();
    let mut ranges = Vec::new();
    let mut start = 0;
    if let Some(bs) = block_size {
        while start < n {
            let end = (start + bs).min(n);
            ranges.push((start, end));
            start = end;
        }
        return ranges;
    }
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

/// Compress a column already lowered to its `u64` lane (see [`DType`] for the
/// mapping). `vals` holds only the **valid** values (nulls already compacted
/// out); `logical_n` is the full length including nulls; `validity`, when present,
/// is the bitmap to store. The `dtype` selects the codec family.
pub(crate) fn compress_lane(
    vals: &[u64],
    dtype: DType,
    logical_n: usize,
    validity: Option<&[u8]>,
    cfg: Config,
) -> Vec<u8> {
    let predictor_log2 = cfg.clamped_predictor_log2();

    let frames = build_frames(vals, predictor_log2, dtype, &cfg);

    let total: usize = frames.iter().map(Vec::len).sum();
    let mut out = Vec::with_capacity(crate::format::HEADER_LEN + total + 16);
    Header {
        predictor_log2,
        dtype,
        has_validity: validity.is_some(),
        n_values: logical_n as u64,
    }
    .write(&mut out);
    // Validity bitmap (if any) sits between the header and the value frames.
    if let Some(bitmap) = validity {
        let vblob = crate::validity::encode(bitmap, logical_n);
        crate::varint::write_u64(&mut out, vblob.len() as u64);
        out.extend_from_slice(&vblob);
    }
    for f in &frames {
        out.extend_from_slice(f);
    }
    out
}

#[cfg(feature = "parallel")]
fn build_frames(vals: &[u64], predictor_log2: u8, dtype: DType, cfg: &Config) -> Vec<Vec<u8>> {
    use rayon::prelude::*;
    let (sel, level) = (cfg.selection, cfg.level);
    let ranges = plan_blocks(vals, cfg.fixed_block_size());
    let run = || {
        ranges
            .par_iter()
            .map(|&(s, e)| encode_block(&vals[s..e], predictor_log2, sel, dtype, level))
            .collect::<Vec<_>>()
    };
    match cfg.threads {
        Some(n) if n > 0 => match rayon::ThreadPoolBuilder::new().num_threads(n).build() {
            Ok(pool) => pool.install(run),
            Err(_) => run(),
        },
        _ => run(), // None or Some(0): use rayon's global pool (all cores)
    }
}

#[cfg(not(feature = "parallel"))]
fn build_frames(vals: &[u64], predictor_log2: u8, dtype: DType, cfg: &Config) -> Vec<Vec<u8>> {
    plan_blocks(vals, cfg.fixed_block_size())
        .iter()
        .map(|&(s, e)| encode_block(&vals[s..e], predictor_log2, cfg.selection, dtype, cfg.level))
        .collect()
}

fn encode_block(
    block: &[u64],
    predictor_log2: u8,
    sel: Selection,
    dtype: DType,
    level: Level,
) -> Vec<u8> {
    match sel {
        Selection::Full => encode_block_full(block, predictor_log2, dtype, level),
        Selection::Sample => encode_block_sampled(block, predictor_log2, dtype, level),
    }
}

fn encode_block_full(block: &[u64], predictor_log2: u8, dtype: DType, level: Level) -> Vec<u8> {
    let family = dtype.family();
    let entropy = level.allows_entropy();
    let predictors = level.allows_predictors();
    let reduced = level.reduced_pool();
    let lambda = level.lambda();
    let allow_lz = level.allows_lz_cascade();
    let decoded_bytes = block.len() * dtype.lane_bytes();
    let raw_bytes = block.len() * 8;
    let feats = probe_block_features(block);

    // Early-out: a genuinely incompressible integer block (full value *and* delta
    // range, high-distinct, no repeats) can't beat RAW at any level — skip the
    // whole competition, including the costly predictor pass. Float families keep
    // trying (ALP/ALP-RD/predictors can win on data this probe can't read).
    if family == Family::Int && looks_incompressible(block, &feats, dtype.lane_bytes()) {
        crate::diag::record_win(Mode::Raw.id());
        return frame_bytes(Mode::Raw, block.len(), &raw::encode(block, dtype));
    }

    // RAW is the always-available baseline; every other mode must beat its score.
    let mut best = Best::new(
        Mode::Raw,
        raw::encode(block, dtype),
        lambda,
        decoded_bytes,
    );

    if let Some(p) = const_block::encode(block) {
        best.consider(Mode::Const, p);
    }
    if let Some(p) = stride::encode(block) {
        best.consider(Mode::Stride, p);
    }
    // XORZ is cheap but rarely the sole winner; reserve it for `Fast`-and-up so
    // `Fastest` stays a minimal pool.
    if !reduced {
        best.consider(Mode::Xorz, xorz::encode(block));
    }

    // The sequential PREDICTORS (FCM/DFCM, polynomial-float, 2nd-order int) live
    // at `High`/`Max` only: they give the best ratio on smooth/structured data but
    // decode as a non-vectorizable recurrence, so `Balanced` (fast decode) skips
    // them even though its entropy coder is on.

    // FCM predictor (a full hash pass per value); its payoff is the entropy-coded
    // PRED_RC.
    if predictors {
        let fcm_res = pred::encode(block, predictor_log2);
        if looks_compressible(fcm_res.len(), raw_bytes) {
            best.consider(Mode::PredRc, code_residuals(&fcm_res, lambda, allow_lz));
        }
        best.consider(Mode::Pred, fcm_res);
    }

    // DFCM predictor: wins on smooth/ramp data where FCM's exact-repeat prediction
    // fails but the deltas are predictable.
    if predictors {
        let dfcm_res = pred::dfcm_encode(block, predictor_log2);
        if looks_compressible(dfcm_res.len(), raw_bytes) {
            best.consider(Mode::Pred2, code_residuals(&dfcm_res, lambda, allow_lz));
        }
    }

    // Float polynomial predictor (Newton forward-difference extrapolation): wins
    // on smooth signals (sensors, audio, climate) where integer-delta predictors
    // fail at zero crossings. The order (linear..cubic) is chosen per block by a
    // finite-difference probe, so finely-sampled / curved data gets a sharper
    // prediction while noise backs off to a low order. Float-only.
    if family == Family::Float && predictors {
        let order = linear::select_order(block);
        let lin2_res = linear::encode(block, order);
        if looks_compressible(lin2_res.len(), raw_bytes) {
            best.consider(Mode::Delta2, code_residuals(&lin2_res, lambda, allow_lz));

            // DELTA_DP: exact float residual of the same predictor. Wins big on
            // polynomial/smooth data (constant difference); self-bails via `None`
            // when float subtraction isn't exactly invertible.
            if let Some(dp_res) = linear::dp_encode(block, order) {
                best.consider(Mode::DeltaDp, code_residuals(&dp_res, lambda, allow_lz));
            }
        }
    }

    // Second-order integer delta: wins on monotone/polynomial data (ramps,
    // parabola) where the bit-pattern second difference is near-constant.
    if predictors {
        let idelta2_res = linear::idelta2_encode(block);
        if looks_compressible(idelta2_res.len(), raw_bytes) {
            best.consider(Mode::OrderedDelta, code_residuals(&idelta2_res, lambda, allow_lz));
        }
    }

    let block_compressible = looks_compressible(best.payload.len(), raw_bytes);

    // FLOAT_MULT / ALP compress via decimal *value*, not predictability, so they
    // are tried regardless of `block_compressible` (random decimals defeat the
    // predictors but pack fine as scaled integers). Both self-bail cheaply on
    // non-decimal data. Float-only: they reinterpret the lane as `f64`. ALP is
    // pure bit-packing (no entropy coder), so it survives the fast levels.
    if family == Family::Float && !reduced {
        // FLOAT_MULT now self-codes its `k` stream (bit-pack or entropy), so it
        // competes at the fast levels too, not only when the entropy coder is on.
        if let Some(p) = float_mult::encode(block, entropy, lambda, allow_lz) {
            best.consider(Mode::FloatMult, p);
        }
        if let Some(p) = alp::encode(block) {
            best.consider(Mode::Alp, p);
        }
        // ALP-RD: real-double split-dictionary. Pure bit-packing (no entropy),
        // so it survives the fast levels; self-bails on non-real-double data.
        if let Some(p) = alp_rd::encode(block) {
            best.consider(Mode::AlpRd, p);
        }
    }

    // LZ: only worth its match finder + entropy pass on low-distinct or
    // repetitive data (dictionaries, quantized levels, cent-rounded prices).
    // Skipping it on high-distinct noisy floats is where most of the encode
    // speedup comes from — LZ finds no matches there and always loses.
    // At `Max` (λ = 0, best-ratio and encode-cost-tolerant) run LZ on any
    // compressible block — it wins on repeated multi-value *sequences* even when
    // value cardinality is high (medical-billing-style data) where the
    // distinct/repeats gate would skip it. Faster levels keep the cheap gate
    // since LZ's match-finding is expensive and loses on non-repetitive data.
    let lz_worth = lambda == 0 || feats.distinct_low || feats.looks_like_repeats;
    if entropy && block_compressible && lz_worth {
        best.consider(Mode::Lz, code_residuals(&lz::encode(block), lambda, allow_lz));
    }

    // Byte-plane transpose: helps when a byte position is low-entropy across
    // values (similar-magnitude floats share sign/exponent bytes). Skip on
    // full-exponent-range data (random, polynomial) where it can't win and the
    // entropy pass over the transposed block would be wasted.
    if entropy
        && block_compressible
        && (feats.exp_range <= TRANSPOSE_EXP_LIMIT || feats.looks_like_repeats)
    {
        best.consider(
            Mode::ByteTranspose,
            code_residuals(&transpose::encode(block), lambda, allow_lz),
        );
    }

    // Integer-column codecs: FoR+bitpack (bounded-range) and delta+bitpack
    // (monotonic / regularly-stepped). Pure bit-packing — random-access and
    // available at every level. Rarely win on f64 bit patterns but cheap to try.
    // FoR+bitpack (bounded-range) and delta+bitpack (monotonic / regularly
    // stepped) are the cheap random-access workhorses: always tried, at every
    // level. They self-bail to ~raw when they can't help, so `Best` just keeps
    // the incumbent — they must NOT hang off `block_compressible` (which is set
    // by the entropy modes and so is false at `Fastest`, where these are the
    // only codecs that can win an integer column).
    best.consider(Mode::ForBitpack, for_bitpack::encode(block, dtype.signed()));
    best.consider(Mode::DeltaBitpack, delta_bitpack::encode(block));
    // Dictionary (scattered low-cardinality) and RLE (grouped runs); both
    // self-bail on high-distinct / run-poor data. Pure bit-packing, but a hash
    // pass / run scan, so they sit out of the minimal `Fastest` pool.
    if block_compressible && !reduced {
        if let Some(p) = dict::encode(block, entropy, lambda, allow_lz) {
            best.consider(Mode::Dict, p);
        }
        if let Some(p) = rle::encode(block) {
            best.consider(Mode::Rle, p);
        }
    }

    // pco (vendored pcodec): a heavyweight numeric backend — latent decomposition
    // (auto delta order, int/float multiples) + bin-packing + interleaved ANS. It
    // captures smooth/structured numeric columns that quoin's own transforms only
    // partially model. Gated to `High`/`Max`: its encode searches hard and its
    // decode, while vectorized, costs more than the cheap bit-packers. Self-bails
    // (`None`) on empty blocks or internal errors, and competes on pure size
    // (λ = 0 at these levels), so it only wins when it is strictly smaller.
    if level.allows_pco()
        && let Some(p) = pcodec::encode(block, dtype, level.pco_level())
    {
        best.consider(Mode::Pco, p);
    }

    crate::diag::record_win(best.mode.id());
    frame_bytes(best.mode, block.len(), &best.payload)
}

// ---------------------------------------------------------------------------
// Sampling-based selection (Selection::Sample): estimate every mode on a small
// stratified sample, then encode only the winner in full. The BtrBlocks/Vortex
// approach — much cheaper than encoding every mode in full.
// ---------------------------------------------------------------------------

/// Modes ranked by sample estimate. CONST/STRIDE/RAW (global structure) and LZ
/// (long-range repeats) are handled on the full block instead — a non-contiguous
/// sample can't see that structure. The rest estimate well on a sample.
const SAMPLE_MODES: [Mode; 15] = [
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
    Mode::DeltaBitpack,
    Mode::Alp,
    Mode::AlpRd,
    Mode::Dict,
    Mode::Rle,
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
fn encode_mode(
    mode: Mode,
    block: &[u64],
    predictor_log2: u8,
    dtype: DType,
    level: Level,
) -> Option<Vec<u8>> {
    let lambda = level.lambda();
    let allow_lz = level.allows_lz_cascade();
    let entropy = level.allows_entropy();
    match mode {
        Mode::Raw => Some(raw::encode(block, dtype)),
        Mode::Const => const_block::encode(block),
        Mode::Stride => stride::encode(block),
        Mode::Xorz => Some(xorz::encode(block)),
        Mode::Pred => Some(pred::encode(block, predictor_log2)),
        Mode::PredRc => Some(code_residuals(&pred::encode(block, predictor_log2), lambda, allow_lz)),
        Mode::Pred2 => Some(code_residuals(
            &pred::dfcm_encode(block, predictor_log2),
            lambda,
            allow_lz,
        )),
        Mode::Delta2 => Some(code_residuals(&linear::encode(block, linear::select_order(block)), lambda, allow_lz)),
        Mode::DeltaDp => {
            linear::dp_encode(block, linear::select_order(block)).map(|r| code_residuals(&r, lambda, allow_lz))
        }
        Mode::OrderedDelta => Some(code_residuals(&linear::idelta2_encode(block), lambda, allow_lz)),
        Mode::FloatMult => float_mult::encode(block, entropy, lambda, allow_lz),
        Mode::Lz => Some(code_residuals(&lz::encode(block), lambda, allow_lz)),
        Mode::ByteTranspose => Some(code_residuals(&transpose::encode(block), lambda, allow_lz)),
        Mode::ForBitpack => Some(for_bitpack::encode(block, dtype.signed())),
        Mode::Alp => alp::encode(block),
        Mode::AlpRd => alp_rd::encode(block),
        Mode::Dict => dict::encode(block, entropy, lambda, allow_lz),
        Mode::Rle => rle::encode(block),
        Mode::DeltaBitpack => Some(delta_bitpack::encode(block)),
        Mode::Pco => pcodec::encode(block, dtype, level.pco_level()),
    }
}

fn encode_block_sampled(block: &[u64], predictor_log2: u8, dtype: DType, level: Level) -> Vec<u8> {
    let family = dtype.family();
    let decoded_bytes = block.len() * dtype.lane_bytes();
    let feats = probe_block_features(block);
    let mut best = Best::new(
        Mode::Raw,
        raw::encode(block, dtype),
        level.lambda(),
        decoded_bytes,
    );

    let consider_full = |m: Mode, best: &mut Best| {
        if let Some(p) = encode_mode(m, block, predictor_log2, dtype, level) {
            best.consider(m, p);
        }
    };
    // Exact O(n) global-structure modes — always on the full block.
    consider_full(Mode::Const, &mut best);
    consider_full(Mode::Stride, &mut best);
    // LZ's long-range repeats are invisible to a sample, so run it on the full
    // block when the cheap features say the data is dictionary-like.
    if mode_runs(Mode::Lz, family, level) && (feats.distinct_low || feats.looks_like_repeats) {
        consider_full(Mode::Lz, &mut best);
    }
    // pco runs on the full block too: it has fixed chunk framing and auto-detects
    // its latent decomposition over the whole sequence, so a tiny stratified
    // sample would mis-estimate it. Gated to High/Max like in the full path.
    if level.allows_pco() {
        consider_full(Mode::Pco, &mut best);
    }

    // Rank the remaining modes by their estimate on a small sample, then encode
    // only the winner in full and let it challenge the structural best.
    let sample = build_sample(block);
    let mut win = None;
    let mut win_est = usize::MAX;
    for &m in SAMPLE_MODES
        .iter()
        .filter(|&&m| mode_runs(m, family, level))
    {
        if let Some(p) = encode_mode(m, &sample, SAMPLE_PLOG2, dtype, level)
            && p.len() < win_est
        {
            win_est = p.len();
            win = Some(m);
        }
    }
    if let Some(m) = win
        && let Some(p) = encode_mode(m, block, predictor_log2, dtype, level)
    {
        best.consider(m, p);
    }

    crate::diag::record_win(best.mode.id());
    frame_bytes(best.mode, block.len(), &best.payload)
}

/// Cheap gate for the expensive range-coded modes: only bother when the
/// predictor already shrank the stream below ~95% of raw. Skips the slow
/// arithmetic coder on essentially-incompressible blocks (e.g. random data),
/// where it can't help anyway.
fn looks_compressible(residual_bytes: usize, raw_bytes: usize) -> bool {
    residual_bytes.saturating_mul(20) < raw_bytes.saturating_mul(19)
}

/// Cheap, conservative incompressibility probe for integer blocks: true only
/// when *both* the value range and the consecutive-delta range nearly fill the
/// lane width (so neither frame-of-reference nor delta can pack it), the block
/// is high-distinct, and not run-heavy. That's genuine noise — RAW wins at every
/// level, so the whole competition (including the predictor pass) is skipped.
/// Bounded ranges (FoR), smooth/monotone data (small deltas), and low-cardinality
/// data (dict/LZ) all fail this test and proceed normally.
fn looks_incompressible(block: &[u64], feats: &BlockFeatures, lane_bytes: usize) -> bool {
    if feats.distinct_low || feats.looks_like_repeats || block.len() < 256 {
        return false;
    }
    let n = block.len();
    let lane_bits = (lane_bytes * 8) as u32;
    let thr = lane_bits.saturating_sub(4);
    let step = (n / 2048).max(1);
    let mut vmin = block[0];
    let mut vmax = block[0];
    let mut dmax_zz = 0u64;
    let mut i = step.max(1);
    while i < n {
        let v = block[i];
        vmin = vmin.min(v);
        vmax = vmax.max(v);
        let d = v.wrapping_sub(block[i - 1]);
        let zz = (d << 1) ^ ((d as i64 >> 63) as u64); // zigzag |delta|
        dmax_zz = dmax_zz.max(zz);
        i += step;
    }
    let value_width = 64 - (vmax - vmin).leading_zeros();
    let delta_width = if dmax_zz == 0 {
        0
    } else {
        64 - dmax_zz.leading_zeros()
    };
    value_width >= thr && delta_width >= thr
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
