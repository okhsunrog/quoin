//! `quoin` — a lossless compressor for typed columns of numbers (`f64`, `i64`,
//! `u64` today; see [`DType`]).
//!
//! It runs a per-block competition between specialized codecs and emits the
//! smallest result. The engine works on a single `u64` lane internally; the
//! column [`DType`] selects the codec family (integer vs float) and is recorded
//! in the stream so the decoder restores the original type. Use
//! [`compress_column`]/[`decompress_column`] for the typed path, or
//! [`compress`]/[`decompress`] for the `f64` convenience case.
//!
//! It began as a from-scratch Rust port of the `fc` floating-point compressor
//! (Apache-2.0, © Praveen Vaddadi). The original is a single ~6k-line C file
//! that runs a competition between ~50 specialized codecs per block and emits
//! the smallest result. This crate rebuilds that design incrementally with a
//! clean, safe API and a deliberate SIMD strategy:
//!
//! * **Hot, irregular kernels** (the FCM/DFCM predictor hash, gather) use raw
//!   `core::arch` intrinsics behind runtime feature detection — see [`hash`].
//! * **Lane-wise transforms** use autovectorized scalar Rust dispatched per
//!   CPU via the `multiversion` crate — see [`transform`].
//! * **Everything else** is plain scalar Rust the compiler can autovectorize.
//!
//! See `ROADMAP.md` for which of the original 50 modes are implemented.
//!
//! # Example
//! ```
//! let data: Vec<f64> = (0..10_000).map(|i| (i as f64) * 0.5).collect();
//! let packed = quoin::compress(&data, quoin::Config::default());
//! let restored = quoin::decompress(&packed).unwrap();
//! assert_eq!(data, restored);
//! ```

#[cfg(feature = "arrow")]
pub mod arrow;
mod bitio;
mod bitpack;
mod codecs;
mod decimal;
mod decoder;
mod diag;
mod dtype;
mod encoder;
mod entropy;
mod error;
mod format;
mod hash;
mod mode;
mod transform;
mod validity;
mod varint;

use std::borrow::Cow;

use zerocopy::{FromBytes, Immutable, IntoBytes};

pub use dtype::DType;
pub use error::Error;
pub use mode::{Mode, mode_name};

/// Version string, mirroring the original `fc_ver`.
pub const VERSION: &str = concat!("quoin ", env!("CARGO_PKG_VERSION"));

/// How the encoder picks a mode for each block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Selection {
    /// Encode every applicable mode in full and keep the smallest. Best ratio,
    /// higher encode cost. The default.
    Full,
    /// Estimate each mode's size on a small stratified sample of the block, then
    /// encode only the winner in full (the BtrBlocks/Vortex approach). Much
    /// faster encode for a slight ratio risk. Opt-in so it can be A/B-benchmarked.
    Sample,
}

/// Speed/ratio trade-off. Each level sets which codecs compete and how strongly
/// slow-to-decode codecs are penalized, dialing from maximum ratio to maximum
/// speed. The cost-aware selection picks `argmin(size + λ·decode_cost)`; `λ` and
/// the entropy-coder gate both come from the level.
///
/// `Max` is the default and keeps the pure-ratio selection policy (λ = 0, all
/// codecs). The penalty bias is most effective with [`Selection::Full`]; under
/// [`Selection::Sample`] a level mainly gates which codecs are estimated.
/// The five levels form a ladder by **decode-cost class** — each step admits one
/// more, slower-to-decode tier of codec:
///
/// | level | adds | decode |
/// | --- | --- | --- |
/// | `Fastest` | minimal pool (RAW/CONST/STRIDE + FoR/delta bit-pack) | fastest |
/// | `Fast` | + XORZ/ALP/ALP-RD/dict/RLE (still no entropy) | fast, random-access |
/// | `Balanced` | + rANS entropy on the *vectorizable* modes + pco (fast decode) | fast-ish |
/// | `High` | + the sequential predictors and the range coder | slower, best "normal" ratio |
/// | `Max` | + the LZ-over-residual cascade, no speed penalty | slowest, any cost |
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    /// Minimal codec pool — RAW/CONST/STRIDE plus the FoR/delta bit-packers
    /// only — and no entropy coder. Fastest encode, random-access decode, lowest
    /// ratio. Skips the moderate-cost probes (XORZ, ALP, ALP-RD, dictionary, RLE)
    /// that `Fast` runs, so it is genuinely cheaper, not just `Fast` with a
    /// different penalty.
    Fastest,
    /// Full non-entropy competition (adds XORZ, ALP, ALP-RD, dictionary, RLE);
    /// still no entropy coder, so decode stays random-access and fast.
    Fast,
    /// Adds the rANS entropy coder on the *vectorizable* entropy modes
    /// (byte-transpose, dictionary, FLOAT_MULT) **and the pco backend** (whose
    /// decode is fast/vectorized — it fits here even though its encode is heavy),
    /// but **not** the sequential predictors and **not** the bit-serial range
    /// coder. Decode stays fast (no value-to-value recurrence); ratio improves
    /// substantially over `Fast` on structured numeric data.
    Balanced,
    /// Adds the sequential predictors (FCM/DFCM/polynomial-float/2nd-order int)
    /// and the bit-serial range coder, chosen only when their ratio gain beats
    /// their decode cost. Best ratio short of the LZ cascade; decode is slower
    /// (the predictor recurrence + range coder are not vectorizable).
    High,
    /// Maximum ratio at any cost: everything `High` has, plus the LZ-over-residual
    /// cascade, and no decode-speed penalty (`λ = 0`). The default.
    Max,
}

impl Level {
    /// Weight on decode cost in the selection score. `0` disables the penalty.
    /// Also the single knob the entropy stage reads to derive its policy: the
    /// range coder is allowed below [`RC_LAMBDA_CUTOFF`](crate::entropy::RC_LAMBDA_CUTOFF)
    /// (so `Balanced` at `2` stays rANS-only) and the LZ cascade only at `0`.
    pub(crate) fn lambda(self) -> u64 {
        match self {
            // `High` and `Max` both score by pure size (λ = 0); they differ only
            // by the LZ cascade (see [`allows_lz_cascade`]), so each level's codec
            // set is a superset of the previous one and ratio stays monotonic —
            // a decode-cost penalty here would let a cheap-to-decode mode beat a
            // much smaller one on hyper-compressible data (a confusing inversion).
            Level::Max | Level::High => 0,
            Level::Balanced => 2,
            Level::Fast => 4,
            Level::Fastest => 16,
        }
    }

    /// Whether the LZ-over-residual cascade (the most encode-expensive, slowest-
    /// decoding final stage) may run — `Max` only. This is the sole thing
    /// separating `Max` from `High`.
    pub(crate) fn allows_lz_cascade(self) -> bool {
        matches!(self, Level::Max)
    }

    /// Whether the entropy coders (rANS, and — gated by `λ` — the range coder)
    /// may compete. `Balanced` and up.
    pub(crate) fn allows_entropy(self) -> bool {
        matches!(self, Level::Balanced | Level::High | Level::Max)
    }

    /// Whether the **sequential predictors** (FCM/DFCM, polynomial-float
    /// DELTA2/DELTA_DP, 2nd-order int) may compete. They give the best ratio on
    /// smooth/structured data but decode as a non-vectorizable recurrence, so
    /// they are reserved for `High`/`Max` — `Balanced` stays recurrence-free.
    pub(crate) fn allows_predictors(self) -> bool {
        matches!(self, Level::High | Level::Max)
    }

    /// `Fastest` runs only the cheapest `O(n)` codecs (RAW/CONST/STRIDE and the
    /// FoR/delta bit-packers) — the random-access workhorses — skipping the
    /// moderate-cost probes (XORZ, ALP, ALP-RD, dictionary, RLE). This is what
    /// separates it from `Fast`: without the entropy gate alone the two would run
    /// the identical competition and emit identical streams.
    pub(crate) fn reduced_pool(self) -> bool {
        matches!(self, Level::Fastest)
    }

    /// Whether the vendored **pco** (pcodec) backend may compete — a heavyweight
    /// numeric codec (latent decomposition + bin-packing + ANS). It is unusual:
    /// **fast vectorized decode** (multi-GB/s) but *expensive encode*. Because its
    /// decode is cheap it belongs in `Balanced` (whose contract is fast decode, not
    /// fast encode) and up — `Balanced` already trades encode for ratio. Keeping it
    /// out of `Balanced` was the cause of the "Balanced decodes slower than Max"
    /// inversion (Max could pick pco, Balanced couldn't). `Fastest`/`Fast` still
    /// skip it (their contract *is* fast encode + random access).
    pub(crate) fn allows_pco(self) -> bool {
        matches!(self, Level::Balanced | Level::High | Level::Max)
    }

    /// pco search level (`0..=12`). Decode cost is level-independent — only trades
    /// encode time for ratio. (Measured: 8 vs 12 barely moves ratio on the corpus,
    /// so it is *not* a useful level-spreader; `Max` searches the top, the rest use
    /// pco's balanced default.)
    pub(crate) fn pco_level(self) -> usize {
        match self {
            Level::Max => 12,
            _ => 8,
        }
    }
}

/// Encoder configuration.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// log2 of the predictor table size. Clamped to `[10, 16]` (as in `fc`).
    pub predictor_log2: u8,
    /// Encoder worker threads. `None` lets rayon use the global pool (all
    /// cores); `Some(n)` runs on a local pool of `n`. Ignored without the
    /// `parallel` feature. Blocks are independent, so this scales nearly
    /// linearly.
    pub threads: Option<usize>,
    /// Mode-selection strategy per block (see [`Selection`]).
    pub selection: Selection,
    /// Speed/ratio trade-off (see [`Level`]).
    pub level: Level,
    /// Fixed block size in **values**, or `None` for the adaptive default
    /// (a base quantum grown for low-entropy regions). When `Some(n)`, every
    /// block holds exactly `n` values (the last may be shorter), clamped to
    /// `[1, `[`MAX_BLOCK_SIZE`]`]`.
    ///
    /// This is a granularity knob, not a pure-ratio one. **Smaller** blocks give
    /// cheaper random access (decode one block for a point lookup), finer
    /// parallelism and lower latency — ideal when quoin's block is aligned with a
    /// storage chunk/page. **Larger** blocks give dictionary/LZ a wider window and
    /// the entropy models more data to adapt to, for a better ratio. The adaptive
    /// default already grows low-entropy blocks, so leave this `None` unless you
    /// need a specific granularity.
    pub block_size: Option<usize>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            predictor_log2: 16,
            threads: None,
            selection: Selection::Full,
            level: Level::Max,
            block_size: None,
        }
    }
}

impl Config {
    pub(crate) fn clamped_predictor_log2(&self) -> u8 {
        self.predictor_log2.clamp(10, 16)
    }

    /// The configured fixed block size clamped to the valid range, or `None`
    /// for adaptive sizing.
    pub(crate) fn fixed_block_size(&self) -> Option<usize> {
        self.block_size.map(|n| n.clamp(1, MAX_BLOCK_SIZE))
    }
}

/// The largest block size (in values) the decoder accepts — a fixed
/// [`Config::block_size`] is clamped to this. It bounds per-block allocation and
/// stops a tiny frame from claiming a huge value count (a decompression bomb).
pub const MAX_BLOCK_SIZE: usize = format::MAX_BLOCK_VALUES;

/// A borrowed typed column — the input to [`compress_column`]. The variant
/// selects the codec family and the [`DType`] recorded in the stream.
#[derive(Clone, Copy, Debug)]
pub enum ColumnRef<'a> {
    /// IEEE-754 binary64.
    F64(&'a [f64]),
    /// Signed 64-bit integers (also `Timestamp`/`Date64`/`Duration` values).
    I64(&'a [i64]),
    /// Unsigned 64-bit integers.
    U64(&'a [u64]),
    /// Signed 32-bit integers (also `Date32`/`Time32` values).
    I32(&'a [i32]),
    /// Unsigned 32-bit integers.
    U32(&'a [u32]),
    /// IEEE-754 binary32.
    F32(&'a [f32]),
    /// 128-bit decimal significands with a fixed `scale` and `precision`
    /// (Arrow semantics: the logical value is `significand * 10^-scale`).
    Decimal128 {
        /// The integer significands.
        values: &'a [i128],
        /// Number of fractional digits (`10^-scale`); Arrow allows it negative.
        scale: i8,
        /// Total significant digits (Arrow: 1..=38).
        precision: u8,
    },
    /// 256-bit decimal significands as little-endian two's-complement `[u8; 32]`
    /// (matches `arrow_buffer::i256`'s byte layout).
    Decimal256 {
        /// The integer significands, little-endian 32-byte two's-complement.
        values: &'a [[u8; 32]],
        /// Number of fractional digits; Arrow allows it negative.
        scale: i8,
        /// Total significant digits (Arrow: 1..=76).
        precision: u8,
    },
}

/// An owned typed column — the output of [`decompress_column`].
#[derive(Clone, Debug, PartialEq)]
pub enum Column {
    F64(Vec<f64>),
    I64(Vec<i64>),
    U64(Vec<u64>),
    I32(Vec<i32>),
    U32(Vec<u32>),
    F32(Vec<f32>),
    /// 128-bit decimal significands with a fixed `scale`/`precision`.
    Decimal128 {
        values: Vec<i128>,
        scale: i8,
        precision: u8,
    },
    /// 256-bit decimal significands as little-endian two's-complement `[u8; 32]`.
    Decimal256 {
        values: Vec<[u8; 32]>,
        scale: i8,
        precision: u8,
    },
}

impl Column {
    /// The logical type of this column.
    pub fn dtype(&self) -> DType {
        match self {
            Column::F64(_) => DType::F64,
            Column::I64(_) => DType::I64,
            Column::U64(_) => DType::U64,
            Column::I32(_) => DType::I32,
            Column::U32(_) => DType::U32,
            Column::F32(_) => DType::F32,
            Column::Decimal128 { .. } => DType::Decimal128,
            Column::Decimal256 { .. } => DType::Decimal256,
        }
    }

    /// Number of values.
    pub fn len(&self) -> usize {
        match self {
            Column::F64(v) => v.len(),
            Column::I64(v) => v.len(),
            Column::U64(v) => v.len(),
            Column::I32(v) => v.len(),
            Column::U32(v) => v.len(),
            Column::F32(v) => v.len(),
            Column::Decimal128 { values, .. } => values.len(),
            Column::Decimal256 { values, .. } => values.len(),
        }
    }

    /// Whether the column has no values.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<'a> ColumnRef<'a> {
    /// The logical type of this column.
    pub fn dtype(&self) -> DType {
        match self {
            ColumnRef::F64(_) => DType::F64,
            ColumnRef::I64(_) => DType::I64,
            ColumnRef::U64(_) => DType::U64,
            ColumnRef::I32(_) => DType::I32,
            ColumnRef::U32(_) => DType::U32,
            ColumnRef::F32(_) => DType::F32,
            ColumnRef::Decimal128 { .. } => DType::Decimal128,
            ColumnRef::Decimal256 { .. } => DType::Decimal256,
        }
    }

    /// Number of values (including nulls).
    pub fn len(&self) -> usize {
        match self {
            ColumnRef::F64(s) => s.len(),
            ColumnRef::I64(s) => s.len(),
            ColumnRef::U64(s) => s.len(),
            ColumnRef::I32(s) => s.len(),
            ColumnRef::U32(s) => s.len(),
            ColumnRef::F32(s) => s.len(),
            ColumnRef::Decimal128 { values, .. } => values.len(),
            ColumnRef::Decimal256 { values, .. } => values.len(),
        }
    }

    /// Whether the column has no values.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Lower every value to the `u64` lane (sign-extending `i32`, widening `f32`
    /// to its exact `f64` so the float-value codecs apply directly).
    ///
    /// The 8-byte lanes (`f64`/`i64`/`u64`) are **reinterpreted in place** — the
    /// bit pattern of an `f64`/`i64` is exactly its `u64` lane word — so they
    /// return a borrowed view with no allocation. The narrower / re-typed lanes
    /// (`i32`/`u32`/`f32`) genuinely change width, so they must allocate.
    fn to_lane(self) -> Cow<'a, [u64]> {
        match self {
            ColumnRef::F64(s) => Cow::Borrowed(reinterpret_as_u64(s)),
            ColumnRef::I64(s) => Cow::Borrowed(reinterpret_as_u64(s)),
            ColumnRef::U64(s) => Cow::Borrowed(s),
            ColumnRef::I32(s) => Cow::Owned(s.iter().map(|&i| i as i64 as u64).collect()),
            ColumnRef::U32(s) => Cow::Owned(s.iter().map(|&u| u64::from(u)).collect()),
            ColumnRef::F32(s) => Cow::Owned(s.iter().map(|&f| (f as f64).to_bits()).collect()),
            // Decimals are wider than the lane; they are routed to the decimal
            // container in `compress_column` before `to_lane` is ever reached.
            ColumnRef::Decimal128 { .. } | ColumnRef::Decimal256 { .. } => {
                unreachable!("decimals use the decimal container")
            }
        }
    }
}

/// Reinterpret a slice of 8-byte `zerocopy`-compatible values as the `u64` lane
/// with **no copy**. Used for `f64`/`i64`/`u64`, whose bit patterns *are* their
/// lane words; the size/alignment match `u64` exactly, so the cast never fails.
fn reinterpret_as_u64<T: IntoBytes + Immutable>(s: &[T]) -> &[u64] {
    <[u64]>::ref_from_bytes(s.as_bytes()).expect("8-byte lane reinterprets as u64")
}

/// A decoded column: the typed values plus an optional Arrow-style validity
/// bitmap (LSB-first; a clear bit marks a null). Null slots in `values` hold the
/// type's default (`0` / `0.0`); use `validity` to tell which are null.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodedColumn {
    pub values: Column,
    pub validity: Option<Vec<u8>>,
}

/// Compress a typed column losslessly, with an optional Arrow-style validity
/// bitmap (`1` bit = valid, `0` = null; LSB-first; length `ceil(n/8)` bytes).
///
/// Nulls are compacted out — the value codec sees only the valid values — and
/// the bitmap is stored as its own (run-length-friendly) stream. Float values
/// are compressed via their raw IEEE-754 bit patterns (exact for NaN/±0);
/// integer columns run the integer codec family.
pub fn compress_column(col: ColumnRef, validity: Option<&[u8]>, cfg: Config) -> Vec<u8> {
    // Decimals are wider than the `u64` lane and own a dedicated container.
    match col {
        ColumnRef::Decimal128 {
            values,
            scale,
            precision,
        } => return decimal::compress128(values, scale, precision, validity, cfg),
        ColumnRef::Decimal256 {
            values,
            scale,
            precision,
        } => return decimal::compress256(values, scale, precision, validity, cfg),
        _ => {}
    }
    let dtype = col.dtype();
    let n = col.len();
    let has_nulls = validity.is_some_and(|bm| validity::count_valid(bm, n) < n);
    if !has_nulls {
        // No nulls (or an all-valid bitmap): every value is encoded. `to_lane`
        // borrows the input for the 8-byte lanes (f64/i64/u64) — no copy.
        return encoder::compress_lane(&col.to_lane(), dtype, n, None, cfg);
    }
    let bitmap = validity.unwrap();
    let valid = validity::compact(&col.to_lane(), bitmap);
    encoder::compress_lane(&valid, dtype, n, Some(bitmap), cfg)
}

/// Decompress a stream produced by [`compress_column`] (or [`compress`]),
/// returning the typed values and optional validity bitmap.
///
/// Unlike the original C library (which silently decodes unknown mode IDs to
/// zeros), this returns [`Error::UnknownMode`] on any unrecognized block.
pub fn decompress_column(src: &[u8]) -> Result<DecodedColumn, Error> {
    // Decimal containers carry a distinct flag and are decoded out of line; the
    // width is the dtype byte in the (otherwise unread) header.
    if decimal::is_decimal_stream(src) {
        return match DType::from_wire(*src.get(7).ok_or(Error::Truncated)?)? {
            DType::Decimal256 => {
                let d = decimal::decompress256(src)?;
                Ok(DecodedColumn {
                    values: Column::Decimal256 {
                        values: d.values,
                        scale: d.scale,
                        precision: d.precision,
                    },
                    validity: d.validity,
                })
            }
            _ => {
                let d = decimal::decompress128(src)?;
                Ok(DecodedColumn {
                    values: Column::Decimal128 {
                        values: d.values,
                        scale: d.scale,
                        precision: d.precision,
                    },
                    validity: d.validity,
                })
            }
        };
    }
    let (dtype, bits, validity) = decoder::decompress_lane(src)?;
    let values = match dtype {
        DType::F64 => Column::F64(bits.into_iter().map(f64::from_bits).collect()),
        DType::I64 => Column::I64(bits.into_iter().map(|w| w as i64).collect()),
        DType::U64 => Column::U64(bits),
        // Narrow back to the low lane bytes.
        DType::I32 => Column::I32(bits.into_iter().map(|w| w as u32 as i32).collect()),
        DType::U32 => Column::U32(bits.into_iter().map(|w| w as u32).collect()),
        // The lane holds the widened `f64`; narrow back to `f32` (exact).
        DType::F32 => Column::F32(bits.into_iter().map(|w| f64::from_bits(w) as f32).collect()),
        // A decimal dtype without the container flag is a corrupt stream (the
        // container path above handles every legitimate decimal).
        DType::Decimal128 | DType::Decimal256 => {
            return Err(Error::CorruptPayload("decimal lane without container"));
        }
    };
    Ok(DecodedColumn { values, validity })
}

/// Compress a stream of `f64` values losslessly.
///
/// Convenience wrapper over [`compress_column`] for the non-null float case; the
/// round trip is exact for every input including NaNs and signed zeros.
pub fn compress(src: &[f64], cfg: Config) -> Vec<u8> {
    compress_column(ColumnRef::F64(src), None, cfg)
}

/// Decompress an `f64` stream produced by [`compress`].
///
/// Returns [`Error::DTypeMismatch`] if the stream holds a non-`f64` column; use
/// [`decompress_column`] for the type-generic path.
pub fn decompress(src: &[u8]) -> Result<Vec<f64>, Error> {
    match decompress_column(src)?.values {
        Column::F64(v) => Ok(v),
        _ => Err(Error::DTypeMismatch),
    }
}

/// The number of `f64` values a stream will decode to, read cheaply from its
/// header without decompressing. Lets a caller size the output buffer up front.
pub fn decompressed_len(src: &[u8]) -> Result<usize, Error> {
    let header = format::Header::read(src)?;
    usize::try_from(header.n_values).map_err(|_| Error::Truncated)
}

/// Per-mode win counts since the last [`reset_mode_win_counts`], indexed by
/// mode ID (see [`mode_name`]). Counts how many blocks each mode won the
/// encoder competition for — diagnostics only. Updated atomically across the
/// encode thread pool.
pub fn mode_win_counts() -> [u64; 64] {
    diag::snapshot()
}

/// Reset the [`mode_win_counts`] counters to zero.
pub fn reset_mode_win_counts() {
    diag::reset();
}

/// Internal kernels re-exported so the `benches/kernels.rs` criterion harness
/// (a separate crate) can measure them. **Not a stable API** — hidden from
/// docs and exempt from semver.
#[doc(hidden)]
pub mod bench_internals {
    use crate::Error;

    /// Fold the runtime-selected (hardware where available) CRC32C hash over a block.
    pub fn hash_fold_best(vals: &[u64]) -> u32 {
        let h = crate::hash::best_hash_fn();
        vals.iter()
            .fold(0u32, |c, &v| c ^ h(crate::hash::HASH_SEED, v))
    }
    /// Fold the bit-exact software CRC32C over a block (for hw-vs-sw comparison).
    pub fn hash_fold_sw(vals: &[u64]) -> u32 {
        vals.iter().fold(0u32, |c, &v| {
            c ^ crate::hash::crc32c_u64_sw(crate::hash::HASH_SEED, v)
        })
    }

    pub fn rc_compress(bytes: &[u8]) -> Vec<u8> {
        crate::entropy::rc::compress_bytes(bytes)
    }
    pub fn rc_decompress(bytes: &[u8], max_len: usize) -> Result<Vec<u8>, Error> {
        crate::entropy::rc::decompress_bytes(bytes, max_len)
    }
    pub fn tans_compress(bytes: &[u8]) -> Option<Vec<u8>> {
        crate::entropy::tans::compress_bytes(bytes)
    }
    pub fn tans_decompress(bytes: &[u8], max_len: usize) -> Result<Vec<u8>, Error> {
        crate::entropy::tans::decompress_bytes(bytes, max_len)
    }
    /// The 4-way interleaved rANS coder — the actual default entropy coder at
    /// `Balanced` (unlike the legacy `tans`, which is decode-only).
    pub fn rans_compress(bytes: &[u8]) -> Option<Vec<u8>> {
        crate::entropy::rans::compress_bytes(bytes)
    }
    pub fn rans_decompress(bytes: &[u8], max_len: usize) -> Result<Vec<u8>, Error> {
        crate::entropy::rans::decompress_bytes(bytes, max_len)
    }

    pub fn fcm_encode(vals: &[u64], predictor_log2: u8) -> Vec<u8> {
        crate::codecs::pred::encode(vals, predictor_log2)
    }
    pub fn dfcm_encode(vals: &[u64], predictor_log2: u8) -> Vec<u8> {
        crate::codecs::pred::dfcm_encode(vals, predictor_log2)
    }

    /// Multiversion-dispatched byte-transpose (for tracking SIMD speed).
    pub fn byte_transpose(src: &[u8], n: usize, dst: &mut [u8]) {
        crate::transform::byte_transpose(src, n, dst);
    }

    /// Cascade-lab: the raw (codes, rights) streams ALP-RD currently bit-packs,
    /// so the lab can measure whether entropy-coding them is smaller.
    pub fn alp_rd_streams(vals: &[u64]) -> Option<(Vec<u64>, Vec<u64>)> {
        crate::codecs::alp_rd::debug_streams(vals)
    }
    /// Cascade-lab: ALP's full bit-packed output (digits + metadata inline), so the
    /// lab can gauge a lower bound on digits→entropy savings (rANS over the whole).
    pub fn alp_encode(vals: &[u64]) -> Option<Vec<u8>> {
        crate::codecs::alp::encode(vals)
    }
    pub fn for_bitpack_encode(vals: &[u64]) -> Vec<u8> {
        crate::codecs::for_bitpack::encode(vals, false)
    }
    pub fn for_bitpack_decode(payload: &[u8], n: usize) -> Result<Vec<u64>, Error> {
        crate::codecs::for_bitpack::decode(payload, n, false)
    }
    pub fn delta_bitpack_encode(vals: &[u64]) -> Vec<u8> {
        crate::codecs::delta_bitpack::encode(vals)
    }
    pub fn delta_bitpack_decode(payload: &[u8], n: usize) -> Result<Vec<u64>, Error> {
        crate::codecs::delta_bitpack::decode(payload, n)
    }

    /// FastLanes-style 1024-value bit-pack / unpack (for tracking SIMD speed).
    pub const BITPACK_BLOCK: usize = crate::bitpack::BLOCK;
    pub fn bitpack(values: &[u32; 1024], width: u32, out: &mut [u32]) {
        crate::bitpack::pack(values, width, out);
    }
    pub fn bitunpack(packed: &[u32], width: u32, out: &mut [u32; 1024]) {
        crate::bitpack::unpack(packed, width, out);
    }
}
