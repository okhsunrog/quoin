//! `quoin` — a lossless compressor for streams of IEEE-754 `f64`.
//!
//! This is a from-scratch Rust port of the `fc` floating-point compressor
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

mod bitio;
mod bitpack;
mod codecs;
mod decoder;
mod diag;
mod encoder;
mod entropy;
mod error;
mod format;
mod hash;
mod mode;
mod transform;
mod varint;

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
}

impl Default for Config {
    fn default() -> Self {
        Config {
            predictor_log2: 16,
            threads: None,
            selection: Selection::Full,
        }
    }
}

impl Config {
    pub(crate) fn clamped_predictor_log2(&self) -> u8 {
        self.predictor_log2.clamp(10, 16)
    }
}

/// Compress a stream of `f64` values losslessly.
///
/// The values are treated as their raw IEEE-754 bit patterns, so the round
/// trip is exact for every input including NaNs and signed zeros.
pub fn compress(src: &[f64], cfg: Config) -> Vec<u8> {
    encoder::compress(src, cfg)
}

/// Decompress a stream produced by [`compress`].
///
/// Unlike the original C library (which silently decodes unknown mode IDs to
/// zeros), this returns [`Error::UnknownMode`] on any unrecognized block.
pub fn decompress(src: &[u8]) -> Result<Vec<f64>, Error> {
    decoder::decompress(src)
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

    /// FastLanes-style 1024-value bit-pack / unpack (for tracking SIMD speed).
    pub const BITPACK_BLOCK: usize = crate::bitpack::BLOCK;
    pub fn bitpack(values: &[u32; 1024], width: u32, out: &mut [u32]) {
        crate::bitpack::pack(values, width, out);
    }
    pub fn bitunpack(packed: &[u32], width: u32, out: &mut [u32; 1024]) {
        crate::bitpack::unpack(packed, width, out);
    }
}
