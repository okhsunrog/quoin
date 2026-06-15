//! `fp-compressor` — a lossless compressor for streams of IEEE-754 `f64`.
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
//! let packed = fp_compressor::compress(&data, fp_compressor::Config::default());
//! let restored = fp_compressor::decompress(&packed).unwrap();
//! assert_eq!(data, restored);
//! ```

mod codecs;
mod decoder;
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
pub const VERSION: &str = concat!("fp-compressor ", env!("CARGO_PKG_VERSION"));

/// Encoder configuration.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// log2 of the predictor table size. Clamped to `[10, 16]` (as in `fc`).
    pub predictor_log2: u8,
}

impl Default for Config {
    fn default() -> Self {
        Config { predictor_log2: 16 }
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
