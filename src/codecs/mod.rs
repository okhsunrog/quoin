//! Per-block codecs. Each operates on the raw `u64` bit patterns of a block of
//! `f64` values. Encoders that may not apply return `None`; [`raw`] always
//! succeeds and is the competition's fallback.
//!
//! The original `fc` defines ~50 modes; quoin implements the core predictor,
//! delta, transpose, LZ, ALP, dictionary/RLE, and bit-packing families used by
//! the block competition today.

pub(crate) mod alp;
pub(crate) mod alp_rd;
pub(crate) mod const_block;
pub(crate) mod delta_bitpack;
pub(crate) mod dict;
pub(crate) mod float_mult;
pub(crate) mod for_bitpack;
pub(crate) mod linear;
pub(crate) mod lz;
pub(crate) mod pcodec;
pub(crate) mod pred;
pub(crate) mod raw;
pub(crate) mod rle;
pub(crate) mod stride;
pub(crate) mod transpose;
pub(crate) mod xorz;
