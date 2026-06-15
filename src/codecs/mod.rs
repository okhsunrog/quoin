//! Per-block codecs. Each operates on the raw `u64` bit patterns of a block of
//! `f64` values. Encoders that may not apply return `None`; [`raw`] always
//! succeeds and is the competition's fallback.
//!
//! Roadmap: the original `fc` defines ~50 modes. Implemented so far: RAW,
//! CONST, STRIDE, XORZ, PRED. Next up are the predictor family with tANS/range
//! residual coders (PRED_TANS, PRED_RC) and the delta/transpose transforms.

pub(crate) mod const_block;
pub(crate) mod linear;
pub(crate) mod pred;
pub(crate) mod raw;
pub(crate) mod stride;
pub(crate) mod xorz;
