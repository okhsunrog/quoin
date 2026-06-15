//! Entropy coders shared by the residual-coding modes.
//!
//! Currently: a binary range coder (`rc`) with an adaptive order-1 byte model.
//! Roadmap: tANS (table ANS) for the tag streams, to match `fc`'s PRED_TANS /
//! BWT_MTF_TANS family.

pub(crate) mod rc;
