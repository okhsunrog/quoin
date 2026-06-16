//! Entropy coders shared by the residual-coding modes.
//!
//! * [`rc`] — binary range coder, adaptive order-1 byte model. Best ratio,
//!   slow decode (eight model updates per byte).
//! * [`tans`] — table ANS. Slightly weaker ratio, much faster decode.
//!
//! [`code_residuals`]/[`decode_residuals`] run both on a predictor residual
//! stream and keep the smaller, tagging the choice in a leading byte so the
//! predictor modes don't each need two mode IDs.
//!
//! (A "prefer tANS within N% for faster decode" policy was tried and reverted:
//! at safe margins it's a no-op because RC's order-1 model beats order-0 tANS
//! by >6% on the byte-transpose streams the noisy datasets use, and larger
//! margins cost real ratio. Faster decode on those would need a faster range
//! decoder or an order-1 tANS — deferred.)

pub(crate) mod rc;
pub(crate) mod tans;

use crate::error::Error;

const TAG_RC: u8 = 0;
const TAG_TANS: u8 = 1;

/// Entropy-code a residual byte stream, choosing the smaller of range-coding
/// and tANS. Output is `[tag] ++ coded`.
pub(crate) fn code_residuals(residuals: &[u8]) -> Vec<u8> {
    let rc = rc::compress_bytes(residuals);
    let mut best_tag = TAG_RC;
    let mut best = rc;
    if let Some(t) = tans::compress_bytes(residuals)
        && t.len() < best.len()
    {
        best_tag = TAG_TANS;
        best = t;
    }
    let mut out = Vec::with_capacity(best.len() + 1);
    out.push(best_tag);
    out.extend_from_slice(&best);
    out
}

/// Inverse of [`code_residuals`]. `max_len` bounds the decoded length.
pub(crate) fn decode_residuals(payload: &[u8], max_len: usize) -> Result<Vec<u8>, Error> {
    let (&tag, rest) = payload.split_first().ok_or(Error::Truncated)?;
    match tag {
        TAG_RC => rc::decompress_bytes(rest, max_len),
        TAG_TANS => tans::decompress_bytes(rest, max_len),
        _ => Err(Error::CorruptPayload("unknown entropy tag")),
    }
}
