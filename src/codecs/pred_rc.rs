//! PRED_RC: the FCM predictor's XOR/LEB128 residual stream, entropy-coded with
//! the adaptive order-1 range coder instead of stored verbatim. This is where
//! the predictor family starts paying off — the residual bytes are far from
//! uniform, so the range coder recovers the redundancy LEB128 leaves behind.

use crate::codecs::pred;
use crate::entropy::rc;
use crate::error::Error;

/// Upper bound on the residual byte stream: at most 10 LEB128 bytes per value.
fn resid_bound(n: usize) -> usize {
    n.saturating_mul(10) + 16
}

/// Range-code an already-computed predictor residual stream. The encoder
/// competition reuses the `PRED` output instead of running the predictor twice.
pub(crate) fn encode_from_residuals(residuals: &[u8]) -> Vec<u8> {
    rc::compress_bytes(residuals)
}

pub(crate) fn decode(payload: &[u8], n: usize, predictor_log2: u8) -> Result<Vec<u64>, Error> {
    let residuals = rc::decompress_bytes(payload, resid_bound(n))?;
    pred::decode(&residuals, n, predictor_log2)
}
