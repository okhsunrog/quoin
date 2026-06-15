//! RAW: verbatim little-endian `u64` words. The always-available fallback.

use crate::error::Error;

pub(crate) fn encode(vals: &[u64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 8);
    for &v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

pub(crate) fn decode(payload: &[u8], n: usize) -> Result<Vec<u64>, Error> {
    if payload.len() != n * 8 {
        return Err(Error::CorruptPayload("raw payload length"));
    }
    let mut out = Vec::with_capacity(n);
    for chunk in payload.chunks_exact(8) {
        out.push(u64::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(out)
}
