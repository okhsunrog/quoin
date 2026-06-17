//! RAW: verbatim little-endian lane words. The always-available fallback.
//!
//! Width is the column's element size (4 or 8): a narrow column emits only its
//! low bytes so the baseline isn't doubled by the internal `u64` lane. `F32` is
//! special — its lane holds a *widened `f64`* (8 meaningful bytes), so RAW can't
//! just truncate; it narrows each value back to `f32` (exact, since the lane came
//! from one) and emits the compact 4-byte form.

use crate::dtype::DType;
use crate::error::Error;

pub(crate) fn encode(vals: &[u64], dtype: DType) -> Vec<u8> {
    if dtype == DType::F32 {
        let mut out = Vec::with_capacity(vals.len() * 4);
        for &v in vals {
            let narrowed = (f64::from_bits(v) as f32).to_bits();
            out.extend_from_slice(&narrowed.to_le_bytes());
        }
        return out;
    }
    let lane_bytes = dtype.lane_bytes();
    let mut out = Vec::with_capacity(vals.len() * lane_bytes);
    for &v in vals {
        out.extend_from_slice(&v.to_le_bytes()[..lane_bytes]);
    }
    out
}

pub(crate) fn decode(payload: &[u8], n: usize, dtype: DType) -> Result<Vec<u64>, Error> {
    let lane_bytes = dtype.lane_bytes();
    if payload.len() != n * lane_bytes {
        return Err(Error::CorruptPayload("raw payload length"));
    }
    if dtype == DType::F32 {
        let mut out = Vec::with_capacity(n);
        for chunk in payload.chunks_exact(4) {
            let bits = u32::from_le_bytes(chunk.try_into().unwrap());
            out.push((f32::from_bits(bits) as f64).to_bits());
        }
        return Ok(out);
    }
    let mut out = Vec::with_capacity(n);
    for chunk in payload.chunks_exact(lane_bytes) {
        let mut word = [0u8; 8];
        word[..lane_bytes].copy_from_slice(chunk);
        out.push(u64::from_le_bytes(word));
    }
    Ok(out)
}
