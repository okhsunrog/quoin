//! STRIDE: arithmetic progression in `u64` bit-pattern space,
//! `v[i] = v[0] + i*stride` (wrapping). Payload is `(first, stride)`.

use crate::error::Error;

pub(crate) fn encode(vals: &[u64]) -> Option<Vec<u8>> {
    if vals.len() < 2 {
        return None;
    }
    let first = vals[0];
    let stride = vals[1].wrapping_sub(vals[0]);
    for (i, &v) in vals.iter().enumerate() {
        let expect = first.wrapping_add((i as u64).wrapping_mul(stride));
        if v != expect {
            return None;
        }
    }
    let mut out = Vec::with_capacity(16);
    out.extend_from_slice(&first.to_le_bytes());
    out.extend_from_slice(&stride.to_le_bytes());
    Some(out)
}

pub(crate) fn decode(payload: &[u8], n: usize) -> Result<Vec<u64>, Error> {
    if payload.len() != 16 {
        return Err(Error::CorruptPayload("stride payload length"));
    }
    let first = u64::from_le_bytes(payload[0..8].try_into().unwrap());
    let stride = u64::from_le_bytes(payload[8..16].try_into().unwrap());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(first.wrapping_add((i as u64).wrapping_mul(stride)));
    }
    Ok(out)
}
