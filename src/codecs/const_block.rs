//! CONST: every value in the block is identical. Payload is the single value.

use crate::error::Error;

pub(crate) fn encode(vals: &[u64]) -> Option<Vec<u8>> {
    let first = *vals.first()?;
    if vals.iter().all(|&v| v == first) {
        Some(first.to_le_bytes().to_vec())
    } else {
        None
    }
}

pub(crate) fn decode(payload: &[u8], n: usize) -> Result<Vec<u64>, Error> {
    if payload.len() != 8 {
        return Err(Error::CorruptPayload("const payload length"));
    }
    let v = u64::from_le_bytes(payload.try_into().unwrap());
    Ok(vec![v; n])
}
