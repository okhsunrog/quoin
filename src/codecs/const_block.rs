//! CONST: every value in the block is identical. Payload is the single lane word.

use crate::error::Error;
use crate::lane::Lane;

pub(crate) fn encode<L: Lane>(vals: &[L]) -> Option<Vec<u8>> {
    let first = *vals.first()?;
    if vals.iter().all(|&v| v == first) {
        let mut out = Vec::with_capacity(L::BYTES);
        first.write_le(&mut out);
        Some(out)
    } else {
        None
    }
}

pub(crate) fn decode<L: Lane>(payload: &[u8], n: usize) -> Result<Vec<L>, Error> {
    if payload.len() != L::BYTES {
        return Err(Error::CorruptPayload("const payload length"));
    }
    Ok(vec![L::read_le(payload); n])
}
