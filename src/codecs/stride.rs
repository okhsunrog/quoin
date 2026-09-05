//! STRIDE: arithmetic progression in lane bit-pattern space,
//! `v[i] = v[0] + i*stride` (wrapping at the lane width). Payload is
//! `(first, stride)` as two lane words.

use crate::error::Error;
use crate::lane::Lane;

pub(crate) fn encode<L: Lane>(vals: &[L]) -> Option<Vec<u8>> {
    if vals.len() < 2 {
        return None;
    }
    let first = vals[0];
    let stride = vals[1].wrapping_sub(vals[0]);
    for (i, &v) in vals.iter().enumerate() {
        let expect = first.wrapping_add(L::from_u64(i as u64).wrapping_mul(stride));
        if v != expect {
            return None;
        }
    }
    let mut out = Vec::with_capacity(2 * L::BYTES);
    first.write_le(&mut out);
    stride.write_le(&mut out);
    Some(out)
}

pub(crate) fn decode<L: Lane>(payload: &[u8], n: usize) -> Result<Vec<L>, Error> {
    if payload.len() != 2 * L::BYTES {
        return Err(Error::CorruptPayload("stride payload length"));
    }
    let first = L::read_le(&payload[..L::BYTES]);
    let stride = L::read_le(&payload[L::BYTES..]);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(first.wrapping_add(L::from_u64(i as u64).wrapping_mul(stride)));
    }
    Ok(out)
}
