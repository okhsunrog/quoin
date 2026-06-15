//! XORZ: XOR each value with its predecessor, LEB128-code the result. Cheap
//! win on streams where neighbours share most of their bit pattern (repeats,
//! slowly varying integer-valued data).

use crate::error::Error;
use crate::varint;

pub(crate) fn encode(vals: &[u64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len());
    let mut prev = 0u64;
    for &v in vals {
        varint::write_u64(&mut out, v ^ prev);
        prev = v;
    }
    out
}

pub(crate) fn decode(payload: &[u8], n: usize) -> Result<Vec<u64>, Error> {
    let mut out = Vec::with_capacity(n);
    let mut prev = 0u64;
    let mut pos = 0usize;
    for _ in 0..n {
        let x = varint::read_u64(payload, &mut pos)?;
        let v = x ^ prev;
        out.push(v);
        prev = v;
    }
    if pos != payload.len() {
        return Err(Error::CorruptPayload("xorz trailing bytes"));
    }
    Ok(out)
}
