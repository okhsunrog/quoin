//! XORZ: XOR each value with its predecessor, LEB128-code the result. Cheap
//! win on streams where neighbours share most of their bit pattern (repeats,
//! slowly varying integer-valued data). A 32-bit lane's residual is at most
//! 32 bits, so it never costs more than 5 varint bytes.

use crate::error::Error;
use crate::lane::Lane;
use crate::varint;

pub(crate) fn encode<L: Lane>(vals: &[L]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len());
    let mut prev = L::ZERO;
    for &v in vals {
        varint::write_u64(&mut out, (v ^ prev).to_u64());
        prev = v;
    }
    out
}

pub(crate) fn decode<L: Lane>(payload: &[u8], n: usize) -> Result<Vec<L>, Error> {
    let mut out = Vec::with_capacity(n);
    let mut prev = L::ZERO;
    let mut pos = 0usize;
    for _ in 0..n {
        let x = L::try_from_u64(varint::read_u64(payload, &mut pos)?)?;
        let v = x ^ prev;
        out.push(v);
        prev = v;
    }
    if pos != payload.len() {
        return Err(Error::CorruptPayload("xorz trailing bytes"));
    }
    Ok(out)
}
