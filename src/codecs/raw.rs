//! RAW: verbatim little-endian lane words. The always-available fallback.
//!
//! Width is the lane's element size (4 or 8 bytes): a 32-bit column is stored
//! at 4 B/value because its lane *is* 32-bit — nothing is widened.

use crate::error::Error;
use crate::lane::Lane;

pub(crate) fn encode<L: Lane>(vals: &[L]) -> Vec<u8> {
    L::le_bytes(vals).into_owned()
}

pub(crate) fn decode<L: Lane>(payload: &[u8], n: usize) -> Result<Vec<L>, Error> {
    if payload.len() != n * L::BYTES {
        return Err(Error::CorruptPayload("raw payload length"));
    }
    Ok(L::from_le_bytes(payload))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_widths() {
        let v64 = [1u64, u64::MAX, 0x0102_0304_0506_0708];
        let e = encode(&v64);
        assert_eq!(e.len(), 24);
        assert_eq!(&e[16..24], &0x0102_0304_0506_0708u64.to_le_bytes());
        assert_eq!(decode::<u64>(&e, 3).unwrap(), v64);
        let v32 = [1u32, u32::MAX, 0x0102_0304];
        let e = encode(&v32);
        assert_eq!(e.len(), 12);
        assert_eq!(&e[8..12], &0x0102_0304u32.to_le_bytes());
        assert_eq!(decode::<u32>(&e, 3).unwrap(), v32);
        assert!(decode::<u32>(&e, 2).is_err());
    }
}
