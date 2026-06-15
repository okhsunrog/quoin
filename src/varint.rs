//! Unsigned LEB128 varints over `u64`. Used by the XOR/predictor residual
//! coders: small residuals (the common case for compressible streams) cost a
//! single byte.

use crate::error::Error;

/// Append `v` to `out` as an unsigned LEB128 varint (1..=10 bytes).
#[inline]
pub(crate) fn write_u64(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 {
            out.push(byte | 0x80);
        } else {
            out.push(byte);
            return;
        }
    }
}

/// Read an unsigned LEB128 varint starting at `*pos`, advancing `*pos`.
#[inline]
pub(crate) fn read_u64(input: &[u8], pos: &mut usize) -> Result<u64, Error> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    loop {
        let byte = *input.get(*pos).ok_or(Error::Truncated)?;
        *pos += 1;
        result |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(result);
        }
        shift += 7;
        if shift >= 64 {
            return Err(Error::CorruptPayload("leb128 overflow"));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_varints() {
        let cases = [
            0u64,
            1,
            127,
            128,
            300,
            u32::MAX as u64,
            u64::MAX,
            0x0123_4567_89ab_cdef,
        ];
        let mut buf = Vec::new();
        for &c in &cases {
            write_u64(&mut buf, c);
        }
        let mut pos = 0;
        for &c in &cases {
            assert_eq!(read_u64(&buf, &mut pos).unwrap(), c);
        }
        assert_eq!(pos, buf.len());
    }
}
