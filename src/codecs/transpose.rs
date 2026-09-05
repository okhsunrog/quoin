//! BYTE_TRANSPOSE: regroup the block into one byte-plane per lane byte (4 for a
//! 32-bit lane, 8 for a 64-bit one), then entropy-code. Wins on data where a
//! byte position is low-entropy across values (e.g. the sign/exponent bytes of
//! a smooth float stream) — transposing turns that into a compressible run.
//! The transpose itself runs on the multiversion'd [`crate::transform`] kernels.

use crate::error::Error;
use crate::lane::Lane;
use crate::transform::{byte_transpose, byte_untranspose};

pub(crate) fn encode<L: Lane>(vals: &[L]) -> Vec<u8> {
    let n = vals.len();
    let aos = L::le_bytes(vals);
    let mut soa = vec![0u8; n * L::BYTES];
    byte_transpose(&aos, n, L::BYTES, &mut soa);
    soa
}

pub(crate) fn decode<L: Lane>(soa: &[u8], n: usize) -> Result<Vec<L>, Error> {
    if soa.len() != n * L::BYTES {
        return Err(Error::CorruptPayload("transpose payload length"));
    }
    let mut aos = vec![0u8; n * L::BYTES];
    byte_untranspose(soa, n, L::BYTES, &mut aos);
    Ok(L::from_le_bytes(&aos))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planes_per_lane_width() {
        let v32: Vec<u32> = (0..100).map(|i| 0x0102_0300 | i).collect();
        let soa = encode(&v32);
        assert_eq!(soa.len(), 400);
        // Plane 1..3 are constant (0x03, 0x02, 0x01); plane 0 is the counter.
        assert!(soa[100..200].iter().all(|&b| b == 0x03));
        assert!(soa[300..400].iter().all(|&b| b == 0x01));
        assert_eq!(decode::<u32>(&soa, 100).unwrap(), v32);
        assert!(decode::<u32>(&soa, 99).is_err());

        let v64: Vec<u64> = (0..100).map(|i| 0x0102_0304_0506_0700 | i).collect();
        let soa = encode(&v64);
        assert_eq!(soa.len(), 800);
        assert_eq!(decode::<u64>(&soa, 100).unwrap(), v64);
    }
}
