//! BYTE_TRANSPOSE: regroup the block into 8 byte-planes, then entropy-code.
//! Wins on data where a byte position is low-entropy across values (e.g. the
//! sign/exponent bytes of a smooth float stream) — transposing turns that into
//! a compressible run. The transpose itself runs on the multiversion'd
//! [`crate::transform`] kernels (AVX2/NEON where available).

use crate::error::Error;
use crate::transform::{byte_transpose, byte_untranspose};

pub(crate) fn encode(vals: &[u64]) -> Vec<u8> {
    let n = vals.len();
    let mut aos = Vec::with_capacity(n * 8);
    for &v in vals {
        aos.extend_from_slice(&v.to_le_bytes());
    }
    let mut soa = vec![0u8; n * 8];
    byte_transpose(&aos, n, &mut soa);
    soa
}

pub(crate) fn decode(soa: &[u8], n: usize) -> Result<Vec<u64>, Error> {
    if soa.len() != n * 8 {
        return Err(Error::CorruptPayload("transpose payload length"));
    }
    let mut aos = vec![0u8; n * 8];
    byte_untranspose(soa, n, &mut aos);
    let mut out = Vec::with_capacity(n);
    for chunk in aos.chunks_exact(8) {
        out.push(u64::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(out)
}
