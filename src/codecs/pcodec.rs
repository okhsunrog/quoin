//! PCO: bridge to the vendored [`pco`] (pcodec) numeric codec.
//!
//! pco is a strong general numeric compressor — it decomposes each number into
//! latent variables (auto-detecting delta order, integer/float multiples), bins
//! them, and entropy-codes with interleaved ANS. It wins on smooth/structured
//! numeric columns (sensor streams, slowly-varying series) that quoin's own
//! transforms only partially capture, so it competes as a heavyweight block mode
//! at `Balanced`+ (see [`Level::allows_pco`](crate::Level::allows_pco)).
//!
//! The block arrives as a lane slice; the [`DType`] names which pco
//! [`Number`](quoin_pco::data_types::Number) type the lane *is* (`u32` lane →
//! `f32`/`i32`/`u32`, `u64` lane → `f64`/`i64`/`u64`), and the slice is
//! **reinterpreted in place** — no conversion, no copy, no widening. pco is
//! lossless (exact for NaN / ±0 / subnormals), so the lane round-trips exactly.

use quoin_pco::ChunkConfig;
use quoin_pco::data_types::Number;
use quoin_pco::standalone::{simple_compress, simple_decompress};
use zerocopy::{FromBytes, Immutable, IntoBytes};

use crate::dtype::DType;
use crate::error::Error;

const PCO_CORRUPT: Error = Error::CorruptPayload("pco decode");

/// Compress a lane slice as pco numbers of type `T` (same width and alignment
/// as `L`, so the reinterpretation is a no-op).
fn compress_as<L, T>(block: &[L], clevel: usize) -> Option<Vec<u8>>
where
    L: IntoBytes + Immutable,
    T: Number + FromBytes + Immutable,
{
    let nums = <[T]>::ref_from_bytes(block.as_bytes()).ok()?;
    simple_compress::<T>(nums, &ChunkConfig::default().with_compression_level(clevel)).ok()
}

/// Decompress pco numbers of type `T` and return them as lane words (a
/// same-width reinterpretation of each value; `Vec::into_iter().map().collect()`
/// runs in place for same-size elements).
fn decompress_as<L, T>(payload: &[u8], n: usize, to_lane: fn(T) -> L) -> Result<Vec<L>, Error>
where
    T: Number,
{
    let nums: Vec<T> = simple_decompress::<T>(payload).map_err(|_| PCO_CORRUPT)?;
    if nums.len() != n {
        return Err(PCO_CORRUPT);
    }
    Ok(nums.into_iter().map(to_lane).collect())
}

/// Compress a 64-bit-lane block (`f64`/`i64`/`u64`) at compression level
/// `clevel`. `None` if pco errors or the block is empty (RAW covers those).
pub(crate) fn compress64(block: &[u64], dtype: DType, clevel: usize) -> Option<Vec<u8>> {
    if block.is_empty() {
        return None;
    }
    match dtype {
        DType::F64 => compress_as::<u64, f64>(block, clevel),
        DType::I64 => compress_as::<u64, i64>(block, clevel),
        DType::U64 => compress_as::<u64, u64>(block, clevel),
        _ => None,
    }
}

pub(crate) fn decompress64(payload: &[u8], n: usize, dtype: DType) -> Result<Vec<u64>, Error> {
    match dtype {
        DType::F64 => decompress_as::<u64, f64>(payload, n, f64::to_bits),
        DType::I64 => decompress_as::<u64, i64>(payload, n, |x| x as u64),
        DType::U64 => decompress_as::<u64, u64>(payload, n, |x| x),
        _ => Err(PCO_CORRUPT),
    }
}

/// Compress a 32-bit-lane block (`f32`/`i32`/`u32`) — pco's native `f32`
/// codec path, straight from the lane.
pub(crate) fn compress32(block: &[u32], dtype: DType, clevel: usize) -> Option<Vec<u8>> {
    if block.is_empty() {
        return None;
    }
    match dtype {
        DType::F32 => compress_as::<u32, f32>(block, clevel),
        DType::I32 => compress_as::<u32, i32>(block, clevel),
        DType::U32 => compress_as::<u32, u32>(block, clevel),
        _ => None,
    }
}

pub(crate) fn decompress32(payload: &[u8], n: usize, dtype: DType) -> Result<Vec<u32>, Error> {
    match dtype {
        DType::F32 => decompress_as::<u32, f32>(payload, n, f32::to_bits),
        DType::I32 => decompress_as::<u32, i32>(payload, n, |x| x as u32),
        DType::U32 => decompress_as::<u32, u32>(payload, n, |x| x),
        _ => Err(PCO_CORRUPT),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip64(block: &[u64], dtype: DType) {
        let enc = compress64(block, dtype, 8).expect("pco should encode");
        let dec = decompress64(&enc, block.len(), dtype).expect("pco should decode");
        assert_eq!(dec, block, "lane roundtrip for {dtype:?}");
    }

    fn roundtrip32(block: &[u32], dtype: DType) -> usize {
        let enc = compress32(block, dtype, 8).expect("pco should encode");
        let dec = decompress32(&enc, block.len(), dtype).expect("pco should decode");
        assert_eq!(dec, block, "lane roundtrip for {dtype:?}");
        enc.len()
    }

    #[test]
    fn roundtrips_all_lanes() {
        // f64: smooth ramp + a NaN and ±0 to check bit-exactness.
        let f64s: Vec<u64> = (0..1000)
            .map(|i| (i as f64 * 0.5).to_bits())
            .chain([f64::NAN.to_bits(), 0.0_f64.to_bits(), (-0.0_f64).to_bits()])
            .collect();
        roundtrip64(&f64s, DType::F64);
        let i64s: Vec<u64> = (-500i64..500).map(|i| i as u64).collect();
        roundtrip64(&i64s, DType::I64);
        let u64s: Vec<u64> = (0..1000u64).map(|i| i.wrapping_mul(7)).collect();
        roundtrip64(&u64s, DType::U64);

        // f32 straight from the lane, with every special pattern bit-exact.
        let f32s: Vec<u32> = (0..1000)
            .map(|i| (i as f32 * 1.25).to_bits())
            .chain([0x7F80_0001, 0xFFC0_1234, 0x8000_0000, 1, 0x7F7F_FFFF])
            .collect();
        roundtrip32(&f32s, DType::F32);
        let i32s: Vec<u32> = (-500i32..500).map(|i| i as u32).collect();
        roundtrip32(&i32s, DType::I32);
        let u32s: Vec<u32> = (0..1000u32).map(|i| i.wrapping_mul(7)).collect();
        roundtrip32(&u32s, DType::U32);
    }

    #[test]
    fn f32_is_encoded_as_f32() {
        // The f32 encoding must be identical to what pco produces for the same
        // `&[f32]` directly — proof that no f64 bridge is involved.
        let vals: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.01).sin() * 100.0).collect();
        let lane: Vec<u32> = vals.iter().map(|v| v.to_bits()).collect();
        let via_lane = compress32(&lane, DType::F32, 8).unwrap();
        let direct =
            simple_compress::<f32>(&vals, &ChunkConfig::default().with_compression_level(8))
                .unwrap();
        assert_eq!(via_lane, direct);
    }

    #[test]
    fn empty_block_bails() {
        assert!(compress64(&[], DType::F64, 8).is_none());
        assert!(compress32(&[], DType::F32, 8).is_none());
        assert!(compress32(&[1, 2], DType::F64, 8).is_none());
    }
}
