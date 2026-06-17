//! PCO: bridge to the vendored [`pco`] (pcodec) numeric codec.
//!
//! pco is a strong general numeric compressor — it decomposes each number into
//! latent variables (auto-detecting delta order, integer/float multiples), bins
//! them, and entropy-codes with interleaved ANS. It wins on smooth/structured
//! numeric columns (sensor streams, slowly-varying series) that quoin's own
//! transforms only partially capture, so it competes as a heavyweight block mode
//! at `High`/`Max` (see [`Level::allows_pco`](crate::Level::allows_pco)).
//!
//! The block arrives as quoin's internal `u64` lane (see [`DType`]); we map each
//! lane word to the concrete pco [`Number`](pco::data_types::Number) type — using
//! the *same* convention as [`ColumnRef::to_lane`](crate::ColumnRef) — compress,
//! and on decode widen back to the lane bit-for-bit. pco is lossless (exact for
//! NaN / ±0 / subnormals), so the lane round-trips exactly.

use pco::ChunkConfig;
use pco::standalone::{simple_compress, simple_decompress};

use crate::dtype::DType;
use crate::error::Error;

/// Map a quoin `u64` lane to pco numbers of type `$t` (via `$to`), compress, and
/// return the bytes. `$to` mirrors the narrowing in `ColumnRef::to_lane`.
macro_rules! compress_as {
    ($block:expr, $cfg:expr, $t:ty, $to:expr) => {{
        let nums: Vec<$t> = $block.iter().map(|&v| $to(v)).collect();
        simple_compress::<$t>(&nums, $cfg).ok()
    }};
}

/// Compress `block` (quoin `u64` lane) with pco at compression level `clevel`.
/// Returns `None` if pco errors or the block is empty (RAW covers those).
pub(crate) fn encode(block: &[u64], dtype: DType, clevel: usize) -> Option<Vec<u8>> {
    if block.is_empty() {
        return None;
    }
    let cfg = &ChunkConfig::default().with_compression_level(clevel);
    match dtype {
        DType::F64 => compress_as!(block, cfg, f64, f64::from_bits),
        DType::F32 => compress_as!(block, cfg, f32, |v| f64::from_bits(v) as f32),
        DType::I64 => compress_as!(block, cfg, i64, |v| v as i64),
        DType::U64 => compress_as!(block, cfg, u64, |v| v),
        DType::I32 => compress_as!(block, cfg, i32, |v| v as u32 as i32),
        DType::U32 => compress_as!(block, cfg, u32, |v| v as u32),
        // Decimals never reach here directly: their limb engine lowers each limb
        // to a U64/I64 lane, which is handled above.
        DType::Decimal128 | DType::Decimal256 => None,
    }
}

/// Map a decoded pco `Vec<$t>` back to the quoin `u64` lane (via `$from`, the
/// inverse of `to_lane`), checking the count.
macro_rules! decompress_to {
    ($payload:expr, $n:expr, $t:ty, $from:expr) => {{
        let nums: Vec<$t> = simple_decompress::<$t>($payload).map_err(|_| PCO_CORRUPT)?;
        if nums.len() != $n {
            return Err(PCO_CORRUPT);
        }
        nums.into_iter().map($from).collect()
    }};
}

const PCO_CORRUPT: Error = Error::CorruptPayload("pco decode");

/// Decompress a pco block back to the quoin `u64` lane (`n` values).
pub(crate) fn decode(payload: &[u8], n: usize, dtype: DType) -> Result<Vec<u64>, Error> {
    Ok(match dtype {
        DType::F64 => decompress_to!(payload, n, f64, |x: f64| x.to_bits()),
        DType::F32 => decompress_to!(payload, n, f32, |x: f32| (x as f64).to_bits()),
        DType::I64 => decompress_to!(payload, n, i64, |x: i64| x as u64),
        DType::U64 => decompress_to!(payload, n, u64, |x: u64| x),
        DType::I32 => decompress_to!(payload, n, i32, |x: i32| x as i64 as u64),
        DType::U32 => decompress_to!(payload, n, u32, |x: u32| u64::from(x)),
        DType::Decimal128 | DType::Decimal256 => return Err(PCO_CORRUPT),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(block: &[u64], dtype: DType) {
        let enc = encode(block, dtype, 8).expect("pco should encode");
        let dec = decode(&enc, block.len(), dtype).expect("pco should decode");
        assert_eq!(dec, block, "lane roundtrip for {dtype:?}");
    }

    #[test]
    fn roundtrips_all_lanes() {
        // f64: smooth ramp + a NaN and ±0 to check bit-exactness.
        let f64s: Vec<u64> = (0..1000)
            .map(|i| (i as f64 * 0.5).to_bits())
            .chain([f64::NAN.to_bits(), 0.0_f64.to_bits(), (-0.0_f64).to_bits()])
            .collect();
        roundtrip(&f64s, DType::F64);

        // f32: lane holds the widened f64.
        let f32s: Vec<u64> = (0..1000)
            .map(|i| ((i as f32 * 1.25) as f64).to_bits())
            .collect();
        roundtrip(&f32s, DType::F32);

        // i64 / u64 / i32 / u32, including negatives (sign-extended lanes).
        let i64s: Vec<u64> = (-500i64..500).map(|i| i as u64).collect();
        roundtrip(&i64s, DType::I64);
        let u64s: Vec<u64> = (0..1000u64).map(|i| i.wrapping_mul(7)).collect();
        roundtrip(&u64s, DType::U64);
        let i32s: Vec<u64> = (-500i32..500).map(|i| i as i64 as u64).collect();
        roundtrip(&i32s, DType::I32);
        let u32s: Vec<u64> = (0..1000u32).map(u64::from).collect();
        roundtrip(&u32s, DType::U32);
    }

    #[test]
    fn empty_block_bails() {
        assert!(encode(&[], DType::F64, 8).is_none());
    }
}
