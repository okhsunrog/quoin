//! Physical lanes: the machine word each codec operates on.
//!
//! A column is lowered to a slice of one **lane type** — [`u64`] for the 64-bit
//! types (`f64`/`i64`/`u64`) and [`u32`] for the 32-bit types
//! (`f32`/`i32`/`u32`) — by a zero-copy reinterpretation of its bytes. Every
//! codec is generic over [`Lane`], so a 32-bit column is packed, hashed, delta-
//! coded and transposed at its **native width**: masks, bit widths, dictionary
//! entries, byte planes and float arithmetic are all 32-bit. Nothing is widened
//! to 64 bits on the way through (the previous design widened `f32` to `f64`,
//! which doubled the lane and ran `f64` arithmetic on `f32` data).
//!
//! [`LaneFloat`] is the float view of a lane (`f32` for `u32`, `f64` for `u64`)
//! used by the float-value codecs (ALP, FLOAT_MULT, the polynomial predictors).
//! Bit patterns move between the two views with `to_bits`/`from_bits` only —
//! never through a numeric conversion — so every pattern, NaN payloads
//! included, is preserved exactly.

use std::fmt::Debug;
use std::hash::Hash;
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Shl, Shr, Sub};

use zerocopy::{FromBytes, Immutable, IntoBytes};

use crate::dtype::DType;
use crate::error::Error;

/// A lane word. Implemented for `u32` and `u64`.
pub(crate) trait Lane:
    Copy
    + Eq
    + Ord
    + Hash
    + Default
    + Debug
    + Send
    + Sync
    + 'static
    + FromBytes
    + IntoBytes
    + Immutable
    + BitAnd<Output = Self>
    + BitOr<Output = Self>
    + BitXor<Output = Self>
    + Not<Output = Self>
    + Shl<u32, Output = Self>
    + Shr<u32, Output = Self>
{
    /// Lane width in bits (32 or 64).
    const BITS: u32;
    /// Lane width in bytes (4 or 8).
    const BYTES: usize;
    /// All-ones lane word.
    const MAX: Self;
    /// The value of the IEEE-754 exponent field when it is all ones (inf/NaN).
    const EXP_ALL_ONES: u32;
    /// The float view of this lane.
    type Float: LaneFloat<Lane = Self>;

    /// The zero word.
    const ZERO: Self;
    /// Zero-extend to `u64`.
    fn to_u64(self) -> u64;
    /// Truncate a `u64` to the lane (the low bits).
    fn from_u64(v: u64) -> Self;
    /// `from_u64`, failing when `v` doesn't fit — for values read from a stream.
    fn try_from_u64(v: u64) -> Result<Self, Error>;
    /// Sign-extend the lane to `i64` (the lane read as a two's-complement int).
    fn as_signed(self) -> i64;
    /// Truncate an `i64` to the lane's two's-complement pattern.
    fn from_signed(v: i64) -> Self;
    fn wrapping_add(self, o: Self) -> Self;
    fn wrapping_sub(self, o: Self) -> Self;
    fn wrapping_mul(self, o: Self) -> Self;
    fn leading_zeros(self) -> u32;
    /// The IEEE-754 exponent field of the lane read as its float view.
    fn exponent_field(self) -> u32;
    /// Append the lane as `BYTES` little-endian bytes.
    fn write_le(self, out: &mut Vec<u8>);
    /// Read a lane from exactly `BYTES` little-endian bytes.
    fn read_le(bytes: &[u8]) -> Self;
    /// Bit pattern → float view (exact, every pattern).
    fn to_float(self) -> Self::Float {
        Self::Float::from_bits(self)
    }
    /// Bit width of `self` (`BITS - leading_zeros`, 0 for zero).
    fn bit_width(self) -> u32 {
        Self::BITS - self.leading_zeros()
    }
    /// Zigzag a lane-width two's-complement value to an unsigned magnitude.
    fn zigzag(self) -> Self {
        (self << 1) ^ Self::from_signed(self.as_signed() >> 63)
    }
    fn unzigzag(self) -> Self {
        (self >> 1) ^ Self::ZERO.wrapping_sub(self & Self::from_u64(1))
    }
    /// Predictor-table hash of the lane (CRC32C; identical on every platform).
    fn hash_step(self, hash: crate::hash::HashFn) -> u32 {
        hash(crate::hash::HASH_SEED, self.to_u64())
    }
    /// A slice of lane words as its little-endian byte image. Zero-copy on
    /// little-endian targets; an owned copy elsewhere (the wire is always LE).
    fn le_bytes(vals: &[Self]) -> std::borrow::Cow<'_, [u8]> {
        #[cfg(target_endian = "little")]
        {
            std::borrow::Cow::Borrowed(vals.as_bytes())
        }
        #[cfg(not(target_endian = "little"))]
        {
            let mut out = Vec::with_capacity(vals.len() * Self::BYTES);
            for &v in vals {
                v.write_le(&mut out);
            }
            std::borrow::Cow::Owned(out)
        }
    }
    /// Parse a little-endian byte image (`n * BYTES` bytes) into lane words.
    fn from_le_bytes(bytes: &[u8]) -> Vec<Self> {
        bytes.chunks_exact(Self::BYTES).map(Self::read_le).collect()
    }

    /// pco (pcodec) bridge: compress `block` as the concrete number type named
    /// by `dtype` (which must be one of this lane's types). The lane words are
    /// reinterpreted in place — no conversion, no copy.
    fn pco_compress(block: &[Self], dtype: DType, clevel: usize) -> Option<Vec<u8>>;
    /// Inverse of [`pco_compress`](Lane::pco_compress); checks the count.
    fn pco_decompress(payload: &[u8], n: usize, dtype: DType) -> Result<Vec<Self>, Error>;
}

/// The float view of a lane (`f32` / `f64`).
pub(crate) trait LaneFloat:
    'static
    + Copy
    + PartialOrd
    + PartialEq
    + Debug
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    type Lane: Lane<Float = Self>;
    const ZERO: Self;

    // --- ALP (adaptive lossless floating-point) constants ---
    /// Largest decimal exponent searched (`e` in `10^e`).
    const ALP_MAX_EXP: usize;
    /// `1.5 · 2^(mantissa bits)`: adding then subtracting it rounds to the
    /// nearest integer (ties-to-even) in one add and one sub.
    const ALP_MAGIC: Self;
    /// Magnitude bound for the magic-number rounding to be exact.
    const ALP_UPPER: Self;
    /// `10^e` / `10^-f` tables, `0..=ALP_MAX_EXP`.
    fn exp10(e: usize) -> Self;
    fn frac10(f: usize) -> Self;

    // --- FLOAT_MULT constants ---
    /// Candidate decimal scales (index is stored in the payload).
    const FM_SCALES: &'static [Self];
    /// `|k|` bound so the integer `k` fits the lane as a signed value.
    const FM_LIMIT: Self;

    fn to_bits(self) -> Self::Lane;
    fn from_bits(b: Self::Lane) -> Self;
    fn from_i32(i: i32) -> Self;
    fn from_i64(i: i64) -> Self;
    /// `as i64` (saturating, NaN → 0).
    fn to_i64(self) -> i64;
    fn to_f64(self) -> f64;
    fn abs(self) -> Self;
    /// Round half away from zero (`f32::round` / `f64::round`).
    fn round(self) -> Self;
    fn is_finite(self) -> bool;
}

const U64_MASK_EXP: u64 = 0x7FF;
const U32_MASK_EXP: u32 = 0xFF;

impl Lane for u64 {
    const BITS: u32 = 64;
    const BYTES: usize = 8;
    const MAX: u64 = u64::MAX;
    const EXP_ALL_ONES: u32 = 0x7FF;
    type Float = f64;

    const ZERO: u64 = 0;
    #[inline]
    fn to_u64(self) -> u64 {
        self
    }
    #[inline]
    fn from_u64(v: u64) -> u64 {
        v
    }
    #[inline]
    fn try_from_u64(v: u64) -> Result<u64, Error> {
        Ok(v)
    }
    #[inline]
    fn as_signed(self) -> i64 {
        self as i64
    }
    #[inline]
    fn from_signed(v: i64) -> u64 {
        v as u64
    }
    #[inline]
    fn wrapping_add(self, o: u64) -> u64 {
        u64::wrapping_add(self, o)
    }
    #[inline]
    fn wrapping_sub(self, o: u64) -> u64 {
        u64::wrapping_sub(self, o)
    }
    #[inline]
    fn wrapping_mul(self, o: u64) -> u64 {
        u64::wrapping_mul(self, o)
    }
    #[inline]
    fn leading_zeros(self) -> u32 {
        u64::leading_zeros(self)
    }
    #[inline]
    fn exponent_field(self) -> u32 {
        ((self >> 52) & U64_MASK_EXP) as u32
    }
    #[inline]
    fn write_le(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
    #[inline]
    fn read_le(bytes: &[u8]) -> u64 {
        u64::from_le_bytes(bytes.try_into().expect("8 bytes"))
    }

    fn pco_compress(block: &[u64], dtype: DType, clevel: usize) -> Option<Vec<u8>> {
        crate::codecs::pcodec::compress64(block, dtype, clevel)
    }
    fn pco_decompress(payload: &[u8], n: usize, dtype: DType) -> Result<Vec<u64>, Error> {
        crate::codecs::pcodec::decompress64(payload, n, dtype)
    }
}

impl Lane for u32 {
    const BITS: u32 = 32;
    const BYTES: usize = 4;
    const MAX: u32 = u32::MAX;
    const EXP_ALL_ONES: u32 = 0xFF;
    type Float = f32;

    const ZERO: u32 = 0;
    #[inline]
    fn to_u64(self) -> u64 {
        u64::from(self)
    }
    #[inline]
    fn from_u64(v: u64) -> u32 {
        v as u32
    }
    #[inline]
    fn try_from_u64(v: u64) -> Result<u32, Error> {
        u32::try_from(v).map_err(|_| Error::CorruptPayload("value exceeds 32-bit lane"))
    }
    #[inline]
    fn as_signed(self) -> i64 {
        i64::from(self as i32)
    }
    #[inline]
    fn from_signed(v: i64) -> u32 {
        v as u32
    }
    #[inline]
    fn wrapping_add(self, o: u32) -> u32 {
        u32::wrapping_add(self, o)
    }
    #[inline]
    fn wrapping_sub(self, o: u32) -> u32 {
        u32::wrapping_sub(self, o)
    }
    #[inline]
    fn wrapping_mul(self, o: u32) -> u32 {
        u32::wrapping_mul(self, o)
    }
    #[inline]
    fn leading_zeros(self) -> u32 {
        u32::leading_zeros(self)
    }
    #[inline]
    fn exponent_field(self) -> u32 {
        (self >> 23) & U32_MASK_EXP
    }
    #[inline]
    fn write_le(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
    #[inline]
    fn read_le(bytes: &[u8]) -> u32 {
        u32::from_le_bytes(bytes.try_into().expect("4 bytes"))
    }

    fn pco_compress(block: &[u32], dtype: DType, clevel: usize) -> Option<Vec<u8>> {
        crate::codecs::pcodec::compress32(block, dtype, clevel)
    }
    fn pco_decompress(payload: &[u8], n: usize, dtype: DType) -> Result<Vec<u32>, Error> {
        crate::codecs::pcodec::decompress32(payload, n, dtype)
    }
}

static EXP10_F64: [f64; 19] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16,
    1e17, 1e18,
];
static FRAC10_F64: [f64; 19] = [
    1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14,
    1e-15, 1e-16, 1e-17, 1e-18,
];
/// `f32` ALP tables: the reference (CWI) float constants — exponents to 10.
static EXP10_F32: [f32; 11] = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10];
static FRAC10_F32: [f32; 11] = [
    1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10,
];

static FM_SCALES_F64: [f64; 7] = [
    10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0,
];
static FM_SCALES_F32: [f32; 7] = [
    10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0,
];

impl LaneFloat for f64 {
    type Lane = u64;
    const ZERO: f64 = 0.0;
    const ALP_MAX_EXP: usize = 18;
    /// 1.5 · 2^52 (ALP `MAGIC_NUMBER`).
    const ALP_MAGIC: f64 = 6_755_399_441_055_744.0;
    /// ~2^63, ALP's encoding limit for doubles.
    const ALP_UPPER: f64 = 9.223_372_036_854_776e18;
    #[inline]
    fn exp10(e: usize) -> f64 {
        EXP10_F64[e]
    }
    #[inline]
    fn frac10(f: usize) -> f64 {
        FRAC10_F64[f]
    }
    const FM_SCALES: &'static [f64] = &FM_SCALES_F64;
    const FM_LIMIT: f64 = 9.0e18;

    #[inline]
    fn to_bits(self) -> u64 {
        f64::to_bits(self)
    }
    #[inline]
    fn from_bits(b: u64) -> f64 {
        f64::from_bits(b)
    }
    #[inline]
    fn from_i32(i: i32) -> f64 {
        f64::from(i)
    }
    #[inline]
    fn from_i64(i: i64) -> f64 {
        i as f64
    }
    #[inline]
    fn to_i64(self) -> i64 {
        self as i64
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline]
    fn abs(self) -> f64 {
        f64::abs(self)
    }
    #[inline]
    fn round(self) -> f64 {
        f64::round(self)
    }
    #[inline]
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }
}

impl LaneFloat for f32 {
    type Lane = u32;
    const ZERO: f32 = 0.0;
    const ALP_MAX_EXP: usize = 10;
    /// 1.5 · 2^23 (ALP `MAGIC_NUMBER` for floats).
    const ALP_MAGIC: f32 = 12_582_912.0;
    /// 2^22: below this the magic-number add/sub rounds to an integer exactly
    /// (the sum stays in the binade whose ulp is 1).
    const ALP_UPPER: f32 = 4_194_304.0;
    #[inline]
    fn exp10(e: usize) -> f32 {
        EXP10_F32[e]
    }
    #[inline]
    fn frac10(f: usize) -> f32 {
        FRAC10_F32[f]
    }
    const FM_SCALES: &'static [f32] = &FM_SCALES_F32;
    /// 2^31: `k` must fit the lane as an `i32`.
    const FM_LIMIT: f32 = 2_147_483_648.0;

    #[inline]
    fn to_bits(self) -> u32 {
        f32::to_bits(self)
    }
    #[inline]
    fn from_bits(b: u32) -> f32 {
        f32::from_bits(b)
    }
    #[inline]
    fn from_i32(i: i32) -> f32 {
        i as f32
    }
    #[inline]
    fn from_i64(i: i64) -> f32 {
        i as f32
    }
    #[inline]
    fn to_i64(self) -> i64 {
        self as i64
    }
    #[inline]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    #[inline]
    fn abs(self) -> f32 {
        f32::abs(self)
    }
    #[inline]
    fn round(self) -> f32 {
        f32::round(self)
    }
    #[inline]
    fn is_finite(self) -> bool {
        f32::is_finite(self)
    }
}

/// Reinterpret a slice of `T` as its lane `L` with **no copy**: `T` and `L` have
/// the same size and alignment (`f32`↔`u32`, `i64`↔`u64`, …), so the byte image
/// is the lane image and the cast never fails.
pub(crate) fn reinterpret<T: IntoBytes + Immutable, L: Lane>(s: &[T]) -> &[L] {
    debug_assert_eq!(std::mem::size_of::<T>(), L::BYTES);
    <[L]>::ref_from_bytes(s.as_bytes()).expect("same-width lane reinterprets in place")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zigzag_roundtrips_both_lanes() {
        for &v in &[0u64, 1, 2, u64::MAX, u64::MAX - 1, 1u64 << 63, 12345] {
            assert_eq!(Lane::unzigzag(Lane::zigzag(v)), v);
        }
        for &v in &[0u32, 1, 2, u32::MAX, u32::MAX - 1, 1u32 << 31, 12345] {
            assert_eq!(Lane::unzigzag(Lane::zigzag(v)), v);
        }
        // Small negatives zigzag to small magnitudes at either width.
        assert_eq!(Lane::zigzag((-1i64) as u64), 1);
        assert_eq!(Lane::zigzag((-1i32) as u32), 1);
        assert_eq!(Lane::zigzag(3u32), 6);
    }

    #[test]
    fn signed_views() {
        assert_eq!((-5i32 as u32).as_signed(), -5);
        assert_eq!(u32::from_signed(-5), -5i32 as u32);
        assert_eq!((-5i64 as u64).as_signed(), -5);
        assert_eq!(u32::MAX.as_signed(), -1);
    }

    #[test]
    fn exponent_fields() {
        assert_eq!(1.0f32.to_bits().exponent_field(), 127);
        assert_eq!(f32::INFINITY.to_bits().exponent_field(), u32::EXP_ALL_ONES);
        assert_eq!(f32::NAN.to_bits().exponent_field(), u32::EXP_ALL_ONES);
        assert_eq!(1.0f64.to_bits().exponent_field(), 1023);
        assert_eq!(f64::NAN.to_bits().exponent_field(), u64::EXP_ALL_ONES);
    }

    #[test]
    fn float_view_preserves_every_pattern() {
        for bits in [0u32, 0x8000_0000, 0x7F80_0001, 0xFFAB_CDEF, 1, 0x7F7F_FFFF] {
            assert_eq!(bits.to_float().to_bits(), bits);
        }
        let s: &[f32] = &[1.5, -0.0, f32::from_bits(0x7F80_0001)];
        let lane: &[u32] = reinterpret(s);
        assert_eq!(lane, &[1.5f32.to_bits(), 0x8000_0000, 0x7F80_0001]);
    }

    #[test]
    fn try_from_u64_rejects_overflow() {
        assert!(u32::try_from_u64(1 << 32).is_err());
        assert_eq!(u32::try_from_u64(u32::MAX as u64).unwrap(), u32::MAX);
    }
}
