//! Logical column type.
//!
//! The engine works on a single physical lane (a `u64` word per value) plus a
//! small descriptor — **width, family (int vs float), signedness** — that
//! decides which codecs apply and how the arithmetic transforms interpret the
//! lane. The many Apache Arrow numeric logical types collapse onto this
//! descriptor: e.g. `Int64`/`Timestamp`/`Duration`/`Date64` all map to [`I64`],
//! `Float64` to [`F64`]. See `docs/TYPES.md` for the full mapping plan.
//!
//! [`I64`]: DType::I64
//! [`F64`]: DType::F64

use crate::error::Error;

/// The logical type of a compressed column. Stored in the stream header so the
/// decoder reconstructs the right output type.
///
/// Only 64-bit lanes are implemented today (`F64`, `I64`, `U64`); narrower
/// integers, `F32`, and decimals are planned (the wire IDs are reserved so they
/// stay stable when added).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DType {
    /// IEEE-754 binary64. Compressed via its raw bit pattern.
    F64,
    /// Signed 64-bit integer. Also the lane for `Timestamp`/`Date64`/`Duration`.
    I64,
    /// Unsigned 64-bit integer.
    U64,
    /// Signed 32-bit integer. Also the lane for `Date32`/`Time32`.
    I32,
    /// Unsigned 32-bit integer.
    U32,
    /// IEEE-754 binary32. Each value is widened to its exact `f64` on the lane so
    /// the float-value codecs (ALP, FLOAT_MULT, …) apply unchanged; the round trip
    /// narrows back to `f32`. RAW emits the compact 4-byte form.
    ///
    /// Every finite value, infinity, signed zero and subnormal round-trips
    /// **bit-exactly**. The one exception is **NaN payload bits**: the float-
    /// prediction codecs reconstruct through `f64` arithmetic whose NaN-payload
    /// propagation is unspecified, so a signaling NaN may come back quieted (its
    /// *value* — NaN — is preserved). Real Arrow `Float32` data never carries
    /// meaningful NaN payloads, and neither Parquet nor Vortex preserve them.
    F32,
    /// 128-bit decimal significand (`i128`) with a fixed scale/precision. Handled
    /// by the [`crate::decimal`] container, which splits the value into 64-bit
    /// limbs run through the ordinary integer engine — it is never a per-lane
    /// `DType` the block codecs see directly. Family/width below are placeholders.
    Decimal128,
    /// 256-bit decimal significand (little-endian `[u8; 32]`). Same limb-split
    /// container as [`Decimal128`](DType::Decimal128) with four 64-bit limbs.
    Decimal256,
}

/// Codec family: which arithmetic interpretation is valid for a column.
///
/// `Float` columns run float-value transforms (ALP, FLOAT_MULT, float-linear);
/// `Int` columns run integer transforms (FoR, delta, bit-pack) on the lane.
/// The type-agnostic byte/bit codecs (RAW/CONST/STRIDE/XORZ/predictors/LZ/
/// transpose) apply to both.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Family {
    Float,
    Int,
}

impl DType {
    /// Stable on-wire identifier. Reserved IDs leave room for the planned types.
    pub(crate) fn wire_id(self) -> u8 {
        match self {
            DType::F64 => 0,
            DType::I64 => 1,
            DType::U64 => 2,
            DType::I32 => 3,
            DType::U32 => 4,
            DType::F32 => 5,
            DType::Decimal128 => 6,
            DType::Decimal256 => 7,
        }
    }

    pub(crate) fn from_wire(id: u8) -> Result<DType, Error> {
        Ok(match id {
            0 => DType::F64,
            1 => DType::I64,
            2 => DType::U64,
            3 => DType::I32,
            4 => DType::U32,
            5 => DType::F32,
            6 => DType::Decimal128,
            7 => DType::Decimal256,
            other => return Err(Error::UnsupportedDType(other)),
        })
    }

    pub(crate) fn family(self) -> Family {
        match self {
            DType::F64 | DType::F32 => Family::Float,
            DType::I64 | DType::U64 | DType::I32 | DType::U32 | DType::Decimal128 | DType::Decimal256 => {
                Family::Int
            }
        }
    }

    /// Bytes per value on the wire for the lane-literal codecs (RAW). The engine
    /// widens every value to a `u64` internally, but RAW emits only the
    /// significant low bytes so a narrow column's baseline isn't doubled.
    /// `F32` stores a widened `f64` on the lane (8 meaningful bytes for the
    /// agnostic codecs), but RAW narrows it back to its compact 4-byte form, so
    /// its wire width is 4 — see [`raw`](crate::codecs::raw).
    pub(crate) fn lane_bytes(self) -> usize {
        match self {
            DType::I32 | DType::U32 | DType::F32 => 4,
            DType::F64 | DType::I64 | DType::U64 | DType::Decimal128 | DType::Decimal256 => 8,
        }
    }

    /// Whether the lane should be interpreted as a signed integer by the
    /// frame-of-reference codec (so a mixed-sign column references its signed
    /// minimum instead of treating negatives as huge unsigned values). `f64`
    /// is unsigned here — its bit pattern is FoR'd as-is.
    pub(crate) fn signed(self) -> bool {
        matches!(self, DType::I32 | DType::I64)
    }
}
