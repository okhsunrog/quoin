//! Logical column type.
//!
//! The engine works on a physical **lane** — a `u64` word per value for the
//! 64-bit types, a `u32` word for the 32-bit types (see [`crate::lane`]) — plus
//! a small descriptor — **width, family (int vs float), signedness** — that
//! decides which codecs apply and how the arithmetic transforms interpret the
//! lane. The many Apache Arrow numeric logical types collapse onto this
//! descriptor: e.g. `Int64`/`Timestamp`/`Duration`/`Date64` all map to [`I64`],
//! `Float64` to [`F64`], `Float32` to [`F32`]. See `docs/TYPES.md` for the full
//! mapping plan.
//!
//! [`I64`]: DType::I64
//! [`F64`]: DType::F64
//! [`F32`]: DType::F32

use crate::error::Error;

/// The logical type of a compressed column. Stored in the stream header so the
/// decoder reconstructs the right output type.
///
/// The 64-bit types run on the `u64` lane and the 32-bit types on the `u32`
/// lane, each at its native width; decimals use a limb container. Narrower
/// integers (`I8`/`I16`, …) are planned (the wire IDs are reserved so they stay
/// stable when added).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DType {
    /// IEEE-754 binary64. Compressed via its raw bit pattern.
    F64,
    /// Signed 64-bit integer. Also the lane for `Timestamp`/`Date64`/`Duration`.
    I64,
    /// Unsigned 64-bit integer.
    U64,
    /// Signed 32-bit integer on the native 32-bit lane. Also the lane for
    /// `Date32`/`Time32`.
    I32,
    /// Unsigned 32-bit integer on the native 32-bit lane.
    U32,
    /// IEEE-754 binary32 on the native 32-bit lane: the bit pattern *is* the
    /// lane word, and the float-value codecs (ALP, FLOAT_MULT, the polynomial
    /// predictors, pco) run in `f32` arithmetic with `f32` constants. No value
    /// is ever converted to `f64`.
    ///
    /// **Every** bit pattern round-trips exactly — finite values, ±0, subnormals,
    /// infinities, and NaNs *including their payload and signaling bit*. The
    /// float-arithmetic codecs either verify their reconstruction bit-for-bit
    /// per value (ALP, FLOAT_MULT, DELTA_DP; a value that fails becomes an
    /// exception or the mode bails) or are skipped for a block containing any
    /// non-finite value (DELTA2/DELTA_DP), so no path depends on the platform's
    /// NaN-propagation rules.
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
            DType::I64
            | DType::U64
            | DType::I32
            | DType::U32
            | DType::Decimal128
            | DType::Decimal256 => Family::Int,
        }
    }

    /// Bytes per value of the type's physical lane (4 or 8): the width every
    /// codec works at and RAW's per-value cost. Decimals report the limb width
    /// (their container never reaches the lane codecs as a whole value).
    pub(crate) fn lane_bytes(self) -> usize {
        match self {
            DType::I32 | DType::U32 | DType::F32 => 4,
            DType::F64 | DType::I64 | DType::U64 | DType::Decimal128 | DType::Decimal256 => 8,
        }
    }

    /// Whether the lane should be interpreted as a signed integer by the
    /// frame-of-reference codec (so a mixed-sign column references its signed
    /// minimum instead of treating negatives as huge unsigned values). Floats
    /// are unsigned here — their bit pattern is FoR'd as-is.
    pub(crate) fn signed(self) -> bool {
        matches!(self, DType::I32 | DType::I64)
    }
}
