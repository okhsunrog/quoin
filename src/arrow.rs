//! Apache Arrow adapter (feature `arrow`): compress and decompress
//! [`arrow_array::Array`]s directly, preserving their validity (null) bitmap.
//!
//! Supported today: the primitive numeric arrays that map 1:1 to a
//! [`DType`](crate::DType) — `Float64`, `Float32`, `Int64`, `UInt64`, `Int32`,
//! `UInt32`, and `Decimal128`/`Decimal256` (with precision/scale preserved).
//! Temporal and decimal types (which need their Arrow logical type preserved in
//! the stream) are planned. An Arrow validity bitmap *is* quoin's validity
//! format (LSB-first), so nulls round-trip directly.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Decimal128Type, Decimal256Type, Float32Type, Float64Type, Int32Type, Int64Type, UInt32Type,
    UInt64Type,
};
use arrow_array::{
    Array, ArrayRef, Decimal128Array, Decimal256Array, Float32Array, Float64Array, Int32Array,
    Int64Array, UInt32Array, UInt64Array,
};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer, i256};
use arrow_schema::DataType;

use crate::error::Error;
use crate::{Column, ColumnRef, Config, DecodedColumn, compress_column, decompress_column};

/// Extract an Arrow null buffer as a quoin validity bitmap (LSB-first, 1=valid).
fn validity_of(array: &dyn Array) -> Option<Vec<u8>> {
    let nb = array.nulls()?;
    let n = nb.len();
    let inner = nb.inner();
    if inner.offset() == 0 {
        // Unsliced: the buffer is already the bitmap quoin wants.
        let len = n.div_ceil(8);
        let mut bm = inner.values()[..len].to_vec();
        let rem = n & 7;
        if rem != 0 {
            bm[len - 1] &= (1u8 << rem) - 1;
        }
        Some(bm)
    } else {
        // Sliced (bit offset): re-pack from logical bit 0.
        let mut bm = vec![0u8; n.div_ceil(8)];
        for i in 0..n {
            if nb.is_valid(i) {
                bm[i >> 3] |= 1 << (i & 7);
            }
        }
        Some(bm)
    }
}

/// Build an Arrow null buffer from a quoin validity bitmap of `n` bits.
fn null_buffer(bm: Vec<u8>, n: usize) -> NullBuffer {
    NullBuffer::new(BooleanBuffer::new(Buffer::from_vec(bm), 0, n))
}

/// Compress an Arrow array — values + validity — with quoin codecs chosen by the
/// array's type. Returns [`Error::UnsupportedArrowType`] for unsupported types.
pub fn compress_array(array: &dyn Array, cfg: Config) -> Result<Vec<u8>, Error> {
    let validity = validity_of(array);
    let v = validity.as_deref();
    let out = match array.data_type() {
        DataType::Float64 => compress_column(
            ColumnRef::F64(array.as_primitive::<Float64Type>().values()),
            v,
            cfg,
        ),
        DataType::Int64 => compress_column(
            ColumnRef::I64(array.as_primitive::<Int64Type>().values()),
            v,
            cfg,
        ),
        DataType::UInt64 => compress_column(
            ColumnRef::U64(array.as_primitive::<UInt64Type>().values()),
            v,
            cfg,
        ),
        DataType::Int32 => compress_column(
            ColumnRef::I32(array.as_primitive::<Int32Type>().values()),
            v,
            cfg,
        ),
        DataType::UInt32 => compress_column(
            ColumnRef::U32(array.as_primitive::<UInt32Type>().values()),
            v,
            cfg,
        ),
        DataType::Float32 => compress_column(
            ColumnRef::F32(array.as_primitive::<Float32Type>().values()),
            v,
            cfg,
        ),
        &DataType::Decimal128(precision, scale) => compress_column(
            ColumnRef::Decimal128 {
                values: array.as_primitive::<Decimal128Type>().values(),
                scale,
                precision,
            },
            v,
            cfg,
        ),
        &DataType::Decimal256(precision, scale) => {
            // i256 → little-endian 32-byte two's-complement, the core's repr.
            let vals: Vec<[u8; 32]> = array
                .as_primitive::<Decimal256Type>()
                .values()
                .iter()
                .map(|d| d.to_le_bytes())
                .collect();
            compress_column(
                ColumnRef::Decimal256 {
                    values: &vals,
                    scale,
                    precision,
                },
                v,
                cfg,
            )
        }
        _ => return Err(Error::UnsupportedArrowType),
    };
    Ok(out)
}

/// Decompress a stream produced by [`compress_array`] back into an Arrow array
/// (with its validity; null slots hold the type default).
pub fn decompress_array(bytes: &[u8]) -> Result<ArrayRef, Error> {
    let DecodedColumn { values, validity } = decompress_column(bytes)?;
    let n = values.len();
    let nulls = validity.map(|bm| null_buffer(bm, n));
    let array: ArrayRef = match values {
        Column::F64(v) => Arc::new(Float64Array::new(v.into(), nulls)),
        Column::I64(v) => Arc::new(Int64Array::new(v.into(), nulls)),
        Column::U64(v) => Arc::new(UInt64Array::new(v.into(), nulls)),
        Column::I32(v) => Arc::new(Int32Array::new(v.into(), nulls)),
        Column::U32(v) => Arc::new(UInt32Array::new(v.into(), nulls)),
        Column::F32(v) => Arc::new(Float32Array::new(v.into(), nulls)),
        Column::Decimal128 {
            values,
            scale,
            precision,
        } => Arc::new(
            Decimal128Array::new(values.into(), nulls)
                .with_precision_and_scale(precision, scale)
                .map_err(|_| Error::UnsupportedArrowType)?,
        ),
        Column::Decimal256 {
            values,
            scale,
            precision,
        } => {
            let vals: Vec<i256> = values.iter().map(|b| i256::from_le_bytes(*b)).collect();
            Arc::new(
                Decimal256Array::new(vals.into(), nulls)
                    .with_precision_and_scale(precision, scale)
                    .map_err(|_| Error::UnsupportedArrowType)?,
            )
        }
    };
    Ok(array)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip<A: Array + PartialEq + Clone + 'static>(a: A) {
        let bytes = compress_array(&a, Config::default()).unwrap();
        let back = decompress_array(&bytes).unwrap();
        let got = back.as_any().downcast_ref::<A>().expect("type round-trips");
        assert_eq!(got, &a);
    }

    #[test]
    fn primitive_with_nulls_roundtrip() {
        roundtrip(Int64Array::from(vec![
            Some(1),
            None,
            Some(3),
            Some(-4),
            None,
            Some(6),
        ]));
        roundtrip(UInt32Array::from(vec![
            Some(10u32),
            Some(20),
            None,
            Some(40),
        ]));
        roundtrip(Int32Array::from(vec![None, Some(-2), Some(3)]));
        // no nulls — the Arrow type and values must round-trip exactly.
        roundtrip(Float64Array::from(vec![1.5, -2.0, 3.25, 4.0]));
        roundtrip(UInt64Array::from(vec![1u64, 1 << 40, 3]));
        // float NaN / signed zero with a null.
        roundtrip(Float64Array::from(vec![
            Some(f64::NAN),
            None,
            Some(-0.0),
            Some(1.0),
        ]));
        // f32: decimal-ish values, plus NaN / signed zero / a null.
        roundtrip(Float32Array::from(vec![1.5f32, -2.0, 3.25, 4.0, 0.1, 100.01]));
        roundtrip(Float32Array::from(vec![
            Some(f32::NAN),
            None,
            Some(-0.0f32),
            Some(1.0),
        ]));
    }

    #[test]
    fn decimal128_roundtrip() {
        // Prices in cents (scale 2), with a null.
        let a = Decimal128Array::from(vec![Some(12_345i128), None, Some(-9_999), Some(0)])
            .with_precision_and_scale(18, 2)
            .unwrap();
        roundtrip(a);
        // No nulls, larger precision/scale, 128-bit magnitudes.
        let b = Decimal128Array::from(vec![
            1i128 << 100,
            -(1i128 << 99),
            0,
            i128::MAX,
            i128::MIN + 1,
        ])
        .with_precision_and_scale(38, 10)
        .unwrap();
        roundtrip(b);
    }

    #[test]
    fn decimal256_roundtrip() {
        let big = i256::from_i128(1i128 << 120);
        let a = Decimal256Array::from(vec![
            Some(i256::from_i128(12_345)),
            None,
            Some(i256::from_i128(-9_999)),
            Some(big.wrapping_mul(big)), // genuine >128-bit magnitude
            Some(i256::MAX),
            Some(i256::MIN),
        ])
        .with_precision_and_scale(50, 4)
        .unwrap();
        roundtrip(a);
    }

    #[test]
    fn sliced_array_roundtrip() {
        // A sliced array has a non-zero validity bit offset.
        let full = Int64Array::from(vec![Some(1), None, Some(3), None, Some(5), Some(6)]);
        let sliced = full.slice(2, 3); // [Some(3), None, Some(5)]
        let bytes = compress_array(&sliced, Config::default()).unwrap();
        let back = decompress_array(&bytes).unwrap();
        let got = back.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(got, &sliced);
    }

    #[test]
    fn unsupported_type_errors() {
        let a = arrow_array::BooleanArray::from(vec![true, false]);
        assert_eq!(
            compress_array(&a, Config::default()),
            Err(Error::UnsupportedArrowType)
        );
    }
}
