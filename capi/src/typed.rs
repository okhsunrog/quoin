//! Typed, Arrow-native columnar C ABI (Phase 0).
//!
//! Unlike the f64-only [`crate`] entry points, these compress a *typed* fixed-
//! width column given as **Arrow-layout buffers**:
//! * `values` — a contiguous native-endian buffer of `n` values, each
//!   `quoin_dtype_width(dtype)` bytes (exactly an Arrow primitive values buffer).
//! * `validity` — an Arrow validity bitmap (LSB-first, **1 = valid**), `ceil(n/8)`
//!   bytes, or `NULL` for "all valid".
//!
//! This matches a column-store block layout one-for-one, so no transcoding is
//! needed at the call site. Narrow integer/bool lanes are widened to quoin's
//! internal 32/64-bit lanes for the value codecs (frame-of-reference packs them
//! back down), and narrowed again on decode, so the caller always sees the
//! native width.
//!
//! Decode writes **into caller-provided buffers** (no allocation handed across
//! the boundary): `values_out` (`n * width` bytes) and an optional
//! `validity_out` (`ceil(n/8)` bytes).
//!
//! Decimals (which need scale/precision) are intentionally not in this first
//! increment.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::os::raw::c_int;
use std::{ptr, slice};

use quoin::{Column, ColumnRef, Config, DecodedColumn, Error, compress_column, decompress_column};

use crate::{QUOIN_ERR_CORRUPT, QUOIN_ERR_NULL, QUOIN_ERR_PANIC, QUOIN_OK};

// Arrow-native dtype tags (kept in sync with `quoin.h`'s QuoinDType).
pub const QUOIN_DTYPE_BOOL: i32 = 0;
pub const QUOIN_DTYPE_I8: i32 = 1;
pub const QUOIN_DTYPE_I16: i32 = 2;
pub const QUOIN_DTYPE_I32: i32 = 3;
pub const QUOIN_DTYPE_I64: i32 = 4;
pub const QUOIN_DTYPE_U8: i32 = 5;
pub const QUOIN_DTYPE_U16: i32 = 6;
pub const QUOIN_DTYPE_U32: i32 = 7;
pub const QUOIN_DTYPE_U64: i32 = 8;
pub const QUOIN_DTYPE_F32: i32 = 9;
pub const QUOIN_DTYPE_F64: i32 = 10;

/// Native width in bytes of one value of `dtype` (the `values` buffer stride and
/// the `values_out` element size). Returns 0 for an unknown dtype.
#[unsafe(no_mangle)]
pub extern "C" fn quoin_dtype_width(dtype: i32) -> usize {
    match dtype {
        QUOIN_DTYPE_BOOL | QUOIN_DTYPE_I8 | QUOIN_DTYPE_U8 => 1,
        QUOIN_DTYPE_I16 | QUOIN_DTYPE_U16 => 2,
        QUOIN_DTYPE_I32 | QUOIN_DTYPE_U32 | QUOIN_DTYPE_F32 => 4,
        QUOIN_DTYPE_I64 | QUOIN_DTYPE_U64 | QUOIN_DTYPE_F64 => 8,
        _ => 0,
    }
}

/// Upper bound on the compressed size of `n` values of `dtype`. Covers RAW
/// (≤ 8 B per widened lane value) plus all framing.
#[unsafe(no_mangle)]
pub extern "C" fn quoin_typed_compress_bound(_dtype: i32, n: usize) -> usize {
    n.saturating_mul(9).saturating_add(1024)
}

fn cfg() -> Config {
    Config::default()
}

unsafe fn validity_slice<'a>(validity: *const u8, n: usize) -> Option<&'a [u8]> {
    if validity.is_null() {
        None
    } else {
        Some(unsafe { slice::from_raw_parts(validity, n.div_ceil(8)) })
    }
}

/// Compress a typed column. Returns the compressed byte count written to `dst`,
/// or **0** on any failure — unknown dtype, `dst` too small, a panic, or the data
/// not shrinking enough to fit — so the caller can fall back to storing raw
/// (matching the `tt_compress` contract).
///
/// # Safety
/// `values` points to `n * quoin_dtype_width(dtype)` readable bytes; `validity`
/// (if non-null) to `ceil(n/8)` bytes; `dst` to `dst_cap` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quoin_typed_compress(
    dtype: i32,
    n: usize,
    values: *const u8,
    validity: *const u8,
    dst: *mut u8,
    dst_cap: usize,
) -> usize {
    if values.is_null() || dst.is_null() {
        return 0;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let valid = unsafe { validity_slice(validity, n) };
        // Build a ColumnRef over the input buffer. Same-width lanes borrow it
        // directly (zero-copy); narrow lanes widen into a temporary.
        macro_rules! borrow {
            ($t:ty, $variant:ident) => {{
                let s = unsafe { slice::from_raw_parts(values as *const $t, n) };
                compress_column(ColumnRef::$variant(s), valid, cfg())
            }};
        }
        macro_rules! widen {
            ($from:ty, $to:ty, $variant:ident) => {{
                let s = unsafe { slice::from_raw_parts(values as *const $from, n) };
                let w: Vec<$to> = s.iter().map(|&x| x as $to).collect();
                compress_column(ColumnRef::$variant(&w), valid, cfg())
            }};
        }
        Some(match dtype {
            QUOIN_DTYPE_F64 => borrow!(f64, F64),
            QUOIN_DTYPE_F32 => borrow!(f32, F32),
            QUOIN_DTYPE_I64 => borrow!(i64, I64),
            QUOIN_DTYPE_U64 => borrow!(u64, U64),
            QUOIN_DTYPE_I32 => borrow!(i32, I32),
            QUOIN_DTYPE_U32 => borrow!(u32, U32),
            QUOIN_DTYPE_I16 => widen!(i16, i32, I32),
            QUOIN_DTYPE_I8 => widen!(i8, i32, I32),
            QUOIN_DTYPE_U16 => widen!(u16, u32, U32),
            QUOIN_DTYPE_U8 => widen!(u8, u32, U32),
            // Bool arrives as one byte per value (0/1); pack into the U32 lane.
            QUOIN_DTYPE_BOOL => widen!(u8, u32, U32),
            _ => return None,
        })
    }));
    match result {
        Ok(Some(packed)) if packed.len() <= dst_cap => unsafe {
            ptr::copy_nonoverlapping(packed.as_ptr(), dst, packed.len());
            packed.len()
        },
        _ => 0,
    }
}

/// Write `n` values of `dtype` decoded from `dec.values` into `values_out`,
/// narrowing the widened lane back to the native width. Returns false on a
/// dtype/lane mismatch.
unsafe fn scatter_values(dtype: i32, n: usize, values: &Column, values_out: *mut u8) -> bool {
    macro_rules! write_direct {
        ($v:expr, $t:ty) => {{
            if $v.len() != n {
                return false;
            }
            unsafe { ptr::copy_nonoverlapping($v.as_ptr(), values_out as *mut $t, n) };
        }};
    }
    macro_rules! write_narrow {
        ($v:expr, $to:ty) => {{
            if $v.len() != n {
                return false;
            }
            let out = unsafe { slice::from_raw_parts_mut(values_out as *mut $to, n) };
            for (o, &x) in out.iter_mut().zip($v.iter()) {
                *o = x as $to;
            }
        }};
    }
    match (dtype, values) {
        (QUOIN_DTYPE_F64, Column::F64(v)) => write_direct!(v, f64),
        (QUOIN_DTYPE_F32, Column::F32(v)) => write_direct!(v, f32),
        (QUOIN_DTYPE_I64, Column::I64(v)) => write_direct!(v, i64),
        (QUOIN_DTYPE_U64, Column::U64(v)) => write_direct!(v, u64),
        (QUOIN_DTYPE_I32, Column::I32(v)) => write_direct!(v, i32),
        (QUOIN_DTYPE_U32, Column::U32(v)) => write_direct!(v, u32),
        (QUOIN_DTYPE_I16, Column::I32(v)) => write_narrow!(v, i16),
        (QUOIN_DTYPE_I8, Column::I32(v)) => write_narrow!(v, i8),
        (QUOIN_DTYPE_U16, Column::U32(v)) => write_narrow!(v, u16),
        (QUOIN_DTYPE_U8, Column::U32(v)) => write_narrow!(v, u8),
        (QUOIN_DTYPE_BOOL, Column::U32(v)) => write_narrow!(v, u8),
        _ => return false,
    }
    true
}

/// Fill `validity_out` (`ceil(n/8)` bytes, LSB-first, 1 = valid) from the decoded
/// bitmap, or all-valid when the column had no nulls. Trailing bits past `n` are
/// cleared.
unsafe fn write_validity(n: usize, decoded: &Option<Vec<u8>>, validity_out: *mut u8) {
    if validity_out.is_null() {
        return;
    }
    let nbytes = n.div_ceil(8);
    let out = unsafe { slice::from_raw_parts_mut(validity_out, nbytes) };
    match decoded {
        Some(bm) => {
            let take = bm.len().min(nbytes);
            out[..take].copy_from_slice(&bm[..take]);
            out[take..].fill(0);
        }
        None => {
            out.fill(0xFF);
            // Clear bits beyond the last value in the final byte.
            if !n.is_multiple_of(8) {
                out[nbytes - 1] = (1u8 << (n % 8)) - 1;
            }
        }
    }
}

/// Decompress a stream produced by [`quoin_typed_compress`] **into caller
/// buffers**. `values_out` must hold `n * quoin_dtype_width(dtype)` bytes;
/// `validity_out` (optional, may be NULL) `ceil(n/8)` bytes. `dtype`/`n` must
/// match the original column (the stream self-describes and is validated against
/// them). Returns `QUOIN_OK` or a negative `QUOIN_ERR_*`.
///
/// # Safety
/// Buffers must be sized as documented; `src` points to `src_len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quoin_typed_decompress(
    dtype: i32,
    n: usize,
    src: *const u8,
    src_len: usize,
    values_out: *mut u8,
    validity_out: *mut u8,
) -> c_int {
    if src.is_null() || values_out.is_null() {
        return QUOIN_ERR_NULL;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let input = unsafe { slice::from_raw_parts(src, src_len) };
        decompress_column(input)
    }));
    match result {
        Ok(Ok(DecodedColumn { values, validity })) => {
            if !unsafe { scatter_values(dtype, n, &values, values_out) } {
                return QUOIN_ERR_CORRUPT;
            }
            unsafe { write_validity(n, &validity, validity_out) };
            QUOIN_OK
        }
        Ok(Err(Error::UnknownMode(_))) | Ok(Err(_)) => QUOIN_ERR_CORRUPT,
        Err(_) => QUOIN_ERR_PANIC,
    }
}
