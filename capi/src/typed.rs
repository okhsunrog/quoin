//! Typed, Arrow-native columnar C ABI (Phase 0).
//!
//! Unlike the f64-only [`crate`] entry points, these compress a *typed* fixed-
//! width column given as **Arrow-layout buffers**:
//! * `values` — a contiguous native-endian buffer of `n` values, each
//!   `quoin_dtype_width(dtype)` bytes (exactly an Arrow primitive values buffer).
//!   Need **not** be aligned: a column-store packs values right after the
//!   validity bitmap, so the pointer is often unaligned — we read it byte-wise.
//! * `validity` — an Arrow validity bitmap (LSB-first, **1 = valid**), `ceil(n/8)`
//!   bytes, or `NULL` for "all valid".
//!
//! This matches a column-store block layout one-for-one, so no transcoding is
//! needed at the call site. Narrow integer/bool lanes are widened to quoin's
//! internal 32/64-bit lanes for the value codecs (frame-of-reference packs them
//! back down) and narrowed again on decode; decimals widen to the 128-bit
//! significand lane. The caller always sees the native width.
//!
//! Decode writes **into caller-provided buffers** (no allocation handed across
//! the boundary): `values_out` (`n * width` bytes) and an optional
//! `validity_out` (`ceil(n/8)` bytes).

use std::os::raw::c_int;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::{ptr, slice};

use quoin::{Column, ColumnRef, Config, DecodedColumn, compress_column, decompress_column};

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
pub const QUOIN_DTYPE_DECIMAL32: i32 = 11;
pub const QUOIN_DTYPE_DECIMAL64: i32 = 12;
pub const QUOIN_DTYPE_DECIMAL128: i32 = 13;
pub const QUOIN_DTYPE_DECIMAL256: i32 = 14;

/// Native width in bytes of one value of `dtype` (the `values` buffer stride and
/// the `values_out` element size). Returns 0 for an unknown dtype.
#[unsafe(no_mangle)]
pub extern "C" fn quoin_dtype_width(dtype: i32) -> usize {
    match dtype {
        QUOIN_DTYPE_BOOL | QUOIN_DTYPE_I8 | QUOIN_DTYPE_U8 => 1,
        QUOIN_DTYPE_I16 | QUOIN_DTYPE_U16 => 2,
        QUOIN_DTYPE_I32 | QUOIN_DTYPE_U32 | QUOIN_DTYPE_F32 | QUOIN_DTYPE_DECIMAL32 => 4,
        QUOIN_DTYPE_I64 | QUOIN_DTYPE_U64 | QUOIN_DTYPE_F64 | QUOIN_DTYPE_DECIMAL64 => 8,
        QUOIN_DTYPE_DECIMAL128 => 16,
        QUOIN_DTYPE_DECIMAL256 => 32,
        _ => 0,
    }
}

/// Upper bound on the compressed size of `n` values of `dtype`. Covers RAW plus
/// all framing; generous for the widened lanes.
#[unsafe(no_mangle)]
pub extern "C" fn quoin_typed_compress_bound(dtype: i32, n: usize) -> usize {
    let w = quoin_dtype_width(dtype).max(8);
    n.saturating_mul(w + 1).saturating_add(1024)
}

fn cfg() -> Config {
    Config::default()
}

/// Copy `n` `T`-values out of a possibly-unaligned byte buffer into an aligned
/// `Vec<T>`. Sound regardless of `ptr` alignment (byte copy into Vec storage).
unsafe fn load<T: Copy>(ptr: *const u8, n: usize) -> Vec<T> {
    let mut v = Vec::<T>::with_capacity(n);
    unsafe {
        ptr::copy_nonoverlapping(ptr, v.as_mut_ptr() as *mut u8, n * std::mem::size_of::<T>());
        v.set_len(n);
    }
    v
}

unsafe fn validity_slice<'a>(validity: *const u8, n: usize) -> Option<&'a [u8]> {
    if validity.is_null() {
        None
    } else {
        Some(unsafe { slice::from_raw_parts(validity, n.div_ceil(8)) })
    }
}

fn finish(packed: Vec<u8>, dst: *mut u8, dst_cap: usize) -> usize {
    if packed.len() <= dst_cap {
        unsafe { ptr::copy_nonoverlapping(packed.as_ptr(), dst, packed.len()) };
        packed.len()
    } else {
        0
    }
}

/// Compress a typed (non-decimal) column. Returns the compressed byte count
/// written to `dst`, or **0** on any failure (unknown dtype, `dst` too small, a
/// panic, or the data not shrinking to fit) so the caller can fall back to
/// storing raw (matching the `tt_compress` contract).
///
/// # Safety
/// `values` points to `n * quoin_dtype_width(dtype)` readable bytes (any
/// alignment); `validity` (if non-null) to `ceil(n/8)` bytes; `dst` to `dst_cap`
/// writable bytes.
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
        // `direct`: load into the matching lane. `widen`: load the narrow type,
        // then map into a wider lane (FoR re-packs it).
        macro_rules! direct {
            ($t:ty, $variant:ident) => {{
                let v = unsafe { load::<$t>(values, n) };
                compress_column(ColumnRef::$variant(&v), valid, cfg())
            }};
        }
        macro_rules! widen {
            ($from:ty, $to:ty, $variant:ident) => {{
                let v = unsafe { load::<$from>(values, n) };
                let w: Vec<$to> = v.iter().map(|&x| x as $to).collect();
                compress_column(ColumnRef::$variant(&w), valid, cfg())
            }};
        }
        Some(match dtype {
            QUOIN_DTYPE_F64 => direct!(f64, F64),
            QUOIN_DTYPE_F32 => direct!(f32, F32),
            QUOIN_DTYPE_I64 => direct!(i64, I64),
            QUOIN_DTYPE_U64 => direct!(u64, U64),
            QUOIN_DTYPE_I32 => direct!(i32, I32),
            QUOIN_DTYPE_U32 => direct!(u32, U32),
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
        Ok(Some(packed)) => finish(packed, dst, dst_cap),
        _ => 0,
    }
}

/// Compress a decimal column. Like [`quoin_typed_compress`] but for the decimal
/// dtypes, which carry a fixed `scale`/`precision` (Arrow semantics; the logical
/// value is `significand * 10^-scale`). `values` is the native-endian
/// significand buffer (`int32`/`int64`/`int128`/`int256-LE` per dtype).
///
/// # Safety
/// As [`quoin_typed_compress`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quoin_typed_compress_decimal(
    dtype: i32,
    n: usize,
    values: *const u8,
    validity: *const u8,
    scale: i8,
    precision: u8,
    dst: *mut u8,
    dst_cap: usize,
) -> usize {
    if values.is_null() || dst.is_null() {
        return 0;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let valid = unsafe { validity_slice(validity, n) };
        let packed = match dtype {
            QUOIN_DTYPE_DECIMAL128 => {
                let v = unsafe { load::<i128>(values, n) };
                compress_column(ColumnRef::Decimal128 { values: &v, scale, precision }, valid, cfg())
            }
            QUOIN_DTYPE_DECIMAL256 => {
                let v = unsafe { load::<[u8; 32]>(values, n) };
                compress_column(ColumnRef::Decimal256 { values: &v, scale, precision }, valid, cfg())
            }
            QUOIN_DTYPE_DECIMAL64 => {
                let v = unsafe { load::<i64>(values, n) };
                let w: Vec<i128> = v.iter().map(|&x| x as i128).collect();
                compress_column(ColumnRef::Decimal128 { values: &w, scale, precision }, valid, cfg())
            }
            QUOIN_DTYPE_DECIMAL32 => {
                let v = unsafe { load::<i32>(values, n) };
                let w: Vec<i128> = v.iter().map(|&x| x as i128).collect();
                compress_column(ColumnRef::Decimal128 { values: &w, scale, precision }, valid, cfg())
            }
            _ => return None,
        };
        Some(packed)
    }));
    match result {
        Ok(Some(packed)) => finish(packed, dst, dst_cap),
        _ => 0,
    }
}

/// Write `n` values of `dtype` from a decoded `Column` into `values_out`
/// (possibly unaligned), narrowing widened lanes back to the native width.
/// Returns false on a dtype/lane mismatch or length mismatch.
unsafe fn scatter_values(dtype: i32, n: usize, values: &Column, values_out: *mut u8) -> bool {
    // Copy a `&[T]` verbatim as bytes (T-sized native-endian elements).
    macro_rules! write_bytes {
        ($v:expr) => {{
            if $v.len() != n {
                return false;
            }
            let bytes = std::mem::size_of_val(&$v[..]);
            unsafe { ptr::copy_nonoverlapping($v.as_ptr() as *const u8, values_out, bytes) };
        }};
    }
    // Narrow each element to `$to` and write its native-endian bytes.
    macro_rules! write_narrow {
        ($v:expr, $to:ty) => {{
            if $v.len() != n {
                return false;
            }
            let w = std::mem::size_of::<$to>();
            let out = unsafe { slice::from_raw_parts_mut(values_out, n * w) };
            for (i, &x) in $v.iter().enumerate() {
                out[i * w..(i + 1) * w].copy_from_slice(&(x as $to).to_ne_bytes());
            }
        }};
    }
    match (dtype, values) {
        (QUOIN_DTYPE_F64, Column::F64(v)) => write_bytes!(v),
        (QUOIN_DTYPE_F32, Column::F32(v)) => write_bytes!(v),
        (QUOIN_DTYPE_I64, Column::I64(v)) => write_bytes!(v),
        (QUOIN_DTYPE_U64, Column::U64(v)) => write_bytes!(v),
        (QUOIN_DTYPE_I32, Column::I32(v)) => write_bytes!(v),
        (QUOIN_DTYPE_U32, Column::U32(v)) => write_bytes!(v),
        (QUOIN_DTYPE_I16, Column::I32(v)) => write_narrow!(v, i16),
        (QUOIN_DTYPE_I8, Column::I32(v)) => write_narrow!(v, i8),
        (QUOIN_DTYPE_U16, Column::U32(v)) => write_narrow!(v, u16),
        (QUOIN_DTYPE_U8, Column::U32(v)) => write_narrow!(v, u8),
        (QUOIN_DTYPE_BOOL, Column::U32(v)) => write_narrow!(v, u8),
        (QUOIN_DTYPE_DECIMAL128, Column::Decimal128 { values: v, .. }) => write_bytes!(v),
        (QUOIN_DTYPE_DECIMAL256, Column::Decimal256 { values: v, .. }) => write_bytes!(v),
        (QUOIN_DTYPE_DECIMAL64, Column::Decimal128 { values: v, .. }) => write_narrow!(v, i64),
        (QUOIN_DTYPE_DECIMAL32, Column::Decimal128 { values: v, .. }) => write_narrow!(v, i32),
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

/// Decompress a stream produced by [`quoin_typed_compress`] or
/// [`quoin_typed_compress_decimal`] **into caller buffers**. `values_out` must
/// hold `n * quoin_dtype_width(dtype)` bytes; `validity_out` (optional, may be
/// NULL) `ceil(n/8)` bytes. `dtype`/`n` must match the original column (the
/// stream self-describes and is validated against them). Returns `QUOIN_OK` or a
/// negative `QUOIN_ERR_*`.
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
        Ok(Err(_)) => QUOIN_ERR_CORRUPT,
        Err(_) => QUOIN_ERR_PANIC,
    }
}
