//! C ABI for `fp-compressor`. See `include/fp_compressor.h` for the C-side
//! declarations and usage notes.
//!
//! Design (per the parent's `TODO.md`):
//! * Context-free [`fp_compress`]/[`fp_decompress`] use rayon's global pool.
//! * An opaque [`FpCtx`] owns a persistent thread pool so frequent C callers
//!   get bounded threads without per-call pool churn (they can't use rayon's
//!   `install` themselves).
//! * Every `extern "C"` entry point wraps the work in `catch_unwind` — a panic
//!   (incl. a rayon worker panic) must never unwind across the FFI boundary.
//! * The caller sizes output buffers ([`fp_compress_bound`],
//!   [`fp_decompressed_value_count`]); we copy into them.
#![allow(clippy::missing_safety_doc)]

use std::os::raw::{c_char, c_int};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::{ptr, slice};

use fp_compressor::{Config, compress, decompress, decompressed_len};

pub const FP_OK: c_int = 0;
pub const FP_ERR_NULL: c_int = -1;
pub const FP_ERR_BUFFER_TOO_SMALL: c_int = -2;
pub const FP_ERR_CORRUPT: c_int = -3;
pub const FP_ERR_PANIC: c_int = -4;

fn cfg() -> Config {
    Config {
        threads: None,
        ..Default::default()
    }
}

/// Upper bound on the compressed size of `n_values` doubles. `9*n` covers the
/// RAW payload (8 B/value) plus all header / per-block framing.
#[unsafe(no_mangle)]
pub extern "C" fn fp_compress_bound(n_values: usize) -> usize {
    n_values.saturating_mul(9).saturating_add(1024)
}

/// NUL-terminated version string (static; do not free).
#[unsafe(no_mangle)]
pub extern "C" fn fp_version() -> *const c_char {
    concat!("fp-compressor-capi ", env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

fn compress_core(
    src: *const f64,
    n_values: usize,
    dst: *mut u8,
    dst_cap: usize,
    out_len: *mut usize,
    pool: Option<&rayon::ThreadPool>,
) -> c_int {
    if src.is_null() || dst.is_null() || out_len.is_null() {
        return FP_ERR_NULL;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let input = unsafe { slice::from_raw_parts(src, n_values) };
        match pool {
            Some(p) => p.install(|| compress(input, cfg())),
            None => compress(input, cfg()),
        }
    }));
    match result {
        Ok(packed) => unsafe {
            *out_len = packed.len();
            if packed.len() > dst_cap {
                return FP_ERR_BUFFER_TOO_SMALL;
            }
            ptr::copy_nonoverlapping(packed.as_ptr(), dst, packed.len());
            FP_OK
        },
        Err(_) => FP_ERR_PANIC,
    }
}

fn decompress_core(
    src: *const u8,
    src_len: usize,
    dst: *mut f64,
    dst_cap_values: usize,
    out_values: *mut usize,
    pool: Option<&rayon::ThreadPool>,
) -> c_int {
    if src.is_null() || dst.is_null() || out_values.is_null() {
        return FP_ERR_NULL;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let input = unsafe { slice::from_raw_parts(src, src_len) };
        match pool {
            Some(p) => p.install(|| decompress(input)),
            None => decompress(input),
        }
    }));
    match result {
        Ok(Ok(vals)) => unsafe {
            *out_values = vals.len();
            if vals.len() > dst_cap_values {
                return FP_ERR_BUFFER_TOO_SMALL;
            }
            ptr::copy_nonoverlapping(vals.as_ptr(), dst, vals.len());
            FP_OK
        },
        Ok(Err(_)) => FP_ERR_CORRUPT,
        Err(_) => FP_ERR_PANIC,
    }
}

/// Compress `n_values` doubles from `src` into `dst` (capacity `dst_cap`
/// bytes). Writes the compressed length to `*out_len`. If `dst_cap` is too
/// small, returns `FP_ERR_BUFFER_TOO_SMALL` with the required size in
/// `*out_len`. Uses the global thread pool.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fp_compress(
    src: *const f64,
    n_values: usize,
    dst: *mut u8,
    dst_cap: usize,
    out_len: *mut usize,
) -> c_int {
    compress_core(src, n_values, dst, dst_cap, out_len, None)
}

/// Decompress `src_len` bytes into `dst` (capacity `dst_cap_values` doubles).
/// Writes the value count to `*out_values`. Returns `FP_ERR_CORRUPT` on a
/// malformed stream, `FP_ERR_BUFFER_TOO_SMALL` (with the needed count in
/// `*out_values`) if `dst` is too small. Uses the global thread pool.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fp_decompress(
    src: *const u8,
    src_len: usize,
    dst: *mut f64,
    dst_cap_values: usize,
    out_values: *mut usize,
) -> c_int {
    decompress_core(src, src_len, dst, dst_cap_values, out_values, None)
}

/// Number of doubles a stream decodes to, read from its header. Returns the
/// count (>= 0), or a negative `FP_ERR_*` code on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fp_decompressed_value_count(src: *const u8, src_len: usize) -> isize {
    if src.is_null() {
        return FP_ERR_NULL as isize;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let input = unsafe { slice::from_raw_parts(src, src_len) };
        decompressed_len(input)
    }));
    match result {
        Ok(Ok(n)) => n as isize,
        Ok(Err(_)) => FP_ERR_CORRUPT as isize,
        Err(_) => FP_ERR_PANIC as isize,
    }
}

/// Opaque compression context owning a persistent thread pool.
pub struct FpCtx {
    pool: Option<rayon::ThreadPool>,
}

/// Create a context with a persistent pool of `threads` workers (`0` = use the
/// global pool). Free with [`fp_ctx_free`]. Returns NULL only on pool-build
/// failure.
#[unsafe(no_mangle)]
pub extern "C" fn fp_ctx_create(threads: usize) -> *mut FpCtx {
    let pool = if threads == 0 {
        None
    } else {
        match rayon::ThreadPoolBuilder::new().num_threads(threads).build() {
            Ok(p) => Some(p),
            Err(_) => return ptr::null_mut(),
        }
    };
    Box::into_raw(Box::new(FpCtx { pool }))
}

/// Free a context created by [`fp_ctx_create`] (joins its threads). NULL is a no-op.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fp_ctx_free(ctx: *mut FpCtx) {
    if !ctx.is_null() {
        drop(unsafe { Box::from_raw(ctx) });
    }
}

/// Like [`fp_compress`] but runs on `ctx`'s persistent pool (no per-call pool
/// churn). A NULL `ctx` falls back to the global pool.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fp_compress_ctx(
    ctx: *const FpCtx,
    src: *const f64,
    n_values: usize,
    dst: *mut u8,
    dst_cap: usize,
    out_len: *mut usize,
) -> c_int {
    let pool = if ctx.is_null() {
        None
    } else {
        unsafe { &*ctx }.pool.as_ref()
    };
    compress_core(src, n_values, dst, dst_cap, out_len, pool)
}

/// Like [`fp_decompress`] but runs on `ctx`'s persistent pool.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fp_decompress_ctx(
    ctx: *const FpCtx,
    src: *const u8,
    src_len: usize,
    dst: *mut f64,
    dst_cap_values: usize,
    out_values: *mut usize,
) -> c_int {
    let pool = if ctx.is_null() {
        None
    } else {
        unsafe { &*ctx }.pool.as_ref()
    };
    decompress_core(src, src_len, dst, dst_cap_values, out_values, pool)
}
