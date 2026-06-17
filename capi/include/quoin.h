/*
 * quoin — C ABI for the quoin floating-point compressor.
 *
 * Lossless compressor for streams of IEEE-754 doubles. Link against
 * libquoin_capi.a (static) or libquoin_capi.so (shared).
 *
 * Threading: quoin_compress / quoin_decompress use a process-global thread pool
 * (created lazily, lives for the process). For frequent calls with a bounded
 * thread budget, create an quoin_ctx (persistent pool) and use the _ctx variants.
 *
 * Caveats:
 *  - Not async-signal-safe; do not call from a signal handler.
 *  - fork() without exec: the worker threads do not survive a fork, so do not
 *    compress/decompress in a child after the pool has been used in the parent.
 *  - Errors are returned as codes; panics are caught and reported as
 *    QUOIN_ERR_PANIC (never unwind across this boundary).
 */
#ifndef QUOIN_H
#define QUOIN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Return / error codes. */
#define QUOIN_OK                    0
#define QUOIN_ERR_NULL             (-1) /* a required pointer argument was NULL  */
#define QUOIN_ERR_BUFFER_TOO_SMALL (-2) /* dst too small; needed size in out_*   */
#define QUOIN_ERR_CORRUPT          (-3) /* malformed / truncated input stream    */
#define QUOIN_ERR_PANIC            (-4) /* internal error caught at the boundary  */

/* NUL-terminated version string (static; do not free). */
const char *quoin_version(void);

/* Upper bound on the compressed size, in bytes, of n_values doubles. */
size_t quoin_compress_bound(size_t n_values);

/*
 * Compress n_values doubles from src into dst (capacity dst_cap bytes).
 * On success returns QUOIN_OK and writes the compressed length to *out_len.
 * If dst is too small, returns QUOIN_ERR_BUFFER_TOO_SMALL and sets *out_len to
 * the required size. Uses the global thread pool.
 */
int quoin_compress(const double *src, size_t n_values,
                uint8_t *dst, size_t dst_cap, size_t *out_len);

/*
 * Decompress src_len bytes into dst (capacity dst_cap_values doubles).
 * On success returns QUOIN_OK and writes the value count to *out_values.
 * Returns QUOIN_ERR_CORRUPT on a malformed stream, or QUOIN_ERR_BUFFER_TOO_SMALL
 * (with the needed count in *out_values) if dst is too small.
 */
int quoin_decompress(const uint8_t *src, size_t src_len,
                  double *dst, size_t dst_cap_values, size_t *out_values);

/*
 * Number of doubles a stream decodes to, read cheaply from its header.
 * Returns the count (>= 0) or a negative QUOIN_ERR_* code. Use it to size the
 * quoin_decompress output buffer.
 */
ptrdiff_t quoin_decompressed_value_count(const uint8_t *src, size_t src_len);

/* Opaque compression context owning a persistent thread pool. */
typedef struct QuoinCtx QuoinCtx;

/* Create a context with `threads` workers (0 = use the global pool). Returns
 * NULL only on pool-creation failure. Free with quoin_ctx_free. */
QuoinCtx *quoin_ctx_create(size_t threads);

/* Free a context (joins its threads). NULL is a no-op. */
void quoin_ctx_free(QuoinCtx *ctx);

/* As quoin_compress / quoin_decompress but on ctx's persistent pool (no per-call
 * pool churn). A NULL ctx falls back to the global pool. */
int quoin_compress_ctx(const QuoinCtx *ctx, const double *src, size_t n_values,
                    uint8_t *dst, size_t dst_cap, size_t *out_len);
int quoin_decompress_ctx(const QuoinCtx *ctx, const uint8_t *src, size_t src_len,
                      double *dst, size_t dst_cap_values, size_t *out_values);

/* ------------------------------------------------------------------------- *
 * Typed, Arrow-native columnar API (Phase 0)
 *
 * Compress a typed fixed-width column given as Arrow-layout buffers:
 *   values   - contiguous native-endian buffer of `n` values, each
 *              quoin_dtype_width(dtype) bytes (an Arrow primitive values buffer).
 *   validity - Arrow validity bitmap (LSB-first, 1 = valid), ceil(n/8) bytes,
 *              or NULL for "all valid".
 * This matches a column-store block one-for-one (no transcoding at the seam).
 * Decode writes INTO caller-provided buffers (no allocation crosses the ABI).
 * ------------------------------------------------------------------------- */

/* Arrow-native dtype tags. */
typedef enum {
    QUOIN_DTYPE_BOOL = 0, /* one byte per value (0/1) */
    QUOIN_DTYPE_I8   = 1,
    QUOIN_DTYPE_I16  = 2,
    QUOIN_DTYPE_I32  = 3,
    QUOIN_DTYPE_I64  = 4,
    QUOIN_DTYPE_U8   = 5,
    QUOIN_DTYPE_U16  = 6,
    QUOIN_DTYPE_U32  = 7,
    QUOIN_DTYPE_U64  = 8,
    QUOIN_DTYPE_F32  = 9,
    QUOIN_DTYPE_F64  = 10,
    /* decimals carry scale/precision; use quoin_typed_compress_decimal.
     * significand widths: 4 / 8 / 16 / 32-byte-LE. */
    QUOIN_DTYPE_DECIMAL32  = 11,
    QUOIN_DTYPE_DECIMAL64  = 12,
    QUOIN_DTYPE_DECIMAL128 = 13,
    QUOIN_DTYPE_DECIMAL256 = 14
} QuoinDType;

/* Native width in bytes of one value of `dtype` (values-buffer stride and
 * values_out element size). 0 for an unknown dtype. */
size_t quoin_dtype_width(int32_t dtype);

/* Upper bound on the compressed size of `n` values of `dtype`. */
size_t quoin_typed_compress_bound(int32_t dtype, size_t n);

/* Compress a typed column into `dst` (capacity `dst_cap`). Returns the
 * compressed byte count, or 0 on any failure (unknown dtype, dst too small,
 * panic, or the data not shrinking to fit) so the caller can store raw. */
size_t quoin_typed_compress(int32_t dtype, size_t n, const void *values,
                            const uint8_t *validity, uint8_t *dst, size_t dst_cap);

/* Compress a decimal column (DECIMAL32/64/128/256). Like quoin_typed_compress
 * but with the column's scale/precision; `values` is the native-endian
 * significand buffer (32/64/128-bit, or 256-bit little-endian). Decode uses the
 * same quoin_typed_decompress (the stream self-describes scale/precision). */
size_t quoin_typed_compress_decimal(int32_t dtype, size_t n, const void *values,
                                    const uint8_t *validity, int8_t scale,
                                    uint8_t precision, uint8_t *dst, size_t dst_cap);

/* Decompress a quoin_typed_compress stream into caller buffers: `values_out`
 * (n * quoin_dtype_width(dtype) bytes) and optional `validity_out` (ceil(n/8)
 * bytes, may be NULL). `dtype`/`n` must match the original (validated against
 * the self-describing stream). Returns QUOIN_OK or a negative QUOIN_ERR_*. */
int quoin_typed_decompress(int32_t dtype, size_t n, const uint8_t *src,
                           size_t src_len, void *values_out, uint8_t *validity_out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* QUOIN_H */
