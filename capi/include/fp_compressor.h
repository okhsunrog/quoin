/*
 * fp_compressor — C ABI for the fp-compressor floating-point compressor.
 *
 * Lossless compressor for streams of IEEE-754 doubles. Link against
 * libfp_compressor_capi.a (static) or libfp_compressor_capi.so (shared).
 *
 * Threading: fp_compress / fp_decompress use a process-global thread pool
 * (created lazily, lives for the process). For frequent calls with a bounded
 * thread budget, create an fp_ctx (persistent pool) and use the _ctx variants.
 *
 * Caveats:
 *  - Not async-signal-safe; do not call from a signal handler.
 *  - fork() without exec: the worker threads do not survive a fork, so do not
 *    compress/decompress in a child after the pool has been used in the parent.
 *  - Errors are returned as codes; panics are caught and reported as
 *    FP_ERR_PANIC (never unwind across this boundary).
 */
#ifndef FP_COMPRESSOR_H
#define FP_COMPRESSOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Return / error codes. */
#define FP_OK                    0
#define FP_ERR_NULL             (-1) /* a required pointer argument was NULL  */
#define FP_ERR_BUFFER_TOO_SMALL (-2) /* dst too small; needed size in out_*   */
#define FP_ERR_CORRUPT          (-3) /* malformed / truncated input stream    */
#define FP_ERR_PANIC            (-4) /* internal error caught at the boundary  */

/* NUL-terminated version string (static; do not free). */
const char *fp_version(void);

/* Upper bound on the compressed size, in bytes, of n_values doubles. */
size_t fp_compress_bound(size_t n_values);

/*
 * Compress n_values doubles from src into dst (capacity dst_cap bytes).
 * On success returns FP_OK and writes the compressed length to *out_len.
 * If dst is too small, returns FP_ERR_BUFFER_TOO_SMALL and sets *out_len to
 * the required size. Uses the global thread pool.
 */
int fp_compress(const double *src, size_t n_values,
                uint8_t *dst, size_t dst_cap, size_t *out_len);

/*
 * Decompress src_len bytes into dst (capacity dst_cap_values doubles).
 * On success returns FP_OK and writes the value count to *out_values.
 * Returns FP_ERR_CORRUPT on a malformed stream, or FP_ERR_BUFFER_TOO_SMALL
 * (with the needed count in *out_values) if dst is too small.
 */
int fp_decompress(const uint8_t *src, size_t src_len,
                  double *dst, size_t dst_cap_values, size_t *out_values);

/*
 * Number of doubles a stream decodes to, read cheaply from its header.
 * Returns the count (>= 0) or a negative FP_ERR_* code. Use it to size the
 * fp_decompress output buffer.
 */
ptrdiff_t fp_decompressed_value_count(const uint8_t *src, size_t src_len);

/* Opaque compression context owning a persistent thread pool. */
typedef struct FpCtx FpCtx;

/* Create a context with `threads` workers (0 = use the global pool). Returns
 * NULL only on pool-creation failure. Free with fp_ctx_free. */
FpCtx *fp_ctx_create(size_t threads);

/* Free a context (joins its threads). NULL is a no-op. */
void fp_ctx_free(FpCtx *ctx);

/* As fp_compress / fp_decompress but on ctx's persistent pool (no per-call
 * pool churn). A NULL ctx falls back to the global pool. */
int fp_compress_ctx(const FpCtx *ctx, const double *src, size_t n_values,
                    uint8_t *dst, size_t dst_cap, size_t *out_len);
int fp_decompress_ctx(const FpCtx *ctx, const uint8_t *src, size_t src_len,
                      double *dst, size_t dst_cap_values, size_t *out_values);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* FP_COMPRESSOR_H */
