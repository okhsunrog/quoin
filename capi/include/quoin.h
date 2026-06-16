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

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* QUOIN_H */
