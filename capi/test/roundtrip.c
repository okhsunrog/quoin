/* End-to-end C ABI test: compress -> decompress -> verify bit-exact.
 * Build/run via capi/test/run.sh. */
#include "quoin.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int check_roundtrip(const char *name, const double *data, size_t n,
                           const QuoinCtx *ctx) {
    size_t cap = quoin_compress_bound(n);
    uint8_t *packed = malloc(cap);
    size_t clen = 0;
    int rc = ctx ? quoin_compress_ctx(ctx, data, n, packed, cap, &clen)
                 : quoin_compress(data, n, packed, cap, &clen);
    if (rc != QUOIN_OK) {
        printf("  %-12s FAIL compress rc=%d\n", name, rc);
        free(packed);
        return 1;
    }

    ptrdiff_t want = quoin_decompressed_value_count(packed, clen);
    if (want != (ptrdiff_t)n) {
        printf("  %-12s FAIL value_count %td != %zu\n", name, want, n);
        free(packed);
        return 1;
    }

    double *out = malloc(n * sizeof(double) + 8);
    size_t got = 0;
    rc = ctx ? quoin_decompress_ctx(ctx, packed, clen, out, n, &got)
             : quoin_decompress(packed, clen, out, n, &got);
    free(packed);
    if (rc != QUOIN_OK || got != n) {
        printf("  %-12s FAIL decompress rc=%d got=%zu\n", name, rc, got);
        free(out);
        return 1;
    }

    /* Compare bit patterns so NaN/-0 are checked exactly. */
    int bad = memcmp(data, out, n * sizeof(double)) != 0;
    free(out);
    double ratio = (double)(n * 8) / (double)clen;
    printf("  %-12s %s  %zu vals -> %zu bytes (%.1fx)%s\n", name,
           bad ? "MISMATCH" : "ok", n, clen, ratio, "");
    return bad;
}

/* Typed Arrow-native API: compress -> decompress into caller buffers -> verify.
 * For the nulls case, null-slot input values must be 0 (quoin compacts nulls and
 * restores the type default at null positions). */
static int check_typed(const char *name, int dtype, size_t n,
                       const void *values, const uint8_t *validity) {
    size_t width = quoin_dtype_width(dtype);
    size_t cap = quoin_typed_compress_bound(dtype, n);
    uint8_t *packed = malloc(cap ? cap : 1);
    size_t clen = quoin_typed_compress(dtype, n, values, validity, packed, cap);
    if (clen == 0) {
        printf("  %-16s FAIL compress returned 0\n", name);
        free(packed);
        return 1;
    }
    size_t vbytes = (n + 7) / 8;
    void *vout = malloc(n * width + 8);
    uint8_t *valout = validity ? malloc(vbytes ? vbytes : 1) : NULL;
    int rc = quoin_typed_decompress(dtype, n, packed, clen, vout, valout);
    free(packed);
    int bad = (rc != QUOIN_OK) || (memcmp(values, vout, n * width) != 0);
    if (validity && !bad)
        bad = memcmp(validity, valout, vbytes) != 0;
    double ratio = (double)(n * width) / (double)clen;
    printf("  %-16s %s  %zu vals x%zuB -> %zu bytes (%.1fx)\n", name,
           bad ? "MISMATCH" : "ok", n, width, clen, ratio);
    free(vout);
    free(valout);
    return bad;
}

/* Decimal typed API: scale/precision on compress; decode is the shared path. */
static int check_typed_decimal(const char *name, int dtype, size_t n,
                               const void *values, int8_t scale, uint8_t precision) {
    size_t width = quoin_dtype_width(dtype);
    size_t cap = quoin_typed_compress_bound(dtype, n);
    uint8_t *packed = malloc(cap);
    size_t clen = quoin_typed_compress_decimal(dtype, n, values, NULL, scale, precision, packed, cap);
    if (clen == 0) {
        printf("  %-16s FAIL compress returned 0\n", name);
        free(packed);
        return 1;
    }
    void *vout = malloc(n * width + 8);
    int rc = quoin_typed_decompress(dtype, n, packed, clen, vout, NULL);
    free(packed);
    int bad = (rc != QUOIN_OK) || (memcmp(values, vout, n * width) != 0);
    double ratio = (double)(n * width) / (double)clen;
    printf("  %-16s %s  %zu vals x%zuB -> %zu bytes (%.1fx)\n", name,
           bad ? "MISMATCH" : "ok", n, width, clen, ratio);
    free(vout);
    return bad;
}

int main(void) {
    printf("quoin C ABI test: %s\n", quoin_version());
    const size_t N = 100000;
    double *ramp = malloc(N * sizeof(double));
    double *cents = malloc(N * sizeof(double));
    double *sine = malloc(N * sizeof(double));
    double *special = malloc(8 * sizeof(double));
    for (size_t i = 0; i < N; i++) {
        ramp[i] = (double)i * 0.5;
        cents[i] = (double)(i % 1000) / 100.0;
        sine[i] = sin((double)i * 1e-4);
    }
    special[0] = 0.0; special[1] = -0.0; special[2] = NAN;
    special[3] = INFINITY; special[4] = -INFINITY;
    special[5] = 1e308; special[6] = 1e-308; special[7] = 3.14159;

    int fails = 0;
    printf("global pool:\n");
    fails += check_roundtrip("ramp", ramp, N, NULL);
    fails += check_roundtrip("cents", cents, N, NULL);
    fails += check_roundtrip("sine", sine, N, NULL);
    fails += check_roundtrip("special", special, 8, NULL);
    fails += check_roundtrip("empty", ramp, 0, NULL);

    QuoinCtx *ctx = quoin_ctx_create(4);
    printf("context pool (4 threads):\n");
    fails += check_roundtrip("ramp", ramp, N, ctx);
    fails += check_roundtrip("cents", cents, N, ctx);
    quoin_ctx_free(ctx);

    /* Error path: corrupt input must not crash, must report an error. */
    uint8_t junk[32];
    memset(junk, 0xAB, sizeof junk);
    double tmp[4];
    size_t got = 0;
    int rc = quoin_decompress(junk, sizeof junk, tmp, 4, &got);
    printf("corrupt input -> rc=%d (expect %d)%s\n", rc, QUOIN_ERR_CORRUPT,
           rc == QUOIN_ERR_CORRUPT ? "  ok" : "  FAIL");
    if (rc != QUOIN_ERR_CORRUPT) fails++;

    /* Typed Arrow-native API. */
    printf("typed API:\n");
    const size_t M = 50000;
    int64_t *i64 = malloc(M * sizeof(int64_t));
    int16_t *i16 = malloc(M * sizeof(int16_t));
    uint8_t *u8 = malloc(M * sizeof(uint8_t));
    float *f32 = malloc(M * sizeof(float));
    for (size_t i = 0; i < M; i++) {
        i64[i] = 1000000 + (int64_t)(i % 5000);
        i16[i] = (int16_t)(-100 + (int)(i % 200));
        u8[i] = (uint8_t)(i % 7);
        f32[i] = sinf((float)i * 1e-4f) * 100.0f;
    }
    fails += check_typed("i64", QUOIN_DTYPE_I64, M, i64, NULL);
    fails += check_typed("i16 (widen)", QUOIN_DTYPE_I16, M, i16, NULL);
    fails += check_typed("u8 (widen)", QUOIN_DTYPE_U8, M, u8, NULL);
    fails += check_typed("f32", QUOIN_DTYPE_F32, M, f32, NULL);
    fails += check_typed("i64 empty", QUOIN_DTYPE_I64, 0, i64, NULL);

    /* Nulls: every 3rd value null; null-slot values set to 0 to match decode. */
    size_t NN = 1000;
    int32_t *i32n = calloc(NN, sizeof(int32_t));
    uint8_t *valid = malloc((NN + 7) / 8);
    memset(valid, 0, (NN + 7) / 8);
    for (size_t i = 0; i < NN; i++) {
        if (i % 3 != 0) { /* valid */
            i32n[i] = 7000 + (int32_t)(i % 50);
            valid[i >> 3] |= (uint8_t)(1u << (i & 7));
        } /* else null: value stays 0, bit stays 0 */
    }
    fails += check_typed("i32 +nulls", QUOIN_DTYPE_I32, NN, i32n, valid);

    /* Decimals (scale/precision). */
    printf("typed decimals:\n");
    int64_t *d64 = malloc(M * sizeof(int64_t));
    __int128 *d128 = malloc(M * sizeof(__int128));
    uint8_t *d256 = malloc(M * 32);
    memset(d256, 0, M * 32);
    for (size_t i = 0; i < M; i++) {
        d64[i] = 100000 + (int64_t)(i % 5000);            /* DECIMAL64 sig */
        d128[i] = (__int128)100000000 + (__int128)(i % 10000); /* DECIMAL128 sig */
        uint64_t lo = 1000000 + (uint64_t)(i % 5000);     /* DECIMAL256: small LE */
        memcpy(&d256[i * 32], &lo, 8);
    }
    fails += check_typed_decimal("dec64", QUOIN_DTYPE_DECIMAL64, M, d64, 4, 18);
    fails += check_typed_decimal("dec128", QUOIN_DTYPE_DECIMAL128, M, d128, 8, 38);
    fails += check_typed_decimal("dec256", QUOIN_DTYPE_DECIMAL256, M, d256, 8, 50);
    free(d64); free(d128); free(d256);

    /* Unaligned in AND out buffers (a column store packs values after a mask,
     * so the value pointer is typically unaligned). */
    uint8_t *ibuf = malloc(M * 8 + 1);
    uint8_t *obuf = malloc(M * 8 + 1);
    for (size_t i = 0; i < M; i++) {
        int64_t v = 5000 + (int64_t)(i % 137);
        memcpy(ibuf + 1 + i * 8, &v, 8);
    }
    size_t ucap = quoin_typed_compress_bound(QUOIN_DTYPE_I64, M);
    uint8_t *upacked = malloc(ucap);
    size_t uclen = quoin_typed_compress(QUOIN_DTYPE_I64, M, ibuf + 1, NULL, upacked, ucap);
    int urc = quoin_typed_decompress(QUOIN_DTYPE_I64, M, upacked, uclen, obuf + 1, NULL);
    int ubad = (uclen == 0) || (urc != QUOIN_OK) || (memcmp(ibuf + 1, obuf + 1, M * 8) != 0);
    printf("  %-16s %s  (unaligned in+out)\n", "i64 misaligned", ubad ? "MISMATCH" : "ok");
    fails += ubad;
    free(upacked); free(ibuf); free(obuf);

    /* Typed error path: corrupt stream must not crash. */
    uint8_t tjunk[32];
    memset(tjunk, 0xAB, sizeof tjunk);
    int64_t tout[4];
    int trc = quoin_typed_decompress(QUOIN_DTYPE_I64, 4, tjunk, sizeof tjunk, tout, NULL);
    printf("  typed corrupt    -> rc=%d (expect %d)%s\n", trc, QUOIN_ERR_CORRUPT,
           trc == QUOIN_ERR_CORRUPT ? "  ok" : "  FAIL");
    if (trc != QUOIN_ERR_CORRUPT) fails++;

    free(i64); free(i16); free(u8); free(f32); free(i32n); free(valid);

    free(ramp); free(cents); free(sine); free(special);
    printf(fails ? "\nFAILED (%d)\n" : "\nAll C ABI tests passed.\n", fails);
    return fails ? 1 : 0;
}
