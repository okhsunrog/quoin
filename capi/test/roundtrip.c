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

    free(ramp); free(cents); free(sine); free(special);
    printf(fails ? "\nFAILED (%d)\n" : "\nAll C ABI tests passed.\n", fails);
    return fails ? 1 : 0;
}
