# Roadmap

Porting the `fc` floating-point compressor to Rust, module by module. The
original is one ~6,200-line C file with ~50 codecs; this tracks the port.

## Done (v0.1)

- [x] Project skeleton, safe public API (`compress` / `decompress` / `Config`).
- [x] Stream format v1 (16-byte header + per-block frames) with bounds-checked
      decode.
- [x] Mode competition driver (single-threaded, fixed quantum).
- [x] SIMD plumbing: CRC32C FCM hash kernel (hw `_mm_crc32_u64` + bit-exact
      software fallback, runtime-dispatched); `multiversion` lane-wise transform.
- [x] Codecs: `RAW`, `CONST`, `STRIDE`, `XORZ`, `PRED` (FCM + XOR + LEB128).
- [x] Round-trip tests across synthetic datasets (incl. NaN / ±0 / inf).
- [x] Benchmark harness vs zstd (vendored) and the C `fc` (FFI), 17 datasets.
- [x] **Binary range coder** (LZMA-style) + adaptive order-1 byte model.
- [x] `PRED_RC` (range-coded predictor residuals). Aggregate ratio 1.73→2.02×.

## Building blocks to port next

These unlock most of the remaining modes:

- [ ] **Bit reader/writer** (`bw_t`/`br_t`) — needed by tANS and bit-packers.
- [ ] **tANS** (table ANS, 8-bit symbols) — `PRED_TANS`, `FUZZY_STRIDE_ANS`,
      `BWT_MTF_TANS`. Faster decode than the binary RC; `fc`'s common winner.
- [x] **Binary range coder** — done; reused by `BWT_MTF_RC`, future direct models.
- [ ] **DFCM predictor** + 2-way set-associative variants — `PRED2`, `PRED4`.
- [ ] **AVX2 gather predictor** (`_mm256_i64gather_epi64`) — the second hot
      kernel; `PRED_SIMD_INTERLEAVED`, `PRED_INTERLEAVED`.

## Codec backlog (by family)

- **Predictors**: `PRED_TANS`, `PRED_RC`, `PRED2`, `PRED4`, `PRED_ADAPTIVE`,
  `PRED_INTERLEAVED`, `PRED_SIMD_INTERLEAVED`, `VITERBI`, `LSB_STRIP`.
- **XOR / delta**: `XOR128`, `LOOKBACK_DELTA`, `ORDERED_DELTA`, `DELTA2`,
  `DELTA_BINNED`, `DELTA_DP_BINNED`.
- **Const / stride / dict**: `FUZZY_STRIDE`, `FUZZY_STRIDE_ANS`, `DICT`,
  `LZ_DICT`, `MTF_LZ`, `FCM_RLE`.
- **Lempel-Ziv**: `LZ`, `LZ_SPLIT`.
- **FP-specific**: `FLOAT32`, `FLOAT_MULT`, `INT_MULT`, `ALP`, `ELF`.
- **Transforms**: `BYTE_TRANSPOSE`, `BITPLANE`, `TRAILING_ZERO_BP`, `SIGN_CONV`,
  `BWT`, `BWT_MTF_TANS`, `BWT_MTF_RC` (needs SA-IS suffix array).
- **Convolutional**: `CONV1`, `CONV_N`, `CONV_DOUBLE`, `CONV_DOUBLE_BP`,
  `CONV_N_BINNED`, `CONV_N_DP_BINNED`.
- **Mixers**: `PAQ_MIXER`, `PAQ4_MIXER`.

## Framework work

- [ ] Adaptive block sizing (256 KiB–1 MiB quantum probe, like `fc`).
- [ ] Multi-threaded encode + decode (`rayon` or a `std::thread` work queue).
- [ ] Feature-gated mode selection (block stats decide which modes to try),
      mirroring `fc`'s `exp_range` / `sign_flips` / `distinct_count` gates.
- [ ] ARM/NEON path for the hot kernels.
- [ ] Diagnostics counters (`fc_enc_mode_wins` equivalent).
- [ ] Optional `fc`-wire-compatible profile for cross-testing against the C lib.
- [ ] Benchmark harness vs. zstd / lz4 / the C `fc`.
