# Roadmap

Porting the `fc` floating-point compressor to Rust, module by module. The
original is one ~6,200-line C file with ~50 codecs; this tracks the port.

## Done

- [x] Project skeleton, safe public API (`compress` / `decompress` / `Config`).
- [x] Stream format v1 (16-byte header + per-block frames) with bounds-checked
      decode.
- [x] Mode competition driver with cheap per-block gating.
- [x] SIMD plumbing: CRC32C FCM hash kernel (hw `_mm_crc32_u64` + bit-exact
      software fallback, runtime-dispatched). `multiversion`-dispatched
      byte-transpose kernel (clones verified in disassembly).
- [x] Round-trip tests across synthetic datasets (incl. NaN / ±0 / inf) plus
      `tests/robustness.rs` and `fuzz/` (cargo-fuzz) hardening the decoder.
- [x] Benchmark harness vs zstd (vendored) and the C `fc` (FFI), 17 datasets,
      plus criterion kernel micro-benchmarks (`benches/kernels.rs`).
- [x] Entropy coders: **binary range coder** (LZMA-style, order-1) and **tANS**
      (ported from `fc`) on **MSB-first bit I/O**; residuals pick the smaller.
- [x] Codecs: `RAW`, `CONST`, `STRIDE`, `XORZ`, `PRED` (FCM), `PRED2` (DFCM),
      `DELTA2` (float-linear), `ORDERED_DELTA` (integer 2nd delta), `DELTA_DP`
      (exact float-residual delta), `LZ` (LZ77), `BYTE_TRANSPOSE`.
- [x] **Block-parallel** encode/decode via rayon (default-on `parallel`
      feature; `Config.threads: Option<usize>`).
- [x] Aggregate ratio 1.59 → **3.00×** (vs C fc 3.07×, zstd-9 2.09×). Outright
      ratio wins vs both on linear (96×), piecewise (83×), int-x1000 (7127×);
      DELTA_DP took parabolic 23.9→2006×.

## Prioritized plan (canonical next-steps order)

Goal: match/beat `fc` on ratio while being faster on encode **and** decode, in
safe portable Rust. Sequencing matters — gating comes first because it removes
the throughput tax that every additional mode otherwise imposes.

**Tier 1 — biggest gaps; do in order:**

1. [x] **Feature-based mode gating** — done. `probe_block_features`
   (`exp_range`, sampled `distinct`, repeat detection) gates LZ and
   byte-transpose. **Encode 2.5–4.8× faster, ratio unchanged (3.00×)**, winners
   verified identical. Room to extend gating to more families later.
2. [x] **Adaptive block sizing** — done (`plan_blocks`, 256 KiB→1 MiB). constant
   14979→55188× (beats fc), dict-16 →13797× (beats fc), decimal/quantized up.
3. [x] **`FLOAT_MULT`** — done. stocks 7.3→18.0× (beats fc 15.6×), quantized
   →3666× (beats fc). Aggregate **3.05×** (vs fc 3.07×). Now beat fc on 7 of 17.

**Tier 2 — strong follow-ups:**

4. [~] **Prefer tANS over RC when close** — investigated & deferred. No-op at
   safe margins (RC's order-1 beats order-0 tANS by >6% on the noisy
   byte-transpose streams); larger margins cost real ratio. Faster
   decode-on-noisy needs a faster range decoder or order-1 tANS.
5. [ ] **`ALP`** (Adaptive Lossless FP) — strong general-purpose FP codec.
6. [ ] **`LSB_STRIP`** / smarter byte-transpose-plane entropy — close the noisy
   long-tail (climate/sensor/ar2/random-walk, where `fc` is ~2–3% ahead).

**Tier 3 — lower ROI / specialized:**

7. [ ] `CONV_N` (N-tap linear predictor) for smooth/periodic (sin, audio).
8. [ ] Real SIMD (AVX2-gather predictor, explicit-SIMD transpose) — **hold**:
   gather feeds FCM which wins nothing here; transpose isn't the bottleneck.
9. [ ] Remaining `fc` modes (BWT, PAQ, ELF, PRED variants) — long-tail.
10. [ ] ARM/NEON path; C ABI (scoped in `TODO.md`); bitplane mode.

## Building blocks to port next

These unlock most of the remaining modes:

- [x] **Bit reader/writer** (`bitio`, MSB-first) — done.
- [x] **tANS** (table ANS, 8-bit symbols) — done; competes with RC per block.
      Still to leverage for `FUZZY_STRIDE_ANS`, `BWT_MTF_TANS`.
- [x] **Binary range coder** — done; reused by `BWT_MTF_RC`, future direct models.
- [x] **DFCM predictor** (`PRED2`) — done. Still: 2-way set-associative `PRED4`.
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

- [ ] Adaptive block sizing (256 KiB–1 MiB quantum probe, like `fc`). Would help
      `constant` (fewer headers) toward `fc`'s 39756×.
- [x] LZ / RLE / dictionary mode — done (`LZ`, hash-chain LZ77 + entropy).
      dict-16 3519×, quantized 1108×, stocks 6.9×; `decimal-cents` (122×) still
      trails zstd — wants a bigger window / better 1024-value dictionary.
- [x] Multi-threaded encode + decode — done (rayon).
- [x] Benchmark harness vs. zstd / the C `fc` — done (`examples/compare.rs`)
      plus criterion kernel benches (`benches/kernels.rs`).
- [x] **Decoder robustness / fuzzing** — done. `tests/robustness.rs` (stable
      randomized) + `fuzz/` (cargo-fuzz: `decompress`, `roundtrip`). Fixed three
      crash/DoS vectors (tANS model validation, `predictor_log2` range, a
      decompression bomb via oversized block counts).
- [x] Lossless double-precision delta (`DELTA_DP`) — done; parabolic 23.9→2006×.
- [x] Byte-transpose mode using `multiversion` — done (replaced the dead demo).
      Note: LLVM doesn't autovectorize the transpose; an explicit-SIMD rewrite
      (core::arch / std::simd / macerator) is the remaining upgrade, low ROI
      since the transpose isn't the bottleneck.
- [ ] **bitplane** split mode (finer-grained than byte-transpose).
- [ ] Adaptive block sizing (256 KiB–1 MiB quantum probe, like `fc`). Would help
      `constant` (fewer headers) toward `fc`'s 39756×.
- [ ] Feature-gated mode selection (block stats decide which modes to try),
      mirroring `fc`'s `exp_range` / `sign_flips` / `distinct_count` gates.
- [ ] ARM/NEON path for the hot kernels.
- [ ] Diagnostics counters (`fc_enc_mode_wins` equivalent).
- [ ] Optional `fc`-wire-compatible profile for cross-testing against the C lib.
