# Roadmap

`quoin` started as a Rust port of the `fc` floating-point compressor and is
evolving into a **type-aware columnar codec engine**: a per-block competition of
lightweight, type-specialized codecs with adaptive sizing, fast SIMD bit-packing,
and a fuzz-hardened safe decoder. The compression-landscape survey (Vortex,
BtrBlocks, FastLanes, ALP, pcodec) and the prioritized "steal list" live in
[`docs/LANDSCAPE.md`](docs/LANDSCAPE.md).

## Done

**Framework**
- [x] Safe public API (`compress`/`decompress`/`Config`/`decompressed_len`).
- [x] Stream format v1 (16-byte header + per-block frames), bounds-checked decode.
- [x] **Adaptive block sizing** (`plan_blocks`, 256 KiB base → 1 MiB low-entropy).
- [x] **Feature-based mode gating** (`probe_block_features`: exp-range, sampled
      distinct, repeats) — 2.5–4.8× faster encode, ratio unchanged.
- [x] **Two selection strategies** (`Config.selection`): `Full` (encode all, keep
      smallest) and **`Sample`** (estimate on a stratified sample, encode only the
      winner — BtrBlocks/Vortex style). Sample is ~5–9× faster encode for ~2%
      ratio. A/B-able via `FCBENCH_SELECT=sample`.
- [x] **Block-parallel** encode/decode (rayon, default-on `parallel` feature).
- [x] Diagnostics: per-mode win counters (`mode_win_counts`).

**Entropy & SIMD substrate**
- [x] **Binary range coder** (LZMA-style, order-1) + **tANS** on MSB-first bit I/O;
      residuals pick the smaller.
- [x] **CRC32C FCM hash** — hardware `_mm_crc32_u64` + bit-exact software fallback,
      runtime-dispatched.
- [x] **FastLanes-style bit-packing** (`src/bitpack.rs`) — 1024-value lane-transposed
      pack/unpack, `multiversion`-dispatched, **verified to autovectorize** (AVX2
      clone emits `ymm`, unlike byte-transpose); ~30 GiB/s.

**Codecs (16)**
- [x] Float/general: `RAW`, `CONST`, `STRIDE`, `XORZ`, `PRED` (FCM), `PRED2` (DFCM),
      `PRED_RC`, `DELTA2` (float-linear), `ORDERED_DELTA` (int 2nd-delta),
      `DELTA_DP` (exact float-residual delta), `LZ` (LZ77), `BYTE_TRANSPOSE`
      (= shuffle / BYTE_STREAM_SPLIT), `FLOAT_MULT`.
- [x] Integer/decimal (columnar foundation): **`FOR_BITPACK`** (frame-of-reference
      + bit-pack), **`ALP`** main scheme (scaled-int decimals + exceptions),
      **`DELTA_BITPACK`** (delta → FoR+bitpack cascade = Parquet DELTA_BINARY_PACKED).

**Quality**
- [x] Round-trip tests (incl. NaN/±0/inf), `tests/robustness.rs`, and `fuzz/`
      (cargo-fuzz) — fixed 3 decoder crash/DoS vectors.
- [x] Harnesses: `examples/compare.rs` (ratio + throughput vs zstd / C `fc`),
      criterion kernel benches (`benches/kernels.rs`).
- [x] **C ABI** (`capi/` crate: cdylib + staticlib, context handle, catch_unwind,
      C round-trip test). See [`TODO.md`](TODO.md).

## Results (bundled harness, 1 Mi values/dataset, 18-core box)

- **f64 suite (17 datasets): aggregate 3.05×** vs C `fc` 3.07×, zstd-9 2.09× — an
  **8–8 split** with `fc` (+1 tie). Outright wins: constant 55,188× (beats fc),
  linear 96×, piecewise 218×, int-x1000 7,127×, decimal 487×, dict-16 13,797×,
  quantized 3,666×, stocks 18.0×. `fc` edges parabolic/sin/geo and the noisy
  long-tail (~2%, near entropy floor).
- **Integer/decimal columns** (demos): `int-narrow` (bounded ids) → FOR_BITPACK
  5.30×; `timestamps` (irregular) → DELTA_BITPACK 4.88×; `decimal-outliers`
  → ALP 1.83× (where FLOAT_MULT bails on a single outlier).
- Encode several× faster post-gating; decode beats `fc` on structured data
  (parallel + tANS), trails on the noisy datasets (sequential range decode).

## Next (the columnar arc, prioritized)

1. [ ] **ALP-RD** — the "real doubles" split-dictionary scheme (left=dict, right=
       bitpack) for non-decimal floats; could beat byte-transpose on the noisy
       long-tail. Algorithm captured in `docs/LANDSCAPE.md`.
2. [ ] **PFOR patching** — move range-outliers to exceptions in FOR_BITPACK/ALP so
       one large value doesn't widen (or raw-fall-back) a whole sub-block.
3. [ ] **Dictionary** + **RLE** encodings (explicit, recursive like BtrBlocks) —
       needed for low-cardinality and, later, strings.
4. [ ] **Typed column API** — generalize off `f64` to `u32`/`u64`/`i*` lanes + a
       type tag, so the integer codecs operate on real columns, not f64 bits.
5. [ ] **Nulls / validity** as a first-class compressed stream (Arrow needs it).
6. [ ] **Arrow adapter** (feature-gated) — map `arrow::Array` → (type, values,
       validity) → codecs; benchmark vs Parquet/Vortex.
7. [ ] **Compute-on-encoded / predicate pushdown** (filter/take over encoded
       arrays) — Vortex's killer feature; the thing that makes us useful *inside*
       a query engine (DataFusion or any columnar store).

## Future ideas (worth considering)

- **Two-stage option** (Parquet/ClickHouse model): lightweight encodings that keep
  random-access + pushdown, with optional LZ4_RAW/ZSTD page compression on top —
  vs our current "own range/tANS coder per stream". Better for DB-storage layers.
- **bitshuffle** (bit-level transpose) as a generic preprocess; note: overlaps our
  bit-packing layout and competes with explicit cascades for the same redundancy.
- **Cascading expression model** (FastLanes RPN / Vortex scheme trait) — generalize
  the flat mode list into composable encodings (dict→bitpack, RLE→recurse).
- **u64 bit-packing** variant (current substrate is u32) for wide integer columns.
- **FSST** for strings; **CONV_N** linear predictors for smooth/periodic floats;
  **order-1 tANS** for faster noisy-data decode; **ARM/NEON** hash path.
- Faster range decoder, or replace with interleaved rANS for SIMD decode.

## Known gaps / honest notes

- The integer/decimal codecs win nothing on the `f64` suite (float bit patterns
  aren't FoR/delta-friendly) — they're for real integer/decimal columns.
- `Sample` selection under-ranks LZ-style long-range structure unless the
  `distinct_low`/`repeats` feature routes it to the full block (already handled).
- Real wide-SIMD lives only in `crc32` + the bit-packing layout; the sequential
  hot loops (predictors, entropy coders) don't vectorize and that's inherent.
