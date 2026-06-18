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
- [x] **Speed/ratio levels** (`Config.level`: `Fastest`/`Fast`/`Balanced`/`High`/`Max`):
      cost-aware selection `argmin(size + λ·decode_cost)` + per-level entropy-coder
      gate. `Max` is the default and bit-identical to the historical output. See
      [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).
- [x] Diagnostics: per-mode win counters (`mode_win_counts`).

**Entropy & SIMD substrate**
- [x] **Binary range coder** (LZMA-style, order-1) + **rANS** interleaved coder;
      residuals pick the smaller.
- [x] **CRC32C FCM hash** — hardware `_mm_crc32_u64` + bit-exact software fallback,
      runtime-dispatched.
- [x] **FastLanes-style bit-packing** (`src/bitpack.rs`) — 1024-value lane-transposed
      pack/unpack, `multiversion`-dispatched, **verified to autovectorize** (AVX2
      clone emits `ymm`, unlike byte-transpose); ~30 GiB/s. Both a `u32` kernel
      (widths 0..=32) and a `u64` kernel (`pack64`/`unpack64`, 0..=64) for wide
      integer columns; `FOR_BITPACK` packs wide ranges instead of raw-fallback.

**Codecs (20 modes)**
- [x] Float/general: `RAW`, `CONST`, `STRIDE`, `XORZ`, `PRED` (FCM), `PRED2` (DFCM),
      `PRED_RC`, `DELTA2` (float-linear), `ORDERED_DELTA` (int 2nd-delta),
      `DELTA_DP` (exact float-residual delta), `LZ` (LZ77), `BYTE_TRANSPOSE`
      (= shuffle / BYTE_STREAM_SPLIT), `FLOAT_MULT`.
- [x] Integer/decimal (columnar foundation): **`FOR_BITPACK`** (frame-of-reference
      + bit-pack), **`ALP`** main scheme (scaled-int decimals + exceptions),
      **`ALP_RD`** (real-double split-dict), **`DELTA_BITPACK`** (delta → FoR+bitpack
      cascade = Parquet DELTA_BINARY_PACKED), **`DICT`**, **`RLE`**.
- [x] **`PCO`** — vendored [pcodec](https://github.com/mwlon/pcodec) numeric backend
      (`vendor/pco`, gated to `High`/`Max`); three vectorizable decode leaves
      annotated `#[multiversion]` so a stock build gets the SIMD fast path without
      `-C target-cpu=native`. Adds a `High` level between `Balanced` and `Max`.

**Quality**
- [x] Round-trip tests (incl. NaN/±0/inf), `tests/robustness.rs`, and `fuzz/`
      (cargo-fuzz) — fixed 3 decoder crash/DoS vectors.
- [x] Harnesses: `examples/compare.rs` (ratio + throughput vs zstd / C `fc`),
      criterion kernel benches (`benches/kernels.rs`).
- [x] **C ABI** (`capi/` crate: cdylib + staticlib). Global-pool
      `quoin_compress`/`quoin_decompress` **and** an opaque context handle
      (`quoin_ctx_create(threads)` / `*_ctx` / `quoin_ctx_free`) owning a persistent
      rayon pool for bounded-thread, churn-free calls (like `ZSTD_CCtx`). Every
      entry point `catch_unwind`s (no unwind across `extern "C"`); errors are codes;
      caller-sized buffers + `quoin_compress_bound`; `cbindgen` header. Fork-without-exec
      and `dlclose`-while-threads-alive caveats documented in the header.
- [x] **Typed, Arrow-native C ABI** (`capi/src/typed.rs`): `QuoinDType` for
      Bool/I8–64/U8–64/F32/F64/Decimal32–256, decode-into-caller-buffer,
      alignment-safe with a zero-copy fast path when the input is aligned. Narrow
      types widen to existing lanes (I8/16→I32, U8/16/Bool→U32, Dec32/64→Dec128).

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
  (parallel + rANS), trails on the noisy datasets (sequential range decode).

## Next (the columnar arc, prioritized)

1. [x] **ALP-RD** — the "real doubles" split-dictionary scheme (`alp_rd.rs`):
       cut-search + 8-entry left dictionary + exceptions, wide `right` bit-packed.
       Real-double columns now compress (and decode fast) instead of RAW.
2. [x] **Dictionary** + **RLE** encodings (`dict.rs`, `rle.rs`). DICT compresses
       both streams: codes (byte-plane entropy cascade) **and values** (sorted +
       delta / transpose / entropy) — the value compression lifted the aggregate
       to **2.70×** (beats zstd-9) since the raw dictionary dominated high-card
       columns.
3. ◑ **Cascade + rANS** — ✅ **interleaved rANS** (`entropy/rans.rs`, 4-way),
       cost-chosen with the range coder/rANS: q-balanced decodes ~40% faster than
       q-max. ✅ **dict→entropy cascade** for ≤256-cardinality blocks (basel_temp
       2.13→3.22×). **Pending:** a wider-alphabet cascade for high-cardinality
       columns (`medicare1`, ~46 K distinct/block), and `ALP digits → entropy`.
4. [ ] **PFOR patching** — move range-outliers to exceptions in FOR_BITPACK/ALP so
       one large value doesn't widen a whole sub-block.
4. [x] **Typed column API** — generalize off `f64` to real typed columns + a type
       tag, so the integer codecs operate on real columns, not f64 bits.
       `DType`/`ColumnRef`/`Column` + `compress_column`, format v2 carries the
       column type, family-aware gating, **64-bit** (`F64`/`I64`/`U64`), **32-bit**
       (`I32`/`U32`/`F32`) lanes with signedness-aware FoR and width-aware RAW, and
       **`Decimal128`/`Decimal256`** containers (scale/precision preserved). Narrow
       `8/16`-bit are widened at the C-ABI boundary (no native lane yet). See
       [`docs/TYPES.md`](docs/TYPES.md).

   Benchmarked against zstd/lz4/deflate on the ALP float corpus — see
   [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md). The losses there set the order
   below: **ALP-RD** (real doubles fall back to RAW), **dictionary+RLE**
   (repeated-value columns), and **cascading the ALP/FLOAT_MULT digits**.
5. [x] **Nulls / validity** — Arrow-compatible bitmap (`src/validity.rs`), nulls
       compacted out of the value stream, bitmap stored RLE/raw. `compress_column`
       takes `Option<&[u8]>`; `decompress_column` returns values + validity.
6. [x] **Arrow adapter** (`src/arrow.rs`, feature `arrow`) — `compress_array`/
       `decompress_array` for primitive numeric **and `Decimal128`/`Decimal256`**
       arrays with validity (incl. sliced), zero-copy from Arrow buffers.
       Benchmarked vs Parquet-zstd (`examples/real_parquet.rs`) and Vortex
       (`examples/vs_vortex.rs`). Pending: temporal types.
7. [ ] **Compute-on-encoded / predicate pushdown** (filter/take over encoded
       arrays) — Vortex's killer feature; the thing that makes us useful *inside*
       a query engine (DataFusion or any columnar store).

## Future ideas (worth considering)

- **Two-stage option** (Parquet/ClickHouse model): lightweight encodings that keep
  random-access + pushdown, with optional LZ4_RAW/ZSTD page compression on top —
  vs our current "own range/rANS coder per stream". Better for DB-storage layers.
- **bitshuffle** (bit-level transpose) as a generic preprocess; note: overlaps our
  bit-packing layout and competes with explicit cascades for the same redundancy.
- **Cascading expression model** (FastLanes RPN / Vortex scheme trait) — generalize
  the flat mode list into composable encodings (dict→bitpack, RLE→recurse).
- **u128 bit-packing** variant (substrate is now u32 + u64) for `Decimal128`.
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
