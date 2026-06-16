# Landscape: competitors, inspirations, and what to steal

Reference for evolving `quoin` from an f64 compressor into a type-aware columnar
codec engine. Compiled from reading the source of the projects below
(cloned/read June 2026) plus public knowledge of the columnar DBs.

## Where quoin sits today vs. where this points

`quoin` today: a per-block **codec competition** for `f64` — it *fully encodes*
each block with every applicable mode and keeps the smallest, gated by cheap
block features. That's the right idea but the expensive variant of it. The
research lineage below (CWI/DuckDB: BtrBlocks → FastLanes → ALP, and Vortex,
pcodec) shows the mature shape: **cascading lightweight schemes, chosen by
sampling-based estimation, over typed columns, with compute pushdown on the
encoded form.**

---

## Tier 1 — direct architecture targets (study hard)

### Vortex — the architecture to grow into
- **What:** Rust, Apache-2.0, production (SpiralDB; LF AI & Data). Arrow-compatible
  columnar format with cascading compression. Used via DuckDB/Spark/DataFusion/Polars.
- **Model:** unified `Scheme` trait across all types; **cascading** encodings wrap
  encodings up to `MAX_CASCADE=3` (e.g. FoR → BitPacking). Schemes declare children
  and recurse; push/pull exclusion rules prevent bad chains.
- **Selection:** two-pass — (1) cheap heuristic *verdicts* (`AlwaysUse` for Constant,
  or a `Ratio`), then (2) **deferred ~1% stratified sampling** only if needed, with
  threshold-aware early-exit. Not full-encode-everything.
- **Encodings:** ints (Constant, Dict, FoR, BitPacking[FastLanes], ZigZag, Sparse,
  RunEnd, Sequence, RLE), floats (ALP, ALP-RD, Dict, RLE, Sparse), strings (Dict,
  FSST, Zstd), bool, datetime-parts, decimal-byte-parts; nulls = compressed bool.
- **Killer feature:** **compute kernels over encoded arrays** (`take`/`filter`/
  `compare`/`slice`) → query pushdown without decompressing. Chunked arrays;
  chunk boundaries set by the caller, not the compressor.
- **Steal:** the whole shape — cascading scheme trait, two-pass+sampling selection,
  compute-on-encoded, caller-controlled chunks.
- Path: `/tmp/quoin-research/vortex` · `vortex-compressor/src/compressor.rs`,
  `vortex-btrblocks/src/builder.rs`.

### BtrBlocks — the research basis for selection-by-sampling
- **What:** C++, MIT, SIGMOD 2023 (CWI/TUM). The canonical "cascade lightweight
  schemes, pick by sampling" system.
- **Selection (the key bit):** sample **10 partitions × 64 elements**, compress the
  sample with each candidate, pick highest *estimated* ratio — then fully encode only
  the winner. A `ThreadCache.estimation_level` flag stops nested schemes from
  re-sampling (top level samples; deeper levels use full intermediate data).
- **Cascading:** depth 3; e.g. RLE emits (run-values, run-lengths) and **recursively
  scheme-picks each stream independently**; FoR → PFOR → BP.
- **Schemes:** ints (One-value, Dict, RLE, PFOR, BP/FoR-bitpack, Frequency, FOR),
  doubles (+ **Pseudodecimal** = exponent/decimal-mantissa + exception bitmap),
  strings (Dict, FSST). Block = 65 536 tuples. Nullmap via Roaring bitmap.
- **Steal:** sampling-based estimation (the fix for our full-encode cost),
  recursive per-stream cascading, Frequency scheme, Pseudodecimal.
- Path: `/tmp/quoin-research/btrblocks` · `compression/SchemePicker.hpp`,
  `stats/NumberStats.hpp`.

### FastLanes — autovectorizable bit-packing (solves our SIMD problem)
- **What:** C++, MIT, VLDB 2023 (CWI). "Decode >100B ints/s with scalar code."
- **The crown jewel:** the **1024-value unified transposed layout** — values are
  interleaved so that *plain scalar* bit-(un)pack loops over fixed strides
  **autovectorize to AVX2/AVX-512/NEON with no intrinsics.** This is exactly the
  fix for our finding that byte-transpose doesn't autovectorize: the *layout* is the
  optimization, not hand-SIMD.
- **Also:** RPN "expression" cascading (TRS→ANALYZE→FFOR etc.), per-bitwidth
  specialized kernels, patching (LL/S/B) for outliers, FFOR/Dict/RLE/Delta.
- **Steal:** the transposed bit-packing layout — adopt it as quoin's integer
  bit-pack/unpack substrate so packing is both small *and* fast on any ISA. Rust
  ports exist; the layout is reimplementable.
- Path: `/tmp/quoin-research/FastLanes` · `src/include/fls/cfg/cfg.hpp` (VEC_SZ=1024),
  generated kernels under `src/primitives/fls_generated/`.

### ALP — the floating-point codec to reimplement
- **What:** C++, MIT, SIGMOD 2024 (CWI). Two schemes; vector=1024, rowgroup=100 vectors.
- **ALP (decimal doubles):** `digit = round(v · 10^e · 10^-f)`; two-level sampling
  (rowgroup picks top-K `(e,f)` pairs from 256 samples; per-vector picks best of K),
  round-trip **exception** detection, then **FOR + bit-packing**. Magic-number rounding
  trick; exponent/factor tables.
- **ALP-RD (real doubles):** split each double at bit `k` (try k∈[1,16]); **dictionary**
  the high "left" part (≤8 entries, 3-bit index) + exceptions; bit-pack the low "right"
  part. This is the principled version of what we sketched for noisy floats.
- **Switch:** if ALP's estimated cost ≥ threshold (48 b/val·32 samples) → use ALP-RD.
- **Steal:** reimplement both faithfully (we captured the constants). Replaces our
  ad-hoc FLOAT_MULT/DELTA_DP with the state-of-the-art.
- Path: `/tmp/quoin-research/ALP` · `include/alp/{encoder,rd,constants,sampler}.hpp`.

### pcodec (pco) — best-in-class *numerical* compression, Rust
- **What:** Rust, Apache-2.0, v1.0 (mwlon). f16/f32/f64 + all int widths. Different
  philosophy from us: **mode → delta → binning → tANS**, not predictor-competition.
- **Modes (auto-detected by sampling):** Classic; **IntMult** (`base·mult+adj`, finds a
  shared multiplier via GCD + z-test significance); **FloatMult** (detect float base via
  trailing-zeros *and* **approximate Euclidean GCD**, then **snap** to nice numbers like
  1/100 or 10ⁿ); **FloatQuant** (drop k low mantissa bits when over-precise).
- **Binning:** quantize each latent into histogram bins, store (bin, offset), tANS the
  bins — efficient for smooth distributions. Delta orders 1–7 / lookback (LZ77-ish) /
  Conv1 auto-chosen. Chunk≤256K / page / batch=256 (caller-friendly).
- **Steal:** the **FloatMult auto-detection** (far better than our fixed decimal
  scales) and the binning+tANS latent model as an alternative codec family.
- Path: `/tmp/quoin-research/pcodec` · `pco/src/mode/{int_mult,float_mult,float_quant}.rs`.

---

## Tier 2 — columnar DBs (how they compress; benchmark/inspiration)

- **DuckDB** — the CWI/DuckDB-Labs lineage productionized. Storage uses lightweight
  per-rowgroup schemes chosen by an analyze pass: Constant, RLE, Bit-packing, Dictionary,
  **FoR**, **Chimp/Patas** then **ALP** for floats, **FSST** for strings. Closest
  "type-aware lightweight" reference. (We already FFI-compare against C `fc`; DuckDB is
  the better *columnar* baseline to benchmark against.)
- **ClickHouse** — composable per-column **codecs** in DDL: `Delta`, `DoubleDelta`
  (timestamps), `Gorilla` (floats), `T64` (bit-transpose for ints/time), `FPC`, then
  general `LZ4`/`ZSTD`. Good source of the classic time-series codecs.
- **Apache Parquet** (arrow-rs `parquet`) — encodings: PLAIN, RLE_DICTIONARY,
  `DELTA_BINARY_PACKED` (ints), `DELTA_BYTE_ARRAY` (strings), and **`BYTE_STREAM_SPLIT`**
  for floats (byte-plane split — *our byte-transpose mode is this idea*), + page codec
  (Snappy/ZSTD). The interop baseline everyone reads/writes.
- **Apache ORC** — RLE v1/v2 for ints, dictionary for strings, + zlib/zstd.
- **InfluxDB (IOx/TSM)** — Gorilla-style: delta-of-delta timestamps, XOR floats, Simple8b
  ints, RLE. The time-series canon.
- **Lance** (lancedb, Rust) — columnar format for ML/AI; v2 "structural encoding" with
  bit-packing, FSST, etc. Another Rust columnar target to compare against.
- **Apache Arrow** itself — in-memory uncompressed, but RunEnd/Dictionary arrays and
  IPC LZ4/ZSTD buffer compression; our Arrow adapter must speak its validity bitmaps &
  buffer layout.

## Tier 3 — floating-point / scientific (context & baselines)

- **fpzip**, **zfp** (LLNL) — predictive (Lorenzo) / transform FP array compressors;
  fpzip is the lossless baseline (already in the `fc` benchmark). zfp is mostly lossy.
- **Gorilla / Chimp / Patas / Elf / SPDP / FPC** — the FP-codec lineage; several overlap
  modes we already ported from `fc`. ALP supersedes most for columnar use.

---

## The prioritized "steal list" (drives the columnar roadmap)

1. **Selection by sampling, not full-encode** *(biggest architectural fix).* Replace
   "encode with every mode, keep smallest" with BtrBlocks/Vortex-style: estimate each
   candidate's ratio on a ~1% stratified sample, fully encode only the winner. Cuts
   encode cost *and* removes the per-mode throughput tax that currently caps how many
   modes we can add.
2. **FastLanes 1024-transposed bit-packing** as the integer substrate — small *and*
   autovectorizing on any ISA (fixes our SIMD finding).
3. **ALP + ALP-RD** for floats (faithful reimpl; retire ad-hoc FLOAT_MULT/DELTA_DP or
   keep as fast-path).
4. **Cascading scheme model** (depth ~3) with a unified trait + recursion (RLE splits
   streams and recompresses each), à la BtrBlocks/Vortex — generalizes our flat mode list.
5. **pcodec FloatMult auto-detection** (trailing-zeros + approx-GCD + snap) to upgrade
   integer/decimal scale detection.
6. **Integer schemes**: FoR, dictionary, RLE(+recurse), Frequency, PFOR/patching.
7. **Compute-on-encoded kernels** (filter/take/compare) for DB pushdown — the feature
   that makes us useful *inside* a query engine, not just a blob codec.
8. **Nulls/validity** as a first-class compressed stream (Roaring/RLE), required for Arrow.
9. **FSST** + dictionary for strings (later phase).

## Benchmark targets

Ratio + enc/dec throughput, per Arrow type, against: **pco** (numeric, Rust — direct),
**Parquet** encodings (`BYTE_STREAM_SPLIT`, `DELTA_BINARY_PACKED`, RLE/dict via arrow-rs),
**Vortex** (Rust, same niche), **DuckDB** storage, **ClickHouse** codecs, **zstd/lz4**
(general baseline), and the C **`fc`** (current FP baseline). Reuse our `examples/compare.rs`
harness, extended to typed columns.
