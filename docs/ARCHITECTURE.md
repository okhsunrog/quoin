# Architecture

How quoin works internally. For *what* it is and *how to use it*, see the
[README](../README.md); for status and plans, [ROADMAP](../ROADMAP.md).

---

## 1. Overview

quoin is a lossless compressor for **typed columns of numbers**. The core idea is
a **per-block competition**: a column is split into independent blocks, and for
each block every applicable codec encodes it; the smallest output wins. Because
the column's *type* is known, only codecs that make sense for that type compete.

Two design decisions shape everything:

1. **One lane abstraction, two widths.** Every typed column is lowered — by a
   zero-copy reinterpret — to a physical lane word: `u64` for the 64-bit types,
   `u32` for the 32-bit types (decimals to a wider container). The codecs are
   written once, generic over the [`Lane`](../src/lane.rs) trait, so a 32-bit
   column is packed, hashed, delta-coded, transposed and float-predicted at its
   **native width**; the [`DType`] selects which *family* of codecs may compete
   and is recorded in the stream so the decoder restores the original type.
2. **Independent blocks.** Each block carries its own winning codec and entropy
   coder. This is what makes encode/decode parallel and gives random access at
   block granularity.

```
        ┌─ compress ──────────────────────────────────────────────┐
typed   │  ColumnRef ──► lane (u32/u64) ► plan blocks ──► per block:│   byte
column  │  (zero-copy   (DType,         (adaptive      competition │   stream
 (+nulls)│   reinterpret) validity)     sizing)        → frame     │  (header
        │                                              ◄───────────┤   + frames)
        └──────────────────────────────────────────────────────────┘
        ┌─ decompress ─────────────────────────────────────────────┐
byte    │  header ──► frames (parallel) ──► per frame: mode decode  │   typed
stream  │             ──► lane words ──► reinterpret as DType ──► validity │ column
        └──────────────────────────────────────────────────────────┘
```

---

## 2. Module map

| Module | Responsibility |
| --- | --- |
| `lib.rs` | Public API (`compress`/`decompress`, `compress_column`, `Config`, `Level`, `DType`, `ColumnRef`). |
| `dtype.rs` | `DType` and `Family` (Int/Float); lane width and signedness per type. |
| `encoder.rs` | Block planning + the **mode competition** (`encode_block_full`, the scoring). |
| `decoder.rs` | Frame parsing and per-mode decode dispatch. |
| `mode.rs` | The `Mode` enum (codec IDs) and `mode_name`. |
| `format.rs` | Stream header + frame layout, block-size constants. |
| `codecs/` | One module per codec (see [§7](#7-codecs)). |
| `entropy/` | `rc` (range coder), `rans` (rANS), the `code_residuals` cascade, the order-1 size estimator. |
| `bitpack.rs` | FastLanes transposed bit-pack/unpack kernels (`u32` and `u64`). |
| `transform.rs` | Lane-wise maps (byte-transpose), `multiversion`-dispatched. |
| `hash.rs` | CRC32C predictor hash (hardware intrinsic + scalar fallback). |
| `decimal.rs` | `Decimal128`/`Decimal256` limb-split container. |
| `validity.rs` | Arrow-style null bitmap (compaction + RLE/raw storage). |
| `arrow.rs` | Arrow `Array` adapter (feature `arrow`). |
| `vendor/quoin-pco/` | Vendored pco (pcodec) fork — the numeric backend. |
| `capi/` | C ABI (cdylib + staticlib), separate crate. |

---

## 3. Lanes & types

The codec engine sees a slice of lane words — `&[u64]` or `&[u32]` — through the
`Lane` trait (`lane.rs`), which also exposes the lane's float view (`f64`/`f32`)
for the float-value codecs. Lowering a typed column (`compress_column`):

| Input type | Lane | How |
| --- | --- | --- |
| `f64` / `i64` / `u64` | `u64` | **zero-copy** bit-cast of the slice (no allocation) |
| `f32` / `i32` / `u32` | `u32` | **zero-copy** bit-cast of the slice (no allocation) |
| `Decimal128/256` | 128/256-bit | dedicated `decimal.rs` container (not a lane) |

Nothing is widened: a 32-bit column's constants, dictionary entries, FoR minima
and byte planes are 4 bytes on the wire, its predictor tables hold `u32`, its
ALP/FLOAT_MULT/DELTA2 arithmetic is `f32` with `f32` constants (ALP exponents to
10, magic `1.5·2^23`), and pco is handed the `&[f32]`/`&[i32]`/`&[u32]` slice as
is. Block quanta are counted in **bytes** (256 KiB base, 1 MiB max), so a 32-bit
lane holds twice the values per block. Every `f32` bit pattern round-trips
exactly — NaN payloads and signaling bits included: the float-arithmetic codecs
either verify each value's reconstruction bit-for-bit (ALP, FLOAT_MULT,
DELTA_DP) or are skipped for a block containing any inf/NaN (DELTA2/DELTA_DP),
so no path depends on the platform's NaN-propagation rules.

`Family` (from `DType`) gates the competition: `Float` unlocks the float-value
codecs (ALP, ALP-RD, FLOAT_MULT) and the float predictors; `Int` runs
frame-of-reference and the signed-delta cascade. Type-agnostic codecs (RAW, CONST,
bit-packers, LZ, transpose, dict, pco) apply to both.

---

## 4. Block framing

A column is split by `plan_blocks` (`encoder.rs`). The **ratio-first levels**
(`High`/`Max`) use full ~1 MiB blocks outright — the wider LZ/dict/entropy
window never lost ratio on the corpus (and won up to +14% where the probe
under-grew), and coarse random access is exactly what those levels trade away.
The fast levels keep an adaptive base of ~256 KiB that grows toward ~1 MiB only
when a cheap probe says the data is low-entropy (dictionary-like / constant /
run-heavy), preserving fine-grained random access and parallelism on noisy
data. A fixed `Config.block_size` overrides both (for storage-chunk alignment /
random access).

Each block becomes a **frame**: `[mode byte | payload]`. An unknown mode byte is a
hard decode error, never a silent zero-fill.

---

## 5. The mode competition (encode)

`encode_block_full` is the heart. It maintains a `Best` (smallest scorer so far,
plus the runner-up mode) and runs codecs roughly in **decode-cost order**, so a
tight incumbent is established early:

1. **Structural / cheap** — RAW (the baseline every mode must beat), CONST,
   STRIDE, XORZ.
2. **Cheap strong** — FoR+bitpack, delta+bitpack, (float) FLOAT_MULT / ALP /
   ALP-RD, and DICT_SHARED when the column-wide table exists. These give a
   tight `best` *before* the expensive predictors, which is what lets the
   estimate gate (below) prune.
3. **Predictors** (`High`/`Max` only) — FCM/DFCM, polynomial-float, 2nd-order int.
   Each is gated by `coded_if_competitive`: an **order-1 entropy estimate** of its
   residual; the full range-code is skipped when even an optimistic estimate can't
   beat the incumbent. (This is why `Max` encode isn't dominated by range-coding
   every candidate.)
4. **Dict / RLE / byte-transpose** — gated by a compressibility probe.
5. **pco** (`Balanced`+) — the heavyweight numeric backend.
6. **LZ cascade** (`Max` only) — applied to the **top-2** base winners only (not
   every candidate), via `encode_mode`. This is the dominant encode cost, so
   running it on two modes instead of ~8 is a large speedup at ~zero ratio cost.

### Scoring

Each candidate scores `payload_size + penalty(mode, λ, decoded_bytes)`, where
`penalty = (λ · decode_weight(mode) · decoded_bytes) >> 8`. `λ` comes from the
[level](#9-levels): `0` at `High`/`Max` → pure size; higher `λ` at the fast levels
biases toward cheap-to-decode modes. `decode_weight` is a per-mode relative
decode-cost class.

### Selection strategies (`Config.selection`)

- **`Full`** (default) — run the competition above.
- **`Sample`** — rank modes by their size on a stratified sample, fully encode
  only the winner (BtrBlocks/Vortex style). Much faster encode, slight ratio risk.

---

## 6. Decode (reverse)

`decode_frames` (`decoder.rs`) scans frame boundaries, then decodes frames in
parallel. Per frame: read the mode byte → dispatch to that codec's `decode` →
producing lane words → reinterpret in place as the column `DType` → reattach
validity. The
decoder is fuzz-hardened: every length/offset is bounds-checked, and a corrupt
stream returns `Error`, never panics or over-allocates.

---

## 7. Codecs

21 block modes, by family (`mode.rs`, `codecs/`):

| Group | Modes | Notes |
| --- | --- | --- |
| Structural | `Raw`, `Const`, `Stride`, `Xorz` | O(n), trivial; RAW is the baseline. |
| Integer bit-pack | `ForBitpack`, `DeltaBitpack` | frame-of-reference / delta + FastLanes bit-pack. Random-access, fast decode. |
| Float value | `Alp`, `AlpRd`, `FloatMult` | doubles that are really decimals → scaled integers (ALP) or split-dictionary (ALP-RD). |
| Predictors | `Pred`, `PredRc`, `Pred2`, `Delta2`, `DeltaDp`, `OrderedDelta` | FCM/DFCM hash + XOR residual; polynomial-float; 2nd-order int. Sequential decode. |
| Dictionary | `Dict`, `DictShared`, `Rle` | low-cardinality / run-heavy; `DictShared` codes into the **column-wide shared table** (see below). |
| Generic | `ByteTranspose`, `Lz` | AoS→SoA byte planes; LZ77 over the block. |
| Numeric backend | `Pco` | vendored pcodec — latent decomposition + bin-packing + ANS. |

### The shared value dictionary

Per-block `Dict` pays for its value table in every block — on columns whose
distinct values recur *across* blocks (repeated coordinates, IDs, quantized
readings) that either loses the competition or re-stores the same values per
block. `DictShared` lifts the table to a **column preamble**: `build_shared`
(one hash pass, bails above 50% cardinality or 2²⁰ distinct) stores the sorted
distinct values once, compressed by the same raw/delta/transpose choice as
`Dict`'s local table; each `DictShared` frame then holds only the codes
section. A **net-win gate** keeps the preamble only when the blocks that chose
`DictShared` saved more (vs their runner-up) than the preamble costs —
otherwise those blocks are re-encoded without it, so streams never pay for a
dead table. Blocks stay independently decodable given the read-only table, so
parallel decode is preserved. Measured: `poi_lat` (424 K values, ~100 K
distinct) went 1.20× → **2.00×**, reclaiming the whole-column-scope win inside
the normal block pipeline; columns that never pick it are byte-identical.

`Decimal128/256` are handled by `decimal.rs` (limb split → each limb through the
integer engine), not a `Mode`.

---

## 8. Entropy layer & cascades

The reusable cascade primitive is `entropy::code_residuals(bytes, λ, allow_lz)`:

```
residual bytes ─► entropy_pick (rANS vs range coder) ─► [+ LZ cascade at Max]
```

- **rANS** (`entropy/rans.rs`) — 4-way interleaved table-ANS. Fast decode. The
  default entropy coder at `Balanced`.
- **Range coder** (`entropy/rc.rs`) — bit-serial, adaptive order-1 byte model.
  Best ratio (~6% over rANS on correlated residuals), slow decode.
- **LZ cascade** (`Max` only) — LZ77 over the *transformed residual*, then
  entropy-code the LZ stream; kept only if strictly smaller. Captures long-range
  repeats a transform leaves behind.

**Where cascades are used:** the seven predictor/transpose/LZ modes cascade their
residual through `code_residuals`; `Dict` cascades both its code-plane and its
(sorted) value stream; `FloatMult` cascades its `k` stream; `ALP-RD` cascades its
codes (entropy on the ≤8-cardinality code bytes) and rights. Each is chosen by
size. Measured rule of thumb: cascades pay on **low-cardinality / skewed** streams
(codes), not on dense bit-packed ones (digits) — the latter are already near
minimal-width.

There is also an **order-1 size estimator** (`estimate_order1_bytes`): a single
joint-histogram pass that approximates the range coder's output, used by the
competition to prune candidates without fully range-coding each.

---

## 9. Levels

`Level` is a speed/ratio knob — a ladder by **decode-cost class**, where each step
admits one more (slower-to-decode) tier:

| Level | Adds over the previous | Decode |
| --- | --- | --- |
| `Fastest` | minimal pool (RAW/CONST/STRIDE + FoR/delta bit-pack), no entropy | fastest, random-access |
| `Fast` | + XORZ / ALP / ALP-RD / dict / RLE (still no entropy) | fast, random-access |
| `Balanced` | + rANS entropy on the vectorizable modes + **pco** | fast (no recurrence) |
| `High` | + the sequential predictors + the range coder | slower |
| `Max` (default) | + the LZ-over-residual cascade, `λ = 0` | slowest, best ratio |

`λ` per level: `Fastest 16, Fast 4, Balanced 2, High 0, Max 0`. The entropy-coder
choice and the predictor/pco/LZ gates are derived from the level. `Max` reproduces
the pure-ratio (`λ = 0`, all codecs) policy.

---

## 10. Stream format

```
stream  = header ++ [validity] ++ [shared-dict preamble] ++ frame*
header  = magic ++ version ++ dtype ++ flags ++ n_values ++ predictor_log2
frame   = mode:u8 ++ payload          (one per block)
```

The header records the column `DType`, which fixes the lane width of every
frame; flags mark the optional validity bitmap and the optional
shared-dictionary preamble (the column-wide value table for `DictShared`
frames), each of which follows the header in that order. Each frame's payload
layout is mode-specific and laid out in lane words. The version byte is **3**;
v2 streams of the 64-bit types and the decimal containers (whose layout did not
change) still decode, while v2 streams of the 32-bit types (widened-`f64`
payloads) are rejected with `UnsupportedVersion` rather than misread. The format
is **internal** (not stabilized
across versions) — the decoder always matches the encoder in the same build.

---

## 11. Parallelism

Blocks are independent, so with the default `parallel` feature both
`build_frames` (encode) and `decode_frames` (decode) fan out over a rayon pool —
`Config.threads` caps it, `None` uses the global pool. Without the `parallel`
feature the crate compiles fully single-threaded (sequential fallbacks via `cfg`),
dropping the rayon dependency.

---

## 12. SIMD strategy

Following *"The State of SIMD in Rust in 2025"* — raw intrinsics for the hot
irregular kernels, portable autovectorization for lane-wise work:

| Tier | Used for | Approach |
| --- | --- | --- |
| Hot / irregular | FCM predictor hash (`_mm_crc32_u64`), gather | raw `core::arch` + runtime feature detection (`hash.rs`) |
| Lane-wise maps | delta, transpose, bit-planes | autovectorized scalar Rust + `multiversion` per-CPU clones (`transform.rs`, `bitpack.rs`) |
| pco decode leaves | offset unpack, latent reconstruct, center-toggle | `multiversion` clones (vendored pco) |
| Everything else | entropy coders, framing | plain scalar Rust, LLVM autovectorizes |

Every SIMD kernel has a **bit-exact scalar fallback**, so a stream decodes
identically regardless of CPU features. `multiversion` compiles AVX2+BMI2 / AVX2 /
SSE4.2 / NEON clones selected at run time — the vectorized path is available
**without** a `-C target-cpu=native` build.

---

## 13. Extensions

- **Arrow adapter** (`arrow.rs`, feature `arrow`) — `compress_array`/
  `decompress_array` for primitive numeric and `Decimal128/256` arrays, reading
  values zero-copy from Arrow buffers and round-tripping the (LSB-first) validity
  bitmap with no transcoding.
- **C ABI** (`capi/`) — a separate crate exposing both a context-free path
  (global rayon pool) and an opaque context handle (persistent pool, like
  `ZSTD_CCtx`), plus a typed Arrow-native path (`QuoinDType`) that decodes into a
  caller buffer (alignment-safe, zero-copy fast path when aligned). Every entry
  point `catch_unwind`s — no panic crosses the FFI boundary.

---

## 14. Diagnostics & testing

- `mode_win_counts` / `reset_mode_win_counts` — per-mode win histogram (which
  codec won how many blocks). Drives the `diag_modes` example.
- `tests/ratio_regression.rs` — a guard that fails if the ratio on reference
  columns drops below a floor (catches silent selector regressions).
- `fuzz/` (cargo-fuzz) — decoder robustness against corrupt input.
- `examples/` — `bench_readme`/`bench_typed` (the benchmark harness),
  `profile_encode` / `diag_modes` / `cascade_lab` (profiling & investigation).
