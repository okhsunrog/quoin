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

1. **One physical lane.** Every typed column is lowered to a single `u64` lane
   (decimals to a wider container). The codecs are written once against that lane;
   the [`DType`] just selects which *family* of codecs may compete and is recorded
   in the stream so the decoder restores the original type.
2. **Independent blocks.** Each block carries its own winning codec and entropy
   coder. This is what makes encode/decode parallel and gives random access at
   block granularity.

```
        ┌─ compress ──────────────────────────────────────────────┐
typed   │  ColumnRef ──► u64 lane ──► plan blocks ──► per block:   │   byte
column  │  (zero-copy   (DType,         (adaptive      competition │   stream
 (+nulls)│   reinterpret) validity)     sizing)        → frame     │  (header
        │                                              ◄───────────┤   + frames)
        └──────────────────────────────────────────────────────────┘
        ┌─ decompress ─────────────────────────────────────────────┐
byte    │  header ──► frames (parallel) ──► per frame: mode decode  │   typed
stream  │             ──► u64 lane ──► widen to DType ──► validity  │   column
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

The codec engine only ever sees a `u64` slice. Lowering a typed column to that
lane (`ColumnRef::to_lane`):

| Input type | Lane | How |
| --- | --- | --- |
| `f64` / `i64` / `u64` | `u64` | **zero-copy** bit-cast of the slice (no allocation) |
| `i32` | `u64` | sign-extend, narrow back on decode |
| `u32` | `u64` | zero-extend |
| `f32` | `u64` | exact-widen to `f64`, then bit-cast |
| `Decimal128/256` | 128/256-bit | dedicated `decimal.rs` container (not the `u64` lane) |

`Family` (from `DType`) gates the competition: `Float` unlocks the float-value
codecs (ALP, ALP-RD, FLOAT_MULT) and the float predictors; `Int` runs
frame-of-reference and the signed-delta cascade. Type-agnostic codecs (RAW, CONST,
bit-packers, LZ, transpose, dict, pco) apply to both.

---

## 4. Block framing

A column is split by `plan_blocks` (`encoder.rs`): an adaptive base of ~256 KiB
that grows toward ~1 MiB when a cheap probe says the data is low-entropy
(dictionary-like / constant / run-heavy), so cheap columns pay less framing
overhead. A fixed `Config.block_size` pins every block to a row count (for
storage-chunk alignment / random access).

Each block becomes a **frame**: `[mode byte | payload]`. An unknown mode byte is a
hard decode error, never a silent zero-fill.

---

## 5. The mode competition (encode)

`encode_block_full` is the heart. It maintains a `Best` (smallest scorer so far,
plus the runner-up mode) and runs codecs roughly in **decode-cost order**, so a
tight incumbent is established early:

1. **Structural / cheap** — RAW (the baseline every mode must beat), CONST,
   STRIDE, XORZ.
2. **Cheap strong** — FoR+bitpack, delta+bitpack, and (float) FLOAT_MULT / ALP /
   ALP-RD. These give a tight `best` *before* the expensive predictors, which is
   what lets the estimate gate (below) prune.
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
producing the `u64` lane → widen to the column `DType` → reattach validity. The
decoder is fuzz-hardened: every length/offset is bounds-checked, and a corrupt
stream returns `Error`, never panics or over-allocates.

---

## 7. Codecs

20 block modes, by family (`mode.rs`, `codecs/`):

| Group | Modes | Notes |
| --- | --- | --- |
| Structural | `Raw`, `Const`, `Stride`, `Xorz` | O(n), trivial; RAW is the baseline. |
| Integer bit-pack | `ForBitpack`, `DeltaBitpack` | frame-of-reference / delta + FastLanes bit-pack. Random-access, fast decode. |
| Float value | `Alp`, `AlpRd`, `FloatMult` | doubles that are really decimals → scaled integers (ALP) or split-dictionary (ALP-RD). |
| Predictors | `Pred`, `PredRc`, `Pred2`, `Delta2`, `DeltaDp`, `OrderedDelta` | FCM/DFCM hash + XOR residual; polynomial-float; 2nd-order int. Sequential decode. |
| Dictionary | `Dict`, `Rle` | low-cardinality / run-heavy. |
| Generic | `ByteTranspose`, `Lz` | AoS→SoA byte planes; LZ77 over the block. |
| Numeric backend | `Pco` | vendored pcodec — latent decomposition + bin-packing + ANS. |

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
stream  = header ++ frame*
header  = magic ++ version ++ dtype ++ flags ++ n_values ++ predictor_log2 ++ [validity]
frame   = mode:u8 ++ payload          (one per block)
```

The header records the column `DType` and (if present) the validity bitmap. Each
frame's payload layout is mode-specific. The format is **internal** (not stabilized
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
