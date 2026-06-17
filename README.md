# quoin

A lossless compressor for columns of numbers with a **safe public API**. It runs
a per-block competition of lightweight, type-specialized codecs (predictors,
deltas, LZ, frame-of-reference + SIMD bit-packing, ALP) and emits the smallest.
quoin's own codecs are written in safe Rust (with a bit-exact software fallback
for every SIMD kernel); the vendored **pco** backend (`vendor/pco`) uses internal
`unsafe` for its fast bit-reader, with scalar fallbacks and per-CPU dispatch.
Started as a from-scratch port of the [`fc`](https://github.com/xtellect/fc)
floating-point compressor (Apache-2.0, © Praveen Vaddadi); now evolving toward a
**type-aware columnar codec engine** (see [`docs/LANDSCAPE.md`](docs/LANDSCAPE.md)).

## Status

Working and competitive. Typed columns
(`f64`/`f32`/`i64`/`u64`/`i32`/`u32`/`Decimal128`/`Decimal256`) with
**nulls/validity**, a five-level speed/ratio knob (Fastest→Fast→Balanced→High→Max,
a decode-cost-class ladder), 20 codecs (+ a 4-way interleaved
rANS coder, + a vendored **pco**/pcodec numeric backend at the top two levels),
adaptive **or fixed/configurable** block sizing
(`Config.block_size`, for storage-chunk alignment and random access),
feature-gated competition (with an optional sampling-based selector), an
autovectorizing FastLanes bit-packing substrate,
block-parallel encode/decode, a fuzz-hardened decoder, an optional **Apache
Arrow adapter** (`--features arrow`), and a **C ABI** (`capi/`). On a typed
column suite quoin beats Parquet and is competitive with Vortex — winning the
fast tier outright, the decimal types decisively, and (via the pco backend at
`High`/`Max`) smooth numeric columns where it previously trailed: on a
synthetic `sensor_f64` it now reaches **9.07×** vs Vortex's BtrBlocks+zstd
8.75×, decoding several times faster (see [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)).

- **f64 (17-dataset harness): aggregate 3.05×** vs the C `fc` 3.07× and zstd-9
  2.09× — an **8–8 split** with `fc` (+1 tie). Outright wins: constant 55,188×
  (beats `fc`), linear 96×, piecewise 218×, int-x1000 7,127×, decimal 487×,
  dict-16 13,797×, quantized 3,666×, stocks 18×.
- **Integer/decimal columns:** FoR+bit-packing, ALP (decimals, robust to
  outliers), and a delta→bitpack cascade — the columnar foundation.

Status, results, next steps, and future ideas are tracked in
[`ROADMAP.md`](ROADMAP.md).

Run `cargo run --release --example compare --features bench-zstd,bench-fc`
(with `FC_SRC_DIR` pointing at an `fc` checkout) to reproduce the comparison.
Kernel-level micro-benchmarks live in `benches/kernels.rs` (`cargo bench`).

```rust
use quoin::{compress, decompress, Config};

let data: Vec<f64> = (0..10_000).map(|i| i as f64 * 0.5).collect();
let packed = compress(&data, Config::default());
let restored = decompress(&packed).unwrap();
assert_eq!(data, restored);
```

## SIMD strategy

The port follows the conclusion of *"The State of SIMD in Rust in 2025"*: for C
ports targeting specific hardware, raw intrinsics are the right call for the
hot, irregular kernels, with portable approaches reserved for simple lane-wise
work. Concretely:

| Tier | Used for | Approach |
| --- | --- | --- |
| **Hot / irregular** | FCM predictor hash (`_mm_crc32_u64`), gather | Raw `core::arch` intrinsics + runtime feature detection (`src/hash.rs`) |
| **Lane-wise maps** | delta, transpose, bitplane | Autovectorized scalar Rust + `multiversion` per-CPU clones (`src/transform.rs`) |
| **pco decode leaves** | offset unpack, latent reconstruct, center-toggle | `multiversion` per-CPU clones (`vendor/pco`) — see below |
| **Everything else** | entropy coders, framing | Plain scalar Rust, let LLVM autovectorize |

Unlike the C original, there is a bit-exact software fallback for every SIMD
kernel, so the same stream decodes identically regardless of CPU features — and
an ARM/NEON path is in scope (the C version is x86-only).

The vendored **pco** backend upstream relies on a global `-C target-cpu=native`
build to vectorize its decode hot loops (its `build.rs` even warns when those
instruction sets are missing). The fork instead annotates pco's three
vectorizable decode leaves — offset bit-unpack, latent→number reconstruction,
and the center-toggle map — with `#[multiversion]`, so AVX2+BMI2 / AVX2 / SSE4.2
/ NEON clones are selected at run time. A stock (non-native) build then gets the
fast path *and* runs correctly on older CPUs via the scalar baseline clone. On a
profiled `sensor_f64` decode this recovered **+27%** over the non-native build —
faster than even a `target-cpu=native` build — while pco's two genuinely serial
hot spots (interleaved-ANS symbol decode and the delta prefix-sum) are left
scalar, since neither vectorizes.

## Differences from the C `fc`

- **Safe API**: `&[f64]` in/out, `Result`-based errors. Unknown block modes are
  a hard error, not silently decoded to zeros.
- **Not yet wire-compatible** with `fc`'s on-disk format. The mode IDs match so
  a compatible profile can be added later for cross-testing.
- Adaptive block sizing, typed columns, nulls, and block-parallel encode/decode
  are quoin extensions rather than `fc` wire-format features.

## Building

```bash
cargo build --release
cargo test            # round-trip + ratio checks across synthetic datasets
```

Requires stable Rust (edition 2024). No nightly features.

## License

Apache-2.0, matching the upstream `fc` project.
