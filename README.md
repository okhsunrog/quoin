# quoin

A lossless, safe-Rust compressor for columns of numbers. It runs a per-block
competition of lightweight, type-specialized codecs (predictors, deltas, LZ,
frame-of-reference + SIMD bit-packing, ALP) and emits the smallest. Started as a
from-scratch port of the [`fc`](https://github.com/xtellect/fc) floating-point
compressor (Apache-2.0, © Praveen Vaddadi); now evolving toward a **type-aware
columnar codec engine** (see [`docs/LANDSCAPE.md`](docs/LANDSCAPE.md)).

## Status

Working and competitive. 16 codecs, adaptive block sizing, feature-gated
competition (with an optional sampling-based selector), range-coder + tANS
entropy stage, an autovectorizing FastLanes bit-packing substrate, block-parallel
encode/decode, a fuzz-hardened decoder, and a **C ABI** (`capi/`).

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
| **Everything else** | entropy coders, framing | Plain scalar Rust, let LLVM autovectorize |

Unlike the C original, there is a bit-exact software fallback for every SIMD
kernel, so the same stream decodes identically regardless of CPU features — and
an ARM/NEON path is in scope (the C version is x86-only).

## Differences from the C `fc`

- **Safe API**: `&[f64]` in/out, `Result`-based errors. Unknown block modes are
  a hard error, not silently decoded to zeros.
- **Not yet wire-compatible** with `fc`'s on-disk format. The mode IDs match so
  a compatible profile can be added later for cross-testing.
- Adaptive block sizing and multi-threading are not yet implemented.

## Building

```bash
cargo build --release
cargo test            # round-trip + ratio checks across synthetic datasets
```

Requires stable Rust (edition 2024). No nightly features.

## License

Apache-2.0, matching the upstream `fc` project.
