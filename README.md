# quoin

A lossless compressor for streams of IEEE-754 `f64`, in safe Rust. This is a
from-scratch port of the [`fc`](https://github.com/xtellect/fc) floating-point
compressor (Apache-2.0, © Praveen Vaddadi), which runs a competition between
~50 specialized codecs per block and emits the smallest result.

## Status

Working and competitive — at near-parity with the C `fc`. Framework, stream
format, feature-gated adaptive-block mode competition, entropy coders (binary
range coder + tANS), 13 codecs/predictors (incl. LZ77, byte-transpose,
FLOAT_MULT), and block-parallel encode/decode are in place; the decoder is
fuzz-hardened. On the bundled 17-dataset harness the aggregate ratio is
**3.05×** (vs the C `fc` 3.07× and zstd-9 2.09×) — an **8–8 split** with `fc`
(plus 1 tie): we win constant (55,188×, beating `fc`), linear (96×),
piecewise (218×), int-x1000 (7,127×), decimal (487×), dict-16 (13,797×),
quantized (3,666×), stocks (18×); `fc` edges parabolic/sin/geo and the noisy
datasets (ar2/audio/random-walk/climate/sensor) by ~2%, near their entropy
floor. Remaining `fc` modes are tracked in [`ROADMAP.md`](ROADMAP.md).

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
