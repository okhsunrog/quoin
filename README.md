# fp-compressor

A lossless compressor for streams of IEEE-754 `f64`, in safe Rust. This is a
from-scratch port of the [`fc`](https://github.com/xtellect/fc) floating-point
compressor (Apache-2.0, © Praveen Vaddadi), which runs a competition between
~50 specialized codecs per block and emits the smallest result.

## Status

Early. The framework, stream format, and the mode competition are in place,
with five real, round-tripping codecs (`RAW`, `CONST`, `STRIDE`, `XORZ`,
`PRED`). The remaining ~45 modes from `fc` are tracked in
[`ROADMAP.md`](ROADMAP.md).

```rust
use fp_compressor::{compress, decompress, Config};

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
