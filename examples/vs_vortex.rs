//! quoin (via the Arrow adapter) vs **Vortex** on identical Arrow columns:
//! compression ratio × encode MB/s × decode MB/s. Vortex is the closest direct
//! competitor — a Rust columnar format with cascading typed encodings (ALP /
//! FastLanes / dict / FSST) chosen by a sampling compressor, same niche as quoin.
//!
//! FAIRNESS NOTE: Vortex's compressed size is measured as `array.nbytes()` — the
//! in-memory footprint of the compressed array tree, which is exactly the metric
//! Vortex's own README/benchmarks report as "compression ratio". It excludes the
//! file container/footer, so it slightly FAVORS Vortex relative to quoin's and
//! Parquet's serialized-bytes numbers. A quoin win here is therefore conservative.
//! Two Vortex configs are benched, analogous to quoin-fast/quoin-max:
//!   - vortex-btr     : default BtrBlocks (lightweight cascading encodings)
//!   - vortex-compact : BtrBlocks + zstd value pages (higher ratio, slower)
//!
//! Usage: cargo run --release --example vs_vortex --features bench-vortex
//! Env: VSV_N (values/column, default 1<<20), VSV_TRIALS (default 3).

use std::time::Instant;

use arrow_array::{
    ArrayRef, Decimal128Array, Decimal256Array, Float32Array, Float64Array, Int32Array, Int64Array,
    UInt32Array,
};
use arrow_buffer::i256;
use arrow_schema::DataType;

use vortex::array::ArrayRef as VxArray;
use vortex::array::arrow::FromArrowArray;
use vortex::array::{LEGACY_SESSION, VortexSessionExecute};
use vortex::compressor::{BtrBlocksCompressor, BtrBlocksCompressorBuilder};

use quoin::Config;
use quoin::arrow::{compress_array, decompress_array};

fn time_median<T>(trials: usize, mut f: impl FnMut() -> T) -> (f64, T) {
    let mut times = Vec::with_capacity(trials);
    let mut last = None;
    for _ in 0..trials {
        let t0 = Instant::now();
        let v = f();
        times.push(t0.elapsed().as_secs_f64());
        last = Some(v);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (times[trials / 2], last.unwrap())
}

fn mbps(bytes: usize, secs: f64) -> f64 {
    if secs <= 0.0 {
        f64::INFINITY
    } else {
        bytes as f64 / 1e6 / secs
    }
}

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s
}

fn raw_bytes(a: &ArrayRef) -> usize {
    a.len()
        * match a.data_type() {
            DataType::Decimal256(..) => 32,
            DataType::Decimal128(..) => 16,
            DataType::Float64 | DataType::Int64 => 8,
            _ => 4,
        }
}

/// Same column generators as `vs_columnar.rs`, so the two harnesses are directly
/// cross-readable (quoin vs Parquet there, quoin vs Vortex here).
fn datasets(n: usize) -> Vec<(&'static str, ArrayRef)> {
    use std::sync::Arc;
    let mut s = 0x1234_5678u64;
    let mut out: Vec<(&'static str, ArrayRef)> = Vec::new();

    out.push((
        "ids_i64",
        Arc::new(Int64Array::from_iter_values((0..n).map(|i| 1000 + (i % 300) as i64))),
    ));
    let mut t = 1_700_000_000_000i64;
    out.push((
        "timestamps_i64",
        Arc::new(Int64Array::from_iter_values((0..n).map(|_| {
            let r = lcg(&mut s);
            t += 1000 + (r >> 40) as i64 % 4096;
            t
        }))),
    ));
    out.push((
        "lowcard_i64",
        Arc::new(Int64Array::from_iter_values(
            (0..n).map(|_| 5_000_000 + (lcg(&mut s) >> 60) as i64),
        )),
    ));
    out.push((
        "random_i64",
        Arc::new(Int64Array::from_iter_values((0..n).map(|_| lcg(&mut s) as i64))),
    ));
    out.push((
        "decimals_f64",
        Arc::new(Float64Array::from_iter_values(
            (0..n).map(|_| ((lcg(&mut s) >> 40) % 1_000_000) as f64 / 100.0),
        )),
    ));
    out.push((
        "sensor_f64",
        Arc::new(Float64Array::from_iter_values(
            (0..n).map(|i| (i as f64 * 0.001).sin() * 100.0 + (i as f64) * 0.01),
        )),
    ));
    out.push((
        "categorical_u32",
        Arc::new(UInt32Array::from_iter_values(
            (0..n).map(|_| 100 + (lcg(&mut s) >> 61) as u32),
        )),
    ));
    out.push((
        "narrow_i32",
        Arc::new(Int32Array::from_iter_values((0..n).map(|i| 50_000 + (i % 1000) as i32))),
    ));
    out.push((
        "decimals_f32",
        Arc::new(Float32Array::from_iter_values(
            (0..n).map(|_| ((lcg(&mut s) >> 40) % 1_000_000) as f32 / 100.0),
        )),
    ));
    out.push((
        "sensor_f32",
        Arc::new(Float32Array::from_iter_values(
            (0..n).map(|i| (i as f32 * 0.001).sin() * 100.0 + (i as f32) * 0.01),
        )),
    ));
    out.push((
        "amounts_dec128",
        Arc::new(
            Decimal128Array::from_iter_values(
                (0..n).map(|_| 1_000_000i128 + ((lcg(&mut s) >> 40) % 5_000_000) as i128),
            )
            .with_precision_and_scale(18, 2)
            .unwrap(),
        ),
    ));
    out.push((
        "amounts_dec256",
        Arc::new(
            Decimal256Array::from_iter_values((0..n).map(|_| {
                i256::from_i128(1i128 << 100)
                    .wrapping_add(i256::from_i128(((lcg(&mut s) >> 40) % 5_000_000) as i128))
            }))
            .with_precision_and_scale(50, 8)
            .unwrap(),
        ),
    ));
    out
}

/// Import an Arrow primitive column (no nulls) into a Vortex array, zero-copy where
/// the layouts allow it.
fn to_vortex(a: &ArrayRef) -> VxArray {
    let r = match a.data_type() {
        DataType::Int64 => VxArray::from_arrow(a.as_any().downcast_ref::<Int64Array>().unwrap(), false),
        DataType::Float64 => {
            VxArray::from_arrow(a.as_any().downcast_ref::<Float64Array>().unwrap(), false)
        }
        DataType::Int32 => VxArray::from_arrow(a.as_any().downcast_ref::<Int32Array>().unwrap(), false),
        DataType::UInt32 => {
            VxArray::from_arrow(a.as_any().downcast_ref::<UInt32Array>().unwrap(), false)
        }
        DataType::Float32 => {
            VxArray::from_arrow(a.as_any().downcast_ref::<Float32Array>().unwrap(), false)
        }
        DataType::Decimal128(..) => {
            VxArray::from_arrow(a.as_any().downcast_ref::<Decimal128Array>().unwrap(), false)
        }
        DataType::Decimal256(..) => {
            VxArray::from_arrow(a.as_any().downcast_ref::<Decimal256Array>().unwrap(), false)
        }
        dt => panic!("unsupported dtype {dt:?}"),
    };
    r.unwrap()
}

#[derive(Default, Clone, Copy)]
struct Acc {
    raw: usize,
    comp: usize,
    enc_s: f64,
    dec_s: f64,
}

fn print_row(ds: &str, codec: &str, raw: usize, comp: usize, enc_s: f64, dec_s: f64) {
    println!(
        "{ds:<16} {codec:<16} {:>7.2}x {:>9.0} {:>9.0}",
        raw as f64 / comp.max(1) as f64,
        mbps(raw, enc_s),
        mbps(raw, dec_s),
    );
}

fn main() {
    let n: usize = std::env::var("VSV_N").ok().and_then(|s| s.parse().ok()).unwrap_or(1 << 20);
    let trials: usize = std::env::var("VSV_TRIALS").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    // Shared block size in rows (0 = native: quoin adaptive, Vortex whole array);
    // clamped to quoin's max so both sides stay aligned.
    let block_values: usize = std::env::var("BLOCK_VALUES")
        .ok()
        .and_then(|s| s.parse().ok())
        .map(|b: usize| if b == 0 { 0 } else { b.min(quoin::MAX_BLOCK_SIZE) })
        .unwrap_or(0);
    let block_rows = (block_values > 0).then_some(block_values);

    println!("{}", quoin::VERSION);
    let bd = match block_rows {
        Some(b) => format!("fixed {b} rows/block (both)"),
        None => "native per-tool blocks".to_string(),
    };
    println!("quoin (Arrow adapter) vs Vortex, {n} values/column, median of {trials}, {bd}");
    println!("(Vortex size = compressed array nbytes, its own ratio metric; see file header)\n");
    println!("{:<16} {:<16} {:>8} {:>9} {:>9}", "dataset", "codec", "ratio", "enc MB/s", "dec MB/s");

    let btr_light = BtrBlocksCompressor::default();
    let btr_compact = BtrBlocksCompressorBuilder::default().with_compact().build();

    let mut accs: std::collections::BTreeMap<&'static str, Acc> = std::collections::BTreeMap::new();
    let mut add = |codec: &'static str, raw, comp, enc_s, dec_s| {
        let a = accs.entry(codec).or_default();
        a.raw += raw;
        a.comp += comp;
        a.enc_s += enc_s;
        a.dec_s += dec_s;
    };

    for (ds, array) in datasets(n) {
        let raw = raw_bytes(&array);

        for (codec, cfg) in [
            (
                "quoin-fast",
                Config { level: quoin::Level::Fast, block_size: block_rows, ..Config::default() },
            ),
            ("quoin-max", Config { block_size: block_rows, ..Config::default() }),
        ] {
            let (enc_s, packed) = time_median(trials, || compress_array(array.as_ref(), cfg).unwrap());
            let (dec_s, back) = time_median(trials, || decompress_array(&packed).unwrap());
            assert_eq!(back.len(), array.len(), "{ds}/{codec} round-trip len");
            print_row(ds, codec, raw, packed.len(), enc_s, dec_s);
            add(codec, raw, packed.len(), enc_s, dec_s);
        }

        // Slice the column into the shared block size so Vortex compresses each
        // chunk independently — no whole-column window quoin doesn't also get.
        let chunk_rows = block_rows.unwrap_or_else(|| array.len().max(1));
        let vx_chunks: Vec<VxArray> = (0..array.len())
            .step_by(chunk_rows)
            .map(|off| to_vortex(&array.slice(off, chunk_rows.min(array.len() - off))))
            .collect();
        for (codec, compressor) in [("vortex-btr", &btr_light), ("vortex-compact", &btr_compact)] {
            let mut ctx = LEGACY_SESSION.create_execution_ctx();
            let (enc_s, compressed) = time_median(trials, || {
                vx_chunks
                    .iter()
                    .map(|c| compressor.compress(c, &mut ctx).unwrap())
                    .collect::<Vec<_>>()
            });
            let comp: usize = compressed.iter().map(|c| c.nbytes() as usize).sum();
            // `to_canonical` (the decode path) is marked deprecated in 0.75 in favour of the
            // session-based `execute::<Canonical>`, but it is the stable in-memory decode for a
            // pinned bench and decodes the full compressed tree identically.
            #[allow(deprecated)]
            let (dec_s, total) = time_median(trials, || {
                compressed.iter().map(|c| c.to_canonical().unwrap().len()).sum::<usize>()
            });
            assert_eq!(total, array.len(), "{ds}/{codec} round-trip len");
            print_row(ds, codec, raw, comp, enc_s, dec_s);
            add(codec, raw, comp, enc_s, dec_s);
        }
        println!();
    }

    println!("AGGREGATE (throughput-weighted):");
    println!("{:<16} {:>8} {:>9} {:>9}", "codec", "ratio", "enc MB/s", "dec MB/s");
    for (codec, a) in &accs {
        println!(
            "{codec:<16} {:>7.2}x {:>9.0} {:>9.0}",
            a.raw as f64 / a.comp.max(1) as f64,
            mbps(a.raw, a.enc_s),
            mbps(a.raw, a.dec_s),
        );
    }
}
