//! quoin (via the Arrow adapter) vs **Parquet** on identical Arrow columns:
//! compression ratio × encode MB/s × decode MB/s. Parquet is the universal
//! columnar baseline (dictionary / RLE / DELTA_BINARY_PACKED / BYTE_STREAM_SPLIT
//! plus page compression); like quoin it is internally paged, so this is a fair,
//! column-store-realistic comparison — unlike whole-stream zstd.
//!
//! Usage: cargo run --release --example vs_columnar --features bench-parquet
//! Env: VSC_N (values/column, default 1<<20), VSC_TRIALS (default 3).

use std::sync::Arc;
use std::time::Instant;

use arrow_array::{
    ArrayRef, Decimal128Array, Decimal256Array, Float32Array, Float64Array, Int32Array, Int64Array,
    RecordBatch, UInt32Array,
};
use arrow_buffer::i256;
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;

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

fn datasets(n: usize) -> Vec<(&'static str, ArrayRef)> {
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
            // Money in cents (scale 2): clustered around a base, billing-like.
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
            // High-precision amounts (scale 8): a large 256-bit base + spread.
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

/// `block_rows: Some(n)` forces Parquet to bound each row group / data page to
/// `n` rows, so its codec sees the same block as quoin (no wider window);
/// `None` keeps Parquet's native defaults.
fn parquet_bytes(array: &ArrayRef, compression: Compression, block_rows: Option<usize>) -> Vec<u8> {
    let field = Field::new("c", array.data_type().clone(), array.null_count() > 0);
    let schema = Arc::new(Schema::new(vec![field]));
    let batch = RecordBatch::try_new(schema.clone(), vec![array.clone()]).unwrap();
    let mut builder = WriterProperties::builder().set_compression(compression);
    if let Some(rows) = block_rows {
        builder = builder
            .set_max_row_group_row_count(Some(rows))
            .set_data_page_row_count_limit(rows);
    }
    let props = builder.build();
    let mut buf: Vec<u8> = Vec::new();
    let mut w = ArrowWriter::try_new(&mut buf, schema, Some(props)).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
    buf
}

fn parquet_read_rows(bytes: &[u8]) -> usize {
    let reader = ParquetRecordBatchReaderBuilder::try_new(Bytes::from(bytes.to_vec()))
        .unwrap()
        .build()
        .unwrap();
    reader.map(|b| b.unwrap().num_rows()).sum()
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
        "{ds:<16} {codec:<14} {:>7.2}x {:>9.0} {:>9.0}",
        raw as f64 / comp.max(1) as f64,
        mbps(raw, enc_s),
        mbps(raw, dec_s),
    );
}

fn main() {
    let n: usize = std::env::var("VSC_N").ok().and_then(|s| s.parse().ok()).unwrap_or(1 << 20);
    let trials: usize = std::env::var("VSC_TRIALS").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    // Shared block size in rows (0 = native: quoin adaptive, Parquet defaults);
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
    println!("quoin (Arrow adapter) vs Parquet, {n} values/column, median of {trials}, {bd}\n");
    println!("{:<16} {:<14} {:>8} {:>9} {:>9}", "dataset", "codec", "ratio", "enc MB/s", "dec MB/s");

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

        for (codec, comp) in [
            ("parquet-zstd9", Compression::ZSTD(ZstdLevel::try_new(9).unwrap())),
            ("parquet-snappy", Compression::SNAPPY),
            ("parquet-plain", Compression::UNCOMPRESSED),
        ] {
            let (enc_s, packed) = time_median(trials, || parquet_bytes(&array, comp, block_rows));
            let (dec_s, rows) = time_median(trials, || parquet_read_rows(&packed));
            assert_eq!(rows, array.len(), "{ds}/{codec} rows");
            print_row(ds, codec, raw, packed.len(), enc_s, dec_s);
            add(codec, raw, packed.len(), enc_s, dec_s);
        }
        println!();
    }

    println!("AGGREGATE (throughput-weighted):");
    println!("{:<14} {:>8} {:>9} {:>9}", "codec", "ratio", "enc MB/s", "dec MB/s");
    for (codec, a) in &accs {
        println!(
            "{codec:<14} {:>7.2}x {:>9.0} {:>9.0}",
            a.raw as f64 / a.comp.max(1) as f64,
            mbps(a.raw, a.enc_s),
            mbps(a.raw, a.dec_s),
        );
    }
}
