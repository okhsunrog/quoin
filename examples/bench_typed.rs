//! README benchmark for the *integer* and *decimal* lanes — the columns where
//! type-awareness pays off most. Emits the same CSV shape as bench_readme.rs.
//!
//! Integers: real i64/i32 columns from a ClickBench `hits` parquet file.
//! Decimals: real f64 columns (ALP corpus) rounded to fixed-point Decimal128,
//!           which is exactly the "store prices/measurements as DECIMAL" case.
//!
//! Baselines (lz4/zlib/zstd) compress the raw little-endian value bytes — the
//! byte-blind view a generic compressor gets. quoin uses its typed column API.
//!
//! Usage:
//!   PARQUET_FILE=datasets/parquet/clickbench_hits_0.parquet \
//!   cargo run --release --example bench_typed \
//!       --features bench-parquet,bench-zstd,bench-lz4,bench-deflate > typed.csv

use std::path::Path;
use std::time::Instant;

use arrow_array::{cast::AsArray, types::Int32Type, types::Int64Type};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use quoin::{Column, ColumnRef, Config, Level, compress_column, decompress_column};

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
    (times[times.len() / 2], last.unwrap())
}

fn mbps(orig: usize, secs: f64) -> f64 {
    if secs <= 0.0 { f64::INFINITY } else { orig as f64 / 1e6 / secs }
}

struct Row { ratio: f64, enc: f64, dec: f64, ok: bool }

fn emit(label: &str, dtype: &str, n: usize, codec: &str, r: Row) {
    println!("{label},{dtype},{n},{codec},{:.4},{:.1},{:.1},{}", r.ratio, r.enc, r.dec, r.ok as u8);
}

// ---- generic byte-blind baselines over a raw &[u8] view of the column ----
fn bench_lz4(bytes: &[u8], trials: usize) -> Row {
    let mut p = Vec::new();
    let (enc, _) = time_median(trials, || { p = lz4_flex::compress_prepend_size(bytes); p.len() });
    let (dec, out) = time_median(trials, || lz4_flex::decompress_size_prepended(&p).unwrap());
    Row { ratio: bytes.len() as f64 / p.len().max(1) as f64, enc: mbps(bytes.len(), enc), dec: mbps(bytes.len(), dec), ok: out == bytes }
}
fn bench_zstd(bytes: &[u8], level: i32, trials: usize) -> Row {
    let mut p = Vec::new();
    let (enc, _) = time_median(trials, || { p = zstd::bulk::compress(bytes, level).unwrap(); p.len() });
    let (dec, out) = time_median(trials, || zstd::bulk::decompress(&p, bytes.len()).unwrap());
    Row { ratio: bytes.len() as f64 / p.len().max(1) as f64, enc: mbps(bytes.len(), enc), dec: mbps(bytes.len(), dec), ok: out == bytes }
}
fn bench_zlib(bytes: &[u8], level: u32, trials: usize) -> Row {
    use flate2::{Compression, read::{ZlibDecoder, ZlibEncoder}};
    use std::io::Read;
    let mut p = Vec::new();
    let (enc, _) = time_median(trials, || { p.clear(); ZlibEncoder::new(bytes, Compression::new(level)).read_to_end(&mut p).unwrap(); p.len() });
    let (dec, out) = time_median(trials, || { let mut o = Vec::with_capacity(bytes.len()); ZlibDecoder::new(&p[..]).read_to_end(&mut o).unwrap(); o });
    Row { ratio: bytes.len() as f64 / p.len().max(1) as f64, enc: mbps(bytes.len(), enc), dec: mbps(bytes.len(), dec), ok: out == bytes }
}

fn bench_quoin(col: ColumnRef, orig: usize, level: Level, trials: usize) -> Row {
    let cfg = Config { level, ..Config::default() };
    let mut p = Vec::new();
    let (enc, _) = time_median(trials, || { p = compress_column(col, None, cfg); p.len() });
    let (dec, dc) = time_median(trials, || decompress_column(&p).unwrap());
    Row { ratio: orig as f64 / p.len().max(1) as f64, enc: mbps(orig, enc), dec: mbps(orig, dec), ok: dc.values == col_owned(col) }
}

// Owned copy of the input column for round-trip verification.
fn col_owned(c: ColumnRef) -> Column {
    match c {
        ColumnRef::I64(s) => Column::I64(s.to_vec()),
        ColumnRef::I32(s) => Column::I32(s.to_vec()),
        ColumnRef::Decimal128 { values, scale, precision } => Column::Decimal128 { values: values.to_vec(), scale, precision },
        _ => unreachable!("bench_typed only drives I64/I32/Decimal128"),
    }
}

fn run_int(label: &str, dtype: &str, n: usize, col: ColumnRef, orig: usize, bytes: &[u8], trials: usize) {
    emit(label, dtype, n, "quoin-fastest", bench_quoin(col, orig, Level::Fastest, trials));
    emit(label, dtype, n, "quoin-fast", bench_quoin(col, orig, Level::Fast, trials));
    emit(label, dtype, n, "quoin-balanced", bench_quoin(col, orig, Level::Balanced, trials));
    emit(label, dtype, n, "quoin-high", bench_quoin(col, orig, Level::High, trials));
    emit(label, dtype, n, "quoin-max", bench_quoin(col, orig, Level::Max, trials));
    emit(label, dtype, n, "lz4", bench_lz4(bytes, trials));
    emit(label, dtype, n, "zlib-6", bench_zlib(bytes, 6, trials));
    emit(label, dtype, n, "zstd-3", bench_zstd(bytes, 3, trials));
    emit(label, dtype, n, "zstd-19", bench_zstd(bytes, 19, trials));
}

fn load_f64_bin(p: &Path) -> Vec<f64> {
    std::fs::read(p).unwrap().chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
}

fn main() {
    let trials: usize = std::env::var("BR_TRIALS").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let cap = 2_000_000usize;
    println!("section,dtype,n,codec,ratio,enc_mbps,dec_mbps,ok");

    // ---- Integer columns from real ClickBench parquet ----
    let pq = std::env::var("PARQUET_FILE").unwrap_or_else(|_| "datasets/parquet/clickbench_hits_0.parquet".into());
    if Path::new(&pq).exists() {
        let file = std::fs::File::open(&pq).unwrap();
        let b = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let rows = b.metadata().file_metadata().num_rows() as usize;
        let mut reader = b.with_batch_size(rows + 1).build().unwrap();
        let batch = reader.next().unwrap().unwrap();
        let want_i64 = ["EventTime", "WatchID", "UserID"]; // timestamp, random ID, clustered ID
        let want_i32 = ["CounterID", "RegionID", "IPNetworkID"]; // low-card / structured
        for name in want_i64 {
            if let Some(c) = batch.column_by_name(name) {
                let a = c.as_primitive::<Int64Type>();
                let v: Vec<i64> = a.values().iter().take(cap).copied().collect();
                let n = v.len();
                let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, n * 8) };
                eprintln!("i64 {name}: {n}");
                run_int(name, "i64", n, ColumnRef::I64(&v), n * 8, bytes, trials);
            }
        }
        for name in want_i32 {
            if let Some(c) = batch.column_by_name(name) {
                let a = c.as_primitive::<Int32Type>();
                let v: Vec<i32> = a.values().iter().take(cap).copied().collect();
                let n = v.len();
                let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, n * 4) };
                eprintln!("i32 {name}: {n}");
                run_int(name, "i32", n, ColumnRef::I32(&v), n * 4, bytes, trials);
            }
        }
    } else {
        eprintln!("no parquet at {pq}, skipping integer columns");
    }

    // ---- Decimal128 columns: real f64 rounded to fixed point ----
    let alp = std::env::var("ALP_DIR").unwrap_or_else(|_| "datasets/alp".into());
    let decimals = [("food_prices", "food_prices.bin", 2i8), ("city_temperature", "city_temperature_f.bin", 1), ("bitcoin_tx", "bitcoin_transactions_f.bin", 2)];
    for (label, fname, scale) in decimals {
        let path = Path::new(&alp).join(fname);
        if !path.exists() { eprintln!("skip decimal {label} (missing)"); continue; }
        let f = load_f64_bin(&path);
        let mul = 10f64.powi(scale as i32);
        let v: Vec<i128> = f.iter().take(cap).map(|x| (x * mul).round() as i128).collect();
        let n = v.len();
        let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, n * 16) };
        eprintln!("dec128 {label}: {n} (scale {scale})");
        let col = ColumnRef::Decimal128 { values: &v, scale, precision: 38 };
        run_int(label, "decimal128", n, col, n * 16, bytes, trials);
    }
}
