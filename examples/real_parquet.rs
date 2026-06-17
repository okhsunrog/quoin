//! quoin vs Parquet on the **real** numeric/temporal columns of an actual
//! Parquet file (e.g. the ClickBench `hits` web-analytics dataset). Every
//! non-string column is fed through quoin's Arrow adapter and compared to
//! Parquet-zstd on the same column. String (binary/utf8) columns are skipped —
//! quoin has no string lane yet.
//!
//! Usage:
//!   PARQUET_FILE=datasets/parquet/clickbench_hits_0.parquet \
//!   cargo run --release --example real_parquet --features bench-parquet
//! Env: PARQUET_FILE (required), RP_TRIALS (default 1), RP_MAXCOLS (default all),
//!      BLOCK_VALUES (shared block size in rows; 0 = native).

use std::sync::Arc;
use std::time::Instant;

use arrow_array::{
    Array, ArrayRef, Int32Array, Int64Array, RecordBatch, UInt32Array, cast::AsArray,
};
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
    if secs <= 0.0 { f64::INFINITY } else { bytes as f64 / 1e6 / secs }
}

/// Widen a parquet column to a quoin-supported Arrow array (Int64/Int32/UInt32/
/// Float64/Float32), preserving nulls. Returns `None` for unsupported types
/// (strings/binary/bool/etc.), which are skipped.
fn to_quoin_array(col: &ArrayRef) -> Option<ArrayRef> {
    match col.data_type() {
        DataType::Int64 | DataType::Int32 | DataType::UInt32 | DataType::Float64
        | DataType::Float32 => Some(col.clone()),
        DataType::Int16 => {
            let a = col.as_primitive::<arrow_array::types::Int16Type>();
            Some(Arc::new(Int32Array::from_iter(a.iter().map(|o| o.map(|v| v as i32)))))
        }
        DataType::UInt16 => {
            let a = col.as_primitive::<arrow_array::types::UInt16Type>();
            Some(Arc::new(UInt32Array::from_iter(a.iter().map(|o| o.map(|v| v as u32)))))
        }
        DataType::Int8 => {
            let a = col.as_primitive::<arrow_array::types::Int8Type>();
            Some(Arc::new(Int32Array::from_iter(a.iter().map(|o| o.map(|v| v as i32)))))
        }
        DataType::UInt8 => {
            let a = col.as_primitive::<arrow_array::types::UInt8Type>();
            Some(Arc::new(UInt32Array::from_iter(a.iter().map(|o| o.map(|v| v as u32)))))
        }
        DataType::UInt64 => {
            let a = col.as_primitive::<arrow_array::types::UInt64Type>();
            // quoin's adapter takes i64/u64; pass through as Int64 bit-for-bit is
            // wrong for huge u64, so feed an actual UInt64 is unsupported — widen
            // is impossible. Re-emit as Int64 only when it fits, else skip.
            if a.iter().flatten().all(|v| v <= i64::MAX as u64) {
                Some(Arc::new(Int64Array::from_iter(a.iter().map(|o| o.map(|v| v as i64)))))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn elem_bytes(dt: &DataType) -> usize {
    match dt {
        DataType::Float64 | DataType::Int64 | DataType::UInt64 => 8,
        _ => 4,
    }
}

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
    ParquetRecordBatchReaderBuilder::try_new(Bytes::from(bytes.to_vec()))
        .unwrap()
        .build()
        .unwrap()
        .map(|b| b.unwrap().num_rows())
        .sum()
}

#[derive(Default, Clone, Copy)]
struct Acc {
    raw: usize,
    comp: usize,
    enc_s: f64,
    dec_s: f64,
    cols: usize,
}

fn main() {
    let path = std::env::var("PARQUET_FILE").expect("set PARQUET_FILE");
    let trials: usize = std::env::var("RP_TRIALS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let max_cols: usize = std::env::var("RP_MAXCOLS").ok().and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    let block_values: usize = std::env::var("BLOCK_VALUES")
        .ok()
        .and_then(|s| s.parse().ok())
        .map(|b: usize| if b == 0 { 0 } else { b.min(quoin::MAX_BLOCK_SIZE) })
        .unwrap_or(0);
    let block_rows = (block_values > 0).then_some(block_values);

    // Read the whole file as one batch (one array per column).
    let file = std::fs::File::open(&path).expect("open parquet");
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    let total_rows = builder.metadata().file_metadata().num_rows() as usize;
    let schema = builder.schema().clone();
    let mut reader = builder.with_batch_size(total_rows + 1).build().unwrap();
    let batch = reader.next().expect("at least one batch").unwrap();

    println!("{}", quoin::VERSION);
    let bd = match block_rows {
        Some(b) => format!("fixed {b} rows/block"),
        None => "native blocks".to_string(),
    };
    println!(
        "real Parquet: {path}\n{} rows, {} cols ({} numeric), median of {trials}, {bd}\n",
        batch.num_rows(),
        schema.fields().len(),
        schema.fields().iter().filter(|f| to_quoin_array_dt(f.data_type())).count(),
    );
    println!("{:<28} {:<10} {:<14} {:>8} {:>9} {:>9}", "column", "type", "codec", "ratio", "enc MB/s", "dec MB/s");

    let mut accs: std::collections::BTreeMap<&'static str, Acc> = std::collections::BTreeMap::new();
    let mut shown = 0usize;

    for (i, field) in schema.fields().iter().enumerate() {
        if shown >= max_cols {
            break;
        }
        let col = batch.column(i);
        let Some(arr) = to_quoin_array(col) else { continue };
        shown += 1;
        let raw = arr.len() * elem_bytes(arr.data_type());
        let orig_ty = format!("{}", field.data_type());

        let mut row = |codec: &'static str, comp: usize, enc_s: f64, dec_s: f64| {
            println!(
                "{:<28} {:<10} {codec:<14} {:>7.2}x {:>9.0} {:>9.0}",
                truncate(field.name(), 28),
                truncate(&orig_ty, 10),
                raw as f64 / comp.max(1) as f64,
                mbps(raw, enc_s),
                mbps(raw, dec_s),
            );
            let a = accs.entry(codec).or_default();
            a.raw += raw;
            a.comp += comp;
            a.enc_s += enc_s;
            a.dec_s += dec_s;
            a.cols += 1;
        };

        for (codec, cfg) in [
            ("quoin-fast", Config { level: quoin::Level::Fast, block_size: block_rows, ..Config::default() }),
            ("quoin-max", Config { block_size: block_rows, ..Config::default() }),
        ] {
            let (enc_s, packed) = time_median(trials, || compress_array(arr.as_ref(), cfg).unwrap());
            let (dec_s, back) = time_median(trials, || decompress_array(&packed).unwrap());
            assert_eq!(back.len(), arr.len(), "{}/{codec} len", field.name());
            row(codec, packed.len(), enc_s, dec_s);
        }
        let (enc_s, packed) = time_median(trials, || {
            parquet_bytes(&arr, Compression::ZSTD(ZstdLevel::try_new(9).unwrap()), block_rows)
        });
        let (dec_s, rows) = time_median(trials, || parquet_read_rows(&packed));
        assert_eq!(rows, arr.len());
        row("parquet-zstd9", packed.len(), enc_s, dec_s);
        println!();
    }

    println!("AGGREGATE over {shown} numeric columns (throughput-weighted):");
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

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n { s.to_string() } else { format!("{}…", &s[..n - 1]) }
}

fn to_quoin_array_dt(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Int64 | DataType::Int32 | DataType::UInt32 | DataType::UInt64
            | DataType::Float64 | DataType::Float32 | DataType::Int16 | DataType::UInt16
            | DataType::Int8 | DataType::UInt8
    )
}
