//! README benchmark harness: emits CSV of ratio + encode/decode throughput for
//! quoin (three levels) vs lz4 / zlib / zstd on real f64 columns from the ALP
//! corpus. Two sections:
//!   VOLUME — one real column truncated to a sweep of sizes (10K..full).
//!   BREADTH — a spread of real columns at a fixed size.
//!
//! Usage:
//!   cargo run --release --example bench_readme \
//!       --features bench-zstd,bench-lz4,bench-deflate > /tmp/bench.csv
//! Env: ALP_DIR (default datasets/alp), BR_TRIALS (default 3).

use std::path::Path;
use std::time::Instant;

use quoin::{Config, Level, compress, decompress};

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

struct Row {
    ratio: f64,
    enc_mbps: f64,
    dec_mbps: f64,
    ok: bool,
}

fn as_bytes(data: &[f64]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 8) }
}

fn bench_quoin(data: &[f64], level: Level, trials: usize) -> Row {
    let orig = data.len() * 8;
    let cfg = Config { level, ..Config::default() };
    let mut packed = Vec::new();
    let (enc_s, _) = time_median(trials, || {
        packed = compress(data, cfg);
        packed.len()
    });
    let (dec_s, restored) = time_median(trials, || decompress(&packed).expect("decode"));
    let ok = data.iter().map(|f| f.to_bits()).eq(restored.iter().map(|f| f.to_bits()));
    Row { ratio: orig as f64 / packed.len().max(1) as f64, enc_mbps: mbps(orig, enc_s), dec_mbps: mbps(orig, dec_s), ok }
}

#[cfg(feature = "bench-lz4")]
fn bench_lz4(data: &[f64], trials: usize) -> Row {
    let orig = data.len() * 8;
    let bytes = as_bytes(data);
    let mut packed = Vec::new();
    let (enc_s, _) = time_median(trials, || { packed = lz4_flex::compress_prepend_size(bytes); packed.len() });
    let (dec_s, restored) = time_median(trials, || lz4_flex::decompress_size_prepended(&packed).expect("lz4 dec"));
    Row { ratio: orig as f64 / packed.len().max(1) as f64, enc_mbps: mbps(orig, enc_s), dec_mbps: mbps(orig, dec_s), ok: restored == bytes }
}

#[cfg(feature = "bench-zstd")]
fn bench_zstd(data: &[f64], level: i32, trials: usize) -> Row {
    let orig = data.len() * 8;
    let bytes = as_bytes(data);
    let mut packed = Vec::new();
    let (enc_s, _) = time_median(trials, || { packed = zstd::bulk::compress(bytes, level).expect("zstd enc"); packed.len() });
    let (dec_s, restored) = time_median(trials, || zstd::bulk::decompress(&packed, orig).expect("zstd dec"));
    Row { ratio: orig as f64 / packed.len().max(1) as f64, enc_mbps: mbps(orig, enc_s), dec_mbps: mbps(orig, dec_s), ok: restored == bytes }
}

#[cfg(feature = "bench-deflate")]
fn bench_zlib(data: &[f64], level: u32, trials: usize) -> Row {
    use flate2::Compression;
    use flate2::read::{ZlibDecoder, ZlibEncoder};
    use std::io::Read;
    let orig = data.len() * 8;
    let bytes = as_bytes(data);
    let mut packed = Vec::new();
    let (enc_s, _) = time_median(trials, || {
        packed.clear();
        ZlibEncoder::new(bytes, Compression::new(level)).read_to_end(&mut packed).expect("zlib enc");
        packed.len()
    });
    let (dec_s, restored) = time_median(trials, || {
        let mut out = Vec::with_capacity(orig);
        ZlibDecoder::new(&packed[..]).read_to_end(&mut out).expect("zlib dec");
        out
    });
    Row { ratio: orig as f64 / packed.len().max(1) as f64, enc_mbps: mbps(orig, enc_s), dec_mbps: mbps(orig, dec_s), ok: restored == bytes }
}

fn load_f64_bin(path: &Path) -> Vec<f64> {
    let bytes = std::fs::read(path).expect("read .bin");
    bytes.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
}

// Each codec under test, as (name, closure). zstd/lz4/zlib gated by feature.
fn run_codecs(label: &str, n: usize, data: &[f64], trials: usize) {
    let emit = |codec: &str, r: Row| {
        println!("{label},{n},{codec},{:.4},{:.1},{:.1},{}", r.ratio, r.enc_mbps, r.dec_mbps, r.ok as u8);
    };
    emit("quoin-balanced", bench_quoin(data, Level::Balanced, trials));
    emit("quoin-high", bench_quoin(data, Level::High, trials));
    emit("quoin-max", bench_quoin(data, Level::Max, trials));
    #[cfg(feature = "bench-lz4")]
    emit("lz4", bench_lz4(data, trials));
    #[cfg(feature = "bench-deflate")]
    emit("zlib-6", bench_zlib(data, 6, trials));
    #[cfg(feature = "bench-zstd")]
    {
        emit("zstd-3", bench_zstd(data, 3, trials));
        emit("zstd-19", bench_zstd(data, 19, trials));
    }
}

fn main() {
    let dir = std::env::var("ALP_DIR").unwrap_or_else(|_| "datasets/alp".into());
    let trials: usize = std::env::var("BR_TRIALS").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let dir = Path::new(&dir);

    // CSV header.
    println!("section,n,codec,ratio,enc_mbps,dec_mbps,ok");

    // --- VOLUME sweep: one real column truncated to a range of sizes. ---
    let vol_col = load_f64_bin(&dir.join("arade4.bin")); // ~9.9M f64
    eprintln!("volume column arade4.bin: {} values", vol_col.len());
    for &n in &[10_000usize, 100_000, 1_000_000, vol_col.len()] {
        let n = n.min(vol_col.len());
        eprintln!("  volume n={n}");
        run_codecs("volume", n, &vol_col[..n], trials);
    }

    // --- BREADTH: a spread of real columns at a fixed cap. ---
    let breadth = [
        "air_sensor_f.bin",
        "bird_migration_f.bin",
        "basel_wind_f.bin",
        "poi_lat.bin",
        "city_temperature_f.bin",
        "food_prices.bin",
        "neon_dew_point_temp.bin",
        "bitcoin_transactions_f.bin",
    ];
    let cap = 2_000_000usize;
    for name in breadth {
        let path = dir.join(name);
        if !path.exists() { eprintln!("  skip {name} (missing)"); continue; }
        let col = load_f64_bin(&path);
        let n = col.len().min(cap);
        eprintln!("  breadth {name}: {n} values");
        // section field carries the dataset name (strip .bin).
        run_codecs(name.trim_end_matches(".bin"), n, &col[..n], trials);
    }
}
