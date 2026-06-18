//! Encode-only loop for profiling (perf / VTune). Reads one raw little-endian
//! f64 `.bin` column and compresses it N times at a fixed level — no decode, no
//! baselines, so a profiler attributes time to the encode path only.
//!
//! Build:  cargo build --release --no-default-features --example profile_encode
//! Run:    PROF_FILE=datasets/alp/poi_lat.bin PROF_LEVEL=fast PROF_ITERS=200 \
//!           taskset -c 0 ./target/release/examples/profile_encode
//! Env: PROF_FILE (required), PROF_LEVEL (fastest|fast|balanced|high|max),
//!      PROF_ITERS (default 200), PROF_N (cap values, default all).

use std::time::Instant;
use quoin::{Config, Level, compress};

fn main() {
    let path = std::env::var("PROF_FILE").expect("set PROF_FILE");
    let iters: usize = std::env::var("PROF_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(200);
    let level = match std::env::var("PROF_LEVEL").as_deref() {
        Ok("fastest") => Level::Fastest,
        Ok("fast") => Level::Fast,
        Ok("balanced") => Level::Balanced,
        Ok("high") => Level::High,
        _ => Level::Max,
    };
    let bytes = std::fs::read(&path).expect("read .bin");
    let mut data: Vec<f64> = bytes.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
    if let Some(n) = std::env::var("PROF_N").ok().and_then(|s| s.parse::<usize>().ok()) {
        data.truncate(n);
    }
    let cfg = Config { level, ..Config::default() };

    // Warm up, then timed loop.
    let mut sink = 0usize;
    for _ in 0..5 { sink ^= compress(&data, cfg).len(); }
    let t = Instant::now();
    for _ in 0..iters { sink ^= compress(std::hint::black_box(&data), cfg).len(); }
    let secs = t.elapsed().as_secs_f64();
    let mbps = (data.len() * 8 * iters) as f64 / secs / 1e6;
    eprintln!("{path}: {} values, {level:?}, {iters} iters, {mbps:.0} MB/s encode (sink={sink})",
              data.len());
}
