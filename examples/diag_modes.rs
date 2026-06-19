//! Diagnostic: per level, which modes win the competition and the real decode
//! throughput. Used to investigate the "Balanced decodes slower than Max" anomaly
//! and to sanity-check decode_weight calibration.
//!
//! Run: PROF_FILE=datasets/alp/bitcoin_transactions_f.bin \
//!        taskset -c 0 cargo run --release --no-default-features --example diag_modes

use std::time::Instant;
use quoin::{Config, Level, compress, decompress, mode_name, mode_win_counts, reset_mode_win_counts};

fn main() {
    let path = std::env::var("PROF_FILE").expect("set PROF_FILE");
    let iters: usize = std::env::var("DIAG_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(50);
    let bytes = std::fs::read(&path).expect("read .bin");
    let mut data: Vec<f64> = bytes.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
    if let Some(n) = std::env::var("PROF_N").ok().and_then(|s| s.parse::<usize>().ok()) {
        data.truncate(n);
    }
    let orig = data.len() * 8;
    println!("{path}  ({} values)\n", data.len());
    println!("{:<10} {:>7} {:>10} {:>10}   winning modes (name×blocks)", "level", "ratio", "enc MB/s", "dec MB/s");

    for (level, name) in [
        (Level::Fastest, "fastest"),
        (Level::Fast, "fast"),
        (Level::Balanced, "balanced"),
        (Level::High, "high"),
        (Level::Max, "max"),
    ] {
        let cfg = Config { level, ..Config::default() };
        reset_mode_win_counts();
        let t = Instant::now();
        let mut packed = Vec::new();
        for _ in 0..3 { packed = compress(&data, cfg); }
        let enc_s = t.elapsed().as_secs_f64() / 3.0;
        let counts = mode_win_counts();

        // warm + timed decode
        for _ in 0..3 { let _ = decompress(&packed).unwrap(); }
        let t = Instant::now();
        for _ in 0..iters { let _ = std::hint::black_box(decompress(&packed).unwrap()); }
        let dec_s = t.elapsed().as_secs_f64() / iters as f64;

        let mut modes: Vec<(usize, u64)> =
            counts.iter().enumerate().filter(|(_, c)| **c > 0).map(|(i, c)| (i, *c)).collect();
        modes.sort_by_key(|&(_, c)| std::cmp::Reverse(c));
        let mstr: Vec<String> =
            modes.iter().take(4).map(|&(i, c)| format!("{}×{}", mode_name(i as u8), c)).collect();
        let ratio = orig as f64 / packed.len() as f64;
        println!("{:<10} {:>6.2}x {:>10.0} {:>10.0}   {}",
                 name, ratio, orig as f64 / 1e6 / enc_s, orig as f64 / 1e6 / dec_s, mstr.join(" "));
    }
}
