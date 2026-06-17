//! Benchmark harness: compression ratio + encode/decode throughput.
//!
//! Two modes:
//!   * synthetic — the same 17 generators as the upstream C `fc` test harness.
//!   * real files — set `ALP_DIR` to a directory of raw little-endian `f64`
//!     `.bin` columns (e.g. the ALP benchmark corpus); each file is benchmarked
//!     across the four [`quoin::Level`]s plus the enabled baselines.
//!
//! Usage:
//!   cargo run --release --example compare
//!   cargo run --release --example compare --features bench-zstd,bench-lz4,bench-deflate
//!   ALP_DIR=datasets/alp FCBENCH_TRIALS=2 cargo run --release --example compare \
//!       --features bench-zstd,bench-lz4,bench-deflate,bench-fc
//!
//! Env: FCBENCH_N (synthetic values/dataset, default 1<<20),
//!      FCBENCH_TRIALS (default 5), ALP_DIR (real-file mode), FCBENCH_SELECT=sample.

// The dataset generators are ported 1:1 from fc/test_fc.c, including its
// hand-written `3.14159265358979` literal — keep it for byte-identical data.
#![allow(clippy::approx_constant)]

use std::path::Path;
use std::time::Instant;

use quoin::{Config, Level, compress, decompress};

// ---------------------------------------------------------------------------
// Synthetic datasets, ported 1:1 from fc/test_fc.c so results are comparable.
// ---------------------------------------------------------------------------

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s
}

// ((double)((int32_t)(s>>32)) / (double)INT32_MAX), matching the C generators.
fn rnd_unit(s: &mut u64) -> f64 {
    let v = lcg(s);
    let i = (v >> 32) as i32;
    f64::from(i) / f64::from(i32::MAX)
}

type Gen = fn(usize) -> Vec<f64>;

fn g_constant(n: usize) -> Vec<f64> {
    vec![3.14159265358979; n]
}
fn g_linear(n: usize) -> Vec<f64> {
    (0..n).map(|i| 1.0 + 1e-3 * i as f64).collect()
}
fn g_parabolic(n: usize) -> Vec<f64> {
    (0..n).map(|i| 0.5 * i as f64 * i as f64).collect()
}
fn g_ar2(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64;
            (0.05 * t).sin() * (-1e-6 * t).exp()
        })
        .collect()
}
fn g_piecewise(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            if i < n / 2 {
                1.0
            } else {
                1.0 + 1e-3 * (i - n / 2) as f64
            }
        })
        .collect()
}
fn g_intmul(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i as i64 * 1000) as f64).collect()
}
fn g_decimal(n: usize) -> Vec<f64> {
    (0..n).map(|i| 0.01 * (i & 1023) as f64).collect()
}
fn g_dict16(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i & 15) as f64).collect()
}
fn g_sin_lo(n: usize) -> Vec<f64> {
    (0..n).map(|i| (1e-4 * i as f64).sin()).collect()
}
fn g_audio(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64;
            0.4 * (1e-4 * t).sin()
                + 0.3 * (5e-4 * t).sin()
                + 0.2 * (1.3e-3 * t).sin()
                + 0.1 * (4.1e-3 * t).sin()
                + 0.05 * (1.07e-2 * t).sin()
        })
        .collect()
}
fn g_walk(n: usize) -> Vec<f64> {
    let mut s = 0xfeedfaceu64;
    let mut pos = 0.0;
    (0..n)
        .map(|_| {
            pos += rnd_unit(&mut s) * 0.01;
            pos
        })
        .collect()
}
fn g_quant4(n: usize) -> Vec<f64> {
    let l = [0.0, 1.5, 2.7, 4.1];
    (0..n).map(|i| l[(i / 1024) % 4]).collect()
}
fn g_climate(n: usize) -> Vec<f64> {
    let mut s = 0xc0ffeeu64;
    (0..n)
        .map(|i| {
            let t = i as f64;
            let trend = 1e-5 * t;
            let seasonal = 0.5 * (2.0 * std::f64::consts::PI * t / 1024.0).sin();
            let noise = rnd_unit(&mut s) * 0.01;
            trend + seasonal + noise
        })
        .collect()
}
fn g_geo(n: usize) -> Vec<f64> {
    let mut s = 0xa11ce5u64;
    let mut lat = 37.7749;
    (0..n)
        .map(|_| {
            lat += rnd_unit(&mut s) * 1e-5;
            lat
        })
        .collect()
}
fn g_stocks(n: usize) -> Vec<f64> {
    let mut s = 0xbeef01u64;
    let mut p = 100.0;
    (0..n)
        .map(|_| {
            let r = rnd_unit(&mut s) * 5e-4;
            p *= 1.0 + r;
            (p * 100.0).floor() / 100.0
        })
        .collect()
}
fn g_sensor(n: usize) -> Vec<f64> {
    let mut s = 0x5e1750u64;
    (0..n)
        .map(|i| {
            let t = i as f64;
            let base = 20.0 + 5.0 * (t * 1e-4).sin();
            let noise = rnd_unit(&mut s) * 0.05;
            base + noise
        })
        .collect()
}
fn g_random(n: usize) -> Vec<f64> {
    let mut s = 0xdeadbeefu64;
    (0..n).map(|_| f64::from_bits(lcg(&mut s))).collect()
}

// Integer columns smuggled through the f64-bits API (compress() reads to_bits),
// to exercise the integer codecs (FoR+bitpack). NOT floats — these are i64
// columns as quoin sees them internally.
fn g_int_narrow(n: usize) -> Vec<f64> {
    // base + uniform 12-bit noise: bounded range, high entropy, not periodic or
    // delta-predictable — frame-of-reference + bit-packing's niche.
    let mut s = 0x1234_abcdu64;
    (0..n)
        .map(|_| {
            s = lcg(&mut s);
            f64::from_bits(1_000_000 + ((s >> 20) & 0xFFF))
        })
        .collect()
}
// Random cent-values with ~0.5% non-decimal outliers — FLOAT_MULT bails on a
// single bad value, but ALP stores them as exceptions and keeps the scaled-int
// encoding for the rest.
fn g_decimal_outliers(n: usize) -> Vec<f64> {
    let mut s = 0xabc_defu64;
    (0..n)
        .map(|_| {
            s = lcg(&mut s);
            if s.is_multiple_of(200) {
                f64::from_bits(s) // rare arbitrary outlier
            } else {
                (s % 100_000) as f64 / 100.0 // random 0.00..=999.99
            }
        })
        .collect()
}
// Monotonic timestamp column (ms): regular ~1s step + small jitter. First-order
// delta + bit-packing's niche (Parquet DELTA_BINARY_PACKED).
fn g_timestamps(n: usize) -> Vec<f64> {
    let mut s = 0x7777u64;
    let mut t = 1_700_000_000_000u64;
    (0..n)
        .map(|_| {
            s = lcg(&mut s);
            t = t.wrapping_add(1000 + (s >> 32) % 4096); // bounded but high-entropy gaps
            f64::from_bits(t)
        })
        .collect()
}
fn g_int_walk(n: usize) -> Vec<f64> {
    // slowly increasing ids with small per-row deltas.
    let mut s = 0x51edu64;
    let mut v = 5_000_000u64;
    (0..n)
        .map(|_| {
            s = lcg(&mut s);
            v = v.wrapping_add(s % 16);
            f64::from_bits(v)
        })
        .collect()
}

const DATASETS: &[(&str, Gen)] = &[
    ("constant", g_constant),
    ("linear", g_linear),
    ("parabolic", g_parabolic),
    ("ar2-damped", g_ar2),
    ("piecewise", g_piecewise),
    ("int-x1000", g_intmul),
    ("decimal-cents", g_decimal),
    ("dict-16", g_dict16),
    ("sin-low-freq", g_sin_lo),
    ("audio-mix", g_audio),
    ("random-walk", g_walk),
    ("quantized-4lvl", g_quant4),
    ("climate", g_climate),
    ("geo-coords", g_geo),
    ("stocks", g_stocks),
    ("sensor-noisy", g_sensor),
    ("pseudo-random", g_random),
    ("int-narrow", g_int_narrow),
    ("int-walk", g_int_walk),
    ("decimal-outliers", g_decimal_outliers),
    ("timestamps", g_timestamps),
];

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

/// Run `f` `trials` times, returning the median elapsed seconds and the last value.
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

struct Row {
    comp: usize,
    ratio: f64,
    enc_mbps: f64,
    dec_mbps: f64,
    ok: bool,
    modes: String,
}

/// Top winning modes from a `mode_win_counts` snapshot, as `NAME×n` pairs.
fn top_modes(counts: &[u64; 64]) -> String {
    let mut v: Vec<(usize, u64)> = counts
        .iter()
        .enumerate()
        .filter(|(_, c)| **c > 0)
        .map(|(i, c)| (i, *c))
        .collect();
    v.sort_by_key(|&(_, c)| std::cmp::Reverse(c));
    v.iter()
        .take(3)
        .map(|(m, c)| format!("{}×{}", quoin::mode_name(*m as u8), c))
        .collect::<Vec<_>>()
        .join(" ")
}

fn bits_eq(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter()
            .map(|f| f.to_bits())
            .eq(b.iter().map(|f| f.to_bits()))
}

fn mbps(orig_bytes: usize, secs: f64) -> f64 {
    if secs <= 0.0 {
        f64::INFINITY
    } else {
        orig_bytes as f64 / 1e6 / secs
    }
}

// ---------------------------------------------------------------------------
// Codecs under test
// ---------------------------------------------------------------------------

/// Encoder config, with FCBENCH_SELECT=sample switching to sampling selection
/// so the two strategies can be A/B-benchmarked on the same datasets.
fn our_config() -> Config {
    let mut c = Config::default();
    if std::env::var("FCBENCH_SELECT").as_deref() == Ok("sample") {
        c.selection = quoin::Selection::Sample;
    }
    c
}

fn bench_ours(data: &[f64], trials: usize) -> Row {
    bench_ours_cfg(data, our_config(), trials)
}

fn bench_ours_cfg(data: &[f64], cfg: Config, trials: usize) -> Row {
    let orig = data.len() * 8;
    let mut packed = Vec::new();
    quoin::reset_mode_win_counts();
    let (enc_s, _) = time_median(trials, || {
        packed = compress(data, cfg);
        packed.len()
    });
    let modes = top_modes(&quoin::mode_win_counts());
    let (dec_s, restored) = time_median(trials, || decompress(&packed).expect("decode"));
    Row {
        comp: packed.len(),
        ratio: orig as f64 / packed.len().max(1) as f64,
        enc_mbps: mbps(orig, enc_s),
        dec_mbps: mbps(orig, dec_s),
        ok: bits_eq(data, &restored),
        modes,
    }
}

#[cfg(feature = "bench-lz4")]
fn bench_lz4(data: &[f64], trials: usize) -> Row {
    let orig = data.len() * 8;
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, orig) };
    let mut packed = Vec::new();
    let (enc_s, _) = time_median(trials, || {
        packed = lz4_flex::compress_prepend_size(bytes);
        packed.len()
    });
    let (dec_s, restored) = time_median(trials, || {
        lz4_flex::decompress_size_prepended(&packed).expect("lz4 dec")
    });
    Row {
        comp: packed.len(),
        ratio: orig as f64 / packed.len().max(1) as f64,
        enc_mbps: mbps(orig, enc_s),
        dec_mbps: mbps(orig, dec_s),
        ok: restored == bytes,
        modes: String::new(),
    }
}

#[cfg(feature = "bench-deflate")]
fn bench_deflate(data: &[f64], level: u32, trials: usize) -> Row {
    use flate2::Compression;
    use flate2::read::{ZlibDecoder, ZlibEncoder};
    use std::io::Read;
    let orig = data.len() * 8;
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, orig) };
    let mut packed = Vec::new();
    let (enc_s, _) = time_median(trials, || {
        packed.clear();
        ZlibEncoder::new(bytes, Compression::new(level))
            .read_to_end(&mut packed)
            .expect("deflate enc");
        packed.len()
    });
    let (dec_s, restored) = time_median(trials, || {
        let mut out = Vec::with_capacity(orig);
        ZlibDecoder::new(&packed[..])
            .read_to_end(&mut out)
            .expect("deflate dec");
        out
    });
    Row {
        comp: packed.len(),
        ratio: orig as f64 / packed.len().max(1) as f64,
        enc_mbps: mbps(orig, enc_s),
        dec_mbps: mbps(orig, dec_s),
        ok: restored == bytes,
        modes: String::new(),
    }
}

/// Load a column of raw little-endian `f64` (the ALP corpus `.bin` format).
fn load_f64_bin(path: &Path) -> Vec<f64> {
    let bytes = std::fs::read(path).expect("read .bin");
    assert!(
        bytes.len().is_multiple_of(8),
        "{path:?}: not a multiple of 8 bytes"
    );
    bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[cfg(feature = "bench-zstd")]
fn bench_zstd(data: &[f64], level: i32, trials: usize) -> Row {
    let orig = data.len() * 8;
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, orig) };
    let mut packed = Vec::new();
    let (enc_s, _) = time_median(trials, || {
        packed = zstd::bulk::compress(bytes, level).expect("zstd enc");
        packed.len()
    });
    let (dec_s, restored) = time_median(trials, || {
        zstd::bulk::decompress(&packed, orig).expect("zstd dec")
    });
    Row {
        comp: packed.len(),
        ratio: orig as f64 / packed.len().max(1) as f64,
        enc_mbps: mbps(orig, enc_s),
        dec_mbps: mbps(orig, dec_s),
        ok: restored == bytes,
        modes: String::new(),
    }
}

#[cfg(feature = "bench-fc")]
mod fc_ffi {
    use std::os::raw::{c_int, c_void};
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct FcCfg {
        pub p: c_int,
        pub t: c_int,
        pub c: c_int,
    }
    unsafe extern "C" {
        pub fn fc_enc(src: *const c_void, bytes: usize, dst: *mut c_void, cfg: FcCfg) -> usize;
        pub fn fc_dec(src: *const c_void, bytes: usize, dst: *mut c_void) -> usize;
    }
}

#[cfg(feature = "bench-fc")]
fn bench_fc(data: &[f64], trials: usize) -> Row {
    use std::os::raw::c_void;
    let orig = data.len() * 8;
    let mut dst = vec![0u8; 2 * orig + 65536];
    let cfg = fc_ffi::FcCfg { p: 18, t: 8, c: 0 };
    let mut clen = 0usize;
    let (enc_s, _) = time_median(trials, || {
        clen = unsafe {
            fc_ffi::fc_enc(
                data.as_ptr() as *const c_void,
                orig,
                dst.as_mut_ptr() as *mut c_void,
                cfg,
            )
        };
        clen
    });
    let mut out = vec![0f64; data.len()];
    let (dec_s, _) = time_median(trials, || unsafe {
        fc_ffi::fc_dec(
            dst.as_ptr() as *const c_void,
            clen,
            out.as_mut_ptr() as *mut c_void,
        )
    });
    Row {
        comp: clen,
        ratio: orig as f64 / clen.max(1) as f64,
        enc_mbps: mbps(orig, enc_s),
        dec_mbps: mbps(orig, dec_s),
        ok: bits_eq(data, &out),
        modes: String::new(),
    }
}

// ---------------------------------------------------------------------------

fn print_row(ds: &str, codec: &str, r: &Row) {
    let flag = if r.ok { "" } else { "  !! MISMATCH" };
    let tail = if r.modes.is_empty() {
        flag.to_string()
    } else {
        format!("  {}{flag}", r.modes)
    };
    println!(
        "{ds:<22} {codec:<11} {:>12} {:>7.2}x {:>9.0} {:>9.0}{tail}",
        r.comp, r.ratio, r.enc_mbps, r.dec_mbps
    );
}

/// Print the baseline rows (zstd/lz4/deflate/fc) for one dataset.
fn print_baselines(name: &str, data: &[f64], trials: usize) {
    #[cfg(feature = "bench-zstd")]
    {
        print_row(name, "zstd-9", &bench_zstd(data, 9, trials));
        print_row(name, "zstd-19", &bench_zstd(data, 19, trials));
    }
    #[cfg(feature = "bench-lz4")]
    print_row(name, "lz4", &bench_lz4(data, trials));
    #[cfg(feature = "bench-deflate")]
    print_row(name, "deflate-6", &bench_deflate(data, 6, trials));
    #[cfg(feature = "bench-fc")]
    print_row(name, "fc(C)", &bench_fc(data, trials));
    let _ = (name, data, trials);
}

fn header() {
    println!(
        "{:<22} {:<11} {:>12} {:>8} {:>9} {:>9}",
        "dataset", "codec", "bytes", "ratio", "enc MB/s", "dec MB/s"
    );
}

/// The four levels, fastest → max-ratio, with the column label used in output.
const LEVELS: [(&str, Level); 4] = [
    ("q-fastest", Level::Fastest),
    ("q-fast", Level::Fast),
    ("q-balanced", Level::Balanced),
    ("q-max", Level::Max),
];

fn main() {
    let trials: usize = std::env::var("FCBENCH_TRIALS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    println!("{}", quoin::VERSION);

    // Aggregates for the headline config (q-max / ours).
    let mut tot_orig = 0usize;
    let mut tot_comp = 0usize;

    if let Ok(dir) = std::env::var("ALP_DIR") {
        // Real-file mode: sweep all levels per column + baselines.
        let mut files: Vec<_> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("ALP_DIR {dir:?}: {e}"))
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "bin"))
            .collect();
        files.sort();
        println!(
            "ALP corpus: {} columns from {dir}, median of {trials}\n",
            files.len()
        );
        header();
        for path in &files {
            let stem = path.file_stem().unwrap().to_string_lossy().to_string();
            let name: String = stem
                .strip_suffix("_f")
                .unwrap_or(&stem)
                .chars()
                .take(22)
                .collect();
            let data = load_f64_bin(path);
            let base = our_config();
            for (label, level) in LEVELS {
                let row = bench_ours_cfg(&data, Config { level, ..base }, trials);
                if level == Level::Max {
                    tot_orig += data.len() * 8;
                    tot_comp += row.comp;
                }
                print_row(&name, label, &row);
            }
            print_baselines(&name, &data, trials);
            println!();
        }
        println!(
            "AGGREGATE (q-max): {} -> {} bytes, overall ratio {:.2}x",
            tot_orig,
            tot_comp,
            tot_orig as f64 / tot_comp.max(1) as f64
        );
        return;
    }

    // Synthetic mode (default).
    let n: usize = std::env::var("FCBENCH_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1 << 20);
    println!(
        "datasets: {} x {n} values ({} MiB each), median of {trials}\n",
        DATASETS.len(),
        n * 8 / (1 << 20)
    );
    header();
    for (name, make) in DATASETS {
        let data = make(n);
        let ours = bench_ours(&data, trials);
        tot_orig += data.len() * 8;
        tot_comp += ours.comp;
        print_row(name, "ours", &ours);
        print_baselines(name, &data, trials);
        println!();
    }
    println!(
        "AGGREGATE (ours): {} -> {} bytes, overall ratio {:.2}x",
        tot_orig,
        tot_comp,
        tot_orig as f64 / tot_comp as f64
    );
}
