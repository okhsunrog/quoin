//! Benchmark harness: compression ratio + encode/decode throughput across the
//! same 17 synthetic datasets as the upstream C `fc` test harness.
//!
//! Usage:
//!   cargo run --release --example compare
//!   cargo run --release --example compare --features bench-zstd
//!   FC_SRC_DIR=../fc cargo run --release --example compare --features bench-zstd,bench-fc
//!
//! Env: FCBENCH_N (values per dataset, default 1<<20), FCBENCH_TRIALS (default 5).

// The dataset generators are ported 1:1 from fc/test_fc.c, including its
// hand-written `3.14159265358979` literal — keep it for byte-identical data.
#![allow(clippy::approx_constant)]

use std::time::Instant;

use quoin::{Config, compress, decompress};

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

fn bench_ours(data: &[f64], trials: usize) -> Row {
    let orig = data.len() * 8;
    let mut packed = Vec::new();
    quoin::reset_mode_win_counts();
    let (enc_s, _) = time_median(trials, || {
        packed = compress(data, Config::default());
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
        "{ds:<16} {codec:<10} {:>10} {:>8.2}x {:>9.0} {:>9.0}{tail}",
        r.comp, r.ratio, r.enc_mbps, r.dec_mbps
    );
}

fn main() {
    let n: usize = std::env::var("FCBENCH_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1 << 20);
    let trials: usize = std::env::var("FCBENCH_TRIALS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    println!("{}", quoin::VERSION);
    println!(
        "datasets: {} x {n} values ({} MiB each), median of {trials}\n",
        DATASETS.len(),
        n * 8 / (1 << 20)
    );
    println!(
        "{:<16} {:<10} {:>10} {:>9} {:>9} {:>9}",
        "dataset", "codec", "bytes", "ratio", "enc MB/s", "dec MB/s"
    );

    // Aggregates for `ours`.
    let mut tot_orig = 0usize;
    let mut tot_comp = 0usize;

    for (name, make) in DATASETS {
        let data = make(n);
        let orig = data.len() * 8;

        let ours = bench_ours(&data, trials);
        tot_orig += orig;
        tot_comp += ours.comp;
        print_row(name, "ours", &ours);

        #[cfg(feature = "bench-zstd")]
        {
            print_row(name, "zstd-3", &bench_zstd(&data, 3, trials));
            print_row(name, "zstd-9", &bench_zstd(&data, 9, trials));
        }
        #[cfg(feature = "bench-fc")]
        print_row(name, "fc(C)", &bench_fc(&data, trials));

        println!();
    }

    println!(
        "AGGREGATE (ours): {} -> {} bytes, overall ratio {:.2}x",
        tot_orig,
        tot_comp,
        tot_orig as f64 / tot_comp as f64
    );
}
