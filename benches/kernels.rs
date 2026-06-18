//! Kernel-level micro-benchmarks (criterion).
//!
//! These isolate the hot kernels so SIMD / codec decisions rest on
//! statistically sound measurements (warmup, many samples, confidence
//! intervals) rather than the noisy one-shot timing in `src/bin/bench.rs`.
//! That noise is exactly why the `target-cpu=native` A/B was inconclusive.
//!
//! Run: `cargo bench --bench kernels`
//! Compare scalar vs AVX2 codegen: `RUSTFLAGS="-C target-cpu=native" cargo bench`

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use quoin::bench_internals as bi;
use quoin::{Config, compress, decompress};
use std::hint::black_box;

const N: usize = 1 << 16; // 64Ki values = 512 KiB per input

fn smooth_bits() -> Vec<u64> {
    (0..N).map(|i| (i as f64 * 1e-4).sin().to_bits()).collect()
}

fn dict_bits() -> Vec<u64> {
    (0..N).map(|i| ((i & 15) as f64).to_bits()).collect()
}

fn smooth_f64() -> Vec<f64> {
    (0..N).map(|i| (i as f64 * 1e-4).sin()).collect()
}

fn bench_hash(c: &mut Criterion) {
    let vals = smooth_bits();
    let mut g = c.benchmark_group("crc32c_hash");
    g.throughput(Throughput::Bytes((vals.len() * 8) as u64));
    g.bench_function("hardware", |b| {
        b.iter(|| bi::hash_fold_best(black_box(&vals)))
    });
    g.bench_function("software", |b| {
        b.iter(|| bi::hash_fold_sw(black_box(&vals)))
    });
    g.finish();
}

fn bench_entropy(c: &mut Criterion) {
    // A realistic residual stream: DFCM residuals of a smooth signal.
    let residuals = bi::dfcm_encode(&smooth_bits(), 16);
    let rc = bi::rc_compress(&residuals);
    let rans = bi::rans_compress(&residuals);

    let mut g = c.benchmark_group("entropy");
    g.throughput(Throughput::Bytes(residuals.len() as u64));
    // The order-1 binary range coder: best ratio, slowest decode.
    g.bench_function("rc_compress", |b| {
        b.iter(|| bi::rc_compress(black_box(&residuals)))
    });
    g.bench_function("rc_decompress", |b| {
        b.iter(|| bi::rc_decompress(black_box(&rc), residuals.len()).unwrap())
    });
    // The 4-way interleaved rANS coder: the default at `Balanced`. Benchmarking
    // it (not the decode-only legacy tANS) is what calibrates W_RC vs W_RANS.
    if let Some(r) = rans {
        g.bench_function("rans_compress", |b| {
            b.iter(|| bi::rans_compress(black_box(&residuals)))
        });
        g.bench_function("rans_decompress", |b| {
            b.iter(|| bi::rans_decompress(black_box(&r), residuals.len()).unwrap())
        });
    }
    g.finish();
}

fn bench_transpose(c: &mut Criterion) {
    let aos: Vec<u8> = smooth_bits().iter().flat_map(|v| v.to_le_bytes()).collect();
    let n = aos.len() / 8;
    let mut dst = vec![0u8; aos.len()];
    let mut g = c.benchmark_group("transpose");
    g.throughput(Throughput::Bytes(aos.len() as u64));
    g.bench_function("byte_transpose", |b| {
        b.iter(|| bi::byte_transpose(black_box(&aos), n, black_box(&mut dst)))
    });
    g.finish();
}

fn bench_bitpack(c: &mut Criterion) {
    // 1024 values needing ~11 bits (a typical FoR residual width).
    let mut s = 1u32;
    let mut values = [0u32; 1024];
    for v in values.iter_mut() {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        *v = (s >> 21) & 0x7FF;
    }
    let width = 11u32;
    let mut packed = vec![0u32; 32 * width as usize];
    bi::bitpack(&values, width, &mut packed);
    let mut out = [0u32; 1024];

    let mut g = c.benchmark_group("bitpack");
    g.throughput(Throughput::Bytes(1024 * 4));
    g.bench_function("pack_w11", |b| {
        b.iter(|| bi::bitpack(black_box(&values), width, black_box(&mut packed)))
    });
    g.bench_function("unpack_w11", |b| {
        b.iter(|| bi::bitunpack(black_box(&packed), width, black_box(&mut out)))
    });
    g.finish();
}

fn bench_predictors(c: &mut Criterion) {
    let vals = smooth_bits();
    let mut g = c.benchmark_group("predictors");
    g.throughput(Throughput::Bytes((vals.len() * 8) as u64));
    g.bench_function("fcm_encode", |b| {
        b.iter(|| bi::fcm_encode(black_box(&vals), 16))
    });
    g.bench_function("dfcm_encode", |b| {
        b.iter(|| bi::dfcm_encode(black_box(&vals), 16))
    });
    g.finish();
}

fn bench_pipeline(c: &mut Criterion) {
    let smooth = smooth_f64();
    let dict: Vec<f64> = dict_bits().iter().map(|&b| f64::from_bits(b)).collect();
    let packed_smooth = compress(&smooth, Config::default());

    let mut g = c.benchmark_group("pipeline");
    g.throughput(Throughput::Bytes((N * 8) as u64));
    g.bench_function("compress_smooth", |b| {
        b.iter(|| compress(black_box(&smooth), Config::default()))
    });
    g.bench_function("compress_dict", |b| {
        b.iter(|| compress(black_box(&dict), Config::default()))
    });
    g.bench_function("decompress_smooth", |b| {
        b.iter(|| decompress(black_box(&packed_smooth)).unwrap())
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_hash,
    bench_entropy,
    bench_transpose,
    bench_bitpack,
    bench_predictors,
    bench_pipeline
);
criterion_main!(benches);
