//! Pareto benchmark: compression **ratio** × **encode MB/s** × **decode MB/s**
//! for each quoin [`Level`] and the byte-level baselines (lz4 / zstd / deflate),
//! on realistic **typed** columns through `compress_column` — *not* f64-smuggled
//! integers. Reports the per-codec aggregate and marks the Pareto frontier, so
//! "does quoin dominate lz4?" is answered directly.
//!
//! Usage:
//!   cargo run --release --example pareto --features bench-zstd,bench-lz4,bench-deflate
//! Env: PARETO_N (values/column, default 1<<20), PARETO_TRIALS (default 3).

use std::collections::BTreeMap;
use std::time::Instant;

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

/// Bitshuffle filter (the HDF5/blosc one): per-block **bit-level** transpose —
/// regroup the i-th bit of every element into a contiguous plane — so a slowly
/// varying high bit becomes a long run that LZ4 then crushes. Block size 8192
/// elements (blosc default). NOTE: this naive transpose is *not* SIMD, so its
/// MB/s is a floor, not representative of the real C bitshuffle — read the
/// **ratio** as faithful and the speed as a lower bound.
#[cfg(feature = "bench-lz4")]
mod bitshuffle {
    const BLK: usize = 8192;

    pub fn transpose(src: &[u8], elem_bytes: usize) -> Vec<u8> {
        let n = src.len() / elem_bytes;
        let bits = elem_bytes * 8;
        // The transposed planes are padded to a byte per 8 elements, so for an
        // `n` that is not a multiple of 8 the last block is slightly larger than
        // the input — size `out` to the exact transposed length, not `src.len()`.
        let mut tlen = 0usize;
        let mut e = 0usize;
        while e < n {
            let cnt = (n - e).min(BLK);
            tlen += bits * cnt.div_ceil(8);
            e += cnt;
        }
        let mut out = vec![0u8; tlen];
        let mut written = 0usize;
        let mut e0 = 0usize;
        while e0 < n {
            let cnt = (n - e0).min(BLK);
            let plane_bytes = cnt.div_ceil(8);
            let block = &mut out[written..written + bits * plane_bytes];
            for j in 0..cnt {
                let elem = &src[(e0 + j) * elem_bytes..(e0 + j + 1) * elem_bytes];
                for b in 0..bits {
                    if (elem[b >> 3] >> (b & 7)) & 1 != 0 {
                        block[b * plane_bytes + (j >> 3)] |= 1 << (j & 7);
                    }
                }
            }
            written += bits * plane_bytes;
            e0 += cnt;
        }
        out
    }

    pub fn untranspose(src: &[u8], elem_bytes: usize, n: usize) -> Vec<u8> {
        let bits = elem_bytes * 8;
        let mut out = vec![0u8; n * elem_bytes];
        let mut read = 0usize;
        let mut e0 = 0usize;
        while e0 < n {
            let cnt = (n - e0).min(BLK);
            let plane_bytes = cnt.div_ceil(8);
            let block = &src[read..read + bits * plane_bytes];
            for j in 0..cnt {
                let elem = &mut out[(e0 + j) * elem_bytes..(e0 + j + 1) * elem_bytes];
                for b in 0..bits {
                    if (block[b * plane_bytes + (j >> 3)] >> (j & 7)) & 1 != 0 {
                        elem[b >> 3] |= 1 << (b & 7);
                    }
                }
            }
            read += bits * plane_bytes;
            e0 += cnt;
        }
        out
    }
}

/// A typed column under test.
enum Col {
    F64(Vec<f64>),
    I64(Vec<i64>),
    I32(Vec<i32>),
    U32(Vec<u32>),
}

impl Col {
    /// Honest raw column size: `n × sizeof(T)` (4 B for the 32-bit lanes).
    fn raw_bytes(&self) -> usize {
        match self {
            Col::F64(v) => v.len() * 8,
            Col::I64(v) => v.len() * 8,
            Col::I32(v) => v.len() * 4,
            Col::U32(v) => v.len() * 4,
        }
    }

    /// Element width in bytes (for paging the byte-level baselines / bitshuffle).
    #[cfg(any(feature = "bench-lz4", feature = "bench-zstd", feature = "bench-deflate"))]
    fn elem_bytes(&self) -> usize {
        match self {
            Col::F64(_) | Col::I64(_) => 8,
            Col::I32(_) | Col::U32(_) => 4,
        }
    }

    /// The raw value bytes, for the byte-level baselines.
    #[cfg(any(feature = "bench-lz4", feature = "bench-zstd", feature = "bench-deflate"))]
    fn as_bytes(&self) -> &[u8] {
        let (ptr, len) = match self {
            Col::F64(v) => (v.as_ptr() as *const u8, v.len() * 8),
            Col::I64(v) => (v.as_ptr() as *const u8, v.len() * 8),
            Col::I32(v) => (v.as_ptr() as *const u8, v.len() * 4),
            Col::U32(v) => (v.as_ptr() as *const u8, v.len() * 4),
        };
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }

    fn as_ref(&self) -> ColumnRef<'_> {
        match self {
            Col::F64(v) => ColumnRef::F64(v),
            Col::I64(v) => ColumnRef::I64(v),
            Col::I32(v) => ColumnRef::I32(v),
            Col::U32(v) => ColumnRef::U32(v),
        }
    }

    fn matches(&self, d: &Column) -> bool {
        match (self, d) {
            (Col::F64(a), Column::F64(b)) => a.iter().map(|x| x.to_bits()).eq(b.iter().map(|x| x.to_bits())),
            (Col::I64(a), Column::I64(b)) => a == b,
            (Col::I32(a), Column::I32(b)) => a == b,
            (Col::U32(a), Column::U32(b)) => a == b,
            _ => false,
        }
    }
}

/// Representative typed columns — the shapes where type-awareness matters.
fn datasets(n: usize) -> Vec<(&'static str, Col)> {
    let mut s = 0x1234_5678u64;
    let mut out: Vec<(&'static str, Col)> = Vec::new();

    out.push(("ids_i64", Col::I64((0..n).map(|i| 1000 + (i % 300) as i64).collect())));

    let mut t = 1_700_000_000_000i64;
    out.push((
        "timestamps_i64",
        Col::I64(
            (0..n)
                .map(|_| {
                    let r = lcg(&mut s);
                    t += 1000 + (r >> 40) as i64 % 4096;
                    t
                })
                .collect(),
        ),
    ));

    out.push((
        "lowcard_i64",
        Col::I64((0..n).map(|_| 5_000_000 + (lcg(&mut s) >> 60) as i64).collect()),
    ));

    out.push(("seq_i64", Col::I64((0..n as i64).collect())));

    out.push(("random_i64", Col::I64((0..n).map(|_| lcg(&mut s) as i64).collect())));

    out.push((
        "decimals_f64",
        Col::F64(
            (0..n)
                .map(|_| ((lcg(&mut s) >> 40) % 1_000_000) as f64 / 100.0)
                .collect(),
        ),
    ));

    out.push((
        "sensor_f64",
        Col::F64(
            (0..n)
                .map(|i| (i as f64 * 0.001).sin() * 100.0 + (i as f64) * 0.01)
                .collect(),
        ),
    ));

    out.push((
        "categorical_u32",
        Col::U32((0..n).map(|_| 100 + (lcg(&mut s) >> 61) as u32).collect()),
    ));

    out.push(("narrow_i32", Col::I32((0..n).map(|i| 50_000 + (i % 1000) as i32).collect())));

    out
}

#[derive(Default, Clone, Copy)]
struct Acc {
    raw: usize,
    comp: usize,
    enc_s: f64,
    dec_s: f64,
}

impl Acc {
    fn ratio(&self) -> f64 {
        self.raw as f64 / self.comp.max(1) as f64
    }
    fn enc(&self) -> f64 {
        mbps(self.raw, self.enc_s)
    }
    fn dec(&self) -> f64 {
        mbps(self.raw, self.dec_s)
    }
}

fn print_row(ds: &str, codec: &str, raw: usize, comp: usize, enc_s: f64, dec_s: f64) {
    println!(
        "{ds:<16} {codec:<11} {:>7.2}x {:>9.0} {:>9.0}",
        raw as f64 / comp.max(1) as f64,
        mbps(raw, enc_s),
        mbps(raw, dec_s),
    );
}

fn record(
    accs: &mut BTreeMap<&'static str, Acc>,
    codec: &'static str,
    raw: usize,
    comp: usize,
    enc_s: f64,
    dec_s: f64,
) {
    let a = accs.entry(codec).or_default();
    a.raw += raw;
    a.comp += comp;
    a.enc_s += enc_s;
    a.dec_s += dec_s;
}

const LEVELS: [(&str, Level); 5] = [
    ("q-fastest", Level::Fastest),
    ("q-fast", Level::Fast),
    ("q-balanced", Level::Balanced),
    ("q-high", Level::High),
    ("q-max", Level::Max),
];

/// Compress `bytes` in independent pages of `page_bytes` (0 ⇒ one whole-buffer
/// page = the tool's native mode), returning `(compressed_size, enc_s, dec_s)`
/// and verifying the round trip. Every byte-level baseline goes through this so
/// they all see the **same** block size as quoin — no tool gets a wider
/// cross-block window than another.
#[cfg(any(feature = "bench-lz4", feature = "bench-zstd", feature = "bench-deflate"))]
fn run_paged<C, D>(
    bytes: &[u8],
    page_bytes: usize,
    trials: usize,
    compress: C,
    decompress: D,
) -> (usize, f64, f64)
where
    C: Fn(&[u8]) -> Vec<u8>,
    D: Fn(&[u8], usize) -> Vec<u8>,
{
    let cl = if page_bytes == 0 { bytes.len().max(1) } else { page_bytes };
    let (enc_s, pages) = time_median(trials, || bytes.chunks(cl).map(&compress).collect::<Vec<_>>());
    let comp: usize = pages.iter().map(Vec::len).sum();
    let (dec_s, restored) = time_median(trials, || {
        let mut out = Vec::with_capacity(bytes.len());
        let mut off = 0usize;
        for p in &pages {
            let rl = (bytes.len() - off).min(cl);
            out.extend_from_slice(&decompress(p, rl));
            off += rl;
        }
        out
    });
    assert_eq!(restored, bytes);
    (comp, enc_s, dec_s)
}

/// Load the real-data ALP corpus: a directory of raw little-endian `f64` `.bin`
/// columns (the ALP/Vortex benchmark format). Returns each file as an `F64`
/// column named by its file stem. Used when `ALP_DIR` is set, so the level
/// ladder and the pco backend are exercised on real smooth/decimal doubles
/// rather than only synthetic shapes.
fn alp_datasets(dir: &str) -> Vec<(String, Col)> {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read ALP_DIR {dir}: {e}"))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "bin"))
        .collect();
    files.sort();
    files
        .into_iter()
        .map(|p| {
            let name = p.file_stem().unwrap().to_string_lossy().into_owned();
            let bytes = std::fs::read(&p).expect("read .bin");
            let vals: Vec<f64> = bytes
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            (name, Col::F64(vals))
        })
        .collect()
}

fn main() {
    let n: usize = std::env::var("PARETO_N").ok().and_then(|s| s.parse().ok()).unwrap_or(1 << 20);
    let trials: usize = std::env::var("PARETO_TRIALS").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    // Block size in *values*, shared by every tool so the comparison is fair.
    // 0 (default) = native: quoin adaptive, byte tools whole-buffer. Clamped to
    // quoin's max so quoin and the byte tools never diverge on a huge setting.
    let block_values: usize = std::env::var("BLOCK_VALUES")
        .ok()
        .and_then(|s| s.parse().ok())
        .map(|b: usize| if b == 0 { 0 } else { b.min(quoin::MAX_BLOCK_SIZE) })
        .unwrap_or(0);

    println!("{}", quoin::VERSION);
    let block_desc = if block_values == 0 {
        "native per-tool blocks".to_string()
    } else {
        format!("fixed {block_values} values/block (all tools)")
    };
    println!("typed columns, {n} values each, median of {trials}, {block_desc}\n");
    println!("{:<16} {:<11} {:>8} {:>9} {:>9}", "dataset", "codec", "ratio", "enc MB/s", "dec MB/s");

    let mut accs: BTreeMap<&'static str, Acc> = BTreeMap::new();

    // Real ALP f64 corpus when ALP_DIR is set, else the synthetic typed shapes.
    let data: Vec<(String, Col)> = match std::env::var("ALP_DIR") {
        Ok(dir) => alp_datasets(&dir),
        Err(_) => datasets(n).into_iter().map(|(s, c)| (s.to_string(), c)).collect(),
    };

    for (ds, col) in &data {
        let raw = col.raw_bytes();

        for (label, level) in LEVELS {
            let cfg = Config {
                level,
                block_size: (block_values > 0).then_some(block_values),
                ..Config::default()
            };
            let cref = col.as_ref();
            let (enc_s, packed) = time_median(trials, || compress_column(cref, None, cfg));
            let (dec_s, dec) = time_median(trials, || decompress_column(&packed).unwrap());
            assert!(col.matches(&dec.values), "{ds}/{label} round-trip");
            print_row(ds, label, raw, packed.len(), enc_s, dec_s);
            record(&mut accs, label, raw, packed.len(), enc_s, dec_s);
        }

        // Byte-level baselines: all paged at the same block size as quoin (in
        // bytes = block_values × element width); page_bytes 0 ⇒ whole buffer.
        #[cfg(any(feature = "bench-lz4", feature = "bench-zstd", feature = "bench-deflate"))]
        let bytes = col.as_bytes();
        #[cfg(any(feature = "bench-lz4", feature = "bench-zstd", feature = "bench-deflate"))]
        let page_bytes = if block_values == 0 { 0 } else { block_values * col.elem_bytes() };

        #[cfg(feature = "bench-lz4")]
        {
            let eb = col.elem_bytes();
            let (comp, enc_s, dec_s) = run_paged(
                bytes,
                page_bytes,
                trials,
                lz4_flex::compress_prepend_size,
                |p, _| lz4_flex::decompress_size_prepended(p).unwrap(),
            );
            print_row(ds, "lz4", raw, comp, enc_s, dec_s);
            record(&mut accs, "lz4", raw, comp, enc_s, dec_s);

            // Bitshuffle + lz4 (the float/numeric baseline). Ratio is faithful;
            // MB/s is a non-SIMD floor (see the `bitshuffle` module).
            let (comp, enc_s, dec_s) = run_paged(
                bytes,
                page_bytes,
                trials,
                |c| lz4_flex::compress_prepend_size(&bitshuffle::transpose(c, eb)),
                |p, rl| {
                    let shuf = lz4_flex::decompress_size_prepended(p).unwrap();
                    bitshuffle::untranspose(&shuf, eb, rl / eb)
                },
            );
            print_row(ds, "bitshuf-lz4", raw, comp, enc_s, dec_s);
            record(&mut accs, "bitshuf-lz4", raw, comp, enc_s, dec_s);
        }
        #[cfg(feature = "bench-zstd")]
        for lvl in [1, 3, 9, 19] {
            let name: &'static str = match lvl {
                1 => "zstd-1",
                3 => "zstd-3",
                9 => "zstd-9",
                _ => "zstd-19",
            };
            let (comp, enc_s, dec_s) = run_paged(
                bytes,
                page_bytes,
                trials,
                |c| zstd::bulk::compress(c, lvl).unwrap(),
                |p, rl| zstd::bulk::decompress(p, rl).unwrap(),
            );
            print_row(ds, name, raw, comp, enc_s, dec_s);
            record(&mut accs, name, raw, comp, enc_s, dec_s);
        }
        #[cfg(feature = "bench-deflate")]
        {
            use flate2::Compression;
            use flate2::read::{ZlibDecoder, ZlibEncoder};
            use std::io::Read;
            let (comp, enc_s, dec_s) = run_paged(
                bytes,
                page_bytes,
                trials,
                |c| {
                    let mut out = Vec::new();
                    ZlibEncoder::new(c, Compression::new(6)).read_to_end(&mut out).unwrap();
                    out
                },
                |p, rl| {
                    let mut out = Vec::with_capacity(rl);
                    ZlibDecoder::new(p).read_to_end(&mut out).unwrap();
                    out
                },
            );
            print_row(ds, "deflate-6", raw, comp, enc_s, dec_s);
            record(&mut accs, "deflate-6", raw, comp, enc_s, dec_s);
        }
        println!();
    }

    // ---- Aggregate + Pareto analysis ----
    let points: Vec<(&'static str, Acc)> = accs.into_iter().collect();
    println!("AGGREGATE (throughput-weighted), * = on the Pareto frontier:");
    println!("{:<11} {:>8} {:>9} {:>9}  frontier", "codec", "ratio", "enc MB/s", "dec MB/s");
    let dominated = |b: &Acc, by: &[(&str, Acc)], self_name: &str| {
        by.iter().any(|(nm, a)| {
            *nm != self_name
                && a.ratio() >= b.ratio()
                && a.enc() >= b.enc()
                && a.dec() >= b.dec()
                && (a.ratio() > b.ratio() || a.enc() > b.enc() || a.dec() > b.dec())
        })
    };
    for (name, a) in &points {
        let front = if dominated(a, &points, name) { "" } else { "  *" };
        println!(
            "{name:<11} {:>7.2}x {:>9.0} {:>9.0}{front}",
            a.ratio(),
            a.enc(),
            a.dec(),
        );
    }

    // Direct verdict: which quoin levels Pareto-dominate lz4?
    if let Some((_, lz4)) = points.iter().find(|(n, _)| *n == "lz4") {
        println!("\nvs lz4 (ratio {:.2}x, enc {:.0}, dec {:.0} MB/s):", lz4.ratio(), lz4.enc(), lz4.dec());
        for (name, a) in points.iter().filter(|(n, _)| n.starts_with("q-")) {
            let dom = a.ratio() >= lz4.ratio() && a.enc() >= lz4.enc() && a.dec() >= lz4.dec()
                && (a.ratio() > lz4.ratio() || a.enc() > lz4.enc() || a.dec() > lz4.dec());
            let verdict = if dom {
                "DOMINATES lz4 (≥ on all three axes)"
            } else if a.ratio() >= lz4.ratio() && a.dec() >= lz4.dec() {
                "≥ lz4 on ratio + decode (trails on encode)"
            } else {
                "does not dominate"
            };
            println!("  {name:<11} {verdict}");
        }
    }
}
