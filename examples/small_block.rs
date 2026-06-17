//! Small-block break-even: at what block size does quoin's per-block framing
//! stop costing more than the ratio it buys, vs raw lz4 on the same block? This
//! matters for a column store whose native block is tiny (e.g. ~100 values):
//! below the break-even quoin loses to lz4, above it quoin wins.
//!
//! Usage: cargo run --release --example small_block --features bench-lz4

use quoin::{ColumnRef, Config, Level, compress_column};

/// Local typed column holder so we can sweep block sizes over one buffer.
enum Col {
    F64(Vec<f64>),
    I64(Vec<i64>),
    I32(Vec<i32>),
}

fn lcg(s: &mut u64) -> u64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *s
}

impl Col {
    fn as_ref(&self) -> ColumnRef<'_> {
        match self {
            Col::F64(v) => ColumnRef::F64(v),
            Col::I64(v) => ColumnRef::I64(v),
            Col::I32(v) => ColumnRef::I32(v),
        }
    }
    fn raw_bytes(&self) -> usize {
        match self {
            Col::F64(v) => v.len() * 8,
            Col::I64(v) => v.len() * 8,
            Col::I32(v) => v.len() * 4,
        }
    }
    fn elem(&self) -> usize {
        match self {
            Col::I32(_) => 4,
            _ => 8,
        }
    }
    fn as_bytes(&self) -> &[u8] {
        match self {
            Col::F64(v) => bytemuck_cast(v),
            Col::I64(v) => bytemuck_cast(v),
            Col::I32(v) => bytemuck_cast(v),
        }
    }
}

fn bytemuck_cast<T>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// lz4 each `page_bytes`-sized chunk independently; return summed compressed len.
#[cfg(feature = "bench-lz4")]
fn lz4_paged(bytes: &[u8], page_bytes: usize) -> usize {
    bytes
        .chunks(page_bytes.max(1))
        .map(|c| lz4_flex::compress_prepend_size(c).len())
        .sum()
}

fn quoin_paged(col: &Col, bs: usize, level: Level) -> usize {
    let cfg = Config { level, block_size: Some(bs), ..Config::default() };
    compress_column(col.as_ref(), None, cfg).len()
}

fn datasets(n: usize) -> Vec<(&'static str, Col)> {
    let mut s = 0x9e37_79b9u64;
    vec![
        (
            "sensor_f64",
            Col::F64((0..n).map(|i| (i as f64 * 0.001).sin() * 100.0 + i as f64 * 0.01).collect()),
        ),
        (
            "timestamps_i64",
            Col::I64({
                let mut t = 1_700_000_000_000i64;
                (0..n).map(|_| { t += 1000 + (lcg(&mut s) >> 40) as i64 % 4096; t }).collect()
            }),
        ),
        (
            "lowcard_i32",
            Col::I32((0..n).map(|_| 5000 + (lcg(&mut s) >> 60) as i32).collect()),
        ),
    ]
}

fn main() {
    let n: usize = std::env::var("SB_N").ok().and_then(|s| s.parse().ok()).unwrap_or(1 << 18);
    let blocks = [64usize, 100, 256, 512, 1024, 4096, 16384, 65536];

    println!("{}", quoin::VERSION);
    println!("small-block break-even vs lz4, {n} values/column\n");

    for (name, col) in datasets(n) {
        let raw = col.raw_bytes();
        println!("== {name} (raw {} KiB) ==", raw / 1024);
        println!("{:>8}  {:>10} {:>10} {:>10}  winner", "block", "q-balanced", "q-high", "lz4");
        for &bs in &blocks {
            let qb = raw as f64 / quoin_paged(&col, bs, Level::Balanced) as f64;
            let qh = raw as f64 / quoin_paged(&col, bs, Level::High) as f64;
            #[cfg(feature = "bench-lz4")]
            let l4 = raw as f64 / lz4_paged(col.as_bytes(), bs * col.elem()) as f64;
            #[cfg(not(feature = "bench-lz4"))]
            let l4 = f64::NAN;
            let best_q = qb.max(qh);
            let mark = if best_q >= l4 { "quoin" } else { "lz4" };
            println!("{bs:>8}  {qb:>9.2}x {qh:>9.2}x {l4:>9.2}x  {mark}");
        }
        println!();
    }
}
