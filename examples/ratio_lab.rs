//! Ratio lab: a fast, deterministic size/time sweep for codec experiments.
//!
//! Compresses every `.bin` (raw little-endian `f64`) column under `ALP_DIR`
//! (default `datasets/alp`, capped at `LAB_N` values, default 1 Mi), plus a
//! few synthetic integer columns and, when `LAB_BOOX` names PointDocument
//! files, their `x`/`y` (f32) and `pressure`/`time` (i32) columns, at the
//! levels in `LAB_LEVELS` (default `fast,balanced,max`), with `LAB_SELECTION=sample`
//! for the sampled selector and `LAB_BIAS=<percent>` to override the decode bias.
//! Emits CSV:
//! `column,dtype,level,n,raw_bytes,bytes,enc_ms,dec_ms,modes`. Every decode is
//! checked bit-exact. Meant for A/B: run once on the baseline commit, once on
//! the change, diff the CSVs.

use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use quoin::{Column, ColumnRef, Config, Level, compress_column, decompress_column};

fn load_f64(p: &Path, cap: usize) -> Vec<f64> {
    let b = std::fs::read(p).expect("read");
    b.chunks_exact(8)
        .take(cap)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s
}

enum Col {
    F64(Vec<f64>),
    F32(Vec<f32>),
    I32(Vec<i32>),
    I64(Vec<i64>),
}

impl Col {
    fn n(&self) -> usize {
        match self {
            Col::F64(v) => v.len(),
            Col::F32(v) => v.len(),
            Col::I32(v) => v.len(),
            Col::I64(v) => v.len(),
        }
    }
    fn raw(&self) -> usize {
        match self {
            Col::F64(v) => v.len() * 8,
            Col::I64(v) => v.len() * 8,
            Col::F32(v) => v.len() * 4,
            Col::I32(v) => v.len() * 4,
        }
    }
    fn dtype(&self) -> &'static str {
        match self {
            Col::F64(_) => "f64",
            Col::F32(_) => "f32",
            Col::I32(_) => "i32",
            Col::I64(_) => "i64",
        }
    }
    fn encode(&self, cfg: Config) -> Vec<u8> {
        match self {
            Col::F64(v) => compress_column(ColumnRef::F64(v), None, cfg),
            Col::F32(v) => compress_column(ColumnRef::F32(v), None, cfg),
            Col::I32(v) => compress_column(ColumnRef::I32(v), None, cfg),
            Col::I64(v) => compress_column(ColumnRef::I64(v), None, cfg),
        }
    }
    fn check(&self, bytes: &[u8]) {
        let dec = decompress_column(bytes).expect("decode").values;
        let ok = match (self, &dec) {
            (Col::F64(v), Column::F64(g)) => v
                .iter()
                .map(|x| x.to_bits())
                .eq(g.iter().map(|x| x.to_bits())),
            (Col::F32(v), Column::F32(g)) => v
                .iter()
                .map(|x| x.to_bits())
                .eq(g.iter().map(|x| x.to_bits())),
            (Col::I32(v), Column::I32(g)) => v == g,
            (Col::I64(v), Column::I64(g)) => v == g,
            _ => false,
        };
        assert!(ok, "bit-exact round trip");
    }
}

fn boox_columns(path: &str) -> Vec<(String, Col)> {
    let b = std::fs::read(path).expect("read boox");
    let xref = i32::from_be_bytes(b[b.len() - 4..].try_into().unwrap()) as usize;
    let strokes = (b.len() - 4 - xref) / 44;
    let (mut x, mut y, mut p, mut t) = (vec![], vec![], vec![], vec![]);
    for k in 0..strokes {
        let e = &b[xref + k * 44..xref + (k + 1) * 44];
        let off = i32::from_be_bytes(e[36..40].try_into().unwrap()) as usize;
        let len = i32::from_be_bytes(e[40..44].try_into().unwrap()) as usize;
        for r in b[off + 4..off + len].chunks_exact(16) {
            x.push(f32::from_bits(u32::from_be_bytes(
                r[0..4].try_into().unwrap(),
            )));
            y.push(f32::from_bits(u32::from_be_bytes(
                r[4..8].try_into().unwrap(),
            )));
            p.push(i32::from(i16::from_be_bytes([r[10], r[11]])));
            t.push(i32::from_be_bytes(r[12..16].try_into().unwrap()));
        }
    }
    let stem = Path::new(path)
        .file_stem()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    vec![
        (format!("boox-{stem}-x"), Col::F32(x)),
        (format!("boox-{stem}-y"), Col::F32(y)),
        (format!("boox-{stem}-pressure"), Col::I32(p)),
        (format!("boox-{stem}-time"), Col::I32(t)),
    ]
}

fn modes() -> String {
    let counts = quoin::mode_win_counts();
    let mut parts: Vec<String> = counts
        .iter()
        .enumerate()
        .filter(|(_, c)| **c > 0)
        .map(|(id, c)| format!("{}:{c}", quoin::mode_name(id as u8)))
        .collect();
    parts.sort();
    parts.join("|")
}

fn main() {
    let dir = std::env::var("ALP_DIR").unwrap_or_else(|_| "datasets/alp".into());
    let cap: usize = std::env::var("LAB_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1 << 20);
    let levels = std::env::var("LAB_LEVELS").unwrap_or_else(|_| "fast,balanced,max".into());
    let levels: Vec<(String, Level)> = levels
        .split(',')
        .map(|l| {
            let lv = match l {
                "fastest" => Level::Fastest,
                "fast" => Level::Fast,
                "balanced" => Level::Balanced,
                "high" => Level::High,
                _ => Level::Max,
            };
            (l.to_string(), lv)
        })
        .collect();

    let mut cols: Vec<(String, Col)> = Vec::new();
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .expect("ALP_DIR")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "bin"))
        .collect();
    files.sort();
    for f in files {
        let name = f.file_stem().unwrap().to_string_lossy().into_owned();
        cols.push((name, Col::F64(load_f64(&f, cap))));
    }
    // Synthetic integer columns (deterministic).
    let mut s = 0x1234_5678u64;
    let mut t = 1_700_000_000_000i64;
    let ts: Vec<i64> = (0..cap.min(500_000))
        .map(|_| {
            t += 1000 + (lcg(&mut s) >> 52) as i64;
            t
        })
        .collect();
    cols.push(("syn-timestamps".into(), Col::I64(ts)));
    let ids: Vec<i32> = (0..cap.min(500_000))
        .map(|_| 5000 + (lcg(&mut s) >> 60) as i32)
        .collect();
    cols.push(("syn-lowcard".into(), Col::I32(ids)));
    // Bounded ids with rare outliers: the PFOR case.
    let pf: Vec<i32> = (0..cap.min(500_000))
        .map(|i| {
            if i % 997 == 0 {
                1 << 24
            } else {
                (lcg(&mut s) >> 54) as i32
            }
        })
        .collect();
    cols.push(("syn-outliers".into(), Col::I32(pf)));
    if let Ok(list) = std::env::var("LAB_BOOX") {
        for p in list.split(',').filter(|s| !s.is_empty()) {
            cols.extend(boox_columns(p));
        }
    }

    let selection = match std::env::var("LAB_SELECTION").as_deref() {
        Ok("sample") => quoin::Selection::Sample,
        _ => quoin::Selection::Full,
    };
    let decode_bias: Option<u32> = std::env::var("LAB_BIAS").ok().and_then(|s| s.parse().ok());
    println!("column,dtype,level,n,raw_bytes,bytes,enc_ms,dec_ms,modes");
    for (name, col) in &cols {
        for (lname, level) in &levels {
            let cfg = Config {
                level: *level,
                selection,
                decode_bias,
                ..Config::default()
            };
            quoin::reset_mode_win_counts();
            let t0 = Instant::now();
            let enc = black_box(col.encode(cfg));
            let enc_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let m = modes();
            col.check(&enc);
            let t0 = Instant::now();
            black_box(decompress_column(&enc).unwrap());
            let dec_ms = t0.elapsed().as_secs_f64() * 1000.0;
            println!(
                "{name},{},{lname},{},{},{},{enc_ms:.3},{dec_ms:.3},{m}",
                col.dtype(),
                col.n(),
                col.raw(),
                enc.len()
            );
        }
        eprintln!("{name} done");
    }
}
