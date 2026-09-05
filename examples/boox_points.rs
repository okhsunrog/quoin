//! Lossless columnar compression of ONYX BOOX Notes `PointDocument` V1 files
//! (the `#points` geometry files), measured column by column with quoin's
//! native typed lanes, with the vendored pco as the reference numeric codec.
//!
//! The files are parsed with an independent reader (big-endian header, stroke
//! records, trailing index), re-serialized and compared byte-for-byte, and then
//! split into their six physical columns — `x`/`y` as `f32`, `pressure`
//! (`i16`), `tiltX`/`tiltY` (`i8`) and the per-stroke relative `time` (`i32`) —
//! which are compressed as typed quoin columns (`F32` / `I32`) and decoded
//! back **bit-exactly**. Everything outside the point columns (header, stroke
//! attributes, the index, the xref) is kept verbatim as a sidecar so the whole
//! original file can be rebuilt.
//!
//! Usage (paths only; no sample data lives in the repository):
//!
//! ```sh
//! cargo run --release --example boox_points -- points-1.bin points-2.bin > boox.csv
//! ```
//!
//! `BOOX_REPEATS` (default 5) sets the timing repeats; every reported time is
//! the median of that many ~3 ms batches, like the ink benchmark harness.

use std::hint::black_box;
use std::time::Instant;

use quoin::{Column, ColumnRef, Config, Level, Selection, compress_column, decompress_column};

const HEADER: usize = 76;
const POINT: usize = 16;
const INDEX_ENTRY: usize = 44;

struct Stroke {
    id: [u8; 36],
    attr_a: i16,
    attr_b: i16,
    n: usize,
}

/// A parsed PointDocument: the columns across all strokes (in file order) plus
/// everything needed to rebuild the file exactly.
struct Doc {
    header: [u8; HEADER],
    strokes: Vec<Stroke>,
    x: Vec<f32>,
    y: Vec<f32>,
    tilt_y: Vec<i8>,
    tilt_x: Vec<i8>,
    pressure: Vec<i16>,
    time: Vec<i32>,
}

fn be_i32(b: &[u8]) -> i32 {
    i32::from_be_bytes(b.try_into().unwrap())
}

fn parse(bytes: &[u8]) -> Result<Doc, String> {
    if bytes.len() < HEADER + 4 {
        return Err("too short".into());
    }
    let header: [u8; HEADER] = bytes[..HEADER].try_into().unwrap();
    if u16::from_be_bytes([header[2], header[3]]) != 1 {
        return Err("unsupported PointDocument version".into());
    }
    let xref = be_i32(&bytes[bytes.len() - 4..]) as usize;
    if xref < HEADER || xref > bytes.len() - 4 || !(bytes.len() - 4 - xref).is_multiple_of(INDEX_ENTRY) {
        return Err("bad xref/index".into());
    }
    let n_strokes = (bytes.len() - 4 - xref) / INDEX_ENTRY;
    let mut doc = Doc {
        header,
        strokes: Vec::with_capacity(n_strokes),
        x: Vec::new(),
        y: Vec::new(),
        tilt_y: Vec::new(),
        tilt_x: Vec::new(),
        pressure: Vec::new(),
        time: Vec::new(),
    };
    let mut expect = HEADER;
    for k in 0..n_strokes {
        let e = &bytes[xref + k * INDEX_ENTRY..xref + (k + 1) * INDEX_ENTRY];
        let offset = be_i32(&e[36..40]) as usize;
        let length = be_i32(&e[40..44]) as usize;
        if offset != expect || length < 4 || !(length - 4).is_multiple_of(POINT) || offset + length > xref {
            return Err(format!("stroke {k}: non-contiguous or malformed record"));
        }
        let rec = &bytes[offset..offset + length];
        let n = (length - 4) / POINT;
        doc.strokes.push(Stroke {
            id: e[..36].try_into().unwrap(),
            attr_a: i16::from_be_bytes([rec[0], rec[1]]),
            attr_b: i16::from_be_bytes([rec[2], rec[3]]),
            n,
        });
        for p in rec[4..].chunks_exact(POINT) {
            doc.x.push(f32::from_bits(u32::from_be_bytes(p[0..4].try_into().unwrap())));
            doc.y.push(f32::from_bits(u32::from_be_bytes(p[4..8].try_into().unwrap())));
            doc.tilt_y.push(p[8] as i8);
            doc.tilt_x.push(p[9] as i8);
            doc.pressure.push(i16::from_be_bytes([p[10], p[11]]));
            doc.time.push(be_i32(&p[12..16]));
        }
        expect = offset + length;
    }
    if expect != xref {
        return Err("gap between last stroke and index".into());
    }
    Ok(doc)
}

fn serialize(doc: &Doc) -> Vec<u8> {
    let n: usize = doc.strokes.iter().map(|s| s.n).sum();
    let mut out = Vec::with_capacity(80 + 48 * doc.strokes.len() + 16 * n);
    out.extend_from_slice(&doc.header);
    let mut index = Vec::with_capacity(doc.strokes.len() * INDEX_ENTRY);
    let mut i = 0usize;
    for s in &doc.strokes {
        let offset = out.len();
        out.extend_from_slice(&s.attr_a.to_be_bytes());
        out.extend_from_slice(&s.attr_b.to_be_bytes());
        for k in i..i + s.n {
            out.extend_from_slice(&doc.x[k].to_bits().to_be_bytes());
            out.extend_from_slice(&doc.y[k].to_bits().to_be_bytes());
            out.push(doc.tilt_y[k] as u8);
            out.push(doc.tilt_x[k] as u8);
            out.extend_from_slice(&doc.pressure[k].to_be_bytes());
            out.extend_from_slice(&doc.time[k].to_be_bytes());
        }
        i += s.n;
        index.extend_from_slice(&s.id);
        index.extend_from_slice(&(offset as i32).to_be_bytes());
        index.extend_from_slice(&((out.len() - offset) as i32).to_be_bytes());
    }
    let xref = out.len() as i32;
    out.extend_from_slice(&index);
    out.extend_from_slice(&xref.to_be_bytes());
    out
}

/// Bytes that are not point columns: header + per-stroke attrs + index + xref.
fn sidecar_bytes(doc: &Doc) -> usize {
    HEADER + doc.strokes.len() * (4 + INDEX_ENTRY) + 4
}

/// Per-point delta of the per-stroke relative time (the transform the earlier
/// SoA experiment used): `dt[i] = t[i] - t[i-1]` within a stroke, `t[0]` at a
/// stroke start. Exactly invertible given the stroke lengths.
fn time_deltas(doc: &Doc) -> Vec<i32> {
    let mut out = Vec::with_capacity(doc.time.len());
    let mut i = 0usize;
    for s in &doc.strokes {
        let mut prev = 0i32;
        for &t in &doc.time[i..i + s.n] {
            out.push(t.wrapping_sub(prev));
            prev = t;
        }
        i += s.n;
    }
    out
}

/// Median of `reps` batch timings (ms per call), ~3 ms per batch.
fn timed<T>(reps: usize, mut f: impl FnMut() -> T) -> f64 {
    let t0 = Instant::now();
    black_box(f());
    let one = t0.elapsed().as_secs_f64();
    let batch = ((0.003 / one.max(1e-9)).ceil() as usize).clamp(1, 64);
    let mut ms: Vec<f64> = (0..reps)
        .map(|_| {
            let t0 = Instant::now();
            for _ in 0..batch {
                black_box(f());
            }
            t0.elapsed().as_secs_f64() * 1000.0 / batch as f64
        })
        .collect();
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ms[ms.len() / 2]
}

#[derive(Clone, Copy)]
enum Col<'a> {
    F32(&'a [f32]),
    I32(&'a [i32]),
}

impl Col<'_> {
    fn n(&self) -> usize {
        match self {
            Col::F32(v) => v.len(),
            Col::I32(v) => v.len(),
        }
    }
    fn raw_bytes(&self) -> usize {
        self.n() * 4
    }
}

fn quoin_encode(col: Col<'_>, cfg: Config) -> Vec<u8> {
    match col {
        Col::F32(v) => compress_column(ColumnRef::F32(v), None, cfg),
        Col::I32(v) => compress_column(ColumnRef::I32(v), None, cfg),
    }
}

fn quoin_check(col: Col<'_>, bytes: &[u8]) {
    let dec = decompress_column(bytes).expect("decode");
    match (col, dec.values) {
        (Col::F32(v), Column::F32(got)) => {
            assert_eq!(v.len(), got.len());
            for (a, b) in v.iter().zip(&got) {
                assert_eq!(a.to_bits(), b.to_bits(), "f32 bit-exact");
            }
        }
        (Col::I32(v), Column::I32(got)) => assert_eq!(v, &got[..]),
        _ => panic!("dtype mismatch"),
    }
}

fn pco_encode(col: Col<'_>, level: usize) -> Vec<u8> {
    let cfg = quoin_pco::ChunkConfig::default().with_compression_level(level);
    match col {
        Col::F32(v) => quoin_pco::standalone::simple_compress(v, &cfg).unwrap(),
        Col::I32(v) => quoin_pco::standalone::simple_compress(v, &cfg).unwrap(),
    }
}

fn pco_check(col: Col<'_>, bytes: &[u8]) {
    match col {
        Col::F32(v) => {
            let got: Vec<f32> = quoin_pco::standalone::simple_decompress(bytes).unwrap();
            assert_eq!(v.len(), got.len());
            for (a, b) in v.iter().zip(&got) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }
        Col::I32(v) => {
            let got: Vec<i32> = quoin_pco::standalone::simple_decompress(bytes).unwrap();
            assert_eq!(v, &got[..]);
        }
    }
}

fn winning_modes() -> String {
    let counts = quoin::mode_win_counts();
    let mut parts: Vec<String> = counts
        .iter()
        .enumerate()
        .filter(|(_, c)| **c > 0)
        .map(|(id, &c)| format!("{}:{c}", quoin::mode_name(id as u8)))
        .collect();
    parts.sort();
    parts.join("|")
}

fn main() {
    let reps: usize = std::env::var("BOOX_REPEATS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("usage: boox_points <PointDocument>...");
        std::process::exit(2);
    }
    let methods: Vec<(&str, Option<Config>, usize)> = vec![
        ("quoin-fastest", Some(Config { level: Level::Fastest, ..Config::default() }), 0),
        ("quoin-fast", Some(Config { level: Level::Fast, ..Config::default() }), 0),
        ("quoin-balanced", Some(Config { level: Level::Balanced, ..Config::default() }), 0),
        (
            "quoin-balanced-sample",
            Some(Config { level: Level::Balanced, selection: Selection::Sample, ..Config::default() }),
            0,
        ),
        ("quoin-high", Some(Config { level: Level::High, ..Config::default() }), 0),
        ("quoin-max", Some(Config { level: Level::Max, ..Config::default() }), 0),
        ("pco-8", None, 8),
        ("pco-12", None, 12),
    ];
    println!("file,method,column,n,raw_bytes,bytes,encode_ms,decode_ms,modes");
    for path in &paths {
        let bytes = std::fs::read(path).expect("read");
        let doc = parse(&bytes).expect("parse PointDocument");
        assert_eq!(serialize(&doc), bytes, "{path}: exact re-serialization");
        let n = doc.x.len();
        let file = std::path::Path::new(path)
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        eprintln!(
            "{file}: {} strokes, {n} points, {} bytes ({} sidecar)",
            doc.strokes.len(),
            bytes.len(),
            sidecar_bytes(&doc)
        );
        println!("{file},sidecar,header+attrs+index,{},{},{},0,0,", doc.strokes.len(), sidecar_bytes(&doc), sidecar_bytes(&doc));
        let pressure: Vec<i32> = doc.pressure.iter().map(|&p| i32::from(p)).collect();
        let tilt_x: Vec<i32> = doc.tilt_x.iter().map(|&t| i32::from(t)).collect();
        let tilt_y: Vec<i32> = doc.tilt_y.iter().map(|&t| i32::from(t)).collect();
        let dt = time_deltas(&doc);
        let columns: Vec<(&str, Col<'_>)> = vec![
            ("x", Col::F32(&doc.x)),
            ("y", Col::F32(&doc.y)),
            ("pressure", Col::I32(&pressure)),
            ("tiltX", Col::I32(&tilt_x)),
            ("tiltY", Col::I32(&tilt_y)),
            ("time", Col::I32(&doc.time)),
            ("time-delta", Col::I32(&dt)),
        ];
        for (mname, cfg, pco_level) in &methods {
            for (cname, col) in &columns {
                let (enc, modes): (Vec<u8>, String) = match cfg {
                    Some(cfg) => {
                        quoin::reset_mode_win_counts();
                        let e = quoin_encode(*col, *cfg);
                        let modes = winning_modes();
                        quoin_check(*col, &e);
                        (e, modes)
                    }
                    None => {
                        let e = pco_encode(*col, *pco_level);
                        pco_check(*col, &e);
                        (e, String::new())
                    }
                };
                let encode_ms = match cfg {
                    Some(cfg) => timed(reps, || quoin_encode(*col, *cfg)),
                    None => timed(reps, || pco_encode(*col, *pco_level)),
                };
                let decode_ms = match cfg {
                    Some(_) => timed(reps, || decompress_column(&enc).unwrap()),
                    None => match col {
                        Col::F32(_) => timed(reps, || {
                            quoin_pco::standalone::simple_decompress::<f32>(&enc).unwrap()
                        }),
                        Col::I32(_) => timed(reps, || {
                            quoin_pco::standalone::simple_decompress::<i32>(&enc).unwrap()
                        }),
                    },
                };
                println!(
                    "{file},{mname},{cname},{},{},{},{encode_ms:.6},{decode_ms:.6},{modes}",
                    col.n(),
                    col.raw_bytes(),
                    enc.len()
                );
            }
        }
    }
}
