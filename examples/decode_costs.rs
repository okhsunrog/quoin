//! Calibration of the selection's decode-cost classes: per-mode decode ns/value
//! on real corpus blocks (128 Ki values, the Max-level block), every mode that
//! applies, median of 5. `ALP_DIR` as in the other examples.
use std::collections::BTreeMap;
fn main() {
    let dir = std::env::var("ALP_DIR").unwrap_or_else(|_| "datasets/alp".into());
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "bin"))
        .collect();
    files.sort();
    let level = match std::env::var("DC_LEVEL").as_deref() {
        Ok("balanced") => quoin::Level::Balanced,
        _ => quoin::Level::Max,
    };
    let mut agg: BTreeMap<&'static str, Vec<f64>> = BTreeMap::new();
    println!("column,mode,bytes,dec_ns_per_value");
    for f in files {
        let b = std::fs::read(&f).unwrap();
        let n = (b.len() / 8).min(128 * 1024);
        let vals: Vec<f64> = b[..n * 8]
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let name = f.file_stem().unwrap().to_string_lossy().into_owned();
        for (m, bytes, ns) in quoin::bench_internals::mode_decode_costs(&vals, level, 5) {
            println!("{name},{m},{bytes},{ns:.2}");
            agg.entry(m).or_default().push(ns);
        }
    }
    eprintln!("\nmode                 median ns/value  (n columns)");
    for (m, mut v) in agg {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!("{m:20} {:8.2}  ({})", v[v.len() / 2], v.len());
    }
}
