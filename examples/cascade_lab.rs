//! Cascade lab: for the ALP-RD value-codec, measure whether entropy-coding its
//! internal streams (currently raw `for_bitpack`) is smaller — i.e. is a
//! `bitpack → entropy` cascade worth wiring in? Reports per real column the
//! codes/rights stream sizes under {raw bitpack, +rANS, +range coder}.
//!
//! Run: ALP_DIR=datasets/alp taskset -c 0 \
//!        cargo run --release --no-default-features --example cascade_lab

use quoin::bench_internals as bi;
use std::path::Path;

fn load(p: &Path) -> Vec<u64> {
    std::fs::read(p).unwrap().chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect()
}

// Smallest of {raw for_bitpack, entropy of the byte view}. For low-cardinality
// streams (codes) the bytes are tiny ints; for wide streams (rights) we entropy
// the already-bit-packed blob.
fn best(stream: &[u64], byte_view: Vec<u8>) -> (usize, usize, usize, &'static str) {
    let raw = bi::for_bitpack_encode(stream).len();
    let rans = bi::rans_compress(&byte_view).map(|v| v.len() + 1).unwrap_or(usize::MAX);
    let rc = bi::rc_compress(&byte_view).len() + 1;
    let mut win = ("bitpack", raw);
    if rans < win.1 { win = ("rans", rans); }
    if rc < win.1 { win = ("rc", rc); }
    (raw, rans.min(rc), win.1, win.0)
}

fn main() {
    let dir = std::env::var("ALP_DIR").unwrap_or_else(|_| "datasets/alp".into());
    let cols = [
        "air_sensor_f", "bird_migration_f", "basel_wind_f", "poi_lat",
        "city_temperature_f", "food_prices", "neon_dew_point_temp", "bitcoin_transactions_f",
    ];
    println!("{:<22} {:>8}   {:>26}   {:>26}", "column", "n",
             "CODES raw→ent (save%, win)", "RIGHTS raw→ent (save%, win)");
    let (mut tot_craw, mut tot_cbest, mut tot_rraw, mut tot_rbest) = (0usize, 0, 0, 0);
    for c in cols {
        let path = Path::new(&dir).join(format!("{c}.bin"));
        if !path.exists() { continue; }
        let mut data = load(&path);
        data.truncate(2_000_000);
        let Some((codes, rights)) = bi::alp_rd_streams(&data) else {
            println!("{c:<22} {:>8}   (ALP-RD n/a)", data.len());
            continue;
        };
        // codes ≤ MAX_DICT (8) → fit in a byte.
        let code_bytes: Vec<u8> = codes.iter().map(|&v| v as u8).collect();
        let (craw, cbest, _, cwin) = best(&codes, code_bytes);
        // rights are wide → entropy the bit-packed blob.
        let rbp = bi::for_bitpack_encode(&rights);
        let (rraw, rbest, _, rwin) = best(&rights, rbp);
        let csave = 100.0 * (craw - cbest) as f64 / craw as f64;
        let rsave = 100.0 * (rraw.saturating_sub(rbest)) as f64 / rraw as f64;
        println!("{c:<22} {:>8}   {:>9}→{:<7} {:>5.1}% {:<5}   {:>9}→{:<7} {:>5.1}% {:<5}",
                 data.len(), craw, cbest, csave, cwin, rraw, rbest, rsave, rwin);
        tot_craw += craw; tot_cbest += cbest; tot_rraw += rraw; tot_rbest += rbest;
    }
    println!("\nTOTAL codes:  {} → {} ({:.1}% saved)   rights: {} → {} ({:.1}% saved)",
             tot_craw, tot_cbest, 100.0 * (tot_craw - tot_cbest) as f64 / tot_craw.max(1) as f64,
             tot_rraw, tot_rbest, 100.0 * (tot_rraw.saturating_sub(tot_rbest)) as f64 / tot_rraw.max(1) as f64);
}
