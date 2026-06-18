//! Ratio regression guard. Compresses a few deterministic reference columns at
//! each level and asserts the ratio hasn't dropped below a committed floor. This
//! catches a silent selector/weights regression that the bit-exact roundtrip
//! tests can't see. Floors are set ~3% below the measured baseline, so normal
//! ratio-neutral tuning passes but a real degradation trips the test.
//!
//! To refresh after an intentional change: run `PRINT_RATIOS=1 cargo test
//! --test ratio_regression -- --nocapture` and copy the printed ratios (minus a
//! few %) into FLOORS.

use quoin::{Column, ColumnRef, Config, Level, compress, compress_column, decompress, decompress_column};

fn lcg(s: &mut u64) -> u64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *s
}

// Deterministic reference columns (no RNG crate, no external data).
fn sine_f64() -> Vec<f64> {
    (0..50_000).map(|i| (i as f64 * 1e-3).sin() * 100.0 + i as f64 * 0.01).collect()
}
fn timestamps_i64() -> Vec<i64> {
    let mut s = 0x1234_5678u64;
    let mut t = 1_700_000_000_000i64;
    (0..50_000).map(|_| { t += 1000 + (lcg(&mut s) >> 52) as i64; t }).collect()
}
fn lowcard_i32() -> Vec<i32> {
    let mut s = 0x9e37u64;
    (0..50_000).map(|_| 5000 + (lcg(&mut s) >> 60) as i32).collect()
}
fn cents_dec128() -> Vec<i128> {
    let mut s = 0xfeedu64;
    (0..50_000).map(|_| 100_00 + (lcg(&mut s) >> 56) as i128).collect()
}

const LEVELS: [(Level, &str); 3] =
    [(Level::Fastest, "fastest"), (Level::Balanced, "balanced"), (Level::Max, "max")];

// (column, level) -> floor ratio. Baseline measured 2026-06; floors are ~3% under.
fn floor(col: &str, level: &str) -> f64 {
    match (col, level) {
        ("sine_f64", "fastest") => 1.40,
        ("sine_f64", "balanced") => 1.40,
        ("sine_f64", "max") => 7.90,
        ("timestamps_i64", "fastest") => 4.70,
        ("timestamps_i64", "balanced") => 4.70,
        ("timestamps_i64", "max") => 5.10,
        ("lowcard_i32", "fastest") => 7.50,
        ("lowcard_i32", "balanced") => 7.50,
        ("lowcard_i32", "max") => 7.70,
        ("cents_dec128", "fastest") => 15.30,
        ("cents_dec128", "balanced") => 15.30,
        ("cents_dec128", "max") => 15.40,
        _ => 0.0,
    }
}

fn check(col: &str, raw_bytes: usize, packed: usize, level: &str) {
    let ratio = raw_bytes as f64 / packed as f64;
    if std::env::var("PRINT_RATIOS").is_ok() {
        println!("{col:18} {level:9} {ratio:.3}");
        return; // print-only mode: don't assert, so all rows print
    }
    let f = floor(col, level);
    assert!(ratio >= f, "{col} @ {level}: ratio {ratio:.3} below floor {f:.3} — selector regression?");
}

#[test]
fn ratio_floors_hold() {
    let sine = sine_f64();
    let ts = timestamps_i64();
    let lc = lowcard_i32();
    let cents = cents_dec128();

    for (level, name) in LEVELS {
        let cfg = Config { level, ..Config::default() };

        // f64 via the simple API.
        let p = compress(&sine, cfg);
        assert_eq!(decompress(&p).unwrap(), sine, "sine roundtrip");
        check("sine_f64", sine.len() * 8, p.len(), name);

        // typed columns via compress_column.
        let p = compress_column(ColumnRef::I64(&ts), None, cfg);
        assert_eq!(decompress_column(&p).unwrap().values, Column::I64(ts.clone()));
        check("timestamps_i64", ts.len() * 8, p.len(), name);

        let p = compress_column(ColumnRef::I32(&lc), None, cfg);
        assert_eq!(decompress_column(&p).unwrap().values, Column::I32(lc.clone()));
        check("lowcard_i32", lc.len() * 4, p.len(), name);

        let p = compress_column(
            ColumnRef::Decimal128 { values: &cents, scale: 2, precision: 18 },
            None,
            cfg,
        );
        check("cents_dec128", cents.len() * 16, p.len(), name);
    }
}
