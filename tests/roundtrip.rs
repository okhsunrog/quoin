//! End-to-end round-trip and ratio checks across synthetic datasets, mirroring
//! the spirit of the original `fc` test harness.

use quoin::{Config, compress, decompress};

fn assert_roundtrip(name: &str, data: &[f64]) -> f64 {
    let packed = compress(data, Config::default());
    let restored = decompress(&packed).expect("decode");
    // Compare bit patterns so NaN / -0.0 are checked exactly.
    let a: Vec<u64> = data.iter().map(|f| f.to_bits()).collect();
    let b: Vec<u64> = restored.iter().map(|f| f.to_bits()).collect();
    assert_eq!(a, b, "round-trip mismatch on dataset `{name}`");

    let original = data.len() * 8;
    let ratio = original as f64 / packed.len().max(1) as f64;
    println!(
        "{name:>14}: {original:>9} -> {:>9} bytes  ratio {ratio:6.2}x",
        packed.len()
    );
    ratio
}

fn series(n: usize, f: impl FnMut(usize) -> f64) -> Vec<f64> {
    (0..n).map(f).collect()
}

// Cheap deterministic PRNG so tests need no dependencies.
fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed
}

#[test]
fn empty_and_tiny() {
    assert_roundtrip("empty", &[]);
    assert_roundtrip("one", &[2.5]);
    assert_roundtrip("two", &[1.0, 2.0]);
}

#[test]
fn structured_datasets_roundtrip_and_compress() {
    const N: usize = 1 << 16;

    let constant = series(N, |_| 42.0);
    let linear = series(N, |i| i as f64);
    let scaled = series(N, |i| i as f64 * 0.5);
    let int_x1000 = series(N, |i| (i % 1000) as f64 * 1000.0);
    let sine = series(N, |i| (i as f64 * 0.01).sin());
    let piecewise = series(N, |i| ((i / 256) as f64) * 7.0);

    // These have real structure; expect to beat the 9-byte frame overhead amply.
    assert!(assert_roundtrip("constant", &constant) > 100.0);
    assert!(assert_roundtrip("linear", &linear) > 2.0);
    assert!(assert_roundtrip("scaled", &scaled) > 2.0);
    assert!(assert_roundtrip("int_x1000", &int_x1000) > 1.5);
    assert_roundtrip("sine", &sine);
    assert!(assert_roundtrip("piecewise", &piecewise) > 4.0);
}

#[test]
fn random_roundtrips_without_expanding_much() {
    const N: usize = 1 << 16;
    let mut seed = 0x1234_5678_9abc_def0u64;
    let random = series(N, |_| f64::from_bits(lcg(&mut seed)));
    // Incompressible: ratio ~1.0, must still round-trip and not blow up.
    let ratio = assert_roundtrip("random_bits", &random);
    assert!(ratio > 0.95, "random data expanded too much: {ratio}");
}

#[test]
fn multi_block_roundtrip() {
    // Several blocks worth, to exercise frame boundaries.
    const N: usize = 200_000;
    let data = series(N, |i| (i as f64).sqrt());
    assert_roundtrip("multiblock", &data);
}

#[test]
fn special_values() {
    let data = vec![
        0.0,
        -0.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MIN,
        f64::MAX,
        f64::MIN_POSITIVE,
    ];
    assert_roundtrip("special", &data);
}
