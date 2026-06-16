#![no_main]
//! For any input, compress -> decompress must reproduce the exact bit patterns.
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Interpret the fuzz bytes as a sequence of f64 (8 bytes each).
    let vals: Vec<f64> =
        data.chunks_exact(8).map(|c| f64::from_bits(u64::from_le_bytes(c.try_into().unwrap()))).collect();

    let packed = quoin::compress(&vals, quoin::Config::default());
    let back = quoin::decompress(&packed).expect("our own stream must decode");

    assert_eq!(vals.len(), back.len(), "length changed across round trip");
    for (i, (a, b)) in vals.iter().zip(&back).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "bit mismatch at index {i}");
    }
});
