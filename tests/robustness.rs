//! Robustness / fuzz-style tests: `decompress` parses untrusted bytes, so it
//! must never panic — only ever return `Ok` or `Err`. We hammer it with random
//! buffers and with bit-flipped / truncated valid streams (which reach deep
//! into the codecs because they keep a valid header).

use fp_compressor::{Config, compress, decompress};
use std::panic::{AssertUnwindSafe, catch_unwind};

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s
}

/// Valid compressed streams from varied data, to mutate into malformed ones.
fn seeds() -> Vec<Vec<u8>> {
    let mut s = 1u64;
    let gens: Vec<Vec<f64>> = vec![
        (0..4000).map(|i| i as f64 * 0.5).collect(), // ramp -> idelta/dfcm
        (0..4000).map(|i| (i as f64 * 0.01).sin()).collect(), // smooth -> delta2/rc/tans
        (0..4000).map(|i| (i & 15) as f64).collect(), // dict -> lz
        (0..4000).map(|_| f64::from_bits(lcg(&mut s))).collect(), // random -> raw
        vec![42.0; 4000],                            // const
    ];
    gens.iter()
        .map(|d| compress(d, Config::default()))
        .collect()
}

fn assert_no_panic(label: &str, iters: usize, mut make: impl FnMut() -> Vec<u8>) {
    for _ in 0..iters {
        let buf = make();
        let r = catch_unwind(AssertUnwindSafe(|| {
            let _ = decompress(&buf);
        }));
        if r.is_err() {
            // The default panic hook already printed the location above; stop
            // at the first one so the output isn't spammed.
            panic!("{label}: decompress panicked on input {buf:?}");
        }
    }
}

#[test]
fn random_bytes_never_panic() {
    let mut s = 0xABCDEF01u64;
    assert_no_panic("random", 30_000, || {
        let len = (lcg(&mut s) % 80) as usize;
        (0..len).map(|_| (lcg(&mut s) >> 24) as u8).collect()
    });
}

#[test]
fn mutated_streams_never_panic() {
    let seeds = seeds();
    let mut s = 0x1357_9bdfu64;
    assert_no_panic("mutated", 50_000, || {
        let mut buf = seeds[(lcg(&mut s) as usize) % seeds.len()].clone();
        if buf.is_empty() {
            return buf;
        }
        for _ in 0..(1 + lcg(&mut s) % 4) {
            if buf.is_empty() {
                break; // an earlier truncate emptied it
            }
            match lcg(&mut s) % 3 {
                0 => {
                    let i = (lcg(&mut s) as usize) % buf.len();
                    buf[i] = (lcg(&mut s) >> 20) as u8;
                }
                1 => {
                    let i = (lcg(&mut s) as usize) % buf.len();
                    buf[i] ^= 1 << (lcg(&mut s) % 8);
                }
                _ => {
                    let keep = (lcg(&mut s) as usize) % buf.len();
                    buf.truncate(keep);
                }
            }
        }
        buf
    });
}

#[test]
fn adversarial_roundtrip() {
    // Patterns that exercise edge cases in the predictors/coders.
    let mut s = 7u64;
    let cases: Vec<Vec<f64>> = vec![
        vec![],
        vec![0.0],
        vec![f64::NAN; 1000],
        (0..5000)
            .map(|i| {
                if i % 2 == 0 {
                    f64::INFINITY
                } else {
                    f64::NEG_INFINITY
                }
            })
            .collect(),
        (0..5000)
            .map(|_| f64::from_bits(lcg(&mut s) & 0xFFF0_0000_0000_0000))
            .collect(),
        (0..70_000).map(|i| (i as f64).sqrt()).collect(), // multi-block
    ];
    for data in cases {
        let packed = compress(&data, Config::default());
        let back = decompress(&packed).expect("valid stream must decode");
        let a: Vec<u64> = data.iter().map(|f| f.to_bits()).collect();
        let b: Vec<u64> = back.iter().map(|f| f.to_bits()).collect();
        assert_eq!(a, b);
    }
}
