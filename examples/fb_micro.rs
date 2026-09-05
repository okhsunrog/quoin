//! Micro-benchmark of the FoR / delta bit-packers via `bench_internals`
//! (encode + decode, median of 7 on 1 Mi values), for codec-level A/Bs.
use quoin::bench_internals as bi;
use std::hint::black_box;
use std::time::Instant;

fn med(mut f: impl FnMut()) -> f64 {
    let mut t: Vec<f64> = (0..7)
        .map(|_| {
            let s = Instant::now();
            f();
            s.elapsed().as_secs_f64() * 1000.0
        })
        .collect();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t[3]
}

fn main() {
    let n = 1 << 20;
    let mut s = 1u64;
    let mut lcg = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        s
    };
    let cols: Vec<(&str, Vec<u64>)> = vec![
        ("bounded10", (0..n).map(|_| lcg() >> 54).collect()),
        ("bounded10+outliers", (0..n).map(|i| if i % 500 == 7 { 1 << 40 } else { lcg() >> 54 }).collect()),
        ("wide40", (0..n).map(|_| lcg() >> 24).collect()),
        ("timestamps", { let mut t = 0u64; (0..n).map(|_| { t += 1000 + (lcg() >> 52); t }).collect() }),
    ];
    println!("column,codec,bytes,enc_ms,dec_ms");
    for (name, v) in &cols {
        let e = bi::for_bitpack_encode(v);
        let enc = med(|| { black_box(bi::for_bitpack_encode(black_box(v))); });
        let dec = med(|| { black_box(bi::for_bitpack_decode(black_box(&e), v.len()).unwrap()); });
        assert_eq!(bi::for_bitpack_decode(&e, v.len()).unwrap(), *v);
        println!("{name},for_bitpack,{},{enc:.3},{dec:.3}", e.len());
        let e = bi::delta_bitpack_encode(v);
        let enc = med(|| { black_box(bi::delta_bitpack_encode(black_box(v))); });
        let dec = med(|| { black_box(bi::delta_bitpack_decode(black_box(&e), v.len()).unwrap()); });
        println!("{name},delta_bitpack,{},{enc:.3},{dec:.3}", e.len());
    }
}
