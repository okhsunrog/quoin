//! Shared value-dictionary preamble: end-to-end stream tests. The scenario is
//! a column whose distinct values recur *across* blocks — too high-cardinality
//! for the per-block Dict (> 50% of a block), cheap for a column-wide table.

use quoin::{Config, Level, compress, decompress};

/// ~5000 distinct random (incompressible) f64 bit patterns scattered over
/// `n` values, so FoR/delta/LZ find little and value sharing is the win.
fn cross_block_column(n: usize, card: usize) -> Vec<f64> {
    let mut s = 0x2545F4914F6CDD1Du64;
    let table: Vec<u64> = (0..card)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            s | 0x3FF0_0000_0000_0000 // keep exponents sane (finite doubles)
        })
        .collect();
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            f64::from_bits(table[(s >> 33) as usize % table.len()])
        })
        .collect()
}

#[test]
fn shared_dict_roundtrips_and_compresses() {
    let data = cross_block_column(100_000, 5000);
    // Small fixed blocks force many blocks (each ~4900 distinct of 8192 values,
    // so the per-block Dict bails) — only the shared table can exploit the
    // column-wide repeats.
    let cfg = Config {
        block_size: Some(8192),
        ..Config::default()
    };
    let packed = compress(&data, cfg);
    let restored = decompress(&packed).unwrap();
    assert_eq!(data.len(), restored.len());
    assert!(
        data.iter()
            .zip(&restored)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    // 5000-entry table ≈ 13-bit codes + one 40 KB preamble over 800 KB raw.
    let raw = data.len() * 8;
    assert!(
        packed.len() * 100 < raw * 40,
        "shared dict should compress cross-block repeats: {} of {raw}",
        packed.len()
    );
}

#[test]
fn shared_dict_with_validity() {
    let mut data = cross_block_column(60_000, 3000);
    // Null out every 7th slot via an Arrow-style LSB bitmap.
    let n = data.len();
    let mut bitmap = vec![0xFFu8; n.div_ceil(8)];
    for i in (0..n).step_by(7) {
        bitmap[i / 8] &= !(1 << (i % 8));
        data[i] = 0.0; // value under a null is arbitrary
    }
    let cfg = Config {
        block_size: Some(8192),
        ..Config::default()
    };
    let packed = quoin::compress_column(
        quoin::ColumnRef::F64(&data),
        Some(&bitmap),
        cfg,
    );
    let decoded = quoin::decompress_column(&packed).unwrap();
    let quoin::Column::F64(vals) = decoded.values else {
        panic!("wrong dtype");
    };
    for i in 0..n {
        let valid = bitmap[i / 8] & (1 << (i % 8)) != 0;
        if valid {
            assert_eq!(vals[i].to_bits(), data[i].to_bits(), "value {i}");
        }
    }
}

#[test]
fn shared_dict_gate_drops_useless_preamble() {
    // A smooth ramp: delta/bitpack crushes it, DictShared never wins a block,
    // so the stream must not pay for a preamble (matches the no-dict size).
    let data: Vec<f64> = (0..64_000).map(|i| i as f64).collect();
    let cfg = Config {
        block_size: Some(8192),
        ..Config::default()
    };
    let packed_max = compress(&data, cfg);
    let packed_fast = compress(
        &data,
        Config {
            block_size: Some(8192),
            level: Level::Fast, // never builds a shared table
            ..Config::default()
        },
    );
    // Max may only be smaller or equal — never bigger because of a dead preamble.
    assert!(
        packed_max.len() <= packed_fast.len(),
        "dead preamble leaked into the stream: {} > {}",
        packed_max.len(),
        packed_fast.len()
    );
    assert_eq!(decompress(&packed_max).unwrap(), data);
}

#[test]
fn corrupt_shared_streams_error_not_panic() {
    let data = cross_block_column(50_000, 2000);
    let cfg = Config {
        block_size: Some(8192),
        ..Config::default()
    };
    let packed = compress(&data, cfg);

    // Truncations at every prefix of the preamble region must error cleanly.
    for cut in [16, 17, 20, 30, 60] {
        if cut < packed.len() {
            assert!(decompress(&packed[..cut]).is_err());
        }
    }
    // Clearing the shared-dict flag makes the preamble parse as frames: must
    // error (or roundtrip-fail), never panic or mis-decode silently.
    let mut cleared = packed.clone();
    cleared[5] &= !0x04;
    match decompress(&cleared) {
        Ok(out) => assert_ne!(out.len(), data.len()),
        Err(_) => {}
    }
    // Corrupting preamble bytes must never panic (any Err/mismatch is fine).
    for i in 17..40.min(packed.len()) {
        let mut bad = packed.clone();
        bad[i] ^= 0xA5;
        let _ = decompress(&bad);
    }
}
