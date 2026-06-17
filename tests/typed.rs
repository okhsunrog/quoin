//! Round-trip and behavior tests for the typed-column API (i64/u64 alongside f64).

use quoin::{
    Column, ColumnRef, Config, DType, Level, Selection, compress_column, decompress_column,
};

fn cfg_full() -> Config {
    Config::default()
}
fn cfg_sample() -> Config {
    Config {
        selection: Selection::Sample,
        ..Config::default()
    }
}

fn roundtrip_i64(vals: &[i64], cfg: Config) -> usize {
    let packed = compress_column(ColumnRef::I64(vals), None, cfg);
    match decompress_column(&packed).unwrap().values {
        Column::I64(got) => assert_eq!(got, vals),
        other => panic!("expected I64, got {:?}", other.dtype()),
    }
    packed.len()
}

fn roundtrip_u64(vals: &[u64], cfg: Config) -> usize {
    let packed = compress_column(ColumnRef::U64(vals), None, cfg);
    match decompress_column(&packed).unwrap().values {
        Column::U64(got) => assert_eq!(got, vals),
        other => panic!("expected U64, got {:?}", other.dtype()),
    }
    packed.len()
}

fn roundtrip_i32(vals: &[i32], cfg: Config) -> usize {
    let packed = compress_column(ColumnRef::I32(vals), None, cfg);
    match decompress_column(&packed).unwrap().values {
        Column::I32(got) => assert_eq!(got, vals),
        other => panic!("expected I32, got {:?}", other.dtype()),
    }
    packed.len()
}

fn roundtrip_u32(vals: &[u32], cfg: Config) -> usize {
    let packed = compress_column(ColumnRef::U32(vals), None, cfg);
    match decompress_column(&packed).unwrap().values {
        Column::U32(got) => assert_eq!(got, vals),
        other => panic!("expected U32, got {:?}", other.dtype()),
    }
    packed.len()
}

#[test]
fn i64_shapes_roundtrip() {
    for &cfg in &[cfg_full(), cfg_sample()] {
        roundtrip_i64(&[], cfg);
        roundtrip_i64(&[42], cfg);
        roundtrip_i64(&[-1, 0, 1, i64::MIN, i64::MAX], cfg);
        // constant
        roundtrip_i64(&vec![7i64; 5000], cfg);
        // monotone ramp
        roundtrip_i64(&(0..5000i64).collect::<Vec<_>>(), cfg);
        // negative ramp
        roundtrip_i64(&(-5000..0i64).collect::<Vec<_>>(), cfg);
        // bounded ids (FoR-friendly)
        let ids: Vec<i64> = (0..8000).map(|i| 1000 + (i % 300)).collect();
        roundtrip_i64(&ids, cfg);
        // pseudo-random wide
        let mut s = 1u64;
        let wide: Vec<i64> = (0..8000)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                s as i64
            })
            .collect();
        roundtrip_i64(&wide, cfg);
    }
}

#[test]
fn u64_shapes_roundtrip() {
    for &cfg in &[cfg_full(), cfg_sample()] {
        roundtrip_u64(&[], cfg);
        roundtrip_u64(&[u64::MAX], cfg);
        roundtrip_u64(&vec![9u64; 5000], cfg);
        roundtrip_u64(&(0..5000u64).collect::<Vec<_>>(), cfg);
        let ids: Vec<u64> = (0..8000).map(|i| 1_000_000 + (i % 250)).collect();
        roundtrip_u64(&ids, cfg);
    }
}

#[test]
fn i32_u32_shapes_roundtrip() {
    for &cfg in &[cfg_full(), cfg_sample()] {
        roundtrip_i32(&[], cfg);
        roundtrip_i32(&[i32::MIN, -1, 0, 1, i32::MAX], cfg);
        roundtrip_i32(&vec![-7i32; 5000], cfg);
        roundtrip_i32(&(-2500..2500i32).collect::<Vec<_>>(), cfg);
        roundtrip_u32(&[], cfg);
        roundtrip_u32(&[u32::MAX], cfg);
        roundtrip_u32(&(0..5000u32).collect::<Vec<_>>(), cfg);
        let ids: Vec<u32> = (0..8000).map(|i| 100_000 + (i % 300)).collect();
        roundtrip_u32(&ids, cfg);
    }
}

#[test]
fn i32_mixed_sign_packs() {
    // Signed FoR: small +/- values pack tight (would bail to 64-bit unsigned).
    let vals: Vec<i32> = (0..16384i32).map(|i| (i % 400) - 200).collect();
    let size = roundtrip_i32(&vals, cfg_full());
    assert!(
        size < vals.len() * 2,
        "mixed-sign i32 should pack <2 B/value, got {}",
        size as f64 / vals.len() as f64
    );
}

#[test]
fn u32_raw_baseline_is_four_bytes() {
    // Incompressible 32-bit data: the RAW baseline must be ~4 B/value, not 8
    // (the internal u64 lane must not double a narrow column's floor).
    let mut s = 0x9E37_79B1u32;
    let noise: Vec<u32> = (0..16384)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            s
        })
        .collect();
    let size = roundtrip_u32(&noise, cfg_full());
    assert!(
        size < noise.len() * 5,
        "incompressible u32 should stay near 4 B/value, got {}",
        size as f64 / noise.len() as f64
    );
}

#[test]
fn timestamps_compress_well() {
    // Irregular-but-monotone i64 timestamps (the Timestamp -> i64 lane case):
    // delta+bitpack territory.
    let mut t = 1_700_000_000_000i64;
    let mut s = 12345u64;
    let ts: Vec<i64> = (0..16384)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            t += 1000 + (s >> 40) as i64 % 4096;
            t
        })
        .collect();
    let size = roundtrip_i64(&ts, cfg_full());
    assert!(
        size < ts.len() * 8 / 2,
        "monotone timestamps should at least halve: {size} vs raw {}",
        ts.len() * 8
    );
}

#[test]
fn bounded_ids_pack_small() {
    let ids: Vec<i64> = (0..16384).map(|i| 50_000 + (i % 1000)).collect();
    let size = roundtrip_i64(&ids, cfg_full());
    // ~10 bits/value + frame overhead, far under 8 bytes/value.
    assert!(
        size < ids.len() * 8 / 4,
        "bounded ids should pack to <2 B/value: {size}"
    );
}

#[test]
fn header_records_dtype() {
    let packed_i = compress_column(ColumnRef::I64(&[1, 2, 3]), None, cfg_full());
    let packed_u = compress_column(ColumnRef::U64(&[1, 2, 3]), None, cfg_full());
    let packed_f = compress_column(ColumnRef::F64(&[1.0, 2.0, 3.0]), None, cfg_full());
    // Byte [7] is the dtype wire id; versions all share magic + version=2.
    assert_eq!(&packed_i[0..5], &packed_f[0..5]); // magic + version
    assert_eq!(packed_f[7], 0); // F64
    assert_eq!(packed_i[7], 1); // I64
    assert_eq!(packed_u[7], 2); // U64
    assert_eq!(
        decompress_column(&packed_i).unwrap().values.dtype(),
        DType::I64
    );
    assert_eq!(
        decompress_column(&packed_u).unwrap().values.dtype(),
        DType::U64
    );
    assert_eq!(
        decompress_column(&packed_f).unwrap().values.dtype(),
        DType::F64
    );
}

#[test]
fn f64_convenience_matches_typed() {
    let data: Vec<f64> = (0..10_000).map(|i| (i as f64) * 0.5).collect();
    let via_compress = quoin::compress(&data, cfg_full());
    let via_column = compress_column(ColumnRef::F64(&data), None, cfg_full());
    assert_eq!(
        via_compress, via_column,
        "compress() must equal the typed path"
    );
    assert_eq!(quoin::decompress(&via_compress).unwrap(), data);
}

fn sine_f64() -> Vec<f64> {
    (0..200_000)
        .map(|i| (i as f64 * 0.01).sin() * 1000.0 + (i as f64) * 0.5)
        .collect()
}

#[test]
fn level_max_is_default_behavior() {
    // Level::Max must reproduce the historical (λ=0, all codecs) output exactly.
    let data = sine_f64();
    let default = quoin::compress(&data, Config::default());
    let max = quoin::compress(
        &data,
        Config {
            level: Level::Max,
            ..Config::default()
        },
    );
    assert_eq!(default, max, "Max level must equal the default");
}

#[test]
fn levels_trade_ratio_for_speed_and_roundtrip() {
    let data = sine_f64();
    let mut sizes = Vec::new();
    for level in [
        Level::Fastest,
        Level::Fast,
        Level::Balanced,
        Level::High,
        Level::Max,
    ] {
        let cfg = Config {
            level,
            ..Config::default()
        };
        let packed = quoin::compress(&data, cfg);
        assert_eq!(
            quoin::decompress(&packed).unwrap(),
            data,
            "{level:?} round-trip"
        );
        sizes.push((level, packed.len()));
    }
    // Faster levels must not beat Max on ratio (Max is the most thorough).
    let max_size = sizes.last().unwrap().1;
    for (level, size) in &sizes {
        assert!(
            *size >= max_size,
            "{level:?} ({size}) should not beat Max ({max_size}) on ratio"
        );
    }
    // And the fastest level should be meaningfully larger here (entropy off).
    let fastest = sizes[0].1;
    assert!(
        fastest > max_size,
        "Fastest ({fastest}) should trade ratio vs Max ({max_size})"
    );
    // Fastest runs a strictly smaller codec pool than Fast (no XORZ/ALP/ALP-RD/
    // dict/RLE), so on this float column it must not out-compress Fast — and the
    // two must no longer be identical (the bug this guards against).
    let fast = sizes[1].1;
    assert!(fastest >= fast, "Fastest ({fastest}) ≥ Fast ({fast}) in pool");
    assert!(fastest > fast, "Fastest and Fast must differ (distinct levels)");
}

#[test]
fn levels_apply_to_integer_columns() {
    let ids: Vec<i64> = (0..100_000).map(|i| 1000 + (i % 500)).collect();
    for level in [Level::Fastest, Level::Fast, Level::Balanced, Level::Max] {
        let cfg = Config {
            level,
            ..Config::default()
        };
        let packed = compress_column(ColumnRef::I64(&ids), None, cfg);
        match decompress_column(&packed).unwrap().values {
            Column::I64(got) => assert_eq!(got, ids, "{level:?} i64 round-trip"),
            other => panic!("expected I64, got {:?}", other.dtype()),
        }
        // FoR+bitpack is a non-entropy codec, so even Fastest packs these well.
        assert!(packed.len() < ids.len() * 2, "{level:?}: ids should pack");
    }
}

fn bitmap_from_bools(bits: &[bool]) -> Vec<u8> {
    let mut bm = vec![0u8; bits.len().div_ceil(8)];
    for (i, &b) in bits.iter().enumerate() {
        if b {
            bm[i >> 3] |= 1 << (i & 7);
        }
    }
    bm
}

#[test]
fn nullable_roundtrip_and_compaction() {
    let n = 8000usize;
    let mut s = 1u64;
    // ~1/8 nulls, scattered + a clustered run.
    let valid: Vec<bool> = (0..n)
        .map(|i| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            !(2000..2500).contains(&i) && s & 7 != 0
        })
        .collect();
    let bm = bitmap_from_bools(&valid);

    // i64 (with the value codec only seeing valid values → tight pack).
    let vals: Vec<i64> = (0..n as i64).map(|i| 1000 + (i % 200)).collect();
    let packed = compress_column(ColumnRef::I64(&vals), Some(&bm), cfg_full());
    let dec = decompress_column(&packed).unwrap();
    assert_eq!(
        dec.validity.as_deref(),
        Some(&bm[..]),
        "validity round-trips"
    );
    match dec.values {
        Column::I64(got) => {
            for i in 0..n {
                if valid[i] {
                    assert_eq!(got[i], vals[i], "valid slot {i}");
                } else {
                    assert_eq!(got[i], 0, "null slot {i} must be 0");
                }
            }
        }
        other => panic!("expected I64, got {:?}", other.dtype()),
    }
    // The compacted dictionary-like ids should still pack well under raw.
    assert!(
        packed.len() < n * 4,
        "nullable column should compress: {}",
        packed.len()
    );

    // f64 nullable round-trips too.
    let fvals: Vec<f64> = (0..n).map(|i| (i as f64) * 0.25).collect();
    let fpacked = compress_column(ColumnRef::F64(&fvals), Some(&bm), cfg_sample());
    let fdec = decompress_column(&fpacked).unwrap();
    assert_eq!(fdec.validity.as_deref(), Some(&bm[..]));
    match fdec.values {
        Column::F64(got) => {
            for i in 0..n {
                assert_eq!(
                    got[i].to_bits(),
                    if valid[i] { fvals[i] } else { 0.0 }.to_bits()
                );
            }
        }
        _ => panic!(),
    }
}

#[test]
fn nullable_edges() {
    // all-valid bitmap is normalized to None (no nulls).
    let vals = vec![5i64; 1000];
    let allset = bitmap_from_bools(&vec![true; 1000]);
    let dec = decompress_column(&compress_column(
        ColumnRef::I64(&vals),
        Some(&allset),
        cfg_full(),
    ))
    .unwrap();
    assert_eq!(dec.validity, None, "all-valid → no validity");

    // all-null: every slot decodes to 0, validity all-clear.
    let allnull = bitmap_from_bools(&vec![false; 1000]);
    let dec = decompress_column(&compress_column(
        ColumnRef::U64(&vec![9u64; 1000]),
        Some(&allnull),
        cfg_full(),
    ))
    .unwrap();
    assert_eq!(dec.validity.as_deref(), Some(&allnull[..]));
    match dec.values {
        Column::U64(got) => assert!(got.iter().all(|&v| v == 0)),
        _ => panic!(),
    }

    // count not a multiple of 8.
    let valid = [true, false, true, true, false, true, true];
    let bm = bitmap_from_bools(&valid);
    let dec = decompress_column(&compress_column(
        ColumnRef::I32(&[10, 20, 30, 40, 50, 60, 70]),
        Some(&bm),
        cfg_full(),
    ))
    .unwrap();
    assert_eq!(dec.validity.as_deref(), Some(&bm[..]));
    match dec.values {
        Column::I32(got) => assert_eq!(got, vec![10, 0, 30, 40, 0, 60, 70]),
        _ => panic!(),
    }
}

#[test]
fn configurable_block_size_roundtrips_and_changes_layout() {
    // A column with enough values to span several fixed blocks.
    let vals: Vec<i64> = (0..100_000).map(|i| 1000 + (i % 777) as i64).collect();

    // Every fixed size round-trips exactly, and a tiny size (point-access
    // granularity) and an over-large size (clamped) both work.
    for bs in [64usize, 1024, 8192, 1 << 30] {
        let cfg = Config {
            block_size: Some(bs),
            ..Config::default()
        };
        let packed = compress_column(ColumnRef::I64(&vals), None, cfg);
        match decompress_column(&packed).unwrap().values {
            Column::I64(got) => assert_eq!(got, vals, "block_size {bs}"),
            other => panic!("expected I64, got {:?}", other.dtype()),
        }
    }

    // Smaller blocks ⇒ more frames ⇒ a (weakly) larger stream than the adaptive
    // default; both still compress well below raw.
    let raw = vals.len() * 8;
    let small = compress_column(
        ColumnRef::I64(&vals),
        None,
        Config {
            block_size: Some(256),
            ..Config::default()
        },
    )
    .len();
    let adaptive = compress_column(ColumnRef::I64(&vals), None, Config::default()).len();
    assert!(small >= adaptive, "tiny blocks cost some ratio: {small} vs {adaptive}");
    assert!(small < raw && adaptive < raw);

    // block_size: None is the adaptive default and is byte-identical to omitting it.
    let explicit_none = compress_column(
        ColumnRef::I64(&vals),
        None,
        Config {
            block_size: None,
            ..Config::default()
        },
    );
    assert_eq!(explicit_none.len(), adaptive);
}

#[test]
fn f64_decompress_rejects_other_types() {
    let packed = compress_column(ColumnRef::I64(&[1, 2, 3]), None, cfg_full());
    assert_eq!(quoin::decompress(&packed), Err(quoin::Error::DTypeMismatch));
}

#[test]
fn f32_roundtrip_and_dtype() {
    // Decimal-ish f32 column (ALP / FLOAT_MULT territory), plus exotic bit
    // patterns: every finite value, infinity, signed zero and subnormal must be
    // bit-exact; NaN inputs must come back as NaN (payload not guaranteed — see
    // `DType::F32`).
    let mut vals: Vec<f32> = (0..4096).map(|i| 100.0 + (i % 700) as f32 / 100.0).collect();
    let finite_len = vals.len();
    vals.push(f32::from_bits(0x0000_0001)); // smallest subnormal
    vals.push(-0.0);
    vals.push(f32::INFINITY);
    vals.push(f32::NEG_INFINITY);
    vals.push(1.0e-30); // tiny normal
    let nan_idx = vals.len();
    vals.push(f32::from_bits(0x7F80_0001)); // signaling NaN
    vals.push(f32::from_bits(0xFFAB_CDEF)); // negative NaN w/ payload

    for cfg in [cfg_full(), cfg_sample(), Config { level: Level::Fast, ..cfg_full() }] {
        let packed = compress_column(ColumnRef::F32(&vals), None, cfg);
        let dec = decompress_column(&packed).unwrap();
        assert_eq!(dec.values.dtype(), DType::F32);
        match dec.values {
            Column::F32(got) => {
                assert_eq!(got.len(), vals.len());
                for (i, (a, b)) in got.iter().zip(&vals).enumerate() {
                    if i >= nan_idx {
                        // NaN value preserved; payload bits are not guaranteed.
                        assert!(a.is_nan(), "f32 NaN value preserved");
                    } else {
                        // Everything non-NaN is bit-exact, including ±0 / inf / subnormal.
                        assert_eq!(a.to_bits(), b.to_bits(), "f32 bit-exact at idx {i}");
                    }
                }
                let _ = finite_len;
            }
            other => panic!("expected F32, got {:?}", other.dtype()),
        }
    }

    // A smooth decimal f32 column should beat its raw 4-byte size.
    let smooth: Vec<f32> = (0..8192).map(|i| (i as f32) * 0.01 + 5.0).collect();
    let packed = compress_column(ColumnRef::F32(&smooth), None, cfg_full());
    assert!(packed.len() < smooth.len() * 4, "f32 column should compress");
}

#[test]
fn f32_incompressible_does_not_expand_much() {
    // Random-ish *finite* f32 falls to RAW; RAW must emit the compact 4-byte form,
    // not the widened 8-byte lane, so the stream stays near the 4-byte/value
    // baseline. Force the exponent into [1, 254] so every value is a finite normal
    // (and the round trip is bit-exact — no NaN-payload caveat in play).
    let vals: Vec<f32> = (0..4096u32)
        .map(|i| {
            let b = 0x4000_0000u32.wrapping_add(i.wrapping_mul(2_654_435_761));
            let exp = (b >> 23) % 254 + 1;
            f32::from_bits((b & 0x807F_FFFF) | (exp << 23))
        })
        .collect();
    let packed = compress_column(ColumnRef::F32(&vals), None, cfg_full());
    let dec = decompress_column(&packed).unwrap();
    match dec.values {
        Column::F32(got) => {
            for (a, b) in got.iter().zip(&vals) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }
        other => panic!("expected F32, got {:?}", other.dtype()),
    }
    // Allow modest framing overhead, but nowhere near the 8-byte widened lane.
    assert!(packed.len() < vals.len() * 5, "incompressible f32 must not double");
}
