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
    // dict/RLE), so on this float column it must not out-compress Fast.
    let fast = sizes[1].1;
    assert!(
        fastest >= fast,
        "Fastest ({fastest}) ≥ Fast ({fast}) in pool"
    );
    // And the two must not run the identical competition (the bug this guards
    // against): on a decimal column Fast's ALP/FLOAT_MULT win where Fastest's
    // bit-packers cannot follow. (The smooth sine above no longer separates
    // them: patched delta-bitpack wins at both.)
    let cents: Vec<f64> = (0..100_000).map(|i| ((i * 7) % 100_000) as f64 / 100.0).collect();
    let size_at = |level| {
        quoin::compress(
            &cents,
            Config {
                level,
                ..Config::default()
            },
        )
        .len()
    };
    let (fastest_c, fast_c) = (size_at(Level::Fastest), size_at(Level::Fast));
    assert!(
        fastest_c > fast_c,
        "Fastest ({fastest_c}) and Fast ({fast_c}) must differ on decimals"
    );
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
    assert!(
        small >= adaptive,
        "tiny blocks cost some ratio: {small} vs {adaptive}"
    );
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
    // patterns: every value — finite, infinity, signed zero, subnormal and NaN
    // with any payload — must be bit-exact (see `DType::F32`).
    let mut vals: Vec<f32> = (0..4096)
        .map(|i| 100.0 + (i % 700) as f32 / 100.0)
        .collect();
    let finite_len = vals.len();
    vals.push(f32::from_bits(0x0000_0001)); // smallest subnormal
    vals.push(-0.0);
    vals.push(f32::INFINITY);
    vals.push(f32::NEG_INFINITY);
    vals.push(1.0e-30); // tiny normal
    let nan_idx = vals.len();
    vals.push(f32::from_bits(0x7F80_0001)); // signaling NaN
    vals.push(f32::from_bits(0xFFAB_CDEF)); // negative NaN w/ payload

    for cfg in [
        cfg_full(),
        cfg_sample(),
        Config {
            level: Level::Fast,
            ..cfg_full()
        },
    ] {
        let packed = compress_column(ColumnRef::F32(&vals), None, cfg);
        let dec = decompress_column(&packed).unwrap();
        assert_eq!(dec.values.dtype(), DType::F32);
        match dec.values {
            Column::F32(got) => {
                assert_eq!(got.len(), vals.len());
                for (i, (a, b)) in got.iter().zip(&vals).enumerate() {
                    // Everything is bit-exact — ±0 / inf / subnormal, and (since
                    // the native 32-bit lane) NaN payloads and signaling bits too.
                    assert_eq!(a.to_bits(), b.to_bits(), "f32 bit-exact at idx {i}");
                }
                let _ = (finite_len, nan_idx);
            }
            other => panic!("expected F32, got {:?}", other.dtype()),
        }
    }

    // A smooth decimal f32 column should beat its raw 4-byte size.
    let smooth: Vec<f32> = (0..8192).map(|i| (i as f32) * 0.01 + 5.0).collect();
    let packed = compress_column(ColumnRef::F32(&smooth), None, cfg_full());
    assert!(
        packed.len() < smooth.len() * 4,
        "f32 column should compress"
    );
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
    assert!(
        packed.len() < vals.len() * 5,
        "incompressible f32 must not double"
    );
}

// ---------------------------------------------------------------------------
// Native 32-bit lanes (format v3): f32 / i32 / u32 are compressed at their own
// width. These tests pin the bit-exactness contract and the wire behaviour.
// ---------------------------------------------------------------------------

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s
}

fn all_configs() -> Vec<Config> {
    let mut v = Vec::new();
    for level in [
        Level::Fastest,
        Level::Fast,
        Level::Balanced,
        Level::High,
        Level::Max,
    ] {
        for selection in [Selection::Full, Selection::Sample] {
            v.push(Config {
                level,
                selection,
                ..Config::default()
            });
        }
    }
    v
}

fn roundtrip_f32_bits(vals: &[f32], cfg: Config, what: &str) -> usize {
    let packed = compress_column(ColumnRef::F32(vals), None, cfg);
    assert_eq!(packed[4], 3, "{what}: native 32-bit streams are format v3");
    let dec = decompress_column(&packed).unwrap();
    assert_eq!(dec.validity, None);
    match dec.values {
        Column::F32(got) => {
            assert_eq!(got.len(), vals.len(), "{what}: length");
            for (i, (a, b)) in got.iter().zip(vals).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{what} @ {cfg:?}: bit-exact at index {i} ({:#010x} vs {:#010x})",
                    a.to_bits(),
                    b.to_bits()
                );
            }
        }
        other => panic!("expected F32, got {:?}", other.dtype()),
    }
    packed.len()
}

/// Every IEEE-754 binary32 class, including signaling NaNs and NaN payloads —
/// the previous widened-f64 design could quiet these; the native lane cannot.
fn f32_edge_patterns() -> Vec<f32> {
    [
        0x0000_0000u32, // +0
        0x8000_0000,    // -0
        0x0000_0001,    // smallest subnormal
        0x807F_FFFF,    // largest negative subnormal
        0x0080_0000,    // smallest normal
        0x7F7F_FFFF,    // f32::MAX
        0xFF7F_FFFF,    // f32::MIN
        0x7F80_0000,    // +inf
        0xFF80_0000,    // -inf
        0x7FC0_0000,    // canonical quiet NaN
        0xFFC0_0000,    // negative quiet NaN
        0x7F80_0001,    // signaling NaN, minimal payload
        0xFF80_0001,    // negative signaling NaN
        0x7FBF_FFFF,    // signaling NaN, full payload
        0x7FC0_1234,    // quiet NaN with payload
        0xFFAB_CDEF,    // negative NaN with payload
        0x3F80_0000,    // 1.0
        0xBF80_0000,    // -1.0
        0x3DCC_CCCD,    // 0.1f32
        0x4049_0FDB,    // pi
    ]
    .into_iter()
    .map(f32::from_bits)
    .collect()
}

#[test]
fn f32_every_bit_pattern_class_is_exact_at_every_level() {
    let edges = f32_edge_patterns();
    // The edge set alone, the edge set repeated (dict/RLE/LZ territory), and the
    // edge set spliced into smooth decimal data (ALP/FLOAT_MULT/predictors with
    // exceptions and the non-finite gate).
    let repeated: Vec<f32> = (0..4000).map(|i| edges[i % edges.len()]).collect();
    let mut spliced: Vec<f32> = (0..6000)
        .map(|i| 100.0 + (i % 700) as f32 / 100.0)
        .collect();
    for (k, &e) in edges.iter().enumerate() {
        spliced[k * 250 + 7] = e;
    }
    let mut smooth_with_nan: Vec<f32> =
        (0..5000).map(|i| (i as f32 * 0.01).sin() * 100.0).collect();
    smooth_with_nan[2500] = f32::from_bits(0x7F80_0001);
    smooth_with_nan[4000] = f32::NEG_INFINITY;
    for cfg in all_configs() {
        roundtrip_f32_bits(&edges, cfg, "edges");
        roundtrip_f32_bits(&repeated, cfg, "repeated edges");
        roundtrip_f32_bits(&spliced, cfg, "spliced edges");
        roundtrip_f32_bits(&smooth_with_nan, cfg, "smooth with sNaN/inf");
    }
}

#[test]
fn f32_random_bit_patterns_roundtrip_at_every_level() {
    // Deterministic random 32-bit patterns: ~1/256 are NaN/inf, the rest span
    // every exponent — RAW territory, but every codec must at least be exact.
    let mut s = 0xF32F_32F3u64;
    let noise: Vec<f32> = (0..20_000)
        .map(|_| f32::from_bits((lcg(&mut s) >> 32) as u32))
        .collect();
    for cfg in all_configs() {
        let size = roundtrip_f32_bits(&noise, cfg, "random bits");
        assert!(
            size < noise.len() * 4 + 512,
            "noise must stay near 4 B/value: {size}"
        );
    }
}

#[test]
fn f32_shapes_and_multiblock() {
    // Several full blocks (the 32-bit lane packs 256 Ki values per 1 MiB block)
    // plus a short tail, across shapes that pick different modes.
    let n = 600_000;
    let mut s = 7u64;
    let shapes: Vec<(&str, Vec<f32>)> = vec![
        ("const", vec![2.5f32; n]),
        ("stride-like ramp", (0..n).map(|i| i as f32).collect()),
        (
            "decimal sensor",
            (0..n).map(|i| 1234.5 + ((i % 5000) as f32) * 0.1).collect(),
        ),
        (
            "smooth",
            (0..n).map(|i| (i as f32 * 0.001).sin() * 2000.0).collect(),
        ),
        (
            "low-card",
            (0..n).map(|_| (lcg(&mut s) >> 60) as f32 * 0.125).collect(),
        ),
        ("runs", (0..n).map(|i| (i / 300) as f32 * 0.5).collect()),
    ];
    for (what, vals) in &shapes {
        for level in [Level::Fastest, Level::Balanced, Level::Max] {
            let cfg = Config {
                level,
                ..Config::default()
            };
            let size = roundtrip_f32_bits(vals, cfg, what);
            // `Fastest` has no dictionary codec, so scattered low-cardinality
            // floats (wide bit-pattern range) legitimately stay at raw there.
            if !(*what == "low-card" && level == Level::Fastest) {
                assert!(
                    size < vals.len() * 4,
                    "{what} @ {level:?} must beat raw: {size}"
                );
            }
        }
        // Sampled selection and a small fixed block size.
        roundtrip_f32_bits(
            vals,
            Config {
                selection: Selection::Sample,
                ..Config::default()
            },
            what,
        );
        roundtrip_f32_bits(
            &vals[..50_000],
            Config {
                block_size: Some(777),
                level: Level::Fast,
                ..Config::default()
            },
            what,
        );
    }
    // Empty, single, and a block-boundary-straddling length.
    for cfg in all_configs() {
        roundtrip_f32_bits(&[], cfg, "empty");
        roundtrip_f32_bits(&[-0.0], cfg, "single");
        roundtrip_f32_bits(&[f32::from_bits(0x7F80_0001)], cfg, "single sNaN");
    }
    let straddle: Vec<f32> = (0..(256 * 1024 + 3))
        .map(|i| (i % 97) as f32 * 0.25)
        .collect();
    roundtrip_f32_bits(&straddle, Config::default(), "block straddle");
}

#[test]
fn f32_nullable_roundtrip_is_exact() {
    let n = 9_001usize;
    let mut s = 3u64;
    let valid: Vec<bool> = (0..n)
        .map(|i| {
            s = lcg(&mut s);
            !(4000..4300).contains(&i) && s & 3 != 0
        })
        .collect();
    let bm = bitmap_from_bools(&valid);
    let edges = f32_edge_patterns();
    let vals: Vec<f32> = (0..n)
        .map(|i| {
            if i % 500 == 0 {
                edges[(i / 500) % edges.len()]
            } else {
                i as f32 * 0.01
            }
        })
        .collect();
    for cfg in all_configs() {
        let packed = compress_column(ColumnRef::F32(&vals), Some(&bm), cfg);
        let dec = decompress_column(&packed).unwrap();
        assert_eq!(dec.validity.as_deref(), Some(&bm[..]));
        match dec.values {
            Column::F32(got) => {
                for i in 0..n {
                    let want = if valid[i] { vals[i].to_bits() } else { 0 };
                    assert_eq!(got[i].to_bits(), want, "{cfg:?} slot {i}");
                }
            }
            _ => panic!(),
        }
    }
    // All-null and all-valid edges.
    let allnull = bitmap_from_bools(&vec![false; 100]);
    let dec = decompress_column(&compress_column(
        ColumnRef::F32(&[f32::NAN; 100]),
        Some(&allnull),
        Config::default(),
    ))
    .unwrap();
    assert_eq!(dec.validity.as_deref(), Some(&allnull[..]));
    match dec.values {
        Column::F32(got) => assert!(got.iter().all(|v| v.to_bits() == 0)),
        _ => panic!(),
    }
}

#[test]
fn f32_is_not_a_widened_f64_stream() {
    // A constant f32 block must be a 4-byte CONST payload, not an 8-byte f64.
    let packed = compress_column(ColumnRef::F32(&[1.5f32; 1000]), None, Config::default());
    // header(16) + frame header(9) + 4-byte payload.
    assert_eq!(
        packed.len(),
        16 + 9 + 4,
        "CONST payload is one 32-bit lane word"
    );
    assert_eq!(&packed[25..29], &1.5f32.to_bits().to_le_bytes());
    // An f32 column and the same values as f64 give different streams (the
    // f64 one is at least as large on every mode that stores lane words).
    let vals32: Vec<f32> = (0..5000).map(|i| (i as f32 * 0.7).sin()).collect();
    let vals64: Vec<f64> = vals32.iter().map(|&v| f64::from(v)).collect();
    for level in [Level::Fastest, Level::Fast, Level::Balanced, Level::Max] {
        let cfg = Config {
            level,
            ..Config::default()
        };
        let p32 = compress_column(ColumnRef::F32(&vals32), None, cfg);
        let p64 = compress_column(ColumnRef::F64(&vals64), None, cfg);
        assert!(
            p32.len() <= p64.len(),
            "{level:?}: native f32 ({}) ≤ exact f64 ({})",
            p32.len(),
            p64.len()
        );
    }
}

#[test]
fn legacy_v2_narrow_streams_are_rejected_wide_ones_decode() {
    // A v2 f32 stream held widened-f64 payloads; misreading it as v3 would be
    // silent garbage, so the decoder refuses it explicitly.
    let mut p = compress_column(ColumnRef::F32(&[1.0, 2.0, 3.0]), None, Config::default());
    p[4] = 2;
    assert_eq!(
        decompress_column(&p),
        Err(quoin::Error::UnsupportedVersion(2))
    );
    let mut p = compress_column(ColumnRef::I32(&[1, 2, 3]), None, Config::default());
    p[4] = 2;
    assert_eq!(
        decompress_column(&p),
        Err(quoin::Error::UnsupportedVersion(2))
    );
    // 64-bit lanes and decimals did not change layout: a v2 stamp still decodes.
    let mut p = compress_column(ColumnRef::F64(&[1.0, 2.0, 3.0]), None, Config::default());
    assert_eq!(p[4], 3);
    p[4] = 2;
    assert_eq!(
        decompress_column(&p).unwrap().values,
        Column::F64(vec![1.0, 2.0, 3.0])
    );
    let mut p = compress_column(
        ColumnRef::Decimal128 {
            values: &[1, 2, 3],
            scale: 2,
            precision: 10,
        },
        None,
        Config::default(),
    );
    p[4] = 2;
    assert!(decompress_column(&p).is_ok());
    p[4] = 1;
    assert_eq!(
        decompress_column(&p),
        Err(quoin::Error::UnsupportedVersion(1))
    );
}

#[test]
fn i32_u32_native_lane_behaviour() {
    // CONST on the 32-bit lane is a 4-byte payload; the lane is signed-aware.
    let packed = compress_column(ColumnRef::I32(&[-7i32; 1000]), None, Config::default());
    assert_eq!(packed.len(), 16 + 9 + 4);
    assert_eq!(&packed[25..29], &(-7i32).to_le_bytes());
    // Every level/selection, several shapes, exact.
    let mut s = 11u64;
    let shapes: Vec<Vec<i32>> = vec![
        (0..70_000).map(|i| (i % 4093) - 2000).collect(), // mixed sign, bounded
        (0..70_000).map(|_| (lcg(&mut s) >> 32) as i32).collect(), // noise
        (0..70_000).map(|i| i * 2 + (i % 3)).collect(),   // monotone (relative time)
        vec![i32::MIN, i32::MAX, 0, -1, 1],
        vec![],
    ];
    for vals in &shapes {
        for cfg in all_configs() {
            roundtrip_i32(vals, cfg);
            let u: Vec<u32> = vals.iter().map(|&v| v as u32).collect();
            roundtrip_u32(&u, cfg);
        }
    }
    // The 32-bit lane allows 2× the 64-bit block value cap (same byte budget).
    let big: Vec<u32> = (0..300_000u32).map(|i| i % 1000).collect();
    let packed = compress_column(
        ColumnRef::U32(&big),
        None,
        Config {
            block_size: Some(1 << 30),
            ..Config::default()
        },
    );
    roundtrip_u32(&big, Config::default());
    match decompress_column(&packed).unwrap().values {
        Column::U32(got) => assert_eq!(got, big),
        _ => panic!(),
    }
    // Frame value counts: 262144 (the 32-bit cap) then the remainder.
    let n0 = u32::from_le_bytes(packed[17..21].try_into().unwrap());
    assert_eq!(n0, 256 * 1024, "32-bit lane blocks hold 256 Ki values");
}

#[test]
fn f32_mode_coverage_is_native() {
    // Sanity that the float-value codecs actually engage on f32 (not just RAW/
    // pco): a decimal column at `Fast` (no entropy, no pco) must compress well —
    // only ALP / FLOAT_MULT / bit-packers are available there.
    let vals: Vec<f32> = (0..50_000)
        .map(|i| 1000.0 + (i % 9000) as f32 * 0.1)
        .collect();
    let cfg = Config {
        level: Level::Fast,
        ..Config::default()
    };
    let size = roundtrip_f32_bits(&vals, cfg, "decimal @ Fast");
    assert!(
        size < vals.len() * 2,
        "ALP/FLOAT_MULT must engage on f32 decimals: {size}"
    );
    // And a smooth non-decimal signal at High (float predictors, f32 arithmetic).
    let smooth: Vec<f32> = (0..50_000)
        .map(|i| ((i as f32) * 0.0007).sin() * 1.0e-3)
        .collect();
    let cfg = Config {
        level: Level::High,
        ..Config::default()
    };
    let size = roundtrip_f32_bits(&smooth, cfg, "smooth @ High");
    assert!(
        size < smooth.len() * 3,
        "predictors must engage on smooth f32: {size}"
    );
}
