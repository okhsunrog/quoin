//! Decimal columns (`Decimal128` today; `Decimal256` by the same scheme).
//!
//! A decimal is an integer significand `v` plus a fixed `scale`/`precision`. The
//! significand is wider than the engine's `u64` lane, so the container:
//!   1. subtracts a global `vmin` (the column minimum over the valid values) →
//!      a **non-negative** offset,
//!   2. splits that offset into 64-bit limbs (2 for `Decimal128`, 4 for
//!      `Decimal256`), little-endian,
//!   3. compresses each limb as an ordinary `U64` column through the full engine.
//!
//! In the common case every value fits in 64 bits after the `vmin` shift, so the
//! high limb(s) are all-zero → `CONST` → ~free, and the low limb gets the
//! engine's per-block frame-of-reference, delta, dict, … No special wide-range
//! path is needed — a genuinely 128-bit-spread column simply has a non-trivial
//! high limb. `vmin`, `scale` and `precision` ride in the container header.
//!
//! Nulls are compacted out once at the container level (the limbs carry only
//! valid significands); the bitmap is stored once and re-applied on decode.

use crate::Config;
use crate::decoder::decompress_lane;
use crate::dtype::DType;
use crate::error::Error;
use crate::format::{FLAG_DECIMAL, FLAG_VALIDITY, HEADER_LEN, MAGIC, VERSION};

/// Offset of the decimal metadata within the container (right after the header).
const META_OFF: usize = HEADER_LEN;

/// A decoded `Decimal128` column.
pub(crate) struct Decoded128 {
    pub values: Vec<i128>,
    pub scale: i8,
    pub precision: u8,
    pub validity: Option<Vec<u8>>,
}

/// `true` if `src` is a decimal container (so the top-level decoder must route it
/// here rather than to the ordinary lane decoder).
pub(crate) fn is_decimal_stream(src: &[u8]) -> bool {
    src.len() > 5 && src[5] & FLAG_DECIMAL != 0
}

fn write_header(out: &mut Vec<u8>, dtype: DType, n_values: usize, has_validity: bool) {
    out.extend_from_slice(&MAGIC);
    out.push(VERSION);
    out.push(FLAG_DECIMAL | if has_validity { FLAG_VALIDITY } else { 0 });
    out.push(0); // predictor_log2 unused at the container level
    out.push(dtype.wire_id());
    out.extend_from_slice(&(n_values as u64).to_le_bytes());
}

/// Shared container layout written by every decimal width. `vmin_le` is the
/// minimum as little-endian two's-complement bytes (`n_limbs * 8` of them); each
/// entry of `limbs` is one 64-bit limb across the valid values.
#[allow(clippy::too_many_arguments)]
fn assemble(
    dtype: DType,
    logical_n: usize,
    scale: i8,
    precision: u8,
    vmin_le: &[u8],
    validity: Option<&[u8]>,
    limbs: &[Vec<u64>],
    n_valid: usize,
    cfg: Config,
) -> Vec<u8> {
    let has_validity = validity.is_some_and(|bm| crate::validity::count_valid(bm, logical_n) < logical_n);
    let mut out = Vec::new();
    write_header(&mut out, dtype, logical_n, has_validity);
    out.push(scale as u8);
    out.push(precision);
    out.push(limbs.len() as u8);
    out.extend_from_slice(vmin_le);
    if has_validity {
        let vblob = crate::validity::encode(validity.unwrap(), logical_n);
        crate::varint::write_u64(&mut out, vblob.len() as u64);
        out.extend_from_slice(&vblob);
    }
    for limb in limbs {
        let sub = crate::encoder::compress_lane(limb, DType::U64, n_valid, None, cfg);
        crate::varint::write_u64(&mut out, sub.len() as u64);
        out.extend_from_slice(&sub);
    }
    out
}

/// Parsed container shell: the metadata plus the decoded `u64` limbs (each of
/// length `n_valid`) and the validity bitmap, ready for width-specific recombine.
struct Shell {
    scale: i8,
    precision: u8,
    logical_n: usize,
    n_valid: usize,
    vmin_off: usize,
    n_limbs: usize,
    limbs: Vec<Vec<u64>>,
    validity: Option<Vec<u8>>,
}

fn parse_shell(src: &[u8], expect: DType) -> Result<Shell, Error> {
    if src.len() < META_OFF + 3 {
        return Err(Error::Truncated);
    }
    if src[0..4] != MAGIC {
        return Err(Error::BadMagic);
    }
    if src[4] != VERSION {
        return Err(Error::UnsupportedVersion(src[4]));
    }
    if src[5] & FLAG_DECIMAL == 0 {
        return Err(Error::CorruptPayload("not a decimal container"));
    }
    let has_validity = src[5] & FLAG_VALIDITY != 0;
    let dtype = DType::from_wire(src[7])?;
    if dtype != expect {
        return Err(Error::DTypeMismatch);
    }
    let logical_n =
        usize::try_from(u64::from_le_bytes(src[8..16].try_into().unwrap())).map_err(|_| Error::Truncated)?;
    let scale = src[META_OFF] as i8;
    let precision = src[META_OFF + 1];
    let n_limbs = src[META_OFF + 2] as usize;
    if n_limbs == 0 || n_limbs > 4 {
        return Err(Error::CorruptPayload("decimal limb count"));
    }
    let vmin_off = META_OFF + 3;
    let mut pos = vmin_off + n_limbs * 8;
    if pos > src.len() {
        return Err(Error::Truncated);
    }
    let validity = if has_validity {
        let vlen = usize::try_from(crate::varint::read_u64(src, &mut pos)?)
            .map_err(|_| Error::CorruptPayload("validity length too large"))?;
        let end = pos.checked_add(vlen).ok_or(Error::Truncated)?;
        let vblob = src.get(pos..end).ok_or(Error::Truncated)?;
        pos = end;
        Some(crate::validity::decode(vblob, logical_n)?)
    } else {
        None
    };
    let n_valid = match &validity {
        Some(bm) => crate::validity::count_valid(bm, logical_n),
        None => logical_n,
    };
    let mut limbs = Vec::with_capacity(n_limbs);
    for _ in 0..n_limbs {
        let sublen = usize::try_from(crate::varint::read_u64(src, &mut pos)?)
            .map_err(|_| Error::CorruptPayload("decimal sub-stream length"))?;
        let end = pos.checked_add(sublen).ok_or(Error::Truncated)?;
        let sub = src.get(pos..end).ok_or(Error::Truncated)?;
        pos = end;
        let (_dt, lane, _val) = decompress_lane(sub)?;
        if lane.len() != n_valid {
            return Err(Error::CorruptPayload("decimal limb length"));
        }
        limbs.push(lane);
    }
    Ok(Shell {
        scale,
        precision,
        logical_n,
        n_valid,
        vmin_off,
        n_limbs,
        limbs,
        validity,
    })
}

// ---- Decimal128 ---------------------------------------------------------------

const N_LIMBS_128: usize = 2;

pub(crate) fn compress128(
    values: &[i128],
    scale: i8,
    precision: u8,
    validity: Option<&[u8]>,
    cfg: Config,
) -> Vec<u8> {
    let logical_n = values.len();
    let has_nulls = validity.is_some_and(|bm| crate::validity::count_valid(bm, logical_n) < logical_n);
    let valid: Vec<i128> = if has_nulls {
        let bm = validity.unwrap();
        (0..logical_n)
            .filter(|&i| crate::validity::is_set(bm, i))
            .map(|i| values[i])
            .collect()
    } else {
        values.to_vec()
    };
    let vmin = valid.iter().copied().min().unwrap_or(0);

    let mut limbs: [Vec<u64>; N_LIMBS_128] =
        [Vec::with_capacity(valid.len()), Vec::with_capacity(valid.len())];
    for &v in &valid {
        let off = v.wrapping_sub(vmin) as u128;
        limbs[0].push(off as u64);
        limbs[1].push((off >> 64) as u64);
    }
    assemble(
        DType::Decimal128,
        logical_n,
        scale,
        precision,
        &vmin.to_le_bytes(),
        validity,
        &limbs,
        valid.len(),
        cfg,
    )
}

pub(crate) fn decompress128(src: &[u8]) -> Result<Decoded128, Error> {
    let shell = parse_shell(src, DType::Decimal128)?;
    if shell.n_limbs != N_LIMBS_128 {
        return Err(Error::CorruptPayload("decimal128 limb count"));
    }
    let vmin = i128::from_le_bytes(
        src[shell.vmin_off..shell.vmin_off + 16]
            .try_into()
            .map_err(|_| Error::Truncated)?,
    );
    let mut valid = Vec::with_capacity(shell.n_valid);
    for i in 0..shell.n_valid {
        let off = u128::from(shell.limbs[0][i]) | (u128::from(shell.limbs[1][i]) << 64);
        valid.push(vmin.wrapping_add(off as i128));
    }
    let values = match &shell.validity {
        Some(bm) => scatter_i128(&valid, bm, shell.logical_n)?,
        None => valid,
    };
    Ok(Decoded128 {
        values,
        scale: shell.scale,
        precision: shell.precision,
        validity: shell.validity,
    })
}

/// Scatter compacted valid values back to the logical length; null slots → 0.
fn scatter_i128(valid: &[i128], bitmap: &[u8], n: usize) -> Result<Vec<i128>, Error> {
    if crate::validity::count_valid(bitmap, n) != valid.len() {
        return Err(Error::CorruptPayload("decimal validity/value count mismatch"));
    }
    let mut out = Vec::with_capacity(n);
    let mut k = 0;
    for i in 0..n {
        if crate::validity::is_set(bitmap, i) {
            out.push(valid[k]);
            k += 1;
        } else {
            out.push(0);
        }
    }
    Ok(out)
}

// ---- Decimal256 ---------------------------------------------------------------
//
// Values are little-endian two's-complement `[u8; 32]`. We work on `[u64; 4]`
// limbs (limb 0 = least significant). Only signed-min, subtract and add are
// needed, all implemented here so the core stays free of any 256-bit dependency.

const N_LIMBS_256: usize = 4;

type Limbs4 = [u64; 4];

fn to_limbs(b: &[u8; 32]) -> Limbs4 {
    let mut l = [0u64; 4];
    for (i, limb) in l.iter_mut().enumerate() {
        *limb = u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap());
    }
    l
}

fn from_limbs(l: &Limbs4) -> [u8; 32] {
    let mut b = [0u8; 32];
    for (i, limb) in l.iter().enumerate() {
        b[i * 8..i * 8 + 8].copy_from_slice(&limb.to_le_bytes());
    }
    b
}

/// Signed (two's-complement) `a < b` over 256 bits.
fn lt_signed(a: &Limbs4, b: &Limbs4) -> bool {
    // Sign bit lives in the top limb; a negative number is "less" than a
    // non-negative one. With equal signs the unsigned ordering of the limbs
    // (high to low) matches the signed ordering.
    let sa = a[3] >> 63;
    let sb = b[3] >> 63;
    if sa != sb {
        return sa == 1; // a negative, b non-negative
    }
    for i in (0..4).rev() {
        if a[i] != b[i] {
            return a[i] < b[i];
        }
    }
    false
}

/// `a - b` mod 2^256 (wrapping). For `a >= b` this is the exact difference.
fn sub_limbs(a: &Limbs4, b: &Limbs4) -> Limbs4 {
    // Limb-wise subtract with borrow. Uses `overflowing_sub` rather than a u128
    // intermediate so the wrapping case (`a < b`, e.g. two's-complement operands)
    // doesn't trip debug overflow checks.
    let mut out = [0u64; 4];
    let mut borrow = false;
    for i in 0..4 {
        let (d1, b1) = a[i].overflowing_sub(b[i]);
        let (d2, b2) = d1.overflowing_sub(borrow as u64);
        out[i] = d2;
        borrow = b1 || b2;
    }
    out
}

/// `a + b` mod 2^256 (wrapping).
fn add_limbs(a: &Limbs4, b: &Limbs4) -> Limbs4 {
    let mut out = [0u64; 4];
    let mut carry = 0u128;
    for i in 0..4 {
        let cur = u128::from(a[i]) + u128::from(b[i]) + carry;
        out[i] = cur as u64;
        carry = cur >> 64;
    }
    out
}

/// A decoded `Decimal256` column.
pub(crate) struct Decoded256 {
    pub values: Vec<[u8; 32]>,
    pub scale: i8,
    pub precision: u8,
    pub validity: Option<Vec<u8>>,
}

pub(crate) fn compress256(
    values: &[[u8; 32]],
    scale: i8,
    precision: u8,
    validity: Option<&[u8]>,
    cfg: Config,
) -> Vec<u8> {
    let logical_n = values.len();
    let has_nulls = validity.is_some_and(|bm| crate::validity::count_valid(bm, logical_n) < logical_n);
    let valid: Vec<Limbs4> = if has_nulls {
        let bm = validity.unwrap();
        (0..logical_n)
            .filter(|&i| crate::validity::is_set(bm, i))
            .map(|i| to_limbs(&values[i]))
            .collect()
    } else {
        values.iter().map(to_limbs).collect()
    };
    let vmin = valid.iter().copied().reduce(|m, v| if lt_signed(&v, &m) { v } else { m }).unwrap_or([0; 4]);

    let mut limbs: [Vec<u64>; N_LIMBS_256] = Default::default();
    for v in &valid {
        let off = sub_limbs(v, &vmin); // >= 0
        for (j, limb) in limbs.iter_mut().enumerate() {
            limb.push(off[j]);
        }
    }
    assemble(
        DType::Decimal256,
        logical_n,
        scale,
        precision,
        &from_limbs(&vmin),
        validity,
        &limbs,
        valid.len(),
        cfg,
    )
}

pub(crate) fn decompress256(src: &[u8]) -> Result<Decoded256, Error> {
    let shell = parse_shell(src, DType::Decimal256)?;
    if shell.n_limbs != N_LIMBS_256 {
        return Err(Error::CorruptPayload("decimal256 limb count"));
    }
    let vmin_bytes: [u8; 32] = src[shell.vmin_off..shell.vmin_off + 32]
        .try_into()
        .map_err(|_| Error::Truncated)?;
    let vmin = to_limbs(&vmin_bytes);
    let mut valid = Vec::with_capacity(shell.n_valid);
    for i in 0..shell.n_valid {
        let off = [
            shell.limbs[0][i],
            shell.limbs[1][i],
            shell.limbs[2][i],
            shell.limbs[3][i],
        ];
        valid.push(from_limbs(&add_limbs(&vmin, &off)));
    }
    let values = match &shell.validity {
        Some(bm) => scatter_i256(&valid, bm, shell.logical_n)?,
        None => valid,
    };
    Ok(Decoded256 {
        values,
        scale: shell.scale,
        precision: shell.precision,
        validity: shell.validity,
    })
}

fn scatter_i256(valid: &[[u8; 32]], bitmap: &[u8], n: usize) -> Result<Vec<[u8; 32]>, Error> {
    if crate::validity::count_valid(bitmap, n) != valid.len() {
        return Err(Error::CorruptPayload("decimal validity/value count mismatch"));
    }
    let mut out = Vec::with_capacity(n);
    let mut k = 0;
    for i in 0..n {
        if crate::validity::is_set(bitmap, i) {
            out.push(valid[k]);
            k += 1;
        } else {
            out.push([0u8; 32]);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rt(values: &[i128], scale: i8, precision: u8) -> usize {
        let packed = compress128(values, scale, precision, None, Config::default());
        let dec = decompress128(&packed).unwrap();
        assert_eq!(dec.values, values);
        assert_eq!(dec.scale, scale);
        assert_eq!(dec.precision, precision);
        packed.len()
    }

    #[test]
    fn small_decimals_compress() {
        // Prices in cents: clustered, small range → low limb FoR, high limb const.
        let vals: Vec<i128> = (0..4096).map(|i| 100_000 + (i % 500) as i128).collect();
        let size = rt(&vals, 2, 18);
        assert!(size < vals.len() * 4, "clustered decimals should compress: {size}");
    }

    #[test]
    fn mixed_sign_and_vmin_shift() {
        let vals: Vec<i128> = (0..2000).map(|i| (i as i128 - 1000) * 7).collect();
        rt(&vals, 4, 20);
    }

    #[test]
    fn wide_128bit_spread_is_lossless() {
        // Genuinely 128-bit values: the high limb carries real data.
        let vals: Vec<i128> = (0..1000)
            .map(|i| ((i as i128) << 90) | ((i as i128) * 0x1_0000_0001))
            .collect();
        rt(&vals, 0, 38);
        // extremes
        rt(&[i128::MIN, 0, i128::MAX, -1, 1], 0, 38);
    }

    #[test]
    fn nullable_roundtrip() {
        let logical: Vec<i128> = (0..500).map(|i| 50_000 - (i % 300) as i128).collect();
        // every 3rd is null
        let mut bm = vec![0u8; logical.len().div_ceil(8)];
        let mut expect = logical.clone();
        for (i, e) in expect.iter_mut().enumerate() {
            if i % 3 == 0 {
                *e = 0; // null slot → 0
            } else {
                bm[i >> 3] |= 1 << (i & 7);
            }
        }
        let packed = compress128(&logical, 2, 18, Some(&bm), Config::default());
        let dec = decompress128(&packed).unwrap();
        assert_eq!(dec.values, expect);
        assert!(dec.validity.is_some());
    }

    #[test]
    fn empty_and_single() {
        rt(&[], 0, 10);
        rt(&[42], 3, 12);
        rt(&[-123_456_789_012_345_678i128], 6, 30);
    }

    // ---- Decimal256 ----

    /// Sign-extend an `i128` to a little-endian 256-bit two's-complement value.
    fn b256(v: i128) -> [u8; 32] {
        let mut b = [if v < 0 { 0xFF } else { 0x00 }; 32];
        b[0..16].copy_from_slice(&v.to_le_bytes());
        b
    }

    fn rt256(values: &[[u8; 32]], scale: i8, precision: u8) -> usize {
        let packed = compress256(values, scale, precision, None, Config::default());
        assert!(is_decimal_stream(&packed));
        let dec = decompress256(&packed).unwrap();
        assert_eq!(dec.values, values);
        assert_eq!((dec.scale, dec.precision), (scale, precision));
        packed.len()
    }

    #[test]
    fn decimal256_small_and_signed() {
        let vals: Vec<[u8; 32]> = (0..4096).map(|i| b256(1_000_000 + (i % 500) as i128)).collect();
        let size = rt256(&vals, 2, 40);
        assert!(size < vals.len() * 8, "clustered dec256 should compress: {size}");
        // mixed sign through the vmin shift
        let mixed: Vec<[u8; 32]> = (0..2000).map(|i| b256((i as i128 - 1000) * 7)).collect();
        rt256(&mixed, 4, 50);
    }

    #[test]
    fn decimal256_full_width_lossless() {
        // Genuine 256-bit magnitudes: set high limbs.
        let mut vals = Vec::new();
        for i in 0..500u64 {
            let mut b = [0u8; 32];
            b[24..32].copy_from_slice(&i.to_le_bytes()); // top limb
            b[0..8].copy_from_slice(&(i.wrapping_mul(0x9E37_79B9)).to_le_bytes());
            vals.push(b);
        }
        rt256(&vals, 0, 70);
        // extremes: i256::MIN, i256::MAX, -1, 0, 1
        let max = [0xFFu8; 32];
        let mut imax = [0xFFu8; 32];
        imax[31] = 0x7F; // 2^255 - 1
        let mut imin = [0x00u8; 32];
        imin[31] = 0x80; // -2^255
        let neg1 = max; // all-ones = -1
        rt256(&[imin, [0u8; 32], imax, neg1, b256(1)], 0, 76);
    }

    #[test]
    fn decimal256_nullable() {
        let logical: Vec<[u8; 32]> = (0..500).map(|i| b256(50_000 - (i % 300) as i128)).collect();
        let mut bm = vec![0u8; logical.len().div_ceil(8)];
        let mut expect = logical.clone();
        for (i, e) in expect.iter_mut().enumerate() {
            if i % 4 == 0 {
                *e = [0u8; 32];
            } else {
                bm[i >> 3] |= 1 << (i & 7);
            }
        }
        let packed = compress256(&logical, 2, 40, Some(&bm), Config::default());
        let dec = decompress256(&packed).unwrap();
        assert_eq!(dec.values, expect);
        assert!(dec.validity.is_some());
    }

    #[test]
    fn decimal256_arithmetic_helpers() {
        // add/sub are inverse; lt_signed matches i128 ordering on small values.
        for &(a, b) in &[(5i128, 3i128), (-5, 3), (3, -5), (-7, -2), (0, 0)] {
            let (la, lb) = (to_limbs(&b256(a)), to_limbs(&b256(b)));
            assert_eq!(lt_signed(&la, &lb), a < b, "lt {a} {b}");
            assert_eq!(from_limbs(&add_limbs(&la, &lb)), b256(a + b));
            assert_eq!(from_limbs(&sub_limbs(&la, &lb)), b256(a - b));
        }
    }
}
