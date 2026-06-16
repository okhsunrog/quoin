//! ALP — Adaptive Lossless floating-Point (the main scheme), ported from the
//! CWI reference (SIGMOD'24). For decimal-like doubles it represents each value
//! as a scaled integer `digit = round(v · 10^e · 10^-f)` and stores the digits
//! via frame-of-reference + FastLanes bit-packing. Values that don't round-trip
//! (and NaN/inf) are stored verbatim as **exceptions**, so a few outliers don't
//! force the whole block to raw — the key robustness win over [`super::float_mult`].
//!
//! Decoding is exact: encode verifies `decode(encode(v)) == v` bit-for-bit per
//! value (else it's an exception), and the decoder recomputes the same product.
//!
//! Not yet implemented: ALP-RD (the "real doubles" split-dictionary scheme for
//! non-decimal floats) — captured in `docs/LANDSCAPE.md` as a follow-up.

use crate::bitpack::{self, BLOCK};
use crate::error::Error;
use crate::varint;

const MAX_EXP: usize = 18;
/// 1.5 · 2^52 — the round-to-nearest-int magic constant (ALP `MAGIC_NUMBER`).
const MAGIC: f64 = 6_755_399_441_055_744.0;
const UPPER: f64 = 9.223_372_036_854_776e18; // ~2^63, ALP's encoding limit

static EXP10: [f64; MAX_EXP + 1] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16,
    1e17, 1e18,
];
static FRAC10: [f64; MAX_EXP + 1] = [
    1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14,
    1e-15, 1e-16, 1e-17, 1e-18,
];

#[inline]
fn encode_value(v: f64, e: usize, f: usize) -> Option<i64> {
    let tmp = v * EXP10[e] * FRAC10[f];
    if !tmp.is_finite() || tmp >= UPPER || tmp <= -UPPER {
        return None;
    }
    Some((tmp + MAGIC - MAGIC) as i64)
}

#[inline]
fn decode_value(digit: i64, e: usize, f: usize) -> f64 {
    (digit as f64) * EXP10[f] * FRAC10[e]
}

/// Pick `(e, f)` minimizing estimated size on a 32-value sample of the sub-block.
/// Returns `None` when even the best fit leaves > half the sample as exceptions
/// (non-decimal data) — lets the mode bail cheaply before a full encode.
fn find_ef(sub: &[u64]) -> Option<(usize, usize)> {
    let n = sub.len();
    let stride = (n / 32).max(1);
    let sample: Vec<f64> = (0..n).step_by(stride).take(32).map(|i| f64::from_bits(sub[i])).collect();

    let mut best = (0usize, 0usize);
    let mut best_cost = usize::MAX;
    let mut best_exc = sample.len();
    for e in 0..=MAX_EXP {
        for f in 0..=e {
            let mut exc = 0usize;
            let (mut lo, mut hi) = (i64::MAX, i64::MIN);
            for &v in &sample {
                match encode_value(v, e, f) {
                    Some(d) if decode_value(d, e, f).to_bits() == v.to_bits() => {
                        lo = lo.min(d);
                        hi = hi.max(d);
                    }
                    _ => exc += 1,
                }
            }
            let width = if exc == sample.len() || hi <= lo {
                if exc == sample.len() { 64 } else { 0 }
            } else {
                (64 - (hi.wrapping_sub(lo) as u64).leading_zeros()) as usize
            };
            let cost = width * sample.len() + exc * 80; // ~exception bytes in bits
            if cost < best_cost {
                best_cost = cost;
                best = (e, f);
                best_exc = exc;
            }
        }
    }
    if best_exc * 2 > sample.len() {
        return None;
    }
    Some(best)
}

pub(crate) fn encode(vals: &[u64]) -> Option<Vec<u8>> {
    if vals.is_empty() {
        return None;
    }
    let mut out = Vec::with_capacity(vals.len());
    varint::write_u64(&mut out, vals.len() as u64);
    let mut i = 0;
    while i < vals.len() {
        let end = (i + BLOCK).min(vals.len());
        encode_subblock(&vals[i..end], &mut out)?;
        i = end;
    }
    Some(out)
}

fn encode_subblock(sub: &[u64], out: &mut Vec<u8>) -> Option<()> {
    let (e, f) = find_ef(sub)?;

    let mut digits = [0i64; BLOCK];
    let mut exceptions: Vec<(u16, u64)> = Vec::new();
    let mut first_valid: Option<i64> = None;
    for (k, &bits) in sub.iter().enumerate() {
        let v = f64::from_bits(bits);
        match encode_value(v, e, f) {
            Some(d) if decode_value(d, e, f).to_bits() == bits => {
                digits[k] = d;
                first_valid.get_or_insert(d);
            }
            _ => exceptions.push((k as u16, bits)),
        }
    }
    // Exceptions (and padding) get a filler digit so they don't widen the range.
    let filler = first_valid.unwrap_or(0);
    for &(pos, _) in &exceptions {
        digits[pos as usize] = filler;
    }
    for d in digits.iter_mut().take(BLOCK).skip(sub.len()) {
        *d = filler;
    }

    let min = *digits.iter().min().unwrap();
    let max = *digits.iter().max().unwrap();
    let range = max.wrapping_sub(min) as u64;
    let width = if range == 0 { 0 } else { 64 - range.leading_zeros() };
    if width > 32 {
        return None; // digits too spread for the u32 packer — ALP-RD territory
    }

    out.push(e as u8);
    out.push(f as u8);
    out.extend_from_slice(&(exceptions.len() as u16).to_le_bytes());
    out.extend_from_slice(&min.to_le_bytes());
    out.push(width as u8);
    if width > 0 {
        let mut residuals = [0u32; BLOCK];
        for (r, &d) in residuals.iter_mut().zip(digits.iter()) {
            *r = d.wrapping_sub(min) as u32;
        }
        let mut packed = vec![0u32; 32 * width as usize];
        bitpack::pack(&residuals, width, &mut packed);
        for w in &packed {
            out.extend_from_slice(&w.to_le_bytes());
        }
    }
    for &(pos, bits) in &exceptions {
        out.extend_from_slice(&pos.to_le_bytes());
        out.extend_from_slice(&bits.to_le_bytes());
    }
    Some(())
}

pub(crate) fn decode(payload: &[u8], n_values: usize) -> Result<Vec<u64>, Error> {
    let mut pos = 0usize;
    let n = varint::read_u64(payload, &mut pos)? as usize;
    if n != n_values {
        return Err(Error::CorruptPayload("alp length mismatch"));
    }
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let e = *payload.get(pos).ok_or(Error::Truncated)? as usize;
        let f = *payload.get(pos + 1).ok_or(Error::Truncated)? as usize;
        if e > MAX_EXP || f > e {
            return Err(Error::CorruptPayload("alp exponent/factor"));
        }
        let exc_count =
            u16::from_le_bytes(payload.get(pos + 2..pos + 4).ok_or(Error::Truncated)?.try_into().unwrap())
                as usize;
        let min = i64::from_le_bytes(
            payload.get(pos + 4..pos + 12).ok_or(Error::Truncated)?.try_into().unwrap(),
        );
        let width = *payload.get(pos + 12).ok_or(Error::Truncated)?;
        pos += 13;
        if width > 32 {
            return Err(Error::CorruptPayload("alp width"));
        }
        let count = (n - out.len()).min(BLOCK);

        let mut digits = [0i64; BLOCK];
        if width > 0 {
            let nwords = 32 * width as usize;
            let pb = payload.get(pos..pos + nwords * 4).ok_or(Error::Truncated)?;
            pos += nwords * 4;
            let mut packed = vec![0u32; nwords];
            for (k, c) in pb.chunks_exact(4).enumerate() {
                packed[k] = u32::from_le_bytes(c.try_into().unwrap());
            }
            let mut residuals = [0u32; BLOCK];
            bitpack::unpack(&packed, u32::from(width), &mut residuals);
            for (d, &r) in digits.iter_mut().zip(residuals.iter()) {
                *d = min.wrapping_add(i64::from(r));
            }
        } else {
            digits.fill(min);
        }

        let start = out.len();
        for &d in digits.iter().take(count) {
            out.push(decode_value(d, e, f).to_bits());
        }
        // Patch exceptions over the decoded digits.
        for _ in 0..exc_count {
            let p = u16::from_le_bytes(
                payload.get(pos..pos + 2).ok_or(Error::Truncated)?.try_into().unwrap(),
            ) as usize;
            let bits = u64::from_le_bytes(
                payload.get(pos + 2..pos + 10).ok_or(Error::Truncated)?.try_into().unwrap(),
            );
            pos += 10;
            if p >= count {
                return Err(Error::CorruptPayload("alp exception position"));
            }
            out[start + p] = bits;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(vals: &[u64]) {
        if let Some(enc) = encode(vals) {
            let dec = decode(&enc, vals.len()).unwrap();
            assert_eq!(dec, vals);
        }
    }

    #[test]
    fn decimal_with_exceptions() {
        // cent-rounded prices with a few NaN/huge outliers → exceptions.
        let mut v: Vec<u64> =
            (0..5000).map(|i| (100.0_f64 + (i % 700) as f64 / 100.0).to_bits()).collect();
        v[10] = f64::NAN.to_bits();
        v[2000] = 1e300_f64.to_bits();
        v[4999] = f64::INFINITY.to_bits();
        roundtrip(&v);

        roundtrip(&(0..3000).map(|i| ((i % 100) as f64 * 0.25).to_bits()).collect::<Vec<_>>());
        roundtrip(&[]);
        roundtrip(&[3.14159_f64.to_bits()]);
        roundtrip(&vec![0.0f64.to_bits(); 2000]);
    }
}
