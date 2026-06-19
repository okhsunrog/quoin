//! ALP-RD: the "real doubles" scheme for floats that aren't decimals (ALP,
//! SIGMOD'24). Where [`super::alp`] fails (coordinates, scientific reals), the
//! high bits of the IEEE-754 pattern still cluster into a handful of distinct
//! values while the low bits are high-entropy.
//!
//! Each value is split at a cut point into a `left` (high) and `right` (low)
//! part. The `left` part is **dictionary**-coded — the top [`MAX_DICT`] distinct
//! values get a small code, the rest become **exceptions** — and the `right`
//! part is **bit-packed** (the wide ~48-bit packing uses the `u64` FastLanes
//! kernel). The cut is chosen by estimating the encoded size on a sample.

use crate::codecs::for_bitpack;
use crate::entropy::{code_residuals, decode_residuals};
use crate::error::Error;
use crate::varint;

// Per-stream encoding tag (codes / rights). The streams used to be raw bit-packed
// only; at the entropy levels they can instead be entropy-coded (codes are
// ≤MAX_DICT-cardinality, rights have residual structure), chosen by size.
const T_BITPACK: u8 = 0;
const T_ENT: u8 = 1;
// FxHashMap (non-cryptographic) instead of std's SipHash map: the keys are
// integer `left`-parts and cardinality is tiny, so SipHash was ~30% of ALP-RD
// encode time (profiled). FxHash collisions can't affect correctness here.
use rustc_hash::FxHashMap as HashMap;

/// Dictionary capacity for the `left` parts (matches the ALP reference). With ≤8
/// entries the code is ≤3 bits.
const MAX_DICT: usize = 8;
/// Largest `left` (cut) bit width searched. The reference caps at 16; we search
/// a little deeper and let the size estimate pick, since a deeper cut sometimes
/// pays off and the estimate rejects it when exceptions explode.
const CUT_MAX: u32 = 24;
/// Approximate bits charged per exception in the estimate: varint position (~32)
/// + the `u64` left value (64).
const EXC_BITS: u64 = 96;

#[inline]
fn right_mask(right_bw: u32) -> u64 {
    if right_bw >= 64 {
        u64::MAX
    } else {
        (1u64 << right_bw) - 1
    }
}

/// Frequency of each `left` part over an iterator of values, given `right_bw`.
fn left_freq<'a>(vals: impl Iterator<Item = &'a u64>, right_bw: u32) -> HashMap<u64, u32> {
    let mut freq: HashMap<u64, u32> = HashMap::default();
    for &v in vals {
        *freq.entry(v >> right_bw).or_insert(0) += 1;
    }
    freq
}

/// The top-`MAX_DICT` `left` values by frequency, plus how many sampled values
/// they cover.
fn top_dict(freq: &HashMap<u64, u32>) -> (Vec<u64>, u64) {
    let mut by_freq: Vec<(u64, u32)> = freq.iter().map(|(&k, &c)| (k, c)).collect();
    by_freq.sort_unstable_by_key(|&(k, c)| (std::cmp::Reverse(c), k));
    let dict: Vec<u64> = by_freq.iter().take(MAX_DICT).map(|&(k, _)| k).collect();
    let covered: u64 = by_freq
        .iter()
        .take(MAX_DICT)
        .map(|&(_, c)| u64::from(c))
        .sum();
    (dict, covered)
}

#[inline]
fn code_bits(dict_len: usize) -> u32 {
    match dict_len {
        0 | 1 => 1,
        n => (usize::BITS - (n - 1).leading_zeros()).max(1),
    }
}

/// Choose the `left` (cut) bit width minimizing estimated size on a sample, or
/// `None` if no cut beats storing the values verbatim.
fn choose_cut(vals: &[u64]) -> Option<u32> {
    let n = vals.len();
    let step = (n / 2048).max(1);
    let sample: Vec<u64> = vals.iter().step_by(step).copied().collect();
    let sn = sample.len() as u64;

    let mut best_cut = 0u32;
    let mut best_bits = f64::INFINITY;
    for left_bw in 1..=CUT_MAX {
        let right_bw = 64 - left_bw;
        let freq = left_freq(sample.iter(), right_bw);
        let (dict, covered) = top_dict(&freq);
        let exceptions = sn - covered;
        let per_value = (sn * u64::from(right_bw + code_bits(dict.len())) + exceptions * EXC_BITS)
            as f64
            / sn as f64;
        if per_value < best_bits {
            best_bits = per_value;
            best_cut = left_bw;
        }
    }
    // Must beat verbatim storage (64 bits) with margin for the dict/headers.
    if best_cut == 0 || best_bits >= 63.0 {
        None
    } else {
        Some(best_cut)
    }
}

/// Diagnostic (cascade-lab): the raw `(codes, rights)` integer streams for a
/// block, before they're bit-packed — so the lab can measure whether entropy-
/// coding them beats the current raw `for_bitpack`. Mirrors `encode`'s split.
pub(crate) fn debug_streams(vals: &[u64]) -> Option<(Vec<u64>, Vec<u64>)> {
    if vals.is_empty() {
        return None;
    }
    let left_bw = choose_cut(vals)?;
    let right_bw = 64 - left_bw;
    let mask = right_mask(right_bw);
    let freq = left_freq(vals.iter(), right_bw);
    let (dict, _) = top_dict(&freq);
    let code_of: HashMap<u64, u32> =
        dict.iter().enumerate().map(|(i, &v)| (v, i as u32)).collect();
    let mut codes = Vec::with_capacity(vals.len());
    let mut rights = Vec::with_capacity(vals.len());
    for &v in vals {
        rights.push(v & mask);
        codes.push(u64::from(code_of.get(&(v >> right_bw)).copied().unwrap_or(0)));
    }
    Some((codes, rights))
}

pub(crate) fn encode(vals: &[u64], entropy: bool, lambda: u64, allow_lz: bool) -> Option<Vec<u8>> {
    if vals.is_empty() {
        return None;
    }
    let left_bw = choose_cut(vals)?;
    let right_bw = 64 - left_bw;
    let mask = right_mask(right_bw);

    // Build the dictionary over the full block for an accurate top-MAX_DICT.
    let freq = left_freq(vals.iter(), right_bw);
    let (dict, _) = top_dict(&freq);
    let code_of: HashMap<u64, u32> = dict
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i as u32))
        .collect();

    let mut codes = Vec::with_capacity(vals.len());
    let mut rights = Vec::with_capacity(vals.len());
    let mut exc_pos = Vec::new();
    let mut exc_left = Vec::new();
    for (i, &v) in vals.iter().enumerate() {
        let left = v >> right_bw;
        rights.push(v & mask);
        if let Some(&c) = code_of.get(&left) {
            codes.push(u64::from(c));
        } else {
            codes.push(0); // placeholder; patched from the exception list
            exc_pos.push(i as u64);
            exc_left.push(left);
        }
    }
    // Too many exceptions → the dictionary isn't capturing the data; bail so
    // another mode wins.
    if exc_pos.len() * 4 > vals.len() {
        return None;
    }

    let mut out = Vec::with_capacity(vals.len() + 64);
    out.push(left_bw as u8);
    out.push(dict.len() as u8);
    for &d in &dict {
        out.extend_from_slice(&d.to_le_bytes());
    }
    varint::write_u64(&mut out, exc_pos.len() as u64);
    for (&p, &l) in exc_pos.iter().zip(&exc_left) {
        varint::write_u64(&mut out, p);
        out.extend_from_slice(&l.to_le_bytes());
    }
    // Cascade each stream through the entropy coder at the entropy levels; raw
    // bit-pack stays the fast-decode path for Fast/Fastest. Codes are
    // ≤MAX_DICT-cardinality → entropy the raw code bytes; rights are wide → entropy
    // the bit-packed blob. Keep whichever is smaller.
    let codes_bp = for_bitpack::encode(&codes, false);
    let (codes_tag, codes_blob) = if entropy {
        let code_bytes: Vec<u8> = codes.iter().map(|&c| c as u8).collect();
        let ent = code_residuals(&code_bytes, lambda, allow_lz);
        if ent.len() < codes_bp.len() { (T_ENT, ent) } else { (T_BITPACK, codes_bp) }
    } else {
        (T_BITPACK, codes_bp)
    };
    let rights_bp = for_bitpack::encode(&rights, false);
    let (rights_tag, rights_blob) = if entropy {
        let ent = code_residuals(&rights_bp, lambda, allow_lz);
        if ent.len() < rights_bp.len() { (T_ENT, ent) } else { (T_BITPACK, rights_bp) }
    } else {
        (T_BITPACK, rights_bp)
    };
    out.push(codes_tag);
    varint::write_u64(&mut out, codes_blob.len() as u64);
    out.extend_from_slice(&codes_blob);
    out.push(rights_tag);
    out.extend_from_slice(&rights_blob);
    Some(out)
}

pub(crate) fn decode(payload: &[u8], n: usize) -> Result<Vec<u64>, Error> {
    let mut pos = 0usize;
    let left_bw = u32::from(*payload.get(pos).ok_or(Error::Truncated)?);
    pos += 1;
    if !(1..=63).contains(&left_bw) {
        return Err(Error::CorruptPayload("alp_rd left_bw out of range"));
    }
    let right_bw = 64 - left_bw;
    let dict_size = usize::from(*payload.get(pos).ok_or(Error::Truncated)?);
    pos += 1;
    if dict_size > MAX_DICT {
        return Err(Error::CorruptPayload("alp_rd dict too large"));
    }
    let mut dict = Vec::with_capacity(dict_size);
    for _ in 0..dict_size {
        let b = payload.get(pos..pos + 8).ok_or(Error::Truncated)?;
        dict.push(u64::from_le_bytes(b.try_into().unwrap()));
        pos += 8;
    }

    let n_exc = varint::read_u64(payload, &mut pos)? as usize;
    if n_exc > n {
        return Err(Error::CorruptPayload("alp_rd too many exceptions"));
    }
    let mut exceptions = Vec::with_capacity(n_exc);
    for _ in 0..n_exc {
        let p = varint::read_u64(payload, &mut pos)? as usize;
        let b = payload.get(pos..pos + 8).ok_or(Error::Truncated)?;
        let l = u64::from_le_bytes(b.try_into().unwrap());
        pos += 8;
        if p >= n {
            return Err(Error::CorruptPayload("alp_rd exception position"));
        }
        exceptions.push((p, l));
    }

    let codes_tag = *payload.get(pos).ok_or(Error::Truncated)?;
    pos += 1;
    let codes_len = varint::read_u64(payload, &mut pos)? as usize;
    let codes_blob = payload.get(pos..pos + codes_len).ok_or(Error::Truncated)?;
    pos += codes_len;
    let rights_tag = *payload.get(pos).ok_or(Error::Truncated)?;
    pos += 1;
    let rights_blob = payload.get(pos..).ok_or(Error::Truncated)?;

    let codes = match codes_tag {
        T_BITPACK => for_bitpack::decode(codes_blob, n, false)?,
        T_ENT => {
            let bytes = decode_residuals(codes_blob, n)?;
            if bytes.len() != n {
                return Err(Error::CorruptPayload("alp_rd codes length"));
            }
            bytes.into_iter().map(u64::from).collect()
        }
        _ => return Err(Error::CorruptPayload("alp_rd codes tag")),
    };
    let rights = match rights_tag {
        T_BITPACK => for_bitpack::decode(rights_blob, n, false)?,
        T_ENT => {
            let bp = decode_residuals(rights_blob, n.saturating_mul(8) + 64)?;
            for_bitpack::decode(&bp, n, false)?
        }
        _ => return Err(Error::CorruptPayload("alp_rd rights tag")),
    };

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let code = codes[i] as usize;
        if code >= dict.len() {
            return Err(Error::CorruptPayload("alp_rd code out of range"));
        }
        out.push((dict[code] << right_bw) | rights[i]);
    }
    for (p, l) in exceptions {
        out[p] = (l << right_bw) | rights[p];
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(vals: &[u64]) -> Option<usize> {
        // Exercise both the raw bit-pack path (Fast) and the entropy cascade
        // (Balanced rANS + Max range-coder/LZ); all must round-trip exactly.
        let raw = encode(vals, false, 0, false)?;
        assert_eq!(decode(&raw, vals.len()).unwrap(), vals);
        let bal = encode(vals, true, 2, false)?;
        assert_eq!(decode(&bal, vals.len()).unwrap(), vals);
        let max = encode(vals, true, 0, true)?;
        assert_eq!(decode(&max, vals.len()).unwrap(), vals);
        Some(raw.len())
    }

    #[test]
    fn real_double_like_packs() {
        // Coordinates: a few distinct high parts (sign+exp+top mantissa), random
        // low mantissa bits — the ALP-RD target.
        let mut s = 0x1234_5678u64;
        let base = 40.0_f64.to_bits() & 0xFFFF_0000_0000_0000; // fixed high bits
        let vals: Vec<u64> = (0..8192)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                base | (s >> 20) // vary the low ~44 bits
            })
            .collect();
        let size = roundtrip(&vals).expect("should encode");
        assert!(
            size < vals.len() * 8,
            "ALP-RD must beat raw 8 B/value, got {}",
            size as f64 / vals.len() as f64
        );
    }

    #[test]
    fn exact_roundtrip_edges() {
        // Mixed: exceptions (rare high parts) must round-trip exactly.
        let mut v: Vec<u64> = (0..2000).map(|i| 0x4045_0000_0000_0000 | i).collect();
        v.push(0xDEAD_BEEF_CAFE_1234); // outlier -> exception
        v.push(0x0000_0000_0000_0001);
        let _ = roundtrip(&v); // may or may not pick ALP-RD; must be exact if it does
    }
}
