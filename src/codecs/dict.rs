//! DICT: dictionary-encode low-cardinality columns. Distinct values map to small
//! integer codes; the dictionary holds the distinct values verbatim. Wins on
//! columns with few distinct values whose repeats are scattered (where RLE's
//! runs don't help). Type-agnostic — operates on the raw `u64` lane.
//!
//! Two independent streams are each stored in whichever form is smallest:
//!
//! * **codes** — [`CODES_BITPACK`] (FoR + bit-packing, random-access) or
//!   [`CODES_ENTROPY`] (split into byte-planes — 1 for ≤256-cardinality, 2 for
//!   ≤65536 — each entropy-coded, capturing their frequency skew).
//! * **dictionary values** — the distinct values are **sorted** (codes remapped
//!   to the sorted order, free), then stored [`VAL_RAW`], [`VAL_DELTA`]
//!   (delta→bit-pack, exploiting the now-monotonic sequence), or [`VAL_TRANSPOSE`]
//!   (byte-transpose → entropy, exploiting shared high bytes of similar-magnitude
//!   decimals). For high-cardinality columns the raw dictionary dominates the
//!   output, so compressing it is the bigger win.

use crate::codecs::{delta_bitpack, for_bitpack, transpose};
use crate::entropy::{code_residuals, decode_residuals};
use crate::error::Error;
use crate::varint;
use std::collections::HashMap;

const CODES_BITPACK: u8 = 0;
const CODES_ENTROPY: u8 = 1;
/// Largest cardinality the byte-plane entropy cascade handles (2 planes).
const MAX_ENTROPY_CARD: usize = 1 << 16;

const VAL_RAW: u8 = 0;
const VAL_DELTA: u8 = 1;
const VAL_TRANSPOSE: u8 = 2;

/// Number of byte-planes needed to represent codes `0..card`.
fn plane_count(card: usize) -> usize {
    if card <= 256 { 1 } else { 2 }
}

/// Encode the (sorted) dictionary values, smallest of raw / delta→bitpack /
/// transpose→entropy. Returns `(tag, blob)`.
fn encode_values(sorted: &[u64], entropy: bool, lambda: u64, allow_lz: bool) -> (u8, Vec<u8>) {
    let mut raw = Vec::with_capacity(sorted.len() * 8);
    for &d in sorted {
        raw.extend_from_slice(&d.to_le_bytes());
    }
    let mut tag = VAL_RAW;
    let mut blob = raw;

    let delta = delta_bitpack::encode(sorted);
    if delta.len() < blob.len() {
        tag = VAL_DELTA;
        blob = delta;
    }
    if entropy {
        let tr = code_residuals(&transpose::encode(sorted), lambda, allow_lz);
        if tr.len() < blob.len() {
            tag = VAL_TRANSPOSE;
            blob = tr;
        }
    }
    (tag, blob)
}

/// Inverse of [`encode_values`] — reconstruct `card` dictionary values.
fn decode_values(tag: u8, blob: &[u8], card: usize) -> Result<Vec<u64>, Error> {
    match tag {
        VAL_RAW => {
            if blob.len() != card * 8 {
                return Err(Error::CorruptPayload("dict raw values length"));
            }
            Ok(blob
                .chunks_exact(8)
                .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
                .collect())
        }
        VAL_DELTA => delta_bitpack::decode(blob, card),
        VAL_TRANSPOSE => {
            let bytes = decode_residuals(blob, card * 8)?;
            if bytes.len() != card * 8 {
                return Err(Error::CorruptPayload("dict transpose values length"));
            }
            transpose::decode(&bytes, card)
        }
        _ => Err(Error::CorruptPayload("dict value tag")),
    }
}

pub(crate) fn encode(vals: &[u64], entropy: bool, lambda: u64, allow_lz: bool) -> Option<Vec<u8>> {
    if vals.is_empty() {
        return None;
    }
    // Above 50% cardinality the codes approach raw width and the dictionary is
    // huge — not worth it. Bail early to cap the cost on high-distinct blocks.
    let max_card = vals.len() / 2 + 1;
    let mut map: HashMap<u64, u32> = HashMap::new();
    let mut dict: Vec<u64> = Vec::new();
    let mut codes: Vec<u64> = Vec::with_capacity(vals.len());
    for &v in vals {
        let code = *map.entry(v).or_insert_with(|| {
            let c = dict.len() as u32;
            dict.push(v);
            c
        });
        codes.push(u64::from(code));
        if dict.len() > max_card {
            return None;
        }
    }

    // Sort the dictionary ascending and remap the codes to sorted positions — a
    // monotonic dictionary delta- and transpose-compresses far better, and the
    // codes are just a permutation either way.
    let mut order: Vec<u32> = (0..dict.len() as u32).collect();
    order.sort_unstable_by_key(|&i| dict[i as usize]);
    let mut remap = vec![0u32; dict.len()];
    for (new, &old) in order.iter().enumerate() {
        remap[old as usize] = new as u32;
    }
    let sorted: Vec<u64> = order.iter().map(|&old| dict[old as usize]).collect();
    for c in codes.iter_mut() {
        *c = u64::from(remap[*c as usize]);
    }

    let (val_tag, val_blob) = encode_values(&sorted, entropy, lambda, allow_lz);

    // Pick the smaller code representation: bit-packed (random-access) or, when
    // the level allows entropy, byte-plane entropy-coded.
    let mut code_tag = CODES_BITPACK;
    let mut code_blob = for_bitpack::encode(&codes, false);
    if entropy && sorted.len() <= MAX_ENTROPY_CARD {
        let nbytes = plane_count(sorted.len());
        let mut blob = vec![nbytes as u8];
        for p in 0..nbytes {
            let plane: Vec<u8> = codes.iter().map(|&c| (c >> (8 * p)) as u8).collect();
            let coded = code_residuals(&plane, lambda, allow_lz);
            varint::write_u64(&mut blob, coded.len() as u64);
            blob.extend_from_slice(&coded);
        }
        if blob.len() < code_blob.len() {
            code_tag = CODES_ENTROPY;
            code_blob = blob;
        }
    }

    let mut out = Vec::with_capacity(val_blob.len() + code_blob.len() + 16);
    varint::write_u64(&mut out, sorted.len() as u64);
    out.push(val_tag);
    varint::write_u64(&mut out, val_blob.len() as u64);
    out.extend_from_slice(&val_blob);
    out.push(code_tag);
    out.extend_from_slice(&code_blob);
    Some(out)
}

pub(crate) fn decode(payload: &[u8], n: usize) -> Result<Vec<u64>, Error> {
    let mut pos = 0usize;
    let card = varint::read_u64(payload, &mut pos)? as usize;
    if card > n {
        return Err(Error::CorruptPayload(
            "dict cardinality exceeds value count",
        ));
    }
    let val_tag = *payload.get(pos).ok_or(Error::Truncated)?;
    pos += 1;
    let val_len = varint::read_u64(payload, &mut pos)? as usize;
    let val_blob = payload.get(pos..pos + val_len).ok_or(Error::Truncated)?;
    pos += val_len;
    let dict = decode_values(val_tag, val_blob, card)?;

    let tag = *payload.get(pos).ok_or(Error::Truncated)?;
    pos += 1;

    let codes: Vec<u64> = match tag {
        CODES_BITPACK => for_bitpack::decode(&payload[pos..], n, false)?,
        CODES_ENTROPY => {
            let nbytes = usize::from(*payload.get(pos).ok_or(Error::Truncated)?);
            pos += 1;
            if !(1..=2).contains(&nbytes) {
                return Err(Error::CorruptPayload("dict plane count"));
            }
            let mut codes = vec![0u64; n];
            for p in 0..nbytes {
                let len = varint::read_u64(payload, &mut pos)? as usize;
                let blob = payload.get(pos..pos + len).ok_or(Error::Truncated)?;
                pos += len;
                let plane = decode_residuals(blob, n)?;
                if plane.len() != n {
                    return Err(Error::CorruptPayload("dict plane length"));
                }
                for (c, &b) in codes.iter_mut().zip(&plane) {
                    *c |= u64::from(b) << (8 * p);
                }
            }
            codes
        }
        _ => return Err(Error::CorruptPayload("dict code tag")),
    };

    let mut out = Vec::with_capacity(n);
    for c in codes {
        let idx = c as usize;
        if idx >= card {
            return Err(Error::CorruptPayload("dict code out of range"));
        }
        out.push(dict[idx]);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(vals: &[u64]) -> Option<usize> {
        // Exercise both code representations.
        for (entropy, lambda) in [(false, 0u64), (true, 0u64), (true, 4u64)] {
            let enc = encode(vals, entropy, lambda, true)?;
            assert_eq!(decode(&enc, vals.len()).unwrap(), vals);
        }
        Some(encode(vals, true, 0, true)?.len())
    }

    #[test]
    fn low_cardinality_packs() {
        // 16 distinct values scattered across the block (4-bit codes).
        let mut s = 1u64;
        let vals: Vec<u64> = (0..16384)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                1_000_000_000u64 + (s >> 60) // 16 distinct, big values
            })
            .collect();
        let size = roundtrip(&vals).expect("should encode");
        assert!(
            size < vals.len(),
            "16-distinct column should pack to <1 B/value, got {}",
            size as f64 / vals.len() as f64
        );
    }

    #[test]
    fn high_cardinality_bails() {
        let vals: Vec<u64> = (0..10000u64).collect(); // all distinct
        assert!(encode(&vals, true, 0, true).is_none());
    }

    #[test]
    fn entropy_cascade_beats_bitpack_on_skew() {
        // Skewed low-cardinality (8 distinct, very uneven): entropy should win.
        let mut s = 1u64;
        let vals: Vec<u64> = (0..16384)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                // 90% one value, rest spread over 7 others
                if s & 0xF == 0 { 100 + (s >> 60) } else { 42 }
            })
            .collect();
        let bp = encode(&vals, false, 0, false).unwrap().len();
        let ent = encode(&vals, true, 0, true).unwrap().len();
        assert!(
            ent < bp,
            "entropy cascade ({ent}) should beat bitpack ({bp})"
        );
    }

    #[test]
    fn dict_value_compression_helps() {
        // Sorted distinct decimal-like values — the high-cardinality dictionary
        // that dominates the output and must be compressed below raw.
        let sorted: Vec<u64> = (0..20000)
            .map(|i| (1000.0_f64 + i as f64 * 0.01).to_bits())
            .collect();
        let (tag, blob) = encode_values(&sorted, true, 0, true);
        assert_ne!(
            tag, VAL_RAW,
            "sorted decimals should delta/transpose-compress"
        );
        assert!(
            blob.len() < sorted.len() * 8,
            "value compression should beat raw: {} vs {}",
            blob.len(),
            sorted.len() * 8
        );
        assert_eq!(decode_values(tag, &blob, sorted.len()).unwrap(), sorted);

        // And a full high-cardinality round-trip through the codec.
        let mut s = 1u64;
        let vals: Vec<u64> = (0..80000)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                sorted[(s >> 49) as usize % sorted.len()]
            })
            .collect();
        let enc = encode(&vals, true, 0, true).unwrap();
        assert_eq!(decode(&enc, vals.len()).unwrap(), vals);
    }

    #[test]
    fn mid_cardinality_two_planes() {
        // ~2000 distinct values (needs 2 byte-planes) repeated across the block.
        let mut s = 1u64;
        let vals: Vec<u64> = (0..40000)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                500_000_000u64 + (s >> 53) % 2000 // 2000 distinct
            })
            .collect();
        // exercises the 2-plane entropy path and bitpack, both must round-trip
        roundtrip(&vals);
    }

    #[test]
    fn edges() {
        assert!(encode(&[], true, 0, true).is_none());
        roundtrip(&vec![7u64; 1000]);
    }
}
