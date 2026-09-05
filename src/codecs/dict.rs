//! DICT: dictionary-encode low-cardinality columns. Distinct values map to small
//! integer codes; the dictionary holds the distinct values verbatim (one lane
//! word each — 4 bytes on a 32-bit lane). Wins on columns with few distinct
//! values whose repeats are scattered (where RLE's runs don't help).
//! Type-agnostic — operates on the raw lane.
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
//!
//! The **shared** variant ([`build_shared`] / [`encode_shared`]) lifts the value
//! table to a column-wide preamble stored once per stream; `DICT_SHARED` frames
//! hold only the codes section. This wins when the same values recur across
//! blocks: per-block `Dict` pays the dictionary per block (or loses the
//! competition because of it), while the shared table amortizes it — measured
//! +50% ratio on `poi_lat` (repeated-coordinate data, ~100 K distinct across
//! 424 K values). Blocks stay independently decodable given the (read-only)
//! table, so parallel decode is preserved.

use crate::codecs::{delta_bitpack, for_bitpack, transpose};
use crate::entropy::{code_residuals, decode_residuals};
use crate::error::Error;
use crate::lane::Lane;
use crate::varint;
use rustc_hash::FxHashMap;
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
pub(crate) fn encode_values<L: Lane>(
    sorted: &[L],
    entropy: bool,
    lambda: u64,
    allow_lz: bool,
) -> (u8, Vec<u8>) {
    let mut tag = VAL_RAW;
    let mut blob = L::le_bytes(sorted).into_owned();

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
pub(crate) fn decode_values<L: Lane>(tag: u8, blob: &[u8], card: usize) -> Result<Vec<L>, Error> {
    match tag {
        VAL_RAW => {
            if blob.len() != card * L::BYTES {
                return Err(Error::CorruptPayload("dict raw values length"));
            }
            Ok(L::from_le_bytes(blob))
        }
        VAL_DELTA => delta_bitpack::decode::<L>(blob, card),
        VAL_TRANSPOSE => {
            let bytes = decode_residuals(blob, card * L::BYTES)?;
            if bytes.len() != card * L::BYTES {
                return Err(Error::CorruptPayload("dict transpose values length"));
            }
            transpose::decode::<L>(&bytes, card)
        }
        _ => Err(Error::CorruptPayload("dict value tag")),
    }
}

pub(crate) fn encode<L: Lane>(
    vals: &[L],
    entropy: bool,
    lambda: u64,
    allow_lz: bool,
) -> Option<Vec<u8>> {
    if vals.is_empty() {
        return None;
    }
    // Above 50% cardinality the codes approach raw width and the dictionary is
    // huge — not worth it. Bail early to cap the cost on high-distinct blocks.
    let max_card = vals.len() / 2 + 1;
    let mut map: HashMap<L, u32> = HashMap::new();
    let mut dict: Vec<L> = Vec::new();
    let mut codes: Vec<u32> = Vec::with_capacity(vals.len());
    for &v in vals {
        let code = *map.entry(v).or_insert_with(|| {
            let c = dict.len() as u32;
            dict.push(v);
            c
        });
        codes.push(code);
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
    let sorted: Vec<L> = order.iter().map(|&old| dict[old as usize]).collect();
    let codes: Vec<L> = codes
        .iter()
        .map(|&c| L::from_u64(u64::from(remap[c as usize])))
        .collect();

    let (val_tag, val_blob) = encode_values(&sorted, entropy, lambda, allow_lz);
    let codes_section = encode_codes_section(&codes, sorted.len(), entropy, lambda, allow_lz);

    let mut out = Vec::with_capacity(val_blob.len() + codes_section.len() + 16);
    varint::write_u64(&mut out, sorted.len() as u64);
    out.push(val_tag);
    varint::write_u64(&mut out, val_blob.len() as u64);
    out.extend_from_slice(&val_blob);
    out.extend_from_slice(&codes_section);
    Some(out)
}

/// Encode the codes stream: `code_tag ++ blob` — the smaller of bit-packed
/// (random-access) or, when the level allows entropy, byte-plane entropy-coded.
fn encode_codes_section<L: Lane>(
    codes: &[L],
    card: usize,
    entropy: bool,
    lambda: u64,
    allow_lz: bool,
) -> Vec<u8> {
    let mut code_tag = CODES_BITPACK;
    let mut code_blob = for_bitpack::encode(codes, false);
    if entropy && card <= MAX_ENTROPY_CARD {
        let nbytes = plane_count(card);
        let mut blob = vec![nbytes as u8];
        for p in 0..nbytes {
            let plane: Vec<u8> = codes
                .iter()
                .map(|&c| (c.to_u64() >> (8 * p)) as u8)
                .collect();
            let coded = code_residuals(&plane, lambda, allow_lz);
            varint::write_u64(&mut blob, coded.len() as u64);
            blob.extend_from_slice(&coded);
        }
        if blob.len() < code_blob.len() {
            code_tag = CODES_ENTROPY;
            code_blob = blob;
        }
    }
    let mut out = Vec::with_capacity(code_blob.len() + 1);
    out.push(code_tag);
    out.extend_from_slice(&code_blob);
    out
}

/// Inverse of [`encode_codes_section`]: decode `n` codes from `code_tag ++ blob`.
fn decode_codes_section<L: Lane>(payload: &[u8], n: usize) -> Result<Vec<L>, Error> {
    let mut pos = 0usize;
    let tag = *payload.get(pos).ok_or(Error::Truncated)?;
    pos += 1;
    match tag {
        CODES_BITPACK => for_bitpack::decode::<L>(&payload[pos..], n, false),
        CODES_ENTROPY => {
            let nbytes = usize::from(*payload.get(pos).ok_or(Error::Truncated)?);
            pos += 1;
            if !(1..=2).contains(&nbytes) {
                return Err(Error::CorruptPayload("dict plane count"));
            }
            let mut codes = vec![L::ZERO; n];
            for p in 0..nbytes {
                let len = varint::read_u64(payload, &mut pos)? as usize;
                let blob = payload.get(pos..pos + len).ok_or(Error::Truncated)?;
                pos += len;
                let plane = decode_residuals(blob, n)?;
                if plane.len() != n {
                    return Err(Error::CorruptPayload("dict plane length"));
                }
                for (c, &b) in codes.iter_mut().zip(&plane) {
                    *c = *c | L::from_u64(u64::from(b) << (8 * p));
                }
            }
            Ok(codes)
        }
        _ => Err(Error::CorruptPayload("dict code tag")),
    }
}

pub(crate) fn decode<L: Lane>(payload: &[u8], n: usize) -> Result<Vec<L>, Error> {
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
    let dict = decode_values::<L>(val_tag, val_blob, card)?;

    let codes = decode_codes_section::<L>(&payload[pos..], n)?;

    let mut out = Vec::with_capacity(n);
    for c in codes {
        let idx = c.to_u64() as usize;
        if idx >= card {
            return Err(Error::CorruptPayload("dict code out of range"));
        }
        out.push(dict[idx]);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Shared (column-wide) dictionary: the value table lives once in the stream
// preamble; DICT_SHARED frames carry only a codes section into it.
// ---------------------------------------------------------------------------

/// Cap on the shared table's cardinality — bounds the encoder's hash pass and
/// the table memory; columns with more distinct values than this don't benefit
/// from value sharing anyway (the codes approach raw width).
const SHARED_MAX_CARD: usize = 1 << 20;

/// Column-wide dictionary context, built once per column by the encoder and
/// shared (read-only) across the per-block competitions.
pub(crate) struct SharedDict<L: Lane> {
    /// value → code in the sorted table.
    map: FxHashMap<L, u32>,
    /// Cardinality of the table.
    pub(crate) card: usize,
    /// Encoded preamble body: `varint(card) ++ val_tag ++ varint(len) ++ blob`.
    pub(crate) preamble: Vec<u8>,
}

/// Build the column-wide table over the full (valid-values) lane, or `None`
/// when the column is too high-cardinality to profit (> 50% distinct, like the
/// per-block bail) or exceeds [`SHARED_MAX_CARD`].
pub(crate) fn build_shared<L: Lane>(
    vals: &[L],
    entropy: bool,
    lambda: u64,
    allow_lz: bool,
) -> Option<SharedDict<L>> {
    if vals.is_empty() {
        return None;
    }
    let max_card = (vals.len() / 2 + 1).min(SHARED_MAX_CARD);
    let mut map: FxHashMap<L, u32> = FxHashMap::default();
    let mut dict: Vec<L> = Vec::new();
    for &v in vals {
        map.entry(v).or_insert_with(|| {
            let c = dict.len() as u32;
            dict.push(v);
            c
        });
        if dict.len() > max_card {
            return None;
        }
    }

    // Sort ascending and remap (same rationale as the per-block path: a
    // monotonic table compresses far better, codes are a permutation either way).
    let mut order: Vec<u32> = (0..dict.len() as u32).collect();
    order.sort_unstable_by_key(|&i| dict[i as usize]);
    let mut remap = vec![0u32; dict.len()];
    for (new, &old) in order.iter().enumerate() {
        remap[old as usize] = new as u32;
    }
    let sorted: Vec<L> = order.iter().map(|&old| dict[old as usize]).collect();
    for c in map.values_mut() {
        *c = remap[*c as usize];
    }

    let (val_tag, val_blob) = encode_values(&sorted, entropy, lambda, allow_lz);
    let mut preamble = Vec::with_capacity(val_blob.len() + 12);
    varint::write_u64(&mut preamble, sorted.len() as u64);
    preamble.push(val_tag);
    varint::write_u64(&mut preamble, val_blob.len() as u64);
    preamble.extend_from_slice(&val_blob);
    Some(SharedDict {
        map,
        card: sorted.len(),
        preamble,
    })
}

/// Decode the preamble body back to the value table. `n_total` bounds the
/// cardinality (distinct ≤ valid values) against corrupt streams; the encoder
/// never emits more than [`SHARED_MAX_CARD`], so that also hard-caps the
/// table allocation regardless of the declared column size.
pub(crate) fn decode_shared_preamble<L: Lane>(
    blob: &[u8],
    n_total: usize,
) -> Result<Vec<L>, Error> {
    let mut pos = 0usize;
    let card = varint::read_u64(blob, &mut pos)? as usize;
    if card == 0 || card > n_total || card > SHARED_MAX_CARD {
        return Err(Error::CorruptPayload("shared dict cardinality"));
    }
    let val_tag = *blob.get(pos).ok_or(Error::Truncated)?;
    pos += 1;
    let vlen = varint::read_u64(blob, &mut pos)? as usize;
    let end = pos.checked_add(vlen).ok_or(Error::Truncated)?;
    if end != blob.len() {
        return Err(Error::CorruptPayload("shared dict preamble length"));
    }
    decode_values::<L>(val_tag, &blob[pos..end], card)
}

/// Encode one block as codes into the shared table (the frame payload is just a
/// codes section). Returns `None` on an empty block or a value missing from the
/// table (impossible for a table built over the same column; defensive).
pub(crate) fn encode_shared<L: Lane>(
    vals: &[L],
    sd: &SharedDict<L>,
    entropy: bool,
    lambda: u64,
    allow_lz: bool,
) -> Option<Vec<u8>> {
    if vals.is_empty() {
        return None;
    }
    let mut codes: Vec<L> = Vec::with_capacity(vals.len());
    for v in vals {
        debug_assert!(sd.map.contains_key(v), "shared dict must cover the column");
        codes.push(L::from_u64(u64::from(*sd.map.get(v)?)));
    }
    Some(encode_codes_section(
        &codes, sd.card, entropy, lambda, allow_lz,
    ))
}

/// Decode a `DICT_SHARED` frame given the stream's decoded value table.
pub(crate) fn decode_shared<L: Lane>(
    payload: &[u8],
    n: usize,
    table: &[L],
) -> Result<Vec<L>, Error> {
    let codes = decode_codes_section::<L>(payload, n)?;
    let mut out = Vec::with_capacity(n);
    for c in codes {
        let idx = c.to_u64() as usize;
        if idx >= table.len() {
            return Err(Error::CorruptPayload("shared dict code out of range"));
        }
        out.push(table[idx]);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip<L: Lane>(vals: &[L]) -> Option<usize> {
        // Exercise both code representations.
        for (entropy, lambda) in [(false, 0u64), (true, 0u64), (true, 4u64)] {
            let enc = encode(vals, entropy, lambda, true)?;
            assert_eq!(decode::<L>(&enc, vals.len()).unwrap(), vals);
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
        // Same on the 32-bit lane (e.g. 16 distinct f32 pressure levels).
        let vals32: Vec<u32> = vals
            .iter()
            .map(|&v| ((v % 16) as f32 * 0.125).to_bits())
            .collect();
        let size32 = roundtrip(&vals32).expect("should encode");
        assert!(size32 < vals32.len());
    }

    #[test]
    fn high_cardinality_bails() {
        let vals: Vec<u64> = (0..10000u64).collect(); // all distinct
        assert!(encode(&vals, true, 0, true).is_none());
        let vals32: Vec<u32> = (0..10000u32).collect();
        assert!(encode(&vals32, true, 0, true).is_none());
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
        assert_eq!(
            decode_values::<u64>(tag, &blob, sorted.len()).unwrap(),
            sorted
        );

        // And a full high-cardinality round-trip through the codec.
        let mut s = 1u64;
        let vals: Vec<u64> = (0..80000)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                sorted[(s >> 49) as usize % sorted.len()]
            })
            .collect();
        let enc = encode(&vals, true, 0, true).unwrap();
        assert_eq!(decode::<u64>(&enc, vals.len()).unwrap(), vals);

        // 32-bit lane: the sorted f32 table also compresses below 4 B/entry.
        let sorted32: Vec<u32> = (0..20000)
            .map(|i| (1000.0_f32 + i as f32 * 0.01).to_bits())
            .collect();
        let (tag, blob) = encode_values(&sorted32, true, 0, true);
        assert!(blob.len() < sorted32.len() * 4);
        assert_eq!(
            decode_values::<u32>(tag, &blob, sorted32.len()).unwrap(),
            sorted32
        );
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
        let vals32: Vec<u32> = vals.iter().map(|&v| v as u32).collect();
        roundtrip(&vals32);
    }

    #[test]
    fn edges() {
        assert!(encode::<u64>(&[], true, 0, true).is_none());
        roundtrip(&vec![7u64; 1000]);
        roundtrip(&vec![7u32; 1000]);
    }

    #[test]
    fn shared_roundtrip_across_blocks() {
        // A value table too big for any *block* (cardinality > block/2) but
        // small for the column — the case the shared dictionary exists for.
        let mut s = 1u64;
        let table: Vec<u64> = (0..3000)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                s
            })
            .collect();
        let vals: Vec<u64> = (0..40000)
            .map(|i| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                table[((s >> 40) as usize + i) % table.len()]
            })
            .collect();
        let vals32: Vec<u32> = vals.iter().map(|&v| (v >> 32) as u32).collect();

        for (entropy, lambda) in [(true, 0u64), (true, 2u64), (false, 0u64)] {
            let sd = build_shared(&vals, entropy, lambda, false).expect("should build");
            let decoded_table =
                decode_shared_preamble::<u64>(&sd.preamble, vals.len()).expect("preamble");
            assert_eq!(decoded_table.len(), sd.card);
            for block in vals.chunks(4096) {
                let payload = encode_shared(block, &sd, entropy, lambda, false).unwrap();
                let out = decode_shared(&payload, block.len(), &decoded_table).unwrap();
                assert_eq!(out, block);
            }
            let sd = build_shared(&vals32, entropy, lambda, false).expect("should build");
            let decoded_table =
                decode_shared_preamble::<u32>(&sd.preamble, vals32.len()).expect("preamble");
            for block in vals32.chunks(4096) {
                let payload = encode_shared(block, &sd, entropy, lambda, false).unwrap();
                let out = decode_shared(&payload, block.len(), &decoded_table).unwrap();
                assert_eq!(out, block);
            }
        }
    }

    #[test]
    fn shared_bails_on_high_cardinality() {
        let vals: Vec<u64> = (0..10000u64).map(|i| i.wrapping_mul(0x9E3779B9)).collect();
        assert!(build_shared(&vals, true, 0, false).is_none());
    }

    #[test]
    fn shared_decode_rejects_out_of_range_code() {
        let vals = vec![1u64, 2, 3, 1, 2, 3, 1, 2];
        let sd = build_shared(&vals, false, 0, false).unwrap();
        let payload = encode_shared(&vals, &sd, false, 0, false).unwrap();
        // A table shorter than the codes reference must error, not index OOB.
        let short_table = vec![1u64];
        assert!(decode_shared(&payload, vals.len(), &short_table).is_err());
    }
}
