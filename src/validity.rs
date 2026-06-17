//! Validity (null) bitmaps — Arrow-compatible: one bit per value, **LSB-first**
//! within each byte, `1` = valid, `0` = null; length `ceil(n/8)` bytes with
//! unused trailing bits cleared.
//!
//! Nulls almost always cluster, so the bitmap is stored as the smaller of a
//! run-length encoding (alternating run lengths, starting from a 0/null run) or
//! the raw bitmap. The values themselves are stored **compacted** — only the
//! valid ones — by the caller; this module just (de)serializes the bitmap and
//! provides the compact/scatter helpers.

use crate::error::Error;
use crate::varint;

const TAG_RAW: u8 = 0;
const TAG_RLE: u8 = 1;

#[inline]
pub(crate) fn is_set(bitmap: &[u8], i: usize) -> bool {
    (bitmap[i >> 3] >> (i & 7)) & 1 == 1
}

#[inline]
fn set_bit(bitmap: &mut [u8], i: usize) {
    bitmap[i >> 3] |= 1 << (i & 7);
}

/// Number of valid (set) bits among the first `n`.
pub(crate) fn count_valid(bitmap: &[u8], n: usize) -> usize {
    let full = n / 8;
    let mut c: usize = bitmap[..full].iter().map(|b| b.count_ones() as usize).sum();
    let rem = n & 7;
    if rem != 0 {
        let mask = (1u8 << rem) - 1;
        c += (bitmap[full] & mask).count_ones() as usize;
    }
    c
}

/// Serialize a validity bitmap (the smaller of raw / run-length).
pub(crate) fn encode(bitmap: &[u8], n: usize) -> Vec<u8> {
    // Run-length: alternating runs of equal bits, starting with the 0 (null)
    // value — a leading `1`-bit yields an initial zero-length run.
    let mut rle = Vec::new();
    let mut cur = 0u8;
    let mut run: u64 = 0;
    for i in 0..n {
        let bit = u8::from(is_set(bitmap, i));
        if bit == cur {
            run += 1;
        } else {
            varint::write_u64(&mut rle, run);
            cur ^= 1;
            run = 1;
        }
    }
    varint::write_u64(&mut rle, run);

    let raw_len = n.div_ceil(8);
    if rle.len() < raw_len {
        let mut out = Vec::with_capacity(rle.len() + 1);
        out.push(TAG_RLE);
        out.extend_from_slice(&rle);
        out
    } else {
        let mut out = Vec::with_capacity(raw_len + 1);
        out.push(TAG_RAW);
        out.extend_from_slice(&bitmap[..raw_len]);
        out
    }
}

/// Inverse of [`encode`]: reconstruct the `ceil(n/8)`-byte bitmap.
pub(crate) fn decode(blob: &[u8], n: usize) -> Result<Vec<u8>, Error> {
    let raw_len = n.div_ceil(8);
    let mut bitmap = Vec::new();
    bitmap
        .try_reserve_exact(raw_len)
        .map_err(|_| Error::CorruptPayload("validity bitmap too large"))?;
    bitmap.resize(raw_len, 0);
    let (&tag, rest) = blob.split_first().ok_or(Error::Truncated)?;
    match tag {
        TAG_RAW => {
            let src = rest.get(..raw_len).ok_or(Error::Truncated)?;
            bitmap.copy_from_slice(src);
            // Clear unused trailing bits so the output is canonical.
            let rem = n & 7;
            if rem != 0 {
                bitmap[raw_len - 1] &= (1u8 << rem) - 1;
            }
        }
        TAG_RLE => {
            let mut pos = 0usize;
            let mut i = 0usize;
            let mut cur = 0u8;
            while i < n {
                let run = usize::try_from(varint::read_u64(rest, &mut pos)?)
                    .map_err(|_| Error::CorruptPayload("validity run too large"))?;
                let end = i
                    .checked_add(run)
                    .ok_or(Error::CorruptPayload("validity run overruns"))?;
                if end > n {
                    return Err(Error::CorruptPayload("validity run overruns"));
                }
                if cur == 1 {
                    for k in i..end {
                        set_bit(&mut bitmap, k);
                    }
                }
                i = end;
                cur ^= 1;
            }
            if i != n {
                return Err(Error::CorruptPayload("validity runs mismatch"));
            }
        }
        _ => return Err(Error::CorruptPayload("validity tag")),
    }
    Ok(bitmap)
}

/// Keep only the lane words at valid positions (compaction for the value codec).
pub(crate) fn compact(lane: &[u64], bitmap: &[u8]) -> Vec<u64> {
    (0..lane.len())
        .filter(|&i| is_set(bitmap, i))
        .map(|i| lane[i])
        .collect()
}

/// Scatter `valid` back into `n` positions per `bitmap`; null slots become 0.
pub(crate) fn scatter(valid: &[u64], bitmap: &[u8], n: usize) -> Result<Vec<u64>, Error> {
    if count_valid(bitmap, n) != valid.len() {
        return Err(Error::CorruptPayload("validity/value count mismatch"));
    }
    let mut out = Vec::new();
    out.try_reserve_exact(n)
        .map_err(|_| Error::CorruptPayload("decoded column too large"))?;
    out.resize(n, 0);
    let mut j = 0usize;
    for (i, slot) in out.iter_mut().enumerate() {
        if is_set(bitmap, i) {
            *slot = valid[j];
            j += 1;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(bits: &[bool]) {
        let n = bits.len();
        let mut bm = vec![0u8; n.div_ceil(8)];
        for (i, &b) in bits.iter().enumerate() {
            if b {
                set_bit(&mut bm, i);
            }
        }
        let enc = encode(&bm, n);
        let dec = decode(&enc, n).unwrap();
        assert_eq!(dec, bm, "validity round-trip n={n}");
        assert_eq!(count_valid(&bm, n), bits.iter().filter(|b| **b).count());
    }

    #[test]
    fn shapes() {
        roundtrip(&[]);
        roundtrip(&[true]);
        roundtrip(&[false]);
        roundtrip(&vec![true; 1000]); // all valid -> RLE tiny
        roundtrip(&vec![false; 1000]); // all null
        // clustered nulls
        let mut v = vec![true; 1000];
        for b in v.iter_mut().take(700).skip(500) {
            *b = false;
        }
        roundtrip(&v);
        // scattered (raw likely wins)
        let mut s = 1u64;
        let scattered: Vec<bool> = (0..1000)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                s & 1 == 0
            })
            .collect();
        roundtrip(&scattered);
        // not a multiple of 8
        roundtrip(&[true, false, true, true, false]);
    }

    #[test]
    fn compact_scatter() {
        let lane = vec![10u64, 20, 30, 40, 50];
        let mut bm = vec![0u8];
        for i in [0, 2, 4] {
            set_bit(&mut bm, i);
        }
        let c = compact(&lane, &bm);
        assert_eq!(c, vec![10, 30, 50]);
        let s = scatter(&c, &bm, 5).unwrap();
        assert_eq!(s, vec![10, 0, 30, 0, 50]); // null slots -> 0
    }
}
