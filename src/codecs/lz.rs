//! LZ: byte-level LZ77 over the block's little-endian bytes. Targets the
//! repeating-value datasets (dictionaries, quantized levels, cent-rounded
//! prices) where the predictors find little but whole values or byte runs
//! recur. The token stream is entropy-coded by the caller (LZ + range/rANS,
//! analogous to zstd's LZ + FSE).
//!
//! LZSS framing: a control byte holds 8 flags (MSB-first); a 1 flag is a match
//! `(varint offset, varint length-MIN_MATCH)`, a 0 flag is a literal byte.
//! Decode tolerates overlapping copies (offset < length).

use crate::error::Error;
use crate::varint;

const MIN_MATCH: usize = 4;
const HASH_BITS: u32 = 17;
const HASH_SIZE: usize = 1 << HASH_BITS;
const MAX_CHAIN: usize = 64;

#[inline]
fn hash4(window: &[u8]) -> usize {
    let v = u32::from_le_bytes([window[0], window[1], window[2], window[3]]);
    (v.wrapping_mul(2654435761) >> (32 - HASH_BITS)) as usize
}

enum Token {
    Lit(u8),
    Match { off: u32, len: u32 },
}

pub(crate) fn lz_compress(input: &[u8]) -> Vec<u8> {
    let n = input.len();
    let mut tokens: Vec<Token> = Vec::new();
    let mut head = vec![-1i64; HASH_SIZE];
    let mut prev = vec![-1i64; n.max(1)];

    let mut i = 0usize;
    while i < n {
        let mut best_len = 0usize;
        let mut best_off = 0usize;
        if i + MIN_MATCH <= n {
            let h = hash4(&input[i..]);
            let max_len = n - i;
            let mut cand = head[h];
            let mut chain = MAX_CHAIN;
            while cand >= 0 && chain > 0 {
                let c = cand as usize;
                // Quick reject (zlib's trick): a candidate can only beat the
                // current best if the byte at the `best_len` boundary matches.
                // One compare skips the full scan for the vast majority of
                // candidates — ratio-neutral, it finds the same best match.
                if best_len == 0 || (c + best_len < n && input[c + best_len] == input[i + best_len])
                {
                    let mut l = 0;
                    while l < max_len && input[c + l] == input[i + l] {
                        l += 1;
                    }
                    if l > best_len {
                        best_len = l;
                        best_off = i - c;
                        if l == max_len {
                            break;
                        }
                    }
                }
                cand = prev[c];
                chain -= 1;
            }
        }

        if best_len >= MIN_MATCH {
            tokens.push(Token::Match {
                off: best_off as u32,
                len: best_len as u32,
            });
            let end = i + best_len;
            while i < end {
                if i + MIN_MATCH <= n {
                    let h = hash4(&input[i..]);
                    prev[i] = head[h];
                    head[h] = i as i64;
                }
                i += 1;
            }
        } else {
            tokens.push(Token::Lit(input[i]));
            if i + MIN_MATCH <= n {
                let h = hash4(&input[i..]);
                prev[i] = head[h];
                head[h] = i as i64;
            }
            i += 1;
        }
    }

    // Serialize: varint(original length) then control-byte groups.
    let mut out = Vec::with_capacity(n / 2 + 16);
    varint::write_u64(&mut out, n as u64);
    for group in tokens.chunks(8) {
        let mut ctrl = 0u8;
        for (k, t) in group.iter().enumerate() {
            if matches!(t, Token::Match { .. }) {
                ctrl |= 1 << (7 - k);
            }
        }
        out.push(ctrl);
        for t in group {
            match *t {
                Token::Lit(b) => out.push(b),
                Token::Match { off, len } => {
                    varint::write_u64(&mut out, u64::from(off));
                    varint::write_u64(&mut out, u64::from(len) - MIN_MATCH as u64);
                }
            }
        }
    }
    out
}

pub(crate) fn lz_decompress(stream: &[u8], expected: usize) -> Result<Vec<u8>, Error> {
    let mut pos = 0usize;
    let n = varint::read_u64(stream, &mut pos)? as usize;
    if n != expected {
        return Err(Error::CorruptPayload("lz length mismatch"));
    }
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let ctrl = *stream.get(pos).ok_or(Error::Truncated)?;
        pos += 1;
        for k in 0..8u32 {
            if out.len() >= n {
                break;
            }
            if (ctrl >> (7 - k)) & 1 == 1 {
                let off = varint::read_u64(stream, &mut pos)? as usize;
                let len = varint::read_u64(stream, &mut pos)? as usize + MIN_MATCH;
                if off == 0 || off > out.len() {
                    return Err(Error::CorruptPayload("lz bad offset"));
                }
                if out.len() + len > n {
                    return Err(Error::CorruptPayload("lz match overruns output"));
                }
                let start = out.len() - off;
                for j in 0..len {
                    let b = out[start + j]; // overlapping copy is intentional
                    out.push(b);
                }
            } else {
                out.push(*stream.get(pos).ok_or(Error::Truncated)?);
                pos += 1;
            }
        }
    }
    Ok(out)
}

pub(crate) fn encode(vals: &[u64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(vals.len() * 8);
    for &v in vals {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    lz_compress(&bytes)
}

pub(crate) fn decode(stream: &[u8], n_values: usize) -> Result<Vec<u64>, Error> {
    let bytes = lz_decompress(stream, n_values * 8)?;
    let mut out = Vec::with_capacity(n_values);
    for chunk in bytes.chunks_exact(8) {
        out.push(u64::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_repetitive() {
        // 16-value dictionary cycled — classic LZ target.
        let vals: Vec<u64> = (0..50_000u64)
            .map(|i| (i & 15) as f64)
            .map(f64::to_bits)
            .collect();
        let enc = encode(&vals);
        assert!(
            enc.len() < vals.len() * 8 / 4,
            "repetitive data should shrink a lot"
        );
        assert_eq!(decode(&enc, vals.len()).unwrap(), vals);
    }

    #[test]
    fn roundtrip_random_and_edges() {
        let mut s = 1u64;
        let vals: Vec<u64> = (0..5000)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                s
            })
            .collect();
        assert_eq!(decode(&encode(&vals), vals.len()).unwrap(), vals);
        assert_eq!(decode(&encode(&[]), 0).unwrap(), Vec::<u64>::new());
        assert_eq!(decode(&encode(&[42]), 1).unwrap(), vec![42]);
    }
}
