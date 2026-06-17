//! Interleaved rANS — a fast order-0 entropy coder.
//!
//! Four independent rANS states are interleaved over the symbol stream so the
//! decode loop has four parallel dependency chains (ILP, and SIMD-friendly),
//! decoding at GB/s — far faster than the bit-serial range coder ([`super::rc`]).
//! Order-0 with a static, normalized frequency table (12-bit precision), 32-bit
//! states, byte-wise renormalization (the canonical `rans_byte` configuration).
//!
//! Ratio is order-0, so on byte streams with strong context the range coder
//! still wins; rANS shines on near-i.i.d. streams (dictionary codes, scaled-int
//! residuals) and whenever decode speed is the priority — see the level-aware
//! choice in [`super::code_residuals`].

use crate::error::Error;
use crate::varint;

const SCALE_BITS: u32 = 12;
const M: u32 = 1 << SCALE_BITS; // total frequency
const MASK: u32 = M - 1;
const RANS_L: u32 = 1 << 23; // lower renorm bound (8-bit renorm)
const NSTREAMS: usize = 4;

/// Normalize symbol counts to sum to exactly `M`, with every present symbol
/// keeping a frequency ≥ 1. Returns per-symbol frequencies.
fn normalize(counts: &[u32; 256], total: u64) -> [u32; 256] {
    let mut freq = [0u32; 256];
    let mut sum = 0u32;
    for s in 0..256 {
        if counts[s] > 0 {
            let f = ((u64::from(counts[s]) * u64::from(M)) / total) as u32;
            freq[s] = f.max(1);
            sum += freq[s];
        }
    }
    // Correct the total to exactly M without zeroing any present symbol.
    while sum > M {
        // Drop one from the current largest freq that is > 1.
        let s = (0..256)
            .filter(|&s| freq[s] > 1)
            .max_by_key(|&s| freq[s])
            .unwrap();
        freq[s] -= 1;
        sum -= 1;
    }
    if sum < M {
        let s = (0..256).max_by_key(|&s| freq[s]).unwrap();
        freq[s] += M - sum;
    }
    freq
}

pub(crate) fn compress_bytes(data: &[u8]) -> Option<Vec<u8>> {
    let n = data.len();
    if n == 0 {
        return None;
    }
    let mut counts = [0u32; 256];
    for &b in data {
        counts[b as usize] += 1;
    }
    let freq = normalize(&counts, n as u64);
    let mut cum = [0u32; 257];
    for s in 0..256 {
        cum[s + 1] = cum[s] + freq[s];
    }

    // Encode backward into a scratch buffer. Worst case is `SCALE_BITS` bits per
    // symbol (a min-frequency symbol) → ≤ 1.5·n bytes; size to 2·n + flush so the
    // backward write never underruns even on incompressible input.
    let mut buf = vec![0u8; 2 * n + 4 * NSTREAMS + 64];
    let mut p = buf.len();
    let mut x = [RANS_L; NSTREAMS];
    let mut i = n;
    while i > 0 {
        i -= 1;
        let s = data[i] as usize;
        let f = freq[s];
        let c = cum[s];
        let idx = i & (NSTREAMS - 1);
        let mut xi = x[idx];
        // renorm: emit bytes until xi < x_max
        let x_max = ((RANS_L >> SCALE_BITS) << 8) * f;
        while xi >= x_max {
            p -= 1;
            buf[p] = xi as u8;
            xi >>= 8;
        }
        x[idx] = ((xi / f) << SCALE_BITS) + (xi % f) + c;
    }
    // Flush states so the decoder reads stream 0,1,2,3 from the front.
    for idx in (0..NSTREAMS).rev() {
        p -= 4;
        buf[p..p + 4].copy_from_slice(&x[idx].to_le_bytes());
    }
    let stream = &buf[p..];

    // Header: n, frequency table (present symbols only), then the stream.
    let mut out = Vec::with_capacity(stream.len() + 64);
    varint::write_u64(&mut out, n as u64);
    let present: Vec<usize> = (0..256).filter(|&s| freq[s] > 0).collect();
    varint::write_u64(&mut out, present.len() as u64);
    for &s in &present {
        out.push(s as u8);
        varint::write_u64(&mut out, u64::from(freq[s]));
    }
    out.extend_from_slice(stream);
    Some(out)
}

pub(crate) fn decompress_bytes(payload: &[u8], max_len: usize) -> Result<Vec<u8>, Error> {
    let mut pos = 0usize;
    let n = varint::read_u64(payload, &mut pos)? as usize;
    if n > max_len {
        return Err(Error::CorruptPayload("rans length exceeds bound"));
    }

    let num_syms = varint::read_u64(payload, &mut pos)? as usize;
    if num_syms == 0 || num_syms > 256 {
        return Err(Error::CorruptPayload("rans symbol count"));
    }
    let mut freq = [0u32; 256];
    let mut sum: u32 = 0;
    for _ in 0..num_syms {
        let s = *payload.get(pos).ok_or(Error::Truncated)?;
        pos += 1;
        let f = varint::read_u64(payload, &mut pos)? as u32;
        if f == 0 || f > M {
            return Err(Error::CorruptPayload("rans frequency"));
        }
        freq[s as usize] = f;
        sum += f;
    }
    if sum != M {
        return Err(Error::CorruptPayload("rans frequencies do not sum to M"));
    }
    let mut cum = [0u32; 257];
    for s in 0..256 {
        cum[s + 1] = cum[s] + freq[s];
    }
    // Slot → symbol decode table.
    let mut slot_to_sym = vec![0u8; M as usize];
    for s in 0..256 {
        for slot in cum[s]..cum[s + 1] {
            slot_to_sym[slot as usize] = s as u8;
        }
    }

    // Init the four states from the front of the stream.
    let mut x = [0u32; NSTREAMS];
    for xi in x.iter_mut() {
        let b = payload.get(pos..pos + 4).ok_or(Error::Truncated)?;
        *xi = u32::from_le_bytes(b.try_into().unwrap());
        pos += 4;
    }

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let idx = i & (NSTREAMS - 1);
        let xi = x[idx];
        let slot = xi & MASK;
        let s = slot_to_sym[slot as usize];
        out.push(s);
        let f = freq[s as usize];
        let c = cum[s as usize];
        let mut xn = f * (xi >> SCALE_BITS) + slot - c;
        // renorm: pull bytes until xn >= RANS_L
        while xn < RANS_L {
            let b = *payload.get(pos).ok_or(Error::Truncated)?;
            pos += 1;
            xn = (xn << 8) | u32::from(b);
        }
        x[idx] = xn;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(data: &[u8]) {
        match compress_bytes(data) {
            Some(enc) => {
                let dec = decompress_bytes(&enc, data.len()).unwrap();
                assert_eq!(dec, data, "rANS round-trip mismatch (len {})", data.len());
            }
            None => assert!(data.is_empty()),
        }
    }

    #[test]
    fn roundtrips() {
        roundtrip(&[]);
        roundtrip(&[0]);
        roundtrip(&[42; 1000]); // single symbol
        roundtrip(&[1, 2, 3, 4, 5]); // n not a multiple of 4
        // skewed distribution
        let mut s = 1u64;
        let skewed: Vec<u8> = (0..20000)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                // bias toward small symbols
                ((s >> 60) as u8) & if s & 0xF == 0 { 0xFF } else { 0x07 }
            })
            .collect();
        roundtrip(&skewed);
        // all 256 symbols, uniform-ish
        let allsyms: Vec<u8> = (0..256 * 40).map(|i| (i % 256) as u8).collect();
        roundtrip(&allsyms);
        // incompressible uniform-random bytes — rANS expands; must not panic.
        let mut s = 0x9E37_79B9u64;
        let noise: Vec<u8> = (0..30000)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                (s >> 56) as u8
            })
            .collect();
        roundtrip(&noise);
    }

    #[test]
    fn compresses_skewed() {
        // 90% zeros → should compress well below 1 B/symbol.
        let mut data = vec![0u8; 9000];
        data.extend((0..1000).map(|i| (i % 255 + 1) as u8));
        let enc = compress_bytes(&data).unwrap();
        assert!(
            enc.len() < data.len() / 2,
            "skewed should compress, got {}",
            enc.len()
        );
        assert_eq!(decompress_bytes(&enc, data.len()).unwrap(), data);
    }
}
