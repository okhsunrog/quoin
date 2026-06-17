//! Binary range coder (LZMA-style, carry via cache) plus an adaptive order-1
//! byte model. This is the reusable arithmetic-coding backend for the
//! residual-coding modes (e.g. PRED_RC).
//!
//! The design is the well-known LZMA range coder: 11-bit probabilities,
//! shift-rate-5 adaptation, 32-bit range renormalized a byte at a time. It is
//! deterministic and exact; encode and decode evolve identical models.

use crate::error::Error;
use crate::varint;

const PROB_BITS: u32 = 11;
const PROB_INIT: u16 = 1 << (PROB_BITS - 1); // 1024 (probability 0.5)
const MOVE_BITS: u32 = 5;
const TOP: u32 = 1 << 24;

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

struct RangeEncoder {
    low: u64,
    range: u32,
    cache: u8,
    cache_size: u64,
    out: Vec<u8>,
}

impl RangeEncoder {
    fn new() -> Self {
        RangeEncoder {
            low: 0,
            range: 0xFFFF_FFFF,
            cache: 0,
            cache_size: 1,
            out: Vec::new(),
        }
    }

    fn shift_low(&mut self) {
        if (self.low >> 32) != 0 || self.low < 0xFF00_0000 {
            let carry = (self.low >> 32) as u8;
            let mut temp = self.cache;
            loop {
                self.out.push(temp.wrapping_add(carry));
                temp = 0xFF;
                self.cache_size -= 1;
                if self.cache_size == 0 {
                    break;
                }
            }
            self.cache = ((self.low >> 24) & 0xFF) as u8;
        }
        self.cache_size += 1;
        self.low = (self.low & 0x00FF_FFFF) << 8;
    }

    #[inline]
    fn encode_bit(&mut self, prob: &mut u16, bit: u32) {
        let bound = (self.range >> PROB_BITS) * u32::from(*prob);
        // Branchless: the per-bit decision is data-dependent and unpredictable
        // (~37% of slots were lost to bad speculation here). `m` is all-ones for
        // bit==1, zero for bit==0; both candidate updates are computed and
        // mask-selected, producing output bit-identical to the branched form.
        let m = 0u32.wrapping_sub(bit);
        self.low += u64::from(bound & m);
        self.range = (bound & !m) | ((self.range - bound) & m);
        let p = u32::from(*prob);
        let p0 = p + (((1 << PROB_BITS) - p) >> MOVE_BITS);
        let p1 = p - (p >> MOVE_BITS);
        *prob = ((p0 & !m) | (p1 & m)) as u16;
        while self.range < TOP {
            self.range <<= 8;
            self.shift_low();
        }
    }

    fn finish(mut self) -> Vec<u8> {
        for _ in 0..5 {
            self.shift_low();
        }
        self.out
    }
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

struct RangeDecoder<'a> {
    code: u32,
    range: u32,
    input: &'a [u8],
    pos: usize,
}

impl<'a> RangeDecoder<'a> {
    fn new(input: &'a [u8]) -> Self {
        let mut d = RangeDecoder {
            code: 0,
            range: 0xFFFF_FFFF,
            input,
            pos: 0,
        };
        d.next_byte(); // skip the encoder's initial dummy byte
        for _ in 0..4 {
            d.code = (d.code << 8) | d.next_byte();
        }
        d
    }

    #[inline]
    fn next_byte(&mut self) -> u32 {
        // Reads past the end yield 0; the encoder's 5-byte flush guarantees
        // enough real bytes for every decoded symbol.
        let b = self.input.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        u32::from(b)
    }

    #[inline]
    fn decode_bit(&mut self, prob: &mut u16) -> u32 {
        let bound = (self.range >> PROB_BITS) * u32::from(*prob);
        // Branchless mask-select (see `encode_bit`): the `code < bound` test is
        // the unpredictable, mispredict-heavy branch. `bit` is derived without a
        // branch and used as a mask; the math matches the branched form exactly.
        let bit = u32::from(self.code >= bound);
        let m = 0u32.wrapping_sub(bit);
        self.code -= bound & m;
        self.range = (bound & !m) | ((self.range - bound) & m);
        let p = u32::from(*prob);
        let p0 = p + (((1 << PROB_BITS) - p) >> MOVE_BITS);
        let p1 = p - (p >> MOVE_BITS);
        *prob = ((p0 & !m) | (p1 & m)) as u16;
        while self.range < TOP {
            self.range <<= 8;
            self.code = (self.code << 8) | self.next_byte();
        }
        bit
    }
}

// ---------------------------------------------------------------------------
// Order-1 byte model: 256 contexts (previous byte) × a 256-node bit tree.
// ---------------------------------------------------------------------------

struct ByteModel {
    probs: Vec<u16>,
}

impl ByteModel {
    fn new() -> Self {
        ByteModel {
            probs: vec![PROB_INIT; 256 * 256],
        }
    }

    #[inline]
    fn encode(&mut self, enc: &mut RangeEncoder, ctx: usize, byte: u8) {
        debug_assert!(ctx < 256 && self.probs.len() == 256 * 256);
        let base = ctx * 256;
        let mut node = 1usize;
        for i in (0..8).rev() {
            let bit = u32::from((byte >> i) & 1);
            // SAFETY: ctx < 256 ⇒ base ≤ 65280, and node ∈ [1, 255] across the
            // 8-step bit-tree walk, so base + node ≤ 65535 < probs.len() (65536).
            // This per-bit access is the range coder's hot path; eliding the
            // bounds check is a measured, behavior-preserving win.
            let p = unsafe { self.probs.get_unchecked_mut(base + node) };
            enc.encode_bit(p, bit);
            node = (node << 1) | bit as usize;
        }
    }

    #[inline]
    fn decode(&mut self, dec: &mut RangeDecoder<'_>, ctx: usize) -> u8 {
        debug_assert!(ctx < 256 && self.probs.len() == 256 * 256);
        let base = ctx * 256;
        let mut node = 1usize;
        for _ in 0..8 {
            // SAFETY: see `encode` — base + node ≤ 65535 < probs.len().
            let p = unsafe { self.probs.get_unchecked_mut(base + node) };
            let bit = dec.decode_bit(p);
            node = (node << 1) | bit as usize;
        }
        (node & 0xFF) as u8
    }
}

// ---------------------------------------------------------------------------
// Public byte-stream API
// ---------------------------------------------------------------------------

/// Range-code `src`. Output is `varint(len) ++ range_coded_bytes`.
pub(crate) fn compress_bytes(src: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len() / 2 + 16);
    varint::write_u64(&mut out, src.len() as u64);
    let mut enc = RangeEncoder::new();
    let mut model = ByteModel::new();
    let mut ctx = 0usize;
    for &b in src {
        model.encode(&mut enc, ctx, b);
        ctx = b as usize;
    }
    out.extend_from_slice(&enc.finish());
    out
}

/// Inverse of [`compress_bytes`]. `max_len` bounds the declared length to avoid
/// pathological allocation on a corrupt stream.
pub(crate) fn decompress_bytes(src: &[u8], max_len: usize) -> Result<Vec<u8>, Error> {
    let mut pos = 0usize;
    let len = varint::read_u64(src, &mut pos)? as usize;
    if len > max_len {
        return Err(Error::CorruptPayload("range-coder length exceeds bound"));
    }
    let mut dec = RangeDecoder::new(&src[pos..]);
    let mut model = ByteModel::new();
    let mut out = Vec::with_capacity(len);
    let mut ctx = 0usize;
    for _ in 0..len {
        let b = model.decode(&mut dec, ctx);
        out.push(b);
        ctx = b as usize;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(s: &mut u64) -> u64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *s
    }

    #[test]
    fn roundtrip_random_bytes() {
        let mut s = 1u64;
        let data: Vec<u8> = (0..50_000).map(|_| (lcg(&mut s) >> 33) as u8).collect();
        let packed = compress_bytes(&data);
        let restored = decompress_bytes(&packed, data.len()).unwrap();
        assert_eq!(restored, data);
    }

    #[test]
    fn roundtrip_skewed_bytes() {
        // Mostly zeros with occasional spikes — should compress well.
        let mut s = 7u64;
        let data: Vec<u8> = (0..50_000)
            .map(|_| {
                if lcg(&mut s).is_multiple_of(16) {
                    (lcg(&mut s) >> 40) as u8
                } else {
                    0
                }
            })
            .collect();
        let packed = compress_bytes(&data);
        assert!(packed.len() < data.len(), "skewed data should shrink");
        let restored = decompress_bytes(&packed, data.len()).unwrap();
        assert_eq!(restored, data);
    }

    #[test]
    fn empty() {
        let packed = compress_bytes(&[]);
        assert_eq!(decompress_bytes(&packed, 0).unwrap(), Vec::<u8>::new());
    }
}
