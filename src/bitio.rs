//! MSB-first bit writer/reader. Used by the tANS coder (and, later, the
//! bit-packing transform modes). Mirrors the original `fc`'s `bw_t`/`br_t`.

/// Writes bits most-significant-first into a byte buffer.
pub(crate) struct BitWriter {
    out: Vec<u8>,
    acc: u64,
    nbits: u32,
}

impl BitWriter {
    pub(crate) fn new() -> Self {
        BitWriter { out: Vec::new(), acc: 0, nbits: 0 }
    }

    /// Append the low `bits` bits of `val` (0..=57 bits per call is safe).
    #[inline]
    pub(crate) fn put(&mut self, val: u64, bits: u32) {
        if bits == 0 {
            return;
        }
        let mask = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
        self.acc = (self.acc << bits) | (val & mask);
        self.nbits += bits;
        while self.nbits >= 8 {
            self.nbits -= 8;
            self.out.push((self.acc >> self.nbits) as u8);
        }
        // Drop the already-flushed high bits so `acc` can't overflow.
        self.acc &= (1u64 << self.nbits) - 1;
    }

    /// Flush any partial final byte (zero-padded low bits) and return the buffer.
    pub(crate) fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            self.out.push((self.acc << (8 - self.nbits)) as u8);
        }
        self.out
    }
}

/// Reads bits most-significant-first. Reads past the end yield zero bits.
pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    acc: u64,
    nbits: u32,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        BitReader { data, pos: 0, acc: 0, nbits: 0 }
    }

    #[inline]
    pub(crate) fn get(&mut self, bits: u32) -> u64 {
        if bits == 0 {
            return 0;
        }
        while self.nbits < bits {
            let byte = self.data.get(self.pos).copied().unwrap_or(0);
            self.pos += 1;
            self.acc = (self.acc << 8) | u64::from(byte);
            self.nbits += 8;
        }
        self.nbits -= bits;
        let mask = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
        let v = (self.acc >> self.nbits) & mask;
        self.acc &= (1u64 << self.nbits) - 1;
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_varied_widths() {
        let items: &[(u64, u32)] =
            &[(0, 1), (1, 1), (5, 3), (0, 0), (1023, 10), (255, 8), (0x1_2345, 17), (3, 2)];
        let mut w = BitWriter::new();
        for &(v, b) in items {
            w.put(v, b);
        }
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        for &(v, b) in items {
            assert_eq!(r.get(b), v, "mismatch reading {b} bits");
        }
    }

    #[test]
    fn long_stream() {
        let mut s = 12345u64;
        let mut vals = Vec::new();
        let mut w = BitWriter::new();
        for _ in 0..10_000 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let bits = 1 + (s % 11) as u32;
            let mask = (1u64 << bits) - 1;
            let v = (s >> 20) & mask;
            vals.push((v, bits));
            w.put(v, bits);
        }
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        for (v, bits) in vals {
            assert_eq!(r.get(bits), v);
        }
    }
}
