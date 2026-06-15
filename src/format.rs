//! On-disk stream format (v1).
//!
//! ```text
//! Header (16 bytes):
//!   [0..4]  magic = "FCR1"
//!   [4]     format version = 1
//!   [5]     flags (reserved, 0)
//!   [6]     predictor_log2 (clamped 10..=16)
//!   [7]     reserved (0)
//!   [8..16] value count (u64 LE) — number of f64 values
//!
//! Then one frame per block until `value count` values are decoded:
//!   [0]      mode id (u8)
//!   [1..5]   value count for this block (u32 LE)
//!   [5..9]   payload length in bytes (u32 LE)
//!   [9..]    payload
//! ```

use crate::error::Error;

pub(crate) const MAGIC: [u8; 4] = *b"FCR1";
pub(crate) const VERSION: u8 = 1;
pub(crate) const HEADER_LEN: usize = 16;
pub(crate) const FRAME_HEADER_LEN: usize = 9;

pub(crate) struct Header {
    pub predictor_log2: u8,
    pub n_values: u64,
}

impl Header {
    pub(crate) fn write(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&MAGIC);
        out.push(VERSION);
        out.push(0); // flags
        out.push(self.predictor_log2);
        out.push(0); // reserved
        out.extend_from_slice(&self.n_values.to_le_bytes());
    }

    pub(crate) fn read(src: &[u8]) -> Result<Header, Error> {
        if src.len() < HEADER_LEN {
            return Err(Error::Truncated);
        }
        if src[0..4] != MAGIC {
            return Err(Error::BadMagic);
        }
        if src[4] != VERSION {
            return Err(Error::UnsupportedVersion(src[4]));
        }
        let predictor_log2 = src[6];
        let n_values = u64::from_le_bytes(src[8..16].try_into().unwrap());
        Ok(Header { predictor_log2, n_values })
    }
}
