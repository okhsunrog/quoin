//! On-disk stream format (v2).
//!
//! ```text
//! Header (16 bytes):
//!   [0..4]  magic = "FCR1"
//!   [4]     format version = 2
//!   [5]     flags (reserved, 0)
//!   [6]     predictor_log2 (clamped 10..=16)
//!   [7]     column type id (see `DType::wire_id`)
//!   [8..16] value count (u64 LE) — number of values
//!
//! Then one frame per block until `value count` values are decoded:
//!   [0]      mode id (u8)
//!   [1..5]   value count for this block (u32 LE)
//!   [5..9]   payload length in bytes (u32 LE)
//!   [9..]    payload
//! ```
//!
//! v2 added the column-type byte at [7] (v1 carried only `f64` and reserved it).

use crate::dtype::DType;
use crate::error::Error;

pub(crate) const MAGIC: [u8; 4] = *b"FCR1";
pub(crate) const VERSION: u8 = 2;
pub(crate) const HEADER_LEN: usize = 16;

/// Flag bit (header[5]): a compressed validity bitmap follows the header, before
/// the value frames, and the frames hold only the **valid** values (compacted).
pub(crate) const FLAG_VALIDITY: u8 = 0x01;
/// Flag bit (header[5]): this is a **decimal container** (see [`crate::decimal`]),
/// not an ordinary lane stream. After the 16-byte header comes a decimal metadata
/// section (scale, precision, vmin) and one `U64` sub-stream per 64-bit limb. The
/// ordinary [`Header::read`] rejects this flag, so the top-level dispatcher must
/// route decimal streams to the decimal decoder before reading the header.
pub(crate) const FLAG_DECIMAL: u8 = 0x02;
pub(crate) const FRAME_HEADER_LEN: usize = 9;

/// Maximum values a single block may declare. The encoder grows low-entropy
/// blocks up to this (adaptive sizing), so the decoder rejects anything larger —
/// this bounds per-block allocation and stops a tiny `CONST`/`STRIDE` frame from
/// claiming a huge value count (a decompression bomb). 128 Ki * 8 B = 1 MiB,
/// matching `fc`'s max quantum.
pub(crate) const MAX_BLOCK_VALUES: usize = 128 * 1024;

pub(crate) struct Header {
    pub predictor_log2: u8,
    pub dtype: DType,
    /// A validity bitmap follows the header (see [`FLAG_VALIDITY`]).
    pub has_validity: bool,
    /// Logical value count (including nulls).
    pub n_values: u64,
}

impl Header {
    pub(crate) fn write(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&MAGIC);
        out.push(VERSION);
        out.push(if self.has_validity { FLAG_VALIDITY } else { 0 }); // flags
        out.push(self.predictor_log2);
        out.push(self.dtype.wire_id());
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
        if src[5] & !FLAG_VALIDITY != 0 {
            return Err(Error::CorruptPayload("unknown header flags"));
        }
        let has_validity = src[5] & FLAG_VALIDITY != 0;
        // Must match the encoder's clamp; the predictor codecs use this as a
        // shift amount (`1 << predictor_log2`) and table size, so an
        // out-of-range value from a corrupt stream would overflow / over-allocate.
        let predictor_log2 = src[6];
        if !(10..=16).contains(&predictor_log2) {
            return Err(Error::CorruptPayload("predictor_log2 out of range"));
        }
        let dtype = DType::from_wire(src[7])?;
        let n_values = u64::from_le_bytes(src[8..16].try_into().unwrap());
        Ok(Header {
            predictor_log2,
            dtype,
            has_validity,
            n_values,
        })
    }
}
