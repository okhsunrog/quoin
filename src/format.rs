//! On-disk stream format (v3).
//!
//! ```text
//! Header (16 bytes):
//!   [0..4]  magic = "FCR1"
//!   [4]     format version = 3
//!   [5]     flags (validity / decimal / shared-dict)
//!   [6]     predictor_log2 (clamped 10..=16)
//!   [7]     column type id (see `DType::wire_id`)
//!   [8..16] value count (u64 LE) — number of values
//!
//! Then, if the shared-dictionary flag is set, the column preamble (after the
//! validity section when both are present):
//!   varint(preamble length) ++ preamble
//!   preamble = varint(cardinality) ++ val_tag ++ varint(len) ++ value blob
//!              (the value blob is `dict.rs`'s compressed sorted-values stream)
//!
//! Then one frame per block until `value count` values are decoded:
//!   [0]      mode id (u8)
//!   [1..5]   value count for this block (u32 LE)
//!   [5..9]   payload length in bytes (u32 LE)
//!   [9..]    payload
//! ```
//!
//! Every payload is laid out in **lane words**: a 32-bit column's constants,
//! dictionary entries, frame-of-reference minima and byte planes are 4 bytes
//! wide, a 64-bit column's are 8. The column type id therefore fixes the lane
//! width of every frame in the stream.
//!
//! **Versions.** v2 added the column-type byte at [7] (v1 carried only `f64`).
//! v3 made the 32-bit types (`f32`/`i32`/`u32`) native 32-bit lanes; in v2
//! they were widened to 64-bit lane words (`f32` to its exact `f64`) and their
//! payloads are laid out accordingly. The 64-bit types and the decimal
//! containers are byte-for-byte identical between v2 and v3, so v2 streams of
//! those types still decode; a v2 stream of a 32-bit type is rejected with
//! [`Error::UnsupportedVersion`] rather than misread under the new layout.

use crate::dtype::DType;
use crate::error::Error;

pub(crate) const MAGIC: [u8; 4] = *b"FCR1";
pub(crate) const VERSION: u8 = 3;
/// The previous format version, still readable for the 64-bit lanes and the
/// decimal containers (whose layout did not change).
pub(crate) const LEGACY_VERSION: u8 = 2;
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
/// Flag bit (header[5]): a **shared value dictionary** preamble follows the
/// header (after the validity section, when both are present): the column-wide
/// sorted distinct values, stored once; `DICT_SHARED` frames hold only codes
/// into it. See [`crate::codecs::dict`].
pub(crate) const FLAG_SHARED_DICT: u8 = 0x04;
pub(crate) const FRAME_HEADER_LEN: usize = 9;

/// Maximum bytes of lane data a single block may hold: the encoder grows
/// low-entropy blocks up to this (adaptive sizing), and the decoder rejects any
/// frame declaring more values than fit — bounding per-block allocation and
/// stopping a tiny `CONST`/`STRIDE` frame from claiming a huge value count (a
/// decompression bomb). 1 MiB, matching `fc`'s max quantum.
pub(crate) const MAX_BLOCK_BYTES: usize = 1024 * 1024;

/// Maximum values a single block may declare on a lane of `lane_bytes` bytes
/// per value: `MAX_BLOCK_BYTES / lane_bytes` (128 Ki for 8-byte lanes, 256 Ki
/// for 4-byte lanes — the same byte budget either way).
pub(crate) const fn max_block_values(lane_bytes: usize) -> usize {
    MAX_BLOCK_BYTES / lane_bytes
}

/// [`max_block_values`] for the 64-bit lanes (the public [`crate::MAX_BLOCK_SIZE`]).
pub(crate) const MAX_BLOCK_VALUES: usize = max_block_values(8);

/// Whether a stream written at `version` is readable for `dtype` by this build.
pub(crate) fn version_supported(version: u8, dtype: DType) -> bool {
    version == VERSION || (version == LEGACY_VERSION && dtype.lane_bytes() == 8)
}

pub(crate) struct Header {
    pub predictor_log2: u8,
    pub dtype: DType,
    /// A validity bitmap follows the header (see [`FLAG_VALIDITY`]).
    pub has_validity: bool,
    /// A shared-dictionary preamble follows the validity section (see
    /// [`FLAG_SHARED_DICT`]).
    pub has_shared_dict: bool,
    /// Logical value count (including nulls).
    pub n_values: u64,
}

impl Header {
    pub(crate) fn write(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&MAGIC);
        out.push(VERSION);
        let mut flags = 0u8;
        if self.has_validity {
            flags |= FLAG_VALIDITY;
        }
        if self.has_shared_dict {
            flags |= FLAG_SHARED_DICT;
        }
        out.push(flags);
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
        let version = src[4];
        if version != VERSION && version != LEGACY_VERSION {
            return Err(Error::UnsupportedVersion(version));
        }
        if src[5] & !(FLAG_VALIDITY | FLAG_SHARED_DICT) != 0 {
            return Err(Error::CorruptPayload("unknown header flags"));
        }
        let has_validity = src[5] & FLAG_VALIDITY != 0;
        let has_shared_dict = src[5] & FLAG_SHARED_DICT != 0;
        // Must match the encoder's clamp; the predictor codecs use this as a
        // shift amount (`1 << predictor_log2`) and table size, so an
        // out-of-range value from a corrupt stream would overflow / over-allocate.
        let predictor_log2 = src[6];
        if !(10..=16).contains(&predictor_log2) {
            return Err(Error::CorruptPayload("predictor_log2 out of range"));
        }
        let dtype = DType::from_wire(src[7])?;
        if !version_supported(version, dtype) {
            return Err(Error::UnsupportedVersion(version));
        }
        let n_values = u64::from_le_bytes(src[8..16].try_into().unwrap());
        Ok(Header {
            predictor_log2,
            dtype,
            has_validity,
            has_shared_dict,
            n_values,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_version_only_for_wide_lanes() {
        let mut h = Vec::new();
        Header {
            predictor_log2: 16,
            dtype: DType::F32,
            has_validity: false,
            has_shared_dict: false,
            n_values: 3,
        }
        .write(&mut h);
        assert_eq!(h[4], VERSION);
        assert!(Header::read(&h).is_ok());
        // The same header stamped v2 describes a widened-f64 stream: rejected.
        h[4] = LEGACY_VERSION;
        assert!(matches!(
            Header::read(&h),
            Err(Error::UnsupportedVersion(2))
        ));
        // A v2 f64 header is fine (identical layout).
        h[7] = DType::F64.wire_id();
        assert!(Header::read(&h).is_ok());
        h[4] = 1;
        assert!(matches!(
            Header::read(&h),
            Err(Error::UnsupportedVersion(1))
        ));
    }

    #[test]
    fn block_budget_is_bytes() {
        assert_eq!(max_block_values(8), 128 * 1024);
        assert_eq!(max_block_values(4), 256 * 1024);
        assert_eq!(MAX_BLOCK_VALUES, max_block_values(8));
    }
}
