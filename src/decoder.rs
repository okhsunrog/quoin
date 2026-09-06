//! Stream decoder. Two phases: scan the frame table sequentially (frame
//! lengths are variable, so boundaries must be found in order), then decode the
//! located blocks — in parallel with the `parallel` feature, since each block
//! is self-contained. Generic over the [`Lane`]: the caller picks `u32` or
//! `u64` from the header's column type, and every frame decodes straight into
//! that lane.

use crate::codecs::{
    alp, alp_rd, const_block, delta_bitpack, dict, float_mult, for_bitpack, linear, lz, pred, raw,
    rle, stride, transpose, xorz,
};
use crate::dtype::DType;
use crate::entropy::decode_residuals;
use crate::error::Error;
use crate::format::{FRAME_HEADER_LEN, HEADER_LEN, Header, max_block_values};
use crate::lane::Lane;
use crate::mode::Mode;

/// Upper bound on a predictor residual stream: at most 10 LEB128 bytes/value.
/// Guards range-coder decode against a corrupt length field.
fn resid_bound(n: usize) -> usize {
    n.saturating_mul(10) + 16
}

/// A located block: its mode, value count, and payload slice.
struct Frame<'a> {
    mode: Mode,
    n: usize,
    payload: &'a [u8],
}

/// A decoded column lane: type, lane words (null slots zero), and an optional
/// validity bitmap.
pub(crate) type DecodedLane<L> = (DType, Vec<L>, Option<Vec<u8>>);

fn take<'a>(src: &'a [u8], pos: &mut usize, len: usize) -> Result<&'a [u8], Error> {
    let end = pos.checked_add(len).ok_or(Error::Truncated)?;
    let out = src.get(*pos..end).ok_or(Error::Truncated)?;
    *pos = end;
    Ok(out)
}

/// Decode a stream to its lane words, the column type, and an optional
/// validity bitmap. Null slots in the returned lane are `0`; callers reinterpret
/// the lane per [`DType`] (e.g. `f32::from_bits`). The header's type must be
/// one of `L`'s (a 4-byte type for `u32`, 8-byte for `u64`).
pub(crate) fn decompress_lane<L: Lane>(src: &[u8]) -> Result<DecodedLane<L>, Error> {
    let header = Header::read(src)?;
    if header.dtype.lane_bytes() != L::BYTES {
        return Err(Error::DTypeMismatch);
    }
    let predictor_log2 = header.predictor_log2;
    let logical_n = usize::try_from(header.n_values).map_err(|_| Error::Truncated)?;
    let max_block = max_block_values(L::BYTES);

    // The validity bitmap (if any) precedes the value frames; the frames hold
    // only the valid (compacted) values, so they decode to `n_valid`, not the
    // logical length.
    let mut pos = HEADER_LEN;
    let validity = if header.has_validity {
        let vlen = usize::try_from(crate::varint::read_u64(src, &mut pos)?)
            .map_err(|_| Error::CorruptPayload("validity length too large"))?;
        let vblob = take(src, &mut pos, vlen)?;
        Some(crate::validity::decode(vblob, logical_n)?)
    } else {
        None
    };
    let n_total = match &validity {
        Some(bm) => crate::validity::count_valid(bm, logical_n),
        None => logical_n,
    };

    // Shared-dictionary preamble (if any) follows the validity section: the
    // column-wide value table, decoded once and shared read-only by the frames.
    let shared: Option<Vec<L>> = if header.has_shared_dict {
        let plen = usize::try_from(crate::varint::read_u64(src, &mut pos)?)
            .map_err(|_| Error::CorruptPayload("shared dict preamble too large"))?;
        let blob = take(src, &mut pos, plen)?;
        Some(dict::decode_shared_preamble::<L>(blob, n_total)?)
    } else {
        None
    };

    // Phase 1: scan frame boundaries (sequential, cheap, fully validated).
    let mut frames: Vec<Frame<'_>> = Vec::new();
    let mut counted = 0usize;
    while counted < n_total {
        let fh = take(src, &mut pos, FRAME_HEADER_LEN)?;
        let mode = Mode::from_id(fh[0])?;
        let n = u32::from_le_bytes(fh[1..5].try_into().unwrap()) as usize;
        let plen = u32::from_le_bytes(fh[5..9].try_into().unwrap()) as usize;

        // Reject blocks larger than the encoder ever emits — bounds per-block
        // allocation and prevents a tiny frame from claiming a huge value count.
        if n > max_block {
            return Err(Error::CorruptPayload("block value count exceeds maximum"));
        }

        let payload = take(src, &mut pos, plen)?;
        counted = counted
            .checked_add(n)
            .ok_or(Error::CorruptPayload("block overruns declared length"))?;
        if counted > n_total {
            return Err(Error::CorruptPayload("block overruns declared length"));
        }
        frames.push(Frame { mode, n, payload });
    }
    if counted != n_total {
        return Err(Error::LengthMismatch {
            expected: n_total,
            got: counted,
        });
    }

    // Phase 2: decode blocks (parallel when enabled), then concatenate in order.
    let mut decoded = decode_frames::<L>(&frames, predictor_log2, header.dtype, shared.as_deref())?;

    // Single block (common for small columns / a large fixed `block_size`): move
    // its buffer out instead of copying it into a fresh lane.
    let bits: Vec<L> = if decoded.len() == 1 {
        decoded.pop().unwrap()
    } else {
        let mut bits: Vec<L> = Vec::new();
        bits.try_reserve_exact(n_total)
            .map_err(|_| Error::CorruptPayload("decoded column too large"))?;
        for block in &decoded {
            bits.extend_from_slice(block);
        }
        bits
    };
    // Scatter the valid values back into the logical positions (null slots → 0).
    let lane = match &validity {
        Some(bm) => crate::validity::scatter(&bits, bm, logical_n)?,
        None => bits,
    };
    Ok((header.dtype, lane, validity))
}

#[cfg(feature = "parallel")]
fn decode_frames<L: Lane>(
    frames: &[Frame<'_>],
    predictor_log2: u8,
    dtype: DType,
    shared: Option<&[L]>,
) -> Result<Vec<Vec<L>>, Error> {
    use rayon::prelude::*;
    frames
        .par_iter()
        .map(|f| decode_frame(f, predictor_log2, dtype, shared))
        .collect()
}

#[cfg(not(feature = "parallel"))]
fn decode_frames<L: Lane>(
    frames: &[Frame<'_>],
    predictor_log2: u8,
    dtype: DType,
    shared: Option<&[L]>,
) -> Result<Vec<Vec<L>>, Error> {
    frames
        .iter()
        .map(|f| decode_frame(f, predictor_log2, dtype, shared))
        .collect()
}

/// Decode one block payload of `mode` (diagnostics / calibration; no shared
/// dictionary).
pub(crate) fn decode_block<L: Lane>(
    mode: Mode,
    payload: &[u8],
    n: usize,
    predictor_log2: u8,
    dtype: DType,
) -> Result<Vec<L>, Error> {
    decode_frame(&Frame { mode, n, payload }, predictor_log2, dtype, None)
}

fn decode_frame<L: Lane>(
    f: &Frame<'_>,
    predictor_log2: u8,
    dtype: DType,
    shared: Option<&[L]>,
) -> Result<Vec<L>, Error> {
    let (payload, n) = (f.payload, f.n);
    Ok(match f.mode {
        Mode::Raw => raw::decode(payload, n)?,
        Mode::Const => const_block::decode(payload, n)?,
        Mode::Stride => stride::decode(payload, n)?,
        Mode::Xorz => xorz::decode(payload, n)?,
        Mode::Pred => pred::decode(payload, n, predictor_log2)?,
        Mode::PredRc => {
            let resid = decode_residuals(payload, resid_bound(n))?;
            pred::decode(&resid, n, predictor_log2)?
        }
        Mode::Pred2 => {
            let resid = decode_residuals(payload, resid_bound(n))?;
            pred::dfcm_decode(&resid, n, predictor_log2)?
        }
        Mode::Delta2 => {
            let resid = decode_residuals(payload, resid_bound(n))?;
            linear::decode(&resid, n)?
        }
        Mode::DeltaDp => {
            let resid = decode_residuals(payload, resid_bound(n))?;
            linear::dp_decode(&resid, n)?
        }
        Mode::OrderedDelta => {
            let resid = decode_residuals(payload, resid_bound(n))?;
            linear::idelta2_decode(&resid, n)?
        }
        Mode::Lz => {
            let lz_bytes = decode_residuals(payload, n.saturating_mul(2 * L::BYTES) + 1024)?;
            lz::decode(&lz_bytes, n)?
        }
        Mode::ByteTranspose => {
            let soa = decode_residuals(payload, n.saturating_mul(L::BYTES) + 16)?;
            transpose::decode(&soa, n)?
        }
        Mode::FloatMult => float_mult::decode(payload, n)?,
        Mode::ForBitpack => for_bitpack::decode(payload, n, dtype.signed())?,
        Mode::Alp => alp::decode(payload, n)?,
        Mode::DeltaBitpack => delta_bitpack::decode(payload, n)?,
        Mode::AlpRd => alp_rd::decode(payload, n)?,
        Mode::Dict => dict::decode(payload, n)?,
        Mode::DictShared => {
            let table = shared.ok_or(Error::CorruptPayload(
                "shared-dict frame without a preamble",
            ))?;
            dict::decode_shared(payload, n, table)?
        }
        Mode::Rle => rle::decode(payload, n)?,
        Mode::Pco => L::pco_decompress(payload, n, dtype)?,
    })
}
