//! Stream decoder. Two phases: scan the frame table sequentially (frame
//! lengths are variable, so boundaries must be found in order), then decode the
//! located blocks — in parallel with the `parallel` feature, since each block
//! is self-contained.

use crate::codecs::{const_block, linear, lz, pred, raw, stride, transpose, xorz};
use crate::entropy::decode_residuals;
use crate::error::Error;
use crate::format::{FRAME_HEADER_LEN, HEADER_LEN, Header, MAX_BLOCK_VALUES};
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

pub(crate) fn decompress(src: &[u8]) -> Result<Vec<f64>, Error> {
    let header = Header::read(src)?;
    let predictor_log2 = header.predictor_log2;
    let n_total = usize::try_from(header.n_values).map_err(|_| Error::Truncated)?;

    // Phase 1: scan frame boundaries (sequential, cheap, fully validated).
    let mut frames: Vec<Frame<'_>> = Vec::new();
    let mut counted = 0usize;
    let mut pos = HEADER_LEN;
    while counted < n_total {
        if pos + FRAME_HEADER_LEN > src.len() {
            return Err(Error::Truncated);
        }
        let mode = Mode::from_id(src[pos])?;
        let n = u32::from_le_bytes(src[pos + 1..pos + 5].try_into().unwrap()) as usize;
        let plen = u32::from_le_bytes(src[pos + 5..pos + 9].try_into().unwrap()) as usize;
        pos += FRAME_HEADER_LEN;

        // Reject blocks larger than the encoder ever emits — bounds per-block
        // allocation and prevents a tiny frame from claiming a huge value count.
        if n > MAX_BLOCK_VALUES {
            return Err(Error::CorruptPayload("block value count exceeds maximum"));
        }

        let end = pos.checked_add(plen).ok_or(Error::Truncated)?;
        if end > src.len() {
            return Err(Error::Truncated);
        }
        if counted + n > n_total {
            return Err(Error::CorruptPayload("block overruns declared length"));
        }
        frames.push(Frame {
            mode,
            n,
            payload: &src[pos..end],
        });
        counted += n;
        pos = end;
    }
    if counted != n_total {
        return Err(Error::LengthMismatch {
            expected: n_total,
            got: counted,
        });
    }

    // Phase 2: decode blocks (parallel when enabled), then concatenate in order.
    let decoded = decode_frames(&frames, predictor_log2)?;

    let mut bits: Vec<u64> = Vec::with_capacity(n_total);
    for block in &decoded {
        bits.extend_from_slice(block);
    }
    Ok(bits.into_iter().map(f64::from_bits).collect())
}

#[cfg(feature = "parallel")]
fn decode_frames(frames: &[Frame<'_>], predictor_log2: u8) -> Result<Vec<Vec<u64>>, Error> {
    use rayon::prelude::*;
    frames
        .par_iter()
        .map(|f| decode_frame(f, predictor_log2))
        .collect()
}

#[cfg(not(feature = "parallel"))]
fn decode_frames(frames: &[Frame<'_>], predictor_log2: u8) -> Result<Vec<Vec<u64>>, Error> {
    frames
        .iter()
        .map(|f| decode_frame(f, predictor_log2))
        .collect()
}

fn decode_frame(f: &Frame<'_>, predictor_log2: u8) -> Result<Vec<u64>, Error> {
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
            let lz_bytes = decode_residuals(payload, n.saturating_mul(16) + 1024)?;
            lz::decode(&lz_bytes, n)?
        }
        Mode::ByteTranspose => {
            let soa = decode_residuals(payload, n.saturating_mul(8) + 16)?;
            transpose::decode(&soa, n)?
        }
    })
}
