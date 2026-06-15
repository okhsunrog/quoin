//! Stream decoder: walk the block frames and dispatch each to its codec.

use crate::codecs::{const_block, pred, pred_rc, raw, stride, xorz};
use crate::error::Error;
use crate::format::{Header, FRAME_HEADER_LEN, HEADER_LEN};
use crate::mode::Mode;

pub(crate) fn decompress(src: &[u8]) -> Result<Vec<f64>, Error> {
    let header = Header::read(src)?;
    let predictor_log2 = header.predictor_log2;
    let n_total = usize::try_from(header.n_values).map_err(|_| Error::Truncated)?;

    let mut bits: Vec<u64> = Vec::with_capacity(n_total);
    let mut pos = HEADER_LEN;

    while bits.len() < n_total {
        // Frame header.
        if pos + FRAME_HEADER_LEN > src.len() {
            return Err(Error::Truncated);
        }
        let mode = Mode::from_id(src[pos])?;
        let n = u32::from_le_bytes(src[pos + 1..pos + 5].try_into().unwrap()) as usize;
        let plen = u32::from_le_bytes(src[pos + 5..pos + 9].try_into().unwrap()) as usize;
        pos += FRAME_HEADER_LEN;

        // Payload.
        let end = pos.checked_add(plen).ok_or(Error::Truncated)?;
        if end > src.len() {
            return Err(Error::Truncated);
        }
        let payload = &src[pos..end];
        pos = end;

        if bits.len() + n > n_total {
            return Err(Error::CorruptPayload("block overruns declared length"));
        }

        let decoded = match mode {
            Mode::Raw => raw::decode(payload, n)?,
            Mode::Const => const_block::decode(payload, n)?,
            Mode::Stride => stride::decode(payload, n)?,
            Mode::Xorz => xorz::decode(payload, n)?,
            Mode::Pred => pred::decode(payload, n, predictor_log2)?,
            Mode::PredRc => pred_rc::decode(payload, n, predictor_log2)?,
        };
        bits.extend_from_slice(&decoded);
    }

    if bits.len() != n_total {
        return Err(Error::LengthMismatch { expected: n_total, got: bits.len() });
    }

    Ok(bits.into_iter().map(f64::from_bits).collect())
}
