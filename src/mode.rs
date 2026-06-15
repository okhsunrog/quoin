use crate::error::Error;

/// Block codec identifiers. Numeric values match the original `fc` mode IDs so
/// that a future wire-compatible mode can reuse them. Only the variants listed
/// here are implemented today; the rest of the 0..=49 space is reserved (see
/// [`mode_name`]).
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Mode {
    /// FCM hash predictor, XOR residuals, LEB128-coded.
    Pred = 0,
    /// All values in the block are identical.
    Const = 1,
    /// Arithmetic progression in `u64` bit-pattern space.
    Stride = 2,
    /// XOR with previous value, LEB128-coded.
    Xorz = 3,
    /// Verbatim little-endian `u64` words. Always-available fallback.
    Raw = 5,
}

impl Mode {
    pub(crate) fn id(self) -> u8 {
        self as u8
    }

    pub(crate) fn from_id(id: u8) -> Result<Mode, Error> {
        Ok(match id {
            0 => Mode::Pred,
            1 => Mode::Const,
            2 => Mode::Stride,
            3 => Mode::Xorz,
            5 => Mode::Raw,
            other => return Err(Error::UnknownMode(other)),
        })
    }
}

/// Human-readable name for a mode ID, covering the full original `fc` table for
/// diagnostics and parity. Returns `"?"` for unused IDs.
pub fn mode_name(id: u8) -> &'static str {
    match id {
        0 => "PRED",
        1 => "CONST",
        2 => "STRIDE",
        3 => "XORZ",
        4 => "LZ",
        5 => "RAW",
        6 => "FLOAT32",
        7 => "ORDERED_DELTA",
        8 => "FUZZY_STRIDE",
        9 => "ALP",
        10 => "TRAILING_ZERO_BP",
        11 => "BYTE_TRANSPOSE",
        13 => "XOR128",
        15 => "LSB_STRIP",
        16 => "LOOKBACK_DELTA",
        17 => "FLOAT_MULT",
        18 => "FCM_RLE",
        19 => "DICT",
        20 => "DELTA2",
        21 => "BITPLANE",
        22 => "INT_MULT",
        23 => "CONV1",
        24 => "PRED_TANS",
        25 => "PRED2",
        26 => "PRED_ADAPTIVE",
        27 => "VITERBI",
        28 => "DELTA_BINNED",
        29 => "PRED_RC",
        30 => "PRED_INTERLEAVED",
        31 => "BWT",
        32 => "LZ_DICT",
        33 => "CONV_N",
        34 => "SIGN_CONV",
        35 => "CONV_DOUBLE",
        36 => "MTF_LZ",
        37 => "CONV_DOUBLE_BP",
        38 => "CONV_N_BINNED",
        39 => "PRED_SIMD_INTERLEAVED",
        40 => "FUZZY_STRIDE_ANS",
        41 => "PAQ_MIXER",
        42 => "PAQ4_MIXER",
        43 => "BWT_MTF_TANS",
        44 => "PRED4",
        45 => "DELTA_DP_BINNED",
        46 => "CONV_N_DP_BINNED",
        47 => "ELF",
        48 => "LZ_SPLIT",
        49 => "BWT_MTF_RC",
        _ => "?",
    }
}
