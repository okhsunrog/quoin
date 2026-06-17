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
    /// LZ77 over the block bytes, token stream entropy-coded.
    Lz = 4,
    /// Byte-plane transpose (AoS→SoA), then entropy-coded.
    ByteTranspose = 11,
    /// Values are integer multiples of 1/scale; store the integers (verified).
    FloatMult = 17,
    /// Second-order integer delta of bit patterns (zigzag), residuals entropy-coded.
    OrderedDelta = 7,
    /// Second-order linear extrapolation in float space, residuals entropy-coded.
    Delta2 = 20,
    /// DFCM (differential FCM) predictor residuals, range-coded.
    Pred2 = 25,
    /// Second-order linear prediction storing the exact float residual (verified
    /// lossless). Nails polynomial/smooth data (constant second difference).
    DeltaDp = 45,
    /// FCM predictor residuals, range-coded (order-1 adaptive model).
    PredRc = 29,
    /// Frame-of-reference + FastLanes bit-packing over 1024-value sub-blocks.
    ForBitpack = 50,
    /// ALP: scaled-integer encoding of decimal-like doubles (FoR+bitpack digits
    /// + exceptions).
    Alp = 9,
    /// First-order delta + FoR + bit-packing (Parquet DELTA_BINARY_PACKED).
    DeltaBitpack = 51,
    /// ALP-RD: real-double split — dictionary the high bits, bit-pack the low.
    AlpRd = 52,
    /// Dictionary: distinct values → bit-packed codes (low-cardinality columns).
    Dict = 19,
    /// Run-length encoding: (value, run-length) pairs (grouped/repeated columns).
    Rle = 53,
    /// Vendored pco (pcodec) numeric backend: latent decomposition + bin-packing
    /// + ANS. Heavyweight ratio mode, gated to `High`/`Max`.
    Pco = 54,
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
            4 => Mode::Lz,
            5 => Mode::Raw,
            11 => Mode::ByteTranspose,
            17 => Mode::FloatMult,
            7 => Mode::OrderedDelta,
            20 => Mode::Delta2,
            25 => Mode::Pred2,
            45 => Mode::DeltaDp,
            29 => Mode::PredRc,
            50 => Mode::ForBitpack,
            9 => Mode::Alp,
            51 => Mode::DeltaBitpack,
            52 => Mode::AlpRd,
            19 => Mode::Dict,
            53 => Mode::Rle,
            54 => Mode::Pco,
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
        50 => "FOR_BITPACK", // quoin extension (fc reserves 50-63)
        51 => "DELTA_BITPACK",
        52 => "ALP_RD",
        53 => "RLE",
        54 => "PCO",
        _ => "?",
    }
}
