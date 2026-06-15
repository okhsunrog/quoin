use core::fmt;

/// Errors returned by [`crate::decompress`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The stream does not start with the expected magic bytes.
    BadMagic,
    /// The stream header declares a format version this build cannot read.
    UnsupportedVersion(u8),
    /// The stream ended in the middle of a header, frame, or payload.
    Truncated,
    /// A block declared a mode ID this build does not implement.
    UnknownMode(u8),
    /// A payload was malformed for its declared mode.
    CorruptPayload(&'static str),
    /// The decoded value count did not match the header.
    LengthMismatch { expected: usize, got: usize },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::BadMagic => write!(f, "bad magic: not an fp-compressor stream"),
            Error::UnsupportedVersion(v) => write!(f, "unsupported format version {v}"),
            Error::Truncated => write!(f, "truncated stream"),
            Error::UnknownMode(m) => write!(f, "unknown block mode {m}"),
            Error::CorruptPayload(w) => write!(f, "corrupt payload: {w}"),
            Error::LengthMismatch { expected, got } => {
                write!(f, "length mismatch: expected {expected} values, decoded {got}")
            }
        }
    }
}

impl std::error::Error for Error {}
