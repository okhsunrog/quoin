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
    /// The header declares a column type this build does not implement.
    UnsupportedDType(u8),
    /// A type-specific decompress (e.g. [`crate::decompress`], which returns
    /// `f64`) was called on a stream holding a different column type.
    DTypeMismatch,
    /// The Arrow adapter was given an array of a type it does not support yet.
    UnsupportedArrowType,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::BadMagic => write!(f, "bad magic: not an quoin stream"),
            Error::UnsupportedVersion(v) => write!(f, "unsupported format version {v}"),
            Error::Truncated => write!(f, "truncated stream"),
            Error::UnknownMode(m) => write!(f, "unknown block mode {m}"),
            Error::CorruptPayload(w) => write!(f, "corrupt payload: {w}"),
            Error::LengthMismatch { expected, got } => {
                write!(
                    f,
                    "length mismatch: expected {expected} values, decoded {got}"
                )
            }
            Error::UnsupportedDType(d) => write!(f, "unsupported column type id {d}"),
            Error::DTypeMismatch => {
                write!(f, "stream column type does not match the requested type")
            }
            Error::UnsupportedArrowType => write!(f, "unsupported Arrow array type"),
        }
    }
}

impl std::error::Error for Error {}
