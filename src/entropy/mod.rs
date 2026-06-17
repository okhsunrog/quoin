//! Entropy coders shared by the residual-coding modes.
//!
//! * [`rc`] — binary range coder, adaptive order-1 byte model. Best ratio,
//!   slow decode (eight model updates per byte).
//! * [`tans`] — table ANS. Slightly weaker ratio, much faster decode.
//!
//! [`code_residuals`]/[`decode_residuals`] run both on a predictor residual
//! stream and keep the smaller, tagging the choice in a leading byte so the
//! predictor modes don't each need two mode IDs.
//!
//! (A "prefer tANS within N% for faster decode" policy was tried and reverted:
//! at safe margins it's a no-op because RC's order-1 model beats order-0 tANS
//! by >6% on the byte-transpose streams the noisy datasets use, and larger
//! margins cost real ratio. Faster decode on those would need a faster range
//! decoder or an order-1 tANS — deferred.)

pub(crate) mod rans;
pub(crate) mod rc;
pub(crate) mod tans;

use crate::error::Error;
use crate::varint;

const TAG_RC: u8 = 0;
const TAG_TANS: u8 = 1; // legacy: still decoded, no longer produced
const TAG_RANS: u8 = 2;
/// Cascade tag: the residual was LZ-compressed, then the LZ stream entropy-coded.
/// Payload after the tag is `varint(residual_len) ++ entropy_pick(lz_stream)`.
const TAG_LZ: u8 = 3;

// Relative decode-cost weights per coder (higher = slower decode). The range
// coder is bit-serial (~8 model updates/byte); rANS is four interleaved
// table-lookup chains. Same penalty scale as mode selection.
const W_RC: u64 = 30;
const W_RANS: u64 = 3;

/// `λ` at or above this keeps the entropy stage **rANS-only** (no bit-serial
/// range coder), for fast decode. `Level::Balanced` sits here (`λ = 2`);
/// `High`/`Max` (`λ ≤ 1`) admit the range coder. Deriving the policy from the
/// single decode-cost knob avoids threading a separate flag everywhere.
pub(crate) const RC_LAMBDA_CUTOFF: u64 = 2;

/// Entropy-code `bytes`, choosing the coder by the decode-cost policy in `λ`:
/// above [`RC_LAMBDA_CUTOFF`] only rANS is considered (fast decode); below it the
/// bit-serial range coder competes, chosen via `argmin(size + λ·W·n)`. The range
/// coder is always available as the fall-back when rANS can't compress. Output is
/// `[TAG_RC|TAG_RANS] ++ coded`; never emits [`TAG_LZ`], so it is safe on the
/// inner LZ stream of a cascade without recursion.
fn entropy_pick(bytes: &[u8], lambda: u64) -> Vec<u8> {
    let rans = rans::compress_bytes(bytes);

    // rANS-only tiers (Balanced): take rANS when it compressed; the range coder
    // is used only when rANS can't (incompressible — and that block then loses
    // the mode competition to RAW anyway), so no fast-decode promise is broken.
    if lambda >= RC_LAMBDA_CUTOFF {
        if let Some(r) = rans {
            return tagged(TAG_RANS, &r);
        }
        return tagged(TAG_RC, &rc::compress_bytes(bytes));
    }

    // High/Max: cost-aware choice between the range coder and rANS.
    let dec_len = bytes.len() as u64;
    let penalty = |w: u64| (lambda.saturating_mul(w).saturating_mul(dec_len) >> 8) as usize;
    let mut best_tag = TAG_RC;
    let mut best = rc::compress_bytes(bytes);
    let best_score = best.len() + penalty(W_RC);
    if let Some(r) = rans
        && r.len() + penalty(W_RANS) < best_score
    {
        best_tag = TAG_RANS;
        best = r;
    }
    tagged(best_tag, &best)
}

fn tagged(tag: u8, coded: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(coded.len() + 1);
    out.push(tag);
    out.extend_from_slice(coded);
    out
}

pub(crate) fn code_residuals(residuals: &[u8], lambda: u64, allow_lz: bool) -> Vec<u8> {
    let basic = entropy_pick(residuals, lambda);

    // LZ cascade — `Max` only (`allow_lz`): run our byte LZ over the *transformed*
    // residual, then entropy-code the result. This is our own "final stage": it
    // captures repeats a transform leaves behind (periodic / recurring sequences)
    // that the order-1 model alone misses — without an external compressor. Kept
    // only if strictly smaller. (Gated explicitly, not by `λ == 0`, because `High`
    // also scores at `λ = 0` but must not run the cascade.)
    if allow_lz && residuals.len() > 64 {
        let lz_stream = crate::codecs::lz::lz_compress(residuals);
        if lz_stream.len() < residuals.len() {
            let inner = entropy_pick(&lz_stream, lambda);
            let mut cand = Vec::with_capacity(inner.len() + 11);
            cand.push(TAG_LZ);
            varint::write_u64(&mut cand, residuals.len() as u64);
            cand.extend_from_slice(&inner);
            if cand.len() < basic.len() {
                return cand;
            }
        }
    }
    basic
}

/// Decode an `[TAG_RC|TAG_TANS|TAG_RANS] ++ coded` stream (no LZ cascade).
fn decode_entropy_only(payload: &[u8], max_len: usize) -> Result<Vec<u8>, Error> {
    let (&tag, rest) = payload.split_first().ok_or(Error::Truncated)?;
    match tag {
        TAG_RC => rc::decompress_bytes(rest, max_len),
        TAG_TANS => tans::decompress_bytes(rest, max_len),
        TAG_RANS => rans::decompress_bytes(rest, max_len),
        _ => Err(Error::CorruptPayload("unknown entropy tag")),
    }
}

/// Inverse of [`code_residuals`]. `max_len` bounds the decoded length.
pub(crate) fn decode_residuals(payload: &[u8], max_len: usize) -> Result<Vec<u8>, Error> {
    if payload.first() == Some(&TAG_LZ) {
        let mut pos = 0usize;
        let orig_len = varint::read_u64(&payload[1..], &mut pos)? as usize;
        if orig_len > max_len {
            return Err(Error::CorruptPayload("lz cascade length"));
        }
        // The LZ stream is at most ~the residual plus token overhead; bound it
        // generously so a corrupt declared length can't over-allocate.
        let lz_bound = orig_len.saturating_mul(2).saturating_add(64);
        let lz_stream = decode_entropy_only(&payload[1 + pos..], lz_bound)?;
        return crate::codecs::lz::lz_decompress(&lz_stream, orig_len);
    }
    decode_entropy_only(payload, max_len)
}

#[cfg(test)]
mod cascade_tests {
    use super::*;

    #[test]
    fn lz_cascade_triggers_and_roundtrips() {
        // Long-range repeats of a high-entropy pattern: the order-1 model can't
        // capture the repeat (the bytes look random locally) but LZ can, so the
        // Max cascade (λ = 0) should win and kick in. A 1 KiB pseudo-random block
        // repeated many times.
        let block: Vec<u8> = (0..1024u32).map(|i| (i.wrapping_mul(2_654_435_761) >> 16) as u8).collect();
        let mut data = Vec::new();
        for _ in 0..50 {
            data.extend_from_slice(&block);
        }
        let coded = code_residuals(&data, 0, true);
        assert_eq!(coded[0], TAG_LZ, "long-range repeats should use the LZ cascade");
        assert!(coded.len() < data.len() / 4, "cascade should compress hard");
        assert_eq!(decode_residuals(&coded, data.len()).unwrap(), data);

        // High-entropy residual (LCG, no short repeats): the cascade must not be
        // selected, and round-trips.
        let mut s = 0x1234_5678u64;
        let noise: Vec<u8> = (0..4000)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (s >> 33) as u8
            })
            .collect();
        let c2 = code_residuals(&noise, 0, true);
        assert_ne!(c2[0], TAG_LZ);
        assert_eq!(decode_residuals(&c2, noise.len()).unwrap(), noise);

        // Balanced (λ > 0) never runs the cascade, but still round-trips.
        let c3 = code_residuals(&data, 1, false);
        assert_ne!(c3[0], TAG_LZ, "cascade is Max-only");
        assert_eq!(decode_residuals(&c3, data.len()).unwrap(), data);
    }
}
