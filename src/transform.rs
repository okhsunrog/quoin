//! Byte-transpose kernels — the lane-wise tier, with `multiversion` dispatch.
//!
//! Transposing `n` `w`-byte values (`w` = the lane width, 4 or 8) from
//! array-of-structs to struct-of-arrays (`w` contiguous byte-planes) is a
//! genuinely lane-wise map, so it carries a `#[multiversion]` attribute: the
//! binary ships baseline / SSE4.2 / AVX2 / NEON clones plus a runtime
//! dispatcher, giving "fast on any CPU without a rebuild" *for free* once the
//! kernel vectorizes.
//!
//! Honest status (measured via `objdump`): LLVM does **not** currently
//! autovectorize this — a byte-matrix transpose is a strided `u8` gather that
//! needs explicit shuffle intrinsics (`pshufb`/`punpck`), so the AVX2 clone is
//! presently ~scalar. That's fine for now: the transpose is one O(n) pass per
//! block while entropy coding dominates the block's time, so a SIMD transpose
//! would barely move end-to-end throughput. The `multiversion` wiring stays so
//! an explicit-SIMD rewrite (core::arch / std::simd / macerator) drops in later
//! with dispatch already done. See `benches/kernels.rs` to track its speed.

use multiversion::multiversion;

/// AoS → SoA: regroup `n` `w`-byte values into `w` contiguous byte-planes, so a
/// low-entropy byte position (e.g. the sign/exponent bytes of similar floats)
/// becomes a compressible run for the downstream entropy coder.
#[multiversion(targets("x86_64+avx2", "x86_64+sse4.2", "aarch64+neon"))]
pub(crate) fn byte_transpose(src: &[u8], n: usize, w: usize, dst: &mut [u8]) {
    debug_assert_eq!(src.len(), n * w);
    debug_assert_eq!(dst.len(), n * w);
    // Plane-major: contiguous writes, stride-w reads. More autovectorizable
    // than a scatter (the compiler can hoist the strided gather).
    for plane in 0..w {
        let out = &mut dst[plane * n..plane * n + n];
        for (i, o) in out.iter_mut().enumerate() {
            *o = src[i * w + plane];
        }
    }
}

/// Inverse of [`byte_transpose`]: SoA → AoS.
#[multiversion(targets("x86_64+avx2", "x86_64+sse4.2", "aarch64+neon"))]
pub(crate) fn byte_untranspose(src: &[u8], n: usize, w: usize, dst: &mut [u8]) {
    debug_assert_eq!(src.len(), n * w);
    debug_assert_eq!(dst.len(), n * w);
    for (i, chunk) in dst.chunks_exact_mut(w).enumerate() {
        for (plane, b) in chunk.iter_mut().enumerate() {
            *b = src[plane * n + i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_roundtrips() {
        for w in [4usize, 8] {
            let n = 1000usize;
            let mut s = 0x9e3779b9u64;
            let aos: Vec<u8> = (0..n * w)
                .map(|_| {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                    (s >> 33) as u8
                })
                .collect();
            let mut soa = vec![0u8; n * w];
            byte_transpose(&aos, n, w, &mut soa);
            // Plane 0 holds byte 0 of every value.
            assert!(soa[..n].iter().enumerate().all(|(i, &b)| b == aos[i * w]));
            let mut back = vec![0u8; n * w];
            byte_untranspose(&soa, n, w, &mut back);
            assert_eq!(back, aos);
        }
    }
}
