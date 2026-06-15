//! Lane-wise transforms — the "autovectorization + multiversion" tier.
//!
//! These are pure data-parallel maps with no gather and no cross-lane
//! dependency on the forward pass, so we let LLVM autovectorize plain scalar
//! Rust and use the `multiversion` crate to emit per-CPU clones (baseline /
//! SSE4.2 / AVX2 / NEON) selected at runtime. This is the alternative to
//! pulling in a portable-SIMD crate like `macerator` for simple kernels.

use multiversion::multiversion;

/// In-place forward delta over `u64` bit patterns: `v[i] -= v[i-1]` (wrapping).
///
/// Walked high→low so each step still reads the *original* predecessor; the
/// shifted-slice subtraction autovectorizes cleanly.
#[multiversion(targets("x86_64+avx2", "x86_64+sse4.2", "aarch64+neon"))]
pub(crate) fn delta_encode_u64(v: &mut [u64]) {
    for i in (1..v.len()).rev() {
        v[i] = v[i].wrapping_sub(v[i - 1]);
    }
}

/// Inverse of [`delta_encode_u64`]. Prefix sum — inherently sequential, so it
/// stays scalar. Roadmap building block for the upcoming DELTA modes.
#[allow(dead_code)]
pub(crate) fn delta_decode_u64(v: &mut [u64]) {
    for i in 1..v.len() {
        v[i] = v[i].wrapping_add(v[i - 1]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_roundtrips() {
        let original: Vec<u64> = (0..1000).map(|i| (i * i) as u64 ^ 0xdead_beef).collect();
        let mut buf = original.clone();
        delta_encode_u64(&mut buf);
        delta_decode_u64(&mut buf);
        assert_eq!(buf, original);
    }
}
