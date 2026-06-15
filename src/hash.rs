//! FCM predictor hash — the first "hot, irregular" kernel.
//!
//! The original `fc` indexes its predictor tables with `_mm_crc32_u64`, the
//! SSE4.2 CRC32C instruction. That is exactly the kind of primitive portable
//! SIMD abstractions don't model, so we take the raw-intrinsic route the
//! 2025 SIMD survey recommends for C ports: a `core::arch` fast path gated by
//! runtime feature detection, plus a bit-exact software fallback so a stream
//! hashed on one machine decodes identically on another.

/// Golden-ratio seed (low 32 bits), as in `fc`'s `0x9e3779b97f4a7c15`.
pub(crate) const HASH_SEED: u32 = 0x9e37_79b9;

/// A hash step: `(running_crc, value) -> running_crc`.
pub(crate) type HashFn = fn(u32, u64) -> u32;

/// Bit-exact software CRC32C (Castagnoli, reflected poly `0x82F63B78`). Matches
/// the hardware `crc32` instruction byte-for-byte over the 8 bytes of `value`.
#[inline]
pub(crate) fn crc32c_u64_sw(crc: u32, value: u64) -> u32 {
    let mut crc = crc;
    let mut v = value;
    for _ in 0..8 {
        crc ^= (v & 0xff) as u32;
        v >>= 8;
        for _ in 0..8 {
            let mask = 0u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0x82F6_3B78 & mask);
        }
    }
    crc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn crc32c_u64_hw(crc: u32, value: u64) -> u32 {
    use core::arch::x86_64::_mm_crc32_u64;
    // Safe to call here: the enclosing `#[target_feature]` guarantees SSE4.2,
    // and on Rust >= 1.87 the intrinsic itself is no longer `unsafe`.
    _mm_crc32_u64(u64::from(crc), value) as u32
}

#[cfg(target_arch = "x86_64")]
fn crc32c_u64_hw_safe(crc: u32, value: u64) -> u32 {
    // SAFETY: only installed by `best_hash_fn` after `is_x86_feature_detected!`
    // confirms SSE4.2 at runtime.
    unsafe { crc32c_u64_hw(crc, value) }
}

/// Pick the fastest available hash step once, so per-value calls don't re-run
/// feature detection (the overhead the SIMD survey warns about).
pub(crate) fn best_hash_fn() -> HashFn {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("sse4.2") {
            return crc32c_u64_hw_safe;
        }
    }
    crc32c_u64_sw
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sw_matches_hw() {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("sse4.2") {
            for &v in &[0u64, 1, 42, 255, 256, u64::MAX, 0x0102_0304_0506_0708] {
                assert_eq!(
                    crc32c_u64_sw(HASH_SEED, v),
                    crc32c_u64_hw_safe(HASH_SEED, v),
                    "sw/hw CRC32C disagree on {v:#x}"
                );
            }
        }
    }

    #[test]
    fn deterministic_and_spread() {
        // Sanity: distinct inputs mostly produce distinct table indices.
        let h = best_hash_fn();
        let mask = (1u32 << 16) - 1;
        let a = h(HASH_SEED, 1) & mask;
        let b = h(HASH_SEED, 2) & mask;
        assert_ne!(a, b);
        assert_eq!(h(HASH_SEED, 1), h(HASH_SEED, 1));
    }
}
