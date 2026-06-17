//! FastLanes-style vertical bit-packing.
//!
//! Packs a block of 1024 values (each fitting in `width` bits) into a
//! **lane-transposed** layout: value at position `row*LANES + lane` is stored in
//! lane `lane`'s bit-stream. Because each lane is independent and every step
//! uses the *same* shift over contiguous memory, the plain scalar pack/unpack
//! loops autovectorize to AVX2/AVX-512/NEON with **no intrinsics** — the layout
//! is the optimization (FastLanes, VLDB'23). This is the substrate for fast
//! integer columns (FoR residuals, ALP digits) in the columnar work.
//!
//! Two element widths share the same 1024-bit register abstraction:
//! [`pack`]/[`unpack`] over `u32` (32 lanes × 32 rows, widths 0..=32) and
//! [`pack64`]/[`unpack64`] over `u64` (16 lanes × 64 rows, widths 0..=64) for
//! wide integer columns.
//!
//! Values must already fit in `width` bits (e.g. after frame-of-reference);
//! this layer does not do FoR or patching.

use multiversion::multiversion;

/// Values per FastLanes block.
pub(crate) const BLOCK: usize = 1024;
const LANES: usize = 32;
const ROWS: usize = BLOCK / LANES; // 32
/// Lanes for the `u64` variant: 16 lanes × 64 bits = the same 1024-bit register.
const LANES64: usize = 16;
const ROWS64: usize = BLOCK / LANES64; // 64

#[inline]
fn mask32(width: u32) -> u32 {
    if width >= 32 {
        u32::MAX
    } else {
        (1u32 << width) - 1
    }
}

#[inline]
fn mask64(width: u32) -> u64 {
    if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

/// Pack `BLOCK` values into `out` (`32 * width` words), lane-transposed.
#[multiversion(targets("x86_64+avx2", "x86_64+avx512f", "aarch64+neon"))]
pub(crate) fn pack(values: &[u32; BLOCK], width: u32, out: &mut [u32]) {
    debug_assert_eq!(out.len(), LANES * width as usize);
    for w in out.iter_mut() {
        *w = 0;
    }
    if width == 0 {
        return;
    }
    let mask = mask32(width);
    for r in 0..ROWS {
        let bit_off = r * width as usize;
        let word = bit_off / 32;
        let shift = (bit_off % 32) as u32;
        let (lo, rest) = out.split_at_mut((word + 1) * LANES);
        let lo = &mut lo[word * LANES..]; // exactly LANES wide
        let row = &values[r * LANES..r * LANES + LANES];
        for l in 0..LANES {
            lo[l] |= (row[l] & mask) << shift;
        }
        if shift + width > 32 {
            let hi = &mut rest[..LANES]; // word+1 plane
            for l in 0..LANES {
                hi[l] |= (row[l] & mask) >> (32 - shift);
            }
        }
    }
}

/// Inverse of [`pack`].
#[multiversion(targets("x86_64+avx2", "x86_64+avx512f", "aarch64+neon"))]
pub(crate) fn unpack(packed: &[u32], width: u32, out: &mut [u32; BLOCK]) {
    debug_assert_eq!(packed.len(), LANES * width as usize);
    if width == 0 {
        out.fill(0);
        return;
    }
    let mask = mask32(width);
    for r in 0..ROWS {
        let bit_off = r * width as usize;
        let word = bit_off / 32;
        let shift = (bit_off % 32) as u32;
        let lo = &packed[word * LANES..word * LANES + LANES];
        let dst = &mut out[r * LANES..r * LANES + LANES];
        if shift + width > 32 {
            let hi = &packed[(word + 1) * LANES..(word + 1) * LANES + LANES];
            for l in 0..LANES {
                dst[l] = ((lo[l] >> shift) | (hi[l] << (32 - shift))) & mask;
            }
        } else {
            for l in 0..LANES {
                dst[l] = (lo[l] >> shift) & mask;
            }
        }
    }
}

/// Pack `BLOCK` `u64` values into `out` (`16 * width` words), lane-transposed.
#[multiversion(targets("x86_64+avx2", "x86_64+avx512f", "aarch64+neon"))]
pub(crate) fn pack64(values: &[u64; BLOCK], width: u32, out: &mut [u64]) {
    debug_assert_eq!(out.len(), LANES64 * width as usize);
    for w in out.iter_mut() {
        *w = 0;
    }
    if width == 0 {
        return;
    }
    let mask = mask64(width);
    for r in 0..ROWS64 {
        let bit_off = r * width as usize;
        let word = bit_off / 64;
        let shift = (bit_off % 64) as u32;
        let (lo, rest) = out.split_at_mut((word + 1) * LANES64);
        let lo = &mut lo[word * LANES64..]; // exactly LANES64 wide
        let row = &values[r * LANES64..r * LANES64 + LANES64];
        for l in 0..LANES64 {
            lo[l] |= (row[l] & mask) << shift;
        }
        if shift + width > 64 {
            let hi = &mut rest[..LANES64]; // word+1 plane
            for l in 0..LANES64 {
                hi[l] |= (row[l] & mask) >> (64 - shift);
            }
        }
    }
}

/// Inverse of [`pack64`].
#[multiversion(targets("x86_64+avx2", "x86_64+avx512f", "aarch64+neon"))]
pub(crate) fn unpack64(packed: &[u64], width: u32, out: &mut [u64; BLOCK]) {
    debug_assert_eq!(packed.len(), LANES64 * width as usize);
    if width == 0 {
        out.fill(0);
        return;
    }
    let mask = mask64(width);
    for r in 0..ROWS64 {
        let bit_off = r * width as usize;
        let word = bit_off / 64;
        let shift = (bit_off % 64) as u32;
        let lo = &packed[word * LANES64..word * LANES64 + LANES64];
        let dst = &mut out[r * LANES64..r * LANES64 + LANES64];
        if shift + width > 64 {
            let hi = &packed[(word + 1) * LANES64..(word + 1) * LANES64 + LANES64];
            for l in 0..LANES64 {
                dst[l] = ((lo[l] >> shift) | (hi[l] << (64 - shift))) & mask;
            }
        } else {
            for l in 0..LANES64 {
                dst[l] = (lo[l] >> shift) & mask;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_all_widths() {
        let mut s = 0x1234_5678u32;
        let mut next = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            s
        };
        for width in 0..=32u32 {
            let mask = mask32(width);
            let mut values = [0u32; BLOCK];
            for v in values.iter_mut() {
                *v = next() & mask;
            }
            let mut packed = vec![0u32; LANES * width as usize];
            pack(&values, width, &mut packed);
            let mut out = [0u32; BLOCK];
            unpack(&packed, width, &mut out);
            assert_eq!(values, out, "round-trip failed at width {width}");
        }
    }

    #[test]
    fn boundary_values() {
        for width in 1..=32u32 {
            let mask = mask32(width);
            // all-max and all-zero patterns stress the straddle paths
            for fill in [0u32, mask, 0xAAAA_AAAA & mask, 1] {
                let values = [fill; BLOCK];
                let mut packed = vec![0u32; LANES * width as usize];
                pack(&values, width, &mut packed);
                let mut out = [0u32; BLOCK];
                unpack(&packed, width, &mut out);
                assert_eq!(values, out, "width {width} fill {fill:#x}");
            }
        }
    }

    #[test]
    fn pack_unpack64_all_widths() {
        let mut s = 0x1234_5678_9ABC_DEF0u64;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            s
        };
        for width in 0..=64u32 {
            let mask = mask64(width);
            let mut values = [0u64; BLOCK];
            for v in values.iter_mut() {
                *v = next() & mask;
            }
            let mut packed = vec![0u64; LANES64 * width as usize];
            pack64(&values, width, &mut packed);
            let mut out = [0u64; BLOCK];
            unpack64(&packed, width, &mut out);
            assert_eq!(values, out, "u64 round-trip failed at width {width}");
        }
    }

    #[test]
    fn boundary_values64() {
        for width in 1..=64u32 {
            let mask = mask64(width);
            for fill in [0u64, mask, 0xAAAA_AAAA_AAAA_AAAA & mask, 1] {
                let values = [fill; BLOCK];
                let mut packed = vec![0u64; LANES64 * width as usize];
                pack64(&values, width, &mut packed);
                let mut out = [0u64; BLOCK];
                unpack64(&packed, width, &mut out);
                assert_eq!(values, out, "u64 width {width} fill {fill:#x}");
            }
        }
    }
}
