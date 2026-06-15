//! tANS (table-driven asymmetric numeral systems) for 8-bit symbols, ported
//! from the original `fc`'s `fc_ans_*`. Table size 1024, state in [1024, 2048).
//! Encode is LIFO (reverse), decode is forward.
//!
//! Decode is markedly faster than the binary range coder (one table lookup +
//! one bit read per byte vs. eight model updates), which is why `fc` leans on
//! it for its common predictor-residual modes. Here it competes with `rc` per
//! block; the encoder keeps whichever is smaller.
//!
//! Self-describing wire format:
//! ```text
//!   varint(src_len)
//!   [num_distinct - 1]            (u8)
//!   which_syms[num_distinct]      (u8 each)
//!   weights[num_distinct]         (u16 LE each)
//!   final_state                   (u32 LE, in [1024, 2048))
//!   renorm bitstream              (MSB-first)
//! ```

use crate::bitio::{BitReader, BitWriter};
use crate::error::Error;
use crate::varint;

const SIZE_LOG: u32 = 10;
const TABLE_SIZE: u32 = 1 << SIZE_LOG; // 1024
const MAX_SYMS: usize = 256;

/// Normalize symbol counts to weights summing to exactly `TABLE_SIZE`.
fn quantize(counts: &[u32], which: &[u8], total: u64) -> Option<Vec<u16>> {
    let num = which.len();
    if num == 0 || num > MAX_SYMS {
        return None;
    }
    let mult = if total > 0 { TABLE_SIZE as f32 / total as f32 } else { 0.0 };
    let mut desired = vec![0.0f32; num];
    let mut total_desired = 0.0f32;
    for (i, &sym) in which.iter().enumerate() {
        let v = (counts[sym as usize] as f32 * mult - 1.0).max(0.0);
        desired[i] = v;
        total_desired += v;
    }
    let surplus_mult = if total_desired > 0.0 {
        (TABLE_SIZE as f32 - num as f32) / total_desired
    } else {
        0.0
    };
    let mut weights = vec![0u16; num];
    let mut sum: i64 = 0;
    for i in 0..num {
        let w = 1.0 + desired[i] * surplus_mult;
        let qw = ((w + 0.5) as u16).max(1);
        weights[i] = qw;
        sum += i64::from(qw);
    }
    // Nudge the weights so they sum to exactly TABLE_SIZE.
    let table = i64::from(TABLE_SIZE);
    let mut guard = num * 8 + 16;
    let mut i = 0usize;
    while sum > table && guard > 0 {
        if weights[i] > 1 {
            weights[i] -= 1;
            sum -= 1;
        }
        i = (i + 1) % num;
        guard -= 1;
    }
    guard = num * 8 + 16;
    i = 0;
    while sum < table && guard > 0 {
        weights[i] += 1;
        sum += 1;
        i = (i + 1) % num;
        guard -= 1;
    }
    if sum == table { Some(weights) } else { None }
}

/// Scatter symbols across the state table (same stride schedule as `fc`).
fn spread(weights: &[u16], which: &[u8]) -> Vec<u8> {
    let mut state_symbols = vec![0u8; TABLE_SIZE as usize];
    let mut stride = (3 * TABLE_SIZE) / 5;
    if stride & 1 == 0 {
        stride += 1;
    }
    let mut step: u32 = 0;
    for (s, &sym) in which.iter().enumerate() {
        for _ in 0..weights[s] {
            let idx = (stride.wrapping_mul(step)) & (TABLE_SIZE - 1);
            state_symbols[idx as usize] = sym;
            step += 1;
        }
    }
    state_symbols
}

struct SymInfo {
    renorm_cutoff: u32,
    min_renorm_bits: u8,
    weight: u16,
    cum: u16,
}

pub(crate) fn compress_bytes(src: &[u8]) -> Option<Vec<u8>> {
    if src.is_empty() {
        return None;
    }
    // Histogram + distinct symbols in increasing byte order.
    let mut counts = [0u32; 256];
    for &b in src {
        counts[b as usize] += 1;
    }
    let which: Vec<u8> = (0..256u32).filter(|&b| counts[b as usize] > 0).map(|b| b as u8).collect();
    let total: u64 = src.len() as u64;
    let weights = quantize(&counts, &which, total)?;
    let state_symbols = spread(&weights, &which);

    // Encoder tables.
    let mut infos: Vec<SymInfo> =
        (0..256).map(|_| SymInfo { renorm_cutoff: 0, min_renorm_bits: 0, weight: 0, cum: 0 }).collect();
    let mut cum: u16 = 0;
    let mut fill_pos = vec![0u16; which.len()];
    for (s, &sym) in which.iter().enumerate() {
        let w = weights[s];
        let max_xs = if w > 0 { 2 * u32::from(w) - 1 } else { 1 };
        let mut log_max = 0u32;
        while (max_xs >> log_max) > 1 {
            log_max += 1;
        }
        let mrb = (SIZE_LOG - log_max) as u8;
        infos[sym as usize] = SymInfo {
            renorm_cutoff: (u32::from(w) * 2) << mrb,
            min_renorm_bits: mrb,
            weight: w,
            cum,
        };
        fill_pos[s] = cum;
        cum = cum.wrapping_add(w);
    }
    let mut next_states = vec![0u32; TABLE_SIZE as usize];
    for (i, &sym) in state_symbols.iter().enumerate() {
        let slot = which.iter().position(|&x| x == sym).unwrap();
        next_states[fill_pos[slot] as usize] = TABLE_SIZE + i as u32;
        fill_pos[slot] += 1;
    }

    // Encode in reverse, recording (low, bits) renorm pairs.
    let mut state = TABLE_SIZE;
    let mut pairs: Vec<(u16, u8)> = Vec::with_capacity(src.len());
    for &sym in src.iter().rev() {
        let info = &infos[sym as usize];
        if info.weight == 0 {
            return None;
        }
        let renorm_bits = info.min_renorm_bits + u8::from(state >= info.renorm_cutoff);
        let low = if renorm_bits == 0 { 0 } else { state & ((1 << renorm_bits) - 1) };
        let shifted = state >> renorm_bits;
        state = next_states[info.cum as usize + (shifted - u32::from(info.weight)) as usize];
        pairs.push((low as u16, renorm_bits));
    }

    // Header.
    let mut out = Vec::with_capacity(src.len() / 2 + which.len() * 3 + 16);
    varint::write_u64(&mut out, src.len() as u64);
    out.push((which.len() - 1) as u8);
    out.extend_from_slice(&which);
    for &w in &weights {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out.extend_from_slice(&state.to_le_bytes());

    // Renorm bits, written so the decoder reads them in forward symbol order.
    let mut bw = BitWriter::new();
    for &(low, bits) in pairs.iter().rev() {
        bw.put(u64::from(low), u32::from(bits));
    }
    out.extend_from_slice(&bw.finish());
    Some(out)
}

pub(crate) fn decompress_bytes(src: &[u8], max_len: usize) -> Result<Vec<u8>, Error> {
    let mut pos = 0usize;
    let expected = varint::read_u64(src, &mut pos)? as usize;
    if expected > max_len {
        return Err(Error::CorruptPayload("tans length exceeds bound"));
    }
    let num = usize::from(*src.get(pos).ok_or(Error::Truncated)?) + 1;
    pos += 1;
    let which = src.get(pos..pos + num).ok_or(Error::Truncated)?.to_vec();
    pos += num;
    let mut weights = vec![0u16; num];
    for w in &mut weights {
        let b = src.get(pos..pos + 2).ok_or(Error::Truncated)?;
        *w = u16::from_le_bytes([b[0], b[1]]);
        pos += 2;
    }
    let sb = src.get(pos..pos + 4).ok_or(Error::Truncated)?;
    let mut state = u32::from_le_bytes([sb[0], sb[1], sb[2], sb[3]]);
    pos += 4;
    if !(TABLE_SIZE..2 * TABLE_SIZE).contains(&state) {
        return Err(Error::CorruptPayload("tans state out of range"));
    }

    // Validate the model from the (untrusted) header: weights must be >= 1,
    // symbols distinct, and weights sum to exactly TABLE_SIZE. The encoder
    // guarantees this; a corrupt stream might not, and the table-build /
    // decode loops rely on it to stay in range (else integer underflow / OOB).
    let mut seen = [false; 256];
    let mut wsum: u32 = 0;
    for (i, &sym) in which.iter().enumerate() {
        if weights[i] == 0 {
            return Err(Error::CorruptPayload("tans zero weight"));
        }
        if seen[sym as usize] {
            return Err(Error::CorruptPayload("tans duplicate symbol"));
        }
        seen[sym as usize] = true;
        wsum += u32::from(weights[i]);
    }
    if wsum != TABLE_SIZE {
        return Err(Error::CorruptPayload("tans weights do not sum to table size"));
    }

    let state_symbols = spread(&weights, &which);

    // Decoder nodes.
    let mut symbol_xs = [0u32; 256];
    for (s, &sym) in which.iter().enumerate() {
        symbol_xs[sym as usize] = u32::from(weights[s]);
    }
    let mut node_base = vec![0u16; TABLE_SIZE as usize];
    let mut node_bits = vec![0u8; TABLE_SIZE as usize];
    let mut node_sym = vec![0u8; TABLE_SIZE as usize];
    let clz_table = TABLE_SIZE.leading_zeros();
    for (i, &sym) in state_symbols.iter().enumerate() {
        let base = symbol_xs[sym as usize];
        if base == 0 {
            return Err(Error::CorruptPayload("tans symbol weight zero"));
        }
        let bits = base.leading_zeros() - clz_table;
        node_base[i] = ((base << bits) - TABLE_SIZE) as u16;
        node_bits[i] = bits as u8;
        node_sym[i] = sym;
        symbol_xs[sym as usize] += 1;
    }

    let mut br = BitReader::new(&src[pos..]);
    let mut out = Vec::with_capacity(expected);
    for _ in 0..expected {
        // `state` is built as TABLE_SIZE + node_base + extra; a valid model
        // keeps it in [TABLE_SIZE, 2*TABLE_SIZE), but guard against a corrupt
        // model pushing the index out of the node tables.
        let idx = (state - TABLE_SIZE) as usize;
        if idx >= TABLE_SIZE as usize {
            return Err(Error::CorruptPayload("tans state index out of range"));
        }
        let bits = u32::from(node_bits[idx]);
        let extra = br.get(bits) as u32;
        out.push(node_sym[idx]);
        state = TABLE_SIZE + u32::from(node_base[idx]) + extra;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(s: &mut u64) -> u64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *s
    }

    fn check(data: &[u8]) {
        match compress_bytes(data) {
            Some(packed) => {
                let restored = decompress_bytes(&packed, data.len()).unwrap();
                assert_eq!(restored, data, "tans round-trip mismatch (len {})", data.len());
            }
            None => { /* coder declined; caller falls back to rc */ }
        }
    }

    #[test]
    fn roundtrip_skewed() {
        let mut s = 1u64;
        let data: Vec<u8> =
            (0..40_000).map(|_| if lcg(&mut s).is_multiple_of(8) { (lcg(&mut s) >> 50) as u8 } else { 0 }).collect();
        let packed = compress_bytes(&data).expect("skewed compresses");
        assert!(packed.len() < data.len());
        assert_eq!(decompress_bytes(&packed, data.len()).unwrap(), data);
    }

    #[test]
    fn roundtrip_uniform_and_single() {
        let mut s = 99u64;
        check(&(0..40_000).map(|_| (lcg(&mut s) >> 40) as u8).collect::<Vec<_>>());
        check(&vec![7u8; 5000]); // single distinct symbol
        check(&[42u8]);
        check(&[]);
    }
}
