//! RLE: run-length encode columns with long identical runs. Each run stores its
//! value (verbatim) and length (bit-packed via [`super::for_bitpack`]). Wins on
//! grouped/sorted columns with repeated values; bails when runs are short.
//! Type-agnostic — operates on the raw `u64` lane.

use crate::codecs::for_bitpack;
use crate::error::Error;
use crate::varint;

pub(crate) fn encode(vals: &[u64]) -> Option<Vec<u8>> {
    if vals.is_empty() {
        return None;
    }
    // Bail once runs exceed half the values — RLE then loses to storing verbatim.
    let max_runs = vals.len() / 2 + 1;
    let mut run_vals: Vec<u64> = Vec::new();
    let mut run_lens: Vec<u64> = Vec::new();
    let mut prev = vals[0];
    let mut len = 0u64;
    for &v in vals {
        if v == prev {
            len += 1;
        } else {
            run_vals.push(prev);
            run_lens.push(len);
            prev = v;
            len = 1;
            if run_vals.len() > max_runs {
                return None;
            }
        }
    }
    run_vals.push(prev);
    run_lens.push(len);

    let mut out = Vec::with_capacity(run_vals.len() * 8 + 16);
    varint::write_u64(&mut out, run_vals.len() as u64);
    for &v in &run_vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&for_bitpack::encode(&run_lens, false));
    Some(out)
}

pub(crate) fn decode(payload: &[u8], n: usize) -> Result<Vec<u64>, Error> {
    let mut pos = 0usize;
    let num_runs = varint::read_u64(payload, &mut pos)? as usize;
    if num_runs > n {
        return Err(Error::CorruptPayload("rle run count exceeds value count"));
    }
    let mut run_vals = Vec::with_capacity(num_runs);
    for _ in 0..num_runs {
        let b = payload.get(pos..pos + 8).ok_or(Error::Truncated)?;
        run_vals.push(u64::from_le_bytes(b.try_into().unwrap()));
        pos += 8;
    }
    let run_lens = for_bitpack::decode(&payload[pos..], num_runs, false)?;

    let mut out = Vec::with_capacity(n);
    for (&v, &l) in run_vals.iter().zip(run_lens.iter()) {
        // Guard against a corrupt length claiming a huge run (decompression bomb).
        let l = l as usize;
        if out.len() + l > n {
            return Err(Error::CorruptPayload("rle run overruns value count"));
        }
        out.extend(std::iter::repeat_n(v, l));
    }
    if out.len() != n {
        return Err(Error::LengthMismatch {
            expected: n,
            got: out.len(),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(vals: &[u64]) -> Option<usize> {
        let enc = encode(vals)?;
        assert_eq!(decode(&enc, vals.len()).unwrap(), vals);
        Some(enc.len())
    }

    #[test]
    fn long_runs_pack() {
        // 50 runs of ~200 each.
        let mut vals = Vec::new();
        for r in 0..50u64 {
            vals.extend(std::iter::repeat_n(1_000_000 + r * 7, 200));
        }
        let size = roundtrip(&vals).expect("should encode");
        assert!(size < vals.len(), "long runs should pack tiny, got {size}");
    }

    #[test]
    fn no_runs_bails() {
        let vals: Vec<u64> = (0..10000u64).collect();
        assert!(encode(&vals).is_none());
    }

    #[test]
    fn edges() {
        assert!(encode(&[]).is_none());
        roundtrip(&vec![3u64; 5000]);
        roundtrip(&[1, 1, 2, 2, 2, 3]);
    }
}
