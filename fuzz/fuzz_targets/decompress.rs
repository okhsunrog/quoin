#![no_main]
//! `decompress` parses untrusted bytes: it must never panic, abort, read OOB,
//! or allocate more than the *declared* (bounded here) column size.
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Bomb-guard, mirroring what callers must do with untrusted input (see the
    // `decompress` docs): a tiny *valid* stream can legitimately declare a huge
    // column (e.g. a run-length-coded all-null bitmap), and decoding it means
    // materializing it. Bound the columns we try to materialize; everything
    // else about the stream is still parsed and must fail gracefully.
    match quoin::decompressed_len(data) {
        Ok(n) if n > (1 << 24) => return,
        _ => {}
    }
    let _ = quoin::decompress(data);
});
