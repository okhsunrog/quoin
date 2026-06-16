#![no_main]
//! `decompress` parses untrusted bytes: it must never panic, abort, read OOB,
//! or over-allocate — only ever return `Ok`/`Err`.
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = quoin::decompress(data);
});
