//! Build script. When the `bench-fc` feature is active, compile and link the
//! upstream C `fc` library so the bench harness can call it via FFI.
//!
//! Point `FC_SRC_DIR` at the `fc` checkout (defaults to `../fc`). Requires a
//! C compiler and an x86-64 CPU with AVX2 + SSE4.2 + BMI + LZCNT, matching the
//! original Makefile's flags.

fn main() {
    // Cargo exposes activated features to build scripts via CARGO_FEATURE_<NAME>.
    if std::env::var_os("CARGO_FEATURE_BENCH_FC").is_none() {
        return;
    }

    let dir = std::env::var("FC_SRC_DIR").unwrap_or_else(|_| "../fc".to_string());
    println!("cargo:rerun-if-env-changed=FC_SRC_DIR");
    println!("cargo:rerun-if-changed={dir}/fc.c");
    println!("cargo:rerun-if-changed={dir}/fc.h");
    println!("cargo:rerun-if-changed={dir}/gorilla.c");

    cc::Build::new()
        .file(format!("{dir}/fc.c"))
        .file(format!("{dir}/gorilla.c"))
        .opt_level(3)
        .flag("-mavx2")
        .flag("-msse4.2")
        .flag("-mbmi")
        .flag("-mlzcnt")
        .flag_if_supported("-std=c11")
        .warnings(false)
        .compile("fc_c");

    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=pthread");
}
