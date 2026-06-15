# TODO

Scoped work items not yet on the main [ROADMAP](ROADMAP.md). These capture
design decisions reached in discussion so the context isn't lost.

## C ABI / FFI layer

Add a C-callable API (new `cdylib` + `staticlib` crate-type and a thin
`extern "C"` layer) so the compressor can be consumed from C/C++/etc. The
threading model needs care at the boundary — notes below.

### Threading across the C boundary

- **Default global pool is C-friendly.** `Config { threads: None }` runs on
  rayon's process-global pool (lazily created once, lives for the process), so
  a context-free `fp_compress()` gets zero per-call thread churn — same as a
  Rust caller.
- **The Rust escape hatch doesn't cross FFI.** "Build your own `ThreadPool` and
  wrap calls in `pool.install()`" is Rust-only. A C caller who wants *bounded*
  threads is otherwise stuck with `Some(n)`, which rebuilds a transient pool
  **every call** — real churn for high-frequency/small-input use.
- **Fix: opaque context handle owning a persistent pool** (like `ZSTD_CCtx`):
  ```c
  fp_ctx* ctx = fp_ctx_create(/*threads=*/8); // builds & holds one ThreadPool
  fp_compress_ctx(ctx, src, len, dst, cap);    // runs in ctx.pool.install(...)
  fp_ctx_free(ctx);                            // deterministic thread teardown
  ```
  Also expose a context-free `fp_compress()` that uses the global pool for
  convenience. Avoid making a one-shot `build_global()` init the *only* option
  (it's process-wide, one-shot, clobbers the host's rayon config).

### FFI correctness footguns (must handle)

- [ ] **Panics must not cross `extern "C"`** — a rayon worker can panic (e.g.
      OOM in a block); unwinding across the boundary is UB. Wrap every
      `extern "C"` fn in `catch_unwind` → error code.
- [ ] **`fork()` without `exec`** — rayon worker threads don't survive a fork,
      so compressing in a child after the pool was started deadlocks. Document
      the caveat; the context handle lets the host control when threads exist.
- [ ] **Thread lifetime vs teardown** — global-pool threads persist; `dlclose`
      while they're alive is unsafe. Context handle gives deterministic
      teardown via `fp_ctx_free`.

### API surface

- [ ] `extern "C"` wrappers: `fp_compress` / `fp_decompress` (global pool),
      `fp_ctx_create(threads)` / `fp_compress_ctx` / `fp_ctx_free`.
- [ ] Error codes (no panics, no `Result` across the boundary); map
      [`Error`](src/error.rs) variants to integers.
- [ ] Caller-sized output buffers + a `comp_bound`-style sizing helper.
- [ ] Generate a C header (e.g. `cbindgen`).
- [ ] `cdylib` + `staticlib` crate-types; keep the `parallel` feature working
      both on and off across the C API.
