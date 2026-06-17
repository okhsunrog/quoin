# Performance & the speed/ratio knob

Design notes for making `quoin` fast and for giving callers an explicit
speed-vs-ratio dial. The level knob, entropy gating, rANS path, and cost-aware
selection are implemented; analytic estimators and further tuning remain.

There are **two distinct levers**, and they act on different phases:

1. **Pure speedups** — make the engine faster *without* giving up ratio.
2. **A speed/ratio knob** — deliberately trade ratio for speed (the "codec
   weight" idea).

The knob itself has two independent axes:

- **encode-effort** — how hard we search for the best codec.
- **decode-cost** — which codec we ultimately pick, weighted by how expensive
  it is to *decode*.

For a write-once-read-many columnar workload, **decode** is usually the thing
worth winning, so the codec "weight" is computed primarily from decode cost.

---

## Part 1 — pure speedups (no ratio cost)

Ranked by expected impact.

### 1. Analytic size estimators → kill full-encode competition (encode)
Today `Full` selection encodes *all* applicable codecs and keeps the smallest —
the dominant encode cost. But most lightweight codecs have a **closed-form size**
that needs no encoding:

- bit-pack / FoR: `width = bits(max − min)` → exact packed size from min/max.
- RLE: one pass counts runs.
- dictionary: count distinct (the sampled bitset in `probe_block_features`
  already estimates this).
- delta→bit-pack: min/max over the deltas.

Turn the competition from "encode 16 times" into "estimate 16 times ~for free,
encode only the winner." This is the principled extension of the existing
`Sample` selection.

### 2. Tiered / optional entropy stage; interleaved rANS for the kept path (decode)
The order-1 binary range coder is sequential and the slowest path on both encode
and decode (~an order of magnitude under bit-packing). Two moves:

- Make the entropy stage **optional** (driven by the knob below): fast levels
  skip it entirely — a lightweight transform (FoR / delta / ALP / bit-pack) and
  done, so decode is pure SIMD/scalar.
- Where entropy coding stays, replace the binary range coder with **interleaved
  rANS** (4–8 streams, SIMD-decodable). Biggest decode win at ~unchanged ratio.

### 3. Buffer reuse / arena
Zero per-block allocations in the competition hot loop. The C ABI already keeps a
persistent pool; thread the same reuse through the Rust path.

### 4. Finish the SIMD substrate
✅ A `u64` bit-pack variant (`bitpack::pack64`/`unpack64`, verified to
autovectorize to AVX2) now backs wide integer columns. Remaining: a `u128`
variant for `Decimal128`. Explicit-shuffle `byte_transpose` is low ROI — defer.

### 5. Block-parallelism (already done)
Block-parallel via rayon; just keep block sizes large enough to keep lanes full.

Items 1–3 are ~80% of the win and none of them costs ratio.

---

## Part 2 — the speed/ratio knob (cost-aware selection)

The intuition: every codec has a **weight** (how heavy it is), and selection
should weigh not just output size but that cost. This is cost-aware / Lagrangian
selection.

Today:

```
pick  argmin_c  size(c, B)
```

Becomes:

```
pick  argmin_c  [ size(c,B)/size_raw  +  λ · dec_time(c,B)/time_raw ]
```

- Both terms are normalized against RAW → dimensionless → addable.
- **`λ` is the priority dial.** `λ = 0` → today's behavior (pure ratio).
  `λ → ∞` → pure speed (cheapest codec). `λ = 1` → equal weight.
- `dec_time(c,B) ≈ decoded_bytes / throughput(c)`, and `throughput(c)` is the
  codec **weight**: a static table of throughput classes calibrated once from
  `benches/kernels.rs`. Roughly: RAW (memcpy) ≫ bit-pack/FoR (~30 GiB/s) >
  byte_transpose (~6) > rANS > LZ > order-1 range coder (slow). Per-codec-family,
  applied for ~free.

**Equivalent human-friendly form** ("only upgrade to a heavier codec if it earns
it"): switch to a heavier codec only when its size win exceeds a threshold.
E.g. bit-pack 4.0× vs range coder 4.1× — a 2.5% gain doesn't justify a 10×
slower decode, so keep bit-pack. Same as λ, expressed as a minimum %-gain to
justify a heavier tier.

### Practical wrapper: levels, not a raw λ

A raw λ is hard to tune blind. Wrap it in a coarse **level enum** (like
`zstd -1..22`) where each level sets three things at once:

| Level      | enabled codecs                  | entropy stage        | encode-effort           | λ      |
|------------|---------------------------------|----------------------|-------------------------|--------|
| `Fastest`  | RAW/CONST/FoR/bit-pack/delta    | none                 | analytic estimators     | high   |
| `Fast`     | + ALP, byte_transpose           | rANS (optional)      | sample                  | medium |
| `Balanced` | + LZ, rANS                      | yes                  | sample → encode winner  | low    |
| `Max`      | all                             | range coder + rANS   | full competition        | 0      |

A level dials **encode speed** (how many codecs we try + sample vs full) *and*
**decode speed** (whether the heavy entropy path is allowed + the λ tie-break)
with a single number. λ / the threshold stay exposed for power users.

### Nuances

- **Encode-cost and decode-cost are separate axes.** The weights in the formula
  are about *decode* (the read-heavy value). Encode speed is governed separately
  by effort (analytic estimators / sample / full). The level enum turns both at
  once; don't conflate them.
- The **rANS replacement** for the range coder is the single change that improves
  *both* speed and the legibility of the knob: "no entropy / fast rANS / max
  range coder" is the clearest, highest-impact speed↔ratio dial.

---

## Implementation status

1. ✅ `Config.level: Level` (`Fastest`/`Fast`/`Balanced`/`Max`). `Max` is the
   default and reproduces the historical output bit-for-bit (regression test
   `level_max_is_default_behavior`).
2. ✅ Cost-aware selection: winner is `argmin(size + λ·weight·decoded_bytes)`
   via the `Best` selector + `decode_weight` table in `src/encoder.rs`. `λ` comes
   from the level (`Max` → 0).
3. [ ] **Analytic size estimators** for bit-pack / FoR / delta / RLE / dict so the
   competition stops fully encoding everything (the remaining *encode*-speed win).
4. ✅ **Entropy stage gated per level**: `Fast`/`Fastest` skip the entropy-coded
   modes entirely (`is_entropy_mode`), keeping decode vectorized/random-access.
5. ✅ **Interleaved rANS** (`entropy/rans.rs`, 4-way, order-0) joins the range
   coder + rANS, chosen by the same `size + λ·decode_cost` rule. Speed-leaning
   levels prefer it: q-balanced decodes ~40% faster than q-max (and faster than
   zstd-19) for ~2% ratio. See [`BENCHMARKS.md`](BENCHMARKS.md).

Remaining: analytic estimators (3) and tuning `decode_weight` against
`benches/kernels.rs`.

## Profiling notes

A `perf` profile of the encode/decode loop on entropy-heavy columns:

- **Range coder ≈ 50–64%** of CPU (the order-1 `encode_bit` model). This is the
  encode path of the best-ratio `Max` level — bit-serial and inherent.
- **LZ match-finding ≈ 18–21%** (now with a zlib-style one-byte quick-reject).
- **rANS ≈ 6%**; decode is dominated by bit-packing (SIMD) and rANS — already fast.

Ratio-neutral wins applied: tANS dropped from the encode path (rANS supersedes
it), and the LZ quick-reject.

### Branchless range coder (biggest win)

A `toplev` top-down breakdown showed the workload was **Bad-Speculation-bound at
37.5% of slots** — the range coder's per-bit decision (`bit` on encode, `code <
bound` on decode) is data-dependent and unpredictable, so the branch predictor
can't learn it. Rewriting `encode_bit`/`decode_bit` to compute both candidate
updates and **mask-select** them (no branch) — output **bit-identical**, so ratio
is unchanged — gave:

| Metric | Before | After |
| --- | --- | --- |
| Bad-speculation (slots) | 37.5% | **18.4%** |
| Branch-miss rate | 5.1% | **1.29%** |
| IPC | 2.38 | **3.87** |

Mispredicts dropped ~4×; this speeds both encode and decode of every range-coded
mode at zero ratio cost. (Tooling: `perf stat`/`record` for hardware counters,
`pmu-tools`' `toplev` and Intel **VTune** (`uarch-exploration`) for the top-down
breakdown; pin to one P-core with `taskset -c 0` + `RAYON_NUM_THREADS=1` on this
hybrid CPU, and build `--profile profiling` for source-level attribution.)

### Bounds-check elision in the model access

VTune showed the range coder still dominant after the branchless fix. Its per-bit
hot operation is the model lookup `self.probs[ctx*256 + node]`, a bounds-checked
`Vec` index done 8× per byte. The index is provably in-bounds (`ctx < 256`,
`node ∈ [1,255]`), so `get_unchecked_mut` is sound. Measured: **instructions
retired 250.5 B → 239.85 B, clockticks 64.55 B → 61.75 B (~4.3% fewer)**,
output bit-identical.

### Negative results (measured, then reverted)

VTune is also useful for *killing* ideas cheaply:

- **Per-thread model arena** — the range coder allocates a 128 KiB order-1 model
  per call; a thread-local reuse looked promising from the cache-miss profile. But
  VTune showed memory is **not** the bottleneck (DRAM Bound 0.3%, Memory Bound
  ~6%) — glibc already caches the freed chunk, so reuse was flat (clockticks
  61.75 B → 61.77 B). Reverted. This also retires the broader "arena" idea: the
  "43% of cache misses in malloc/free" was 43% of a small number.
- **Cross-mode rANS estimator** — rank entropy modes by cheap rANS size, range-code
  only the best few. The free `looks_compressible` gate already prunes to ~3
  candidates, so it added rANS work without saving RC calls. Reverted.

After the branchless coder + bounds-check elision the range coder runs at high IPC
(~3.9, ~58% retiring); the remaining bad-spec (renorm/`shift_low`, LZ branches) and
core-bound execution are diminishing returns.
