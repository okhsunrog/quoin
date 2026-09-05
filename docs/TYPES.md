# Typed columns

`quoin` compresses typed numeric columns. The engine works on a **physical
lane** — a `u64` word per value for the 64-bit types, a `u32` word for the
32-bit types; every codec is generic over the lane — plus a small descriptor
that decides which codecs apply:

- **width** — the lane size (32- and 64-bit today; 8/16 planned, 128/256 via
  the decimal container).
- **family** — int vs float. Decisive: integer columns run frame-of-reference /
  delta / bit-packing on the lane; float columns run the float-value schemes
  (ALP, FLOAT_MULT, float-linear). The type-agnostic codecs (RAW, CONST, STRIDE,
  XORZ, predictors, LZ, byte-transpose) apply to both.
- **signedness** — within integers, governs zigzag for delta (a ratio concern;
  reconstruction stays correct with wrapping arithmetic either way).

Width alone is **not** enough: the same 64 bits compress completely differently
as `1.0, 1.1, 1.2` (float delta `0.1, 0.1` → tiny) vs the integer delta of their
bit patterns (huge, irregular). The engine must be told the family.

## Arrow type mapping

Apache Arrow's many logical types collapse onto a handful of physical lanes plus
metadata:

| Lane | Family | Arrow logical types |
| --- | --- | --- |
| 64-bit | float | `Float64` |
| 64-bit | int | `Int64`, `UInt64`, `Timestamp(*)`, `Date64`, `Time64`, `Duration`, `Decimal64` |
| 32-bit | float | `Float32` |
| 32-bit | int | `Int32`, `UInt32`, `Date32`, `Time32`, `Decimal32` |
| 16-bit | int | `Int16`, `UInt16`, `Float16` (bits) |
| 8-bit | int | `Int8`, `UInt8` |
| 128-bit | int | `Decimal128` |
| 256-bit | int | `Decimal256` |

Decimals are the *easy* case, not the hard one: an Arrow decimal is an integer
unscaled value plus a `scale` carried in the schema. The codec compresses the
integer lane (FoR / delta / bit-pack, exact); `scale`/`precision` ride as opaque
metadata.

## Status

Implemented: **`F64`, `F32`, `I64`, `U64`, `I32`, `U32`, `Decimal128`,
`Decimal256`**, via
[`DType`](../src/dtype.rs), `ColumnRef`/`Column`, and
`compress_column`/`decompress_column`. `f64` streams are unchanged but for the
format version bump. Integer columns gate out the float-only codecs; decimals use
the limb-split container.

## Roadmap

1. ✅ **Type scaffold + 64-bit lanes** — `DType` in the stream header (format
   v2), typed API, family-aware codec gating. `F64`/`I64`/`U64`.
2. ✅ **64-bit integers** — i64/u64 through FoR/delta/bit-pack, with a `u64`
   bit-pack kernel (`bitpack::pack64`/`unpack64`, 16 lanes × 64 rows,
   autovectorized) so wide ranges (33..=64 bits) pack instead of falling back to
   raw, and **signedness-aware FoR** (`for_bitpack` references the signed
   minimum, so mixed-sign columns pack instead of bailing to 64-bit). Temporal
   types map onto `I64` for free.
3. ◑ **32-bit lanes** — ✅ `Int32`/`UInt32` run on a **native `u32` lane**
   (format v3): the slice is reinterpreted in place, FoR is signed-aware at 32
   bits, CONST/STRIDE/dictionary/RLE entries and byte planes are 4 bytes, the
   predictor tables hold `u32`, and pco gets the `&[i32]`/`&[u32]` directly.
   Remaining: narrow `Int8/16`/`UInt8/16` (lane_bytes 1/2).
4. ✅ **`Float32`** — the same native `u32` lane, with the float-value codecs
   running in **`f32` arithmetic with `f32` constants**: ALP uses the reference's
   float parameters (exponents to 10, magic `1.5·2^23`), FLOAT_MULT verifies
   `k/scale` in `f32`, DELTA2/DELTA_DP extrapolate in `f32`, and pco compresses
   the `&[f32]` as is. No value is ever converted to `f64`. **Every** bit pattern
   round-trips exactly — finite values, ±0, subnormals, infinities and NaNs with
   any payload or signaling bit: the float-arithmetic codecs verify each value's
   reconstruction bit-for-bit (ALP/FLOAT_MULT/DELTA_DP) or are skipped for a
   block containing any inf/NaN (DELTA2/DELTA_DP), so decoding never depends on
   the platform's NaN-propagation rules. (Before v3, `f32` was widened to its
   exact `f64` on the `u64` lane; those v2 streams are rejected, not misread.)
5. ✅ **`Decimal128`** (`src/decimal.rs`) — a **limb-split container**, not a new
   lane kernel. Subtract a global `vmin` (column min) → a non-negative offset,
   split it into 64-bit limbs (2 for `Decimal128`), and run **each limb as an
   ordinary `U64` column through the full engine**. In the common case every value
   fits in 64 bits after the shift, so the high limb is all-zero → `CONST` → ~free,
   and the low limb gets per-block FoR / delta / dict — no special wide-range path
   (a genuinely 128-bit-spread column just has a non-trivial high limb). `vmin`,
   `scale` and `precision` ride in the container header (`FLAG_DECIMAL`); nulls are
   compacted once at the container level. Lossless across the full `i128` range
   (incl. `MIN`/`MAX`). `Decimal32/64` fold onto the 32/64 lanes.
   ✅ **`Decimal256`** is the *same* container with 4 limbs and a 32-byte `vmin`.
   Values are little-endian two's-complement `[u8; 32]` in the core API (matching
   `arrow_buffer::i256`'s layout); the few needed 256-bit ops (signed-min,
   add/sub) are implemented on `[u64; 4]` in `decimal.rs`, so the core stays free of
   any big-integer dependency. Lossless across the full `i256` range.
6. ✅ **Nulls / validity** (`src/validity.rs`) — an Arrow-compatible validity
   bitmap (LSB-first, 1 = valid). Nulls are **compacted out** so the value codec
   only sees valid values (best ratio); the bitmap is stored as the smaller of a
   run-length encoding (nulls cluster) or raw, between the header and the frames
   (`FLAG_VALIDITY`). `compress_column` takes `validity: Option<&[u8]>`;
   `decompress_column` returns `DecodedColumn { values, validity }` (null slots →
   0). All lanes nullable; all-valid normalizes to no-validity; non-null streams
   are byte-identical to before.
7. ◑ **Arrow adapter** (`src/arrow.rs`, feature `arrow`) — ✅ `compress_array`/
   `decompress_array` for the primitive numeric arrays (`Float64`/`Float32`/
   `Int64`/`UInt64`/`Int32`/`UInt32`) plus `Decimal128`/`Decimal256`
   (precision/scale preserved), with validity, including sliced arrays.
   **Pending:** temporal arrays (need the Arrow logical type preserved in the
   stream).
8. [ ] **`Float16`, `Boolean`**, then temporal-type preservation.
