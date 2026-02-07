# Benchmarks

All benchmarks use [Criterion.rs](https://github.com/bgruber/criterion.rs) and live under
each crate's `benches/` directory.

## Quick Reference

| Crate | Bench name | What it measures |
|---|---|---|
| `zip-plus` | `zip_benches` | Zip PCS (integer coefficients): RAA & IPRS encoding, commit, 128-bit & 256-bit encoding |
| `zip-plus` | `zip_plus_benches` | Zip+ PCS (polynomial coefficients): RAA, IPRS (multiple fields/depths/rates), commit |
| `piop` | `sumcheck` | Sumcheck prover & verifier over random fields |
| `poly` | `mle_evaluation` | Multilinear extension evaluation (field & inner representation) |
| `poly` | `nat_evaluation` | Natural evaluation domain polynomial evaluation |

## Cargo Feature Flags

Several features affect benchmark performance. Pass them with `--features`:

| Feature | Crate | Effect |
|---|---|---|
| `parallel` | all | Enables Rayon parallelism for encoding and other operations |
| `simd` | `zip-plus` (via `zinc-poly`) | Enables SIMD-optimised polynomial operations |
| `asm` | `zip-plus` (via `zinc-transcript`) | Uses assembly-optimised hashing |
| `unchecked-butterfly` | `zip-plus` | Skips overflow checks in NTT butterfly operations (faster, requires valid input ranges) |
| `pntt-timing` | `zip-plus` | Prints a detailed PNTT timing breakdown (useful with `profile_encode` example) |
| `bench-64` | `zip-plus` | Enables 64-bit coefficient benchmarks (disabled by default) |

A good default for maximum performance:

```bash
--features "parallel simd asm unchecked-butterfly"
```

---

## `zip-plus` Benchmarks

### `zip_benches` — Zip PCS (integer coefficients)

Benchmarks encoding and commitment with **integer** evaluation/codeword types
(`i32` → `i64`).

```bash
cargo bench --bench zip_benches -p zip-plus
```

**Benchmark groups:**

| Group name | Code | Description |
|---|---|---|
| `Zip+` | RAA code, rate 1/2 | Commit with RAA linear code for poly sizes 2^12 – 2^16 |
| `Zip IPRS` | IPRS depth-2, rate 1/2 | Commit with IPRS (F65537) for poly sizes 2^13 – 2^17 |
| `Zip IPRS Matrix Shapes` | IPRS depth-1, rate 1/2 | Single-row encode with base 1rr6/32/64 matrix shapes |
| `Zip IPRS rate1_4` | IPRS depth-2, rate 1/4 | Commit with IPRS (F65537) at rate 1/4 |
| `Zip IPRS rate1_4 Matrix Shapes` | IPRS depth-1, rate 1/4 | Single-row encode with base 16/32/64 at rate 1/4 |
| `Zip Encode 128-bit` | IPRS depth-2, 128-bit ints | Encode `Int<2>` → `Int<3>` for sizes 2^11–2^14 at rate 1/2 and 1/4 |
| `Zip Encode 256-bit` | IPRS depth-3, 256-bit ints | Encode `Int<4>` → `Int<5>` at rate 1/2 and 1/4 |
| `Zip Encode 128-bit Selected` | IPRS depth-2/3, 128-bit ints | Encode `Int<2>` → `Int<3>` at sizes 2^12, 2^13 and `Int<2>` → `Int<4>` at size 2^14 (rate 1/2) |

**Filter examples:**

```bash
# Only the basic Zip+ RAA benchmarks
cargo bench --bench zip_benches -p zip-plus -- "Zip\+"
# Only IPRS rate 1/4
cargo bench --bench zip_benches -p zip-plus -- "rate1_4"
# Only 128-bit encoding
cargo bench --bench zip_benches -p zip-plus -- "128-bit"
# Only 256-bit encoding
cargo bench --bench zip_benches -p zip-plus -- "256-bit"
# Only 128-bit selected (depth-2 for 2^12/2^13, depth-3 for 2^14)
cargo bench --bench zip_benches -p zip-plus -- "128-bit Selected"
```

### `zip_plus_benches` — Zip+ PCS (polynomial coefficients)

Benchmarks encoding and commitment with **polynomial** evaluation/codeword
types (`BinaryPoly<32>` → `DensePolynomial<i64, 32>`).

```bash
cargo bench --bench zip_plus_benches -p zip-plus
```

**Benchmark groups:**

| Group name | Description |
|---|---|
| `Zip+ RAA` | Commit with RAA code, poly sizes 2^12 – 2^16 |
| `Zip+ IPRS` | Commit with IPRS depth-2 (F65537, rate 1/2), poly sizes 2^13 – 2^17 |
| `Zip+ IPRS Matrix Shapes` | Single-row commit with depth-1 base 16/32/64 (rate 1/2) |
| `Zip+ IPRS rate1_4` | Commit with IPRS depth-2 (F65537, rate 1/4), poly sizes 2^13 – 2^17 |
| `Zip+ IPRS rate1_4 Matrix Shapes` | Single-row commit with depth-1 base 16/32/64 (rate 1/4) |
| `Zip+ IPRS F65537 Depth3` | IPRS depth-3, row_len=4096, rate 1/2. Matrix sizes from 2×4096 to 32×4096 |
| `Zip+ IPRS F65537 Depth2` | IPRS depth-2, row_len=2048, rate 1/2. Matrix sizes from 4×2048 to 64×2048 |
| `Zip+ IPRS F12289 Depth3 2^11` | IPRS depth-3 over F12289, row_len=2048, rate 1/2. Matrix sizes 4×2048 to 64×2048 |
| `Zip+ IPRS F65537 Depth2 Rate1_4` | IPRS depth-2, row_len=4096, rate 1/4 (i64-safe). Matrix sizes 4×4096 to 32×4096 |
| `Zip+ IPRS F65537 Depth2 Wide` | IPRS depth-2, rate 1/2, fixed 4 rows with increasing row_len (4096→32768). Explores wide matrices: 4×4096, 4×8192, 4×16384, 4×32768 |
| `Zip+ Commit Comparison` | Commit with optimal matrix shapes: 2×2^12, 2×2^13, 4×2^13, 4×2^14, 8×2^14. Uses depth-2 for msg sizes 2^12–2^14 and depth-3 for msg size 2^14 |
| `Zip+ Test Comparison` | Test phase (proximity test) with the same optimal matrix shapes as Commit Comparison |
| `Zip+ Evaluate Comparison` | Evaluate phase (evaluation proof generation) with the same optimal matrix shapes as Commit Comparison |
| `Zip+ Commit Comparison Rate1_4` | Same matrix shapes as Commit Comparison but with rate 1/4 IPRS codes (all depth-2) |
| `Zip+ Test Comparison Rate1_4` | Test phase with rate 1/4 IPRS codes (same matrix shapes as Commit Comparison) |
| `Zip+ Evaluate Comparison Rate1_4` | Evaluate phase with rate 1/4 IPRS codes (same matrix shapes as Commit Comparison) |
| `Zip+ Commit Comparison Rate1_4 Depth3` | Same matrix shapes as Commit Comparison with rate 1/4, depth-3 IPRS codes and `i128` codeword coefficients (avoids i64 overflow) |
| `Zip+ Test Comparison Rate1_4 Depth3` | Test phase with rate 1/4 depth-3 IPRS codes (i128 coefficients) |
| `Zip+ Evaluate Comparison Rate1_4 Depth3` | Evaluate phase with rate 1/4 depth-3 IPRS codes (i128 coefficients) |
| `Zip+ Commit Comparison Rate1_4 Depth4` | Same matrix shapes with rate 1/4, depth-4 IPRS codes and `i128` coefficients. Smallest possible base matrices (4×1, 8×2, 16×4) |
| `Zip+ Test Comparison Rate1_4 Depth4` | Test phase with rate 1/4 depth-4 IPRS codes (i128 coefficients) |
| `Zip+ Evaluate Comparison Rate1_4 Depth4` | Evaluate phase with rate 1/4 depth-4 IPRS codes (i128 coefficients) |

**Filter examples:**

```bash
# Only F65537 Depth3
cargo bench --bench zip_plus_benches -p zip-plus -- "F65537 Depth3"
# Only F65537 Depth2 at rate 1/4
cargo bench --bench zip_plus_benches -p zip-plus -- "F65537 Depth2 Rate1_4"
# Only F65537 Depth2 Wide (fixed 4 rows, increasing row_len)
cargo bench --bench zip_plus_benches -p zip-plus -- "Depth2 Wide"
# Only Commit Comparison (optimal matrix shapes)
cargo bench --bench zip_plus_benches -p zip-plus -- "Commit Comparison"
# Only Commit Comparison at rate 1/4
cargo bench --bench zip_plus_benches -p zip-plus -- "Commit Comparison Rate1_4"
# Only Commit Comparison at rate 1/4 depth 3
cargo bench --bench zip_plus_benches -p zip-plus -- "Comparison Rate1_4 Depth3"
# Only Commit Comparison at rate 1/4 depth 4
cargo bench --bench zip_plus_benches -p zip-plus -- "Comparison Rate1_4 Depth4"
# Only F12289
cargo bench --bench zip_plus_benches -p zip-plus -- "F12289"
# Only RAA
cargo bench --bench zip_plus_benches -p zip-plus -- "RAA"
# A specific matrix size
cargo bench --bench zip_plus_benches -p zip-plus -- "matrix=16x4096"
```

### Benchmark Results: Commit Comparison

Results from the `Zip+ Commit Comparison` group with `--features "parallel simd asm unchecked-butterfly"` (Apple M-series):

| Matrix | Entries | IPRS Config | Depth | msg_size | Time |
|--------|---------|-------------|-------|----------|------|
| 2×4096 | 2^13 | Base64 Depth2 | 2 | 2^12 | ~2.67 ms |
| 2×8192 | 2^14 | Base128 Depth2 | 2 | 2^13 | ~6.77 ms |
| 4×8192 | 2^15 | Base128 Depth2 | 2 | 2^13 | ~13.0 ms |
| 4×16384 | 2^16 | Depth3 | 3 | 2^14 | ~15.9 ms |
| 8×16384 | 2^17 | Depth3 | 3 | 2^14 | ~30.5 ms |

### Benchmark Results: Test Comparison

Results from the `Zip+ Test Comparison` group with `--features "parallel simd asm unchecked-butterfly"` (Apple M-series).
The test phase performs the proximity test: it re-encodes each row and checks that the committed codeword is close to the encoding.

| Matrix | Entries | IPRS Config | Depth | msg_size | Time |
|--------|---------|-------------|-------|----------|------|
| 2×4096 | 2^13 | Base64 Depth2 | 2 | 2^12 | ~562 µs |
| 2×8192 | 2^14 | Base128 Depth2 | 2 | 2^13 | ~1.14 ms |
| 4×8192 | 2^15 | Base128 Depth2 | 2 | 2^13 | ~1.58 ms |
| 4×16384 | 2^16 | Depth3 | 3 | 2^14 | ~2.40 ms |
| 8×16384 | 2^17 | Depth3 | 3 | 2^14 | ~3.05 ms |

### Benchmark Results: Evaluate Comparison

Results from the `Zip+ Evaluate Comparison` group with `--features "parallel simd asm unchecked-butterfly"` (Apple M-series).
The evaluate phase generates the evaluation proof: it reduces the polynomial evaluation claim to column openings via the sumcheck protocol and computes the final field evaluation.

| Matrix | Entries | IPRS Config | Depth | msg_size | Time |
|--------|---------|-------------|-------|----------|------|
| 2×4096 | 2^13 | Base64 Depth2 | 2 | 2^12 | ~1.58 ms |
| 2×8192 | 2^14 | Base128 Depth2 | 2 | 2^13 | ~2.25 ms |
| 4×8192 | 2^15 | Base128 Depth2 | 2 | 2^13 | ~3.96 ms |
| 4×16384 | 2^16 | Depth3 | 3 | 2^14 | ~6.22 ms |
| 8×16384 | 2^17 | Depth3 | 3 | 2^14 | ~10.8 ms |

### Benchmark Results: 128-bit Encoding (Selected)

Results from the `Zip Encode 128-bit Selected` group (`Int<2>` evaluations, rate 1/2, F65537):

| msg_size | IPRS Config | Depth | Cw type | Time |
|----------|-------------|-------|---------|------|
| 2^12 = 4096 | Base64 Depth2 | 2 | `Int<3>` | ~1.39 ms |
| 2^13 = 8192 | Base128 Depth2 | 2 | `Int<3>` | ~4.40 ms |
| 2^14 = 16384 | Depth3 | 3 | `Int<4>` | ~7.81 ms |

> **Note:** The depth-3 case for 2^14 requires `Int<4>` (256-bit) codewords because
> 3 recursion levels overflow the default `Int<3>` (192-bit) codeword type.

### Proof Size Analysis

The Zip+ proof size (in bits) is dominated by two terms:

```
proof_size ≈ 2 × 32 × 200 × 26 × num_rows + 128 × num_columns
           = 332,800 × num_rows + 128 × num_columns
```

The first term accounts for the Merkle authentication paths (one per sampled column, per row), while the second term accounts for the column openings themselves. These are the dominant costs for wide matrices; other minor contributions exist but become negligible as the matrix width increases.

**Choosing matrix dimensions:** Given a fixed number of matrix entries (`num_rows × num_columns`), the optimal shape minimizes the proof size. Taking the derivative and solving yields:

```
optimal_num_rows ≈ √(total_entries / 2600)
```

In practice, `num_rows` must be a power of two. The table below shows optimal configurations:

| Total Entries | Best num_rows | num_columns | Proof Size |
|---------------|---------------|-------------|------------|
| 2^14 = 16,384 | 2 | 8,192 | ~209 KB |
| 2^15 = 32,768 | 4 | 8,192 | ~290 KB |
| 2^16 = 65,536 | 4 | 16,384 | ~418 KB |

**Trade-off with verifier time:** Wider matrices (more columns, fewer rows) reduce proof size but increase verifier time, since the verifier must read and hash more column data. When configuring parameters, consider both proof size and verification latency requirements for your use case.

### `profile_encode` — Profiling Example

Not a Criterion benchmark, but a standalone binary for profiling `encode_rows`
(IPRS depth-2, rate 1/2, poly_size=2^16, 100 iterations).

```bash
cargo run --example profile_encode --release -p zip-plus --features "asm simd parallel"
```

Add `--features pntt-timing` for a detailed PNTT stage breakdown:

```bash
cargo run --example profile_encode --release -p zip-plus --features "asm simd parallel pntt-timing"
```

---

## `piop` Benchmarks

### `sumcheck` — Sumcheck Protocol

Benchmarks the sumcheck prover and verifier for a simple product relation
(`a·b − c`) over random Montgomery fields (`MontyField<3>` and `MontyField<4>`),
with witness sizes from 2^13 to 2^17.

```bash
cargo bench --bench sumcheck -p zinc-piop
```

Supports `parallel` feature:

```bash
cargo bench --bench sumcheck -p zinc-piop --features "parallel"
```

**Filter examples:**

```bash
# Only prover benchmarks
cargo bench --bench sumcheck -p zinc-piop -- "Prover"
# Only verifier benchmarks
cargo bench --bench sumcheck -p zinc-piop -- "Verifier"
```

---

## `poly` Benchmarks

### `mle_evaluation` — Multilinear Extension Evaluation

Benchmarks `DenseMultilinearExtension::evaluate` and `evaluate_with_config`
over a 256-bit Montgomery field for 0 to 19 variables.

```bash
cargo bench --bench mle_evaluation -p zinc-poly
```

### `nat_evaluation` — Natural Evaluation Domain

Benchmarks `NatEvaluatedPoly::evaluate_at_point` over a 256-bit Montgomery
field for degrees from 2^0 to 2^15.

```bash
cargo bench --bench nat_evaluation -p zinc-poly
```

---

## Running All Benchmarks

```bash
# All benchmarks, all crates, with performance features
cargo bench --features "parallel simd asm unchecked-butterfly"
```

## Viewing Reports

Criterion generates HTML reports under `target/criterion/`. Open the top-level
index to browse all results:

```bash
open target/criterion/report/index.html
```

Each benchmark group also has its own report with violin plots and
regression analysis at `target/criterion/<group-name>/report/index.html`.
