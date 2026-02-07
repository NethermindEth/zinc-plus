# Benchmarks

All benchmarks use [Criterion.rs](https://github.com/bgruber/criterion.rs) and live under
each crate's `benches/` directory.

## Quick Reference

| Crate | Bench name | What it measures |
|---|---|---|
| `zip-plus` | `zip_benches` | Zip PCS (integer coefficients): RAA & IPRS encoding, commit, 256-bit encoding |
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
| `Zip IPRS Matrix Shapes` | IPRS depth-1, rate 1/2 | Single-row encode with base 16/32/64 matrix shapes |
| `Zip IPRS rate1_4` | IPRS depth-2, rate 1/4 | Commit with IPRS (F65537) at rate 1/4 |
| `Zip IPRS rate1_4 Matrix Shapes` | IPRS depth-1, rate 1/4 | Single-row encode with base 16/32/64 at rate 1/4 |
| `Zip Encode 256-bit` | IPRS depth-3, 256-bit ints | Encode `Int<4>` → `Int<5>` at rate 1/2 and 1/4 |

**Filter examples:**

```bash
# Only the basic Zip+ RAA benchmarks
cargo bench --bench zip_benches -p zip-plus -- "Zip\+"
# Only IPRS rate 1/4
cargo bench --bench zip_benches -p zip-plus -- "rate1_4"
# Only 256-bit encoding
cargo bench --bench zip_benches -p zip-plus -- "256-bit"
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

**Filter examples:**

```bash
# Only F65537 Depth3
cargo bench --bench zip_plus_benches -p zip-plus -- "F65537 Depth3"
# Only F65537 Depth2 at rate 1/4
cargo bench --bench zip_plus_benches -p zip-plus -- "F65537 Depth2 Rate1_4"
# Only F12289
cargo bench --bench zip_plus_benches -p zip-plus -- "F12289"
# Only RAA
cargo bench --bench zip_plus_benches -p zip-plus -- "RAA"
# A specific matrix size
cargo bench --bench zip_plus_benches -p zip-plus -- "matrix=16x4096"
```

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
