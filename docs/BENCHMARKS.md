# Benchmarks

All benchmarks use [Criterion.rs](https://github.com/bgruber/criterion.rs) and live under
each crate's `benches/` directory.

## Quick Reference

| Crate | Bench name | What it measures |
|---|---|---|
| `zip-plus` | `zip_benches` | Zip PCS (integer coefficients): RAA & IPRS encoding, commit, 128-bit & 256-bit encoding |
| `zip-plus` | `zip_plus_benches` | Zip+ PCS (polynomial coefficients): RAA, IPRS (multiple fields/depths/rates), commit/test/evaluate comparisons |
| `zip-plus` | `zip_plus_commit_10` | Zip+ 10 polys (F65537): commit, test, evaluate, verify across 6–11 vars |
| `zip-plus` | `zip_plus_commit_15` | Zip+ 15 polys (F65537): commit, test, evaluate, verify across 6–11 vars |
| `zip-plus` | `zip_plus_commit_40` | Zip+ 40 polys (F65537): commit, test, evaluate, verify across 6–11 vars |
| `zip-plus` | `zip_plus_commit_55` | Zip+ 55 polys (F65537): commit, test, evaluate, verify across 6–11 vars |
| `zip-plus` | `zip_commit_5` | Zip 5 polys (F65537): commit, test, evaluate, verify across 6–11 vars |
| `zip-plus` | `zip_commit_40` | Zip 40 polys (F65537): commit, test, evaluate, verify across 6–11 vars |
| `zip-plus` | `zip_plus_commit_10_f12289` | Zip+ 10 polys (F12289): commit, test, evaluate, verify across 6–10 vars |
| `zip-plus` | `zip_plus_commit_15_f12289` | Zip+ 15 polys (F12289): commit, test, evaluate, verify across 6–10 vars |
| `zip-plus` | `zip_plus_commit_40_f12289` | Zip+ 40 polys (F12289): commit, test, evaluate, verify across 6–10 vars |
| `zip-plus` | `zip_plus_commit_55_f12289` | Zip+ 55 polys (F12289): commit, test, evaluate, verify across 6–10 vars |
| `zip-plus` | `zip_commit_5_f12289` | Zip 5 polys (F12289): commit, test, evaluate, verify across 6–10 vars |
| `zip-plus` | `zip_commit_40_f12289` | Zip 40 polys (F12289): commit, test, evaluate, verify across 6–10 vars |
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
| `Zip+ IPRS F65537 Depth2` | IPRS depth-2, row_len=2048, rate 1/2. Matrix sizes from 4×2048 to 64×2048. Includes commit, test, evaluate |
| `Zip+ IPRS F12289 Depth3 2^11` | IPRS depth-3 over F12289, row_len=2048, rate 1/2. Matrix sizes 4×2048 to 64×2048 |
| `Zip+ IPRS F65537 Depth2 Rate1_4` | IPRS depth-2, row_len=4096, rate 1/4 (i64-safe). Matrix sizes 4×4096 to 32×4096 |
| `Zip+ IPRS F65537 Depth2 Wide` | IPRS depth-2, rate 1/2, fixed 4 rows with increasing row_len (4096→32768). Explores wide matrices: 4×4096, 4×8192, 4×16384, 4×32768 |
| `Zip+ Commit Comparison` | Commit with optimal matrix shapes: 2×2^12, 2×2^13, 4×2^13, 4×2^14, 8×2^14. Uses depth-2 for msg sizes 2^12–2^14 and depth-3 for msg size 2^14 |
| `Zip+ Test Comparison` | Test phase (proximity test) with the same optimal matrix shapes as Commit Comparison |
| `Zip+ Evaluate Comparison` | Evaluate phase (evaluation proof generation) with the same optimal matrix shapes as Commit Comparison |
| `Zip+ Commit Comparison Rate1_4` | Same matrix shapes as Commit Comparison but with rate 1/4 IPRS codes (all depth-2, i64 coefficients) |
| `Zip+ Test Comparison Rate1_4` | Test phase with rate 1/4 IPRS codes (same matrix shapes as Commit Comparison) |
| `Zip+ Evaluate Comparison Rate1_4` | Evaluate phase with rate 1/4 IPRS codes (same matrix shapes as Commit Comparison) |
| `Zip+ Commit Comparison Rate1_4 Depth3` | Same matrix shapes as Commit Comparison but with rate 1/4 depth-3 IPRS codes (i128 coefficients) |
| `Zip+ Test Comparison Rate1_4 Depth3` | Test phase with rate 1/4 depth-3 IPRS codes |
| `Zip+ Evaluate Comparison Rate1_4 Depth3` | Evaluate phase with rate 1/4 depth-3 IPRS codes |
| `Zip+ Commit Comparison Rate1_4 Depth4` | Same matrix shapes as Commit Comparison but with rate 1/4 depth-4 IPRS codes (i128 coefficients, smallest base matrices) |
| `Zip+ Test Comparison Rate1_4 Depth4` | Test phase with rate 1/4 depth-4 IPRS codes |
| `Zip+ Evaluate Comparison Rate1_4 Depth4` | Evaluate phase with rate 1/4 depth-4 IPRS codes |
| `Zip+ Commit 10 Polys 8 Vars` | Commit 10 polys with 8 vars (1×256 matrix), depth-1 & depth-2 rate 1/4 |

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
| 2×4096 | 2^13 | Base64 Depth2 | 2 | 2^12 | ~3.69 ms |
| 2×8192 | 2^14 | Base128 Depth2 | 2 | 2^13 | ~7.97 ms |
| 4×8192 | 2^15 | Base128 Depth2 | 2 | 2^13 | ~14.4 ms |
| 4×16384 | 2^16 | Depth3 | 3 | 2^14 | ~20.5 ms |
| 8×16384 | 2^17 | Depth3 | 3 | 2^14 | ~39.9 ms |

### Benchmark Results: Test Comparison

Results from the `Zip+ Test Comparison` group with `--features "parallel simd asm unchecked-butterfly"` (Apple M-series).
The test phase performs the proximity test: it re-encodes each row and checks that the committed codeword is close to the encoding.

| Matrix | Entries | IPRS Config | Depth | msg_size | Time |
|--------|---------|-------------|-------|----------|------|
| 2×4096 | 2^13 | Base64 Depth2 | 2 | 2^12 | ~535 µs |
| 2×8192 | 2^14 | Base128 Depth2 | 2 | 2^13 | ~1.01 ms |
| 4×8192 | 2^15 | Base128 Depth2 | 2 | 2^13 | ~1.64 ms |
| 4×16384 | 2^16 | Depth3 | 3 | 2^14 | ~2.40 ms |
| 8×16384 | 2^17 | Depth3 | 3 | 2^14 | ~3.05 ms |

### Benchmark Results: Evaluate Comparison

Results from the `Zip+ Evaluate Comparison` group with `--features "parallel simd asm unchecked-butterfly"` (Apple M-series).
The evaluate phase generates the evaluation proof: it reduces the polynomial evaluation claim to column openings via the sumcheck protocol and computes the final field evaluation.

| Matrix | Entries | IPRS Config | Depth | msg_size | Time |
|--------|---------|-------------|-------|----------|------|
| 2×4096 | 2^13 | Base64 Depth2 | 2 | 2^12 | ~1.60 ms |
| 2×8192 | 2^14 | Base128 Depth2 | 2 | 2^13 | ~2.25 ms |
| 4×8192 | 2^15 | Base128 Depth2 | 2 | 2^13 | ~3.96 ms |
| 4×16384 | 2^16 | Depth3 | 3 | 2^14 | ~6.22 ms |
| 8×16384 | 2^17 | Depth3 | 3 | 2^14 | ~10.8 ms |

### Benchmark Results: Commit Comparison Rate 1/4

Results from the `Zip+ Commit Comparison Rate1_4` group (depth-2, i64 coefficients, rate 1/4):

| Matrix | Entries | IPRS Config | Depth | msg_size | Time |
|--------|---------|-------------|-------|----------|------|
| 2×4096 | 2^13 | Base64 Depth2 | 2 | 2^12 | ~7.26 ms |
| 2×8192 | 2^14 | Base128 Depth2 | 2 | 2^13 | ~13.4 ms |
| 4×8192 | 2^15 | Base128 Depth2 | 2 | 2^13 | ~26.3 ms |
| 4×16384 | 2^16 | Base256 Depth2 | 2 | 2^14 | ~77.3 ms |
| 8×16384 | 2^17 | Base256 Depth2 | 2 | 2^14 | ~312 ms |

### Benchmark Results: Test Comparison Rate 1/4

Results from the `Zip+ Test Comparison Rate1_4` group (depth-2, rate 1/4):

| Matrix | Entries | msg_size | Time |
|--------|---------|----------|------|
| 2×4096 | 2^13 | 2^12 | ~263 µs |
| 2×8192 | 2^14 | 2^13 | ~347 µs |
| 4×8192 | 2^15 | 2^13 | ~447 µs |
| 4×16384 | 2^16 | 2^14 | ~722 µs |
| 8×16384 | 2^17 | 2^14 | ~1.51 ms |

### Benchmark Results: Evaluate Comparison Rate 1/4

Results from the `Zip+ Evaluate Comparison Rate1_4` group (depth-2, rate 1/4):

| Matrix | Entries | msg_size | Time |
|--------|---------|----------|------|
| 2×4096 | 2^13 | 2^12 | ~1.61 ms |
| 2×8192 | 2^14 | 2^13 | ~2.37 ms |
| 4×8192 | 2^15 | 2^13 | ~3.90 ms |
| 4×16384 | 2^16 | 2^14 | ~6.47 ms |
| 8×16384 | 2^17 | 2^14 | ~5.12 ms |

### Benchmark Results: 128-bit Encoding (Selected)

Results from the `Zip Encode 128-bit Selected` group (`Int<2>` evaluations, rate 1/2, F65537):

| msg_size | IPRS Config | Depth | Cw type | Time |
|----------|-------------|-------|---------|------|
| 2^12 = 4096 | Base64 Depth2 | 2 | `Int<3>` | ~4.40 ms |
| 2^13 = 8192 | Base128 Depth2 | 2 | `Int<3>` | ~16.7 ms |
| 2^14 = 16384 | Depth2 | 2 | `Int<3>` | ~14.2 ms |
| 2^14 = 16384 | Depth3 | 3 | `Int<4>` | ~25.0 ms |

> **Note:** The depth-3 case for 2^14 requires `Int<4>` (256-bit) codewords because
> 3 recursion levels overflow the default `Int<3>` (192-bit) codeword type.

### Benchmark Results: IPRS F65537 Depth2

Results from the `Zip+ IPRS F65537 Depth2` group (row_len=2048, rate 1/2):

| Matrix | Entries | Operation | Time |
|--------|---------|-----------|------|
| 4×2048 | 2^13 | Commit | ~2.08 ms |
| 8×2048 | 2^14 | Commit | ~3.99 ms |
| 16×2048 | 2^15 | Commit | ~6.65 ms |
| 32×2048 | 2^16 | Commit | ~12.9 ms |
| 64×2048 | 2^17 | Commit | ~24.1 ms |
| 1×2048 | 2^11 | Test | ~62.6 µs |
| 1×2048 | 2^11 | Evaluate | ~542 µs |

### Benchmark Results: IPRS F65537 Depth3

Results from the `Zip+ IPRS F65537 Depth3` group (row_len=4096, rate 1/2):

| Matrix | Entries | Time |
|--------|---------|------|
| 2×4096 | 2^13 | ~2.49 ms |
| 4×4096 | 2^14 | ~3.60 ms |
| 8×4096 | 2^15 | ~6.31 ms |
| 16×4096 | 2^16 | ~13.0 ms |
| 32×4096 | 2^17 | ~22.1 ms |

### Benchmark Results: IPRS F65537 Depth2 Rate 1/4

Results from the `Zip+ IPRS F65537 Depth2 Rate1_4` group (row_len=4096, rate 1/4):

| Matrix | Entries | Time |
|--------|---------|------|
| 4×4096 | 2^14 | ~7.83 ms |
| 8×4096 | 2^15 | ~14.6 ms |
| 16×4096 | 2^16 | ~28.5 ms |
| 32×4096 | 2^17 | ~56.8 ms |

### Benchmark Results: IPRS F65537 Depth2 Wide

Results from the `Zip+ IPRS F65537 Depth2 Wide` group (fixed 4 rows, increasing row_len, rate 1/2):

| Matrix | Entries | Time |
|--------|---------|------|
| 4×4096 | 2^14 | ~4.16 ms |
| 4×8192 | 2^15 | ~11.4 ms |
| 4×16384 | 2^16 | ~36.8 ms |
| 4×32768 | 2^17 | ~124 ms |

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

---

### Multi-Polynomial Benchmarks (`zip_plus_commit_*` / `zip_commit_*`)

These standalone bench targets measure **commit + test + evaluate + verify** for multiple
multilinear polynomials simultaneously. Each one sweeps over `num_vars` = 6–11 (F65537) or
6–10 (F12289), using a **1-row matrix** (1 × 2^num_vars). The IPRS codes use depth-1 and
depth-2 at rate 1/4.

Two families exist:

- **Zip+** (`zip_plus_commit_*`): polynomial evaluations (`BinaryPoly<32>` → `DensePolynomial<i64, 32>`)
- **Zip** (`zip_commit_*`): integer evaluations (`i32` → `Int<2>` codewords)

```bash
# Example: Zip+ 55 polys over F65537
cargo bench --bench zip_plus_commit_55 -p zip-plus --features "parallel simd asm unchecked-butterfly"

# Example: Zip+ 15 polys over F12289
cargo bench --bench zip_plus_commit_15_f12289 -p zip-plus --features "parallel simd asm unchecked-butterfly"

# Example: Zip (integer) 5 polys over F65537
cargo bench --bench zip_commit_5 -p zip-plus --features "parallel simd asm unchecked-butterfly"
```

**Available bench targets and polynomial counts:**

| Bench name | Eval type | Polys | Field | Vars |
|---|---|---|---|---|
| `zip_plus_commit_10` | Zip+ (`BinaryPoly<32>`) | 10 | F65537 | 6–11 |
| `zip_plus_commit_15` | Zip+ (`BinaryPoly<32>`) | 15 | F65537 | 6–11 |
| `zip_plus_commit_40` | Zip+ (`BinaryPoly<32>`) | 40 | F65537 | 6–11 |
| `zip_plus_commit_55` | Zip+ (`BinaryPoly<32>`) | 55 | F65537 | 6–11 |
| `zip_commit_5` | Zip (`i32`) | 5 | F65537 | 6–11 |
| `zip_commit_40` | Zip (`i32`) | 40 | F65537 | 6–11 |
| `zip_plus_commit_10_f12289` | Zip+ (`BinaryPoly<32>`) | 10 | F12289 | 6–10 |
| `zip_plus_commit_15_f12289` | Zip+ (`BinaryPoly<32>`) | 15 | F12289 | 6–10 |
| `zip_plus_commit_40_f12289` | Zip+ (`BinaryPoly<32>`) | 40 | F12289 | 6–10 |
| `zip_plus_commit_55_f12289` | Zip+ (`BinaryPoly<32>`) | 55 | F12289 | 6–10 |
| `zip_commit_5_f12289` | Zip (`i32`) | 5 | F12289 | 6–10 |
| `zip_commit_40_f12289` | Zip (`i32`) | 40 | F12289 | 6–10 |

Each bench file defines one benchmark group **per code type per phase**.
The group names follow the pattern:

- F65537: `Zip+ {Op} {N} Polys IPRS-{depth}-1/4-F65537`
- F12289: `Zip+ {Op} F12289 {N} Polys IPRS-{depth}-{rate}-F12289`
- Zip (non-plus): `Zip {Op} {N} Polys IPRS-...`

The IPRS tag format is `IPRS-X-Y-Z` where **X** = depth, **Y** = rate, **Z** = base field.

For **F65537** bench files there are 8 groups (4 phases × 2 codes):

| Group name pattern | Code |
|---|---|
| `Zip+ {Op} {N} Polys IPRS-1-1/4-F65537` | Depth-1, rate 1/4 |
| `Zip+ {Op} {N} Polys IPRS-2-1/4-F65537` | Depth-2, rate 1/4 |

For **F12289** bench files there are 16 groups (4 phases × 4 codes):

| Group name pattern | Code |
|---|---|
| `Zip+ {Op} F12289 {N} Polys IPRS-1-1/2-F12289` | Depth-1, rate 1/2 |
| `Zip+ {Op} F12289 {N} Polys IPRS-2-1/2-F12289` | Depth-2, rate 1/2 |
| `Zip+ {Op} F12289 {N} Polys IPRS-1-1/4-F12289` | Depth-1, rate 1/4 |
| `Zip+ {Op} F12289 {N} Polys IPRS-2-1/4-F12289` | Depth-2, rate 1/4 |

**Filter examples:**

```bash
# Only depth-1 rate 1/4 over F65537 (all phases)
cargo bench --bench zip_plus_commit_55 -p zip-plus -- "IPRS-1-1/4-F65537"
# Only depth-2 (both rates, all phases)
cargo bench --bench zip_plus_commit_55_f12289 -p zip-plus -- "IPRS-2"
# Only commit phase (all codes)
cargo bench --bench zip_plus_commit_55 -p zip-plus -- "Commit"
# Only rate 1/2 benchmarks (F12289)
cargo bench --bench zip_plus_commit_55_f12289 -p zip-plus -- "1/2-F12289"
# Only verify phase for a specific code
cargo bench --bench zip_plus_commit_55 -p zip-plus -- "Verify.*IPRS-2"
```

### Benchmark Results: Zip+ Multi-Polynomial (8 Vars, matrix=1×256)

Results from the `*_8 Vars` groups with `--features "parallel simd asm unchecked-butterfly"` (Apple M-series).
These use a fixed 1×256 matrix (8 vars) at rate 1/4 over F65537.

#### Commit (8 Vars)

| Polys | Depth-1 | Depth-2 |
|-------|---------|---------|
| 10 | ~4.25 ms | ~4.15 ms |
| 15 | ~6.41 ms | ~6.54 ms |
| 40 | ~17.4 ms | ~17.5 ms |
| 50 | ~24.0 ms | ~21.9 ms |
| 55 | ~26.9 ms | ~24.2 ms |

#### Test (8 Vars)

| Polys | Depth-1 | Depth-2 |
|-------|---------|---------|
| 10 | ~770 µs | ~625 µs |
| 15 | ~904 µs | ~1.16 ms |
| 40 | ~2.53 ms | ~2.66 ms |
| 50 | ~3.70 ms | ~3.60 ms |
| 55 | ~3.84 ms | ~3.75 ms |

#### Evaluate (8 Vars)

| Polys | Depth-1 | Depth-2 |
|-------|---------|---------|
| 10 | ~3.52 ms | ~3.48 ms |
| 15 | ~6.01 ms | ~6.05 ms |
| 50 | ~17.6 ms | ~17.9 ms |
| 55 | ~19.3 ms | ~19.3 ms |

#### Verify (8 Vars)

| Polys | Depth-1 | Depth-2 |
|-------|---------|---------|
| 10 | ~15.7 ms | ~14.6 ms |
| 15 | ~24.4 ms | ~22.1 ms |
| 50 | ~75.2 ms | ~73.3 ms |
| 55 | ~83.4 ms | ~79.9 ms |

### Benchmark Results: Zip+ 55 Polys — Varying Matrix Width

Results from the `Zip+ {Op} 55 Polys` groups, sweeping num_vars from 6 to 11
(matrix widths 64 to 2048, all with 1 row). Depth-1 vs depth-2 over F65537, rate 1/4.

#### Commit 55 Polys

| Matrix | Depth-1 | Depth-2 |
|--------|---------|---------|
| 1×64 | ~12.8 ms | ~14.3 ms |
| 1×128 | ~16.5 ms | ~18.2 ms |
| 1×256 | ~24.0 ms | ~24.5 ms |
| 1×512 | ~39.0 ms | ~37.6 ms |
| 1×1024 | ~72.5 ms | ~57.3 ms |
| 1×2048 | ~187 ms | ~93.6 ms |

#### Test 55 Polys

| Matrix | Depth-1 | Depth-2 |
|--------|---------|---------|
| 1×64 | ~2.91 ms | ~3.71 ms |
| 1×128 | ~4.27 ms | ~3.72 ms |
| 1×256 | ~5.44 ms | ~4.81 ms |
| 1×512 | ~5.69 ms | ~4.94 ms |
| 1×1024 | ~6.00 ms | ~5.45 ms |
| 1×2048 | ~6.84 ms | ~6.68 ms |

#### Evaluate 55 Polys

| Matrix | Depth-1 | Depth-2 |
|--------|---------|---------|
| 1×64 | ~15.7 ms | ~17.1 ms |
| 1×128 | ~18.5 ms | ~23.4 ms |
| 1×256 | ~22.3 ms | ~20.4 ms |
| 1×512 | ~28.4 ms | ~27.3 ms |
| 1×1024 | ~37.2 ms | ~36.5 ms |
| 1×2048 | ~47.5 ms | ~43.3 ms |

#### Verify 55 Polys

| Matrix | Depth-1 |
|--------|---------|
| 1×64 | ~52.6 ms |
| 1×128 | ~63.3 ms |
| 1×256 | ~81.9 ms |
| 1×512 | ~143 ms |

### Benchmark Results: Zip+ 15 Polys — Varying Matrix Width

Results from the `Zip+ {Op} 15 Polys` groups, sweeping num_vars from 6 to 11.

#### Commit 15 Polys

| Matrix | Depth-1 | Depth-2 |
|--------|---------|---------|
| 1×64 | ~3.55 ms | ~3.93 ms |
| 1×128 | ~4.50 ms | ~4.96 ms |
| 1×256 | ~6.51 ms | ~6.48 ms |
| 1×512 | ~10.4 ms | ~9.06 ms |
| 1×1024 | ~22.6 ms | ~13.7 ms |
| 1×2048 | ~51.3 ms | ~25.6 ms |

#### Test 15 Polys

| Matrix | Depth-1 | Depth-2 |
|--------|---------|---------|
| 1×64 | ~844 µs | ~794 µs |
| 1×128 | ~893 µs | ~990 µs |
| 1×256 | ~943 µs | ~1.23 ms |
| 1×512 | ~1.04 ms | ~1.25 ms |
| 1×1024 | ~1.22 ms | ~1.17 ms |
| 1×2048 | ~1.48 ms | ~1.32 ms |

#### Evaluate 15 Polys

| Matrix | Depth-1 | Depth-2 |
|--------|---------|---------|
| 1×64 | ~3.84 ms | ~3.85 ms |
| 1×128 | ~4.64 ms | ~4.36 ms |
| 1×256 | ~5.14 ms | ~5.16 ms |
| 1×512 | ~6.25 ms | ~6.13 ms |
| 1×1024 | ~8.03 ms | ~7.61 ms |
| 1×2048 | ~11.5 ms | ~10.5 ms |

#### Verify 15 Polys

| Matrix | Depth-1 | Depth-2 |
|--------|---------|---------|
| 1×64 | ~13.6 ms | ~14.0 ms |
| 1×128 | ~16.2 ms | ~17.0 ms |
| 1×256 | ~22.0 ms | ~21.3 ms |
| 1×512 | ~38.2 ms | ~29.9 ms |
| 1×1024 | ~88.0 ms | ~49.8 ms |
| 1×2048 | ~277 ms | ~97.1 ms |

### Benchmark Results: Zip (baseline) — 5 and 40 Polys (8 Vars, matrix=1×256)

Results from the Zip baseline benchmarks using integer evaluations (`i32` → `Int<2>`):

#### Zip 5 Polys (8 Vars)

| Operation | Depth-1 | Depth-2 |
|-----------|---------|---------|
| Commit | ~2.10 ms | ~2.02 ms |
| Test | ~319 µs | ~337 µs |
| Evaluate | ~1.74 ms | ~1.75 ms |
| Verify | ~5.93 ms | ~5.99 ms |

#### Zip 40 Polys (8 Vars)

| Operation | Depth-1 | Depth-2 |
|-----------|---------|---------|
| Commit | ~16.9 ms | ~15.5 ms |
| Test | ~2.70 ms | ~2.54 ms |
| Evaluate | ~14.0 ms | ~14.2 ms |
| Verify | ~46.4 ms | ~45.0 ms |

### Automated Benchmark Script: Depth-1 F12289 (10 & 55 Polys)

The script `scripts/run_depth2_benchmarks.py` runs depth-1 IPRS benchmarks over F12289
for 10 and 55 polynomials at both rates 1/2 and 1/4, then collects the Criterion results
and generates a LaTeX table.

```bash
# Run all four benchmark commands and print the LaTeX table
python3 scripts/run_depth2_benchmarks.py

# Skip running benchmarks; only collect existing Criterion results
python3 scripts/run_depth2_benchmarks.py --dry-run

# Save the LaTeX table to a file
python3 scripts/run_depth2_benchmarks.py --output table.tex
```

The script executes these four commands in sequence:

```bash
cargo bench --bench zip_plus_commit_10_f12289 --features "asm parallel simd unchecked-butterfly" -- "IPRS-1-1/4-F12289"
cargo bench --bench zip_plus_commit_10_f12289 --features "asm parallel simd unchecked-butterfly" -- "IPRS-1-1/2-F12289"
cargo bench --bench zip_plus_commit_55_f12289 --features "asm parallel simd unchecked-butterfly" -- "IPRS-1-1/2-F12289"
cargo bench --bench zip_plus_commit_55_f12289 --features "asm parallel simd unchecked-butterfly" -- "IPRS-1-1/4-F12289"
```

It then reads median timings from Criterion's JSON output and produces a table with
columns for Commit, Test, Evaluate, and Verify across `num_vars` 6–10, grouped by
polynomial count and rate.

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
