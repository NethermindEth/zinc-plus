# Agent Setup Instructions

Two agents are needed to complete the Zinc+ SNARK implementation. Each tackles a distinct missing piece.

---

## Current State

**Workspace:** `/Users/albertgarretafontelles/dev/zinc-plus-new/`

### What works today

- **SHA-256 UAIR** (`sha256-uair/src/lib.rs`): 6 of 14 constraints implemented over F₂[X]/(X³²−1) — 4 rotation constraints (Σ₀, Σ₁, σ₀, σ₁) + 2 shift decompositions.
- **Full pipeline** (`snark/src/pipeline.rs`): `prove()` and `verify()` functions wire IdealCheck → CombinedPolyResolver → PCS. Serializes/deserializes all PIOP proof data.
- **Full pipeline test** (`snark/tests/full_pipeline.rs`): prove→verify round-trip passes.
- **ECDSA UAIR** (`ecdsa-uair/src/lib.rs`): 258×14 trace dimensions, random witness generation, constraint math specified in `ecdsa-uair/src/constraints.rs`. Zero constraints implemented.
- **Benchmarks** (`snark/benches/e2e_sha256.rs`): Full pipeline benchmark.
- **15 tests pass, 0 warnings.**

### Benchmark numbers (1×SHA-256, R4B16 DEPTH=1, poly=2⁷)

| Metric        | Value     |
|---------------|-----------|
| Full Prover   | 13.9 ms   |
| Full Verifier | 1.8 ms    |
| PCS proof     | 405.5 KB  |
| PIOP proof    | 11.7 KB   |
| Total raw     | 417.2 KB  |
| Compressed    | 190.6 KB  |
| Target limit  | 800 KB ✓  |

---

## Agent 1: Multi-Ring IdealCheck + SHA-256 Q[X] Constraints

### Goal

Implement SHA-256 constraints 7–14 which operate in Z[X] (integer polynomial arithmetic) and cannot be expressed in F₂[X] because the constant `2` is zero in F₂.

### The Problem

The `IdealCheckProtocol` in `piop/src/ideal_check.rs` is hardcoded to traces of type `&[DenseMultilinearExtension<BinaryPoly<DEGREE_PLUS_ONE>>]`. The Q[X] constraints need a trace over `DensePolynomial<i64, D>` (integer-coefficient polynomials) where `D` ≥ 64 (to accommodate `μ·X³²` terms with degree 63).

### Key Files to Modify

1. **`piop/src/ideal_check.rs`** — Generalize `prove_as_subprotocol` and `verify_as_subprotocol` from `BinaryPoly<D>` to a generic ring `R` that implements `CoefficientProjectable<C, D>`. The `CombinedPolyResolver` in `piop/src/combined_poly_resolver.rs` is *already* generic over `R: Semiring` — follow its pattern.

2. **`piop/src/ideal_check/utils.rs`** — `project_trace_matrix` and `project_scalars` are hardcoded to `BinaryPoly<D>`. Generalize them.

3. **`piop/src/ideal_check/combined_poly_builder.rs`** — `compute_combined_polynomials` takes `BinaryPoly<D>`. Generalize.

4. **`sha256-uair/src/lib.rs`** — Add `impl Uair<DensePolynomial<i64, 64>> for Sha256Uair` with the 8 Q[X] constraints (see the detailed spec already in the file at the bottom of `constrain_general`).

5. **`snark/src/pipeline.rs`** — Extend `prove()` and `verify()` to run **two** IdealCheck sub-protocols on the same transcript: one for F₂[X] constraints (`Uair<BinaryPoly<32>>`), one for Z[X] constraints (`Uair<DensePolynomial<i64, 64>>`). The CPR then combines the subclaims.

### Existing Infrastructure You Can Use

- **`DensePolynomial<i64, D>`** already implements `Semiring` (`poly/src/univariate/dense.rs`, line ~378).
- **`CoefficientProjectable<R, D>` for `DensePolynomial<R, D>`** already exists (`poly/src/univariate/dense.rs`, line ~600).
- **`DegreeOneIdeal<i64>`** (`poly/src/univariate/ideal.rs`) handles the `(X−2)` ideal. Create it with `DegreeOneIdeal::new(2_i64)`. Its `IdealCheck<DynamicPolynomialF<F>>` evaluates the polynomial at `2` in F_p — this IS sound for large primes.
- **`IdealOrZero<DegreeOneIdeal<F>>`** already has `IdealCheck<DynamicPolynomialF<F>>`.

### Constraint Details (constraints 7–14)

All mathematical details are already written in `sha256-uair/src/lib.rs` (lines ~298–380). Summary:

- **C7–C9 (BitPoly lookups):** `e + f − 2·ch_ef` etc. — verify AND/Maj by checking all coefficients are in {0,1}. Express as `assert_zero(c_i · (c_i − 1))` per coefficient, or as a polynomial identity.
- **C10–C11 (carry propagation):** State updates `â[t+1] − h − Σ₁ − Ch − ... + μ_a·X³² ∈ (X−2)`. Use `assert_in_ideal` with `DegreeOneIdeal::new(2)`.
- **C12–C13 (register delay):** `d[t+1] − a[t] = 0`. These need the `down` row. Express as `assert_zero(down[COL_D] − up[COL_A])`.
- **C14 (message schedule):** `Ŵ[t] − Ŵ[t−16] − σ̂₀ − Ŵ[t−7] − σ̂₁ + μ_W·X³² ∈ (X−2)`. Requires multi-row lookback — may need intermediate columns or a sliding-window trace layout.

### Architecture Constraint

The `Uair::constrain_general` API only gives access to `up` (current row) and `down` (next row). Multi-row lookbacks (t−3, t−7, t−15, t−16) need either:
- **Additional intermediate columns** that relay values forward using 1-step delay chains.
- **A modified UAIR trait** with wider row access (would be a bigger refactor).

The simpler path is adding relay columns to the trace.

### Verification Approach

After generalizing `IdealCheckProtocol`, the IC verifier's `ideal_over_f_from_ref` closure will use the REAL `DegreeOneIdeal` (not `TrivialIdeal`) because `p(2) = 0` in Z[X] lifts correctly to `p(2) = 0` in F_p for any prime p > 2³⁶. This is a meaningful ideal check — unlike the F₂[X] case where the cyclotomic ideal doesn't lift.

### Testing

- Extend the witness generator (`sha256-uair/src/witness.rs`) to produce valid carry polynomials (μ_a, μ_e, μ_W).
- Add a test in `snark/tests/full_pipeline.rs` that runs the dual-IC pipeline.
- Run `cargo test -p zinc-snark -p zinc-sha256-uair` — all existing 15 tests must still pass.

---

## Agent 2: ECDSA UAIR Constraints

### Goal

Implement the 11 ECDSA verification constraints and wire them through the pipeline so that a combined SHA-256 + ECDSA benchmark can run.

### The Problem

ECDSA constraints operate in F_p (secp256k1 base field, 256-bit) and F_n (group order, 256-bit). These are fundamentally different from F₂[X] and require:
- 256-bit field arithmetic (Montgomery multiplication)
- A Jacobian point doubling/addition formula with conditional logic (Shamir selector)
- Two distinct finite fields (F_p for coordinates, F_n for scalars)

### Key Files

1. **`ecdsa-uair/src/lib.rs`** — Currently has `NUM_CONSTRAINTS: usize = 0` and an empty `constrain_general`. Implement the 11 constraints.

2. **`ecdsa-uair/src/constraints.rs`** — Full mathematical specification of all 11 constraints already written. Use as reference.

3. **`ecdsa-uair/src/witness.rs`** — Currently generates random traces. Must generate a *valid* ECDSA verification witness: pick a key pair, sign a message, compute the Shamir double-and-add trace with correct intermediate values.

4. **`snark/src/pipeline.rs`** — The combined 8×SHA-256 + ECDSA benchmark needs a pipeline that can handle 33 columns (19 SHA + 14 ECDSA) at poly_size 2¹⁰.

### Ring Choice

ECDSA constraints fundamentally need a 256-bit prime field, not F₂[X]. The approach depends on whether Agent 1 has completed the multi-ring `IdealCheckProtocol` generalization:

**If multi-ring IdealCheck is available:** Use `DensePolynomial<FieldElement, 1>` as the ring (degree-0 polynomials = field elements). Define `impl Uair<Fp256> for EcdsaUair` where `Fp256` is a secp256k1 field element type.

**If multi-ring IdealCheck is NOT yet available:** Encode F_p elements as `BinaryPoly<256>` (256-bit binary representation) and express constraints using the `(X−2)` ideal for integer arithmetic. Each multiplication `a·b mod p` becomes a constraint.  This is less clean but works within the existing framework.

**Simplest interim approach:** Keep constraints as documentation. Use random traces for PCS benchmarking (which is already implemented and working — the benchmark creates random 33-column traces). The PCS timing is the bottleneck and is ring-agnostic. PIOP timing for ECDSA is a small additive cost.

### secp256k1 Parameters

Already documented in `ecdsa-uair/src/lib.rs` module docs:

```
p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
Curve: y² = x³ + 7
```

### Constraint Summary

All 11 constraints are `assert_zero` (exact equality in F_p or F_n):

| #  | Name              | Ring | Formula sketch                           |
|----|-------------------|------|------------------------------------------|
| 1  | u₁ scalar accum   | F_n  | u₁[t+1] − 2·u₁[t] − b₁[t] = 0         |
| 2  | u₂ scalar accum   | F_n  | u₂[t+1] − 2·u₂[t] − b₂[t] = 0         |
| 3  | Doubling S        | F_p  | S[t] − Y[t]² = 0                        |
| 4  | Doubling Z_mid    | F_p  | Z_mid[t] − 2·Y[t]·Z[t] = 0             |
| 5  | Doubling X_mid    | F_p  | X_mid[t] − M² + 2·U = 0                 |
| 6  | Doubling Y_mid    | F_p  | Y_mid[t] − M·(U−X_mid) + 8·S² = 0      |
| 7  | Addition H        | F_p  | H[t] − Tx·Z_mid² + X_mid = 0           |
| 8  | Addition R_a      | F_p  | R_a[t] − Ty·Z_mid³ + Y_mid = 0         |
| 9  | Result Z          | F_p  | Z[t+1] − selector(b₁,b₂,Z_mid,H) = 0   |
| 10 | Result X          | F_p  | X[t+1] − add_formula_x = 0              |
| 11 | Result Y          | F_p  | Y[t+1] − add_formula_y = 0              |

Full formulas: `ecdsa-uair/src/constraints.rs`.

### Witness Generation

A valid witness requires:
1. Generate a secp256k1 key pair `(sk, Q)`.
2. Sign a message: compute `(r, s)` per ECDSA.
3. Compute `u₁ = e·s⁻¹ mod n`, `u₂ = r·s⁻¹ mod n`.
4. Run the 256-step Shamir double-and-add loop in Jacobian coordinates, recording all intermediate values (X, Y, Z, X_mid, Y_mid, Z_mid, S, H, R_a) at each step.
5. Encode each F_p element as a `BinaryPoly<256>` (or the appropriate ring element).

Consider using the `k256` crate (secp256k1 in pure Rust) for reference arithmetic, or implement from scratch using `crypto-bigint` which is already a dependency.

### Testing

- Add a `witness_valid_ecdsa_verification` test that generates a real signature, builds the trace, and verifies the constraints hold.
- The combined benchmark (`snark/benches/e2e_sha256.rs`, `sha256_8x_ecdsa` group) already creates random 33-column traces at poly=2¹⁰ for PCS timing. Once ECDSA constraints are wired, this should use real traces.

---

## Shared Context

### Architecture quick reference

```
UAIR (sha256/ecdsa)          defines constraints
  ↓
IdealCheckProtocol (piop/)   evaluates constraints at random field point
  ↓
CombinedPolyResolver (piop/) sumcheck reduces to evaluation claims
  ↓
BatchedZipPlus PCS (zip-plus/) polynomial commitment: commit, test, evaluate, verify
  ↓
pipeline.rs (snark/)         orchestrates all steps, serializes proof
```

### Key types

| Type | Location | Description |
|------|----------|-------------|
| `PiopField` = `MontyField<4>` | `snark/src/pipeline.rs` | 256-bit Montgomery field for PIOP (random prime per FS transcript) |
| `Zt::Pt` = `i128` | ZipTypes | PCS evaluation point type |
| `BinaryPoly<D>` | `poly/src/univariate/binary*.rs` | F₂[X] truncated at degree D−1 |
| `DensePolynomial<R, D>` | `poly/src/univariate/dense.rs` | R[X] truncated at degree D−1 |
| `DynamicPolynomialF<F>` | `poly/src/univariate/dynamic/over_field.rs` | F[X] with variable degree |
| `DegreeOneIdeal<R>` | `poly/src/univariate/ideal.rs` | Ideal `(X − r)` in `R[X]` |
| `CyclotomicIdeal` | `sha256-uair/src/lib.rs` | Ideal `(X³² − 1)` |
| `TrivialIdeal` | `snark/src/pipeline.rs` | Always-pass ideal (for F₂ constraints) |
| `IdealOrZero<I>` | `uair/src/ideal_collector.rs` | Either `Ideal(I)` or `Zero` |

### Running tests

```bash
# All tests
cargo test -p zinc-snark -p zinc-sha256-uair -p zinc-ecdsa-uair

# Full pipeline only
cargo test --test full_pipeline -p zinc-snark -- --nocapture

# Benchmarks
cargo bench --bench e2e_sha256 -p zinc-snark -- "Full Pipeline"
```

### Important gotcha

The IC verifier's ideal check over F_p does NOT work for F₂[X] constraints (XOR = addition in F₂, but 1+1=2≠0 in F_p). That's why `verify()` uses `TrivialIdeal` for the F₂[X] UAIR. But Q[X] constraints with `DegreeOneIdeal::new(2)` DO lift correctly to F_p — the `(X−2)` ideal check evaluates `p(2)` and this is meaningful in any field of characteristic > 2.

### PCS evaluation check

The PCS verify evaluation check is **commented out** in `zip-plus/src/batched_pcs/phase_verify.rs` (lines 79–92). The parameter `_point_f` has an underscore prefix (unused). This means PCS verify currently only checks proximity (FRI), not that evaluations match CPR claims. Re-enabling this is an optional hardening step.
