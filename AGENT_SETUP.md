# Agent Setup Instructions

Two agents are needed to complete the Zinc+ SNARK implementation. Each tackles a distinct missing piece.

---

## Current State

**Workspace:** `/Users/albertgarretafontelles/dev/zinc-plus-new/`

### What works today

- **SHA-256 UAIR** (`sha256-uair/src/lib.rs`): 6 of 14 constraints implemented over F₂[X]/(X³²−1) — 4 rotation constraints (Σ₀, Σ₁, σ₀, σ₁) + 2 shift decompositions.
- **Full pipeline** (`snark/src/pipeline.rs`): `prove()` and `verify()` for `BinaryPoly<D>`; `prove_generic()` and `verify_generic()` for any ring `R: ProjectToField<PiopField>`; `prove_dual_ring()` and `verify_dual_ring()` for two-ring PIOP.
- **ECDSA UAIR** (`ecdsa-uair/src/lib.rs`): All 11 constraints implemented over both `DensePolynomial<i64, 1>` and `Int<4>`. Unified single-ring `Int<4>` pipeline tested end-to-end.
- **ECDSA Int<4> pipeline test** (`snark/tests/ecdsa_pipeline.rs`): `prove_generic()` → `verify_generic()` round-trip passes with `EcdsaScalarZipTypes` (PCS field = `MontyField<8>`).
- **ProjectToField for Int<LIMBS>** (`piop/src/ideal_check/utils.rs`): Bridges `Int<N>` to the PIOP field for IdealCheck.
- **Benchmarks** (`snark/benches/e2e_sha256.rs`): Full pipeline benchmark.
- **37 tests pass, 0 warnings.**

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

## Agent 2: ECDSA UAIR Constraints ✅ COMPLETED

### What Was Done

The ECDSA trace now uses `Int<4>` (256-bit integer) throughout — for PCS commitment, PIOP constraint checking, and the end-to-end pipeline. This eliminates the prior dual-ring architecture (`BinaryPoly<32>` PCS + `DensePolynomial<i64, 1>` constraints).

### Changes Made

1. **`piop/src/ideal_check/utils.rs`** — Added `ProjectToField<F> for Int<LIMBS>`. Bridges `Int<N>` to the PIOP field (`MontyField<4>`) for the IdealCheck protocol.

2. **`ecdsa-uair/src/lib.rs`** — Added `impl Uair<Int<4>> for EcdsaUair` with all 11 constraints (C1–C11). Uses `Int::<4>::from_ref(&v)` for constants. `type Ideal = ImpossibleIdeal` (all constraints are `assert_zero`).

3. **`ecdsa-uair/src/witness.rs`** — Added `GenerateWitness<Int<4>>` producing a fixed-point witness (X=1, Y=1, Z=0, S=1, Z_mid=0, X_mid=1, Y_mid=1, H=−1, R_a=−1).

4. **`snark/src/pipeline.rs`** — Added `prove_generic<U, R, Zt, Lc, PcsF, CHECK>()` and `verify_generic<...>()`. These accept any ring `R: ProjectToField<PiopField>` and a separate PCS field `PcsF` (e.g. `MontyField<8>` for 512-bit Zt::Fmod).

5. **`snark/tests/ecdsa_pipeline.rs`** — Rewritten to use `Int<4>` single-ring pipeline with `EcdsaScalarZipTypes` (Eval=Int<4>, Cw=Int<5>, CombR=Int<8>, PcsF=MontyField<8>).

---

## Agent 3: Split SHA-256 Trace into BinaryPoly and Int Columns

### Goal

Split the SHA-256 trace into two PCS batches: one with `BinaryPoly<32>` for columns that participate in F₂[X] rotation/shift constraints, and one with `Int<1>` (64-bit integer, sufficient for 32-bit values + carries) for columns that only participate in integer arithmetic (carry propagation, BitPoly checks). This reduces the codeword size from 256 bytes to 8 bytes for 9 out of 20 columns.

### The Problem

Today, all 20 SHA-256 columns are committed together with `Eval = BinaryPoly<32>`, `Cw = DensePolynomial<i64, 32>` (256 bytes/codeword). But many columns never appear in rotation constraints — they only appear in integer addition carry constraints (C10–C11) and BitPoly checks (C7–C9). Committing those as 256-byte polynomials wastes ~2,016 bytes per column opening (9 cols × 256 B – 9 cols × 32 B = ~2,016 B), which at 147 column openings is ~289 KB of waste.

### Column Classification

Constraints by column usage:

| Constraint | Type | Columns Used |
|---|---|---|
| C1: Σ₀ rotation | F₂[X], cyclotomic | 0 (a), 3 (Σ₀) |
| C2: Σ₁ rotation | F₂[X], cyclotomic | 1 (e), 4 (Σ₁) |
| C3: σ₀ rotation+shift | F₂[X], cyclotomic | 2 (W), 8 (σ₀_w), 15 (S0) |
| C4: σ₁ rotation+shift | F₂[X], cyclotomic | 2 (W), 9 (σ₁_w), 16 (S1) |
| C5: σ₀ decomposition | F₂[X], assert_zero | 2 (W), 15 (S0), 17 (R0) |
| C6: σ₁ decomposition | F₂[X], assert_zero | 2 (W), 16 (S1), 18 (R1) |
| C7: Ch_ef BitPoly | Q[X], BitPoly | 6 (ch_ef) |
| C8: Ch_neg_eg BitPoly | Q[X], BitPoly | 7 (ch_neg_eg) |
| C9: Maj BitPoly | Q[X], BitPoly | 5 (Maj) |
| C10: a-update carry | Q[X], (X−2) | **down[0]**, 3 (Σ₀), 4 (Σ₁), 5 (Maj), 6 (ch_ef), 7 (ch_neg_eg), 11 (h), 12 (μ_a), 19 (K_t), 2 (W) |
| C11: e-update carry | Q[X], (X−2) | **down[1]**, 4 (Σ₁), 6 (ch_ef), 7 (ch_neg_eg), 10 (d), 11 (h), 13 (μ_e), 19 (K_t), 2 (W) |

**Columns requiring `BinaryPoly<32>` (used in F₂[X] constraints C1–C6):**
0 (a_hat), 1 (e_hat), 2 (W_hat), 3 (Σ₀_hat), 4 (Σ₁_hat), 8 (σ₀_w_hat), 9 (σ₁_w_hat), 15 (S0), 16 (S1), 17 (R0), 18 (R1)
→ **11 columns** in BinaryPoly PCS batch

**Columns that could be `Int<1>` (only used in Q[X] constraints C7–C11):**
5 (Maj), 6 (ch_ef), 7 (ch_neg_eg), 10 (d), 11 (h), 12 (μ_a), 13 (μ_e), 14 (μ_W), 19 (K_t)
→ **9 columns** in Int PCS batch

### Proof Size Impact

Current (all BinaryPoly, `Cw = DensePolynomial<i64, 32>` = 256 B):
- Column opening data per opening: 20 cols × 256 B = 5,120 B

After split:
- BinaryPoly batch: 11 cols × 256 B = 2,816 B per opening
- Int batch: 9 cols × 8 B = 72 B per opening (Cw = Int<1+> ≈ 8–16 B)
- Total per opening: ~2,888 B vs 5,120 B → **1.77× smaller**

At 147 column openings: saves ~328 KB raw in a single SHA-256 proof.

### Implementation Plan

#### Step 1 (Agent 1): Refactor trace generation for split columns

In `sha256-uair/src/witness.rs`:
- The existing `GenerateWitness<BinaryPoly<32>>` stays as-is but only returns the 11 polynomial columns (indices 0–4, 8–9, 15–18).
- Add `GenerateWitness<Int<1>>` (or a new trait) that returns the 9 integer columns (indices 5–7, 10–14, 19), each as `DenseMultilinearExtension<Int<1>>`.
- The integer value for each cell is `BinaryPoly::to_u64() as i64`, which fits in `Int<1>`.

#### Step 2 (Agent 1): Define `Sha256IntZipTypes`

In `snark/benches/e2e_sha256.rs` (and eventually a shared types module):
```rust
struct Sha256IntZipTypes;
impl ZipTypes for Sha256IntZipTypes {
    const NUM_COLUMN_OPENINGS: usize = 147;
    type Eval = Int<1>;     // 64-bit integer
    type Cw = Int<2>;       // 128-bit codeword (to avoid overflow in NTT)
    type Fmod = Uint<4>;    // 256-bit modulus search
    type PrimeTest = MillerRabin;
    type Chal = i128;
    type Pt = i128;
    type CombR = Int<4>;    // 256-bit combination ring
    type Comb = Self::CombR;
    type EvalDotChal = ScalarProduct;
    type CombDotChal = ScalarProduct;
    type ArrCombRDotChal = MBSInnerProduct;
}
```

#### Step 3 (Agent 1): Two PCS batches in benchmarks

Modify `sha256_8x_ecdsa` (and `sha256_single` / `sha256_full_pipeline`) to:
1. Generate `sha_poly_trace: Vec<DenseMultilinearExtension<BinaryPoly<32>>>` (11 cols).
2. Generate `sha_int_trace: Vec<DenseMultilinearExtension<Int<1>>>` (9 cols).
3. Commit/test/evaluate the BinaryPoly batch with `Sha256ZipTypes`.
4. Commit/test/evaluate the Int batch with `Sha256IntZipTypes` (using `MontyField<4>` as PcsF).
5. Sum both PCS proof sizes.

#### Step 4 (Agent 2): Pipeline support for split-trace proofs

In `snark/src/pipeline.rs`:
- Add `prove_split_sha256()` that runs PIOP on the full 20-column `BinaryPoly<32>` trace (constraints only need the BinaryPoly view), but commits the two trace halves to separate PCS batches.
- The PIOP (IdealCheck + CPR) operates on the `BinaryPoly<32>` trace since `Int<1>` values are trivially embeddable as `BinaryPoly<32>`.
- The PCS evaluation claims from CPR are split: polynomial columns evaluated with `Sha256ZipTypes`, integer columns evaluated with `Sha256IntZipTypes`.
- The verifier reconstructs both PCS proofs and checks them independently.

### Key Insight: PIOP Still Uses One Trace Type

The PIOP (IdealCheck + CPR) doesn't care about PCS types — it operates on `BinaryPoly<32>` for all 20 columns. The split only affects **PCS commitment and proof size**. The integer columns store `BinaryPoly::from(u32)` values where only bit 0 determines the integer value; the polynomial structure is trivial but the PCS doesn't need to know that.

### Interaction Protocol

1. **Agent 1** implements Steps 1–3 (trace split, new ZipTypes, benchmark changes).
2. **Agent 1** runs `cargo test -p zinc-sha256-uair` and `cargo check -p zinc-snark`.
3. **Agent 2** implements Step 4 (pipeline support).
4. **Agent 2** runs `cargo test -p zinc-snark` — all existing tests must pass.
5. **Agent 1** runs `cargo bench --bench e2e_sha256 -p zinc-snark -- "Full Pipeline"` to measure new proof sizes.
6. Both agents confirm the size reduction.

### Expected Result

For 1×SHA-256 (num_vars=7, 128 rows, R4B16 DEPTH=1):
- BinaryPoly batch (11 cols): ~11/20 of current PCS size
- Int batch (9 cols): much smaller due to 8–16 B codewords vs 256 B
- Combined: significantly smaller than current 805 KB (mono-batch)

### Verification

```bash
cargo test -p zinc-sha256-uair -p zinc-snark
cargo bench --bench e2e_sha256 -p zinc-snark --features=zinc-snark/parallel,zinc-snark/simd -- "Full Pipeline"
```

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
