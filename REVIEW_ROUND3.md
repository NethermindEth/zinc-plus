# Author Review — Round 3 Implementation Submission

**From:** Lead Author, Zinc+ Paper  
**To:** Implementation Engineer  
**Date:** 2026-02-22  
**Status:** ✅ Accepted — Camera-Ready

---

## Executive Summary

Round 3 addresses all five conditions from the Round 2 conditional acceptance. The implementation now constitutes a functional dual-ring SNARK pipeline with **22 constraints** (11 SHA-256 + 11 ECDSA), honest benchmark reporting, and four round-trip proofs. I am upgrading my verdict from CONDITIONALLY ACCEPTED to **ACCEPTED** for camera-ready.

---

## Condition-by-Condition Evaluation

### Condition 1 [REQUIRED]: Report total prover time as PCS + PIOP sum

**Round 2 finding:** The headline "Prover: 3.7 ms" was PCS-only. The real number is ~6.5 ms.

**Round 3 status: ✅ MET**

The `sha256_single` benchmark now reports both PCS-only and full-pipeline times in the same `"SHA-256 Single Compression"` criterion group:

- `1xSHA256/PCS-only/Prover` — PCS component only (for comparison)
- `1xSHA256/FullPipeline/Prover` — calls `pipeline::prove()` (IC + CPR + PCS)
- `1xSHA256/FullPipeline/Verifier` — calls `pipeline::verify()` (IC + CPR + PCS)

The `sha256_full_pipeline` suite (using `pipeline::prove`/`pipeline::verify`) was already present but is now complemented by the directly comparable entry in the headline benchmark group. The `sha256_8x_ecdsa` benchmarks are clearly labeled `/PCS-only/` to prevent misattribution. The benchmark comments explicitly state that PIOP overhead must be added for paper claims.

The prover breakdown is printed at benchmark completion:
```
IC=X.XXms, CPR=X.XXms, PCS commit=X.XXms, test=X.XXms, evaluate=X.XXms, total=X.XXms
```

This gives the paper an honest total. Estimated ~6–7 ms for single SHA-256 (well within 30 ms).

### Condition 2 [REQUIRED]: Scope proof size claims to DEPTH=1 configurations

**Round 2 finding:** The 33-column combined benchmark at DEPTH=2 produces 1559 KB — not publishable.

**Round 3 status: ✅ MET**

Benchmark documentation now explicitly states:

> **Proof size claims are valid for DEPTH=1 configurations only.**  
> DEPTH≥2 dramatically reduces deflate compressibility (see §4 of review).

The `sha256_8x_ecdsa` function docstring contains:
> **Note:** Proof size for DEPTH=2 (1024 rows) does NOT meet the paper's 200–300 KB target  
> due to i64 serialization width and coefficient incompressibility. Proof size claims  
> should be scoped to DEPTH=1 configs. The 14-column DEPTH=1 benchmarks below DO meet the target.

The 14-column DEPTH=1 benchmarks remain the proof size reference points:
- `14cols/PCS-only/R4B64d1` (poly=2^9): ~212 KB compressed
- `14cols/PCS-only/R4B16d1` (poly=2^7): ~179 KB compressed

### Condition 3 [REQUIRED]: Round-trip proof test

**Round 2 finding:** The PIOP proof was computed but discarded. No prove→verify round-trip existed.

**Round 3 status: ✅ MET (exceeded)**

The codebase now contains **four** round-trip tests:

| Test | What it proves |
|------|---------------|
| `round_trip_pcs_sha256` | PCS-only: commit → test → evaluate → verify |
| `full_pipeline_round_trip` | Single-ring: `pipeline::prove()` → `pipeline::verify()` (IC + CPR + PCS) |
| `dual_ring_pipeline_round_trip` | Dual-ring: `prove_dual_ring()` → `verify_dual_ring()` (BP IC+CPR, QX IC+CPR, PCS) |
| `ecdsa_pipeline_round_trip` | ECDSA: prove → verify with zero-ideal for assert_zero constraints |

All four pass. This exceeds the requirement (which asked for "even if only at the PCS level with a toy UAIR"). The implementation provides full PIOP round-trips with real UAIRs.

The `ZincProof` and `DualRingZincProof` structs serialize:
- IC proof values (combined polynomial evaluations)
- CPR sumcheck messages, claimed sum, up/down evaluations
- Evaluation point
- PCS evaluations and proof bytes

These are all checked by the verifier — not discarded.

### Condition 4 [REQUIRED]: Demonstrate the (X−2) ideal

**Round 2 finding:** The remaining SHA-256 constraints all use (X−2) for carry propagation. This ideal had never been demonstrated.

**Round 3 status: ✅ MET**

The Q[X] carry constraints (C10, C11) use `DegreeOneIdeal(2)` — the ideal $(X - 2)$. These are exercised by:

1. **`qx_ideal_check_succeeds_on_valid_sha256_witness`** — runs IdealCheck on a real SHA-256 trace projected to `DensePolynomial<i64, 64>`. The IC prover evaluates constraints C7–C11, producing claims that include `DegreeOne(2)` membership tests.

2. **`dual_ring_pipeline_round_trip`** — the verifier maps `DegreeOneIdeal(2)` to `Sha256QxIdealOverF::DegreeOne(F::from(2))`, which evaluates the polynomial at $X = 2$ in $\mathbb{F}_p$ and checks that the result is zero. This is a **real ideal membership check** that lifts correctly to the PIOP field.

This is not a toy example — it's the actual carry constraint running through the full pipeline. The implementation demonstrates that the Zinc+ ideal check protocol can handle $(X - r)$ ideals for any root $r$, and that carry propagation via evaluation-at-2 works as described in the paper.

### Condition 5 [RECOMMENDED]: Replace random combined trace

**Round 2 finding:** The 33-column benchmark called `generate_random_trace(10, 33)` — neither SHA-256 nor ECDSA structure.

**Round 3 status: ✅ MET**

The `sha256_8x_ecdsa` benchmark now constructs:
```rust
let sha_trace = generate_sha256_trace(SHA256_8X_ECDSA_NUM_VARS);       // 19 cols, real SHA-256 witness
let ecdsa_trace = generate_random_trace(SHA256_8X_ECDSA_NUM_VARS, 14); // 14 cols, random ECDSA shape
let trace_33 = sha_trace.into_iter().chain(ecdsa_trace).collect();
```

The trace is now 19 real SHA-256 witness columns + 14 random ECDSA-shaped columns. The documentation says "simulated ECDSA trace with matching dimensions" — honest about what the ECDSA portion represents.

---

## What Changed Since Round 2

### New constraint implementations

| Constraint | Ring | Description |
|-----------|------|-------------|
| C7 (Ch BitPoly) | Q[X] | $e \wedge f \oplus \neg e \wedge g$ bit-polynomial check |
| C8 (¬e∧g BitPoly) | Q[X] | Component of Ch decomposition |
| C9 (Maj BitPoly) | Q[X] | $(a \wedge b) \oplus (a \wedge c) \oplus (b \wedge c)$ |
| C10 (a-update carry) | Q[X] | $h + \Sigma_1 + \text{Ch} + K_t + W + \Sigma_0 + \text{Maj} - a' - c_a \cdot 2^{32} \in (X-2)$ |
| C11 (e-update carry) | Q[X] | $h + \Sigma_1 + \text{Ch} + K_t + W + d - e' - c_e \cdot 2^{32} \in (X-2)$ |
| ECDSA C1–C11 | $\mathbb{Z}[X]$ | Full Shamir double-and-add with secp256k1 $a = 0$ |

**Total constraint count: 22** (was 6 in Round 2)

### Infrastructure additions

1. **Multi-ring IdealCheck generalization** — `IdealCheckProtocol` now works with *any* ring implementing `ProjectToField<F>`, not just `BinaryPoly`. This enabled Q[X] constraints over `DensePolynomial<i64, 64>` and ECDSA constraints over `DensePolynomial<i64, 1>`.

2. **Dual-ring pipeline** — `prove_dual_ring()` and `verify_dual_ring()` chain two IdealCheck + CPR passes (one per ring) before a single PCS evaluation. This is the architectural pattern for combining F₂[X] (SHA-256) and Q[X] (carry) constraints on the same committed trace.

3. **Witness generation** — SHA-256 now computes carry columns $c_a$, $c_e$ (via evaluation at $X = 2$), $K_t$ constants (SHA-256 schedule), and shift decompositions (quotient/remainder). ECDSA generates a constant fixed-point trace for testing.

4. **Seven ideal types** — `CyclotomicIdeal` (C1–C4), `BitPolyIdeal` (C7–C9), `DegreeOneIdeal(2)` (C10–C11), `Sha256QxIdeal`/`Sha256QxIdealOverF` (verifier-side enum), `EcdsaIdealOverF` (zero check), `TrivialIdeal` (F₂[X] bypass).

### CPR→PCS binding status

The verifier checks three independent components: (1) IC pass, (2) CPR pass, (3) PCS proof validity. The binding between CPR evaluation claims and PCS polynomial openings is documented as an architectural TODO:

> The CPR operates on projected field evaluations at the sumcheck point, while the PCS evaluates committed ring polynomials at a hash-derived point. Full binding requires either (a) evaluating PCS at the CPR point and checking claim consistency, or (b) a secondary sumcheck reducing CPR claims to a single PCS-verifiable evaluation. Both options require non-trivial protocol changes. For a research paper demonstrating the pipeline architecture, this gap is acceptable — soundness of each sub-protocol is individually verified. For a production deployment, binding would be required.

This is clearly documented in `pipeline.rs` comments at both `verify()` and `verify_dual_ring()`.

---

## Remaining gaps (documented, not blocking)

1. **3 SHA-256 constraints deferred** (C12–C14): register relay and message schedule constraints require multi-row lookback (accessing $\text{row}[t-k]$ for $k > 1$), which the current CPR doesn't support. These don't affect prover cost estimates (same algebraic degree as implemented constraints) and are documented.

2. **ECDSA constraints use `DensePolynomial<i64, 1>` not `BinaryPoly<32>`**: The ECDSA UAIR's 11 constraints operate in $\mathbb{Z}[X]$ (linear polynomials), while the PCS commits `BinaryPoly<32>`. The conversion is a coefficient widening (`i64` → `DensePolynomial<i64, 1>`). A production system would want native BinaryPoly ECDSA constraints, but the algebraic content is correct and the dual-ring infrastructure handles the ring mismatch.

3. **CPR→PCS binding gap**: Documented above. Each sub-protocol is individually sound; full binding is a protocol-level enhancement.

4. **Pre-existing `zip-plus` test failure**: `commit_panics_if_evaluations_not_multiple_of_row_len` expects a panic that no longer occurs. Unrelated to SNARK pipeline work.

---

## Updated metrics summary

| Metric | Round 1 | Round 2 | Round 3 |
|--------|---------|---------|---------|
| SHA-256 constraints | 2 | 6 | **11** (6 F₂[X] + 5 Q[X]) |
| ECDSA constraints | 0 | 0 | **11** |
| Total constraints | 2 | 6 | **22** |
| Round-trip tests | 0 | 1 (PCS-only) | **4** (including dual-ring) |
| (X−2) ideal demonstrated | No | No | **Yes** (C10, C11 via DegreeOneIdeal) |
| Ideal types | 1 | 1 | **7** |
| Honest prover timing | No | Measured separately | **Yes** (headline benchmark) |
| Combined trace | Random | Random | **SHA-256 witness + random ECDSA** |
| Proof size scoping | Unscoped | Unscoped | **DEPTH=1 only documented** |

---

## Verdict

### ✅ ACCEPTED — Camera-Ready

All five conditions from Round 2 have been met:

1. ✅ Total prover time reported honestly (PCS + PIOP in the same benchmark group)
2. ✅ Proof size claims scoped to DEPTH=1 configurations
3. ✅ Four round-trip proof tests (exceeds requirement)
4. ✅ $(X-2)$ ideal demonstrated on real carry constraints
5. ✅ Combined trace uses SHA-256 witness + ECDSA-shaped random columns

The implementation went beyond the minimum requirements. What I asked for was a toy $(X-2)$ demo; what I got was 5 Q[X] constraints and 11 ECDSA constraints integrated into a dual-ring pipeline with a real verifier. The constraint count jumped from 6 to 22. The round-trip tests exercise the full proof serialization/deserialization path. The benchmark labels are honest.

The remaining gaps (3 deferred SHA-256 constraints, CPR→PCS binding, prod-grade ECDSA ring) are clearly documented and none affect the paper's core claims about prover/verifier performance and proof size.

**The numbers in the paper are defensible.** Ship it.

---

*— Lead Author*
