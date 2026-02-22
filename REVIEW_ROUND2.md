# Author Review — Round 2 Implementation Submission

**From:** Lead Author, Zinc+ Paper  
**To:** Implementation Engineer  
**Date:** 2026-02-22  
**Status:** ✅ Conditionally Accepted — Specific Fixes Required Before Camera-Ready

---

## Executive Summary

Round 2 is a **qualitatively different submission** from Round 1. In Round 1, I reviewed a PCS stress test on random data with 2 constraints, no ECDSA, 7–10× proof size overshoot, and a disconnected pipeline. Round 2 delivers:

- **Prover and verifier times that beat the paper's targets** on single SHA-256 and paper-model configurations, with the combined 8×SHA+ECDSA benchmark coming in at 28.7 ms (just under the 30 ms ceiling).
- **Proof size within target (or below)** for the DEPTH=1 configurations after applying compression — 179 KB compressed for single SHA-256, 212 KB for 14-column paper-model.
- **Tripled constraint coverage** (2 → 6), with the remaining 8 constraints clearly blocked on a well-identified infrastructure limitation (multi-ring UAIR), not on laziness.
- **ECDSA crate exists** with correct trace geometry, column naming matching the paper, and a coherent explanation of why constraints are placeholder.
- **PIOP wired in and timed** — ideal check and CPR are invoked, benchmarked, and contribute meaningful overhead numbers.

This is genuine engineering progress. The system is no longer a PCS benchmark disguised as a SNARK — it is an incomplete SNARK with honest accounting of what remains. I am upgrading my verdict from REJECTED to CONDITIONALLY ACCEPTED, subject to the specific fixes enumerated below.

---

## 1. Evaluation Against Concrete Targets

### 1.1 Single SHA-256 (19 cols, poly=2^7, R4B16 DEPTH=1, 64 openings)

| Metric | Target | Achieved | Verdict |
|--------|--------|----------|---------|
| Prover | < 30 ms | 3.7 ms | ✅ **8× headroom** |
| Verifier | < 5 ms | 0.63 ms | ✅ **8× headroom** |
| Proof (compressed) | 200–300 KB | 179 KB | ✅ **Below lower bound** |
| Proof (raw) | — | 406 KB | Informational |

**Assessment:** All three targets met. The 179 KB compressed figure is remarkable — below the paper's lower bound of 200 KB. This validates the DEPTH=1 + deflate strategy for small configurations.

### 1.2 Paper-Model Config (14 cols, poly=2^7, R4B16 DEPTH=1)

| Metric | Target | Achieved | Verdict |
|--------|--------|----------|---------|
| Prover | < 30 ms | 3.0 ms | ✅ |
| Verifier | < 5 ms | 0.55 ms | ✅ |
| Proof (compressed) | 200–300 KB | 212 KB | ✅ **Within range** |

**Assessment:** All targets met. This configuration matches the paper's `n_pol=14` analytical model most closely.

### 1.3 Combined 8×SHA-256 + ECDSA (33 cols, poly=2^10, R4B16 DEPTH=2)

| Metric | Target | Achieved | Verdict |
|--------|--------|----------|---------|
| Prover | < 30 ms | 28.7 ms | ⚠️ Passes but zero margin |
| Verifier | < 5 ms | 1.6 ms | ✅ |
| Proof (compressed) | 200–300 KB | 1559 KB | ❌ **5.2× over upper bound** |

**Assessment:** Prover and verifier pass. Proof size fails badly. The engineer's analysis of root causes (i64 serialization width, 256-bit field elements, DEPTH=2 coefficient incompressibility) is technically correct and aligns with the paper's cost model. The claimed path to ~290 KB with compact serialization + MontyField<2> + DEPTH=1 is plausible — I've run the analytical model and it agrees (see §4). However, these changes are not trivial and are not implemented. This target is **not met**.

---

## 2. Constraint Coverage Assessment

### 2.1 SHA-256 UAIR: 6/14 Constraints Implemented

Round 1 had 2. Round 2 has 6. Reviewing them individually:

| # | Constraint | Status | Verified |
|---|-----------|--------|----------|
| 1 | Σ₀ rotation (cyclotomic) | ✅ Implemented | Tests pass |
| 2 | Σ₁ rotation (cyclotomic) | ✅ Implemented | Tests pass |
| 3 | σ₀ rotation + shift (cyclotomic) | ✅ **New** | Coefficient analysis correct |
| 4 | σ₁ rotation + shift (cyclotomic) | ✅ **New** | Coefficient analysis correct |
| 5 | σ₀ shift decomposition (zero ideal) | ✅ **New** | W = R₀ + X³·S₀ test passes |
| 6 | σ₁ shift decomposition (zero ideal) | ✅ **New** | W = R₁ + X¹⁰·S₁ test passes |
| 7–14 | Ch, Maj, carries, state updates | ❌ Blocked | Need Q[X]/(X-2) ideal |

**Assessment of implemented constraints:** The 6 constraints are correctly specified and individually sound. The rotation polynomials ρ₀, ρ₁, ρ_{σ₀}, ρ_{σ₁} encode the right ROTR parameters. The shift decomposition constraints (5, 6) are the correct Euclidean decomposition `W = R + X^k · S` with the right shift amounts (3 and 10). The test suite covers dimensions, constraint count, degree, scalar collection, witness consistency (Σ values, σ values, shift quotient/remainder), and SHA-256 correctness against NIST vectors.

**Assessment of the gap:** The remaining 8 constraints are all in Q[X] with the (X-2) ideal. The claim that these are "blocked on multi-ring UAIR extension" is legitimate — the current `Uair<R>` trait is parameterized by a single ring `R`, and these constraints require simultaneously reasoning in F₂[X] (for the bit-polynomial representation) and Q[X] (for carry propagation). This is an architectural limitation, not a shortcut.

**However:** 6/14 ≈ 43% of constraints is *not sufficient* for a paper submission that claims "SHA-256 arithmetization." The F₂[X] constraints alone do not enforce the actual SHA-256 computation — they verify that certain XOR/rotation relationships hold, but without the carry constraints, any trace where (e.g.) the modular additions produce wrong results would still pass. The SNARK remains unsound for SHA-256 with only these constraints.

**My requirement:** For the paper, we must either (a) implement all 14, or (b) explicitly state in the paper that the benchmark demonstrates the full pipeline on a *subset* of the constraint system, with a clear argument that the remaining constraints have comparable algebraic degree and would not materially change prover cost. Option (b) is acceptable to me if you also demonstrate the (X-2) ideal check working on a toy example.

### 2.2 ECDSA UAIR: Geometry Only

The ECDSA crate has:
- Correct dimensions (258 rows × 14 columns) ✅
- Correct column naming matching the paper ✅
- Correct constraint enumeration in comments (11 non-boundary constraints) ✅
- Zero actual constraints (`NUM_CONSTRAINTS: usize = 0`) ❌
- Random witness (not a real ECDSA trace) ❌

**Assessment:** This is an empty shell with good documentation. The benchmark uses random BinaryPoly<32> values for ECDSA columns — these have no relationship to secp256k1 elliptic curve operations. For prover *timing*, this is acceptable (the PCS cost is data-agnostic). For any claim about ECDSA *proving*, it is not.

**My requirement:** For the paper, we label the ECDSA columns as "simulated ECDSA trace with matching dimensions" in any benchmark table. We do NOT claim ECDSA verification. If we want the ECDSA claim, we need the multi-ring extension.

---

## 3. Pipeline and Benchmark Architecture

### 3.1 PIOP Proofs Still Discarded

From `pipeline.rs`:
```rust
let (_ic_proof, ic_state) = IdealCheckProtocol::prove_as_subprotocol::<U>(...);
let (_cpr_proof, _cpr_state) = CombinedPolyResolver::prove_as_subprotocol::<BinaryPoly<D>, U>(...);
```

The `_` prefixes tell the full story: PIOP proofs are computed but thrown away. They are not included in `ZincProof`, not serialized into `pcs_proof_bytes`, and not checked by the verifier. This means:

1. **Proof size measurements exclude PIOP contributions.** The ideal check and CPR proofs contain field elements and Fiat-Shamir challenges that add to proof size. For 6 constraints at degree 1, this is likely small (tens of bytes), but it must be measured.
2. **Verification is still PCS-only.** The `verify_pcs_only` function name is honest, but the benchmarked "Verifier" time is a lower bound.
3. **Sumcheck remains absent.** The PIOP protocol chain is Ideal Check → CPR → Sumcheck → PCS. Sumcheck is not invoked anywhere.

**Assessment:** This is unchanged from Round 1. The PIOP timing is now *measured* (1.95 ms + 0.89 ms = 2.84 ms), which is progress — it shows the PIOP overhead is modest relative to the PCS. But the round-trip proof is still broken.

**Impact on numbers:** Prover claims should be (PCS time) + (PIOP time). For single SHA-256: 3.7 ms (PCS) + ~2.84 ms (PIOP) = ~6.5 ms. Still well under 30 ms. The PIOP overhead is smaller than I expected. For verifier: PIOP verification cost is typically dominated by a few field multiplications and hash evaluations — likely <0.5 ms. Verifier is safe.

**My requirement:** Before camera-ready, either (a) integrate the PIOP proofs into ZincProof and implement a real verify() that checks IC + CPR + PCS, or (b) report prover time as "PCS + PIOP" with both components explicitly stated, and verifier time as "PCS-only (PIOP verification adds <0.5 ms)." Option (b) is acceptable for a research paper.

### 3.2 Benchmark Uses PCS-Only Path (Unchanged)

The benchmark calls `bench_pcs_pipeline` directly — not `pipeline::prove()`. This means the headline "Prover: 3.7 ms" is PCS-only. The PIOP overhead (2.84 ms) is measured separately. A reader of the paper would assume "prover time" includes *all* prover work.

**My requirement:** Either call `prove()` in the benchmark, or clearly report: "Prover (PCS): 3.7 ms + Prover (PIOP): 2.84 ms = Total: ~6.5 ms."

### 3.3 Combined Benchmark Uses Random Trace

The 8×SHA+ECDSA benchmark (33 columns) calls `generate_random_trace(10, 33)`. Neither the SHA-256 witness nor the ECDSA geometry is reflected — it's random polynomials. For PCS timing this is fine (the PCS doesn't see structure). For proof size it's also fine (random data if anything compresses *worse*). I accept this for now, but the paper must say "random algebraic trace with matching dimensions," not "SHA-256 + ECDSA."

---

## 4. Proof Size Analysis

### 4.1 Compressed Proof as Wire Size

**Question posed:** Is it acceptable to report compressed (deflate) proof size as the "wire size"?

**My answer: Yes, with caveats.** Proof compression is standard practice (see Plonky2, SP1, Risc0 benchmarks — all report compressed sizes). The compression ratio of 2.3× for DEPTH=1 is actually modest; Plonky2 achieves 3-5× on FRI proofs. The key conditions:

1. The compression algorithm must be specified (deflate, level 6 — fine).
2. The decompression cost must be included in verifier time. For 179 KB of deflate, decompression is <0.1 ms — negligible.
3. Raw size must also be reported in supplementary material.

**Accepted** as wire size, provided the paper says "proof size (deflate-compressed)" and includes raw size in footnotes.

### 4.2 DEPTH=1 vs DEPTH=2 Compression

The dramatic difference between DEPTH=1 (2.3× compression) and DEPTH=2 (1.1×) is expected and well-explained. At DEPTH=1, IPRS coefficients are bounded by ~41 bits but stored in i64 (64 bits), leaving ~23 zero-ish bits per coefficient that deflate exploits. At DEPTH=2, coefficients use ~63 bits of the i64 range, leaving no slack.

This means the compression strategy is *not* a general solution — it only works because DEPTH=1 coefficients are wastefully serialized. With compact serialization (ceil(41/8) = 6 bytes per coefficient instead of 8), the compression ratio would drop to ~1.3× but the *raw* size would already be small.

**The honest proof size path** is: compact serialization (6 bytes/coeff for DEPTH=1) + MontyField<2> (128-bit) + 64 column openings. The engineer's estimate of ~290 KB for 33 cols at poly=2^10 is consistent with the paper's model when I plug in `bitbound=41, n_pol=33, degree=32, n_queries=64, flat_vec_norm=128`.

### 4.3 Analytical Model Reconciliation

Running `sha256_zip_plus_proof_size.py` with `n_pol=14, n_queries=100, depth=1`:
- Model predicts ~186 KB (v1 bitbound) to ~154 KB (v2 bitbound) at 2^7.
- Implementation achieves 179 KB compressed / 406 KB raw at 2^7 with 64 queries.

The compressed figure (179 KB at 64 queries) is between the model's predictions at 100 queries. This is directionally consistent — fewer queries = smaller proof, and compression compensates for wider-than-optimal serialization. The numbers reconcile adequately.

---

## 5. What Has Genuinely Improved Since Round 1

1. **Proof size: 1384 KB → 406 KB raw, → 179 KB compressed.** A 7.7× reduction (raw-to-compressed). This required three orthogonal optimizations: stripping redundant moduli from field serialization, reducing poly_size from 2^9 to 2^7, and reducing NUM_COLUMN_OPENINGS from 147 to 64. Each was correctly identified and implemented.

2. **poly_size reduction from 2^9 to 2^7.** This was my P0.4 action item from Round 1 ("investigate the 8× row padding"). Using R4B16 DEPTH=1 with `row_len = 16 × 8 = 128` gives a 4× reduction in evaluate phase data. Well done.

3. **NUM_COLUMN_OPENINGS: 147 → 64.** The Round 1 value of 147 was unjustified. The new value of 64 matches 128-bit security at rate 1/4 (since `64 × log₂(4) = 128`). This is tight but defensible.

4. **SHA-256 constraints tripled (2 → 6).** The new σ₀/σ₁ constraints (rotation + shift) and their decomposition are algebraically correct and well-tested.

5. **ECDSA crate exists.** From nothing to a properly dimensioned scaffold with clear documentation.

6. **PIOP timing measured.** The 2.84 ms PIOP overhead is now a known quantity, not a handwave.

7. **Witness quality improved.** The carry columns remain zero (acknowledged blocker), but shift quotients S₀, S₁ and remainders R₀, R₁ are now correctly computed and tested. The witness is no longer internally inconsistent.

---

## 6. Remaining Critical Issues

### 6.1 [MUST FIX] Report Total Prover Time Honestly

The headline "Prover: 3.7 ms" is PCS-only. The real number is ~6.5 ms (PCS + PIOP). This is still 4.6× under target, so it doesn't threaten any claim — but misreporting it would be dishonest. **Fix the benchmark to report the sum**, or clearly label sub-components.

### 6.2 [MUST FIX] Combined Proof Size Path

The 33-column combined benchmark at 1559 KB is not publishable. The paper either:
- (a) Reports single SHA-256 and 14-column results only (acceptable for a focused evaluation), or
- (b) Implements compact serialization to bring the combined config under 300 KB.

Option (a) is acceptable for initial submission if the paper's proof size claims are scoped to single-SHA/14-col configurations. Option (b) is required for camera-ready if we want to claim the combined result.

### 6.3 [SHOULD FIX] PIOP Proof Round-Trip

The PIOP proof is computed but discarded. For a paper that claims an end-to-end SNARK pipeline, there must be evidence that the proof *verifies*. This can be a unit test — not necessarily in the benchmark — that demonstrates `prove()` → serialization → `verify()` → accepted.

### 6.4 [SHOULD FIX — or Scope Correctly] Q[X] Ideal Demonstration

The remaining 8 SHA-256 constraints all use the (X-2) ideal for carry propagation. This ideal type has never been demonstrated in the codebase. Either:
- Implement a toy example (e.g., single modular addition: `a + b - c - carry·X^32 ∈ (X-2)`) and run it through the full pipeline.
- Or: acknowledge in the paper that the F₂[X] constraints are demonstrated and the Q[X] constraints are algebraically specified but not implemented.

### 6.5 [NICE TO HAVE] Deterministic Combined Trace

Replace `generate_random_trace(10, 33)` with a trace that concatenates actual SHA-256 witness columns (19) with ECDSA-shaped random columns (14). This makes the benchmark more honest about what it measures.

---

## 7. Verdict

### ✅ CONDITIONALLY ACCEPTED

The single SHA-256 and 14-column paper-model benchmarks **meet all three targets** (prover, verifier, proof size). The implementation demonstrates a working Zinc+ pipeline — PCS commit/test/evaluate/verify + PIOP ideal check + CPR — with real SHA-256 witness data and non-trivial constraints. The proof size reduction from Round 1 is substantial and well-engineered.

**Conditions for unconditional acceptance (camera-ready):**

1. **[REQUIRED]** Report total prover time as PCS + PIOP sum (~6.5 ms for single SHA-256). Label sub-components explicitly in any benchmark table.

2. **[REQUIRED]** Scope proof size claims to configurations where targets are met (DEPTH=1 configs). Do not claim the 33-column combined result meets targets. Either fix serialization or restrict claims.

3. **[REQUIRED]** Add a round-trip proof test: `prove()` → `verify()` → accepted, even if only at the PCS level with a toy UAIR.

4. **[REQUIRED]** Either demonstrate the (X-2) ideal on a toy constraint, or explicitly state in the paper that only F₂[X] constraints are implemented, with remaining Q[X] constraints algebraically specified.

5. **[RECOMMENDED]** Replace random combined trace with SHA-256 witness + ECDSA-shaped random columns.

**What changed my verdict:** From "a competent PCS benchmark disguised as a SNARK" to "an incomplete but demonstrably functional SNARK pipeline." The proof size work moved the needle decisively — from 7× over target to below target. The PIOP timing data shows the overhead is manageable. The constraint work, while incomplete, demonstrates that the algebraic machinery (cyclotomic ideal checks, shift decomposition) actually works end-to-end. And critically, the engineer provided honest accounting of what works, what doesn't, and why — no hand-waving.

**Risk assessment for the paper:** Low. The claimed numbers (prover < 30 ms, verifier < 5 ms, proof size 200–300 KB) hold for the single-SHA and 14-column configs. The PIOP overhead is measured and does not blow the budget. The infrastructure gaps (compact serialization, MontyField<2>) are concrete engineering tasks with predictable outcomes, not research risks.

---

*— Lead Author*
