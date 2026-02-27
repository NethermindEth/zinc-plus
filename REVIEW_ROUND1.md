# Author Review — Round 1 Implementation Attempt

**From:** Lead Author, Zinc+ Paper  
**To:** Implementation Engineer  
**Date:** 2026-02-22  
**Status:** ❌ Not Acceptable — Major Issues  

---

## Executive Summary

The scaffolding is recognizable as an attempt at the system I described, and some of the plumbing (PCS-level pipeline, Criterion benchmarks, SHA-256 witness generation) is structurally sound. However, this is nowhere near the standard I asked for. The UAIR — the one thing that must be correct for this to be a *Zinc+* proof system rather than a PCS stress test — has 2 out of ~53 constraints. The "8×SHA-256+ECDSA" benchmark is a random trace with zero constraints enforced. The proof size is 7–10× the paper's target, and the root cause is deeper than a knob to tune.

I am returning this with mandatory rework. Below is my detailed analysis.

---

## 1. Critical Issues (Blockers)

### 1.1 The SHA-256 UAIR is a Skeleton, Not an Arithmetization

**What was delivered:** 2 constraints out of ~53 — only the Σ₀ and Σ₁ rotation checks.

**What is missing (non-exhaustive):**

| Category | Constraints | Status |
|---|---|---|
| Σ₀, Σ₁ rotation checks (cyclotomic ideal) | 2 | ✅ Implemented |
| σ₀, σ₁ rotation+shift (cyclotomic ideal, with shift remainder) | 2 | ❌ Commented out as "future work" |
| σ₀, σ₁ shift quotient/remainder decomposition | 2 | ❌ Missing |
| Ch decomposition (AND constraints over F₂) | ~4 | ❌ Missing |
| Maj decomposition (AND constraints over F₂) | ~4 | ❌ Missing |
| State-update carry constraints (eval-at-2 ideal) | ~6 | ❌ Missing |
| Message schedule addition constraint (eval-at-2 ideal) | ~3 | ❌ Missing |
| Remaining rotation/XOR decomposition constraints | ~30 | ❌ Missing |

**Severity:** Fatal. Without these constraints, the system proves nothing. A SNARK that accepts every witness is not a SNARK. The benchmark numbers are *meaningless* as evidence for the paper's claims because no soundness property is being enforced.

**What I specifically asked for:** "IMPLEMENT ALL 53 CONSTRAINTS HERE" (DIRECTIVE §2.4). Known-answer tests against NIST test vectors. Constraint counting tests. Ideal distribution tests. None of this was done.

**The witness generation also has placeholders:** Carry columns (`mu_a`, `mu_e`, `mu_W`) are all zeros. Shift quotients (`S0`, `S1`) are zeros. These columns are not just unused — they would cause the missing constraints to *fail* if they were added. The witness generator and the constraint system are inconsistent.

### 1.2 No ECDSA Arithmetization — Not Even Started

The 8×SHA-256+ECDSA benchmark generates a random trace:

```rust
let trace = generate_random_trace(SHA256_8X_ECDSA_NUM_VARS, SHA256_8X_ECDSA_BATCH_SIZE);
```

This measures PCS throughput on random data, not SNARK proving time. I can benchmark `memcpy` too, but I wouldn't put it in a paper. The column count is 19 (same as SHA-256 alone), not the 33 the paper requires for the combined statement. The ECDSA UAIR crate doesn't exist.

### 1.3 The `NUM_COLUMN_OPENINGS = 147` is Hardcoded and Unjustified

In the benchmark's `ZipTypes` implementation:

```rust
const NUM_COLUMN_OPENINGS: usize = 147;
```

This value appears to be copied from the batched Zip+ benchmark without analysis. The paper's proof size script (`sha256_zip_plus_proof_size.py`) compares `n_queries ∈ {100, 142}`, corresponding to ~96-bit security. The value 147 exceeds both.

**Why this matters critically for proof size:** Each column opening in the batched PCS writes `batch_size × num_rows × Cw::NUM_BYTES` bytes of column values plus a Merkle authentication path. With `batch_size = 19`, `DensePolynomial<i64, 32>` (256 bytes per codeword element), and 147 openings:

$$\text{column\_opening\_cost} = 147 \times (19 \times 1 \times 256 + \text{merkle}) \approx 757 \text{ KB}$$

This alone exceeds the paper's entire proof size target of 200–300 KB.

---

## 2. Proof Size Analysis — Why It's 2 MB Instead of 200 KB

The measured proof sizes are:

| Config | Measured | Paper Target | Ratio |
|---|---|---|---|
| Single SHA-256 (2^9, R4B64d1) | 1384 KB | — | — |
| 8×SHA+ECDSA (2^10, R4B16d2) | 2021 KB | 211 KB | 9.6× |
| 8×SHA+ECDSA (2^10, R4B64d1, 2rows) | 2083 KB | 211 KB | 9.9× |

The proof bytes are extracted via:

```rust
let pcs_transcript: zip_plus::pcs_transcript::PcsTranscript = proof.into();
pcs_transcript.stream.into_inner()
```

This captures the PCS test transcript + evaluation phase. The dominant cost components are:

### 2.1 Evaluation Phase Dominates

The batched PCS `evaluate` phase serializes $\text{batch\_size} \times \text{row\_len}$ field elements of `PiopField = MontyField<4>`. Each Montgomery field element is serialized as a `(value, modulus)` pair:

$$\text{eval\_phase} = 19 \times 1024 \times (2 \times 4 \times 8) = 19 \times 1024 \times 64 \approx 1216 \text{ KB}$$

This single phase already **exceeds the paper's target by 4×**. The paper's cost model puts `flat_vec_norm = 133` bits per column for this phase — roughly 17 KB. The implementation is sending the full Montgomery representation (512 bits/element) for every polynomial in the batch.

**Root cause:** The evaluation phase serialization was designed for correctness testing, not for compact proofs. The paper assumes a much tighter encoding: only the *inner* field element values (not the modulus) and only at the necessary bit-width.

### 2.2 Column Openings Are Expensive at batch_size=19

Per column opening: 19 polynomials × 1 row × 256 bytes + ~408 bytes Merkle proof ≈ 5.3 KB per opening. At 147 openings ≈ 757 KB. The paper's model assumes `n_pol × degree × bitbound` bits per row per query, with bitbound ≈ 63 bits at depth 2, which yields a very similar number at i64 granularity. This component is inherently correct but large.

### 2.3 Proximity Phase

1024 elements of `Int<6>` (48 bytes each) = 48 KB. This is small.

### 2.4 Summary

| Component | Approx. Size | % of Total |
|---|---|---|
| Evaluation phase (batch_size × row_len × F_size) | ~1216 KB | 60% |
| Column openings (147 × per-opening) | ~757 KB | 37% |
| Proximity phase | ~48 KB | 2.4% |
| Total | **~2021 KB** | 100% |

**Conclusion:** The proof is dominated by the evaluation phase, which serializes field elements at 8× the necessary bit-width. Even eliminating this overhead entirely, the column openings at 757 KB would still be 2.5× the target. Both must be addressed.

---

## 3. Additional Technical Issues

### 3.1 SHA-256 Padded to 512 Rows (8×) Due to IPRS DEPTH Requirement

The natural trace for single SHA-256 is 64 rows (`num_vars = 6`). But the benchmark uses `num_vars = 9` (512 rows):

> "IPRS codes need DEPTH ≥ 1 (radix-8 NTT), and row_len = BASE_LEN × 8^DEPTH.
> So the minimum practical poly_size with num_rows=1 is 512 (R4B64 DEPTH=1)."

This 8× padding is wasteful. The paper handles 64-row traces at `2^6` with DEPTH=1 and a different base/radix configuration. The constraint `poly_size ≥ 512` for a 64-row trace means the proof is eating the cost of 448 dummy rows. **Investigate whether a smaller IPRS configuration (smaller base, or num_rows > 1) can handle 64-row traces without this blowup.**

### 3.2 The Prover Pipeline Does Not Produce a Complete Proof

Looking at `pipeline.rs`, the `prove()` function runs both the PIOP (ideal check + CPR) and the PCS, but the returned `ZincProof` only contains the PCS bytes:

```rust
pub struct ZincProof {
    pub pcs_proof_bytes: Vec<u8>,
    pub commitment: BatchedZipPlusCommitment,
    pub timing: TimingBreakdown,
}
```

Where are the ideal check proof and CPR proof? They are computed (`_ic_proof`, `_cpr_proof`) but **discarded** (note the `_` prefix). The verifier has nothing to verify for the PIOP. This means:

1. Proof size measurements exclude the PIOP proof components.
2. Round-trip verification is impossible — the verifier can't check ideal-check or CPR claims.
3. The sumcheck phase is entirely absent from both prover and verifier.

### 3.3 Verification Is PCS-Only

The benchmark's "Verifier" benchmark only runs `BatchedZipPlus::verify`. There is no PIOP verification (ideal check verification, CPR verification, sumcheck verification). The measured verifier time of 1.5–2.1 ms is a lower bound; the real verifier will be slower.

### 3.4 Benchmark Does Not Use `prove()` / `verify()` from pipeline.rs

The benchmark file (`e2e_sha256.rs`) directly calls the PCS commit/test/evaluate/verify cycle via `bench_pcs_pipeline`. It does **not** use the `prove()` and `verify()` functions from `pipeline.rs`. This means:
- The PIOP cost is not measured in the "Prover" benchmark.
- The prover times (12–18 ms) are PCS-only, not end-to-end.
- The "PIOP only" micro-benchmark exists separately but is not composed into the main benchmark.

### 3.5 The 8×SHA+ECDSA Config Uses Wrong Column Count

The paper specifies 8×SHA-256+ECDSA as ~1024 rows × **33 columns** (19 SHA + 14 ECDSA). The benchmark uses 19 columns. This under-counts the column openings cost and makes the PCS benchmark optimistic.

---

## 4. What Is Actually On Track

To be fair:

1. **Prover time is in the right ballpark.** 18 ms for PCS-only on random data at 2^10 is encouraging. The paper claims <30 ms end-to-end. Adding the PIOP overhead, real constraints, and 33 columns will inflate this, but sub-30ms remains plausible on M4 with careful optimization.

2. **Verifier time is good.** 1.7 ms for PCS-verify is well under the 5 ms target. Even with PIOP verification added, this should stay under budget.

3. **The pipeline architecture is correct.** The data flow (commit → ideal check → CPR → test → evaluate → verify) matches the paper. The `pipeline.rs` structure is sound; it just needs to actually include all the pieces.

4. **The SHA-256 witness generation is correct.** The 2 implemented constraints (Σ₀, Σ₁) are verified by tests, the intermediate values match NIST test vectors at round 0, and the message schedule is correct. The foundation for the full witness is there.

5. **The BinaryPoly<32> arithmetic and IPRS code infrastructure work.** This was the hard part, and it's done.

---

## 5. Prioritized Action Items

### P0: Fix Proof Size (Blocks Paper Submission)

1. **Implement compact serialization for the evaluation phase.** The `MontyField` `(value, modulus)` encoding wastes 50% on the modulus (which is constant and known to the verifier). Strip it. Further, evaluate whether the inner value can be sent at reduced bit-width (the paper's `flat_vec_norm = 133` bits suggests significant savings).

2. **Reduce `NUM_COLUMN_OPENINGS` to 100.** This matches 96-bit security per the paper's analysis. At 100 instead of 147, column opening cost drops by 32%.

3. **Verify proof size against the paper's analytical model.** Run `sha256_zip_plus_proof_size.py` with the *actual* implementation parameters (`n_pol=19`, `degree=32`, `base_field_size=16`, `depth=2`, `n_queries=100`, `flat_vec_norm=133`) and compare byte-for-byte with the measured output. The model and implementation must reconcile before we publish.

4. **Investigate the 8× row padding.** For single SHA-256 (64 rows), find an IPRS configuration that supports `poly_size = 64` or `poly_size = 128` instead of 512.

### P1: Complete the SHA-256 UAIR (Blocks Correctness Claims)

5. **Implement all ~53 constraints,** following the exact specification in DIRECTIVE §2.2. Start with the `(X-2)` ideal (EvalAt2Ideal) for modular addition and carry constraints — these are the most novel part of the arithmetization.

6. **Implement the missing ideal types:** `EvalAt2Ideal` for `(X-2)` in `Q[X]`, and `ZeroIdeal` for exact-zero constraints over `F₂[X]`.

7. **Fix the witness generator:** Fill in carry columns (`mu_a`, `mu_e`, `mu_W`) and shift quotients (`S0`, `S1`) with correct values.

8. **Known-answer tests:** SHA-256 of `"abc"` → `ba7816bf...`. All 53 constraints satisfied on this trace. Constraint counter = 53.

### P2: Fix the Pipeline (Blocks Verification)

9. **Include PIOP proof in `ZincProof`.** The ideal check proof and CPR proof must be serialized and returned. The sumcheck must be integrated.

10. **Implement real verification.** The verifier must check ideal-check, CPR, sumcheck, AND PCS opening. Currently it only checks PCS.

11. **Make the benchmark use `pipeline::prove()` / `pipeline::verify()`.** The end-to-end benchmark must measure the actual end-to-end pipeline, not a manual PCS-only loop.

### P3: ECDSA (Blocks Paper Claims)

12. **Create `ecdsa-uair/` crate** per DIRECTIVE §3.

13. **Use 33 columns in the combined benchmark,** not 19.

---

## 6. Assessment: On Track or Fundamentally Broken?

**Structurally on track. Implementation incomplete.**

The architecture (UAIR → PIOP → PCS composition) is correct. The prover timing on PCS-only is consistent with the paper's projections. The polynomial infrastructure works.

The proof size problem is **not** fundamental — it's a serialization issue (bloated field element encoding + high query count). Cutting `NUM_COLUMN_OPENINGS` to 100 and implementing compact evaluation-phase serialization should bring proof size to ~300–400 KB range, which is close enough to optimize further.

The real risk is the SHA-256 UAIR. 51 missing constraints represents 95% of the arithmetization work. The `(X-2)` ideal constraints (modular addition) are the intellectually hard part and have never been implemented in this framework. If those don't compose cleanly with the ideal-check PIOP, we have a fundamental problem. **This must be tested immediately.**

**Bottom line:** I see a competent PCS benchmark disguised as a SNARK. Strip away the SHA-256 window dressing and what you have is `BatchedZipPlus::bench(random_data, 19_polynomials)`. That's fine as an intermediate step, but do not confuse it with the deliverable. The deliverable is an end-to-end SNARK that proves SHA-256+ECDSA computations with 200-300 KB proofs in under 30 ms, and the distance from here to there is measured in weeks, not hours.

---

*— Lead Author*
