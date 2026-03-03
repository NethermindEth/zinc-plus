# Proof Size Optimization — Two-Agent Loop

## Problem Statement

The 8×SHA-256 + ECDSA dual-PCS benchmark produces proofs that are ~4× larger
than the paper's 200–300 KB target:

| Batch | Proof Size | Target |
|-------|-----------|--------|
| SHA PCS (20 cols) | 686 KB | — |
| ECDSA PCS (14 cols) | 537 KB | — |
| Combined (deflate) | 718 KB | 200–300 KB |

Root-cause analysis (`scripts/proof_size_analysis.py`) shows two dominant costs:

1. **Eval-phase bloat (768 KB, 63%)** — The batched evaluate phase writes one
   full `row_len`-sized field-element row **per polynomial** to the transcript.
   That's `batch_size × row_len × F_bytes` per batch.
2. **SHA codeword serialization (342 KB, 28%)** — `Cw = DensePolynomial<i64,32>`
   serializes each codeword element as 32 × 8 = 256 bytes, but a
   `BinaryPoly<32>` carries only 32 bits = 4 bytes of information (**64× waste**).

## Optimization Plan (3 Tasks)

### Task A — Batch the eval phase (saves ~750 KB)

Instead of writing `row_len` field elements per polynomial, the prover should:

1. Sample a Fiat-Shamir batching challenge `β` **after** the test-phase transcript.
2. Compute a **single** batched row: $\hat{r} = \sum_{i=0}^{m-1} \beta^i \cdot r_i$
   where $r_i$ is the field-projected, row-combined evaluation row of polynomial $i$.
3. Write only $\hat{r}$ (512 field elements) to the transcript.
4. Send individual evaluations $v_i = \langle r_i, q_1 \rangle$ as scalars.

The verifier checks:
- The batched row encodes correctly under the linear code.
- $\sum_i \beta^i \cdot v_i = \langle \hat{r}, q_1 \rangle$.
- Each opened column's column values, alpha-projected and row-combined per
  polynomial, sum with the $\beta^i$ weights to match the encoded batched row.

**Expected savings:** Eval phase goes from `batch_size × (1 + 512 × F_bytes)` to
`1 + 512 × F_bytes + batch_size × F_bytes` (one combined row + per-poly scalars).

- SHA: 320 KB → 16 KB + 0.6 KB ≈ **17 KB** (saves 303 KB)
- ECDSA: 448 KB → 32 KB + 0.9 KB ≈ **33 KB** (saves 415 KB)

### Task B — Compact codeword serialization for BinaryPoly (saves ~300 KB)

Each `DensePolynomial<i64, 32>` codeword element has coefficients bounded by the
IPRS bitbound (~45 bits at depth=1 with 2048 columns). The `i64` representation
is adequate for arithmetic but wastes bytes in the proof. The NTT lifts each
binary coefficient into an `i64`, producing values in range $[0, 2^{45})$.

These can be serialized as 6 bytes per coefficient (48 bits, rounding up the
45-bit bitbound), giving 32 × 6 = 192 bytes per codeword element instead of 256.

**Alternatively** (simpler): keep `i64` for arithmetic but implement a custom
`Transcribable` that skips leading-zero bytes. Each `i64` coefficient that fits
in 6 bytes writes 6 bytes. This saves 2 bytes per coefficient × 32 coefficients
= 64 bytes per codeword element, reducing SHA column data from 256 → 192 bytes
(25% saving on column openings).

Most aggressively, the codeword coefficients with bitbound ≤ 45 bits could each
be packed into `ceil(45/8) = 6` bytes, giving 192 bytes per Cw element vs 256.
Given that column openings will be ~27 KB after Task A (much smaller), the ROI
here is lower priority than Task A.

### Task C — Narrow the ECDSA field from `MontyField<8>` to `MontyField<4>` (saves ~220 KB eval, optional)

Currently ECDSA uses `CombR = Int<8>` (512-bit) which forces `Fmod = Uint<8>`
and `FScalar = MontyField<8>` (64 bytes/element). If `CombR` can be narrowed to
`Int<6>` (384-bit) — which is feasible since ECDSA has 14 columns × 1 row ×
256-bit evaluations with 128-bit challenges — then `MontyField<4>` (32 bytes)
suffices, halving the ECDSA eval-phase cost.

This is lower priority than Task A but stacks with it.

---

## Two-Agent Loop Setup

### Overview

Use two agents working in alternation:

- **Agent 1 ("Implementer")** — makes code changes, one task at a time.
- **Agent 2 ("Validator")** — runs tests + benchmarks, reports pass/fail + proof sizes.

The loop iterates until all tests pass and proof size meets the target.

### Shared Context (paste into both agents' system prompts)

```
You are working on the Zinc+ SNARK codebase at:
  /Users/albertgarretafontelles/dev/zinc-plus-new

Branch: agentic-approach

Architecture:
- zip-plus/src/batched_pcs/ — Batched PCS implementation (commit, test, evaluate, verify)
- zip-plus/src/pcs/         — Single-polynomial PCS (reference implementation)
- snark/benches/e2e_sha256.rs — Benchmarks (proof size measured here)
- snark/tests/              — Integration tests (round_trip.rs, full_pipeline.rs, etc.)

Key types:
- ZipTypes trait (zip-plus/src/pcs/structs.rs) — defines Eval, Cw, CombR, Fmod, etc.
- BatchedZipPlus (zip-plus/src/batched_pcs/structs.rs) — batched PCS struct
- Proof transcript = PcsTranscript (zip-plus/src/pcs_transcript.rs)

Proof size formula (test phase, from phase_test.rs line 69):
  row_len × CombR::NUM_BYTES
  + NUM_COLUMN_OPENINGS × (batch_size × num_rows × Cw::NUM_BYTES + merkle_proof)

Eval phase writes batch_size × row_len × F_bytes to transcript (the bottleneck).

The goal is to reduce combined proof size from ~1.2 MB to ≤ 300 KB while keeping
all existing tests passing and prover time < 30 ms.
```

---

### Agent 1: Implementer

**System prompt additions:**

```
You are implementing proof-size optimizations for the Zinc+ batched PCS.
You will receive a task (A, B, or C) and must implement it fully.
After each change, describe exactly which files you modified and what tests
the Validator should run.

DO NOT run benchmarks yourself — the Validator handles that.
DO test compilation: `cargo check -p zip-plus && cargo check -p zinc-snark`

CRITICAL CONSTRAINTS:
- The non-batched PCS (zip-plus/src/pcs/) must NOT be modified.
- All existing tests in snark/tests/ must continue to pass.
- The batched_pcs tests in zip-plus/src/batched_pcs/phase_verify.rs::tests must pass.
- Do not change NUM_COLUMN_OPENINGS (64) or the IPRS rate (1/4).
```

**Task sequence to assign (one per turn):**

#### Turn 1 — Task A: Batch the eval phase

```
TASK A: Batch the eval phase to write a single combined row instead of per-polynomial rows.

FILES TO MODIFY:
1. zip-plus/src/batched_pcs/phase_evaluate.rs
   - After computing all per-polynomial projected rows r_i, sample a batching
     challenge β from the Fiat-Shamir transcript.
   - Compute batched_row = Σ β^i × r_i (a single Vec<F> of length row_len).
   - Write ONLY batched_row to the transcript (one call to write_field_elements).
   - Write each per-polynomial evaluation v_i as individual field elements.
   - Return (evals, proof) as before.

2. zip-plus/src/batched_pcs/phase_verify.rs — fn verify_evaluation
   - Read the single batched_row (row_len field elements) from the transcript.
   - Read batch_size individual evaluation scalars.
   - Encode batched_row under the linear code.
   - Verify: Σ β^i × v_i == ⟨batched_row, q_1⟩.
   - For each opened column, compute the expected encoded value from the
     per-polynomial column data (alpha-project, row-combine, sum with β^i weights)
     and check it matches encoded_batched_row[col].

3. zip-plus/src/batched_pcs/phase_test.rs — update estimated_transcript_size
   - The new formula for the eval phase contribution:
     1 + row_len × F_bytes + batch_size × F_bytes  (length prefix + one combined
     row + per-polynomial evaluation scalars)
   - Note: the test-phase transcript size estimate itself doesn't include eval data;
     verify only the test-phase assertion still holds.

4. snark/benches/e2e_sha256.rs
   - No algorithmic changes needed, but verify the proof size printout reflects
     the new smaller proofs.

VERIFICATION:
- `cargo test -p zip-plus` must pass
- `cargo test -p zinc-snark` must pass
- Report which files were changed and the exact diff summary.
```

#### Turn 2 — Task B: Compact codeword serialization (if needed after Task A)

```
TASK B: Reduce BinaryPoly codeword serialization from 256 to 192 bytes per element.

This task's ROI depends on Task A results. If proof size is already ≤ 300 KB
after Task A, skip this task. Otherwise:

FILES TO MODIFY:
1. poly/src/univariate/dense.rs — ConstTranscribable impl for DensePolynomial
   - Currently: NUM_BYTES = R::NUM_BYTES * DEGREE_PLUS_ONE
   - For i64 coefficients with known bitbound, write ceil(bitbound/8) bytes per
     coefficient instead of 8 bytes.
   - This requires knowing the bitbound at compile time. Consider adding a
     const COMPACT_COEFF_BYTES to ZipTypes or to DensePolynomial.

ALTERNATIVE (simpler, recommended):
1. zip-plus/src/pcs_transcript.rs — add a write_compact_cw / read_compact_cw
   method pair that serializes DensePolynomial<i64, D> using a specified number
   of bytes per coefficient.
2. zip-plus/src/batched_pcs/phase_test.rs — use write_compact_cw when writing
   column openings.
3. zip-plus/src/batched_pcs/phase_verify.rs — use read_compact_cw when reading.
4. Update estimated_transcript_size accordingly.

VERIFICATION:
- `cargo test -p zip-plus` must pass
- `cargo test -p zinc-snark` must pass
```

#### Turn 3 — Task C: Narrow ECDSA field (if needed)

```
TASK C: Narrow ECDSA CombR from Int<8> to Int<6> and FScalar from MontyField<8>
to MontyField<4>.

FILES TO MODIFY:
1. snark/benches/e2e_sha256.rs — EcdsaScalarZipTypes
   - Change CombR from Int<{INT_LIMBS * 8}> to Int<{INT_LIMBS * 6}>
   - Change Fmod from Uint<{INT_LIMBS * 8}> to Uint<{INT_LIMBS * 4}>
   - Remove FScalar type alias; reuse F (MontyField<4>) for ECDSA too.
   - Update Comb type accordingly.

2. Verify no overflow: 14 columns × 1 row × 256-bit eval × 128-bit challenge
   = ~384 bits for the inner product, fitting in Int<6> (384-bit).

VERIFICATION:
- `cargo bench -p zinc-snark --bench e2e_sha256 -- "Combined"` must run without
  overflow panics.
- Proof sizes should shrink by ~200 KB on the ECDSA eval phase.
```

---

### Agent 2: Validator

**System prompt additions:**

```
You are the Validator for the Zinc+ proof-size optimization work.
After the Implementer reports changes, you run compilations, tests, and benchmarks.

VALIDATION STEPS (run in order, stop on first failure):

1. Compile check:
   cargo check -p zip-plus -p zinc-snark 2>&1 | tail -20

2. Unit tests:
   cargo test -p zip-plus 2>&1 | tail -30

3. Integration tests:
   cargo test -p zinc-snark 2>&1 | tail -30

4. Benchmark (proof size measurement):
   cargo bench -p zinc-snark --bench e2e_sha256 \
     --features=zinc-snark/parallel,zinc-snark/simd \
     -- "Combined" 2>&1 | grep -E "proof|KB|bytes|Prover|Verifier"

REPORT FORMAT:
- Compile: PASS/FAIL (error excerpt if FAIL)
- Unit tests: PASS/FAIL (N passed, M failed)
- Integration tests: PASS/FAIL (N passed, M failed)
- Proof sizes: SHA=__KB, ECDSA=__KB, Combined=__KB, Compressed=__KB
- Prover time: __ms  Verifier time: __ms
- VERDICT: PASS (all tests green + proof ≤ 300 KB) / FAIL (explain what broke)

If FAIL, describe the errors concisely so the Implementer knows what to fix.
If PASS, confirm the task is complete and instruct the Implementer to proceed
to the next task (or declare victory if all tasks are done).
```

---

### Loop Protocol

```
┌─────────────┐      task + context       ┌─────────────┐
│             │ ─────────────────────────> │             │
│  Operator   │                            │ Implementer │
│  (You)      │ <───────────────────────── │  (Agent 1)  │
│             │   "changed X, Y, Z files"  │             │
└──────┬──────┘                            └─────────────┘
       │
       │  paste Implementer's report
       ▼
┌─────────────┐                            ┌─────────────┐
│             │   "run validation steps"   │             │
│  Operator   │ ─────────────────────────> │  Validator  │
│  (You)      │                            │  (Agent 2)  │
│             │ <───────────────────────── │             │
└──────┬──────┘  "PASS" or "FAIL: ..."     └─────────────┘
       │
       │  if FAIL → paste errors back to Implementer
       │  if PASS → give Implementer the next task
       ▼
     repeat
```

**Step-by-step:**

1. Copy the **Shared Context** block into both agents' conversation preamble.
2. Send **Turn 1 (Task A)** to the Implementer agent.
3. When the Implementer reports completion, send its summary to the Validator.
4. If Validator says FAIL → paste the error report to the Implementer, ask it to fix.
5. If Validator says PASS → check the reported proof size:
   - ≤ 300 KB → **Done.** Commit and skip Tasks B/C.
   - Still > 300 KB → send **Turn 2 (Task B)** to the Implementer.
6. Repeat the Implementer → Validator loop for Task B.
7. If still > 300 KB after Task B, send **Turn 3 (Task C)**.
8. Final commit: `git add -A && git commit -m "proof-size: batch eval phase + compact cw serialization"`

### Success Criteria

| Metric | Target | Current |
|--------|--------|---------|
| Combined proof (compressed) | ≤ 300 KB | 718 KB |
| Combined prover time | < 30 ms | 25.6 ms |
| Combined verifier time | < 5 ms | 3.0 ms |
| `cargo test -p zinc-snark` | all pass | all pass |
| `cargo test -p zip-plus` | all pass | all pass |

### Estimated Impact per Task

| Task | SHA proof | ECDSA proof | Combined (raw) |
|------|----------|-------------|----------------|
| Current | 686 KB | 537 KB | 1,222 KB |
| After A (batch eval) | 383 KB | 121 KB | 504 KB |
| After A + B (compact cw) | 233 KB | 121 KB | 354 KB |
| After A + B + C (narrow F) | 233 KB | 89 KB | 322 KB |

*Note: Deflate compression typically achieves 1.5–1.7× ratio on these proofs,
so 354 KB raw → ~220 KB compressed. Task A alone may suffice.*

---

## File Reference

| File | Role |
|------|------|
| `zip-plus/src/batched_pcs/phase_evaluate.rs` | **Primary target** — eval-phase prover logic |
| `zip-plus/src/batched_pcs/phase_verify.rs` | **Primary target** — eval-phase verifier logic |
| `zip-plus/src/batched_pcs/phase_test.rs` | Test-phase prover (transcript size estimate) |
| `zip-plus/src/batched_pcs/structs.rs` | BatchedZipPlus types and proof structs |
| `zip-plus/src/pcs/structs.rs` | `ZipTypes` trait (DO NOT MODIFY) |
| `zip-plus/src/pcs/phase_evaluate.rs` | Single-poly eval (reference, DO NOT MODIFY) |
| `zip-plus/src/pcs_transcript.rs` | Transcript read/write helpers |
| `zip-plus/src/merkle.rs` | Merkle proof serialization |
| `poly/src/univariate/dense.rs` | `DensePolynomial` ConstTranscribable impl |
| `snark/benches/e2e_sha256.rs` | Benchmark that measures proof size |
| `snark/benches/steps_sha256_8x_folded.rs` | Per-step folded SHA-256 benchmark (full UAIR, 30 cols) |
| `snark/benches/steps_sha256_8x_uc_folded.rs` | Per-step folded SHA-256 benchmark (**underconstrained** UAIR, 26 cols) |
| `sha256-uair/src/underconstrained.rs` | Underconstrained UAIR (removes 4 F₂[X] columns + constraints) |
| `snark/tests/round_trip.rs` | Round-trip proof test |
| `snark/tests/full_pipeline.rs` | Full PIOP+PCS pipeline test |
| `snark/tests/dual_ring_pipeline.rs` | Dual-ring (SHA+ECDSA) pipeline test |
| `scripts/proof_size_analysis.py` | Analytical proof-size calculator |
