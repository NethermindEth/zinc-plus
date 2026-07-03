# Fold-First SumFold V2 (piop core) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the fold-first SumFold V2 prover/verifier core in the `zinc-piop` crate per `documentation/fold-first-sumfold-doc/prover-algorithm.md`: Gram-form univariate-skip booleanity zerocheck (weighted variant, no quotient), Lagrange-weight fold, folded IdealCheck, assembled row-sumcheck target, with differential and tamper tests.

**Architecture:** One new module `piop/src/neutron_nova/fold_first.rs` holds the skip domain (symmetric integers, Lagrange/barycentric helpers), the Gram pass `(G, h)`, the skip-round proof (evaluation-basis message + γ-weighted zerocheck + α → θ/B★), and the V2 orchestration (`prove_/verify_fold_first_sha_sumfold`). Everything downstream reuses existing machinery: `fold_projected_traces` (via a new `_with_weights` entry), `build_sha_ideal_values_at_point`/`verify_fresh_sha_ideal_polys` for the folded IdealCheck, `folded_row_integrand_values` + `MultiDegreeSumcheck` for the row sumcheck.

**Tech Stack:** Rust, `crypto_primitives::PrimeField`/`FromWithConfig`, `zinc_transcript::Transcript` (`absorb_random_field_slice_owned`, `get_field_challenge`), `zinc_utils::powers`, existing `projection_sha` helpers. Tests: `MontyField<4>` + `test_utils::test_config()` + `Blake3Transcript`.

## Global Constraints

- Do not weaken lints: no new `#[allow]` beyond the patterns already used in this crate (`clippy::arithmetic_side_effects` on math-heavy fns is the established pattern).
- Field bound style: `F: PrimeField` wherever possible; add `InnerTransparentField + DelayedFieldProductSum + Send + Sync + 'static` only where reusing row-sumcheck/integrand helpers requires it; `FromWithConfig<u64>` for domain construction.
- All errors via the existing `ShaProjectionError`/`SumFoldError` enums (extend `SumFoldError` with fold-first variants; `thiserror` style).
- Typecheck after every task: `cargo check -p zinc-piop`; full gate: `cargo test -p zinc-piop fold_first`.
- Protocol-crate wiring (PCS, commitments, e2e bench, domain label `PF_CONCISE_SHA256_FRESH_BATCH_V2`) is **out of scope** — follow-up plan. The piop layer contains all new cryptography.
- Commit after each green task; messages `feat(piop): ...` style matching repo history (no scope prefix used in repo — use plain imperative like existing `Add direct native SHA prepare path`).

---

### Task 1: Skip domain with Lagrange helpers

**Files:**
- Create: `piop/src/neutron_nova/fold_first.rs`
- Modify: `piop/src/neutron_nova/mod.rs` (add `pub mod fold_first;` + re-exports)
- Test: inline `#[cfg(test)] mod tests` in `fold_first.rs`

**Interfaces:**
- Produces: `SkipDomain<F>` with `new(n_instances, cfg) -> Result<Self, SumFoldError>`, `n_instances()`, `node_count()` (= 2N−1), `domain_values(cfg) -> &[F]` (first N nodes), `lagrange_at(x, cfg) -> Vec<F>` (length N, the θ weights), `interpolate(node_values, x, cfg) -> Result<F, SumFoldError>` (length-2N−1 barycentric eval).

- [ ] **Step 1: Write failing tests** (in `fold_first.rs` bottom): `skip_domain_nodes_are_distinct_symmetric_integers`, `lagrange_weights_are_indicators_on_domain_and_sum_to_one`, `interpolate_matches_polynomial_evaluation` (build p(Y) = 3Y² + 2Y + 7 via its node values, compare `interpolate` at a random point against direct evaluation).
- [ ] **Step 2: `cargo test -p zinc-piop fold_first` — expect compile FAIL (missing types).**
- [ ] **Step 3: Implement.** Node j (0 ≤ j < N): integer `j as i64 - (N/2) as i64`; off-domain node s (0 ≤ s < N−1): integer `(N - N/2) as i64 + s as i64` (continues upward, distinct from domain max `N-1-N/2`). Signed embed: `fn signed_field(v: i64, cfg) -> F { if v >= 0 { F::from_with_cfg(v as u64, cfg) } else { F::zero_with_cfg(cfg) - F::from_with_cfg(v.unsigned_abs(), cfg) } }`. Store `nodes: Vec<F>` (len 2N−1, domain first). `lagrange_at(x)`: over the first N nodes with prefix/suffix products of (x−uᵢ) and precomputed denominators dⱼ = Πᵢ≠ⱼ(uⱼ−uᵢ); divide via field `/`. `interpolate`: same formula over all 2N−1 nodes weighting stored values. Validate `n_instances ≥ 2`.
- [ ] **Step 4: Tests pass.**
- [ ] **Step 5: Commit** `Add fold-first skip domain with Lagrange helpers`.

### Task 2: Booleanity value matrix + Gram pass + node values

**Files:**
- Modify: `piop/src/neutron_nova/projection_sha.rs` — make `booleanity_source_value_at_row_with_virtuals`, `sources_need_virtuals`, `reconstruct_virtual_ch_maj_at_row_unchecked` `pub(crate)`.
- Modify: `piop/src/neutron_nova/fold_first.rs`

**Interfaces:**
- Produces: `GramAccumulator<F>` with `accumulate(traces, publics_unused_ok, row_weights, rho_powers, sources, cfg) -> Result<Self, ShaProjectionError>` building `g: Vec<F>` (upper-tri, `tri(j,k)=k*(k+1)/2+j`, j≤k) and `h: Vec<F>`; `skip_node_values(&self, domain, cfg) -> Vec<F>` — domain node j value = `g[tri(j,j)] − h[j]` (exact for any witness), off-domain via quadratic form `Σⱼ Lⱼ²·G[jj] + 2Σ_{j<k} LⱼLₖ·G[jk] − Σⱼ Lⱼ·hⱼ`.
- Consumes: Task 1 `SkipDomain`.

- [ ] **Step 1: Failing tests:** `gram_matches_naive_q_at_all_nodes` (N=4 synthetic boolean traces; naive oracle: for each node x, for each (q,z), Lagrange-extend the per-instance value vector to x via `SkipDomain::lagrange_at`-style weights over domain nodes, compute ω·(v²−v), sum; compare against `skip_node_values`), `honest_bits_give_zero_domain_values`, `tampered_bit_gives_matching_nonzero_domain_value` (set one bit to 2; domain value j* equals directly computed Σω(d²−d) for that instance).
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement.** Per (q,z): ω = rho_powers[idx(q)]·row_weights[z]; compute per-instance values dⱼ once (reuse `booleanity_source_value_at_row_with_virtuals`; hoist `reconstruct_virtual_ch_maj_at_row_unchecked` per (trace,row) when `sources_need_virtuals`); skip when all dⱼ zero; for j≤k with dⱼ,dₖ ≠ 0: `g[tri] += ω·dⱼ·dₖ` with a fast path `if dⱼ==1 && dₖ==1 { g[tri] += ω }`; `h[j] += ω·dⱼ` (fast path add ω). Loop order: row outer, source inner, virtuals per (trace,row) cached in a `Vec<Option<VirtualChMajValues<F>>>` per row.
- [ ] **Step 4: Tests pass.** Also add `gram_diagonal_equals_h_for_honest_bits`.
- [ ] **Step 5: Commit** `Add fold-first Gram accumulator for the booleanity zerocheck`.

### Task 3: Skip-round proof (prove/verify over Transcript)

**Files:** `piop/src/neutron_nova/fold_first.rs`

**Interfaces:**
- Produces:
  - `pub struct FoldFirstSkipRoundProof<F> { pub node_values: Vec<F> }`
  - `pub struct SkipRoundVerdict<F> { pub theta: Vec<F>, pub b_star: F, pub alpha: F }`
  - `prove_skip_round(gram, domain, transcript, cfg) -> (FoldFirstSkipRoundProof<F>, SkipRoundVerdict<F>)` — computes node values, absorbs via `absorb_random_field_slice_owned`, samples α via `get_field_challenge`, θ = `domain.lagrange_at(α)`, B★ = `domain.interpolate(&node_values, α)`.
  - `verify_skip_round(proof, gamma, domain, transcript, cfg) -> Result<SkipRoundVerdict<F>, SumFoldError>` — checks `node_values.len() == 2N−1`, checks `Σⱼ γʲ·node_values[j] == 0` over the N domain slots (`zinc_utils::powers`), absorbs, samples α, returns verdict.
- Consumes: Tasks 1–2.

- [ ] **Step 1: Failing tests:** `skip_round_roundtrip_agrees` (prover/verifier transcripts produce identical θ, B★, α; Σθ = 1), `gamma_check_rejects_tampered_witness` (tamper a bit, honest-prover message → `verify_skip_round` errs with `SkipRoundZeroCheckFailed`), `zeroed_domain_message_passes_gamma_but_shifts_b_star` (adversary zeroes domain slots of the tampered message: verify succeeds but returned `b_star` ≠ the true folded booleanity residue computed directly — asserted via direct Σ_z ω·D'(D'−1) on the θ-folded trace).
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** with new `SumFoldError` variants `SkipRoundMessageLength { got, expected }` and `SkipRoundZeroCheckFailed`.
- [ ] **Step 4: Tests pass.**
- [ ] **Step 5: Commit** `Add fold-first skip-round prover and verifier`.

### Task 4: Weighted fold entry + fold-commutes identity

**Files:**
- Modify: `piop/src/neutron_nova/projection_sha.rs` — extract body of `fold_projected_traces` into `pub fn fold_projected_traces_with_weights(traces, publics, weights: &[F], field_cfg)`, old fn delegates passing `sumfold.eq_instance_weights`.
- Modify: `piop/src/neutron_nova/fold_first.rs` (test only)

**Interfaces:**
- Produces: `fold_projected_traces_with_weights<F, Trace, Public>(&[Trace], &[Public], &[F], &F::Config) -> Result<(ProjectionFoldWitness<F>, ProjectedPublic<F>), ShaProjectionError>` — same bounds as existing (`F: ShaBinaryFoldField`).

- [ ] **Step 1: Failing test** `b_star_equals_folded_booleanity_row_sum` — the completeness identity: run Tasks 2–3 on N=4 synthetic traces, θ-fold via new entry, then compute `Σ_z row_weights[z] · Σ_q ρ^idx(q) · D'(D'−1)` on the folded trace via `booleanity_source_value_at_row_with_virtuals` and assert equality with `b_star`.
- [ ] **Step 2: Run — FAIL (missing fn).**
- [ ] **Step 3: Refactor + implement.** Weight-count validation moves into the `_with_weights` fn.
- [ ] **Step 4: Tests pass (including all existing `projection_sha` tests: `cargo test -p zinc-piop`).**
- [ ] **Step 5: Commit** `Expose weighted trace folding for fold-first SumFold`.

### Task 5: Folded IdealCheck helper

**Files:** `piop/src/neutron_nova/fold_first.rs`

**Interfaces:**
- Produces: `sha_nonzero_target_at(ideal_polys: &[DynamicPolynomialF<F>; NUM_NONZERO_SHA_FAMILIES], a, lambda, cfg) -> Result<F, ShaProjectionError>` — Σ_{f∈F≠0} λᶠ·E'_f(a), computed via a single-instance `FreshIdealEvaluationCache` + existing `evaluate_fresh_sha_targets` (or direct poly evaluation if cache fields are private — resolve at implementation; add the helper regardless so verifier code never touches cache internals).
- Consumes: existing `build_sha_ideal_values_at_point`, `verify_fresh_sha_ideal_polys`, `validate_fresh_sha_ideal_polys_canonical`.

- [ ] **Step 1: Failing tests:** `zero_trace_folded_ideal_polys_pass_membership` (all-zero trace → `build_sha_ideal_values_at_point` → `verify_fresh_sha_ideal_polys` Ok), `synthetic_trace_ideal_polys_fail_membership` (documents that garbage bits are rejected — negative control), `nonzero_target_matches_manual_lambda_sum` (evaluate each poly at a directly, weight by the nonzero-family λ powers, compare).
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement helper.**
- [ ] **Step 4: Tests pass.**
- [ ] **Step 5: Commit** `Add folded IdealCheck target helper for fold-first`.

### Task 6: V2 orchestration + integration/tamper tests

**Files:** `piop/src/neutron_nova/fold_first.rs`, `piop/src/neutron_nova/mod.rs` (exports)

**Interfaces:**
- Produces:
  - `pub struct FoldFirstSumFoldProof<F: PrimeField> { pub skip_round: FoldFirstSkipRoundProof<F>, pub folded_ideal_polys: Vec<DynamicPolynomialF<F>>, pub row_sumcheck: MultiDegreeSumcheckProof<F> }`
  - `pub struct FoldFirstProverArtifacts<F: PrimeField> { pub theta: Vec<F>, pub folded_witness: ProjectionFoldWitness<F>, pub folded_public: ProjectedPublic<F>, pub b_star: F, pub target: F }`
  - `pub struct FoldFirstVerifierClaims<F: PrimeField> { pub theta: Vec<F>, pub b_star: F, pub target: F, pub row_point: Vec<F>, pub expected_row_eval: F }`
  - `prove_fold_first_sha_sumfold(traces, publics, sources, transcript, cfg) -> Result<(FoldFirstSumFoldProof<F>, FoldFirstProverArtifacts<F>), ShaProjectionError>`
  - `verify_fold_first_sha_sumfold(proof, n_instances, sources_len_bound_unused, transcript, cfg) -> Result<FoldFirstVerifierClaims<F>, ShaProjectionError>`
  - Transcript order (both sides identical): sample r_ic (7×), ρ, γ → skip round (absorb node values, sample α) → fold → absorb each folded ideal poly's coefficient slice → verifier: canonical+membership checks → sample a, λ, ξ → T' := `sha_nonzero_target_at(...)` + ξ·B★ → row sumcheck as `MultiDegreeSumcheck` subprotocol over `build_folded_row_sumcheck_group(folded_row_integrand_values(...))`; verifier checks `proof.row_sumcheck.claimed_sums()[0] == T'` via `verify_folded_row_sumcheck_claim` and returns the endpoint subclaim.
- Consumes: everything above.

- [ ] **Step 1: Failing integration tests:**
  - `fold_first_zero_traces_prove_and_verify` (N=4 all-zero traces + zero publics: full prove/verify Ok; T' = 0; endpoint eval equals folded integrand MLE at row_point — evaluated directly in the test).
  - `fold_first_algebraic_identity_on_synthetic_traces` (synthetic boolean traces; prover-side only — skip verifier membership check by asserting the *identity* `folded_row_integrand_sum == sha_nonzero_target_at + ξ·B★ + Σ_{f∈F₀} λᶠ·(zero-family row-sum)`, with the zero-family term computed via `residual_polys_at_row` evaluated at a with row weights; documents exactly which completeness term honest SHA traces zero out).
  - `fold_first_rejects_tampered_bit` (zero traces, one bit → 1 in a rotation source: verify fails — γ-check or membership or row-claim, assert Err).
  - `fold_first_rejects_wrong_ideal_poly` (perturb one transmitted coefficient: membership or T'-mismatch → Err).
  - `fold_first_rejects_forged_skip_message` (zero out tampered domain slots: row-claim mismatch → Err).
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement orchestration** (≈150 lines; reuse `build_sha_lambda_powers`, `build_eq_x_r_vec`, `powers`, `folded_row_integrand_values_with_row_weights`, `build_folded_row_sumcheck_group`, `MultiDegreeSumcheck::{prove,verify}_as_subprotocol`).
- [ ] **Step 4: All tests pass: `cargo test -p zinc-piop`.**
- [ ] **Step 5: Commit** `Add fold-first SumFold V2 prove/verify orchestration`.

### Task 7: Lint/format gate + exports polish

- [ ] **Step 1:** `cargo fmt --all` and `cargo clippy -p zinc-piop --all-targets` — fix findings without new allows (except established `arithmetic_side_effects` pattern on math fns).
- [ ] **Step 2:** `cargo test -p zinc-piop` green.
- [ ] **Step 3: Commit** `Polish fold-first module exports and lints`.

## Self-Review

- Spec coverage: P0 transpose is subsumed by Task 2's value-matrix walk (bit-packing is a later perf optimization — O-10 — not required for correctness; documented in module docs). P1–P3 → Tasks 1–3; P4 → Task 4; P5 → Task 5; P6/P7 target assembly → Task 6 (scalarization of the folded trace happens inside `fold_projected_traces` via folded `scalarized` table — note: existing fold already folds the per-instance scalarized tables, which for shared `a`… **V2 samples `a` after folding, so folded `scalarized` from pre-`a` tables is WRONG — Task 6 must rebuild `scalarized` on the folded trace via `scalarize_bit_slices(&folded.bit_slices, &a, cfg)` after sampling `a`**; test `fold_first_zero_traces_prove_and_verify` would catch a mismatch only trivially, so `fold_first_algebraic_identity_on_synthetic_traces` asserts with rebuilt scalarized tables). P8 unchanged/out of scope. Weighted-vs-quotient: weighted implemented; quotient not implemented (documented optional).
- Placeholder scan: none.
- Type consistency: `SkipRoundVerdict` consumed by Task 6; `fold_projected_traces_with_weights` bounds `ShaBinaryFoldField` (synthetic tests use `MontyField<4>` which already implements it — used by existing fold tests).
