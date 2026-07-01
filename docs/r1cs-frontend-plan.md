# Plan: R1CS variant of UCS — a Spartan-style `ConstraintSystem` frontend

## Status (as of implementation)

**Milestone M1 (stages 1–3) is implemented and green** in
`protocol/src/r1cs_frontend.rs` (+ e2e tests in `protocol/src/lib.rs`). R1CS is
provable/verifiable end-to-end over \(\mathbb{Q}[X]\) with the zero ideal through
the full substrate, reusing it exactly as UAIR does. **Both open risks (f) and
(g) are resolved** (see the Open-risks section). The substrate required **zero
signature changes**; the only seam additions were `Layout::new` and
`ProtocolError::R1cs`.

- **Stage 1 (scaffolding + seam plumbing):** done. `Layout::new`,
  `ProtocolError::R1cs(String)`, and the `r1cs_frontend` module with all types +
  `Transcribable` + the `ConstraintSystem` surface.
- **Stage 2 (two-sumcheck argument, frontend-only):** done. `prove_constraints`
  / `verify_constraints` / `verify_lifted_evals` fully implemented; 7 frontend
  tests (satisfiable ±public input, 4 negatives, serialization round-trip).
- **Stage 3 (full-substrate integration):** done. 4 e2e tests via
  `ZincPlusPiop::<_, R1csFrontend<i64,F>, _, _, _>::prove/verify` (2 satisfiable,
  2 tamper negatives).
- **Stages 4 (M2, ideal membership) and 5 (M3, F_q/F_q[X] multi-prime):** not
  started.

Test gate: `cargo test -p zinc-protocol --lib` (41 tests: 30 UAIR + 11 R1CS),
`cargo check --all-targets`, `cargo clippy` (no new warnings).

**Notable deviations from the original sketch below** (all recorded inline in
the relevant sections): M1 witnesses are **scalars committed as one `int`
column** (not poly-valued — that is M3); the frontend stores **`public_values`**
and `new` takes them (the seam does not pass `io`); a handful of free helper
functions were added (`constraint_axis_vars`, `mz_inner`, `matrix_mle_eval`,
`witness_column_scalars`).

## Context

`zinc-plus`'s `protocol` crate now proves/verifies over a generic `ConstraintSystem`
seam (`protocol/src/constraint_system.rs`), with `UairFrontend` as the only
implementation. The lead researcher wants a **second frontend implementing R1CS**:

> "R1CS equality is not to zero but component-wise membership to an ideal. If you
> recall Spartan, it should be more or less clear how to prove such a variant of
> R1CS with integer matrices, witnesses in Z[X] and ideal membership. Can also be
> over F_q and F_q[X]."

The paper (`zinc-plus-paper/UCS/structural_instantiations.tex:147-166`) already
frames R1CS as a UCS specialization: matrices \(M_1,M_2,M_3 \in R_0^{n\times(m+\ell)}\),
vector \(z=(y,w)\), constraint
\[(M_1 z)\circ(M_2 z)-(M_3 z)\in\mathfrak{n}\quad\text{component-wise}\]
encoded as \(Q_{R1CS}(k,Z)=L_{M_1}(k,Z)\cdot L_{M_2}(k,Z)-L_{M_3}(k,Z)\) with
\(L_M(k,Z)=\sum_j\tilde M(k,j)Z_j\). The intended outcome: a working R1CS frontend
that reuses the substrate (commit / φ_q-projection / lift-and-project / Zip+ PCS)
exactly as UAIR does, proving R1CS via Spartan's two sumchecks, written **generic
over a `Semiring`** so it later covers \(\mathbb{Z}, \mathbb{Z}[X], \mathbb{F}_q, \mathbb{F}_q[X]\).

## Why R1CS needs a new argument (not CPR reuse)

UAIR constraints are *local* (a row touches a few columns + shifted neighbours) →
one combined sumcheck (CPR) suffices. R1CS constraints are *global* —
\((M_1z)(b)=\sum_j M_1(b,j)z(j)\) touches all of \(z\) — so it needs **Spartan's two
sumchecks**: an **outer** one over the constraint hypercube (whose claimed value is
checked \(=0\), later ideal-membership) reducing to \(r_x\); then an **inner** one
reducing \(\{\widetilde{M_iz}(r_x)\}\) to a single \(\tilde z(r_y)\) claim, where
\(r_y = r_0\) is exactly the witness-opening point the substrate lifts and opens.

## Locked decisions (from the user)

- **First milestone:** R1CS over \(\mathbb{Q}[X]\) with the **zero ideal** (classic
  Spartan equality), plugged into the full substrate as UAIR uses it.
- **Non-succinct verifier:** A,B,C are public index data; the verifier evaluates
  \(\tilde A(r_x,r_y),\tilde B,\tilde C\) directly in \(O(\text{nnz})\). No sparse-matrix
  commitment (Spark/computation-commitment deferred).
- **Match ProveKit's shape:** \(z=[1,\text{public},\text{private},\dots]\), matrices
  sparse-CSR, \(z\) committed as an MLE over the **variable** axis.
- **Generic over `Semiring S`** from the start (only ℚ[X]/zero-ideal tested first).
- **Hand-built matrices + unit tests** (no external compiler).

## Axis mapping (the crux, verified against code)

The substrate binds `num_vars` to the **witness/variable axis**: it commits the
witness column-MLEs over `num_vars` vars and opens them at the returned `r_0`
(`prover.rs:277,400-408`; `verifier.rs:432`). Spartan's outer sumcheck runs over the
**constraint axis** (`s_x = log2(#constraints)`), producing `r_x` **internally**; the
inner sumcheck reduces to `r_y` over the variable axis. Therefore:

- `num_vars = log2(#witnesses padded)`, `r_0 = r_y`, `r_0_fq = vec![]`.
- The constraint hypercube (`s_x`, `τ`, `r_x`) never crosses the seam — it lives
  entirely inside `prove_constraints`/`verify_constraints`. `#constraints` and
  `#witnesses` may be different powers of two; that's fine.

## Architecture — new module `protocol/src/r1cs_frontend.rs`

Mirror `uair_frontend.rs`; declare `pub mod r1cs_frontend;` in `protocol/src/lib.rs`.
**No `D`/`FD` const generics, no `BinaryPoly`/booleanity/`alpha'` machinery** — the
witness is a plain `int`-group column (unfolded, no booleanity).

```rust
pub struct R1csInstance<S> {            // public index data
    pub a: SparseMatrix<S>, pub b: SparseMatrix<S>, pub c: SparseMatrix<S>,
    pub num_public_inputs: usize,       // excludes constant-1 at z[0]
}                                       // num_rows = 2^s_x, num_cols = 2^num_vars

pub struct R1csFrontend<S, F: PrimeField> {
    instance: R1csInstance<S>,          // the O(nnz) verifier index (analog of UAIR's signature)
    public_values: Vec<S>,              // the `io` (len = num_public_inputs); see note below
    layout: Layout<F::Integer>,         // 1 int witness col, 0 public cols, 0 primes
    _marker: PhantomData<F>,
}
// AS BUILT: `new(instance, public_values)` (asserts public_values.len() ==
// num_public_inputs). `public_values` is stored because the seam does not pass
// `io` and both prover and verifier need it to build z_pub. Derives Clone/Debug
// (F: PrimeField ⟹ Clone+Debug via Semiring; no manual impls). Both prover and
// verifier build an identical frontend.

pub struct R1csConstraintProof<F: PrimeField> {   // : Transcribable (mirror UairConstraintProof)
    pub outer_sumcheck: MultiDegreeSumcheckProof<F>,  // degree 3, one group
    pub az_rx: F, pub bz_rx: F, pub cz_rx: F,         // prover-sent evals at r_x
    pub inner_sumcheck: MultiDegreeSumcheckProof<F>,  // degree 2, one group
    pub z_ry: F,                                      // prover-sent z̃_full(r_y)
}

pub struct R1csVerifierClaims<F: PrimeField> { z_ry_claimed: F, z_pub_ry: F }
```

**`impl ConstraintSystem for R1csFrontend`** associated types:
`Prime = F::Integer` (= `Zt::Fmod`, satisfies the substrate bound at `prover.rs:209`);
`Field = F`; `ConstraintProof = R1csConstraintProof<F>`;
`VerifierClaims = R1csVerifierClaims<F>`;
`IdealSource = FqIdealSource = Scalar = ()` (zero ideal ⇒ no ideal-check, no ψ_a;
the `project_*` closures on `verify_constraints` are simply never called, and the
`IdealOverF: Ideal + IdealCheck` bound is satisfied vacuously — **no trait method
signature changes needed**). `needs_decoupled_pcs_prime` keeps its default (`false`).

**Layout:** `total = witness = ColumnLayout::new(0,0,1)` (one `int` column),
`public = (0,0,0)`, `primes = vec![]`. Public inputs are bound inside the argument
(below), not via substrate "public columns", so `assemble_all` interleaving stays trivial.

## `prove_constraints` (steps, reusing existing primitives)

1. **Extract** the single `int` witness column MLE from `projected_traces[0]`
   (`piop/src/projections.rs`); assemble `z_full = [1, public, private, pad]`
   (len `2^num_vars`). Using the projected column guarantees the `z̃(r_y)` claim
   equals what the PCS opens.
2. **Outer sumcheck (degree 3).** Squeeze `τ` (len `s_x`) first. Build `Az,Bz,Cz`
   in `O(nnz)` via `SparseMatrix::cells()` (`crypto-primitives/src/matrix.rs`), wrap
   as `DenseMultilinearExtension` over `s_x` (`poly/src/mle/dense.rs:50`);
   `eq_tau = build_eq_x_r_inner(&τ)` (`poly/src/utils.rs`).
   `MultiDegreeSumcheck::prove_as_subprotocol` (`piop/src/sumcheck/multi_degree.rs:322`),
   one group: `poly=[eq_tau,az,bz,cz]`, `comb_fn=|v| v[0]*(v[1]*v[2]-v[3])`,
   claimed sum `0`. `r_x = state.point()` (internal); compute `az_rx,bz_rx,cz_rx`.
3. **Inner sumcheck (degree 2).** Absorb `az_rx,bz_rx,cz_rx`; squeeze `r_A,r_B,r_C`.
   Build `m_row[j] = Σ_i eq_rx[i]·(r_A·A+r_B·B+r_C·C)[i][j]` (`O(nnz)`), over `num_vars`;
   `prove_as_subprotocol` with `poly=[m_row, z_full_mle]`, `comb_fn=|v| v[0]*v[1]`,
   proving `Σ_y m_row(y)·z_full(y) = r_A·az_rx+r_B·bz_rx+r_C·cz_rx`.
   `r_y = state.point()`, `z_ry = z_full_mle.eval(r_y)`.
4. **Return** `(proof, ConstraintEndpoints::new(r_y, vec![]))`.

**AS BUILT (M1 = scalars, not \(\mathbb{Q}[X]\) polys).** For M1 the witness is
**scalars in \(F\)** committed as one `int` column; poly-valued witnesses
(\(\mathbb{Z}[X]/\mathbb{F}_q[X]\)) are M3. Consequences baked into the code:
- The projected `int`-column entries are **degree-0** `DynamicPolynomialF<F>`;
  `witness_column_scalars` extracts each scalar via `evaluate_at_point(&0)`
  (Horner ⇒ constant term). Handles both `RowMajor` and `ColumnMajor` traces.
- The **committed** witness column has the public prefix (indices \(0..=\ell\))
  **zeroed**; `prove_constraints` re-adds \([1, \text{io}]\) to form the full
  \(z\) (so \(z = z_{\text{pub,padded}} + z_{\text{wit,padded}}\), MLE-linear).
- Matrix-entry / public-value projection \(S \to F\) is `F::from_with_cfg(&s,
  cfg)` — impl bound `F: for<'a> FromWithConfig<&'a S>`. Constraint count must be
  a power of two \(\ge 2\) (`constraint_axis_vars` errors otherwise, since the
  outer sumcheck needs \(s_x \ge 1\)); helper `mz_inner` builds \(Mz\).
- Challenges are drawn with `transcript.get_field_challenges::<F>(n, cfg)` and
  the three `az/bz/cz` are bound via `absorb_random_field` — see risk (g).

## `verify_constraints` + `verify_lifted_evals`

- **`verify_constraints`** replays the identical FS order: squeeze `τ`; replay outer
  via `verify_as_subprotocol` (`multi_degree.rs:521`) → `r_x`; check
  `claimed_sum == 0` and `expected == eq(τ,r_x)·(az_rx·bz_rx − cz_rx)`; absorb evals;
  squeeze `r_A,r_B,r_C`; replay inner → `r_y`; check
  `claimed_sum == r_A·az_rx+r_B·bz_rx+r_C·cz_rx` and
  `inner_expected == (r_A·Ã+r_B·B̃+r_C·C̃)(r_x,r_y)·z_ry`, where the matrix MLEs are
  evaluated directly: `M̃(r_x,r_y) = Σ_{(i,j,val)} eq_rx[i]·eq_ry[j]·project(val)`
  (`O(nnz)`, via `build_eq_x_r_vec`, `poly/src/utils.rs`). Returns
  `(ConstraintEndpoints::new(r_y,vec![]), R1csVerifierClaims{ z_ry, z_pub_ry })`.
- **Public-input binding (Spartan-standard):** committed MLE is the private part only;
  `z̃_full(r_y) = z̃_pub(r_y) + z̃_wit(r_y)`. Verifier computes
  `z_pub_ry = Σ_{k∈public} eq_ry[k]·z_full[k]` (`O(#public)`) in `verify_constraints`;
  `z̃_wit(r_y)` arrives from the substrate. Reconciliation
  `z_ry == z_pub_ry + z_wit_ry` happens in **`verify_lifted_evals`** (hence `z_pub_ry`
  rides on `VerifierClaims`), reading the single entry of `per_family_all_lifted[0]`.

## Seam changes (both additive; neither touches UAIR)

1. **REQUIRED — `Layout::new`** in `constraint_system.rs`: `Layout`'s fields are
   private with only `from_signature` (`constraint_system.rs:96-106`). Add
   `pub fn new(total_cols, public_cols, witness_cols, primes) -> Self` so a non-UAIR
   frontend can build its layout. (`TotalColumnLayout::new` etc. are already public.)
2. **RECOMMENDED — `ProtocolError::R1cs(...)`** variant (`lib.rs`) rather than
   overloading UAIR/MP-eval error variants.

## Staged implementation (atomic, multi-session)

The work is split into five **atomic** stages. *Atomic* means: after the stage lands,
the whole workspace builds and the existing UAIR suite (30 tests) plus every test the
stage itself adds pass — so each stage can be implemented, reviewed, and merged in its
own session, and a later session can start from just this doc + the merged code.
**Shared gate for every stage:** `cargo test -p zinc-protocol --lib`,
`cargo check --all-targets`, `cargo clippy` (no new warnings), existing 30 tests green.
Stages 1–3 are milestone **M1** (ℚ[X], zero ideal); stages 4–5 are **M2/M3**.
The frontend is written generic over `Semiring S` from stage 1 even though only the
ℚ[X]/zero-ideal instantiation is exercised until M3.

### Stage 1 — Scaffolding + seam plumbing  *(M1)* — ✅ DONE
Landed as planned. `R1csFrontend` uses **derived** `Clone`/`Debug` (not manual —
`F: PrimeField ⟹ Clone+Debug`); the byte-offset multiply in the `Transcribable`
impl uses the `mul!` macro (crate denies `arithmetic_side_effects`).

The only stage that touches shared, UAIR-adjacent files; kept small and purely additive
so it lands with zero behavioural change.
- **Entry state:** current `main`; substrate already generic over `CS: ConstraintSystem`.
- **Scope:**
  - `constraint_system.rs`: add `pub fn Layout::new(total_cols, public_cols, witness_cols, primes)`.
  - `lib.rs`: add `ProtocolError::R1cs(String)`; declare `pub mod r1cs_frontend;`.
  - New `protocol/src/r1cs_frontend.rs`: define `R1csInstance<S>`, `R1csFrontend<S,F>`
    (+ constructor `new`, manual `Clone`/`Debug`), `R1csConstraintProof<F>`
    (+ `Transcribable`/`GenTranscribable`, mirroring `UairConstraintProof`),
    `R1csVerifierClaims<F>`. Write `impl ConstraintSystem for R1csFrontend`: all
    associated types (`IdealSource=FqIdealSource=Scalar=()`), `layout()`, and the three
    protocol methods **stubbed** to `Err(ProtocolError::R1cs("unimplemented".into()))`.
- **Gate:** shared gate (nothing calls the stubs yet). No new behavioural test.
- **Unblocks:** the type surface + seam glue for the argument.

### Stage 2 — Two-sumcheck argument, frontend-only  *(M1 — the algorithmic heart)* — ✅ DONE
Landed as planned. Risk (g) resolved (see Open-risks). Frontend stores
`public_values` and `new` takes them (the constant `1` at z[0] is always public;
`public_values` are the additional `io`). 7 tests: satisfiable (no-public &
with-public), 4 negatives (non-satisfying, tampered `z_ry`, tampered matrix
cell, tampered lift), serialization round-trip. `comb_fn`s use `.clone() * &…`
(field ops are `Mul<&Self>`, not `Mul<Self>`).

Implements the actual protocol; fully testable without the substrate.
- **Entry state:** stage 1 merged.
- **Scope:** implement `prove_constraints` (outer degree-3 + inner degree-2 sumchecks
  via `MultiDegreeSumcheck::prove_as_subprotocol`), `verify_constraints` (replay both,
  O(nnz) direct `Ã/B̃/C̃(r_x,r_y)` eval, public-prefix `z_pub_ry`), and
  `verify_lifted_evals` (reconcile `z_ry == z_pub_ry + z_wit_ry`). Resolve **risk (g)**
  (τ squeezed before the sumcheck metadata; `r_A,r_B,r_C` exactly between the two
  sumchecks — byte-identical prove/verify).
- **Gate:** shared gate + **frontend-only tests** in `r1cs_frontend.rs` (style of
  `multi_degree.rs` tests): tiny satisfiable instance (`z=[1,x,w]`, `w=x·x`),
  `prove_constraints` then `verify_constraints`+`verify_lifted_evals` on a fresh
  transcript with a **hand-assembled** `per_family_all_lifted`, assert `Ok`; negatives:
  corrupt `z_ry`, corrupt a matrix cell, non-satisfying witness (outer claimed-sum ≠ 0).
  Add a `R1csConstraintProof` serialization roundtrip test (real proof now exists).
- **Unblocks:** the argument is proven correct in isolation; stage 3 only has to confirm
  the substrate feeds it the witness eval it expects.

### Stage 3 — Full-substrate integration  *(M1 — completes the milestone)* — ✅ DONE
Landed as planned; **risk (f) confirmed — no code change needed** (see
Open-risks). 4 e2e tests in `lib.rs` (`do_r1cs_test` harness reusing
`TestZincTypesIprs`/`setup_pp`/`make_iprs`): `test_e2e_r1cs`,
`test_e2e_r1cs_with_public_input`, and tamper negatives on `z_ry` and on
`witness_lifted_evals[0][0]`. The hand-built `UairTrace` has empty
`binary_poly`/`arbitrary_poly` and one `int` witness column; the public
`UairTrace` is entirely empty (R1CS declares 0 substrate public columns). The
three `project_*` closures passed to `verify` are `|_, _| unreachable!()`.

Additive test-wiring; the substrate needs no signature changes.
- **Entry state:** stage 2 merged (frontend proven in isolation).
- **Scope:** an e2e test (style of `lib.rs` e2e): build `R1csInstance` + `R1csFrontend::new`,
  a `UairTrace` whose single `int` witness column is the private-witness vector, run
  `ZincPlusPiop::<Zt, R1csFrontend<S,F>, F, D, FD>::prove`/`::verify`. Exercises commit
  (int-group, unfolded) + lift-and-project + PCS open at `r_y`, and
  `needs_decoupled_pcs_prime=false`. Resolve **risk (f)** here: confirm the
  `DynamicPolynomialF<F>` in `per_family_all_lifted[0]` is the scalar `z̃_wit(r_y)`
  (constant poly, no ψ_a element); if a trivial projecting element must be squeezed for
  FS consistency, add it symmetrically in `verify_lifted_evals` (frontend-internal — no
  seam change). If the stage-2 lifted-eval assumption was off, the fix lives entirely in
  `verify_lifted_evals`.
- **Gate:** shared gate + e2e prove/verify passes + a tamper negative (mutate a proof
  field ⇒ verify rejects).
- **Unblocks:** M1 done — R1CS provable/verifiable end-to-end over ℚ[X] with zero ideal.

### Stage 4 — Ideal membership  *(M2)*
- **Entry state:** M1 (stages 1–3) merged.
- **Scope:** replace the outer sumcheck's `=0` check with an ideal-membership check
  (reuse `IdealCheckProtocol`-style membership at the claimed sum); set `IdealSource`
  to the real ideal type and wire `project_ideal`. Generalizes `=0` per the paper's
  ideal-batching lemma (`crypto/crypto_piop.tex:115-119`).
- **Gate:** shared gate + frontend + e2e tests extended with a non-zero principal ideal
  (satisfying and non-satisfying instances).

### Stage 5 — F_q / F_q[X] multi-prime families  *(M3)*
- **Entry state:** M2 merged.
- **Scope:** populate `primes`, override `needs_decoupled_pcs_prime`, run per-prime
  sumchecks (the substrate's per-family lift path already exists), and exercise
  `S = F_q[X]` / `Z[X]` witnesses (move the witness column to the `arbitrary_poly` group).
- **Gate:** shared gate + e2e over at least one declared prime family.

### Dependency chain
`1 → 2 → 3 (= M1)  →  4 (M2)  →  5 (M3)`. Stages 4 and 5 are optional follow-ons; M1
is a self-contained, shippable deliverable.

## Open risks — both RESOLVED in M1

- **(f) lifted-eval vs scalar `z̃_wit(r_y)` — RESOLVED, no code change.** Confirmed
  against the substrate (`verifier.rs` `finish_verify`): with 0 substrate public
  columns, `recompute_public_lifted` returns empty and `assemble_all` yields
  `per_family_all_lifted[0] = [witness_lifted_evals[0][0]]` — the single `int`
  column's lift, a **degree-0** `DynamicPolynomialF<F>` = the scalar `z̃_wit(r_y)`,
  with **no ψ_a projecting element** (zero ideal). `verify_lifted_evals` extracts
  it via `evaluate_at_point(&0)`; `test_e2e_r1cs_tamper_lifted_eval` exercises
  this exact path end-to-end.
- **(g) Fiat–Shamir squeeze positions — RESOLVED.** `MultiDegreeSumcheck::
  prove/verify_as_subprotocol` absorb their metadata `(num_vars, num_families,
  per-group degrees)` at the very start (`multi_degree.rs:344–390` / `540–581`).
  So `τ` is squeezed **before** the outer call and `r_A,r_B,r_C` **between** the
  two calls (never interleaved with a sumcheck's own rounds). `az/bz/cz(r_x)` are
  bound via `absorb_random_field` before `r_A,r_B,r_C`. Prover and verifier use
  the identical `get_field_challenges` / `absorb_random_field` calls, so the
  transcript is byte-identical (validated by every passing test).

## Critical files

- `protocol/src/constraint_system.rs` — seam trait; `Layout::new` ✅ added.
- `protocol/src/r1cs_frontend.rs` — the frontend + 7 frontend-only tests ✅.
- `protocol/src/lib.rs` — `pub mod r1cs_frontend;` + `ProtocolError::R1cs` ✅;
  R1CS e2e tests live in its `mod tests` (`do_r1cs_test` + 4 tests).
- `piop/src/sumcheck/multi_degree.rs` — the sumcheck primitive for BOTH sumchecks.
- `poly/src/utils.rs` — `build_eq_x_r_inner`/`build_eq_x_r_vec`/`eq_eval`/`mle_eval_with_eq_table`.
- `crypto-primitives/src/matrix.rs` — `SparseMatrix` fixed-density row iteration (`O(nnz)`);
  needs the `Matrix` trait in scope for `.cells()`.

---

# Appendix A — API reference (verified signatures)

Everything below was read from the tree and is exact as of this writing. Line
numbers may drift; the **symbol names** are stable — grep for them.

## A.1 The seam — `protocol/src/constraint_system.rs`

- **Trait `ConstraintSystem: Clone + Debug`** (l.199). Associated types and their
  R1CS choices are in the Architecture section above. The three methods:
  ```rust
  fn prove_constraints(&self, transcript: &mut impl Transcript,
      projected_traces: &[ProjectedTrace<Self::Field>],
      field_cfgs: &[<Self::Field as HasPrimeFieldConfig>::Config],
      num_vars: usize,
  ) -> Result<(Self::ConstraintProof, ConstraintEndpoints<Self::Field>), ProtocolError<Self::Field>>;

  fn verify_constraints<IdealOverF>(&self, transcript: &mut impl Transcript,
      proof: &Self::ConstraintProof,
      field_cfgs: &[...::Config], num_vars: usize,
      project_ideal:    impl Fn(&Self::IdealSource,   &...::Config) -> IdealOverF,
      project_fq_ideal: impl Fn(&Self::FqIdealSource, &...::Config) -> IdealOverF,
      project_scalar:   impl Fn(&Self::Scalar,        &...::Config) -> DynamicPolynomialF<Self::Field>,
  ) -> Result<(ConstraintEndpoints<Self::Field>, Self::VerifierClaims), ProtocolError<Self::Field>>
  where IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<Self::Field>>;

  fn verify_lifted_evals(&self, claims: &Self::VerifierClaims,
      per_family_all_lifted: &[Vec<DynamicPolynomialF<Self::Field>>],
      field_cfgs: &[...::Config],
  ) -> Result<(), ProtocolError<Self::Field>>;
  ```
  For R1CS the three `project_*` closures are ignored and `IdealOverF` is never
  instantiated (bound satisfied vacuously).
- **`Layout<P>`** (l.63): fields `total_cols/public_cols/witness_cols/primes` are
  **private**; accessors `total_cols()/public_cols()/witness_cols()/primes()`;
  builders `from_signature(&UairSignature<P>)` (l.96) and slicers `public_of`/
  `witness_of` (l.114/132). **Stage 1 adds** `pub fn new(total_cols, public_cols,
  witness_cols, primes)`. The three column-layout wrappers come from `zinc_uair`:
  `TotalColumnLayout::new(n_bin, n_arb, n_int)` etc. (all `::new` are public).
- **`ConstraintEndpoints<F>`** (l.159, `#[non_exhaustive]`): `pub r_0: Vec<F>`,
  `pub r_0_fq: Vec<Vec<F>>`; ctor `ConstraintEndpoints::new(r_0, r_0_fq)` (l.171).

## A.2 The sumcheck primitive — `piop/src/sumcheck/multi_degree.rs`

Used for **both** sumchecks. `CombFn<F>` (from `zinc_piop::CombFn`) is
`Box<dyn Fn(&[F]) -> F + Send + Sync>`; the slice holds one value per group MLE, in
the order the MLEs were listed.

```rust
// group: poly MLEs are over F::Inner (store field values via `.inner()`)
MultiDegreeSumcheckGroup::new(degree: usize,
    poly: Vec<DenseMultilinearExtension<F::Inner>>, comb_fn: CombFn<F>) -> Self  // l.86

MultiDegreeSumcheck::<F>::prove_as_subprotocol(
    transcript: &mut impl Transcript,
    families: Vec<(Vec<MultiDegreeSumcheckGroup<F>>, &F::Config)>,
    num_vars: usize, q_star_cfg: &F::Config,
) -> Vec<(MultiDegreeSumcheckProof<F>, Vec<SumcheckProverState<F>>)>              // l.322

MultiDegreeSumcheck::<F>::verify_as_subprotocol(
    transcript: &mut impl Transcript, num_vars: usize,
    proofs: &[(&MultiDegreeSumcheckProof<F>, &F::Config)], q_star_cfg: &F::Config,
) -> Result<Vec<MultiDegreeSubClaims<F>>, SumCheckError<F>>                       // l.521
```

- **Single-family form:** prove with `vec![(vec![group], cfg)]`, `q_star_cfg = cfg`;
  verify with `&[(&proof, cfg)]`, `q_star_cfg = cfg`. Pop the single element.
- **Recover the point on the prover:** `SumcheckProverState.randomness: Vec<F>`
  (`piop/src/sumcheck/prover.rs:53`, public) is the shared point (length `num_vars`)
  after finalize → `r_x = states[0].randomness` (outer), `r_y = states[0].randomness`
  (inner). **On the verifier:** `MultiDegreeSubClaims::point() -> &[F]` and
  `expected_evaluations() -> &[F]` (l.255/259).
- **Claimed sum:** the prover derives it in round 1; the verifier reads it via
  `MultiDegreeSumcheckProof::claimed_sums() -> &[F]` (l.131) and must assert it equals
  the expected value (outer: `0`; inner: `r_A·az_rx+r_B·bz_rx+r_C·cz_rx`).
- **`MultiDegreeSumcheckProof<F>`** derives `Clone/Debug/PartialEq/Eq` and already
  impls `GenTranscribable`/`Transcribable` (l.141/227) — `R1csConstraintProof` delegates
  to these for its two sumcheck fields.
- **FS metadata (risk g):** both `prove/verify_as_subprotocol` absorb
  `(num_vars, num_families, per-family num_groups, per-group degrees)` at the very
  start. So squeeze `τ` **before** the outer call and `r_A,r_B,r_C` **between** the two
  calls — never interleaved with a sumcheck's own rounds.
- **Reference usage:** the `#[cfg(test)]` block at l.637–859 is a near-verbatim
  template (build MLEs from `.inner()`, single/two-family prove+verify, check the
  subclaim against the polynomial identity with `eq_eval`).

## A.3 eq / MLE helpers — `poly/src/utils.rs`, `poly/src/mle/dense.rs`

```rust
build_eq_x_r_vec<F: PrimeField>(r: &[F], cfg) -> Result<Vec<F>, ArithErrors>          // utils l.41
build_eq_x_r_inner<F: InnerTransparentField>(r: &[F], cfg)
    -> Result<DenseMultilinearExtension<F::Inner>, ArithErrors>                        // utils l.82
eq_eval<R: Semiring>(x: &[R], y: &[R], one: R) -> Result<R, ArithErrors>              // utils l.164
mle_eval_with_eq_table<F: InnerTransparentField>(evals: &[F::Inner], eq_table: &[F], cfg) -> F // utils l.191
DenseMultilinearExtension::from_evaluations_vec(num_vars, evals: Vec<T>, zero: T)     // dense l.50
    // or the struct literal `DenseMultilinearExtension { num_vars, evaluations }`
mle.evaluate_with_config(point: &[F], cfg) -> Result<F, _>                            // MLE-with-config trait
```
- Build `eq(τ,·)`/`eq(r_x,·)` as **inner** MLEs (for use as a group `poly`); use
  `build_eq_x_r_vec` when you need an `eq(r,·)` table of `F` values to (a) evaluate
  `Az/Bz/Cz` at `r_x` via `mle_eval_with_eq_table` and (b) compute the matrix MLEs
  `M̃(r_x,r_y) = Σ_{(i,j,val)} eq_rx[i]·eq_ry[j]·project(val)`.
- `eq_eval(τ, r_x, one)` gives the scalar `eq(τ,r_x)` for the outer consistency check.

## A.4 Sparse matrices — `crypto-primitives/src/matrix.rs`

```rust
pub struct SparseMatrix<T> { pub num_rows, pub num_cols, pub density, pub cells: Vec<(usize, T)> } // l.46
```
- **Fixed density:** every row has exactly `density` entries; `cells()` (l.74) yields
  one iterator per row of `(col_idx, &val)` via `chunks(density)`. This is *not*
  ProveKit's variable-density CSR — for hand-built test matrices, pad short rows to
  `density` with a filler `(0, S::zero())` (adds `0·z[0]`, harmless). Derives
  `Clone/Debug/Default/Eq/PartialEq` (good for `R1csInstance` derives).
- Building `Az`: `for (row, cells_in_row) in a.cells().enumerate() { Az[row] =
  Σ (col,val) project(val)·z_full[col] }`, `O(nnz)`.

# Appendix B — reference reading (open these for detail)

Too much to inline; read the file/symbol when implementing the relevant stage.

- **Structural template to mirror (all stages):** `protocol/src/uair_frontend.rs` —
  `UairConstraintProof` + its hand-written `GenTranscribable`/`Transcribable` (the
  pattern for `R1csConstraintProof`), the frontend struct's manual `Clone`/`Debug`
  (bound only on stored types, not `F`), `new`/`new_verifier`, and the
  `impl ConstraintSystem` method bodies.
- **Substrate contract (understand; do NOT modify) — stage 3:**
  `protocol/src/prover.rs` `prove(pp, trace, num_vars, cs)` → `step_constraints`
  (calls `cs.prove_constraints`) → lift via `compute_lifted_evals`;
  `protocol/src/verifier.rs` `finish_verify` (calls `verify_constraints`, then
  `assemble_all` + `recompute_public_lifted` to build `per_family_all_lifted`, then
  `verify_lifted_evals`, then `pcs_verify`).
- **`ProjectedTrace` — stage 2:** `piop/src/projections.rs` — enum
  `RowMajor(Vec<Vec<DynamicPolynomialF<F>>>)` / `ColumnMajor(Vec<DenseMultilinearExtension<DynamicPolynomialF<F>>>)`.
  `prove_constraints` reads `projected_traces[0]` for z's witness-column MLE(s).
- **Test-harness patterns:** `protocol/src/lib.rs` e2e/tamper tests (the concrete `Zt`
  types, `ZincPlusPiop::prove/verify`, `UairTrace` construction) for stage 3;
  `piop/src/sumcheck/multi_degree.rs` tests (`const_monty_params!`, `ConstMontyField`,
  `Blake3Transcript::new()`) for stage 2.
- **Paper (math for M2/M3):** `zinc-plus-paper/UCS/structural_instantiations.tex:147-166`
  (R1CS as UCS); `zinc-plus-paper/crypto/crypto_piop.tex:98-134` (three-step reduction /
  Spartan structure) and `:115-119` (ideal-membership lemma). Expand macros when
  reading: `\Q`→\(\mathbb{Q}\), `\ff`→\(\mathbb{F}\), `\rr`→challenge vector, `\meq`→\(\widetilde{eq}\).
- **ProveKit R1CS shape (reference for the z layout):**
  `provekit/provekit/common/src/r1cs.rs` (the `R1CS` struct: `num_public_inputs`, `a/b/c`),
  `.../sparse_matrix.rs` (their CSR), `.../utils/sumcheck.rs`
  (`calculate_witness_bounds`, `transpose_r1cs_matrices`, `calculate_eq`). Their
  `z = [1, public_inputs, private, ...]` ordering is the shape we mirror.
