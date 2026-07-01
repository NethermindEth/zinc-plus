# Plan: a ProveKit-shaped R1CS `ConstraintSystem` frontend

## Status & pivot notice

**Pivoted (this revision).** The goal is now narrowed to: *make the R1CS shapes
that ProveKit actually defines provable* — plain R1CS over a **fixed prime field**,
zero ideal, scalar witnesses — reusing the Zinc+ substrate. All ideal /
polynomial / multi-prime machinery that ProveKit's R1CS does **not** use is
**postponed** (kept in [§Postponed](#postponed-original-general-ucs-plan) for the
future general-UCS work) and its speculative code scaffolding is to be **dropped**
(see [§What to drop from code](#what-to-drop-from-code)).

**What already exists (the reusable base).** The two-sumcheck Spartan argument
(`protocol/src/r1cs_frontend.rs`) and its substrate wiring are implemented and
green — this was "M1" (R1CS, zero ideal, scalar witnesses) with **integer**
matrix entries projected to the substrate's **randomly-sampled** prime. The pivot
**refactors** that base:

| Aspect | M1 as-built (base) | Pivot target |
|---|---|---|
| Matrix entries | generic `S` (`i64` in tests), projected `S→F` | native field elements, stored as **`F`** |
| Matrix type | crypto-primitives fixed-density `SparseMatrix<S>` | **our own variable-density CSR** (ProveKit shape) |
| Field | substrate **samples** a random prime `q0` | **fixed** known prime field `F` (the R1CS field) |
| Soundness basis | integer-relation via random-prime projection | native large-field R1CS (Zinc+ "add-on", `R = F_q`) |

The **argument itself does not change** — only the entry/field/type surface and
one substrate hook. Both previously-open risks (f)/(g) remain resolved
([§Resolved risks](#resolved-risks-carried-over-from-the-base)).

Test gate for every step: `cargo test -p zinc-protocol --lib`,
`cargo check --all-targets`, `cargo clippy` (no new warnings), UAIR suite green.

## Context

`zinc-plus`'s `protocol` crate proves/verifies over a generic `ConstraintSystem`
seam (`protocol/src/constraint_system.rs`), with `UairFrontend` as the first
implementation. We add a **second frontend implementing R1CS**, shaped after
ProveKit so ProveKit-style instances are directly expressible.

ProveKit's R1CS (`provekit/provekit/common/src/r1cs.rs`,
`.../sparse_matrix.rs`) is **plain field R1CS**: three sparse matrices `A,B,C`
with **field-element** entries, a witness `z` with the constant-`1` at column 0,
`num_public_inputs`, and exact semantics `(Az)∘(Bz) = Cz`. **No ideal, no
polynomials.** (It also carries `num_virtual` solver-only columns and an
`Interner`/delta-encoding — optimizations we do **not** mirror.)

The lead researcher's original, broader vision — *"R1CS with integer matrices,
witnesses in Z[X] and ideal membership … also over F_q and F_q[X]"* — is the
general UCS R1CS and is **postponed** below.

## Target relation & soundness

**Relation.** Given a fixed prime field `F` (large, see below), public matrices
\(A,B,C \in F^{n\times m}\), public inputs \(\text{io}\in F^{\ell}\), and a private
witness, form \(z = [\,1,\ \text{io}_1,\dots,\text{io}_\ell,\ \text{private}\dots\,]\in F^m\)
and prove
\[(Az)\circ(Bz) - (Cz) = 0 \quad\text{(component-wise, over } F).\]
This is the **zero-ideal, scalar-witness** specialization — exactly ProveKit's
R1CS semantics.

**Why proving directly over a fixed `F` is sound.** This is the Zinc+ paper's
**"add-on" / `R = F_q` case** (`crypto/crypto_piop.tex`), *not* the ℚ/ℤ case:

- The random-prime projection (Step 1, `φ_{q0}`) exists only "for the soundness
  of this projection itself in the `R = ℚ` case" (l.60). For `R = F_q` it is the
  canonical embedding, and **"if q is already large enough, ℓ = 1 and this step is
  trivial"** (l.80-81) — no random prime, "no prime projection or well-definedness
  concerns arise" (l.16). The add-on "requir[es] only Steps 2 and 3 … with no
  modifications to existing components" (l.18).
- Soundness then rests on **Schwartz–Zippel over `F`**; the only requirement is
  \(|F| = \Omega(2^\lambda)\) so the Step 2/3 bounds are negligible (l.80, l.88).
  For the **zero ideal** the membership check "reduces to a standard equality
  check" (l.5), and for **scalar** witnesses Step 3's evaluate-at-`a` is trivial —
  so the whole thing collapses to a standard finite-field R1CS PIOP (Spartan) over
  `F`.
- The PCS/IOPP over a fixed field is standard: for `R = F_q`, "an IOPP … over
  F_q[X] can be constructed from any IOPP over F_q … with no modifications"
  (l.161, l.179). No random `q''` is needed.

Fixing `q0` (known before commitment) is safe here because there is **no integer
relation to cheat on** — the relation is native over `F`; the prover commits
`F`-elements; the sumcheck's own `F`-randomness catches non-satisfaction. The
"sample the prime *after* commit" discipline only matters when lifting a
ℤ-relation.

> **Hard requirement — large field.** `F` must be cryptographically large
> (\(|F| = \Omega(2^\lambda)\); ProveKit's BN254 scalar ≈ 254 bits qualifies).
> A *small* prime field would need the paper's `ℓ>1` extension embedding, which
> the scalar single-field path does **not** implement. The frontend should
> require/document a large base field and not silently accept a tiny one.

> **Out of scope (postponed): real BN254-mod-p instances.** An *actual* ProveKit
> proof holds only *mod p*; embedding it into Zinc+'s integer world would make the
> constraint `(Az∘Bz−Cz) ∈ ⟨p⟩` (mod-p membership), which is ideal machinery. We
> prove instances that hold **exactly over `F`**; the ⟨p⟩ bridge is postponed.

## Axis mapping (unchanged, verified against code)

The substrate binds `num_vars` to the **witness/variable axis**: it commits the
witness column MLE over `num_vars` vars and opens it at the returned `r_0`
(`prover.rs`, `verifier.rs`). Spartan's **outer** sumcheck runs over the
**constraint axis** (`s_x = log2(#constraints)`), producing `r_x` **internally**;
the **inner** sumcheck reduces to `r_y` over the variable axis. Hence
`num_vars = log2(#witnesses padded)`, `r_0 = r_y`, `r_0_fq = vec![]`, and the
constraint hypercube (`s_x`, `τ`, `r_x`) never crosses the seam.

## ProveKit R1CS shape we mirror (with our own types)

- **Sparse matrix** — variable-density CSR: `num_rows`, `num_cols`, per-row
  **sorted** `(col, value)` entries, `value ∈ F`. (ProveKit's `new_row_indices`
  + `col_indices` + `values`; we skip its `Interner` and delta-encoded serde.)
  Provide `iter_row`, `iter`, `num_entries`, a builder, and right-multiply.
- **`z` layout** — `z[0] = 1` (constant), then `num_public_inputs` public inputs,
  then private witnesses. `num_witnesses = num_cols` (**includes** `z[0]`),
  `num_constraints = num_rows`.
  - **To confirm during impl:** ProveKit's exact public-input *placement* in `z`
    (we adopt `[1, io, private]`; verify against ProveKit's witness solver so a
    ProveKit instance drops in unchanged).
- **`num_virtual`** — solver-only, uncommitted, not in matrices: **ignored**.

## Architecture (revised types) — `protocol/src/r1cs_frontend.rs`

```rust
// Our own variable-density sparse matrix (module-local; NOT crypto-primitives').
// Generic over the value type as a plain container; instantiated at `F`.
pub struct SparseMatrix<T> {
    pub num_rows: usize,
    pub num_cols: usize,
    row_starts: Vec<usize>,      // len num_rows + 1 (CSR row offsets)
    entries: Vec<(usize, T)>,    // (col, value), sorted by col within each row
}

pub struct R1csInstance<F: PrimeField> {   // the O(nnz) public verifier index
    pub a: SparseMatrix<F>, pub b: SparseMatrix<F>, pub c: SparseMatrix<F>,
    pub num_public_inputs: usize,          // excludes the constant-1 at z[0]
}

pub struct R1csFrontend<F: PrimeField> {
    instance: R1csInstance<F>,
    public_values: Vec<F>,                 // io (len = num_public_inputs)
    field_cfg: F::Config,                  // the FIXED R1CS field (supplied to `new`)
    layout: Layout<F::Integer>,            // 1 int witness col, 0 public cols, 0 primes
}

// Unchanged from the base:
pub struct R1csConstraintProof<F: PrimeField> {   // : Transcribable
    pub outer_sumcheck: MultiDegreeSumcheckProof<F>,  // degree 3, one group
    pub az_rx: F, pub bz_rx: F, pub cz_rx: F,
    pub inner_sumcheck: MultiDegreeSumcheckProof<F>,  // degree 2, one group
    pub z_ry: F,
}
pub struct R1csVerifierClaims<F: PrimeField> { z_ry_claimed: F, z_pub_ry: F }
```

`impl ConstraintSystem` associated types are unchanged: `Prime = F::Integer`,
`Field = F`, `ConstraintProof/VerifierClaims` as above, and
`IdealSource = FqIdealSource = Scalar = ()` (zero ideal ⇒ the `project_*` closures
are never called, `IdealOverF` never instantiated — **these stay `()`; the trait
keeps them for UAIR, we simply ignore them**). `needs_decoupled_pcs_prime` keeps
its default (`false`); `layout()` is one `int` witness column, no primes.

**Key simplification vs the base:** with entries already in `F`, the `S→F`
projection (`FromWithConfig<&S>`) **drops out** — `mz_inner` / `matrix_mle_eval`
use the stored `F` values directly. `public_values: Vec<F>`.

## The two-sumcheck argument (unchanged — reuse the base)

`prove_constraints` / `verify_constraints` / `verify_lifted_evals` are exactly as
implemented (the "M1" argument). Summary:

- **Assemble** `z = z_pub + z_wit`: read the committed (private) witness column
  from `projected_traces[0]` (degree-0 int-column lift → `F` scalar via
  `evaluate_at_point(&0)`), then add the public prefix `[1, io]` (indices
  `0..=ℓ` are zeroed in the committed column).
- **Outer (deg 3):** squeeze `τ` (len `s_x`) *before* the sumcheck absorbs its
  metadata; prove `Σ_x eq(τ,x)(Ãz·B̃z − C̃z)(x) = 0`; recover `r_x`; send
  `az_rx/bz_rx/cz_rx`.
- **Inner (deg 2):** absorb the three evals, squeeze `r_A,r_B,r_C`; build
  `m_row[j] = Σ_i eq_rx[i]·(r_A A + r_B B + r_C C)[i][j]`; prove
  `Σ_y m_row(y)·z̃(y) = r_A az_rx + …`; recover `r_y`; send `z̃(r_y)`.
- **verify_constraints** replays identically, checks outer claimed-sum `= 0`,
  `expected = eq(τ,r_x)(az·bz − cz)`, inner claimed-sum `= r_A az + …`,
  `expected = M̃(r_x,r_y)·z_ry` with `Ã/B̃/C̃(r_x,r_y)` in `O(nnz)`; computes
  `z_pub(r_y)`.
- **verify_lifted_evals** reconciles `z̃(r_y) = z_pub(r_y) + z_wit(r_y)` against
  the substrate lift.

The `O(nnz)` matrix-MLE evaluation `M̃(r_x,r_y) = Σ_{(i,j,val)}
eq_rx[i]·eq_ry[j]·val` now reads `val ∈ F` directly (no projection).

## Substrate field-fixing (the one non-additive-behavior change)

The frontend's matrices/witness live in the fixed `F`; the substrate must
therefore **use `F` as its working field** rather than sampling a random `q0`
(otherwise the substrate projects the witness to some `q0 ≠ F` and the field
arithmetic is inconsistent). This is *required* — it is what makes Option 1
(entries as `F`) coherent, and re-projecting `F`-entries to a random `q0` would be
the unsound mod-p path.

Plan:
1. **Add a seam method** to `ConstraintSystem`, e.g.
   `fn working_field(&self) -> Option<F::Config> { None }` (default `None`).
   `R1csFrontend` returns `Some(self.field_cfg.clone())`; `UairFrontend` keeps the
   default `None`.
2. **Substrate uses it** where it currently samples: `prover.rs` `project_common`
   and `verifier.rs` `step1_prime_projection` do
   `let field_cfg = cs.working_field().unwrap_or_else(|| sample_random_prime());`.
   **UAIR path unchanged** (`None` ⇒ samples as before) — additive in effect.
3. **`q''` needs no change:** with `primes = []` (R1CS layout) the substrate
   already aliases `q'' := q0`, so the PCS opens over `F` too.
4. **Witness commitment:** the witness column is committed via the existing `int`
   group with `Int = F::Integer` (canonical field-element reps in `[0,p)`); since
   `q0 = F`'s prime, the substrate's `φ_{q0}` projection is the identity, so
   `projected_traces[0]` carries the `F`-element witness the argument expects.
5. **FS binding (soundness hygiene):** absorb the fixed field modulus (and,
   ideally, a digest of `A,B,C`) into the transcript so the statement is bound —
   the random-prime path got the field "for free" from the transcript; a fixed
   field is public input and should be absorbed. (Confirm placement; the matrices
   being public-but-unabsorbed is a pre-existing property of the non-succinct
   verifier — decide whether to harden here or note it.)

## Staged steps (revised)

Atomic = after the step, the workspace builds and the UAIR suite + this step's
tests pass. The argument (proof types, sumcheck logic) is inherited from the base
and does not change; these steps are the **refactor + substrate hook**.

### Step R1 — Own variable-density sparse matrix *(additive, standalone)*
- New `SparseMatrix<T>` (variable-density CSR) in the r1cs module, **not yet
  wired** into `R1csInstance`. Builder, `iter_row`/`iter`/`num_entries`,
  right-multiply, `Clone/Debug/(Eq/PartialEq)`, unit tests (build, iterate,
  multiply, empty/short rows).
- **Gate:** shared gate; new matrix unit tests pass. Purely additive.

### Step R2 — Refactor frontend to fixed-field `F` + substrate hook *(the pivot core)*
Coupled by necessity (types + field must move together for e2e).
- Drop generic `S`; `R1csInstance<F>` / `R1csFrontend<F>` use `SparseMatrix<F>` and
  store `field_cfg`; `public_values: Vec<F>`. Drop the `FromWithConfig<&S>`
  projection in `mz_inner` / `matrix_mle_eval` (use `F` directly).
- Add `ConstraintSystem::working_field` (default `None`); return `Some(cfg)` from
  `R1csFrontend`; wire the `unwrap_or_else(sample)` in prover/verifier.
- Instantiate the witness `int` group with `Int = F::Integer`; absorb the field
  modulus into the transcript.
- Update **frontend-only** tests (build over a fixed large `F`, `F`-element
  entries) and the **e2e** tests (fixed-field `Zt`, large `Int`).
- **Gate:** shared gate; all frontend-only + e2e R1CS tests pass; UAIR green.
- **Risks to resolve here:** large-field `Zt`/Zip+ instantiation (see open
  questions); ProveKit public-input placement.

### Step R3 — Cleanup & finalize
- Remove any remaining base scaffolding oriented at the postponed generality
  (see [§What to drop](#what-to-drop-from-code)); confirm no dead `S`/projection
  paths remain.
- Update inline docs and this plan's status.
- **Gate:** shared gate; clippy clean.

### Dependency chain
`R1 → R2 → R3`. R1 is a safe additive warm-up; R2 is the substantive change.

## What to drop from code

Purely the base's speculative generality that ProveKit R1CS does not use:
- The generic **`S`** matrix-entry parameter and the `F: for<'a>
  FromWithConfig<&'a S>` projection (entries are `F`).
- Use of crypto-primitives' **fixed-density** `SparseMatrix` (and the `Matrix`
  trait import) → our variable-density type.
- Integer-entry (`i64`) test instances → fixed-field `F` instances.
- (Nothing ideal/poly-specific was ever implemented; `IdealSource` etc. **stay
  `()`** — they are trait-required for UAIR and merely ignored here.)

## Open questions to resolve during implementation

- **(P1) Large-field `Zt` / Zip+ instantiation.** Committing the witness `int`
  column with `Int = F::Integer` (~256-bit) needs `ZipTypes`/linear-code params
  that handle the width. Confirm the existing IPRS/RAA codes + inner-product
  types instantiate at that size, or add the needed `Zt`. (This is the main
  integration risk — the substrate's PCS is integer-oriented; we reuse it with a
  fixed prime = `F`, per the paper's "field IOPP" reading.)
- **(P2) Public-input placement** in `z` vs ProveKit's witness solver
  (`[1, io, private]` assumed).
- **(P3) FS binding** of the fixed field modulus (and possibly a matrix digest) —
  decide the exact absorb point; weigh hardening the (pre-existing) unabsorbed
  public matrices.

## Resolved risks (carried over from the base)

- **(f) lifted-eval vs scalar `z̃_wit(r_y)` — RESOLVED, no code change.** With 0
  substrate public columns, `assemble_all` yields `per_family_all_lifted[0] =
  [witness_lifted_evals[0][0]]` — a **degree-0** `DynamicPolynomialF<F>` = the
  scalar `z̃_wit(r_y)`, no ψ_a element. `verify_lifted_evals` extracts it via
  `evaluate_at_point(&0)`; a tamper test exercises the path e2e.
- **(g) Fiat–Shamir squeeze positions — RESOLVED.** `MultiDegreeSumcheck::
  prove/verify_as_subprotocol` absorb their metadata at the very start, so `τ` is
  squeezed **before** the outer call and `r_A,r_B,r_C` **between** the two calls;
  `az/bz/cz(r_x)` are bound via `absorb_random_field` before `r_A,r_B,r_C`.
  Prover/verifier use identical `get_field_challenges`/`absorb_random_field`
  calls ⇒ byte-identical transcript.

## Critical files

- `protocol/src/r1cs_frontend.rs` — the frontend + our `SparseMatrix<T>` + tests.
- `protocol/src/constraint_system.rs` — seam trait; `Layout::new` (done); add
  `working_field` default method (R2).
- `protocol/src/prover.rs`, `protocol/src/verifier.rs` — the `working_field`
  hook at the prime-sampling step (R2; UAIR path unchanged).
- `protocol/src/lib.rs` — `pub mod r1cs_frontend;`, `ProtocolError::R1cs` (done);
  e2e tests in `mod tests`.
- `piop/src/sumcheck/multi_degree.rs` — the sumcheck primitive for both sumchecks.
- `poly/src/utils.rs` — `build_eq_x_r_inner`/`build_eq_x_r_vec`/`eq_eval`/`mle_eval_with_eq_table`.
- **Reference (do not import):** `provekit/provekit/common/src/{r1cs.rs,sparse_matrix.rs}`.
- **Soundness reference:** `zinc-plus-paper/crypto/crypto_piop.tex` (`R = F_q`
  add-on, l.16-19, 79-88, 161, 179).

---

# Postponed (original general-UCS plan)

Kept verbatim-ish for when we generalize beyond ProveKit's plain-field R1CS. The
code scaffolding for these is intentionally **not** built now.

**Original goal (lead researcher).** *"R1CS equality is not to zero but
component-wise membership to an ideal … R1CS with integer matrices, witnesses in
Z[X] and ideal membership. Can also be over F_q and F_q[X]."* Paper framing
(`UCS/structural_instantiations.tex`): matrices \(M_1,M_2,M_3\in R_0^{n\times(m+\ell)}\),
\(z=(y,w)\), \((M_1z)\circ(M_2z)-(M_3z)\in\mathfrak{n}\) component-wise, with
\(\mathfrak{n}=\{0\}\) recovering plain R1CS.

**Postponed M2 — ideal membership.** Replace the outer sumcheck's `=0` check with
ideal-membership. Nontrivial ideal membership (e.g. ⟨X−c⟩) requires
**polynomial-valued** constraint values, hence poly-valued witnesses/matrices —
with scalar witnesses membership in ⟨X−c⟩ collapses to `=0`. For `DegreeOneIdeal`
(`W=1`, `RotationIdeal<F,1>`), `contains(p) ⇔ p.evaluate_at_point(root)=0`
(`uair/src/ideal/rotation.rs`), so it reduces to evaluating the relation at the
ideal root. **Decision taken before the pivot:** use **poly witnesses (Z[X])**
committed as an `arbitrary_poly` column, reconciling the poly-valued substrate
lift at the root. (Reuse `IdealCheckProtocol`-style membership / the ideal-batching
lemma, `crypto/crypto_piop.tex`.) General `W>1` needs the "membership on the
sumcheck's claimed polynomial" path.

**Postponed M3 — F_q[X] / Z[X] witnesses & multi-prime.** Populate `primes`,
override `needs_decoupled_pcs_prime`, run per-prime sumchecks (substrate per-family
lift path already exists), and move the witness column to the `arbitrary_poly`
group for poly-valued witnesses.

**Postponed — random-prime projection for ℚ/ℤ relations.** The base's original
approach (integer entries projected to a *randomly sampled* prime) is the sound
path for **integer** R1CS/UCS. The pivot uses a fixed field instead; the
random-prime layer returns for the general ℚ/ℤ case.

**Postponed — real ProveKit-over-BN254 (mod-p) instances.** Proving instances
that hold only *mod p* means the ℤ-relation `(Az∘Bz−Cz) ∈ ⟨p⟩` — the ⟨p⟩ ideal
bridge (ideal machinery).

---

# Appendix A — API reference (verified signatures)

Read from the tree; symbol names are stable (grep). Line numbers may drift.

## A.1 The seam — `protocol/src/constraint_system.rs`

- **Trait `ConstraintSystem: Clone + Debug`.** Three methods:
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
  For R1CS the three `project_*` closures are ignored, `IdealOverF` never
  instantiated. **R2 adds** `fn working_field(&self) -> Option<F::Config> { None }`.
- **`Layout<P>`:** private fields; accessors `total_cols/public_cols/witness_cols/
  primes`; `from_signature`; **`new(total_cols, public_cols, witness_cols, primes)`**
  (added). Column wrappers `TotalColumnLayout::new(n_bin,n_arb,n_int)` etc.
- **`ConstraintEndpoints<F>`** (`#[non_exhaustive]`): `pub r_0`, `pub r_0_fq`;
  `ConstraintEndpoints::new(r_0, r_0_fq)`.

## A.2 The sumcheck primitive — `piop/src/sumcheck/multi_degree.rs`

`CombFn<F> = Box<dyn Fn(&[F]) -> F + Send + Sync>` (from `zinc_piop`); slice holds
one value per group MLE in listed order. Field ops are `Mul<&Self>` (not
`Mul<Self>`) ⇒ write `v[0].clone() * &v[1]`.

```rust
MultiDegreeSumcheckGroup::new(degree: usize,
    poly: Vec<DenseMultilinearExtension<F::Inner>>, comb_fn: CombFn<F>) -> Self

MultiDegreeSumcheck::<F>::prove_as_subprotocol(transcript, 
    families: Vec<(Vec<MultiDegreeSumcheckGroup<F>>, &F::Config)>, num_vars, q_star_cfg)
    -> Vec<(MultiDegreeSumcheckProof<F>, Vec<SumcheckProverState<F>>)>

MultiDegreeSumcheck::<F>::verify_as_subprotocol(transcript, num_vars,
    proofs: &[(&MultiDegreeSumcheckProof<F>, &F::Config)], q_star_cfg)
    -> Result<Vec<MultiDegreeSubClaims<F>>, SumCheckError<F>>
```
- Single-family: prove `vec![(vec![group], cfg)]`, `q_star_cfg=cfg`; verify
  `&[(&proof, cfg)]`. Pop the single element.
- Prover recovers the point via `SumcheckProverState.randomness: Vec<F>` (public);
  verifier via `MultiDegreeSubClaims::point()`/`expected_evaluations()`. Claimed
  sums via `MultiDegreeSumcheckProof::claimed_sums()`.
- **FS metadata (risk g):** both calls absorb `(num_vars, num_families, per-group
  degrees)` at the very start — squeeze `τ`/`r_A,r_B,r_C` outside them.
- The `#[cfg(test)]` block is a near-verbatim usage template.

## A.3 eq / MLE helpers — `poly/src/utils.rs`, `poly/src/mle/dense.rs`

```rust
build_eq_x_r_vec<F: PrimeField>(r, cfg) -> Result<Vec<F>, ArithErrors>
build_eq_x_r_inner<F: InnerTransparentField>(r, cfg) -> Result<DenseMultilinearExtension<F::Inner>, _>
eq_eval<R: Semiring>(x, y, one: R) -> Result<R, ArithErrors>
mle_eval_with_eq_table<F: InnerTransparentField>(evals: &[F::Inner], eq_table: &[F], cfg) -> F
DenseMultilinearExtension::from_evaluations_vec(num_vars, evals: Vec<T>, zero: T)
DynamicPolynomialF::evaluate_at_point(&F) -> Result<F, _>   // Horner; at 0 ⇒ constant term
```

## A.4 Our sparse matrix (to build in R1) — `protocol/src/r1cs_frontend.rs`

Variable-density CSR mirroring ProveKit (no interner, no delta serde). Sketch:
```rust
pub struct SparseMatrix<T> { pub num_rows, pub num_cols, row_starts: Vec<usize>, entries: Vec<(usize, T)> }
impl<T> SparseMatrix<T> {
    fn from_rows(num_cols, rows: impl IntoIterator<Item = Vec<(usize, T)>>) -> Self;  // sorts each row by col
    fn iter_row(&self, r) -> impl Iterator<Item = (usize, &T)>;
    fn iter(&self) -> impl Iterator<Item = (usize, usize, &T)>;
    fn num_entries(&self) -> usize;
}
```
Building `Az` and `M̃(r_x,r_y)` now reads `value ∈ F` directly (no projection);
the outer sumcheck needs `#constraints` a power of two `≥ 2` (`s_x ≥ 1`).

# Appendix B — reference reading

- **Structural template:** `protocol/src/uair_frontend.rs` — `UairConstraintProof`
  `Transcribable`, `impl ConstraintSystem` bodies.
- **Substrate contract:** `protocol/src/prover.rs` (`step_constraints` →
  `prove_constraints`; `project_common` samples the prime — R2 hook here) and
  `protocol/src/verifier.rs` (`finish_verify`; `step1_prime_projection` samples —
  R2 hook here; `assemble_all` + `verify_lifted_evals`).
- **`ProjectedTrace`:** `piop/src/projections.rs`.
- **Test harness:** `protocol/src/lib.rs` e2e/tamper tests (`Zt` types,
  `ZincPlusPiop::prove/verify`, `UairTrace`); `multi_degree.rs` tests.
- **ProveKit (reference, do not import):** `provekit/provekit/common/src/r1cs.rs`
  (`R1CS`, `num_public_inputs`, `num_witnesses = num_cols`), `.../sparse_matrix.rs`
  (variable-density CSR, right-multiply), `.../utils/sumcheck.rs`.
- **Paper:** `crypto/crypto_piop.tex` (`R = F_q` add-on soundness);
  `UCS/structural_instantiations.tex` (R1CS as UCS, for the postponed general case).
