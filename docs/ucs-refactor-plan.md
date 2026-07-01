# Plan: abstract UAIR behind a UCS constraint-system trait

Status: **in progress** — Phases 1–3, 3.5, and 4a done (§12): **both** prover and verifier now route their constraint argument through the `UairFrontend` `ConstraintSystem` seam (`prove_constraints` / `verify_constraints` + `verify_lifted_evals`), proof-preserving (full 30-test suite green, incl. all tamper/soundness negatives). The seam hands the substrate only the shared endpoint `r_0` (`ConstraintEndpoints`); the substrate's post-constraint job is lift-and-project + the PCS open (+ assembling the per-family lifted evals the verifier's `verify_lifted_evals` binds). **Remaining (Phase 4b):** genericize `ZincPlusPiop` over `CS` (drop the `U` pin) and embed `UairConstraintProof` inside `Proof<F>` (retiring the flat fields + the repack/unpack on both sides). Revise freely. Owner: Alexander.

> **Self-containment note.** This document is written to survive a context reset:
> the Background, Repository map, Current pipeline, Authoring-layer, and
> Constraint-engine sections capture the facts the plan depends on, gathered by
> reading the code (commit on `main`, mid-2026). File/symbol references are
> point-in-time — re-verify against the tree before relying on a specific
> `path:symbol`. A glossary is at the end.

---

## 1. Goal

Make the `protocol` crate prove/verify over a generic **constraint system** rather
than hard-wiring `U: Uair`. UAIR becomes the one (current) implementation behind a
trait. No R1CS implementation now, but the trait is shaped so a future Spartan/R1CS
frontend slots in without a second seam.

---

## 2. Background / orientation

**What Zinc+ is.** Zinc+ is a SNARK framework (paper: `Zinc+: SNARKs for Polynomial
Rings`, sources under `/home/fs/code/zinc-plus-paper`). It proves a relation called
**UCS (Universal Constraint System)**: algebraic constraints *and* ideal-membership
predicates over several rings at once — `Q[X]` (hence `Z[X]`, `Z`) and one or more
`F_{q_i}[X]` (hence `F_{q_i}`). Witness entries are *polynomials* (degree `< d`,
bounded coefficient bit-size), not field elements. Ideal-membership encodes bit
operations cheaply: `X-2` links a bit-polynomial to the integer it represents,
`X^w-1` encodes rotation, `X-1` Hamming weight, etc.

**The Zinc+ compiler pipeline** (the spine of the implementation; paper "Technical
Overview / Zinc+ PIOP"):

```
 Z[X] --phi_q--> F_q[X] --(ideal-membership batched to equality)--> F_q[X] --psi_a--> F_q --> field PIOP
       step 1                       step 2                                 step 3
```

- `phi_q`: coefficient-wise reduction (a random ~Omega(lambda)-bit prime `q0` when
  the ring is `Q`; an embedding/declared prime for `F_q`). Brings constraints into
  `F_q[X]`.
- ideal-membership -> strict equality: prover sends the supposed remainder `e`,
  verifier checks `e` is in the ideal, constraint becomes `Q(...) - e = 0`.
- `psi_a`: evaluate at a random `X = a`, dropping `F_q[X] -> F_q`.
- The resulting `F_q` equality is proved by any finite-field PIOP. Witness columns
  are committed once over the ring via **Zip+** (a Brakedown/Ligero-style hash-based
  IOPP) using **IPRS codes** (MDS codes over `Q` with FFT encoding and bounded norm
  growth); the committed oracle answers *projected* evaluation queries.

**UCS vs UAIR vs R1CS.**
- **UCS** is the general *relation*.
- **UAIR (Universal AIR with lookups)** is the AIR-like *specialization* this
  codebase implements: a uniform per-row constraint applied across a trace, with
  access to the current row (`up`) and a shifted row (`down`). This is what the
  `Uair` trait and the whole `protocol` pipeline are built around today.
- **R1CS** is a different specialization (future). Crucially, UAIR and R1CS share
  the *compiler + commitment substrate* (phi_q / psi_a / Zip+) but have different
  *constraint arguments* (UAIR = one width-bounded sumcheck over the trace via the
  "CPR" engine; R1CS = Spartan-style nested sumchecks over matrix MLEs). The paper's
  architecture figure shows the finite-field PIOP as a pluggable black box for
  exactly this reason.

**Why this refactor.** Today `protocol::ZincPlusPiop<Zt, U, F, D, FD>` is generic
over `U: Uair`, and UAIR-specific logic is threaded throughout. We want `protocol`
generic over a `ConstraintSystem` *seam* so UAIR is one implementation and R1CS can
be added later as another, reusing the substrate unchanged. The substrate already
reaches everything UAIR-specific through a small set of touchpoints (`U::signature()`,
`count_constraints::<U>`, the `CombinedPolyResolver` / `IdealCheckProtocol` engines,
and the `project_scalar` / `project_ideal` / `project_fq_ideal` closures), so the
seam is largely latent already.

**Paper sources (for deeper reference).** `crypto/crypto_ucs.tex` (UCS + encodings),
`crypto/crypto_piop.tex` (the compiler steps), `crypto/crypto_iopp.tex` (Zip+),
`crypto/crypto_iprs.tex` (IPRS codes), `UCS/structural_instantiations.tex` (R1CS /
lookup / AIR as UCS), `PIOR/zincplus_pior.tex` (the PIOR). Note: KaTeX in this repo's
rustdoc does not support `\Q`/`\FF` macros — use `\mathbb{Q}` / `\mathbb{F}`.

---

## 3. Repository map

Cargo workspace at `/home/fs/code/zinc-plus`. Members:

| Crate | Role |
|---|---|
| `protocol` | **The Zinc+ PIOP end-to-end** (prover/verifier pipeline). The crate this refactor mainly touches. |
| `piop` | PIOP building blocks: sumcheck, the constraint engines (CPR, ideal-check), multipoint-eval, projections, shift predicate, lookup/booleanity. |
| `uair` | The `Uair` authoring trait + `UairSignature`/`UairTrace`/`TraceRow` data model + `ConstraintBuilder` and the static-analysis builders. |
| `zip-plus` | The Zip+ IOPP/PCS + IPRS/RAA codes. |
| `poly` | MLEs, dense/binary/dynamic polynomials. |
| `primality` | Prime sampling/testing (Miller–Rabin). |
| `transcript` | Fiat–Shamir transcript traits. |
| `utils` | Misc (inner products, `FromRef`, `MulByScalar`, etc.). |
| `test-uair` | Test `Uair` impls (`ShaProxy`, `BigLinearUair`, `TestUair*`, `BinaryDecompositionUair`, …) used by the e2e tests. |

Key files/symbols:
- `protocol/src/lib.rs` — `ZincPlusPiop<Zt, U, F, const D, const FD>` (`D` =
  `DEGREE_PLUS_ONE`, `FD` = folded degree+1), the `ZincTypes` type-bundle trait,
  `Proof<F>` and its (de)serialization, helper fns (`compute_lifted_evals`,
  `build_all_cfgs`, `absorb_public_columns`).
- `protocol/src/prover.rs` — type-state prover, `step0_fold` … `step8_pcs_open`,
  `finish`, and `prove(...)`.
- `protocol/src/verifier.rs` — `step0_reconstruct_transcript` … `step7_pcs_verify`,
  `finish`, and `verify(...)`.
- `protocol/src/shared_challenge.rs` — `compute_q_star_idx`,
  `sample_shared_field_challenge(s)` (one shared integer point lifted into each
  family's field).
- `protocol/src/fold.rs` — `FoldTrace` (binary-column folding).
- `piop/src/combined_poly_resolver.rs` (+ `folder.rs` = `ConstraintFolder`,
  `structs.rs`) — **CPR**, the constraint-aggregation sumcheck engine.
- `piop/src/ideal_check.rs` (+ `combined_poly_builder.rs` = `CombinedPolyRowBuilder`,
  `batched_ideal_check.rs`, `structs.rs`) — **IdealCheckProtocol**.
- `piop/src/multipoint_eval.rs` — `MultipointEval`, `MultipointEvalFamilyInputs`.
- `piop/src/projections.rs` — `project_trace_coeffs_row_major/column_major`,
  `project_scalars`, `project_scalars_to_field`, `build_bit_op_virtual_mle`,
  `evaluate_trace_to_column_mles`, `collect_scalars`, `ProjectedTrace`.
- `piop/src/sumcheck/` (`multi_degree.rs`, `prover.rs`, `verifier.rs`),
  `shift_predicate.rs`, `random_field_sumcheck.rs`, `lookup/booleanity.rs`.
- `uair/src/lib.rs` — `Uair`, `ConstraintBuilder`, `UairSignature`, `UairTrace`,
  `TraceRow`, `ColumnLayout` (+ Total/Public/Virtual/Witness wrappers), `ShiftSpec`,
  `BitOpSpec`/`BitOp`. `uair/src/{constraint_counter, degree_counter, collect_scalars,
  ideal_collector, do_nothing_builder, dummy_semiring, ideal, lookup_types}.rs`.

---

## 4. The current protocol pipeline (what the refactor reshapes)

The protocol is a **type-state machine**: each prover step consumes a `ProverX`
struct and returns `ProverX+1`. The verifier mirrors it. Tag = which side of the
planned seam each step lands on.

**Prover** (`protocol/src/prover.rs`):

| Step | Fn | Does | Seam |
|---|---|---|---|
| 0 | `step0_fold` | Fold the binary witness columns (`Zt::BinaryFold`). | substrate |
| 1 | `step1_commit` | Zip+ commit the witness as **three** column groups → `(bin, arb, int)` commitments; absorb public columns. | substrate |
| 2 | `step2_combined` / `step2_mle_first` | `phi_q`: sample random prime `q0`, project the trace to `F_{q0}[X]`; stage per-declared-prime projections (`fq_staging`); compute `q*` (min prime) for shared-challenge ranges. Two layouts: row-major ("combined") vs column-major ("mle_first"). | substrate |
| 3 | `step3_ideal_check` | Ideal-membership check per family via `IdealCheckProtocol::prove_*`. | **frontend** |
| 4 | `step4_eval_projection` | `psi_a`: sample shared `a`, project trace MLEs to `F_q`; **+** project UAIR scalars (`project_scalar`) and build bit-op virtual MLEs (`build_bit_op_virtual_mle`). | mixed (psi_a = substrate; scalars/bit-ops = frontend) |
| 5 | `step5_sumcheck` | Constraint sumcheck via **CPR** (`prepare_sumcheck_group` + `finalize_prover`); optional **booleanity** group over witness binary columns; lookup groups (TODO). Lockstep multi-degree sumcheck across families. Squeezes `alpha'` for the booleanity→MP bridge. | **frontend** |
| 6 | (inside `prove_constraints`) | Lockstep multipoint-eval over all families: combine each family's `up_evals`/`down_evals`/`bit_op_evals` (+ the appended `alpha'` booleanity-bridge columns) into a single shared endpoint `r_0`. **Now the closing move of the frontend's constraint argument** (Phase 3.5); the generic `MultipointEval` algorithm stays in `piop`, but the substrate pipeline no longer invokes it — it receives only `r_0` via `ConstraintEndpoints`. | **frontend** |
| 7 | `step7_lift_and_project` | Sample PCS-only prime `q''` (aliased to `q0` when no `F_q[X]` families — the `n_fq == 0` trick); compute per-family witness-only **lifted evals** at the `r_0` the frontend returned; `r* = r_0 mod q''`. Uses layout for witness-column extraction. | substrate |
| 8 | `step8_pcs_open` | Zip+ PCS open the three commitments at `r*` (binary uses extended point with folding challenges). | substrate |
| — | `finish` | Assemble `Proof<F>`. | substrate |

**Verifier** (`protocol/src/verifier.rs`) — **as landed in Phase 4a** (mirror of the
prover): `step0_reconstruct_transcript` (substrate; also repacks the flat `Proof`
fields into a `&UairConstraintProof`), `step1_prime_projection` (substrate), then a
single `finish_verify` that:
- calls `frontend.verify_constraints(..)` (**frontend**, former verifier steps 2–5:
  ideal-check verify, `psi_a` eval projection, CPR + booleanity verify, `alpha'`
  squeeze, and the lockstep `MultipointEval::verify_as_subprotocol` → `r_0`);
- keeps the **substrate** lift-assembly (sample `q''`; length guards; recompute
  public lifted evals + interleave with the sent witness lifts → per-family
  all-column lifts; `r* = r_0 mod q''`; absorb lift coeffs);
- calls `frontend.verify_lifted_evals(claims, per_family_all_lifted, cfgs)`
  (**frontend**: `psi_a`-project + bit-op / `alpha'` reconstruction + `verify_subclaim`
  per family) between assembly and coeff-absorption;
- runs the substrate `pcs_verify` (`q''`/Zip+ verify).

The intermediate verifier type-states (`VerifierIdealChecked`/`EvalProjected`/
`Sumchecked`/`MultipointEvaled`/`LiftedEvalsChecked`/`PcsVerified`) collapsed, mirroring
the prover. FS transcript order is byte-identical (`verify_subclaim` is transcript-free,
so relocating it is safe). Test-only stepwise entry (`step2_ideal_check` /
`step3_eval_projection`) survives in a `#[cfg(test)] mod test_helpers` for the tamper
replay tests.

**Family model.** Families = `[Q[X] (q0), q_1, ..., q_n]` where `q_1..q_n =
layout.primes()`. One shared integer evaluation point/challenge is sampled in
`[0, q*)` (`q* = min q_i`) and lifted into each family's field; the lockstep
sumcheck/MP-eval keep all families on a single shared `r_0`. `q''` is the PCS-only
prime, decoupled from constraint primes (or aliased to `q0` when `n = 0`).

`Proof<F>` fields today: `commitments: (Zip,Zip,Zip)`, `zip: Vec<u8>`,
`ideal_check`, `cpr_proof`, `combined_sumcheck`, `multipoint_eval`,
`witness_lifted_evals: Vec<Vec<DynamicPolynomialF<F>>>` (per family),
`lookup_proof: Option`, `booleanity_proof: Option`, and per-prime vectors
`ideal_checks_fq`, `cpr_proofs_fq`, `combined_sumchecks_fq`, `multipoint_evals_fq`,
plus `witness_lifted_evals_pp: Option` (the `q''` family; `None` ⟺ `q'' := q0`).

---

## 5. Authoring layer: the `Uair` data model (`uair/src/lib.rs`)

This is what arithmetizations write today and what the `UairFrontend<U>` adapter
will wrap. The refactor keeps it.

- **`trait Uair`**: associated types `Ideal`, `FqIdeal` (ideal types for `Q[X]` and
  `F_{q_i}[X]` constraints), `Scalar` (usually `DensePolynomial<Ring, 32>`), `Prime`;
  `fn signature() -> UairSignature<Self::Prime>`; `fn constrain_general<B: ConstraintBuilder>(b, up: TraceRow, down: TraceRow, from_ref, mbs, ideal_from_ref, fq_ideal_from_ref)`
  (and a convenience `constrain`). The constraint logic is *code* run against an
  opaque builder — there are no stored matrices.
- **`trait ConstraintBuilder`**: `type Expr: Semiring`, `type Ideal`, `type FqIdeal`;
  `assert_in_ideal`, `assert_zero`, `assert_in_fq_ideal(prime_idx, expr, ideal)`.
  The protocol/engines supply concrete builders.
- **`UairSignature<Prime>`**: `total_cols`/`public_cols`/`witness_cols`/`down_cols`
  (each a `ColumnLayout` = counts of `binary_poly` / `arbitrary_poly` / `int`
  columns; public precede witness within each type), `shifts: Vec<ShiftSpec>`,
  `bit_op_specs: Vec<BitOpSpec>`, `lookup_specs: Vec<LookupColumnSpec>`,
  `primes: Vec<Prime>`. Flattened trace order: `[pub_bin, wit_bin, pub_arb, wit_arb,
  pub_int, wit_int]`; down-row order: `[shifted_bin, bitop_bin, shifted_arb, shifted_int]`.
- **`ShiftSpec { source_col, shift_amount }`** — AIR row shift (a virtual `down`
  column). **`BitOpSpec { source_col, op: BitOp::{Rot(c), ShR(c)} }`** — an
  `R`-linear entry-wise map on `R^{<W}[X]` (rotation / right-shift) materialized as a
  *virtual, uncommitted* column; the verifier reconstructs its MLE eval from the
  source column's opening (paper Lemma: `R`-linear coordinate-wise maps commute with
  MLE). Bit-ops only on `binary_poly` columns.
- **`UairTrace`**: the three column groups as `DenseMultilinearExtension`s
  (`binary_poly: BinaryPoly<DB>`, `arbitrary_poly: DensePolynomial<PolyCoeff, DA>`,
  `int: Int`); `.public(sig)` / `.witness(sig)` slice it.
- **Static-analysis builders** (all run `U::constrain[_general]` once against a
  special `ConstraintBuilder` to extract facts — all UAIR-specific, all become
  frontend-internal): `constraint_counter` (count per family),
  `degree_counter` (per-constraint degree → `count_max_degree`), `collect_scalars`
  (gather the scalar set for projection), `ideal_collector` (gather `Ideal`/`FqIdeal`
  refs), `do_nothing_builder` + `dummy_semiring` (no-op sinks).

---

## 6. Constraint-engine interface (what `UairFrontend<U>` wraps)

The seam's prover/verifier methods will delegate to these. Their current interface:

- **`CombinedPolyResolver` (CPR)** — the constraint sumcheck.
  - `prepare_sumcheck_group::<U>(trace_mles, bit_op_mles, eval_point, projected_scalars,
    family_idx, num_constraints, num_vars, max_degree, folding_challenge, cfg)
    -> (MultiDegreeSumcheckGroup, CprProverAncillary)` — builds the combined MLE
    group; its `comb_fn` runs `U::constrain_general` into `ConstraintFolder` (RLC of
    constraints by powers of a folding challenge).
  - `finalize_prover::<U>(ts, sumcheck_state, ancillary, cfg) -> (CprProof, CprProverState)`
    — yields `up_evals`, `down_evals`, `bit_op_evals`, eval point.
  - `prepare_verifier::<U>(...)`, `finalize_verifier::<U>(...)` — verifier mirror;
    `finalize_verifier` re-runs `U::constrain_general` on the evals to check the
    combination at the shared point.
- **`IdealCheckProtocol`** — the ideal-membership check.
  - `prove_combined::<_, D>` (row-major) / `prove_mle_first::<_, D>` (column-major),
    `verify_as_subprotocol::<_, IdealOverF, _>` (uses `batched_ideal_check`,
    generic over the ideal type). Uses `collect_ideals::<U>` + `count_constraint_degrees::<U>`.
- **`MultipointEval`** (a generic `piop` primitive, **invoked by the frontend** as of
  Phase 3.5) — `prove_as_subprotocol(ts, families: Vec<MultipointEvalFamilyInputs>,
  shifts, q_star_cfg) -> per-family (proof, state{eval_point})`; `verify_subclaim(...)`.
  `MultipointEvalFamilyInputs { field_cfg, trace_mles, bit_op_mles, eval_point,
  up_evals, bit_op_evals, down_evals }` is the shape of the frontend's *internal*
  `FamilyEvalClaims` (no longer crosses the seam — see §9).

So: `prove_constraints` ≈ (step3 ideal-check) + (step4 scalar/bit-op) + (step5 CPR +
booleanity) + (step6 multipoint-eval), returning the
`CprProof`/`IdealCheckProof`/`booleanity`/`MultipointEvalProof` bundle plus the shared
endpoint `r_0`. The verifier mirror is **split across two frontend calls** because
MP-eval verify itself splits: `verify_constraints` runs `verify_as_subprotocol`
(→ `r_0` + subclaims), and — since `verify_subclaim` binds those subclaims to the
substrate's *lifted evals*, which only exist after the substrate lifts at `r_0` —
`verify_lifted_evals` runs `verify_subclaim` post-lift. (This asymmetry has no prover
analog: the prover trusts its own lifts by construction; the verifier must re-check
them.) The `MultipointEval` *algorithm* remains a reusable `piop` primitive any future
frontend can call (R1CS/Spartan, which terminates at a single point, won't need it).

---

## 7. Settled architectural decisions

1. **Seam = constraint-argument level.** The trait owns the *constraint argument*:
   ideal-check (step 3), the `psi_a` scalar-projection / bit-op-virtual parts of
   step 4, the constraint sumcheck + booleanity (step 5), and — as of Phase 3.5 — the
   lockstep multipoint-eval (step 6) that reduces all per-family claims to a single
   shared point `r_0`. The substrate keeps: fold, commit, `phi_q` prime projection,
   lift-and-project, Zip+ PCS open. The frontend hands the substrate only the shared
   endpoint `r_0`; the substrate binds it to the commitment via lift + PCS open.
   - Rejected: "thin per-row seam" (keep CPR/IdealCheck as the engine, trait =
     `constrain_general` only) — it hard-wires the uniform up/down trace and R1CS
     can't fit.
2. **Keep the 3-group witness layout** (`binary_poly` / `arbitrary_poly` / `int`)
   and the binary-folding as a substrate fixture, exposed via the trait's layout
   accessor. A future R1CS frontend reuses a group (flat witness in `int` /
   `arbitrary`). Full N-typed-column abstraction is deferred.

---

## 8. Two-layer trait design (the key idea)

Keep `Uair` as the *authoring* trait (per-row ergonomics — what arithmetizations
write today) and add `ConstraintSystem` as the *protocol-facing* trait (the seam).
Bridge them with a blanket adapter so **no existing arithmetization changes**:

The trait surface as **landed** (`protocol/src/constraint_system.rs`):

```rust
// authoring layer (unchanged): write a UAIR row-by-row
trait Uair { type Ideal; type FqIdeal; type Scalar; type Prime;
             fn signature(); fn constrain_general(..); }

// protocol seam (landed): the thing `protocol` will be generic over
trait ConstraintSystem {
    type Prime: Semiring;                  // UairSignature prime, e.g. Zt::Fmod
    type Field: PrimeField;                // the projection field F (note 1)
    type ConstraintProof: Transcribable;   // frontend sub-proof (Transcribable only; note 3)

    // --- verifier-side associated types (Phase 4a) ---
    type VerifierClaims;                   // opaque frontend verifier tail-state,
                                           // carried verify_constraints -> verify_lifted_evals
    type IdealSource;                      // UAIR: IdealOrZero<U::Ideal>   (Q[X] family)
    type FqIdealSource;                    // UAIR: IdealOrZero<U::FqIdeal> (F_q[X] families)
    type Scalar;                           // UAIR: U::Scalar               (psi_a scalar projection)

    fn layout(&self) -> &Layout<Self::Prime>;            // Layout = alias of UairSignature
    fn needs_decoupled_pcs_prime(&self) -> bool {        // default reproduces UAIR (decision 6)
        !self.layout().primes().is_empty()
    }

    // returns the sub-proof + the shared endpoint r_0 the frontend's own
    // multipoint-eval reduced to (Phase 3.5). FamilyEvalClaims no longer crosses
    // the seam — it is frontend-internal scaffolding for that reduction.
    fn prove_constraints(&self, ts: &mut impl Transcript,
        projected_traces: &[ProjectedTrace<Self::Field>],
        field_cfgs: &[<Self::Field as HasPrimeFieldConfig>::Config],
        num_vars: usize,
    ) -> Result<(Self::ConstraintProof, ConstraintEndpoints<Self::Field>),
                ProtocolError<Self::Field>>;

    // verifier steps 2-5: ideal-check verify, psi_a projection, CPR+booleanity
    // verify, MP-eval verify_as_subprotocol -> r_0. The IdealOverF + projection
    // closures mirror the current verify() wrapper; passed (not stored) so the
    // verify-side frontend needn't carry the prover's scalar map.
    fn verify_constraints<IdealOverF>(&self, ts: &mut impl Transcript,
        proof: &Self::ConstraintProof,
        field_cfgs: &[<Self::Field as HasPrimeFieldConfig>::Config],
        num_vars: usize,
        project_ideal:    impl Fn(&Self::IdealSource,   &Cfg) -> IdealOverF,
        project_fq_ideal: impl Fn(&Self::FqIdealSource, &Cfg) -> IdealOverF,
        project_scalar:   impl Fn(&Self::Scalar,        &Cfg) -> DynamicPolynomialF<Self::Field>,
    ) -> Result<(ConstraintEndpoints<Self::Field>, Self::VerifierClaims),
                ProtocolError<Self::Field>>
    where IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<Self::Field>>;

    // post-lift binding (replaces the old `reconstruct_virtual_evals`): the
    // substrate hands per-family FULL (public+witness, layout-interleaved) lifted
    // evals at r_0; the frontend projects them, reconstructs bit-op / alpha'
    // virtuals, and runs MP-eval verify_subclaim per family.
    fn verify_lifted_evals(&self, claims: &Self::VerifierClaims,
        per_family_all_lifted: &[Vec<DynamicPolynomialF<Self::Field>>],
        field_cfgs: &[<Self::Field as HasPrimeFieldConfig>::Config],
    ) -> Result<(), ProtocolError<Self::Field>>;
}   // (Cfg = <Self::Field as HasPrimeFieldConfig>::Config, elided above)

// adapter (protocol/src/uair_frontend.rs; prover impl landed Phase 3/3.5, verifier 4a).
// Verify-side is constructed via `UairFrontend::new_verifier()` (empty witness cols +
// an unreachable stored project_scalar; verify_constraints uses its closure params).
struct UairFrontend<'a, U: Uair, F, const D: usize, const FD: usize> { /* borrows trace + cached signature */ }
```

**Resolved during the abstract-surface milestone:**
1. **`Field` is an associated type, not a method generic.** A method `fn
   prove_constraints<F: PrimeField>` cannot add, in the impl, the prover's extra field
   bounds (`InnerTransparentField`, `FromWithConfig<&_>`, `MulByScalar`, …,
   `Integer = Zt::Fmod`). `type Field: PrimeField` lets `UairFrontend<U, F, …>` pick a
   concrete, fully-bounded `F`, mirroring `ZincPlusPiop<Zt, U, F, D, FD>`.
2. `Config` is reached via `<Self::Field as HasPrimeFieldConfig>::Config` (`Config` is a
   `PrimeField` *supertrait* assoc type, not on `PrimeField` itself).
3. **`ConstraintProof: Transcribable` only.** `GenTranscribable` exists purely for
   deserialization code-dedup; it is not part of the seam contract.
4. Const generics `D`/`FD` live on the **frontend type** (decision 5), not the methods.

- R1CS later: `impl ConstraintSystem for R1csFrontend` directly (Spartan inside),
  no `Uair`, single family, trivial `verify_lifted_evals` (no virtual columns; it
  terminates at a single point).
- Bridge for UAIR: `impl<'a, U: Uair, F, const D, const FD> ConstraintSystem for
  UairFrontend<'a, U, F, D, FD>` — prover bodies landed Phase 3/3.5, verifier bodies
  (`verify_constraints` + `verify_lifted_evals`) Phase 4a.

---

## 9. Shared data contract (the types that cross the seam)

| Type | Today | Role |
|---|---|---|
| `Layout<Prime>` | `UairSignature` (column layout + `shifts` + `primes`) | What the substrate needs for commit / projection / lift. **Bit-ops, lookups, shifts stay frontend-internal** (the frontend's own MP-eval consumes them; they never surface across the seam). |
| `ConstraintEndpoints<F>` | `{ r_0: Vec<F>, r_0_fq: Vec<Vec<F>> }` (`#[non_exhaustive]`) | The single shared evaluation point the frontend's multipoint-eval reduced all per-family claims to. The substrate's lift-and-project + PCS open bind the commitment to it. R1CS/Spartan returns its single terminating point here directly. |
| `Self::ConstraintProof` | `{ ic_proof(+_fq), cpr_proof(+_fq), combined_sumcheck(+_fq), multipoint_eval(+_fq), booleanity_proof, lookup_proof }` | Frontend sub-proof; embedded in the substrate `Proof`. **Includes the MP-eval proofs as of Phase 3.5.** |
| `Self::VerifierClaims` (verifier only) | UAIR: `{ mp_subclaim(+_fq), projecting_elements, alpha_prime }` (opaque to substrate) | Frontend verifier tail-state from `verify_constraints` → `verify_lifted_evals`. The substrate carries it uninterpreted. |
| `per_family_all_lifted` (verifier only) | `&[Vec<DynamicPolynomialF<F>>]` | Substrate → frontend: each family's FULL (public-recomputed + sent-witness, layout-interleaved) lifted evals at `r_0`, which `verify_lifted_evals` binds via `verify_subclaim`. |

The substrate `Proof<F>` **will** become (Phase 4b, not yet):
`{ commitments, zip, witness_lifted_evals(+_pp), constraint_proof: CS::ConstraintProof }`
(the `multipoint_eval(+_fq)` proofs now live *inside* `ConstraintProof`).

**Landed:** `ConstraintEndpoints<F>` is in `constraint_system.rs`; `VerifierClaims`/
`IdealSource`/`FqIdealSource`/`Scalar` are trait associated types (UAIR:
`UairVerifierClaims<F>`, `IdealOrZero<U::Ideal>`, `IdealOrZero<U::FqIdeal>`, `U::Scalar`).
`FamilyEvalClaims<F>` moved into `uair_frontend.rs` as a **private** struct (`new(..)` +
`as_inputs(&self) -> MultipointEvalFamilyInputs<'_, F>`); it is now internal scaffolding
for the frontend's multipoint-eval, not a seam type. `UairConstraintProof<F>` carries
`multipoint_eval` + `multipoint_evals_fq`, with a hand-written
`GenTranscribable`/`Transcribable` that faithfully mirrors the `Proof<F>` byte layout.
**Still not exercised (Phase 4b):** the substrate `Proof` remains flat and
hand-serialized; both sides repack (prover `finish` unpacks the frontend sub-proof into
`Proof`, verifier `step0` repacks the flat `Proof` into a `&UairConstraintProof`).
`ConstraintProof` is the trait's associated type bounded `Transcribable`.

---

## 10. Substrate vs frontend partition (grounded in current code)

**Substrate (stays generic; only learns the layout via `cs.layout()`):**
- `protocol` steps: `step0_fold`, `step1_commit`, `step2_*` (`phi_q` prime
  projection + per-family staging), `step7_lift_and_project`, `step8_pcs_open`,
  `finish`; verifier `step0_reconstruct_transcript`/`step1_prime_projection`, the
  lift-assembly inside `finish_verify` (sample `q''`, recompute public lifts +
  interleave with sent witness lifts, absorb, `r*`), and `pcs_verify`.
- `piop`: `sumcheck/*`, `multipoint_eval`, `shift_predicate`, `random_field_sumcheck`,
  `ideal_check/batched_ideal_check`, `lookup/booleanity` (generic), trace-projection
  in `projections.rs`, the `*/structs.rs` proof/state types. These stay generic `piop`
  primitives; the `multipoint_eval` *algorithm* lives here but is now *invoked* by the
  frontend (Phase 3.5).
- Multi-family structure (Q[X] + per-prime F_q[X]) + `q''` PCS prime + Zip+/transcript.

**Frontend = `UairFrontend<U>` (wraps the UAIR-specific engine):**
- `piop`: `combined_poly_resolver` (CPR) + `folder.rs` (`ConstraintFolder`),
  `ideal_check` prove paths + `combined_poly_builder` (`CombinedPolyRowBuilder`),
  `projections::collect_scalars`.
- `uair`: the `Uair` trait + `constrain_general`, and the four static-analysis
  builders (`constraint_counter`, `degree_counter`, `collect_scalars`,
  `ideal_collector`) + `do_nothing_builder` + `dummy_semiring`.
- The `project_scalar` / `project_ideal` / `project_fq_ideal` closures (today passed
  into `prove`/`verify`) move *inside* the frontend.
- Bit-op virtual MLE building (`build_bit_op_virtual_mle`, step 4) and verifier-side
  `derive_bit_op_open_evals` (now lives in `uair_frontend.rs`); booleanity `alpha'` bridge.
- **Invoking** the lockstep `MultipointEval` over its per-family `FamilyEvalClaims`:
  `prove_as_subprotocol` in `prove_constraints` (Phase 3.5) and `verify_as_subprotocol`
  in `verify_constraints` + `verify_subclaim` in `verify_lifted_evals` (Phase 4a). The
  algorithm stays in `piop`.

Per-module verdicts from the coupling survey (S=substrate, F=frontend, M=mixed):
`combined_poly_resolver.rs` F · `.../folder.rs` F · `.../structs.rs` S ·
`ideal_check.rs` M (prove paths + `collect_ideals`/`count_constraint_degrees` F;
`verify_as_subprotocol` ideal-membership S) · `ideal_check/combined_poly_builder.rs` F ·
`ideal_check/batched_ideal_check.rs` S · `ideal_check/structs.rs` S ·
`multipoint_eval.rs` S (generic algorithm in `piop`; **invoked by the frontend** as of
Phase 3.5) · `projections.rs` M (trace projection S; `collect_scalars` F) ·
`sumcheck/*` S · `lookup/booleanity.rs` S (layout passed in) · `shift_predicate.rs` S ·
`random_field_sumcheck.rs` S.

---

## 11. Data flow with the seam marked

```
                    SUBSTRATE                         |   FRONTEND (CS)
 fold -> commit -> phi_q project (per family) ........|
                                                      |  prove_constraints:
                                                      |   step3 ideal-check
                                                      |   step4 psi_a scalars + bit-op virtuals
                                                      |   step5 constraint sumcheck (+booleanity)
                                                      |   step6 multipoint-eval (lockstep, all
                                                      |         families) over internal
                                                      |         FamilyEvalClaims -> r_0
                                                      |  -> ConstraintProof + ConstraintEndpoints{r_0}
 step7 lift-and-project (witness cols via layout) <---|  (consumes only r_0)
   at r_0                                             |
 step8 Zip+ PCS open at r* ...........................|
```
Verifier mirror (Phase 4a, landed):
```
                    SUBSTRATE                              |   FRONTEND (CS)
 reconstruct transcript -> phi_q prime projection .........|
                                                           |  verify_constraints:
                                                           |   ideal-check verify / psi_a /
                                                           |   CPR+booleanity verify /
                                                           |   MP-eval verify_as_subprotocol
                                                           |  -> ConstraintEndpoints{r_0} + VerifierClaims
 sample q''; recompute public lifts + interleave with      |
   sent witness lifts -> per_family_all_lifted ............|
                                                           |  verify_lifted_evals(claims, lifts):
                                                           |   psi_a-project + bit-op/alpha'
                                                           |   reconstruct + verify_subclaim/family
 absorb lift coeffs; pcs_verify at r* .....................|
```
`verify_subclaim` is transcript-free, so running it inside `verify_lifted_evals`
(after the substrate lift) preserves the exact FS order.

---

## 12. Refactor sequence (each phase behavior-preserving, `test_e2e_*` green)

**Phase 1 — carve contract types. ✓ DONE.**
`Layout` (alias of `UairSignature`), `FamilyEvalClaims` (owned mirror of
`MultipointEvalFamilyInputs` + `as_inputs()` view), and the `ConstraintProof`
associated type — all in `protocol/src/constraint_system.rs`.

**Phase 2 — declare `ConstraintSystem` + `UairFrontend`. ✓ DONE (abstract surface).**
Trait defined; `UairFrontend` declared (no trait impl yet). Compiles, no behavior
change, verifier untouched.

**Phase 3 — implement the UAIR frontend + route the PROVER through it. ✓ DONE.**
`impl ConstraintSystem for UairFrontend` carries the **prover** constraint argument
(old steps 3–5 + the booleanity/`alpha'` bridge + scalar/bit-op projection), with the
projections internalized (decision 9 / Option A). The prover type-state machine was
restructured: the six intermediate states (`ProverProjected{Combined,MleFirst}`,
`FqProjStaging`, `ProverIdealChecked`, `ProverEvalProjected`, `ProverSumchecked`)
collapsed into one substrate `ProverProjected` (step 2 = `phi_q` projection only); steps
3–5 are gone; `prove()` calls `frontend.prove_constraints(...)`; step 6 consumes the
per-family `FamilyEvalClaims`; step 7/8 thread the opaque `UairConstraintProof`; `finish`
unpacks it into the **unchanged** `Proof<F>`. **Invariant held: `Proof<F>` is
byte-identical, so the current, unmodified verifier still accepts** — validated by the
full `test_e2e_*` + tamper + serialization-roundtrip suite (30 tests, green), clippy
clean.

  *Scope note:* `ZincPlusPiop<Zt, U, F, D, FD>` still pins `U`; `prove()` constructs
  `UairFrontend::<Zt, U, F, D, FD>::new(trace)` internally and routes through the
  `ConstraintSystem` trait. The struct is **not yet** generic over an arbitrary `CS`
  (dropping `U`) — that genericization is deferred to fold in with Phase 4, because
  changing the struct's generics ripples into the verifier's `impl` block, which we keep
  untouched until last.

**Phase 3.5 — move multipoint-eval into the frontend. ✓ DONE.**
The lockstep multipoint-eval (former substrate `step6_multipoint_eval`) is now the
closing move of `UairFrontend::prove_constraints`. The seam contract changed: the
frontend returns `(ConstraintProof, ConstraintEndpoints{r_0, r_0_fq})` instead of
`(ConstraintProof, Vec<FamilyEvalClaims>)`. `FamilyEvalClaims` moved into
`uair_frontend.rs` as a private helper (no longer a seam type); the
`MultipointEval` proofs moved into `UairConstraintProof`; the substrate's
`step_constraints_and_multipoint_eval` shrank to `step_constraints` (call the frontend,
unpack, thread `r_0` to step 7). The `MultipointEval` *algorithm* stays in `piop`.
**Transcript order is unchanged** (MP-eval still runs immediately after the `alpha'`
bridge squeeze), so `Proof<F>` is byte-identical and the unmodified verifier still
accepts — validated by the full suite (30 tests, green). Rationale: the substrate is now
purely "commit + project + bind a single point"; the frontend owns *everything that
reduces the relation to one witness-eval claim", which is exactly the R1CS-uniform shape
(R1CS/Spartan terminates at a single point with no MP-eval needed).

**Phase 4a — route the VERIFIER through the seam. ✓ DONE.**
`verify_constraints` (verifier steps 2–5: ideal-check verify, `psi_a`, CPR+booleanity
verify, MP-eval `verify_as_subprotocol` → `r_0`) + `verify_lifted_evals` (post-lift
`verify_subclaim` + bit-op/`alpha'` reconstruction) landed on `UairFrontend`. The
`reconstruct_virtual_evals` stub was **replaced** by `verify_lifted_evals` (see §13 /
decision 3, 11). The verifier type-state machine collapsed (mirror of Phase 3): steps
2–7 fold into `finish_verify`, which calls the two frontend methods around the retained
substrate lift-assembly + `pcs_verify`. New trait members: `type VerifierClaims`,
`type IdealSource`/`FqIdealSource`/`Scalar`, an `IdealOverF` method generic on
`verify_constraints` + the three projection closures as params. Verify-side frontend
built via `UairFrontend::new_verifier()` (empty witness cols + an unreachable stored
`project_scalar`). **Proof-preserving:** `Proof<F>`, `ZincPlusPiop<…, U, …>`, and the
whole prover are byte-for-byte untouched (verifier repacks the flat `Proof` into a
`&UairConstraintProof`); FS order preserved (`verify_subclaim` is transcript-free).
Validated by the full 30-test suite incl. **all tamper/soundness negatives** (green),
clippy clean.

  *Deviations to remember:* `benches/e2e.rs` collapsed its granular per-step verifier
  benches into "0/1" + one "2–7"; a stale rustdoc intra-link `[step7_pcs_verify]`
  remains in `prover.rs` (dangling, `cargo doc`-only, left per "don't touch prover");
  the tamper-replay tests keep a **duplicated** step-2/3 ordering in
  `#[cfg(test)] test_helpers`.

**Phase 4b — genericize + embed (LAST). NOT STARTED.** Genericize
`ZincPlusPiop<Zt, U, F, D, FD>` over `CS: ConstraintSystem` (drop the `U` pin) and embed
`UairConstraintProof` inside `Proof<F>` — retiring the flat `Proof` fields and the
repack/unpack on both sides, and exercising `UairConstraintProof`'s
`GenTranscribable`/`Transcribable` (currently written but unused). Both prove and verify
already route through the seam, so this is now purely a struct-generics + serialization
change (no algorithm change).

**Phase 5 — crate/module reorg (optional, can trail).**
Keep generic primitives in `piop` (sumcheck, mp_eval, shift_predicate, booleanity,
batched_ideal_check). Move UAIR-specific composition (CPR, `ConstraintFolder`,
`ideal_check` prove paths, `combined_poly_builder`, `collect_scalars`) next to
`UairFrontend`.

Regression harness throughout: the `test_e2e_*` + tamper/soundness suite
(`ShaProxy`, `BigLinearUair`, `TestUair*`, …) stays green after every phase. Since
Phases 3/3.5/4a are each proof-preserving, that suite (prove → verify, plus the
tamper *negatives* that assert rejection) is the strongest guarantee neither refactor
changed behaviour — the tamper cases specifically exercise the reworked verifier's
reject paths.

---

## 13. Risks / sharp edges

- **Virtual-column reconstruction. ✓ RESOLVED (Phase 4a).** Bit-op / booleanity virtual
  columns are frontend-defined and bound at `r_0`. The old `reconstruct_virtual_evals`
  stub was replaced by `verify_lifted_evals`, which does the full per-family
  `verify_subclaim` binding inside the frontend. It could *not* be folded into
  `verify_constraints` (the binding needs the substrate's lifted evals, which only exist
  after the substrate lifts at `r_0` — which `verify_constraints` produces): hence the
  **two-phase** verifier seam (`verify_constraints` → `r_0`+`VerifierClaims`; substrate
  lift; `verify_lifted_evals`). This is the one irreducible asymmetry vs the prover.
- **Shifts live in `Layout`** (= `UairSignature`), consumed by the frontend's own
  MP-eval shift predicate — now fully frontend-internal (the substrate's lift no longer
  reads them). Harmless for R1CS (empty list). A later cleanup could drop `shifts` from
  the shared `Layout` entirely.
- **`D`/`FD` const-generic threading** through trait methods — mechanical but noisy.
- **`Proof<F>` serialization** (`GenTranscribable`/`Transcribable`) must follow the
  `ConstraintProof` associated type cleanly; today it hand-serializes every field.
- **`q'' := q0` PCS optimization** (steps 7/8): the substrate aliases the PCS-only
  prime to the Q[X] prime when there are no F_q[X] families. It is a substrate-side
  PCS optimization keyed on `cs.layout().primes().len() == 0` (data the substrate
  already reads), so it survives the abstraction unchanged and cannot desync (the
  substrate both *builds* families from, and *counts* them via, `layout().primes()`).
  It is only ever an optimization, never required for soundness. Fix: name the
  decision (Open decision 6) rather than leaving a bare `n_fq == 0` that bakes in the
  UAIR family model.

---

## 14. Out of scope (explicit)

- No R1CS/Spartan implementation (only design-for).
- No N-typed-column witness abstraction (deferred; keep 3 groups).
- No lookup completion (it's already TODO/None in the proof).
- No zero-knowledge, no sparse-matrix commitment.

---

## 15. Open sub-decisions (recommendations; revise as you like)

1. **Names:** trait `ConstraintSystem`; adapter `UairFrontend<U, F, D, FD>`; shared
   layout `Layout` (alias of `UairSignature`). — *landed as written.*
2. **Where CPR/IdealCheck live after reorg:** move to the UAIR frontend (now
   UAIR-specific) vs leave in `piop`. — *rec: move, but only in Phase 5.*
3. **Verifier virtual-eval hook shape.** — *✓ resolved (Phase 4a): the
   `reconstruct_virtual_evals` stub was **replaced** by `verify_lifted_evals`, which owns
   the full per-family `verify_subclaim` binding (not just reconstructing evals). See
   decision 11 for why it's a separate post-lift method rather than folded into
   `verify_constraints`.*
4. **Keep `Uair` or fold into `ConstraintSystem`:** — *✓ keep both (authoring vs seam);
   R1CS bypasses `Uair`.*
5. **Const generics on trait vs frontend type:** — *✓ on the frontend type
   (`UairFrontend<U, F, D, FD>`).*
6. **`q''`-decoupling predicate.** — *✓ landed: `needs_decoupled_pcs_prime()` on the
   trait, default `!self.layout().primes().is_empty()`.*
7. **`Field` as associated type** (vs method generic). — *✓ resolved: associated type,
   so the impl supplies the full field-bound bundle (see §8 note 1).*
8. **`ConstraintProof` bound.** — *✓ resolved: `Transcribable` only; `GenTranscribable`
   is deserialization dedup, not part of the contract.*
9. **Seam input set / frontend statefulness. ✓ resolved: Option A** — the seam carries
   only field-projected witness data (`ProjectedTrace<F>` + cfgs); each frontend carries
   its own private state. Chosen as best (and for UAIR, the only viable) fit across all
   three target consumers:
   - **UAIR — forces it.** Raw `binary_poly` columns are `BinaryPoly<D>`-typed, and a
     const generic cannot appear in a trait-method signature (no `generic_const_exprs`),
     so they cannot cross the seam. `UairFrontend<U,F,D,FD>` (with `D` at the type level)
     holds them; the `U::Scalar`/`U::Ideal`-typed projections are internalized so they
     don't couple the trait. (Option B is impossible here.)
   - **R1CS — purely seam-fed.** Spartan runs on the field-projected witness from
     `projected_traces` plus the frontend's own matrices `A,B,C`; it never needs the raw
     witness, so Option A imposes nothing extra.
   - **ZK — additive.** Blinding columns are declared in the layout and committed by the
     substrate as ordinary witness, so masking sumchecks consume the same seam data;
     `#[non_exhaustive]` on `FamilyEvalClaims` absorbs any ZK metadata. No trait-method
     change.

   The landed trait already matches this (prove/verify take only `projected_traces`,
   `field_cfgs`, `num_vars`). Phase-3 refinement to consider: feed UAIR booleanity from
   the `phi_q`-projected trace (bits are 0/1 in `F_{q0}` too), which would drop even the
   UAIR raw-trace borrow and make the seam maximally clean.

10. **Multipoint-eval placement.** Substrate pipeline step vs frontend-invoked. — *✓
    resolved (Phase 3.5): **frontend-invoked**. MP-eval reduces the relation to a single
    witness-eval claim — constraint-argument work — and R1CS/Spartan needs no MP-eval at
    all (it terminates at a single point). The substrate becomes purely "commit +
    project + bind one point". The `MultipointEval` algorithm stays a reusable `piop`
    primitive; the seam now carries only `ConstraintEndpoints{r_0}`, and
    `FamilyEvalClaims` became frontend-internal scaffolding.*

11. **Verifier seam shape (the two-phase binding).** — *✓ resolved (Phase 4a).* The
    verifier's MP-eval verify splits (`verify_as_subprotocol` produces `r_0`;
    `verify_subclaim` binds it to the substrate's post-lift lifted evals), so the seam is
    **two frontend calls** around the substrate lift: `verify_constraints` →
    `(ConstraintEndpoints, VerifierClaims)`, then `verify_lifted_evals(claims,
    per_family_all_lifted, cfgs)`. `VerifierClaims` is an opaque associated type
    (the substrate never interprets it). The ideal/scalar projection closures are
    **method params** on `verify_constraints` (with an `IdealOverF` method generic +
    `IdealSource`/`FqIdealSource`/`Scalar` associated types to name the source types),
    mirroring the current `verify()` wrapper rather than being stored on the frontend —
    so the verify-side `UairFrontend::new_verifier()` needs no real scalar map (it stores
    an unreachable `project_scalar`, used only by the prove path). Alternatives
    considered & rejected: (a) fold `verify_subclaim` into `verify_constraints` —
    impossible, lifted evals don't exist yet; (b) keep `verify_subclaim` in the substrate
    with a returned `VerifierClaims` + a narrow `reconstruct_virtual_evals` hook — leaves
    MP-eval-verify split across the seam, less symmetric with the prover.

---

## 16. Glossary

- **UCS** — Universal Constraint System: the general relation (algebraic + ideal
  membership constraints over `Q[X]` / `F_{q_i}[X]`). **UAIR** — the AIR-like
  uniform-trace specialization implemented here. **R1CS** — rank-1 constraint system
  (future frontend).
- **family** — one constraint domain: `Q[X]` (prime `q0`) or a declared `F_{q_i}[X]`
  (`q_i` from `layout.primes()`). Total families = `1 + primes().len()`.
- **phi_q** — coefficient reduction `Z[X]/Q[X] -> F_q[X]` (random prime for `Q`).
  **psi_a** — evaluation at `X = a`, `F_q[X] -> F_q`. (Paper compiler steps 1 & 3.)
- **q0** — the random prime for the `Q[X]` family (sampled at step 2). **q_i** —
  declared primes for `F_q[X]` families. **q\*** — `min q_i`, the range for shared
  challenges. **q''** — the PCS-only opening prime (sampled at step 7, decoupled from
  constraint primes; aliased to `q0` when there are no `F_q[X]` families).
- **r_0** — the shared multipoint-eval endpoint (one integer point across families).
  **r\*** — `r_0 mod q''`, the PCS evaluation point.
- **MLE** — multilinear extension. **ideal-membership** — `Q(...) in (g)` (vs `= 0`),
  generalizing the standard zero check; reduced to an equality by sending the
  remainder and checking it lies in the ideal.
- **projected oracle** — a committed witness oracle that answers evaluation queries
  *under a projection* `phi_q` / `psi_a`.
- **CPR** — `CombinedPolyResolver`, the constraint-aggregation sumcheck (folds all
  per-row constraints by powers of a challenge). **IdealCheckProtocol** — the
  ideal-membership sub-protocol. **MultipointEval** — combines per-family up/down/bit-op
  evals into one `r_0`; a generic `piop` primitive, invoked by the frontend (Phase 3.5).
  Its verify splits: `verify_as_subprotocol` (→ `r_0`, in `verify_constraints`) +
  `verify_subclaim` (binds to lifted evals, in `verify_lifted_evals`).
- **`verify_constraints` / `verify_lifted_evals`** — the two-phase verifier seam
  (Phase 4a): the first re-derives `r_0` + opaque `VerifierClaims`; the second, after the
  substrate assembles per-family lifted evals, runs the per-family `verify_subclaim`
  binding (see decision 11).
- **lift-and-project** — recompute witness-column MLE evals at `r_0` per family
  (lifting the shared integer point into each field), used to discharge the eval
  claims via the PCS.
- **booleanity** — argument that witness `binary_poly` columns are 0/1; its
  `bit_slice_evals` are bridged into MP-eval via a fresh challenge `alpha'`.
- **bit-op virtual / shift** — uncommitted columns derived by an `R`-linear map
  (`Rot`/`ShR`) or a row shift; the verifier reconstructs their evals from committed
  columns (MLE commutes with `R`-linear maps).
- **Zip+ / IPRS** — the hash-based polynomial-commitment IOPP and the MDS code over
  `Q` it uses. **3-group witness layout** — `binary_poly` / `arbitrary_poly` / `int`,
  each committed separately; binary columns are folded (`Zt::BinaryFold`).
- **`Zt: ZincTypes`** — the type bundle (Int, Chal, Pt, Fmod, PrimeTest, the three
  `ZipTypes`, `BinaryFold`, the three `LinearCode`s). **`ZincPlusPiop<Zt, U, F, D, FD>`**
  — the protocol struct; `D = DEGREE_PLUS_ONE`, `FD` = folded degree+1.
