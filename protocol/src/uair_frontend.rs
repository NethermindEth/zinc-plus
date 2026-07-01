//! UAIR adapter for the
//! [`ConstraintSystem`](crate::constraint_system::ConstraintSystem) seam.
//!
//! [`UairFrontend`] bridges any `U: Uair` to the protocol-facing
//! [`ConstraintSystem`] trait by delegating to the existing UAIR constraint
//! engine (`CombinedPolyResolver` + `IdealCheckProtocol` + booleanity).
//!
//! **Prover side.** The body of `prove_constraints` is the constraint argument
//! that used to live in prover steps 3--5 (plus the step-4 scalar / bit-op
//! pieces and the step-6 booleanity-bridge + per-prime zero-padding) and the
//! lockstep multipoint-eval (former substrate step 6). The substrate keeps only
//! commit / prime projection / lift-and-project / PCS-open and feeds the
//! frontend the `phi_q`-projected per-family traces.
//!
//! **Verifier side (Phase 4).** [`ConstraintSystem::verify_constraints`]
//! relocates verifier steps 2--5 (ideal-check verify, `psi_a` scalar
//! projection, constraint sumcheck + booleanity verify, lockstep
//! multipoint-eval verify), returning the shared endpoint(s) plus the opaque
//! [`UairVerifierClaims`] tail-state.
//! [`ConstraintSystem::verify_lifted_evals`] then closes every per-family
//! multipoint-eval subclaim against the substrate-assembled lifted evals at
//! `r_0` (bit-op reconstruction + `alpha'` bridge / zero-pad).

use crate::{
    CombinedPolyResolverProof, IdealCheckError, IdealCheckProof, MultiDegreeSumcheckProof,
    ProtocolError, alpha_prime_bridge_up_evals,
    constraint_system::{ConstraintEndpoints, ConstraintSystem, Layout},
};
use core::marker::PhantomData;
use crypto_primitives::{FromPrimitiveWithConfig, HasPrimeFieldConfig, PrimeField};
use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use zinc_piop::{
    combined_poly_resolver::{CombinedPolyResolver, CombinedPolyResolverError},
    ideal_check::{self, IdealCheckProtocol},
    lookup::{
        BatchedLookupProof,
        booleanity::{BooleanityChecker, BooleanityProof},
    },
    multipoint_eval::{
        self, MultipointEval, MultipointEvalFamilyInputs, Proof as MultipointEvalProof,
    },
    projections::{
        ProjectedScalars, ProjectedTrace, build_bit_op_virtual_mle, evaluate_trace_to_column_mles,
        project_scalars, project_scalars_to_field,
    },
    sumcheck::multi_degree::{MultiDegreeSumcheck, MultiDegreeSumcheckGroup},
};
use zinc_poly::{
    EvaluatablePolynomial,
    mle::DenseMultilinearExtension,
    univariate::{binary::BinaryPoly, dynamic::over_field::DynamicPolynomialF},
};
use zinc_transcript::traits::{ConstTranscribable, GenTranscribable, Transcribable, Transcript};
use zinc_uair::{
    Uair, UairSignature, UairTrace,
    constraint_counter::count_constraints,
    degree_counter::count_max_degree,
    ideal::{Ideal, IdealCheck},
    ideal_collector::IdealOrZero,
};
use zinc_utils::{
    add, cfg_iter, from_ref::FromRef, inner_transparent_field::InnerTransparentField,
    mul_by_scalar::MulByScalar, powers,
};

/// The constraint-argument sub-proof produced by [`UairFrontend`].
///
/// Holds exactly the constraint-argument outputs that the substrate `Proof`
/// embeds today: the Q[X]-family ideal-check / CPR / multi-degree sumcheck,
/// the per-prime `F_{q_i}[X]` mirrors, and the optional booleanity / lookup
/// arguments.
///
/// The [`GenTranscribable`] / [`Transcribable`] impls serialize the full
/// constraint-argument bundle; the substrate [`Proof`](crate::Proof) embeds
/// this type as its `constraint_proof` field and delegates the sub-proof
/// bytes to these impls (via the trait's `ConstraintProof: Transcribable`
/// bound).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UairConstraintProof<F: PrimeField> {
    /// Randomized ideal check proof ($Q[X]$ constraint family).
    pub ideal_check: IdealCheckProof<F>,
    /// Per-prime $F_{q_i}[X]$ ideal-check proofs.
    pub ideal_checks_fq: Vec<IdealCheckProof<F>>,
    /// Combined polynomial resolver proof ($Q[X]$ constraint family).
    pub cpr_proof: CombinedPolyResolverProof<F>,
    /// Per-prime $F_{q_i}[X]$ CPR proofs.
    pub cpr_proofs_fq: Vec<CombinedPolyResolverProof<F>>,
    /// Multi-degree sumcheck proof ($Q[X]$ constraint family).
    pub combined_sumcheck: MultiDegreeSumcheckProof<F>,
    /// Per-prime $F_{q_i}[X]$ multi-degree sumcheck proofs.
    pub combined_sumchecks_fq: Vec<MultiDegreeSumcheckProof<F>>,
    /// Multipoint-eval proof ($Q[X]$ constraint family): collapses the
    /// per-column / shift / bit-op-virtual claims at `r*` into one claim at the
    /// shared endpoint `r_0`.
    pub multipoint_eval: MultipointEvalProof<F>,
    /// Per-prime $F_{q_i}[X]$ multipoint-eval proofs, produced by the same
    /// lockstep reduction.
    pub multipoint_evals_fq: Vec<MultipointEvalProof<F>>,
    /// Binary-polynomial booleanity argument proof. `None` when the UAIR has
    /// no witness binary-poly columns.
    pub booleanity_proof: Option<BooleanityProof<F>>,
    /// Lookup argument proof (`None`; lookup is not yet implemented).
    pub lookup_proof: Option<BatchedLookupProof<F>>,
}

impl<F> GenTranscribable for UairConstraintProof<F>
where
    F: PrimeField,
    F::Integer: ConstTranscribable,
{
    fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
        let (ideal_check, bytes) = IdealCheckProof::<F>::read_transcription_bytes_subset(bytes);
        let (cpr_proof, bytes) =
            CombinedPolyResolverProof::<F>::read_transcription_bytes_subset(bytes);
        let (combined_sumcheck, bytes) =
            MultiDegreeSumcheckProof::<F>::read_transcription_bytes_subset(bytes);
        let (multipoint_eval, bytes) =
            MultipointEvalProof::<F>::read_transcription_bytes_subset(bytes);

        // booleanity_proof: presence flag (u32) + optional length-prefixed body.
        let (presence, bytes) = u32::read_transcription_bytes_subset(bytes);
        let (booleanity_proof, bytes) = if presence != 0 {
            let (bp, rest) = BooleanityProof::<F>::read_transcription_bytes_subset(bytes);
            (Some(bp), rest)
        } else {
            (None, bytes)
        };

        // ideal_checks_fq: u32 count + length-prefixed entries.
        let (n_ic, mut bytes) = u32::read_transcription_bytes_subset(bytes);
        let n_ic = usize::try_from(n_ic).expect("fits usize");
        let mut ideal_checks_fq = Vec::with_capacity(n_ic);
        for _ in 0..n_ic {
            let (ic, rest) = IdealCheckProof::<F>::read_transcription_bytes_subset(bytes);
            ideal_checks_fq.push(ic);
            bytes = rest;
        }

        // cpr_proofs_fq: u32 count + length-prefixed entries.
        let (n_cpr_fq, mut bytes) = u32::read_transcription_bytes_subset(bytes);
        let n_cpr_fq = usize::try_from(n_cpr_fq).expect("n_cpr_fq must fit into usize");
        let mut cpr_proofs_fq = Vec::with_capacity(n_cpr_fq);
        for _ in 0..n_cpr_fq {
            let (cpr, rest) =
                CombinedPolyResolverProof::<F>::read_transcription_bytes_subset(bytes);
            cpr_proofs_fq.push(cpr);
            bytes = rest;
        }

        // combined_sumchecks_fq: u32 count + length-prefixed entries.
        let (n_sum_fq, mut bytes) = u32::read_transcription_bytes_subset(bytes);
        let n_sum_fq = usize::try_from(n_sum_fq).expect("n_sum_fq must fit into usize");
        let mut combined_sumchecks_fq = Vec::with_capacity(n_sum_fq);
        for _ in 0..n_sum_fq {
            let (sumcheck, rest) =
                MultiDegreeSumcheckProof::<F>::read_transcription_bytes_subset(bytes);
            combined_sumchecks_fq.push(sumcheck);
            bytes = rest;
        }

        // multipoint_evals_fq: u32 count + length-prefixed entries.
        let (n_mp_fq, mut bytes) = u32::read_transcription_bytes_subset(bytes);
        let n_mp_fq = usize::try_from(n_mp_fq).expect("n_mp_fq must fit into usize");
        let mut multipoint_evals_fq = Vec::with_capacity(n_mp_fq);
        for _ in 0..n_mp_fq {
            let (mp, rest) = MultipointEvalProof::<F>::read_transcription_bytes_subset(bytes);
            multipoint_evals_fq.push(mp);
            bytes = rest;
        }

        assert!(bytes.is_empty(), "All bytes should be consumed");
        Self {
            ideal_check,
            ideal_checks_fq,
            cpr_proof,
            cpr_proofs_fq,
            combined_sumcheck,
            combined_sumchecks_fq,
            multipoint_eval,
            multipoint_evals_fq,
            booleanity_proof,
            lookup_proof: None,
        }
    }

    fn write_transcription_bytes_exact(&self, mut buf: &mut [u8]) {
        buf = self.ideal_check.write_transcription_bytes_subset(buf);
        buf = self.cpr_proof.write_transcription_bytes_subset(buf);
        buf = self.combined_sumcheck.write_transcription_bytes_subset(buf);
        buf = self.multipoint_eval.write_transcription_bytes_subset(buf);

        let presence = u32::from(self.booleanity_proof.is_some());
        buf = presence.write_transcription_bytes_subset(buf);
        if let Some(ref bp) = self.booleanity_proof {
            buf = bp.write_transcription_bytes_subset(buf);
        }

        let n_ic = u32::try_from(self.ideal_checks_fq.len()).expect("fits u32");
        buf = n_ic.write_transcription_bytes_subset(buf);
        for ic in &self.ideal_checks_fq {
            buf = ic.write_transcription_bytes_subset(buf);
        }

        let n_cpr = u32::try_from(self.cpr_proofs_fq.len()).expect("fits u32");
        buf = n_cpr.write_transcription_bytes_subset(buf);
        for cpr in &self.cpr_proofs_fq {
            buf = cpr.write_transcription_bytes_subset(buf);
        }

        let n_sc = u32::try_from(self.combined_sumchecks_fq.len()).expect("fits u32");
        buf = n_sc.write_transcription_bytes_subset(buf);
        for sc in &self.combined_sumchecks_fq {
            buf = sc.write_transcription_bytes_subset(buf);
        }

        let n_mp = u32::try_from(self.multipoint_evals_fq.len()).expect("fits u32");
        buf = n_mp.write_transcription_bytes_subset(buf);
        for mp in &self.multipoint_evals_fq {
            buf = mp.write_transcription_bytes_subset(buf);
        }
        let _ = buf;
    }
}

impl<F> Transcribable for UairConstraintProof<F>
where
    F: PrimeField,
    F::Integer: ConstTranscribable,
{
    #[allow(clippy::arithmetic_side_effects)]
    fn get_num_bytes(&self) -> usize {
        let booleanity_bytes = match &self.booleanity_proof {
            Some(bp) => BooleanityProof::<F>::LENGTH_NUM_BYTES + bp.get_num_bytes(),
            None => 0,
        };
        let ideal_checks_fq_bytes: usize = self
            .ideal_checks_fq
            .iter()
            .map(|ic| IdealCheckProof::<F>::LENGTH_NUM_BYTES + ic.get_num_bytes())
            .sum();
        let cpr_proofs_fq_bytes: usize = self
            .cpr_proofs_fq
            .iter()
            .map(|cpr| CombinedPolyResolverProof::<F>::LENGTH_NUM_BYTES + cpr.get_num_bytes())
            .sum();
        let combined_sumchecks_fq_bytes: usize = self
            .combined_sumchecks_fq
            .iter()
            .map(|sc| MultiDegreeSumcheckProof::<F>::LENGTH_NUM_BYTES + sc.get_num_bytes())
            .sum();
        let multipoint_evals_fq_bytes: usize = self
            .multipoint_evals_fq
            .iter()
            .map(|mp| MultipointEvalProof::<F>::LENGTH_NUM_BYTES + mp.get_num_bytes())
            .sum();
        IdealCheckProof::<F>::LENGTH_NUM_BYTES
            + self.ideal_check.get_num_bytes()
            + CombinedPolyResolverProof::<F>::LENGTH_NUM_BYTES
            + self.cpr_proof.get_num_bytes()
            + MultiDegreeSumcheckProof::<F>::LENGTH_NUM_BYTES
            + self.combined_sumcheck.get_num_bytes()
            + MultipointEvalProof::<F>::LENGTH_NUM_BYTES
            + self.multipoint_eval.get_num_bytes()
            // ideal_checks_fq: count + sum of (length-prefix + body) per entry
            + u32::NUM_BYTES
            + ideal_checks_fq_bytes
            // cpr_proofs_fq: count + sum of (length-prefix + body) per entry
            + u32::NUM_BYTES
            + cpr_proofs_fq_bytes
            // combined_sumchecks_fq: count + sum of (length-prefix + body) per entry
            + u32::NUM_BYTES
            + combined_sumchecks_fq_bytes
            // booleanity presence flag + optional payload
            + u32::NUM_BYTES
            + booleanity_bytes
            // multipoint_evals_fq: count + sum of (length-prefix + body) per entry
            + u32::NUM_BYTES
            + multipoint_evals_fq_bytes
    }
}

/// Opaque frontend verifier tail-state produced by
/// [`UairFrontend::verify_constraints`] and consumed by
/// [`UairFrontend::verify_lifted_evals`].
///
/// Carries the multipoint-eval subclaims (the full
/// [`multipoint_eval::Subclaim`] per family — `verify_subclaim` needs more than
/// just `r_0`, which lives in [`ConstraintEndpoints`]), the per-family `psi_a`
/// projecting elements, and the booleanity `alpha'` challenge (`None` when
/// there are no witness binary-poly columns).
pub struct UairVerifierClaims<F: PrimeField> {
    /// Q[X]-family multipoint-eval subclaim.
    mp_subclaim: multipoint_eval::Subclaim<F>,
    /// Per-prime $F_{q_i}[X]$ multipoint-eval subclaims (in `primes()` order).
    mp_subclaims_fq: Vec<multipoint_eval::Subclaim<F>>,
    /// Per-family $\psi$-projecting elements (family order, `[0] = Q[X]`).
    projecting_elements: Vec<F>,
    /// Booleanity `alpha'` bridge challenge (`None` iff no witness binary-poly
    /// columns).
    alpha_prime_f: Option<F>,
}

/// Adapter wrapping a `U: Uair` so it can serve as a
/// [`ConstraintSystem`](crate::constraint_system::ConstraintSystem).
///
/// Type parameters:
/// - `U` — the UAIR description (authoring trait).
/// - `F` — the projection field (becomes `ConstraintSystem::Field`); fixed on
///   the type so the impl can require the full field-bound bundle.
/// - `D` (= `DEGREE_PLUS_ONE`) / `FD` (folded degree+1) — const generics, on
///   the type per decision 5 of `docs/ucs-refactor-plan.md`.
///
/// It borrows the original (un-projected) witness binary-poly columns from the
/// trace — needed for the booleanity argument and the `alpha'` booleanity
/// bridge — and carries the `project_scalar` projection map, the full
/// [`UairSignature`] (for the frontend's own `shifts()` / `bit_op_specs()` /
/// column access), and the minimal substrate-only [`Layout`] (so
/// [`ConstraintSystem::layout`] can hand back a borrow that carries no
/// frontend-internal specs).
pub struct UairFrontend<'a, U: Uair, F: PrimeField, const D: usize, const FD: usize> {
    /// The original (un-projected) witness binary-poly columns, borrowed from
    /// the trace. Used for booleanity + the `alpha'` bridge.
    witness_binary_cols: &'a [DenseMultilinearExtension<BinaryPoly<D>>],
    /// Projection map `phi_q` for UAIR scalars (`U::Scalar -> F[X]` under a
    /// per-family field cfg). Internalized here so the seam stays
    /// constraint-system-agnostic.
    project_scalar: fn(&U::Scalar, &<F as HasPrimeFieldConfig>::Config) -> DynamicPolynomialF<F>,
    /// The full UAIR signature, kept for the frontend's own use (`shifts()`,
    /// `bit_op_specs()`, column layouts) — data the substrate never reads.
    signature: UairSignature<U::Prime>,
    /// The minimal substrate-only layout, derived from `signature`, returned by
    /// [`ConstraintSystem::layout`].
    layout: Layout<U::Prime>,
    _marker: PhantomData<F>,
}

// Manual `Clone` / `Debug` bounding only the concretely-stored types (never
// `U` or `F` themselves): the borrowed slice is `Copy`, the fn pointer is
// `Copy`, and the signature is `Clone`/`Debug` since `U::Prime: Semiring`.
// This lets the prover's substrate type-states derive `Clone`/`Debug` with a
// plain `CS: Clone`/`Debug` bound.
impl<'a, U: Uair, F: PrimeField, const D: usize, const FD: usize> Clone
    for UairFrontend<'a, U, F, D, FD>
{
    fn clone(&self) -> Self {
        Self {
            witness_binary_cols: self.witness_binary_cols,
            project_scalar: self.project_scalar,
            signature: self.signature.clone(),
            layout: self.layout.clone(),
            _marker: PhantomData,
        }
    }
}

impl<'a, U: Uair, F: PrimeField, const D: usize, const FD: usize> core::fmt::Debug
    for UairFrontend<'a, U, F, D, FD>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("UairFrontend")
            .field("signature", &self.signature)
            .finish_non_exhaustive()
    }
}

impl<'a, U: Uair, F: PrimeField, const D: usize, const FD: usize> UairFrontend<'a, U, F, D, FD> {
    /// Build a UAIR frontend.
    ///
    /// `witness_binary_cols` is the original (un-projected) witness slice of
    /// the trace's binary-poly columns (i.e.
    /// `&trace.binary_poly[num_pub_bin..num_total_bin]`).
    pub fn new(
        witness_binary_cols: &'a [DenseMultilinearExtension<BinaryPoly<D>>],
        project_scalar: fn(
            &U::Scalar,
            &<F as HasPrimeFieldConfig>::Config,
        ) -> DynamicPolynomialF<F>,
    ) -> Self {
        let signature = U::signature();
        let layout = Layout::from_signature(&signature);
        Self {
            witness_binary_cols,
            project_scalar,
            signature,
            layout,
            _marker: PhantomData,
        }
    }

    /// Build a prove-side UAIR frontend directly from the caller's trace.
    ///
    /// Slices the original (un-projected) witness binary-poly columns out of
    /// `trace` — `&trace.binary_poly[num_pub_bin..num_total_bin]`, using
    /// [`U::signature`](Uair::signature) for the column counts — and forwards
    /// to [`new`](Self::new). This lets callers build the prove-side
    /// frontend without hand-slicing.
    ///
    /// Generic over the trace's coefficient/int types (they never surface in
    /// [`UairFrontend`], which only borrows the binary-poly columns).
    pub fn from_trace<'t, PolyCoeff: Clone, Int: Clone>(
        trace: &'t UairTrace<'static, PolyCoeff, Int, D, D>,
        project_scalar: fn(
            &U::Scalar,
            &<F as HasPrimeFieldConfig>::Config,
        ) -> DynamicPolynomialF<F>,
    ) -> UairFrontend<'t, U, F, D, FD> {
        let sig = U::signature();
        let num_pub_bin = sig.public_cols().num_binary_poly_cols();
        let num_total_bin = sig.total_cols().num_binary_poly_cols();
        let witness_binary_cols = &trace.binary_poly[num_pub_bin..num_total_bin];
        UairFrontend::new(witness_binary_cols, project_scalar)
    }

    /// Build a verify-side UAIR frontend.
    ///
    /// The verifier has no witness columns (they are committed, not held in the
    /// clear) and receives its scalar projection as a `verify_constraints`
    /// parameter rather than through the stored fn ptr — so this construction
    /// uses an empty witness slice and an intentionally-unreachable stored
    /// `project_scalar`. This keeps the prover's [`new`](Self::new) and
    /// `prove_constraints` untouched (the stored fn ptr is never invoked on the
    /// verify path).
    pub fn new_verifier() -> Self {
        fn unreachable_project_scalar<U: Uair, F: PrimeField>(
            _scalar: &U::Scalar,
            _cfg: &<F as HasPrimeFieldConfig>::Config,
        ) -> DynamicPolynomialF<F> {
            unreachable!("verify-side UairFrontend never invokes the stored project_scalar")
        }
        let signature = U::signature();
        let layout = Layout::from_signature(&signature);
        Self {
            witness_binary_cols: &[],
            project_scalar: unreachable_project_scalar::<U, F>,
            signature,
            layout,
            _marker: PhantomData,
        }
    }

    /// The frontend's full UAIR signature — including the UAIR-specific
    /// `shifts` / `bit_op_specs` / `lookup_specs` / `down_cols` /
    /// `affine_virtual_specs` that are deliberately **not** exposed on the
    /// generic seam ([`ConstraintSystem::layout`] returns the minimal
    /// substrate-facing [`Layout`], which carries only column layouts +
    /// primes).
    ///
    /// UAIR-side consumers reach those specs through this concrete accessor
    /// rather than the generic seam, keeping the seam R1CS-agnostic while the
    /// specs remain UAIR-only.
    pub fn signature(&self) -> &UairSignature<U::Prime> {
        &self.signature
    }
}

impl<'a, U, F, const D: usize, const FD: usize> ConstraintSystem for UairFrontend<'a, U, F, D, FD>
where
    U: Uair + 'static,
    F: InnerTransparentField
        + FromPrimitiveWithConfig
        + for<'b> MulByScalar<&'b F>
        + FromRef<F>
        + Send
        + Sync
        + 'static,
    F::Integer: ConstTranscribable + Ord + Zero + Default + Send + Sync,
{
    type Prime = U::Prime;
    type Field = F;
    type ConstraintProof = UairConstraintProof<F>;
    type VerifierClaims = UairVerifierClaims<F>;
    type IdealSource = IdealOrZero<U::Ideal>;
    type FqIdealSource = IdealOrZero<U::FqIdeal>;
    type Scalar = U::Scalar;

    fn layout(&self) -> &Layout<Self::Prime> {
        &self.layout
    }

    /// Prover side of the UAIR constraint argument.
    ///
    /// Faithfully relocates prover steps 3--6: the constraint argument proper,
    /// the step-6 booleanity-bridge / per-prime zero-padding, and the lockstep
    /// multipoint-eval itself (former substrate step 6). Transcript
    /// squeeze/absorb order is preserved verbatim relative to the original step
    /// chain:
    ///
    /// 1. shared ideal-check eval points (`mu` per family), per-family ideal
    ///    check (Q then each prime);
    /// 2. shared `psi_a` projecting element `a`;
    /// 3. shared CPR folding challenge `alpha`, booleanity `prepare`, lockstep
    ///    multi-degree sumcheck, per-family CPR/booleanity finalize, then the
    ///    `alpha'` booleanity-bridge squeeze;
    /// 4. assemble the per-family eval claims (Q-family with appended `alpha'`
    ///    bridge columns/up-evals, per-prime zero-padded);
    /// 5. lockstep multipoint-eval over those claims, reducing them to the
    ///    shared endpoint `r_0` returned as [`ConstraintEndpoints`].
    #[allow(clippy::too_many_lines, clippy::arithmetic_side_effects)]
    fn prove_constraints(
        &self,
        transcript: &mut impl Transcript,
        projected_traces: &[ProjectedTrace<Self::Field>],
        field_cfgs: &[<Self::Field as HasPrimeFieldConfig>::Config],
        num_vars: usize,
    ) -> Result<(Self::ConstraintProof, ConstraintEndpoints<Self::Field>), ProtocolError<Self::Field>>
    {
        let sig = &self.signature;
        let num_constraints = count_constraints::<U>();
        let max_degree = count_max_degree::<U>();
        let n_fq = projected_traces.len().saturating_sub(1);

        let field_cfg = field_cfgs[0].clone();
        let q_star_idx = crate::shared_challenge::compute_q_star_idx::<F>(field_cfgs);
        let q_star_cfg = field_cfgs[q_star_idx].clone();

        // ---- per-family scalar projections (phi_q) -------------------------
        // The substrate no longer projects scalars; the frontend recomputes
        // them per family from its own `project_scalar` map.
        let projected_scalars_fx: Vec<ProjectedScalars<U::Scalar, DynamicPolynomialF<F>>> =
            field_cfgs
                .iter()
                .map(|cfg| project_scalars::<F, U>(|s| (self.project_scalar)(s, cfg)))
                .collect();

        // =================== step 3: ideal check ===========================
        let shared_eval_points: Vec<Vec<F>> =
            crate::shared_challenge::sample_shared_field_challenges::<F>(
                transcript,
                num_vars,
                &q_star_cfg,
                field_cfgs,
            );

        let ideal_check = prove_ideal_family::<U, F, _, D>(
            transcript,
            &projected_traces[0],
            &projected_scalars_fx[0],
            0,
            num_constraints.q,
            &shared_eval_points[0],
            &field_cfg,
        )?;

        let mut ideal_checks_fq: Vec<IdealCheckProof<F>> = Vec::with_capacity(n_fq);
        for prime_idx in 0..n_fq {
            let family_idx = add!(prime_idx, 1);
            let cfg_q_i = &field_cfgs[family_idx];
            let ic_proof_i = prove_ideal_family::<U, F, _, D>(
                transcript,
                &projected_traces[family_idx],
                &projected_scalars_fx[family_idx],
                family_idx,
                num_constraints.for_prime(prime_idx),
                &shared_eval_points[family_idx],
                cfg_q_i,
            )
            .map_err(|source| ProtocolError::FqIdealCheck {
                prime_idx,
                q: F::modulus(cfg_q_i).to_string(),
                source,
            })?;
            ideal_checks_fq.push(ic_proof_i);
        }

        // =================== step 4: eval projection (psi_a) ================
        let projecting_elements: Vec<F> = crate::shared_challenge::sample_shared_field_challenge::<F>(
            transcript,
            &q_star_cfg,
            field_cfgs,
        );

        let bit_op_specs = sig.bit_op_specs().to_vec();

        // Per-family psi_a-projected trace MLEs, bit-op virtual MLEs, and
        // psi_a-projected scalars.
        let mut projected_trace_f: Vec<Vec<DenseMultilinearExtension<F::Inner>>> =
            Vec::with_capacity(projected_traces.len());
        let mut bit_op_mles_per_family: Vec<Vec<DenseMultilinearExtension<F::Inner>>> =
            Vec::with_capacity(projected_traces.len());
        let mut projected_scalars_f: Vec<ProjectedScalars<U::Scalar, F>> =
            Vec::with_capacity(projected_traces.len());

        for (family_idx, trace_i) in projected_traces.iter().enumerate() {
            let cfg_i = &field_cfgs[family_idx];
            let elem_i = &projecting_elements[family_idx];

            let trace_f_i = evaluate_trace_to_column_mles(trace_i, elem_i);
            let bit_op_mles_i = bit_op_specs
                .iter()
                .map(|spec| build_bit_op_virtual_mle::<F, D>(trace_i, spec, elem_i, cfg_i))
                .collect();
            let scalars_f_i =
                project_scalars_to_field(projected_scalars_fx[family_idx].clone(), elem_i)
                    .map_err(|(_s, _f, e)| ProtocolError::ScalarProjection(e))?;

            projected_trace_f.push(trace_f_i);
            bit_op_mles_per_family.push(bit_op_mles_i);
            projected_scalars_f.push(scalars_f_i);
        }

        // =================== step 5: sumcheck ===============================
        let folding_challenges: Vec<F> = crate::shared_challenge::sample_shared_field_challenge::<F>(
            transcript,
            &q_star_cfg,
            field_cfgs,
        );

        // ---- Q[X] family groups: CPR + optional booleanity ----
        let (q_cpr_group, q_cpr_ancillary) = CombinedPolyResolver::prepare_sumcheck_group::<U>(
            projected_trace_f[0].clone(),
            bit_op_mles_per_family[0].clone(),
            &shared_eval_points[0],
            &projected_scalars_f[0],
            0,
            num_constraints.q,
            num_vars,
            max_degree,
            &folding_challenges[0],
            &field_cfg,
        )?;

        let mut q_groups = vec![q_cpr_group];

        let trace_wit_bin_poly = self.witness_binary_cols;
        let bool_ancillary = if !trace_wit_bin_poly.is_empty() {
            let (bool_group, anc) = BooleanityChecker::prepare_sumcheck_group::<D>(
                transcript,
                trace_wit_bin_poly,
                num_vars,
                &field_cfg,
            )
            .map_err(ProtocolError::Booleanity)?;
            q_groups.push(bool_group);
            Some(anc)
        } else {
            None
        };

        // ---- Per-prime F_q[X] family groups: one CPR each ----
        let mut fq_cpr_ancillaries = Vec::with_capacity(n_fq);
        let mut fq_family_groups: Vec<Vec<MultiDegreeSumcheckGroup<F>>> = Vec::with_capacity(n_fq);
        for prime_idx in 0..n_fq {
            let family_idx = add!(prime_idx, 1);
            let cfg_i = &field_cfgs[family_idx];
            let (cpr_group_i, cpr_ancillary_i) = CombinedPolyResolver::prepare_sumcheck_group::<U>(
                projected_trace_f[family_idx].clone(),
                bit_op_mles_per_family[family_idx].clone(),
                &shared_eval_points[family_idx],
                &projected_scalars_f[family_idx],
                family_idx,
                num_constraints.for_prime(prime_idx),
                num_vars,
                max_degree,
                &folding_challenges[family_idx],
                cfg_i,
            )?;
            fq_family_groups.push(vec![cpr_group_i]);
            fq_cpr_ancillaries.push(cpr_ancillary_i);
        }

        // ---- Lockstep multi-degree sumcheck ----
        let mut md_sc_families: Vec<(Vec<MultiDegreeSumcheckGroup<F>>, &F::Config)> =
            Vec::with_capacity(add!(n_fq, 1));
        md_sc_families.push((q_groups, &field_cfg));
        for (prime_idx, groups) in fq_family_groups.into_iter().enumerate() {
            let family_idx = add!(prime_idx, 1);
            md_sc_families.push((groups, &field_cfgs[family_idx]));
        }

        let mut sumcheck_outputs = MultiDegreeSumcheck::prove_as_subprotocol(
            transcript,
            md_sc_families,
            num_vars,
            &q_star_cfg,
        )
        .into_iter();

        // ---- Q[X] family finalize ----
        let (combined_sumcheck, md_states) =
            sumcheck_outputs.next().expect("Q[X] family always present");
        let mut md_iter = md_states.into_iter();

        let (cpr_proof, cpr_prover_state) = CombinedPolyResolver::finalize_prover::<U>(
            transcript,
            md_iter.next().expect("CPR group always present"),
            q_cpr_ancillary,
            &field_cfg,
        )?;

        let booleanity_proof = if let Some(anc) = bool_ancillary {
            let state = md_iter.next().expect("booleanity group present");
            Some(
                BooleanityChecker::finalize_prover(transcript, state, anc, &field_cfg)
                    .map_err(ProtocolError::Booleanity)?,
            )
        } else {
            None
        };

        // TODO: build BatchedLookupProof from collected lookup proofs + metas.
        let lookup_proof = None;

        // ---- Per-prime family finalize ----
        let mut cpr_proofs_fq: Vec<CombinedPolyResolverProof<F>> = Vec::with_capacity(n_fq);
        let mut cpr_eval_points_fq: Vec<Vec<F>> = Vec::with_capacity(n_fq);
        let mut combined_sumchecks_fq: Vec<MultiDegreeSumcheckProof<F>> = Vec::with_capacity(n_fq);
        for (prime_idx, cpr_ancillary_i) in fq_cpr_ancillaries.into_iter().enumerate() {
            let family_idx = add!(prime_idx, 1);
            let cfg_i = &field_cfgs[family_idx];
            let (sumcheck_i, states_i) =
                sumcheck_outputs.next().expect("fq family sumcheck output");
            let mut states_iter_i = states_i.into_iter();
            let (cpr_proof_i, cpr_state_i) = CombinedPolyResolver::finalize_prover::<U>(
                transcript,
                states_iter_i.next().expect("CPR group always present"),
                cpr_ancillary_i,
                cfg_i,
            )?;
            combined_sumchecks_fq.push(sumcheck_i);
            cpr_proofs_fq.push(cpr_proof_i);
            cpr_eval_points_fq.push(cpr_state_i.evaluation_point);
        }

        // ---- booleanity -> multipoint-eval `alpha'` bridge squeeze ----
        let alpha_prime_f: Option<F> = booleanity_proof
            .as_ref()
            .map(|_| transcript.get_field_challenge(&field_cfg));

        // ============ step 6 (frontend part): family eval claims ============
        // Q[X] family: append the alpha'-bridge columns/up-evals when booleanity
        // ran; per-prime families: zero-pad to match the Q-family column count.
        let mut q_trace_mles = projected_trace_f[0].clone();
        let (q_up_evals, num_wit_bin) = if let Some(alpha_prime) = &alpha_prime_f {
            let num_wit_bin = trace_wit_bin_poly.len();
            let one = F::one_with_cfg(&field_cfg);
            let alpha_powers: Vec<F> = powers(alpha_prime.clone(), one, D);
            let extra_trace_mles: Vec<DenseMultilinearExtension<F::Inner>> =
                cfg_iter!(trace_wit_bin_poly)
                    .map(|col| project_binary_col_at_field::<F, D>(col, &alpha_powers, &field_cfg))
                    .collect();
            debug_assert_eq!(extra_trace_mles.len(), num_wit_bin);

            let bp = booleanity_proof
                .as_ref()
                .expect("booleanity_proof present iff alpha_prime_f is Some");
            let extra_up_evals = alpha_prime_bridge_up_evals::<F, D>(
                &bp.bit_slice_evals,
                num_wit_bin,
                alpha_prime,
                &field_cfg,
            );

            q_trace_mles.extend(extra_trace_mles);
            let mut up_evals = cpr_proof.up_evals.clone();
            up_evals.extend(extra_up_evals);
            (up_evals, num_wit_bin)
        } else {
            (cpr_proof.up_evals.clone(), 0)
        };

        let extension_size = 1usize << num_vars;
        let mut claims: Vec<FamilyEvalClaims<F>> = Vec::with_capacity(add!(n_fq, 1));
        claims.push(FamilyEvalClaims::new(
            field_cfg.clone(),
            q_trace_mles,
            bit_op_mles_per_family[0].clone(),
            cpr_prover_state.evaluation_point.clone(),
            q_up_evals,
            cpr_proof.bit_op_evals.clone(),
            cpr_proof.down_evals.clone(),
        ));

        for prime_idx in 0..n_fq {
            let family_idx = add!(prime_idx, 1);
            let cfg_i = &field_cfgs[family_idx];
            let zero_i = F::zero_with_cfg(cfg_i);
            let zero_inner_i = zero_i.inner();

            let mut trace_i = projected_trace_f[family_idx].clone();
            if num_wit_bin > 0 {
                let zero_mle = DenseMultilinearExtension::from_evaluations_vec(
                    num_vars,
                    vec![zero_inner_i.clone(); extension_size],
                    zero_inner_i.clone(),
                );
                trace_i.extend((0..num_wit_bin).map(|_| zero_mle.clone()));
            }

            let mut up_i = cpr_proofs_fq[prime_idx].up_evals.clone();
            up_i.extend((0..num_wit_bin).map(|_| zero_i.clone()));

            claims.push(FamilyEvalClaims::new(
                cfg_i.clone(),
                trace_i,
                bit_op_mles_per_family[family_idx].clone(),
                cpr_eval_points_fq[prime_idx].clone(),
                up_i,
                cpr_proofs_fq[prime_idx].bit_op_evals.clone(),
                cpr_proofs_fq[prime_idx].down_evals.clone(),
            ));
        }

        // ============ step 6: lockstep multipoint-eval ============
        // Reduce every per-family up/down/bit-op evaluation claim (at the
        // per-family points `r*`) to the single shared endpoint `r_0`. This is
        // the former substrate step 6, now the closing move of the constraint
        // argument; the substrate consumes only the returned `r_0` for its
        // lift-and-project + PCS open. Transcript order is unchanged: the
        // reduction runs immediately after the `alpha'` bridge squeeze, exactly
        // as it did when the substrate invoked it after `prove_constraints`.
        let shifts = sig.shifts();
        let mp_inputs: Vec<MultipointEvalFamilyInputs<'_, F>> =
            claims.iter().map(FamilyEvalClaims::as_inputs).collect();

        let mut mp_outputs =
            MultipointEval::prove_as_subprotocol(transcript, mp_inputs, shifts, &q_star_cfg)?
                .into_iter();

        let (multipoint_eval, q_state) = mp_outputs.next().expect("Q-family present");
        let r_0 = q_state.eval_point;

        let mut multipoint_evals_fq: Vec<MultipointEvalProof<F>> = Vec::with_capacity(n_fq);
        let mut r_0_fq: Vec<Vec<F>> = Vec::with_capacity(n_fq);
        for (proof_i, state_i) in mp_outputs {
            multipoint_evals_fq.push(proof_i);
            r_0_fq.push(state_i.eval_point);
        }

        let proof = UairConstraintProof {
            ideal_check,
            cpr_proof,
            combined_sumcheck,
            multipoint_eval,
            ideal_checks_fq,
            cpr_proofs_fq,
            combined_sumchecks_fq,
            multipoint_evals_fq,
            booleanity_proof,
            lookup_proof,
        };

        Ok((proof, ConstraintEndpoints::new(r_0, r_0_fq)))
    }

    /// Verifier side of the UAIR constraint argument.
    ///
    /// Faithfully relocates verifier steps 2--5 (ideal-check verify, `psi_a`
    /// scalar projection, constraint sumcheck + booleanity verify, lockstep
    /// multipoint-eval verify), preserving the transcript squeeze/absorb order
    /// verbatim relative to the original verifier step chain. Returns the
    /// shared endpoint(s) `r_0` plus the opaque [`UairVerifierClaims`]
    /// tail-state that [`verify_lifted_evals`](Self::verify_lifted_evals)
    /// consumes.
    #[allow(clippy::too_many_lines, clippy::arithmetic_side_effects)]
    fn verify_constraints<IdealOverF>(
        &self,
        transcript: &mut impl Transcript,
        proof: &Self::ConstraintProof,
        field_cfgs: &[<Self::Field as HasPrimeFieldConfig>::Config],
        num_vars: usize,
        project_ideal: impl Fn(
            &Self::IdealSource,
            &<Self::Field as HasPrimeFieldConfig>::Config,
        ) -> IdealOverF,
        project_fq_ideal: impl Fn(
            &Self::FqIdealSource,
            &<Self::Field as HasPrimeFieldConfig>::Config,
        ) -> IdealOverF,
        project_scalar: impl Fn(
            &Self::Scalar,
            &<Self::Field as HasPrimeFieldConfig>::Config,
        ) -> DynamicPolynomialF<F>,
    ) -> Result<(ConstraintEndpoints<Self::Field>, Self::VerifierClaims), ProtocolError<Self::Field>>
    where
        IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<F>>,
    {
        let sig = &self.signature;
        let num_constraints = count_constraints::<U>();
        let field_cfg = field_cfgs[0].clone();
        let n_fq = field_cfgs.len().saturating_sub(1);

        // Honest prover emits one sub-proof per declared prime.
        if sig.primes().len() != proof.ideal_checks_fq.len() {
            return Err(ProtocolError::FqIdealCheck {
                prime_idx: proof.ideal_checks_fq.len(),
                q: "<Length mismatch>".to_owned(),
                source: IdealCheckError::IdealCollectorError(
                    ideal_check::BatchedIdealCheckError::LengthMismatch {
                        num_ideals: sig.primes().len(),
                        provided_values: proof.ideal_checks_fq.len(),
                    },
                ),
            });
        }

        let q_star_idx = crate::shared_challenge::compute_q_star_idx::<F>(field_cfgs);
        let q_star_cfg = field_cfgs[q_star_idx].clone();

        // ===================== step 2: ideal check =========================
        let shared_eval_points: Vec<Vec<F>> =
            crate::shared_challenge::sample_shared_field_challenges::<F>(
                transcript,
                num_vars,
                &q_star_cfg,
                field_cfgs,
            );

        let mut ic_subclaims: Vec<ideal_check::VerifierSubclaim<F>> =
            Vec::with_capacity(field_cfgs.len());
        let q_subclaim = IdealCheckProtocol::<U>::verify_as_subprotocol::<_, IdealOverF, _, _>(
            transcript,
            proof.ideal_check.clone(),
            /* family_idx = */ 0,
            num_constraints.q,
            &shared_eval_points[0],
            |ideal| project_ideal(ideal, &field_cfg),
            |_| unreachable!("Q[X] family"),
        )?;
        ic_subclaims.push(q_subclaim);

        for (prime_idx, (cfg_q_i, fq_proof)) in field_cfgs[1..]
            .iter()
            .zip(proof.ideal_checks_fq.iter())
            .enumerate()
        {
            let family_idx = add!(prime_idx, 1);
            let fq_subclaim =
                IdealCheckProtocol::<U>::verify_as_subprotocol::<_, IdealOverF, _, _>(
                    transcript,
                    fq_proof.clone(),
                    family_idx,
                    num_constraints.for_prime(prime_idx),
                    &shared_eval_points[family_idx],
                    |_| unreachable!("F_q[X] family"),
                    |ideal| project_fq_ideal(ideal, cfg_q_i),
                )
                .map_err(|source| ProtocolError::FqIdealCheck {
                    prime_idx,
                    q: F::modulus(cfg_q_i).to_string(),
                    source,
                })?;
            ic_subclaims.push(fq_subclaim);
        }

        // ==================== step 3: eval projection ======================
        let projecting_elements: Vec<F> = crate::shared_challenge::sample_shared_field_challenge::<F>(
            transcript,
            &q_star_cfg,
            field_cfgs,
        );

        // Q[X] family.
        let projected_scalars_fx = project_scalars::<F, U>(|s| project_scalar(s, &field_cfg));
        let projected_scalars_f =
            project_scalars_to_field(projected_scalars_fx, &projecting_elements[0])
                .map_err(|(_s, _f, e)| ProtocolError::ScalarProjection(e))?;

        // Per-prime F_q[X] families.
        let projected_scalars_f_fq: Vec<ProjectedScalars<U::Scalar, F>> = field_cfgs
            .iter()
            .zip(projecting_elements.iter())
            .skip(1)
            .map(|(cfg_i, projecting_element)| {
                let projected_scalars_fx_i = project_scalars::<F, U>(|s| project_scalar(s, cfg_i));
                project_scalars_to_field(projected_scalars_fx_i, projecting_element)
                    .map_err(|(_s, _f, e)| ProtocolError::ScalarProjection(e))
            })
            .collect::<Result<_, _>>()?;

        // ======================= step 4: sumcheck ==========================
        if proof.cpr_proofs_fq.len() != n_fq || proof.combined_sumchecks_fq.len() != n_fq {
            return Err(ProtocolError::FqIdealCheck {
                prime_idx: proof.cpr_proofs_fq.len(),
                q: "<fq sub-proof length mismatch>".to_owned(),
                source: IdealCheckError::IdealCollectorError(
                    ideal_check::BatchedIdealCheckError::LengthMismatch {
                        num_ideals: n_fq,
                        provided_values: proof.cpr_proofs_fq.len(),
                    },
                ),
            });
        }

        // Sample one shared CPR batching challenge alpha in [0, q*).
        let folding_challenges: Vec<F> = crate::shared_challenge::sample_shared_field_challenge::<F>(
            transcript,
            &q_star_cfg,
            field_cfgs,
        );

        // -------- Q[X] family: CPR pre-sumcheck ------------------------
        let q_cpr_verifier_ancillary = CombinedPolyResolver::prepare_verifier::<U>(
            &proof.cpr_proof,
            proof.combined_sumcheck.claimed_sums()[0].clone(),
            &ic_subclaims[0],
            num_constraints.q,
            num_vars,
            &projecting_elements[0],
            &folding_challenges[0],
            &field_cfg,
        )?;

        // Booleanity pre-sumcheck (group index 1 on the Q-family).
        let num_pub_bin = sig.public_cols().num_binary_poly_cols();
        let num_total_bin = sig.total_cols().num_binary_poly_cols();
        let bin_wit_present = num_total_bin > num_pub_bin;

        let bool_verifier_ancillary = if bin_wit_present {
            let bool_claimed_sum = proof
                .combined_sumcheck
                .claimed_sums()
                .get(1)
                .ok_or(ProtocolError::BooleanityProofMissing)?;
            let num_wit_bin = num_total_bin.saturating_sub(num_pub_bin);
            let anc = BooleanityChecker::<F>::prepare_verifier(
                transcript,
                bool_claimed_sum,
                num_wit_bin,
                D,
                num_vars,
                &field_cfg,
            )
            .map_err(ProtocolError::Booleanity)?;
            Some(anc)
        } else {
            None
        };

        // -------- Per-prime families: CPR pre-sumcheck ------------------
        let mut fq_cpr_ancillaries = Vec::with_capacity(n_fq);
        for prime_idx in 0..n_fq {
            let family_idx = add!(prime_idx, 1);
            let cfg_i = &field_cfgs[family_idx];
            let anc_i = CombinedPolyResolver::prepare_verifier::<U>(
                &proof.cpr_proofs_fq[prime_idx],
                proof.combined_sumchecks_fq[prime_idx].claimed_sums()[0].clone(),
                &ic_subclaims[family_idx],
                num_constraints.for_prime(prime_idx),
                num_vars,
                &projecting_elements[family_idx],
                &folding_challenges[family_idx],
                cfg_i,
            )?;
            fq_cpr_ancillaries.push(anc_i);
        }

        // -------- Lockstep multi-degree sumcheck verify ----------------
        let mut family_proofs: Vec<(&MultiDegreeSumcheckProof<F>, &F::Config)> =
            Vec::with_capacity(add!(n_fq, 1));
        family_proofs.push((&proof.combined_sumcheck, &field_cfg));
        for prime_idx in 0..n_fq {
            let family_idx = add!(prime_idx, 1);
            family_proofs.push((
                &proof.combined_sumchecks_fq[prime_idx],
                &field_cfgs[family_idx],
            ));
        }
        let all_md_subclaims = MultiDegreeSumcheck::verify_as_subprotocol(
            transcript,
            num_vars,
            &family_proofs,
            &q_star_cfg,
        )
        .map_err(CombinedPolyResolverError::SumcheckError)?;

        // -------- Q[X] family finalize ---------------------------------
        let q_md_subclaims = all_md_subclaims
            .first()
            .expect("Q[X] family subclaim always present");

        let cpr_subclaim = CombinedPolyResolver::finalize_verifier::<U>(
            transcript,
            proof.cpr_proof.clone(),
            q_md_subclaims.point().to_vec(),
            q_md_subclaims.expected_evaluations()[0].clone(),
            q_cpr_verifier_ancillary,
            &projected_scalars_f,
            /* family_idx = */ 0,
            &field_cfg,
        )?;

        // Booleanity -> multipoint-eval `alpha_prime` bridge.
        let bool_bit_slice_evals: Option<Vec<F>> = if let Some(anc) = bool_verifier_ancillary {
            let booleanity_proof = proof
                .booleanity_proof
                .clone()
                .ok_or(ProtocolError::BooleanityProofMissing)?;
            let expected_eval = q_md_subclaims
                .expected_evaluations()
                .get(1)
                .ok_or(ProtocolError::BooleanityProofMissing)?;
            let bool_subclaim = BooleanityChecker::<F>::finalize_verifier(
                transcript,
                booleanity_proof,
                q_md_subclaims.point().to_vec(),
                expected_eval,
                anc,
                &field_cfg,
            )
            .map_err(ProtocolError::Booleanity)?;
            Some(bool_subclaim.bit_slice_evals)
        } else {
            None
        };

        // -------- Per-prime family finalize ---------------------------
        let mut cpr_eval_points_fq: Vec<Vec<F>> = Vec::with_capacity(n_fq);
        let mut cpr_up_evals_fq: Vec<Vec<F>> = Vec::with_capacity(n_fq);
        let mut cpr_bit_op_evals_fq: Vec<Vec<F>> = Vec::with_capacity(n_fq);
        let mut cpr_down_evals_fq: Vec<Vec<F>> = Vec::with_capacity(n_fq);
        for (prime_idx, (cpr_ancillary_i, md_subclaims_i)) in fq_cpr_ancillaries
            .into_iter()
            .zip(all_md_subclaims.iter().skip(1))
            .enumerate()
        {
            let family_idx = add!(prime_idx, 1);
            let cfg_i = &field_cfgs[family_idx];
            let cpr_subclaim_i = CombinedPolyResolver::finalize_verifier::<U>(
                transcript,
                proof.cpr_proofs_fq[prime_idx].clone(),
                md_subclaims_i.point().to_vec(),
                md_subclaims_i.expected_evaluations()[0].clone(),
                cpr_ancillary_i,
                &projected_scalars_f_fq[prime_idx],
                family_idx,
                cfg_i,
            )?;
            cpr_eval_points_fq.push(cpr_subclaim_i.evaluation_point);
            cpr_up_evals_fq.push(cpr_subclaim_i.up_evals);
            cpr_bit_op_evals_fq.push(cpr_subclaim_i.bit_op_evals);
            cpr_down_evals_fq.push(cpr_subclaim_i.down_evals);
        }

        // Squeeze alpha_prime in the same transcript order as the prover.
        let alpha_prime_f: Option<F> = bool_bit_slice_evals
            .as_ref()
            .map(|_| transcript.get_field_challenge(&field_cfg));

        assert!(
            proof.lookup_proof.is_none(),
            "Arbitrary lookup argument is not supported yet!"
        );

        // ================= step 5: lockstep multipoint-eval ================
        // Length-mismatch guard.
        if proof.multipoint_evals_fq.len() != n_fq {
            return Err(ProtocolError::FqIdealCheck {
                prime_idx: proof.multipoint_evals_fq.len(),
                q: "<fq mp-eval sub-proof length mismatch>".to_owned(),
                source: IdealCheckError::IdealCollectorError(
                    ideal_check::BatchedIdealCheckError::LengthMismatch {
                        num_ideals: n_fq,
                        provided_values: proof.multipoint_evals_fq.len(),
                    },
                ),
            });
        }

        // Q[X] family up_evals (with booleanity-bridge appended when
        // alpha_prime is present) + num_wit_bin extension width.
        let (q_up_evals, num_wit_bin): (Vec<F>, usize) =
            if let (Some(bit_slice_evals), Some(alpha_prime)) =
                (&bool_bit_slice_evals, &alpha_prime_f)
            {
                let num_wit_bin = num_total_bin.saturating_sub(num_pub_bin);
                let extra = alpha_prime_bridge_up_evals::<F, D>(
                    bit_slice_evals,
                    num_wit_bin,
                    alpha_prime,
                    &field_cfg,
                );
                (
                    cpr_subclaim.up_evals.iter().cloned().chain(extra).collect(),
                    num_wit_bin,
                )
            } else {
                (cpr_subclaim.up_evals.clone(), 0)
            };

        // Per-prime families: zero-pad up_evals to match Q's column count.
        let fq_up_evals_ext: Vec<Vec<F>> = (0..n_fq)
            .map(|prime_idx| {
                let family_idx = add!(prime_idx, 1);
                let zero_i = F::zero_with_cfg(&field_cfgs[family_idx]);
                let mut up_i = cpr_up_evals_fq[prime_idx].clone();
                up_i.extend((0..num_wit_bin).map(|_| zero_i.clone()));
                up_i
            })
            .collect();

        let shifts = sig.shifts();

        // Single (n+1)-family lockstep MP-eval verifier.
        let mut all_proofs: Vec<multipoint_eval::Proof<F>> = Vec::with_capacity(add!(n_fq, 1));
        all_proofs.push(proof.multipoint_eval.clone());
        all_proofs.extend(proof.multipoint_evals_fq.iter().cloned());

        let mut all_families: Vec<MultipointEvalFamilyInputs<'_, F>> =
            Vec::with_capacity(add!(n_fq, 1));
        // The verifier-side MP precheck only consumes the claimed eval vectors;
        // bit-op MLE bodies exist on the prover side only.
        all_families.push(MultipointEvalFamilyInputs {
            field_cfg: &field_cfg,
            trace_mles: &[],
            bit_op_mles: &[],
            eval_point: &cpr_subclaim.evaluation_point,
            up_evals: &q_up_evals,
            bit_op_evals: &cpr_subclaim.bit_op_evals,
            down_evals: &cpr_subclaim.down_evals,
        });
        for (prime_idx, up_evals) in fq_up_evals_ext.iter().enumerate() {
            let family_idx = add!(prime_idx, 1);
            all_families.push(MultipointEvalFamilyInputs {
                field_cfg: &field_cfgs[family_idx],
                trace_mles: &[],
                bit_op_mles: &[],
                eval_point: &cpr_eval_points_fq[prime_idx],
                up_evals,
                bit_op_evals: &cpr_bit_op_evals_fq[prime_idx],
                down_evals: &cpr_down_evals_fq[prime_idx],
            });
        }

        let mut subclaims_iter = MultipointEval::verify_as_subprotocol(
            transcript,
            all_proofs,
            all_families,
            shifts,
            num_vars,
            &q_star_cfg,
        )?
        .into_iter();

        let mp_subclaim = subclaims_iter.next().expect("Q-family subclaim present");
        let mp_subclaims_fq: Vec<multipoint_eval::Subclaim<F>> = subclaims_iter.collect();
        debug_assert_eq!(mp_subclaims_fq.len(), n_fq);

        // Extract the shared endpoints. `r_0` lives in `ConstraintEndpoints`;
        // the full subclaims travel in `UairVerifierClaims` for the later
        // `verify_subclaim` calls.
        let r_0 = mp_subclaim.r0.clone();
        let r_0_fq: Vec<Vec<F>> = mp_subclaims_fq.iter().map(|s| s.r0.clone()).collect();

        Ok((
            ConstraintEndpoints::new(r_0, r_0_fq),
            UairVerifierClaims {
                mp_subclaim,
                mp_subclaims_fq,
                projecting_elements,
                alpha_prime_f,
            },
        ))
    }

    /// Verifier hook: close the per-family multipoint-eval subclaims against
    /// the substrate-assembled lifted evaluations at `r_0`.
    ///
    /// Faithfully relocates the per-family `verify_subclaim` + bit-op
    /// reconstruction + `alpha'`/zero-pad open-eval logic from the old verifier
    /// step 6. `per_family_all_lifted[i]` is family `i`'s FULL (public +
    /// witness, layout-interleaved) lifted evals at `r_0`, assembled by the
    /// substrate.
    #[allow(clippy::arithmetic_side_effects)]
    fn verify_lifted_evals(
        &self,
        claims: &Self::VerifierClaims,
        per_family_all_lifted: &[Vec<DynamicPolynomialF<Self::Field>>],
        field_cfgs: &[<Self::Field as HasPrimeFieldConfig>::Config],
    ) -> Result<(), ProtocolError<Self::Field>> {
        let sig = &self.signature;
        let n_fq = claims.mp_subclaims_fq.len();
        let wit_cols = sig.witness_cols();
        let num_wit_bin = wit_cols.num_binary_poly_cols();

        // Q-family (i = 0).
        let field_cfg = &field_cfgs[0];
        let q_all_lifted = &per_family_all_lifted[0];

        let mut q_open_evals: Vec<F> = q_all_lifted
            .iter()
            .map(|bar_u| {
                bar_u
                    .evaluate_at_point(&claims.projecting_elements[0])
                    .map_err(ProtocolError::LiftedEvalProjection)
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Booleanity bridge: append alpha'-projected witness-bin lifts.
        // Q-family witness columns are laid out as `[wit_bin..., wit_arb...,
        // wit_int...]` after the public interleave; the substrate already
        // interleaves public+witness, so the witness-bin block starts right
        // after the public-bin block.
        if let Some(alpha_prime) = &claims.alpha_prime_f {
            let num_pub_bin = sig.public_cols().num_binary_poly_cols();
            let num_total_bin = sig.total_cols().num_binary_poly_cols();
            for bar_u in &q_all_lifted[num_pub_bin..num_total_bin] {
                q_open_evals.push(
                    bar_u
                        .evaluate_at_point(alpha_prime)
                        .map_err(ProtocolError::LiftedEvalProjection)?,
                );
            }
        }

        MultipointEval::verify_subclaim(
            &claims.mp_subclaim,
            &q_open_evals,
            &derive_bit_op_open_evals::<F, D>(
                sig.bit_op_specs(),
                q_all_lifted,
                &claims.projecting_elements[0],
                field_cfg,
            )?,
            sig.shifts(),
            field_cfg,
        )?;

        // Per-prime families (i >= 1).
        for prime_idx in 0..n_fq {
            let family_idx = add!(prime_idx, 1);
            let cfg_i = &field_cfgs[family_idx];
            let all_lifted_i = &per_family_all_lifted[family_idx];

            let mut open_evals_i: Vec<F> = all_lifted_i
                .iter()
                .map(|bar_u| {
                    bar_u
                        .evaluate_at_point(&claims.projecting_elements[family_idx])
                        .map_err(ProtocolError::LiftedEvalProjection)
                })
                .collect::<Result<Vec<_>, _>>()?;

            // Booleanity-bridge slots on fq families are zero-padded
            // (witness-bin lives in Q[X] only).
            if claims.alpha_prime_f.is_some() {
                let zero_i = F::zero_with_cfg(cfg_i);
                open_evals_i.extend((0..num_wit_bin).map(|_| zero_i.clone()));
            }

            MultipointEval::verify_subclaim(
                &claims.mp_subclaims_fq[prime_idx],
                &open_evals_i,
                &derive_bit_op_open_evals::<F, D>(
                    sig.bit_op_specs(),
                    all_lifted_i,
                    &claims.projecting_elements[family_idx],
                    cfg_i,
                )?,
                sig.shifts(),
                cfg_i,
            )?;
        }

        Ok(())
    }
}

/// Reconstruct the bit-op virtual-column open-evals at the `psi_a` projecting
/// element from the family's committed lifted evals.
///
/// For each bit-op spec, apply its `R`-linear map (`Rot`/`ShR`) to the source
/// column's lifted eval (MLE commutes with the map), then evaluate the result
/// at the projecting element. Moved verbatim from the substrate verifier.
fn derive_bit_op_open_evals<F: PrimeField, const D: usize>(
    specs: &[zinc_uair::BitOpSpec],
    all_lifted: &[DynamicPolynomialF<F>],
    projecting_element: &F,
    field_cfg: &F::Config,
) -> Result<Vec<F>, ProtocolError<F>> {
    specs
        .iter()
        .map(|spec| {
            let source = &all_lifted[spec.source_col()];
            let transformed = spec.op().transform::<F, D>(source, field_cfg);
            transformed
                .evaluate_at_point(projecting_element)
                .map_err(ProtocolError::LiftedEvalProjection)
        })
        .collect()
}

/// Project a single witness binary-poly column at a field element by
/// evaluating each bit-packed cell $\sum_i \text{bit}_i \cdot X^i$ at
/// $X = \alpha$.
///
/// Used to build the appended $\alpha'$-projected witness-binary-poly MLEs
/// that participate in `MultipointEval` as the Schwartz-Zippel bridge from
/// booleanity into the PCS chain. Moved here verbatim from the prover.
#[allow(clippy::arithmetic_side_effects)]
fn project_binary_col_at_field<F, const D: usize>(
    col: &DenseMultilinearExtension<BinaryPoly<D>>,
    alpha_powers: &[F],
    field_cfg: &F::Config,
) -> DenseMultilinearExtension<F::Inner>
where
    F: PrimeField,
    F::Integer: Clone + Send + Sync,
{
    debug_assert_eq!(alpha_powers.len(), D);
    let zero = F::zero_with_cfg(field_cfg);

    let evaluations: Vec<F::Inner> = col
        .evaluations
        .iter()
        .map(|entry| {
            let mut acc = zero.clone();
            for (i, bit) in entry.iter().enumerate() {
                if bit.into_inner() {
                    acc += &alpha_powers[i];
                }
            }
            acc.into_inner()
        })
        .collect();

    DenseMultilinearExtension {
        num_vars: col.num_vars,
        evaluations,
    }
}

/// Run the per-family ideal check, dispatching on the projected trace layout
/// (`RowMajor` -> `prove_combined`, `ColumnMajor` -> `prove_mle_first`).
#[allow(clippy::too_many_arguments)]
fn prove_ideal_family<U, F, T, const D: usize>(
    transcript: &mut T,
    trace: &ProjectedTrace<F>,
    scalars: &ProjectedScalars<U::Scalar, DynamicPolynomialF<F>>,
    family_idx: usize,
    num_constraints: usize,
    eval_point: &[F],
    cfg: &F::Config,
) -> Result<IdealCheckProof<F>, zinc_piop::ideal_check::IdealCheckError<F>>
where
    U: Uair,
    F: InnerTransparentField,
    F::Integer: ConstTranscribable,
    T: Transcript,
{
    match trace {
        ProjectedTrace::RowMajor(t) => IdealCheckProtocol::<U>::prove_combined::<_, D>(
            transcript,
            t,
            scalars,
            family_idx,
            num_constraints,
            eval_point,
            cfg,
        ),
        ProjectedTrace::ColumnMajor(t) => IdealCheckProtocol::<U>::prove_mle_first::<_, D>(
            transcript,
            t,
            scalars,
            family_idx,
            num_constraints,
            eval_point,
            cfg,
        ),
    }
}

/// Owned, per-family bundle of the MLE-evaluation claims the constraint
/// argument produces and feeds into the frontend's multipoint-eval.
///
/// This is the owned counterpart of [`MultipointEvalFamilyInputs`] (which
/// borrows); [`FamilyEvalClaims::as_inputs`] hands a borrowing view to
/// [`MultipointEval`]. It is purely internal scaffolding for
/// [`UairFrontend::prove_constraints`] — now that multipoint-eval runs inside
/// the frontend, these claims never cross the [`ConstraintSystem`] seam.
struct FamilyEvalClaims<F: PrimeField> {
    /// Field configuration for this family.
    field_cfg: F::Config,
    /// Trace MLEs, projected into this family's field.
    trace_mles: Vec<DenseMultilinearExtension<F::Inner>>,
    /// Bit-op virtual-column MLEs, projected into this family's field.
    bit_op_mles: Vec<DenseMultilinearExtension<F::Inner>>,
    /// Evaluation point `r*` for this family.
    eval_point: Vec<F>,
    /// `up_eval_j = v_j(r*)` per committed column `j`.
    up_evals: Vec<F>,
    /// `bit_op_eval_l = bit_op_l(r*)` per bit-op virtual column `l`.
    bit_op_evals: Vec<F>,
    /// `down_eval_k = v_{src_k}^{<<c_k}(r*)` per shift `k`.
    down_evals: Vec<F>,
}

impl<F: PrimeField> FamilyEvalClaims<F> {
    /// Assemble a per-family claim bundle.
    #[allow(clippy::too_many_arguments)]
    fn new(
        field_cfg: F::Config,
        trace_mles: Vec<DenseMultilinearExtension<F::Inner>>,
        bit_op_mles: Vec<DenseMultilinearExtension<F::Inner>>,
        eval_point: Vec<F>,
        up_evals: Vec<F>,
        bit_op_evals: Vec<F>,
        down_evals: Vec<F>,
    ) -> Self {
        Self {
            field_cfg,
            trace_mles,
            bit_op_mles,
            eval_point,
            up_evals,
            bit_op_evals,
            down_evals,
        }
    }

    /// Borrowing view consumed by [`MultipointEval::prove_as_subprotocol`].
    fn as_inputs(&self) -> MultipointEvalFamilyInputs<'_, F> {
        MultipointEvalFamilyInputs {
            field_cfg: &self.field_cfg,
            trace_mles: &self.trace_mles,
            bit_op_mles: &self.bit_op_mles,
            eval_point: &self.eval_point,
            up_evals: &self.up_evals,
            bit_op_evals: &self.bit_op_evals,
            down_evals: &self.down_evals,
        }
    }
}
