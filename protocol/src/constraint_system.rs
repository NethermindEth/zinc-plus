//! Abstract constraint-system seam for the Zinc+ protocol.
//!
//! This module introduces [`ConstraintSystem`] — the protocol-facing boundary
//! that the substrate (commit / `phi_q` & `psi_a` projection / multipoint-eval
//! / lift-and-project / Zip+ PCS) is to be made generic over. UAIR will be the
//! first (and, for now, only) implementation, wired up via the
//! [`UairFrontend`](crate::uair_frontend::UairFrontend) adapter.
//!
//! **Scope of this commit:** trait + contract types only. No prover-step bodies
//! are moved yet, the protocol is not yet generic over the trait, and the
//! verifier is untouched. Items marked `PROPOSED` are expected to be refined
//! when the step bodies move (Phase 2 of `docs/ucs-refactor-plan.md`).
//!
//! ## Why `Field` is an associated type (not a method generic)
//!
//! The UAIR implementation of [`ConstraintSystem::prove_constraints`] needs the
//! same large field-bound bundle the current prover carries (see the
//! `impl_with_type_bounds!` macro in [`crate::prover`]:
//! `InnerTransparentField`, `FromWithConfig<&_>`, `MulByScalar`, …, plus
//! `Integer = Zt::Fmod` tying the field to the type bundle). A trait *method*
//! generic over `F` could not add those bounds in the impl. Exposing the field
//! as `type Field: PrimeField` lets the concrete impl (`UairFrontend<U, F, …>`)
//! pick an `F` that already satisfies every extra bound, exactly as
//! `ZincPlusPiop<Zt, U, F, D, FD>` is monomorphized today.

use crypto_primitives::{HasPrimeFieldConfig, PrimeField, Semiring};
use zinc_piop::{multipoint_eval::MultipointEvalFamilyInputs, projections::ProjectedTrace};
use zinc_poly::{
    mle::DenseMultilinearExtension, univariate::dynamic::over_field::DynamicPolynomialF,
};
use zinc_transcript::traits::{Transcribable, Transcript};
use zinc_uair::UairSignature;

use crate::ProtocolError;

/// The trace/column layout the substrate needs to commit, project, lift, and
/// run multipoint-eval over.
///
/// Kept identical to [`UairSignature`] for now — the 3-group
/// `binary_poly` / `arbitrary_poly` / `int` layout plus `shifts` and `primes`
/// (the "keep 3 groups" decision in `docs/ucs-refactor-plan.md`). Bit-op and
/// lookup specs stay *inside* the frontend; they only surface across the seam
/// as opaque MLEs/evals in [`FamilyEvalClaims`].
pub type Layout<P> = UairSignature<P>;

/// Owned, per-family bundle of the MLE-evaluation claims the constraint
/// argument produces and the substrate's multipoint-eval binds.
///
/// This is the owned counterpart of [`MultipointEvalFamilyInputs`] (which
/// borrows); use [`FamilyEvalClaims::as_inputs`] to hand a borrowing view to
/// the substrate.
///
/// `#[non_exhaustive]` so future additions (e.g. ZK blinding metadata) stay
/// purely additive.
#[non_exhaustive]
pub struct FamilyEvalClaims<F: PrimeField> {
    /// Field configuration for this family.
    pub field_cfg: F::Config,
    /// Trace MLEs, projected into this family's field.
    pub trace_mles: Vec<DenseMultilinearExtension<F::Inner>>,
    /// Bit-op virtual-column MLEs, projected into this family's field.
    pub bit_op_mles: Vec<DenseMultilinearExtension<F::Inner>>,
    /// Evaluation point `r*` for this family.
    pub eval_point: Vec<F>,
    /// `up_eval_j = v_j(r*)` per committed column `j`.
    pub up_evals: Vec<F>,
    /// `bit_op_eval_l = bit_op_l(r*)` per bit-op virtual column `l`.
    pub bit_op_evals: Vec<F>,
    /// `down_eval_k = v_{src_k}^{<<c_k}(r*)` per shift `k`.
    pub down_evals: Vec<F>,
}

impl<F: PrimeField> FamilyEvalClaims<F> {
    /// Assemble a per-family claim bundle.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
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

    /// Borrowing view consumed by the substrate's multipoint-eval
    /// (`MultipointEval::prove_as_subprotocol` / `verify_subclaim`).
    pub fn as_inputs(&self) -> MultipointEvalFamilyInputs<'_, F> {
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

/// The protocol-facing constraint-system seam.
///
/// An implementation owns the **constraint argument** — today, for UAIR: the
/// ideal-membership check (prover step 3), the `psi_a` scalar projection and
/// bit-op virtual-column materialization (part of step 4), and the constraint
/// sumcheck + booleanity (step 5). It consumes the per-family `phi_q`-projected
/// traces the substrate produced and returns its sub-proof plus one
/// [`FamilyEvalClaims`] per constraint family for the substrate to bind via
/// multipoint-eval, lift-and-project, and the Zip+ PCS open.
///
/// **Seam discipline (decision 9 of `docs/ucs-refactor-plan.md`, Option A).** Only
/// field-projected witness data crosses the seam (`ProjectedTrace<Self::Field>` +
/// cfgs). Frontend-private state stays on the implementing type: UAIR's raw
/// `BinaryPoly<D>` columns (a const generic cannot appear in a trait-method
/// signature) and its projection maps; R1CS's `A,B,C` matrices. This keeps the
/// trait constraint-system-agnostic and lets a future ZK variant add blinding via
/// layout-declared columns without changing these methods.
///
/// Families are ordered `[Q[X] (q0), q_1, .., q_n]` with `q_1..q_n =
/// self.layout().primes()`.
pub trait ConstraintSystem {
    /// Prime-power type carried by the layout (`UairSignature::primes()`),
    /// e.g. the field integer type `Zt::Fmod`.
    type Prime: Semiring;

    /// The projection field `F` the constraint argument runs over. The impl
    /// supplies an `F` with all the extra bounds the engine needs (see the
    /// module docs); the trait only requires `PrimeField`.
    type Field: PrimeField;

    /// Frontend sub-proof, embedded in the substrate `Proof`.
    ///
    /// PROPOSED: bound is `Transcribable` here; Phase 2 may also require
    /// `GenTranscribable` once (de)serialization is wired through the substrate
    /// `Proof`.
    type ConstraintProof: Transcribable;

    /// The trace/column layout (drives substrate commit / projection / lift /
    /// multipoint-eval). See [`Layout`].
    fn layout(&self) -> &Layout<Self::Prime>;

    /// Whether the PCS-only prime `q''` must be sampled fresh (decoupled from
    /// the constraint primes) rather than aliased to `q0`.
    ///
    /// Default reproduces the current UAIR behaviour exactly: decouple iff
    /// there is at least one declared `F_q[X]` family. A frontend with a
    /// different family/commitment relationship may override.
    fn needs_decoupled_pcs_prime(&self) -> bool {
        !self.layout().primes().is_empty()
    }

    /// Prover side of the constraint argument.
    ///
    /// PROPOSED signature (finalized when step bodies move in Phase 2). Inputs
    /// shown are the substrate-produced, field-projected data; the UAIR impl
    /// additionally relies on its own state (the borrowed original trace for
    /// booleanity / bit-op materialization, and the projection closures), which
    /// it carries on the concrete frontend type rather than across the seam.
    ///
    /// `projected_traces` / `field_cfgs` are indexed by family
    /// (`[0] = Q[X]`, `[i] = q_i`).
    fn prove_constraints(
        &self,
        transcript: &mut impl Transcript,
        projected_traces: &[ProjectedTrace<Self::Field>],
        field_cfgs: &[<Self::Field as HasPrimeFieldConfig>::Config],
        num_vars: usize,
    ) -> Result<
        (Self::ConstraintProof, Vec<FamilyEvalClaims<Self::Field>>),
        ProtocolError<Self::Field>,
    >;

    /// Verifier side: re-derive the per-family expected claim bundles from the
    /// sub-proof, for the substrate to bind via multipoint-eval verify.
    ///
    /// PROPOSED — see [`Self::prove_constraints`]. (Wiring this is part of the
    /// later verifier work; declaring it here does not modify the verifier.)
    fn verify_constraints(
        &self,
        transcript: &mut impl Transcript,
        proof: &Self::ConstraintProof,
        field_cfgs: &[<Self::Field as HasPrimeFieldConfig>::Config],
        num_vars: usize,
    ) -> Result<Vec<FamilyEvalClaims<Self::Field>>, ProtocolError<Self::Field>>;

    /// Verifier hook: reconstruct the frontend-defined *virtual* column
    /// evaluations (bit-op virtuals, booleanity `alpha'` bridge) at the
    /// substrate's multipoint-eval endpoint `r_0`, from the committed columns'
    /// lifted evaluations there.
    ///
    /// Virtual columns are not committed; this closes their subclaim at `r_0`.
    /// A frontend without virtual columns (e.g. R1CS) returns an empty vector.
    /// PROPOSED.
    fn reconstruct_virtual_evals(
        &self,
        committed_lifted_at_r0: &[DynamicPolynomialF<Self::Field>],
        field_cfg: &<Self::Field as HasPrimeFieldConfig>::Config,
    ) -> Vec<DynamicPolynomialF<Self::Field>>;
}
