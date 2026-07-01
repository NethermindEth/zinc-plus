//! Abstract constraint-system seam for the Zinc+ protocol.
//!
//! This module introduces [`ConstraintSystem`] — the protocol-facing boundary
//! that the substrate (commit / `phi_q` projection / lift-and-project / Zip+
//! PCS) is made generic over. UAIR is the first (and, for now, only)
//! implementation, wired up via the
//! [`UairFrontend`](crate::uair_frontend::UairFrontend) adapter.
//!
//! **Seam shape.** The constraint argument owns everything that reduces the
//! relation to a *single* witness-evaluation claim — including the lockstep
//! multipoint-eval (former substrate step 6, now run inside
//! [`UairFrontend::prove_constraints`](crate::uair_frontend::UairFrontend)). It
//! hands the substrate only the shared endpoint [`ConstraintEndpoints`]; the
//! substrate's remaining job is lift-and-project + the Zip+ PCS open at that
//! point, which keeps it constraint-system-agnostic (a future R1CS/Spartan
//! frontend terminates at a single point directly).
//!
//! Verifier-side methods are still `todo!` — wiring them is the later verifier
//! phase and does not modify the existing verifier (Phase 4 of
//! `docs/ucs-refactor-plan.md`).
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
use zinc_piop::projections::ProjectedTrace;
use zinc_poly::univariate::dynamic::over_field::DynamicPolynomialF;
use zinc_transcript::traits::{Transcribable, Transcript};
use zinc_uair::{
    UairSignature,
    ideal::{Ideal, IdealCheck},
};

use crate::ProtocolError;

/// The trace/column layout the substrate needs to commit, project, lift, and
/// run multipoint-eval over.
///
/// Kept identical to [`UairSignature`] for now — the 3-group
/// `binary_poly` / `arbitrary_poly` / `int` layout plus `shifts` and `primes`
/// (the "keep 3 groups" decision in `docs/ucs-refactor-plan.md`). Bit-op and
/// lookup specs stay *inside* the frontend; they never surface across the seam
/// (the frontend's own multipoint-eval consumes them).
pub type Layout<P> = UairSignature<P>;

/// The single evaluation endpoint the constraint argument reduces all of its
/// per-family claims down to.
///
/// The UAIR frontend's lockstep multipoint-eval (the reduction over committed
/// columns, shifts, and bit-op virtuals) collapses every per-column / shift /
/// virtual evaluation claim at the per-family points `r*` into a single shared
/// point `r_0` (lifted into each family's field). The substrate consumes only
/// this endpoint: its lift-and-project + Zip+ PCS open bind the committed
/// witness to `r_0`. A frontend whose argument already terminates at a single
/// point (e.g. R1CS/Spartan) returns that point directly, with no multipoint
/// reduction.
///
/// `#[non_exhaustive]` so future additions (e.g. ZK blinding metadata) stay
/// purely additive.
#[non_exhaustive]
pub struct ConstraintEndpoints<F: PrimeField> {
    /// Q[X]-family (`q_0`) evaluation point `r_0`.
    pub r_0: Vec<F>,
    /// Per-prime `F_{q_i}[X]` evaluation points — the shared `r_0` lifted into
    /// each declared prime's field, in `layout().primes()` order. Empty for a
    /// `Q[X]`-only layout.
    pub r_0_fq: Vec<Vec<F>>,
}

impl<F: PrimeField> ConstraintEndpoints<F> {
    /// Assemble the constraint-argument endpoint bundle.
    pub fn new(r_0: Vec<F>, r_0_fq: Vec<Vec<F>>) -> Self {
        Self { r_0, r_0_fq }
    }
}

/// The protocol-facing constraint-system seam.
///
/// An implementation owns the **constraint argument** — today, for UAIR: the
/// ideal-membership check (prover step 3), the `psi_a` scalar projection and
/// bit-op virtual-column materialization (part of step 4), the constraint
/// sumcheck + booleanity (step 5), and the lockstep multipoint-eval that
/// reduces every per-family evaluation claim to a single shared point (former
/// substrate step 6). It consumes the per-family `phi_q`-projected traces the
/// substrate produced and returns its sub-proof plus the
/// [`ConstraintEndpoints`] the substrate binds via lift-and-project and the
/// Zip+ PCS open.
///
/// **Seam discipline (decision 9 of `docs/ucs-refactor-plan.md`, Option A).**
/// Only field-projected witness data crosses the seam
/// (`ProjectedTrace<Self::Field>` + cfgs). Frontend-private state stays on the
/// implementing type: UAIR's raw `BinaryPoly<D>` columns (a const generic
/// cannot appear in a trait-method signature) and its projection maps; R1CS's
/// `A,B,C` matrices. This keeps the trait constraint-system-agnostic and lets a
/// future ZK variant add blinding via layout-declared columns without changing
/// these methods.
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

    /// Opaque frontend verifier tail-state, carried from
    /// [`verify_constraints`](Self::verify_constraints) to
    /// [`verify_lifted_evals`](Self::verify_lifted_evals).
    ///
    /// It holds whatever the frontend's multipoint-eval verify produced that
    /// the per-family lifted-eval consistency check needs but the substrate
    /// does not interpret (for UAIR: the multipoint-eval subclaims, the
    /// per-family $\psi$-projecting elements, and the booleanity `alpha'`
    /// challenge).
    type VerifierClaims;

    /// Source ideal type the frontend's projection closure maps into the field
    /// ideal `IdealOverF` (UAIR: `IdealOrZero<U::Ideal>` for the $Q[X]$
    /// family).
    type IdealSource;

    /// Source ideal type for the per-prime $F_q[X]$ families (UAIR:
    /// `IdealOrZero<U::FqIdeal>`).
    type FqIdealSource;

    /// Source scalar type the frontend's `psi_a` scalar projection consumes
    /// (UAIR: `U::Scalar`).
    type Scalar;

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
    /// Inputs are the substrate-produced, field-projected data; the UAIR impl
    /// additionally relies on its own state (the borrowed original trace for
    /// booleanity / bit-op materialization, and the projection closures), which
    /// it carries on the concrete frontend type rather than across the seam.
    ///
    /// Returns the sub-proof together with the [`ConstraintEndpoints`] — the
    /// shared point `r_0` (and its per-prime liftings) the multipoint-eval
    /// reduced to — which the substrate's lift-and-project + PCS open bind.
    ///
    /// `projected_traces` / `field_cfgs` are indexed by family
    /// (`[0] = Q[X]`, `[i] = q_i`).
    fn prove_constraints(
        &self,
        transcript: &mut impl Transcript,
        projected_traces: &[ProjectedTrace<Self::Field>],
        field_cfgs: &[<Self::Field as HasPrimeFieldConfig>::Config],
        num_vars: usize,
    ) -> Result<(Self::ConstraintProof, ConstraintEndpoints<Self::Field>), ProtocolError<Self::Field>>;

    /// Verifier side of the constraint argument (mirror of
    /// [`prove_constraints`](Self::prove_constraints)).
    ///
    /// Runs the ideal-check verify, the `psi_a` scalar projection, the
    /// constraint sumcheck + booleanity verify, and the lockstep
    /// multipoint-eval verify — re-deriving the shared evaluation endpoint
    /// [`ConstraintEndpoints`] the substrate binds via lift-and-project + PCS
    /// verify. The opaque [`VerifierClaims`](Self::VerifierClaims) tail-state
    /// carries the multipoint-eval subclaims (and, for UAIR, the per-family
    /// projecting elements + `alpha'`) forward to
    /// [`verify_lifted_evals`](Self::verify_lifted_evals).
    ///
    /// The `project_ideal` / `project_fq_ideal` / `project_scalar` closures are
    /// frontend-specific projections (mirroring the current `verify()`
    /// wrapper): they map the source ideal/scalar types into the field.
    /// They are passed as params (not stored) so a verify-side frontend
    /// need not be constructed with the prover's scalar map.
    ///
    /// `field_cfgs` are indexed by family (`[0] = Q[X]`, `[i] = q_i`).
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
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
        ) -> DynamicPolynomialF<Self::Field>,
    ) -> Result<(ConstraintEndpoints<Self::Field>, Self::VerifierClaims), ProtocolError<Self::Field>>
    where
        IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<Self::Field>>;

    /// Verifier hook: close the per-family multipoint-eval subclaims against
    /// the substrate-assembled lifted evaluations at the shared endpoint `r_0`.
    ///
    /// `per_family_all_lifted[i]` is family `i`'s FULL (public + witness,
    /// layout-interleaved) lifted evals at `r_0`, assembled by the substrate.
    /// The frontend runs, per family: the `psi_a` projection at its projecting
    /// element, the bit-op-virtual reconstruction + the `alpha'` bridge append
    /// (Q-family) / zero-pad (fq families), and
    /// `MultipointEval::verify_subclaim`.
    ///
    /// A frontend without virtual columns (e.g. R1CS) still uses this to close
    /// its single-family subclaim (or is a no-op if it terminates at a single
    /// point directly).
    fn verify_lifted_evals(
        &self,
        claims: &Self::VerifierClaims,
        per_family_all_lifted: &[Vec<DynamicPolynomialF<Self::Field>>],
        field_cfgs: &[<Self::Field as HasPrimeFieldConfig>::Config],
    ) -> Result<(), ProtocolError<Self::Field>>;
}
