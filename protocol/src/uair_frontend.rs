//! UAIR adapter for the [`ConstraintSystem`](crate::constraint_system::ConstraintSystem)
//! seam.
//!
//! [`UairFrontend`] is the bridge that will implement `ConstraintSystem` for any
//! `U: Uair` by delegating to the existing UAIR constraint engine
//! (`CombinedPolyResolver` + `IdealCheckProtocol` + booleanity).
//!
//! **This commit declares the adapter type only.** The `impl ConstraintSystem
//! for UairFrontend` — i.e. moving the bodies of prover steps 3–5 (and their
//! verifier counterparts) behind the trait — is Phase 2 and is deliberately
//! left out so the abstract surface can be reviewed first.

use core::marker::PhantomData;

use zinc_uair::Uair;

/// Adapter wrapping a `U: Uair` so it can serve as a
/// [`ConstraintSystem`](crate::constraint_system::ConstraintSystem).
///
/// Type parameters:
/// - `U` — the UAIR description (authoring trait).
/// - `F` — the projection field (becomes `ConstraintSystem::Field`); fixed on
///   the type so the impl can require the full field-bound bundle.
/// - `D` (= `DEGREE_PLUS_ONE`) / `FD` (folded degree+1) — const generics, on the
///   type per decision 5 of `docs/ucs-refactor-plan.md`.
///
/// Phase 2 will likely extend this to borrow the original trace (needed for
/// booleanity / bit-op virtual materialization) and to carry the
/// `project_scalar` / `project_ideal` / `project_fq_ideal` closures.
pub struct UairFrontend<U: Uair, F, const D: usize, const FD: usize> {
    #[allow(dead_code)] // populated in Phase 2 when the impl lands
    marker: PhantomData<(U, F)>,
}
