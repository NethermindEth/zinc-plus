//! R1CS (Spartan-style) adapter for the
//! [`ConstraintSystem`](crate::constraint_system::ConstraintSystem) seam.
//!
//! [`R1csFrontend`] implements the UCS specialization of R1CS described in the
//! paper (`zinc-plus-paper/UCS/structural_instantiations.tex`): matrices
//! $A, B, C \in R_0^{n \times m}$, a witness vector $z = (1, \text{public},
//! \text{private})$, and the constraint $(Az) \circ (Bz) - (Cz) \in
//! \mathfrak{n}$ component-wise (ideal membership generalizing $= 0$). Unlike
//! UAIR — whose constraints are *local* and fold into a single combined
//! sumcheck — R1CS constraints are *global* (a row of $Az$ touches every entry
//! of $z$), so this frontend proves satisfiability with **Spartan's two
//! sumchecks**: an outer sumcheck over the constraint hypercube reducing to a
//! point $r_x$, then an inner sumcheck reducing $\{\widetilde{Az},
//! \widetilde{Bz}, \widetilde{Cz}\}(r_x)$ to a single witness-evaluation claim
//! $\tilde z(r_y)$, where $r_y = r_0$ is exactly the point the substrate binds
//! via lift-and-project + the Zip+ PCS open.
//!
//! # Axis mapping
//!
//! The substrate binds `num_vars` to the **witness/variable** axis (it commits
//! the witness column MLE over `num_vars` variables and opens it at the
//! returned `r_0`). The R1CS **constraint** axis (`s_x =
//! log2(#constraints)`, the challenge `tau`, and the outer point `r_x`) is
//! entirely internal to [`prove_constraints`](ConstraintSystem::prove_constraints)
//! / [`verify_constraints`](ConstraintSystem::verify_constraints) and never
//! crosses the seam. Hence `num_vars = log2(#witnesses)`, `r_0 = r_y`, and
//! `r_0_fq = vec![]`.
//!
//! # Milestone status
//!
//! This is the **stage-1 scaffolding**: the data types, their
//! [`Transcribable`] serialization, and the [`ConstraintSystem`] impl surface.
//! The three protocol methods are stubbed to return
//! [`ProtocolError::R1cs`](crate::ProtocolError::R1cs); stage 2 fills in the
//! two-sumcheck argument and stage 3 wires it through the full substrate. The
//! frontend is generic over a [`Semiring`] `S` (the matrix-entry type) so it
//! later covers $\mathbb{Z}$, $\mathbb{Z}[X]$, $\mathbb{F}_q$, and
//! $\mathbb{F}_q[X]$; only the $\mathbb{Q}[X]$ / zero-ideal instantiation is
//! exercised in M1.

use crate::{
    MultiDegreeSumcheckProof, ProtocolError,
    constraint_system::{ConstraintEndpoints, ConstraintSystem, Layout},
};
use core::marker::PhantomData;
use crypto_primitives::{HasPrimeFieldConfig, PrimeField, Semiring, SparseMatrix};
use zinc_piop::projections::ProjectedTrace;
use zinc_poly::univariate::dynamic::over_field::DynamicPolynomialF;
use zinc_transcript::traits::{ConstTranscribable, GenTranscribable, Transcribable, Transcript};
use zinc_uair::{
    PublicColumnLayout, TotalColumnLayout, WitnessColumnLayout,
    ideal::{Ideal, IdealCheck},
};
use zinc_utils::mul;

/// The public R1CS index: the three constraint matrices and the public-input
/// count.
///
/// The matrices are the verifier's $O(\text{nnz})$ index (the non-succinct
/// analog of UAIR's [`UairSignature`](zinc_uair::UairSignature)): the verifier
/// evaluates $\tilde A, \tilde B, \tilde C$ at $(r_x, r_y)$ directly from
/// `cells`, so no sparse-matrix (Spark) commitment is needed. Rows index the
/// constraint axis (`num_rows = 2^{s_x}`) and columns the variable axis
/// (`num_cols = 2^{\text{num\_vars}}`); [`SparseMatrix`] is fixed-density, so
/// hand-built matrices pad each row to `density` with `(0, S::zero())` fillers.
///
/// `num_public_inputs` counts the public entries of $z$ **after** the leading
/// constant $1$ at index `0`; the committed witness carries only the private
/// tail, and [`verify_lifted_evals`](ConstraintSystem::verify_lifted_evals)
/// reconciles $\tilde z(r_y) = \tilde z_{\text{pub}}(r_y) + \tilde
/// z_{\text{wit}}(r_y)$ (Spartan-standard public-input binding).
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct R1csInstance<S> {
    /// Left multiplication matrix $A$.
    pub a: SparseMatrix<S>,
    /// Right multiplication matrix $B$.
    pub b: SparseMatrix<S>,
    /// Output matrix $C$.
    pub c: SparseMatrix<S>,
    /// Number of public inputs, excluding the constant $1$ at $z[0]$.
    pub num_public_inputs: usize,
}

/// Adapter exposing an [`R1csInstance`] as a
/// [`ConstraintSystem`](crate::constraint_system::ConstraintSystem).
///
/// Type parameters:
/// - `S` — the matrix-entry [`Semiring`] ($\mathbb{Z}$, $\mathbb{Z}[X]$,
///   $\mathbb{F}_q$, $\mathbb{F}_q[X]$, ...).
/// - `F` — the projection field (becomes [`ConstraintSystem::Field`]); fixed on
///   the type so the impl can require the extra field bounds the argument needs.
///
/// Carries the public [`R1csInstance`] (the verifier index) and the minimal
/// substrate-facing [`Layout`] — a single `int` witness column and no declared
/// primes.
#[derive(Clone, Debug)]
pub struct R1csFrontend<S, F: PrimeField> {
    /// The public R1CS index ($A$, $B$, $C$, public-input count).
    instance: R1csInstance<S>,
    /// The minimal substrate-only layout: one `int` witness column, no primes.
    layout: Layout<F::Integer>,
    _marker: PhantomData<F>,
}

impl<S, F: PrimeField> R1csFrontend<S, F> {
    /// Build an R1CS frontend from its public index.
    ///
    /// The layout is fixed: the full $z$ vector is committed as a single `int`
    /// witness column (public inputs are bound *inside* the argument, not via
    /// substrate public columns), and no $\mathbb{F}_q[X]$ families are declared
    /// (M1 is $\mathbb{Q}[X]$-only).
    pub fn new(instance: R1csInstance<S>) -> Self {
        let layout = Layout::new(
            TotalColumnLayout::new(0, 0, 1),
            PublicColumnLayout::new(0, 0, 0),
            WitnessColumnLayout::new(0, 0, 1),
            Vec::new(),
        );
        Self {
            instance,
            layout,
            _marker: PhantomData,
        }
    }

    /// The public R1CS index this frontend proves against.
    pub fn instance(&self) -> &R1csInstance<S> {
        &self.instance
    }
}

/// The constraint-argument sub-proof produced by [`R1csFrontend`].
///
/// Holds Spartan's two sumcheck transcripts plus the prover-sent evaluations
/// they bind: the outer (degree-3) sumcheck over the constraint hypercube, the
/// claimed $\widetilde{Az}, \widetilde{Bz}, \widetilde{Cz}$ at $r_x$, the inner
/// (degree-2) sumcheck over the variable hypercube, and the claimed
/// $\tilde z(r_y)$.
///
/// The [`GenTranscribable`] / [`Transcribable`] impls delegate the two sumcheck
/// fields to their own serialization and append the four field scalars behind a
/// single field-cfg header; the substrate [`Proof`](crate::Proof) embeds this
/// as its `constraint_proof` field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct R1csConstraintProof<F: PrimeField> {
    /// Outer sumcheck (degree 3): $\sum_x \mathrm{eq}(\tau, x) \cdot
    /// (\widetilde{Az}(x) \cdot \widetilde{Bz}(x) - \widetilde{Cz}(x)) = 0$.
    pub outer_sumcheck: MultiDegreeSumcheckProof<F>,
    /// Prover-sent $\widetilde{Az}(r_x)$.
    pub az_rx: F,
    /// Prover-sent $\widetilde{Bz}(r_x)$.
    pub bz_rx: F,
    /// Prover-sent $\widetilde{Cz}(r_x)$.
    pub cz_rx: F,
    /// Inner sumcheck (degree 2): $\sum_y (r_A \tilde A + r_B \tilde B + r_C
    /// \tilde C)(r_x, y) \cdot \tilde z(y) = r_A \cdot az_{rx} + r_B \cdot
    /// bz_{rx} + r_C \cdot cz_{rx}$.
    pub inner_sumcheck: MultiDegreeSumcheckProof<F>,
    /// Prover-sent $\tilde z(r_y)$ — the full-witness MLE evaluation the
    /// substrate binds via the PCS open.
    pub z_ry: F,
}

impl<F> GenTranscribable for R1csConstraintProof<F>
where
    F: PrimeField,
    F::Integer: ConstTranscribable,
{
    fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
        let (outer_sumcheck, bytes) =
            MultiDegreeSumcheckProof::<F>::read_transcription_bytes_subset(bytes);
        let (inner_sumcheck, bytes) =
            MultiDegreeSumcheckProof::<F>::read_transcription_bytes_subset(bytes);

        // Four scalars behind one shared field-cfg header:
        // [field_cfg][az_rx][bz_rx][cz_rx][z_ry].
        let mod_size = F::Integer::NUM_BYTES;
        let cfg = zinc_transcript::read_field_cfg::<F>(&bytes[..mod_size]);
        let bytes = &bytes[mod_size..];
        let scalars_len = mul!(4, mod_size);
        let scalars = zinc_transcript::read_field_vec_with_cfg::<F>(&bytes[..scalars_len], &cfg);
        let bytes = &bytes[scalars_len..];
        let [az_rx, bz_rx, cz_rx, z_ry] = <[F; 4]>::try_from(scalars)
            .unwrap_or_else(|_| unreachable!("read exactly four R1CS scalars"));

        assert!(bytes.is_empty(), "All bytes should be consumed");
        Self {
            outer_sumcheck,
            az_rx,
            bz_rx,
            cz_rx,
            inner_sumcheck,
            z_ry,
        }
    }

    fn write_transcription_bytes_exact(&self, mut buf: &mut [u8]) {
        buf = self.outer_sumcheck.write_transcription_bytes_subset(buf);
        buf = self.inner_sumcheck.write_transcription_bytes_subset(buf);

        // Four scalars behind one shared field-cfg header (all four share the
        // same modulus; take the cfg from `az_rx`).
        buf = zinc_transcript::append_field_cfg::<F>(buf, &F::modulus(self.az_rx.cfg()));
        buf = zinc_transcript::append_field_vec_lifted::<F>(
            buf,
            &[
                self.az_rx.clone(),
                self.bz_rx.clone(),
                self.cz_rx.clone(),
                self.z_ry.clone(),
            ],
        );
        let _ = buf;
    }
}

impl<F> Transcribable for R1csConstraintProof<F>
where
    F: PrimeField,
    F::Integer: ConstTranscribable,
{
    #[allow(clippy::arithmetic_side_effects)]
    fn get_num_bytes(&self) -> usize {
        MultiDegreeSumcheckProof::<F>::LENGTH_NUM_BYTES
            + self.outer_sumcheck.get_num_bytes()
            + MultiDegreeSumcheckProof::<F>::LENGTH_NUM_BYTES
            + self.inner_sumcheck.get_num_bytes()
            // field-cfg header + four integer-encoded scalars
            + F::Integer::NUM_BYTES
            + 4 * F::Integer::NUM_BYTES
    }
}

/// Opaque frontend verifier tail-state produced by
/// [`R1csFrontend::verify_constraints`] and consumed by
/// [`R1csFrontend::verify_lifted_evals`].
///
/// Carries the prover-claimed $\tilde z(r_y)$ and the verifier-computed public
/// prefix $\tilde z_{\text{pub}}(r_y)$, so the lifted-eval hook can reconcile
/// $\tilde z(r_y) \overset{?}{=} \tilde z_{\text{pub}}(r_y) + \tilde
/// z_{\text{wit}}(r_y)$ against the substrate-assembled witness eval at $r_0$.
// Stage 1: constructed only in stage 2's `verify_constraints`; the fields are
// read in stage 2's `verify_lifted_evals`. Suppress dead-code until then.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct R1csVerifierClaims<F: PrimeField> {
    /// Prover-claimed $\tilde z(r_y)$ (the full-witness MLE eval).
    z_ry_claimed: F,
    /// Verifier-computed public prefix $\tilde z_{\text{pub}}(r_y)$.
    z_pub_ry: F,
}

impl<S, F> ConstraintSystem for R1csFrontend<S, F>
where
    S: Semiring,
    F: PrimeField,
    F::Integer: ConstTranscribable,
{
    type Prime = F::Integer;
    type Field = F;
    type ConstraintProof = R1csConstraintProof<F>;
    type VerifierClaims = R1csVerifierClaims<F>;
    // R1CS over the zero ideal (M1): no ideal-membership check, no `psi_a`
    // scalar projection. The three `project_*` closures on `verify_constraints`
    // are never called and `IdealOverF` is never instantiated, so these source
    // types are all `()` and the `IdealOverF: Ideal + IdealCheck` bound is
    // satisfied vacuously.
    type IdealSource = ();
    type FqIdealSource = ();
    type Scalar = ();

    fn layout(&self) -> &Layout<Self::Prime> {
        &self.layout
    }

    fn prove_constraints(
        &self,
        _transcript: &mut impl Transcript,
        _projected_traces: &[ProjectedTrace<Self::Field>],
        _field_cfgs: &[<Self::Field as HasPrimeFieldConfig>::Config],
        _num_vars: usize,
    ) -> Result<(Self::ConstraintProof, ConstraintEndpoints<Self::Field>), ProtocolError<Self::Field>>
    {
        // Stage 2: outer (degree-3) + inner (degree-2) Spartan sumchecks.
        Err(ProtocolError::R1cs(
            "prove_constraints: not yet implemented (stage 2)".to_owned(),
        ))
    }

    fn verify_constraints<IdealOverF>(
        &self,
        _transcript: &mut impl Transcript,
        _proof: &Self::ConstraintProof,
        _field_cfgs: &[<Self::Field as HasPrimeFieldConfig>::Config],
        _num_vars: usize,
        _project_ideal: impl Fn(
            &Self::IdealSource,
            &<Self::Field as HasPrimeFieldConfig>::Config,
        ) -> IdealOverF,
        _project_fq_ideal: impl Fn(
            &Self::FqIdealSource,
            &<Self::Field as HasPrimeFieldConfig>::Config,
        ) -> IdealOverF,
        _project_scalar: impl Fn(
            &Self::Scalar,
            &<Self::Field as HasPrimeFieldConfig>::Config,
        ) -> DynamicPolynomialF<Self::Field>,
    ) -> Result<(ConstraintEndpoints<Self::Field>, Self::VerifierClaims), ProtocolError<Self::Field>>
    where
        IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<Self::Field>>,
    {
        // Stage 2: replay both sumchecks, evaluate A/B/C(r_x, r_y) in O(nnz),
        // and compute the public prefix z_pub(r_y).
        Err(ProtocolError::R1cs(
            "verify_constraints: not yet implemented (stage 2)".to_owned(),
        ))
    }

    fn verify_lifted_evals(
        &self,
        _claims: &Self::VerifierClaims,
        _per_family_all_lifted: &[Vec<DynamicPolynomialF<Self::Field>>],
        _field_cfgs: &[<Self::Field as HasPrimeFieldConfig>::Config],
    ) -> Result<(), ProtocolError<Self::Field>> {
        // Stage 2: reconcile z(r_y) == z_pub(r_y) + z_wit(r_y).
        Err(ProtocolError::R1cs(
            "verify_lifted_evals: not yet implemented (stage 2)".to_owned(),
        ))
    }
}
