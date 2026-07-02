//! R1CS (Spartan-style) adapter for the
//! [`ConstraintSystem`](crate::constraint_system::ConstraintSystem) seam,
//! shaped after ProveKit's plain-field R1CS (`provekit/.../r1cs.rs`).
//!
//! [`R1csFrontend`] proves R1CS over a **fixed prime field** $F$: matrices
//! $A, B, C \in F^{n \times m}$, a witness vector $z = (1, \text{public},
//! \text{private})$, and the exact constraint $(Az) \circ (Bz) - (Cz) = 0$
//! component-wise (the **zero ideal**). This is the Zinc+ "add-on" /
//! $R = \mathbb{F}_q$ case (`zinc-plus-paper/crypto/crypto_piop.tex`): the
//! relation is native over $F$, so the substrate uses $F$ directly as its
//! working field (via [`ConstraintSystem::working_field`]) rather than sampling
//! a random prime. Soundness needs $F$ to be a large prime
//! ($|F| = \Omega(2^\lambda)$); see `docs/r1cs-frontend-plan.md`. Ideal
//! membership / polynomial witnesses / multi-prime families are **postponed**
//! (kept in that doc's "Postponed" section).
//!
//! Unlike UAIR — whose constraints are *local* and fold into a single combined
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
//! entirely internal to
//! [`prove_constraints`](ConstraintSystem::prove_constraints)
//! / [`verify_constraints`](ConstraintSystem::verify_constraints) and never
//! crosses the seam. Hence `num_vars = log2(#witnesses)`, `r_0 = r_y`, and
//! `r_0_fq = vec![]`.

use crate::{
    MultiDegreeSumcheckProof, ProtocolError,
    constraint_system::{ConstraintEndpoints, ConstraintSystem, Layout},
    r1cs_sparse_matrix::SparseMatrix,
};
use crypto_primitives::{FromPrimitiveWithConfig, HasPrimeFieldConfig, PrimeField};
use num_traits::Zero;
use zinc_piop::{
    CombFn,
    projections::ProjectedTrace,
    sumcheck::multi_degree::{MultiDegreeSumcheck, MultiDegreeSumcheckGroup},
};
use zinc_poly::{
    EvaluatablePolynomial,
    mle::DenseMultilinearExtension,
    univariate::dynamic::over_field::DynamicPolynomialF,
    utils::{build_eq_x_r_inner, build_eq_x_r_vec, eq_eval, mle_eval_with_eq_table},
};
use zinc_transcript::traits::{ConstTranscribable, GenTranscribable, Transcribable, Transcript};
use zinc_uair::{
    PublicColumnLayout, TotalColumnLayout, WitnessColumnLayout,
    ideal::{Ideal, IdealCheck},
};
use zinc_utils::{inner_transparent_field::InnerTransparentField, mul};

/// The public R1CS index: the three constraint matrices and the public-input
/// count, shaped after ProveKit's `R1CS` (`a/b/c`, `num_public_inputs`).
///
/// The matrices are the verifier's $O(\text{nnz})$ index (the non-succinct
/// analog of UAIR's [`UairSignature`](zinc_uair::UairSignature)): the verifier
/// evaluates $\tilde A, \tilde B, \tilde C$ at $(r_x, r_y)$ directly from the
/// entries, so no sparse-matrix (Spark) commitment is needed. Rows index the
/// constraint axis (`num_rows = 2^{s_x}`) and columns the variable axis
/// (`num_cols = 2^{\text{num\_vars}}`). Entries are **field elements** in the
/// fixed R1CS field `F` (ProveKit stores field elements; we mirror that), held
/// in our variable-density
/// [`SparseMatrix`](crate::r1cs_sparse_matrix::SparseMatrix).
///
/// `num_public_inputs` counts the public entries of $z$ **after** the leading
/// constant $1$ at index `0`; the committed witness carries only the private
/// tail, and [`verify_lifted_evals`](ConstraintSystem::verify_lifted_evals)
/// reconciles $\tilde z(r_y) = \tilde z_{\text{pub}}(r_y) + \tilde
/// z_{\text{wit}}(r_y)$ (Spartan-standard public-input binding).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct R1csInstance<F: PrimeField> {
    /// Left multiplication matrix $A$.
    pub a: SparseMatrix<F>,
    /// Right multiplication matrix $B$.
    pub b: SparseMatrix<F>,
    /// Output matrix $C$.
    pub c: SparseMatrix<F>,
    /// Number of public inputs, excluding the constant $1$ at $z[0]$.
    pub num_public_inputs: usize,
}

/// Adapter exposing an [`R1csInstance`] as a
/// [`ConstraintSystem`](crate::constraint_system::ConstraintSystem).
///
/// `F` — the **fixed** prime field the R1CS lives over (becomes
/// [`ConstraintSystem::Field`]). The relation is native over `F` (ProveKit's
/// `R = F_q` / Zinc+ "add-on" case): the frontend supplies `F` to the substrate
/// via [`ConstraintSystem::working_field`], so the substrate uses `F` as its
/// working field instead of sampling a random prime. `F` must be a large prime
/// for soundness (`docs/r1cs-frontend-plan.md`).
///
/// Carries the public [`R1csInstance`] (the verifier index), the public-input
/// values, the fixed field cfg, and the minimal substrate-facing [`Layout`] —
/// a single `int` witness column and no declared primes.
#[derive(Clone, Debug)]
pub struct R1csFrontend<F: PrimeField> {
    /// The public R1CS index ($A$, $B$, $C$, public-input count).
    instance: R1csInstance<F>,
    /// The public-input values $[\text{io}_1, \dots, \text{io}_\ell]$ (the
    /// `io` part of $z$, **excluding** the constant $1$ at $z[0]$). Length must
    /// equal `instance.num_public_inputs`. Part of the public statement, so
    /// both prover and verifier hold it; the seam does not pass it, so it lives
    /// on the frontend. The full public prefix of $z$ is $[1, \text{io}_1,
    /// \dots, \text{io}_\ell]$ occupying indices $0..=\ell$.
    public_values: Vec<F>,
    /// The fixed R1CS field, handed to the substrate via `working_field`.
    field_cfg: F::Config,
    /// The minimal substrate-only layout: one `int` witness column, no primes.
    layout: Layout<F::Integer>,
}

impl<F: PrimeField> R1csFrontend<F> {
    /// Build an R1CS frontend from its public index, public-input values, and
    /// fixed field.
    ///
    /// `public_values` are the $\ell = $ `instance.num_public_inputs` public
    /// inputs (the `io` of $z$, excluding the constant $1$ at $z[0]$); the full
    /// public prefix is $[1] \mathbin{+\!+} \text{public\_values}$. The witness
    /// (private) part of $z$ is committed as a single `int` column (with the
    /// public prefix slots zeroed in the committed column); public inputs are
    /// bound *inside* the argument, not via substrate public columns.
    /// `field_cfg` is the fixed prime field the relation lives over (and
    /// that the substrate will use as its working field).
    ///
    /// # Panics
    /// If `public_values.len() != instance.num_public_inputs`.
    pub fn new(instance: R1csInstance<F>, public_values: Vec<F>, field_cfg: F::Config) -> Self {
        assert_eq!(
            public_values.len(),
            instance.num_public_inputs,
            "public_values length must equal instance.num_public_inputs",
        );
        let layout = Layout::new(
            TotalColumnLayout::new(0, 0, 1),
            PublicColumnLayout::new(0, 0, 0),
            WitnessColumnLayout::new(0, 0, 1),
            Vec::new(),
        );
        Self {
            instance,
            public_values,
            field_cfg,
            layout,
        }
    }

    /// The public R1CS index this frontend proves against.
    pub fn instance(&self) -> &R1csInstance<F> {
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
#[derive(Clone, Debug)]
pub struct R1csVerifierClaims<F: PrimeField> {
    /// Prover-claimed $\tilde z(r_y)$ (the full-witness MLE eval).
    z_ry_claimed: F,
    /// Verifier-computed public prefix $\tilde z_{\text{pub}}(r_y)$.
    z_pub_ry: F,
}

impl<F> ConstraintSystem for R1csFrontend<F>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync + 'static,
    F::Integer: ConstTranscribable + Zero,
{
    type Prime = F::Integer;
    type Field = F;
    type ConstraintProof = R1csConstraintProof<F>;
    type VerifierClaims = R1csVerifierClaims<F>;
    // R1CS over the zero ideal: no ideal-membership check, no `psi_a` scalar
    // projection. The three `project_*` closures on `verify_constraints` are
    // never called and `IdealOverF` is never instantiated, so these source
    // types are all `()` and the `IdealOverF: Ideal + IdealCheck` bound is
    // satisfied vacuously.
    type IdealSource = ();
    type FqIdealSource = ();
    type Scalar = ();

    fn layout(&self) -> &Layout<Self::Prime> {
        &self.layout
    }

    /// The relation is native over the fixed field `F`; hand it to the
    /// substrate so it uses `F` as the working field rather than sampling a
    /// random prime.
    fn working_field(&self) -> Option<<Self::Field as HasPrimeFieldConfig>::Config> {
        Some(self.field_cfg.clone())
    }

    /// Prover side: Spartan's two sumchecks over the fixed field $F$.
    ///
    /// First, assemble the full witness $z = z_{\text{pub}} + z_{\text{wit}}$
    /// (the committed witness from `projected_traces[0]` plus the public prefix
    /// $[1, \text{io}]$).
    ///
    /// The **outer** sumcheck (degree 3) squeezes $\tau$ (length $s_x$) and
    /// proves $\sum_x \mathrm{eq}(\tau, x)(\widetilde{Az} \cdot \widetilde{Bz}
    /// - \widetilde{Cz})(x) = 0$ over the constraint hypercube, yielding
    /// $r_x$ internally; it then sends $\widetilde{Az}, \widetilde{Bz},
    /// \widetilde{Cz}$ at $r_x$.
    ///
    /// The **inner** sumcheck (degree 2) binds those three evals, squeezes
    /// $r_A, r_B, r_C$, and proves $\sum_y m_{\text{row}}(y) \tilde z(y) = r_A
    /// az_{rx} + r_B bz_{rx} + r_C cz_{rx}$ over the variable hypercube,
    /// yielding $r_y = r_0$; it sends $\tilde z(r_y)$.
    #[allow(
        clippy::arithmetic_side_effects,
        clippy::too_many_lines,
        clippy::needless_range_loop
    )]
    fn prove_constraints(
        &self,
        transcript: &mut impl Transcript,
        projected_traces: &[ProjectedTrace<Self::Field>],
        field_cfgs: &[<Self::Field as HasPrimeFieldConfig>::Config],
        num_vars: usize,
    ) -> Result<(Self::ConstraintProof, ConstraintEndpoints<Self::Field>), ProtocolError<Self::Field>>
    {
        let instance = &self.instance;
        let cfg = &field_cfgs[0];
        let s_x = constraint_axis_vars::<F>(instance)?;
        let z_len = 1usize << num_vars;

        // ---- assemble the full witness z = z_pub + z_wit -------------------
        let trace = projected_traces
            .first()
            .ok_or_else(|| ProtocolError::R1cs("no projected trace for R1CS witness".to_owned()))?;
        let mut z_full = witness_column_scalars::<F>(trace, cfg)?;
        if z_full.len() != z_len {
            return Err(ProtocolError::R1cs(format!(
                "witness column length {} != 2^num_vars = {z_len}",
                z_full.len()
            )));
        }
        // Public prefix: z[0] = 1, z[1..=ell] = public inputs. The committed
        // witness column is zero in these slots, so we add the public values.
        let one = F::one_with_cfg(cfg);
        z_full[0] += &one;
        for (k, iv) in self.public_values.iter().enumerate() {
            z_full[k + 1] += iv;
        }

        // ---- outer sumcheck (degree 3) -------------------------------------
        // Squeeze tau BEFORE the sumcheck absorbs its own metadata (FS order).
        let tau = transcript.get_field_challenges::<F>(s_x, cfg);
        let eq_tau = build_eq_x_r_inner(&tau, cfg)
            .map_err(|e| ProtocolError::R1cs(format!("eq(tau, .) build failed: {e}")))?;

        let az_inner = mz_inner::<F>(&instance.a, &z_full, cfg);
        let bz_inner = mz_inner::<F>(&instance.b, &z_full, cfg);
        let cz_inner = mz_inner::<F>(&instance.c, &z_full, cfg);

        let zero_inner = F::zero_with_cfg(cfg).into_inner();
        // eq_tau * (Az * Bz - Cz)
        let outer_comb: CombFn<F> = Box::new(|v: &[F]| {
            let az_bz = v[1].clone() * &v[2];
            v[0].clone() * &(az_bz - &v[3])
        });
        let outer_group = MultiDegreeSumcheckGroup::new(
            3,
            vec![
                eq_tau,
                DenseMultilinearExtension::from_evaluations_vec(
                    s_x,
                    az_inner.clone(),
                    zero_inner.clone(),
                ),
                DenseMultilinearExtension::from_evaluations_vec(
                    s_x,
                    bz_inner.clone(),
                    zero_inner.clone(),
                ),
                DenseMultilinearExtension::from_evaluations_vec(
                    s_x,
                    cz_inner.clone(),
                    zero_inner.clone(),
                ),
            ],
            outer_comb,
        );
        let (outer_sumcheck, outer_states) = MultiDegreeSumcheck::<F>::prove_as_subprotocol(
            transcript,
            vec![(vec![outer_group], cfg)],
            s_x,
            cfg,
        )
        .pop()
        .expect("single family");

        let r_x = outer_states[0].randomness.clone();
        let eq_rx = build_eq_x_r_vec(&r_x, cfg)
            .map_err(|e| ProtocolError::R1cs(format!("eq(r_x, .) build failed: {e}")))?;
        let az_rx = mle_eval_with_eq_table::<F>(&az_inner, &eq_rx, cfg);
        let bz_rx = mle_eval_with_eq_table::<F>(&bz_inner, &eq_rx, cfg);
        let cz_rx = mle_eval_with_eq_table::<F>(&cz_inner, &eq_rx, cfg);

        // ---- bind Az/Bz/Cz(r_x), then squeeze r_A, r_B, r_C ----------------
        let mut buf = vec![0u8; F::Integer::NUM_BYTES];
        transcript.absorb_random_field(&az_rx, &mut buf);
        transcript.absorb_random_field(&bz_rx, &mut buf);
        transcript.absorb_random_field(&cz_rx, &mut buf);
        let batch = transcript.get_field_challenges::<F>(3, cfg);
        let (r_a, r_b, r_c) = (batch[0].clone(), batch[1].clone(), batch[2].clone());

        // ---- inner sumcheck (degree 2) -------------------------------------
        // m_row[j] = sum_i eq_rx[i] * (r_A A + r_B B + r_C C)[i][j].
        let mut m_row = vec![F::zero_with_cfg(cfg); z_len];
        for (matrix, coeff) in [
            (&instance.a, &r_a),
            (&instance.b, &r_b),
            (&instance.c, &r_c),
        ] {
            for i in 0..matrix.num_rows {
                let w = coeff.clone() * &eq_rx[i];
                for (j, val) in matrix.iter_row(i) {
                    let term = w.clone() * val;
                    m_row[j] += &term;
                }
            }
        }
        let m_row_inner: Vec<F::Inner> = m_row.into_iter().map(F::into_inner).collect();
        let z_full_inner: Vec<F::Inner> = z_full.iter().map(|f| f.inner().clone()).collect();

        let inner_comb: CombFn<F> = Box::new(|v: &[F]| v[0].clone() * &v[1]);
        let inner_group = MultiDegreeSumcheckGroup::new(
            2,
            vec![
                DenseMultilinearExtension::from_evaluations_vec(
                    num_vars,
                    m_row_inner,
                    zero_inner.clone(),
                ),
                DenseMultilinearExtension::from_evaluations_vec(
                    num_vars,
                    z_full_inner.clone(),
                    zero_inner,
                ),
            ],
            inner_comb,
        );
        let (inner_sumcheck, inner_states) = MultiDegreeSumcheck::<F>::prove_as_subprotocol(
            transcript,
            vec![(vec![inner_group], cfg)],
            num_vars,
            cfg,
        )
        .pop()
        .expect("single family");

        let r_y = inner_states[0].randomness.clone();
        let eq_ry = build_eq_x_r_vec(&r_y, cfg)
            .map_err(|e| ProtocolError::R1cs(format!("eq(r_y, .) build failed: {e}")))?;
        let z_ry = mle_eval_with_eq_table::<F>(&z_full_inner, &eq_ry, cfg);

        let proof = R1csConstraintProof {
            outer_sumcheck,
            az_rx,
            bz_rx,
            cz_rx,
            inner_sumcheck,
            z_ry,
        };
        Ok((proof, ConstraintEndpoints::new(r_y, Vec::new())))
    }

    /// Verifier side: replay both sumchecks in the identical FS order, evaluate
    /// $\tilde A, \tilde B, \tilde C$ at $(r_x, r_y)$ directly in
    /// $O(\text{nnz})$, and compute the public prefix $\tilde
    /// z_{\text{pub}}(r_y)$ (carried on [`R1csVerifierClaims`] to
    /// [`verify_lifted_evals`](Self::verify_lifted_evals)).
    #[allow(clippy::arithmetic_side_effects, clippy::too_many_lines)]
    fn verify_constraints<IdealOverF>(
        &self,
        transcript: &mut impl Transcript,
        proof: &Self::ConstraintProof,
        field_cfgs: &[<Self::Field as HasPrimeFieldConfig>::Config],
        num_vars: usize,
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
        let instance = &self.instance;
        let cfg = &field_cfgs[0];
        let s_x = constraint_axis_vars::<F>(instance)?;

        // ---- outer sumcheck replay -----------------------------------------
        let tau = transcript.get_field_challenges::<F>(s_x, cfg);
        let outer_subclaims = MultiDegreeSumcheck::<F>::verify_as_subprotocol(
            transcript,
            s_x,
            &[(&proof.outer_sumcheck, cfg)],
            cfg,
        )
        .map_err(|e| ProtocolError::R1cs(format!("outer sumcheck verify failed: {e}")))?;
        let outer = &outer_subclaims[0];

        // Outer claimed sum must be 0: Az * Bz - Cz vanishes on the hypercube.
        let zero = F::zero_with_cfg(cfg);
        if proof.outer_sumcheck.claimed_sums()[0] != zero {
            return Err(ProtocolError::R1cs(
                "outer sumcheck claimed sum is nonzero (R1CS not satisfied)".to_owned(),
            ));
        }
        let r_x = outer.point().to_vec();

        // Outer consistency: expected == eq(tau, r_x) * (az_rx * bz_rx - cz_rx).
        let one = F::one_with_cfg(cfg);
        let eq_tau_rx = eq_eval(&tau, &r_x, one)
            .map_err(|e| ProtocolError::R1cs(format!("eq(tau, r_x) failed: {e}")))?;
        let az_bz = proof.az_rx.clone() * &proof.bz_rx;
        let outer_expected = eq_tau_rx * &(az_bz - &proof.cz_rx);
        if outer.expected_evaluations()[0] != outer_expected {
            return Err(ProtocolError::R1cs(
                "outer sumcheck final evaluation mismatch".to_owned(),
            ));
        }

        // ---- bind Az/Bz/Cz(r_x), squeeze r_A, r_B, r_C ---------------------
        let mut buf = vec![0u8; F::Integer::NUM_BYTES];
        transcript.absorb_random_field(&proof.az_rx, &mut buf);
        transcript.absorb_random_field(&proof.bz_rx, &mut buf);
        transcript.absorb_random_field(&proof.cz_rx, &mut buf);
        let batch = transcript.get_field_challenges::<F>(3, cfg);
        let (r_a, r_b, r_c) = (batch[0].clone(), batch[1].clone(), batch[2].clone());

        // ---- inner sumcheck replay -----------------------------------------
        let inner_subclaims = MultiDegreeSumcheck::<F>::verify_as_subprotocol(
            transcript,
            num_vars,
            &[(&proof.inner_sumcheck, cfg)],
            cfg,
        )
        .map_err(|e| ProtocolError::R1cs(format!("inner sumcheck verify failed: {e}")))?;
        let inner = &inner_subclaims[0];

        // Inner claimed sum must equal r_A az_rx + r_B bz_rx + r_C cz_rx (this
        // binds the prover-sent evals to the outer sumcheck's reduction).
        let inner_claimed = r_a.clone() * &proof.az_rx
            + &(r_b.clone() * &proof.bz_rx)
            + &(r_c.clone() * &proof.cz_rx);
        if proof.inner_sumcheck.claimed_sums()[0] != inner_claimed {
            return Err(ProtocolError::R1cs(
                "inner sumcheck claimed sum mismatch (Az/Bz/Cz(r_x) not bound)".to_owned(),
            ));
        }
        let r_y = inner.point().to_vec();

        // Inner consistency: expected == M~(r_x, r_y) * z_ry with
        // M = r_A A + r_B B + r_C C, evaluated directly in O(nnz).
        let eq_rx = build_eq_x_r_vec(&r_x, cfg)
            .map_err(|e| ProtocolError::R1cs(format!("eq(r_x, .) build failed: {e}")))?;
        let eq_ry = build_eq_x_r_vec(&r_y, cfg)
            .map_err(|e| ProtocolError::R1cs(format!("eq(r_y, .) build failed: {e}")))?;
        let a_eval = matrix_mle_eval::<F>(&instance.a, &eq_rx, &eq_ry, cfg);
        let b_eval = matrix_mle_eval::<F>(&instance.b, &eq_rx, &eq_ry, cfg);
        let c_eval = matrix_mle_eval::<F>(&instance.c, &eq_rx, &eq_ry, cfg);
        let m_eval = r_a * &a_eval + &(r_b * &b_eval) + &(r_c * &c_eval);
        if inner.expected_evaluations()[0] != m_eval * &proof.z_ry {
            return Err(ProtocolError::R1cs(
                "inner sumcheck final evaluation mismatch (matrix MLE)".to_owned(),
            ));
        }

        // ---- public prefix: z_pub(r_y) = eq_ry[0]*1 + sum_k eq_ry[k+1]*io[k] --
        let mut z_pub_ry = eq_ry[0].clone();
        for (k, iv) in self.public_values.iter().enumerate() {
            z_pub_ry += &(eq_ry[k + 1].clone() * iv);
        }

        Ok((
            ConstraintEndpoints::new(r_y, Vec::new()),
            R1csVerifierClaims {
                z_ry_claimed: proof.z_ry.clone(),
                z_pub_ry,
            },
        ))
    }

    /// Verifier hook: reconcile the prover-claimed $\tilde z(r_y)$ against the
    /// substrate-assembled witness eval, $\tilde z(r_y) \overset{?}{=} \tilde
    /// z_{\text{pub}}(r_y) + \tilde z_{\text{wit}}(r_y)$.
    #[allow(clippy::arithmetic_side_effects)]
    fn verify_lifted_evals(
        &self,
        claims: &Self::VerifierClaims,
        per_family_all_lifted: &[Vec<DynamicPolynomialF<Self::Field>>],
        field_cfgs: &[<Self::Field as HasPrimeFieldConfig>::Config],
    ) -> Result<(), ProtocolError<Self::Field>> {
        let cfg = &field_cfgs[0];
        let family = per_family_all_lifted
            .first()
            .ok_or_else(|| ProtocolError::R1cs("no lifted evals for the R1CS family".to_owned()))?;
        let z_wit = family.first().ok_or_else(|| {
            ProtocolError::R1cs("R1CS family has no witness-column lift".to_owned())
        })?;
        // The int-column lift is degree-0, so its value at any point is the
        // scalar z_wit(r_y) (confirmed against the substrate; no psi_a
        // projecting element is applied for the zero ideal).
        let z_wit_ry = z_wit
            .evaluate_at_point(&F::zero_with_cfg(cfg))
            .map_err(|e| ProtocolError::R1cs(format!("witness lift eval failed: {e}")))?;

        if claims.z_ry_claimed != claims.z_pub_ry.clone() + &z_wit_ry {
            return Err(ProtocolError::R1cs(
                "witness eval reconciliation failed: z(r_y) != z_pub(r_y) + z_wit(r_y)".to_owned(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Frontend-internal helpers
// ---------------------------------------------------------------------------

/// Constraint-axis variable count $s_x = \log_2(\text{num\\_rows})$, validating
/// that $A, B, C$ share a row count that is a power of two $\ge 2$ (the outer
/// sumcheck needs $s_x \ge 1$).
fn constraint_axis_vars<F: PrimeField>(
    instance: &R1csInstance<F>,
) -> Result<usize, ProtocolError<F>> {
    let n = instance.a.num_rows;
    if n < 2 || !n.is_power_of_two() {
        return Err(ProtocolError::R1cs(format!(
            "R1CS constraint count must be a power of two >= 2, got {n}"
        )));
    }
    if instance.b.num_rows != n || instance.c.num_rows != n {
        return Err(ProtocolError::R1cs(
            "A, B, C must share the same row count".to_owned(),
        ));
    }
    Ok(n.trailing_zeros() as usize)
}

/// Extract the single `int` witness column from a projected trace as scalars in
/// `F`. The committed witness entries are degree-0 (the substrate's projection
/// of an int column), so each `DynamicPolynomialF<F>` collapses to its constant
/// term (its value at $0$).
fn witness_column_scalars<F: PrimeField>(
    trace: &ProjectedTrace<F>,
    cfg: &F::Config,
) -> Result<Vec<F>, ProtocolError<F>> {
    let zero = F::zero_with_cfg(cfg);
    let to_scalar = |p: &DynamicPolynomialF<F>| {
        p.evaluate_at_point(&zero)
            .map_err(|e| ProtocolError::R1cs(format!("witness column projection failed: {e}")))
    };
    match trace {
        ProjectedTrace::ColumnMajor(cols) => {
            let col = cols
                .first()
                .ok_or_else(|| ProtocolError::R1cs("projected trace has no columns".to_owned()))?;
            col.evaluations.iter().map(to_scalar).collect()
        }
        ProjectedTrace::RowMajor(rows) => rows
            .iter()
            .map(|row| {
                let p = row.first().ok_or_else(|| {
                    ProtocolError::R1cs("projected trace row is empty".to_owned())
                })?;
                to_scalar(p)
            })
            .collect(),
    }
}

/// Compute the length-`num_rows` vector `M z` as `F::Inner` evaluations, with
/// field-element matrix entries multiplied by the (field) witness.
#[allow(clippy::arithmetic_side_effects)]
fn mz_inner<F>(matrix: &SparseMatrix<F>, z_full: &[F], cfg: &F::Config) -> Vec<F::Inner>
where
    F: InnerTransparentField,
{
    let zero = F::zero_with_cfg(cfg);
    (0..matrix.num_rows)
        .map(|i| {
            let mut acc = zero.clone();
            for (j, val) in matrix.iter_row(i) {
                acc += &(val.clone() * &z_full[j]);
            }
            acc.into_inner()
        })
        .collect()
}

/// Evaluate the matrix MLE $\tilde M(r_x, r_y) = \sum_{(i, j)}
/// \mathrm{eq}_{r_x}[i] \cdot \mathrm{eq}_{r_y}[j] \cdot M[i][j]$ directly in
/// $O(\text{nnz})$ (field-element entries).
#[allow(clippy::arithmetic_side_effects)]
fn matrix_mle_eval<F>(matrix: &SparseMatrix<F>, eq_rx: &[F], eq_ry: &[F], cfg: &F::Config) -> F
where
    F: PrimeField,
{
    let mut acc = F::zero_with_cfg(cfg);
    for (i, j, val) in matrix.iter() {
        let mut term = eq_rx[i].clone() * &eq_ry[j];
        term *= val;
        acc += &term;
    }
    acc
}

// ---------------------------------------------------------------------------
// Tests (frontend-only: exercises the two-sumcheck argument in isolation, with
// a hand-assembled `per_family_all_lifted` standing in for the substrate).
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crypto_bigint::U64;
    use crypto_primitives::{
        Field, FromWithConfig, crypto_bigint_monty::MontyField, crypto_bigint_uint::Uint,
    };
    use zinc_transcript::Blake3Transcript;
    use zinc_uair::{ideal::DegreeOneIdeal, ideal_collector::IdealOrZero};

    const FIELD_LIMBS: usize = U64::LIMBS * 3;
    type F = MontyField<FIELD_LIMBS>;
    type FMod = Uint<FIELD_LIMBS>;
    type Cfg = <F as HasPrimeFieldConfig>::Config;

    fn test_cfg() -> Cfg {
        // A 64-bit prime (same one used across the shared-challenge tests).
        F::make_cfg(&FMod::from(0xFFFF_FFFF_FFFF_FFC5_u64)).expect("prime modulus")
    }

    // R1CS uses the zero ideal; these projections are never invoked.
    fn no_ideal(_: &(), _: &Cfg) -> IdealOrZero<DegreeOneIdeal<F>> {
        unreachable!("R1CS uses the zero ideal; projection never invoked")
    }
    fn no_scalar(_: &(), _: &Cfg) -> DynamicPolynomialF<F> {
        unreachable!("R1CS has no psi_a scalar projection")
    }

    /// The satisfiable instance `z = [1, x, w]` with the single (nontrivial)
    /// constraint `x * x = w`, padded to `num_vars = 2` (z-length 4) and to
    /// `s_x = 1` (two constraints; the second is the trivial `0 * 0 = 0`).
    ///
    /// `num_public` decides how much of the public prefix `[1, x]` is treated
    /// as verifier-known `io` (the constant `1` at index 0 is always
    /// public): the committed witness column zeroes indices
    /// `0..=num_public`, and `public_values` carries `z[1..=num_public]`.
    struct Setup {
        cfg: Cfg,
        instance: R1csInstance<F>,
        public_values: Vec<F>,
        projected: Vec<ProjectedTrace<F>>,
        /// The committed witness column in `F` (public slots zeroed).
        z_wit: Vec<F>,
        num_vars: usize,
    }

    fn build(x: i64, w: i64, num_public: usize) -> Setup {
        let cfg = test_cfg();
        let num_vars = 2usize;
        let n = 1usize << num_vars; // z-length 4
        let f = |v: i64| F::from_with_cfg(&v, &cfg);
        let zero = F::zero_with_cfg(&cfg);

        let z_full = [f(1), f(x), f(w), zero.clone()];

        // Committed witness: zero the public prefix [0..=num_public].
        let mut z_wit: Vec<F> = z_full.to_vec();
        for slot in z_wit.iter_mut().take(num_public + 1) {
            *slot = zero.clone();
        }
        let public_values: Vec<F> = z_full[1..=num_public].to_vec();

        // Projected trace: one column-major `int` witness column of degree-0
        // (constant) DynamicPolynomialF entries.
        let col = DenseMultilinearExtension {
            num_vars,
            evaluations: z_wit
                .iter()
                .map(|v| DynamicPolynomialF::constant_poly(v.clone()))
                .collect(),
        };
        let projected = vec![ProjectedTrace::ColumnMajor(vec![col])];

        // A z = x, B z = x, C z = w; second row is the trivial 0 = 0 constraint.
        let mat =
            |col: usize, val: F| SparseMatrix::from_rows(n, vec![vec![(col, val)], Vec::new()]);
        let instance = R1csInstance {
            a: mat(1, f(1)),
            b: mat(1, f(1)),
            c: mat(2, f(1)),
            num_public_inputs: num_public,
        };

        Setup {
            cfg,
            instance,
            public_values,
            projected,
            z_wit,
            num_vars,
        }
    }

    /// Assemble the substrate's `per_family_all_lifted[0]` = the degree-0 lift
    /// of the committed witness column at `r_y`: `sum_b eq(b, r_y) z_wit(b)`.
    fn hand_lifted(z_wit: &[F], r_y: &[F], cfg: &Cfg) -> Vec<Vec<DynamicPolynomialF<F>>> {
        let z_wit_inner: Vec<<F as Field>::Inner> = z_wit.iter().map(|v| *v.inner()).collect();
        let eq_ry = build_eq_x_r_vec(r_y, cfg).unwrap();
        let z_wit_ry = mle_eval_with_eq_table::<F>(&z_wit_inner, &eq_ry, cfg);
        vec![vec![DynamicPolynomialF::constant_poly(z_wit_ry)]]
    }

    fn prove(setup: &Setup) -> (R1csConstraintProof<F>, Vec<F>) {
        let frontend = R1csFrontend::<F>::new(
            setup.instance.clone(),
            setup.public_values.clone(),
            setup.cfg,
        );
        let mut pt = Blake3Transcript::new();
        let (proof, endpoints) = frontend
            .prove_constraints(&mut pt, &setup.projected, &[setup.cfg], setup.num_vars)
            .expect("prove_constraints");
        (proof, endpoints.r_0)
    }

    /// End-to-end (frontend-only): prove then verify + reconcile the witness
    /// eval against a hand-assembled substrate lift, for the constant-only
    /// public prefix (`num_public = 0`).
    #[test]
    fn r1cs_prove_verify_no_public_inputs() {
        let setup = build(3, 9, 0);
        let (proof, r_y_prover) = prove(&setup);

        let frontend = R1csFrontend::<F>::new(
            setup.instance.clone(),
            setup.public_values.clone(),
            setup.cfg,
        );
        let mut vt = Blake3Transcript::new();
        let (endpoints, claims) = frontend
            .verify_constraints::<IdealOrZero<DegreeOneIdeal<F>>>(
                &mut vt,
                &proof,
                &[setup.cfg],
                setup.num_vars,
                no_ideal,
                no_ideal,
                no_scalar,
            )
            .expect("verify_constraints");
        assert_eq!(endpoints.r_0, r_y_prover, "prover/verifier r_y must agree");

        let lifted = hand_lifted(&setup.z_wit, &endpoints.r_0, &setup.cfg);
        frontend
            .verify_lifted_evals(&claims, &lifted, &[setup.cfg])
            .expect("verify_lifted_evals");
    }

    /// Same, but with one genuine public input (`x` at z[1]), exercising the
    /// nonzero `z_pub(r_y)` reconciliation path.
    #[test]
    fn r1cs_prove_verify_with_public_input() {
        let setup = build(3, 9, 1);
        assert_eq!(setup.public_values.len(), 1, "one public input (x)");
        let (proof, _) = prove(&setup);

        let frontend = R1csFrontend::<F>::new(
            setup.instance.clone(),
            setup.public_values.clone(),
            setup.cfg,
        );
        let mut vt = Blake3Transcript::new();
        let (endpoints, claims) = frontend
            .verify_constraints::<IdealOrZero<DegreeOneIdeal<F>>>(
                &mut vt,
                &proof,
                &[setup.cfg],
                setup.num_vars,
                no_ideal,
                no_ideal,
                no_scalar,
            )
            .expect("verify_constraints");

        let lifted = hand_lifted(&setup.z_wit, &endpoints.r_0, &setup.cfg);
        frontend
            .verify_lifted_evals(&claims, &lifted, &[setup.cfg])
            .expect("verify_lifted_evals");
    }

    /// Negative: a non-satisfying witness (`w != x^2`) makes the outer
    /// sumcheck's claimed sum nonzero, so verification rejects.
    #[test]
    fn r1cs_rejects_non_satisfying_witness() {
        let setup = build(3, 10, 0); // 3*3 = 9 != 10
        let (proof, _) = prove(&setup);

        let frontend = R1csFrontend::<F>::new(
            setup.instance.clone(),
            setup.public_values.clone(),
            setup.cfg,
        );
        let mut vt = Blake3Transcript::new();
        let res = frontend.verify_constraints::<IdealOrZero<DegreeOneIdeal<F>>>(
            &mut vt,
            &proof,
            &[setup.cfg],
            setup.num_vars,
            no_ideal,
            no_ideal,
            no_scalar,
        );
        assert!(res.is_err(), "non-satisfying witness must be rejected");
    }

    /// Negative: tampering the prover-sent `z_ry` breaks the inner sumcheck's
    /// `M~(r_x, r_y) * z_ry` consistency check.
    #[test]
    fn r1cs_rejects_tampered_z_ry() {
        let setup = build(3, 9, 0);
        let (mut proof, _) = prove(&setup);
        proof.z_ry += &F::one_with_cfg(&setup.cfg);

        let frontend = R1csFrontend::<F>::new(
            setup.instance.clone(),
            setup.public_values.clone(),
            setup.cfg,
        );
        let mut vt = Blake3Transcript::new();
        let res = frontend.verify_constraints::<IdealOrZero<DegreeOneIdeal<F>>>(
            &mut vt,
            &proof,
            &[setup.cfg],
            setup.num_vars,
            no_ideal,
            no_ideal,
            no_scalar,
        );
        assert!(res.is_err(), "tampered z_ry must be rejected");
    }

    /// Negative: verifying against a different matrix than the prover used
    /// (a `C` with a perturbed coefficient) breaks the inner `M~(r_x, r_y)`
    /// evaluation.
    #[test]
    fn r1cs_rejects_tampered_matrix() {
        let setup = build(3, 9, 0);
        let (proof, _) = prove(&setup);

        // Rebuild the instance with C's coefficient 1 -> 2 at (row 0, col 2).
        let n = 1usize << setup.num_vars;
        let f = |v: i64| F::from_with_cfg(&v, &setup.cfg);
        let mut bad = setup.instance.clone();
        bad.c = SparseMatrix::from_rows(n, vec![vec![(2usize, f(2))], Vec::new()]);
        let frontend = R1csFrontend::<F>::new(bad, setup.public_values.clone(), setup.cfg);
        let mut vt = Blake3Transcript::new();
        let res = frontend.verify_constraints::<IdealOrZero<DegreeOneIdeal<F>>>(
            &mut vt,
            &proof,
            &[setup.cfg],
            setup.num_vars,
            no_ideal,
            no_ideal,
            no_scalar,
        );
        assert!(res.is_err(), "tampered matrix must be rejected");
    }

    /// Negative: a correct witness eval reconciled against the wrong substrate
    /// lift (`z_wit(r_y)` perturbed) must be rejected by `verify_lifted_evals`.
    #[test]
    fn r1cs_rejects_tampered_lifted_eval() {
        let setup = build(3, 9, 0);
        let (proof, _) = prove(&setup);

        let frontend = R1csFrontend::<F>::new(
            setup.instance.clone(),
            setup.public_values.clone(),
            setup.cfg,
        );
        let mut vt = Blake3Transcript::new();
        let (endpoints, claims) = frontend
            .verify_constraints::<IdealOrZero<DegreeOneIdeal<F>>>(
                &mut vt,
                &proof,
                &[setup.cfg],
                setup.num_vars,
                no_ideal,
                no_ideal,
                no_scalar,
            )
            .expect("verify_constraints");

        let mut lifted = hand_lifted(&setup.z_wit, &endpoints.r_0, &setup.cfg);
        lifted[0][0] = DynamicPolynomialF::constant_poly(F::one_with_cfg(&setup.cfg));
        let res = frontend.verify_lifted_evals(&claims, &lifted, &[setup.cfg]);
        assert!(res.is_err(), "tampered witness lift must be rejected");
    }

    /// The `R1csConstraintProof` serialization round-trips exactly.
    #[test]
    fn r1cs_proof_serialization_roundtrip() {
        let setup = build(3, 9, 0);
        let (proof, _) = prove(&setup);

        let n = proof.get_num_bytes();
        let mut buf = vec![0u8; n];
        proof.write_transcription_bytes_exact(&mut buf);
        let proof2 = R1csConstraintProof::<F>::read_transcription_bytes_exact(&buf);
        assert_eq!(proof, proof2);
    }
}
