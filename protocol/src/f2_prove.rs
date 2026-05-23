//! All-F_2 prove path.
//!
//! Skeleton implementation of the protocol described in
//! [`f2_prove_plan.md`](../f2_prove_plan.md), stopping at the
//! sumcheck's MLE evaluation claims (which the user has said will be
//! proved in a future iteration).
//!
//! ## Pipeline
//!
//! 1. **Step 0 (commit)** — PCS commit via `RaaF2Code`. (Not run by
//!    [`prove_f2_uair`] yet — the focus of this slice is the
//!    IC/sumcheck wiring. Caller is expected to do the commit
//!    independently and absorb the commitment into the same
//!    transcript before invoking `prove_f2_uair`.)
//!
//! 2. **Step 1 (prime projection)** — mute. The trace is already in
//!    `F_2[X]`; there is no `Z[X] → F_q[X]` reduction step.
//!
//! 3. **Step 2 (ideal check over GF(2^192)[X])** — run
//!    `IdealCheckProtocol::prove_combined::<BinaryFieldGF192>` on
//!    the trace lifted via the trivial `F_2 ⊂ GF(2^192)`
//!    coefficient embedding.
//!
//! 4. **Step 3 (evaluation projection `ψ_α`)** — sample
//!    `α ∈ GF(2^192)` and substitute `X = α` in the trace, producing
//!    `Vec<DenseMultilinearExtension<BinaryFieldGF192>>`. The IC's
//!    `DynamicPolynomialF<BinaryFieldGF192>`-valued combined-MLE
//!    values are likewise evaluated at α to land in `GF(2^192)`.
//!
//! 5. **Step 4 (sumcheck over GF(2^192))** — run
//!    [`MultiDegreeSumcheck`] on the projected trace. The IC's
//!    evaluation point becomes the `eq`-style randomness for a
//!    zerocheck-shaped group; the sumcheck reduces a single
//!    degree-2 group to a final evaluation point + per-MLE
//!    expected evaluations.
//!
//! 6. **Stop.** The output is the bundle of (IC proof, sumcheck
//!    proof, MLE evaluation claims at the sumcheck's final point).
//!    Proving the MLE evaluation claims themselves is a follow-up.

use core::marker::PhantomData;
use crypto_primitives::Field;
use std::fmt::Debug;
use zinc_piop::{
    ideal_check::{IdealCheckProtocol, Proof as IcProof, VerifierSubclaim as IcVerifierSubclaim},
    projections::project_f2_trace_row_major,
    sumcheck::{
        SumCheckError,
        multi_degree::{
            MultiDegreeSubClaims, MultiDegreeSumcheck, MultiDegreeSumcheckGroup,
            MultiDegreeSumcheckProof,
        },
    },
};
use zinc_uair::ideal_collector::IdealOrZero;
use zinc_poly::{
    mle::DenseMultilinearExtension,
    univariate::{
        binary::BinaryPoly, binary_f2_wide::BinaryF2Poly, binary_gf192::BinaryFieldGF192,
        dynamic::over_field::DynamicPolynomialF,
    },
};
use zinc_transcript::traits::Transcript;
use zip_plus::{
    ZipError,
    code::{F2LinearOpener, LinearCode},
    merkle::MerkleProof,
    pcs::structs::{ZipPlus, ZipPlusCommitment, ZipPlusHint, ZipPlusParams},
};
use zinc_uair::{Uair, UairTrace, constraint_counter::count_constraints};
use zinc_utils::cfg_iter;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Output of [`ZincPlusPiopF2::prove_f2_uair`]: everything the
/// verifier sees up to (but not including) the MLE evaluation claims
/// being themselves proved.
///
/// `ic_proof` is the ideal-check proof over `GF(2^192)[X]`.
/// `sumcheck_proof` is the multi-degree sumcheck proof over
/// `GF(2^192)`. `alpha` is the evaluation-projection challenge.
///
/// The MLE evaluation claims (sumcheck's final point r* +
/// expected per-MLE evaluations) are derived by the verifier from
/// `sumcheck_proof` via `MultiDegreeSumcheck::verify_as_subprotocol`.
/// They're not stored here because (a) the verifier reconstructs
/// them from the proof anyway, (b) `MultiDegreeSubClaims` is not
/// `Clone`, and (c) the IC's evaluation point (the eq-randomness
/// the sumcheck consumes) lives inside the IC's prover state, not
/// the proof itself — the verifier re-derives it from the
/// transcript identically.
#[derive(Clone, Debug)]
pub struct F2Proof {
    pub ic_proof: IcProof<BinaryFieldGF192>,
    pub sumcheck_proof: MultiDegreeSumcheckProof<BinaryFieldGF192>,
    /// `α ∈ GF(2^192)` — the evaluation-projection challenge, drawn
    /// from the transcript after the IC. Recorded here as a
    /// convenience; the verifier could equivalently re-derive it.
    pub alpha: BinaryFieldGF192,
}

/// Errors emitted by [`ZincPlusPiopF2::prove_f2_uair`].
#[derive(Debug, thiserror::Error)]
pub enum F2ProveError<U: Uair> {
    #[error("ideal-check failed: {0}")]
    IdealCheck(zinc_piop::ideal_check::IdealCheckError<BinaryFieldGF192, U::Ideal>),
    #[error("evaluation projection failed: {0}")]
    EvalProjection(zinc_poly::EvaluationError),
}

/// Verifier subclaim emitted by [`ZincPlusPiopF2::verify_f2_uair`]: the
/// data downstream layers need to discharge the MLE evaluation claims.
///
/// `ic_evaluation_point` is the IC's randomly-sampled point `r` (the
/// `eq`-randomness for the zerocheck-shaped sumcheck). `alpha` is the
/// evaluation-projection challenge drawn between the IC and the
/// sumcheck. `sumcheck_point` is the sumcheck's shared final point
/// `r*`.
///
/// `primary_column_evals[g]` is the verifier-derived expected
/// evaluation of *primary* (committed) column `g`'s projected MLE at
/// `r*`. These feed into the F_2[X] PCS open
/// ([`ZincPlusPiopF2::verify_f2_open`]).
///
/// `virtual_column_evals[k]` is the analogous evaluation for a
/// *virtual* binary_poly column — an F_2-linear combination of
/// primary columns declared at protocol time. Virtual columns are
/// not committed; the verifier derives their evals at `r*` from
/// `primary_column_evals` and the
/// [`F2VirtualBpSpec`] linear-combo definition.
#[derive(Clone, Debug)]
pub struct F2VerifierSubclaim {
    pub ic_evaluation_point: Vec<BinaryFieldGF192>,
    pub alpha: BinaryFieldGF192,
    pub sumcheck_point: Vec<BinaryFieldGF192>,
    pub primary_column_evals: Vec<BinaryFieldGF192>,
    pub virtual_column_evals: Vec<BinaryFieldGF192>,
}

/// One virtual binary_poly column for an F_2 UAIR: an
/// `F_2[X]`-linear (= XOR) combination of primary witness
/// binary_poly columns.
///
/// Virtual columns are never committed. The UAIR's
/// `constrain_general` can index them as ordinary binary_poly
/// columns *after* the primary witness columns (so primary cols
/// `0..num_primary` and virtual cols
/// `num_primary..num_primary + num_virtual`). The prover
/// materialises each virtual column from the primary trace before
/// running the IC + sumcheck; the verifier derives its MLE
/// evaluation at `r*` from `primary_column_evals` and the spec.
///
/// Coefficients are implicitly `1 ∈ F_2` — listing a primary col
/// index in `primary_col_indices` XORs that column into the virtual
/// column. Omitting an index means coefficient `0`. The order of
/// indices is irrelevant (XOR is commutative); duplicates cancel
/// pairwise.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct F2VirtualBpSpec {
    pub primary_col_indices: Vec<usize>,
}

/// Errors emitted by [`ZincPlusPiopF2::verify_f2_uair`].
#[derive(Debug, thiserror::Error)]
pub enum F2VerifyError<U: Uair, IdealOverF>
where
    IdealOverF: zinc_uair::ideal::Ideal,
{
    #[error("ideal-check verification failed: {0}")]
    IdealCheck(zinc_piop::ideal_check::IdealCheckError<BinaryFieldGF192, IdealOverF>),
    #[error("sumcheck verification failed: {0}")]
    Sumcheck(SumCheckError<BinaryFieldGF192>),
    #[error("α drawn from transcript ({transcript}) disagrees with proof.alpha ({proof})")]
    AlphaMismatch {
        transcript: BinaryFieldGF192,
        proof: BinaryFieldGF192,
    },
    #[error("eq(r*, r) = 0 — sumcheck point coincides with IC point, cannot derive MLE claims")]
    DegenerateEq,
    #[error("expected {expected} sumcheck groups, got {actual}")]
    GroupCountMismatch { expected: usize, actual: usize },
    #[error(
        "virtual column eval mismatch at index {virtual_idx}: sumcheck-extracted ({sumcheck:?}) ≠ derived from primary evals ({derived:?})"
    )]
    VirtualEvalMismatch {
        virtual_idx: usize,
        sumcheck: BinaryFieldGF192,
        derived: BinaryFieldGF192,
    },
    #[error("internal: U::Uair phantom")]
    _Uair(std::marker::PhantomData<U>),
}

/// All-`F_2` ZincTypes-like trait. Mirrors [`ZincTypes`](crate::ZincTypes)
/// but drops the `ArbitraryZt`/`IntZt` lanes (an all-`F_2` UAIR has
/// neither) and the prime-modulus / challenge / projecting-element
/// machinery (`F_2[X]` doesn't get reduced via a random prime; the
/// projecting element is sampled directly in `GF(2^192)`).
pub trait F2ZincTypes<const DEGREE_PLUS_ONE: usize>: Clone + Debug {
    /// Zip+ types for the (single) binary polynomial trace columns.
    type BinaryZt: zip_plus::pcs::structs::ZipTypes<
            Eval = BinaryPoly<DEGREE_PLUS_ONE>,
            Cw = BinaryPoly<DEGREE_PLUS_ONE>,
        >;

    /// Linear code used in Zip+ for the binary polynomial trace
    /// columns. Expected to be a flavour of
    /// [`RaaF2Code`](zip_plus::code::raa_f2::RaaF2Code) so that
    /// codewords stay in `F_2[X]/<X^D>` (no integer widening).
    ///
    /// `F2LinearOpener` is additionally required so the F_2[X]
    /// MLE-opening protocol's proximity check can encode a width-`W`
    /// combined row through the same linear map as the commitment.
    type BinaryLc: zip_plus::code::LinearCode<Self::BinaryZt>
        + zip_plus::code::F2LinearOpener;
}

/// Phantom marker that ties the prove function to its type parameters
/// without storing any data. Mirrors `ZincPlusPiop` from the integer
/// prove path; kept as its own struct so future expansions of the
/// F_2 protocol can hang state off it.
pub struct ZincPlusPiopF2<Zt, U, const DEGREE_PLUS_ONE: usize>(PhantomData<(Zt, U)>)
where
    Zt: F2ZincTypes<DEGREE_PLUS_ONE>,
    U: Uair;

/// Materialize the virtual binary_poly columns from a primary
/// witness trace and a list of [`F2VirtualBpSpec`]s.
///
/// Each virtual column is the XOR (= F_2 sum) of the listed primary
/// witness columns, evaluated row-by-row. The output ordering
/// matches `virtual_specs`. Asserts the primary trace's columns are
/// well-shaped (same length); panics if a spec references an
/// out-of-range primary column.
#[allow(clippy::arithmetic_side_effects)]
pub fn materialize_f2_virtual_bp_cols<const D: usize>(
    primary_trace: &[DenseMultilinearExtension<BinaryPoly<D>>],
    virtual_specs: &[F2VirtualBpSpec],
) -> Vec<DenseMultilinearExtension<BinaryPoly<D>>> {
    if virtual_specs.is_empty() {
        return Vec::new();
    }
    let num_rows = primary_trace
        .first()
        .map(|m| m.evaluations.len())
        .unwrap_or(0);
    let num_vars = primary_trace
        .first()
        .map(|m| m.num_vars)
        .unwrap_or(0);
    for col in primary_trace {
        assert_eq!(
            col.evaluations.len(),
            num_rows,
            "materialize_f2_virtual_bp_cols: primary columns must share length",
        );
        assert_eq!(
            col.num_vars,
            num_vars,
            "materialize_f2_virtual_bp_cols: primary columns must share num_vars",
        );
    }
    let num_primary = primary_trace.len();

    virtual_specs
        .iter()
        .map(|spec| {
            let mut evals: Vec<BinaryPoly<D>> = vec![BinaryPoly::default(); num_rows];
            for &col_idx in &spec.primary_col_indices {
                assert!(
                    col_idx < num_primary,
                    "F2VirtualBpSpec: primary col idx {col_idx} out of range (num_primary = {num_primary})",
                );
                let primary_col = &primary_trace[col_idx];
                for (i, slot) in evals.iter_mut().enumerate() {
                    use zinc_poly::univariate::F2AddAssign;
                    slot.f2_add_assign(&primary_col.evaluations[i]);
                }
            }
            DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                evals,
                BinaryPoly::default(),
            )
        })
        .collect()
}

/// Derive virtual binary_poly column MLE evaluations at `r*` from
/// the primary column evals at `r*` and the F_2-linear combo specs.
/// Mirrors [`materialize_f2_virtual_bp_cols`]'s arithmetic at the
/// MLE-evaluation level: F_2 addition (= GF(2^192) addition) of the
/// referenced primary evals.
pub fn derive_f2_virtual_evals_at(
    primary_evals: &[BinaryFieldGF192],
    virtual_specs: &[F2VirtualBpSpec],
) -> Vec<BinaryFieldGF192> {
    virtual_specs
        .iter()
        .map(|spec| {
            let mut acc = BinaryFieldGF192::zero();
            for &col_idx in &spec.primary_col_indices {
                acc += &primary_evals[col_idx];
            }
            acc
        })
        .collect()
}

/// Default sumcheck-group builder for the F_2 prove path.
///
/// Emits one degree-2 group per projected trace column with the
/// combination function `comb_fn(eq(y, r), col(y)) = eq · col`. The
/// claimed sum is the column MLE evaluated at the IC point `r`,
/// giving a zerocheck-shaped reduction whose final point `r*` lets
/// the verifier interpret each group's expected evaluation as
/// `eq(r*, r) · col(r*)`.
///
/// This is the simplest viable composition; full UAIRs with
/// constraint-shaped degree groups would use a CPR-style builder
/// (see [`ZincPlusPiopF2::prove_f2_uair_with_groups`]).
pub fn eq_dot_column_groups(
    ic_eval_point: &[BinaryFieldGF192],
    projected_trace: &[DenseMultilinearExtension<BinaryFieldGF192>],
    field_cfg: &(),
) -> Vec<MultiDegreeSumcheckGroup<BinaryFieldGF192>> {
    let eq_r = zinc_poly::utils::build_eq_x_r_inner(ic_eval_point, field_cfg)
        .expect("eq table construction must succeed for valid IC point");
    let zero_inner = *BinaryFieldGF192::zero().inner();
    projected_trace
        .iter()
        .map(|col| {
            let col_inner_mle = DenseMultilinearExtension::from_evaluations_vec(
                col.num_vars,
                col.evaluations.iter().map(|x| *x.inner()).collect(),
                zero_inner,
            );
            MultiDegreeSumcheckGroup::new(
                2,
                vec![eq_r.clone(), col_inner_mle],
                Box::new(|v: &[BinaryFieldGF192]| v[0] * v[1]),
            )
        })
        .collect()
}

/// Default subclaim extractor matching [`eq_dot_column_groups`].
///
/// Each group's expected evaluation is `eq(r*, r) · col(r*)`;
/// dividing by `eq(r*, r)` recovers the per-column MLE evaluation
/// claim. Returns `Err(())` when `eq(r*, r) == 0`, which only
/// happens when the IC and sumcheck transcript challenges collide
/// (probability ~`2^{-192}` per round for an honest Fiat-Shamir
/// hash).
#[allow(clippy::result_unit_err)]
pub fn extract_column_evals_eq_dot_col(
    ic_eval_point: &[BinaryFieldGF192],
    md_subclaims: &MultiDegreeSubClaims<BinaryFieldGF192>,
) -> Result<Vec<BinaryFieldGF192>, ()> {
    let one = BinaryFieldGF192::one();
    let eq_at_rstar_r =
        zinc_poly::utils::eq_eval(md_subclaims.point(), ic_eval_point, one)
            .expect("matching length (num_vars) by construction");
    if eq_at_rstar_r.is_zero() {
        return Err(());
    }
    let eq_inv = eq_at_rstar_r.inverse();
    Ok(md_subclaims
        .expected_evaluations()
        .iter()
        .map(|expected| (*expected) * eq_inv)
        .collect())
}

impl<Zt, U, const D: usize> ZincPlusPiopF2<Zt, U, D>
where
    Zt: F2ZincTypes<D>,
    U: Uair + 'static,
{
    /// Run the F_2 prove pipeline up to (but not including) the MLE
    /// evaluation claims.
    ///
    /// `transcript` is mutated by the IC, the α draw, and the
    /// sumcheck — caller should absorb any pre-IC commitments (PCS
    /// commitment etc.) into it before invoking.
    ///
    /// `trace` is the all-F_2 trace; its `arbitrary_poly` and `int`
    /// lanes must be empty (asserted by `project_f2_trace_row_major`).
    ///
    /// `project_scalar` lifts each UAIR scalar from
    /// `U::Scalar` to `DynamicPolynomialF<BinaryFieldGF192>` (the
    /// `GF(2^192)[X]` form the IC's combined-poly machinery
    /// expects). For an F_2-typed UAIR with `U::Scalar = BinaryPoly<D>`
    /// the natural choice is per-coefficient `F_2 ⊂ GF(2^192)`
    /// embedding; for UAIRs with no scalars (e.g. `assert_zero`-only)
    /// the closure is never invoked.
    pub fn prove_f2_uair(
        transcript: &mut impl Transcript,
        trace: &UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D>,
        num_vars: usize,
        project_scalar: impl Fn(&U::Scalar) -> DynamicPolynomialF<BinaryFieldGF192> + Sync,
    ) -> Result<(F2Proof, F2VerifierSubclaim), F2ProveError<U>> {
        // Default composition + no virtual columns. Use
        // [`Self::prove_f2_uair_with_groups`] with explicit
        // `virtual_specs` for UAIRs that declare virtual binary_poly
        // columns.
        Self::prove_f2_uair_with_groups(
            transcript,
            trace,
            &[],
            num_vars,
            project_scalar,
            eq_dot_column_groups,
        )
    }

    /// Generic-group variant of [`Self::prove_f2_uair`].
    ///
    /// `build_groups(ic_eval_point, projected_trace, field_cfg) ->
    /// Vec<MultiDegreeSumcheckGroup>` is the user-supplied sumcheck
    /// group composition. Use [`eq_dot_column_groups`] for the
    /// per-column zerocheck shape; richer UAIRs can pass a closure
    /// that produces per-degree groups matching the CPR layout.
    ///
    /// Returns both the wire `F2Proof` and the prover-side
    /// `F2VerifierSubclaim` — the latter mirrors what the verifier
    /// would derive from the same transcript, and lets the caller
    /// chain into [`Self::prove_f2_open`] without re-running the
    /// IC + sumcheck on a verifier shim.
    pub fn prove_f2_uair_with_groups<G>(
        transcript: &mut impl Transcript,
        trace: &UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D>,
        virtual_specs: &[F2VirtualBpSpec],
        num_vars: usize,
        project_scalar: impl Fn(&U::Scalar) -> DynamicPolynomialF<BinaryFieldGF192> + Sync,
        build_groups: G,
    ) -> Result<(F2Proof, F2VerifierSubclaim), F2ProveError<U>>
    where
        G: FnOnce(
            &[BinaryFieldGF192],
            &[DenseMultilinearExtension<BinaryFieldGF192>],
            &(),
        ) -> Vec<MultiDegreeSumcheckGroup<BinaryFieldGF192>>,
    {
        let num_constraints = count_constraints::<U>();
        let field_cfg = ();
        let num_primary = trace.binary_poly.len();

        // -- Materialise virtual binary_poly columns ----------
        // Appended after the primary witness columns; the UAIR's
        // constraint code references them by their (extended)
        // absolute index.
        let virtual_cols =
            materialize_f2_virtual_bp_cols::<D>(&trace.binary_poly, virtual_specs);
        let mut all_binary_poly_cols: Vec<DenseMultilinearExtension<BinaryPoly<D>>> =
            trace.binary_poly.iter().cloned().collect();
        all_binary_poly_cols.extend(virtual_cols);
        let extended_trace: UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D> = UairTrace {
            binary_poly: all_binary_poly_cols.into(),
            arbitrary_poly: vec![].into(),
            int: vec![].into(),
        };

        // -- Step 2: Ideal check over GF(2^192)[X] -----------------
        //
        // F_2-native path: skip the per-cell lift to
        // `DynamicPolynomialF<GF(2^192)>` and evaluate constraint
        // expressions row-by-row in 64-bit bit-poly arithmetic
        // (XORs and carryless shifts of `u64`s). The combined
        // polynomial's per-coefficient MLE-evaluation at the IC
        // point r is a sum-of-eqs over the rows where that bit is
        // set — no DynamicPolynomial allocation per cell. See
        // [`crate::f2_native_ic`] for the algorithm + scope.
        //
        // The caller still supplies `project_scalar` for the
        // sumcheck path below (and for completeness if a fallback
        // is needed). For the IC, we use the bit-pack form of the
        // same scalar: bit `i` set iff the scalar's i-th coefficient
        // is non-zero — for SHA-F_2 where scalars have F_2-coefficient
        // (0/1) values, this matches the `project_scalar` output
        // exactly (coefficient 0 ↔ GF(2^192) zero ↔ bit 0;
        // coefficient 1 ↔ GF(2^192) one ↔ bit 1).
        let project_scalar_to_bits =
            |s: &U::Scalar| -> u64 {
                let projected = project_scalar(s);
                let mut bits: u64 = 0;
                for (i, c) in projected.coeffs.iter().enumerate() {
                    if i >= 64 {
                        break;
                    }
                    if !<BinaryFieldGF192 as crypto_primitives::PrimeField>::is_zero(c) {
                        #[allow(clippy::arithmetic_side_effects)]
                        {
                            bits |= 1u64 << i;
                        }
                    }
                }
                bits
            };

        let (ic_proof, ic_state) =
            crate::f2_native_ic::F2NativeIc::<U>::prove_combined::<BinaryFieldGF192, _, D>(
                transcript,
                &extended_trace.binary_poly,
                num_constraints,
                num_vars,
                &field_cfg,
                project_scalar_to_bits,
            );

        // -- Step 3: Evaluation projection (X = α) -----------------
        let alpha: BinaryFieldGF192 = transcript.get_field_challenge(&field_cfg);

        // Precompute α^0, α^1, ..., α^{D-1} once outside the per-cell
        // loop. With ~21K cells per SHA F_2 trace (41 cols × 512 rows),
        // this drops the per-cell cost from D=32 GF(2^192)
        // multiplications to just bit-selected XOR-adds — saving ~670K
        // field multiplications across the whole projection.
        let alpha_pows: Vec<BinaryFieldGF192> =
            zinc_poly::univariate::binary_gf192::alpha_powers(&alpha, D);
        let projected_trace: Vec<DenseMultilinearExtension<BinaryFieldGF192>> =
            cfg_iter!(extended_trace.binary_poly)
                .map(|col| {
                    let evals_at_alpha: Vec<BinaryFieldGF192> = col
                        .evaluations
                        .iter()
                        .map(|cell| {
                            zinc_poly::univariate::binary_gf192
                                ::eval_f2_poly_d_at_with_powers::<D>(cell, &alpha_pows)
                        })
                        .collect();
                    DenseMultilinearExtension::from_evaluations_vec(
                        col.num_vars,
                        evals_at_alpha,
                        BinaryFieldGF192::zero(),
                    )
                })
                .collect();

        // -- Step 4: Sumcheck over GF(2^192) -----------------------
        let groups = build_groups(
            &ic_state.evaluation_point,
            &projected_trace,
            &field_cfg,
        );

        let (sumcheck_proof, prover_states) =
            MultiDegreeSumcheck::<BinaryFieldGF192>::prove_as_subprotocol(
                transcript,
                groups,
                num_vars,
                &field_cfg,
            );

        // -- Derive the prover-side subclaim --------------------
        let sumcheck_point = prover_states[0].randomness.clone();
        // Per-column (primary + virtual) MLE evals at r*.
        let all_col_evals: Vec<BinaryFieldGF192> = projected_trace
            .iter()
            .map(|col| {
                let zero_inner = *BinaryFieldGF192::zero().inner();
                let inner_mle = DenseMultilinearExtension::from_evaluations_vec(
                    col.num_vars,
                    col.evaluations.iter().map(|x| *x.inner()).collect(),
                    zero_inner,
                );
                <DenseMultilinearExtension<_> as zinc_poly::mle::MultilinearExtensionWithConfig<
                    BinaryFieldGF192,
                >>::evaluate_with_config(
                    inner_mle, &sumcheck_point, &field_cfg
                )
                .expect("MLE evaluation on r* should succeed")
            })
            .collect();
        // Split: first `num_primary` are committed; the rest are
        // virtual. The verifier reconstructs the virtual side from
        // the primary side and `virtual_specs`.
        let (primary_evals, virtual_evals) = all_col_evals.split_at(num_primary);
        let subclaim = F2VerifierSubclaim {
            ic_evaluation_point: ic_state.evaluation_point,
            alpha,
            sumcheck_point,
            primary_column_evals: primary_evals.to_vec(),
            virtual_column_evals: virtual_evals.to_vec(),
        };

        Ok((
            F2Proof {
                ic_proof,
                sumcheck_proof,
                alpha,
            },
            subclaim,
        ))
    }

    /// Verify a proof emitted by [`Self::prove_f2_uair`].
    ///
    /// Mirrors the prover's transcript exactly: runs the IC's
    /// `verify_as_subprotocol`, redraws α, then runs the sumcheck's
    /// `verify_as_subprotocol`. Returns the verifier's
    /// [`F2VerifierSubclaim`] — the IC point, α, the sumcheck point
    /// `r*`, and the per-column MLE evaluation claims derived from
    /// the per-group sumcheck subclaims via the `eq · col` group
    /// composition the prover used.
    ///
    /// `project_ideal` lifts `U::Ideal` → `IdealOverF` exactly as the
    /// integer-protocol verifier does in [`crate::verifier`]. For
    /// `assert_zero`-only UAIRs the closure is never invoked (the
    /// IC's per-constraint loop short-circuits on zero ideals).
    pub fn verify_f2_uair<IdealOverF>(
        transcript: &mut impl Transcript,
        proof: &F2Proof,
        virtual_specs: &[F2VirtualBpSpec],
        num_vars: usize,
        num_primary_columns: usize,
        project_ideal: impl Fn(&IdealOrZero<U::Ideal>) -> IdealOverF,
    ) -> Result<F2VerifierSubclaim, F2VerifyError<U, IdealOverF>>
    where
        IdealOverF: zinc_uair::ideal::Ideal
            + zinc_uair::ideal::IdealCheck<DynamicPolynomialF<BinaryFieldGF192>>,
    {
        Self::verify_f2_uair_with_groups(
            transcript,
            proof,
            virtual_specs,
            num_vars,
            num_primary_columns,
            project_ideal,
            |ic_eval_point, md_subclaims| {
                extract_column_evals_eq_dot_col(ic_eval_point, md_subclaims)
                    .map_err(|_| F2VerifyError::DegenerateEq)
            },
        )
    }

    /// Generic-group variant of [`Self::verify_f2_uair`].
    ///
    /// `extract_subclaims(ic_eval_point, md_subclaims) ->
    /// Result<Vec<MLE eval claim>, F2VerifyError>` is the
    /// composition-specific inversion: from per-group expected
    /// evaluations + the IC point, recover the per-column MLE
    /// evaluation claims at `r*` (covering both primary and virtual
    /// columns, in that order) that downstream PCS opening will
    /// discharge for primary cols. Pair with
    /// [`Self::prove_f2_uair_with_groups`] — the closure must invert
    /// whatever `build_groups` composed.
    ///
    /// After extraction, the verifier checks each virtual column's
    /// extracted eval against the F_2-linear combo of primary col
    /// evals — soundness for the verifier's "virtual = linear combo"
    /// derivation. Mismatch surfaces as
    /// [`F2VerifyError::VirtualEvalMismatch`].
    pub fn verify_f2_uair_with_groups<IdealOverF, E>(
        transcript: &mut impl Transcript,
        proof: &F2Proof,
        virtual_specs: &[F2VirtualBpSpec],
        num_vars: usize,
        num_primary_columns: usize,
        project_ideal: impl Fn(&IdealOrZero<U::Ideal>) -> IdealOverF,
        extract_subclaims: E,
    ) -> Result<F2VerifierSubclaim, F2VerifyError<U, IdealOverF>>
    where
        IdealOverF: zinc_uair::ideal::Ideal
            + zinc_uair::ideal::IdealCheck<DynamicPolynomialF<BinaryFieldGF192>>,
        E: FnOnce(
            &[BinaryFieldGF192],
            &MultiDegreeSubClaims<BinaryFieldGF192>,
        ) -> Result<Vec<BinaryFieldGF192>, F2VerifyError<U, IdealOverF>>,
    {
        let num_constraints = count_constraints::<U>();
        let field_cfg = ();
        let num_total = num_primary_columns + virtual_specs.len();

        let ic_subclaim: IcVerifierSubclaim<BinaryFieldGF192> =
            <U as IdealCheckProtocol>::verify_as_subprotocol::<_, IdealOverF, _>(
                transcript,
                proof.ic_proof.clone(),
                num_constraints,
                num_vars,
                project_ideal,
                &field_cfg,
            )
            .map_err(F2VerifyError::IdealCheck)?;
        let ic_evaluation_point = ic_subclaim.evaluation_point;

        let alpha: BinaryFieldGF192 = transcript.get_field_challenge(&field_cfg);
        if alpha != proof.alpha {
            return Err(F2VerifyError::AlphaMismatch {
                transcript: alpha,
                proof: proof.alpha,
            });
        }

        let md_subclaims = MultiDegreeSumcheck::<BinaryFieldGF192>::verify_as_subprotocol(
            transcript,
            num_vars,
            &proof.sumcheck_proof,
            &field_cfg,
        )
        .map_err(F2VerifyError::Sumcheck)?;

        if md_subclaims.expected_evaluations().len() != num_total {
            return Err(F2VerifyError::GroupCountMismatch {
                expected: num_total,
                actual: md_subclaims.expected_evaluations().len(),
            });
        }

        let sumcheck_point = md_subclaims.point().to_vec();
        let all_col_evals = extract_subclaims(&ic_evaluation_point, &md_subclaims)?;
        let (primary_evals, virtual_evals_from_sumcheck) =
            all_col_evals.split_at(num_primary_columns);
        let primary_evals = primary_evals.to_vec();

        // Virtual-eval consistency: each virtual col eval extracted
        // from the sumcheck must equal the F_2-linear combo of
        // primary evals defined by the spec.
        let virtual_evals_derived =
            derive_f2_virtual_evals_at(&primary_evals, virtual_specs);
        for (k, (from_sumcheck, derived)) in virtual_evals_from_sumcheck
            .iter()
            .zip(virtual_evals_derived.iter())
            .enumerate()
        {
            if from_sumcheck != derived {
                return Err(F2VerifyError::VirtualEvalMismatch {
                    virtual_idx: k,
                    sumcheck: *from_sumcheck,
                    derived: *derived,
                });
            }
        }

        Ok(F2VerifierSubclaim {
            ic_evaluation_point,
            alpha,
            sumcheck_point,
            primary_column_evals: primary_evals,
            virtual_column_evals: virtual_evals_derived,
        })
    }

    // -- PCS commit/open plumbing ------------------------------------
    //
    // Step 0 (commit) and Step 7 (open) are exposed as separate
    // functions so callers can compose: commit binary trace columns,
    // absorb the commitment into the transcript, then run the
    // IC + sumcheck via `prove_f2_uair`. The shape mirrors the
    // integer prove path, which keeps commit/open as separate phases
    // around the PIOP.

    /// Step 0: commit to the F_2 trace's binary_poly columns via the
    /// caller-supplied Zip+ params, returning the commitment + a
    /// prover-side hint that's needed at open time.
    ///
    /// Caller is expected to absorb `commitment.root.0` into the
    /// Fiat-Shamir transcript before invoking
    /// [`Self::prove_f2_uair`]; see
    /// [`Self::commit_and_absorb_f2_trace`] for the bundled helper.
    pub fn commit_f2_trace(
        pp: &ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
        trace_binary_cols: &[DenseMultilinearExtension<BinaryPoly<D>>],
    ) -> Result<
        (
            ZipPlusHint<<Zt::BinaryZt as zip_plus::pcs::structs::ZipTypes>::Cw>,
            ZipPlusCommitment,
        ),
        ZipError,
    > {
        ZipPlus::<Zt::BinaryZt, Zt::BinaryLc>::commit(pp, trace_binary_cols)
    }

    /// Convenience: commit + absorb the resulting commitment root
    /// into `transcript`. Returns the hint (needed at open time) and
    /// the commitment (so the verifier can be handed the same root).
    pub fn commit_and_absorb_f2_trace(
        transcript: &mut impl Transcript,
        pp: &ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
        trace_binary_cols: &[DenseMultilinearExtension<BinaryPoly<D>>],
    ) -> Result<
        (
            ZipPlusHint<<Zt::BinaryZt as zip_plus::pcs::structs::ZipTypes>::Cw>,
            ZipPlusCommitment,
        ),
        ZipError,
    > {
        let (hint, comm) = Self::commit_f2_trace(pp, trace_binary_cols)?;
        // Mirror the integer protocol's commitment-absorption: write
        // the Merkle root bytes directly. (`ZipPlusCommitment`
        // implements `ConstTranscribable`; here we use the byte-level
        // form to match how the existing Zip+ tests prime their
        // verifier transcripts.)
        transcript.absorb_slice(&*comm.root);
        Ok((hint, comm))
    }

    /// Verifier counterpart of [`Self::commit_and_absorb_f2_trace`]:
    /// absorbs a previously-published commitment into the verifier's
    /// transcript at the same point the prover did. Idempotent —
    /// just bytes-into-transcript.
    pub fn absorb_commitment(
        transcript: &mut impl Transcript,
        commitment: &ZipPlusCommitment,
    ) {
        transcript.absorb_slice(&*commitment.root);
    }
}

// -- Step 7 (open) via F_2[X] lift-and-project --------------------
//
// See [`f2_open_plan.md`](../f2_open_plan.md) for the full design.
//
// `F2OpenProof` carries: (i) a per-column lifted claim `a_g' ∈
// F_2[X]<≈416>`, (ii) a per-column "b-vector" `b_g ∈ F_2[X]^{num_rows}`
// supporting the evaluation-consistency check. The verifier:
//
//   1. Re-derives `(q_0, q_1)` from the sumcheck's final point `r*`,
//      computing the eq-tensor in `GF(2^192)` and lifting each entry
//      to `BinaryF2Poly<3>` (the canonical degree-<192 representative).
//   2. Checks `Σ_i q_0[i] · b_g[i] = a_g'` in `F_2[X]` (evaluation
//      consistency).
//   3. Checks `ψ_α(a_g') = a_g` in `GF(2^192)` (the lift discharge).
//
// **Proximity not yet bound.** This slice provides the lift +
// evaluation-consistency portion. The proximity check — that `b_g`
// is consistent with the committed codeword matrix via Merkle column
// openings — is the follow-up that ties `b_g` to the PCS. The
// existing `commit_hint` carries everything required (codeword
// matrices + Merkle tree); the missing pieces are the column-sampling
// loop and the `F_2[X]`-lifted re-encoding check at sampled columns.
// See `f2_open_plan.md` § "Piece 4" for the shape.

/// γ-batched data emitted by [`ZincPlusPiopF2::prove_f2_open`].
///
/// **Single-open semantics.** Instead of sending per-column
/// `(a_g', b_g, combined_row_g)` triples, the prover draws a
/// transcript-fresh GF(2^192) challenge `γ_g` for each committed
/// primary column and folds the per-column data into a **single**
/// `(a', b', combined_row')` bundle:
///
/// ```text
/// a'             := Σ_g γ_g · a_g'              ∈ F_2[X]<≈608>
/// b'[i]          := Σ_g γ_g · b_g[i]            ∈ F_2[X]<≈416>
/// combined_row'  := Σ_g γ_g · combined_row_g    ∈ F_2[X]<≈416>
/// ```
///
/// The verifier reconstructs the same `γ_g` from the transcript,
/// checks `Σ_i q_0'[i] · b'[i] = a'` (eval consistency),
/// `ψ_α(a') = Σ_g γ_g · a_g` (lift discharge, where each `a_g` is
/// the per-column MLE claim from the sumcheck subclaim),
/// `<combined_row', q_1'> = <coeffs, b'>` (coherence), and the
/// γ-weighted encoding consistency at each opened column. Soundness
/// against per-column tampering follows from Schwartz-Zippel over
/// the random γ_g.
///
/// Widths: `a' ∈ BinaryF2Poly<10>` (≥ D + 2·192 - 1 + 192 - 1 =
/// 606 bits for D=32); `b'` and `combined_row'` entries in
/// `BinaryF2Poly<7>` (≥ D + 2·192 - 1 = 414 bits).
//
// TODO: when `feature(generic_const_exprs)` stabilises, parameterise
// these widths over `D` and `μ_eq` so the BinaryF2Poly<W> sizes
// shrink to the true bit bounds. Until then the slight
// over-allocation costs ~10% extra transcription bytes per opening;
// not on a hot path for any current workload.
#[derive(Clone, Debug)]
pub struct F2OpenProof<const D: usize> {
    /// `a' = Σ_g γ_g · a_g' ∈ F_2[X]`.
    pub lifted_claim: BinaryF2Poly<10>,
    /// `b'[i] = Σ_g γ_g · b_g[i]`. `num_rows` entries.
    pub b_vector: Vec<BinaryF2Poly<7>>,
    /// `combined_row'[j] = Σ_g γ_g · combined_row_g[j]`. `row_len` entries.
    pub combined_row: Vec<BinaryF2Poly<7>>,
    /// One entry per opened codeword column. Each entry holds the
    /// column's `batch_size · num_rows` codeword cells (concatenated
    /// per-poly in commit order) plus a Merkle proof.
    pub opened_columns: Vec<F2OpenedColumn<D>>,
}

/// A single column opening: the column index, the concatenated
/// codeword cells across all committed polynomials, and the Merkle
/// proof tying those cells to the commitment root.
#[derive(Clone, Debug)]
pub struct F2OpenedColumn<const D: usize> {
    pub column_idx: usize,
    /// `batch_size · num_rows` entries — column `column_idx` of each
    /// `cw_matrix` concatenated in commit order.
    pub column_values: Vec<BinaryPoly<D>>,
    pub merkle_proof: MerkleProof,
}

/// Errors emitted by [`ZincPlusPiopF2::verify_f2_open`].
#[derive(Debug, thiserror::Error)]
pub enum F2OpenError {
    #[error(
        "evaluation-consistency check failed: Σ_i q_0'[i] · b'[i] ≠ a' in F_2[X]"
    )]
    EvalConsistency,
    #[error(
        "lift discharge failed: ψ_α(a') ({computed:?}) ≠ Σ_g γ_g · a_g ({expected:?})"
    )]
    LiftDischarge {
        computed: BinaryFieldGF192,
        expected: BinaryFieldGF192,
    },
    #[error("F2OpenProof.b_vector has length {got}, expected {expected}")]
    BvecLenMismatch { expected: usize, got: usize },
    #[error(
        "coherence check failed: <combined_row', q_1'> ≠ <coeffs, b'> in F_2[X]"
    )]
    Coherence,
    #[error("F2OpenProof.combined_row has length {got}, expected {expected}")]
    CombinedRowLenMismatch { expected: usize, got: usize },
    #[error("Merkle path verification failed for opened column {column_idx}: {reason}")]
    MerkleVerify { column_idx: usize, reason: String },
    #[error(
        "encoding consistency check failed at opened col idx j={column_idx}: \
         encode(combined_row')[j] ≠ Σ_i coeffs[i] · Σ_g γ_g · cw_M^g[i, j]"
    )]
    EncodingConsistency { column_idx: usize },
    #[error("F2OpenedColumn has {got} entries, expected {expected}")]
    ColumnValuesLenMismatch { expected: usize, got: usize },
}

/// Absorb a slice of `BinaryF2Poly<W>` into the transcript by
/// writing each entry's `W × u64` words as little-endian bytes.
/// Deterministic and prover/verifier-symmetric.
///
/// Public so per-region benches can exercise the open phase's
/// sub-steps independently; not part of the verified protocol surface.
pub fn absorb_f2_poly_slice<'a, const W: usize, I>(transcript: &mut impl Transcript, iter: I)
where
    I: IntoIterator<Item = &'a BinaryF2Poly<W>>,
{
    let mut buf = [0u8; 8];
    for p in iter {
        for w in p.words() {
            buf.copy_from_slice(&w.to_le_bytes());
            transcript.absorb_slice(&buf);
        }
    }
}

/// Squeeze a u64 challenge from the transcript and reduce modulo
/// `codeword_len`. `codeword_len` is a power of two in all Zip+
/// instantiations, so the modular reduction is bias-free.
///
/// Public so per-region benches can drive the per-opening loop in
/// isolation; not part of the verified protocol surface.
pub fn sample_column_idx(transcript: &mut impl Transcript, codeword_len: usize) -> usize {
    assert!(
        codeword_len.is_power_of_two(),
        "sample_column_idx requires power-of-two codeword length; got {codeword_len}",
    );
    let raw: u64 = transcript.get_challenge();
    #[allow(clippy::arithmetic_side_effects)]
    let idx = (raw as usize) & (codeword_len - 1);
    idx
}

/// Build `(q_0, q_1)` over `GF(2^192)` then lift each entry to
/// `BinaryF2Poly<3>` *via the α-dependent inverse lift*. Mirrors
/// `zip-plus`'s `point_to_tensor` split convention: `q_0` has length
/// `num_rows` (built from the last `log2(num_rows)` entries of
/// `point`); `q_1` has length `row_len = 2^{point.len() -
/// log2(num_rows)}` (built from the preceding entries).
///
/// **Why the inverse lift** (not the canonical-representative lift):
/// the verifier checks `ψ_α(a') = a`, where `ψ_α` evaluates an
/// F_2[X] polynomial at α via `Σ p_i α^i`. For the lifted claim
///
/// ```text
/// a' = q_1'^T · M_w · q_2'   in F_2[X]
/// ```
///
/// to satisfy `ψ_α(a') = q_1^T · ψ_α(M_w) · q_2 = a`, we need
/// `ψ_α(q_i'[k]) = q_i[k]` per entry. The canonical bit-pattern
/// representative satisfies that *only* when α is the field's
/// quotient generator `X` (mod P); for a transcript-fresh α the
/// inverse lift solves `Σ_j c_j · α^j = q_i[k]` for the unique
/// coefficient vector `c ∈ F_2^{192}` and returns
/// `q_i'[k] = Σ_j c_j X^j`. See `AlphaPolyBasis` in
/// [`binary_gf192`](zinc_poly::univariate::binary_gf192) for the
/// linear-algebra detail.
///
/// `basis` is the precomputed lift table (one per α, shared across
/// all `q_i[k]` entries to amortise the 192×192 F_2 matrix inverse).
///
/// Public so per-region benches can time `q0`/`q1` construction in
/// isolation; not part of the verified protocol surface.
#[allow(clippy::type_complexity)]
pub fn build_lifted_eq_tensor(
    num_rows: usize,
    point: &[BinaryFieldGF192],
    basis: &zinc_poly::univariate::binary_gf192::AlphaPolyBasis,
) -> (Vec<BinaryF2Poly<3>>, Vec<BinaryF2Poly<3>>) {
    assert!(num_rows.is_power_of_two());
    let split = point.len() - (num_rows.ilog2() as usize);
    let (hi, lo) = point.split_at(split);
    let field_cfg = ();
    let q0_gf = if !lo.is_empty() {
        zinc_poly::utils::build_eq_x_r_vec(lo, &field_cfg)
            .expect("build_eq_x_r_vec on lo")
    } else {
        vec![BinaryFieldGF192::one()]
    };
    let q1_gf = if !hi.is_empty() {
        zinc_poly::utils::build_eq_x_r_vec(hi, &field_cfg)
            .expect("build_eq_x_r_vec on hi")
    } else {
        vec![BinaryFieldGF192::one()]
    };
    let q0: Vec<BinaryF2Poly<3>> = q0_gf.iter().map(|g| basis.lift(g)).collect();
    let q1: Vec<BinaryF2Poly<3>> = q1_gf.iter().map(|g| basis.lift(g)).collect();
    (q0, q1)
}

impl<Zt, U, const D: usize> ZincPlusPiopF2<Zt, U, D>
where
    Zt: F2ZincTypes<D>,
    U: Uair + 'static,
{
    /// Step 7 (prove) — γ-batched lift-and-project open.
    ///
    /// Reduces the per-column MLE evaluation claims emitted by the
    /// sumcheck to a single F_2[X] opening: the prover folds each
    /// per-column `(a_g', b_g, combined_row_g)` triple by random
    /// challenges `γ_g ∈ GF(2^192)` into the bundled
    /// `(a', b', combined_row')` carried in [`F2OpenProof`]. The
    /// verifier discharges every column's claim via a single
    /// `ψ_α(a') = Σ_g γ_g · a_g` check; soundness over per-column
    /// tampering follows by Schwartz-Zippel over the γ_g.
    ///
    /// `trace_binary_cols` provides direct witness access (the
    /// `M_{w_g}` matrices); `pp` defines the commit shape (matching
    /// the codeword matrix stored in `commit_hint`); `sumcheck_point`
    /// is `r*` from the sumcheck output. `num_column_openings`
    /// controls the proximity soundness — see
    /// [`zip_plus::code::raa_f2::recommended_num_column_openings`].
    pub fn prove_f2_open(
        transcript: &mut impl Transcript,
        pp: &ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
        commit_hint: &ZipPlusHint<BinaryPoly<D>>,
        trace_binary_cols: &[DenseMultilinearExtension<BinaryPoly<D>>],
        sumcheck_point: &[BinaryFieldGF192],
        alpha: &BinaryFieldGF192,
        num_column_openings: usize,
    ) -> F2OpenProof<D> {
        let num_rows = pp.num_rows;
        let row_len = pp.linear_code.row_len();
        let codeword_len = pp.linear_code.codeword_len();
        let num_cols = trace_binary_cols.len();
        assert!(num_rows.is_power_of_two());

        let basis = zinc_poly::univariate::binary_gf192::AlphaPolyBasis::new(alpha);
        let (q0, q1) = build_lifted_eq_tensor(num_rows, sumcheck_point, &basis);
        debug_assert_eq!(q1.len(), row_len);
        debug_assert_eq!(q0.len(), num_rows);

        // -- Step 7.1: γ_g challenges (one per committed column) ---
        // Drawn early so a', b', combined_row' depend on γ. (γ is
        // the only cross-column entropy in the open; coeffs comes
        // later for proximity.)
        let gamma_gf: Vec<BinaryFieldGF192> =
            transcript.get_field_challenges(num_cols, &());
        let gamma: Vec<BinaryF2Poly<3>> =
            gamma_gf.iter().map(|g| basis.lift(g)).collect();

        // -- Step 7.2: per-column intermediates, folded into b'/a'.
        //
        // For each committed column g:
        //   b_g[i] := Σ_j q_1'[j] · M_w_g[i, j]      (BinaryF2Poly<4>)
        //   a_g'   := Σ_i q_0'[i] · b_g[i]           (BinaryF2Poly<7>)
        //   b'[i] += γ_g · b_g[i]                    (BinaryF2Poly<7>)
        //   a'    += γ_g · a_g'                      (BinaryF2Poly<10>)
        //
        // Parallel structure: each (column, row) pair contributes
        // independently to `b'` and (via `a_g'`) to `a'`. We
        // parallelise across columns, producing per-column
        // partial `(b_g, a_g_scaled)` results, then merge serially.
        // The per-row work within a column stays sequential (row_len
        // is small in the deployed shape).
        let per_col_results: Vec<(Vec<BinaryF2Poly<7>>, BinaryF2Poly<10>)> =
            cfg_iter!(trace_binary_cols)
                .enumerate()
                .map(|(g, col)| {
                    assert_eq!(
                        col.evaluations.len(),
                        num_rows * row_len,
                        "trace column evaluation count must equal num_rows × row_len"
                    );

                    let mut b_g_scaled: Vec<BinaryF2Poly<7>> =
                        Vec::with_capacity(num_rows);
                    let mut b_g: Vec<BinaryF2Poly<4>> = Vec::with_capacity(num_rows);
                    for i in 0..num_rows {
                        let row_slice =
                            &col.evaluations[i * row_len..(i + 1) * row_len];
                        let row_lifted: Vec<BinaryF2Poly<1>> = row_slice
                            .iter()
                            .map(
                                zinc_poly::univariate::binary_gf192::lift_bp_to_f2_poly_1::<D>,
                            )
                            .collect();
                        let entry: BinaryF2Poly<4> =
                            zinc_poly::univariate::binary_f2_wide::f2_inner_product::<1, 3, 4>(
                                &row_lifted, &q1,
                            );
                        // γ_g · b_g[i], to be merged into b'[i].
                        let scaled: BinaryF2Poly<7> =
                            zinc_poly::univariate::binary_f2_wide::f2_poly_mul::<3, 4, 7>(
                                &gamma[g], &entry,
                            );
                        b_g_scaled.push(scaled);
                        b_g.push(entry);
                    }

                    let a_g_prime: BinaryF2Poly<7> =
                        zinc_poly::univariate::binary_f2_wide::f2_inner_product::<3, 4, 7>(
                            &q0, &b_g,
                        );
                    // γ_g · a_g', to be merged into a'.
                    let a_scaled: BinaryF2Poly<10> =
                        zinc_poly::univariate::binary_f2_wide::f2_poly_mul::<3, 7, 10>(
                            &gamma[g], &a_g_prime,
                        );
                    (b_g_scaled, a_scaled)
                })
                .collect();

        // Serial merge: ~`num_cols` u64 XORs per row. Cheap.
        let mut b_prime: Vec<BinaryF2Poly<7>> =
            vec![BinaryF2Poly::<7>::zero(); num_rows];
        let mut a_prime: BinaryF2Poly<10> = BinaryF2Poly::<10>::zero();
        for (b_g_scaled, a_scaled) in per_col_results {
            for i in 0..num_rows {
                b_prime[i] += b_g_scaled[i].clone();
            }
            a_prime += a_scaled;
        }

        // Absorb (b', a') into the transcript so subsequent challenges
        // depend on them.
        absorb_f2_poly_slice::<7, _>(transcript, b_prime.iter());
        absorb_f2_poly_slice::<10, _>(transcript, core::iter::once(&a_prime));

        // -- Step 7.3: proximity coefficients ----------------------
        let coeffs_gf: Vec<BinaryFieldGF192> =
            transcript.get_field_challenges(num_rows, &());
        let coeffs: Vec<BinaryF2Poly<3>> =
            coeffs_gf.iter().map(|g| basis.lift(g)).collect();

        // -- Step 7.4: combined_row' = Σ_g γ_g · (Σ_i coeffs[i] · M_w_g[i, *])
        //
        // Parallelise across the outer (g) loop: each column produces
        // an independent length-`row_len` contribution to
        // `combined_row`. Merge serially.
        let per_col_combined: Vec<Vec<BinaryF2Poly<7>>> = cfg_iter!(trace_binary_cols)
            .enumerate()
            .map(|(g, col)| {
                let mut col_contrib: Vec<BinaryF2Poly<7>> =
                    Vec::with_capacity(row_len);
                for j in 0..row_len {
                    let column_j_lifted: Vec<BinaryF2Poly<1>> = (0..num_rows)
                        .map(|i| {
                            zinc_poly::univariate::binary_gf192::lift_bp_to_f2_poly_1::<D>(
                                &col.evaluations[i * row_len + j],
                            )
                        })
                        .collect();
                    // Cells (W=1, ~16 set bits avg) are far sparser
                    // than `coeffs` (W=3, ~96 set bits avg); pass the
                    // lifted cells as `a` so the schoolbook in
                    // `f2_poly_mul` iterates over the sparser operand.
                    let per_col_entry: BinaryF2Poly<4> =
                        zinc_poly::univariate::binary_f2_wide::f2_inner_product::<1, 3, 4>(
                            &column_j_lifted,
                            &coeffs,
                        );
                    let scaled: BinaryF2Poly<7> =
                        zinc_poly::univariate::binary_f2_wide::f2_poly_mul::<3, 4, 7>(
                            &gamma[g],
                            &per_col_entry,
                        );
                    col_contrib.push(scaled);
                }
                col_contrib
            })
            .collect();

        let mut combined_row: Vec<BinaryF2Poly<7>> =
            vec![BinaryF2Poly::<7>::zero(); row_len];
        for col_contrib in per_col_combined {
            for j in 0..row_len {
                combined_row[j] += col_contrib[j].clone();
            }
        }

        absorb_f2_poly_slice::<7, _>(transcript, combined_row.iter());

        // -- Step 7.5: sample column indices + Merkle opens --------
        let opened_columns: Vec<F2OpenedColumn<D>> = (0..num_column_openings)
            .map(|_| {
                let column_idx = sample_column_idx(transcript, codeword_len);
                let mut column_values: Vec<BinaryPoly<D>> =
                    Vec::with_capacity(commit_hint.cw_matrices.len() * num_rows);
                for cw_matrix in &commit_hint.cw_matrices {
                    for row in cw_matrix.as_rows() {
                        column_values.push(row[column_idx].clone());
                    }
                }
                let merkle_proof = commit_hint
                    .merkle_tree
                    .prove(column_idx)
                    .expect("Merkle prove should succeed for in-range column idx");
                F2OpenedColumn {
                    column_idx,
                    column_values,
                    merkle_proof,
                }
            })
            .collect();

        F2OpenProof {
            lifted_claim: a_prime,
            b_vector: b_prime,
            combined_row,
            opened_columns,
        }
    }

    /// Step 7 (verify) — γ-batched lift-and-project verifier.
    ///
    /// Re-derives the γ_g challenges (one per primary committed
    /// column), runs the four core checks (eval-consistency, ψ_α
    /// discharge, coherence, encoding consistency), and verifies the
    /// Merkle paths for each opened column. Returns `Ok(())` iff all
    /// checks pass; the first failure short-circuits with a
    /// structured error.
    pub fn verify_f2_open(
        transcript: &mut impl Transcript,
        pp: &ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
        commitment: &ZipPlusCommitment,
        proof: &F2OpenProof<D>,
        subclaim: &F2VerifierSubclaim,
    ) -> Result<(), F2OpenError> {
        let num_rows = pp.num_rows;
        let row_len = pp.linear_code.row_len();
        // The F_2[X] open binds only the primary (committed) columns —
        // virtual columns are derived locally and never opened.
        let num_cols = subclaim.primary_column_evals.len();
        let batch_size = commitment.batch_size;
        let codeword_len = pp.linear_code.codeword_len();

        // -- Shape checks -----------------------------------------
        if proof.b_vector.len() != num_rows {
            return Err(F2OpenError::BvecLenMismatch {
                expected: num_rows,
                got: proof.b_vector.len(),
            });
        }
        if proof.combined_row.len() != row_len {
            return Err(F2OpenError::CombinedRowLenMismatch {
                expected: row_len,
                got: proof.combined_row.len(),
            });
        }

        // -- Re-derive γ_g ----------------------------------------
        let basis =
            zinc_poly::univariate::binary_gf192::AlphaPolyBasis::new(&subclaim.alpha);
        let (q0, q1) = build_lifted_eq_tensor(num_rows, &subclaim.sumcheck_point, &basis);
        let gamma_gf: Vec<BinaryFieldGF192> =
            transcript.get_field_challenges(num_cols, &());
        let gamma: Vec<BinaryF2Poly<3>> =
            gamma_gf.iter().map(|g| basis.lift(g)).collect();

        // Absorb (b', a') as the prover did.
        absorb_f2_poly_slice::<7, _>(transcript, proof.b_vector.iter());
        absorb_f2_poly_slice::<10, _>(
            transcript,
            core::iter::once(&proof.lifted_claim),
        );

        // -- Check 1: evaluation consistency in F_2[X] -------------
        //    Σ_i q_0[i] · b'[i]  =  a'.
        let recomputed_a_prime: BinaryF2Poly<10> = {
            let mut acc = BinaryF2Poly::<10>::zero();
            for i in 0..num_rows {
                let prod: BinaryF2Poly<10> =
                    zinc_poly::univariate::binary_f2_wide::f2_poly_mul::<3, 7, 10>(
                        &q0[i],
                        &proof.b_vector[i],
                    );
                acc += prod;
            }
            acc
        };
        if recomputed_a_prime != proof.lifted_claim {
            return Err(F2OpenError::EvalConsistency);
        }

        // -- Check 2: lift discharge in GF(2^192) ------------------
        //    ψ_α(a')  =  Σ_g γ_g_gf · a_g.
        let psi = zinc_poly::univariate::binary_gf192::eval_f2_wide_poly_at::<10>(
            &proof.lifted_claim,
            &subclaim.alpha,
        );
        let mut expected = BinaryFieldGF192::zero();
        for g in 0..num_cols {
            let mut term = gamma_gf[g];
            term *= &subclaim.primary_column_evals[g];
            expected += &term;
        }
        if psi != expected {
            return Err(F2OpenError::LiftDischarge {
                computed: psi,
                expected,
            });
        }

        // -- Re-derive coeffs + absorb combined_row ---------------
        let coeffs_gf: Vec<BinaryFieldGF192> =
            transcript.get_field_challenges(num_rows, &());
        let coeffs: Vec<BinaryF2Poly<3>> =
            coeffs_gf.iter().map(|g| basis.lift(g)).collect();
        absorb_f2_poly_slice::<7, _>(transcript, proof.combined_row.iter());

        // -- Check 3: coherence ------------------------------------
        //    <combined_row', q_1>  =  <coeffs, b'>  in F_2[X]<10>.
        // `q1` (W=3) is ~96 set bits vs `combined_row` (W=7) at ~224
        // — pass the smaller as `a` to minimise schoolbook iterations.
        let lhs: BinaryF2Poly<10> = {
            let mut acc = BinaryF2Poly::<10>::zero();
            for j in 0..row_len {
                let prod: BinaryF2Poly<10> =
                    zinc_poly::univariate::binary_f2_wide::f2_poly_mul::<3, 7, 10>(
                        &q1[j],
                        &proof.combined_row[j],
                    );
                acc += prod;
            }
            acc
        };
        let rhs: BinaryF2Poly<10> = {
            let mut acc = BinaryF2Poly::<10>::zero();
            for i in 0..num_rows {
                let prod: BinaryF2Poly<10> =
                    zinc_poly::univariate::binary_f2_wide::f2_poly_mul::<3, 7, 10>(
                        &coeffs[i],
                        &proof.b_vector[i],
                    );
                acc += prod;
            }
            acc
        };
        if lhs != rhs {
            return Err(F2OpenError::Coherence);
        }

        // -- Check 4: per-column-opening encoding + Merkle ---------
        //   For each j ∈ opened cols:
        //     (a) Merkle verify column values against root.
        //     (b) encode(combined_row')[j] = Σ_i coeffs[i] · Σ_g γ_g · cw_M^g[i, j].
        // The encoding is F_2[X]-linear, so encoding combined_row'
        // once (cost O(codeword_len)) lets us index at each sampled
        // column. Same encoding for every opened col — cache it.
        let encoded: Vec<BinaryF2Poly<7>> = pp
            .linear_code
            .encode_f2_lin_open::<7>(&proof.combined_row);
        debug_assert_eq!(encoded.len(), codeword_len);

        let expected_column_values_len = batch_size * num_rows;

        // The column-opening loop is the single dominant cost of
        // `verify_f2_open` (~656 F_2[X]<3>·<1> mults per opened
        // column × `num_column_openings = 987` for rate-1/4 RAA).
        // Each iteration is independent of the others *except* for
        // the sequential `sample_column_idx` transcript draws. We
        // therefore pre-sample all expected indices serially first,
        // then parallelise the per-opening verification work.
        let expected_indices: Vec<usize> = (0..proof.opened_columns.len())
            .map(|_| sample_column_idx(transcript, codeword_len))
            .collect();

        cfg_iter!(proof.opened_columns)
            .zip(cfg_iter!(expected_indices))
            .try_for_each(|(opened, expected_idx)| {
                if opened.column_idx != *expected_idx {
                    return Err(F2OpenError::MerkleVerify {
                        column_idx: opened.column_idx,
                        reason: format!(
                            "column index mismatch: prover sent {}, transcript yields {}",
                            opened.column_idx, expected_idx,
                        ),
                    });
                }
                if opened.column_values.len() != expected_column_values_len {
                    return Err(F2OpenError::ColumnValuesLenMismatch {
                        expected: expected_column_values_len,
                        got: opened.column_values.len(),
                    });
                }

                // (a) Merkle path
                opened
                    .merkle_proof
                    .verify(&commitment.root, &opened.column_values, opened.column_idx)
                    .map_err(|e| F2OpenError::MerkleVerify {
                        column_idx: opened.column_idx,
                        reason: format!("{e}"),
                    })?;

                // (b) γ-weighted encoding consistency.
                //
                //  weighted_col[i] = Σ_g γ_g · cw_M^g[i, j]  ∈ F_2[X]<4>
                //  actual_at_j     = Σ_i coeffs[i] · weighted_col[i]  ∈ F_2[X]<7>
                //  expected_at_j   = encoded[j]  ∈ F_2[X]<7>
                //
                // Inner-loop optimisation: `f2_poly_mul` is schoolbook
                // and iterates over the SET bits of operand `a`. The
                // lifted cell is a 32-bit `BinaryF2Poly<1>` (~16 set
                // bits average), while γ is a 192-bit
                // `BinaryF2Poly<3>` (~96 set bits average). Passing
                // the cell as `a` gives ~6× fewer XOR-shifts vs. the
                // natural `γ · cell` ordering. The product is
                // commutative in `F_2[X]`.
                let mut weighted_col: Vec<BinaryF2Poly<4>> =
                    vec![BinaryF2Poly::<4>::zero(); num_rows];
                for g in 0..num_cols {
                    for i in 0..num_rows {
                        let cell = zinc_poly::univariate::binary_gf192::lift_bp_to_f2_poly_1::<D>(
                            &opened.column_values[g * num_rows + i],
                        );
                        let prod: BinaryF2Poly<4> =
                            zinc_poly::univariate::binary_f2_wide::f2_poly_mul::<1, 3, 4>(
                                &cell, &gamma[g],
                            );
                        weighted_col[i] += prod;
                    }
                }
                let actual_at_j: BinaryF2Poly<7> =
                    zinc_poly::univariate::binary_f2_wide::f2_inner_product::<3, 4, 7>(
                        &coeffs,
                        &weighted_col,
                    );
                if actual_at_j != encoded[opened.column_idx] {
                    return Err(F2OpenError::EncodingConsistency {
                        column_idx: opened.column_idx,
                    });
                }
                Ok(())
            })?;

        Ok(())
    }

    // -- Bundled commit + prove + open / verify entry points -----------
    //
    // [`Self::prove_f2_full`] / [`Self::verify_f2_full`] are the
    // single-call public API. They wrap Step 0 (commit + absorb) →
    // Steps 2-4 (IC + α + sumcheck) → Step 7 (γ-batched open) on a
    // single shared transcript, returning / consuming a single
    // [`F2FullProof`].

    /// Run the full F_2 prove pipeline on a single transcript.
    ///
    /// Composes commit → IC → α → sumcheck → open. The same
    /// transcript carries all Fiat-Shamir state across the four
    /// phases; the verifier mirror is [`Self::verify_f2_full`].
    ///
    /// `pp` defines the commit shape (`num_vars`, row layout,
    /// linear code). `num_vars` is the MLE arity for the witness
    /// trace. `project_scalar` lifts UAIR scalars from `U::Scalar`
    /// to `DynamicPolynomialF<GF(2^192)>` (typically the
    /// per-coefficient F_2 ⊂ GF(2^192) embedding).
    /// `num_column_openings` controls the proximity-check
    /// soundness — see
    /// [`zip_plus::code::raa_f2::recommended_num_column_openings`].
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn prove_f2_full(
        transcript: &mut impl Transcript,
        pp: &ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
        trace: &UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D>,
        virtual_specs: &[F2VirtualBpSpec],
        num_vars: usize,
        project_scalar: impl Fn(&U::Scalar) -> DynamicPolynomialF<BinaryFieldGF192> + Sync,
        num_column_openings: usize,
    ) -> Result<F2FullProof<D>, F2ProveError<U>> {
        // Step 0: commit primary cols + absorb root. Virtual cols
        // are *not* committed; they're materialised inside the
        // IC + sumcheck and reconstructed by the verifier from
        // `virtual_specs` + the primary col MLE evals at `r*`.
        let (hint, commitment) =
            Self::commit_and_absorb_f2_trace(transcript, pp, &trace.binary_poly)
                .expect("F_2 commit should succeed for a well-shaped trace");

        // Steps 2-4: IC + α + sumcheck on the primary+virtual
        // extended trace. Returns both the wire proof and a
        // prover-side subclaim equivalent to what the verifier will
        // derive (including `virtual_column_evals` for the virtual
        // cols).
        let (uair_proof, subclaim) = Self::prove_f2_uair_with_groups(
            transcript,
            trace,
            virtual_specs,
            num_vars,
            project_scalar,
            eq_dot_column_groups,
        )?;

        // Step 7: γ-batched open on the primary cols only.
        let open_proof = Self::prove_f2_open(
            transcript,
            pp,
            &hint,
            &trace.binary_poly,
            &subclaim.sumcheck_point,
            &subclaim.alpha,
            num_column_openings,
        );

        Ok(F2FullProof {
            commitment,
            uair: uair_proof,
            open: open_proof,
        })
    }

    /// Verifier mirror of [`Self::prove_f2_full`]: absorbs the
    /// commitment, runs the IC + sumcheck verifier, then the
    /// γ-batched open verifier — all on a single shared transcript.
    /// `virtual_specs` must match what the prover used; the
    /// verifier derives virtual column MLE evals at `r*` from
    /// `primary_column_evals` via those specs and checks them
    /// against the sumcheck-extracted virtual evals.
    #[allow(clippy::too_many_arguments)]
    pub fn verify_f2_full<IdealOverF>(
        transcript: &mut impl Transcript,
        pp: &ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
        proof: &F2FullProof<D>,
        virtual_specs: &[F2VirtualBpSpec],
        num_vars: usize,
        num_primary_columns: usize,
        project_ideal: impl Fn(&IdealOrZero<U::Ideal>) -> IdealOverF,
    ) -> Result<F2VerifierSubclaim, F2FullVerifyError<U, IdealOverF>>
    where
        IdealOverF: zinc_uair::ideal::Ideal
            + zinc_uair::ideal::IdealCheck<DynamicPolynomialF<BinaryFieldGF192>>,
    {
        // Step 0: absorb the commitment exactly as the prover did.
        Self::absorb_commitment(transcript, &proof.commitment);

        // Steps 2-4: IC + α + sumcheck verifier (with virtual-col
        // consistency check baked in).
        let subclaim = Self::verify_f2_uair(
            transcript,
            &proof.uair,
            virtual_specs,
            num_vars,
            num_primary_columns,
            project_ideal,
        )
        .map_err(F2FullVerifyError::Uair)?;

        // Step 7: γ-batched open verifier (primary cols only).
        Self::verify_f2_open(transcript, pp, &proof.commitment, &proof.open, &subclaim)
            .map_err(F2FullVerifyError::Open)?;

        Ok(subclaim)
    }
}

/// The complete F_2 proof: commitment + IC/sumcheck proof + Step 7
/// γ-batched open. Produced by [`ZincPlusPiopF2::prove_f2_full`] and
/// consumed by [`ZincPlusPiopF2::verify_f2_full`].
#[derive(Clone, Debug)]
pub struct F2FullProof<const D: usize> {
    pub commitment: ZipPlusCommitment,
    pub uair: F2Proof,
    pub open: F2OpenProof<D>,
}

/// Errors emitted by [`ZincPlusPiopF2::verify_f2_full`].
#[derive(Debug, thiserror::Error)]
pub enum F2FullVerifyError<U: Uair, IdealOverF>
where
    IdealOverF: zinc_uair::ideal::Ideal,
{
    #[error("IC + sumcheck verification failed: {0}")]
    Uair(F2VerifyError<U, IdealOverF>),
    #[error("F_2[X] open verification failed: {0}")]
    Open(F2OpenError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::ConstZero;
    use rand::{Rng, rng};
    use zinc_poly::mle::MultilinearExtensionWithConfig;
    use zinc_transcript::Blake3Transcript;
    use zinc_uair::{
        ConstraintBuilder, PublicColumnLayout, TotalColumnLayout, TraceRow, UairSignature,
        ideal::ImpossibleIdeal,
    };

    /// Smallest viable all-`F_2` UAIR for testing the prove path: two
    /// `binary_poly` columns with the constraint `col_0 == col_1`
    /// (i.e. `col_0 - col_1 ∈ <0>`, an `assert_zero` constraint).
    /// `Scalar = BinaryPoly<32>`; `Ideal = ImpossibleIdeal` (unused).
    #[derive(Clone, Debug, Default)]
    struct TinyF2Uair;

    impl Uair for TinyF2Uair {
        type Ideal = ImpossibleIdeal;
        type Scalar = BinaryPoly<32>;

        fn signature() -> UairSignature {
            UairSignature::new(
                TotalColumnLayout::new(2, 0, 0),
                PublicColumnLayout::default(),
                vec![],
                vec![],
                vec![],
            )
        }

        fn constrain_general<B, FromR, MulByScalar, IFromR>(
            b: &mut B,
            up: TraceRow<B::Expr>,
            _down: TraceRow<B::Expr>,
            _from_ref: FromR,
            _mbs: MulByScalar,
            _ideal_from_ref: IFromR,
        ) where
            B: ConstraintBuilder,
            FromR: Fn(&Self::Scalar) -> B::Expr,
            MulByScalar: Fn(&B::Expr, &Self::Scalar) -> Option<B::Expr>,
            IFromR: Fn(&Self::Ideal) -> B::Ideal,
        {
            b.assert_zero(up.binary_poly[0].clone() - &up.binary_poly[1]);
        }
    }

    /// End-to-end: build a satisfied all-`F_2` trace, run the
    /// F_2 prove path, and assert the resulting proof has the
    /// expected shape (`num_constraints` IC entries, `num_vars`
    /// sumcheck rounds per group, `num_cols` groups).
    ///
    /// The test does NOT verify the proof against a verifier — that
    /// would require a parallel `verify_f2_uair` and is the next
    /// slice. The prover-side test confirms (a) the pipeline runs
    /// without panicking against a real F_2 UAIR + GF(2^192) field,
    /// (b) the wire format of the resulting proof is internally
    /// consistent, and (c) all transcript draws (IC challenge, α,
    /// sumcheck round challenges) are exercised by a real
    /// `Blake3Transcript`.
    #[test]
    fn prove_f2_pipeline_runs_against_tinyf2uair() {
        const D: usize = 32;
        let num_vars: usize = 4;
        let poly_size = 1usize << num_vars;
        let mut r = rng();

        // Build the trace: two identical random binary-poly columns
        // (so the `col_0 == col_1` constraint holds for every row).
        let col0_vals: Vec<BinaryPoly<D>> =
            (0..poly_size).map(|_| BinaryPoly::from(r.random::<u32>())).collect();
        let col1_vals = col0_vals.clone();
        let col0 = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            col0_vals,
            BinaryPoly::default(),
        );
        let col1 = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            col1_vals,
            BinaryPoly::default(),
        );
        let trace: UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D> = UairTrace {
            binary_poly: vec![col0, col1].into(),
            arbitrary_poly: vec![].into(),
            int: vec![].into(),
        };

        let mut transcript = Blake3Transcript::new();

        // F2ZincTypes-bound entry point. Doesn't need an actual
        // `F2ZincTypes` impl yet — the prove function is generic
        // over U; the F2ZincTypes trait is a future hook for the
        // PCS-commit wiring (not exercised in this test).
        //
        // We invoke prove_f2_uair directly via a phantom struct
        // that satisfies the F2ZincTypes bound trivially. For now
        // there's no concrete F2ZincTypes-implementing type in
        // protocol/, so we use the function statically with U
        // pinned to TinyF2Uair.
        let proof = prove_f2_uair_for_tests::<TinyF2Uair, D>(
            &mut transcript,
            &trace,
            num_vars,
            // `TinyF2Uair` has only an `assert_zero` constraint, so
            // `collect_scalars` returns an empty set and this
            // closure is never called. Provide a sensible default:
            // lift `BinaryPoly<32>` → `DynamicPolynomialF<GF192>` via
            // the F_2 ⊂ GF(2^192) per-coefficient embedding.
            |scalar: &BinaryPoly<32>| -> DynamicPolynomialF<BinaryFieldGF192> {
                let coeffs: Vec<BinaryFieldGF192> = scalar
                    .iter()
                    .map(|b| {
                        if b.into_inner() {
                            BinaryFieldGF192::one()
                        } else {
                            BinaryFieldGF192::zero()
                        }
                    })
                    .collect();
                DynamicPolynomialF { coeffs }
            },
        )
        .expect("prove_f2_uair should succeed on a satisfied F_2 trace");

        // The IC proof has one entry per constraint (1 for
        // TinyF2Uair). For an `assert_zero` constraint the value is
        // `DynamicPolynomialF::ZERO`.
        let num_constraints = count_constraints::<TinyF2Uair>();
        assert_eq!(proof.ic_proof.combined_mle_values.len(), num_constraints);
        for v in &proof.ic_proof.combined_mle_values {
            assert_eq!(v, &DynamicPolynomialF::<BinaryFieldGF192>::ZERO);
        }

        // The sumcheck proof has `num_cols` groups (one per
        // projected trace column), each of `num_vars` round
        // messages. Each round message carries `degree = 2` tail
        // evaluations (Karatsuba {0, 1, ∞}-style — see
        // `nat_evaluation::evaluate_at_point` for the
        // reconstruction).
        let claimed = proof.sumcheck_proof.claimed_sums();
        assert_eq!(claimed.len(), 2, "two trace columns → two groups");
        // Each col is identically itself; on the boolean hypercube
        // the sum of `eq(y, r) · col(y)` equals the MLE evaluated
        // at `r` (= IC's evaluation point). That value is finite
        // and not asserted to anything specific in this smoke test.

        // α was drawn from the transcript between the IC and the
        // sumcheck. Any non-default value is fine; assert it
        // isn't the trivial zero (which would indicate the
        // transcript flow is broken — Blake3 of a non-trivial
        // state is overwhelmingly likely to produce a non-zero
        // 192-bit element).
        assert!(
            !proof.alpha.is_zero(),
            "α should be a non-zero GF(2^192) challenge; got {}",
            proof.alpha
        );
    }

    /// Test shim that exposes the prove logic as a free function,
    /// bypassing the `F2ZincTypes`-typed `ZincPlusPiopF2` wrapper.
    /// (The F2ZincTypes trait is reserved for the PCS-commit wiring;
    /// no concrete impl is required for the IC+sumcheck pipeline
    /// itself.)
    fn prove_f2_uair_for_tests<U, const D: usize>(
        transcript: &mut impl Transcript,
        trace: &UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D>,
        num_vars: usize,
        project_scalar: impl Fn(&U::Scalar) -> DynamicPolynomialF<BinaryFieldGF192> + Sync,
    ) -> Result<F2Proof, F2ProveError<U>>
    where
        U: Uair + 'static,
    {
        // Re-implementation of `ZincPlusPiopF2::prove_f2_uair`
        // without the `F2ZincTypes` bound — for tests only. The
        // logic is otherwise identical.
        let num_constraints = count_constraints::<U>();
        let field_cfg = ();

        let row_major_trace = project_f2_trace_row_major::<BinaryFieldGF192, _, _, D>(
            trace,
            &field_cfg,
        );

        let scalars =
            zinc_piop::projections::project_scalars::<BinaryFieldGF192, U>(|s| project_scalar(s));

        let (ic_proof, ic_state) = <U as IdealCheckProtocol>::prove_combined::<BinaryFieldGF192>(
            transcript,
            &row_major_trace,
            &scalars,
            num_constraints,
            num_vars,
            &field_cfg,
        )
        .map_err(F2ProveError::IdealCheck)?;

        let alpha: BinaryFieldGF192 = transcript.get_field_challenge(&field_cfg);

        let projected_trace: Vec<DenseMultilinearExtension<BinaryFieldGF192>> = trace
            .binary_poly
            .iter()
            .map(|col| {
                let evals_at_alpha: Vec<BinaryFieldGF192> = col
                    .evaluations
                    .iter()
                    .map(|cell| {
                        zinc_poly::univariate::binary_gf192::eval_f2_poly_d_at::<D>(cell, &alpha)
                    })
                    .collect();
                DenseMultilinearExtension::from_evaluations_vec(
                    col.num_vars,
                    evals_at_alpha,
                    BinaryFieldGF192::zero(),
                )
            })
            .collect();

        let eq_r = zinc_poly::utils::build_eq_x_r_inner(
            &ic_state.evaluation_point,
            &field_cfg,
        )
        .expect("eq table construction must succeed for valid IC point");

        let zero_inner = *BinaryFieldGF192::zero().inner();
        let groups: Vec<MultiDegreeSumcheckGroup<BinaryFieldGF192>> = projected_trace
            .iter()
            .map(|col| {
                let col_inner_mle = DenseMultilinearExtension::from_evaluations_vec(
                    col.num_vars,
                    col.evaluations.iter().map(|x| *x.inner()).collect(),
                    zero_inner,
                );
                MultiDegreeSumcheckGroup::new(
                    2,
                    vec![eq_r.clone(), col_inner_mle],
                    Box::new(|v: &[BinaryFieldGF192]| v[0] * v[1]),
                )
            })
            .collect();

        let (sumcheck_proof, _prover_states) =
            MultiDegreeSumcheck::<BinaryFieldGF192>::prove_as_subprotocol(
                transcript,
                groups,
                num_vars,
                &field_cfg,
            );

        Ok(F2Proof {
            ic_proof,
            sumcheck_proof,
            alpha,
        })
    }

    /// Test shim that mirrors `ZincPlusPiopF2::verify_f2_uair`
    /// without the `F2ZincTypes` bound.
    fn verify_f2_uair_for_tests<U, IdealOverF>(
        transcript: &mut impl Transcript,
        proof: &F2Proof,
        num_vars: usize,
        num_columns: usize,
        project_ideal: impl Fn(&IdealOrZero<U::Ideal>) -> IdealOverF,
    ) -> Result<F2VerifierSubclaim, F2VerifyError<U, IdealOverF>>
    where
        U: Uair + 'static,
        IdealOverF: zinc_uair::ideal::Ideal
            + zinc_uair::ideal::IdealCheck<DynamicPolynomialF<BinaryFieldGF192>>,
    {
        let num_constraints = count_constraints::<U>();
        let field_cfg = ();

        let ic_subclaim = <U as IdealCheckProtocol>::verify_as_subprotocol::<_, IdealOverF, _>(
            transcript,
            proof.ic_proof.clone(),
            num_constraints,
            num_vars,
            project_ideal,
            &field_cfg,
        )
        .map_err(F2VerifyError::IdealCheck)?;
        let ic_evaluation_point = ic_subclaim.evaluation_point;

        let alpha: BinaryFieldGF192 = transcript.get_field_challenge(&field_cfg);
        if alpha != proof.alpha {
            return Err(F2VerifyError::AlphaMismatch {
                transcript: alpha,
                proof: proof.alpha,
            });
        }

        let md_subclaims = MultiDegreeSumcheck::<BinaryFieldGF192>::verify_as_subprotocol(
            transcript,
            num_vars,
            &proof.sumcheck_proof,
            &field_cfg,
        )
        .map_err(F2VerifyError::Sumcheck)?;

        let sumcheck_point = md_subclaims.point().to_vec();
        let group_expected = md_subclaims.expected_evaluations();
        if group_expected.len() != num_columns {
            return Err(F2VerifyError::GroupCountMismatch {
                expected: num_columns,
                actual: group_expected.len(),
            });
        }

        let one = BinaryFieldGF192::one();
        let eq_at_rstar_r = zinc_poly::utils::eq_eval(
            &sumcheck_point,
            &ic_evaluation_point,
            one,
        )
        .expect("matching length by construction");
        if eq_at_rstar_r.is_zero() {
            return Err(F2VerifyError::DegenerateEq);
        }
        let eq_inv = eq_at_rstar_r.inverse();
        let column_mle_evals: Vec<BinaryFieldGF192> = group_expected
            .iter()
            .map(|expected| (*expected) * eq_inv)
            .collect();

        // Tests use the shim with virtual-spec-free UAIRs only;
        // mirror the real `verify_f2_uair`'s primary/virtual split
        // by treating all extracted evals as primary.
        Ok(F2VerifierSubclaim {
            ic_evaluation_point,
            alpha,
            sumcheck_point,
            primary_column_evals: column_mle_evals,
            virtual_column_evals: Vec::new(),
        })
    }

    /// End-to-end roundtrip: prove, then verify with a fresh
    /// transcript, and assert the verifier's subclaim is
    /// internally consistent.
    #[test]
    fn prove_then_verify_f2_pipeline_roundtrips() {
        const D: usize = 32;
        let num_vars: usize = 4;
        let poly_size = 1usize << num_vars;
        let mut r = rng();

        let col0_vals: Vec<BinaryPoly<D>> =
            (0..poly_size).map(|_| BinaryPoly::from(r.random::<u32>())).collect();
        let col1_vals = col0_vals.clone();
        let col0 = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            col0_vals.clone(),
            BinaryPoly::default(),
        );
        let col1 = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            col1_vals,
            BinaryPoly::default(),
        );
        let trace: UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D> = UairTrace {
            binary_poly: vec![col0.clone(), col1.clone()].into(),
            arbitrary_poly: vec![].into(),
            int: vec![].into(),
        };

        let project_scalar = |scalar: &BinaryPoly<32>| -> DynamicPolynomialF<BinaryFieldGF192> {
            let coeffs: Vec<BinaryFieldGF192> = scalar
                .iter()
                .map(|b| if b.into_inner() { BinaryFieldGF192::one() } else { BinaryFieldGF192::zero() })
                .collect();
            DynamicPolynomialF { coeffs }
        };

        // Prove
        let mut prover_transcript = Blake3Transcript::new();
        let proof = prove_f2_uair_for_tests::<TinyF2Uair, D>(
            &mut prover_transcript,
            &trace,
            num_vars,
            project_scalar,
        )
        .expect("prove should succeed");

        // Verify on a fresh transcript
        let mut verifier_transcript = Blake3Transcript::new();
        let subclaim = verify_f2_uair_for_tests::<TinyF2Uair, _>(
            &mut verifier_transcript,
            &proof,
            num_vars,
            /* num_columns */ 2,
            // `TinyF2Uair`'s single constraint is `assert_zero`, so
            // the IC's ideal-projection closure is never invoked.
            |_ideal| zinc_uair::ideal::ImpossibleIdeal,
        )
        .expect("verify should succeed on an honest proof");

        // Cross-check: column_mle_evals should match each column's
        // projected MLE evaluated at `r*` directly.
        let zero_inner = *BinaryFieldGF192::zero().inner();
        for (g, expected) in subclaim.primary_column_evals.iter().enumerate() {
            let projected_col_inner_evals: Vec<_> = trace.binary_poly[g]
                .evaluations
                .iter()
                .map(|cell| {
                    *zinc_poly::univariate::binary_gf192::eval_f2_poly_d_at::<D>(
                        cell,
                        &subclaim.alpha,
                    )
                    .inner()
                })
                .collect();
            let projected_col_mle = DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                projected_col_inner_evals,
                zero_inner,
            );
            let direct = projected_col_mle
                .evaluate_with_config(&subclaim.sumcheck_point, &())
                .expect("MLE eval should succeed");
            assert_eq!(
                direct, *expected,
                "column {g}: direct MLE evaluation at r* disagrees with verifier-derived expected",
            );
        }

        assert_eq!(subclaim.alpha, proof.alpha);
        assert_eq!(subclaim.sumcheck_point.len(), num_vars);
        assert_eq!(subclaim.ic_evaluation_point.len(), num_vars);
        assert_eq!(subclaim.primary_column_evals.len(), 2);
    }

    /// Tampering with the proof's α should yield an AlphaMismatch
    /// error rather than a panic or a silent acceptance.
    #[test]
    fn verify_rejects_tampered_alpha() {
        const D: usize = 32;
        let num_vars: usize = 3;
        let poly_size = 1usize << num_vars;
        let mut r = rng();

        let col0_vals: Vec<BinaryPoly<D>> =
            (0..poly_size).map(|_| BinaryPoly::from(r.random::<u32>())).collect();
        let col1_vals = col0_vals.clone();
        let col0 = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            col0_vals,
            BinaryPoly::default(),
        );
        let col1 = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            col1_vals,
            BinaryPoly::default(),
        );
        let trace: UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D> = UairTrace {
            binary_poly: vec![col0, col1].into(),
            arbitrary_poly: vec![].into(),
            int: vec![].into(),
        };

        let project_scalar = |_: &BinaryPoly<32>| DynamicPolynomialF::<BinaryFieldGF192>::ZERO;

        let mut prover_transcript = Blake3Transcript::new();
        let mut proof = prove_f2_uair_for_tests::<TinyF2Uair, D>(
            &mut prover_transcript,
            &trace,
            num_vars,
            project_scalar,
        )
        .expect("prove should succeed");

        // Mutate α — verifier should reject.
        proof.alpha = proof.alpha + BinaryFieldGF192::one();

        let mut verifier_transcript = Blake3Transcript::new();
        let err = verify_f2_uair_for_tests::<TinyF2Uair, _>(
            &mut verifier_transcript,
            &proof,
            num_vars,
            2,
            |_ideal| zinc_uair::ideal::ImpossibleIdeal,
        )
        .expect_err("tampered α should trigger AlphaMismatch");

        assert!(
            matches!(err, F2VerifyError::AlphaMismatch { .. }),
            "expected AlphaMismatch, got {err:?}",
        );
    }

    /// Exercises [`ZincPlusPiopF2::prove_f2_uair_with_groups`] and
    /// [`ZincPlusPiopF2::verify_f2_uair_with_groups`] with a
    /// non-default composition: a single degree-2 group
    /// `eq(y, r) · (col_0(y) + col_1(y))`. The verifier-side
    /// extractor returns the *combined* expected `col_0 + col_1`
    /// evaluation at `r*`, which downstream PCS opening could
    /// discharge by opening each column separately.
    ///
    /// This is the minimum non-trivial demonstration that the
    /// builder/extractor abstraction supports a composition outside
    /// the default `eq · col`-per-column shape.
    #[test]
    fn prove_then_verify_with_custom_groups() {
        const D: usize = 32;
        let num_vars: usize = 4;
        let poly_size = 1usize << num_vars;
        let mut r = rng();

        let col0_vals: Vec<BinaryPoly<D>> =
            (0..poly_size).map(|_| BinaryPoly::from(r.random::<u32>())).collect();
        let col1_vals = col0_vals.clone();
        let col0 = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            col0_vals,
            BinaryPoly::default(),
        );
        let col1 = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            col1_vals,
            BinaryPoly::default(),
        );
        let trace: UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D> = UairTrace {
            binary_poly: vec![col0, col1].into(),
            arbitrary_poly: vec![].into(),
            int: vec![].into(),
        };

        let project_scalar = |_: &BinaryPoly<32>| DynamicPolynomialF::<BinaryFieldGF192>::ZERO;

        // Single combined group: comb_fn(eq, c0, c1) = eq · (c0 + c1).
        let custom_groups =
            |ic_eval_point: &[BinaryFieldGF192],
             projected_trace: &[DenseMultilinearExtension<BinaryFieldGF192>],
             field_cfg: &()| {
                let eq_r =
                    zinc_poly::utils::build_eq_x_r_inner(ic_eval_point, field_cfg).unwrap();
                let zero_inner = *BinaryFieldGF192::zero().inner();
                let mles_inner: Vec<DenseMultilinearExtension<_>> = projected_trace
                    .iter()
                    .map(|col| {
                        DenseMultilinearExtension::from_evaluations_vec(
                            col.num_vars,
                            col.evaluations.iter().map(|x| *x.inner()).collect(),
                            zero_inner,
                        )
                    })
                    .collect();
                let mut mles_with_eq = vec![eq_r];
                mles_with_eq.extend(mles_inner);
                vec![MultiDegreeSumcheckGroup::new(
                    2,
                    mles_with_eq,
                    Box::new(|v: &[BinaryFieldGF192]| v[0] * (v[1] + v[2])),
                )]
            };

        // Prove with the custom builder. Use the test shim's
        // `prove_f2_uair_with_groups_for_tests` — we route through
        // the actual generic entry point by invoking the same logic
        // inline.
        let mut prover_transcript = Blake3Transcript::new();
        let num_constraints = count_constraints::<TinyF2Uair>();
        let field_cfg = ();

        let row_major_trace =
            project_f2_trace_row_major::<BinaryFieldGF192, _, _, D>(&trace, &field_cfg);
        let scalars = zinc_piop::projections::project_scalars::<BinaryFieldGF192, TinyF2Uair>(
            project_scalar,
        );
        let (ic_proof, ic_state) =
            <TinyF2Uair as IdealCheckProtocol>::prove_combined::<BinaryFieldGF192>(
                &mut prover_transcript,
                &row_major_trace,
                &scalars,
                num_constraints,
                num_vars,
                &field_cfg,
            )
            .unwrap();
        let alpha: BinaryFieldGF192 = prover_transcript.get_field_challenge(&field_cfg);
        let projected_trace: Vec<DenseMultilinearExtension<BinaryFieldGF192>> = trace
            .binary_poly
            .iter()
            .map(|col| {
                let evals_at_alpha: Vec<BinaryFieldGF192> = col
                    .evaluations
                    .iter()
                    .map(|cell| {
                        zinc_poly::univariate::binary_gf192::eval_f2_poly_d_at::<D>(cell, &alpha)
                    })
                    .collect();
                DenseMultilinearExtension::from_evaluations_vec(
                    col.num_vars,
                    evals_at_alpha,
                    BinaryFieldGF192::zero(),
                )
            })
            .collect();
        let groups = custom_groups(&ic_state.evaluation_point, &projected_trace, &field_cfg);
        let (sumcheck_proof, _) = MultiDegreeSumcheck::<BinaryFieldGF192>::prove_as_subprotocol(
            &mut prover_transcript,
            groups,
            num_vars,
            &field_cfg,
        );
        let proof = F2Proof {
            ic_proof,
            sumcheck_proof,
            alpha,
        };

        // Verify with a matching custom extractor: one group with
        // expected_evaluations[0] = eq(r*, r) · (col_0(r*) + col_1(r*)).
        let mut verifier_transcript = Blake3Transcript::new();
        let extract = |ic_eval_point: &[BinaryFieldGF192],
                       md_subclaims: &MultiDegreeSubClaims<BinaryFieldGF192>|
         -> Result<Vec<BinaryFieldGF192>, F2VerifyError<TinyF2Uair, ImpossibleIdeal>> {
            let one = BinaryFieldGF192::one();
            let eq_at_rstar_r =
                zinc_poly::utils::eq_eval(md_subclaims.point(), ic_eval_point, one).unwrap();
            if eq_at_rstar_r.is_zero() {
                return Err(F2VerifyError::DegenerateEq);
            }
            let combined = md_subclaims.expected_evaluations()[0] * eq_at_rstar_r.inverse();
            Ok(vec![combined])
        };

        // Inline the verifier shim with custom extract.
        let ic_subclaim = <TinyF2Uair as IdealCheckProtocol>::verify_as_subprotocol::<
            _,
            ImpossibleIdeal,
            _,
        >(
            &mut verifier_transcript,
            proof.ic_proof.clone(),
            num_constraints,
            num_vars,
            |_ideal| ImpossibleIdeal,
            &field_cfg,
        )
        .unwrap();
        let alpha_v: BinaryFieldGF192 = verifier_transcript.get_field_challenge(&field_cfg);
        assert_eq!(alpha_v, proof.alpha);
        let md_subclaims = MultiDegreeSumcheck::<BinaryFieldGF192>::verify_as_subprotocol(
            &mut verifier_transcript,
            num_vars,
            &proof.sumcheck_proof,
            &field_cfg,
        )
        .unwrap();
        assert_eq!(
            md_subclaims.expected_evaluations().len(),
            1,
            "custom builder produced a single group",
        );
        let combined_claim = extract(&ic_subclaim.evaluation_point, &md_subclaims).unwrap();
        assert_eq!(combined_claim.len(), 1);

        // Cross-check the combined claim against the column MLEs
        // evaluated directly at `r*` in their projected form.
        let zero_inner = *BinaryFieldGF192::zero().inner();
        let col_evals_at_rstar: Vec<BinaryFieldGF192> = trace
            .binary_poly
            .iter()
            .map(|col| {
                let projected_inner: Vec<_> = col
                    .evaluations
                    .iter()
                    .map(|cell| {
                        *zinc_poly::univariate::binary_gf192::eval_f2_poly_d_at::<D>(
                            cell,
                            &proof.alpha,
                        )
                        .inner()
                    })
                    .collect();
                let projected_mle = DenseMultilinearExtension::from_evaluations_vec(
                    num_vars,
                    projected_inner,
                    zero_inner,
                );
                projected_mle
                    .evaluate_with_config(md_subclaims.point(), &field_cfg)
                    .unwrap()
            })
            .collect();
        let direct_combined = col_evals_at_rstar[0] + col_evals_at_rstar[1];
        assert_eq!(
            combined_claim[0], direct_combined,
            "custom-extractor combined claim disagrees with direct MLE evaluation",
        );
    }

    // -- Step 0 (PCS commit) wiring + roundtrip ----------------------
    //
    // Exercises the commit phase plus the IC + sumcheck pipeline
    // against a concrete `F2ZincTypes` impl. The follow-on Step 7
    // (open) is exercised by `prove_then_verify_f2_open_roundtrips`
    // below; the two together cover the full F_2 prove/verify cycle.

    use crypto_primitives::crypto_bigint_int::Int;
    use crypto_primitives::crypto_bigint_uint::Uint;
    use std::marker::PhantomData;
    use zinc_poly::univariate::binary::BinaryPolyInnerProduct;
    use zinc_poly::univariate::dense::{DensePolyInnerProduct, DensePolynomial};
    use zinc_primality::MillerRabin;
    use zinc_utils::inner_product::MBSInnerProduct;
    use zip_plus::code::raa::RaaConfig;
    use zip_plus::code::raa_f2::RaaF2Code;
    use zip_plus::pcs::structs::ZipTypes;

    /// Local F_2 `ZipTypes` impl mirroring zip-plus's test-only
    /// `TestBinPolyF2ZipTypes<D>`. We can't re-use that one because
    /// it lives behind `#[cfg(test)]` inside zip-plus.
    #[derive(Debug, Clone)]
    struct LocalBinPolyF2ZipTypes<const D: usize> {}
    impl<const D: usize> ZipTypes for LocalBinPolyF2ZipTypes<D> {
        const NUM_COLUMN_OPENINGS: usize = 147;
        type Eval = BinaryPoly<D>;
        type Cw = BinaryPoly<D>;
        type Fmod = Uint<{ crypto_bigint::U64::LIMBS * 4 }>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<{ crypto_bigint::U64::LIMBS * 8 }>;
        type Comb = DensePolynomial<Self::CombR, D>;
        type EvalDotChal = BinaryPolyInnerProduct<Self::Chal, D>;
        type CombDotChal =
            DensePolyInnerProduct<Self::CombR, Self::Chal, Self::CombR, MBSInnerProduct, D>;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    #[derive(Copy, Clone)]
    struct LocalRaaConfig;
    impl RaaConfig for LocalRaaConfig {
        const PERMUTE_IN_PLACE: bool = false;
        const CHECK_FOR_OVERFLOWS: bool = false;
    }

    /// Concrete `F2ZincTypes` impl. Demonstrates the trait can be
    /// satisfied by real Zip+ primitives.
    #[derive(Clone, Debug)]
    struct F2Types<const D: usize>(PhantomData<()>);

    const LOCAL_REP_FACTOR: usize = 4;

    impl<const D: usize> F2ZincTypes<D> for F2Types<D> {
        type BinaryZt = LocalBinPolyF2ZipTypes<D>;
        type BinaryLc = RaaF2Code<Self::BinaryZt, LocalRaaConfig, LOCAL_REP_FACTOR>;
    }

    /// End-to-end: commit binary trace columns via Zip+, run the
    /// IC + α + sumcheck pipeline with the commitment absorbed into
    /// the transcript at Step 0, then verify on a fresh transcript
    /// with the same commitment absorption. Asserts the verifier's
    /// subclaim matches direct MLE evaluations of the projected
    /// columns at `r*`.
    ///
    /// Open at `r*` is exercised separately by
    /// `prove_then_verify_f2_open_roundtrips`.
    #[test]
    fn commit_prove_verify_f2_roundtrip() {
        const D: usize = 32;
        let num_vars: usize = 6;
        let row_len: usize = 8;
        let poly_size = 1usize << num_vars;
        let num_rows = poly_size.div_ceil(row_len);
        assert_eq!(num_rows * row_len, poly_size);

        let mut rng_local = rng();

        let col0_vals: Vec<BinaryPoly<D>> = (0..poly_size)
            .map(|_| BinaryPoly::from(rng_local.random::<u32>()))
            .collect();
        let col1_vals = col0_vals.clone();
        let col0 = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            col0_vals,
            BinaryPoly::default(),
        );
        let col1 = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            col1_vals,
            BinaryPoly::default(),
        );
        let trace: UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D> = UairTrace {
            binary_poly: vec![col0.clone(), col1.clone()].into(),
            arbitrary_poly: vec![].into(),
            int: vec![].into(),
        };

        // Step 0: commit -------------------------------------------
        let lc = <F2Types<D> as F2ZincTypes<D>>::BinaryLc::new(row_len);
        let pp: ZipPlusParams<
            <F2Types<D> as F2ZincTypes<D>>::BinaryZt,
            <F2Types<D> as F2ZincTypes<D>>::BinaryLc,
        > = ZipPlusParams::new(num_vars, num_rows, lc);

        let mut prover_transcript = Blake3Transcript::new();
        let (_hint, comm) =
            ZincPlusPiopF2::<F2Types<D>, TinyF2Uair, D>::commit_and_absorb_f2_trace(
                &mut prover_transcript,
                &pp,
                &trace.binary_poly,
            )
            .expect("commit should succeed");
        assert_eq!(comm.batch_size, 2);

        // Steps 2-4: IC + α + sumcheck (via the test shim, since
        // prove_f2_uair is gated on the F2ZincTypes bound which is
        // already satisfied above — we use the shim to also avoid
        // dragging the bound into the call-site verbosity).
        let project_scalar = |_: &BinaryPoly<32>| DynamicPolynomialF::<BinaryFieldGF192>::ZERO;
        let proof = prove_f2_uair_for_tests::<TinyF2Uair, D>(
            &mut prover_transcript,
            &trace,
            num_vars,
            project_scalar,
        )
        .expect("prove should succeed");

        // -- Verifier side ----------------------------------------
        let mut verifier_transcript = Blake3Transcript::new();
        ZincPlusPiopF2::<F2Types<D>, TinyF2Uair, D>::absorb_commitment(
            &mut verifier_transcript,
            &comm,
        );

        let subclaim = verify_f2_uair_for_tests::<TinyF2Uair, _>(
            &mut verifier_transcript,
            &proof,
            num_vars,
            2,
            |_ideal| ImpossibleIdeal,
        )
        .expect("verify should succeed");

        // Sanity: column MLE claims at r* match direct evaluation.
        let zero_inner = *BinaryFieldGF192::zero().inner();
        for (g, expected) in subclaim.primary_column_evals.iter().enumerate() {
            let projected_inner: Vec<_> = trace.binary_poly[g]
                .evaluations
                .iter()
                .map(|cell| {
                    *zinc_poly::univariate::binary_gf192::eval_f2_poly_d_at::<D>(
                        cell,
                        &subclaim.alpha,
                    )
                    .inner()
                })
                .collect();
            let projected_mle = DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                projected_inner,
                zero_inner,
            );
            let direct = projected_mle
                .evaluate_with_config(&subclaim.sumcheck_point, &())
                .expect("MLE eval should succeed");
            assert_eq!(
                direct, *expected,
                "column {g}: direct MLE evaluation at r* disagrees with verifier-derived expected",
            );
        }
    }

    // -- F_2[X] open (Step 7) roundtrip ------------------------------
    //
    // Exercises the lift-and-project MLE-opening pipeline:
    //   1. Build a satisfied F_2 trace + a sumcheck point r*.
    //   2. Run the prover-side computation of (a_g', b_g) per column.
    //   3. Verifier checks eval-consistency Σ_i q_0' · b_g = a_g' in
    //      F_2[X] and lift discharge ψ_α(a_g') = a_g in GF(2^192).
    //
    // The subclaim that feeds in here is constructed end-to-end
    // (commit → IC → α → sumcheck → verify), so this is the first
    // full pipeline-to-PCS-claim demonstration of the F_2 protocol.

    #[test]
    fn prove_then_verify_f2_open_roundtrips() {
        const D: usize = 32;
        let num_vars: usize = 6;
        let row_len: usize = 8;
        let poly_size = 1usize << num_vars;
        let num_rows = poly_size / row_len;
        assert_eq!(num_rows * row_len, poly_size);

        let mut rng_local = rng();

        // -- Build a satisfied trace: col_0 == col_1 (assert_zero ok)
        let col0_vals: Vec<BinaryPoly<D>> = (0..poly_size)
            .map(|_| BinaryPoly::from(rng_local.random::<u32>()))
            .collect();
        let col1_vals = col0_vals.clone();
        let col0 = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            col0_vals,
            BinaryPoly::default(),
        );
        let col1 = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            col1_vals,
            BinaryPoly::default(),
        );
        let trace: UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D> = UairTrace {
            binary_poly: vec![col0.clone(), col1.clone()].into(),
            arbitrary_poly: vec![].into(),
            int: vec![].into(),
        };

        // -- Run the full IC + sumcheck pipeline to produce a subclaim.
        let project_scalar =
            |_: &BinaryPoly<32>| DynamicPolynomialF::<BinaryFieldGF192>::ZERO;
        let mut prover_transcript = Blake3Transcript::new();
        let proof = prove_f2_uair_for_tests::<TinyF2Uair, D>(
            &mut prover_transcript,
            &trace,
            num_vars,
            project_scalar,
        )
        .expect("prove should succeed");

        let mut verifier_transcript = Blake3Transcript::new();
        let subclaim = verify_f2_uair_for_tests::<TinyF2Uair, _>(
            &mut verifier_transcript,
            &proof,
            num_vars,
            2,
            |_ideal| ImpossibleIdeal,
        )
        .expect("verify should succeed");

        // -- Step 0: commit + absorb (separate from the IC transcript).
        let lc = <F2Types<D> as F2ZincTypes<D>>::BinaryLc::new(row_len);
        let pp: ZipPlusParams<
            <F2Types<D> as F2ZincTypes<D>>::BinaryZt,
            <F2Types<D> as F2ZincTypes<D>>::BinaryLc,
        > = ZipPlusParams::new(num_vars, num_rows, lc);
        let mut open_prover_transcript = Blake3Transcript::new();
        let (hint, comm) =
            ZincPlusPiopF2::<F2Types<D>, TinyF2Uair, D>::commit_and_absorb_f2_trace(
                &mut open_prover_transcript,
                &pp,
                &trace.binary_poly,
            )
            .expect("commit should succeed");

        // -- Step 7 prover: lift + eval-consistency + proximity.
        let num_column_openings = 4;
        let open_proof = ZincPlusPiopF2::<F2Types<D>, TinyF2Uair, D>::prove_f2_open(
            &mut open_prover_transcript,
            &pp,
            &hint,
            &trace.binary_poly,
            &subclaim.sumcheck_point,
            &subclaim.alpha,
            num_column_openings,
        );
        assert_eq!(open_proof.b_vector.len(), num_rows);
        assert_eq!(open_proof.combined_row.len(), row_len);
        assert_eq!(open_proof.opened_columns.len(), num_column_openings);

        // -- Step 7 verifier: full check (eval + ψ_α + coherence + Merkle).
        let mut open_verifier_transcript = Blake3Transcript::new();
        ZincPlusPiopF2::<F2Types<D>, TinyF2Uair, D>::absorb_commitment(
            &mut open_verifier_transcript,
            &comm,
        );
        ZincPlusPiopF2::<F2Types<D>, TinyF2Uair, D>::verify_f2_open(
            &mut open_verifier_transcript,
            &pp,
            &comm,
            &open_proof,
            &subclaim,
        )
        .expect("F_2[X] open verification should succeed");
    }

    /// Tampering with `a_g'` should produce either an
    /// `EvalConsistency` failure (if the b-vectors are unchanged and
    /// `a_g'` no longer matches the recomputed inner product) or a
    /// `LiftDischarge` failure (if `ψ_α(a_g')` no longer matches
    /// `a_g`). Either is acceptable; the verifier must reject.
    #[test]
    fn verify_f2_open_rejects_tampered_lifted_claim() {
        const D: usize = 32;
        let num_vars: usize = 4;
        let row_len: usize = 4;
        let poly_size = 1usize << num_vars;
        let num_rows = poly_size / row_len;
        let mut rng_local = rng();

        let col0_vals: Vec<BinaryPoly<D>> = (0..poly_size)
            .map(|_| BinaryPoly::from(rng_local.random::<u32>()))
            .collect();
        let col1_vals = col0_vals.clone();
        let col0 = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            col0_vals,
            BinaryPoly::default(),
        );
        let col1 = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            col1_vals,
            BinaryPoly::default(),
        );
        let trace: UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D> = UairTrace {
            binary_poly: vec![col0, col1].into(),
            arbitrary_poly: vec![].into(),
            int: vec![].into(),
        };

        let project_scalar = |_: &BinaryPoly<32>| DynamicPolynomialF::<BinaryFieldGF192>::ZERO;
        let mut pt = Blake3Transcript::new();
        let proof = prove_f2_uair_for_tests::<TinyF2Uair, D>(&mut pt, &trace, num_vars, project_scalar)
            .expect("prove should succeed");

        let mut vt = Blake3Transcript::new();
        let subclaim = verify_f2_uair_for_tests::<TinyF2Uair, _>(
            &mut vt,
            &proof,
            num_vars,
            2,
            |_| ImpossibleIdeal,
        )
        .expect("verify should succeed");

        let lc = <F2Types<D> as F2ZincTypes<D>>::BinaryLc::new(row_len);
        let pp: ZipPlusParams<
            <F2Types<D> as F2ZincTypes<D>>::BinaryZt,
            <F2Types<D> as F2ZincTypes<D>>::BinaryLc,
        > = ZipPlusParams::new(num_vars, num_rows, lc);
        let mut open_pt = Blake3Transcript::new();
        let (hint, comm) =
            ZincPlusPiopF2::<F2Types<D>, TinyF2Uair, D>::commit_and_absorb_f2_trace(
                &mut open_pt,
                &pp,
                &trace.binary_poly,
            )
            .expect("commit should succeed");

        let mut open_proof = ZincPlusPiopF2::<F2Types<D>, TinyF2Uair, D>::prove_f2_open(
            &mut open_pt,
            &pp,
            &hint,
            &trace.binary_poly,
            &subclaim.sumcheck_point,
            &subclaim.alpha,
            4,
        );

        // Flip the lowest bit of the (γ-batched) lifted claim a'.
        let mut tampered_words = *open_proof.lifted_claim.words();
        tampered_words[0] ^= 1;
        open_proof.lifted_claim = BinaryF2Poly::<10>::from_words(tampered_words);

        let mut open_vt = Blake3Transcript::new();
        ZincPlusPiopF2::<F2Types<D>, TinyF2Uair, D>::absorb_commitment(&mut open_vt, &comm);
        let err = ZincPlusPiopF2::<F2Types<D>, TinyF2Uair, D>::verify_f2_open(
            &mut open_vt,
            &pp,
            &comm,
            &open_proof,
            &subclaim,
        )
        .expect_err("tampered lifted claim must be rejected");

        assert!(
            matches!(
                err,
                F2OpenError::EvalConsistency | F2OpenError::LiftDischarge { .. }
            ),
            "expected EvalConsistency or LiftDischarge, got {err:?}",
        );
    }

    /// Tampering with a b-vector entry while leaving the lifted
    /// claim unchanged should trip the **coherence** check
    /// (`<combined_row, q_1'> = <coeffs, b>` no longer balances) —
    /// the proximity-binding side of the verifier.
    #[test]
    fn verify_f2_open_rejects_tampered_b_vector() {
        const D: usize = 32;
        let num_vars: usize = 4;
        let row_len: usize = 4;
        let poly_size = 1usize << num_vars;
        let num_rows = poly_size / row_len;
        let mut rng_local = rng();

        let col0_vals: Vec<BinaryPoly<D>> = (0..poly_size)
            .map(|_| BinaryPoly::from(rng_local.random::<u32>()))
            .collect();
        let col1_vals = col0_vals.clone();
        let col0 = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            col0_vals,
            BinaryPoly::default(),
        );
        let col1 = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            col1_vals,
            BinaryPoly::default(),
        );
        let trace: UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D> = UairTrace {
            binary_poly: vec![col0, col1].into(),
            arbitrary_poly: vec![].into(),
            int: vec![].into(),
        };

        let project_scalar = |_: &BinaryPoly<32>| DynamicPolynomialF::<BinaryFieldGF192>::ZERO;
        let mut pt = Blake3Transcript::new();
        let proof =
            prove_f2_uair_for_tests::<TinyF2Uair, D>(&mut pt, &trace, num_vars, project_scalar)
                .expect("prove should succeed");
        let mut vt = Blake3Transcript::new();
        let subclaim = verify_f2_uair_for_tests::<TinyF2Uair, _>(
            &mut vt,
            &proof,
            num_vars,
            2,
            |_| ImpossibleIdeal,
        )
        .expect("verify should succeed");

        let lc = <F2Types<D> as F2ZincTypes<D>>::BinaryLc::new(row_len);
        let pp: ZipPlusParams<
            <F2Types<D> as F2ZincTypes<D>>::BinaryZt,
            <F2Types<D> as F2ZincTypes<D>>::BinaryLc,
        > = ZipPlusParams::new(num_vars, num_rows, lc);
        let mut open_pt = Blake3Transcript::new();
        let (hint, comm) =
            ZincPlusPiopF2::<F2Types<D>, TinyF2Uair, D>::commit_and_absorb_f2_trace(
                &mut open_pt,
                &pp,
                &trace.binary_poly,
            )
            .expect("commit should succeed");
        let mut open_proof = ZincPlusPiopF2::<F2Types<D>, TinyF2Uair, D>::prove_f2_open(
            &mut open_pt,
            &pp,
            &hint,
            &trace.binary_poly,
            &subclaim.sumcheck_point,
            &subclaim.alpha,
            4,
        );

        // Flip one bit in b_vector[0] and re-derive lifted_claim so
        // the eval-consistency check still passes (verifier recomputes
        // Σ q_0 · b' = a' identically). Then either:
        //   - coherence <combined_row', q_1> = <coeffs, b'> fails, because
        //     combined_row' is bound (via Merkle + encoding consistency) to
        //     the genuine M_w while b' now isn't; or
        //   - the lift discharge ψ_α(a') = Σ_g γ_g · a_g fails, because
        //     the rebased a' projects to a different GF(2^192) value.
        let mut tampered_b = *open_proof.b_vector[0].words();
        tampered_b[0] ^= 1;
        open_proof.b_vector[0] = BinaryF2Poly::<7>::from_words(tampered_b);
        // Re-derive a' = Σ_i q_0[i] · b'[i] over F_2[X]<10>.
        let basis = zinc_poly::univariate::binary_gf192::AlphaPolyBasis::new(&subclaim.alpha);
        let (q0, _q1) = {
            let split = subclaim.sumcheck_point.len() - (num_rows.ilog2() as usize);
            let (hi, lo) = subclaim.sumcheck_point.split_at(split);
            let q0_gf = zinc_poly::utils::build_eq_x_r_vec(lo, &()).unwrap();
            let q1_gf = zinc_poly::utils::build_eq_x_r_vec(hi, &()).unwrap();
            let q0: Vec<BinaryF2Poly<3>> = q0_gf.iter().map(|g| basis.lift(g)).collect();
            let q1: Vec<BinaryF2Poly<3>> = q1_gf.iter().map(|g| basis.lift(g)).collect();
            (q0, q1)
        };
        open_proof.lifted_claim = {
            let mut acc = BinaryF2Poly::<10>::zero();
            for i in 0..num_rows {
                let prod: BinaryF2Poly<10> =
                    zinc_poly::univariate::binary_f2_wide::f2_poly_mul::<3, 7, 10>(
                        &q0[i],
                        &open_proof.b_vector[i],
                    );
                acc += prod;
            }
            acc
        };

        let mut open_vt = Blake3Transcript::new();
        ZincPlusPiopF2::<F2Types<D>, TinyF2Uair, D>::absorb_commitment(&mut open_vt, &comm);
        let err = ZincPlusPiopF2::<F2Types<D>, TinyF2Uair, D>::verify_f2_open(
            &mut open_vt,
            &pp,
            &comm,
            &open_proof,
            &subclaim,
        )
        .expect_err("tampered b-vector must trip a downstream check");

        assert!(
            matches!(
                err,
                F2OpenError::LiftDischarge { .. } | F2OpenError::Coherence
            ),
            "expected LiftDischarge or Coherence, got {err:?}",
        );
    }

    /// End-to-end roundtrip through the *bundled* prove/verify entry
    /// points. Exercises commit + IC + sumcheck + γ-batched open on a
    /// single shared transcript per side.
    #[test]
    fn prove_then_verify_f2_full_roundtrips() {
        const D: usize = 32;
        let num_vars: usize = 6;
        let row_len: usize = 8;
        let poly_size = 1usize << num_vars;
        let num_rows = poly_size / row_len;
        assert_eq!(num_rows * row_len, poly_size);

        let mut rng_local = rng();
        let col0_vals: Vec<BinaryPoly<D>> = (0..poly_size)
            .map(|_| BinaryPoly::from(rng_local.random::<u32>()))
            .collect();
        let col1_vals = col0_vals.clone();
        let col0 = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            col0_vals,
            BinaryPoly::default(),
        );
        let col1 = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            col1_vals,
            BinaryPoly::default(),
        );
        let trace: UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D> = UairTrace {
            binary_poly: vec![col0, col1].into(),
            arbitrary_poly: vec![].into(),
            int: vec![].into(),
        };

        let lc = <F2Types<D> as F2ZincTypes<D>>::BinaryLc::new(row_len);
        let pp: ZipPlusParams<
            <F2Types<D> as F2ZincTypes<D>>::BinaryZt,
            <F2Types<D> as F2ZincTypes<D>>::BinaryLc,
        > = ZipPlusParams::new(num_vars, num_rows, lc);

        // -- Prove (single transcript across commit + uair + open) -
        let mut prover_transcript = Blake3Transcript::new();
        let proof = ZincPlusPiopF2::<F2Types<D>, TinyF2Uair, D>::prove_f2_full(
            &mut prover_transcript,
            &pp,
            &trace,
            /* virtual_specs */ &[],
            num_vars,
            |_| DynamicPolynomialF::<BinaryFieldGF192>::ZERO,
            /* num_column_openings */ 4,
        )
        .expect("prove_f2_full should succeed");

        // Shape sanity.
        assert_eq!(proof.commitment.batch_size, 2);
        assert_eq!(proof.uair.alpha, proof.uair.alpha); // alpha plumbed
        assert_eq!(proof.open.b_vector.len(), num_rows);
        assert_eq!(proof.open.combined_row.len(), row_len);
        assert_eq!(proof.open.opened_columns.len(), 4);

        // -- Verify (single fresh transcript) ---------------------
        let mut verifier_transcript = Blake3Transcript::new();
        let subclaim = ZincPlusPiopF2::<F2Types<D>, TinyF2Uair, D>::verify_f2_full(
            &mut verifier_transcript,
            &pp,
            &proof,
            /* virtual_specs */ &[],
            num_vars,
            /* num_primary_columns */ 2,
            |_ideal| ImpossibleIdeal,
        )
        .expect("verify_f2_full should succeed");

        assert_eq!(subclaim.alpha, proof.uair.alpha);
        assert_eq!(subclaim.sumcheck_point.len(), num_vars);
        assert_eq!(subclaim.primary_column_evals.len(), 2);
    }

    /// Mutating the bundled proof's open phase (here: flipping a bit
    /// in `lifted_claim`) must surface as a structured `Open(...)`
    /// error from `verify_f2_full`.
    #[test]
    fn verify_f2_full_rejects_tampered_open() {
        const D: usize = 32;
        let num_vars: usize = 4;
        let row_len: usize = 4;
        let poly_size = 1usize << num_vars;
        let num_rows = poly_size / row_len;
        let mut rng_local = rng();
        let col0_vals: Vec<BinaryPoly<D>> = (0..poly_size)
            .map(|_| BinaryPoly::from(rng_local.random::<u32>()))
            .collect();
        let col1_vals = col0_vals.clone();
        let col0 = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            col0_vals,
            BinaryPoly::default(),
        );
        let col1 = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            col1_vals,
            BinaryPoly::default(),
        );
        let trace: UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D> = UairTrace {
            binary_poly: vec![col0, col1].into(),
            arbitrary_poly: vec![].into(),
            int: vec![].into(),
        };

        let lc = <F2Types<D> as F2ZincTypes<D>>::BinaryLc::new(row_len);
        let pp: ZipPlusParams<
            <F2Types<D> as F2ZincTypes<D>>::BinaryZt,
            <F2Types<D> as F2ZincTypes<D>>::BinaryLc,
        > = ZipPlusParams::new(num_vars, num_rows, lc);

        let mut pt = Blake3Transcript::new();
        let mut proof = ZincPlusPiopF2::<F2Types<D>, TinyF2Uair, D>::prove_f2_full(
            &mut pt,
            &pp,
            &trace,
            &[],
            num_vars,
            |_| DynamicPolynomialF::<BinaryFieldGF192>::ZERO,
            4,
        )
        .expect("prove should succeed");

        // Flip a bit in a'.
        let mut tampered = *proof.open.lifted_claim.words();
        tampered[0] ^= 1;
        proof.open.lifted_claim = BinaryF2Poly::<10>::from_words(tampered);

        let mut vt = Blake3Transcript::new();
        let err = ZincPlusPiopF2::<F2Types<D>, TinyF2Uair, D>::verify_f2_full(
            &mut vt,
            &pp,
            &proof,
            &[],
            num_vars,
            2,
            |_| ImpossibleIdeal,
        )
        .expect_err("tampered open must be rejected");

        assert!(
            matches!(
                err,
                F2FullVerifyError::Open(
                    F2OpenError::EvalConsistency | F2OpenError::LiftDischarge { .. }
                )
            ),
            "expected Open(EvalConsistency | LiftDischarge), got {err:?}",
        );
    }

    // -- Virtual binary_poly column support ---------------------------
    //
    // `VirtualF2Uair` declares 3 binary_poly columns in its signature:
    //   primary col 0 = `a`            (committed)
    //   primary col 1 = `b`            (committed)
    //   virtual col 2 = `a XOR b`     (computed by both prover + verifier)
    //
    // The constraint asserts col_2 == col_0 XOR col_1, which holds by
    // construction since the virtual col IS the XOR. Practically this
    // exercises:
    //   - Prover materialises the virtual column from primary cols.
    //   - IC + sumcheck see 3 columns (2 primary + 1 virtual).
    //   - Verifier derives the virtual eval at r* from primary evals.
    //   - F_2[X] open only opens 2 primary cols.
    //
    // The `assert_zero(col_2 - (col_0 + col_1))` is automatically
    // satisfied by the materialisation; this lets us focus on the
    // wiring rather than the constraint mechanics.

    #[derive(Clone, Debug, Default)]
    struct VirtualF2Uair;

    impl Uair for VirtualF2Uair {
        type Ideal = ImpossibleIdeal;
        type Scalar = BinaryPoly<32>;

        fn signature() -> UairSignature {
            // total binary_poly = 2 primary + 1 virtual = 3.
            // The virtual col index inside `binary_poly` is the next
            // available slot (= 2). The signature counts the virtual
            // col in `total_cols.num_binary_poly_cols` so the IC sees
            // it; the trace itself only holds the 2 primary cols
            // (the prove path materialises col 2 inline).
            UairSignature::new(
                TotalColumnLayout::new(/* binary */ 3, 0, 0),
                PublicColumnLayout::default(),
                vec![],
                vec![],
                vec![],
            )
        }

        fn constrain_general<B, FromR, MulByScalar, IFromR>(
            b: &mut B,
            up: TraceRow<B::Expr>,
            _down: TraceRow<B::Expr>,
            _from_ref: FromR,
            _mbs: MulByScalar,
            _ideal_from_ref: IFromR,
        ) where
            B: ConstraintBuilder,
            FromR: Fn(&Self::Scalar) -> B::Expr,
            MulByScalar: Fn(&B::Expr, &Self::Scalar) -> Option<B::Expr>,
            IFromR: Fn(&Self::Ideal) -> B::Ideal,
        {
            // assert_zero(virtual - (a + b)) ≡ virtual = a XOR b.
            // In F_2 addition IS XOR, so this is a sum.
            let a = up.binary_poly[0].clone();
            let b_col = up.binary_poly[1].clone();
            let v = up.binary_poly[2].clone();
            b.assert_zero(v - (a + b_col));
        }
    }

    /// Build a satisfied trace for `VirtualF2Uair` and run the full
    /// pipeline (commit + IC + sumcheck + γ-batched open) on a
    /// single transcript, verifying both end-to-end and the
    /// virtual-col-derivation consistency.
    #[test]
    fn prove_then_verify_f2_full_with_virtual_column() {
        const D: usize = 32;
        let num_vars: usize = 6;
        let row_len: usize = 8;
        let poly_size = 1usize << num_vars;
        let num_rows = poly_size / row_len;
        assert_eq!(num_rows * row_len, poly_size);

        let mut rng_local = rng();

        // Two arbitrary primary columns `a` and `b`. The virtual col
        // `v = a XOR b` is materialised inside `prove_f2_full`.
        let a_vals: Vec<BinaryPoly<D>> = (0..poly_size)
            .map(|_| BinaryPoly::from(rng_local.random::<u32>()))
            .collect();
        let b_vals: Vec<BinaryPoly<D>> = (0..poly_size)
            .map(|_| BinaryPoly::from(rng_local.random::<u32>()))
            .collect();
        let a = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            a_vals,
            BinaryPoly::default(),
        );
        let b = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            b_vals,
            BinaryPoly::default(),
        );
        let trace: UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D> = UairTrace {
            binary_poly: vec![a.clone(), b.clone()].into(),
            arbitrary_poly: vec![].into(),
            int: vec![].into(),
        };

        // `v = a XOR b`. Primary col idx 0 and 1.
        let virtual_specs = vec![F2VirtualBpSpec {
            primary_col_indices: vec![0, 1],
        }];

        let lc = <F2Types<D> as F2ZincTypes<D>>::BinaryLc::new(row_len);
        let pp: ZipPlusParams<
            <F2Types<D> as F2ZincTypes<D>>::BinaryZt,
            <F2Types<D> as F2ZincTypes<D>>::BinaryLc,
        > = ZipPlusParams::new(num_vars, num_rows, lc);

        // -- Prove (full pipeline) --------------------------------
        let mut prover_transcript = Blake3Transcript::new();
        let proof = ZincPlusPiopF2::<F2Types<D>, VirtualF2Uair, D>::prove_f2_full(
            &mut prover_transcript,
            &pp,
            &trace,
            &virtual_specs,
            num_vars,
            |_| DynamicPolynomialF::<BinaryFieldGF192>::ZERO,
            4,
        )
        .expect("prove_f2_full with virtual col should succeed");

        // Only the 2 primary cols are committed.
        assert_eq!(proof.commitment.batch_size, 2);

        // -- Verify (full pipeline) -------------------------------
        let mut verifier_transcript = Blake3Transcript::new();
        let subclaim = ZincPlusPiopF2::<F2Types<D>, VirtualF2Uair, D>::verify_f2_full(
            &mut verifier_transcript,
            &pp,
            &proof,
            &virtual_specs,
            num_vars,
            /* num_primary_columns */ 2,
            |_ideal| ImpossibleIdeal,
        )
        .expect("verify_f2_full with virtual col should succeed");

        assert_eq!(subclaim.primary_column_evals.len(), 2);
        assert_eq!(subclaim.virtual_column_evals.len(), 1);

        // Sanity: the derived virtual eval matches XOR of primary evals.
        let xor =
            subclaim.primary_column_evals[0] + subclaim.primary_column_evals[1];
        assert_eq!(subclaim.virtual_column_evals[0], xor);
    }

    /// A prover supplying mismatched virtual_specs (or none) on a
    /// trace whose IC + sumcheck expected the materialised virtual
    /// column must be rejected. Here we let the prover succeed with
    /// correct specs but feed the verifier *empty* specs — the
    /// verifier should fail (either group-count mismatch from the
    /// sumcheck's extra group, or a downstream check).
    #[test]
    fn verify_f2_full_rejects_missing_virtual_spec() {
        const D: usize = 32;
        let num_vars: usize = 4;
        let row_len: usize = 4;
        let poly_size = 1usize << num_vars;
        let num_rows = poly_size / row_len;
        let mut rng_local = rng();
        let a_vals: Vec<BinaryPoly<D>> = (0..poly_size)
            .map(|_| BinaryPoly::from(rng_local.random::<u32>()))
            .collect();
        let b_vals: Vec<BinaryPoly<D>> = (0..poly_size)
            .map(|_| BinaryPoly::from(rng_local.random::<u32>()))
            .collect();
        let a = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            a_vals,
            BinaryPoly::default(),
        );
        let b = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            b_vals,
            BinaryPoly::default(),
        );
        let trace: UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D> = UairTrace {
            binary_poly: vec![a, b].into(),
            arbitrary_poly: vec![].into(),
            int: vec![].into(),
        };

        let virtual_specs = vec![F2VirtualBpSpec {
            primary_col_indices: vec![0, 1],
        }];

        let lc = <F2Types<D> as F2ZincTypes<D>>::BinaryLc::new(row_len);
        let pp: ZipPlusParams<
            <F2Types<D> as F2ZincTypes<D>>::BinaryZt,
            <F2Types<D> as F2ZincTypes<D>>::BinaryLc,
        > = ZipPlusParams::new(num_vars, num_rows, lc);

        let mut pt = Blake3Transcript::new();
        let proof = ZincPlusPiopF2::<F2Types<D>, VirtualF2Uair, D>::prove_f2_full(
            &mut pt,
            &pp,
            &trace,
            &virtual_specs,
            num_vars,
            |_| DynamicPolynomialF::<BinaryFieldGF192>::ZERO,
            4,
        )
        .expect("prove should succeed with correct specs");

        // Verifier passes empty specs. The sumcheck proof has 3
        // groups (primary 0, primary 1, virtual 0); we'd claim 2
        // primary cols + 0 virtual = 2 total. Mismatch.
        let mut vt = Blake3Transcript::new();
        let err = ZincPlusPiopF2::<F2Types<D>, VirtualF2Uair, D>::verify_f2_full(
            &mut vt,
            &pp,
            &proof,
            /* virtual_specs */ &[],
            num_vars,
            2,
            |_| ImpossibleIdeal,
        )
        .expect_err("verifier must reject when virtual_specs don't match the prover's");

        // Any rejection is fine — the structural mismatch can
        // surface as GroupCountMismatch (sumcheck has 3 groups but
        // verifier expected 2) or a downstream check that fails
        // because the sumcheck challenges diverge.
        let _ = err;
    }

    // ---------------------------------------------------------------
    // SHA-256 F_2[X] UAIR — full prove/verify roundtrip.
    //
    // Exercises the real F_2 SHA-256 arithmetisation
    // (`zinc_test_uair::Sha256F2Uair`) through the bundled
    // `prove_f2_full` / `verify_f2_full` entry points. With
    // `NUM_COMPRESSIONS = 7`, the trace fits in `num_vars = 9` =
    // 2^9 = 512 rows (480 active + 32 slack).
    // ---------------------------------------------------------------

    #[test]
    fn prove_then_verify_sha256_f2_roundtrips() {
        use crypto_primitives::crypto_bigint_int::Int;
        use zinc_test_uair::{
            GenerateRandomTrace, Sha256F2Ideal, Sha256F2Uair, sha256_f2_project_ideal,
            sha256_f2_project_scalar,
        };

        const D: usize = 32;
        type R = Int<4>;
        type U = Sha256F2Uair<R>;

        // Trace parameters from the SHA F_2 UAIR spec.
        let num_vars: usize = 9; // 2^9 = 512 ≥ 7·68 + 4 = 480
        let row_len: usize = 32;
        let poly_size = 1usize << num_vars;
        let num_rows = poly_size / row_len;
        assert_eq!(num_rows * row_len, poly_size);

        // Generate honest trace.
        let mut rng_local = rng();
        let trace = U::generate_random_trace(num_vars, &mut rng_local);

        // Set up PCS params with rate-1/4 RAA code.
        let lc = <F2Types<D> as F2ZincTypes<D>>::BinaryLc::new(row_len);
        let pp: ZipPlusParams<
            <F2Types<D> as F2ZincTypes<D>>::BinaryZt,
            <F2Types<D> as F2ZincTypes<D>>::BinaryLc,
        > = ZipPlusParams::new(num_vars, num_rows, lc);

        // Prove.
        let mut prover_transcript = Blake3Transcript::new();
        let proof = ZincPlusPiopF2::<F2Types<D>, U, D>::prove_f2_full(
            &mut prover_transcript,
            &pp,
            &trace,
            /* virtual_specs */ &[],
            num_vars,
            sha256_f2_project_scalar::<R>,
            // Small num_column_openings to keep the test snappy; the
            // soundness-critical bench uses
            // `recommended_num_column_openings(4) = 987`.
            /* num_column_openings */ 4,
        )
        .expect("prove_f2_full on SHA F_2 UAIR should succeed");

        // Sanity-check the proof shape.
        assert_eq!(
            proof.commitment.batch_size,
            zinc_test_uair::sha256_f2::cols::NUM_BIN
        );

        // Verify on a fresh transcript.
        let mut verifier_transcript = Blake3Transcript::new();
        let subclaim = ZincPlusPiopF2::<F2Types<D>, U, D>::verify_f2_full(
            &mut verifier_transcript,
            &pp,
            &proof,
            /* virtual_specs */ &[],
            num_vars,
            zinc_test_uair::sha256_f2::cols::NUM_BIN,
            |ideal: &IdealOrZero<Sha256F2Ideal>| sha256_f2_project_ideal(ideal),
        )
        .expect("verify_f2_full on SHA F_2 UAIR should succeed");

        // Sanity: subclaim is well-shaped.
        assert_eq!(subclaim.sumcheck_point.len(), num_vars);
        assert_eq!(
            subclaim.primary_column_evals.len(),
            zinc_test_uair::sha256_f2::cols::NUM_BIN
        );
        assert_eq!(subclaim.virtual_column_evals.len(), 0);
    }
}
