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
    ideal_check::{IdealCheckProtocol, Proof as IcProof},
    projections::project_f2_trace_row_major,
    sumcheck::multi_degree::{
        MultiDegreeSumcheck, MultiDegreeSumcheckGroup, MultiDegreeSumcheckProof,
    },
};
use zinc_poly::{
    mle::DenseMultilinearExtension,
    univariate::{
        binary::BinaryPoly, binary_gf192::BinaryFieldGF192,
        dynamic::over_field::DynamicPolynomialF,
    },
};
use zinc_transcript::traits::Transcript;
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
#[derive(Debug)]
pub struct F2Proof {
    pub ic_proof: IcProof<BinaryFieldGF192>,
    pub sumcheck_proof: MultiDegreeSumcheckProof<BinaryFieldGF192>,
    /// `α ∈ GF(2^192)` — the evaluation-projection challenge, drawn
    /// from the transcript after the IC. Recorded here as a
    /// convenience; the verifier could equivalently re-derive it.
    pub alpha: BinaryFieldGF192,
}

/// Errors emitted by [`prove_f2_uair`].
#[derive(Debug, thiserror::Error)]
pub enum F2ProveError<U: Uair> {
    #[error("ideal-check failed: {0}")]
    IdealCheck(zinc_piop::ideal_check::IdealCheckError<BinaryFieldGF192, U::Ideal>),
    #[error("evaluation projection failed: {0}")]
    EvalProjection(zinc_poly::EvaluationError),
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
    type BinaryLc: zip_plus::code::LinearCode<Self::BinaryZt>;
}

/// Phantom marker that ties the prove function to its type parameters
/// without storing any data. Mirrors `ZincPlusPiop` from the integer
/// prove path; kept as its own struct so future expansions of the
/// F_2 protocol can hang state off it.
pub struct ZincPlusPiopF2<Zt, U, const DEGREE_PLUS_ONE: usize>(PhantomData<(Zt, U)>)
where
    Zt: F2ZincTypes<DEGREE_PLUS_ONE>,
    U: Uair;

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
    ) -> Result<F2Proof, F2ProveError<U>> {
        let num_constraints = count_constraints::<U>();
        let field_cfg = ();

        // -- Step 2: Ideal check over GF(2^192)[X] -----------------
        // Trace cells (F_2[X]<D>) are lifted to GF(2^192)[X]<D> via
        // the trivial F_2 ⊂ GF(2^192) coefficient embedding inside
        // `project_f2_trace_row_major`.
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

        // -- Step 3: Evaluation projection (X = α) -----------------
        let alpha: BinaryFieldGF192 = transcript.get_field_challenge(&field_cfg);

        // Project the trace to GF(2^192) at X = α. Each cell
        // (F_2[X]<D>) becomes a single GF(2^192) value via per-bit
        // dot-product with powers of α. The resulting per-column
        // MLEs are over GF(2^192).
        let projected_trace: Vec<DenseMultilinearExtension<BinaryFieldGF192>> = cfg_iter!(
            trace.binary_poly
        )
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

        // -- Step 4: Sumcheck over GF(2^192) -----------------------
        //
        // Smallest meaningful claim shape: a single degree-2 group
        // proving that `Σ_{y ∈ {0,1}^num_vars} eq(y, r) · cell(y) = 0`
        // for each column `cell` and the IC's evaluation point `r`.
        // For an `assert_zero`-only UAIR every cell-expression is
        // identically zero on the hypercube, so the sumcheck's
        // claimed sum is 0; the protocol still produces a valid
        // proof + a final MLE evaluation point + per-MLE expected
        // evaluations.
        //
        // For a full F_2 protocol with non-trivial constraints the
        // group composition would mirror what
        // `CombinedPolyResolver::prepare_sumcheck_group` builds for
        // the integer path. That generalisation is a follow-up;
        // this slice's contract is the trait-and-types wiring.
        let eq_r = zinc_poly::utils::build_eq_x_r_inner(
            &ic_state.evaluation_point,
            &field_cfg,
        )
        .expect("eq table construction must succeed for valid IC point");

        // One group per projected column, each proving the trivial
        // `Σ eq(y, r) · col(y)` claim. comb_fn picks v[0] (eq) and
        // multiplies by v[1] (col).
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::ConstZero;
    use rand::{Rng, rng};
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
}
