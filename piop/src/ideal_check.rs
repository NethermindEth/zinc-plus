//! Ideal-check subprotocol.
mod batched_ideal_check;
mod combined_poly_builder;
mod structs;

pub use structs::*;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::projections::{ColumnMajorTrace, RowMajorTrace, ScalarMap};
use batched_ideal_check::*;
use crypto_primitives::PrimeField;
use num_traits::ConstZero;
use thiserror::Error;
use zinc_poly::{
    EvaluationError,
    univariate::dynamic::over_field::DynamicPolynomialF,
    utils::{ArithErrors as PolyArithErrors, build_eq_x_r_vec},
};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_uair::{
    Uair,
    degree_counter::linear_constraint_mask,
    ideal::{Ideal, IdealCheck},
    ideal_collector::{IdealOrZero, collect_ideals},
};
use zinc_utils::{cfg_into_iter, inner_transparent_field::InnerTransparentField};

/// Ideal-check subprotocol.
pub trait IdealCheckProtocol: Uair {
    /// Prover for linear-only UAIRs using MLE-first evaluation.
    ///
    /// Uses column-indexed trace for efficient MLE evaluation:
    /// evaluates trace columns at the challenge point first,
    /// then applies constraints to the evaluated values.
    ///
    /// # Parameters
    /// - `transcript`: the Fiat-Shamir transcript.
    /// - `trace_matrix`: input trace for the UAIR `U` projected to
    ///   `DynamicPolynomialF<F>`, column-indexed: `trace_matrix[col][row]`.
    /// - `projected_scalars`: UAIR scalars projected to
    ///   `DynamicPolynomialF<F>`.
    /// - `num_constraints`: number of constraints this UAIR encodes.
    /// - `num_vars`: number of variables in trace MLEs.
    /// - `field_cfg`: random field configuration sampled on the previous steps
    ///   of the overall protocol.
    #[allow(clippy::type_complexity)]
    fn prove_linear<F>(
        transcript: &mut impl Transcript,
        trace_matrix: &ColumnMajorTrace<F>,
        projected_scalars: &ScalarMap<Self::Scalar, DynamicPolynomialF<F>>,
        num_constraints: usize,
        num_vars: usize,
        field_cfg: &F::Config,
    ) -> Result<(Proof<F>, ProverState<F>), IdealCheckError<F, Self::Ideal>>
    where
        F: InnerTransparentField,
        F::Inner: ConstTranscribable,
        F::Modulus: ConstTranscribable;

    /// Prover for any UAIR using combined polynomial construction.
    ///
    /// Uses row-indexed (transposed) trace for efficient row-by-row
    /// combined polynomial construction.
    ///
    /// # Parameters
    /// - `transcript`: the Fiat-Shamir transcript.
    /// - `trace_matrix`: input trace for the UAIR `U` projected to
    ///   `DynamicPolynomialF<F>`, row-indexed: `trace_matrix[row][col]`.
    /// - `projected_scalars`: UAIR scalars projected to
    ///   `DynamicPolynomialF<F>`.
    /// - `num_constraints`: number of constraints this UAIR encodes.
    /// - `num_vars`: number of variables in trace MLEs.
    /// - `field_cfg`: random field configuration sampled on the previous steps
    ///   of the overall protocol.
    #[allow(clippy::type_complexity)]
    fn prove_combined<F>(
        transcript: &mut impl Transcript,
        trace_matrix: &RowMajorTrace<F>,
        projected_scalars: &ScalarMap<Self::Scalar, DynamicPolynomialF<F>>,
        num_constraints: usize,
        num_vars: usize,
        field_cfg: &F::Config,
    ) -> Result<(Proof<F>, ProverState<F>), IdealCheckError<F, Self::Ideal>>
    where
        F: InnerTransparentField,
        F::Inner: ConstTranscribable,
        F::Modulus: ConstTranscribable;

    /// Prover for mixed-degree UAIRs: routes each constraint through the
    /// MLE-first lane (degree ≤ 1, non-zero-ideal) or the combined-poly
    /// lane (degree > 1, non-zero-ideal). Zero-ideal constraints are
    /// substituted with `ZERO` as in the other paths.
    ///
    /// Both lanes evaluate against the same Fiat-Shamir-sampled IC point,
    /// so the merged per-constraint values are mathematically identical to
    /// what the all-Combined path would produce — the verifier sees the
    /// same `Proof` shape and runs unchanged.
    ///
    /// Takes both layouts because the MLE-first lane requires column-major
    /// while the combined-poly lane requires row-major. Step 1 of the
    /// protocol projects the trace twice for this path.
    ///
    /// # Parameters
    /// - `transcript`: the Fiat-Shamir transcript.
    /// - `row_major_trace` / `column_major_trace`: same projected trace in
    ///   both layouts.
    /// - `projected_scalars`: UAIR scalars projected to
    ///   `DynamicPolynomialF<F>`.
    /// - `num_constraints`: number of constraints this UAIR encodes.
    /// - `num_vars`: number of variables in trace MLEs.
    /// - `field_cfg`: random field configuration sampled on the previous
    ///   steps of the overall protocol.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn prove_hybrid<F>(
        transcript: &mut impl Transcript,
        row_major_trace: &RowMajorTrace<F>,
        column_major_trace: &ColumnMajorTrace<F>,
        projected_scalars: &ScalarMap<Self::Scalar, DynamicPolynomialF<F>>,
        num_constraints: usize,
        num_vars: usize,
        field_cfg: &F::Config,
    ) -> Result<(Proof<F>, ProverState<F>), IdealCheckError<F, Self::Ideal>>
    where
        F: InnerTransparentField,
        F::Inner: ConstTranscribable,
        F::Modulus: ConstTranscribable;

    /// The verifier part of the ideal-check subprotocol.
    ///
    /// The verifier samples a random field element
    /// the same way the prover sampled a random field
    /// element for projecting coefficients but disregards it
    /// as the verifier does not need to project anything.
    /// Then it computes the ideals encoded by the UAIR `U`,
    /// samples a random evaluation point and receives
    /// the evaluations of the combined polynomials sent by the prover
    /// and checks they belong to the corresponding ideals defined
    /// by the UAIR `U`.
    ///
    /// # Parameters
    /// - `transcript`: the Fiat-Shamir transcript.
    /// - `proof`: a purported proof produced by the prover.
    /// - `num_constraints`: the number of constraints the UAIR `U` encodes.
    /// - `num_vars`: the number of variables the trace row MLEs have.
    /// - `ideal_over_f_from_ref`: since the UAIR `U` is not aware of the field
    ///   the ideal check is operating on it defines ideals over the ring
    ///   `IcTypes::Witness`. `ideal_over_f_from_ref` allows to convert the
    ///   ideals over `IcTypes::Witness` into ideals over the field
    ///   `IcTypes::F`. Think of this as a projection for ideals.
    /// - `field_cfg`: random field configuration sampled on the previous steps
    ///   of the overall protocol.
    #[allow(clippy::type_complexity)]
    fn verify_as_subprotocol<F, IdealOverF, IdealOverFFromRef>(
        transcript: &mut impl Transcript,
        proof: Proof<F>,
        num_constraints: usize,
        num_vars: usize,
        ideal_over_f_from_ref: IdealOverFFromRef,
        field_cfg: &F::Config,
    ) -> Result<VerifierSubclaim<F>, IdealCheckError<F, IdealOverF>>
    where
        F: InnerTransparentField,
        F::Inner: ConstTranscribable,
        F::Modulus: ConstTranscribable,
        IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<F>>,
        IdealOverFFromRef: Fn(&IdealOrZero<Self::Ideal>) -> IdealOverF;
}

impl<U> IdealCheckProtocol for U
where
    U: Uair,
{
    #[allow(clippy::type_complexity)]
    fn prove_linear<F>(
        transcript: &mut impl Transcript,
        trace_matrix: &ColumnMajorTrace<F>,
        projected_scalars: &ScalarMap<U::Scalar, DynamicPolynomialF<F>>,
        num_constraints: usize,
        num_vars: usize,
        field_cfg: &F::Config,
    ) -> Result<(Proof<F>, ProverState<F>), IdealCheckError<F, U::Ideal>>
    where
        F: InnerTransparentField,
        F::Inner: ConstTranscribable,
        F::Modulus: ConstTranscribable,
    {
        // Zero-ideal (`assert_zero`) constraints are identically zero on the
        // hypercube for an honest prover, and the MLE-first evaluation of a
        // nonlinear constraint produces a meaningless value. Both cases are
        // handled inside `CombinedPolyRowBuilder::assert_zero`, which writes
        // `ZERO` into the corresponding slot regardless of the input
        // expression — so no post-hoc overwrite is needed here.
        let evaluation_point = transcript.get_field_challenges(num_vars, field_cfg);

        // Evaluate combined polynomials using MLE-first approach:
        // evaluate trace columns at the point, then apply constraints.
        let combined_mle_values = combined_poly_builder::evaluate_combined_polynomials::<_, U>(
            trace_matrix,
            projected_scalars,
            num_constraints,
            &evaluation_point,
            field_cfg,
        )?;

        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];

        combined_mle_values.iter().for_each(|combined_mle_value| {
            transcript
                .absorb_random_field_slice(&combined_mle_value.coeffs, &mut transcription_buf);
        });

        Ok((
            Proof {
                combined_mle_values,
            },
            ProverState { evaluation_point },
        ))
    }

    #[allow(clippy::type_complexity)]
    fn prove_combined<F>(
        transcript: &mut impl Transcript,
        trace_matrix: &RowMajorTrace<F>,
        projected_scalars: &ScalarMap<U::Scalar, DynamicPolynomialF<F>>,
        num_constraints: usize,
        num_vars: usize,
        field_cfg: &F::Config,
    ) -> Result<(Proof<F>, ProverState<F>), IdealCheckError<F, U::Ideal>>
    where
        F: InnerTransparentField,
        F::Inner: ConstTranscribable,
        F::Modulus: ConstTranscribable,
    {
        // Build per-coefficient MLEs for every constraint, including
        // zero-ideal ones. The earlier short-circuit that replaced the
        // F[X] expression with `ZERO` for zero-ideal constraints was
        // only sound when the per-row F[X] expression was identically
        // the zero polynomial — see `CombinedPolyRowBuilder::assert_zero`
        // for the rationale.
        let is_zero_ideal: Vec<bool> = vec![false; num_constraints];

        let evaluation_point = transcript.get_field_challenges(num_vars, field_cfg);

        // Build combined polynomial MLEs row-by-row and evaluate them.
        let combined_mles = combined_poly_builder::compute_combined_polynomials::<_, U>(
            trace_matrix,
            projected_scalars,
            num_constraints,
            field_cfg,
            &is_zero_ideal,
        );

        let eq_table = build_eq_x_r_vec(&evaluation_point, field_cfg)?;

        // Evaluate coefficient MLEs at the evaluation point.
        let combined_mle_values: Vec<DynamicPolynomialF<F>> = cfg_into_iter!(combined_mles)
            .enumerate()
            .map(|(i, coeff_mles)| {
                // Skip zero-ideal constraints: their combined polynomial
                // is zero for an honest prover.
                if is_zero_ideal[i] {
                    return DynamicPolynomialF::ZERO;
                }
                let coeffs = coeff_mles
                    .into_iter()
                    .map(|coeff_mle| {
                        zinc_poly::utils::mle_eval_with_eq_table(
                            &coeff_mle.evaluations,
                            &eq_table,
                            field_cfg,
                        )
                    })
                    .collect::<Vec<_>>();
                DynamicPolynomialF::new_trimmed(coeffs)
            })
            .collect::<Vec<_>>();

        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];

        combined_mle_values.iter().for_each(|combined_mle_value| {
            transcript
                .absorb_random_field_slice(&combined_mle_value.coeffs, &mut transcription_buf);
        });

        Ok((
            Proof {
                combined_mle_values,
            },
            ProverState { evaluation_point },
        ))
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn prove_hybrid<F>(
        transcript: &mut impl Transcript,
        row_major_trace: &RowMajorTrace<F>,
        column_major_trace: &ColumnMajorTrace<F>,
        projected_scalars: &ScalarMap<U::Scalar, DynamicPolynomialF<F>>,
        num_constraints: usize,
        num_vars: usize,
        field_cfg: &F::Config,
    ) -> Result<(Proof<F>, ProverState<F>), IdealCheckError<F, U::Ideal>>
    where
        F: InnerTransparentField,
        F::Inner: ConstTranscribable,
        F::Modulus: ConstTranscribable,
    {
        // Per-constraint classification: linear vs non-linear (by degree),
        // and zero-ideal vs non-zero-ideal. The three buckets get filled
        // from different sources — see comment below.
        let linear_mask = linear_constraint_mask::<U>();
        let is_zero_ideal: Vec<bool> = collect_ideals::<U>(num_constraints)
            .ideals
            .iter()
            .map(|i| i.is_zero_ideal())
            .collect();
        debug_assert_eq!(linear_mask.len(), num_constraints);
        debug_assert_eq!(is_zero_ideal.len(), num_constraints);

        let evaluation_point = transcript.get_field_challenges(num_vars, field_cfg);

        // MLE-first lane: produces correct values for linear non-zero-ideal
        // slots (and garbage for the rest, which we discard).
        let linear_values = combined_poly_builder::evaluate_combined_polynomials_unchecked::<_, U>(
            column_major_trace,
            projected_scalars,
            num_constraints,
            &evaluation_point,
            field_cfg,
        )?;

        // Combined-poly lane: build coefficient MLEs only for the
        // non-linear, non-zero-ideal slots (skip linear and zero-ideal —
        // those come from the other sources). Reuses the existing
        // `skip_constraints` plumbing.
        let skip_combined: Vec<bool> = linear_mask
            .iter()
            .zip(is_zero_ideal.iter())
            .map(|(&lin, &zero_ideal)| lin || zero_ideal)
            .collect();
        let combined_mles = combined_poly_builder::compute_combined_polynomials::<_, U>(
            row_major_trace,
            projected_scalars,
            num_constraints,
            field_cfg,
            &skip_combined,
        );

        let eq_table = build_eq_x_r_vec(&evaluation_point, field_cfg)?;

        // Merge: zero-ideal → ZERO; linear → MLE-first value;
        // non-linear → combined-poly value evaluated at the IC point.
        let combined_mle_values: Vec<DynamicPolynomialF<F>> = cfg_into_iter!(0..num_constraints)
            .map(|i| {
                if is_zero_ideal[i] {
                    DynamicPolynomialF::ZERO
                } else if linear_mask[i] {
                    linear_values[i].clone()
                } else {
                    let coeffs = combined_mles[i]
                        .iter()
                        .map(|coeff_mle| {
                            zinc_poly::utils::mle_eval_with_eq_table(
                                &coeff_mle.evaluations,
                                &eq_table,
                                field_cfg,
                            )
                        })
                        .collect::<Vec<_>>();
                    DynamicPolynomialF::new_trimmed(coeffs)
                }
            })
            .collect::<Vec<_>>();

        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];
        combined_mle_values.iter().for_each(|combined_mle_value| {
            transcript
                .absorb_random_field_slice(&combined_mle_value.coeffs, &mut transcription_buf);
        });

        Ok((
            Proof {
                combined_mle_values,
            },
            ProverState { evaluation_point },
        ))
    }

    fn verify_as_subprotocol<F, IdealOverF, IdealOverFFromRef>(
        transcript: &mut impl Transcript,
        proof: Proof<F>,
        num_constraints: usize,
        num_vars: usize,
        ideal_over_f_from_ref: IdealOverFFromRef,
        field_cfg: &F::Config,
    ) -> Result<VerifierSubclaim<F>, IdealCheckError<F, IdealOverF>>
    where
        F: InnerTransparentField,
        F::Inner: ConstTranscribable,
        F::Modulus: ConstTranscribable,
        IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<F>>,
        IdealOverFFromRef: Fn(&IdealOrZero<U::Ideal>) -> IdealOverF,
    {
        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];

        let combined_mle_values = proof.combined_mle_values;

        let evaluation_point = transcript.get_field_challenges(num_vars, field_cfg);

        for mle_value in &combined_mle_values {
            transcript.absorb_random_field_slice(&mle_value.coeffs, &mut transcription_buf);
        }

        let ideal_collector = collect_ideals::<U>(num_constraints);

        // Only check non-trivial ideals. For assert_zero constraints
        // the ideal is the zero ideal and the combined polynomial
        // value is zero by construction; the sumcheck that follows
        // verifies consistency of the claimed evaluations with the
        // actual trace.
        let (non_trivial_ideals, non_trivial_values): (Vec<_>, Vec<_>) = ideal_collector
            .ideals
            .iter()
            .zip(combined_mle_values.iter())
            .filter(|(ideal, _)| !ideal.is_zero_ideal())
            .map(|(ideal, value)| (ideal_over_f_from_ref(ideal), value.clone()))
            .unzip();

        batched_ideal_check(&non_trivial_ideals, &non_trivial_values)?;

        Ok(VerifierSubclaim {
            evaluation_point,
            values: combined_mle_values,
        })
    }
}

#[derive(Clone, Debug, Error)]
pub enum IdealCheckError<F: PrimeField, I> {
    #[error("ideal check prover failed to evaluate an mle: {0}")]
    MleEvaluationError(#[from] EvaluationError),
    #[error("mle evaluation ideal check failure: {0}")]
    IdealCollectorError(#[from] BatchedIdealCheckError<DynamicPolynomialF<F>, I>),
    #[error("`eq` polynomial construction failure: {0}")]
    EqPolyConstructionError(#[from] PolyArithErrors),
}

#[cfg(test)]
mod tests {
    use crypto_primitives::{crypto_bigint_int::Int, crypto_bigint_monty::MontyField};

    use rand::rng;
    use zinc_poly::univariate::{dense::DensePolynomial, dynamic::over_field::DynamicPolynomialF};
    use zinc_test_uair::{
        GenerateRandomTrace, TestUairNoMultiplication, TestUairSimpleMultiplication,
    };
    use zinc_transcript::Blake3Transcript;
    use zinc_uair::{
        constraint_counter::count_constraints,
        ideal::{DegreeOneIdeal, Ideal, IdealCheck},
    };

    use crate::test_utils::{
        LIMBS, run_ideal_check_prover_combined, run_ideal_check_prover_linear, test_config,
    };

    use super::*;

    // TODO(Ilia): These tests are absolute joke.
    //             Once we have time we need to create a comprehensive test suite
    //             akin to the one we have for the PCS or the sumcheck.

    fn test_successful_verification_linear<
        U,
        IdealOverF,
        IdealOverFFromRef,
        const DEGREE_PLUS_ONE: usize,
    >(
        num_vars: usize,
        ideal_over_f_from_ref: IdealOverFFromRef,
    ) where
        U: Uair<Scalar = DensePolynomial<Int<5>, DEGREE_PLUS_ONE>>
            + GenerateRandomTrace<DEGREE_PLUS_ONE, PolyCoeff = Int<5>, Int = Int<5>>
            + IdealCheckProtocol,
        IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<MontyField<LIMBS>>>,
        IdealOverFFromRef: Fn(&IdealOrZero<U::Ideal>) -> IdealOverF,
    {
        let mut rng = rng();
        let transcript = Blake3Transcript::new();

        let (proof, prover_state, ..) = run_ideal_check_prover_linear::<U, DEGREE_PLUS_ONE>(
            num_vars,
            &U::generate_random_trace(num_vars, &mut rng),
            &mut transcript.clone(),
        );

        let num_constraints = count_constraints::<U>();

        let verifier_result = U::verify_as_subprotocol(
            &mut transcript.clone(),
            proof,
            num_constraints,
            num_vars,
            ideal_over_f_from_ref,
            &test_config(),
        )
        .expect("Verification failed");

        assert_eq!(
            prover_state.evaluation_point,
            verifier_result.evaluation_point
        );
    }

    fn test_successful_verification_combined<
        U,
        IdealOverF,
        IdealOverFFromRef,
        const DEGREE_PLUS_ONE: usize,
    >(
        num_vars: usize,
        ideal_over_f_from_ref: IdealOverFFromRef,
    ) where
        U: Uair<Scalar = DensePolynomial<Int<5>, DEGREE_PLUS_ONE>>
            + GenerateRandomTrace<DEGREE_PLUS_ONE, PolyCoeff = Int<5>, Int = Int<5>>
            + IdealCheckProtocol,
        IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<MontyField<LIMBS>>>,
        IdealOverFFromRef: Fn(&IdealOrZero<U::Ideal>) -> IdealOverF,
    {
        let mut rng = rng();
        let transcript = Blake3Transcript::new();

        let (proof, prover_state, ..) = run_ideal_check_prover_combined::<U, DEGREE_PLUS_ONE>(
            num_vars,
            &U::generate_random_trace(num_vars, &mut rng),
            &mut transcript.clone(),
        );

        let num_constraints = count_constraints::<U>();

        let verifier_result = U::verify_as_subprotocol(
            &mut transcript.clone(),
            proof,
            num_constraints,
            num_vars,
            ideal_over_f_from_ref,
            &test_config(),
        )
        .expect("Verification failed");

        assert_eq!(
            prover_state.evaluation_point,
            verifier_result.evaluation_point
        );
    }

    #[test]
    fn test_successful_verification() {
        let field_cfg = test_config();

        let num_vars = 2;

        // Linear UAIR - test both approaches
        test_successful_verification_linear::<TestUairNoMultiplication<Int<5>>, _, _, 32>(
            num_vars,
            |ideal_over_ring| ideal_over_ring.map(|i| DegreeOneIdeal::from_with_cfg(i, &field_cfg)),
        );
        test_successful_verification_combined::<TestUairNoMultiplication<Int<5>>, _, _, 32>(
            num_vars,
            |ideal_over_ring| ideal_over_ring.map(|i| DegreeOneIdeal::from_with_cfg(i, &field_cfg)),
        );

        // Non-linear UAIR - only combined approach works
        test_successful_verification_combined::<TestUairSimpleMultiplication<Int<5>>, _, _, 32>(
            num_vars,
            |_ideal_over_ring| IdealOrZero::<DegreeOneIdeal<_>>::zero(),
        );
    }

    /// Compile-time integration check: the `IdealCheckProtocol`
    /// blanket impl resolves against `F = BinaryFieldGF192`, i.e. the
    /// synthetic `PrimeField` + `InnerTransparentField` impls on the
    /// `GF(2^192)` field satisfy every bound the three IC entry
    /// points demand:
    /// - `F: InnerTransparentField` (impl on `BinaryFieldGF192`).
    /// - `F::Inner: ConstTranscribable` (via `Uint<3>` from
    ///   `crypto_primitives::crypto_bigint_uint`).
    /// - `F::Modulus: ConstTranscribable` (via the `()` impl added
    ///   to `zinc_transcript`).
    ///
    /// We take the method function pointers without invoking them.
    /// If any of the trait bounds fail to resolve, this stops
    /// compiling — which is the real assertion. Building a
    /// `GF(2^192)`-projected trace from an `F_2[X]`-typed UAIR is
    /// the next plumbing step in the F_2 prove-path plan.
    #[test]
    fn ic_compiles_for_gf192() {
        use zinc_poly::univariate::binary_gf192::BinaryFieldGF192;

        // Helper with the same trait bounds as the IC prover methods.
        // Instantiating it with `F = BinaryFieldGF192` asserts at
        // compile time that all three IC `prove_*` bounds resolve:
        //   - F: InnerTransparentField
        //   - F::Inner: ConstTranscribable
        //   - F::Modulus: ConstTranscribable
        //
        // Plus `U: IdealCheckProtocol`, which holds for every Uair via
        // the blanket impl above.
        #[allow(unused)]
        fn _check_bounds<F, U>()
        where
            F: zinc_utils::inner_transparent_field::InnerTransparentField,
            F::Inner: ConstTranscribable,
            F::Modulus: ConstTranscribable,
            U: IdealCheckProtocol,
        {
        }
        _check_bounds::<BinaryFieldGF192, TestUairNoMultiplication<Int<5>>>();
    }

    // -- All-F_2 IC integration test --------------------------------
    //
    // A minimal `Uair` whose trace is all-`F_2[X]`-typed (two
    // `binary_poly` columns, no `arbitrary_poly` or `int`). The
    // single constraint is `col_0 - col_1 ∈ <0>` — i.e., the two
    // columns must be identical. We choose `assert_zero` (zero
    // ideal) so the IC's per-constraint loop short-circuits and we
    // can verify the prove-path runs end-to-end without needing a
    // full verifier round-trip.
    //
    // `Scalar = BinaryPoly<32>` keeps the scalar type purely F_2.
    // `Ideal = ImpossibleIdeal` is unused (`assert_zero` doesn't
    // touch the ideal machinery).

    use std::marker::PhantomData;
    use zinc_poly::univariate::binary::BinaryPoly;
    use zinc_uair::{
        ConstraintBuilder, TraceRow, UairSignature, ideal::ImpossibleIdeal,
        PublicColumnLayout, TotalColumnLayout,
    };

    #[derive(Clone, Debug, Default)]
    pub struct MinimalF2Uair(PhantomData<()>);

    impl Uair for MinimalF2Uair {
        type Ideal = ImpossibleIdeal;
        type Scalar = BinaryPoly<32>;

        fn signature() -> UairSignature {
            UairSignature::new(
                TotalColumnLayout::new(/* binary */ 2, /* arb */ 0, /* int */ 0),
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
            // The single zero-ideal constraint: col_0 = col_1.
            // Subtracting is XOR in F_2; the IC sees this as
            // `is_zero_ideal[0] = true` and emits a ZERO claim.
            b.assert_zero(up.binary_poly[0].clone() - &up.binary_poly[1]);
        }
    }

    /// End-to-end smoke test: `IdealCheckProtocol::prove_combined`
    /// runs against `F = BinaryFieldGF192` on a properly-shaped
    /// `GF(2^192)`-projected all-`F_2` trace. The trace is built
    /// from two identical `BinaryPoly<32>` columns (satisfying the
    /// UAIR's `col_0 = col_1` constraint), projected via
    /// `project_f2_trace_row_major`, and fed into the IC.
    #[test]
    fn ic_runs_against_gf192_with_f2_trace() {
        use crate::projections::{project_f2_trace_row_major, project_scalars};
        use ahash::AHasher;
        use rand::{Rng, rng};
        use std::{collections::HashMap, hash::BuildHasherDefault};
        use zinc_poly::{
            mle::DenseMultilinearExtension,
            univariate::binary_gf192::BinaryFieldGF192,
        };
        use zinc_uair::UairTrace;

        type Gf192 = BinaryFieldGF192;

        let num_vars: usize = 4;
        let poly_size = 1usize << num_vars; // 16 rows
        let mut r = rng();

        // Build column 0 with random F_2[X]<32> values, column 1
        // identical (so the constraint `col_0 - col_1 = 0` holds).
        let col0_vals: Vec<BinaryPoly<32>> =
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
        let trace: UairTrace<'static, BinaryPoly<32>, BinaryPoly<32>, 32> = UairTrace {
            binary_poly: vec![col0, col1].into(),
            arbitrary_poly: vec![].into(),
            int: vec![].into(),
        };

        // Project the trace to GF(2^192).
        let row_major_trace =
            project_f2_trace_row_major::<Gf192, BinaryPoly<32>, BinaryPoly<32>, 32>(
                &trace,
                &(),
            );

        // Scalars: `MinimalF2Uair` has only an `assert_zero`
        // constraint, so `collect_scalars` returns an empty set. The
        // resulting map is empty; the per-scalar projection closure
        // is never called.
        let scalars: ScalarMap<<MinimalF2Uair as Uair>::Scalar, DynamicPolynomialF<Gf192>> =
            project_scalars::<Gf192, MinimalF2Uair>(|scalar| {
                // Lift `BinaryPoly<32>` → `DynamicPolynomialF<GF(2^192)>`
                // via the per-coefficient F_2 ⊂ GF(2^192) embedding.
                let coeffs: Vec<Gf192> = scalar
                    .iter()
                    .map(|b| if b.into_inner() { Gf192::one() } else { Gf192::zero() })
                    .collect();
                DynamicPolynomialF { coeffs }
            });
        // The MinimalF2Uair has zero non-trivial scalars (only assert_zero
        // is used), so this map is expected to be empty.
        assert!(scalars.is_empty(), "MinimalF2Uair should collect no scalars; got {:?}", scalars.len());

        let mut transcript = Blake3Transcript::new();
        let num_constraints = count_constraints::<MinimalF2Uair>();
        let field_cfg = ();

        let result = <MinimalF2Uair as IdealCheckProtocol>::prove_combined::<Gf192>(
            &mut transcript,
            &row_major_trace,
            &scalars,
            num_constraints,
            num_vars,
            &field_cfg,
        )
        .expect("prove_combined should succeed on a satisfied F_2 trace");
        let (proof, prover_state) = result;

        // The single constraint is a zero-ideal, so its combined-MLE
        // value is `DynamicPolynomialF::ZERO`.
        assert_eq!(proof.combined_mle_values.len(), num_constraints);
        for v in &proof.combined_mle_values {
            assert_eq!(
                v,
                &DynamicPolynomialF::<Gf192>::ZERO,
                "zero-ideal constraint should emit DynamicPolynomialF::ZERO",
            );
        }
        // The prover state's evaluation point is `num_vars` field
        // elements; that's all we can sensibly assert without
        // re-deriving it from the transcript.
        assert_eq!(prover_state.evaluation_point.len(), num_vars);

        // Silence unused warnings if `MinimalF2Uair` doesn't use its
        // generic parameter (the PhantomData<()>).
        let _: HashMap<_, _, BuildHasherDefault<AHasher>> = scalars;
    }
}
