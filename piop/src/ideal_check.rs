//! Ideal-check subprotocol.
mod batched_ideal_check;
mod combined_poly_builder;
mod structs;

pub use structs::*;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::projections::{ColumnMajorTrace, RowMajorTrace};
use batched_ideal_check::*;
use crypto_primitives::PrimeField;
use derive_more::From;
use num_traits::ConstZero;
use std::collections::HashMap;
use thiserror::Error;
use zinc_poly::{
    EvaluationError, mle::MultilinearExtensionWithConfig,
    univariate::dynamic::over_field::DynamicPolynomialF,
};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_uair::{
    Uair,
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
        projected_scalars: &HashMap<Self::Scalar, DynamicPolynomialF<F>>,
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
        projected_scalars: &HashMap<Self::Scalar, DynamicPolynomialF<F>>,
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
        projected_scalars: &HashMap<U::Scalar, DynamicPolynomialF<F>>,
        num_constraints: usize,
        num_vars: usize,
        field_cfg: &F::Config,
    ) -> Result<(Proof<F>, ProverState<F>), IdealCheckError<F, U::Ideal>>
    where
        F: InnerTransparentField,
        F::Inner: ConstTranscribable,
        F::Modulus: ConstTranscribable,
    {
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
        projected_scalars: &HashMap<U::Scalar, DynamicPolynomialF<F>>,
        num_constraints: usize,
        num_vars: usize,
        field_cfg: &F::Config,
    ) -> Result<(Proof<F>, ProverState<F>), IdealCheckError<F, U::Ideal>>
    where
        F: InnerTransparentField,
        F::Inner: ConstTranscribable,
        F::Modulus: ConstTranscribable,
    {
        // Collect ideals to identify assert_zero constraints whose
        // combined polynomial is zero by construction (for honest provers).
        let ideal_collector = collect_ideals::<U>(num_constraints);
        let is_zero_ideal: Vec<bool> = ideal_collector
            .ideals
            .iter()
            .map(|i| i.is_zero_ideal())
            .collect();

        let evaluation_point = transcript.get_field_challenges(num_vars, field_cfg);

        // Build combined polynomial MLEs row-by-row and evaluate them.
        let combined_mles = combined_poly_builder::compute_combined_polynomials::<_, U>(
            trace_matrix,
            projected_scalars,
            num_constraints,
            field_cfg,
            &is_zero_ideal,
        );

        // Evaluate coefficient MLEs at the evaluation point.
        let combined_mle_values: Vec<DynamicPolynomialF<F>> = cfg_into_iter!(combined_mles)
            .enumerate()
            .map(|(i, coeff_mles)| {
                // Skip zero-ideal constraints: their combined polynomial
                // is zero for an honest prover.
                if is_zero_ideal[i] {
                    return Ok(DynamicPolynomialF::ZERO);
                }
                let coeffs = coeff_mles
                    .into_iter()
                    .map(|coeff_mle| coeff_mle.evaluate_with_config(&evaluation_point, field_cfg))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(DynamicPolynomialF::new_trimmed(coeffs))
            })
            .collect::<Result<Vec<_>, EvaluationError>>()?;

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

#[derive(Clone, Debug, From, Error)]
pub enum IdealCheckError<F: PrimeField, I> {
    #[error("ideal check prover failed to evaluate an mle: {0}")]
    MleEvaluationError(EvaluationError),
    #[error("mle evaluation ideal check failure: {0}")]
    IdealCollectorError(BatchedIdealCheckError<DynamicPolynomialF<F>, I>),
}

#[cfg(test)]
mod tests {
    use crypto_primitives::{crypto_bigint_int::Int, crypto_bigint_monty::MontyField};

    use rand::rng;
    use zinc_poly::univariate::{dense::DensePolynomial, dynamic::over_field::DynamicPolynomialF};
    use zinc_test_uair::{
        GenerateSingleTypeWitness, TestAirNoMultiplication, TestUairSimpleMultiplication,
    };
    use zinc_transcript::KeccakTranscript;
    use zinc_uair::{
        constraint_counter::count_constraints,
        ideal::{Ideal, IdealCheck, degree_one::DegreeOneIdeal},
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
        U: GenerateSingleTypeWitness<Witness = DensePolynomial<Int<5>, DEGREE_PLUS_ONE>>
            + Uair<Scalar = DensePolynomial<Int<5>, DEGREE_PLUS_ONE>>
            + IdealCheckProtocol,
        IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<MontyField<LIMBS>>>,
        IdealOverFFromRef: Fn(&IdealOrZero<U::Ideal>) -> IdealOverF,
    {
        let mut rng = rng();
        let transcript = KeccakTranscript::new();

        let (proof, prover_state, ..) = run_ideal_check_prover_linear::<U, DEGREE_PLUS_ONE>(
            num_vars,
            &U::generate_witness(num_vars, &mut rng),
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
        U: GenerateSingleTypeWitness<Witness = DensePolynomial<Int<5>, DEGREE_PLUS_ONE>>
            + Uair<Scalar = DensePolynomial<Int<5>, DEGREE_PLUS_ONE>>
            + IdealCheckProtocol,
        IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<MontyField<LIMBS>>>,
        IdealOverFFromRef: Fn(&IdealOrZero<U::Ideal>) -> IdealOverF,
    {
        let mut rng = rng();
        let transcript = KeccakTranscript::new();

        let (proof, prover_state, ..) = run_ideal_check_prover_combined::<U, DEGREE_PLUS_ONE>(
            num_vars,
            &U::generate_witness(num_vars, &mut rng),
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
        test_successful_verification_linear::<TestAirNoMultiplication<Int<5>>, _, _, 32>(
            num_vars,
            |ideal_over_ring| ideal_over_ring.map(|i| DegreeOneIdeal::from_with_cfg(i, &field_cfg)),
        );
        test_successful_verification_combined::<TestAirNoMultiplication<Int<5>>, _, _, 32>(
            num_vars,
            |ideal_over_ring| ideal_over_ring.map(|i| DegreeOneIdeal::from_with_cfg(i, &field_cfg)),
        );

        // Non-linear UAIR - only combined approach works
        test_successful_verification_combined::<TestUairSimpleMultiplication<Int<5>>, _, _, 32>(
            num_vars,
            |_ideal_over_ring| IdealOrZero::zero(),
        );
    }
}
