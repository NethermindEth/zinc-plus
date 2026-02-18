//! Ideal-check subprotocol.
mod batched_ideal_check;
mod combined_poly_builder;
mod structs;

pub use structs::*;

use batched_ideal_check::*;
use crypto_primitives::PrimeField;
use derive_more::From;
use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::collections::HashMap;
use thiserror::Error;
use zinc_poly::{
    EvaluationError,
    mle::{DenseMultilinearExtension, MultilinearExtensionWithConfig},
    univariate::dynamic::over_field::DynamicPolynomialF,
};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_uair::{
    NonLinearUair, Uair,
    ideal::{Ideal, IdealCheck},
    ideal_collector::{IdealOrZero, collect_ideals},
};
use zinc_utils::{cfg_iter, inner_transparent_field::InnerTransparentField};

/// Ideal-check subprotocol.
pub trait IdealCheckProtocol: Uair {
    /// The prover part of the ideal-check subprotocol.
    ///
    /// The prover samples a random field element
    /// and projects the coefficients of coefficients
    /// of the input MLEs. Then it computes the combined polynomials
    /// encoded by the UAIR `U`, samples a random evaluation point
    /// and sends the evaluations of the combined polynomials
    /// to the verifier.
    ///
    /// # Parameters
    /// - `transcript`: the Fiat-Shamir transcript.
    /// - `trace`: the input trace for the UAIR `U` projected to
    ///   `DynamicPolynomialF<F>`.
    /// - `projected_scalars`: the scalars of the UAIR `U` projected to
    ///   `DynamicPolynomialF<F>`.
    /// - `num_constraints`: the number of constraints the UAIR `U` encodes.
    /// - `num_vars`: the number of variables the trace row MLEs have.
    /// - `field_cfg`: random field configuration sampled on the previous steps
    ///   of the overall protocol.
    #[allow(clippy::type_complexity)]
    fn prove_as_subprotocol<F>(
        transcript: &mut impl Transcript,
        trace: &[DenseMultilinearExtension<DynamicPolynomialF<F>>],
        projected_scalars: &HashMap<Self::Scalar, DynamicPolynomialF<F>>,
        num_constraints: usize,
        num_vars: usize,
        field_cfg: &F::Config,
    ) -> Result<(Proof<F>, ProverState<F>), IdealCheckError<F, Self::Ideal>>
    where
        F: InnerTransparentField,
        F::Inner: ConstTranscribable;

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
    ) -> Result<VerifierSubClaim<F>, IdealCheckError<F, IdealOverF>>
    where
        F: InnerTransparentField,
        F::Inner: ConstTranscribable,
        IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<F>>,
        IdealOverFFromRef: Fn(&IdealOrZero<Self::Ideal>) -> IdealOverF;
}

impl<U> IdealCheckProtocol for U
where
    U: NonLinearUair,
{
    #[allow(clippy::type_complexity)]
    fn prove_as_subprotocol<F>(
        transcript: &mut impl Transcript,
        trace: &[DenseMultilinearExtension<DynamicPolynomialF<F>>],
        projected_scalars: &HashMap<U::Scalar, DynamicPolynomialF<F>>,
        num_constraints: usize,
        num_vars: usize,
        field_cfg: &F::Config,
    ) -> Result<(Proof<F>, ProverState<F>), IdealCheckError<F, U::Ideal>>
    where
        F: InnerTransparentField,
        F::Inner: ConstTranscribable,
    {
        let evaluation_point = transcript.get_field_challenges(num_vars, field_cfg);

        let combined_mles = combined_poly_builder::compute_combined_polynomials::<_, U>(
            trace,
            projected_scalars,
            num_constraints,
            field_cfg,
        );

        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];

        let combined_mle_values = cfg_iter!(combined_mles)
            .map(|combined_mle| {
                Ok(DynamicPolynomialF::new_trimmed(
                    cfg_iter!(combined_mle)
                        .map(|coeff_mle| {
                            coeff_mle.evaluate_with_config(&evaluation_point, field_cfg)
                        })
                        .collect::<std::result::Result<Vec<_>, _>>()?,
                ))
            })
            .collect::<std::result::Result<Vec<_>, IdealCheckError<_, _>>>()?;

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
    ) -> Result<VerifierSubClaim<F>, IdealCheckError<F, IdealOverF>>
    where
        F: InnerTransparentField,
        F::Inner: ConstTranscribable,
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

        batched_ideal_check(
            &ideal_collector
                .ideals
                .iter()
                .map(ideal_over_f_from_ref)
                .collect_vec(),
            &combined_mle_values,
        )?;

        Ok(VerifierSubClaim {
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

    use crate::test_utils::{LIMBS, run_ideal_check_prover_single_type, test_config};

    use super::*;

    // TODO(Ilia): These tests are absolute joke.
    //             Once we have time we need to create a comprehensive test suite
    //             akin to the one we have for the PCS or the sumcheck.

    fn test_successful_verification_generic<
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

        let (proof, prover_state, ..) = run_ideal_check_prover_single_type::<U, DEGREE_PLUS_ONE>(
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

        test_successful_verification_generic::<TestAirNoMultiplication<5>, _, _, 32>(
            num_vars,
            |ideal_over_ring| ideal_over_ring.map(|i| DegreeOneIdeal::from_with_cfg(i, &field_cfg)),
        );
        test_successful_verification_generic::<TestUairSimpleMultiplication<Int<5>>, _, _, 32>(
            num_vars,
            |_ideal_over_ring| IdealOrZero::zero(),
        );
    }
}
