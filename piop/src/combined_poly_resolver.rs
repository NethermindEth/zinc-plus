mod folder;
mod structs;

use derive_more::{Display, From};
use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::marker::PhantomData;
use thiserror::Error;
use zinc_poly::mle::MultilinearExtensionWithConfig;
use zinc_poly::mle::dense::CollectDenseMleWithZero;
use zinc_poly::utils::{ArithErrors, build_eq_x_r_inner, eq_eval};
use zinc_poly::{CoefficientProjectable, EvaluatablePolynomial, EvaluationError};
use zinc_uair::ideal::DummyIdeal;
use zinc_utils::{cfg_iter, field, powers};

use crypto_primitives::{
    DenseRowMatrix, FromPrimitiveWithConfig, FromWithConfig, PrimeField, Semiring,
};
use zinc_poly::{
    mle::DenseMultilinearExtension, univariate::dynamic::over_field::DynamicPolynomialF,
};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_uair::Uair;
use zinc_utils::projectable_to_field::ProjectableToField;
use zinc_utils::{cfg_into_iter, inner_transparent_field::InnerTransparentField};

use crate::combined_poly_resolver::folder::ConstraintFolder;
use crate::sumcheck::{self, MLSumcheck, SumCheckError};

pub use structs::*;

pub struct CombinedPolyResolver<F: InnerTransparentField>(PhantomData<F>);

impl<F: InnerTransparentField + FromPrimitiveWithConfig> CombinedPolyResolver<F> {
    #[allow(clippy::arithmetic_side_effects)]
    pub fn prove_as_subprotocol<Rcoeff, R, U, const DEGREE_PLUS_ONE: usize>(
        transcript: &mut impl Transcript,
        trace_matrix: &[DenseMultilinearExtension<DynamicPolynomialF<F>>],
        evaluation_point: &[F],
        num_constraints: usize,
        num_vars: usize,
        max_degree: usize,
        field_cfg: &F::Config,
    ) -> Result<(Proof<F>, ProverState<F>), CombinedPolyResolverError<F>>
    where
        R: Semiring + 'static + CoefficientProjectable<Rcoeff, DEGREE_PLUS_ONE>,
        F: FromWithConfig<Rcoeff>,
        F::Inner: ConstTranscribable + Zero,
        U: Uair<R>,
    {
        let projecting_element: F = transcript.get_field_challenge(field_cfg);

        let zero = F::zero_with_cfg(field_cfg);

        let num_cols = U::num_cols();

        let up: Vec<DenseMultilinearExtension<F::Inner>> = cfg_iter!(trace_matrix)
            .map(|column| {
                cfg_iter!(column)
                    .map(|coeff| {
                        coeff
                            .evaluate_at_point(&projecting_element)
                            .map_err(|err| {
                                CombinedPolyResolverError::ProjectionError(
                                    coeff.clone(),
                                    projecting_element.clone(),
                                    err,
                                )
                            })
                            .expect("todo")
                            .inner()
                            .clone()
                    })
                    .collect_dense_mle_with_zero(zero.inner())
            })
            .collect();

        let down: Vec<DenseMultilinearExtension<F::Inner>> = cfg_iter!(up)
            .map(|column| {
                cfg_iter!(column[1..])
                    .cloned()
                    .collect_dense_mle_with_zero(zero.inner())
            })
            .collect();

        let eq_r = build_eq_x_r_inner(evaluation_point, field_cfg)?;

        let last_row_selector =
            build_eq_x_r_inner(&vec![F::one_with_cfg(field_cfg); num_vars], field_cfg)?;

        let folding_challenge: F = transcript.get_field_challenge(field_cfg);

        let folding_challenge_powers: Vec<F> = powers(
            folding_challenge,
            F::one_with_cfg(field_cfg),
            num_constraints,
        );

        let mles: Vec<DenseMultilinearExtension<F::Inner>> = {
            let mut mles = Vec::with_capacity(2 * num_cols + 2);

            mles.push(last_row_selector);
            mles.push(eq_r);

            mles.extend(up);
            mles.extend(down);

            mles
        };

        let (sumcheck_proof, sumcheck_prover_state) = MLSumcheck::prove_as_subprotocol(
            transcript,
            mles,
            num_vars,
            // we multiply the combined poly by the selector and eq_r which are
            // linear.
            max_degree + 2,
            |mle_values: &[F]| {
                let selector = &mle_values[0];
                let eq_r = &mle_values[1];

                let mut folder = ConstraintFolder::new(&folding_challenge_powers, &zero);

                U::constrain_general(
                    &mut folder,
                    &mle_values[2..num_cols + 2],
                    &mle_values[num_cols + 2..],
                    |x| F::one_with_cfg(field_cfg),
                    |x, y| Some(F::one_with_cfg(field_cfg)),
                    |x| DummyIdeal,
                );

                folder.folded_constraints * selector * eq_r
            },
            field_cfg,
        );

        let sumcheck_eval_point = sumcheck_prover_state.randomness.clone();

        let evals: Vec<F> = cfg_iter!(sumcheck_prover_state.mles[2..])
            .map(|mle| mle.evaluate_with_config(&sumcheck_eval_point, field_cfg))
            .collect::<Result<Vec<_>, _>>()?;

        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];
        transcript.absorb_random_field_slice(&evals, &mut transcription_buf);

        Ok((
            Proof {
                sumcheck_proof,
                up_evals: evals[0..num_cols].to_vec(),
                down_evals: evals[num_cols..].to_vec(),
            },
            ProverState {
                sumcheck_prover_state,
            },
        ))
    }

    pub fn verify_as_subprotocol<R, U>(
        transcript: &mut impl Transcript,
        proof: Proof<F>,
        num_constraints: usize,
        num_vars: usize,
        max_degree: usize,
        evaluation_point: &[F],
        claimed_values: &[DynamicPolynomialF<F>],
        field_cfg: &F::Config,
    ) -> Result<(), CombinedPolyResolverError<F>>
    where
        R: Semiring + 'static,
        F::Inner: ConstTranscribable,
        U: Uair<R>,
    {
        let projecting_element: F = transcript.get_field_challenge(field_cfg);

        let zero = F::zero_with_cfg(field_cfg);
        let one = F::one_with_cfg(field_cfg);

        let folding_challenge: F = transcript.get_field_challenge(field_cfg);

        let folding_challenge_powers: Vec<F> =
            powers(folding_challenge, one.clone(), num_constraints);

        let expected_sum = claimed_values
            .iter()
            .zip(&folding_challenge_powers)
            .map(
                |(claimed_value, random_coeff)| -> Result<F, CombinedPolyResolverError<F>> {
                    Ok(claimed_value
                        .evaluate_at_point(&projecting_element)
                        .map_err(|err| {
                            CombinedPolyResolverError::ProjectionError(
                                claimed_value.clone(),
                                projecting_element.clone(),
                                err,
                            )
                        })?
                        * random_coeff)
                },
            )
            .try_fold(
                zero.clone(),
                |acc, next| -> Result<F, CombinedPolyResolverError<F>> { Ok(acc + next?) },
            )?;

        if proof.sumcheck_proof.claimed_sum != expected_sum {
            return Err(CombinedPolyResolverError::WrongSumcheckSum {
                got: proof.sumcheck_proof.claimed_sum,
                expected: expected_sum,
            });
        }

        let subclaim: sumcheck::verifier::SubClaim<F> = MLSumcheck::verify_as_subprotocol(
            transcript,
            num_vars,
            max_degree,
            &proof.sumcheck_proof,
            field_cfg,
        )?;

        let sumcheck_point = subclaim.point;

        let eq_r_value = eq_eval(&sumcheck_point, evaluation_point, one.clone())?;
        let selector_value = eq_eval(&sumcheck_point, &vec![one.clone(); num_vars], one)?;

        let mut folder = ConstraintFolder::new(&folding_challenge_powers, &zero);

        U::constrain_general(
            &mut folder,
            &proof.up_evals,
            &proof.down_evals,
            |x| F::zero_with_cfg(field_cfg),
            |x, y| Some(F::zero_with_cfg(field_cfg)),
            |_| DummyIdeal,
        );

        let expected_claim_value = eq_r_value * selector_value * folder.folded_constraints;

        if expected_claim_value != subclaim.expected_evaluation {
            return Err(CombinedPolyResolverError::ClaimValueDoesNotMatch {
                got: subclaim.expected_evaluation,
                expected: expected_claim_value,
            });
        }

        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];
        transcript.absorb_random_field_slice(&proof.up_evals, &mut transcription_buf);
        transcript.absorb_random_field_slice(&proof.down_evals, &mut transcription_buf);

        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum CombinedPolyResolverError<F: PrimeField> {
    #[error("failed to build eq_r: {0}")]
    EqrError(ArithErrors),
    #[error("error evaluating MLE: {0}")]
    MleEvaluationError(EvaluationError),
    #[error("error projecting polynomial {0} by point {1}: {2}")]
    ProjectionError(DynamicPolynomialF<F>, F, EvaluationError),
    #[error("sumcheck verification failed: {0}")]
    SumcheckError(SumCheckError<F>),
    #[error("wrong sumcheck claimed sum: received {got}, expected {expected}")]
    WrongSumcheckSum { got: F, expected: F },
    #[error("resulting claim value does not match: received {got}, expected {expected}")]
    ClaimValueDoesNotMatch { got: F, expected: F },
}

impl<F: PrimeField> From<EvaluationError> for CombinedPolyResolverError<F> {
    fn from(eval_error: EvaluationError) -> Self {
        Self::MleEvaluationError(eval_error)
    }
}

impl<F: PrimeField> From<ArithErrors> for CombinedPolyResolverError<F> {
    fn from(arith_error: ArithErrors) -> Self {
        Self::EqrError(arith_error)
    }
}

impl<F: PrimeField> From<SumCheckError<F>> for CombinedPolyResolverError<F> {
    fn from(sumcheck_error: SumCheckError<F>) -> Self {
        Self::SumcheckError(sumcheck_error)
    }
}
