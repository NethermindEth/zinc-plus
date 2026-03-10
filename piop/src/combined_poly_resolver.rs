//! Combined polynomial resolver subprotocol.

mod folder;
mod structs;

pub use structs::*;

use crate::{
    combined_poly_resolver::folder::ConstraintFolder,
    ideal_check,
    sumcheck::{MLSumcheck, SumCheckError},
};
use crypto_primitives::{FromPrimitiveWithConfig, PrimeField};
use itertools::Itertools;
use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::{collections::HashMap, marker::PhantomData, slice};
use thiserror::Error;
use zinc_poly::{
    EvaluatablePolynomial, EvaluationError,
    mle::{DenseMultilinearExtension, MultilinearExtensionWithConfig},
    univariate::dynamic::over_field::DynamicPolynomialF,
    utils::{ArithErrors, build_eq_x_r_inner, eq_eval},
};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_uair::{TraceRow, Uair, ideal::ImpossibleIdeal};
use zinc_utils::{
    cfg_iter, from_ref::FromRef, inner_transparent_field::InnerTransparentField, powers,
};

pub struct CombinedPolyResolver<F: InnerTransparentField>(PhantomData<F>);

impl<F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync> CombinedPolyResolver<F> {
    /// The prover part of the combined polynomial resolver subprotocol.
    /// It projects the trace matrix along the projection F[X] -> F
    /// defined by a random challenge and runs the prover part
    /// of a sumcheck protocol of the form:
    /// $$
    /// \sum_{b \in H} (f_0(b, x_0[b],...,x_n[b], x_0ˆdown[b],...,x_nˆdown[b])
    ///                 + \alpha f_1(...) + ... + \alpha^k f_k(...)) = v_0 +
    ///                   \alpha * v_1 + ... + \alphaˆk * v_k,
    /// $$
    /// where $f_i(b, x_0[b],...,x_n[b], x_0ˆdown[b],...,x_nˆdown[b])
    ///         = eq(r, b) * (1 - eq(r, 1,...1))
    ///             * g_i(x_0[b],...,x_n[b], x_0ˆdown[b],...,x_nˆdown[b])$
    /// and `g_i` is a constraint polynomial given by the UAIR `U`.
    /// `v_0,...,v_k` are the claimed evaluations of the combined polynomials.
    ///
    /// # Parameters
    /// - `transcript`: FS-transcript.
    /// - `trace_matrix`: The trace that have been projected to F.
    /// - `evaluation_point`: The evaluation point for the claims.
    /// - `projected_scalars`: The UAIR scalars projected to `F`.
    /// - `num_constraints`: The number of constraint polynomials in the UAIR
    ///   `U`.
    /// - `num_vars`: The number of variables of the trace MLEs.
    /// - `max_degree`: The degree of the UAIR `U`.
    /// - `field_cfg`: The random field config.
    #[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
    pub fn prove_as_subprotocol<U>(
        transcript: &mut impl Transcript,
        trace_matrix: Vec<DenseMultilinearExtension<F::Inner>>,
        evaluation_point: &[F],
        projected_scalars: &HashMap<U::Scalar, F>,
        num_constraints: usize,
        num_vars: usize,
        max_degree: usize,
        field_cfg: &F::Config,
    ) -> Result<(Proof<F>, ProverState<F>), CombinedPolyResolverError<F>>
    where
        F::Inner: ConstTranscribable + Send + Sync + Zero + Default,
        F::Modulus: ConstTranscribable,
        U: Uair,
    {
        debug_assert_ne!(
            num_vars, 1,
            "The protocol is not needed when the number of variables is 1 :)"
        );

        let zero = F::zero_with_cfg(field_cfg);
        let one = F::one_with_cfg(field_cfg);

        // Shifted trace. Just take the trace, drop the first row
        // and append 0 to the end. Note, that the latter happens
        // thanks to the FromIterator implementation for `DenseMultilinearExtension`
        // as it always pads to the next power of two.
        // It might lead to a problem when `num_vars = 1` but that is not going to
        // happen on real world traces.
        let down: Vec<DenseMultilinearExtension<F::Inner>> = cfg_iter!(trace_matrix)
            .map(|column| column[1..].iter().cloned().collect())
            .collect();

        let eq_r = build_eq_x_r_inner(evaluation_point, field_cfg)?;

        // To get the constraints on the last row ignored
        // we multiply each constraint polynomial
        // by the selector (1 - eq(1,...,1, x))
        let last_row_selector = build_eq_x_r_inner(&vec![one.clone(); num_vars], field_cfg)?;

        // The challenge '\alpha' to batch multiple evaluation claims
        let folding_challenge: F = transcript.get_field_challenge(field_cfg);

        let folding_challenge_powers: Vec<F> =
            powers(folding_challenge, one.clone(), num_constraints);

        let num_cols = trace_matrix.len();

        let mles: Vec<DenseMultilinearExtension<F::Inner>> = {
            let mut mles = Vec::with_capacity(2 * num_cols + 2);

            mles.push(last_row_selector);
            mles.push(eq_r);

            mles.extend(trace_matrix);
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

                let project = |scalar: &U::Scalar| {
                    projected_scalars
                        .get(scalar)
                        .cloned()
                        .expect("all scalars should have been projected at this point")
                };

                U::constrain_general(
                    &mut folder,
                    TraceRow::from_slice_with_signature(
                        &mle_values[2..num_cols + 2],
                        &U::signature(),
                    ),
                    TraceRow::from_slice_with_signature(
                        &mle_values[num_cols + 2..],
                        &U::signature(),
                    ),
                    project,
                    |x, y| Some(project(y) * x),
                    ImpossibleIdeal::from_ref,
                );

                folder.folded_constraints * (one.clone() - selector) * eq_r
            },
            field_cfg,
        );

        // Sumcheck prover stops evaluating MLEs
        // at the second to last challenge
        // leaving all MLEs in num_vars=1
        // state. We need to evaluate them up
        // and send to the verifier.
        debug_assert!(
            sumcheck_prover_state
                .mles
                .iter()
                .all(|mle| mle.num_vars == 1)
        );

        let last_sumcheck_challenge = sumcheck_prover_state
            .randomness
            .last()
            .expect("sumcheck could not have had 0 rounds");

        let mut mles = sumcheck_prover_state.mles;
        let evals: Vec<F> = mles
            .drain(2..)
            .map(|mle| {
                mle.evaluate_with_config(slice::from_ref(last_sumcheck_challenge), field_cfg)
            })
            .try_collect()?;

        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];
        transcript.absorb_random_field_slice(&evals, &mut transcription_buf);

        Ok((
            Proof {
                sumcheck_proof,
                up_evals: evals[0..num_cols].to_vec(),
                down_evals: evals[num_cols..].to_vec(),
            },
            ProverState {
                evaluation_point: sumcheck_prover_state.randomness,
            },
        ))
    }

    /// The verifier part of the combined polynomial resolver
    /// subprotocol.
    ///
    /// # Parameters
    /// - `transcript`: FS-transcript.
    /// - `proof`: The prover's proof.
    /// - `num_constraints`: The number of constraints of the UAIR `U`.
    /// - `max_degree`: The degree of the UAIR `U`.
    /// - `projecting_element`: The random challenge used to project F[X]->F.
    /// - `projected_scalars`: The scalars of the UAIR `U` projected onto `F`.
    /// - `ic_check_subclaim`: The subclaim left after the ideal check
    ///   subprotocol. The subclaim is resolved by this protocol.
    /// - `field_cfg`: The random field config.
    #[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
    // TODO(Ilia): sanitise too_many_arguments ^ once we have time
    pub fn verify_as_subprotocol<U>(
        transcript: &mut impl Transcript,
        proof: Proof<F>,
        num_constraints: usize,
        num_vars: usize,
        max_degree: usize,
        projecting_element: &F,
        projected_scalars: &HashMap<U::Scalar, F>,
        ic_check_subclaim: ideal_check::VerifierSubclaim<F>,
        field_cfg: &F::Config,
    ) -> Result<VerifierSubclaim<F>, CombinedPolyResolverError<F>>
    where
        F::Inner: ConstTranscribable,
        F::Modulus: ConstTranscribable,
        U: Uair,
    {
        proof.validate_evaluation_sizes(U::signature().total_cols())?;

        let zero = F::zero_with_cfg(field_cfg);
        let one = F::one_with_cfg(field_cfg);

        let folding_challenge: F = transcript.get_field_challenge(field_cfg);

        let folding_challenge_powers: Vec<F> =
            powers(folding_challenge, one.clone(), num_constraints);

        // TODO(Alex): investigate if parallelising this is beneficial.
        // Compute v_0 + \alpha * v_1 + ... + \alpha ^ k * v_k.
        let expected_sum = ic_check_subclaim
            .values
            .iter()
            .zip(&folding_challenge_powers)
            .map(
                |(claimed_value, random_coeff)| -> Result<F, CombinedPolyResolverError<F>> {
                    Ok(claimed_value
                        .evaluate_at_point(projecting_element)
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

        let subclaim = MLSumcheck::verify_as_subprotocol(
            transcript,
            num_vars,
            max_degree + 2,
            &proof.sumcheck_proof,
            field_cfg,
        )?;

        let sumcheck_point = subclaim.point;

        let eq_r_value = eq_eval(
            &sumcheck_point,
            &ic_check_subclaim.evaluation_point,
            one.clone(),
        )?;
        let selector_value = eq_eval(&sumcheck_point, &vec![one.clone(); num_vars], one.clone())?;

        let mut folder = ConstraintFolder::new(&folding_challenge_powers, &zero);

        let project = |scalar: &U::Scalar| {
            projected_scalars
                .get(scalar)
                .cloned()
                .expect("all scalars should have been projected at this point")
        };

        U::constrain_general(
            &mut folder,
            TraceRow::from_slice_with_signature(&proof.up_evals, &U::signature()),
            TraceRow::from_slice_with_signature(&proof.down_evals, &U::signature()),
            project,
            |x, y| Some(project(y) * x),
            ImpossibleIdeal::from_ref,
        );

        let expected_claim_value = eq_r_value * (one - selector_value) * folder.folded_constraints;

        if expected_claim_value != subclaim.expected_evaluation {
            return Err(CombinedPolyResolverError::ClaimValueDoesNotMatch {
                got: subclaim.expected_evaluation,
                expected: expected_claim_value,
            });
        }

        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];
        transcript.absorb_random_field_slice(&proof.up_evals, &mut transcription_buf);
        transcript.absorb_random_field_slice(&proof.down_evals, &mut transcription_buf);

        Ok(VerifierSubclaim {
            up_evals: proof.up_evals,
            down_evals: proof.down_evals,
            evaluation_point: sumcheck_point,
        })
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
    #[error("wrong trace columns evaluations number: got {got}, expected {expected}")]
    WrongUpEvalsNumber { got: usize, expected: usize },
    #[error("wrong shifted trace columns evaluations number: got {got}, expected {expected}")]
    WrongDownEvalsNumber { got: usize, expected: usize },
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

#[cfg(test)]
mod tests {
    use crate::{
        ideal_check::IdealCheckProtocol,
        projections::{evaluate_trace_to_column_mles, project_scalars_to_field},
        test_utils::{LIMBS, run_ideal_check_prover_combined, test_config},
    };
    use crypto_primitives::{crypto_bigint_int::Int, crypto_bigint_monty::MontyField};
    use rand::rng;
    use zinc_poly::univariate::dense::DensePolynomial;
    use zinc_test_uair::{
        GenerateSingleTypeWitness, TestAirNoMultiplication, TestUairSimpleMultiplication,
    };
    use zinc_transcript::KeccakTranscript;
    use zinc_uair::{
        constraint_counter::count_constraints,
        degree_counter::count_max_degree,
        ideal::{Ideal, IdealCheck, degree_one::DegreeOneIdeal},
        ideal_collector::IdealOrZero,
    };

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

        let mut prover_transcript = KeccakTranscript::new();
        let mut verifier_transcript = prover_transcript.clone();

        let trace = U::generate_witness(num_vars, &mut rng);

        let (ic_proof, ic_prover_state, projected_scalars, projected_trace) =
            run_ideal_check_prover_combined::<U, DEGREE_PLUS_ONE>(
                num_vars,
                &trace,
                &mut prover_transcript,
            );

        let num_constraints = count_constraints::<U>();

        let ic_check_subclaim = U::verify_as_subprotocol(
            &mut verifier_transcript,
            ic_proof,
            num_constraints,
            num_vars,
            ideal_over_f_from_ref,
            &test_config(),
        )
        .expect("Verification failed");

        let max_degree = count_max_degree::<U>();

        let projecting_element: MontyField<4> =
            prover_transcript.get_field_challenge(&test_config());

        let projected_scalars =
            project_scalars_to_field(projected_scalars, &projecting_element).unwrap();

        let (proof, _) = CombinedPolyResolver::prove_as_subprotocol::<U>(
            &mut prover_transcript,
            evaluate_trace_to_column_mles(&projected_trace, &projecting_element),
            &ic_prover_state.evaluation_point,
            &projected_scalars,
            num_constraints,
            num_vars,
            max_degree,
            &test_config(),
        )
        .expect("CombinedPolyResolver prover failed");

        let projecting_element: MontyField<LIMBS> =
            verifier_transcript.get_field_challenge(&test_config());

        assert!(
            CombinedPolyResolver::verify_as_subprotocol::<U>(
                &mut verifier_transcript,
                proof,
                num_constraints,
                num_vars,
                max_degree,
                &projecting_element,
                &projected_scalars,
                ic_check_subclaim,
                &test_config(),
            )
            .is_ok()
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
