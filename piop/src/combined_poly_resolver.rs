//! Combined polynomial resolver subprotocol.

mod folder;
mod structs;
mod utils;

use itertools::Itertools;
use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::{collections::HashMap, marker::PhantomData, slice};
use thiserror::Error;
use zinc_poly::{
    EvaluatablePolynomial, EvaluationError,
    mle::MultilinearExtensionWithConfig,
    utils::{ArithErrors, build_eq_x_r_inner, eq_eval},
};
use zinc_uair::ideal::ImpossibleIdeal;
use zinc_utils::{cfg_iter, from_ref::FromRef, powers};

use crypto_primitives::{FromPrimitiveWithConfig, PrimeField, Semiring};
use zinc_poly::{
    mle::DenseMultilinearExtension, univariate::dynamic::over_field::DynamicPolynomialF,
};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_uair::Uair;
use zinc_utils::inner_transparent_field::InnerTransparentField;

use crate::{
    combined_poly_resolver::{folder::ConstraintFolder, utils::project_scalars_to_field},
    ideal_check,
    sumcheck::{MLSumcheck, SumCheckError},
};

pub use structs::*;

pub struct CombinedPolyResolver<F: InnerTransparentField>(PhantomData<F>);

impl<F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync> CombinedPolyResolver<F> {
    /// The prover part of the combined polynomial resolver subprotocol.
    /// It projects the trace matrix along the projection F[X] -> F
    /// defined by a random challenge and runs the prover part
    /// of a sumcheck protocol of the form:
    /// $$
    /// \sum_{b \in H} (f_0(b, x_0[b],...,x_n[b], x_0ˆdown[b],...,x_nˆdown[b])
    ///                 + \alpha f_1(...) + ... + \alpha^k f_k(...))
    ///                     = v_0 + \alpha * v_1 + ... + \alphaˆk * v_k,
    /// $$
    /// where $f_i(b, x_0[b],...,x_n[b], x_0ˆdown[b],...,x_nˆdown[b])
    ///         = eq(r, b) * (1 - eq(r, 1,...1))
    ///             * g_i(x_0[b],...,x_n[b], x_0ˆdown[b],...,x_nˆdown[b])$
    /// and `g_i` is a constraint polynomial given by the UAIR `U`.
    /// `v_0,...,v_k` are the claimed evaluations of the combined polynomials.
    ///
    /// # Parameters
    /// - `transcript`: FS-transcript.
    /// - `trace_matrix`: The trace that have been projected to F[X].
    /// - `evaluation_point`: The evaluation point for the claims.
    /// - `projected_scalars`: The UAIR scalars projected to `F[X]`.
    /// - `num_constraints`: The number of constraint polynomials in the UAIR `U`.
    /// - `num_vars`: The number of variables of the trace MLEs.
    /// - `max_degree`: The degree of the UAIR `U`.
    /// - `field_cfg`: The random field config.
    #[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
    pub fn prove_as_subprotocol<R, U>(
        transcript: &mut impl Transcript,
        trace_matrix: &[DenseMultilinearExtension<DynamicPolynomialF<F>>],
        evaluation_point: &[F],
        projected_scalars: HashMap<R, DynamicPolynomialF<F>>,
        num_constraints: usize,
        num_vars: usize,
        max_degree: usize,
        field_cfg: &F::Config,
    ) -> Result<(Proof<F>, ProverState<F>), CombinedPolyResolverError<F>>
    where
        R: Semiring + Send + Sync + 'static,
        F::Inner: ConstTranscribable + Send + Sync + Zero + Default,
        U: Uair<R>,
    {
        let projecting_element: F = transcript.get_field_challenge(field_cfg);

        // Project scalars along F[X] -> F.
        let projected_scalars: HashMap<R, F> =
            project_scalars_to_field(projected_scalars, &projecting_element)?;

        let zero = F::zero_with_cfg(field_cfg);
        let one = F::one_with_cfg(field_cfg);

        let num_cols = U::num_cols();

        // Project trace along F[X] -> F.
        let up: Vec<DenseMultilinearExtension<F::Inner>> = cfg_iter!(trace_matrix)
            .map(|column| {
                cfg_iter!(column)
                    .map(|coeff| {
                        coeff
                            .evaluate_at_point(&projecting_element)
                            .expect("evaluation cannot fail here")
                            .inner()
                            .clone()
                    })
                    .collect()
            })
            .collect();

        // Shifted trace. Just take the trace, drop the first row
        // and append 0 to the end.
        let down: Vec<DenseMultilinearExtension<F::Inner>> = cfg_iter!(up)
            .map(|column| cfg_iter!(column[1..]).cloned().collect())
            .collect();

        let eq_r = build_eq_x_r_inner(evaluation_point, field_cfg)?;

        // To get the constraints on the last row ignored
        // we multiply each constraint polynomial
        // by the selector (1 - eq(1,...,1, x))
        let last_row_selector =
            build_eq_x_r_inner(&vec![F::one_with_cfg(field_cfg); num_vars], field_cfg)?;

        // The challenge '\alpha' to batch multiple evaluation claims
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

            mles.extend(up.clone());
            mles.extend(down.clone());

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

                let project = |scalar: &R| {
                    projected_scalars
                        .get(scalar)
                        .cloned()
                        .expect("all scalars should have been projected at this point")
                };

                U::constrain_general(
                    &mut folder,
                    &mle_values[2..num_cols + 2],
                    &mle_values[num_cols + 2..],
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

        let last_sumcheck_challenge = sumcheck_prover_state
            .randomness
            .last()
            .expect("sumcheck could not have had 0 rounds");

        let evals: Vec<F> = sumcheck_prover_state.mles[2..]
            .iter()
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
                up,
                down,
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
    /// - `ic_check_subclaim`: The subclaim left after the ideal check
    ///   subprotocol. The subclaim is resolved by this protocol.
    /// - `field_cfg`: The random field config.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn verify_as_subprotocol<R, U>(
        transcript: &mut impl Transcript,
        proof: Proof<F>,
        num_constraints: usize,
        num_vars: usize,
        max_degree: usize,
        ic_check_subclaim: ideal_check::VerifierSubClaim<R, F>,
        field_cfg: &F::Config,
    ) -> Result<VerifierSubclaim<F>, CombinedPolyResolverError<F>>
    where
        R: Semiring + 'static,
        F::Inner: ConstTranscribable,
        U: Uair<R>,
    {
        let projecting_element: F = transcript.get_field_challenge(field_cfg);

        let projected_scalars: HashMap<R, F> =
            project_scalars_to_field(ic_check_subclaim.projected_scalars, &projecting_element)?;

        let zero = F::zero_with_cfg(field_cfg);
        let one = F::one_with_cfg(field_cfg);

        let folding_challenge: F = transcript.get_field_challenge(field_cfg);

        let folding_challenge_powers: Vec<F> =
            powers(folding_challenge, one.clone(), num_constraints);

        // Compute v_0 + \alpha * v_1 + ... + \alpha ^ k * v_k.
        let expected_sum = ic_check_subclaim
            .values
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

        let project = |scalar: &R| {
            projected_scalars
                .get(scalar)
                .cloned()
                .expect("all scalars should have been projected at this point")
        };

        U::constrain_general(
            &mut folder,
            &proof.up_evals,
            &proof.down_evals,
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
    use crypto_primitives::{crypto_bigint_int::Int, crypto_bigint_monty::MontyField};
    use rand::rng;
    use zinc_poly::univariate::{dense::DensePolynomial, ideal::DegreeOneIdeal};
    use zinc_test_uair::{GenerateWitness, TestAirNoMultiplication, TestUairSimpleMultiplication};
    use zinc_transcript::KeccakTranscript;
    use zinc_uair::{
        constraint_counter::count_constraints,
        degree_counter::count_max_degree,
        ideal::{Ideal, IdealCheck},
        ideal_collector::IdealOrZero,
    };

    use crate::{
        ideal_check::{IdealCheckProtocol, IdealCheckTypes},
        test_utils::{LIMBS, TestIcTypes, run_ideal_check_prover, test_config},
    };

    use super::*;

    fn test_successful_verification_generic<
        U,
        IdealOverF,
        IdealOverFFromRef,
        const DEGREE_PLUS_ONE: usize,
    >(
        num_vars: usize,
        ideal_over_f_from_ref: IdealOverFFromRef,
    ) where
        U: GenerateWitness<DensePolynomial<Int<5>, DEGREE_PLUS_ONE>>,
        IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<MontyField<LIMBS>>>,
        IdealOverFFromRef: Fn(&IdealOrZero<U::Ideal>) -> IdealOverF,
    {
        let mut rng = rng();

        let mut prover_transcript = KeccakTranscript::new();
        let mut verifier_transcript = prover_transcript.clone();

        let trace = U::generate_witness(num_vars, &mut rng);

        let (ic_proof, ic_prover_state) =
            run_ideal_check_prover::<U, DEGREE_PLUS_ONE>(num_vars, &trace, &mut prover_transcript);

        let num_constraints =
            count_constraints::<<TestIcTypes as IdealCheckTypes<_>>::Witness, U>();

        let ic_check_subclaim =
            IdealCheckProtocol::<TestIcTypes, _>::verify_as_subprotocol::<U, _, _>(
                &mut verifier_transcript,
                ic_proof,
                num_constraints,
                num_vars,
                ideal_over_f_from_ref,
                &test_config(),
            )
            .expect("Verification failed");

        let max_degree = count_max_degree::<_, U>();

        let (proof, _) = CombinedPolyResolver::prove_as_subprotocol::<_, U>(
            &mut prover_transcript,
            &ic_prover_state.trace_matrix,
            &ic_prover_state.evaluation_point,
            ic_prover_state.projected_scalars,
            num_constraints,
            num_vars,
            max_degree,
            &test_config(),
        )
        .expect("CombinedPolyResolver prover failed");

        assert!(
            CombinedPolyResolver::verify_as_subprotocol::<_, U>(
                &mut verifier_transcript,
                proof,
                num_constraints,
                num_vars,
                max_degree,
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

        test_successful_verification_generic::<TestAirNoMultiplication, _, _, 32>(
            num_vars,
            |ideal_over_ring| ideal_over_ring.map(|i| DegreeOneIdeal::from_with_cfg(i, &field_cfg)),
        );
        test_successful_verification_generic::<TestUairSimpleMultiplication, _, _, 32>(
            num_vars,
            |_ideal_over_ring| IdealOrZero::zero(),
        );
    }
}
