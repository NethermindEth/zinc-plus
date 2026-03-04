//! Ideal-check subprotocol.
mod batched_ideal_check;
mod combined_poly_builder;
mod structs;

use batched_ideal_check::*;
use crypto_primitives::PrimeField;
use derive_more::From;
use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::{collections::HashMap, marker::PhantomData};
pub use structs::*;
use thiserror::Error;
use zinc_poly::{
    EvaluationError,
    mle::{DenseMultilinearExtension, MultilinearExtensionWithConfig},
    univariate::{binary::BinaryPoly, dynamic::over_field::DynamicPolynomialF},
};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_uair::{
    Uair,
    degree_counter::count_max_degree,
    ideal::{Ideal, IdealCheck},
    ideal_collector::{IdealOrZero, collect_ideals},
};
use zinc_utils::{cfg_iter, inner_transparent_field::InnerTransparentField};

/// Ideal-check subprotocol.
pub struct IdealCheckProtocol<F: InnerTransparentField>(PhantomData<F>);

impl<F: InnerTransparentField> IdealCheckProtocol<F> {
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
    pub fn prove_as_subprotocol<U>(
        transcript: &mut impl Transcript,
        trace: &[DenseMultilinearExtension<DynamicPolynomialF<F>>],
        projected_scalars: &HashMap<U::Scalar, DynamicPolynomialF<F>>,
        num_constraints: usize,
        num_vars: usize,
        field_cfg: &F::Config,
    ) -> Result<(Proof<F>, ProverState<F>), IdealCheckError<F, U::Ideal>>
    where
        U: Uair,
        F::Inner: ConstTranscribable,
    {
        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];

        let evaluation_point = transcript.get_field_challenges(num_vars, field_cfg);

        let max_constraint_degree = count_max_degree::<U>();

        let (combined_mle_values, combined_mles) = if max_constraint_degree <= 1 {
            // For linear constraints (degree ≤ 1), evaluate column MLEs first
            // and then apply the linear combination, avoiding row-by-row
            // DynamicPolynomialF arithmetic.
            let values =
                combined_poly_builder::evaluate_combined_polynomials_linear::<_, U>(
                    trace,
                    projected_scalars,
                    &evaluation_point,
                    num_constraints,
                    field_cfg,
                )?;
            (values, vec![])
        } else {
            let mles = combined_poly_builder::compute_combined_polynomials::<_, U>(
                trace,
                projected_scalars,
                num_constraints,
                field_cfg,
            );

            let values = cfg_iter!(mles)
                .map(|combined_mle| {
                    Ok(DynamicPolynomialF::new_trimmed(
                        cfg_iter!(combined_mle)
                            .map(|coeff_mle| {
                                coeff_mle
                                    .evaluate_with_config(&evaluation_point, field_cfg)
                            })
                            .collect::<std::result::Result<Vec<_>, _>>()?,
                    ))
                })
                .collect::<std::result::Result<Vec<_>, IdealCheckError<_, _>>>()?;

            (values, mles)
        };

        combined_mle_values.iter().for_each(|combined_mle_value| {
            transcript
                .absorb_random_field_slice(&combined_mle_value.coeffs, &mut transcription_buf);
        });

        Ok((
            Proof {
                combined_mle_values,
            },
            ProverState {
                evaluation_point,
                combined_mles,
            },
        ))
    }

    /// MLE-first prover for ideal-check on binary-polynomial traces.
    ///
    /// When the UAIR `U` has only linear constraints (`max_degree == 1`)
    /// and all trace columns are `BinaryPoly<D>`, this method evaluates
    /// each column MLE at the random point **before** projecting to F,
    /// then applies the constraints once.  This avoids the O(N × C × D)
    /// full trace projection and the O(N × num_constraints × D)
    /// row-by-row constraint evaluation.
    ///
    /// The Fiat-Shamir transcript and `Proof` format are identical to
    /// [`prove_as_subprotocol`]; the verifier is unchanged.
    #[allow(clippy::type_complexity)]
    pub fn prove_mle_first<U, const D: usize>(
        transcript: &mut impl Transcript,
        binary_poly_trace: &[DenseMultilinearExtension<BinaryPoly<D>>],
        projected_scalars: &HashMap<U::Scalar, DynamicPolynomialF<F>>,
        num_constraints: usize,
        num_vars: usize,
        field_cfg: &F::Config,
    ) -> Result<(Proof<F>, ProverState<F>), IdealCheckError<F, U::Ideal>>
    where
        U: Uair,
        F::Inner: ConstTranscribable + Default + Send + Sync + num_traits::Zero,
    {
        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];

        let evaluation_point = transcript.get_field_challenges(num_vars, field_cfg);

        let combined_mle_values =
            combined_poly_builder::compute_combined_values_mle_first::<F, U, D>(
                binary_poly_trace,
                projected_scalars,
                &evaluation_point,
                num_constraints,
                field_cfg,
            );

        combined_mle_values.iter().for_each(|combined_mle_value| {
            transcript
                .absorb_random_field_slice(&combined_mle_value.coeffs, &mut transcription_buf);
        });

        Ok((
            Proof {
                combined_mle_values,
            },
            ProverState {
                evaluation_point,
                combined_mles: vec![],
            },
        ))
    }

    /// Like [`prove_mle_first`], but uses a **supplied** evaluation point.
    #[allow(clippy::type_complexity)]
    pub fn prove_mle_first_at_point<U, const D: usize>(
        transcript: &mut impl Transcript,
        binary_poly_trace: &[DenseMultilinearExtension<BinaryPoly<D>>],
        projected_scalars: &HashMap<U::Scalar, DynamicPolynomialF<F>>,
        num_constraints: usize,
        evaluation_point: &[F],
        field_cfg: &F::Config,
    ) -> Result<(Proof<F>, ProverState<F>), IdealCheckError<F, U::Ideal>>
    where
        U: Uair,
        F::Inner: ConstTranscribable + Default + Send + Sync + num_traits::Zero,
    {
        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];

        let combined_mle_values =
            combined_poly_builder::compute_combined_values_mle_first::<F, U, D>(
                binary_poly_trace,
                projected_scalars,
                evaluation_point,
                num_constraints,
                field_cfg,
            );

        combined_mle_values.iter().for_each(|combined_mle_value| {
            transcript
                .absorb_random_field_slice(&combined_mle_value.coeffs, &mut transcription_buf);
        });

        Ok((
            Proof {
                combined_mle_values,
            },
            ProverState {
                evaluation_point: evaluation_point.to_vec(),
                combined_mles: vec![],
            },
        ))
    }

    /// Prove ideal-check for constraints of **any degree** directly from
    /// a `BinaryPoly<D>` trace, without a separate projection pass.
    ///
    /// Combines `project_trace_coeffs` + eq-weighted constraint evaluation
    /// into a single fused pass:
    ///
    /// 1. Draws an evaluation point from the transcript.
    /// 2. Precomputes `eq(r, j)` weights.
    /// 3. For each row: projects `BinaryPoly<D>` values to
    ///    `DynamicPolynomialF<F>` inline, evaluates constraints, and
    ///    accumulates the eq-weighted result.
    ///
    /// This avoids:
    /// - Allocating the full projected trace matrix.
    /// - Intermediate combined-polynomial MLEs.
    /// - A separate MLE evaluation pass.
    ///
    /// The Fiat-Shamir transcript and [`Proof`] format are identical to
    /// [`prove_as_subprotocol`]; the verifier is unchanged.
    #[allow(clippy::type_complexity)]
    pub fn prove_from_binary_poly<U, const D: usize>(
        transcript: &mut impl Transcript,
        binary_poly_trace: &[DenseMultilinearExtension<BinaryPoly<D>>],
        projected_scalars: &HashMap<U::Scalar, DynamicPolynomialF<F>>,
        num_constraints: usize,
        num_vars: usize,
        field_cfg: &F::Config,
    ) -> Result<(Proof<F>, ProverState<F>), IdealCheckError<F, U::Ideal>>
    where
        U: Uair,
        F::Inner: ConstTranscribable + Default + Send + Sync + num_traits::Zero,
    {
        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];

        let evaluation_point = transcript.get_field_challenges(num_vars, field_cfg);

        let max_constraint_degree = count_max_degree::<U>();

        let combined_mle_values = if max_constraint_degree <= 1 {
            // For linear constraints: use MLE-first path.
            combined_poly_builder::compute_combined_values_mle_first::<F, U, D>(
                binary_poly_trace,
                projected_scalars,
                &evaluation_point,
                num_constraints,
                field_cfg,
            )
        } else {
            // For higher-degree constraints: fused projection +
            // row-by-row evaluation + eq-weighted accumulation.
            combined_poly_builder::compute_combined_evals_from_binary_poly_with_eq::<F, U, D>(
                binary_poly_trace,
                projected_scalars,
                &evaluation_point,
                num_constraints,
                field_cfg,
            )
        };

        combined_mle_values.iter().for_each(|combined_mle_value| {
            transcript
                .absorb_random_field_slice(&combined_mle_value.coeffs, &mut transcription_buf);
        });

        Ok((
            Proof {
                combined_mle_values,
            },
            ProverState {
                evaluation_point,
                combined_mles: vec![],
            },
        ))
    }

    /// Like [`prove_from_binary_poly`], but uses a **supplied** evaluation
    /// point instead of drawing one from the transcript.
    ///
    /// This is the fused BinaryPoly path for pipelines that share a
    /// single evaluation point across multiple IC passes (e.g.
    /// dual-ring pipeline).
    ///
    /// **Important:** the caller is responsible for ensuring that
    /// `evaluation_point` was produced via the transcript *before*
    /// this function is called so that Fiat-Shamir consistency is
    /// maintained.
    #[allow(clippy::type_complexity)]
    pub fn prove_from_binary_poly_at_point<U, const D: usize>(
        transcript: &mut impl Transcript,
        binary_poly_trace: &[DenseMultilinearExtension<BinaryPoly<D>>],
        projected_scalars: &HashMap<U::Scalar, DynamicPolynomialF<F>>,
        num_constraints: usize,
        evaluation_point: &[F],
        field_cfg: &F::Config,
    ) -> Result<(Proof<F>, ProverState<F>), IdealCheckError<F, U::Ideal>>
    where
        U: Uair,
        F::Inner: ConstTranscribable + Default + Send + Sync + num_traits::Zero,
    {
        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];

        let max_constraint_degree = count_max_degree::<U>();

        let combined_mle_values = if max_constraint_degree <= 1 {
            combined_poly_builder::compute_combined_values_mle_first::<F, U, D>(
                binary_poly_trace,
                projected_scalars,
                evaluation_point,
                num_constraints,
                field_cfg,
            )
        } else {
            combined_poly_builder::compute_combined_evals_from_binary_poly_with_eq::<F, U, D>(
                binary_poly_trace,
                projected_scalars,
                evaluation_point,
                num_constraints,
                field_cfg,
            )
        };

        combined_mle_values.iter().for_each(|combined_mle_value| {
            transcript
                .absorb_random_field_slice(&combined_mle_value.coeffs, &mut transcription_buf);
        });

        Ok((
            Proof {
                combined_mle_values,
            },
            ProverState {
                evaluation_point: evaluation_point.to_vec(),
                combined_mles: vec![],
            },
        ))
    }

    /// Like [`prove_as_subprotocol`], but uses a **supplied** evaluation
    /// point instead of drawing one from the transcript.
    ///
    /// This is useful when two or more IC passes should share the same
    /// evaluation point (e.g. dual-ring pipeline).
    ///
    /// **Important:** the caller is responsible for ensuring that
    /// `evaluation_point` was produced via the transcript *before*
    /// this function is called so that Fiat-Shamir consistency is
    /// maintained.
    #[allow(clippy::type_complexity)]
    pub fn prove_at_point<U>(
        transcript: &mut impl Transcript,
        trace: &[DenseMultilinearExtension<DynamicPolynomialF<F>>],
        projected_scalars: &HashMap<U::Scalar, DynamicPolynomialF<F>>,
        num_constraints: usize,
        evaluation_point: &[F],
        field_cfg: &F::Config,
    ) -> Result<(Proof<F>, ProverState<F>), IdealCheckError<F, U::Ideal>>
    where
        U: Uair,
        F::Inner: ConstTranscribable,
    {
        let max_constraint_degree = count_max_degree::<U>();

        let combined_mle_values = if max_constraint_degree <= 1 {
            // For linear constraints: evaluate column MLEs first, then apply
            // the linear constraint once.  Avoids row-by-row
            // DynamicPolynomialF arithmetic entirely.
            combined_poly_builder::evaluate_combined_polynomials_at_point::<_, U>(
                trace,
                projected_scalars,
                evaluation_point,
                num_constraints,
                field_cfg,
            )?
        } else {
            // For higher-degree constraints: evaluate row-by-row but
            // accumulate with eq-weights directly, avoiding intermediate
            // combined-MLE storage and the separate evaluation pass.
            combined_poly_builder::compute_combined_polynomial_evals_with_eq::<_, U>(
                trace,
                projected_scalars,
                evaluation_point,
                num_constraints,
                field_cfg,
            )
        };

        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];

        combined_mle_values.iter().for_each(|combined_mle_value| {
            transcript
                .absorb_random_field_slice(&combined_mle_value.coeffs, &mut transcription_buf);
        });

        Ok((
            Proof {
                combined_mle_values,
            },
            ProverState {
                evaluation_point: evaluation_point.to_vec(),
                combined_mles: vec![],
            },
        ))
    }

    /// Like [`verify_as_subprotocol`], but uses a **supplied** evaluation
    /// point instead of drawing one from the transcript.
    ///
    /// Mirror of [`prove_at_point`] for the verifier side.
    #[allow(clippy::type_complexity)]
    pub fn verify_at_point<U, IdealOverF, IdealOverFFromRef>(
        transcript: &mut impl Transcript,
        proof: Proof<F>,
        num_constraints: usize,
        evaluation_point: Vec<F>,
        ideal_over_f_from_ref: IdealOverFFromRef,
        _field_cfg: &F::Config,
    ) -> Result<VerifierSubClaim<F>, IdealCheckError<F, IdealOverF>>
    where
        U: Uair,
        F::Inner: ConstTranscribable,
        IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<F>>,
        IdealOverFFromRef: Fn(&IdealOrZero<U::Ideal>) -> IdealOverF,
    {
        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];

        let combined_mle_values = proof.combined_mle_values;

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

    /// Like [`verify_at_point`] but skips the ideal membership check.
    ///
    /// Absorbs the proof values into the transcript (keeping it in sync
    /// with the prover) and returns the sub-claim directly.  This is
    /// useful when the ideal check is known to fail due to field
    /// mismatch (e.g. ECDSA constraints evaluated over a different
    /// prime) but the rest of the PIOP still needs to proceed.
    pub fn verify_at_point_absorb_only(
        transcript: &mut impl Transcript,
        proof: Proof<F>,
        evaluation_point: Vec<F>,
        _field_cfg: &F::Config,
    ) -> VerifierSubClaim<F>
    where
        F::Inner: ConstTranscribable,
    {
        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];

        let combined_mle_values = proof.combined_mle_values;

        for mle_value in &combined_mle_values {
            transcript.absorb_random_field_slice(&mle_value.coeffs, &mut transcription_buf);
        }

        VerifierSubClaim {
            evaluation_point,
            values: combined_mle_values,
        }
    }

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
    pub fn verify_as_subprotocol<U, IdealOverF, IdealOverFFromRef>(
        transcript: &mut impl Transcript,
        proof: Proof<F>,
        num_constraints: usize,
        num_vars: usize,
        ideal_over_f_from_ref: IdealOverFFromRef,
        field_cfg: &F::Config,
    ) -> Result<VerifierSubClaim<F>, IdealCheckError<F, IdealOverF>>
    where
        U: Uair,
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
    use zinc_poly::univariate::{
        dense::DensePolynomial, dynamic::over_field::DynamicPolynomialF, ideal::DegreeOneIdeal,
    };
    use zinc_test_uair::{
        GenerateSingleTypeWitness, TestAirNoMultiplication, TestUairSimpleMultiplication,
    };
    use zinc_transcript::KeccakTranscript;
    use zinc_uair::{
        constraint_counter::count_constraints,
        ideal::{Ideal, IdealCheck},
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
            + Uair<Scalar = DensePolynomial<Int<5>, DEGREE_PLUS_ONE>>,
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

        let verifier_result = IdealCheckProtocol::verify_as_subprotocol::<U, _, _>(
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
            |_ideal_over_ring| IdealOrZero::Zero,
        );
    }
}
