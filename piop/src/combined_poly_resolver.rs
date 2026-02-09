mod folder;

use derive_more::{Display, From};
use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::marker::PhantomData;
use thiserror::Error;
use zinc_poly::EvaluatablePolynomial;
use zinc_poly::mle::dense::CollectDenseMleWithZero;
use zinc_poly::utils::{ArithErrors, build_eq_x_r_inner};
use zinc_uair::ideal::DummyIdeal;
use zinc_utils::{cfg_iter, field, powers};

use crypto_primitives::{DenseRowMatrix, FromPrimitiveWithConfig, Semiring};
use zinc_poly::{
    mle::DenseMultilinearExtension, univariate::dynamic::over_field::DynamicPolynomialF,
};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_uair::Uair;
use zinc_utils::projectable_to_field::ProjectableToField;
use zinc_utils::{cfg_into_iter, inner_transparent_field::InnerTransparentField};

use crate::combined_poly_resolver::folder::ConstraintFolder;
use crate::sumcheck::MLSumcheck;

pub struct CombinedPolyResolver<F: InnerTransparentField>(PhantomData<F>);

impl<F: InnerTransparentField + FromPrimitiveWithConfig> CombinedPolyResolver<F> {
    #[allow(clippy::arithmetic_side_effects)]
    pub fn prove_as_subprotocol<R: Semiring + 'static, U: Uair<R>>(
        transcript: &mut impl Transcript,
        trace_matrix: &[DenseMultilinearExtension<DynamicPolynomialF<F>>],
        evaluation_point: &[F],
        num_constraints: usize,
        num_vars: usize,
        max_degree: usize,

        field_cfg: &F::Config,
    ) -> Result<(), CombinedPolyResolverError>
    where
        F::Inner: ConstTranscribable + Zero,
    {
        let projecting_element: F = transcript.get_field_challenge(field_cfg);

        let zero = F::zero_with_cfg(field_cfg);

        let up: Vec<DenseMultilinearExtension<F::Inner>> = cfg_iter!(trace_matrix)
            .map(|column| {
                cfg_iter!(column)
                    .map(|coeff| {
                        coeff
                            .evaluate_at_point(&projecting_element)
                            .expect("should be fine")
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

        let (sumcheck_proof, sumcheck_prover_state) = MLSumcheck::prove_as_subprotocol(
            transcript,
            vec![last_row_selector, eq_r],
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
                    &mle_values[2..num_constraints + 2],
                    &mle_values[num_constraints + 2..],
                    |x| F::one_with_cfg(field_cfg),
                    |x, y| Some(F::one_with_cfg(field_cfg)),
                    |x| DummyIdeal,
                );

                folder.folded_constraints * selector * eq_r
            },
            field_cfg,
        );

        Ok(())
    }
}

#[derive(Debug, Error, From)]
pub enum CombinedPolyResolverError {
    #[error("failed to build eq_r: {0}")]
    EqrError(ArithErrors),
}
