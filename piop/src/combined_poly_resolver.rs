use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::marker::PhantomData;
use zinc_poly::EvaluatablePolynomial;
use zinc_poly::mle::dense::CollectDenseMleWithZero;
use zinc_poly::utils::build_eq_x_r_inner;
use zinc_utils::{cfg_iter, field};

use crypto_primitives::{DenseRowMatrix, Semiring};
use zinc_poly::{
    mle::DenseMultilinearExtension, univariate::dynamic::over_field::DynamicPolynomialF,
};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_uair::Uair;
use zinc_utils::projectable_to_field::ProjectableToField;
use zinc_utils::{cfg_into_iter, inner_transparent_field::InnerTransparentField};

pub struct CombinedPolyResolver<F: InnerTransparentField>(PhantomData<F>);

impl<F: InnerTransparentField> CombinedPolyResolver<F> {
    pub fn prove_as_subprotocol<R: Semiring + 'static, U: Uair<R>>(
        transcript: &mut impl Transcript,
        trace_matrix: &[DenseMultilinearExtension<DynamicPolynomialF<F>>],
        evaluation_point: &[F],
        field_cfg: &F::Config,
    ) where
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

        let eq_r = build_eq_x_r_inner(evaluation_point, field_cfg);

        let folding_challenge: F = transcript.get_field_challenge(field_cfg);
    }
}
