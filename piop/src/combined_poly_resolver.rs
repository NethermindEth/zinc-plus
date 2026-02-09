use std::marker::PhantomData;

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
        trace_matrix: &DenseRowMatrix<DynamicPolynomialF<F>>,
        field_cfg: &F::Config,
    ) where
        F::Inner: ConstTranscribable,
    {
        let projecting_element: F = transcript.get_field_challenge(field_cfg);

        // let up: Vec<DenseMultilinearExtension<F::Inner>> =
        //     cfg_into_iter!(0..trace_matrix.num_cols())
        //         .map(|col_idx| {
        //             cfg_into_iter!(0..trace_matrix.num_rows())
        //                 .map(|row_idx| (trace_matrix[row_idx][col_idx]))
        //                 .collect()
        //         })
        //         .collect();
    }
}
