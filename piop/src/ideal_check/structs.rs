use crypto_primitives::{
    ConstIntSemiring, Field, FromWithConfig, PrimeField, Semiring, boolean::Boolean,
};
use std::collections::HashMap;
use zinc_poly::{
    mle::DenseMultilinearExtension,
    univariate::{binary::BinaryPoly, dynamic::over_field::DynamicPolynomialF},
};
use zinc_transcript::traits::ConstTranscribable;
use zinc_utils::{from_ref::FromRef, inner_transparent_field::InnerTransparentField};

pub trait IdealCheckField:
    InnerTransparentField
    + Field<Inner: ConstIntSemiring + ConstTranscribable + FromRef<Self::Inner>>
    + FromWithConfig<Boolean>
    + Send
    + Sync
    + 'static
{
}
impl<T> IdealCheckField for T where
    T: InnerTransparentField
        + Field<Inner: ConstIntSemiring + ConstTranscribable + FromRef<Self::Inner>>
        + FromWithConfig<Boolean>
        + Send
        + Sync
        + 'static
{
}

#[derive(Clone, Debug)]
pub struct Proof<F: IdealCheckField, const DEGREE_PLUS_ONE: usize> {
    pub combined_mle_values: Vec<DynamicPolynomialF<F>>,
}

#[derive(Clone, Debug)]
pub struct ProverState<F: IdealCheckField, const DEGREE_PLUS_ONE: usize> {
    pub evaluation_point: Vec<F>,
    pub combined_mles: Vec<Vec<DenseMultilinearExtension<F::Inner>>>,
    pub trace_matrix: Vec<DenseMultilinearExtension<DynamicPolynomialF<F>>>,
    pub projected_scalars: HashMap<BinaryPoly<DEGREE_PLUS_ONE>, DynamicPolynomialF<F>>,
}

pub struct VerifierSubClaim<R: Semiring, F: PrimeField> {
    pub evaluation_point: Vec<F>,
    pub values: Vec<DynamicPolynomialF<F>>,
    pub projected_scalars: HashMap<R, DynamicPolynomialF<F>>,
}
