use std::collections::HashMap;

use crypto_primitives::{FromWithConfig, PrimeField, boolean::Boolean};
use zinc_poly::{
    mle::DenseMultilinearExtension,
    univariate::{binary::BinaryPoly, dynamic::over_field::DynamicPolynomialF},
};
use zinc_utils::inner_transparent_field::InnerTransparentField;

pub trait IdealCheckField:
    InnerTransparentField + FromWithConfig<Boolean> + Send + Sync + 'static
{
}
impl<T> IdealCheckField for T where
    T: InnerTransparentField + FromWithConfig<Boolean> + Send + Sync + 'static
{
}

#[derive(Clone, Debug)]
pub struct Proof<F: PrimeField, const DEGREE_PLUS_ONE: usize> {
    pub combined_mle_values: Vec<DynamicPolynomialF<F>>,
}

#[derive(Clone, Debug)]
pub struct ProverState<F: PrimeField, const DEGREE_PLUS_ONE: usize> {
    pub evaluation_point: Vec<F>,
    pub combined_mles: Vec<Vec<DenseMultilinearExtension<F::Inner>>>,
    pub projected_scalars: HashMap<BinaryPoly<DEGREE_PLUS_ONE>, DynamicPolynomialF<F>>,
}

pub struct VerifierSubClaim<F: PrimeField> {
    pub evaluation_point: Vec<F>,
    pub values: Vec<DynamicPolynomialF<F>>,
    pub coefficient_projecting_element: F,
}
