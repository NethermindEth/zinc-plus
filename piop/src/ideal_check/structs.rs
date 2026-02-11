use std::collections::HashMap;

use crypto_primitives::{Field, FromWithConfig, boolean::Boolean};
use zinc_poly::{
    mle::DenseMultilinearExtension,
    univariate::{binary::BinaryPoly, dynamic::over_field::DynamicPolynomialF},
};
use zinc_utils::inner_transparent_field::InnerTransparentField;

pub trait IdealCheckTypes {
    type F: InnerTransparentField + FromWithConfig<Boolean> + Send + Sync + 'static;
}

#[derive(Clone, Debug)]
pub struct Proof<IcTypes: IdealCheckTypes, const DEGREE_PLUS_ONE: usize> {
    pub combined_mle_values: Vec<DynamicPolynomialF<IcTypes::F>>,
}

#[derive(Clone, Debug)]
pub struct ProverState<IcTypes: IdealCheckTypes, const DEGREE_PLUS_ONE: usize> {
    pub evaluation_point: Vec<IcTypes::F>,
    pub combined_mles: Vec<Vec<DenseMultilinearExtension<<IcTypes::F as Field>::Inner>>>,
    pub projected_scalars: HashMap<BinaryPoly<DEGREE_PLUS_ONE>, DynamicPolynomialF<IcTypes::F>>,
}

pub struct VerifierSubClaim<IcTypes: IdealCheckTypes> {
    pub evaluation_point: Vec<IcTypes::F>,
    pub values: Vec<DynamicPolynomialF<IcTypes::F>>,
    pub coefficient_projecting_element: IcTypes::F,
}
