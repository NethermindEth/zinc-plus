use std::collections::HashMap;

use crypto_primitives::{
    Field, FromWithConfig, Semiring, boolean::Boolean, crypto_bigint_int::Int,
};
use zinc_poly::{
    CoefficientProjectable, Polynomial,
    mle::DenseMultilinearExtension,
    univariate::{dense::DensePolynomial, dynamic::over_field::DynamicPolynomialF},
};
use zinc_transcript::traits::ConstTranscribable;
use zinc_utils::{
    inner_transparent_field::InnerTransparentField, projectable_to_field::ProjectableToField,
};

pub trait IdealCheckTypes<R: Semiring, const DEGREE_PLUS_ONE: usize> {
    type Witness: Polynomial<R>
        + Semiring
        + CoefficientProjectable<R, DEGREE_PLUS_ONE>
        + ProjectableToField<Self::F>
        + ConstTranscribable
        + Send
        + Sync
        + 'static;

    type F: InnerTransparentField + FromWithConfig<R> + Send + Sync + 'static;
}

#[derive(Clone, Debug)]
pub struct Proof<
    R: Semiring,
    IcTypes: IdealCheckTypes<R, DEGREE_PLUS_ONE>,
    const DEGREE_PLUS_ONE: usize,
> {
    pub combined_mle_values: Vec<DynamicPolynomialF<IcTypes::F>>,
}

#[derive(Clone, Debug)]
pub struct ProverState<
    R: Semiring,
    IcTypes: IdealCheckTypes<R, DEGREE_PLUS_ONE>,
    const DEGREE_PLUS_ONE: usize,
> {
    pub evaluation_point: Vec<IcTypes::F>,
    pub combined_mles: Vec<Vec<DenseMultilinearExtension<<IcTypes::F as Field>::Inner>>>,
    pub projected_scalars: HashMap<IcTypes::Witness, DynamicPolynomialF<IcTypes::F>>,
}

pub struct VerifierSubClaim<
    R: Semiring,
    IcTypes: IdealCheckTypes<R, DEGREE_PLUS_ONE>,
    const DEGREE_PLUS_ONE: usize,
> {
    pub evaluation_point: Vec<IcTypes::F>,
    pub values: Vec<DynamicPolynomialF<IcTypes::F>>,
    pub coefficient_projecting_element: IcTypes::F,
}
