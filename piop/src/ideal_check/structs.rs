use std::collections::HashMap;

use crypto_primitives::{Field, FromWithConfig, Semiring};
use zinc_poly::{
    CoefficientProjectable, mle::DenseMultilinearExtension,
    univariate::dynamic::over_field::DynamicPolynomialF,
};
use zinc_transcript::traits::ConstTranscribable;
use zinc_utils::{
    inner_transparent_field::InnerTransparentField, projectable_to_field::ProjectableToField,
};

pub trait IdealCheckTypes<const DEGREE_PLUS_ONE: usize> {
    type WitnessCoeff;
    type Witness: Semiring
        + CoefficientProjectable<Self::WitnessCoeff, DEGREE_PLUS_ONE>
        + ProjectableToField<Self::F>
        + ConstTranscribable
        + Send
        + Sync
        + 'static;

    type F: InnerTransparentField + FromWithConfig<Self::WitnessCoeff> + Send + Sync + 'static;
}

#[derive(Clone, Debug)]
pub struct Proof<IcTypes: IdealCheckTypes<DEGREE_PLUS_ONE>, const DEGREE_PLUS_ONE: usize> {
    pub combined_mle_values: Vec<DynamicPolynomialF<IcTypes::F>>,
}

#[derive(Clone, Debug)]
pub struct ProverState<IcTypes: IdealCheckTypes<DEGREE_PLUS_ONE>, const DEGREE_PLUS_ONE: usize> {
    pub evaluation_point: Vec<IcTypes::F>,
    pub combined_mles: Vec<Vec<DenseMultilinearExtension<<IcTypes::F as Field>::Inner>>>,
    pub projected_scalars: HashMap<IcTypes::Witness, DynamicPolynomialF<IcTypes::F>>,
}

pub struct VerifierSubClaim<IcTypes: IdealCheckTypes<DEGREE_PLUS_ONE>, const DEGREE_PLUS_ONE: usize>
{
    pub evaluation_point: Vec<IcTypes::F>,
    pub values: Vec<DynamicPolynomialF<IcTypes::F>>,
}
