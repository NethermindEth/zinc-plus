use crypto_primitives::{Field, FromWithConfig, PrimeField, Semiring};
use std::collections::HashMap;
use zinc_poly::{
    mle::DenseMultilinearExtension, univariate::dynamic::over_field::DynamicPolynomialF,
};
use zinc_transcript::traits::ConstTranscribable;
use zinc_utils::{
    inner_transparent_field::InnerTransparentField, projectable_to_field::ProjectableToField,
};

#[derive(Clone, Debug)]
pub struct Proof<F: PrimeField> {
    pub combined_mle_values: Vec<DynamicPolynomialF<F>>,
}

#[derive(Clone, Debug)]
pub struct ProverState<R: Semiring, F: PrimeField> {
    pub evaluation_point: Vec<F>,
    pub combined_mles: Vec<Vec<DenseMultilinearExtension<F::Inner>>>,
    pub trace_matrix: Vec<DenseMultilinearExtension<DynamicPolynomialF<F>>>,
    pub projected_scalars: HashMap<R, DynamicPolynomialF<F>>,
}

pub struct VerifierSubClaim<R: Semiring, F: PrimeField> {
    pub evaluation_point: Vec<F>,
    pub values: Vec<DynamicPolynomialF<F>>,
    pub projected_scalars: HashMap<R, DynamicPolynomialF<F>>,
}
