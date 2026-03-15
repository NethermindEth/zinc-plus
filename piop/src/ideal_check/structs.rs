use crypto_primitives::PrimeField;
use zinc_poly::{
    mle::DenseMultilinearExtension, univariate::dynamic::over_field::DynamicPolynomialF,
};

#[derive(Clone, Debug)]
pub struct Proof<F: PrimeField> {
    pub combined_mle_values: Vec<DynamicPolynomialF<F>>,
}

#[derive(Clone, Debug)]
pub struct ProverState<F: PrimeField> {
    pub evaluation_point: Vec<F>,
    pub combined_mles: Vec<Vec<DenseMultilinearExtension<F::Inner>>>,
}

#[derive(Clone, Debug)]
pub struct VerifierSubClaim<F: PrimeField> {
    pub evaluation_point: Vec<F>,
    pub values: Vec<DynamicPolynomialF<F>>,
}
