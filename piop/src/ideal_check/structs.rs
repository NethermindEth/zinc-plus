use crypto_primitives::PrimeField;
use zinc_poly::univariate::dynamic::over_field::DynamicPolynomialF;

#[derive(Clone, Debug)]
pub struct Proof<F: PrimeField> {
    pub combined_mle_values: Vec<DynamicPolynomialF<F>>,
}

#[derive(Clone, Debug)]
pub struct ProverState<F: PrimeField> {
    pub evaluation_point: Vec<F>,
}

#[derive(Clone, Debug)]
pub struct VerifierSubclaim<F: PrimeField> {
    pub evaluation_point: Vec<F>,
    pub values: Vec<DynamicPolynomialF<F>>,
}
