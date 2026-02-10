use crypto_primitives::PrimeField;
use zinc_poly::mle::DenseMultilinearExtension;

use crate::sumcheck::{self, SumcheckProof};

#[derive(Debug, Clone)]
pub struct Proof<F: PrimeField> {
    pub sumcheck_proof: SumcheckProof<F>,
    pub up_evals: Vec<F>,
    pub down_evals: Vec<F>,
}

pub struct ProverState<F: PrimeField> {
    pub up: Vec<DenseMultilinearExtension<F::Inner>>,
    pub down: Vec<DenseMultilinearExtension<F::Inner>>,
    pub sumcheck_prover_state: sumcheck::prover::ProverState<F>,
}

pub struct VerifierSubclaim<F: PrimeField> {
    pub evaluation_point: Vec<F>,
    pub up_evals: Vec<F>,
    pub down_evals: Vec<F>,
}
