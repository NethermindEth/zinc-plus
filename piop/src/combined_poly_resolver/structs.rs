use crypto_primitives::PrimeField;

use crate::sumcheck::{self, SumcheckProof};

#[derive(Debug, Clone)]
pub struct Proof<F: PrimeField> {
    pub sumcheck_proof: SumcheckProof<F>,
    pub up_evals: Vec<F>,
    pub down_evals: Vec<F>,
}

pub struct ProverState<F: PrimeField> {
    pub sumcheck_prover_state: sumcheck::prover::ProverState<F>,
}
