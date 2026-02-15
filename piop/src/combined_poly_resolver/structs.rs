use crypto_primitives::PrimeField;
use zinc_poly::mle::DenseMultilinearExtension;

use crate::{combined_poly_resolver::CombinedPolyResolverError, sumcheck::SumcheckProof};

/// The proof type of the combined polynomial resolver
/// subprotocol.
#[derive(Debug, Clone)]
pub struct Proof<F: PrimeField> {
    /// A proof of the inner sumcheck subprotocol used
    /// for resolving combined polynomial evaluation claims.
    pub sumcheck_proof: SumcheckProof<F>,
    /// The evaluation of the projected trace columns MLEs.
    pub up_evals: Vec<F>,
    /// The evaluations of the shifted projected trace columns MLEs.
    pub down_evals: Vec<F>,
}

impl<F: PrimeField> Proof<F> {
    /// Check if `up_evals` and `down_evals` vectors
    /// has the length `num_cols`.
    pub fn validate_evaluation_sizes(
        &self,
        num_cols: usize,
    ) -> Result<(), CombinedPolyResolverError<F>> {
        if self.up_evals.len() != num_cols {
            return Err(CombinedPolyResolverError::WrongUpEvalsNumber {
                got: self.up_evals.len(),
                expected: num_cols,
            });
        }

        if self.down_evals.len() != num_cols {
            return Err(CombinedPolyResolverError::WrongDownEvalsNumber {
                got: self.down_evals.len(),
                expected: num_cols,
            });
        }

        Ok(())
    }
}

/// Expensive data computed in the course
/// of the combined polynomial resolver subprotocol
/// that is passed further to the next subprotocol.
pub struct ProverState<F: PrimeField> {
    /// The projected trace columns MLEs.
    pub up: Vec<DenseMultilinearExtension<F::Inner>>,
    /// The projected shifted trace columns MLEs.
    pub down: Vec<DenseMultilinearExtension<F::Inner>>,
    /// The evaluation point yielded by the sumcheck
    /// subprotocol.
    pub evaluation_point: Vec<F>,
}

/// The claim that is left to be proven
/// after the combined polynomial resolver
/// verifier has succeeded verifying.
/// In this case, it is several evaluation claims
/// about the trace columns and the shifted trace columns
/// on the same evaluation point.
#[derive(Clone, Debug)]
pub struct VerifierSubclaim<F: PrimeField> {
    /// Evaluation point for the claims.
    pub evaluation_point: Vec<F>,
    /// Evaluation claims about the trace columns.
    pub up_evals: Vec<F>,
    /// Evaluation claims about the shifted trace columns.
    pub down_evals: Vec<F>,
}
