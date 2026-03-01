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
    /// Check if `up_evals` and `down_evals` vectors have the expected
    /// lengths.
    pub fn validate_evaluation_sizes(
        &self,
        num_up_cols: usize,
        num_down_cols: usize,
    ) -> Result<(), CombinedPolyResolverError<F>> {
        if self.up_evals.len() != num_up_cols {
            return Err(CombinedPolyResolverError::WrongUpEvalsNumber {
                got: self.up_evals.len(),
                expected: num_up_cols,
            });
        }

        if self.down_evals.len() != num_down_cols {
            return Err(CombinedPolyResolverError::WrongDownEvalsNumber {
                got: self.down_evals.len(),
                expected: num_down_cols,
            });
        }

        Ok(())
    }
}

pub struct ProverState<F: PrimeField> {
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

/// Intermediate state from CPR's pre-sumcheck phase.
///
/// Contains the sumcheck degree group (degree, MLEs, combination function)
/// plus metadata needed for finalization after the sumcheck completes.
pub struct CprSumcheckGroup<F: PrimeField> {
    /// The degree for this sumcheck group: `max_degree + 2`.
    pub degree: usize,
    /// The MLEs: `[selector, eq_r, up_cols..., down_cols...]`.
    pub mles: Vec<DenseMultilinearExtension<F::Inner>>,
    /// The combination function (captures α powers, constraint evaluation, etc.).
    pub comb_fn: Box<dyn Fn(&[F]) -> F + Send + Sync>,
    /// Number of trace columns (needed to split up/down evals after sumcheck).
    pub num_cols: usize,
}

/// Pre-sumcheck verification state for CPR.
///
/// Holds data computed before the sumcheck that is needed after
/// the subclaim is generated.
pub struct CprVerifierPreSumcheck<F: PrimeField> {
    /// The α-folding challenge powers.
    pub folding_challenge_powers: Vec<F>,
    /// The IC evaluation point (needed for eq_eval).
    pub ic_evaluation_point: Vec<F>,
}
