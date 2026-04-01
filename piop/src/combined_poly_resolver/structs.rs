use crypto_primitives::PrimeField;

use crate::combined_poly_resolver::CombinedPolyResolverError;

/// The proof type of the combined polynomial resolver subprotocol.
///
/// Note: the sumcheck proof now lives at the protocol
/// level as part of `MultiDegreeSumcheckProof`.
#[derive(Debug, Clone)]
pub struct Proof<F: PrimeField> {
    /// The evaluation of the projected trace columns MLEs at the shared point.
    pub up_evals: Vec<F>,
    /// The evaluations of the shifted projected trace columns MLEs at the
    /// shared point.
    pub down_evals: Vec<F>,
}

impl<F: PrimeField> Proof<F> {
    /// Check if `up_evals` and `down_evals` vectors have the expected lengths.
    pub fn validate_evaluation_sizes(
        &self,
        num_cols: usize,
        num_down_cols: usize,
    ) -> Result<(), CombinedPolyResolverError<F>> {
        if self.up_evals.len() != num_cols {
            return Err(CombinedPolyResolverError::WrongUpEvalsNumber {
                got: self.up_evals.len(),
                expected: num_cols,
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
    /// The shared evaluation point yielded by the multi-degree sumcheck.
    pub evaluation_point: Vec<F>,
}

/// Ancillary data produced by `prepare_sumcheck_group` and consumed by
/// `finalize_prover`. Holds everything needed to extract `up_evals` /
/// `down_evals` after the shared sumcheck completes.
pub struct CprProverAncillary {
    /// Number of trace (up) columns — used to split the flat evals vec.
    pub num_cols: usize,
    /// Number of shifted (down) columns.
    pub num_down_cols: usize,
    /// Number of variables — used to index the last challenge.
    pub num_vars: usize,
}

/// Ancillary data produced by `prepare_verifier` and consumed by
/// `finalize_verifier`. Holds state that bridges the pre-sumcheck and
/// post-sumcheck halves of the CPR verifier.
pub struct CprVerifierAncillary<F: PrimeField> {
    /// Powers of the folding challenge α: [1, α, α², ..., α^{k-1}].
    pub folding_challenge_powers: Vec<F>,
    /// Evaluation point from the ideal check subclaim (for eq_r computation).
    pub ic_evaluation_point: Vec<F>,
    /// Number of variables (for selector computation).
    pub num_vars: usize,
}

/// The claim that is left to be proven after the combined polynomial resolver
/// verifier has succeeded. It is several evaluation claims about the trace
/// columns and the shifted trace columns at the same evaluation point.
#[derive(Clone, Debug)]
pub struct VerifierSubclaim<F: PrimeField> {
    /// Evaluation point for the claims.
    pub evaluation_point: Vec<F>,
    /// Evaluation claims about the trace columns.
    pub up_evals: Vec<F>,
    /// Evaluation claims about the shifted trace columns.
    pub down_evals: Vec<F>,
}
