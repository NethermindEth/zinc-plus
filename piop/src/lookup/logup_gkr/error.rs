//! Errors for the logup-GKR subprotocol.

use thiserror::Error;

use crate::sumcheck::SumCheckError;

/// Errors returned by the logup-GKR verifier.
#[derive(Debug, Error)]
pub enum LogupGkrError<F> {
    /// Inner sumcheck at round `i` rejected.
    #[error("sumcheck failed at layer {layer}: {source}")]
    Sumcheck {
        layer: usize,
        #[source]
        source: SumCheckError<F>,
    },
    /// Prover claimed a cumulative sum that does not match the root
    /// layer's numerator / denominator.
    #[error("root cumulative sum mismatch")]
    RootSumMismatch,
    /// The denominator is zero at some point used by the verifier (so
    /// the rational is undefined / the argument is vacuous).
    #[error("zero denominator at layer {0}")]
    ZeroDenominator(usize),
    /// The sumcheck's final evaluation does not match the
    /// verifier-reconstructed combination at layer `i`.
    #[error("final-evaluation mismatch at layer {0}")]
    FinalEvalMismatch(usize),
    /// The number of round proofs does not match the expected count.
    #[error("invalid proof shape: expected {expected} round proofs, got {got}")]
    InvalidShape { expected: usize, got: usize },
}
