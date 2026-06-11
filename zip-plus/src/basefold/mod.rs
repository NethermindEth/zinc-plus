//! BaseFold-style limbwise-folding IOPP over the integers ("Zip++").
//!
//! This module implements the polylogarithmic-verifier variant of Zip+:
//! a FRI/BaseFold-style folding IOPP for multilinear evaluation claims over
//! the integers, where the per-round bitsize blowup of naive integer folding
//! is avoided by re-decomposing the folded vector into balanced base-`2^p`
//! limbs each round and folding all limb halves with fresh independent
//! challenges. Projected evaluation claims (values in `F_{q0}` for a
//! transcript-sampled prime `q0`) are tracked linearly through the rounds;
//! no sumcheck is run.
//!
//! Components:
//! - [`limbs`]: balanced base-`2^p` digit decomposition / recombination;
//! - [`chain`]: the radix-2 *foldable IPRS chain* (full-depth lifted FFT,
//!   one butterfly level per fold round, lifted-Vandermonde base case);
//! - [`iopp`]: the interactive prover/verifier (folding rounds, claim
//!   tracking, cross-round consistency checks in cleared-denominator
//!   integer arithmetic).
//!
//! Design ledger: `documentation/zip-plus-basefold-design.md`. The protocol
//! and its analysis follow the working draft *"Zip++: Hash-Based IOPPs over
//! the Integers with Polylogarithmic Verification via Limbwise Folding"*.

pub mod chain;
pub mod compiled;
pub mod iopp;
pub mod limbs;

/// Errors of the basefold IOPP components.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BasefoldError {
    /// Digit width `p` does not fit the digit integer type.
    DigitWidth { p: u32, capacity: u32 },
    /// A value does not fit `num_limbs` balanced base-`2^p` digits.
    ValueTooWide { num_limbs: usize, p: u32 },
    /// Invalid or inconsistent chain / protocol parameters.
    InvalidParams(String),
    /// A tracked-claim equation failed at the given round.
    ClaimCheckFailed { round: usize },
    /// A cross-round consistency check failed.
    ConsistencyCheckFailed { round: usize, position: usize },
    /// A Merkle opening failed to verify against its round root.
    MerkleCheckFailed { round: usize, query: usize },
    /// The cleartext tail failed its checks.
    TailCheckFailed(String),
}

impl core::fmt::Display for BasefoldError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DigitWidth { p, capacity } => {
                write!(f, "digit width p={p} exceeds capacity {capacity} bits")
            }
            Self::ValueTooWide { num_limbs, p } => {
                write!(f, "value not representable in {num_limbs} balanced base-2^{p} digits")
            }
            Self::InvalidParams(msg) => write!(f, "invalid basefold parameters: {msg}"),
            Self::ClaimCheckFailed { round } => {
                write!(f, "tracked-claim equation failed at round {round}")
            }
            Self::ConsistencyCheckFailed { round, position } => {
                write!(
                    f,
                    "cross-round consistency check failed at round {round}, position {position}"
                )
            }
            Self::MerkleCheckFailed { round, query } => {
                write!(
                    f,
                    "merkle opening verification failed at round {round}, query {query}"
                )
            }
            Self::TailCheckFailed(msg) => write!(f, "cleartext tail check failed: {msg}"),
        }
    }
}

impl std::error::Error for BasefoldError {}
