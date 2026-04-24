//! Standalone logup-GKR subprotocol.
//!
//! Proves `sum_{x in {0,1}^n} N(x) / D(x) = S` for two multilinear
//! extensions `N` (numerator) and `D` (denominator) via a grand-sum
//! GKR circuit. Each layer of the binary tree folds pairs of
//! rationals `(n_0/d_0), (n_1/d_1)` into `(n_0 d_1 + n_1 d_0, d_0 d_1)`
//! and each layer is reduced with one sumcheck of degree 3.
//!
//! This module is the standalone Phase 1 of the logup-GKR port: it
//! takes arbitrary `(N, D)` as input and produces a subclaim
//! `(point, N(point), D(point))` at the leaf level. Wiring the leaves
//! to actual trace columns + multiplicity is done by the caller in a
//! later phase.

pub mod argument;
pub mod circuit;
pub mod error;
pub mod leaves;
pub mod proof;
pub mod prover;
pub mod verifier;

#[cfg(test)]
mod tests;

pub use argument::{
    LookupArgument, LookupArgumentError, LookupArgumentProof, LookupArgumentSubclaim,
};
pub use leaves::{LeafComponentEvals, LookupLeaves, build_lookup_leaves, expected_leaf_evals};

pub use circuit::{GrandSumCircuit, GrandSumLayer};
pub use error::LogupGkrError;
pub use proof::{LogupGkrProof, LogupGkrRoundProof};
pub use prover::{LogupGkrProver, LogupGkrSubclaim};
pub use verifier::LogupGkrVerifier;
