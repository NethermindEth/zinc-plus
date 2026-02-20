//! Zinc+ SNARK composition layer.
//!
//! This crate composes the PIOP (Polynomial IOP) layer with the PCS
//! (Polynomial Commitment Scheme) layer to form an end-to-end SNARK.
//!
//! # Architecture
//!
//! The Zinc+ SNARK proves that a given witness satisfies a UAIR
//! (Universal Algebraic Intermediate Representation) constraint system.
//! It does so by:
//!
//! 1. **Committing** all trace column polynomials via the batched Zip+ PCS
//!    (one shared Merkle tree per type batch).
//! 2. **Running the PIOP** (ideal check + combined polynomial resolver)
//!    to reduce constraint satisfaction to polynomial evaluation claims.
//! 3. **Opening** the committed polynomials at the PIOP's evaluation point
//!    using the batched Zip+ PCS, proving the evaluation claims.
//!
//! Both prover and verifier share a Fiat-Shamir transcript that absorbs
//! all commitment data before producing PIOP challenges, ensuring soundness.

pub mod structs;

pub use structs::*;

use std::fmt;

use crypto_primitives::PrimeField;
use zinc_piop::combined_poly_resolver::CombinedPolyResolverError;
use zip_plus::ZipError;

/// Errors produced by the Zinc+ SNARK composition layer.
#[derive(Debug)]
pub enum ZincSnarkError<F: PrimeField> {
    /// Error during the ideal check subprotocol.
    IdealCheckError(String),
    /// Error during the combined polynomial resolver subprotocol.
    ResolverError(CombinedPolyResolverError<F>),
    /// Scalar projection error during F[X] → F mapping.
    ScalarProjectionError(String),
    /// PCS error for a specific column.
    PcsError {
        /// Column index (up columns first, then down columns).
        column: usize,
        /// The underlying PCS error.
        source: ZipError,
    },
    /// Invalid input to prove or verify.
    InvalidInput(String),
}

impl<F: PrimeField> fmt::Display for ZincSnarkError<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IdealCheckError(e) => write!(f, "ideal check error: {e}"),
            Self::ResolverError(e) => write!(f, "combined poly resolver error: {e}"),
            Self::ScalarProjectionError(msg) => write!(f, "scalar projection error: {msg}"),
            Self::PcsError { column, source } => {
                write!(f, "PCS error for column {column}: {source}")
            }
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
        }
    }
}

impl<F: PrimeField> std::error::Error for ZincSnarkError<F> {}

impl<F: PrimeField> From<CombinedPolyResolverError<F>> for ZincSnarkError<F> {
    fn from(e: CombinedPolyResolverError<F>) -> Self {
        Self::ResolverError(e)
    }
}
