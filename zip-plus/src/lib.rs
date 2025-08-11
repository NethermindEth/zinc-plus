use crypto_primitives::IntRing;

pub mod code;
pub mod code_raa;
pub mod pcs;
pub mod pcs_transcript;
pub mod utils;
pub mod traits;
mod macros;

#[cfg(test)]
mod tests;
pub mod poly_z;
mod poly;
pub mod transcript;
mod conversion;
pub mod field;
mod const_helpers;
mod poly_f;

use ark_std::string::String;
use thiserror::Error;
#[derive(Clone, Debug, PartialEq, Error)]
pub enum Error {
    #[error("Invalid PCS param: {0}")]
    InvalidPcsParam(String),
    #[error("Invalid commitment opening: {0}")]
    InvalidPcsOpen(String),
    #[error("Bad Snark: {0}")]
    InvalidSnark(String),
    #[error("Serialization Error: {0}")]
    Serialization(String),
    #[error("Transcript failure: {1}")]
    Transcript(ark_std::io::ErrorKind, String),
}

pub trait ZipTypes: Send + Sync {
    /// Width of elements in witness/polynomial evaluations on hypercube
    type N: IntRing;

    /// Width of elements in the encoding matrices
    type L: IntRing + for<'a> From<&'a Self::N>;

    /// Width of elements in the code
    type K: IntRing + for<'a> From<&'a Self::N> + for<'a> From<&'a Self::L>;

    /// Width of elements in linear combination of code rows
    type M: IntRing
        + for<'a> From<&'a Self::N>
        + for<'a> From<&'a Self::L>
        + for<'a> From<&'a Self::K>
        + for<'a> From<&'a Self::M>;
}
