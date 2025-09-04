use crypto_primitives::IntRing;

pub mod code;
pub mod code_raa;
mod macros;
pub mod pcs;
pub mod pcs_transcript;
pub mod traits;
pub mod utils;

mod const_helpers;
pub mod field;
mod poly;
mod poly_f;
pub mod poly_z;
#[cfg(test)]
mod tests;
pub mod transcript;

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
