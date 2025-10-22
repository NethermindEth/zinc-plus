pub mod code;
pub mod pcs;
pub mod pcs_transcript;
pub mod traits;
pub mod utils;

pub mod field;
pub mod merkle;
pub mod merkle_poc;
pub mod poly;
pub mod primality;
pub mod transcript;

use ark_std::string::String;
use crypto_primitives::FieldError;
use thiserror::Error;

#[derive(Clone, Debug, PartialEq, Error)]
pub enum ZipError {
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
    #[error("Error during polynomial evaluation: {0}")]
    PolynomialEvaluationError(poly::EvaluationError),
}

impl From<poly::EvaluationError> for ZipError {
    fn from(err: poly::EvaluationError) -> Self {
        ZipError::PolynomialEvaluationError(err)
    }
}

impl From<FieldError> for ZipError {
    fn from(err: FieldError) -> Self {
        ZipError::InvalidSnark(format!("Field error: {err}"))
    }
}
