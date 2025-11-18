pub mod code;
pub mod pcs;
pub mod pcs_transcript;
pub mod utils;

pub mod merkle;

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
    PolynomialEvaluationError(zinc_poly::EvaluationError),
    #[error("Error during inner product computation: {0}")]
    InnerProductError(zinc_utils::inner_product::InnerProductError),
}

impl From<zinc_poly::EvaluationError> for ZipError {
    fn from(err: zinc_poly::EvaluationError) -> Self {
        ZipError::PolynomialEvaluationError(err)
    }
}

impl From<FieldError> for ZipError {
    fn from(err: FieldError) -> Self {
        ZipError::InvalidSnark(format!("Field error: {err}"))
    }
}

impl From<zinc_utils::inner_product::InnerProductError> for ZipError {
    fn from(err: zinc_utils::inner_product::InnerProductError) -> Self {
        ZipError::InnerProductError(err)
    }
}
