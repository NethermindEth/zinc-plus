pub mod dense;
pub mod mle;

use ark_std::string::String;
use displaydoc::Display;
use thiserror::Error;

extern crate alloc;

pub type RefCounter<T> = alloc::sync::Arc<T>;

/// A `enum` specifying the possible failure modes of the arithmetics.
#[derive(Display, Debug, Error)]
pub enum ArithErrors {
    /// Invalid parameters: {0}
    InvalidParameters(String),
    /// Should not arrive to this point
    ShouldNotArrive,
    /// An error during (de)serialization: {0}
    SerializationErrors(ark_serialize::SerializationError),
}

impl From<ark_serialize::SerializationError> for ArithErrors {
    fn from(e: ark_serialize::SerializationError) -> Self {
        Self::SerializationErrors(e)
    }
}
