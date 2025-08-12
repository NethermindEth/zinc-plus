// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// Adapted for rings by Nethermind

//! Error module.

use displaydoc::Display;
use thiserror::Error;

/// A `enum` specifying the possible failure modes of the arithmetics.
#[derive(Display, Debug, Error)]
pub enum ArithErrors {
    /// An error during (de)serialization: {0}
    SerializationErrors(ark_serialize::SerializationError),
}

impl From<ark_serialize::SerializationError> for ArithErrors {
    fn from(e: ark_serialize::SerializationError) -> Self {
        Self::SerializationErrors(e)
    }
}
