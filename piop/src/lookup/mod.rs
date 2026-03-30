//! Lookup argument for the Zinc+ PIOP.
pub mod structs;

pub use structs::*;

// Re-export spec types from zinc-uair for convenience.
pub use zinc_uair::{LookupColumnSpec, LookupTableType};
