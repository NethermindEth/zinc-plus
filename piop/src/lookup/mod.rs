//! Lookup argument for the Zinc+ PIOP.
//!
//! `structs` carries the legacy stub types from main (different
//! design, currently unused). `gkr_logup` is the new GKR-LogUp module
//! with polynomial-valued chunk lifts (chunks neither sent nor
//! committed; bound via the parent column's PCS commitment).
pub mod gkr_logup;
pub mod structs;

pub use structs::*;
