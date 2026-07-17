//! Lookup argument for the Zinc+ PIOP.
//!
//! `structs` carries the legacy stub types from main (a different,
//! currently-unused decomposed-inverse design). `gkr_logup` is the GKR-LogUp
//! module with polynomial-valued chunk lifts (chunks neither sent nor
//! committed; bound via the parent column's PCS commitment). `booleanity` is
//! the algebraic v·(v-1) approach (a separate multi-degree sumcheck group on
//! main-beta).
pub mod booleanity;
pub mod gkr_logup;
pub mod structs;

pub use structs::*;
