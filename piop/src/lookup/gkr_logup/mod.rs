//! GKR-LogUp lookup with polynomial-valued chunk lifts.
//!
//! Re-exports the GKR primitives and proof types. The protocol
//! integration layer (`protocol.rs`) and the test UAIR live in
//! follow-on commits; this commit only lands the primitives so the
//! cumulative diff stays reviewable.

pub mod gkr;
pub mod structs;
pub mod tables;

pub use structs::{
    BatchedGkrFractionProof, BatchedGkrLayerProof, GkrFractionProof, GkrLayerProof,
    GkrLogupError,
};
