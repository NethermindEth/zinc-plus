//! GKR-LogUp lookup with polynomial-valued chunk lifts.
//!
//! Modules:
//! * `gkr` — fraction tree primitives + layered GKR fractional sumcheck.
//! * `tables` — projected table generators + multiplicity helpers.
//! * `structs` — proof, error, and intermediate types.
//! * `protocol` — top-level prove/verify per lookup group with the
//!   chunks-in-clear polynomial-valued lift design.

pub mod gkr;
pub mod protocol;
pub mod structs;
pub mod tables;

pub use protocol::{prove_group, verify_group, BinaryPolyLookupInstance};
pub use structs::{
    BatchedGkrFractionProof, BatchedGkrLayerProof, GkrFractionProof, GkrLayerProof,
    GkrLogupError, GkrLogupGroupMeta, GkrLogupGroupProof, GkrLogupGroupSubclaim,
    GkrLogupLookupProof,
};
