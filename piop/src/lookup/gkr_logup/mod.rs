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

pub use protocol::{
    BinaryPolyLookupInstance, IntLookupInstance, combine_chunks, combine_chunks_word,
    compute_binary_poly_lift, compute_binary_poly_lifts, prove_group, prove_group_prescribed,
    prove_group_word, verify_group,
};
pub use structs::{
    BatchedGkrFractionProof, BatchedGkrLayerProof, GkrFractionProof, GkrLayerProof,
    GkrLogupError, GkrLogupGroupMeta, GkrLogupGroupProof, GkrLogupGroupSubclaim,
    GkrLogupLookupProof, dump_size_breakdown,
};
