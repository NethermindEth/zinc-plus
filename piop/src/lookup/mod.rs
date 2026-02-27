//! Lookup argument for the Zinc+ PIOP.
//!
//! This module implements the **lookup argument** that enforces typing
//! constraints on trace columns — i.e., that every entry of a witness
//! vector belongs to a prescribed finite set (a "lookup table").
//!
//! ## Architecture
//!
//! - [`tables`]: Projected lookup table generation (`BitPoly`, `Word`)
//!   and utility functions (batch inversion, multiplicity computation).
//! - [`logup`]: Core LogUp protocol (`prove_as_subprotocol` /
//!   `verify_as_subprotocol`).
//! - [`decomposition`]: Decomposition + LogUp for large tables (e.g.
//!   `BitPoly(32)` → two `BitPoly(16)` sub-tables).
//! - [`structs`]: Proof types, prover state, verifier sub-claims, and
//!   error definitions.
//!
//! ## Protocol overview
//!
//! The lookup step receives projected trace columns over F_q and
//! verifies that each column's entries belong to the appropriate
//! projected lookup table. For small tables, the core LogUp protocol
//! is used directly. For large tables (e.g. 2^32 entries), the
//! Lasso-style decomposition reduces to two sub-table lookups of
//! size 2^16 each.
//!
//! All auxiliary vectors (multiplicities, inverse vectors,
//! decomposition chunks) are sent **in the clear** — no polynomial
//! commitment is needed.

pub mod batched_decomposition;
pub mod decomposition;
pub mod logup;
pub mod structs;
pub mod tables;

pub use batched_decomposition::BatchedDecompLogupProtocol;
pub use decomposition::DecompLogupProtocol;
pub use logup::LogupProtocol;
pub use structs::*;
