//! Shift-by-constant sumcheck reduction.
//!
//! Given evaluation claims about shifted MLEs — i.e. claims of the form
//!
//!   MLE\[shift_c(v)\](r) = a
//!
//! where shift_c(v)\[i\] = v\[i − c\] for i ≥ c (and 0 otherwise) — this
//! module reduces them to evaluation claims about the **unshifted** MLEs
//! at a fresh random point via sumcheck.
//!
//! # Batched reduction
//!
//! Multiple shift claims (possibly with different shift amounts and
//! different source polynomials) are batched into a single sumcheck
//! instance using random linear combination.  The proof size is
//! independent of the number of claims.
//!
//! After the sumcheck, the verifier needs:
//! 1. For each claim `i`, the value `S_{c_i}(s, r_i)` — computable
//!    in O(m + c · log c) by [`eval_shift_predicate`].
//! 2. For each claim `i`, the value `MLE[v_i](s)` — deferred to the
//!    PCS as an opening claim about the **unshifted** committed column.
//!
//! # References
//!
//! Adapted from the `shift-sumcheck-bench` repository.

mod predicate;
mod prover;
mod structs;
mod verifier;

pub use predicate::*;
pub use prover::*;
pub use structs::*;
pub use verifier::*;
