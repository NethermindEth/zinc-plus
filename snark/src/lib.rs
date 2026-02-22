//! End-to-end Zinc+ SNARK combining the PIOP pipeline with the Zip+ PCS.
//!
//! The pipeline for a UAIR over `BinaryPoly<DEGREE_PLUS_ONE>`:
//!
//! 1. **Witness generation** — produce a trace as MLEs of `BinaryPoly`.
//! 2. **PCS commit** — encode the trace columns with a linear code and
//!    build a Merkle commitment (batched Zip+).
//! 3. **Ideal check** (PIOP step 1) — project the trace into a random
//!    prime field and verify ideal-membership constraints.
//! 4. **Combined polynomial resolver** (PIOP step 2) — fold the
//!    multilinear constraint polynomials via sumcheck.
//! 5. **PCS evaluate + test** — open the committed MLEs at the
//!    evaluation points produced by the PIOP.
//! 6. **Verification** — the verifier replays the Fiat-Shamir transcript.

#![allow(clippy::arithmetic_side_effects)]

pub mod pipeline;

