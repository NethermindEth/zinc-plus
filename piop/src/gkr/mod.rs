//! Layered-sumcheck (GKR) primitives.
//!
//! Phase-1 deliverable per
//! [`protocol/src/f2_gkr_plan.md`](../../../protocol/src/f2_gkr_plan.md):
//! a single, special-purpose GKR for the Sklansky parallel-prefix carry
//! tree at width 32 over GF(2^192). The trait surface for a general
//! GKR substrate is intentionally NOT crystallised here yet — Phase 2's
//! integration with the F_2 prover is where the right shape will fall
//! out. For now, the carry-specific implementation in
//! [`sklansky_carry`] is enough to exercise the layered-sumcheck idea
//! end-to-end.
//!
//! See the module docs in [`sklansky_carry`] for the protocol details
//! and the per-stage transition equations.

pub mod sklansky_carry;
