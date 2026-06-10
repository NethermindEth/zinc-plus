//! # The inner ZK-IOP for mask consistency (`R_lift`)
//!
//! Foundation for replacing the transparent v0 seed revelation
//! ([`super::mask::MaskConsistencyProof::Transparent`]) with the
//! zero-knowledge argument of paper §5. The relation
//!
//! ```text
//! (lift)       <G_l, s_i> = X_{i,l} + p * Y_{i,l}        over Z
//! (aggregate)  rem_l      = X_{0,l} + sum_j rho_j X_{j,l} over Z
//! (ranges)     s_i, X in [0, 2^(bits(p)-1)),  Y in [0, 2^y_bits)
//! ```
//!
//! is embedded into a prime field `F_p'` with `p'` large enough that no
//! range-checked value wraps (Lemma 5.2 of the note). The key structural
//! fact (Lemma 5.3): after substituting each value by its bit decomposition,
//! the entire relation is **linear in the bits** — the only non-linear
//! constraint left is booleanity, which a Ligero-style quadratic test
//! discharges uniformly over the whole witness.
//!
//! Layering:
//! - [`field`] — selection of an FFT-friendly inner prime `p'` and a radix-2
//!   NTT over the runtime-modulus Montgomery field (the Ligero encoder and
//!   the test-polynomial arithmetic of the next increment run on these);
//! - [`constraints`] — the `R_lift`-to-bits compiler: witness layout, sparse
//!   linear constraints over `F_p'`, and the honest-witness builder;
//! - `ligero` (next increment) — the ZK-Ligero IOP over `F_p'` (interleaved
//!   RS commitment on a coset domain, blinded linear + booleanity tests,
//!   column spot checks) consuming the two layers above.
//!
//! Window convention: thanks to `p` being the *least* prime above a power of
//! two and to the seed/lift window sampling at commit time
//! (`MaskSeeds::sample`, the resampling loop in `ZkZip::commit_single`),
//! every range is a power-of-two window and range checks are pure bit
//! decompositions — no comparison against an arbitrary modulus is ever
//! needed inside the IOP.

pub mod constraints;
pub mod field;
