//! Sklansky parallel-prefix carry tree over F_2 for 32-bit addition.
//!
//! Phase-0 deliverable for the GKR-virtual-columns plan
//! ([`protocol/src/f2_gkr_plan.md`](../../protocol/src/f2_gkr_plan.md)):
//! a reference implementation of the layered circuit that will replace the
//! committed `W_C_*` carry columns in [`crate::sha256_f2`], plus golden
//! vectors against the existing per-bit recurrence
//! [`crate::sha256_f2::binius_carry`].
//!
//! ## Recurrence vs. prefix form
//!
//! The Binius full-adder carry word satisfies the per-bit recurrence
//!
//! ```text
//!   c[i] = (a[i] ∧ b[i]) ⊕ (c[i−1] ∧ (a[i] ⊕ b[i])),   c[-1] = 0,
//! ```
//!
//! which is a linear-depth chain unsuitable for GKR. The standard
//! parallel-prefix reformulation uses per-bit generate / propagate
//! signals
//!
//! ```text
//!   G_i = a[i] ∧ b[i],   P_i = a[i] ⊕ b[i],
//! ```
//!
//! and the associative combine over (G, P) pairs
//!
//! ```text
//!   (G_lo, P_lo) ∘ (G_hi, P_hi) = (G_hi ⊕ P_hi · G_lo,  P_hi · P_lo).
//! ```
//!
//! Folding the 32 per-bit pairs with this operator via the Sklansky tree
//! gives `(G_pref[0..=i], P_pref[0..=i])` at each position `i` in 5
//! stages. The carry-out of bit `i` (= bit `i` of `binius_carry`) is the
//! prefix-G at position `i`.
//!
//! ## Sklansky schedule
//!
//! At stage `k` (k ∈ {0,…,4}) the bit width 32 is partitioned into
//! blocks of size `2·2^k = 2^(k+1)`. Each block has a *source* position
//! (its lower half's top bit, at offset `2^k − 1` from the block base)
//! and `2^k` *target* positions in its upper half. Every target combines
//! the source's `(G, P)` with its own. All combines within a stage are
//! independent — that's the property that makes the depth log₂(32) = 5
//! and gives a uniformly-shaped GKR layer.

/// Bit width of the words we add. SHA-256 uses 32-bit lanes.
pub const WORD_BITS: usize = 32;

/// Number of Sklansky stages for [`WORD_BITS`] = 32.
pub const PREFIX_STAGES: usize = 5;

/// One combine in the Sklansky schedule: the value held at `src` is mixed
/// into the value held at `tgt`. All edges within a single stage are
/// independent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CombineEdge {
    pub src: u8,
    pub tgt: u8,
}

/// Enumerate the combine edges for one Sklansky stage at width
/// [`WORD_BITS`]. Stages are 0-indexed; stage `k` has block size
/// `2^(k+1)` and emits `WORD_BITS / 2 = 16` edges (regardless of `k`).
pub fn sklansky_edges(stage: usize) -> Vec<CombineEdge> {
    assert!(stage < PREFIX_STAGES, "stage out of range");
    let half = 1usize << stage;
    let full = half << 1;
    let mut edges = Vec::with_capacity(WORD_BITS / 2);
    let mut base = 0usize;
    while base + full <= WORD_BITS {
        let src = base + half - 1;
        for j in half..full {
            let tgt = base + j;
            edges.push(CombineEdge {
                src: src as u8,
                tgt: tgt as u8,
            });
        }
        base += full;
    }
    edges
}

/// Compute the Binius carry word `c` for `a + b mod 2^32` via the
/// Sklansky parallel-prefix tree. Bit `i` of the output equals the
/// carry-out of bit `i`, matching [`crate::sha256_f2::binius_carry`].
pub fn prefix_carry(a: u32, b: u32) -> u32 {
    let trace = prefix_carry_trace(a, b);
    trace[PREFIX_STAGES].0
}

/// Run the Sklansky tree and return `(G, P)` after each stage as
/// `WORD_BITS`-wide bit-packed `u32`s. Index 0 is the initial state
/// (the per-bit generates/propagates); index `k+1` is the state after
/// stage `k`. The final `G` word is the carry word; the final `P` word
/// is the all-positions propagate prefix and is mostly useful for
/// cross-checking.
///
/// This is the intermediate-state form that the Phase-1 GKR prover will
/// consume: each entry is one layer's wire values.
pub fn prefix_carry_trace(a: u32, b: u32) -> [(u32, u32); PREFIX_STAGES + 1] {
    let mut g_bits: [u8; WORD_BITS] = core::array::from_fn(|i| ((a & b) >> i) as u8 & 1);
    let mut p_bits: [u8; WORD_BITS] = core::array::from_fn(|i| ((a ^ b) >> i) as u8 & 1);

    let mut out = [(0u32, 0u32); PREFIX_STAGES + 1];
    out[0] = (pack(&g_bits), pack(&p_bits));

    for stage in 0..PREFIX_STAGES {
        for edge in sklansky_edges(stage) {
            let (src, tgt) = (edge.src as usize, edge.tgt as usize);
            let (g_src, p_src) = (g_bits[src], p_bits[src]);
            // (G_tgt, P_tgt) is the "hi" block of this combine; the source
            // holds the "lo" block. See module-level doc for the operator.
            g_bits[tgt] ^= p_bits[tgt] & g_src;
            p_bits[tgt] &= p_src;
        }
        out[stage + 1] = (pack(&g_bits), pack(&p_bits));
    }
    out
}

fn pack(bits: &[u8; WORD_BITS]) -> u32 {
    let mut w = 0u32;
    for (i, &b) in bits.iter().enumerate() {
        w |= (b as u32 & 1) << i;
    }
    w
}

/// SHA-256 majority `Maj(x, y, z) = xy ⊕ xz ⊕ yz`. Mirrors
/// `crate::sha256_f2::maj`; kept here so this module is self-contained
/// for the Phase-1 GKR consumer.
pub const fn maj(x: u32, y: u32, z: u32) -> u32 {
    (x & y) ^ (x & z) ^ (y & z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    /// The honest reference recurrence — same body as
    /// `crate::sha256_f2::binius_carry`, copied here to keep the test
    /// independent of that module's privacy.
    fn binius_carry_ref(a: u32, b: u32) -> u32 {
        let mut c = 0u32;
        for i in 0..32 {
            let ai = (a >> i) & 1;
            let bi = (b >> i) & 1;
            let cprev = if i == 0 { 0 } else { (c >> (i - 1)) & 1 };
            let ci = (ai & bi) ^ (cprev & (ai ^ bi));
            c |= ci << i;
        }
        c
    }

    /// Sanity-check the Sklansky schedule: every stage has exactly
    /// `WORD_BITS / 2 = 16` edges and after all stages every bit
    /// position has been a target at least once (so the prefix is
    /// fully propagated).
    #[test]
    fn sklansky_schedule_shape() {
        let mut touched = [false; WORD_BITS];
        touched[0] = true; // bit 0 is its own prefix; never a target
        for stage in 0..PREFIX_STAGES {
            let edges = sklansky_edges(stage);
            assert_eq!(edges.len(), WORD_BITS / 2, "stage {stage}");
            for e in edges {
                assert!((e.src as usize) < (e.tgt as usize), "edge order");
                touched[e.tgt as usize] = true;
            }
        }
        for (i, t) in touched.iter().enumerate() {
            assert!(*t, "bit {i} never reached the full prefix");
        }
    }

    /// Stage `k` should produce, at position `i`, the prefix combine of
    /// `[i & !mask_k ..= i]` where `mask_k = (1 << (k+1)) - 1`. We
    /// verify this indirectly: after the final stage every position
    /// must hold the full prefix `[0..=i]`, i.e. the carry-out at bit
    /// `i` equals the reference recurrence's bit `i`.
    #[test]
    fn carry_matches_recurrence_edge_cases() {
        let cases = [
            (0u32, 0u32),
            (u32::MAX, 0),
            (0, u32::MAX),
            (u32::MAX, 1),
            (1, u32::MAX),
            (u32::MAX, u32::MAX),
            (0x5555_5555, 0xAAAA_AAAA),
            (0xDEAD_BEEF, 0xCAFE_BABE),
        ];
        for (a, b) in cases {
            assert_eq!(
                prefix_carry(a, b),
                binius_carry_ref(a, b),
                "prefix vs recurrence diverges at a={a:#x}, b={b:#x}",
            );
        }
    }

    /// Fuzz: 1M random pairs must agree byte-for-byte with the
    /// reference recurrence. This is the Phase-0 exit criterion.
    #[test]
    fn carry_matches_recurrence_fuzz() {
        let mut rng = StdRng::seed_from_u64(0xC0FFEE_BABE_F00D_u64);
        for _ in 0..1_000_000 {
            let a: u32 = rng.random();
            let b: u32 = rng.random();
            let got = prefix_carry(a, b);
            let want = binius_carry_ref(a, b);
            assert_eq!(got, want, "fuzz mismatch a={a:#x}, b={b:#x}");
        }
    }

    /// The carry-out and the XOR of inputs reconstruct
    /// `u32::wrapping_add` exactly:
    ///   sum_i = a_i ⊕ b_i ⊕ k_i, where k_i = carry-in at position i.
    /// Carry-in at i equals carry-out at i-1, i.e. bit `i-1` of
    /// `prefix_carry(a, b)`. Output bit 31's carry-out is the overflow
    /// that's discarded for mod-2^32 addition.
    #[test]
    fn carry_reconstructs_wrapping_add() {
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..10_000 {
            let a: u32 = rng.random();
            let b: u32 = rng.random();
            let carry = prefix_carry(a, b);
            let carry_in = carry << 1; // k_i = carry-out at i-1; k_0 = 0
            let sum = a ^ b ^ carry_in;
            assert_eq!(sum, a.wrapping_add(b), "a={a:#x} b={b:#x}");
        }
    }

    /// `Maj` equals bitwise majority on each bit. Trivial but pins
    /// the algebraic identity that the Phase-1 GKR layer will assert.
    #[test]
    fn maj_matches_bitwise_majority() {
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..100_000 {
            let x: u32 = rng.random();
            let y: u32 = rng.random();
            let z: u32 = rng.random();
            let got = maj(x, y, z);
            // Bitwise majority: at least two of (x, y, z) are 1.
            let want = (x & y) | (y & z) | (x & z);
            assert_eq!(got, want, "x={x:#x} y={y:#x} z={z:#x}");
        }
    }
}
