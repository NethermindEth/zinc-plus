//! `GF(2^8)`-accelerated additive-NTT + oblong round message — the prover
//! **speed lever** for the oblong AND zerocheck (port plan
//! `documentation/f2-hadamard-oblong-port-plan.md`, prerequisite P4).
//!
//! The naive path ([`super::oblong_and`]) runs the additive-NTT word extension
//! and the per-extension-point products in `GF(2^128)` (CLMUL + 16-byte XORs).
//! Here we run them in [`Gf8`] (`GF(2^8)`: log/antilog mult, 1-byte XORs) via a
//! precomputed **byte-lookup table** (binius `ntt_lookup.rs`), and lift to
//! `GF(2^128)` only for the final `eq`-weighted accumulation.
//!
//! ## Why this is sound: the embed subspace
//!
//! The NTT runs over the `GF(2^8)` subspace `H₈ = {Gf8(0), …, Gf8(63)}` (the
//! `Gf8` monomial basis `{1,α,…,α⁵}`). Its image `embed(H₈) ⊂ GF(2^128)` is
//! the `(SKIPPED_VARS+1)`-dim subspace the **verifier** extrapolates over. Since
//! [`embed`](super::binary_gf8::embed) is a field homomorphism, every `Gf8`
//! NTT value / product equals (after `embed`) the corresponding `GF(2^128)`
//! computation over `embed(H₈)` — so the `Gf8` path produces exactly the round
//! message a `GF(2^128)` prover would over that subspace, only faster. This is
//! a **different subspace** than the naive path's monomial `{X⁰…X⁵}`; the two
//! are not meant to produce equal round messages, only each internally
//! consistent with its own verifier.
//!
//! Current status: the NTT + products run in `GF(2^8)`; the `eq`-weighting
//! still runs per-word in `GF(2^128)`. Binius's further optimisation — split
//! `eq` into ≤3 deterministic small-field vars (weighted in `GF(2^8)`,
//! accumulated per chunk) + the big-field remainder (weighted once per chunk) —
//! is a follow-up that cuts the `GF(2^128)` mult count by the chunk size.

use super::binary_gf128::BinaryFieldGF128;
use super::binary_gf8::{Gf8, embed};
use super::binary_subspace::{BinarySubspace, lagrange_evals};
use super::oblong_and::{OblongScheme, PAR_MIN_LEN, SKIPPED_VARS, WORD_BITS};

type F128 = BinaryFieldGF128;

/// Bytes per word (`WORD_BITS / 8 = 4` for the 32-bit SHA words).
pub const WORD_BYTES: usize = WORD_BITS / 8;

/// Lagrange basis evaluations `L_b(z)` over the `GF(2^8)` base subspace
/// `{Gf8(0), …, Gf8(WORD_BITS−1)}` (an `F_2`-subgroup ⇒ single barycentric
/// weight). The `Gf8` analogue of [`super::binary_subspace::lagrange_evals`].
#[allow(clippy::arithmetic_side_effects)]
fn gf8_base_lagrange(z: Gf8) -> [Gf8; WORD_BITS] {
    let domain: [Gf8; WORD_BITS] = std::array::from_fn(|i| Gf8(i as u8));

    // w = (∏_{j=1}^{WORD_BITS−1} domain[j])^{-1}.
    let prod = domain[1..].iter().fold(Gf8::ONE, |acc, &d| acc.mul(d));
    let w = prod.inverse();

    let mut prefixes = [Gf8::ONE; WORD_BITS];
    for i in 1..WORD_BITS {
        prefixes[i] = prefixes[i - 1].mul(z - domain[i - 1]);
    }
    let mut suffixes = [Gf8::ONE; WORD_BITS];
    for i in (0..WORD_BITS - 1).rev() {
        suffixes[i] = suffixes[i + 1].mul(z - domain[i + 1]);
    }
    std::array::from_fn(|i| prefixes[i].mul(suffixes[i]).mul(w))
}

/// Precomputed byte-lookup additive NTT over `GF(2^8)`.
///
/// `lut[byte_idx][byte_val][j]` is the contribution to extension-point `j` of
/// the 8 bits of byte `byte_idx` taking value `byte_val`, i.e.
/// `Σ_{bit set in byte_val} L_{byte_idx·8 + bit}(ext_j)`. Then
/// `extend_word(w)[j] = Σ_{byte_idx} lut[byte_idx][byte(w, byte_idx)][j]` —
/// `WORD_BYTES` lookups + adds, vs the naive `popcount(w)` Lagrange-row XORs.
#[derive(Clone)]
pub struct Gf8Ntt {
    lut: Vec<[[Gf8; WORD_BITS]; 256]>, // length WORD_BYTES
}

impl Default for Gf8Ntt {
    fn default() -> Self {
        Self::new()
    }
}

impl Gf8Ntt {
    #[allow(clippy::arithmetic_side_effects)]
    pub fn new() -> Self {
        // ext_lagrange_by_point[j][b] = L_b(ext_j), ext_j = Gf8(WORD_BITS + j).
        let ext_lagrange_by_point: Vec<[Gf8; WORD_BITS]> = (0..WORD_BITS)
            .map(|j| gf8_base_lagrange(Gf8((WORD_BITS + j) as u8)))
            .collect();

        let mut lut = vec![[[Gf8::ZERO; WORD_BITS]; 256]; WORD_BYTES];
        for (byte_idx, table) in lut.iter_mut().enumerate() {
            for (val, row) in table.iter_mut().enumerate() {
                for (j, slot) in row.iter_mut().enumerate() {
                    let mut acc = Gf8::ZERO;
                    for bit in 0..8 {
                        if (val >> bit) & 1 == 1 {
                            acc = acc + ext_lagrange_by_point[j][byte_idx * 8 + bit];
                        }
                    }
                    *slot = acc;
                }
            }
        }
        Self { lut }
    }

    /// Extension-domain evaluations of the word `w` in `GF(2^8)`:
    /// `out[j] = A(ext_j)` over the `Gf8` subspace.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn extend_word(&self, w: u32) -> [Gf8; WORD_BITS] {
        let bytes = w.to_le_bytes();
        let mut out = [Gf8::ZERO; WORD_BITS];
        for (byte_idx, table) in self.lut.iter().enumerate() {
            let row = &table[bytes[byte_idx] as usize];
            for (o, &r) in out.iter_mut().zip(row.iter()) {
                *o = *o + r;
            }
        }
        out
    }
}

/// The Phase-1 oblong round message computed with the `GF(2^8)` NTT: the
/// `WORD_BITS` extension-domain evaluations of `R₀(Z)=Σ_X (A·B−C)(Z,X)·eq(X;r)`,
/// returned in `GF(2^128)` (the NTT + product run in `GF(2^8)`, then `embed`).
///
/// Output matches [`super::oblong_and::univariate_round_message`] run over the
/// `embed(H₈)` subspace (not the monomial one). `eq` is the `GF(2^128)` row-var
/// equality indicator (length `2ⁿ`).
/// Accumulate one word-triple's contribution into the round-message
/// accumulator: NTT-extend in `GF(2^8)`, `(A·B − C)` per extension point in
/// `GF(2^8)`, then `embed` + weight by the `GF(2^128)` eq value.
#[inline]
#[allow(clippy::arithmetic_side_effects)]
fn accumulate_gf8_word(
    acc: &mut [F128; WORD_BITS],
    ntt: &Gf8Ntt,
    a: u32,
    b: u32,
    c: u32,
    w: F128,
) {
    let ae = ntt.extend_word(a);
    let be = ntt.extend_word(b);
    let ce = ntt.extend_word(c);
    for j in 0..WORD_BITS {
        acc[j] += embed(ae[j].mul(be[j]) - ce[j]) * w;
    }
}

#[allow(clippy::arithmetic_side_effects)]
pub fn gf8_round_message(
    a_words: &[u32],
    b_words: &[u32],
    c_words: &[u32],
    eq: &[F128],
    ntt: &Gf8Ntt,
) -> [F128; WORD_BITS] {
    let n = a_words.len();
    assert_eq!(b_words.len(), n);
    assert_eq!(c_words.len(), n);
    assert_eq!(eq.len(), n);

    // Parallel map-reduce over the words (each contributes to the 32-element
    // accumulator). The fused discharge this replaces is parallel too.
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        (0..n)
            .into_par_iter()
            .with_min_len(PAR_MIN_LEN)
            .fold(
                || [F128::zero(); WORD_BITS],
                |mut acc, x| {
                    accumulate_gf8_word(&mut acc, ntt, a_words[x], b_words[x], c_words[x], eq[x]);
                    acc
                },
            )
            .reduce(
                || [F128::zero(); WORD_BITS],
                |mut a, b| {
                    for j in 0..WORD_BITS {
                        a[j] += b[j];
                    }
                    a
                },
            )
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut acc = [F128::zero(); WORD_BITS];
        for x in 0..n {
            accumulate_gf8_word(&mut acc, ntt, a_words[x], b_words[x], c_words[x], eq[x]);
        }
        acc
    }
}

/// The `embed(H₈)` subspaces the verifier must use with the `GF(2^8)` path:
/// `(base, full)` of dims `SKIPPED_VARS` and `SKIPPED_VARS+1`, as
/// `GF(2^128)` `BinarySubspace`s. (The naive path uses the monomial subspace
/// instead.)
pub fn embed_subspaces() -> (super::binary_subspace::BinarySubspace, super::binary_subspace::BinarySubspace) {
    use super::binary_subspace::BinarySubspace;
    // embed is F_2-linear, so embed(H₈) is spanned by embed of the Gf8 monomial
    // basis {α⁰…α^{dim-1}} = bytes {1,2,4,…}. The basis (not the points!) goes to
    // new_unchecked: dim-many elements, with get(i) = embed(Gf8(i)).
    let base_basis: Vec<F128> = (0..SKIPPED_VARS).map(|i| embed(Gf8(1u8 << i))).collect();
    let full_basis: Vec<F128> = (0..=SKIPPED_VARS).map(|i| embed(Gf8(1u8 << i))).collect();
    (
        BinarySubspace::new_unchecked(base_basis),
        BinarySubspace::new_unchecked(full_basis),
    )
}

/// The `GF(2^8)` [`OblongScheme`]: the byte-lookup additive NTT over the
/// `embed(H₈)` subspace. Drop-in for [`super::oblong_and::MonomialScheme`] in
/// `prove_oblong_and_channel`; the verifier must pair it with
/// [`Gf8Scheme::full_subspace`]. Holds the precomputed NTT table + the
/// `embed(H₈)` base/full subspaces.
pub struct Gf8Scheme {
    ntt: Gf8Ntt,
    base: BinarySubspace,
    full: BinarySubspace,
}

impl Default for Gf8Scheme {
    fn default() -> Self {
        Self::new()
    }
}

impl Gf8Scheme {
    pub fn new() -> Self {
        let (base, full) = embed_subspaces();
        Self {
            ntt: Gf8Ntt::new(),
            base,
            full,
        }
    }

    /// The full univariate domain (`embed(H₈)`, dim `SKIPPED_VARS+1`) the
    /// verifier extrapolates `R₀` over with this scheme.
    pub fn full_subspace(&self) -> &BinarySubspace {
        &self.full
    }
}

impl OblongScheme for Gf8Scheme {
    fn round_message(&self, a: &[u32], b: &[u32], c: &[u32], eq: &[F128]) -> [F128; WORD_BITS] {
        gf8_round_message(a, b, c, eq, &self.ntt)
    }

    fn base_lagrange(&self, z: F128) -> [F128; WORD_BITS] {
        // L_b(z) over the embed(H₈) base subspace — the fold weights matching
        // the GF(2^8) round message's basis.
        let lag = lagrange_evals(&self.base, z);
        let mut out = [F128::zero(); WORD_BITS];
        out.copy_from_slice(&lag);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::super::binary_subspace::{BinarySubspace, extrapolate_over_subspace};
    use super::*;

    fn sample32(seed: u64) -> u32 {
        (seed.wrapping_mul(0xD1B5_4A32_D192_ED03) >> 17) as u32
    }
    fn sample128(seed: u64) -> F128 {
        let hi = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(29) ^ 0x1234_5678_9ABC_DEF0;
        F128::from_words([seed ^ 0xA5A5_5A5A_0F0F_F0F0, hi])
    }

    /// GF(2^128) reference: extend a word over the embed(H₈) base subspace by
    /// interpolating its bits and extrapolating to embed(ext_j).
    fn gf128_extend_over_embed(w: u32, base: &BinarySubspace, ext_points: &[F128]) -> Vec<F128> {
        let bits: Vec<F128> = (0..WORD_BITS)
            .map(|b| if (w >> b) & 1 == 1 { F128::one() } else { F128::zero() })
            .collect();
        ext_points
            .iter()
            .map(|&p| extrapolate_over_subspace(base, &bits, p))
            .collect()
    }

    #[test]
    fn gf8_extend_embeds_to_gf128_extend() {
        // The core correctness gate: embed(Gf8 NTT) == GF128 NTT over embed(H₈).
        let ntt = Gf8Ntt::new();
        let (base, _full) = embed_subspaces();
        let ext_points: Vec<F128> = (0..WORD_BITS).map(|j| embed(Gf8((WORD_BITS + j) as u8))).collect();

        for &w in &[0u32, 1, 0xFFFF_FFFF, 0x8000_0001, sample32(7), sample32(123), sample32(9999)] {
            let g = ntt.extend_word(w);
            let reference = gf128_extend_over_embed(w, &base, &ext_points);
            for j in 0..WORD_BITS {
                assert_eq!(embed(g[j]), reference[j], "word={w:#x} ext {j}");
            }
        }
    }

    #[test]
    fn gf8_round_message_matches_gf128_over_embed_subspace() {
        // The full round message via GF(2^8) must equal the GF(2^128)
        // computation over the same embed(H₈) subspace.
        let (base, _full) = embed_subspaces();
        let ext_points: Vec<F128> = (0..WORD_BITS).map(|j| embed(Gf8((WORD_BITS + j) as u8))).collect();
        let ntt = Gf8Ntt::new();

        for n in [4usize, 6] {
            let num = 1usize << n;
            let a: Vec<u32> = (0..num).map(|i| sample32(i as u64 * 3 + 1)).collect();
            let b: Vec<u32> = (0..num).map(|i| sample32(i as u64 * 5 + 2)).collect();
            let c: Vec<u32> = a.iter().zip(&b).map(|(&x, &y)| x & y).collect();
            let eq: Vec<F128> = (0..num).map(|i| sample128(i as u64 + 70)).collect();

            let got = gf8_round_message(&a, &b, &c, &eq, &ntt);

            // Reference in GF(2^128) over embed(H₈).
            let mut want = [F128::zero(); WORD_BITS];
            for x in 0..num {
                let ae = gf128_extend_over_embed(a[x], &base, &ext_points);
                let be = gf128_extend_over_embed(b[x], &base, &ext_points);
                let ce = gf128_extend_over_embed(c[x], &base, &ext_points);
                for j in 0..WORD_BITS {
                    want[j] += (ae[j] * be[j] - ce[j]) * eq[x];
                }
            }
            assert_eq!(got, want, "gf8 round message mismatch at n={n}");
        }
    }

    #[test]
    fn gf8_scheme_full_protocol_round_trips() {
        // Prove with the GF(2^8) scheme + verify over embed(H₈): the full
        // Phase-1+2 protocol must round-trip (honest AND), validating that the
        // GF(2^8) round message, the embed-base fold, and the embed-full R₀
        // reconstruction are mutually consistent.
        use super::super::oblong_and::{
            ReplayChannel, prove_oblong_and_channel, verify_oblong_and_channel,
        };
        for n in [4usize, 6] {
            let num = 1usize << n;
            let a: Vec<u32> = (0..num).map(|i| sample32(i as u64 * 3 + 1)).collect();
            let b: Vec<u32> = (0..num).map(|i| sample32(i as u64 * 5 + 2)).collect();
            let c: Vec<u32> = a.iter().zip(&b).map(|(&x, &y)| x & y).collect();
            let scheme = Gf8Scheme::new();

            // Shared challenge sequence [r…, z, γ…] for prove + verify.
            let mut seq: Vec<F128> = (0..n).map(|i| sample128(i as u64 + 1)).collect();
            seq.push(sample128(0xBEEF));
            seq.extend((0..n).map(|k| sample128(0xCAFE + k as u64)));

            let mut pch = ReplayChannel::new(seq.clone());
            let proof = prove_oblong_and_channel(&mut pch, &a, &b, &c, &scheme);

            let mut vch = ReplayChannel::new(seq);
            let out = verify_oblong_and_channel(&mut vch, &proof, n, scheme.full_subspace())
                .expect("GF(2^8) scheme must verify honest AND");
            assert_eq!(out.eval_point.len(), n + 1);
        }
    }

    #[test]
    fn gf8_extend_is_f2_linear() {
        // extend_word = Σ L_b over set bits ⇒ linear: ext(w1^w2)=ext(w1)+ext(w2).
        let ntt = Gf8Ntt::new();
        let w1 = sample32(11);
        let w2 = sample32(22);
        let e1 = ntt.extend_word(w1);
        let e2 = ntt.extend_word(w2);
        let e12 = ntt.extend_word(w1 ^ w2);
        for j in 0..WORD_BITS {
            assert_eq!(e12[j], e1[j] + e2[j]);
        }
    }

    /// Release-mode timing: GF(2^8) byte-lookup round message vs the naive
    /// GF(2^128) one, at the nvars=16 discharge scale. Run with:
    /// `cargo test -p zinc-poly --release --features simd \
    ///   oblong_and_gf8::tests::bench -- --ignored --nocapture`
    #[test]
    #[ignore = "timing benchmark; run with --release --ignored --nocapture"]
    fn bench_gf8_vs_gf128_round_message() {
        use super::super::oblong_and::{AdditiveNtt, eq_indicator, univariate_round_message};
        use std::time::Instant;

        let n = 16usize;
        let num = 1usize << n;
        let a: Vec<u32> = (0..num).map(|i| sample32(i as u64 * 3 + 1)).collect();
        let b: Vec<u32> = (0..num).map(|i| sample32(i as u64 * 5 + 2)).collect();
        let c: Vec<u32> = a.iter().zip(&b).map(|(&x, &y)| x & y).collect();
        let r: Vec<F128> = (0..n).map(|i| sample128(i as u64 + 9)).collect();
        let eq = eq_indicator(&r);

        let gf128_ntt = AdditiveNtt::new();
        let gf8_ntt = Gf8Ntt::new();

        let reps = 5;
        let mut best_128 = f64::MAX;
        let mut best_8 = f64::MAX;
        for _ in 0..reps {
            let t = Instant::now();
            let m1 = univariate_round_message(&a, &b, &c, &eq, &gf128_ntt);
            best_128 = best_128.min(t.elapsed().as_secs_f64());
            std::hint::black_box(m1);

            let t = Instant::now();
            let m2 = gf8_round_message(&a, &b, &c, &eq, &gf8_ntt);
            best_8 = best_8.min(t.elapsed().as_secs_f64());
            std::hint::black_box(m2);
        }
        let per = |s: f64| s * 1e9 / num as f64;
        println!(
            "\n  round message, n={n} ({num} words), best of {reps}:\n    \
             GF(2^128) naive : {:8.2} ms  ({:6.2} ns/word)\n    \
             GF(2^8)  lookup : {:8.2} ms  ({:6.2} ns/word)\n    \
             speedup         : {:.2}x  (NTT+products in GF(2^8); eq-weighting still GF(2^128))\n",
            best_128 * 1e3,
            per(best_128),
            best_8 * 1e3,
            per(best_8),
            best_128 / best_8,
        );
    }
}
