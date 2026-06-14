//! Foldable Reed--Solomon chain over `GF(2^128)` via the additive NTT.
//!
//! **Phase 1 of the proof-size plan** (`documentation/f2-codeswitch-dual-readoff-design.md`,
//! `f2x-sha-todo.md` "Code-switching opening via the Zip++ binary basefold lane"):
//! the binary, limb-free substrate for a Basefold/WHIR inner IOPP that will
//! replace the Brakedown-style `prove_f2_open` and shrink the SHA-256 proof
//! from ~1.1 MiB to the ~150--250 KB class.
//!
//! Built on the in-house [`super::binary_subspace`] additive-NTT primitives
//! over our own [`BinaryFieldGF128`] — *not* the binius64 / bolt-rs RS codes,
//! which live over different `GF(2^128)` representations (an `F_2`-linear
//! isomorphism would have to bridge every claim and the Fiat--Shamir
//! transcript). The Zip++ integer chain (`zip-plus/src/basefold/chain.rs`)
//! also does not port: its butterfly `out[j] = even ± tw·odd` collapses in
//! characteristic 2 (`+ = −`); the correct binary butterfly is the *additive*
//! NTT (Lin--Chung--Han), realised here.
//!
//! ## The fold (the one new primitive)
//!
//! A Reed--Solomon codeword is `c[i] = P(S.get(i))` for a message polynomial
//! `P` of degree `< 2^k` evaluated over a `K`-dimensional `F_2`-subspace `S`
//! (rate `2^{k-K}`). Let `b = S.basis[K-1]` and let `ŝ_b(x) = x·(x+b)` be the
//! 2-to-1 `F_2`-linear map with kernel `{0, b}`. In the novel polynomial
//! basis `P(x) = E(ŝ_b(x)) + x·O(ŝ_b(x))` with `deg E, deg O < 2^{k-1}`, so
//! from the coset pair `(y_i, y_i + b)` (codeword indices `i`, `i + half`):
//! ```text
//!   c[i]      = E(u_i) + y_i·O(u_i)
//!   c[i+half] = E(u_i) + (y_i+b)·O(u_i)          (u_i = ŝ_b(y_i))
//!   ⇒  O(u_i) = (c[i] + c[i+half]) / b
//!      E(u_i) =  c[i] + y_i·O(u_i)
//! ```
//! Folding with a challenge `α` keeps `P'(u) = E(u) + α·O(u)` (degree
//! `< 2^{k-1}`), so the folded codeword over the image subspace
//! `S'' = span{ŝ_b(b_0), …, ŝ_b(b_{K-2})}` is
//! ```text
//!   c'[i] = E(u_i) + α·O(u_i) = c[i] + (y_i + α)·(c[i] + c[i+half]) / b.
//! ```
//! `ŝ_b` is `F_2`-linear (char 2: `(x+x')² = x²+x'²`), injective on
//! `span{b_0,…,b_{K-2}}` (its kernel is `{0,b}` and `b` is outside that
//! span), so the `K-1` images are independent and `S''` is a genuine
//! `(K-1)`-dimensional subspace with `S''.get(0) = 0`.

use super::binary_gf128::BinaryFieldGF128;
use super::binary_subspace::{BinarySubspace, evaluate_univariate, extrapolate_over_subspace};

type F = BinaryFieldGF128;

#[inline]
fn inv(a: F) -> F {
    assert!(!a.is_zero(), "fold direction must be a non-zero field element");
    a.inverse()
}

/// Reed--Solomon encode: evaluate the message polynomial `coeffs` (low-to-high
/// monomial coefficients, `len = 2^k`) over every point of `subspace`
/// (`dim = K ≥ k`), yielding a codeword of length `2^K` and rate `2^{k-K}`.
///
/// This is the additive-NTT image; the naive `O(2^k · 2^K)` Horner form is
/// used for the Phase-1 correctness substrate — a butterfly NTT is a later
/// optimisation (and the ledger shows encode is not the commit bottleneck).
#[allow(clippy::arithmetic_side_effects)]
pub fn rs_encode(coeffs: &[F], subspace: &BinarySubspace) -> Vec<F> {
    assert!(
        coeffs.len() <= (1usize << subspace.dim()),
        "message length must be ≤ subspace size"
    );
    subspace.iter().map(|x| evaluate_univariate(coeffs, x)).collect()
}

/// One additive-NTT fold round: fold a length-`2^K` codeword over `subspace`
/// (dim `K`) with challenge `alpha` into a length-`2^{K-1}` codeword over the
/// image subspace `S'' = span{ŝ_b(b_j)}`. Returns `(c', S'')`.
#[allow(clippy::arithmetic_side_effects)]
pub fn fold_codeword(
    codeword: &[F],
    subspace: &BinarySubspace,
    alpha: F,
) -> (Vec<F>, BinarySubspace) {
    let dim = subspace.dim();
    assert!(dim >= 1, "cannot fold a 0-dimensional subspace");
    assert_eq!(codeword.len(), 1usize << dim, "codeword length must be 2^dim");

    let b = subspace.basis()[dim - 1];
    let b_inv = inv(b);
    let half = codeword.len() / 2;

    // c'[i] = c[i] + (y_i + α)·(c[i] + c[i+half])/b
    let folded: Vec<F> = (0..half)
        .map(|i| {
            let y_i = subspace.get(i);
            let o = (codeword[i] + codeword[i + half]) * b_inv;
            codeword[i] + (y_i + alpha) * o
        })
        .collect();

    // Image subspace basis: ŝ_b(b_j) = b_j·(b_j + b), for j < dim-1.
    let folded_basis: Vec<F> = subspace.basis()[..dim - 1]
        .iter()
        .map(|&bj| bj * (bj + b))
        .collect();

    (folded, BinarySubspace::new_unchecked(folded_basis))
}

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects)]
mod tests {
    use super::*;

    /// Deterministic GF(2^128) sample (mirrors binary_subspace's test helper).
    fn sample(seed: u64) -> F {
        let hi = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(29) ^ 0x1234_5678_9ABC_DEF0;
        F::from_words([seed ^ 0xA5A5_5A5A_0F0F_F0F0, hi])
    }

    /// Test that a single fold maps `RS[k, K]` into `RS[k-1, K-1]`: the folded
    /// codeword must be a valid Reed--Solomon codeword of degree `< 2^{k-1}`
    /// over the image subspace. Verified non-circularly: define the candidate
    /// low-degree polynomial from the first `2^{k-1}` folded values (over the
    /// `(k-1)`-dim sub-subspace) and check *every* folded value matches its
    /// extrapolation. If the fold formula were wrong, `c'` would not be low
    /// degree and the tail values would mismatch.
    #[test]
    fn fold_maps_rs_to_smaller_rs() {
        for (k, r) in [(3usize, 2usize), (4, 1), (4, 2), (2, 3), (5, 1)] {
            let big_dim = k + r;
            let s = BinarySubspace::with_dim(big_dim);
            let coeffs: Vec<F> = (0..(1usize << k)).map(|i| sample(0xA00 + i as u64 + ((k * 17 + r) as u64) << 20)).collect();
            let cw = rs_encode(&coeffs, &s);
            assert_eq!(cw.len(), 1usize << big_dim);

            let alpha = sample(0xF01D ^ ((k << 4) | r) as u64);
            let (cw2, s2) = fold_codeword(&cw, &s, alpha);
            assert_eq!(s2.dim(), big_dim - 1);
            assert_eq!(cw2.len(), 1usize << (big_dim - 1));

            // The folded message has degree < 2^{k-1}. Its codeword over s2 is
            // determined by the first 2^{k-1} points (the (k-1)-dim sub-subspace).
            if k == 0 {
                continue;
            }
            let sub = s2.reduce_dim(k - 1);
            let low = &cw2[..(1usize << (k - 1))];
            for (i, &v) in cw2.iter().enumerate() {
                let got = extrapolate_over_subspace(&sub, low, s2.get(i));
                assert_eq!(v, got, "fold not low-degree at k={k} r={r} i={i}");
            }
        }
    }

    /// Folding a constant codeword (degree-0 message, `k=0`) with any `α`
    /// returns the same constant (E ≡ const, O ≡ 0).
    #[test]
    fn fold_of_constant_is_constant() {
        let big_dim = 4usize;
        let s = BinarySubspace::with_dim(big_dim);
        let c = sample(0x9);
        let cw = rs_encode(&[c], &s); // P ≡ c
        assert!(cw.iter().all(|&v| v == c));
        let (cw2, _s2) = fold_codeword(&cw, &s, sample(0x7));
        assert!(cw2.iter().all(|&v| v == c), "fold of constant changed it");
    }

    /// Full Basefold descent: fold `k` times with random challenges down to a
    /// `RS[0, r]` constant codeword, and check it equals `P'` — the message
    /// successively folded as `P_{j+1}(u) = E_j(u) + α_j·O_j(u)` — which for a
    /// fully-folded poly collapses to a single field element. We validate by
    /// re-deriving that the final codeword is constant (degree-0).
    #[test]
    fn full_descent_reaches_constant() {
        let (k, r) = (5usize, 2usize);
        let mut s = BinarySubspace::with_dim(k + r);
        let coeffs: Vec<F> = (0..(1usize << k)).map(|i| sample(0xBEEF + i as u64)).collect();
        let mut cw = rs_encode(&coeffs, &s);
        for round in 0..k {
            let alpha = sample(0x5151 + round as u64);
            let (cw2, s2) = fold_codeword(&cw, &s, alpha);
            cw = cw2;
            s = s2;
        }
        // After k folds the message is degree-0: codeword constant over the
        // remaining r-dim subspace.
        let c0 = cw[0];
        assert!(cw.iter().all(|&v| v == c0), "did not reach a constant after k folds");
        assert_eq!(cw.len(), 1usize << r);
    }
}
