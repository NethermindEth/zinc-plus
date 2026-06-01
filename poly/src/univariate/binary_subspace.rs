//! `F_2`-linear subspaces of `GF(2^128)` and barycentric univariate
//! evaluation / extrapolation over them.
//!
//! These are the math primitives (P2 + P3 in the port plan
//! `documentation/f2-hadamard-oblong-port-plan.md`) for the Binius64-style
//! **oblong univariate AND zerocheck** that replaces the memory-bound
//! bit-slice Hadamard discharge.
//!
//! Ported from binius64 `crates/math/src/{binary_subspace,univariate}.rs`,
//! specialised to our [`BinaryFieldGF128`] (monomial / LSB-first basis,
//! GHASH reduction `0x87` — the same field as binius `BinaryField128bGhash`,
//! per the port plan §0). The math is self-consistent over any correct
//! `GF(2^128)`; matching binius's exact byte convention only matters for
//! the later `GF(2^8)` subfield embedding and for cross-checking vectors.
//!
//! ## What a subspace gives us
//!
//! A degree-`< 2^k` univariate over a `k`-dimensional `F_2`-subspace
//! `H ⊂ GF(2^128)` is the univariate avatar of a `k`-variate multilinear:
//! the subspace points `{H.get(0), …, H.get(2^k − 1)}` are identified with
//! the hypercube `{0,1}^k` via `get(i) = Σ_j basis[j]·bit(i,j)`, and the
//! univariate's evaluations on those points equal the multilinear's
//! hypercube values. This correspondence is what lets the AND zerocheck
//! fuse the `k` bit-within-word variables into a single univariate-skip
//! round (see [`super::oblong_and`]).

use super::binary_gf128::BinaryFieldGF128;

type F = BinaryFieldGF128;

/// `a^{-1}` if `a ≠ 0`, else `0` (binius's `invert_or_zero`). Used by the
/// barycentric routines, where the single zero subspace point is skipped
/// so the inverse is always of a non-zero product in practice.
#[inline]
fn invert_or_zero(a: F) -> F {
    if a.is_zero() { F::zero() } else { a.inverse() }
}

/// The `i`-th default `F_2` basis element of `GF(2^128)`: the monomial
/// `X^i` (bit `i` set, all others clear). Matches binius `F::basis(i)`.
#[inline]
#[allow(clippy::arithmetic_side_effects)]
fn monomial_basis(i: usize) -> F {
    assert!(i < 128, "GF(2^128) has 128 F_2-basis elements");
    if i < 64 {
        F::from_words([1u64 << i, 0])
    } else {
        F::from_words([0, 1u64 << (i - 64)])
    }
}

/// An `F_2`-linear subspace of `GF(2^128)`, defined by an ordered basis.
/// The ordering induces an ordering on the `2^dim` subspace elements via
/// [`Self::get`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinarySubspace {
    basis: Vec<F>,
}

impl BinarySubspace {
    /// A subspace from an explicit ordered basis. Does **not** check the
    /// basis is `F_2`-linearly independent (binius `new_unchecked`).
    pub fn new_unchecked(basis: Vec<F>) -> Self {
        Self { basis }
    }

    /// The `dim`-dimensional subspace spanned by the default basis
    /// `{X^0, …, X^{dim-1}}` (the low `dim` monomials). For this basis
    /// `get(i)` is the field element whose low-`dim`-bit pattern is `i`.
    pub fn with_dim(dim: usize) -> Self {
        assert!(dim <= 128, "GF(2^128) subspace dim must be ≤ 128");
        Self {
            basis: (0..dim).map(monomial_basis).collect(),
        }
    }

    /// The subspace's dimension (number of basis elements).
    pub fn dim(&self) -> usize {
        self.basis.len()
    }

    /// The ordered basis.
    pub fn basis(&self) -> &[F] {
        &self.basis
    }

    /// The `index`-th subspace element, `Σ_j basis[j]·bit(index, j)`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn get(&self, index: usize) -> F {
        debug_assert!(
            self.dim() >= usize::BITS as usize || index < (1usize << self.dim()),
            "index must be < 2^dim"
        );
        let mut acc = F::zero();
        for (j, &b) in self.basis.iter().enumerate() {
            if (index >> j) & 1 == 1 {
                acc += b;
            }
        }
        acc
    }

    /// All `2^dim` subspace elements, in order.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn iter(&self) -> impl Iterator<Item = F> + '_ {
        (0..(1usize << self.dim())).map(move |i| self.get(i))
    }

    /// The sub-subspace spanned by the first `dim` basis elements.
    pub fn reduce_dim(&self, dim: usize) -> Self {
        assert!(dim <= self.dim(), "reduce_dim: dim must be ≤ this subspace's dim");
        Self {
            basis: self.basis[..dim].to_vec(),
        }
    }
}

/// Evaluate a univariate (low-to-high monomial coefficients) at `x` via
/// Horner's method. Port of binius `evaluate_univariate`.
#[allow(clippy::arithmetic_side_effects)]
pub fn evaluate_univariate(coeffs: &[F], x: F) -> F {
    let Some((highest, rest)) = coeffs.split_last() else {
        return F::zero();
    };
    rest.iter().rev().fold(*highest, |acc, &c| acc * x + c)
}

/// Lagrange basis evaluations `L_i(z)` for a power-of-two binary subspace
/// domain. Port of binius `lagrange_evals_scalars`.
///
/// For a `2^k`-point additive subgroup, every barycentric weight is the
/// same value `w = (∏_{j=1}^{2^k − 1} domain[j])^{-1}` (the additive group
/// structure makes `{i ⊕ j : j ≠ i} = {1, …, 2^k − 1}` for every `i`), so
/// a single `w` plus a prefix/suffix product pass gives all evaluations in
/// `O(n)`.
#[allow(clippy::arithmetic_side_effects)]
pub fn lagrange_evals(subspace: &BinarySubspace, z: F) -> Vec<F> {
    let domain: Vec<F> = subspace.iter().collect();
    let n = domain.len();

    // Single barycentric weight for the additive subgroup (skip the zero
    // element domain[0]).
    let w = invert_or_zero(domain[1..].iter().fold(F::one(), |acc, &d| acc * d));

    // prefix[i] = ∏_{j<i} (z − domain[j]);  suffix[i] = ∏_{j>i} (z − domain[j]).
    let mut prefixes = vec![F::one(); n];
    for i in 1..n {
        prefixes[i] = prefixes[i - 1] * (z - domain[i - 1]);
    }
    let mut suffixes = vec![F::one(); n];
    for i in (0..n.saturating_sub(1)).rev() {
        suffixes[i] = suffixes[i + 1] * (z - domain[i + 1]);
    }

    prefixes
        .iter()
        .zip(&suffixes)
        .map(|(&p, &s)| p * s * w)
        .collect()
}

/// Extrapolate a polynomial from its evaluations over a binary subspace to
/// an arbitrary point `z`. Equivalent to `⟨values, lagrange_evals(z)⟩` but
/// without materialising the Lagrange vector. Port of binius
/// `extrapolate_over_subspace`.
#[allow(clippy::arithmetic_side_effects)]
pub fn extrapolate_over_subspace(subspace: &BinarySubspace, values: &[F], z: F) -> F {
    let n = 1usize << subspace.dim();
    assert_eq!(values.len(), n, "values length must equal subspace size");

    let w = invert_or_zero(subspace.iter().skip(1).fold(F::one(), |acc, d| acc * d));

    // Accumulate Σ_i values[i] · ∏_{j≠i} (z − domain[j]) via a prefix-product fold.
    let (acc, _) = values.iter().zip(subspace.iter()).fold(
        (F::zero(), F::one()),
        |(acc, prod), (&value, point)| {
            let term = z - point;
            (acc * term + prod * value, prod * term)
        },
    );
    acc * w
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic GF(2^128) sample from a u64 seed (low 64 bits + a
    /// scrambled high word), to avoid pulling in an RNG dependency.
    fn sample(seed: u64) -> F {
        let hi = seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .rotate_left(29)
            ^ 0x1234_5678_9ABC_DEF0;
        F::from_words([seed ^ 0xA5A5_5A5A_0F0F_F0F0, hi])
    }

    #[test]
    fn with_dim_standard_basis_get_is_index() {
        // Standard basis {X^0..X^{k-1}} ⇒ get(i) is the element with
        // low-k-bit pattern i, i.e. from_words([i, 0]).
        for dim in 0..=8 {
            let s = BinarySubspace::with_dim(dim);
            assert_eq!(s.dim(), dim);
            for i in 0..(1usize << dim) {
                assert_eq!(s.get(i), F::from_words([i as u64, 0]), "dim={dim} i={i}");
            }
        }
    }

    #[test]
    fn iter_matches_get() {
        let s = BinarySubspace::with_dim(6);
        for (i, e) in s.iter().enumerate() {
            assert_eq!(e, s.get(i));
        }
        assert_eq!(s.iter().count(), 64);
    }

    #[test]
    fn reduce_dim_prefix_basis() {
        let s = BinarySubspace::with_dim(6);
        let r = s.reduce_dim(5);
        assert_eq!(r.dim(), 5);
        for i in 0..32 {
            assert_eq!(r.get(i), s.get(i));
        }
    }

    #[test]
    fn lagrange_partition_of_unity() {
        // Σ_i L_i(z) = 1 for any z.
        for dim in [3usize, 4, 5, 6] {
            let s = BinarySubspace::with_dim(dim);
            let z = sample(0xDEAD_BEEF ^ dim as u64);
            let evals = lagrange_evals(&s, z);
            let sum = evals.iter().fold(F::zero(), |a, &b| a + b);
            assert_eq!(sum, F::one(), "partition of unity failed at dim={dim}");
        }
    }

    #[test]
    fn lagrange_interpolation_property() {
        // L_i(domain[j]) = δ_ij.
        for dim in [3usize, 4, 5] {
            let s = BinarySubspace::with_dim(dim);
            let domain: Vec<F> = s.iter().collect();
            for (j, &dj) in domain.iter().enumerate() {
                let evals = lagrange_evals(&s, dj);
                for (i, &e) in evals.iter().enumerate() {
                    let want = if i == j { F::one() } else { F::zero() };
                    assert_eq!(e, want, "L_{i}(x_{j}) wrong at dim={dim}");
                }
            }
        }
    }

    #[test]
    fn extrapolate_matches_horner() {
        // For a random degree-<n polynomial, extrapolating from its
        // subspace evals must agree with direct Horner evaluation.
        for dim in 0..=6 {
            let n = 1usize << dim;
            let s = BinarySubspace::with_dim(dim);
            let coeffs: Vec<F> = (0..n).map(|i| sample(0x100 + i as u64 + (dim as u64) << 32)).collect();
            let values: Vec<F> = s.iter().map(|p| evaluate_univariate(&coeffs, p)).collect();
            let z = sample(0xC0FFEE ^ dim as u64);
            assert_eq!(
                extrapolate_over_subspace(&s, &values, z),
                evaluate_univariate(&coeffs, z),
                "extrapolate≠horner at dim={dim}"
            );
        }
    }

    #[test]
    fn extrapolate_matches_lagrange_inner_product() {
        // For arbitrary (not-necessarily-polynomial) values, extrapolate
        // equals ⟨values, lagrange_evals(z)⟩.
        for dim in 0..=6 {
            let n = 1usize << dim;
            let s = BinarySubspace::with_dim(dim);
            let values: Vec<F> = (0..n).map(|i| sample(0x777 + i as u64)).collect();
            let z = sample(0x555 ^ dim as u64);
            let direct = extrapolate_over_subspace(&s, &values, z);
            let lag = lagrange_evals(&s, z);
            let ip = values
                .iter()
                .zip(&lag)
                .fold(F::zero(), |acc, (&v, &l)| acc + v * l);
            assert_eq!(direct, ip, "extrapolate≠⟨v,L⟩ at dim={dim}");
        }
    }
}
