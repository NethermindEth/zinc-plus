//! Precomputed parameters for the additive Reed-Solomon FFT over
//! `GF(2^32)` lifted to `Z[X] / f̃`.
//!
//! Every twiddle used by the FFT kernel is a *lifted* `GF(2^32)`
//! element — i.e. has `{0, 1} ⊂ Z` coefficients in `Z[X] / f̃` — so we
//! store all twiddle-shaped data in [`Gf2_32`] form. The kernel calls
//! [`ring_ops::mul_by_lifted_gf32`](super::ring_ops::mul_by_lifted_gf32)
//! to perform the lifted-ring multiplication on the fly, avoiding the
//! storage cost of a separate `Z[X] / f̃`-typed twiddle table.

use crate::ZipError;
use std::{fmt::Debug, marker::PhantomData};

use super::basis::{D, F_TILDE_LOWER_DEGREES, Gf2_32, cantor_basis};

/// Configuration trait parameterising the binary additive FFT.
///
/// Mirrors the role of [`crate::code::iprs::pntt::radix8::params::Config`]
/// for IPRS: declares the algebraic ground (degree of the binary
/// extension, irreducible polynomial, Cantor basis). Fixed at `D = 32`
/// for the current implementation, per the plan.
pub trait Config: Debug + Copy + PartialEq + Eq + Send + Sync + 'static {
    /// Extension degree of the binary base field: `GF(2^DEGREE)`.
    const DEGREE: usize;

    /// Non-leading monomial degrees of `f̃` (degrees `< DEGREE` with
    /// coefficient `1`).
    const F_TILDE_LOWER_DEGREES: &'static [usize];

    /// Compute the Cantor basis `{β_0, …, β_{DEGREE-1}}` in `GF(2^DEGREE)`.
    fn cantor_basis() -> [Gf2_32; D];
}

/// Concrete configuration for `GF(2^32) = F_2[X] / (X^32 + X^7 + X^3 + X^2 + 1)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AddFftConfigGF2_32;

impl Config for AddFftConfigGF2_32 {
    const DEGREE: usize = D;
    const F_TILDE_LOWER_DEGREES: &'static [usize] = &F_TILDE_LOWER_DEGREES;

    fn cantor_basis() -> [Gf2_32; D] {
        cantor_basis()
    }
}

/// Precomputed parameters for the radix-2 additive FFT.
///
/// The FFT evaluates a polynomial of degree `< row_len` at all
/// `codeword_len` points of the affine subspace
/// `V_d = span_{F_2}{β_0, …, β_{d-1}}` of `GF(2^32)`, where
/// `codeword_len = 2^d`. Inputs are padded with zeros to length
/// `codeword_len` before evaluation.
///
/// We require `codeword_len.is_power_of_two()` and
/// `row_len.is_power_of_two()` with `row_len ≤ codeword_len ≤ 2^32`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Radix2AddFftParams<C: Config> {
    /// Number of input coefficients (a power of 2, `≤ codeword_len`).
    pub row_len: usize,

    /// Number of output evaluations (a power of 2, `≤ 2^32`).
    pub codeword_len: usize,

    /// `log2(codeword_len)`. This is also the number of butterfly
    /// stages the FFT performs and the dimension of the evaluation
    /// subspace `V_d`.
    pub log2_codeword_len: usize,

    /// Cantor basis `β_0, …, β_{D-1}` of `GF(2^32)`. The first
    /// `log2_codeword_len` elements span the evaluation subspace.
    pub cantor_basis: [Gf2_32; D],

    /// Evaluation points. `eval_points[k] = ∑_{i=0..log2_codeword_len} k_i · β_i`
    /// where `k_i` is bit `i` of `k`. Length: `codeword_len`.
    pub eval_points: Vec<Gf2_32>,

    /// Subspace polynomials `s_0, …, s_{log2_codeword_len - 1}` in
    /// sparse form. Each `s_i(X)` is `F_2`-linearised, with nonzero
    /// coefficients only at degrees `2^j` for `j = 0..=i`, so we
    /// store `subspace_polys[i][j]` = coefficient of `X^(2^j)` in
    /// `s_i(X)`. The leading coefficient `subspace_polys[i][i]` is
    /// always `Gf2_32::ONE`.
    pub subspace_polys: Vec<Vec<Gf2_32>>,

    _phantom: PhantomData<C>,
}

impl<C: Config> Radix2AddFftParams<C> {
    /// Construct precomputed parameters. Validates that `row_len` and
    /// `codeword_len` are both powers of two, that
    /// `row_len ≤ codeword_len`, and that `codeword_len ≤ 2^DEGREE`
    /// (the Cantor basis only spans a `D`-dimensional subspace).
    pub fn new(row_len: usize, codeword_len: usize) -> Result<Self, ZipError> {
        if !row_len.is_power_of_two() {
            return Err(ZipError::InvalidPcsParam(format!(
                "row_len ({row_len}) must be a power of 2"
            )));
        }
        if !codeword_len.is_power_of_two() {
            return Err(ZipError::InvalidPcsParam(format!(
                "codeword_len ({codeword_len}) must be a power of 2"
            )));
        }
        if row_len > codeword_len {
            return Err(ZipError::InvalidPcsParam(format!(
                "row_len ({row_len}) must be ≤ codeword_len ({codeword_len})"
            )));
        }
        let log2_codeword_len = codeword_len.trailing_zeros() as usize;
        if log2_codeword_len > C::DEGREE {
            return Err(ZipError::InvalidPcsParam(format!(
                "codeword_len ({codeword_len}) exceeds 2^{} addressable by the Cantor basis",
                C::DEGREE
            )));
        }

        let cantor_basis = C::cantor_basis();
        let eval_points = compute_eval_points(&cantor_basis, log2_codeword_len);
        let subspace_polys = compute_subspace_polynomials(&cantor_basis, log2_codeword_len);

        Ok(Self {
            row_len,
            codeword_len,
            log2_codeword_len,
            cantor_basis,
            eval_points,
            subspace_polys,
            _phantom: PhantomData,
        })
    }
}

/// `eval_points[k] = ∑_{i=0..log2_codeword_len} k_i · β_i`.
fn compute_eval_points(basis: &[Gf2_32; D], log2_codeword_len: usize) -> Vec<Gf2_32> {
    let n = 1usize << log2_codeword_len;
    let mut points = Vec::with_capacity(n);
    for k in 0..n {
        let mut v = Gf2_32::ZERO;
        let mut bits = k;
        let mut i = 0usize;
        while bits != 0 {
            if bits & 1 != 0 {
                v = v.add(basis[i]);
            }
            bits >>= 1;
            i += 1;
        }
        points.push(v);
    }
    points
}

/// Compute `s_0, …, s_{m - 1}` in sparse form. Uses the Cantor
/// recurrence: `s_0(X) = X` and
/// `s_{i+1}(X) = s_i(X)² + s_i(X)` (valid because `s_i(β_i) = 1`).
///
/// `s_i(X) = ∑_{j=0..=i} c_{i,j} · X^(2^j)`. Squaring an
/// `F_2`-linearised polynomial squares each coefficient and
/// doubles each exponent: `s_i(X)² = ∑_j c_{i,j}² · X^(2^(j+1))`.
/// Combining gives `c_{i+1, j} = c_{i, j-1}² + c_{i, j}` (with
/// `c_{i, j} = 0` for `j > i` and `c_{i, -1} = 0`).
fn compute_subspace_polynomials(_basis: &[Gf2_32; D], m: usize) -> Vec<Vec<Gf2_32>> {
    let mut polys: Vec<Vec<Gf2_32>> = Vec::with_capacity(m);

    // s_0(X) = X, i.e. coefficient of X^(2^0) is 1.
    let mut current = vec![Gf2_32::ONE];

    for i in 0..m {
        polys.push(current.clone());

        // s_{i+1, j} = s_{i, j-1}² + s_{i, j}
        let next_len = i + 2;
        let mut next = vec![Gf2_32::ZERO; next_len];
        for j in 0..next_len {
            let from_shift = if j >= 1 && j - 1 < current.len() {
                current[j - 1].square()
            } else {
                Gf2_32::ZERO
            };
            let from_self = if j < current.len() {
                current[j]
            } else {
                Gf2_32::ZERO
            };
            next[j] = from_shift.add(from_self);
        }
        current = next;
    }

    polys
}

/// Evaluate a subspace polynomial `s_i` (in the sparse form returned
/// by [`compute_subspace_polynomials`]) at a `Gf2_32` point.
///
/// Uses repeated squaring: `X^(2^j) at α = α^(2^j)`. We accumulate
/// the powers of α as we go.
pub fn evaluate_subspace_poly_at_gf32(s_i: &[Gf2_32], alpha: Gf2_32) -> Gf2_32 {
    let mut acc = Gf2_32::ZERO;
    let mut power = alpha; // α^(2^0) = α
    for (j, &coeff) in s_i.iter().enumerate() {
        acc = acc.add(coeff.mul(power));
        if j + 1 < s_i.len() {
            power = power.square();
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    type P = Radix2AddFftParams<AddFftConfigGF2_32>;

    #[test]
    fn rejects_non_power_of_two_lengths() {
        assert!(P::new(7, 16).is_err());
        assert!(P::new(8, 17).is_err());
    }

    #[test]
    fn rejects_row_len_exceeding_codeword_len() {
        assert!(P::new(64, 32).is_err());
    }

    #[test]
    fn rejects_codeword_len_beyond_basis_dimension() {
        // 2^33 > 2^32 = number of basis elements available
        // (we can't allocate that many anyway, so test the bookkeeping
        // by instantiating just past the limit at log2.)
        assert!(P::new(8, 1usize << 33).is_err());
    }

    #[test]
    fn basic_construction() {
        let p = P::new(64, 256).expect("valid");
        assert_eq!(p.row_len, 64);
        assert_eq!(p.codeword_len, 256);
        assert_eq!(p.log2_codeword_len, 8);
        assert_eq!(p.eval_points.len(), 256);
        assert_eq!(p.subspace_polys.len(), 8);
        for (i, s_i) in p.subspace_polys.iter().enumerate() {
            assert_eq!(s_i.len(), i + 1, "s_{i} should have {} sparse coeffs", i + 1);
            assert_eq!(*s_i.last().unwrap(), Gf2_32::ONE, "s_{i} must be monic");
        }
    }

    /// `s_0(X) = X`, so `s_0(α) = α` for any α.
    #[test]
    fn s_0_is_identity() {
        let p = P::new(2, 4).expect("valid");
        let s_0 = &p.subspace_polys[0];
        for &alpha in &p.eval_points {
            assert_eq!(evaluate_subspace_poly_at_gf32(s_0, alpha), alpha);
        }
    }

    /// `s_i(β_i) = 1` for all `i ≥ 0` (Cantor-basis property).
    #[test]
    fn s_i_at_beta_i_equals_one() {
        let p = P::new(2, 1usize << 10).expect("valid"); // log2_codeword_len = 10
        for i in 0..p.log2_codeword_len {
            let s_i = &p.subspace_polys[i];
            let value = evaluate_subspace_poly_at_gf32(s_i, p.cantor_basis[i]);
            assert_eq!(value, Gf2_32::ONE, "s_{i}(β_{i}) ≠ 1");
        }
    }

    /// `s_i` vanishes on all of `V_i = span(β_0, …, β_{i-1})`.
    #[test]
    fn s_i_vanishes_on_v_i() {
        let m = 6usize;
        let p = P::new(2, 1usize << m).expect("valid");
        for i in 0..m {
            let s_i = &p.subspace_polys[i];
            // Enumerate V_i = {sum k_j β_j : k in 0..2^i}.
            for k in 0..(1usize << i) {
                let mut alpha = Gf2_32::ZERO;
                let mut bits = k;
                let mut j = 0usize;
                while bits != 0 {
                    if bits & 1 != 0 {
                        alpha = alpha.add(p.cantor_basis[j]);
                    }
                    bits >>= 1;
                    j += 1;
                }
                let v = evaluate_subspace_poly_at_gf32(s_i, alpha);
                assert_eq!(v, Gf2_32::ZERO, "s_i vanishes on V_i, k={k}, i={i}");
            }
        }
    }

    /// `eval_points[0] = 0` and `eval_points[k]` for `k = 2^i` is `β_i`.
    #[test]
    fn eval_points_match_basis() {
        let p = P::new(2, 1usize << 5).expect("valid");
        assert_eq!(p.eval_points[0], Gf2_32::ZERO);
        for i in 0..p.log2_codeword_len {
            assert_eq!(p.eval_points[1 << i], p.cantor_basis[i]);
        }
    }

    /// Cantor recurrence: `s_{i+1}(X) = s_i(X)² + s_i(X)`.
    /// Verified at points: `s_{i+1}(α) = s_i(α)² + s_i(α)` for all
    /// enumerated α.
    #[test]
    fn cantor_recurrence_holds_pointwise() {
        let p = P::new(2, 1usize << 7).expect("valid");
        for i in 0..(p.log2_codeword_len - 1) {
            let s_i = &p.subspace_polys[i];
            let s_i1 = &p.subspace_polys[i + 1];
            for &alpha in &p.eval_points {
                let v_i = evaluate_subspace_poly_at_gf32(s_i, alpha);
                let v_i1 = evaluate_subspace_poly_at_gf32(s_i1, alpha);
                assert_eq!(v_i1, v_i.square().add(v_i), "recurrence failed at i={i}");
            }
        }
    }
}
