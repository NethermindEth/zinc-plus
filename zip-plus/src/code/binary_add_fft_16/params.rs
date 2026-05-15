//! Precomputed parameters for the additive Reed-Solomon FFT over
//! `GF(2^16)` lifted to `Z[X] / f̃` with
//! `f̃ = X^16 + X^5 + X^3 + X^2 + 1`.
//!
//! Mirrors `binary_add_fft_8/params.rs` but for D = 16.

use crate::ZipError;
use std::{fmt::Debug, marker::PhantomData};

use super::basis::{D_16, F_TILDE_LOWER_DEGREES_16, Gf2_16, cantor_basis_16};

/// Configuration trait parameterising the binary additive FFT (D = 16).
pub trait Config16: Debug + Copy + PartialEq + Eq + Send + Sync + 'static {
    /// Extension degree of the binary base field: `GF(2^DEGREE)`.
    const DEGREE: usize;

    /// Non-leading monomial degrees of `f̃` (degrees `< DEGREE` with
    /// coefficient `1`).
    const F_TILDE_LOWER_DEGREES: &'static [usize];

    /// Compute the Cantor basis `{β_0, …, β_{DEGREE-1}}` in `GF(2^DEGREE)`.
    fn cantor_basis() -> [Gf2_16; D_16];
}

/// Concrete configuration for `GF(2^16) = F_2[X] / (X^16 + X^5 + X^3 + X^2 + 1)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AddFftConfigGF2_16;

impl Config16 for AddFftConfigGF2_16 {
    const DEGREE: usize = D_16;
    const F_TILDE_LOWER_DEGREES: &'static [usize] = &F_TILDE_LOWER_DEGREES_16;

    fn cantor_basis() -> [Gf2_16; D_16] {
        cantor_basis_16()
    }
}

/// Precomputed parameters for the radix-2 additive FFT.
///
/// We require `codeword_len.is_power_of_two()` and
/// `row_len.is_power_of_two()` with `row_len ≤ codeword_len ≤ 2^16 = 65_536`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Radix2AddFftParams16<C: Config16> {
    /// Number of input coefficients (a power of 2, `≤ codeword_len`).
    pub row_len: usize,

    /// Number of output evaluations (a power of 2, `≤ 2^16 = 65_536`).
    pub codeword_len: usize,

    /// `log2(codeword_len)`. This is also the number of butterfly
    /// stages the FFT performs and the dimension of the evaluation
    /// subspace `V_d`.
    pub log2_codeword_len: usize,

    /// Cantor basis `β_0, …, β_{D-1}` of `GF(2^16)`. The first
    /// `log2_codeword_len` elements span the evaluation subspace.
    pub cantor_basis: [Gf2_16; D_16],

    /// Evaluation points. `eval_points[k] = ∑_{i=0..log2_codeword_len} k_i · β_i`
    /// where `k_i` is bit `i` of `k`. Length: `codeword_len`.
    pub eval_points: Vec<Gf2_16>,

    /// Subspace polynomials `s_0, …, s_{log2_codeword_len - 1}` in
    /// sparse form. Each `s_i(X)` is `F_2`-linearised, with nonzero
    /// coefficients only at degrees `2^j` for `j = 0..=i`, so we
    /// store `subspace_polys[i][j]` = coefficient of `X^(2^j)` in
    /// `s_i(X)`. The leading coefficient `subspace_polys[i][i]` is
    /// always `Gf2_16::ONE`.
    pub subspace_polys: Vec<Vec<Gf2_16>>,

    _phantom: PhantomData<C>,
}

impl<C: Config16> Radix2AddFftParams16<C> {
    /// Construct precomputed parameters. Validates that `row_len` and
    /// `codeword_len` are both powers of two, that
    /// `row_len ≤ codeword_len`, and that `codeword_len ≤ 2^DEGREE = 65_536`.
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
        let eval_points = compute_eval_points_16(&cantor_basis, log2_codeword_len);
        let subspace_polys = compute_subspace_polynomials_16(&cantor_basis, log2_codeword_len);

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
pub fn compute_eval_points_16(basis: &[Gf2_16; D_16], log2_codeword_len: usize) -> Vec<Gf2_16> {
    let n = 1usize << log2_codeword_len;
    let mut points = Vec::with_capacity(n);
    for k in 0..n {
        let mut v = Gf2_16::ZERO;
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
/// `s_{i+1}(X) = s_i(X)² + s_i(X)`.
pub fn compute_subspace_polynomials_16(_basis: &[Gf2_16; D_16], m: usize) -> Vec<Vec<Gf2_16>> {
    let mut polys: Vec<Vec<Gf2_16>> = Vec::with_capacity(m);

    // s_0(X) = X, i.e. coefficient of X^(2^0) is 1.
    let mut current = vec![Gf2_16::ONE];

    for i in 0..m {
        polys.push(current.clone());

        // s_{i+1, j} = s_{i, j-1}² + s_{i, j}
        let next_len = i + 2;
        let mut next = vec![Gf2_16::ZERO; next_len];
        for j in 0..next_len {
            let from_shift = if j >= 1 && j - 1 < current.len() {
                current[j - 1].square()
            } else {
                Gf2_16::ZERO
            };
            let from_self = if j < current.len() {
                current[j]
            } else {
                Gf2_16::ZERO
            };
            next[j] = from_shift.add(from_self);
        }
        current = next;
    }

    polys
}

/// Evaluate a subspace polynomial `s_i` (in the sparse form returned
/// by [`compute_subspace_polynomials_16`]) at a `Gf2_16` point.
pub fn evaluate_subspace_poly_at_gf16(s_i: &[Gf2_16], alpha: Gf2_16) -> Gf2_16 {
    let mut acc = Gf2_16::ZERO;
    let mut power = alpha; // α^(2^0) = α
    for (j, &coeff) in s_i.iter().enumerate() {
        acc = acc.add(coeff.mul(power));
        if j + 1 < s_i.len() {
            power = power.square();
        }
    }
    acc
}

/// Vandermonde matrix for the additive FFT's base layer.
/// `matrix[k][i] = X_i(v_k)` in `GF(2^16)`.
/// Shape: `[codeword_len][2^base_log2]`.
pub fn compute_vandermonde_matrix_gf16(
    eval_points: &[Gf2_16],
    subspace_polys: &[Vec<Gf2_16>],
    base_log2: usize,
) -> Vec<Vec<Gf2_16>> {
    let n = eval_points.len();
    let base = 1usize << base_log2;

    let subspace_evals: Vec<Vec<Gf2_16>> = (0..base_log2)
        .map(|l| {
            eval_points
                .iter()
                .map(|&v_k| evaluate_subspace_poly_at_gf16(&subspace_polys[l], v_k))
                .collect()
        })
        .collect();

    (0..n)
        .map(|k| {
            (0..base)
                .map(|i| {
                    let mut x_i_at_v_k = Gf2_16::ONE;
                    for l in 0..base_log2 {
                        if (i >> l) & 1 != 0 {
                            x_i_at_v_k = x_i_at_v_k.mul(subspace_evals[l][k]);
                        }
                    }
                    x_i_at_v_k
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    type P = Radix2AddFftParams16<AddFftConfigGF2_16>;

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
        // 2^17 > 2^16 = number of basis elements available.
        assert!(P::new(8, 1usize << 17).is_err());
    }

    #[test]
    fn basic_construction() {
        let p = P::new(16, 1024).expect("valid");
        assert_eq!(p.row_len, 16);
        assert_eq!(p.codeword_len, 1024);
        assert_eq!(p.log2_codeword_len, 10);
        assert_eq!(p.eval_points.len(), 1024);
        assert_eq!(p.subspace_polys.len(), 10);
        for (i, s_i) in p.subspace_polys.iter().enumerate() {
            assert_eq!(s_i.len(), i + 1, "s_{i} should have {} sparse coeffs", i + 1);
            assert_eq!(*s_i.last().unwrap(), Gf2_16::ONE, "s_{i} must be monic");
        }
    }

    /// `s_0(X) = X`, so `s_0(α) = α` for any α.
    #[test]
    fn s_0_is_identity() {
        let p = P::new(2, 4).expect("valid");
        let s_0 = &p.subspace_polys[0];
        for &alpha in &p.eval_points {
            assert_eq!(evaluate_subspace_poly_at_gf16(s_0, alpha), alpha);
        }
    }

    /// `s_i(β_i) = 1` for all `i ≥ 0` (Cantor-basis property).
    #[test]
    fn s_i_at_beta_i_equals_one() {
        let p = P::new(2, 1usize << 8).expect("valid");
        for i in 0..p.log2_codeword_len {
            let s_i = &p.subspace_polys[i];
            let value = evaluate_subspace_poly_at_gf16(s_i, p.cantor_basis[i]);
            assert_eq!(value, Gf2_16::ONE, "s_{i}(β_{i}) ≠ 1");
        }
    }

    /// `s_i` vanishes on all of `V_i = span(β_0, …, β_{i-1})`.
    #[test]
    fn s_i_vanishes_on_v_i() {
        let m = 6usize;
        let p = P::new(2, 1usize << m).expect("valid");
        for i in 0..m {
            let s_i = &p.subspace_polys[i];
            for k in 0..(1usize << i) {
                let mut alpha = Gf2_16::ZERO;
                let mut bits = k;
                let mut j = 0usize;
                while bits != 0 {
                    if bits & 1 != 0 {
                        alpha = alpha.add(p.cantor_basis[j]);
                    }
                    bits >>= 1;
                    j += 1;
                }
                let v = evaluate_subspace_poly_at_gf16(s_i, alpha);
                assert_eq!(v, Gf2_16::ZERO, "s_i vanishes on V_i, k={k}, i={i}");
            }
        }
    }

    /// `eval_points[0] = 0` and `eval_points[k]` for `k = 2^i` is `β_i`.
    #[test]
    fn eval_points_match_basis() {
        let p = P::new(2, 1usize << 4).expect("valid");
        assert_eq!(p.eval_points[0], Gf2_16::ZERO);
        for i in 0..p.log2_codeword_len {
            assert_eq!(p.eval_points[1 << i], p.cantor_basis[i]);
        }
    }

    /// Cantor recurrence: `s_{i+1}(X) = s_i(X)² + s_i(X)`.
    #[test]
    fn cantor_recurrence_holds_pointwise() {
        let p = P::new(2, 1usize << 6).expect("valid");
        for i in 0..(p.log2_codeword_len - 1) {
            let s_i = &p.subspace_polys[i];
            let s_i1 = &p.subspace_polys[i + 1];
            for &alpha in &p.eval_points {
                let v_i = evaluate_subspace_poly_at_gf16(s_i, alpha);
                let v_i1 = evaluate_subspace_poly_at_gf16(s_i1, alpha);
                assert_eq!(v_i1, v_i.square().add(v_i), "recurrence failed at i={i}");
            }
        }
    }
}
