//! Efficient evaluation-form product of multilinear polynomials, via
//! recursive extrapolation — Dao et al. "Speeding Up Sum-Check Proving"
//! (cs.nyu.edu/~zd2131/papers/26-587.pdf, §4), Procedures 1
//! (`MultiProductEval`) and 2 (`MultiExtrapolate`).
//!
//! Given `d` multilinear polynomials in `v` variables by their evaluations
//! over the Boolean cube `{0,1}^v` (the natural grid `U_1^v`), this returns
//! the product `g = Π_i p_i`'s evaluations over `U_d^v`, where
//! `U_m = [F::from(0), F::from(1), …, F::from(m)]` is the `(m+1)`-point
//! natural grid (over `GF(2^128)` this is `[0, 1, X, X+1, …]`, the same
//! boundary points the multi-degree sumcheck uses). The recursive halving
//! keeps the extrapolation work at `O(d log d)` big-field multiplications,
//! versus the naïve `O(d^2)`-per-grid-point evaluation.
//!
//! Grids are stored row-major as a flat `Vec<F>`: a `v`-dimensional grid
//! whose axis `j` has size `s_j` uses index `Σ_j idx_j · Π_{l<j} s_l`
//! (axis 0 least significant). Every input axis has size 2 (`{0,1}`); every
//! output axis has size `d+1`.
//!
//! Univariate extrapolation reuses [`NatEvaluatedPoly`], whose evaluations
//! are taken at exactly `F::from(0), F::from(1), …` — so the first `k+1`
//! grid values along an axis are the input and the remaining `d-k` are the
//! interpolant evaluated at `F::from(k+1), …, F::from(d)`.

use crypto_primitives::FromPrimitiveWithConfig;
use zinc_poly::univariate::nat_evaluation::{EvalAux, NatEvaluatedPoly};

/// Extrapolate axis `axis` of the `v`-dim grid `evals` (current per-axis
/// `sizes`) from `sizes[axis]` evaluations up to `new_size`. The first
/// `sizes[axis]` values along each axis line are copied (the natural grid
/// is a prefix); the rest are the unique degree-`(sizes[axis]-1)`
/// interpolant evaluated at `new_pts`. `aux` must be
/// `NatEvaluatedPoly::prepare_eval_aux(sizes[axis], cfg)` and `new_pts` must
/// be `[F::from(sizes[axis]), …, F::from(new_size-1)]`.
#[allow(clippy::arithmetic_side_effects)]
fn extrapolate_axis<F: FromPrimitiveWithConfig>(
    evals: &[F],
    sizes: &[usize],
    axis: usize,
    new_size: usize,
    aux: &EvalAux<F>,
    new_pts: &[F],
) -> Vec<F> {
    let old = sizes[axis];
    let inner: usize = sizes[..axis].iter().product();
    let outer: usize = sizes[axis + 1..].iter().product();
    let mut out = vec![evals[0].clone(); inner * new_size * outer];
    for o in 0..outer {
        for i in 0..inner {
            // Gather the `old` values along this axis line, in order.
            let col: Vec<F> = (0..old)
                .map(|a| evals[(o * old + a) * inner + i].clone())
                .collect();
            let poly = NatEvaluatedPoly::new(col);
            for a in 0..new_size {
                let val = if a < old {
                    poly.evaluations[a].clone()
                } else {
                    poly.evaluate_at_point_with_aux(&new_pts[a - old], aux)
                        .expect("non-empty interpolant")
                };
                out[(o * new_size + a) * inner + i] = val;
            }
        }
    }
    out
}

/// Procedure 2: multivariate polynomial extrapolation. Given `evals` over
/// `U_k^v` (per-axis size `k+1`, row-major), return its evaluations over
/// `U_d^v` (per-axis size `d+1`), `d >= k`, by extrapolating one axis at a
/// time.
#[allow(clippy::arithmetic_side_effects)]
pub fn multi_extrapolate<F: FromPrimitiveWithConfig>(
    mut evals: Vec<F>,
    v: usize,
    k: usize,
    d: usize,
    config: &F::Config,
) -> Vec<F> {
    debug_assert_eq!(evals.len(), (k + 1).pow(v as u32));
    if k >= d {
        return evals;
    }
    let aux = NatEvaluatedPoly::<F>::prepare_eval_aux(k + 1, config);
    let new_pts: Vec<F> = ((k + 1)..=d)
        .map(|m| F::from_with_cfg(m as u64, config))
        .collect();
    let mut sizes = vec![k + 1; v];
    for axis in 0..v {
        evals = extrapolate_axis(&evals, &sizes, axis, d + 1, &aux, &new_pts);
        sizes[axis] = d + 1;
    }
    evals
}

/// Procedure 1: efficient product of multilinear polynomials in evaluation
/// form. `polys[i]` holds `p_i`'s evaluations over `{0,1}^v` (length `2^v`,
/// row-major). Returns `g = Π_i p_i`'s evaluations over `U_d^v` (length
/// `(d+1)^v`), where `d = polys.len()`. Halves the factor set recursively,
/// extrapolates each partial product up to `U_d^v`, then multiplies
/// pointwise.
#[allow(clippy::arithmetic_side_effects)]
pub fn multi_product_eval<F: FromPrimitiveWithConfig>(
    polys: &[Vec<F>],
    v: usize,
    config: &F::Config,
) -> Vec<F> {
    let d = polys.len();
    assert!(d >= 1, "need at least one polynomial");
    if d == 1 {
        debug_assert_eq!(polys[0].len(), 2usize.pow(v as u32));
        // A single multilinear's `{0,1}^v` evals are already its `U_1^v` evals.
        return polys[0].clone();
    }
    let m = d / 2;
    let q_l = multi_product_eval(&polys[..m], v, config); // over U_m^v
    let q_r = multi_product_eval(&polys[m..], v, config); // over U_{d-m}^v
    let q_l = multi_extrapolate(q_l, v, m, d, config); // → U_d^v
    let q_r = multi_extrapolate(q_r, v, d - m, d, config); // → U_d^v
    q_l.iter()
        .zip(&q_r)
        .map(|(a, b)| a.clone() * b.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_primitives::{Field, FromWithConfig};
    use zinc_poly::univariate::binary_gf128::BinaryFieldGF128;

    type Gf = BinaryFieldGF128;

    /// Naïve reference: product of `d` multilinears evaluated at each grid
    /// point of `U_d^v`, each multilinear via its multilinear extension
    /// `Σ_{c∈{0,1}^v} (Π_j eq1(pt_j, c_j))·p[c]`.
    fn naive_grid(polys: &[Vec<Gf>], v: usize) -> Vec<Gf> {
        let cfg = &();
        let d = polys.len();
        let size = d + 1;
        let pts: Vec<Gf> = (0..size).map(|m| Gf::from_with_cfg(m as u64, cfg)).collect();
        let eq1 = |x: Gf, c: usize| if c == 1 { x } else { Gf::one() - x };
        (0..size.pow(v as u32))
            .map(|gi| {
                // Decode the grid multi-index (axis 0 least significant).
                let mut idx = vec![0usize; v];
                let mut t = gi;
                for slot in idx.iter_mut() {
                    *slot = t % size;
                    t /= size;
                }
                let mut prod = Gf::one();
                for p in polys {
                    let mut val = Gf::zero();
                    for c in 0..(1usize << v) {
                        let mut w = Gf::one();
                        for (j, &ij) in idx.iter().enumerate() {
                            w = w * eq1(pts[ij], (c >> j) & 1);
                        }
                        val = val + w * p[c];
                    }
                    prod = prod * val;
                }
                prod
            })
            .collect()
    }

    #[test]
    fn multiproduct_matches_naive_v2_d3() {
        let cfg = &();
        let v = 2;
        let d = 3;
        // d distinct (non-Boolean) multilinears over {0,1}^2.
        let polys: Vec<Vec<Gf>> = (0..d)
            .map(|p| {
                (0..(1usize << v))
                    .map(|c| Gf::from_words([(p * 7 + c * 3 + 1) as u64, (c + 1) as u64]))
                    .collect()
            })
            .collect();
        assert_eq!(
            multi_product_eval(&polys, v, cfg),
            naive_grid(&polys, v),
            "Procedure-1 product over U_3^2 must match the naive grid product"
        );
    }

    #[test]
    fn multiproduct_matches_naive_v3_d4() {
        let cfg = &();
        let v = 3;
        let d = 4;
        let polys: Vec<Vec<Gf>> = (0..d)
            .map(|p| {
                (0..(1usize << v))
                    .map(|c| Gf::from_words([(p * 11 + c * 5 + 2) as u64, (p + c) as u64]))
                    .collect()
            })
            .collect();
        assert_eq!(
            multi_product_eval(&polys, v, cfg),
            naive_grid(&polys, v),
            "Procedure-1 product over U_4^3 must match the naive grid product"
        );
    }

    #[test]
    fn multiproduct_single_factor_is_identity() {
        let cfg = &();
        let v = 2;
        let p: Vec<Gf> = (0..4).map(|c| Gf::from_words([c as u64, 1])).collect();
        // d=1 → over U_1^2 = {0,1}^2, unchanged.
        assert_eq!(multi_product_eval(std::slice::from_ref(&p), v, cfg), p);
    }
}
