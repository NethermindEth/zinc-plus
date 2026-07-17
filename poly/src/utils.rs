use crypto_primitives::SemiringConfig;
use thiserror::Error;
use zinc_utils::{add, mul, sub};

use crate::mle::{DenseMultilinearExtension, dense::CollectDenseMleWithZero};

/// A `enum` specifying the possible failure modes of the arithmetics.
#[derive(Debug, Clone, Error)]
pub enum ArithErrors {
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
}

/// This function build the eq(x, r) polynomial for any given r.
///
/// Evaluate
///      eq(x,y) = \prod_i=1^num_var (x_i * y_i + (1-x_i)*(1-y_i))
/// over r, which is
///      eq(x,y) = \prod_i=1^num_var (x_i * r_i + (1-x_i)*(1-r_i))
pub fn build_eq_x_r<C: SemiringConfig>(
    cfg: &C,
    r: &[C::Element],
) -> Result<DenseMultilinearExtension<C::Element>, ArithErrors> {
    let evals = build_eq_x_r_vec(cfg, r)?;
    let mle = DenseMultilinearExtension::from_evaluations_vec(r.len(), evals, cfg.zero());

    Ok(mle)
}

/// This function builds the eq(x, r) polynomial for any given r, and outputs
/// the evaluation of eq(x, r) in its vector form.
///
/// Evaluate
///      $eq(x,y) = \prod_i=1^num_var (x_i * y_i + (1-x_i)*(1-y_i))$
/// over r, which is
///      $eq(x,y) = \prod_i=1^num_var (x_i * r_i + (1-x_i)*(1-r_i))$
pub fn build_eq_x_r_vec<C: SemiringConfig>(
    cfg: &C,
    r: &[C::Element],
) -> Result<Vec<C::Element>, ArithErrors> {
    // we build eq(x,r) from its evaluations
    // we want to evaluate eq(x,r) over x \in {0, 1}^num_vars
    // for example, with num_vars = 4, x is a binary vector of 4, then
    //  0 0 0 0 -> (1-r0)   * (1-r1)    * (1-r2)    * (1-r3)
    //  1 0 0 0 -> r0       * (1-r1)    * (1-r2)    * (1-r3)
    //  0 1 0 0 -> (1-r0)   * r1        * (1-r2)    * (1-r3)
    //  1 1 0 0 -> r0       * r1        * (1-r2)    * (1-r3)
    //  ....
    //  1 1 1 1 -> r0       * r1        * r2        * r3
    // we will need 2^num_var evaluations

    if r.is_empty() {
        return Err(ArithErrors::InvalidParameters("r length is 0".to_owned()));
    }

    let one = cfg.one();
    let mut eval = vec![one; 1 << r.len()];
    let mut s = 1;
    for ri in r {
        for j in (0..s).rev() {
            let prev = eval[j].clone();
            let hi = cfg.mul(&prev, ri);
            eval[j] = cfg.sub(&prev, &hi);
            eval[add!(j, s)] = hi;
        }
        s = mul!(s, 2);
    }

    Ok(eval)
}

/// Build the shift selector MLE `next_c_mle(r, *)` with the first `num_vars`
/// variables fixed to `r`.
///
/// For each `b in {0,1}^{num_vars}`:
///   next_c_mle(b) = eq(r, b - c)   if b >= c
///   next_c_mle(b) = 0              if b < c
///
/// Uses the identity `next_c_mle(r, b) = eq(r, b - c)` for `b >= c` and
/// `0` for `b < c`.
pub fn build_next_c_r_mle<C: SemiringConfig>(
    cfg: &C,
    r: &[C::Element],
    c: usize,
) -> Result<DenseMultilinearExtension<C::Element>, ArithErrors> {
    let num_vars = r.len();
    let n = 1 << num_vars;
    assert!(c < n, "shift c={c} must be < domain size {n}");

    let eq_r = build_eq_x_r(cfg, r)?;
    if c == 0 {
        return Ok(eq_r);
    }

    // next_c_mle(r, 0) = 0 for b < c
    // next_c_mle(r, b - c) = eq(r, b - c) for b >= c
    let mut evaluations = Vec::with_capacity(n);
    evaluations.resize(c, cfg.zero());
    evaluations.extend_from_slice(&eq_r.evaluations[..sub!(n, c)]);

    Ok(DenseMultilinearExtension {
        num_vars,
        evaluations,
    })
}

/// Evaluate eq polynomial.
pub fn eq_eval<C: SemiringConfig>(
    cfg: &C,
    x: &[C::Element],
    y: &[C::Element],
) -> Result<C::Element, ArithErrors> {
    if x.len() != y.len() {
        return Err(ArithErrors::InvalidParameters(
            "x and y have different length".to_string(),
        ));
    }

    let one = cfg.one();
    let mut res = one.clone();
    for (xi, yi) in x.iter().zip(y.iter()) {
        // xi * yi + (1 - xi) * (1 - yi) = 2 * xi * yi - xi - yi + 1
        let xi_yi = cfg.mul(xi, yi);
        let mut term = cfg.add(&xi_yi, &xi_yi);
        cfg.sub_assign(&mut term, xi);
        cfg.sub_assign(&mut term, yi);
        cfg.add_assign(&mut term, &one);
        cfg.mul_assign(&mut res, &term);
    }

    Ok(res)
}

/// Evaluate an MLE at a point using a precomputed eq table.
///
/// Given `evaluations[b]` and `eq_table[b] = eq(b, r)` (precomputed via
/// [`build_eq_x_r_vec`]), returns `\sum_{b} eq_table[b] * evaluations[b]`.
///
/// This is equivalent to `DenseMultilinearExtension::evaluate`
/// but avoids cloning the evaluation vector (the fix-variables algorithm is
/// destructive). When multiple MLEs share the same evaluation point, build the
/// eq table once and call this function for each MLE.
pub fn mle_eval_with_eq_table<C: SemiringConfig>(
    cfg: &C,
    evaluations: &[C::Element],
    eq_table: &[C::Element],
) -> C::Element {
    let mut acc = cfg.zero();
    assert_eq!(
        evaluations.len(),
        eq_table.len(),
        "evaluations and eq_table must have the same length"
    );
    for (eval, eq_val) in evaluations.iter().zip(eq_table.iter()) {
        let term = cfg.mul(eq_val, eval);
        cfg.add_assign(&mut acc, &term);
    }
    acc
}

/// Returns a multilinear polynomial in 2n variables that evaluates to 1
/// if and only if the second n-bit vector is equal to the first vector plus one
#[allow(clippy::arithmetic_side_effects)]
pub fn next_mle<E: Clone>(
    num_vars: u32,
    zero: E,
    one: E,
) -> Result<DenseMultilinearExtension<E>, ArithErrors> {
    if !num_vars.is_multiple_of(2) {
        return Err(ArithErrors::InvalidParameters(
            "num_vars must be even".to_string(),
        ));
    }

    let mut mle = (0..1 << num_vars)
        .map(|_| zero.clone())
        .collect_dense_mle_with_zero(&zero);

    let half_vars = num_vars / 2;

    for i in 0usize..(1 << half_vars) - 1 {
        let next = i + 1;

        let i_concat_next = (next << half_vars) | i;

        mle[i_concat_next] = one.clone();
    }

    Ok(mle)
}

/// Evaluates the next MLE in O(n), by reusing suffix equality and prefix carry
/// products across carry positions.
///
/// Improved from O(n²) approach here: https://github.com/TomWambsgans/Whirlaway/blob/9e3592b/crates/air/src/utils.rs#L92
///
/// `next_mle(u, v) = 1` iff `Val(v) = Val(u) + 1` and `Val(u) < 2^n - 1`.
///
/// # Arguments
/// - `u`: first n-bit vector (LE convention: index 0 = LSB).
/// - `v`: second n-bit vector. Must have `v.len() == u.len()`.
///
/// # Algorithm
/// Uses prefix/suffix products for O(n) evaluation:
///   `next_mle(u, v) = sum_{j=0}^{n-1}
///       [prod_{i<j} u_i * (1 - v_i)]      -- bits below j: were 1, flip to 0
///     * (1 - u_j) * v_j                   -- bit j: 0 → 1
///     * [prod_{i>j} eq(u_i, v_i)]`        -- bits above j: unchanged
///
/// # Panics
/// Panics if `u.len() != v.len()`.
#[allow(clippy::arithmetic_side_effects)]
pub fn next_mle_eval<C: SemiringConfig>(cfg: &C, u: &[C::Element], v: &[C::Element]) -> C::Element {
    let n = u.len();
    assert_eq!(n, v.len(), "u and v must have the same length");
    if n == 0 {
        return cfg.zero();
    }

    let one = cfg.one();

    // suffix_eq[j] = prod_{i=j}^{n-1} eq(u_i, v_i)
    let mut suffix_eq = vec![one.clone(); n + 1];
    for i in (0..n).rev() {
        // eq(u_i, v_i) = u_i * v_i + (1 - u_i) * (1 - v_i)
        let uv = cfg.mul(&u[i], &v[i]);
        let eq_i = cfg.add(&uv, &cfg.mul(&cfg.sub(&one, &u[i]), &cfg.sub(&one, &v[i])));
        suffix_eq[i] = cfg.mul(&suffix_eq[i + 1], &eq_i);
    }

    // prefix_carry accumulates prod_{i<j} u_i * (1 - v_i)
    let mut prefix_carry = one.clone();
    let mut result = cfg.zero();
    for j in 0..n {
        // prefix_carry * (1 - u_j) * v_j * suffix_eq[j + 1]
        let mut term = cfg.mul(&prefix_carry, &cfg.sub(&one, &u[j]));
        cfg.mul_assign(&mut term, &v[j]);
        cfg.mul_assign(&mut term, &suffix_eq[j + 1]);
        cfg.add_assign(&mut result, &term);

        let carry = cfg.mul(&u[j], &cfg.sub(&one, &v[j]));
        cfg.mul_assign(&mut prefix_carry, &carry);
    }
    result
}

#[cfg(test)]
#[allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::needless_range_loop
)]
mod tests {
    use crypto_bigint::{U128, const_monty_params};
    use crypto_primitives::{FixedConfig, crypto_bigint_const_monty::ConstMontyField};
    use num_traits::{One, Zero};
    use proptest::{prelude::*, proptest};

    use super::*;

    const_monty_params!(Params, U128, "00000000b933426489189cb5b47d567f");

    type F = ConstMontyField<Params, { U128::LIMBS }>;

    const NUM_VARS: u32 = 8;

    fn cfg() -> FixedConfig<F> {
        FixedConfig::default()
    }

    #[test]
    fn build_eq_x_r_vec_matches_product_formula() {
        // For each b in {0,1}^n, eq(b, r) = prod_i (b_i * r_i + (1-b_i) * (1-r_i)).
        // The helper uses little-endian indexing: bit i of the index is x_i.
        let r: Vec<F> = (0..NUM_VARS).map(|i| F::from(i + 11)).collect();
        let eq_vec = build_eq_x_r_vec(&cfg(), &r).unwrap();

        let n = 1usize << NUM_VARS;
        assert_eq!(eq_vec.len(), n);

        let one = F::one();
        for b in 0..n {
            let mut expected = one;
            for (i, ri) in r.iter().enumerate() {
                let bit = (b >> i) & 1;
                expected *= if bit == 1 { *ri } else { one - ri };
            }
            assert_eq!(eq_vec[b], expected, "mismatch at b={b}");
        }
    }

    #[test]
    fn build_eq_x_r_vec_basic() {
        let r: [F; _] = [F::from(3_u64)];
        let evals = build_eq_x_r_vec(&cfg(), &r).unwrap();
        assert_eq!(evals, vec![F::one() - r[0], r[0]]);
    }

    #[test]
    fn build_eq_x_r_vec_two_vars() {
        let r: [F; _] = [F::from(2_u64), F::from(5_u64)];
        let evals = build_eq_x_r_vec(&cfg(), &r).unwrap();
        let e00 = (F::one() - r[0]) * (F::one() - r[1]);
        let e01 = r[0] * (F::one() - r[1]);
        let e10 = (F::one() - r[0]) * r[1];
        let e11 = r[0] * r[1];
        assert_eq!(evals, vec![e00, e01, e10, e11]);
    }

    #[test]
    fn build_eq_x_r_error_on_empty() {
        let r: [F; 0] = [];
        let err = build_eq_x_r_vec(&cfg(), &r).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("Invalid parameters"));
    }

    #[test]
    fn build_eq_x_r_mle_properties() {
        let r: [F; _] = [F::from(7_u64), F::from(11_u64), F::from(13_u64)];
        let mle = build_eq_x_r(&cfg(), &r).unwrap();
        assert_eq!(mle.num_vars, r.len());
        let evals = mle.evaluations;
        let direct = build_eq_x_r_vec(&cfg(), &r).unwrap();
        assert_eq!(evals, direct);
    }

    #[test]
    fn next_mle_is_one_on_successors() {
        let next_mle = next_mle(NUM_VARS, F::zero(), F::one()).unwrap();

        for i in 0..(1 << ((NUM_VARS / 2) - 1)) {
            let mut point: Vec<F> = (0..(NUM_VARS / 2))
                .map(|j| {
                    if i & (1 << j) == 0 {
                        F::zero()
                    } else {
                        F::one()
                    }
                })
                .collect();

            point.extend((0..(NUM_VARS / 2)).map(|j| {
                if (i + 1) & (1 << j) == 0 {
                    F::zero()
                } else {
                    F::one()
                }
            }));

            assert_eq!(next_mle.clone().evaluate(&cfg(), &point), Ok(F::one()));
        }
    }

    #[test]
    fn next_mle_is_one_only_on_successors() {
        let next_mle = next_mle(NUM_VARS, F::zero(), F::one()).unwrap();

        // The number of successors is (1 << (num_vars / 2)) - 1
        // and we know the mle is one on them. So we need to check
        // that it is one only on that many points.
        assert_eq!(
            next_mle.evaluations.iter().filter(|x| !x.is_zero()).count(),
            (1 << (NUM_VARS / 2)) - 1
        );
    }

    fn any_f() -> impl Strategy<Value = F> + 'static {
        any::<u128>().prop_map(F::from)
    }

    fn point_n(n: usize) -> impl Strategy<Value = Vec<F>> {
        prop::collection::vec(any_f(), n)
    }

    #[test]
    fn next_mle_eval_coincides_with_next_mle_evaluated_at_successors() {
        let next_mle = next_mle(NUM_VARS, F::zero(), F::one()).unwrap();

        for i in 0..(1 << ((NUM_VARS / 2) - 1)) {
            let mut point: Vec<F> = (0..(NUM_VARS / 2))
                .map(|j| {
                    if i & (1 << j) == 0 {
                        F::zero()
                    } else {
                        F::one()
                    }
                })
                .collect();

            point.extend((0..(NUM_VARS / 2)).map(|j| {
                if (i + 1) & (1 << j) == 0 {
                    F::zero()
                } else {
                    F::one()
                }
            }));

            let (u, v) = point.split_at(NUM_VARS as usize / 2);
            assert_eq!(
                next_mle.clone().evaluate(&cfg(), &point),
                Ok(next_mle_eval(&cfg(), u, v))
            );
        }
    }

    proptest! {
    #[test]
    #[cfg_attr(miri, ignore)] // long running
    fn prop_next_mle_eval_coincides_with_next_mle_evaluate_at_point(r in point_n(NUM_VARS as usize)) {
        let next_mle = next_mle(NUM_VARS, F::zero(), F::one()).unwrap();

        let (u, v) = r.split_at(NUM_VARS as usize / 2);
        prop_assert_eq!(
            next_mle.evaluate(&cfg(), &r),
            Ok(next_mle_eval(&cfg(), u, v))
        );
    }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // long running
    fn next_c_r_mle_c1_matches_shift_by_1() {
        // c=1 should give the same result as the original build_next_r_mle
        let num_vars: usize = 4;
        let r: Vec<F> = (0..num_vars).map(|i| F::from((i + 3) as u32)).collect();

        let next_1 = build_next_c_r_mle(&cfg(), &r, 1).unwrap();

        // Manually build shift-by-1: evaluations[0] = 0, evaluations[b] = eq(r, b-1)
        let eq_r = build_eq_x_r(&cfg(), &r).unwrap();
        let n = 1 << num_vars;
        let mut expected = vec![F::zero(); 1];
        expected.extend_from_slice(&eq_r.evaluations[..n - 1]);

        assert_eq!(next_1.evaluations, expected);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // long running
    fn next_c_r_mle_c0_is_eq() {
        // c=0 should return eq(r, b)
        let num_vars: usize = 4;
        let r: Vec<F> = (0..num_vars).map(|i| F::from((i + 7) as u32)).collect();

        let next_0 = build_next_c_r_mle(&cfg(), &r, 0).unwrap();
        let eq_r = build_eq_x_r(&cfg(), &r).unwrap();

        assert_eq!(next_0.evaluations, eq_r.evaluations);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // long running
    fn next_c_r_mle_has_correct_structure() {
        // For any c, evaluations[b] should be:
        //   0 for b < c
        //   eq(r, b-c) for b >= c
        let num_vars: usize = 4;
        let n = 1 << num_vars;
        let r: Vec<F> = (0..num_vars).map(|i| F::from((i + 5) as u32)).collect();

        for c in [2, 3, 5, 7] {
            let next_c = build_next_c_r_mle(&cfg(), &r, c).unwrap();
            let eq_r = build_eq_x_r(&cfg(), &r).unwrap();

            // First c entries should be zero
            for b in 0..c {
                assert!(
                    next_c.evaluations[b].is_zero(),
                    "c={c}, b={b}: expected zero"
                );
            }
            // Remaining entries should match eq(r, b-c)
            for b in c..n {
                assert_eq!(
                    next_c.evaluations[b],
                    eq_r.evaluations[b - c],
                    "c={c}, b={b}: mismatch"
                );
            }
        }
    }

    proptest! {
    #[test]
    #[cfg_attr(miri, ignore)] // long running
    fn prop_next_c_r_mle_evaluates_correctly(r in point_n(4), c in 1..15usize) {
        // build_next_c_r_mle(r, c) evaluated at random point should equal
        // the shift-c predicate: sum_b next_c(b) * eq(b, point)
        let next_c = build_next_c_r_mle(&cfg(), &r, c).unwrap();
        let eq_r = build_eq_x_r(&cfg(), &r).unwrap();

        // Verify the table structure holds
        let n = 1 << r.len();
        for b in 0..c.min(n) {
            prop_assert!(next_c.evaluations[b].is_zero());
        }
        for b in c..n {
            prop_assert_eq!(&next_c.evaluations[b], &eq_r.evaluations[b - c]);
        }
    }
    }
}
