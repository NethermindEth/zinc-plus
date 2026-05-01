use crypto_primitives::{Field, PrimeField, Semiring};
use num_traits::Zero;
use thiserror::Error;
use zinc_utils::{cfg_iter_mut, inner_transparent_field::InnerTransparentField, sub};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

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
pub fn build_eq_x_r<F>(
    r: &[F],
    cfg: &F::Config,
) -> Result<DenseMultilinearExtension<F>, ArithErrors>
where
    F: PrimeField,
{
    let evals = build_eq_x_r_vec(r, cfg)?;
    let mle =
        DenseMultilinearExtension::from_evaluations_vec(r.len(), evals, F::zero_with_cfg(cfg));

    Ok(mle)
}

/// This function build the eq(x, r) polynomial for any given r, and output the
/// evaluation of eq(x, r) in its vector form.
///
/// Evaluate
///      eq(x,y) = \prod_i=1^num_var (x_i * y_i + (1-x_i)*(1-y_i))
/// over r, which is
///      eq(x,y) = \prod_i=1^num_var (x_i * r_i + (1-x_i)*(1-r_i))
pub fn build_eq_x_r_vec<F>(r: &[F], cfg: &F::Config) -> Result<Vec<F>, ArithErrors>
where
    F: PrimeField,
{
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

    let mut eval = Vec::new();
    build_eq_x_r_helper(r, &mut eval, cfg)?;

    Ok(eval)
}

/// A helper function to build eq(x, r) recursively.
/// This function takes `r.len()` steps, and for each step it requires a maximum
/// `r.len()-1` multiplications.
fn build_eq_x_r_helper<F>(r: &[F], buf: &mut Vec<F>, cfg: &F::Config) -> Result<(), ArithErrors>
where
    F: PrimeField,
{
    if r.is_empty() {
        return Err(ArithErrors::InvalidParameters("r length is 0".into()));
    } else if r.len() == 1 {
        // initializing the buffer with [1-r_0, r_0]
        buf.push(F::one_with_cfg(cfg) - &r[0]);
        buf.push(r[0].clone());
    } else {
        build_eq_x_r_helper(&r[1..], buf, cfg)?;

        // suppose at the previous step we received [b_1, ..., b_k]
        // for the current step we will need
        // if x_0 = 0:   (1-r0) * [b_1, ..., b_k]
        // if x_0 = 1:   r0 * [b_1, ..., b_k]

        let mut res = vec![F::zero_with_cfg(cfg); buf.len() << 1];
        cfg_iter_mut!(res).enumerate().for_each(|(i, val)| {
            let bi = buf[i >> 1].clone();
            let tmp = r[0].clone() * &bi;
            if (i & 1) == 0 {
                *val = bi - tmp;
            } else {
                *val = tmp;
            }
        });
        *buf = res;
    }

    Ok(())
}

/// This function build the eq(x, r) polynomial for any given r.
///
/// Evaluate
///      eq(x,y) = \prod_i=1^num_var (x_i * y_i + (1-x_i)*(1-y_i))
/// over r, which is
///      eq(x,y) = \prod_i=1^num_var (x_i * r_i + (1-x_i)*(1-r_i))
pub fn build_eq_x_r_inner<F>(
    r: &[F],
    cfg: &F::Config,
) -> Result<DenseMultilinearExtension<F::Inner>, ArithErrors>
where
    F: PrimeField,
    F::Inner: Zero,
{
    let evals = build_eq_x_r_inner_vec(r, cfg)?;
    let mle = DenseMultilinearExtension {
        num_vars: r.len(),
        evaluations: evals,
    };

    Ok(mle)
}

/// This function build the eq(x, r) polynomial for any given r, and output the
/// evaluation of eq(x, r) in its vector form.
///
/// Evaluate
///      eq(x,y) = \prod_i=1^num_var (x_i * y_i + (1-x_i)*(1-y_i))
/// over r, which is
///      eq(x,y) = \prod_i=1^num_var (x_i * r_i + (1-x_i)*(1-r_i))
fn build_eq_x_r_inner_vec<F>(r: &[F], cfg: &F::Config) -> Result<Vec<F::Inner>, ArithErrors>
where
    F: PrimeField,
    F::Inner: Zero,
{
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

    let mut eval = Vec::new();
    build_eq_x_r_inner_helper(r, &mut eval, cfg)?;

    Ok(eval)
}

/// A helper function to build eq(x, r) recursively.
/// This function takes `r.len()` steps, and for each step it requires a maximum
/// `r.len()-1` multiplications.
fn build_eq_x_r_inner_helper<F>(
    r: &[F],
    buf: &mut Vec<F::Inner>,
    cfg: &F::Config,
) -> Result<(), ArithErrors>
where
    F: PrimeField,
    F::Inner: Zero,
{
    let one = F::one_with_cfg(cfg);
    if r.is_empty() {
        return Err(ArithErrors::InvalidParameters("r length is 0".into()));
    } else if r.len() == 1 {
        // initializing the buffer with [1-r_0, r_0]
        buf.push((one - &r[0]).into_inner());
        buf.push(r[0].inner().clone());
    } else {
        build_eq_x_r_inner_helper(&r[1..], buf, cfg)?;

        // suppose at the previous step we received [b_1, ..., b_k]
        // for the current step we will need
        // if x_0 = 0:   (1-r0) * [b_1, ..., b_k]
        // if x_0 = 1:   r0 * [b_1, ..., b_k]

        let mut res = vec![F::Inner::zero(); buf.len() << 1];
        cfg_iter_mut!(res).enumerate().for_each(|(i, val)| {
            let bi = F::new_unchecked_with_cfg(buf[i >> 1].clone(), cfg);
            let tmp = r[0].clone() * &bi;
            if (i & 1) == 0 {
                *val = (bi - tmp).into_inner();
            } else {
                *val = tmp.into_inner();
            }
        });
        *buf = res;
    }

    Ok(())
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
pub fn build_next_c_r_mle<F>(
    r: &[F],
    c: usize,
    field_cfg: &F::Config,
) -> Result<DenseMultilinearExtension<F::Inner>, ArithErrors>
where
    F: PrimeField,
    F::Inner: Zero,
{
    let num_vars = r.len();
    let n = 1 << num_vars;
    assert!(c < n, "shift c={c} must be < domain size {n}");
    let zero_inner = F::zero_with_cfg(field_cfg).into_inner();

    let eq_r = build_eq_x_r_inner(r, field_cfg)?;
    if c == 0 {
        return Ok(eq_r);
    }

    // next_c_mle(r, 0) = 0 for b < c
    // next_c_mle(r, b - c) = eq(r, b - c) for b >= c
    let mut evaluations = Vec::with_capacity(n);
    evaluations.resize(c, zero_inner);
    evaluations.extend_from_slice(&eq_r.evaluations[..sub!(n, c)]);

    Ok(DenseMultilinearExtension {
        num_vars,
        evaluations,
    })
}

/// Evaluate eq polynomial.
#[allow(clippy::arithmetic_side_effects)]
pub fn eq_eval<R: Semiring>(x: &[R], y: &[R], one: R) -> Result<R, ArithErrors> {
    if x.len() != y.len() {
        return Err(ArithErrors::InvalidParameters(
            "x and y have different length".to_string(),
        ));
    }

    let mut res = one.clone();
    for (xi, yi) in x.iter().zip(y.iter()) {
        let xi_yi = xi.clone() * yi;
        res *= xi_yi.clone() + xi_yi - xi - yi + one.clone();
    }

    Ok(res)
}

/// Evaluate an MLE at a point using a precomputed eq table.
///
/// Given `evaluations[b]` (in `F::Inner` form) and `eq_table[b] = eq(b, r)`
/// (precomputed via [`build_eq_x_r_vec`]), returns `\sum_{b} eq_table[b] *
/// evaluations[b]`.
///
/// This is equivalent to `DenseMultilinearExtension::evaluate_with_config`
/// but avoids cloning the evaluation vector (the fix-variables algorithm is
/// destructive). When multiple MLEs share the same evaluation point, build the
/// eq table once and call this function for each MLE.
#[allow(clippy::arithmetic_side_effects)]
pub fn mle_eval_with_eq_table<F: InnerTransparentField>(
    evaluations: &[F::Inner],
    eq_table: &[F],
    cfg: &F::Config,
) -> F {
    let mut acc = F::zero_with_cfg(cfg);
    assert_eq!(
        evaluations.len(),
        eq_table.len(),
        "evaluations and eq_table must have the same length"
    );
    for (eval, eq_val) in evaluations.iter().zip(eq_table.iter()) {
        let mut term = eq_val.clone();
        term.mul_assign_by_inner(eval);
        acc += &term;
    }
    acc
}

/// Returns a multilinear polynomial in 2n variables that evaluates to 1
/// if and only if the second n-bit vector is equal to the first vector plus one
#[allow(clippy::arithmetic_side_effects)]
pub fn next_mle_inner<F: Field>(
    num_vars: u32,
    zero: F,
    one: F,
) -> Result<DenseMultilinearExtension<F::Inner>, ArithErrors> {
    if !num_vars.is_multiple_of(2) {
        return Err(ArithErrors::InvalidParameters(
            "num_vars must be even".to_string(),
        ));
    }

    let mut mle = (0..1 << num_vars)
        .map(|_| zero.inner().clone())
        .collect_dense_mle_with_zero(zero.inner());

    let half_vars = num_vars / 2;

    for i in 0usize..(1 << half_vars) - 1 {
        let next = i + 1;

        let i_concat_next = (next << half_vars) | i;

        mle[i_concat_next] = one.inner().clone();
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
pub fn next_mle_eval<R: Semiring>(u: &[R], v: &[R], zero: R, one: R) -> R {
    let n = u.len();
    assert_eq!(n, v.len(), "u and v must have the same length");
    if n == 0 {
        return zero;
    }

    // suffix_eq[j] = prod_{i=j}^{n-1} eq(u_i, v_i)
    let mut suffix_eq = vec![one.clone(); n + 1];
    for i in (0..n).rev() {
        suffix_eq[i] = suffix_eq[i + 1].clone()
            * (u[i].clone() * &v[i] + (one.clone() - &u[i]) * (one.clone() - &v[i]));
    }

    // prefix_carry accumulates prod_{i<j} u_i * (1 - v_i)
    let mut prefix_carry = one.clone();
    let mut result = zero;
    for j in 0..n {
        result += prefix_carry.clone() * (one.clone() - &u[j]) * &v[j] * &suffix_eq[j + 1];
        prefix_carry *= u[j].clone() * (one.clone() - &v[j]);
    }
    result
}

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects, clippy::cast_possible_truncation)]
mod tests {
    use crypto_bigint::{U128, const_monty_params};
    use crypto_primitives::{IntoWithConfig, crypto_bigint_const_monty::ConstMontyField};
    use num_traits::One;
    use proptest::{prelude::*, proptest};

    use crate::mle::MultilinearExtensionWithConfig;

    use super::*;

    const_monty_params!(Params, U128, "00000000b933426489189cb5b47d567f");

    type F = ConstMontyField<Params, { U128::LIMBS }>;

    const NUM_VARS: u32 = 8;

    #[test]
    fn next_mle_is_one_on_successors() {
        let next_mle = next_mle_inner(NUM_VARS, F::zero(), F::one()).unwrap();

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

            assert_eq!(
                next_mle.clone().evaluate_with_config(&point, &()),
                Ok(F::one())
            );
        }
    }

    #[test]
    fn next_mle_is_one_only_on_successors() {
        let next_mle = next_mle_inner(NUM_VARS, F::zero(), F::one()).unwrap();

        // The number of successors is (1 << (num_vars / 2)) - 1
        // and we know the mle is one on them. So we need to check
        // that it is one only on that many points.
        assert_eq!(
            next_mle.evaluations.iter().filter(|x| !x.is_zero()).count(),
            (1 << (NUM_VARS / 2)) - 1
        );
    }

    fn any_f(cfg: <F as PrimeField>::Config) -> impl Strategy<Value = F> + 'static {
        any::<u128>().prop_map(move |v| v.into_with_cfg(&cfg))
    }

    fn point_n(n: usize) -> impl Strategy<Value = Vec<F>> {
        prop::collection::vec(any_f(()), n)
    }

    #[test]
    fn next_mle_eval_coincides_with_next_mle_evaluated_at_successors() {
        let next_mle = next_mle_inner(NUM_VARS, F::zero(), F::one()).unwrap();

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
                next_mle.clone().evaluate_with_config(&point, &()),
                Ok(next_mle_eval(u, v, F::zero(), F::one()))
            );
        }
    }

    proptest! {
    #[test]
    #[cfg_attr(miri, ignore)] // long running
    fn prop_next_mle_eval_coincides_with_next_mle_evaluate_at_point(r in point_n(NUM_VARS as usize)) {
        let next_mle = next_mle_inner(NUM_VARS, F::zero(), F::one()).unwrap();

        let (u, v) = r.split_at(NUM_VARS as usize / 2);
        prop_assert_eq!(
            next_mle.evaluate_with_config(&r, &()),
            Ok(next_mle_eval(u, v, F::zero(), F::one()))
        );
    }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // long running
    fn next_c_r_mle_c1_matches_shift_by_1() {
        // c=1 should give the same result as the original build_next_r_mle
        let num_vars: usize = 4;
        let r: Vec<F> = (0..num_vars).map(|i| F::from((i + 3) as u32)).collect();

        let next_1 = build_next_c_r_mle(&r, 1, &()).unwrap();

        // Manually build shift-by-1: evaluations[0] = 0, evaluations[b] = eq(r, b-1)
        let eq_r = build_eq_x_r_inner(&r, &()).unwrap();
        let n = 1 << num_vars;
        let mut expected = vec![F::zero().into_inner(); 1];
        expected.extend_from_slice(&eq_r.evaluations[..n - 1]);

        assert_eq!(next_1.evaluations, expected);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // long running
    fn next_c_r_mle_c0_is_eq() {
        // c=0 should return eq(r, b)
        let num_vars: usize = 4;
        let r: Vec<F> = (0..num_vars).map(|i| F::from((i + 7) as u32)).collect();

        let next_0 = build_next_c_r_mle(&r, 0, &()).unwrap();
        let eq_r = build_eq_x_r_inner(&r, &()).unwrap();

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
            let next_c = build_next_c_r_mle(&r, c, &()).unwrap();
            let eq_r = build_eq_x_r_inner(&r, &()).unwrap();

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
        let next_c = build_next_c_r_mle(&r, c, &()).unwrap();
        let eq_r = build_eq_x_r_inner(&r, &()).unwrap();

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
