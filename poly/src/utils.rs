use crypto_primitives::{Field, PrimeField, Semiring};
use num_traits::Zero;
use thiserror::Error;
use zinc_utils::cfg_iter_mut;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::mle::{DenseMultilinearExtension, dense::CollectDenseMleWithZero};

/// A `enum` specifying the possible failure modes of the arithmetics.
#[derive(displaydoc::Display, Debug, Error)]
pub enum ArithErrors {
    /// Invalid parameters: {0}
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
pub(crate) fn build_eq_x_r_vec<F>(r: &[F], cfg: &F::Config) -> Result<Vec<F>, ArithErrors>
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
        buf.push((one - &r[0]).inner().clone());
        buf.push(r[0].inner().clone());
    } else {
        build_eq_x_r_inner_helper(&r[1..], buf, cfg)?;

        // suppose at the previous step we received [b_1, ..., b_k]
        // for the current step we will need
        // if x_0 = 0:   (1-r0) * [b_1, ..., b_k]
        // if x_0 = 1:   r0 * [b_1, ..., b_k]

        let mut bi = F::zero_with_cfg(cfg);

        *buf = (0..(buf.len() << 1))
            .map(|i| {
                *bi.inner_mut() = buf[i >> 1].clone();
                let tmp = r[0].clone() * &bi;
                if (i & 1) == 0 {
                    (bi.clone() - &tmp).inner().clone()
                } else {
                    tmp.inner().clone()
                }
            })
            .collect();
    }

    Ok(())
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

/// Evaluates the next MLE at the point `point` in log-time.
///
/// # Arguments
/// - `point`: A slice of 2n field elements representing two n-bit vectors
///   concatenated. The first n elements are `x` (original vector), the last n
///   are `y` (candidate successor).
///
/// # Behavior
/// Constructs a polynomial P(x, y) such that:
/// \begin{equation}
///     P(x, y) = 1 \quad \text{if and only if} \quad y = x + 1.
/// \end{equation}
///
/// The polynomial sums contributions for each possible carry position `k`,
/// ensuring that:
/// 1. Bits to the left of `k` (more significant) match.
/// 2. Bit at position `k` transitions from 0 (in x) to 1 (in y).
/// 3. Bits to the right of `k` are 1 in x and 0 in y (simulating the carry
///    propagation).
///
/// # Panics
/// Panics if `point.len()` is not even.
///
/// # Returns
/// Field element: 1 if y = x + 1, 0 otherwise.
#[allow(clippy::arithmetic_side_effects)]
pub fn next_mle_eval<R: Semiring>(point: &[R], zero: R, one: R) -> R {
    // Check that the point length is even: we split into x and y of equal length.
    assert_eq!(
        point.len() % 2,
        0,
        "Input point must have an even number of variables."
    );
    let n = point.len() / 2;

    // Split point into x (first n) and y (last n).
    let (x, y) = point.split_at(n);

    // Sum contributions for each possible carry position k = 0..n-1.
    (0..n)
        .map(|k| {
            // Term 1: bits to the left of k match
            //
            // For i > k, enforce x_i == y_i.
            // Using equality polynomial: x_i * y_i + (1 - x_i)*(1 - y_i).
            let eq_high_bits = (k + 1..n)
                .map(|i| x[i].clone() * &y[i] + (one.clone() - &x[i]) * (one.clone() - &y[i]))
                .fold(one.clone(), |acc, next| acc * next);

            // Term 2: carry bit at position k
            //
            // Enforce x_k = 0 and y_k = 1.
            // Condition: (1 - x_k) * y_k.
            let carry_bit = (one.clone() - &x[k]) * &y[k];

            // Term 3: bits to the right of k are 1 in x and 0 in y
            //
            // For i < k, enforce x_i = 1 and y_i = 0.
            // Condition: x_i * (1 - y_i).
            let low_bits_are_one_zero = (0..k)
                .map(|i| (one.clone() - &y[i]) * &x[i])
                .fold(one.clone(), |acc, next| acc * next);

            // Multiply the three terms for this k, representing one "carry pattern".
            eq_high_bits * carry_bit * low_bits_are_one_zero
        })
        // Sum over all carry positions: any valid "k" gives contribution 1.
        .fold(zero, |acc, next| acc + next)
}

#[cfg(test)]
mod tests {
    use crypto_bigint::{U128, const_monty_params};
    use crypto_primitives::{IntoWithConfig, crypto_bigint_const_monty::ConstMontyField};
    use num_traits::One;
    use proptest::{prelude::*, proptest};

    use crate::mle::MultilinearExtensionWithConfig;

    use super::*;

    const N: usize = 2;

    const_monty_params!(Params, U128, "00000000b933426489189cb5b47d567f");

    type F = ConstMontyField<Params, N>;

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

            assert_eq!(next_mle.evaluate_with_config(&point, &()), Ok(F::one()));
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

            assert_eq!(
                next_mle.evaluate_with_config(&point, &()),
                Ok(next_mle_eval(&point, F::zero(), F::one()))
            );
        }
    }

    proptest! {
    #[test]
    fn prop_next_mle_eval_coincides_with_next_mle_evaluate_at_point(r in point_n(NUM_VARS as usize)) {
        let next_mle = next_mle_inner(NUM_VARS, F::zero(), F::one()).unwrap();

        prop_assert_eq!(
            next_mle.evaluate_with_config(&r, &()),
            Ok(next_mle_eval(&r, F::zero(), F::one()))
        );
    }
    }
}
