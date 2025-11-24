#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::convert::identity;

use ark_std::cfg_into_iter;
use crypto_primitives::{FromPrimitiveWithConfig, Semiring};

use crate::{EvaluatablePolynomial, EvaluationError, Polynomial};

/// Polynomial evaluated on 0, 1, 2, ....
#[derive(Clone, Debug, PartialEq)]
pub struct NatEvaluatedPoly<F> {
    /// Evaluations on P(0), P(1), P(2), ...
    pub evaluations: Vec<F>,
}

impl<F> NatEvaluatedPoly<F> {
    #[inline(always)]
    pub const fn new(evaluations: Vec<F>) -> Self {
        Self { evaluations }
    }
}

impl<F> Polynomial<F> for NatEvaluatedPoly<F> {
    const DEGREE_BOUND: usize = usize::MAX;
}

impl<F: FromPrimitiveWithConfig> EvaluatablePolynomial<F, F, F> for NatEvaluatedPoly<F> {
    type EvaluationPoint = F;

    /// Interpolate the *unique* univariate polynomial of degree *at most*
    /// `evaluations.len()-1` passing through the y-values in `evaluations` at x
    /// = 0,..., evaluations.len()-1
    /// and evaluate this  polynomial at `point`. In other words, efficiently
    /// compute  $\sum_{i=0}^{len\ evaluations - 1} evaluations\[i\] *
    /// (\prod_{j!=i} (\text{point} - j)/(i-j))$.
    // All the arithmetic ops in the function
    // are made sure to not overflow.
    #[allow(clippy::arithmetic_side_effects, clippy::cast_possible_wrap)]
    #[allow(clippy::arithmetic_side_effects, clippy::cast_possible_wrap)]
    fn evaluate_at_point(&self, point: &Self::EvaluationPoint) -> Result<F, EvaluationError> {
        let NatEvaluatedPoly { evaluations } = self;
        // TODO(Alex): Once we have benches, it's worth checking
        //             if we're even winning anything
        //             with specialized branches above.

        // We will need these a few times
        let point = point.clone();
        let config = point.cfg();
        let zero = F::zero_with_cfg(config);
        let one = F::one_with_cfg(config);

        let len = evaluations.len();

        let mut evals = vec![];

        let mut prod = point.clone();
        evals.push(point.clone());

        //`prod = \prod_{j} (x - j)`
        // we return early if 0 <= x < len, i.e. if the desired value has been passed
        let mut j = zero.clone();
        for i in 1..len {
            if point == j {
                return Ok(evaluations[i - 1].clone());
            }
            j += &one;

            let tmp = point.clone() - j.clone();
            evals.push(tmp.clone());
            prod *= tmp;
        }

        if point == j {
            return Ok(evaluations[len - 1].clone());
        }

        let mut res = zero;
        // we want to compute \prod (j!=i) (i-j) for a given i
        //
        // we start from the last step, which is
        //  denom[len-1] = (len-1) * (len-2) *... * 2 * 1
        // the step before that is
        //  denom[len-2] = (len-2) * (len-3) * ... * 2 * 1 * -1
        // and the step before that is
        //  denom[len-3] = (len-3) * (len-4) * ... * 2 * 1 * -1 * -2
        //
        // i.e., for any i, the one before this will be derived from
        //  denom[i-1] = - denom[i] * (len-i) / i
        //
        // that is, we only need to store
        // - the last denom for i = len-1, and
        // - the ratio between the current step and the last step, which is the product
        //   of -(len-i) / i from all previous steps and we store this product as a
        //   fraction number to reduce field divisions.

        // We know
        //  - 2^61 < factorial(20) < 2^62
        //  - 2^122 < factorial(33) < 2^123
        // so we will be able to compute the ratio
        //  - for len <= 20 with i64
        //  - for len <= 33 with i128
        //  - for len >  33 with BigInt
        if evaluations.len() <= 20 {
            let last_denom: F = F::from_with_cfg(factorial(len - 1, identity), config);

            let mut ratio_numerator = 1i64;
            let mut ratio_denominator = 1u64;

            for i in (0..len).rev() {
                let ratio_numerator_f = F::from_with_cfg(ratio_numerator, config);

                let ratio_denominator_f = F::from_with_cfg(ratio_denominator, config);

                let x = prod.clone() * ratio_denominator_f
                    / (last_denom.clone() * ratio_numerator_f * &evals[i]);

                res += &(evaluations[i].clone() * x);

                // compute ratio for the next step which is current_ratio * -(len-i)/i
                if i != 0 {
                    // Using intentionally, overflow isn't possible
                    ratio_numerator *= -(len as i64 - i as i64);
                    ratio_denominator *= i as u64;
                }
            }
        } else if evaluations.len() <= 33 {
            let last_denom = F::from_with_cfg(factorial(len - 1, u128::from), config);
            let mut ratio_numerator = 1i128;
            let mut ratio_denominator = 1u128;

            for i in (0..len).rev() {
                let ratio_numerator_f = F::from_with_cfg(ratio_numerator, config);

                let ratio_denominator_f = F::from_with_cfg(ratio_denominator, config);

                let x: F = prod.clone() * ratio_denominator_f
                    / (last_denom.clone() * ratio_numerator_f * &evals[i]);
                res += &(evaluations[i].clone() * x);

                // compute ratio for the next step which is current_ratio * -(len-i)/i
                if i != 0 {
                    ratio_numerator *= -(len as i128 - i as i128);
                    ratio_denominator *= i as u128;
                }
            }
        } else {
            // since we are using field operations, we can merge
            // `last_denom` and `ratio_numerator` into a single field element.
            let mut denom_up = factorial(len - 1, |u| F::from_with_cfg(u, config));
            let mut denom_down = one;

            for i in (0..len).rev() {
                let x = prod.clone() * &denom_down / (denom_up.clone() * &evals[i]);
                res += &(evaluations[i].clone() * x);

                // compute denom for the next step is -current_denom * (len-i)/i
                if i != 0 {
                    let denom_up_factor = F::from_with_cfg((len - i) as u64, config);
                    denom_up *= -denom_up_factor;

                    let denom_down_factor = F::from_with_cfg(i as u64, config);
                    denom_down *= denom_down_factor;
                }
            }
        }

        Ok(res)
    }
}

/// Compute the factorial(a) = 1 * 2 * ... * a.
fn factorial<R, F>(a: usize, from_u64: F) -> R
where
    R: Semiring,
    F: Fn(u64) -> R + Send + Sync,
{
    cfg_into_iter!((1..=(a as u64))).map(from_u64).product()
}
