use crypto_primitives::{
    FieldConfig, ProjectPrimitiveIntegersWithConfig, Semiring, SetConfig, SetElement,
};

use crate::EvaluationError;

/// Polynomial evaluated on 0, 1, 2, ....
#[derive(Clone, Debug, PartialEq)]
pub struct NatEvaluatedPoly<E: SetElement> {
    /// Evaluations on P(0), P(1), P(2), ...
    pub evaluations: Vec<E>,
}

impl<E: SetElement> NatEvaluatedPoly<E> {
    #[inline(always)]
    pub const fn new(evaluations: Vec<E>) -> Self {
        Self { evaluations }
    }

    /// Interpolate the *unique* univariate polynomial of degree *at most*
    /// `evaluations.len()-1` passing through the y-values in `evaluations` at x
    /// = 0,..., evaluations.len()-1
    /// and evaluate this  polynomial at `point`. In other words, efficiently
    /// compute  $\sum_{i=0}^{len\ evaluations - 1} evaluations\[i\] *
    /// (\prod_{j!=i} (\text{point} - j)/(i-j))$.
    // All the arithmetic ops in the function
    // are made sure to not overflow.
    #[allow(clippy::arithmetic_side_effects, clippy::cast_possible_wrap)]
    pub fn evaluate_at_point<C>(&self, cfg: &C, point: &E) -> Result<E, EvaluationError>
    where
        C: FieldConfig + ProjectPrimitiveIntegersWithConfig + SetConfig<Element = E>,
    {
        let evaluations = &self.evaluations;
        // TODO(Alex): Once we have benches, it's worth checking
        //             if we're even winning anything
        //             with specialized branches above.

        let zero = cfg.zero();
        let one = cfg.one();

        let len = evaluations.len();

        let mut evals = vec![];

        let mut prod = point.clone();
        evals.push(point.clone());

        //`prod = \prod_{j} (x - j)`
        // we return early if 0 <= x < len, i.e. if the desired value has been passed
        let mut j = zero.clone();
        for i in 1..len {
            if *point == j {
                return Ok(evaluations[i - 1].clone());
            }
            cfg.add_assign(&mut j, &one);

            let tmp = cfg.sub(point, &j);
            evals.push(tmp.clone());
            cfg.mul_assign(&mut prod, &tmp);
        }

        if *point == j {
            return Ok(evaluations[len - 1].clone());
        }

        let div = |num: &E, denom: &E| cfg.div(num, denom);

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
        //  - for len >  33 with field elements
        if evaluations.len() <= 20 {
            let last_denom: E = cfg.project(&factorial(len - 1, u64::from));

            let mut ratio_numerator = 1i64;
            let mut ratio_denominator = 1u64;

            for i in (0..len).rev() {
                let ratio_numerator_f = cfg.project(&ratio_numerator);
                let ratio_denominator_f = cfg.project(&ratio_denominator);

                let num = cfg.mul(&prod, &ratio_denominator_f);
                let denom = cfg.mul(&cfg.mul(&last_denom, &ratio_numerator_f), &evals[i]);
                let x = div(&num, &denom);

                let term = cfg.mul(&evaluations[i], &x);
                cfg.add_assign(&mut res, &term);

                // compute ratio for the next step which is current_ratio * -(len-i)/i
                if i != 0 {
                    // Using intentionally, overflow isn't possible
                    ratio_numerator *= -(len as i64 - i as i64);
                    ratio_denominator *= i as u64;
                }
            }
        } else if evaluations.len() <= 33 {
            let last_denom: E = cfg.project(&factorial(len - 1, u128::from));
            let mut ratio_numerator = 1i128;
            let mut ratio_denominator = 1u128;

            for i in (0..len).rev() {
                let ratio_numerator_f = cfg.project(&ratio_numerator);
                let ratio_denominator_f = cfg.project(&ratio_denominator);

                let num = cfg.mul(&prod, &ratio_denominator_f);
                let denom = cfg.mul(&cfg.mul(&last_denom, &ratio_numerator_f), &evals[i]);
                let x = div(&num, &denom);

                let term = cfg.mul(&evaluations[i], &x);
                cfg.add_assign(&mut res, &term);

                // compute ratio for the next step which is current_ratio * -(len-i)/i
                if i != 0 {
                    ratio_numerator *= -(len as i128 - i as i128);
                    ratio_denominator *= i as u128;
                }
            }
        } else {
            // since we are using field operations, we can merge
            // `last_denom` and `ratio_numerator` into a single field element.
            let mut denom_up = cfg.product((1..=(len as u64 - 1)).map(|u: u64| cfg.project(&u)));
            let mut denom_down = one;

            for i in (0..len).rev() {
                let num = cfg.mul(&prod, &denom_down);
                let denom = cfg.mul(&denom_up, &evals[i]);
                let x = div(&num, &denom);

                let term = cfg.mul(&evaluations[i], &x);
                cfg.add_assign(&mut res, &term);

                // compute denom for the next step is -current_denom * (len-i)/i
                if i != 0 {
                    let denom_up_factor = cfg.project(&((len - i) as u64));
                    denom_up = cfg.neg(&cfg.mul(&denom_up, &denom_up_factor));

                    let denom_down_factor = cfg.project(&(i as u64));
                    cfg.mul_assign(&mut denom_down, &denom_down_factor);
                }
            }
        }

        Ok(res)
    }
}

/// Compute the factorial(a) = 1 * 2 * ... * a.
#[allow(clippy::arithmetic_side_effects)]
fn factorial<R, F>(a: usize, from_u64: F) -> R
where
    R: Semiring,
    F: Fn(u64) -> R + Send + Sync,
{
    (1..=(a as u64))
        .map(&from_u64)
        .reduce(|mut acc, next| {
            acc *= next;
            acc
        })
        .unwrap_or(from_u64(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_primitives::{
        BaseFieldConfig, ProjectElementWithConfig, crypto_bigint_monty::MontyField,
        crypto_bigint_uint::Uint,
    };
    use itertools::Itertools;

    const LIMBS: usize = 4;

    fn test_config() -> MontyField<LIMBS> {
        let modulus =
            Uint::from_be_hex("0000000000000000000000000000000000860995AE68FC80E1B1BD1E39D54B33");
        MontyField::new(&modulus).expect("modulus should be a valid odd prime")
    }

    #[test]
    fn evaluate_nat_evaluation() {
        let cfg = test_config();
        let field_elem = cfg.project(&100u64);

        let poly = NatEvaluatedPoly::new((0..1024u64).map(|x| cfg.project(&x)).collect_vec());

        let res = poly.evaluate_at_point(&cfg, &field_elem).unwrap();

        assert_eq!(res, cfg.project(&100u64));
    }
}
