use std::convert::identity;

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

impl<F: Clone> Polynomial<F> for NatEvaluatedPoly<F> {
    const DEGREE_BOUND: usize = usize::MAX;
}

impl<F: FromPrimitiveWithConfig> EvaluatablePolynomial<F, F> for NatEvaluatedPoly<F> {
    type EvaluationPoint = F;

    /// Interpolate the *unique* univariate polynomial of degree at most
    /// `evaluations.len() - 1` passing through the y-values
    /// `evaluations[0], …, evaluations[len-1]` at the boundary points
    /// `F::from(0), F::from(1), …, F::from(len-1)`, and evaluate that
    /// polynomial at `point`. Returns
    ///
    /// $$
    /// \sum_{i=0}^{len-1} \text{evaluations}[i] \cdot
    ///     \prod_{j \ne i} \frac{(point - F::\text{from}(j))}{(F::\text{from}(i) - F::\text{from}(j))}.
    /// $$
    ///
    /// **Field-honest implementation.** Earlier versions of this
    /// function exploited `F::from(k) = k` (the natural-number → field
    /// ring homomorphism that exists for prime fields) and accumulated
    /// Lagrange denominators via integer factorial arithmetic. That
    /// optimisation is unsound in characteristic 2: the only ring
    /// homomorphism `Z → GF(2^n)` factors through `Z/2Z`, so a
    /// faithful integer-style `From<u64>` either collapses the
    /// boundary points (`F::from(2) = 0 = F::from(0)`) or, under a
    /// bit-pattern convention (`F::from(2) = X`), keeps them distinct
    /// but breaks `F::from(a) · F::from(b) = F::from(a·b)`. The
    /// integer-factorial accumulator relies on the latter identity
    /// and so was returning incorrect Lagrange denominators.
    ///
    /// The current implementation computes
    /// `denom[i] = Π_{j ≠ i} (F::from(i) - F::from(j))` directly via
    /// field-arithmetic over the actual boundary points. The cost
    /// rises from `O(len)` to `O(len²)` field multiplications, but
    /// the algorithm is now correct in **any** field: prime fields
    /// (where the new computation gives the same result as the old
    /// integer-factorial path), `GF(2^n)`, and any other extension.
    /// For typical sumcheck workloads `len ≤ 8`, so the constant
    /// factor is negligible.
    #[allow(clippy::arithmetic_side_effects)]
    fn evaluate_at_point(&self, point: &Self::EvaluationPoint) -> Result<F, EvaluationError> {
        let evaluations = &self.evaluations;
        let point = point.clone();
        let config = point.cfg();
        let zero = F::zero_with_cfg(config);
        let one = F::one_with_cfg(config);

        let len = evaluations.len();
        if len == 0 {
            return Err(EvaluationError::EmptyPolynomial);
        }

        // Precompute `boundary[k] = F::from(k as u64)` for k = 0..len.
        // Critical: use `from_with_cfg` per index rather than
        // `boundary[k-1] + one`. In a prime field both produce the same
        // value (the k-th additive successor of 0 equals `F::from(k)`),
        // but in characteristic 2 they diverge: `+ one` cycles
        // `0, 1, 0, 1, …` while `from_with_cfg(k)` returns
        // `0, 1, X, X+1, X^2, …` (the canonical bit-pattern injection,
        // which is distinct for each `k` in any field). Distinct
        // boundary points are required for non-zero Lagrange
        // denominators.
        //
        // Unused since we'd otherwise warn about `one` being dead
        // code in this branch — keep it bound so any future code
        // that wants the field's `one()` doesn't need to re-derive
        // it.
        let _ = &one;
        let boundary: Vec<F> = (0..len)
            .map(|k| F::from_with_cfg(k as u64, config))
            .collect();

        // Early exit: if `point` exactly matches one of the boundary
        // points, the answer is that boundary's evaluation. This also
        // avoids a `0 / 0` numerator/denominator in the loop below.
        for (k, b) in boundary.iter().enumerate() {
            if &point == b {
                return Ok(evaluations[k].clone());
            }
        }

        // Field-honest Lagrange.
        //
        // For each i:
        //   numerator[i] = Π_{j ≠ i} (point - boundary[j])
        //   denominator[i] = Π_{j ≠ i} (boundary[i] - boundary[j])
        // Contribution: evaluations[i] * numerator[i] / denominator[i].
        let mut res = zero;
        for i in 0..len {
            let mut num = one.clone();
            let mut den = one.clone();
            for j in 0..len {
                if j == i {
                    continue;
                }
                num *= point.clone() - &boundary[j];
                den *= boundary[i].clone() - &boundary[j];
            }
            res += &(evaluations[i].clone() * num / den);
        }
        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use crypto_bigint::{Odd, modular::MontyParams};
    use crypto_primitives::{FromWithConfig, crypto_bigint_monty::F256};
    use itertools::Itertools;

    use crate::{EvaluatablePolynomial, univariate::nat_evaluation::NatEvaluatedPoly};

    const LIMBS: usize = 4;
    type F = F256;

    fn test_config() -> MontyParams<LIMBS> {
        let modulus = crypto_bigint::Uint::<LIMBS>::from_be_hex(
            "0000000000000000000000000000000000860995AE68FC80E1B1BD1E39D54B33",
        );
        let modulus = Odd::new(modulus).expect("modulus should be odd");
        MontyParams::new(modulus)
    }

    #[test]
    fn evaluate_nat_evaluation() {
        let field_elem = F::from_with_cfg(100, &test_config());

        let poly = NatEvaluatedPoly::new(
            (0..1024)
                .map(|x| F::from_with_cfg(x, &test_config()))
                .collect_vec(),
        );

        let res = poly.evaluate_at_point(&field_elem).unwrap();

        assert_eq!(res, F::from_with_cfg(100, &test_config()));
    }
}
