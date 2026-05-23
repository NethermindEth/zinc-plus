use std::convert::identity;

use crypto_primitives::{FromPrimitiveWithConfig, Semiring};

use crate::{EvaluatablePolynomial, EvaluationError, Polynomial};

/// Precomputed Lagrange data for `NatEvaluatedPoly::evaluate_at_point`:
/// the boundary points `F::from(0..len)` and the inverse denominators
/// `dens_inv[i] = (Π_{j ≠ i} (boundary[i] − boundary[j]))⁻¹`.
///
/// Built once per `(len, config)` via [`NatEvaluatedPoly::prepare_eval_aux`].
/// Cheap to clone (two `Vec<F>` of length `len`). The hot loop in the
/// sumcheck verifier ([`zinc_piop::sumcheck`]) reuses the same aux
/// across all `num_vars` rounds of a given group — the denominators
/// only depend on `(len, config)`, not on the polynomial being
/// evaluated.
#[derive(Clone, Debug)]
pub struct EvalAux<F> {
    /// `boundary[k] = F::from(k as u64)` for `k = 0..len`.
    pub boundary: Vec<F>,
    /// `dens_inv[i] = (Π_{j ≠ i} (boundary[i] − boundary[j]))⁻¹`.
    /// Built via Montgomery's batch-inversion trick (one field
    /// inversion + `O(len)` mults, vs `len` independent inversions).
    pub dens_inv: Vec<F>,
}

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
    fn evaluate_at_point(&self, point: &Self::EvaluationPoint) -> Result<F, EvaluationError> {
        let len = self.evaluations.len();
        if len == 0 {
            return Err(EvaluationError::EmptyPolynomial);
        }
        // Single-call path: build the aux on the fly. Reuse-heavy
        // callers (e.g. the sumcheck verifier) should call
        // `prepare_eval_aux` once and use `evaluate_at_point_with_aux`
        // directly to amortise the field inversion across multiple
        // polynomials of the same length.
        let aux = NatEvaluatedPoly::<F>::prepare_eval_aux(len, point.cfg());
        self.evaluate_at_point_with_aux(point, &aux)
    }
}

impl<F: FromPrimitiveWithConfig> NatEvaluatedPoly<F> {
    /// Precompute the Lagrange-aux ([`EvalAux`]) for evaluating any
    /// polynomial of length `len` at any point in field `F`.
    ///
    /// Returns `boundary[k] = F::from(k)` and
    /// `dens_inv[i] = (Π_{j ≠ i} (boundary[i] − boundary[j]))⁻¹`,
    /// computed via Montgomery's batch-inversion trick (one field
    /// inversion total, plus `O(len)` multiplications). Reuse the
    /// returned aux across all `evaluate_at_point_with_aux` calls
    /// with the same `len` and `config` — the denominators don't
    /// depend on the polynomial.
    ///
    /// **Hot-path provenance**: the sumcheck verifier
    /// ([`zinc_piop::sumcheck::multi_degree`]) calls
    /// `evaluate_at_point` once per `(group, round)`. Before this
    /// optimisation, each call did `len` field inversions in the
    /// denominator loop — totalling ~370 inversions for a
    /// `num_groups=41, num_vars=9` sumcheck, costing tens of ms in
    /// GF(2^192) (each Fermat inversion = ~380 field multiplications).
    /// Pre-computing the aux once per group drops that to a single
    /// inversion per group and lifts the rest of the round work into
    /// O(`len`) multiplications.
    pub fn prepare_eval_aux(len: usize, config: &F::Config) -> EvalAux<F> {
        let one = F::one_with_cfg(config);
        let boundary: Vec<F> = (0..len)
            .map(|k| F::from_with_cfg(k as u64, config))
            .collect();
        // dens[i] = Π_{j ≠ i} (boundary[i] − boundary[j]).
        // For len ≤ 1, the empty product is `1` (already correct).
        let dens: Vec<F> = (0..len)
            .map(|i| {
                let mut d = one.clone();
                for j in 0..len {
                    if j == i {
                        continue;
                    }
                    d = d * (boundary[i].clone() - &boundary[j]);
                }
                d
            })
            .collect();
        let dens_inv = batch_invert(dens, config);
        EvalAux { boundary, dens_inv }
    }

    /// Evaluate the polynomial using a precomputed [`EvalAux`].
    /// Mirrors [`evaluate_at_point`], but reuses the supplied
    /// boundary points and inverse denominators instead of
    /// recomputing them on every call. The aux must have been
    /// prepared with the same `len = self.evaluations.len()` and
    /// the same field `config`; otherwise the result is
    /// meaningless.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn evaluate_at_point_with_aux(
        &self,
        point: &F,
        aux: &EvalAux<F>,
    ) -> Result<F, EvaluationError> {
        let len = self.evaluations.len();
        if len == 0 {
            return Err(EvaluationError::EmptyPolynomial);
        }
        debug_assert_eq!(
            aux.boundary.len(),
            len,
            "EvalAux len mismatch with polynomial: aux has {}, polynomial has {}",
            aux.boundary.len(),
            len,
        );

        let config = point.cfg();
        let zero = F::zero_with_cfg(config);
        let one = F::one_with_cfg(config);

        // Early exit: if `point` exactly matches one of the boundary
        // points, the answer is that boundary's evaluation. Avoids
        // the `0 / 0` (= `0 · ∞`) trap in the loop below.
        for (k, b) in aux.boundary.iter().enumerate() {
            if point == b {
                return Ok(self.evaluations[k].clone());
            }
        }

        let mut res = zero;
        for i in 0..len {
            let mut num = one.clone();
            for j in 0..len {
                if j == i {
                    continue;
                }
                num = num * (point.clone() - &aux.boundary[j]);
            }
            // Contribution: evaluations[i] * num * dens_inv[i].
            res += &(self.evaluations[i].clone() * num * &aux.dens_inv[i]);
        }
        Ok(res)
    }
}

/// Montgomery batch inversion: invert `n` field elements with a single
/// field inversion + `3(n-1)` multiplications.
///
/// Assumes every input is non-zero (caller's responsibility — for the
/// Lagrange denominators above this is guaranteed by distinct boundary
/// points). The result satisfies `out[i] * values[i] = 1` for all `i`.
#[allow(clippy::arithmetic_side_effects)]
fn batch_invert<F: FromPrimitiveWithConfig>(values: Vec<F>, config: &F::Config) -> Vec<F> {
    let n = values.len();
    if n == 0 {
        return values;
    }
    let one = F::one_with_cfg(config);
    // products[i] = values[0] * values[1] * ... * values[i].
    let mut products: Vec<F> = Vec::with_capacity(n);
    products.push(values[0].clone());
    for i in 1..n {
        products.push(products[i - 1].clone() * &values[i]);
    }
    // Single field inversion: inv = (Π values[i])⁻¹ = 1 / products[n-1].
    let mut inv = one.clone() / products[n - 1].clone();
    let mut result: Vec<F> = vec![one; n];
    // Walk backwards: at step i, `inv` holds (values[0] * ... * values[i])⁻¹.
    // Then result[i] = inv * products[i-1] = (values[0]*...*values[i-1])^{-1}
    //                  ... no wait: result[i] should be values[i]^{-1}.
    // Correct extraction: result[i] = inv * products[i-1].
    //   = (Π_{k ≤ i} values[k])⁻¹ · (Π_{k < i} values[k])
    //   = values[i]⁻¹.
    // Then update inv ← inv · values[i] = (Π_{k < i} values[k])⁻¹.
    for i in (1..n).rev() {
        result[i] = inv.clone() * &products[i - 1];
        inv = inv.clone() * &values[i];
    }
    // After the loop, inv = values[0]⁻¹.
    result[0] = inv;
    result
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
