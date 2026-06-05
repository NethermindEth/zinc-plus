use crypto_primitives::PrimeField;
use zinc_uair::{ConstraintBuilder, ideal::ImpossibleIdeal};
use zinc_utils::{
    UNCHECKED,
    delayed_reduction::DelayedFieldProductSum,
    inner_product::{FieldFieldInnerProduct, InnerProduct},
};

/// There are several situations where we need to
/// compute an RLC `u_0 + \alpha * u_1 + ... + \alpha ^ k * u_k`,
/// where `u_0,...,u_k` are field evaluations of
/// the constraint polynomials of a UAIR on certain values:
/// $$
/// u_0 = f_0(r_0,...,r_n)
/// ...
/// u_k = f_k(r_0,...,r_n)
/// $$
/// This situation happens twice: in the combined poly resolver
/// prover when we instantiate the sumcheck and batch together
/// all the evaluation claims for the combined polynomial MLEs;
/// and, secondly, in the combined poly resolver verifier where
/// check correctness of the resulting sumcheck claim.
///
/// This constraint builder handles those situations.
/// It's `Expr` associated type is the field `F`, so once
/// an `assert_*` method is called it records the residual in order.
/// Call [`ConstraintFolder::finish_folded`] to compute the RLC with the
/// DMR-aware field-field product-sum backend.
pub struct ConstraintFolder<'a, F: PrimeField> {
    /// A reference to precomputed powers of the challenge.
    challenge_powers: &'a [F],
    /// Residuals in the exact order constraints were visited.
    residuals: Vec<F>,
    /// Additive identity used as the product-sum seed.
    zero: F,
}

impl<'a, F: PrimeField> ConstraintFolder<'a, F> {
    pub fn new(challenge_powers: &'a [F], zero: &F) -> Self {
        Self {
            challenge_powers,
            residuals: Vec::with_capacity(challenge_powers.len()),
            zero: zero.clone(),
        }
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn fold_constraint(&mut self, expr: F) {
        debug_assert!(
            self.residuals.len() < self.challenge_powers.len(),
            "more constraint residuals than challenge powers"
        );
        self.residuals.push(expr);
    }

    pub fn finish_folded(self) -> F
    where
        F: DelayedFieldProductSum,
    {
        FieldFieldInnerProduct::inner_product::<UNCHECKED>(
            &self.residuals,
            &self.challenge_powers[..self.residuals.len()],
            self.zero,
        )
        .expect("constraint residuals and challenge powers have matching lengths")
    }
}

impl<'a, F: PrimeField> ConstraintBuilder for ConstraintFolder<'a, F> {
    type Expr = F;

    type Ideal = ImpossibleIdeal;

    #[inline(always)]
    fn assert_in_ideal(&mut self, expr: Self::Expr, _ideal: &Self::Ideal) {
        self.fold_constraint(expr);
    }

    /// `assert_zero` constraints contribute to the folded RLC like any
    /// other constraint. For an honest prover they evaluate to zero on
    /// every row of the hypercube, so their fold-sum is 0 — but the
    /// polynomial expression itself can have per-variable degree > 1
    /// (e.g. `b·(b-1)·s_accum`), so the sumcheck protocol must run at
    /// `count_max_degree::<U>() + 2`, NOT `count_effective_max_degree`.
    /// A previous version of this method was a no-op coupled with
    /// `count_effective_max_degree` for the protocol degree; that
    /// combination silently dropped the binding between assert_zero
    /// constraints and the witness, breaking soundness for every UAIR
    /// with assert_zero constraints.
    #[inline(always)]
    fn assert_zero(&mut self, expr: Self::Expr) {
        self.fold_constraint(expr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_primitives::{FromWithConfig, crypto_bigint_monty::MontyField};
    use zinc_uair::ConstraintBuilder;

    type F = MontyField<4>;

    fn cfg() -> <F as crypto_primitives::PrimeField>::Config {
        crate::test_utils::test_config()
    }

    fn f(value: u64) -> F {
        F::from_with_cfg(value, &cfg())
    }

    fn naive_fold(residuals: &[F], powers: &[F], zero: F) -> F {
        residuals
            .iter()
            .zip(powers)
            .fold(zero, |acc, (residual, power)| {
                acc + residual.clone() * power
            })
    }

    #[test]
    fn finish_folded_matches_naive_empty() {
        let cfg = cfg();
        let zero = F::zero_with_cfg(&cfg);
        let powers = vec![f(1), f(7), f(49)];
        let folder = ConstraintFolder::new(&powers, &zero);

        assert_eq!(folder.finish_folded(), zero);
    }

    #[test]
    fn finish_folded_matches_naive_single_constraint() {
        let cfg = cfg();
        let zero = F::zero_with_cfg(&cfg);
        let powers = vec![f(1), f(7), f(49)];
        let residuals = vec![f(11)];
        let mut folder = ConstraintFolder::new(&powers, &zero);
        folder.assert_zero(residuals[0].clone());

        assert_eq!(
            folder.finish_folded(),
            naive_fold(&residuals, &powers, zero)
        );
    }

    #[test]
    fn finish_folded_matches_naive_multiple_constraints() {
        let cfg = cfg();
        let zero = F::zero_with_cfg(&cfg);
        let powers = vec![f(1), f(7), f(49), f(343)];
        let residuals = vec![f(3), f(5), f(8), f(13)];
        let mut folder = ConstraintFolder::new(&powers, &zero);
        for residual in &residuals {
            folder.assert_zero(residual.clone());
        }

        assert_eq!(
            folder.finish_folded(),
            naive_fold(&residuals, &powers, zero)
        );
    }
}
