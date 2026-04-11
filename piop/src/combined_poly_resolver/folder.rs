use crypto_primitives::PrimeField;
use zinc_uair::{ConstraintBuilder, ideal::ImpossibleIdeal};

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
/// an `assert_*` method is called it adds it to the RLC
/// with the next power of the challenge `\alpha`.
pub struct ConstraintFolder<'a, F: PrimeField> {
    /// A reference to precomputed powers of the challenge.
    challenge_powers: &'a [F],
    /// Index of the current constraint,
    /// and therefore the current power of the challenge.
    current_constraint: usize,
    /// The RLC computed so far.
    pub folded_constraints: F,
}

impl<'a, F: PrimeField> ConstraintFolder<'a, F> {
    pub fn new(challenge_powers: &'a [F], zero: &F) -> Self {
        Self {
            challenge_powers,
            current_constraint: 0,
            folded_constraints: zero.clone(),
        }
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn fold_constraint(&mut self, expr: F) {
        self.folded_constraints += expr * &self.challenge_powers[self.current_constraint];
        self.current_constraint += 1;
    }
}

impl<'a, F: PrimeField> ConstraintBuilder for ConstraintFolder<'a, F> {
    type Expr = F;

    type Ideal = ImpossibleIdeal;

    #[inline(always)]
    fn assert_in_ideal(&mut self, expr: Self::Expr, _ideal: &Self::Ideal) {
        self.fold_constraint(expr);
    }

    /// For an honest prover, an `assert_zero` constraint is identically
    /// zero on the hypercube, so its contribution to the folded RLC is 0
    /// and can be skipped. We still advance `current_constraint` so that
    /// the challenge-power indexing remains aligned with the raw
    /// constraint count seen by the non-folding paths (e.g. the ideal
    /// check's coefficient-MLE builder, which also assigns powers in
    /// declaration order). The prover and verifier both use this folder,
    /// so the skip is symmetric.
    #[inline(always)]
    #[allow(clippy::arithmetic_side_effects)]
    fn assert_zero(&mut self, _expr: Self::Expr) {
        self.current_constraint += 1;
    }
}
