use crypto_primitives::PrimeField;
use zinc_uair::ConstraintBuilder;
use zinc_uair::ideal::ImpossibleIdeal;

pub struct ConstraintFolder<'a, F: PrimeField> {
    challenge_powers: &'a [F],
    current_constraint: usize,
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

    #[inline(always)]
    fn assert_zero(&mut self, expr: Self::Expr) {
        self.fold_constraint(expr);
    }
}
