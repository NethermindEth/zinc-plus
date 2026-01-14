pub mod constraint_counter;

use crypto_primitives::Semiring;
use num_traits::Zero;

pub trait ConstraintBuilder {
    type Expr: Semiring + Zero;

    fn assert_in_ideal(&mut self, expr: Self::Expr, ideal_generator: &Self::Expr);

    fn assert_zero(&mut self, expr: Self::Expr) {
        self.assert_in_ideal(expr, &Self::Expr::zero());
    }
}

pub trait Uair<R: Semiring> {
    fn num_cols() -> usize;

    fn constrain<B: ConstraintBuilder>(b: &mut B, up: &[B::Expr], down: &[B::Expr]);
}
