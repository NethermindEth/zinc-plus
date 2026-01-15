pub mod constraint_counter;
pub mod dummy_semiring;
pub mod ideal;

use crypto_primitives::{FixedSemiring, Semiring};

use crate::ideal::Ideal;

pub trait ConstraintBuilder {
    type Expr: FixedSemiring;
    type Ideal: Ideal<Self::Expr>;

    fn assert_in_ideal<I>(&mut self, expr: Self::Expr, ideal: &I)
    where
        I: Ideal<Self::Expr>,
        Self::Ideal: From<I>;

    fn assert_zero(&mut self, expr: Self::Expr) {
        self.assert_in_ideal(expr, &Self::Ideal::zero_ideal());
    }
}

pub trait Uair<R: Semiring> {
    type Ideal: Ideal<R>;

    fn num_cols() -> usize;

    fn constrain<B>(b: &mut B, up: &[B::Expr], down: &[B::Expr])
    where
        B: ConstraintBuilder,
        B::Ideal: From<Self::Ideal>;
}
