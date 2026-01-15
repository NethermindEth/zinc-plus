pub mod constraint_counter;
pub mod dummy_semiring;
pub mod ideal;

use crypto_primitives::FixedSemiring;
use zinc_utils::from_ref::FromRef;

use crate::ideal::Ideal;

pub trait ConstraintBuilder<R: FixedSemiring> {
    type Expr: FixedSemiring;
    type Ideal: Ideal<R>;

    fn assert_in_ideal(&mut self, expr: Self::Expr, ideal: &Self::Ideal);

    fn assert_zero(&mut self, expr: Self::Expr) {
        self.assert_in_ideal(expr, &Self::Ideal::zero_ideal());
    }
}

pub trait Uair<R: FixedSemiring> {
    type Ideal: Ideal<R>;

    fn num_cols() -> usize;

    fn constrain<B>(b: &mut B, up: &[B::Expr], down: &[B::Expr])
    where
        B: ConstraintBuilder<R>,
        B::Ideal: FromRef<Self::Ideal>;
}
