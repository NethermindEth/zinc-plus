pub mod constraint_counter;
pub mod dummy_semiring;
pub mod ideal;

use crypto_primitives::{FixedSemiring, Semiring};
use zinc_utils::{from_ref::FromRef, mul_by_scalar::MulByScalar};

use crate::ideal::{Ideal, IdealCheck};

pub trait ConstraintBuilder {
    type Expr: IdealCheck<Self::Ideal>;
    type Ideal: Ideal;

    fn assert_in_ideal(&mut self, expr: Self::Expr, ideal: &Self::Ideal);

    fn assert_zero(&mut self, expr: Self::Expr) {
        self.assert_in_ideal(expr, &Self::Ideal::zero_ideal());
    }
}

pub trait Uair<R: Semiring + 'static> {
    type Ideal: Ideal;

    fn num_cols() -> usize;

    fn constrain_general<B, FromR, MulByScalar>(
        b: &mut B,
        up: &[B::Expr],
        down: &[B::Expr],
        from_ref: FromR,
        mbs: MulByScalar,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&R) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &R) -> Option<B::Expr>,
        B::Ideal: FromRef<Self::Ideal>;

    fn constrain<B>(b: &mut B, up: &[B::Expr], down: &[B::Expr])
    where
        B: ConstraintBuilder,
        B::Expr: FromRef<R> + for<'a> MulByScalar<&'a R>,
        B::Ideal: FromRef<Self::Ideal>,
    {
        Self::constrain_general(b, up, down, B::Expr::from_ref, |x, y| {
            B::Expr::mul_by_scalar(x, y)
        })
    }
}
