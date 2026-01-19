use crypto_primitives::{FixedSemiring, crypto_bigint_int::Int};
use zinc_poly::univariate::{dense::DensePolynomial, ideal::DegreeOneIdeal};
use zinc_uair::{
    ConstraintBuilder, Uair,
    ideal::{Ideal, ZeroIdeal},
};
use zinc_utils::from_ref::FromRef;

pub struct TestUair;

impl<R: FixedSemiring + 'static> Uair<R> for TestUair {
    type Ideal = ZeroIdeal;

    fn num_cols() -> usize {
        3
    }

    fn constrain_general<B, FromR, MulByScalar>(
        b: &mut B,
        up: &[B::Expr],
        down: &[B::Expr],
        _from_ref: FromR,
        _mbs: MulByScalar,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&R) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &R) -> Option<B::Expr>,
        B::Ideal: FromRef<Self::Ideal>,
    {
        b.assert_in_ideal(up[0].clone() * &down[1] - &up[1], &B::Ideal::zero_ideal());
        b.assert_in_ideal(up[2].clone(), &B::Ideal::zero_ideal());
    }
}

pub struct TestAirNoMultiplication;

impl<const LIMBS: usize> Uair<DensePolynomial<Int<LIMBS>, 32>> for TestAirNoMultiplication {
    type Ideal = DegreeOneIdeal<Int<LIMBS>, ZeroIdeal, 32>;

    fn num_cols() -> usize {
        3
    }

    fn constrain_general<B, FromR, MulByScalar>(
        b: &mut B,
        up: &[B::Expr],
        _down: &[B::Expr],
        _from_ref: FromR,
        _mbs: MulByScalar,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&DensePolynomial<Int<LIMBS>, 32>) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &DensePolynomial<Int<LIMBS>, 32>) -> Option<B::Expr>,
        B::Ideal: FromRef<Self::Ideal>,
    {
        b.assert_in_ideal(
            up[0].clone() + &up[1] - &up[2],
            &B::Ideal::from_ref(&DegreeOneIdeal::new(Int::from(2))),
        );
    }
}
