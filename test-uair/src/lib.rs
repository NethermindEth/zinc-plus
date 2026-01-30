#![allow(clippy::arithmetic_side_effects)] // UAIRs should not care about overflows
mod generate_witness;

use crypto_primitives::{FixedSemiring, Semiring, crypto_bigint_int::Int};
use rand::{
    Rng,
    distr::{Distribution, StandardUniform},
};
use zinc_poly::{
    mle::{DenseMultilinearExtension, MultilinearExtensionRand},
    univariate::{dense::DensePolynomial, dynamic::DynamicPolynomial, ideal::DegreeOneIdeal},
};
use zinc_uair::{
    ConstraintBuilder, Uair,
    ideal::{Ideal, ZeroIdeal},
};
use zinc_utils::from_ref::FromRef;

pub use generate_witness::*;

pub struct TestUairSimpleMultiplication;

impl<R: Semiring + 'static> Uair<R> for TestUairSimpleMultiplication {
    type Ideal = ZeroIdeal;

    fn num_cols() -> usize {
        3
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: &[B::Expr],
        down: &[B::Expr],
        _from_ref: FromR,
        _mbs: MulByScalar,
        _ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
    {
        b.assert_in_ideal(up[0].clone() * &up[1] - &down[0], &B::Ideal::zero_ideal());
        b.assert_in_ideal(up[1].clone() * &up[2] - &down[1], &B::Ideal::zero_ideal());
        b.assert_in_ideal(up[0].clone() * &up[2] - &down[2], &B::Ideal::zero_ideal());
    }
}

impl<R, const DEGREE_PLUS_ONE: usize> GenerateWitness<DensePolynomial<R, DEGREE_PLUS_ONE>>
    for TestUairSimpleMultiplication
where
    R: FixedSemiring + 'static + FromRef<i8>,
    StandardUniform: Distribution<R>,
{
    fn generate_witness<Rng: rand::RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> Vec<DenseMultilinearExtension<DensePolynomial<R, DEGREE_PLUS_ONE>>> {
        let mut a: Vec<DynamicPolynomial<R>> = vec![DynamicPolynomial::new(vec![R::from_ref(
            &rng.random::<i8>(),
        )])];
        let mut b: Vec<DynamicPolynomial<R>> = vec![DynamicPolynomial::new(vec![
            R::zero(),
            R::from_ref(&rng.random::<i8>()),
        ])];
        let mut c: Vec<DynamicPolynomial<R>> = vec![DynamicPolynomial::new(vec![
            R::zero(),
            R::from_ref(&rng.random::<i8>()),
        ])];

        for i in 1..1 << num_vars {
            let prev_a = a[i - 1].clone();
            let prev_b = b[i - 1].clone();
            let prev_c = c[i - 1].clone();

            a.push(prev_a.clone() * &prev_b);
            b.push(prev_b * &prev_c);
            c.push(prev_a * prev_c);
        }

        vec![
            a.into_iter()
                .map(|x| {
                    assert!(
                        x.degree() < Some(DEGREE_PLUS_ONE),
                        "degree bound exceeded: {}",
                        x.degree().expect("if the degree is large it's not None")
                    );
                    DensePolynomial::new(x.coeffs)
                })
                .collect(),
            b.into_iter()
                .map(|x| {
                    assert!(
                        x.degree() < Some(DEGREE_PLUS_ONE),
                        "degree bound exceeded: {}",
                        x.degree().expect("if the degree is large it's not None"),
                    );
                    DensePolynomial::new(x.coeffs)
                })
                .collect(),
            c.into_iter()
                .map(|x| {
                    assert!(
                        x.degree() < Some(DEGREE_PLUS_ONE),
                        "degree bound exceeded: {}",
                        x.degree().expect("if the degree is large it's not None"),
                    );
                    DensePolynomial::new(x.coeffs)
                })
                .collect(),
        ]
    }
}

pub struct TestAirNoMultiplication;

impl<const LIMBS: usize> Uair<DensePolynomial<Int<LIMBS>, 32>> for TestAirNoMultiplication {
    type Ideal = DegreeOneIdeal<Int<LIMBS>>;

    fn num_cols() -> usize {
        3
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: &[B::Expr],
        _down: &[B::Expr],
        _from_ref: FromR,
        _mbs: MulByScalar,
        ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
    {
        b.assert_in_ideal(
            up[0].clone() + &up[1] - &up[2],
            &ideal_from_ref(&DegreeOneIdeal::new(Int::from(2))),
        );
    }
}

impl<const LIMBS: usize> GenerateWitness<DensePolynomial<Int<LIMBS>, 32>>
    for TestAirNoMultiplication
{
    fn generate_witness<Rng: rand::RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> Vec<DenseMultilinearExtension<DensePolynomial<Int<LIMBS>, 32>>> {
        let a: DenseMultilinearExtension<_> = DenseMultilinearExtension::rand(num_vars, rng)
            .into_iter()
            .map(|x| DensePolynomial::from(Int::from_i8(x)))
            .collect();

        let b: DenseMultilinearExtension<_> = DenseMultilinearExtension::rand(num_vars, rng)
            .into_iter()
            .map(|x| DensePolynomial::from(Int::from_i8(x)))
            .collect();

        let c = a.clone() + b.clone();

        vec![a, b, c]
    }
}
