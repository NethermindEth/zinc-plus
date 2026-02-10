#![allow(clippy::arithmetic_side_effects)] // UAIRs should not care about overflows
mod generate_witness;

use crypto_primitives::{FixedSemiring, Semiring, boolean::Boolean, crypto_bigint_int::Int};
use rand::{
    Rng,
    distr::{Distribution, StandardUniform},
};
use zinc_poly::{
    mle::{DenseMultilinearExtension, MultilinearExtensionRand},
    univariate::{
        binary::BinaryPoly, dense::DensePolynomial,
        dynamic::over_fixed_semiring::DynamicPolynomialFS, ideal::DegreeOneIdeal,
    },
};
use zinc_uair::{ConstraintBuilder, Uair};
use zinc_utils::from_ref::FromRef;

pub use generate_witness::*;
use zinc_uair::ideal::ImpossibleIdeal;

pub struct TestUairSimpleMultiplication;

impl<R: Semiring + 'static> Uair<R> for TestUairSimpleMultiplication {
    type Ideal = ImpossibleIdeal; // Not used

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
        b.assert_zero(up[0].clone() * &up[1] - &down[0]);
        b.assert_zero(up[1].clone() * &up[2] - &down[1]);
        b.assert_zero(up[0].clone() * &up[2] - &down[2]);
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
        let mut a: Vec<DynamicPolynomialFS<R>> = vec![DynamicPolynomialFS::new(vec![R::from_ref(
            &rng.random::<i8>(),
        )])];
        let mut b: Vec<DynamicPolynomialFS<R>> = vec![DynamicPolynomialFS::new(vec![
            R::zero(),
            R::from_ref(&rng.random::<i8>()),
        ])];
        let mut c: Vec<DynamicPolynomialFS<R>> = vec![DynamicPolynomialFS::new(vec![
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
        let a: DenseMultilinearExtension<DensePolynomial<Int<LIMBS>, 32>> =
            DenseMultilinearExtension::rand(num_vars, rng)
                .into_iter()
                .map(|x: u32| {
                    DensePolynomial::from_ref(&DensePolynomial::<Boolean, _>::from(
                        BinaryPoly::<32>::from(x),
                    ))
                })
                .collect();

        let b: DenseMultilinearExtension<_> = DenseMultilinearExtension::rand(num_vars, rng)
            .into_iter()
            .map(|x: u32| {
                DensePolynomial::from_ref(&DensePolynomial::<Boolean, _>::from(
                    BinaryPoly::<32>::from(x),
                ))
            })
            .collect();

        let c = a.clone() + b.clone();

        vec![a, b, c]
    }
}

pub struct TestAirScalarMultiplications;

impl<const LIMBS: usize> Uair<DensePolynomial<Int<LIMBS>, 32>> for TestAirScalarMultiplications {
    type Ideal = DegreeOneIdeal<Int<LIMBS>>;

    fn num_cols() -> usize {
        3
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: &[B::Expr],
        _down: &[B::Expr],
        from_ref: FromR,
        mbs: MulByScalar,
        ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
        FromR: Fn(&DensePolynomial<Int<LIMBS>, 32>) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &DensePolynomial<Int<LIMBS>, 32>) -> Option<B::Expr>,
    {
        b.assert_in_ideal(
            mbs(
                &up[0],
                &DensePolynomial::new([Int::from_i8(-1), Int::from_i8(0), Int::from_i8(1)]),
            )
            .expect("arithmetic overflow")
                + &up[1]
                - &up[2]
                + from_ref(&DensePolynomial::new([
                    Int::from_i8(1),
                    Int::from_i8(2),
                    Int::from_i8(3),
                    Int::from_i8(4),
                ])),
            &ideal_from_ref(&DegreeOneIdeal::new(Int::from(2))),
        );
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use zinc_uair::{
        collect_scalars::collect_scalars, constraint_counter::count_constraints,
        degree_counter::count_max_degree,
    };

    use super::*;

    const LIMBS: usize = 4;

    #[test]
    fn test_uair_simple_multiplication_correct_constraints_number() {
        assert_eq!(
            count_constraints::<DensePolynomial<Int<LIMBS>, 32>, TestUairSimpleMultiplication>(),
            3
        );
    }

    #[test]
    fn test_uair_simple_multiplication_correct_max_degree() {
        assert_eq!(
            count_max_degree::<DensePolynomial<Int<LIMBS>, 32>, TestUairSimpleMultiplication>(),
            2
        );
    }

    #[test]
    fn test_air_no_multiplication_correct_max_degree() {
        assert_eq!(
            count_max_degree::<DensePolynomial<Int<LIMBS>, 32>, TestAirNoMultiplication>(),
            1
        );
    }

    #[test]
    fn test_air_scalar_multiplications_correct_collect_scalars() {
        assert_eq!(
            collect_scalars::<DensePolynomial<Int<LIMBS>, 32>, TestAirScalarMultiplications>(),
            HashSet::from_iter(vec![
                DensePolynomial::new([Int::from_i8(-1), Int::from_i8(0), Int::from_i8(1)]),
                DensePolynomial::new([
                    Int::from_i8(1),
                    Int::from_i8(2),
                    Int::from_i8(3),
                    Int::from_i8(4),
                ])
            ])
        );
    }
}
