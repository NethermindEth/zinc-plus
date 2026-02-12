#![allow(clippy::arithmetic_side_effects)] // UAIRs should not care about overflows
mod generate_witness;

use std::marker::PhantomData;

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
use zinc_uair::{ConstraintBuilder, TraceRow, Uair, UairSignature};
use zinc_utils::from_ref::FromRef;

pub use generate_witness::*;
use zinc_uair::ideal::ImpossibleIdeal;

pub struct TestUairSimpleMultiplication<R>(PhantomData<R>);

impl<R: Semiring + 'static> Uair for TestUairSimpleMultiplication<R> {
    type Ideal = ImpossibleIdeal; // Not used
    type Scalar = R;

    fn signature() -> UairSignature {
        UairSignature {
            binary_poly_cols: 0,
            arbitrary_poly_cols: 3,
            int_cols: 0,
        }
    }

    fn constrain_general<'a, B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<'a, B::Expr>,
        down: TraceRow<'a, B::Expr>,
        _from_ref: FromR,
        _mbs: MulByScalar,
        _ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
    {
        let up = up.arbitrary_poly;
        let down = down.arbitrary_poly;

        b.assert_zero(up[0].clone() * &up[1] - &down[0]);
        b.assert_zero(up[1].clone() * &up[2] - &down[1]);
        b.assert_zero(up[0].clone() * &up[2] - &down[2]);
    }
}

impl<R> GenerateWitness for TestUairSimpleMultiplication<R>
where
    R: FixedSemiring + 'static + FromRef<i8>,
    StandardUniform: Distribution<R>,
{
    type Witness = DensePolynomial<R, 32>;

    fn generate_witness<Rng: rand::RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> Vec<DenseMultilinearExtension<DensePolynomial<R, 32>>> {
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
                        x.degree() < Some(32),
                        "degree bound exceeded: {}",
                        x.degree().expect("if the degree is large it's not None")
                    );
                    DensePolynomial::new(x.coeffs)
                })
                .collect(),
            b.into_iter()
                .map(|x| {
                    assert!(
                        x.degree() < Some(32),
                        "degree bound exceeded: {}",
                        x.degree().expect("if the degree is large it's not None"),
                    );
                    DensePolynomial::new(x.coeffs)
                })
                .collect(),
            c.into_iter()
                .map(|x| {
                    assert!(
                        x.degree() < Some(32),
                        "degree bound exceeded: {}",
                        x.degree().expect("if the degree is large it's not None"),
                    );
                    DensePolynomial::new(x.coeffs)
                })
                .collect(),
        ]
    }
}

pub struct TestAirNoMultiplication<const LIMBS: usize>;

impl<const LIMBS: usize> Uair for TestAirNoMultiplication<LIMBS> {
    type Ideal = DegreeOneIdeal<Int<LIMBS>>;
    type Scalar = DensePolynomial<Int<LIMBS>, 32>;

    fn signature() -> UairSignature {
        UairSignature {
            binary_poly_cols: 0,
            arbitrary_poly_cols: 3,
            int_cols: 0,
        }
    }

    fn constrain_general<'a, B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<'a, B::Expr>,
        _down: TraceRow<'a, B::Expr>,
        _from_ref: FromR,
        _mbs: MulByScalar,
        ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
    {
        let up = up.arbitrary_poly;

        b.assert_in_ideal(
            up[0].clone() + &up[1] - &up[2],
            &ideal_from_ref(&DegreeOneIdeal::new(Int::from(2))),
        );
    }
}

impl<const LIMBS: usize> GenerateWitness for TestAirNoMultiplication<LIMBS> {
    type Witness = DensePolynomial<Int<LIMBS>, 32>;

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

pub struct TestAirScalarMultiplications<const LIMBS: usize>;

impl<const LIMBS: usize> Uair for TestAirScalarMultiplications<LIMBS> {
    type Ideal = DegreeOneIdeal<Int<LIMBS>>;
    type Scalar = DensePolynomial<Int<LIMBS>, 32>;

    fn signature() -> UairSignature {
        UairSignature {
            binary_poly_cols: 0,
            arbitrary_poly_cols: 3,
            int_cols: 0,
        }
    }

    fn constrain_general<'a, B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<'a, B::Expr>,
        _down: TraceRow<'a, B::Expr>,
        from_ref: FromR,
        mbs: MulByScalar,
        ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
        FromR: Fn(&DensePolynomial<Int<LIMBS>, 32>) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &DensePolynomial<Int<LIMBS>, 32>) -> Option<B::Expr>,
    {
        let up = up.arbitrary_poly;

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
    use zinc_uair::{
        collect_scalars::collect_scalars, constraint_counter::count_constraints,
        degree_counter::count_max_degree,
    };

    use super::*;

    const LIMBS: usize = 4;

    #[test]
    fn test_uair_simple_multiplication_correct_constraints_number() {
        assert_eq!(
            count_constraints::<TestUairSimpleMultiplication<Int<LIMBS>>>(),
            3
        );
    }

    #[test]
    fn test_air_no_multiplication_correct_constraints_number() {
        assert_eq!(count_constraints::<TestAirNoMultiplication<LIMBS>>(), 1);
    }

    #[test]
    fn test_uair_simple_multiplication_correct_max_degree() {
        assert_eq!(
            count_max_degree::<TestUairSimpleMultiplication<Int<LIMBS>>>(),
            2
        );
    }

    #[test]
    fn test_air_no_multiplication_correct_max_degree() {
        assert_eq!(count_max_degree::<TestAirNoMultiplication<LIMBS>>(), 1);
    }

    #[test]
    fn test_air_scalar_multiplications_correct_collect_scalars() {
        assert_eq!(
            collect_scalars::<TestAirScalarMultiplications<LIMBS>>(),
            (vec![
                DensePolynomial::new([Int::from_i8(-1), Int::from_i8(0), Int::from_i8(1)]),
                DensePolynomial::new([
                    Int::from_i8(1),
                    Int::from_i8(2),
                    Int::from_i8(3),
                    Int::from_i8(4),
                ])
            ]
            .into_iter()
            .collect())
        );
    }
}
