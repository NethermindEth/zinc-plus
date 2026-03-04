#![allow(clippy::arithmetic_side_effects)] // UAIRs should not care about overflows
mod generate_witness;

use std::marker::PhantomData;

use crypto_primitives::{ConstSemiring, FixedSemiring, Semiring, boolean::Boolean};
use num_traits::Zero;
use rand::{
    Rng,
    distr::{Distribution, StandardUniform},
};
use zinc_poly::{
    EvaluatablePolynomial,
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

impl<R> Uair for TestUairSimpleMultiplication<R>
where
    R: Semiring + 'static,
{
    type Ideal = ImpossibleIdeal; // Not used
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        UairSignature {
            binary_poly_cols: 0,
            arbitrary_poly_cols: 3,
            int_cols: 0,
        }
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        down: TraceRow<B::Expr>,
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

impl<R> GenerateSingleTypeWitness for TestUairSimpleMultiplication<R>
where
    R: FixedSemiring + From<i8> + 'static,
    StandardUniform: Distribution<R>,
{
    type Witness = DensePolynomial<R, 32>;

    fn generate_witness<Rng: rand::RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> Vec<DenseMultilinearExtension<DensePolynomial<R, 32>>> {
        let mut a: Vec<DynamicPolynomialFS<R>> =
            vec![DynamicPolynomialFS::new(vec![R::from(rng.random::<i8>())])];
        let mut b: Vec<DynamicPolynomialFS<R>> = vec![DynamicPolynomialFS::new(vec![
            R::zero(),
            R::from(rng.random::<i8>()),
        ])];
        let mut c: Vec<DynamicPolynomialFS<R>> = vec![DynamicPolynomialFS::new(vec![
            R::zero(),
            R::from(rng.random::<i8>()),
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

pub struct TestAirNoMultiplication<R>(PhantomData<R>);

impl<R> Uair for TestAirNoMultiplication<R>
where
    R: ConstSemiring + From<i32> + 'static,
{
    type Ideal = DegreeOneIdeal<R>;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        UairSignature {
            binary_poly_cols: 0,
            arbitrary_poly_cols: 3,
            int_cols: 0,
        }
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        _down: TraceRow<B::Expr>,
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
            &ideal_from_ref(&DegreeOneIdeal::new(R::from(2))),
        );
    }
}

impl<R> GenerateSingleTypeWitness for TestAirNoMultiplication<R>
where
    R: ConstSemiring + From<i32> + 'static,
{
    type Witness = DensePolynomial<R, 32>;

    fn generate_witness<Rng: rand::RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> Vec<DenseMultilinearExtension<DensePolynomial<R, 32>>> {
        let a: DenseMultilinearExtension<DensePolynomial<R, 32>> =
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

pub struct TestAirScalarMultiplications<R>(PhantomData<R>);

impl<R> Uair for TestAirScalarMultiplications<R>
where
    R: ConstSemiring + From<i8> + 'static,
{
    type Ideal = DegreeOneIdeal<R>;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        UairSignature {
            binary_poly_cols: 0,
            arbitrary_poly_cols: 3,
            int_cols: 0,
        }
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        _down: TraceRow<B::Expr>,
        from_ref: FromR,
        mbs: MulByScalar,
        ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
        FromR: Fn(&DensePolynomial<R, 32>) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &DensePolynomial<R, 32>) -> Option<B::Expr>,
    {
        let up = up.arbitrary_poly;

        b.assert_in_ideal(
            mbs(
                &up[0],
                &DensePolynomial::new([R::from(-1), R::from(0), R::from(1)]),
            )
            .expect("arithmetic overflow")
                + &up[1]
                - &up[2]
                + from_ref(&DensePolynomial::new([
                    R::from(1),
                    R::from(2),
                    R::from(3),
                    R::from(4),
                ])),
            &ideal_from_ref(&DegreeOneIdeal::new(R::from(2))),
        );
    }
}

pub struct BinaryDecompositionUair<R>(PhantomData<R>);

impl<R> Uair for BinaryDecompositionUair<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type Ideal = DegreeOneIdeal<R>;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        UairSignature {
            binary_poly_cols: 1,
            arbitrary_poly_cols: 0,
            int_cols: 1,
        }
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        _down: TraceRow<B::Expr>,
        _from_ref: FromR,
        _mbs: MulByScalar,
        ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&Self::Scalar) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &Self::Scalar) -> Option<B::Expr>,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
    {
        let int_col = &up.int[0];
        let binary_poly_col = &up.binary_poly[0];

        b.assert_in_ideal(
            binary_poly_col.clone() - int_col,
            &ideal_from_ref(&DegreeOneIdeal::new(R::from(2))),
        );
    }
}

impl<R> GenerateMultiTypeWitness for BinaryDecompositionUair<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_witness<Rng: rand::RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> (
        Vec<DenseMultilinearExtension<BinaryPoly<32>>>,
        Vec<DenseMultilinearExtension<DensePolynomial<Self::PolyCoeff, 32>>>,
        Vec<DenseMultilinearExtension<Self::Int>>,
    ) {
        let int_col_u32: DenseMultilinearExtension<u32> =
            DenseMultilinearExtension::rand(num_vars, rng);

        let binary_poly_col: DenseMultilinearExtension<BinaryPoly<32>> =
            int_col_u32.iter().map(|i| BinaryPoly::from(*i)).collect();

        let int_col = int_col_u32.into_iter().map(R::from).collect();

        (vec![binary_poly_col], vec![], vec![int_col])
    }
}

pub struct BigLinearUair<R>(PhantomData<R>);

impl<R> Uair for BigLinearUair<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type Ideal = DegreeOneIdeal<R>;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        UairSignature {
            binary_poly_cols: 16,
            arbitrary_poly_cols: 0,
            int_cols: 1,
        }
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        down: TraceRow<B::Expr>,
        _from_ref: FromR,
        _mbs: MulByScalar,
        ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&Self::Scalar) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &Self::Scalar) -> Option<B::Expr>,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
    {
        let one_ideal = DegreeOneIdeal::new(R::from(1));
        let two_ideal = DegreeOneIdeal::new(R::from(2));

        let sum_of_binary_polys = up.binary_poly[1..]
            .iter()
            .fold(up.binary_poly[0].clone(), |acc, next| acc + next);

        // up.binary_poly[0] + up.binary_poly[1] + ... up.binary_poly[16]
        //      = up.int[0] mod (X - 1)
        b.assert_in_ideal(
            sum_of_binary_polys - &up.int[0],
            &ideal_from_ref(&one_ideal),
        );

        // down.binary_poly[0] = up.int[0] mod (X - 1)
        b.assert_in_ideal(
            down.binary_poly[0].clone() - &up.int[0],
            &ideal_from_ref(&two_ideal),
        );

        // up.binary_poly[i] = down.binary_poly[i], for all i=1,...,15
        up.binary_poly[1..]
            .iter()
            .zip(&down.binary_poly[1..])
            .for_each(|(up, down)| {
                b.assert_zero(up.clone() - down);
            });
    }
}

impl<R> GenerateMultiTypeWitness for BigLinearUair<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_witness<Rng: rand::RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> (
        Vec<DenseMultilinearExtension<BinaryPoly<32>>>,
        Vec<DenseMultilinearExtension<DensePolynomial<Self::PolyCoeff, 32>>>,
        Vec<DenseMultilinearExtension<Self::Int>>,
    ) {
        let mut binary_poly_cols: Vec<DenseMultilinearExtension<BinaryPoly<32>>> =
            vec![(0..(1 << num_vars)).map(|_| BinaryPoly::zero()).collect(); 16];
        let mut int_col: DenseMultilinearExtension<Self::Int> =
            (0..(1 << num_vars)).map(|_| R::ZERO).collect();

        binary_poly_cols.iter_mut().for_each(|col| {
            col[0] = rng.random();
        });

        for i in 0..(1 << num_vars) - 1 {
            let int: u32 = binary_poly_cols
                .iter()
                .map(|col| col[i].evaluate_at_point(&1_u32).expect("should be fine"))
                .sum();
            int_col[i] = R::from(int);

            binary_poly_cols[0][i + 1] = BinaryPoly::from(int);
            binary_poly_cols[1..].iter_mut().for_each(|col| {
                col[i + 1] = col[i].clone();
            });
        }

        let len = int_col.len();

        int_col[len - 1] = R::from(
            binary_poly_cols
                .iter()
                .map(|col| {
                    col[len - 1]
                        .evaluate_at_point(&1_u32)
                        .expect("should be fine")
                })
                .sum::<u32>(),
        );

        (binary_poly_cols, vec![], vec![int_col])
    }
}

#[cfg(test)]
mod tests {
    use crypto_primitives::crypto_bigint_int::Int;
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
        assert_eq!(
            count_constraints::<TestAirNoMultiplication<Int<LIMBS>>>(),
            1
        );
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
        assert_eq!(count_max_degree::<TestAirNoMultiplication<Int<LIMBS>>>(), 1);
    }

    #[test]
    fn test_air_scalar_multiplications_correct_collect_scalars() {
        assert_eq!(
            collect_scalars::<TestAirScalarMultiplications<Int<LIMBS>>>(),
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
