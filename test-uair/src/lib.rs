#![allow(clippy::arithmetic_side_effects)] // UAIRs should not care about overflows
mod generate_witness;

use std::marker::PhantomData;

use crypto_primitives::{FixedSemiring, Semiring, boolean::Boolean, crypto_bigint_int::Int};
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
use zinc_uair::{ConstraintBuilder, ShiftSpec, TraceRow, Uair, UairSignature};
use zinc_utils::from_ref::FromRef;

pub use generate_witness::*;
use zinc_uair::ideal::ImpossibleIdeal;

pub struct TestUairSimpleMultiplication<R>(PhantomData<R>);

impl<R: Semiring + 'static> Uair for TestUairSimpleMultiplication<R> {
    type Ideal = ImpossibleIdeal; // Not used
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        UairSignature {
            binary_poly_cols: 0,
            arbitrary_poly_cols: 3,
            int_cols: 0,
            shifts: vec![],
            public_columns: vec![],
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
            shifts: vec![],
            public_columns: vec![],
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
            &ideal_from_ref(&DegreeOneIdeal::new(Int::from(2))),
        );
    }
}

impl<const LIMBS: usize> GenerateSingleTypeWitness for TestAirNoMultiplication<LIMBS> {
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
            shifts: vec![],
            public_columns: vec![],
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

pub struct BinaryDecompositionUair;

impl Uair for BinaryDecompositionUair {
    type Ideal = DegreeOneIdeal<u32>;
    type Scalar = DensePolynomial<u32, 32>;

    fn signature() -> UairSignature {
        UairSignature {
            binary_poly_cols: 1,
            arbitrary_poly_cols: 0,
            int_cols: 1,
            shifts: vec![],
            public_columns: vec![],
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
            &ideal_from_ref(&DegreeOneIdeal::new(2)),
        );
    }
}

impl GenerateMultiTypeWitness for BinaryDecompositionUair {
    type PolyCoeff = u32;
    type Int = u32;

    fn generate_witness<Rng: rand::RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> (
        Vec<DenseMultilinearExtension<BinaryPoly<32>>>,
        Vec<DenseMultilinearExtension<DensePolynomial<Self::PolyCoeff, 32>>>,
        Vec<DenseMultilinearExtension<Self::Int>>,
    ) {
        let int_col: DenseMultilinearExtension<u32> =
            DenseMultilinearExtension::rand(num_vars, rng);

        let binary_poly_col: DenseMultilinearExtension<BinaryPoly<32>> =
            int_col.iter().map(|i| BinaryPoly::from(*i)).collect();

        (vec![binary_poly_col], vec![], vec![int_col])
    }
}

pub struct BigLinearUair;

impl Uair for BigLinearUair {
    type Ideal = DegreeOneIdeal<u32>;
    type Scalar = DensePolynomial<u32, 32>;

    fn signature() -> UairSignature {
        UairSignature {
            binary_poly_cols: 16,
            arbitrary_poly_cols: 0,
            int_cols: 1,
            shifts: vec![],
            public_columns: vec![],
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
        let sum_of_binary_polys = up.binary_poly[1..]
            .iter()
            .fold(up.binary_poly[0].clone(), |acc, next| acc + next);

        // up.binary_poly[0] + up.binary_poly[1] + ... up.binary_poly[16]
        //      = up.int[0] mod (X - 1)
        b.assert_in_ideal(
            sum_of_binary_polys - &up.int[0],
            &ideal_from_ref(&DegreeOneIdeal::new(1)),
        );

        // down.binary_poly[0] = up.int[0] mod (X - 1)
        b.assert_in_ideal(
            down.binary_poly[0].clone() - &up.int[0],
            &ideal_from_ref(&DegreeOneIdeal::new(2)),
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

impl GenerateMultiTypeWitness for BigLinearUair {
    type PolyCoeff = u32;
    type Int = u32;

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
        let mut int_col: DenseMultilinearExtension<u32> = (0..(1 << num_vars)).map(|_| 0).collect();

        binary_poly_cols.iter_mut().for_each(|col| {
            col[0] = rng.random();
        });

        for i in 0..(1 << num_vars) - 1 {
            int_col[i] = binary_poly_cols
                .iter()
                .map(|col| col[i].evaluate_at_point(&1u32).expect("should be fine"))
                .sum();

            binary_poly_cols[0][i + 1] = BinaryPoly::from(int_col[i]);
            binary_poly_cols[1..].iter_mut().for_each(|col| {
                col[i + 1] = col[i].clone();
            });
        }

        let len = int_col.len();

        int_col[len - 1] = binary_poly_cols
            .iter()
            .map(|col| {
                col[len - 1]
                    .evaluate_at_point(&1u32)
                    .expect("should be fine")
            })
            .sum();

        (binary_poly_cols, vec![], vec![int_col])
    }
}

/// A simple UAIR with one public column for testing the public-inputs
/// pipeline.
///
/// Layout: 2 binary_poly columns.
/// - Column 0 (`a`): private witness.
/// - Column 1 (`b`): public input.
///
/// Single constraint: `a - b = 0` (i.e. `a = b`).
/// In F₂\[X\], subtraction is the same as addition (XOR).
pub struct PublicColumnTestUair;

impl Uair for PublicColumnTestUair {
    type Ideal = ImpossibleIdeal;
    type Scalar = BinaryPoly<32>;

    fn signature() -> UairSignature {
        UairSignature {
            binary_poly_cols: 2,
            arbitrary_poly_cols: 0,
            int_cols: 0,
            shifts: vec![],
            // Column 1 is public.
            public_columns: vec![1],
        }
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        _down: TraceRow<B::Expr>,
        _from_ref: FromR,
        _mbs: MulByScalar,
        _ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
    {
        // a = b
        b.assert_zero(up.binary_poly[0].clone() - &up.binary_poly[1]);
    }
}

impl GenerateMultiTypeWitness for PublicColumnTestUair {
    type PolyCoeff = u32;
    type Int = u32;

    fn generate_witness<R: rand::RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut R,
    ) -> (
        Vec<DenseMultilinearExtension<BinaryPoly<32>>>,
        Vec<DenseMultilinearExtension<DensePolynomial<Self::PolyCoeff, 32>>>,
        Vec<DenseMultilinearExtension<Self::Int>>,
    ) {
        // Both columns have identical values (satisfying a = b).
        let col: DenseMultilinearExtension<BinaryPoly<32>> =
            (0..(1 << num_vars)).map(|_| rng.random()).collect();
        let col_copy = col.clone();
        (vec![col, col_copy], vec![], vec![])
    }
}

/// A UAIR that exercises **shifted public columns**.
///
/// Layout: 2 binary_poly columns.
/// - Column 0 (`a`): private witness.
/// - Column 1 (`b`): public input.
///
/// One shift: left-shift-by-1 of column 1.
///   `shift_1(b)[i] = b[i+1]` for `i < N−1`, `0` for `i = N−1`.
///
/// Constraint: `a - shift_1(b) = 0` (i.e. `a[i] = b[i+1]`).
///
/// The verifier must evaluate the MLE of the (unshifted) source column
/// `b` at the shift sumcheck challenge point by itself, since column 1
/// is public.
pub struct PublicShiftTestUair;

impl Uair for PublicShiftTestUair {
    type Ideal = ImpossibleIdeal;
    type Scalar = BinaryPoly<32>;

    fn signature() -> UairSignature {
        UairSignature {
            binary_poly_cols: 2,
            arbitrary_poly_cols: 0,
            int_cols: 0,
            // Shift-by-1 of column 1 (the public column).
            shifts: vec![ShiftSpec {
                source_col: 1,
                shift_amount: 1,
            }],
            // Column 1 is public.
            public_columns: vec![1],
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
        // a = shift_1(b)
        // down.binary_poly[0] is the shift-by-1 of column 1.
        b.assert_zero(up.binary_poly[0].clone() - &down.binary_poly[0]);
    }
}

impl GenerateMultiTypeWitness for PublicShiftTestUair {
    type PolyCoeff = u32;
    type Int = u32;

    fn generate_witness<R: rand::RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut R,
    ) -> (
        Vec<DenseMultilinearExtension<BinaryPoly<32>>>,
        Vec<DenseMultilinearExtension<DensePolynomial<Self::PolyCoeff, 32>>>,
        Vec<DenseMultilinearExtension<Self::Int>>,
    ) {
        let n = 1usize << num_vars;
        // Column b: random public input.
        let b_vals: Vec<BinaryPoly<32>> = (0..n).map(|_| rng.random()).collect();
        // Column a: a[i] = b[i+1] for i < N-1, a[N-1] = 0.
        let mut a_vals: Vec<BinaryPoly<32>> = Vec::with_capacity(n);
        for i in 0..n {
            if i + 1 < n {
                a_vals.push(b_vals[i + 1].clone());
            } else {
                a_vals.push(BinaryPoly::zero());
            }
        }
        let col_a: DenseMultilinearExtension<BinaryPoly<32>> = a_vals.into_iter().collect();
        let col_b: DenseMultilinearExtension<BinaryPoly<32>> = b_vals.into_iter().collect();
        (vec![col_a, col_b], vec![], vec![])
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
