#![allow(clippy::arithmetic_side_effects)] // UAIRs should not care about overflows
mod generate_trace;

pub use generate_trace::*;

use crypto_primitives::{ConstSemiring, FixedSemiring, Semiring, boolean::Boolean};
use num_traits::Zero;
use rand::{
    distr::{Distribution, StandardUniform},
    prelude::*,
};
use std::marker::PhantomData;
use zinc_poly::{
    EvaluatablePolynomial,
    mle::{DenseMultilinearExtension, MultilinearExtensionRand},
    univariate::{
        binary::BinaryPoly, dense::DensePolynomial,
        dynamic::over_fixed_semiring::DynamicPolynomialFS,
    },
};
use zinc_uair::{
    ConstraintBuilder, LookupColumnSpec, LookupTableType, PublicColumnLayout, ShiftSpec,
    TotalColumnLayout, TraceRow, Uair, UairSignature, UairTrace,
    ideal::{ImpossibleIdeal, degree_one::DegreeOneIdeal},
};
use zinc_utils::from_ref::FromRef;

pub struct TestUairSimpleMultiplication<R>(PhantomData<R>);

impl<R> Uair for TestUairSimpleMultiplication<R>
where
    R: Semiring + 'static,
{
    type Ideal = ImpossibleIdeal; // Not used
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(0, 3, 0);
        let shifts = (0..3).map(|i| ShiftSpec::new(i, 1)).collect();
        UairSignature::new(total, PublicColumnLayout::default(), shifts, vec![])
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

impl<R> GenerateRandomTrace<32> for TestUairSimpleMultiplication<R>
where
    R: FixedSemiring + From<i8> + 'static,
    StandardUniform: Distribution<R>,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_random_trace<Rng: RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, R, R, 32> {
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

        let arbitrary_poly = vec![
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
        .into();
        UairTrace {
            arbitrary_poly,
            ..Default::default()
        }
    }
}

pub struct TestAirNoMultiplication<R>(PhantomData<R>); // TODO: Rename to XxxUairXxx

impl<R> Uair for TestAirNoMultiplication<R>
where
    R: ConstSemiring + From<i32> + 'static,
{
    type Ideal = DegreeOneIdeal<R>;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(0, 3, 0);
        UairSignature::new(total, PublicColumnLayout::default(), vec![], vec![])
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

impl<R> GenerateRandomTrace<32> for TestAirNoMultiplication<R>
where
    R: ConstSemiring + From<i32> + 'static,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_random_trace<Rng: rand::RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, R, R, 32> {
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

        UairTrace {
            arbitrary_poly: vec![a, b, c].into(),
            ..Default::default()
        }
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
        let total = TotalColumnLayout::new(0, 3, 0);
        UairSignature::new(total, PublicColumnLayout::default(), vec![], vec![])
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
        let total = TotalColumnLayout::new(1, 0, 1);
        UairSignature::new(total, PublicColumnLayout::default(), vec![], vec![])
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

impl<R> GenerateRandomTrace<32> for BinaryDecompositionUair<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_random_trace<Rng: rand::RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, R, R, 32> {
        let int_col_u32: DenseMultilinearExtension<u32> =
            DenseMultilinearExtension::rand(num_vars, rng);

        let binary_poly_col: DenseMultilinearExtension<BinaryPoly<32>> =
            int_col_u32.iter().map(|i| BinaryPoly::from(*i)).collect();

        let int_col = int_col_u32.into_iter().map(R::from).collect();

        UairTrace {
            binary_poly: vec![binary_poly_col].into(),
            arbitrary_poly: vec![].into(),
            int: vec![int_col].into(),
        }
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
        let total = TotalColumnLayout::new(16, 0, 1);
        let shifts = (0..16).map(|i| ShiftSpec::new(i, 1)).collect();
        UairSignature::new(total, PublicColumnLayout::default(), shifts, vec![])
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

        // down.binary_poly[i](1) = up.binary_poly[i](1), for all i=1,...,15
        // (preserves popcount across rows, but allows the bit pattern to change)
        up.binary_poly[1..]
            .iter()
            .zip(&down.binary_poly[1..])
            .for_each(|(up, down)| {
                b.assert_in_ideal(up.clone() - down, &ideal_from_ref(&one_ideal));
            });
    }
}

impl<R> GenerateRandomTrace<32> for BigLinearUair<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_random_trace<Rng: rand::RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, R, R, 32> {
        /// Generate a random binary polynomial with the given number of 1-bits.
        fn random_binary_poly_with_popcount(
            popcount: u32,
            rng: &mut (impl rand::RngCore + ?Sized),
        ) -> BinaryPoly<32> {
            let mut positions: [u8; 32] =
                core::array::from_fn(|i| u8::try_from(i).expect("can't fail"));
            for i in 0..popcount as usize {
                let j = i + rng.next_u32() as usize % (32 - i);
                positions.swap(i, j);
            }
            let mut value: u32 = 0;
            for &pos in &positions[..popcount as usize] {
                value |= 1u32 << pos;
            }
            BinaryPoly::from(value)
        }

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
                let popcount = col[i].evaluate_at_point(&1_u32).expect("should be fine");
                col[i + 1] = random_binary_poly_with_popcount(popcount, rng);
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

        UairTrace {
            binary_poly: binary_poly_cols.into(),
            arbitrary_poly: vec![].into(),
            int: vec![int_col].into(),
        }
    }
}

pub struct BigLinearUairWithPublicInput<R>(PhantomData<R>);

impl<R> Uair for BigLinearUairWithPublicInput<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type Ideal = <BigLinearUair<R> as Uair>::Ideal;
    type Scalar = <BigLinearUair<R> as Uair>::Scalar;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(16, 0, 1);
        let public = PublicColumnLayout::new(4, 0, 0);
        let shifts = (0..16).map(|i| ShiftSpec::new(i, 1)).collect();
        UairSignature::new(total, public, shifts, vec![])
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        down: TraceRow<B::Expr>,
        from_ref: FromR,
        mbs: MulByScalar,
        ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&Self::Scalar) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &Self::Scalar) -> Option<B::Expr>,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
    {
        BigLinearUair::<R>::constrain_general(b, up, down, from_ref, mbs, ideal_from_ref)
    }
}

impl<R> GenerateRandomTrace<32> for BigLinearUairWithPublicInput<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type PolyCoeff = <BigLinearUair<R> as GenerateRandomTrace<32>>::PolyCoeff;
    type Int = <BigLinearUair<R> as GenerateRandomTrace<32>>::Int;

    fn generate_random_trace<Rng: RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, Self::PolyCoeff, Self::Int, 32> {
        BigLinearUair::<R>::generate_random_trace(num_vars, rng)
    }
}

/// Test UAIR with mixed shift amounts.
/// 3 columns (a, b, c): column a shifts by 1, column b shifts by 2.
/// Constraints are linear (degree 1).
pub struct TestUairMixedShifts<R>(PhantomData<R>);

impl<R> Uair for TestUairMixedShifts<R>
where
    R: Semiring + 'static,
{
    type Ideal = ImpossibleIdeal;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(0, 3, 0);
        let shifts = vec![
            ShiftSpec::new(0, 1), // a shifted by 1
            ShiftSpec::new(1, 2), // b shifted by 2
        ];
        UairSignature::new(total, PublicColumnLayout::default(), shifts, vec![])
    }

    // Constraints:
    //   a[i+1] = a[i] + b[i]  →  down[0] - up[0] - up[1] = 0
    //   c[i]   = b[i+2]       →  up[2] - down[1] = 0
    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        builder: &mut B,
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

        builder.assert_zero(down[0].clone() - &up[0] - &up[1]);
        builder.assert_zero(up[2].clone() - &down[1]);
    }
}

impl<R> GenerateRandomTrace<32> for TestUairMixedShifts<R>
where
    R: FixedSemiring + From<i8> + 'static,
    StandardUniform: Distribution<R>,
{
    type PolyCoeff = R;
    type Int = R;

    // Witness: random b, derive a from a[i+1] = a[i] + b[i], set c[i] = b[i+2].
    fn generate_random_trace<Rng: rand::RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, R, R, 32> {
        let n = 1 << num_vars;

        // Random b column (degree-0 polynomials to stay under degree 32)
        let b_col: Vec<DynamicPolynomialFS<R>> = (0..n)
            .map(|_| DynamicPolynomialFS::new(vec![R::from(rng.random::<i8>())]))
            .collect();

        // a[0] random, a[i+1] = a[i] + b[i]
        let mut a_col: Vec<DynamicPolynomialFS<R>> =
            vec![DynamicPolynomialFS::new(vec![R::from(rng.random::<i8>())])];
        for i in 0..n - 1 {
            a_col.push(a_col[i].clone() + &b_col[i]);
        }

        // c[i] = b[i+2], zero-padded for last 2 entries
        let mut c_col: Vec<DynamicPolynomialFS<R>> = Vec::with_capacity(n);
        for i in 0..n {
            if i + 2 < n {
                c_col.push(b_col[i + 2].clone());
            } else {
                c_col.push(DynamicPolynomialFS::zero());
            }
        }

        let to_mle = |col: Vec<DynamicPolynomialFS<R>>| -> DenseMultilinearExtension<DensePolynomial<R, 32>> {
            col.into_iter()
                .map(|x| DensePolynomial::new(x.coeffs))
                .collect()
        };

        UairTrace {
            arbitrary_poly: vec![to_mle(a_col), to_mle(b_col), to_mle(c_col)].into(),
            ..Default::default()
        }
    }
}

/// Minimal UAIR with a lookup constraint: one int column looked up
/// against `Word(4)` (values in {0, ..., 15}). Two int columns total,
/// with a trivial constraint `a - b = 0` to keep the CPR non-degenerate.
pub struct SimpleLookupUair<R>(PhantomData<R>);

impl<R> Uair for SimpleLookupUair<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type Ideal = ImpossibleIdeal;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(0, 0, 2);
        UairSignature::new(
            total,
            PublicColumnLayout::default(),
            vec![],
            vec![LookupColumnSpec {
                column_index: 0,
                table_type: LookupTableType::Word {
                    width: 4,
                    chunk_width: None,
                },
            }],
        )
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
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
    {
        b.assert_zero(up.int[0].clone() - &up.int[1]);
    }
}

/// Generate a `GenerateRandomTrace` impl for a UAIR with 2 int columns where
/// column 0 = column 1, values drawn uniformly from `0..modulus`.
macro_rules! impl_int_copy_trace {
    ($name:ident, $modulus:expr) => {
        impl<R> GenerateRandomTrace<32> for $name<R>
        where
            R: ConstSemiring + From<u32> + 'static,
        {
            type PolyCoeff = R;
            type Int = R;

            fn generate_random_trace<Rng: rand::RngCore + ?Sized>(
                num_vars: usize,
                rng: &mut Rng,
            ) -> UairTrace<'static, R, R, 32> {
                let n = 1 << num_vars;
                let vals: Vec<R> = (0..n).map(|_| R::from(rng.next_u32() % $modulus)).collect();
                let a: DenseMultilinearExtension<R> = vals.clone().into_iter().collect();
                let b: DenseMultilinearExtension<R> = vals.into_iter().collect();
                UairTrace {
                    int: vec![a, b].into(),
                    ..Default::default()
                }
            }
        }
    };
}

impl_int_copy_trace!(SimpleLookupUair, 16);

/// UAIR with two int columns both looked up against the same `Word(4)` table.
/// Tests L>1 (multi-column) lookup within a single group.
/// 3 int columns total: a, b looked up; c = a + b (trivial constraint).
pub struct MultiColLookupUair<R>(PhantomData<R>);

impl<R> Uair for MultiColLookupUair<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type Ideal = ImpossibleIdeal;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(0, 0, 3);
        UairSignature::new(
            total,
            PublicColumnLayout::default(),
            vec![],
            vec![
                LookupColumnSpec {
                    column_index: 0,
                    table_type: LookupTableType::Word {
                        width: 4,
                        chunk_width: None,
                    },
                },
                LookupColumnSpec {
                    column_index: 1,
                    table_type: LookupTableType::Word {
                        width: 4,
                        chunk_width: None,
                    },
                },
            ],
        )
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
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
    {
        b.assert_zero(up.int[0].clone() + &up.int[1] - &up.int[2]);
    }
}

/// Generate a `GenerateRandomTrace` impl for a UAIR with 3 int columns where
/// `c = a + b`, with `a` in `0..range_a` and `b` in `0..range_b`.
macro_rules! impl_sum_pair_trace {
    ($name:ident, $range_a:expr, $range_b:expr) => {
        impl<R> GenerateRandomTrace<32> for $name<R>
        where
            R: ConstSemiring + From<u32> + 'static,
        {
            type PolyCoeff = R;
            type Int = R;

            fn generate_random_trace<Rng: rand::RngCore + ?Sized>(
                num_vars: usize,
                rng: &mut Rng,
            ) -> UairTrace<'static, R, R, 32> {
                let n = 1 << num_vars;
                let a_vals: Vec<R> = (0..n).map(|_| R::from(rng.next_u32() % $range_a)).collect();
                let b_vals: Vec<R> = (0..n).map(|_| R::from(rng.next_u32() % $range_b)).collect();
                let c_vals: Vec<R> = a_vals
                    .iter()
                    .zip(b_vals.iter())
                    .map(|(a, b)| a.clone() + b)
                    .collect();
                let a: DenseMultilinearExtension<R> = a_vals.into_iter().collect();
                let b: DenseMultilinearExtension<R> = b_vals.into_iter().collect();
                let c: DenseMultilinearExtension<R> = c_vals.into_iter().collect();
                UairTrace {
                    int: vec![a, b, c].into(),
                    ..Default::default()
                }
            }
        }
    };
}

impl_sum_pair_trace!(MultiColLookupUair, 16, 16);

/// UAIR with two columns looking up different table types:
/// column 0 → Word(4), column 1 → Word(8). Tests multiple lookup groups.
/// 3 int columns: a in {0..15}, b in {0..255}, c = a + b.
pub struct MultiGroupLookupUair<R>(PhantomData<R>);

impl<R> Uair for MultiGroupLookupUair<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type Ideal = ImpossibleIdeal;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(0, 0, 3);
        UairSignature::new(
            total,
            PublicColumnLayout::default(),
            vec![],
            vec![
                LookupColumnSpec {
                    column_index: 0,
                    table_type: LookupTableType::Word {
                        width: 4,
                        chunk_width: None,
                    },
                },
                LookupColumnSpec {
                    column_index: 1,
                    table_type: LookupTableType::Word {
                        width: 8,
                        chunk_width: None,
                    },
                },
            ],
        )
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
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
    {
        b.assert_zero(up.int[0].clone() + &up.int[1] - &up.int[2]);
    }
}

impl_sum_pair_trace!(MultiGroupLookupUair, 16, 256);

/// UAIR with a BitPoly lookup: one binary_poly column looked up against
/// `BitPoly(8)`. Tests the BitPoly table type.
/// 1 binary_poly column + 1 int column, constraint: binary_poly - int ∈ <X-2>.
pub struct BitPolyLookupUair<R>(PhantomData<R>);

impl<R> Uair for BitPolyLookupUair<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type Ideal = DegreeOneIdeal<R>;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(1, 0, 1);
        UairSignature::new(
            total,
            PublicColumnLayout::default(),
            vec![],
            vec![LookupColumnSpec {
                column_index: 0,
                table_type: LookupTableType::BitPoly {
                    width: 8,
                    chunk_width: None,
                },
            }],
        )
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
        b.assert_in_ideal(
            up.binary_poly[0].clone() - &up.int[0],
            &ideal_from_ref(&DegreeOneIdeal::new(R::from(2))),
        );
    }
}

/// Generate a `GenerateRandomTrace` impl for a UAIR with 1 binary_poly column
/// and 1 int column, both carrying the same values in `0..modulus`.
macro_rules! impl_bitpoly_int_trace {
    ($name:ident, $modulus:expr) => {
        impl<R> GenerateRandomTrace<32> for $name<R>
        where
            R: ConstSemiring + From<u32> + 'static,
        {
            type PolyCoeff = R;
            type Int = R;

            fn generate_random_trace<Rng: rand::RngCore + ?Sized>(
                num_vars: usize,
                rng: &mut Rng,
            ) -> UairTrace<'static, R, R, 32> {
                let n = 1 << num_vars;
                let values: Vec<u32> = (0..n).map(|_| rng.next_u32() % $modulus).collect();
                let bp: DenseMultilinearExtension<BinaryPoly<32>> =
                    values.iter().map(|&v| BinaryPoly::from(v)).collect();
                let int: DenseMultilinearExtension<R> = values.into_iter().map(R::from).collect();
                UairTrace {
                    binary_poly: vec![bp].into(),
                    int: vec![int].into(),
                    ..Default::default()
                }
            }
        }
    };
}

impl_bitpoly_int_trace!(BitPolyLookupUair, 256);

/// Decomposed Word lookup: Word(8) with chunk_width=4, K=2.
/// 2 int columns: a looked up (decomposed), b = a (trivial constraint).
pub struct DecomposedWordUair<R>(PhantomData<R>);

impl<R> Uair for DecomposedWordUair<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type Ideal = ImpossibleIdeal;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        UairSignature::new(
            TotalColumnLayout::new(0, 0, 2),
            PublicColumnLayout::default(),
            vec![],
            vec![LookupColumnSpec {
                column_index: 0,
                table_type: LookupTableType::Word {
                    width: 8,
                    chunk_width: Some(4),
                },
            }],
        )
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
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
    {
        b.assert_zero(up.int[0].clone() - &up.int[1]);
    }
}

impl_int_copy_trace!(DecomposedWordUair, 256);

/// Decomposed BitPoly lookup: BitPoly(8) with chunk_width=4, K=2.
/// 1 binary_poly column looked up (decomposed) + 1 int column.
/// Constraint: binary_poly[0] - int[0] ∈ <X-2>.
pub struct DecomposedBitPolyUair<R>(PhantomData<R>);

impl<R> Uair for DecomposedBitPolyUair<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type Ideal = DegreeOneIdeal<R>;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        UairSignature::new(
            TotalColumnLayout::new(1, 0, 1),
            PublicColumnLayout::default(),
            vec![],
            vec![LookupColumnSpec {
                column_index: 0,
                table_type: LookupTableType::BitPoly {
                    width: 8,
                    chunk_width: Some(4),
                },
            }],
        )
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
        b.assert_in_ideal(
            up.binary_poly[0].clone() - &up.int[0],
            &ideal_from_ref(&DegreeOneIdeal::new(R::from(2))),
        );
    }
}

impl_bitpoly_int_trace!(DecomposedBitPolyUair, 256);

// ---------------------------------------------------------------------------
// Comparison UAIRs: lookup vs pure-constraint for the same 8-bit range check
// ---------------------------------------------------------------------------

/// 8-bit range check via lookup: Word(8), 2 int columns, a=b.
/// Compare against `RangeCheck8Uair` to measure lookup vs constraint cost.
pub struct Word8LookupUair<R>(PhantomData<R>);

impl<R> Uair for Word8LookupUair<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type Ideal = ImpossibleIdeal;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        UairSignature::new(
            TotalColumnLayout::new(0, 0, 2),
            PublicColumnLayout::default(),
            vec![],
            vec![LookupColumnSpec {
                column_index: 0,
                table_type: LookupTableType::Word {
                    width: 8,
                    chunk_width: None,
                },
            }],
        )
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
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
    {
        b.assert_zero(up.int[0].clone() - &up.int[1]);
    }
}

impl_int_copy_trace!(Word8LookupUair, 256);

/// 8-bit range check via binary decomposition (no lookup):
/// 1 binary_poly column (8-bit values) + 1 int column, constraint mod 2.
/// Compare against `Word8LookupUair` to measure constraint vs lookup cost.
pub struct RangeCheck8Uair<R>(PhantomData<R>);

impl<R> Uair for RangeCheck8Uair<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type Ideal = DegreeOneIdeal<R>;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        UairSignature::new(
            TotalColumnLayout::new(1, 0, 1),
            PublicColumnLayout::default(),
            vec![],
            vec![],
        )
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
        b.assert_in_ideal(
            up.binary_poly[0].clone() - &up.int[0],
            &ideal_from_ref(&DegreeOneIdeal::new(R::from(2))),
        );
    }
}

impl_bitpoly_int_trace!(RangeCheck8Uair, 256);

#[cfg(test)]
mod tests {
    use crypto_primitives::crypto_bigint_int::Int;
    use zinc_uair::{
        collect_scalars::collect_scalars,
        constraint_counter::count_constraints,
        degree_counter::{count_constraint_degrees, count_max_degree},
    };

    use super::*;

    const LIMBS: usize = 4;

    #[test]
    fn test_constraint_degrees() {
        fn assert_uair_shape<U: Uair>(expected_degrees: &[usize]) {
            assert_eq!(count_constraints::<U>(), expected_degrees.len());
            assert_eq!(count_constraint_degrees::<U>(), expected_degrees);
            assert_eq!(
                count_max_degree::<U>(),
                *expected_degrees.iter().max().unwrap()
            );
        }

        assert_uair_shape::<TestUairSimpleMultiplication<Int<LIMBS>>>(&[2, 2, 2]);
        assert_uair_shape::<TestAirNoMultiplication<Int<LIMBS>>>(&[1]);
        assert_uair_shape::<TestAirScalarMultiplications<Int<LIMBS>>>(&[1]);
        assert_uair_shape::<BinaryDecompositionUair<u32>>(&[1]);
        assert_uair_shape::<BigLinearUair<u32>>(&[1; 17]);
        assert_uair_shape::<TestUairMixedShifts<Int<LIMBS>>>(&[1, 1]);
        assert_uair_shape::<SimpleLookupUair<u32>>(&[1]);
        assert_uair_shape::<MultiColLookupUair<u32>>(&[1]);
        assert_uair_shape::<MultiGroupLookupUair<u32>>(&[1]);
        assert_uair_shape::<BitPolyLookupUair<u32>>(&[1]);
        assert_uair_shape::<DecomposedWordUair<u32>>(&[1]);
        assert_uair_shape::<DecomposedBitPolyUair<u32>>(&[1]);
        assert_uair_shape::<Word8LookupUair<u32>>(&[1]);
        assert_uair_shape::<RangeCheck8Uair<u32>>(&[1]);
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
