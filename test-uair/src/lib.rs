#![allow(clippy::arithmetic_side_effects)] // UAIRs should not care about overflows
pub mod ecdsa;
mod generate_trace;
pub mod sha256;

pub use ecdsa::{ECDSA_INT_LIMBS, EcdsaScalarRing, EcdsaScalarSliceUair};
pub use generate_trace::*;
pub use sha256::{Sha256CompressionSliceUair, Sha256Ideal};

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
    AffineExpr, ConstraintBuilder, LookupColumnSpec, LookupTableType, PublicColumnLayout, ShiftSpec,
    TotalColumnLayout, TraceRow, Uair, UairSignature, UairTrace,
    ideal::{DegreeOneIdeal, ImpossibleIdeal},
};
use zinc_utils::from_ref::FromRef;

#[derive(Clone, Debug)]
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

#[derive(Clone, Debug)]
pub struct TestUairNoMultiplication<R>(PhantomData<R>);

impl<R> Uair for TestUairNoMultiplication<R>
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

impl<R> GenerateRandomTrace<32> for TestUairNoMultiplication<R>
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

#[derive(Clone, Debug)]
pub struct TestUairScalarMultiplications<R>(PhantomData<R>);

impl<R> Uair for TestUairScalarMultiplications<R>
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

#[derive(Clone, Debug)]
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

#[derive(Clone, Debug)]
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

#[derive(Clone, Debug)]
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

/// A second "big linear" UAIR with 14 binary-poly columns and 4 int columns,
/// used as a benchmarking shape distinct from `BigLinearUair`.
///
/// Constraints (0-based; `bp = up.binary_poly`, `int = up.int`):
///
/// - `bp[0][t+1] - bp[1] - bp[2] - bp[3] - int[0] - int[1] - int[2] ∈ (X-2)`
/// - `bp[4][t+4] - bp[5] - bp[6] - bp[7] - int[1] - int[2] - int[3] ∈ (X-2)`
/// - `bp[8] - int[0] ∈ (X-2)`
/// - `bp[9] - int[1] ∈ (X-2)`
/// - `bp[10] - X * bp[11] ∈ (X-1)`
/// - `bp[12] - X * bp[13] ∈ (X-1)`
///
/// Note the asymmetric shift amounts: `bp[0]` is shifted by 1 (used by C1)
/// and `bp[4]` is shifted by 4 (used by C2).
#[derive(Clone, Debug)]
pub struct ShaProxy<R>(PhantomData<R>);

impl<R> Uair for ShaProxy<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type Ideal = DegreeOneIdeal<R>;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        // 14 binary_poly cols, 0 arbitrary_poly cols, 4 int cols.
        let total = TotalColumnLayout::new(14, 0, 4);
        // c_1 (bp[0]) is shifted by 1 (used by C1 as bp[0][t+1]); c_5 (bp[4])
        // is shifted by 4 (used by C2 as bp[4][t+4]).
        let shifts = vec![ShiftSpec::new(0, 1), ShiftSpec::new(4, 4)];
        UairSignature::new(total, PublicColumnLayout::default(), shifts, vec![])
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        down: TraceRow<B::Expr>,
        _from_ref: FromR,
        mbs: MulByScalar,
        ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&Self::Scalar) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &Self::Scalar) -> Option<B::Expr>,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
    {
        let one_ideal = ideal_from_ref(&DegreeOneIdeal::new(R::ONE));
        let two_ideal = ideal_from_ref(&DegreeOneIdeal::new(R::from(2)));
        // The polynomial X = 0 + 1*X, used to express `X * c_k` via `mbs`.
        let x_scalar = DensePolynomial::<R, 32>::new([R::ZERO, R::from(1)]);

        // `down.binary_poly` is indexed by ShiftSpec position, not source col.
        // Our shifts vec is [ShiftSpec::new(0, 1), ShiftSpec::new(4, 1)], so
        // down.binary_poly[0] = bp[0][t+1], down.binary_poly[1] = bp[4][t+1].

        // (C1) dbp[0] - bp[1] - bp[2] - bp[3] - int[0] - int[1] - int[2] ∈ (X-2)
        b.assert_in_ideal(
            down.binary_poly[0].clone()
                - &up.binary_poly[1]
                - &up.binary_poly[2]
                - &up.binary_poly[3]
                - &up.int[0]
                - &up.int[1]
                - &up.int[2],
            &two_ideal,
        );

        // (C2) dbp[4] - bp[5] - bp[6] - bp[7] - int[1] - int[2] - int[3] ∈ (X-2)
        b.assert_in_ideal(
            down.binary_poly[1].clone()
                - &up.binary_poly[5]
                - &up.binary_poly[6]
                - &up.binary_poly[7]
                - &up.int[1]
                - &up.int[2]
                - &up.int[3],
            &two_ideal,
        );

        // (C3) bp[8] - int[0] ∈ (X-2)
        b.assert_in_ideal(up.binary_poly[8].clone() - &up.int[0], &two_ideal);

        // (C4) bp[9] - int[1] ∈ (X-2)
        b.assert_in_ideal(up.binary_poly[9].clone() - &up.int[1], &two_ideal);

        // (C5) bp[10] - X * bp[11] ∈ (X-1)
        b.assert_in_ideal(
            up.binary_poly[10].clone()
                - &mbs(&up.binary_poly[11], &x_scalar).expect("mul-by-X overflow"),
            &one_ideal,
        );

        // (C6) bp[12] - X * bp[13] ∈ (X-1)
        b.assert_in_ideal(
            up.binary_poly[12].clone()
                - &mbs(&up.binary_poly[13], &x_scalar).expect("mul-by-X overflow"),
            &one_ideal,
        );
    }
}

impl<R> GenerateRandomTrace<32> for ShaProxy<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type PolyCoeff = R;
    type Int = R;

    #[allow(clippy::needless_range_loop)]
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
                value |= 1_u32 << pos;
            }
            BinaryPoly::from(value)
        }

        // Bits used by the "small" binary polys that feed into C1/C2 sums.
        // Capping at 28 bits keeps each `eval(2)` value below 2^28 - 1, so the
        // sum used to construct `bp[0]` / `bp[4]` at the next row stays in u32:
        //     3 * (2^28 - 1) + 3 * 31  ≈  8.05 * 10^8  <  2^32 - 1.
        const SMALL_MASK: u32 = (1 << 28) - 1;
        // Range of values for the int columns (small non-negative).
        const INT_MAX_EXCL: u32 = 32;

        let len = 1 << num_vars;

        let mut bp_cols: Vec<DenseMultilinearExtension<BinaryPoly<32>>> =
            vec![(0..len).map(|_| BinaryPoly::zero()).collect(); 14];
        let mut int_cols: Vec<DenseMultilinearExtension<R>> =
            vec![(0..len).map(|_| R::ZERO).collect(); 4];

        // Row 0 / "head row" init for the columns whose value at the head of
        // the trace is unconstrained:
        //   - `bp[0]` is shifted by 1, so only `bp[0][0]` is unconstrained.
        //   - `bp[4]` is shifted by 4, so `bp[4][0..4]` are unconstrained (the C2
        //     fix-up at iteration `i` writes `bp[4][i+4]`, so the first 4 indices are
        //     never written by the loop).
        bp_cols[0][0] = BinaryPoly::from(rng.next_u32() & SMALL_MASK);
        for k in 0..4.min(len) {
            bp_cols[4][k] = BinaryPoly::from(rng.next_u32() & SMALL_MASK);
        }

        for i in 0..len {
            // bp[1..=3]: always small random binary polys (28-bit values).
            for k in 1..=3 {
                bp_cols[k][i] = BinaryPoly::from(rng.next_u32() & SMALL_MASK);
            }

            // For rows where the C2-target `bp[4][i+4]` is past the trace
            // boundary, the protocol reads it as zero-padded. To still
            // satisfy C2 at those rows, the C2 RHS sum must vanish at X = 2;
            // we achieve that by zeroing `bp[5..=7]` and `int[1..=3]` here.
            // (Mirrors the boundary trick used by `TestUairMixedShifts`.)
            // Row `len - 1` is exempt from constraint checking by the
            // protocol's last-row selector, so the zeroing only matters for
            // rows `len - 4 ..= len - 2`.
            let c2_target_oob = i + 4 >= len;

            for k in 5..=7 {
                bp_cols[k][i] = if c2_target_oob {
                    BinaryPoly::zero()
                } else {
                    BinaryPoly::from(rng.next_u32() & SMALL_MASK)
                };
            }

            // int[0..=3]: small non-negative values. Zero out int[1..=3] when
            // we're in the C2 boundary region (int[0] can stay random — it is
            // not used by C2).
            let int_vals: [u32; 4] = if c2_target_oob {
                [rng.next_u32() % INT_MAX_EXCL, 0, 0, 0]
            } else {
                [
                    rng.next_u32() % INT_MAX_EXCL,
                    rng.next_u32() % INT_MAX_EXCL,
                    rng.next_u32() % INT_MAX_EXCL,
                    rng.next_u32() % INT_MAX_EXCL,
                ]
            };
            for (k, v) in int_vals.iter().enumerate() {
                int_cols[k][i] = R::from(*v);
            }

            // (C3)/(C4): bp[8] = BinaryPoly::from(int[0]);
            // bp[9] = BinaryPoly::from(int[1]).
            // Since `BinaryPoly::from(n).evaluate_at_point(2) == n`,
            // this makes `bp[k] - int[k]` vanish at X=2, satisfying the (X-2) ideal check.
            bp_cols[8][i] = BinaryPoly::from(int_vals[0]);
            bp_cols[9][i] = BinaryPoly::from(int_vals[1]);

            // (C5): popcount(bp[10]) == popcount(bp[11]).
            let bp11: BinaryPoly<32> = rng.random();
            let popcount11 = bp11
                .evaluate_at_point(&1_u32)
                .expect("popcount eval should fit in u32");
            bp_cols[11][i] = bp11;
            bp_cols[10][i] = random_binary_poly_with_popcount(popcount11, rng);

            // (C6): popcount(bp[12]) == popcount(bp[13]).
            let bp13: BinaryPoly<32> = rng.random();
            let popcount13 = bp13
                .evaluate_at_point(&1_u32)
                .expect("popcount eval should fit in u32");
            bp_cols[13][i] = bp13;
            bp_cols[12][i] = random_binary_poly_with_popcount(popcount13, rng);

            // Set bp[0][i+1] and bp[4][i+4] so C1 and C2 respectively hold at
            // row i. Each summand fits in u32 (bp eval ≤ 2^28 - 1, ints ≤ 31),
            // so each sum (≤ 3 * (2^28 - 1) + 3 * 31 ≈ 8.05e8) stays well
            // below 2^32. C1 and C2 have different shift amounts now (1 vs 4)
            // and so need separate `if` guards.
            let eval_at_2 = |bp: &BinaryPoly<32>| -> u32 {
                bp.evaluate_at_point(&2_u32)
                    .expect("28-bit binary poly eval at 2 fits in u32")
            };

            // C1 (shift = 1)
            if i + 1 < len {
                let s1 = eval_at_2(&bp_cols[1][i])
                    + eval_at_2(&bp_cols[2][i])
                    + eval_at_2(&bp_cols[3][i])
                    + int_vals[0]
                    + int_vals[1]
                    + int_vals[2];
                bp_cols[0][i + 1] = BinaryPoly::from(s1);
            }

            // C2 (shift = 4)
            if i + 4 < len {
                let s2 = eval_at_2(&bp_cols[5][i])
                    + eval_at_2(&bp_cols[6][i])
                    + eval_at_2(&bp_cols[7][i])
                    + int_vals[1]
                    + int_vals[2]
                    + int_vals[3];
                bp_cols[4][i + 4] = BinaryPoly::from(s2);
            }
        }

        UairTrace {
            binary_poly: bp_cols.into(),
            arbitrary_poly: vec![].into(),
            int: int_cols.into(),
        }
    }
}

/// Test UAIR with mixed shift amounts.
/// 3 columns (a, b, c): column a shifts by 1, column b shifts by 2.
/// Constraints are linear (degree 1).
#[derive(Clone, Debug)]
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

// ---------------------------------------------------------------------------
// IntLookupUair: minimal UAIR declaring a lookup argument.
//
// Layout: 2 int witness columns (no binary_poly, no arbitrary_poly, no shifts).
//   col[0]: `v`, the witness column whose values must all lie in
//           `T = {0, 1, …, 255}` (Word table with `width = 8`).
//   col[1]: `m`, the multiplicity column — `m[j]` is the count of
//           occurrences of `T[j]` in `v`.
//
// Constraint: a trivially-satisfied `0 ∈ ⟨X - 2⟩`, so that the ideal
// check has exactly one (no-op) constraint — required by the protocol
// infrastructure but unrelated to the lookup.
//
// The trace has `2^num_vars` rows. The caller must pass `num_vars = 8`
// so that the row count matches the table size (`2^width = 256`). No
// chunking is used in this test UAIR.
// ---------------------------------------------------------------------------

/// Fixed table width for `IntLookupUair` — rows are 8-bit words.
pub const INT_LOOKUP_TABLE_WIDTH: usize = 8;

#[derive(Clone, Debug)]
pub struct IntLookupUair<R>(PhantomData<R>);

impl<R> Uair for IntLookupUair<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type Ideal = DegreeOneIdeal<R>;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        // 2 int witness columns: v (index 0) and m (index 1).
        let total = TotalColumnLayout::new(0, 0, 2);
        let lookup_specs = vec![LookupColumnSpec {
            expression: AffineExpr::single(0),
            table_type: LookupTableType::Word {
                width: INT_LOOKUP_TABLE_WIDTH,
                chunk_width: None,
            },
        }];
        UairSignature::new(total, PublicColumnLayout::default(), vec![], lookup_specs)
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
        // Trivially-satisfied constraint: (v - v) ∈ ⟨X - 2⟩.
        // Needed because the existing protocol infrastructure expects
        // at least one constraint. The actual lookup is proved outside
        // the ideal-check path.
        let v = &up.int[0];
        b.assert_in_ideal(
            v.clone() - v,
            &ideal_from_ref(&DegreeOneIdeal::new(R::from(2))),
        );
    }
}

impl<R> GenerateRandomTrace<32> for IntLookupUair<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_random_trace<Rng: rand::RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, R, R, 32> {
        // `num_vars >= table_width` is required so every column MLE has
        // `2^num_vars` rows. When `num_vars > table_width` the
        // multiplicity column is padded with zeros for indices past the
        // table (those slots in the logup leaf layout contribute
        // `0 / (α - padded_table_val)` == 0 to the cumulative sum).
        assert!(
            num_vars >= INT_LOOKUP_TABLE_WIDTH,
            "IntLookupUair requires num_vars >= {INT_LOOKUP_TABLE_WIDTH}"
        );
        let row_count = 1usize << num_vars;
        let table_size = 1usize << INT_LOOKUP_TABLE_WIDTH;

        // Witness values: each entry uniform in [0, table_size).
        let witness_raw: Vec<u32> = (0..row_count)
            .map(|_| (rng.random::<u32>()) % (table_size as u32))
            .collect();

        // Multiplicities: m[j] = count of j in witness_raw for
        // j in 0..table_size; 0 for the padding slots j >= table_size.
        let mut mults_raw = vec![0u32; row_count];
        for &v in &witness_raw {
            mults_raw[v as usize] += 1;
        }

        // Build MLEs explicitly at `num_vars` variables (relying on the
        // `FromIterator` pad-to-next-power-of-two would pick the wrong
        // number of variables when `row_count` happens to already be a
        // power of two but smaller than `2^num_vars`).
        let zero = R::from(0u32);
        let v_col = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            witness_raw.into_iter().map(R::from).collect(),
            zero.clone(),
        );
        let m_col = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            mults_raw.into_iter().map(R::from).collect(),
            zero,
        );

        UairTrace {
            binary_poly: vec![].into(),
            arbitrary_poly: vec![].into(),
            int: vec![v_col, m_col].into(),
        }
    }
}

// ---------------------------------------------------------------------------
// IntLookupMultiUair: two int witness columns sharing a single lookup
// group against a `Word { width: 8 }` table.
//
// Layout: 3 int witness columns (no binary_poly, no arbitrary_poly, no
// shifts).
//   col[0]: `v_0` — first lookup witness column
//   col[1]: `v_1` — second lookup witness column (same table type, so
//                   grouped together by `group_lookup_specs`)
//   col[2]: `m`   — shared multiplicity column (one per group)
//
// Constraint: trivially-satisfied `0 ∈ ⟨X - 2⟩` (same workaround as
// `IntLookupUair`).
//
// `num_vars` must satisfy `num_vars >= INT_LOOKUP_TABLE_WIDTH`.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct IntLookupMultiUair<R>(PhantomData<R>);

impl<R> Uair for IntLookupMultiUair<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type Ideal = DegreeOneIdeal<R>;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        // 3 int witness columns. Both v_0 and v_1 look up into the
        // same `Word { width: 8 }` table — `group_lookup_specs` will
        // produce a single group with column_indices = [0, 1].
        let total = TotalColumnLayout::new(0, 0, 3);
        let lookup_specs = vec![
            LookupColumnSpec {
                expression: AffineExpr::single(0),
                table_type: LookupTableType::Word {
                    width: INT_LOOKUP_TABLE_WIDTH,
                    chunk_width: None,
                },
            },
            LookupColumnSpec {
                expression: AffineExpr::single(1),
                table_type: LookupTableType::Word {
                    width: INT_LOOKUP_TABLE_WIDTH,
                    chunk_width: None,
                },
            },
        ];
        UairSignature::new(total, PublicColumnLayout::default(), vec![], lookup_specs)
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
        // Trivially-satisfied constraint (same as IntLookupUair).
        let v = &up.int[0];
        b.assert_in_ideal(
            v.clone() - v,
            &ideal_from_ref(&DegreeOneIdeal::new(R::from(2))),
        );
    }
}

impl<R> GenerateRandomTrace<32> for IntLookupMultiUair<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_random_trace<Rng: rand::RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, R, R, 32> {
        assert!(
            num_vars >= INT_LOOKUP_TABLE_WIDTH,
            "IntLookupMultiUair requires num_vars >= {INT_LOOKUP_TABLE_WIDTH}"
        );
        let row_count = 1usize << num_vars;
        let table_size = 1usize << INT_LOOKUP_TABLE_WIDTH;

        // Two independent witness columns, each uniform in [0, table_size).
        let gen_col = |rng: &mut Rng| -> Vec<u32> {
            (0..row_count)
                .map(|_| (rng.random::<u32>()) % (table_size as u32))
                .collect()
        };
        let v0_raw: Vec<u32> = gen_col(rng);
        let v1_raw: Vec<u32> = gen_col(rng);

        // Shared multiplicities: count occurrences of each table entry
        // across BOTH witness columns.
        let mut mults_raw = vec![0u32; row_count];
        for &v in v0_raw.iter().chain(v1_raw.iter()) {
            mults_raw[v as usize] += 1;
        }

        let zero = R::from(0u32);
        let v0_col = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            v0_raw.into_iter().map(R::from).collect(),
            zero.clone(),
        );
        let v1_col = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            v1_raw.into_iter().map(R::from).collect(),
            zero.clone(),
        );
        let m_col = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            mults_raw.into_iter().map(R::from).collect(),
            zero,
        );

        UairTrace {
            binary_poly: vec![].into(),
            arbitrary_poly: vec![].into(),
            int: vec![v0_col, v1_col, m_col].into(),
        }
    }
}

// ---------------------------------------------------------------------------
// IntLookupWithPublicUair: `IntLookupUair` shape, but prepended with one
// public int column. Exercises the Phase 2g path that reconstructs public
// per-column evaluations at the multipoint-eval / reducer opening points
// and splices them with witness evals from the reducer proof.
//
// Layout: 3 int columns (1 public + 2 witness); the witness columns mirror
// `IntLookupUair`.
//   col[0] (public):  `p` — public data, no constraints refer to it.
//   col[1] (witness): `v` — looks up into `Word { width: 8 }`.
//   col[2] (witness): `m` — multiplicity column (last-col convention).
//
// `num_vars >= INT_LOOKUP_TABLE_WIDTH` is required (same as `IntLookupUair`).
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct IntLookupWithPublicUair<R>(PhantomData<R>);

impl<R> Uair for IntLookupWithPublicUair<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type Ideal = DegreeOneIdeal<R>;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        // Layout: 1 public int col + 2 witness int cols (v, m).
        let total = TotalColumnLayout::new(0, 0, 3);
        let public = PublicColumnLayout::new(0, 0, 1);
        // `v` sits at full-trace index 1 (public col is index 0).
        let lookup_specs = vec![LookupColumnSpec {
            expression: AffineExpr::single(1),
            table_type: LookupTableType::Word {
                width: INT_LOOKUP_TABLE_WIDTH,
                chunk_width: None,
            },
        }];
        UairSignature::new(total, public, vec![], lookup_specs)
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
        // Trivially-satisfied constraint on `v` (witness col, up.int[1]).
        let v = &up.int[1];
        b.assert_in_ideal(
            v.clone() - v,
            &ideal_from_ref(&DegreeOneIdeal::new(R::from(2))),
        );
    }
}

impl<R> GenerateRandomTrace<32> for IntLookupWithPublicUair<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_random_trace<Rng: rand::RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, R, R, 32> {
        assert!(
            num_vars >= INT_LOOKUP_TABLE_WIDTH,
            "IntLookupWithPublicUair requires num_vars >= {INT_LOOKUP_TABLE_WIDTH}"
        );
        let row_count = 1usize << num_vars;
        let table_size = 1usize << INT_LOOKUP_TABLE_WIDTH;

        // Public column: arbitrary u32-valued data, no constraint coupling.
        let public_raw: Vec<u32> = (0..row_count).map(|_| rng.random::<u32>()).collect();

        // Lookup witness.
        let witness_raw: Vec<u32> = (0..row_count)
            .map(|_| (rng.random::<u32>()) % (table_size as u32))
            .collect();

        let mut mults_raw = vec![0u32; row_count];
        for &v in &witness_raw {
            mults_raw[v as usize] += 1;
        }

        let zero = R::from(0u32);
        let p_col = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            public_raw.into_iter().map(R::from).collect(),
            zero.clone(),
        );
        let v_col = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            witness_raw.into_iter().map(R::from).collect(),
            zero.clone(),
        );
        let m_col = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            mults_raw.into_iter().map(R::from).collect(),
            zero,
        );

        UairTrace {
            binary_poly: vec![].into(),
            arbitrary_poly: vec![].into(),
            int: vec![p_col, v_col, m_col].into(),
        }
    }
}

// ---------------------------------------------------------------------------
// ByteDecompLookupUair: decompose a 16-bit witness into two byte chunks,
// each constrained to live in `Word { width: 8 }`. Exercises the canonical
// lookup use case — range-checking chunks of a larger value — with real
// constraint coupling between the witness columns (as opposed to the
// trivially-satisfied constraints in `IntLookupUair` / `IntLookupMultiUair`).
//
// Layout: 4 int witness columns.
//   col[0]: `v`      — 16-bit value, row-wise uniform in [0, 2^16).
//   col[1]: `c_low`  — low byte of v: `c_low[i] = v[i] mod 256`.
//   col[2]: `c_high` — high byte of v: `c_high[i] = v[i] / 256`.
//   col[3]: `m`      — shared multiplicity of both lookups into
//                      `Word { width: 8 }` (group_lookup_specs collapses
//                      c_low and c_high into one group).
//
// Constraint: `v[i] - c_low[i] - 256 * c_high[i]  ∈  ⟨X - 2⟩`
//   (enforced per-row by the ideal check — on the trace, this forces
//    the left-hand side to be exactly 0).
//
// Together with the lookup (c_low, c_high ∈ [0, 256)), this is a real
// range check: any byte outside [0, 256) would have to break either a
// lookup identity or the decomposition constraint.
//
// `num_vars >= INT_LOOKUP_TABLE_WIDTH` is required (same as IntLookupUair).
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ByteDecompLookupUair<R>(PhantomData<R>);

impl<R> Uair for ByteDecompLookupUair<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type Ideal = DegreeOneIdeal<R>;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        // 4 int witness columns: v, c_low, c_high, m.
        let total = TotalColumnLayout::new(0, 0, 4);
        let lookup_specs = vec![
            LookupColumnSpec {
                expression: AffineExpr::single(1),
                table_type: LookupTableType::Word {
                    width: INT_LOOKUP_TABLE_WIDTH,
                    chunk_width: None,
                },
            },
            LookupColumnSpec {
                expression: AffineExpr::single(2),
                table_type: LookupTableType::Word {
                    width: INT_LOOKUP_TABLE_WIDTH,
                    chunk_width: None,
                },
            },
        ];
        UairSignature::new(total, PublicColumnLayout::default(), vec![], lookup_specs)
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        _down: TraceRow<B::Expr>,
        _from_ref: FromR,
        mbs: MulByScalar,
        ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&Self::Scalar) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &Self::Scalar) -> Option<B::Expr>,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
    {
        let v = &up.int[0];
        let c_low = &up.int[1];
        let c_high = &up.int[2];

        // 2^chunk_width = 256 (the "base" of the decomposition).
        let base: DensePolynomial<R, 32> =
            DensePolynomial::new([R::from(1u32 << INT_LOOKUP_TABLE_WIDTH)]);

        // v - c_low - 256 * c_high  ∈  ⟨X - 2⟩
        b.assert_in_ideal(
            v.clone() - c_low - &mbs(c_high, &base).expect("256 * c_high"),
            &ideal_from_ref(&DegreeOneIdeal::new(R::from(2))),
        );
    }
}

impl<R> GenerateRandomTrace<32> for ByteDecompLookupUair<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_random_trace<Rng: rand::RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, R, R, 32> {
        assert!(
            num_vars >= INT_LOOKUP_TABLE_WIDTH,
            "ByteDecompLookupUair requires num_vars >= {INT_LOOKUP_TABLE_WIDTH}"
        );
        let row_count = 1usize << num_vars;
        let table_size = 1u32 << INT_LOOKUP_TABLE_WIDTH;

        // 16-bit values uniform in [0, 2^16). Chunks are derived.
        let v_raw: Vec<u32> = (0..row_count)
            .map(|_| rng.random::<u32>() & 0xFFFF)
            .collect();
        let c_low_raw: Vec<u32> = v_raw.iter().map(|x| x & 0xFF).collect();
        let c_high_raw: Vec<u32> = v_raw.iter().map(|x| (x >> 8) & 0xFF).collect();

        // Shared multiplicity: count across BOTH chunk columns.
        let mut mults_raw = vec![0u32; row_count];
        for &c in c_low_raw.iter().chain(c_high_raw.iter()) {
            debug_assert!(c < table_size, "chunk must lie in the byte table");
            mults_raw[c as usize] += 1;
        }

        let zero = R::from(0u32);
        let build_col = |raw: Vec<u32>| -> DenseMultilinearExtension<R> {
            DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                raw.into_iter().map(R::from).collect(),
                zero.clone(),
            )
        };

        UairTrace {
            binary_poly: vec![].into(),
            arbitrary_poly: vec![].into(),
            int: vec![
                build_col(v_raw),
                build_col(c_low_raw),
                build_col(c_high_raw),
                build_col(mults_raw),
            ]
            .into(),
        }
    }
}

// ---------------------------------------------------------------------------
// AffineSumLookupUair: demonstrates Phase 2i's AffineExpr lookup spec.
//
// Single lookup group with one non-trivial affine expression — `x + y`,
// range-checked into `Word { width: 2 }`. This exercises:
//   * prover step4b synthesizing the combined MLE from the two input
//     column MLEs (not just reading a single committed column);
//   * verifier step5 reconstructing the combined value at `ρ_row` via
//     MLE linearity (`eval[x+y](ρ) = eval[x](ρ) + eval[y](ρ)`), then
//     cross-checking against the lookup argument's `witness_evals[0]`.
//
// Layout (3 int witness columns):
//   col[0]: `x` — row-wise in `{0, 1}`
//   col[1]: `y` — row-wise in `{0, 1}`
//   col[2]: `m` — multiplicity for Word{width:2} (table entries 0..3)
//
// `num_vars >= 2` is required so the 4-entry table fits with zero-padding.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct AffineSumLookupUair<R>(PhantomData<R>);

impl<R> Uair for AffineSumLookupUair<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type Ideal = DegreeOneIdeal<R>;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(0, 0, 3);
        // x + y ∈ Word{2}, column 0 = x, column 1 = y.
        let lookup_specs = vec![LookupColumnSpec {
            expression: AffineExpr {
                terms: vec![(0, 1), (1, 1)],
                constant: 0,
            },
            table_type: LookupTableType::Word {
                width: 2,
                chunk_width: None,
            },
        }];
        UairSignature::new(total, PublicColumnLayout::default(), vec![], lookup_specs)
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
        // Trivially-satisfied constraint (same pattern as IntLookupUair).
        let x = &up.int[0];
        b.assert_in_ideal(
            x.clone() - x,
            &ideal_from_ref(&DegreeOneIdeal::new(R::from(2u32))),
        );
    }
}

impl<R> GenerateRandomTrace<32> for AffineSumLookupUair<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_random_trace<Rng: rand::RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, R, R, 32> {
        assert!(num_vars >= 2, "AffineSumLookupUair requires num_vars >= 2");
        let row_count = 1usize << num_vars;

        let x_raw: Vec<u32> = (0..row_count).map(|_| rng.next_u32() & 1).collect();
        let y_raw: Vec<u32> = (0..row_count).map(|_| rng.next_u32() & 1).collect();

        // m counts occurrences of x[r] + y[r] ∈ {0, 1, 2} in the table
        // domain (index 3 stays 0 since 3 never appears). Indices past
        // the 4-entry table are naturally 0 as well.
        let mut mults_raw = vec![0u32; row_count];
        for (&x, &y) in x_raw.iter().zip(y_raw.iter()) {
            mults_raw[(x + y) as usize] += 1;
        }

        let zero = R::from(0u32);
        let build = |raw: Vec<u32>| -> DenseMultilinearExtension<R> {
            DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                raw.into_iter().map(R::from).collect(),
                zero.clone(),
            )
        };
        UairTrace {
            binary_poly: vec![].into(),
            arbitrary_poly: vec![].into(),
            int: vec![build(x_raw), build(y_raw), build(mults_raw)].into(),
        }
    }
}

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
        assert_uair_shape::<TestUairNoMultiplication<Int<LIMBS>>>(&[1]);
        assert_uair_shape::<TestUairScalarMultiplications<Int<LIMBS>>>(&[1]);
        assert_uair_shape::<BinaryDecompositionUair<u32>>(&[1]);
        assert_uair_shape::<BigLinearUair<u32>>(&[1; 17]);
        assert_uair_shape::<TestUairMixedShifts<Int<LIMBS>>>(&[1, 1]);
    }

    #[test]
    fn test_air_scalar_multiplications_correct_collect_scalars() {
        assert_eq!(
            collect_scalars::<TestUairScalarMultiplications<Int<LIMBS>>>(),
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
