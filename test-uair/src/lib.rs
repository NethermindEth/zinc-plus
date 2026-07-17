#![allow(clippy::arithmetic_side_effects)] // UAIRs should not care about overflows
mod generate_trace;

pub use generate_trace::*;

use crypto_primitives::{ConstSemiring, FixedConfig, Semiring, SemiringConfig, boolean::Boolean};
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
        binary::BinaryPoly,
        dense::DensePolynomial,
        dynamic::{DynamicPolynomial, HasDynamicPolynomialConfig},
    },
};
use zinc_uair::{
    BitOp, BitOpSpec, ConstraintBuilder, PublicColumnLayout, ShiftSpec, TotalColumnLayout,
    TraceRow, Uair, UairSignature, UairTrace,
    ideal::{DegreeOneIdeal, ImpossibleIdeal},
};
use zinc_utils::from_ref::FromRef;

#[derive(Clone, Debug)]
pub struct TestUairSimpleMultiplication<R, P>(PhantomData<(R, P)>);

impl<R, P> Uair for TestUairSimpleMultiplication<R, P>
where
    R: Semiring + 'static,
    P: Semiring + 'static,
{
    type Ideal = ImpossibleIdeal; // Not used
    type FqIdeal = ImpossibleIdeal;
    type Scalar = DensePolynomial<R, 32>;
    type Prime = P;

    fn signature() -> UairSignature<Self::Prime> {
        let total = TotalColumnLayout::new(0, 3, 0);
        let shifts = (0..3).map(|i| ShiftSpec::new(i, 1)).collect();
        UairSignature::new(total, PublicColumnLayout::default(), shifts, vec![])
    }

    fn constrain_general<C, B, FromR, MulByScalar, IFromR, IFqFromR>(
        b: &mut B,
        expr_cfg: &C,
        up: TraceRow<C::Element>,
        down: TraceRow<C::Element>,
        _from_ref: FromR,
        _mbs: MulByScalar,
        _ideal_from_ref: IFromR,
        _fq_ideal_from_ref: IFqFromR,
    ) where
        C: SemiringConfig,
        B: ConstraintBuilder<Expr = C::Element>,
    {
        let up = up.arbitrary_poly;
        let down = down.arbitrary_poly;

        b.assert_zero(expr_cfg.sub(&expr_cfg.mul(&up[0], &up[1]), &down[0]));
        b.assert_zero(expr_cfg.sub(&expr_cfg.mul(&up[1], &up[2]), &down[1]));
        b.assert_zero(expr_cfg.sub(&expr_cfg.mul(&up[0], &up[2]), &down[2]));
    }
}

impl<R, P> GenerateRandomTrace<32> for TestUairSimpleMultiplication<R, P>
where
    R: Semiring + From<i8> + 'static,
    P: Semiring + 'static,
    StandardUniform: Distribution<R>,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_random_trace<G: Rng + ?Sized>(
        num_vars: usize,
        rng: &mut G,
    ) -> UairTrace<'static, R, R, 32, 32> {
        let ring_cfg = FixedConfig::<R>::default();
        let poly_cfg = ring_cfg.dyn_poly_cfg();

        let mut a: Vec<DynamicPolynomial<R>> =
            vec![DynamicPolynomial::new(vec![R::from(rng.random::<i8>())])];
        let mut b: Vec<DynamicPolynomial<R>> = vec![DynamicPolynomial::new(vec![
            R::zero(),
            R::from(rng.random::<i8>()),
        ])];
        let mut c: Vec<DynamicPolynomial<R>> = vec![DynamicPolynomial::new(vec![
            R::zero(),
            R::from(rng.random::<i8>()),
        ])];

        for i in 1..1 << num_vars {
            let prev_a = a[i - 1].clone();
            let prev_b = b[i - 1].clone();
            let prev_c = c[i - 1].clone();

            a.push(poly_cfg.mul(&prev_a, &prev_b));
            b.push(poly_cfg.mul(&prev_b, &prev_c));
            c.push(poly_cfg.mul(&prev_a, &prev_c));
        }

        let arbitrary_poly = vec![
            a.into_iter()
                .map(|x| {
                    let deg = poly_cfg.degree(&x);
                    assert!(
                        deg < Some(32),
                        "degree bound exceeded: {}",
                        deg.expect("if the degree is large it's not None")
                    );
                    DensePolynomial::new_with_zero(x.coeffs, R::zero())
                })
                .collect(),
            b.into_iter()
                .map(|x| {
                    let deg = poly_cfg.degree(&x);
                    assert!(
                        deg < Some(32),
                        "degree bound exceeded: {}",
                        deg.expect("if the degree is large it's not None"),
                    );
                    DensePolynomial::new_with_zero(x.coeffs, R::zero())
                })
                .collect(),
            c.into_iter()
                .map(|x| {
                    let deg = poly_cfg.degree(&x);
                    assert!(
                        deg < Some(32),
                        "degree bound exceeded: {}",
                        deg.expect("if the degree is large it's not None"),
                    );
                    DensePolynomial::new_with_zero(x.coeffs, R::zero())
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
pub struct TestUairNoMultiplication<R, P>(PhantomData<(R, P)>);

impl<R, P> Uair for TestUairNoMultiplication<R, P>
where
    R: ConstSemiring + From<i32> + 'static,
    P: Semiring + 'static,
{
    type Ideal = DegreeOneIdeal<R>;
    type FqIdeal = ImpossibleIdeal;
    type Scalar = DensePolynomial<R, 32>;
    type Prime = P;

    fn signature() -> UairSignature<Self::Prime> {
        let total = TotalColumnLayout::new(0, 3, 0);
        UairSignature::new(total, PublicColumnLayout::default(), vec![], vec![])
    }

    fn constrain_general<C, B, FromR, MulByScalar, IFromR, IFqFromR>(
        b: &mut B,
        expr_cfg: &C,
        up: TraceRow<C::Element>,
        _down: TraceRow<C::Element>,
        _from_ref: FromR,
        _mbs: MulByScalar,
        ideal_from_ref: IFromR,
        _fq_ideal_from_ref: IFqFromR,
    ) where
        C: SemiringConfig,
        B: ConstraintBuilder<Expr = C::Element>,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
    {
        let up = up.arbitrary_poly;

        b.assert_in_ideal(
            expr_cfg.sub(&expr_cfg.add(&up[0], &up[1]), &up[2]),
            &ideal_from_ref(&DegreeOneIdeal::new(R::from(2))),
        );
    }
}

impl<R, P> GenerateRandomTrace<32> for TestUairNoMultiplication<R, P>
where
    R: ConstSemiring + From<i32> + 'static,
    P: Semiring + 'static,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_random_trace<G: Rng + ?Sized>(
        num_vars: usize,
        rng: &mut G,
    ) -> UairTrace<'static, R, R, 32, 32> {
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
pub struct TestUairScalarMultiplications<R, P>(PhantomData<(R, P)>);

impl<R, P> Uair for TestUairScalarMultiplications<R, P>
where
    R: ConstSemiring + From<i8> + 'static,
    P: Semiring + 'static,
{
    type Ideal = DegreeOneIdeal<R>;
    type FqIdeal = ImpossibleIdeal;
    type Scalar = DensePolynomial<R, 32>;
    type Prime = P;

    fn signature() -> UairSignature<Self::Prime> {
        let total = TotalColumnLayout::new(0, 3, 0);
        UairSignature::new(total, PublicColumnLayout::default(), vec![], vec![])
    }

    fn constrain_general<C, B, FromR, MulByScalar, IFromR, IFqFromR>(
        b: &mut B,
        expr_cfg: &C,
        up: TraceRow<C::Element>,
        _down: TraceRow<C::Element>,
        from_ref: FromR,
        mbs: MulByScalar,
        ideal_from_ref: IFromR,
        _fq_ideal_from_ref: IFqFromR,
    ) where
        C: SemiringConfig,
        B: ConstraintBuilder<Expr = C::Element>,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
        FromR: Fn(&DensePolynomial<R, 32>) -> C::Element,
        MulByScalar: Fn(&C::Element, &DensePolynomial<R, 32>) -> Option<C::Element>,
    {
        let up = up.arbitrary_poly;

        let scaled = mbs(
            &up[0],
            &DensePolynomial::new_with_zero([R::from(-1), R::from(0), R::from(1)], R::zero()),
        )
        .expect("arithmetic overflow");
        let constant = from_ref(&DensePolynomial::new_with_zero(
            [R::from(1), R::from(2), R::from(3), R::from(4)],
            R::zero(),
        ));
        // (up_0 * const[-1,0,1]) + up_1 - up_2 + const[1,2,3,4]
        b.assert_in_ideal(
            expr_cfg.add(
                &expr_cfg.sub(&expr_cfg.add(&scaled, &up[1]), &up[2]),
                &constant,
            ),
            &ideal_from_ref(&DegreeOneIdeal::new(R::from(2))),
        );
    }
}

#[derive(Clone, Debug)]
pub struct BinaryDecompositionUair<R, P>(PhantomData<(R, P)>);

impl<R, P> Uair for BinaryDecompositionUair<R, P>
where
    R: ConstSemiring + From<u32> + 'static,
    P: Semiring + 'static,
{
    type Ideal = DegreeOneIdeal<R>;
    type FqIdeal = ImpossibleIdeal;
    type Scalar = DensePolynomial<R, 32>;
    type Prime = P;

    fn signature() -> UairSignature<Self::Prime> {
        let total = TotalColumnLayout::new(1, 0, 1);
        UairSignature::new(total, PublicColumnLayout::default(), vec![], vec![])
    }

    fn constrain_general<C, B, FromR, MulByScalar, IFromR, IFqFromR>(
        b: &mut B,
        expr_cfg: &C,
        up: TraceRow<C::Element>,
        _down: TraceRow<C::Element>,
        _from_ref: FromR,
        _mbs: MulByScalar,
        ideal_from_ref: IFromR,
        _fq_ideal_from_ref: IFqFromR,
    ) where
        C: SemiringConfig,
        B: ConstraintBuilder<Expr = C::Element>,
        FromR: Fn(&Self::Scalar) -> C::Element,
        MulByScalar: Fn(&C::Element, &Self::Scalar) -> Option<C::Element>,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
    {
        let int_col = &up.int[0];
        let binary_poly_col = &up.binary_poly[0];

        b.assert_in_ideal(
            expr_cfg.sub(binary_poly_col, int_col),
            &ideal_from_ref(&DegreeOneIdeal::new(R::from(2))),
        );
    }
}

impl<R, P> GenerateRandomTrace<32> for BinaryDecompositionUair<R, P>
where
    R: ConstSemiring + From<u32> + 'static,
    P: Semiring + 'static,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_random_trace<G: Rng + ?Sized>(
        num_vars: usize,
        rng: &mut G,
    ) -> UairTrace<'static, R, R, 32, 32> {
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
pub struct BigLinearUair<R, P>(PhantomData<(R, P)>);

impl<R, P> Uair for BigLinearUair<R, P>
where
    R: ConstSemiring + From<u32> + 'static,
    P: Semiring + 'static,
{
    type Ideal = DegreeOneIdeal<R>;
    type FqIdeal = ImpossibleIdeal;
    type Scalar = DensePolynomial<R, 32>;
    type Prime = P;

    fn signature() -> UairSignature<Self::Prime> {
        let total = TotalColumnLayout::new(16, 0, 1);
        let shifts = (0..16).map(|i| ShiftSpec::new(i, 1)).collect();
        UairSignature::new(total, PublicColumnLayout::default(), shifts, vec![])
    }

    fn constrain_general<C, B, FromR, MulByScalar, IFromR, IFqFromR>(
        b: &mut B,
        expr_cfg: &C,
        up: TraceRow<C::Element>,
        down: TraceRow<C::Element>,
        _from_ref: FromR,
        _mbs: MulByScalar,
        ideal_from_ref: IFromR,
        _fq_ideal_from_ref: IFqFromR,
    ) where
        C: SemiringConfig,
        B: ConstraintBuilder<Expr = C::Element>,
        FromR: Fn(&Self::Scalar) -> C::Element,
        MulByScalar: Fn(&C::Element, &Self::Scalar) -> Option<C::Element>,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
    {
        let one_ideal = DegreeOneIdeal::new(R::from(1));
        let two_ideal = DegreeOneIdeal::new(R::from(2));

        let sum_of_binary_polys = up.binary_poly[1..]
            .iter()
            .fold(up.binary_poly[0].clone(), |acc, next| {
                expr_cfg.add(&acc, next)
            });

        // up.binary_poly[0] + up.binary_poly[1] + ... up.binary_poly[16]
        //      = up.int[0] mod (X - 1)
        b.assert_in_ideal(
            expr_cfg.sub(&sum_of_binary_polys, &up.int[0]),
            &ideal_from_ref(&one_ideal),
        );

        // down.binary_poly[0] = up.int[0] mod (X - 1)
        b.assert_in_ideal(
            expr_cfg.sub(&down.binary_poly[0], &up.int[0]),
            &ideal_from_ref(&two_ideal),
        );

        // down.binary_poly[i](1) = up.binary_poly[i](1), for all i=1,...,15
        // (preserves popcount across rows, but allows the bit pattern to change)
        up.binary_poly[1..]
            .iter()
            .zip(&down.binary_poly[1..])
            .for_each(|(up, down)| {
                b.assert_in_ideal(expr_cfg.sub(up, down), &ideal_from_ref(&one_ideal));
            });
    }
}

impl<R, P> GenerateRandomTrace<32> for BigLinearUair<R, P>
where
    R: ConstSemiring + From<u32> + 'static,
    P: Semiring + 'static,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_random_trace<G: Rng + ?Sized>(
        num_vars: usize,
        rng: &mut G,
    ) -> UairTrace<'static, R, R, 32, 32> {
        /// Generate a random binary polynomial with the given number of 1-bits.
        fn random_binary_poly_with_popcount(
            popcount: u32,
            rng: &mut (impl rand::Rng + ?Sized),
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
pub struct BigLinearUairWithPublicInput<R, P>(PhantomData<(R, P)>);

impl<R, P> Uair for BigLinearUairWithPublicInput<R, P>
where
    R: ConstSemiring + From<u32> + 'static,
    P: Semiring + 'static,
{
    type Ideal = <BigLinearUair<R, P> as Uair>::Ideal;
    type FqIdeal = <BigLinearUair<R, P> as Uair>::FqIdeal;
    type Scalar = <BigLinearUair<R, P> as Uair>::Scalar;
    type Prime = <BigLinearUair<R, P> as Uair>::Prime;

    fn signature() -> UairSignature<Self::Prime> {
        let total = TotalColumnLayout::new(16, 0, 1);
        let public = PublicColumnLayout::new(4, 0, 0);
        let shifts = (0..16).map(|i| ShiftSpec::new(i, 1)).collect();
        UairSignature::new(total, public, shifts, vec![])
    }

    fn constrain_general<C, B, FromR, MulByScalar, IFromR, IFqFromR>(
        b: &mut B,
        expr_cfg: &C,
        up: TraceRow<C::Element>,
        down: TraceRow<C::Element>,
        from_ref: FromR,
        mbs: MulByScalar,
        ideal_from_ref: IFromR,
        fq_ideal_from_ref: IFqFromR,
    ) where
        C: SemiringConfig,
        B: ConstraintBuilder<Expr = C::Element>,
        FromR: Fn(&Self::Scalar) -> C::Element,
        MulByScalar: Fn(&C::Element, &Self::Scalar) -> Option<C::Element>,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
        IFqFromR: Fn(&Self::FqIdeal) -> B::FqIdeal,
    {
        BigLinearUair::<R, P>::constrain_general(
            b,
            expr_cfg,
            up,
            down,
            from_ref,
            mbs,
            ideal_from_ref,
            fq_ideal_from_ref,
        )
    }
}

impl<R, P> GenerateRandomTrace<32> for BigLinearUairWithPublicInput<R, P>
where
    R: ConstSemiring + From<u32> + 'static,
    P: Semiring + 'static,
{
    type PolyCoeff = <BigLinearUair<R, P> as GenerateRandomTrace<32>>::PolyCoeff;
    type Int = <BigLinearUair<R, P> as GenerateRandomTrace<32>>::Int;

    fn generate_random_trace<G: Rng + ?Sized>(
        num_vars: usize,
        rng: &mut G,
    ) -> UairTrace<'static, Self::PolyCoeff, Self::Int, 32, 32> {
        BigLinearUair::<R, P>::generate_random_trace(num_vars, rng)
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
pub struct ShaProxy<R, P>(PhantomData<(R, P)>);

impl<R, P> Uair for ShaProxy<R, P>
where
    R: ConstSemiring + From<u32> + 'static,
    P: Semiring + 'static,
{
    type Ideal = DegreeOneIdeal<R>;
    type FqIdeal = ImpossibleIdeal;
    type Scalar = DensePolynomial<R, 32>;
    type Prime = P;

    fn signature() -> UairSignature<Self::Prime> {
        // 14 binary_poly cols, 0 arbitrary_poly cols, 4 int cols.
        let total = TotalColumnLayout::new(14, 0, 4);
        // c_1 (bp[0]) is shifted by 1 (used by C1 as bp[0][t+1]); c_5 (bp[4])
        // is shifted by 4 (used by C2 as bp[4][t+4]).
        let shifts = vec![ShiftSpec::new(0, 1), ShiftSpec::new(4, 4)];
        UairSignature::new(total, PublicColumnLayout::default(), shifts, vec![])
    }

    fn constrain_general<C, B, FromR, MulByScalar, IFromR, IFqFromR>(
        b: &mut B,
        expr_cfg: &C,
        up: TraceRow<C::Element>,
        down: TraceRow<C::Element>,
        _from_ref: FromR,
        mbs: MulByScalar,
        ideal_from_ref: IFromR,
        _fq_ideal_from_ref: IFqFromR,
    ) where
        C: SemiringConfig,
        B: ConstraintBuilder<Expr = C::Element>,
        FromR: Fn(&Self::Scalar) -> C::Element,
        MulByScalar: Fn(&C::Element, &Self::Scalar) -> Option<C::Element>,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
    {
        let one_ideal = ideal_from_ref(&DegreeOneIdeal::new(R::ONE));
        let two_ideal = ideal_from_ref(&DegreeOneIdeal::new(R::from(2)));
        // The polynomial X = 0 + 1*X, used to express `X * c_k` via `mbs`.
        let x_scalar = DensePolynomial::<R, 32>::new_with_zero([R::ZERO, R::from(1)], R::zero());

        // `down.binary_poly` is indexed by ShiftSpec position, not source col.
        // Our shifts vec is [ShiftSpec::new(0, 1), ShiftSpec::new(4, 1)], so
        // down.binary_poly[0] = bp[0][t+1], down.binary_poly[1] = bp[4][t+1].

        // (C1) dbp[0] - bp[1] - bp[2] - bp[3] - int[0] - int[1] - int[2] ∈ (X-2)
        let c1 = [
            &up.binary_poly[1],
            &up.binary_poly[2],
            &up.binary_poly[3],
            &up.int[0],
            &up.int[1],
            &up.int[2],
        ]
        .into_iter()
        .fold(down.binary_poly[0].clone(), |acc, term| {
            expr_cfg.sub(&acc, term)
        });
        b.assert_in_ideal(c1, &two_ideal);

        // (C2) dbp[4] - bp[5] - bp[6] - bp[7] - int[1] - int[2] - int[3] ∈ (X-2)
        let c2 = [
            &up.binary_poly[5],
            &up.binary_poly[6],
            &up.binary_poly[7],
            &up.int[1],
            &up.int[2],
            &up.int[3],
        ]
        .into_iter()
        .fold(down.binary_poly[1].clone(), |acc, term| {
            expr_cfg.sub(&acc, term)
        });
        b.assert_in_ideal(c2, &two_ideal);

        // (C3) bp[8] - int[0] ∈ (X-2)
        b.assert_in_ideal(expr_cfg.sub(&up.binary_poly[8], &up.int[0]), &two_ideal);

        // (C4) bp[9] - int[1] ∈ (X-2)
        b.assert_in_ideal(expr_cfg.sub(&up.binary_poly[9], &up.int[1]), &two_ideal);

        // (C5) bp[10] - X * bp[11] ∈ (X-1)
        b.assert_in_ideal(
            expr_cfg.sub(
                &up.binary_poly[10],
                &mbs(&up.binary_poly[11], &x_scalar).expect("mul-by-X overflow"),
            ),
            &one_ideal,
        );

        // (C6) bp[12] - X * bp[13] ∈ (X-1)
        b.assert_in_ideal(
            expr_cfg.sub(
                &up.binary_poly[12],
                &mbs(&up.binary_poly[13], &x_scalar).expect("mul-by-X overflow"),
            ),
            &one_ideal,
        );
    }
}

impl<R, P> GenerateRandomTrace<32> for ShaProxy<R, P>
where
    R: ConstSemiring + From<u32> + 'static,
    P: Semiring + 'static,
{
    type PolyCoeff = R;
    type Int = R;

    #[allow(clippy::needless_range_loop)]
    fn generate_random_trace<G: Rng + ?Sized>(
        num_vars: usize,
        rng: &mut G,
    ) -> UairTrace<'static, R, R, 32, 32> {
        /// Generate a random binary polynomial with the given number of 1-bits.
        fn random_binary_poly_with_popcount(
            popcount: u32,
            rng: &mut (impl rand::Rng + ?Sized),
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
pub struct TestUairMixedShifts<R, P>(PhantomData<(R, P)>);

impl<R, P> Uair for TestUairMixedShifts<R, P>
where
    R: Semiring + 'static,
    P: Semiring + 'static,
{
    type Ideal = ImpossibleIdeal;
    type FqIdeal = ImpossibleIdeal;
    type Scalar = DensePolynomial<R, 32>;
    type Prime = P;

    fn signature() -> UairSignature<Self::Prime> {
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
    fn constrain_general<C, B, FromR, MulByScalar, IFromR, IFqFromR>(
        builder: &mut B,
        expr_cfg: &C,
        up: TraceRow<C::Element>,
        down: TraceRow<C::Element>,
        _from_ref: FromR,
        _mbs: MulByScalar,
        _ideal_from_ref: IFromR,
        _fq_ideal_from_ref: IFqFromR,
    ) where
        C: SemiringConfig,
        B: ConstraintBuilder<Expr = C::Element>,
    {
        let up = up.arbitrary_poly;
        let down = down.arbitrary_poly;

        builder.assert_zero(expr_cfg.sub(&expr_cfg.sub(&down[0], &up[0]), &up[1]));
        builder.assert_zero(expr_cfg.sub(&up[2], &down[1]));
    }
}

impl<R, P> GenerateRandomTrace<32> for TestUairMixedShifts<R, P>
where
    R: Semiring + From<i8> + 'static,
    P: Semiring + 'static,
    StandardUniform: Distribution<R>,
{
    type PolyCoeff = R;
    type Int = R;

    // Witness: random b, derive a from a[i+1] = a[i] + b[i], set c[i] = b[i+2].
    fn generate_random_trace<G: Rng + ?Sized>(
        num_vars: usize,
        rng: &mut G,
    ) -> UairTrace<'static, R, R, 32, 32> {
        let n = 1 << num_vars;

        // Random b column (degree-0 polynomials to stay under degree 32)
        let ring_cfg = FixedConfig::<R>::default();
        let poly_cfg = ring_cfg.dyn_poly_cfg();

        let b_col: Vec<DynamicPolynomial<R>> = (0..n)
            .map(|_| DynamicPolynomial::new(vec![R::from(rng.random::<i8>())]))
            .collect();

        // a[0] random, a[i+1] = a[i] + b[i]
        let mut a_col: Vec<DynamicPolynomial<R>> =
            vec![DynamicPolynomial::new(vec![R::from(rng.random::<i8>())])];
        for i in 0..n - 1 {
            a_col.push(poly_cfg.add(&a_col[i], &b_col[i]));
        }

        // c[i] = b[i+2], zero-padded for last 2 entries
        let mut c_col: Vec<DynamicPolynomial<R>> = Vec::with_capacity(n);
        for i in 0..n {
            if i + 2 < n {
                c_col.push(b_col[i + 2].clone());
            } else {
                c_col.push(DynamicPolynomial::ZERO);
            }
        }

        let to_mle =
            |col: Vec<DynamicPolynomial<R>>| -> DenseMultilinearExtension<DensePolynomial<R, 32>> {
                col.into_iter()
                    .map(|x| DensePolynomial::new_with_zero(x.coeffs, R::zero()))
                    .collect()
            };

        UairTrace {
            arbitrary_poly: vec![to_mle(a_col), to_mle(b_col), to_mle(c_col)].into(),
            ..Default::default()
        }
    }
}

/// Mixed-splice UAIR for bit-op virtual columns.
///
/// It populates three slots of the canonical down-row ordering at once:
/// shifted binary, bit-op binary, and shifted arbitrary. This catches
/// materialization code that appends bit-op virtuals at the tail instead of
/// inserting them into the binary down slice.
#[derive(Clone, Debug)]
pub struct TestUairBitOpsMixedSplice<R, P>(PhantomData<(R, P)>);

impl<R, P> Uair for TestUairBitOpsMixedSplice<R, P>
where
    R: ConstSemiring + 'static,
    P: Semiring + 'static,
{
    type Ideal = DegreeOneIdeal<R>;
    type FqIdeal = ImpossibleIdeal;
    type Scalar = DensePolynomial<R, 32>;
    type Prime = P;

    fn signature() -> UairSignature<Self::Prime> {
        let total = TotalColumnLayout::new(3, 2, 0);
        let shifts = vec![ShiftSpec::new(0, 1), ShiftSpec::new(3, 1)];
        let bit_op_specs = vec![BitOpSpec::new(0, BitOp::ShR(3))];
        let sig = UairSignature::new(total, PublicColumnLayout::default(), shifts, vec![])
            .with_bit_op_specs(bit_op_specs);
        debug_assert_eq!(sig.down_cols().num_binary_poly_cols(), 2);
        debug_assert_eq!(sig.down_cols().num_arbitrary_poly_cols(), 1);
        debug_assert_eq!(sig.down_cols().num_int_cols(), 0);
        sig
    }

    fn constrain_general<C, B, FromR, MulByScalar, IFromR, IFqFromR>(
        b: &mut B,
        expr_cfg: &C,
        up: TraceRow<C::Element>,
        down: TraceRow<C::Element>,
        _from_ref: FromR,
        _mbs: MulByScalar,
        ideal_from_ref: IFromR,
        _fq_ideal_from_ref: IFqFromR,
    ) where
        C: SemiringConfig,
        B: ConstraintBuilder<Expr = C::Element>,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
    {
        let one_ideal = ideal_from_ref(&DegreeOneIdeal::new(R::ONE));
        b.assert_in_ideal(
            expr_cfg.sub(&down.binary_poly[0], &up.binary_poly[2]),
            &one_ideal,
        );
        b.assert_in_ideal(
            expr_cfg.sub(&down.binary_poly[1], &up.binary_poly[1]),
            &one_ideal,
        );
        b.assert_in_ideal(
            expr_cfg.sub(&down.arbitrary_poly[0], &up.arbitrary_poly[1]),
            &one_ideal,
        );
    }
}

impl<R, P> GenerateRandomTrace<32> for TestUairBitOpsMixedSplice<R, P>
where
    R: ConstSemiring + From<i8> + 'static,
    P: Semiring + 'static,
    StandardUniform: Distribution<R>,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_random_trace<G: Rng + ?Sized>(
        num_vars: usize,
        rng: &mut G,
    ) -> UairTrace<'static, R, R, 32, 32> {
        let n = 1usize << num_vars;

        let w_u32: Vec<u32> = (0..n).map(|_| rng.next_u32()).collect();
        let w_col: DenseMultilinearExtension<BinaryPoly<32>> =
            w_u32.iter().map(|w| BinaryPoly::from(*w)).collect();
        let s_shr_col: DenseMultilinearExtension<BinaryPoly<32>> =
            w_u32.iter().map(|w| BinaryPoly::from(w >> 3)).collect();
        let t_col: DenseMultilinearExtension<BinaryPoly<32>> = (0..n)
            .map(|i| {
                if i + 1 < n {
                    BinaryPoly::from(w_u32[i + 1])
                } else {
                    BinaryPoly::from(0u32)
                }
            })
            .collect();

        let a_cells: Vec<DensePolynomial<R, 32>> = (0..n)
            .map(|_| DensePolynomial::new_with_zero([R::from(rng.random::<i8>())], R::zero()))
            .collect();
        let a_next_cells: Vec<DensePolynomial<R, 32>> = (0..n)
            .map(|i| {
                if i + 1 < n {
                    a_cells[i + 1].clone()
                } else {
                    DensePolynomial::<R, 32>::zero()
                }
            })
            .collect();

        UairTrace {
            binary_poly: vec![w_col, s_shr_col, t_col].into(),
            arbitrary_poly: vec![
                a_cells.into_iter().collect(),
                a_next_cells.into_iter().collect(),
            ]
            .into(),
            int: vec![].into(),
        }
    }
}

/// UAIR combining bit-op virtual columns with both Q[X] and F_q[X]
/// constraint families.
///
/// The single bit-op virtual column is `ShR(w, 3)`. Both families constrain it
/// to match the committed expected column `s`, so the full protocol must carry
/// the virtual bit-op column through each family, not just through ideal check.
#[derive(Clone, Debug)]
pub struct TestUairBitOpsFqFamily<R, P>(PhantomData<(R, P)>);

impl<R, P> Uair for TestUairBitOpsFqFamily<R, P>
where
    R: ConstSemiring + 'static,
    P: Semiring + From<u64> + 'static,
{
    type Ideal = DegreeOneIdeal<R>;
    type FqIdeal = DegreeOneIdeal<R>;
    type Scalar = DensePolynomial<R, 32>;
    type Prime = P;

    fn signature() -> UairSignature<Self::Prime> {
        let total = TotalColumnLayout::new(2, 0, 0);
        UairSignature::new(total, PublicColumnLayout::default(), vec![], vec![])
            .with_bit_op_specs(vec![BitOpSpec::new(0, BitOp::ShR(3))])
            .with_primes(vec![P::from(MERSENNE_61_PRIME)])
    }

    fn constrain_general<C, B, FromR, MulByScalar, IFromR, IFqFromR>(
        b: &mut B,
        expr_cfg: &C,
        up: TraceRow<C::Element>,
        down: TraceRow<C::Element>,
        _from_ref: FromR,
        _mbs: MulByScalar,
        ideal_from_ref: IFromR,
        fq_ideal_from_ref: IFqFromR,
    ) where
        C: SemiringConfig,
        B: ConstraintBuilder<Expr = C::Element>,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
        IFqFromR: Fn(&Self::FqIdeal) -> B::FqIdeal,
    {
        let q_ideal = ideal_from_ref(&DegreeOneIdeal::new(R::ONE));
        let fq_ideal = fq_ideal_from_ref(&DegreeOneIdeal::new(R::ONE));
        let bit_op_matches_expected = expr_cfg.sub(&down.binary_poly[0], &up.binary_poly[1]);

        b.assert_in_ideal(bit_op_matches_expected.clone(), &q_ideal);
        b.assert_in_fq_ideal(0, bit_op_matches_expected, &fq_ideal);
    }
}

impl<R, P> GenerateRandomTrace<32> for TestUairBitOpsFqFamily<R, P>
where
    R: ConstSemiring + 'static,
    P: Semiring + From<u64> + 'static,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_random_trace<G: Rng + ?Sized>(
        num_vars: usize,
        rng: &mut G,
    ) -> UairTrace<'static, R, R, 32, 32> {
        let n = 1usize << num_vars;
        let w_u32: Vec<u32> = (0..n).map(|_| rng.next_u32()).collect();
        let w_col: DenseMultilinearExtension<BinaryPoly<32>> =
            w_u32.iter().map(|w| BinaryPoly::from(*w)).collect();
        let s_shr_col: DenseMultilinearExtension<BinaryPoly<32>> =
            w_u32.iter().map(|w| BinaryPoly::from(w >> 3)).collect();

        UairTrace {
            binary_poly: vec![w_col, s_shr_col].into(),
            ..Default::default()
        }
    }
}

/// A UAIR exercising the Flavor-1 $F_{q}[X]$-constraint surface
/// for **multiple large primes**.
///
/// Single arbitrary-poly witness column `a`. Two $F_{q_i}[X]$-
/// constraints, one per declared prime $q_i$:
///
/// $$
///   \phi_{q_0}(a) \in (X - 0) \quad \text{in } F_{q_0}[X],
///   \qquad
///   \phi_{q_1}(a) \in (X - 0) \quad \text{in } F_{q_1}[X],
/// $$
///
/// i.e. the constant term of `a` is zero modulo each of $q_0$ and $q_1$.
/// The trace generator builds `a` with its $X^0$ coefficient set to
/// integer zero, so both constraints hold simultaneously regardless of the
/// chosen primes.
///
/// Used to exercise the per-prime $F_{q_i}[X]$ ideal-check family
/// of the Zinc$+$ protocol end-to-end with multiple primes, including the
/// lockstep multi-family sumcheck/MP-eval driver.
#[derive(Clone, Debug)]
pub struct TestUairFqLargePrime<R, P>(PhantomData<(R, P)>);

/// M61 prime (2^61−1); used as $q_0$ in [`TestUairFqLargePrime`].
pub const MERSENNE_61_PRIME: u64 = (1 << 61) - 1;

/// Goldilocks prime (2^64 - 2^32 + 1); used as $q_1$ in
/// [`TestUairFqLargePrime`].
#[allow(clippy::cast_possible_truncation)]
pub const GOLDILOCKS_PRIME: u64 = ((1_u128 << 64) - (1 << 32) + 1) as u64;

impl<R, P> Uair for TestUairFqLargePrime<R, P>
where
    R: ConstSemiring + From<i32> + 'static,
    P: Semiring + From<u64> + 'static,
{
    type Ideal = ImpossibleIdeal;
    type FqIdeal = DegreeOneIdeal<R>;
    type Scalar = DensePolynomial<R, 32>;
    type Prime = P;

    fn signature() -> UairSignature<Self::Prime> {
        // 1 arbitrary-poly witness column `a`, no shifts, no lookups.
        let total = TotalColumnLayout::new(0, 1, 0);
        UairSignature::new(total, PublicColumnLayout::default(), vec![], vec![])
            .with_primes(vec![P::from(MERSENNE_61_PRIME), P::from(GOLDILOCKS_PRIME)])
    }

    fn constrain_general<C, B, FromR, MulByScalar, IFromR, IFqFromR>(
        b: &mut B,
        _expr_cfg: &C,
        up: TraceRow<C::Element>,
        _down: TraceRow<C::Element>,
        _from_ref: FromR,
        _mbs: MulByScalar,
        _ideal_from_ref: IFromR,
        fq_ideal_from_ref: IFqFromR,
    ) where
        C: SemiringConfig,
        B: ConstraintBuilder<Expr = C::Element>,
        FromR: Fn(&Self::Scalar) -> C::Element,
        MulByScalar: Fn(&C::Element, &Self::Scalar) -> Option<C::Element>,
        IFqFromR: Fn(&Self::FqIdeal) -> B::FqIdeal,
    {
        // One constraint per declared prime: `phi_{q_i}(a) \in (X - 0)`.
        let ideal = fq_ideal_from_ref(&DegreeOneIdeal::<R>::new(R::ZERO));
        b.assert_in_fq_ideal(
            /* prime_idx = */ 0,
            up.arbitrary_poly[0].clone(),
            &ideal,
        );
        b.assert_in_fq_ideal(
            /* prime_idx = */ 1,
            up.arbitrary_poly[0].clone(),
            &ideal,
        );
    }
}

impl<R, P> GenerateRandomTrace<32> for TestUairFqLargePrime<R, P>
where
    R: ConstSemiring + From<i32> + 'static,
    P: Semiring + From<u64> + 'static,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_random_trace<G: Rng + ?Sized>(
        num_vars: usize,
        rng: &mut G,
    ) -> UairTrace<'static, R, R, 32, 32> {
        // Build the witness column: random polynomials whose constant term
        // is forced to zero (so `phi_q(a)(0) = 0` regardless of `q`).
        let a: DenseMultilinearExtension<DensePolynomial<R, 32>> =
            DenseMultilinearExtension::rand(num_vars, rng)
                .into_iter()
                .map(|x: u32| {
                    let poly = DensePolynomial::from_ref(&DensePolynomial::<Boolean, _>::from(
                        BinaryPoly::<32>::from(x),
                    ));
                    // Zero out the X^0 coefficient.
                    let mut coeffs = [R::ZERO; 32];
                    coeffs[1..].clone_from_slice(&poly.coeffs[1..]);
                    DensePolynomial::new(coeffs)
                })
                .collect();

        UairTrace {
            arbitrary_poly: vec![a].into(),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_primitives::crypto_bigint_int::Int;
    use num_traits::ConstZero;
    use zinc_uair::{
        collect_scalars::collect_scalars,
        constraint_counter::count_constraints,
        degree_counter::{count_constraint_degrees_flattened, count_max_degree},
    };

    const LIMBS: usize = 4;

    #[test]
    fn test_constraint_degrees() {
        fn assert_uair_shape<U: Uair>(expected_degrees: &[usize]) {
            assert_eq!(count_constraints::<U>().total(), expected_degrees.len());
            assert_eq!(count_constraint_degrees_flattened::<U>(), expected_degrees);
            assert_eq!(
                count_max_degree::<U>(),
                *expected_degrees.iter().max().unwrap()
            );
        }

        assert_uair_shape::<TestUairSimpleMultiplication<Int<LIMBS>, u64>>(&[2, 2, 2]);
        assert_uair_shape::<TestUairNoMultiplication<Int<LIMBS>, u64>>(&[1]);
        assert_uair_shape::<TestUairScalarMultiplications<Int<LIMBS>, u64>>(&[1]);
        assert_uair_shape::<BinaryDecompositionUair<u32, u64>>(&[1]);
        assert_uair_shape::<BigLinearUair<u32, u64>>(&[1; 17]);
        assert_uair_shape::<TestUairMixedShifts<Int<LIMBS>, u64>>(&[1, 1]);
        assert_uair_shape::<TestUairBitOpsMixedSplice<Int<LIMBS>, u64>>(&[1, 1, 1]);
        assert_uair_shape::<TestUairBitOpsFqFamily<Int<LIMBS>, u64>>(&[1, 1]);
        // TestUairFqLargePrime: two F_{q_i}[X] linear constraints (one per
        // declared prime).
        assert_uair_shape::<TestUairFqLargePrime<Int<LIMBS>, u64>>(&[1, 1]);
    }

    #[test]
    fn test_air_scalar_multiplications_correct_collect_scalars() {
        assert_eq!(
            collect_scalars::<TestUairScalarMultiplications<Int<LIMBS>, u64>>(),
            (vec![
                DensePolynomial::new_with_zero(
                    [Int::from_i8(-1), Int::from_i8(0), Int::from_i8(1)],
                    Int::ZERO
                ),
                DensePolynomial::new_with_zero(
                    [
                        Int::from_i8(1),
                        Int::from_i8(2),
                        Int::from_i8(3),
                        Int::from_i8(4),
                    ],
                    Int::ZERO
                )
            ]
            .into_iter()
            .collect())
        );
    }
}
