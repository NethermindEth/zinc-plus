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
    ConstraintBuilder, PublicColumnLayout, ShiftSpec, TotalColumnLayout, TraceRow, Uair,
    UairSignature, UairTrace,
    ideal::{
        ImpossibleIdeal, degree_one::DegreeOneIdeal, mixed::MixedDegreeOneOrXnMinusOne,
        xn_minus_one::XnMinusOneIdeal,
    },
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
        UairSignature::new(total, PublicColumnLayout::default(), shifts)
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
        UairSignature::new(total, PublicColumnLayout::default(), vec![])
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
        UairSignature::new(total, PublicColumnLayout::default(), vec![])
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
        UairSignature::new(total, PublicColumnLayout::default(), vec![])
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
        UairSignature::new(total, PublicColumnLayout::default(), shifts)
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
        UairSignature::new(total, public, shifts)
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

/// A "wide and shifty" linear UAIR with 14 binary-poly columns and 4 int
/// columns. Designed to stress the protocol's shift-handling, `mbs` machinery,
/// and mixed-ideal support: it uses both the `(X-2)` ideal (for linear
/// recurrences) and the `(X^32 - 1)` ideal (for cyclic-shift relations on
/// `BinaryPoly<32>`).
///
/// The user-facing constraints are written in a `t`-relative frame with
/// **negative shifts** (`bp[k][t-N]`). Since `ShiftSpec` only supports
/// forward shifts, the constraint set is **globally re-indexed** by
/// substituting `s = t - 16` so that every reference becomes a forward
/// shift in `s`. The constraint *shape* (which columns interact, with what
/// coefficients) is preserved exactly. Maximum forward shift is 17.
///
/// Constraints in the user `t`-frame:
///
/// ```text
/// C1: bp[0][t+1] + bp[1][t-3] - bp[2][t]   - bp[3][t]  - bp[4][t]
///                - bp[5][t]   - bp[6][t]   - int[0][t] - int[1][t]    ∈ (X-2)
/// C2: bp[1][t+1] - bp[0][t-3] - bp[1][t-3] - bp[2][t]  - bp[3][t]
///                - bp[7][t]   - int[0][t]  - int[2][t]                 ∈ (X-2)
/// C3: bp[4][t]   - bp[4][t-16]- bp[4][t-7] - bp[8][t]  - bp[9][t]
///                - int[3][t]                                            ∈ (X-2)
/// C4: bp[0][t] * (X^25 + X^14) - bp[5][t]                              ∈ (X^32 - 1)
/// C5: bp[1][t] * (X^25 + X^13) - bp[3][t]                              ∈ (X^32 - 1)
/// C6: bp[10][t] - int[0][t]                                            ∈ (X-2)
/// C7: bp[11][t] - int[1][t]                                            ∈ (X-2)
/// ```
///
/// Note the **plus sign on `bp[1][t-3]` in C1** (everything else in C1's RHS
/// is subtracted). This single sign flip is necessary to break the boundary
/// cascade that would otherwise force the entire trace to zero (see the
/// `generate_random_trace` doc comment for details).
///
/// **Slacks (free variables that absorb constraint residue):**
/// - `int[2]` only appears in C2 → free slack for C2's recurrence.
/// - `int[3]` only appears in C3 → free slack for C3's recurrence.
/// - `bp[6]` only appears in C1 → free `BinaryPoly` slack for C1's
///   recurrence (constrained to non-negative `u32` values, made viable
///   by the bp[1] sign flip + bp[0]/bp[1] pinned to large values).
///
/// `bp[12]` and `bp[13]` are unused by any constraint and exist purely as
/// random fillers.
pub struct SHAProxy<R>(PhantomData<R>);

impl<R> Uair for SHAProxy<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type Ideal = MixedDegreeOneOrXnMinusOne<R, 32>;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        // 14 binary_poly cols, 0 arbitrary_poly cols, 4 int cols.
        let total = TotalColumnLayout::new(14, 0, 4);
        // 17 binary-poly shifts + 4 int shifts = 21 shifts total.
        // Multiple shifts per source_col are supported and `UairSignature::new`
        // sorts them stably by source_col. We write them in source_col order so
        // the table-vs-code mapping is obvious.
        let shifts = vec![
            // bp[0]: shifts 13, 16, 17  -> down.binary_poly[0..3]
            ShiftSpec::new(0, 13),
            ShiftSpec::new(0, 16),
            ShiftSpec::new(0, 17),
            // bp[1]: shifts 13, 16, 17  -> down.binary_poly[3..6]
            ShiftSpec::new(1, 13),
            ShiftSpec::new(1, 16),
            ShiftSpec::new(1, 17),
            // bp[2], bp[3]: shift 16    -> down.binary_poly[6..8]
            ShiftSpec::new(2, 16),
            ShiftSpec::new(3, 16),
            // bp[4]: shifts 9, 16       -> down.binary_poly[8..10]
            ShiftSpec::new(4, 9),
            ShiftSpec::new(4, 16),
            // bp[5..=11]: shift 16      -> down.binary_poly[10..17]
            ShiftSpec::new(5, 16),
            ShiftSpec::new(6, 16),
            ShiftSpec::new(7, 16),
            ShiftSpec::new(8, 16),
            ShiftSpec::new(9, 16),
            ShiftSpec::new(10, 16),
            ShiftSpec::new(11, 16),
            // int[0..=3]: shift 16      -> down.int[0..4]
            // (flat indices 14..=17 since we have 14 bp + 0 ap before int)
            ShiftSpec::new(14, 16),
            ShiftSpec::new(15, 16),
            ShiftSpec::new(16, 16),
            ShiftSpec::new(17, 16),
        ];
        UairSignature::new(total, PublicColumnLayout::default(), shifts)
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
        let two_ideal = MixedDegreeOneOrXnMinusOne::DegreeOne(DegreeOneIdeal::new(R::from(2)));
        let xn_ideal: MixedDegreeOneOrXnMinusOne<R, 32> =
            MixedDegreeOneOrXnMinusOne::XnMinusOne(XnMinusOneIdeal::<32>::new());

        // Scalar polynomials for C4 and C5: X^25 + X^14 and X^25 + X^13.
        // The DensePolynomial<R, 32> coefficient at index k is the coefficient
        // of X^k. The product `bp[k] * scalar` produces a degree-≤56 expression
        // in the underlying `DynamicPolynomialF`, which is fine because the
        // expression type grows dynamically.
        let mut x25_x14 = [R::ZERO; 32];
        x25_x14[14] = R::from(1);
        x25_x14[25] = R::from(1);
        let x25_plus_x14 = DensePolynomial::<R, 32>::new(x25_x14);

        let mut x25_x13 = [R::ZERO; 32];
        x25_x13[13] = R::from(1);
        x25_x13[25] = R::from(1);
        let x25_plus_x13 = DensePolynomial::<R, 32>::new(x25_x13);

        // Convenience aliases for shifted columns. Indices match the shifts
        // vec above (after `UairSignature::new`'s stable sort by source_col).
        let bp0_s13 = &down.binary_poly[0];
        let bp0_s16 = &down.binary_poly[1];
        let bp0_s17 = &down.binary_poly[2];
        let bp1_s13 = &down.binary_poly[3];
        let bp1_s16 = &down.binary_poly[4];
        let bp1_s17 = &down.binary_poly[5];
        let bp2_s16 = &down.binary_poly[6];
        let bp3_s16 = &down.binary_poly[7];
        let bp4_s9 = &down.binary_poly[8];
        let bp4_s16 = &down.binary_poly[9];
        let bp5_s16 = &down.binary_poly[10];
        let bp6_s16 = &down.binary_poly[11];
        let bp7_s16 = &down.binary_poly[12];
        let bp8_s16 = &down.binary_poly[13];
        let bp9_s16 = &down.binary_poly[14];
        let bp10_s16 = &down.binary_poly[15];
        let bp11_s16 = &down.binary_poly[16];
        let int0_s16 = &down.int[0];
        let int1_s16 = &down.int[1];
        let int2_s16 = &down.int[2];
        let int3_s16 = &down.int[3];

        // (C1) bp[0][s+17] + bp[1][s+13] - bp[2..=6][s+16] - int[0..=1][s+16] ∈ (X-2)
        // The PLUS sign on `bp1_s13` (which is `bp[1][t-3]` in the t-frame)
        // is intentional: it lets `bp[6]` serve as a non-negative slack at
        // the boundary row s = len-17, where the LHS `bp[0][s+17]` reads as
        // OOB zero. Without the flip, the slack would have to be negative
        // there, which a `BinaryPoly<32>` cannot represent.
        b.assert_in_ideal(
            bp0_s17.clone() + bp1_s13
                - bp2_s16
                - bp3_s16
                - bp4_s16
                - bp5_s16
                - bp6_s16
                - int0_s16
                - int1_s16,
            &ideal_from_ref(&two_ideal),
        );

        // (C2) bp[1][s+17] - bp[0][s+13] - bp[1][s+13] - bp[2,3,7][s+16]
        //      - int[0][s+16] - int[2][s+16] ∈ (X-2)
        b.assert_in_ideal(
            bp1_s17.clone()
                - bp0_s13
                - bp1_s13
                - bp2_s16
                - bp3_s16
                - bp7_s16
                - int0_s16
                - int2_s16,
            &ideal_from_ref(&two_ideal),
        );

        // (C3) bp[4][s+16] - bp[4][s] - bp[4][s+9] - bp[8][s+16] - bp[9][s+16]
        //      - int[3][s+16] ∈ (X-2)
        // The "bp[4][s]" term is the only reference at the unshifted current
        // row, so it comes from `up.binary_poly[4]` instead of `down`.
        b.assert_in_ideal(
            bp4_s16.clone()
                - &up.binary_poly[4]
                - bp4_s9
                - bp8_s16
                - bp9_s16
                - int3_s16,
            &ideal_from_ref(&two_ideal),
        );

        // (C4) bp[0][s+16] * (X^25 + X^14) - bp[5][s+16] ∈ (X^32 - 1)
        // In the ring R[X]/(X^32 - 1), multiplying by X is a cyclic shift.
        // So this constraint forces `bp[5]` to be the sum of `bp[0]` cyclically
        // shifted by 25 positions and by 14 positions. For binary polys, the
        // sum is well-defined (avoids carry) iff `bp[0]` has no two 1-bits at
        // positions 11 apart (mod 32), where 11 = 25 - 14.
        b.assert_in_ideal(
            mbs(bp0_s16, &x25_plus_x14).expect("mul-by-scalar overflow") - bp5_s16,
            &ideal_from_ref(&xn_ideal),
        );

        // (C5) bp[1][s+16] * (X^25 + X^13) - bp[3][s+16] ∈ (X^32 - 1)
        // Same idea as C4 but with the bit-gap constraint at 12 = 25 - 13.
        b.assert_in_ideal(
            mbs(bp1_s16, &x25_plus_x13).expect("mul-by-scalar overflow") - bp3_s16,
            &ideal_from_ref(&xn_ideal),
        );

        // (C6) bp[10][s+16] - int[0][s+16] ∈ (X-2)
        b.assert_in_ideal(bp10_s16.clone() - int0_s16, &ideal_from_ref(&two_ideal));

        // (C7) bp[11][s+16] - int[1][s+16] ∈ (X-2)
        b.assert_in_ideal(bp11_s16.clone() - int1_s16, &ideal_from_ref(&two_ideal));
    }
}

impl<R> GenerateRandomTrace<32> for SHAProxy<R>
where
    R: ConstSemiring + From<u32> + From<i64> + 'static,
{
    type PolyCoeff = R;
    type Int = R;

    /// Constructs a non-trivial witness using the **slack columns** in the
    /// constraint set:
    ///
    /// - **`int[2]` is a free slack for C2** (it only appears in C2). At
    ///   each row, we set `int[2]` to whatever signed `i64` value makes C2
    ///   hold.
    /// - **`int[3]` is a free slack for C3** (same idea).
    /// - **`bp[6]` is a `BinaryPoly` slack for C1** (it only appears in C1).
    ///   Its `eval(2)` must land in `[0, 2^32 - 1]`, which we engineer by
    ///   pinning `bp[0]` to `2^31` and `bp[1]` to `2^30`, so `bp[0][s+17] +
    ///   bp[1][s+13]` dominates the other (small) RHS terms.
    ///
    /// **C4/C5 (cyclic shifts mod X^32 - 1).** With `bp[0] = 2^31` (single
    /// bit at position 31), `bp[5]` is forced by C4 to be the cyclic shifts
    /// of bp[0] by 25 and 14 positions: bits at `(31+25) mod 32 = 24` and
    /// `(31+14) mod 32 = 13`, i.e., `bp[5] = 2^24 + 2^13`. Same for C5 with
    /// `bp[1] = 2^30` ⟹ `bp[3] = 2^23 + 2^11`. The single-bit pattern
    /// trivially satisfies both bit-gap constraints (no two bits to be
    /// 11 or 12 apart when there's only one bit).
    ///
    /// **C1 boundary cascade and the bp[1] sign flip.** At row `s = len-17`,
    /// the constraint LHS `bp[0][s+17]` reads as zero (out-of-bounds). With
    /// every C1 RHS term having a minus sign, the slack `bp[6][len-1]`
    /// would have to absorb a *negative* value, which a `BinaryPoly<32>`
    /// cannot represent. Flipping `bp[1]` to a plus sign makes
    /// `bp[6][len-1] = bp[1][len-4] - small ≈ 2^30 > 0`, so the slack is
    /// representable. For boundary rows `s ∈ [len-16, len-14]` where bp[6]
    /// itself is OOB, we zero out `bp[1][r]` for `r ∈ [len-3, len-1]` so
    /// that the constraint becomes `0 = 0` trivially. C5 then forces
    /// `bp[3][r] = 0` at those same rows.
    fn generate_random_trace<Rng: rand::RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, R, R, 32> {
        // Cap for "small" random binary polys feeding into C1's slack.
        // Bounded so that the sum of all small terms in C1's RHS plus the
        // dominant `bp[0][s+17].eval(2) + bp[1][s+13].eval(2)` stays comfortably
        // within `u32`. Capping at 24 bits gives plenty of headroom.
        const SMALL_MASK: u32 = (1u32 << 24) - 1;

        let len = 1usize << num_vars;

        // ---- Step 1: allocate columns ----
        let mut bp_cols: Vec<DenseMultilinearExtension<BinaryPoly<32>>> =
            vec![(0..len).map(|_| BinaryPoly::zero()).collect(); 14];
        let mut int_cols: Vec<DenseMultilinearExtension<R>> =
            vec![(0..len).map(|_| R::ZERO).collect(); 4];

        // ---- Step 2: fill bp[0], bp[1], bp[5], bp[3] (C4/C5 pinned) ----
        // bp[0] = 2^31, bp[1] = 2^30 at every row except the last 3 of each
        // (which must be zero so the C1/C2 boundary cascade is trivially
        // satisfied at rows s ∈ [len-16, len-14], where the slack columns
        // bp[6] / int[2] are out of bounds).
        let bp0_value: u32 = 1u32 << 31; // single bit at position 31
        let bp1_value: u32 = 1u32 << 30; // single bit at position 30
        // bp[5] = cyclic shifts of bp[0] by 25 and 14 in R[X]/(X^32-1):
        //   (31+25) mod 32 = 24, (31+14) mod 32 = 13 → bp[5] = 2^24 | 2^13
        let bp5_value: u32 = (1u32 << 24) | (1u32 << 13);
        // bp[3] = cyclic shifts of bp[1] by 25 and 13:
        //   (30+25) mod 32 = 23, (30+13) mod 32 = 11 → bp[3] = 2^23 | 2^11
        let bp3_value: u32 = (1u32 << 23) | (1u32 << 11);

        // Zero the last 3 rows of bp[0]/bp[1] (and dependents bp[5]/bp[3]).
        // This makes the C1/C2 boundary cases at s ∈ [len-16, len-14]
        // (where bp[0..1][s+13] is in-bounds but the slack columns are OOB)
        // satisfiable because every constraint term vanishes there.
        let pair_zero_start = len.saturating_sub(3);
        for r in 0..pair_zero_start {
            bp_cols[0][r] = BinaryPoly::from(bp0_value);
            bp_cols[1][r] = BinaryPoly::from(bp1_value);
            bp_cols[5][r] = BinaryPoly::from(bp5_value);
            bp_cols[3][r] = BinaryPoly::from(bp3_value);
        }
        // (rows pair_zero_start..len of bp[0]/bp[1]/bp[3]/bp[5] are already
        // initialized to BinaryPoly::zero())

        // ---- Step 3: fill bp[2], bp[4], bp[7], bp[8], bp[9], bp[10], bp[11]
        // ---- with small random values ----
        // Special handling for bp[4]: zero out the last 16 rows. C3's slack
        // (int[3], shifted by 16) is OOB at s ∈ [len-16, len-2], so C3 at
        // those rows reads `0 - bp[4][s] - bp[4][s+9] = 0`, which forces
        // both bp[4][s] and bp[4][s+9] to zero whenever in-bounds. The
        // combined zero range is r ∈ [len-16, len-1].
        let bp4_zero_start = len.saturating_sub(16);
        for k in [2usize, 7, 8, 9, 10, 11] {
            for r in 0..len {
                bp_cols[k][r] = BinaryPoly::from(rng.next_u32() & SMALL_MASK);
            }
        }
        for r in 0..bp4_zero_start {
            bp_cols[4][r] = BinaryPoly::from(rng.next_u32() & SMALL_MASK);
        }
        // (rows bp4_zero_start..len of bp[4] are already zero from initialization)

        // ---- Step 4: fill bp[12], bp[13] (unconstrained random fillers) ----
        for k in [12usize, 13] {
            for r in 0..len {
                bp_cols[k][r] = BinaryPoly::from(rng.next_u32() & SMALL_MASK);
            }
        }

        // ---- Step 5: fill int[0], int[1] from C6/C7 ----
        // int[0][r] = bp[10][r].eval(2); int[1][r] = bp[11][r].eval(2).
        for r in 0..len {
            let int0 = bp_cols[10][r]
                .evaluate_at_point(&1_u32) // start with eval at 1 to validate via popcount
                .ok();
            let _ = int0; // not used; we evaluate at 2 below.
            let int0 = bp_cols[10][r]
                .evaluate_at_point(&2_u32)
                .expect("24-bit binary poly eval at 2 fits in u32");
            let int1 = bp_cols[11][r]
                .evaluate_at_point(&2_u32)
                .expect("24-bit binary poly eval at 2 fits in u32");
            int_cols[0][r] = R::from(int0);
            int_cols[1][r] = R::from(int1);
        }

        // ---- Step 6: fill int[3] from C3 slack ----
        // C3: bp[4][t] - bp[4][t-16] - bp[4][t-7] - bp[8][t] - bp[9][t] - int[3][t] ∈ (X-2)
        // ⟹ int[3][t] = bp[4][t].eval(2) - bp[4][t-16].eval(2) - bp[4][t-7].eval(2)
        //               - bp[8][t].eval(2) - bp[9][t].eval(2)
        // For r < 7, bp[4][r-7] reads as 0 (zero-padded).
        // For r < 16, bp[4][r-16] reads as 0 too.
        // int[3] is i64-typed, so it can be negative or large.
        for r in 0..len {
            let bp4_t = bp_cols[4][r]
                .evaluate_at_point(&2_u32)
                .expect("24-bit binary poly eval at 2 fits in u32") as i64;
            let bp4_t_minus_16 = if r >= 16 {
                bp_cols[4][r - 16]
                    .evaluate_at_point(&2_u32)
                    .expect("24-bit binary poly eval at 2 fits in u32") as i64
            } else {
                0
            };
            let bp4_t_minus_7 = if r >= 7 {
                bp_cols[4][r - 7]
                    .evaluate_at_point(&2_u32)
                    .expect("24-bit binary poly eval at 2 fits in u32") as i64
            } else {
                0
            };
            let bp8_t = bp_cols[8][r]
                .evaluate_at_point(&2_u32)
                .expect("24-bit binary poly eval at 2 fits in u32") as i64;
            let bp9_t = bp_cols[9][r]
                .evaluate_at_point(&2_u32)
                .expect("24-bit binary poly eval at 2 fits in u32") as i64;
            let int3_val: i64 = bp4_t - bp4_t_minus_16 - bp4_t_minus_7 - bp8_t - bp9_t;
            int_cols[3][r] = R::from(int3_val);
        }

        // ---- Step 7: fill int[2] from C2 slack ----
        // C2: bp[1][t+1] - bp[0][t-3] - bp[1][t-3] - bp[2][t] - bp[3][t]
        //     - bp[7][t] - int[0][t] - int[2][t] ∈ (X-2)
        // ⟹ int[2][t] = bp[1][t+1].eval(2) - bp[0][t-3].eval(2) - bp[1][t-3].eval(2)
        //               - bp[2][t].eval(2) - bp[3][t].eval(2) - bp[7][t].eval(2) - int[0][t]
        // For boundary cases (t+1 ≥ len or t-3 < 0), the OOB cells read as 0.
        let eval2 = |bp: &BinaryPoly<32>| -> i64 {
            bp.evaluate_at_point(&2_u32)
                .expect("binary poly eval at 2 fits in u32") as i64
        };
        for r in 0..len {
            let bp1_t_plus_1 = if r + 1 < len { eval2(&bp_cols[1][r + 1]) } else { 0 };
            let bp0_t_minus_3 = if r >= 3 { eval2(&bp_cols[0][r - 3]) } else { 0 };
            let bp1_t_minus_3 = if r >= 3 { eval2(&bp_cols[1][r - 3]) } else { 0 };
            let bp2_t = eval2(&bp_cols[2][r]);
            let bp3_t = eval2(&bp_cols[3][r]);
            let bp7_t = eval2(&bp_cols[7][r]);
            let int0_t = eval2(&bp_cols[10][r]); // int[0] = bp[10].eval(2) (C6)
            let int2_val: i64 =
                bp1_t_plus_1 - bp0_t_minus_3 - bp1_t_minus_3 - bp2_t - bp3_t - bp7_t - int0_t;
            int_cols[2][r] = R::from(int2_val);
        }

        // ---- Step 8: fill bp[6] from C1 slack ----
        // C1: bp[0][t+1] + bp[1][t-3] - bp[2][t] - bp[3][t] - bp[4][t]
        //     - bp[5][t] - bp[6][t] - int[0][t] - int[1][t] ∈ (X-2)
        // Note the + on bp[1][t-3] (sign flip from the all-minus form).
        // ⟹ bp[6][t].eval(2) = bp[0][t+1].eval(2) + bp[1][t-3].eval(2)
        //                       - bp[2][t].eval(2) - bp[3][t].eval(2) - bp[4][t].eval(2)
        //                       - bp[5][t].eval(2) - int[0][t] - int[1][t]
        //
        // The protocol reads bp[6][r] at every s ∈ [0, len-2] via the +16
        // shift, i.e., at r = s + 16 ∈ [16, len+14]. The in-bounds part is
        // r ∈ [16, len-1]. We iterate r over [0, len-1] and just compute the
        // formula at every row; the values for r < 16 are unused but harmless.
        for r in 0..len {
            let bp0_t_plus_1 = if r + 1 < len { eval2(&bp_cols[0][r + 1]) } else { 0 };
            let bp1_t_minus_3 = if r >= 3 { eval2(&bp_cols[1][r - 3]) } else { 0 };
            let bp2_t = eval2(&bp_cols[2][r]);
            let bp3_t = eval2(&bp_cols[3][r]);
            let bp4_t = eval2(&bp_cols[4][r]);
            let bp5_t = eval2(&bp_cols[5][r]);
            let int0_t = eval2(&bp_cols[10][r]); // int[0] = bp[10].eval(2) (C6)
            let int1_t = eval2(&bp_cols[11][r]); // int[1] = bp[11].eval(2) (C7)
            let bp6_val: i64 =
                bp0_t_plus_1 + bp1_t_minus_3 - bp2_t - bp3_t - bp4_t - bp5_t - int0_t - int1_t;
            // bp[6][r] needs to be a non-negative u32 for `BinaryPoly::from`.
            // Our parameter choices guarantee this for r ∈ [16, len-1]:
            // bp[0][r+1] = 2^31 (always, except r+1 ≥ len at r = len-1) and
            // bp[1][r-3] = 2^30 (always, except bp[1] zeroed at len-3..len-1
            // and r < 3); the dominant positives sum to ≥ 2^30 ≈ 10^9, which
            // dwarfs the small subtracted terms (each ≤ 2^24).
            let bp6_u32: u32 = u32::try_from(bp6_val).unwrap_or_else(|_| {
                // Fallback for any row where the formula goes negative (e.g.
                // rows where bp[0][r+1] is OOB AND bp[1][r-3] vanishes). The
                // protocol may still read this value at some s, but only if
                // the corresponding constraint already vanishes trivially via
                // OOB-zero on the LHS.
                0
            });
            bp_cols[6][r] = BinaryPoly::from(bp6_u32);
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
        UairSignature::new(total, PublicColumnLayout::default(), shifts)
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

/// Test UAIR exercising the `(X^N - 1)` ideal.
///
/// Same shape as [`TestAirNoMultiplication`]: three `arbitrary_poly` columns
/// `(a, b, c)` and the single constraint `a + b - c ∈ (X^N - 1)`. The honest
/// random trace sets `c = a + b`, making the constraint expression literally
/// zero, which trivially belongs to any ideal.
pub struct XnMinusOneTestUair<R>(PhantomData<R>);

impl<R> Uair for XnMinusOneTestUair<R>
where
    R: ConstSemiring + From<i32> + 'static,
{
    type Ideal = XnMinusOneIdeal<32>;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(0, 3, 0);
        UairSignature::new(total, PublicColumnLayout::default(), vec![])
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
            &ideal_from_ref(&XnMinusOneIdeal::<32>),
        );
    }
}

impl<R> GenerateRandomTrace<32> for XnMinusOneTestUair<R>
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

/// Test UAIR exercising the mixed `(X - a) / (X^N - 1)` ideal.
///
/// Three `arbitrary_poly` columns `(a, b, c)` and two constraints that both
/// evaluate to zero on the honest trace (`c = a + b`):
/// - `a + b - c ∈ (X - 2)`
/// - `a + b - c ∈ (X^N - 1)`
pub struct MixedIdealTestUair<R>(PhantomData<R>);

impl<R> Uair for MixedIdealTestUair<R>
where
    R: ConstSemiring + From<i32> + 'static,
{
    type Ideal = MixedDegreeOneOrXnMinusOne<R, 32>;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(0, 3, 0);
        UairSignature::new(total, PublicColumnLayout::default(), vec![])
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

        let diff = up[0].clone() + &up[1] - &up[2];

        b.assert_in_ideal(
            diff.clone(),
            &ideal_from_ref(&MixedDegreeOneOrXnMinusOne::DegreeOne(
                DegreeOneIdeal::new(R::from(2)),
            )),
        );
        b.assert_in_ideal(
            diff,
            &ideal_from_ref(&MixedDegreeOneOrXnMinusOne::XnMinusOne(
                XnMinusOneIdeal::<32>,
            )),
        );
    }
}

impl<R> GenerateRandomTrace<32> for MixedIdealTestUair<R>
where
    R: ConstSemiring + From<i32> + 'static,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_random_trace<Rng: rand::RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, R, R, 32> {
        XnMinusOneTestUair::<R>::generate_random_trace(num_vars, rng)
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
        assert_uair_shape::<TestAirNoMultiplication<Int<LIMBS>>>(&[1]);
        assert_uair_shape::<TestAirScalarMultiplications<Int<LIMBS>>>(&[1]);
        assert_uair_shape::<BinaryDecompositionUair<u32>>(&[1]);
        assert_uair_shape::<BigLinearUair<u32>>(&[1; 17]);
        assert_uair_shape::<TestUairMixedShifts<Int<LIMBS>>>(&[1, 1]);
        assert_uair_shape::<XnMinusOneTestUair<Int<LIMBS>>>(&[1]);
        assert_uair_shape::<MixedIdealTestUair<Int<LIMBS>>>(&[1, 1]);
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
