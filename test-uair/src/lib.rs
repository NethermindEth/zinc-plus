#![allow(clippy::arithmetic_side_effects)] // UAIRs should not care about overflows
mod generate_trace;

pub use generate_trace::*;

use crypto_primitives::{
    ConstSemiring, FixedSemiring, Semiring, boolean::Boolean,
    crypto_bigint_int::Int as CbInt, crypto_bigint_uint::Uint as CbUint,
};
use num_traits::{ConstOne, ConstZero, Zero};
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

// ---------------------------------------------------------------------------
// EcdsaUair: 11-column / 258-row arithmetization of secp256k1 ECDSA
// signature verification using Shamir's trick for u1·G + u2·Q.
//
// Constraint set follows §6 of the Zinc+ IMPLEMENTATION.md (agentic-approach-v1
// branch). All constraints are `assert_zero` over `Int<4>` exact-integer
// equality. Selectors gate the boundary constraints (B3 at row 0, B4 at row
// 257) and disable the addition step at the final row (C5–C7 are gated by
// `(1 − sel_final)`).
//
// IMPORTANT: the witness produced by `generate_random_trace` is a *real*
// double-and-add walk over secp256k1's base field. Because the constraints
// express integer equality (not equality mod p), the resulting cell values
// only satisfy the doubling/addition relations modulo p, not exactly. The
// witness therefore does NOT satisfy the constraint system. It exists as a
// realistic-shape benchmark fixture for prover/verifier throughput numbers
// where soundness is not required. A satisfying witness for this exact
// constraint set would need auxiliary quotient/carry columns (not included
// here) or the all-zero fixed-point trace.
// ---------------------------------------------------------------------------

mod ecdsa_secp256k1 {
    //! Minimal secp256k1 helpers used by `EcdsaUair`'s witness generator.
    //! All field arithmetic widens to `Uint<8>` so additions and the schoolbook
    //! multiplication never overflow before the modular reduction.

    use super::{CbInt, CbUint};
    use num_traits::ConstZero;

    pub type U4 = CbUint<4>;
    pub type U8 = CbUint<8>;
    pub type I4 = CbInt<4>;

    /// secp256k1 base field prime: p = 2^256 − 2^32 − 977.
    pub const P: U4 = U4::from_be_hex(
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F",
    );

    /// Standard secp256k1 generator.
    pub const GX: U4 = U4::from_be_hex(
        "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798",
    );
    pub const GY: U4 = U4::from_be_hex(
        "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8",
    );

    /// `Q = 2·G` (the chosen "public key" for this benchmark fixture).
    pub const QX: U4 = U4::from_be_hex(
        "C6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5",
    );
    pub const QY: U4 = U4::from_be_hex(
        "1AE168FEA63DC339A3C58419466CEAEEF7F632653266D0E1236431A950CFE52A",
    );

    /// `G + Q = 3·G`.
    pub const GQX: U4 = U4::from_be_hex(
        "F9308A019258C31049344F85F89D5229B531C845836F99B08601F113BCE036F9",
    );
    pub const GQY: U4 = U4::from_be_hex(
        "388F7B0F632DE8140FE337E62A37F3566500A99934C2231B6CB9FD7584B8E672",
    );

    /// Boundary-check x-coordinate (any fixed value is fine — the witness
    /// intentionally violates B4 anyway).
    pub const R_SIG: U4 = GX;

    fn p_wide() -> U8 {
        P.resize::<8>()
    }

    fn fp_reduce(v: U8) -> U4 {
        (v % p_wide()).resize::<4>()
    }

    pub fn fp_sub(a: U4, b: U4) -> U4 {
        let p8 = p_wide();
        fp_reduce(a.resize::<8>() + p8 - b.resize::<8>())
    }

    pub fn fp_mul(a: U4, b: U4) -> U4 {
        fp_reduce(a.resize::<8>() * b.resize::<8>())
    }

    /// Reinterpret a `Uint<4>` bit pattern as `Int<4>` (two's complement).
    /// Values in `[2^255, 2^256)` become negative; this is fine because the
    /// witness is intentionally non-satisfying.
    pub fn to_int4(v: U4) -> I4 {
        *v.as_int()
    }

    #[derive(Clone, Copy)]
    pub struct Point {
        pub x: U4,
        pub y: U4,
        pub z: U4,
    }

    pub const IDENTITY: Point = Point {
        x: U4::from_words([0, 0, 0, 0]),
        y: U4::from_words([1, 0, 0, 0]),
        z: U4::from_words([0, 0, 0, 0]),
    };

    /// Doubling for short Weierstrass curves with `a = 0` (secp256k1).
    /// Matches the spec used by C1/C2/C3:
    ///     Z3 = 2 Y Z
    ///     X3 = 9 X^4 − 8 X Y^2
    ///     Y3 = 12 X^3 Y^2 − 3 X^2 X3 − 8 Y^4
    pub fn point_double(p: Point) -> Point {
        let two = U4::from(2u32);
        let three = U4::from(3u32);
        let eight = U4::from(8u32);
        let nine = U4::from(9u32);
        let twelve = U4::from(12u32);

        let z3 = fp_mul(two, fp_mul(p.y, p.z));
        let x_sq = fp_mul(p.x, p.x);
        let y_sq = fp_mul(p.y, p.y);
        let x_4 = fp_mul(x_sq, x_sq);
        let y_4 = fp_mul(y_sq, y_sq);
        let x3 = fp_sub(fp_mul(nine, x_4), fp_mul(eight, fp_mul(p.x, y_sq)));
        let term1 = fp_mul(twelve, fp_mul(fp_mul(x_sq, p.x), y_sq));
        let term2 = fp_mul(three, fp_mul(x_sq, x3));
        let term3 = fp_mul(eight, y_4);
        let y3 = fp_sub(fp_sub(term1, term2), term3);
        Point { x: x3, y: y3, z: z3 }
    }

    /// Addition of a Jacobian point `mid` and an affine point `(t_x, t_y)`,
    /// matching the spec used by C4–C7. Returns the new Jacobian point and
    /// the scratch value `H` (stored in trace column `H`).
    pub fn point_add_affine(mid: Point, t_x: U4, t_y: U4) -> (Point, U4) {
        let two = U4::from(2u32);
        let z_mid_sq = fp_mul(mid.z, mid.z);
        let z_mid_cu = fp_mul(z_mid_sq, mid.z);
        let h = fp_sub(fp_mul(t_x, z_mid_sq), mid.x);
        let r_a = fp_sub(fp_mul(t_y, z_mid_cu), mid.y);
        let h_sq = fp_mul(h, h);
        let h_cu = fp_mul(h_sq, h);
        let r_a_sq = fp_mul(r_a, r_a);
        let x3 = fp_sub(fp_sub(r_a_sq, h_cu), fp_mul(two, fp_mul(mid.x, h_sq)));
        let y3 = fp_sub(
            fp_mul(r_a, fp_sub(fp_mul(mid.x, h_sq), x3)),
            fp_mul(mid.y, h_cu),
        );
        let z3 = fp_mul(mid.z, h);
        (Point { x: x3, y: y3, z: z3 }, h)
    }

    /// Selects the table point T = {O, G, Q, G+Q}[b1 + 2·b2].
    /// Returns (T_x, T_y). The (false, false) → identity case returns (0, 0)
    /// because s = 0 there and the addition step is skipped anyway.
    pub fn select_table_point(b1: bool, b2: bool) -> (U4, U4) {
        match (b1, b2) {
            (false, false) => (U4::ZERO, U4::ZERO),
            (true, false) => (GX, GY),
            (false, true) => (QX, QY),
            (true, true) => (GQX, GQY),
        }
    }
}

/// Column indices in the flat int-column group (no binary_poly or
/// arbitrary_poly columns). Public columns precede witness columns within
/// the int group.
///
/// `R_A`, `H_SQ`, `H_CU` are *wire* columns: their values are constrained
/// to equal the inlined sub-expressions `T_y·Z_mid³ − Y_mid`, `H²`, `H³`
/// respectively, so the addition-step constraints C5–C7 can reference them
/// at degree 1 instead of recomputing high-degree expressions inline. This
/// drops the maximum constraint degree from 13 to 5.
#[allow(dead_code)]
mod ecdsa_cols {
    pub const B1: usize = 0;
    pub const B2: usize = 1;
    pub const SEL_INIT: usize = 2;
    pub const SEL_FINAL: usize = 3;
    pub const X: usize = 4;
    pub const Y: usize = 5;
    pub const Z: usize = 6;
    pub const X_MID: usize = 7;
    pub const Y_MID: usize = 8;
    pub const Z_MID: usize = 9;
    pub const H: usize = 10;
    pub const R_A: usize = 11;
    pub const H_SQ: usize = 12;
    pub const H_CU: usize = 13;

    pub const NUM_INT_COLS: usize = 14;

    /// `down.int[*]` indices after `UairSignature::new`'s stable sort by
    /// `source_col` (X < Y < Z).
    pub const DOWN_X: usize = 0;
    pub const DOWN_Y: usize = 1;
    pub const DOWN_Z: usize = 2;
}

/// Number of "real" rows in the ECDSA trace: 1 init row + 256 walk rows +
/// 1 final row. The witness is padded out to `1 << num_vars` ≥ 258, so
/// `num_vars ≥ 9`.
pub const ECDSA_NUM_REAL_ROWS: usize = 258;

pub struct EcdsaUair;

impl Uair for EcdsaUair {
    type Ideal = ImpossibleIdeal;
    type Scalar = DensePolynomial<CbInt<4>, 32>;

    fn signature() -> UairSignature {
        // 0 binary-poly, 0 arbitrary-poly, 14 int columns
        // (4 public + 10 witness; the last three witness columns wire R_a,
        // H², H³ to keep the addition-step constraints at low degree).
        let total = TotalColumnLayout::new(0, 0, ecdsa_cols::NUM_INT_COLS);
        // 4 public int columns: b1, b2, sel_init, sel_final.
        let public = PublicColumnLayout::new(0, 0, 4);
        // X, Y, Z each shifted by 1 (next-row accumulator).
        let shifts = vec![
            ShiftSpec::new(ecdsa_cols::X, 1),
            ShiftSpec::new(ecdsa_cols::Y, 1),
            ShiftSpec::new(ecdsa_cols::Z, 1),
        ];
        UairSignature::new(total, public, shifts)
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        down: TraceRow<B::Expr>,
        from_ref: FromR,
        _mbs: MulByScalar,
        _ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&Self::Scalar) -> B::Expr,
    {
        use ecdsa_cols::*;
        use ecdsa_secp256k1::{GQX, GQY, GX, GY, QX, QY, R_SIG, to_int4};

        // Inject an Int<4> constant as a degree-0 scalar polynomial expression.
        let cst = |k: CbInt<4>| -> B::Expr {
            let mut coeffs = [CbInt::<4>::ZERO; 32];
            coeffs[0] = k;
            from_ref(&DensePolynomial::<CbInt<4>, 32>::new(coeffs))
        };
        let cst_uint = |k: ecdsa_secp256k1::U4| cst(to_int4(k));

        // Column expression aliases (current row).
        let b1 = &up.int[B1];
        let b2 = &up.int[B2];
        let sel_init = &up.int[SEL_INIT];
        let sel_final = &up.int[SEL_FINAL];
        let x = &up.int[X];
        let y = &up.int[Y];
        let z = &up.int[Z];
        let x_mid = &up.int[X_MID];
        let y_mid = &up.int[Y_MID];
        let z_mid = &up.int[Z_MID];
        let h = &up.int[H];
        let r_a_col = &up.int[R_A];
        let h_sq_col = &up.int[H_SQ];
        let h_cu_col = &up.int[H_CU];

        // Shifted (next-row) accumulator coordinates.
        let x_next = &down.int[DOWN_X];
        let y_next = &down.int[DOWN_Y];
        let z_next = &down.int[DOWN_Z];

        let one = || cst(CbInt::<4>::ONE);
        let two = || cst(CbInt::<4>::from_i8(2));
        let three = || cst(CbInt::<4>::from_i8(3));
        let eight = || cst(CbInt::<4>::from_i8(8));
        let nine = || cst(CbInt::<4>::from_i8(9));
        let twelve = || cst(CbInt::<4>::from_i8(12));

        // Shamir indicator: s = b1 + b2 − b1·b2  (= 1 iff any bit set).
        let s = || b1.clone() + b2 - &(b1.clone() * b2);
        let one_minus_s = || one() - &s();

        // Inlined table-point sub-expressions:
        //   T_x = (1−b1)b2·Qx + b1(1−b2)·Gx + b1·b2·GQx
        //   T_y = (1−b1)b2·Qy + b1(1−b2)·Gy + b1·b2·GQy
        // (the (1−b1)(1−b2) → identity case contributes 0 to both)
        let t_x = || {
            let s_g = b1.clone() * &(one() - b2);
            let s_q = (one() - b1) * b2;
            let s_gq = b1.clone() * b2;
            s_g * &cst_uint(GX) + &(s_q * &cst_uint(QX)) + &(s_gq * &cst_uint(GQX))
        };
        let t_y = || {
            let s_g = b1.clone() * &(one() - b2);
            let s_q = (one() - b1) * b2;
            let s_gq = b1.clone() * b2;
            s_g * &cst_uint(GY) + &(s_q * &cst_uint(QY)) + &(s_gq * &cst_uint(GQY))
        };

        // -- C1: Z_mid − 2·Y·Z = 0 -------------------------------------------
        b.assert_zero(z_mid.clone() - &(two() * y * z));

        // -- C2: X_mid − (9·X^4 − 8·X·Y^2) = 0 -------------------------------
        let x_sq = || x.clone() * x;
        let y_sq = || y.clone() * y;
        let x_4 = || x_sq() * &x_sq();
        let y_4 = || y_sq() * &y_sq();
        b.assert_zero(x_mid.clone() - &(nine() * &x_4()) + &(eight() * x * &y_sq()));

        // -- C3: Y_mid − (12·X^3·Y^2 − 3·X^2·X_mid − 8·Y^4) = 0 ---------------
        let x_cu = || x_sq() * x;
        b.assert_zero(
            y_mid.clone() - &(twelve() * &x_cu() * &y_sq())
                + &(three() * &x_sq() * x_mid)
                + &(eight() * &y_4()),
        );

        // -- C4: H − (T_x · Z_mid^2 − X_mid) = 0 -----------------------------
        let z_mid_sq = || z_mid.clone() * z_mid;
        b.assert_zero(h.clone() - &(t_x() * &z_mid_sq()) + x_mid);

        // -- W1: H_sq − H·H = 0  (wire H² so C6/C7 can read it at degree 1) -
        b.assert_zero(h_sq_col.clone() - &(h.clone() * h));

        // -- W2: H_cu − H_sq·H = 0 (chained wire so C6/C7 read H³ as degree 1)
        b.assert_zero(h_cu_col.clone() - &(h_sq_col.clone() * h));

        // -- W3: R_a − (T_y·Z_mid^3 − Y_mid) = 0 -----------------------------
        let z_mid_cu = z_mid_sq() * z_mid;
        b.assert_zero(r_a_col.clone() - &(t_y() * &z_mid_cu) + y_mid);

        // The previous (1 − sel_final) gating on C5–C7 is dropped: at the
        // final boundary row the addition step would otherwise read
        // out-of-bounds zero-padded down cells, which is benign here because
        // the witness is non-satisfying anyway. Removing the gate strips
        // one degree off each of C5/C6/C7.

        // -- C5: Z[t+1] − ((1−s)·Z_mid + s·Z_mid·H) = 0 ----------------------
        b.assert_zero(
            z_next.clone() - &(one_minus_s() * z_mid) - &(s() * z_mid * h),
        );

        // -- C6: X[t+1] − ((1−s)·X_mid
        //                  + s·(R_a^2 − H^3 − 2·X_mid·H^2)) = 0 -------------
        let r_a_sq = r_a_col.clone() * r_a_col;
        let c6_addend = r_a_sq - h_cu_col - &(two() * x_mid * h_sq_col);
        b.assert_zero(
            x_next.clone() - &(one_minus_s() * x_mid) - &(s() * &c6_addend),
        );

        // -- C7: Y[t+1] − ((1−s)·Y_mid
        //                  + s·(R_a·(X_mid·H^2 − X[t+1]) − Y_mid·H^3)) = 0 --
        let c7_addend =
            r_a_col.clone() * &(x_mid.clone() * h_sq_col - x_next)
                - &(y_mid.clone() * h_cu_col);
        b.assert_zero(
            y_next.clone() - &(one_minus_s() * y_mid) - &(s() * &c7_addend),
        );

        // -- B3: sel_init · Z = 0 (force identity at row 0) -------------------
        b.assert_zero(sel_init.clone() * z);

        // -- B4: sel_final · Z · (X − R_SIG · Z^2) = 0 ------------------------
        let z_sq = z.clone() * z;
        b.assert_zero(sel_final.clone() * z * &(x.clone() - &(cst_uint(R_SIG) * &z_sq)));
    }
}

impl GenerateRandomTrace<32> for EcdsaUair {
    type PolyCoeff = CbInt<4>;
    type Int = CbInt<4>;

    /// Builds a real Shamir's-trick walk over secp256k1 for fixed scalars
    /// `u1`, `u2` derived deterministically from `rng`. The 256 walk rows
    /// (rows 0..256) each hold the accumulator state, the doubled state,
    /// and the addition scratch `H` for that step. Row 0 is initialised
    /// from the point at infinity. Row 257 is the boundary row holding the
    /// final accumulator. Padding rows beyond row 257 are zero.
    ///
    /// **The witness intentionally does NOT satisfy the constraint
    /// system.** See the module-level note above.
    fn generate_random_trace<Rng: rand::RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, CbInt<4>, CbInt<4>, 32> {
        use ecdsa_cols::*;
        use ecdsa_secp256k1::{
            IDENTITY, Point, fp_mul, fp_sub, point_add_affine, point_double, select_table_point,
            to_int4,
        };

        let len = 1usize << num_vars;
        assert!(
            len >= ECDSA_NUM_REAL_ROWS,
            "ECDSA UAIR requires num_vars >= 9 (got {num_vars}, len = {len})"
        );

        // ---- Step 1: pick deterministic scalars u1, u2 -----------------
        // Sample 32 random bytes per scalar; the constraint system does not
        // depend on the scalars matching any particular signature.
        let mut u1_bytes = [0u8; 32];
        let mut u2_bytes = [0u8; 32];
        rng.fill_bytes(&mut u1_bytes);
        rng.fill_bytes(&mut u2_bytes);

        // Bit `i` of u_k (0 ≤ i < 256) at row position `i`, processed MSB-first
        // so the standard double-and-add walk produces u_k · base.
        let bit_msb_first = |bytes: &[u8; 32], i: usize| -> bool {
            // i = 0 → MSB of bytes[0]; i = 255 → LSB of bytes[31].
            let byte = bytes[i / 8];
            ((byte >> (7 - (i % 8))) & 1) == 1
        };

        // ---- Step 2: allocate columns ---------------------------------
        let zero = CbInt::<4>::ZERO;
        let mut int_cols: Vec<DenseMultilinearExtension<CbInt<4>>> =
            (0..NUM_INT_COLS).map(|_| (0..len).map(|_| zero).collect()).collect();

        // Helper: given a row's mid-point and the chosen table point T,
        // compute the H scratch and the wired R_a / H_sq / H_cu values.
        let aux_at_row = |mid: Point, t_x, t_y| {
            let z_mid_sq = fp_mul(mid.z, mid.z);
            let z_mid_cu = fp_mul(z_mid_sq, mid.z);
            let h = fp_sub(fp_mul(t_x, z_mid_sq), mid.x);
            let r_a = fp_sub(fp_mul(t_y, z_mid_cu), mid.y);
            let h_sq = fp_mul(h, h);
            let h_cu = fp_mul(h_sq, h);
            (h, r_a, h_sq, h_cu)
        };

        // ---- Step 3: walk -------------------------------------------------
        let mut acc: Point = IDENTITY;
        for row in 0..256usize {
            let b1 = bit_msb_first(&u1_bytes, row);
            let b2 = bit_msb_first(&u2_bytes, row);
            let mid = point_double(acc);
            let (t_x, t_y) = select_table_point(b1, b2);
            let s = b1 || b2;
            let (h, r_a, h_sq, h_cu) = aux_at_row(mid, t_x, t_y);
            let next = if s {
                point_add_affine(mid, t_x, t_y).0
            } else {
                // s = 0 → addition step is a no-op; the next-row accumulator
                // is just the doubled point.
                mid
            };

            int_cols[B1][row] = to_int4(if b1 { ecdsa_secp256k1::U4::ONE } else { ecdsa_secp256k1::U4::ZERO });
            int_cols[B2][row] = to_int4(if b2 { ecdsa_secp256k1::U4::ONE } else { ecdsa_secp256k1::U4::ZERO });
            int_cols[X][row] = to_int4(acc.x);
            int_cols[Y][row] = to_int4(acc.y);
            int_cols[Z][row] = to_int4(acc.z);
            int_cols[X_MID][row] = to_int4(mid.x);
            int_cols[Y_MID][row] = to_int4(mid.y);
            int_cols[Z_MID][row] = to_int4(mid.z);
            int_cols[H][row] = to_int4(h);
            int_cols[R_A][row] = to_int4(r_a);
            int_cols[H_SQ][row] = to_int4(h_sq);
            int_cols[H_CU][row] = to_int4(h_cu);

            acc = next;
        }

        // ---- Step 4: row 256 — first "tail" row holding the result --------
        // This row stores the final accumulator (= u1·G + u2·Q in Jacobian).
        // The doubled / H / wire cells are computed from the spec for
        // self-consistent shape; they still won't satisfy the constraint
        // system exactly, since arithmetic is modular.
        {
            let mid = point_double(acc);
            let (t_x, t_y) = (ecdsa_secp256k1::U4::ZERO, ecdsa_secp256k1::U4::ZERO);
            let (h, r_a, h_sq, h_cu) = aux_at_row(mid, t_x, t_y);
            int_cols[X][256] = to_int4(acc.x);
            int_cols[Y][256] = to_int4(acc.y);
            int_cols[Z][256] = to_int4(acc.z);
            int_cols[X_MID][256] = to_int4(mid.x);
            int_cols[Y_MID][256] = to_int4(mid.y);
            int_cols[Z_MID][256] = to_int4(mid.z);
            int_cols[H][256] = to_int4(h);
            int_cols[R_A][256] = to_int4(r_a);
            int_cols[H_SQ][256] = to_int4(h_sq);
            int_cols[H_CU][256] = to_int4(h_cu);
        }

        // ---- Step 5: row 257 — final boundary row -----------------------
        // sel_final fires here. B4 fires; C1–C7 will read mostly zero cells
        // and be violated as documented.
        int_cols[X][257] = to_int4(acc.x);
        int_cols[Y][257] = to_int4(acc.y);
        int_cols[Z][257] = to_int4(acc.z);

        // ---- Step 6: selectors -------------------------------------------
        int_cols[SEL_INIT][0] = to_int4(ecdsa_secp256k1::U4::ONE);
        int_cols[SEL_FINAL][257] = to_int4(ecdsa_secp256k1::U4::ONE);

        UairTrace {
            binary_poly: vec![].into(),
            arbitrary_poly: vec![].into(),
            int: int_cols.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// ShaProxyEcdsaUair: SHAProxy<Int<4>> + EcdsaUair on a single heterogeneous
// trace, with both constraint sets active per row.
//
// Layout (14 binary_poly + 0 arbitrary_poly + 18 int):
//   bp[0..14]   = SHAProxy's 14 binary_poly columns
//   int[0..14]  = EcdsaUair's 14 int columns (4 public, 10 witness),
//                 contiguous so EcdsaUair's hard-coded indices still work.
//   int[14..18] = SHAProxy's 4 int columns.
//
// Both component UAIRs' `constrain_general` impls are reused unchanged via
// sliced sub-views of the incoming `up`/`down` rows. The Ideal type is
// SHAProxy's mixed ideal; EcdsaUair only emits `assert_zero` calls so its
// `IFromR` closure is supplied as an unreachable stub.
//
// IMPORTANT: the witness inherits EcdsaUair's non-satisfaction (real
// secp256k1 walk vs. integer-equality constraints). This UAIR is intended
// for prover-only benchmarking, just like `EcdsaUair`.
// ---------------------------------------------------------------------------

pub struct ShaProxyEcdsaUair;

impl Uair for ShaProxyEcdsaUair {
    type Ideal = MixedDegreeOneOrXnMinusOne<CbInt<4>, 32>;
    type Scalar = DensePolynomial<CbInt<4>, 32>;

    fn signature() -> UairSignature {
        // 14 bp + 0 ap + 18 int. The 18 int cols are arranged as
        // [4 ECDSA public, 10 ECDSA witness, 4 SHAProxy witness].
        let total = TotalColumnLayout::new(14, 0, 18);
        // Same 4 public ECDSA int cols as `EcdsaUair`.
        let public = PublicColumnLayout::new(0, 0, 4);

        // Combined shift list. Source-col indices are flat (bp || ap || int),
        // so the int prefix in this layout starts at flat index 14, and
        // SHAProxy's int cells start at int position 14 (= flat 28). ECDSA's
        // int shifts (originally on flat 4..6 in EcdsaUair alone) get bumped
        // by +14 because of the new bp prefix; SHAProxy's int shifts
        // (originally on flat 14..17 in SHAProxy alone) also get bumped by
        // +14 because SHAProxy's int cells now sit after 14 ECDSA int cells.
        let mut shifts: Vec<ShiftSpec> = SHAProxy::<CbInt<4>>::signature()
            .shifts()
            .iter()
            .map(|s| {
                if s.source_col() < 14 {
                    // SHAProxy bp shift — flat index unchanged.
                    ShiftSpec::new(s.source_col(), s.shift_amount())
                } else {
                    // SHAProxy int shift — bump by +14 because SHAProxy's
                    // int cells are at combined int[14..18] (flat 28..32).
                    ShiftSpec::new(s.source_col() + 14, s.shift_amount())
                }
            })
            .collect();
        for s in EcdsaUair::signature().shifts() {
            // ECDSA int shift — bump by +14 (the bp prefix in the combined
            // layout). EcdsaUair has no bp cells of its own, so all of its
            // shifts are int shifts.
            shifts.push(ShiftSpec::new(s.source_col() + 14, s.shift_amount()));
        }
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
        // ---- ECDSA sub-view ---------------------------------------------
        // EcdsaUair reads up.int[0..14] and down.int[0..3]. Combined int
        // cells [0..14] are exactly EcdsaUair's, contiguous. Combined down
        // int cells [0..3] are exactly EcdsaUair's X/Y/Z shifts (sorted
        // before SHAProxy's int shifts because their flat source_col
        // indices 18,19,20 are smaller than SHAProxy's 28..31).
        let ecdsa_up = TraceRow {
            binary_poly: &[],
            arbitrary_poly: &[],
            int: &up.int[0..14],
        };
        let ecdsa_down = TraceRow {
            binary_poly: &[],
            arbitrary_poly: &[],
            int: &down.int[0..3],
        };
        EcdsaUair::constrain_general(
            b,
            ecdsa_up,
            ecdsa_down,
            &from_ref,
            &mbs,
            // EcdsaUair never calls this — it only emits `assert_zero`.
            |_: &ImpossibleIdeal| -> B::Ideal { unreachable!() },
        );

        // ---- SHAProxy sub-view ------------------------------------------
        // SHAProxy reads up.binary_poly[0..14], up.int[0..4],
        // down.binary_poly[0..17], and down.int[0..4]. Combined bp cells are
        // unchanged, combined int[14..18] are SHAProxy's 4 int cells, and
        // combined down int[3..7] are SHAProxy's 4 int shifts.
        let sha_up = TraceRow {
            binary_poly: up.binary_poly,
            arbitrary_poly: up.arbitrary_poly,
            int: &up.int[14..18],
        };
        let sha_down = TraceRow {
            binary_poly: down.binary_poly,
            arbitrary_poly: down.arbitrary_poly,
            int: &down.int[3..7],
        };
        SHAProxy::<CbInt<4>>::constrain_general(
            b,
            sha_up,
            sha_down,
            &from_ref,
            &mbs,
            &ideal_from_ref,
        );
    }
}

impl GenerateRandomTrace<32> for ShaProxyEcdsaUair {
    type PolyCoeff = CbInt<4>;
    type Int = CbInt<4>;

    /// Concatenates `SHAProxy::<Int<4>>::generate_random_trace` and
    /// `EcdsaUair::generate_random_trace` into a single combined trace.
    /// `num_vars ≥ 9` is required (inherited from `EcdsaUair`).
    ///
    /// **The witness intentionally does NOT satisfy the combined constraint
    /// system**: SHAProxy's part is satisfying, but EcdsaUair's part
    /// violates exact-integer equality (modular vs. integer arithmetic).
    /// Use only for prover-only benchmarking.
    fn generate_random_trace<Rng: rand::RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, CbInt<4>, CbInt<4>, 32> {
        let sha_trace = SHAProxy::<CbInt<4>>::generate_random_trace(num_vars, rng);
        let ecdsa_trace = EcdsaUair::generate_random_trace(num_vars, rng);

        // ECDSA int cols first (so they're contiguous at int[0..14]),
        // then SHAProxy int cols (at int[14..18]).
        let mut int_cols = ecdsa_trace.int.into_owned();
        int_cols.extend(sha_trace.int.into_owned());

        UairTrace {
            binary_poly: sha_trace.binary_poly,
            arbitrary_poly: vec![].into(),
            int: int_cols.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// EcdsaUairLimbs: limb-decomposed variant of EcdsaUair.
//
// Each 256-bit Jacobian coordinate is stored as 8 little-endian u32 limbs in
// 8 separate witness columns of i64 cells. The constraint set mirrors
// EcdsaUair one-for-one, but each Int<4> constraint is emitted as 8
// per-limb constraints. There are no carry columns: high partial-product
// limbs are silently discarded (matches Int<4>'s mod-2^256 wrap-around).
// The witness inherits EcdsaUair's non-satisfaction property.
//
// Cell type is i64 (matching `BenchZincTypes::Int = i64`) so the bench
// reuses `do_bench_uair` infrastructure with no new ZincTypes.
// ---------------------------------------------------------------------------

#[allow(dead_code)]
mod ecdsa_limbs_cols {
    pub const NUM_INT_COLS: usize = 84;

    pub const B1: usize = 0;
    pub const B2: usize = 1;
    pub const SEL_INIT: usize = 2;
    pub const SEL_FINAL: usize = 3;

    // Each 8-cell run is one logical 256-bit value, little-endian limbs.
    pub const X: usize = 4;
    pub const Y: usize = 12;
    pub const Z: usize = 20;
    pub const X_MID: usize = 28;
    pub const Y_MID: usize = 36;
    pub const Z_MID: usize = 44;
    pub const H: usize = 52;
    pub const R_A: usize = 60;
    pub const H_SQ: usize = 68;
    pub const H_CU: usize = 76;

    /// `down.int[*]` index of the start of each shifted-source value's 8
    /// limbs, after `UairSignature::new`'s stable sort by `source_col`.
    pub const DOWN_X: usize = 0;
    pub const DOWN_Y: usize = 8;
    pub const DOWN_Z: usize = 16;
}

/// Decompose a 256-bit `Uint<4>` into 8 little-endian u32 limbs.
fn uint4_to_u32_limbs(v: ecdsa_secp256k1::U4) -> [u32; 8] {
    let w = v.to_words(); // [u64; 4]
    [
        w[0] as u32,
        (w[0] >> 32) as u32,
        w[1] as u32,
        (w[1] >> 32) as u32,
        w[2] as u32,
        (w[2] >> 32) as u32,
        w[3] as u32,
        (w[3] >> 32) as u32,
    ]
}

pub struct EcdsaUairLimbs;

impl Uair for EcdsaUairLimbs {
    type Ideal = DegreeOneIdeal<i64>;
    type Scalar = DensePolynomial<i64, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(0, 0, ecdsa_limbs_cols::NUM_INT_COLS);
        let public = PublicColumnLayout::new(0, 0, 4);

        // 24 shift specs: X[k], Y[k], Z[k] each shifted by 1, for k = 0..8.
        let mut shifts = Vec::with_capacity(24);
        for k in 0..8 {
            shifts.push(ShiftSpec::new(ecdsa_limbs_cols::X + k, 1));
        }
        for k in 0..8 {
            shifts.push(ShiftSpec::new(ecdsa_limbs_cols::Y + k, 1));
        }
        for k in 0..8 {
            shifts.push(ShiftSpec::new(ecdsa_limbs_cols::Z + k, 1));
        }
        UairSignature::new(total, public, shifts)
    }

    #[allow(clippy::too_many_lines)]
    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        down: TraceRow<B::Expr>,
        from_ref: FromR,
        _mbs: MulByScalar,
        _ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&Self::Scalar) -> B::Expr,
    {
        use ecdsa_limbs_cols::*;
        use ecdsa_secp256k1::{GQX, GQY, GX, GY, QX, QY, R_SIG};
        use std::array;

        // Inject an i64 constant as a degree-0 scalar polynomial expression.
        let cst = |k: i64| -> B::Expr {
            let mut coeffs = [0i64; 32];
            coeffs[0] = k;
            from_ref(&DensePolynomial::<i64, 32>::new(coeffs))
        };
        let zero = cst(0);

        // ---- Limb-array helpers ----------------------------------------
        let limbs_at = |start: usize| -> [B::Expr; 8] {
            array::from_fn(|i| up.int[start + i].clone())
        };
        let down_limbs_at = |start: usize| -> [B::Expr; 8] {
            array::from_fn(|i| down.int[start + i].clone())
        };
        let const_limbs = |c: ecdsa_secp256k1::U4| -> [B::Expr; 8] {
            let limbs = uint4_to_u32_limbs(c);
            array::from_fn(|i| cst(limbs[i] as i64))
        };

        let add = |a: &[B::Expr; 8], bb: &[B::Expr; 8]| -> [B::Expr; 8] {
            array::from_fn(|i| a[i].clone() + &bb[i])
        };
        let sub = |a: &[B::Expr; 8], bb: &[B::Expr; 8]| -> [B::Expr; 8] {
            array::from_fn(|i| a[i].clone() - &bb[i])
        };
        let scalar_mul = |k_expr: &B::Expr, a: &[B::Expr; 8]| -> [B::Expr; 8] {
            array::from_fn(|i| k_expr.clone() * &a[i])
        };
        // Schoolbook multi-limb multiplication, truncated to limbs 0..7.
        // Output[k] = sum over (i, j) with i + j = k and i + j < 8 of a[i] * b[j].
        let mul = |a: &[B::Expr; 8], bb: &[B::Expr; 8]| -> [B::Expr; 8] {
            let mut out: [B::Expr; 8] = array::from_fn(|_| zero.clone());
            for i in 0..8 {
                for j in 0..8 {
                    if i + j < 8 {
                        out[i + j] = out[i + j].clone() + &(a[i].clone() * &bb[j]);
                    }
                }
            }
            out
        };

        // ---- Column views ----------------------------------------------
        let b1 = up.int[B1].clone();
        let b2 = up.int[B2].clone();
        let sel_init = up.int[SEL_INIT].clone();
        let sel_final = up.int[SEL_FINAL].clone();

        let x = limbs_at(X);
        let y = limbs_at(Y);
        let z = limbs_at(Z);
        let x_mid = limbs_at(X_MID);
        let y_mid = limbs_at(Y_MID);
        let z_mid = limbs_at(Z_MID);
        let h = limbs_at(H);
        let r_a = limbs_at(R_A);
        let h_sq = limbs_at(H_SQ);
        let h_cu = limbs_at(H_CU);

        let x_next = down_limbs_at(DOWN_X);
        let y_next = down_limbs_at(DOWN_Y);
        let z_next = down_limbs_at(DOWN_Z);

        // ---- Inlined sub-expressions ------------------------------------
        // Shamir indicator s = b1 + b2 - b1*b2 (a single B::Expr, degree 2).
        let s = b1.clone() + &b2 - &(b1.clone() * &b2);
        let one_minus_s = cst(1) - &s;

        // Table-point limbs: T_x = (1-b1)·b2·Qx + b1·(1-b2)·Gx + b1·b2·GQx
        // (and same for T_y). Each limb is a degree-2 expression in (b1, b2).
        let one_minus_b1 = cst(1) - &b1;
        let one_minus_b2 = cst(1) - &b2;
        let s_g = b1.clone() * &one_minus_b2; // b1·(1-b2)
        let s_q = one_minus_b1 * &b2; // (1-b1)·b2
        let s_gq = b1.clone() * &b2; // b1·b2

        let gx_limbs = const_limbs(GX);
        let gy_limbs = const_limbs(GY);
        let qx_limbs = const_limbs(QX);
        let qy_limbs = const_limbs(QY);
        let gqx_limbs = const_limbs(GQX);
        let gqy_limbs = const_limbs(GQY);
        let r_sig_limbs = const_limbs(R_SIG);

        let t_x: [B::Expr; 8] = array::from_fn(|i| {
            s_g.clone() * &gx_limbs[i]
                + &(s_q.clone() * &qx_limbs[i])
                + &(s_gq.clone() * &gqx_limbs[i])
        });
        let t_y: [B::Expr; 8] = array::from_fn(|i| {
            s_g.clone() * &gy_limbs[i]
                + &(s_q.clone() * &qy_limbs[i])
                + &(s_gq.clone() * &gqy_limbs[i])
        });

        // ---- C1: Z_mid - 2·Y·Z = 0 -------------------------------------
        let yz = mul(&y, &z);
        let two = cst(2);
        let two_yz = scalar_mul(&two, &yz);
        let c1 = sub(&z_mid, &two_yz);
        for k in 0..8 {
            b.assert_zero(c1[k].clone());
        }

        // ---- C2: X_mid - (9·X^4 - 8·X·Y^2) = 0 -------------------------
        let x_sq = mul(&x, &x);
        let x_4 = mul(&x_sq, &x_sq);
        let y_sq = mul(&y, &y);
        let x_y_sq = mul(&x, &y_sq);
        let nine = cst(9);
        let eight = cst(8);
        let nine_x_4 = scalar_mul(&nine, &x_4);
        let eight_x_y_sq = scalar_mul(&eight, &x_y_sq);
        // X_mid - 9*X^4 + 8*X*Y^2
        let c2 = add(&sub(&x_mid, &nine_x_4), &eight_x_y_sq);
        for k in 0..8 {
            b.assert_zero(c2[k].clone());
        }

        // ---- C3: Y_mid - (12·X^3·Y^2 - 3·X^2·X_mid - 8·Y^4) = 0 ---------
        let x_cu = mul(&x_sq, &x);
        let y_4 = mul(&y_sq, &y_sq);
        let twelve = cst(12);
        let three = cst(3);
        let x_cu_y_sq = mul(&x_cu, &y_sq);
        let twelve_x3_y2 = scalar_mul(&twelve, &x_cu_y_sq);
        let x_sq_x_mid = mul(&x_sq, &x_mid);
        let three_x2_xmid = scalar_mul(&three, &x_sq_x_mid);
        let eight_y4 = scalar_mul(&eight, &y_4);
        // Y_mid - 12*X^3*Y^2 + 3*X^2*X_mid + 8*Y^4
        let c3 = add(
            &add(&sub(&y_mid, &twelve_x3_y2), &three_x2_xmid),
            &eight_y4,
        );
        for k in 0..8 {
            b.assert_zero(c3[k].clone());
        }

        // ---- C4: H - (T_x · Z_mid^2 - X_mid) = 0 ------------------------
        let z_mid_sq = mul(&z_mid, &z_mid);
        let t_x_z_mid_sq = mul(&t_x, &z_mid_sq);
        // H - T_x*Z_mid^2 + X_mid
        let c4 = add(&sub(&h, &t_x_z_mid_sq), &x_mid);
        for k in 0..8 {
            b.assert_zero(c4[k].clone());
        }

        // ---- W1: H_sq - H·H = 0 ----------------------------------------
        let h_h = mul(&h, &h);
        let w1 = sub(&h_sq, &h_h);
        for k in 0..8 {
            b.assert_zero(w1[k].clone());
        }

        // ---- W2: H_cu - H_sq·H = 0 -------------------------------------
        let h_sq_h = mul(&h_sq, &h);
        let w2 = sub(&h_cu, &h_sq_h);
        for k in 0..8 {
            b.assert_zero(w2[k].clone());
        }

        // ---- W3: R_a - (T_y · Z_mid^3 - Y_mid) = 0 ---------------------
        let z_mid_cu = mul(&z_mid_sq, &z_mid);
        let t_y_z_mid_cu = mul(&t_y, &z_mid_cu);
        // R_a - T_y*Z_mid^3 + Y_mid
        let w3 = add(&sub(&r_a, &t_y_z_mid_cu), &y_mid);
        for k in 0..8 {
            b.assert_zero(w3[k].clone());
        }

        // ---- C5: Z[t+1] - ((1-s)·Z_mid + s·Z_mid·H) = 0 ----------------
        let z_mid_h = mul(&z_mid, &h);
        let one_minus_s_z_mid = scalar_mul(&one_minus_s, &z_mid);
        let s_z_mid_h = scalar_mul(&s, &z_mid_h);
        let c5 = sub(&sub(&z_next, &one_minus_s_z_mid), &s_z_mid_h);
        for k in 0..8 {
            b.assert_zero(c5[k].clone());
        }

        // ---- C6: X[t+1] - ((1-s)·X_mid + s·(R_a^2 - H_cu - 2·X_mid·H_sq)) = 0
        let r_a_sq = mul(&r_a, &r_a);
        let x_mid_h_sq = mul(&x_mid, &h_sq);
        let two_x_mid_h_sq = scalar_mul(&two, &x_mid_h_sq);
        // R_a^2 - H_cu - 2*X_mid*H_sq
        let c6_addend = sub(&sub(&r_a_sq, &h_cu), &two_x_mid_h_sq);
        let one_minus_s_x_mid = scalar_mul(&one_minus_s, &x_mid);
        let s_c6_addend = scalar_mul(&s, &c6_addend);
        let c6 = sub(&sub(&x_next, &one_minus_s_x_mid), &s_c6_addend);
        for k in 0..8 {
            b.assert_zero(c6[k].clone());
        }

        // ---- C7: Y[t+1] - ((1-s)·Y_mid
        //          + s·(R_a·(X_mid·H_sq - X[t+1]) - Y_mid·H_cu)) = 0 ------
        let x_mid_h_sq_minus_x_next = sub(&x_mid_h_sq, &x_next);
        let r_a_times = mul(&r_a, &x_mid_h_sq_minus_x_next);
        let y_mid_h_cu = mul(&y_mid, &h_cu);
        let c7_addend = sub(&r_a_times, &y_mid_h_cu);
        let one_minus_s_y_mid = scalar_mul(&one_minus_s, &y_mid);
        let s_c7_addend = scalar_mul(&s, &c7_addend);
        let c7 = sub(&sub(&y_next, &one_minus_s_y_mid), &s_c7_addend);
        for k in 0..8 {
            b.assert_zero(c7[k].clone());
        }

        // ---- B3: sel_init · Z = 0 (per limb) ----------------------------
        let sel_init_z = scalar_mul(&sel_init, &z);
        for k in 0..8 {
            b.assert_zero(sel_init_z[k].clone());
        }

        // ---- B4: sel_final · Z · (X - R_SIG · Z^2) = 0 (per limb) -------
        let z_sq_b4 = mul(&z, &z);
        let r_sig_z_sq = mul(&r_sig_limbs, &z_sq_b4);
        let x_minus_r_sig_z_sq = sub(&x, &r_sig_z_sq);
        let z_times = mul(&z, &x_minus_r_sig_z_sq);
        let b4 = scalar_mul(&sel_final, &z_times);
        for k in 0..8 {
            b.assert_zero(b4[k].clone());
        }
    }
}

impl GenerateRandomTrace<32> for EcdsaUairLimbs {
    type PolyCoeff = i64;
    type Int = i64;

    /// Real Shamir's-trick walk over secp256k1, decomposed into 8 u32
    /// limbs per 256-bit value. Reuses the existing `ecdsa_secp256k1`
    /// helpers (point doubling/addition, table-point selection). The
    /// witness intentionally does NOT satisfy the constraint system —
    /// see the module-level note above.
    fn generate_random_trace<Rng: rand::RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, i64, i64, 32> {
        use ecdsa_limbs_cols::*;
        use ecdsa_secp256k1::{
            IDENTITY, Point, fp_mul, fp_sub, point_add_affine, point_double, select_table_point,
        };

        let len = 1usize << num_vars;
        assert!(
            len >= ECDSA_NUM_REAL_ROWS,
            "EcdsaUairLimbs requires num_vars >= 9 (got {num_vars}, len = {len})"
        );

        let mut u1_bytes = [0u8; 32];
        let mut u2_bytes = [0u8; 32];
        rng.fill_bytes(&mut u1_bytes);
        rng.fill_bytes(&mut u2_bytes);

        let bit_msb_first = |bytes: &[u8; 32], i: usize| -> bool {
            let byte = bytes[i / 8];
            ((byte >> (7 - (i % 8))) & 1) == 1
        };

        let mut int_cols: Vec<DenseMultilinearExtension<i64>> =
            (0..NUM_INT_COLS).map(|_| (0..len).map(|_| 0i64).collect()).collect();

        // Helper: write the 8 u32 limbs of `v` into columns
        // [base, base+8) at row `row`.
        let write_limbs = |cols: &mut [DenseMultilinearExtension<i64>],
                               base: usize,
                               row: usize,
                               v: ecdsa_secp256k1::U4| {
            let limbs = uint4_to_u32_limbs(v);
            for k in 0..8 {
                cols[base + k][row] = limbs[k] as i64;
            }
        };

        // Aux: from a (mid, t_x, t_y) compute (h, r_a, h_sq, h_cu).
        let aux_at_row = |mid: Point, t_x, t_y| {
            let z_mid_sq = fp_mul(mid.z, mid.z);
            let z_mid_cu = fp_mul(z_mid_sq, mid.z);
            let h = fp_sub(fp_mul(t_x, z_mid_sq), mid.x);
            let r_a = fp_sub(fp_mul(t_y, z_mid_cu), mid.y);
            let h_sq = fp_mul(h, h);
            let h_cu = fp_mul(h_sq, h);
            (h, r_a, h_sq, h_cu)
        };

        let mut acc: Point = IDENTITY;
        for row in 0..256usize {
            let bit1 = bit_msb_first(&u1_bytes, row);
            let bit2 = bit_msb_first(&u2_bytes, row);
            let mid = point_double(acc);
            let (t_x, t_y) = select_table_point(bit1, bit2);
            let s = bit1 || bit2;
            let (h, r_a, h_sq, h_cu) = aux_at_row(mid, t_x, t_y);
            let next = if s {
                point_add_affine(mid, t_x, t_y).0
            } else {
                mid
            };

            int_cols[B1][row] = if bit1 { 1 } else { 0 };
            int_cols[B2][row] = if bit2 { 1 } else { 0 };
            write_limbs(&mut int_cols, X, row, acc.x);
            write_limbs(&mut int_cols, Y, row, acc.y);
            write_limbs(&mut int_cols, Z, row, acc.z);
            write_limbs(&mut int_cols, X_MID, row, mid.x);
            write_limbs(&mut int_cols, Y_MID, row, mid.y);
            write_limbs(&mut int_cols, Z_MID, row, mid.z);
            write_limbs(&mut int_cols, H, row, h);
            write_limbs(&mut int_cols, R_A, row, r_a);
            write_limbs(&mut int_cols, H_SQ, row, h_sq);
            write_limbs(&mut int_cols, H_CU, row, h_cu);

            acc = next;
        }

        // Row 256: tail row holding the final accumulator.
        {
            let mid = point_double(acc);
            let (t_x, t_y) = (ecdsa_secp256k1::U4::ZERO, ecdsa_secp256k1::U4::ZERO);
            let (h, r_a, h_sq, h_cu) = aux_at_row(mid, t_x, t_y);
            write_limbs(&mut int_cols, X, 256, acc.x);
            write_limbs(&mut int_cols, Y, 256, acc.y);
            write_limbs(&mut int_cols, Z, 256, acc.z);
            write_limbs(&mut int_cols, X_MID, 256, mid.x);
            write_limbs(&mut int_cols, Y_MID, 256, mid.y);
            write_limbs(&mut int_cols, Z_MID, 256, mid.z);
            write_limbs(&mut int_cols, H, 256, h);
            write_limbs(&mut int_cols, R_A, 256, r_a);
            write_limbs(&mut int_cols, H_SQ, 256, h_sq);
            write_limbs(&mut int_cols, H_CU, 256, h_cu);
        }

        // Row 257: final boundary row.
        write_limbs(&mut int_cols, X, 257, acc.x);
        write_limbs(&mut int_cols, Y, 257, acc.y);
        write_limbs(&mut int_cols, Z, 257, acc.z);

        // Selectors.
        int_cols[SEL_INIT][0] = 1;
        int_cols[SEL_FINAL][257] = 1;

        UairTrace {
            binary_poly: vec![].into(),
            arbitrary_poly: vec![].into(),
            int: int_cols.into(),
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
        assert_uair_shape::<TestAirNoMultiplication<Int<LIMBS>>>(&[1]);
        assert_uair_shape::<TestAirScalarMultiplications<Int<LIMBS>>>(&[1]);
        assert_uair_shape::<BinaryDecompositionUair<u32>>(&[1]);
        assert_uair_shape::<BigLinearUair<u32>>(&[1; 17]);
        assert_uair_shape::<TestUairMixedShifts<Int<LIMBS>>>(&[1, 1]);
        assert_uair_shape::<XnMinusOneTestUair<Int<LIMBS>>>(&[1]);
        assert_uair_shape::<MixedIdealTestUair<Int<LIMBS>>>(&[1, 1]);

        // EcdsaUair: 12 constraints in the order they are asserted —
        // C1, C2, C3, C4, W1 (H_sq), W2 (H_cu), W3 (R_a),
        // C5, C6, C7, B3, B4. The wire columns (R_a, H_sq, H_cu) drop the
        // max constraint degree from 13 to 5.
        assert_uair_shape::<EcdsaUair>(&[2, 4, 5, 4, 2, 2, 5, 4, 4, 5, 2, 4]);

        // ShaProxyEcdsaUair: 19 constraints. The first 12 entries are the
        // ECDSA degrees (delegation order matches the impl: ECDSA first,
        // then SHAProxy), followed by SHAProxy's 7 linear constraints.
        // Filled in from the first run; adjust if the counter disagrees.
        assert_uair_shape::<ShaProxyEcdsaUair>(&[
            2, 4, 5, 4, 2, 2, 5, 4, 4, 5, 2, 4, // ECDSA (12)
            1, 1, 1, 1, 1, 1, 1, // SHAProxy (7)
        ]);

        // EcdsaUairLimbs: 96 constraints (12 logical × 8 limbs each), in
        // the same order as `EcdsaUair`. Each per-limb constraint has the
        // same degree as the corresponding `EcdsaUair` constraint, since
        // limb cells substitute for Int<4> cells one-for-one inside the
        // same algebraic shape.
        let ecdsa_limbs_expected: Vec<usize> = [
            2, 4, 5, 4, 2, 2, 5, 4, 4, 5, 2, 4,
        ]
        .iter()
        .flat_map(|d| std::iter::repeat(*d).take(8))
        .collect();
        assert_uair_shape::<EcdsaUairLimbs>(&ecdsa_limbs_expected);
    }

    #[test]
    fn ecdsa_uair_limbs_generate_random_trace_smoke() {
        let mut rng = StdRng::seed_from_u64(0xEC05A_11);
        let trace = EcdsaUairLimbs::generate_random_trace(9, &mut rng);
        assert_eq!(trace.binary_poly.len(), 0);
        assert_eq!(trace.arbitrary_poly.len(), 0);
        assert_eq!(trace.int.len(), 84);
        for col in trace.int.iter() {
            assert_eq!(col.len(), 1 << 9);
        }
    }

    #[test]
    fn sha_proxy_ecdsa_uair_generate_random_trace_smoke() {
        let mut rng = StdRng::seed_from_u64(0x5_4A_EC);
        let trace = ShaProxyEcdsaUair::generate_random_trace(9, &mut rng);
        assert_eq!(trace.binary_poly.len(), 14);
        assert_eq!(trace.arbitrary_poly.len(), 0);
        assert_eq!(trace.int.len(), 18);
        for col in trace.binary_poly.iter() {
            assert_eq!(col.len(), 1 << 9);
        }
        for col in trace.int.iter() {
            assert_eq!(col.len(), 1 << 9);
        }
    }

    #[test]
    fn ecdsa_uair_generate_random_trace_smoke() {
        // 1 << 9 = 512 ≥ 258 (the number of real ECDSA rows). The witness
        // is intentionally non-satisfying — this test only checks that the
        // walk runs without panicking and yields the right column shape.
        let mut rng = StdRng::seed_from_u64(0xEC05A);
        let trace = EcdsaUair::generate_random_trace(9, &mut rng);
        assert_eq!(trace.binary_poly.len(), 0);
        assert_eq!(trace.arbitrary_poly.len(), 0);
        assert_eq!(trace.int.len(), 14);
        for col in trace.int.iter() {
            assert_eq!(col.len(), 1 << 9);
        }
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
