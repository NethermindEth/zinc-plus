//! Limber (ePrint 2026/1635) Table 1 "Zinc+" row — the same statement on
//! the current main-beta pipeline.
//!
//! Statement: MultiSwap (ePrint 2019/1494, k = 0 swaps) ≈ 6,209 integer
//! constraints of the moduli-R1CS shape a·b = c + u·N over a 2048-bit
//! modulus. Here encoded honestly: 4 integer witness columns of 2048-bit
//! magnitude (Int<34>), one real modmul constraint per row.
//!
//! Methodology mirrors Limber §7: single-threaded unless the `parallel`
//! feature is on; medians over N runs; prover time includes commitment;
//! proof sizes raw + zstd-22.

#![allow(clippy::arithmetic_side_effects)]

use crypto_bigint::NonZero;
use crypto_primitives::{
    ConstIntRing, ConstIntSemiring, Field, FixedSemiring, FromWithConfig, PrimeField,
    crypto_bigint_int::Int, crypto_bigint_monty::MontyField, crypto_bigint_uint::Uint,
};
use rand::prelude::*;
use std::{fmt::Debug, hint::black_box, marker::PhantomData, ops::Neg, time::Instant};
use zinc_poly::{
    ConstCoeffBitWidth, Polynomial,
    mle::DenseMultilinearExtension,
    univariate::{
        binary::{BinaryPoly, BinaryPolyInnerProduct},
        dense::{DensePolyInnerProduct, DensePolynomial},
        dynamic::over_field::DynamicPolynomialF,
    },
};
use zinc_primality::{MillerRabin, PrimalityTest};
use zinc_protocol::{IntFoldedZincTypes4x, Proof, ZincPlusPiop, ZincTypes};
use zinc_test_uair::GenerateRandomTrace;
use zinc_transcript::traits::ConstTranscribable;
use zinc_uair::{
    ConstraintBuilder, PublicColumnLayout, TotalColumnLayout, TraceRow, Uair, UairSignature,
    UairTrace,
    ideal::ImpossibleIdeal,
    ideal_collector::IdealOrZero,
};
use zinc_utils::{
    field::runtime_monty::Fp,
    from_ref::FromRef,
    inner_product::{InnerProduct, MBSInnerProduct, ScalarProduct},
    mul_by_scalar::MulByScalar,
    named::Named,
};
use zip_plus::{
    code::iprs::{IprsCode, PnttConfigF65537},
    pcs::structs::{ZipPlus, ZipTypes},
    utils::eprint_proof_size,
};

const PERFORM_CHECKS: bool = if cfg!(feature = "unchecked") {
    zinc_utils::UNCHECKED
} else {
    zinc_utils::CHECKED
};

/// Inverse rate, following the repo convention: default 4, `iprs-rate-1-8`
/// switches to 8.
const REP: usize = if cfg!(feature = "iprs-rate-1-16") {
    16
} else if cfg!(feature = "iprs-rate-1-8") {
    8
} else {
    4
};

/// Openings tied to REP (repo convention: 150 @ 1/4, 100 @ 1/8, 75 @ 1/16).
const NUM_COL_OPENINGS_FOR_REP: usize = if cfg!(feature = "iprs-rate-1-16") {
    75
} else if cfg!(feature = "iprs-rate-1-8") {
    100
} else {
    150
};

//
// Generic Zip/Zinc type scaffolding — copied from `protocol/benches/e2e.rs`
// on this branch (same bounds), so the concrete instantiation below is the
// only new part.
//

#[allow(clippy::type_complexity)]
#[derive(Debug, Clone, Copy)]
pub struct GenericBenchZipTypes<
    Eval,
    Cw,
    Fmod,
    PrimeTest,
    Chal,
    Pt,
    CombR,
    Comb,
    EvalDotChal,
    CombDotChal,
    ArrCombRDotChal,
>(
    PhantomData<(
        Eval,
        Cw,
        Fmod,
        PrimeTest,
        Chal,
        Pt,
        CombR,
        Comb,
        EvalDotChal,
        CombDotChal,
        ArrCombRDotChal,
    )>,
);

impl<Eval, Cw, Fmod, PrimeTest, Chal, Pt, CombR, Comb, EvalDotChal, CombDotChal, ArrCombRDotChal>
    ZipTypes
    for GenericBenchZipTypes<
        Eval,
        Cw,
        Fmod,
        PrimeTest,
        Chal,
        Pt,
        CombR,
        Comb,
        EvalDotChal,
        CombDotChal,
        ArrCombRDotChal,
    >
where
    Eval: ConstCoeffBitWidth + Default + Named + Clone + Debug + Send + Sync,
    Cw: FixedSemiring + ConstCoeffBitWidth + ConstTranscribable + FromRef<Eval> + Named + Copy,
    Fmod: ConstIntSemiring + ConstTranscribable + Named,
    PrimeTest: PrimalityTest<Fmod> + Send + Sync,
    Chal: ConstIntRing + ConstTranscribable + Named,
    Pt: ConstIntRing,
    CombR: ConstIntRing
        + Neg<Output = CombR>
        + ConstTranscribable
        + FromRef<CombR>
        + for<'a> MulByScalar<&'a Chal>,
    Comb: FixedSemiring + Polynomial<CombR> + FromRef<Eval> + FromRef<Cw> + Named,
    EvalDotChal: InnerProduct<Eval, Chal, CombR> + Clone + Debug + Send + Sync,
    CombDotChal: InnerProduct<Comb, Chal, CombR> + Clone + Debug + Send + Sync,
    ArrCombRDotChal: InnerProduct<[CombR], Chal, CombR> + Clone + Debug + Send + Sync,
{
    const NUM_COLUMN_OPENINGS: usize = NUM_COL_OPENINGS_FOR_REP;
    type Eval = Eval;
    type Cw = Cw;
    type Fmod = Fmod;
    type PrimeTest = PrimeTest;
    type Chal = Chal;
    type Pt = Pt;
    type CombR = CombR;
    type Comb = Comb;
    type EvalDotChal = EvalDotChal;
    type CombDotChal = CombDotChal;
    type ArrCombRDotChal = ArrCombRDotChal;
}

#[derive(Clone, Debug)]
struct GenericBenchZincTypes<
    Int,
    CwR,
    Chal,
    Pt,
    BinaryCombR,
    CombR,
    IntCombR,
    Fmod,
    PrimeTest,
    const D: usize,
>(
    PhantomData<(
        Int,
        CwR,
        Chal,
        Pt,
        BinaryCombR,
        CombR,
        IntCombR,
        Fmod,
        PrimeTest,
    )>,
);

impl<Int, CwR, Chal, Pt, BinaryCombR, CombR, IntCombR, Fmod, PrimeTest, const D: usize>
    ZincTypes<D>
    for GenericBenchZincTypes<
        Int,
        CwR,
        Chal,
        Pt,
        BinaryCombR,
        CombR,
        IntCombR,
        Fmod,
        PrimeTest,
        D,
    >
where
    Int: ConstIntSemiring
        + for<'a> MulByScalar<&'a i64, CwR>
        + Named
        + ConstCoeffBitWidth
        + ConstTranscribable
        + Default
        + Clone
        + Send
        + Sync
        + 'static,
    CwR: FixedSemiring
        + for<'a> MulByScalar<&'a i64>
        + ConstCoeffBitWidth
        + ConstTranscribable
        + Named
        + FromRef<Int>
        + FromRef<CwR>
        + Copy,
    Chal: ConstIntRing + ConstTranscribable + Named,
    Pt: ConstIntRing,
    BinaryCombR: ConstIntRing
        + Polynomial<BinaryCombR>
        + Neg<Output = BinaryCombR>
        + for<'a> MulByScalar<&'a i64>
        + for<'a> MulByScalar<&'a Chal>
        + ConstTranscribable
        + Named
        + FromRef<i64>
        + FromRef<Int>
        + FromRef<CwR>
        + FromRef<Chal>
        + FromRef<BinaryCombR>,
    CombR: ConstIntRing
        + Polynomial<CombR>
        + Neg<Output = CombR>
        + for<'a> MulByScalar<&'a i64>
        + for<'a> MulByScalar<&'a Chal>
        + ConstTranscribable
        + Named
        + FromRef<i64>
        + FromRef<Int>
        + FromRef<CwR>
        + FromRef<Chal>
        + FromRef<CombR>,
    IntCombR: ConstIntRing
        + Polynomial<IntCombR>
        + Neg<Output = IntCombR>
        + for<'a> MulByScalar<&'a i64>
        + for<'a> MulByScalar<&'a Chal>
        + ConstTranscribable
        + Named
        + FromRef<i64>
        + FromRef<Int>
        + FromRef<CwR>
        + FromRef<Chal>
        + FromRef<IntCombR>,
    Fmod: ConstIntSemiring + ConstTranscribable + Named,
    PrimeTest: PrimalityTest<Fmod> + Debug + Send + Sync,
{
    type Int = Int;
    type Chal = Chal;
    type Pt = Pt;
    type Fmod = Fmod;
    type PrimeTest = PrimeTest;

    type BinaryZt = GenericBenchZipTypes<
        BinaryPoly<D>,
        DensePolynomial<i64, D>,
        Fmod,
        PrimeTest,
        Chal,
        Pt,
        BinaryCombR,
        DensePolynomial<BinaryCombR, D>,
        BinaryPolyInnerProduct<Chal, D>,
        DensePolyInnerProduct<BinaryCombR, Chal, BinaryCombR, MBSInnerProduct, D>,
        MBSInnerProduct,
    >;
    type ArbitraryZt = GenericBenchZipTypes<
        DensePolynomial<Int, D>,
        DensePolynomial<CwR, D>,
        Fmod,
        PrimeTest,
        Chal,
        Pt,
        CombR,
        DensePolynomial<CombR, D>,
        DensePolyInnerProduct<Int, Chal, CombR, MBSInnerProduct, D>,
        DensePolyInnerProduct<CombR, Chal, CombR, MBSInnerProduct, D>,
        MBSInnerProduct,
    >;
    type IntZt = GenericBenchZipTypes<
        Int,
        CwR,
        Fmod,
        PrimeTest,
        Chal,
        Pt,
        IntCombR,
        IntCombR,
        ScalarProduct,
        ScalarProduct,
        MBSInnerProduct,
    >;

    type BinaryLc = IprsCode<Self::BinaryZt, PnttConfigF65537, REP, PERFORM_CHECKS>;
    type ArbitraryLc = IprsCode<Self::ArbitraryZt, PnttConfigF65537, REP, PERFORM_CHECKS>;
    type IntLc = IprsCode<Self::IntZt, PnttConfigF65537, REP, PERFORM_CHECKS>;
}

//
// Concrete instantiation: 2048-bit integer witness cells.
//

const DEGREE_PLUS_ONE: usize = 32;
const FIELD_LIMBS: usize = 4;

zinc_utils::define_modulus!(LimberBenchSlot, FIELD_LIMBS);
type F = Fp<LimberBenchSlot, FIELD_LIMBS>;

const BIG: usize = 34; // Eval limbs: 2176-bit type for 2048-bit values
const BIG_CW: usize = 38; // codeword limbs
const BIG_M: usize = 44; // combination-ring limbs

type RsaInt = Int<BIG>;

type RsaZincTypes = GenericBenchZincTypes<
    /* Int         = */ RsaInt,
    /* CwR         = */ Int<BIG_CW>,
    /* Chal        = */ i128,
    /* Pt          = */ i128,
    /* BinaryCombR = */ Int<5>,
    /* CombR       = */ Int<BIG_M>,
    /* IntCombR    = */ Int<BIG_M>,
    /* Fmod        = */ Uint<FIELD_LIMBS>,
    MillerRabin,
    DEGREE_PLUS_ONE,
>;

//
// The UAIR: 4 int columns (a, b, c, u); a·b − c − u·N = 0 over Z per row.
//

/// Fixed 2048-bit modulus N = 2^2048 − 59 (stands in for the RSA-2048
/// modulus; the factorization is irrelevant to proving cost).
fn modulus_raw() -> crypto_bigint::Uint<32> {
    let hex = format!("{}C5", "F".repeat(510));
    crypto_bigint::Uint::<32>::from_be_hex(&hex)
}

/// Reinterpret a non-negative raw Uint (top bits clear) as the wrapper Int.
fn int34(v: &crypto_bigint::Uint<64>) -> RsaInt {
    *Uint::<BIG>::new(v.resize::<BIG>()).as_int()
}

fn n_scalar() -> DensePolynomial<RsaInt, 32> {
    let mut coeffs = [<RsaInt as num_traits::ConstZero>::ZERO; 32];
    coeffs[0] = int34(&modulus_raw().resize::<64>());
    DensePolynomial::new(coeffs)
}

#[derive(Clone, Debug)]
pub struct RsaModMulUair;

impl Uair for RsaModMulUair {
    type Ideal = ImpossibleIdeal;
    type Scalar = DensePolynomial<RsaInt, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(0, 0, 4);
        UairSignature::new(total, PublicColumnLayout::default(), vec![], vec![], vec![])
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        _down: TraceRow<B::Expr>,
        _from_ref: FromR,
        mbs: MulByScalar,
        _ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&Self::Scalar) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &Self::Scalar) -> Option<B::Expr>,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
    {
        let n = n_scalar();
        // a·b − c − u·N = 0 over Z.
        let ab = up.int[0].clone() * &up.int[1];
        let un = mbs(&up.int[3], &n).expect("u·N overflow");
        b.assert_zero(ab - &up.int[2] - &un);
    }
}

impl GenerateRandomTrace<DEGREE_PLUS_ONE> for RsaModMulUair {
    type PolyCoeff = RsaInt;
    type Int = RsaInt;

    fn generate_random_trace<Rng: rand::RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, RsaInt, RsaInt, DEGREE_PLUS_ONE> {
        let len = 1usize << num_vars;
        let n32 = modulus_raw();
        let nz32: NonZero<crypto_bigint::Uint<32>> =
            Option::from(NonZero::new(n32)).expect("N != 0");
        let n64 = n32.resize::<64>();
        let nz64: NonZero<crypto_bigint::Uint<64>> =
            Option::from(NonZero::new(n64)).expect("N != 0");

        let mut cols: [Vec<RsaInt>; 4] = core::array::from_fn(|_| Vec::with_capacity(len));

        for _ in 0..len {
            let mut rand32 = |rng: &mut Rng| -> crypto_bigint::Uint<32> {
                crypto_bigint::Uint::<32>::from_words(core::array::from_fn(|_| rng.next_u64()))
            };
            let a = rand32(rng).div_rem(&nz32).1;
            let b = rand32(rng).div_rem(&nz32).1;
            let a64 = a.resize::<64>();
            let b64 = b.resize::<64>();
            let (lo, hi) = a64.widening_mul(&b64);
            debug_assert_eq!(hi, crypto_bigint::Uint::<64>::ZERO);
            let (u, c) = lo.div_rem(&nz64);

            cols[0].push(int34(&a64));
            cols[1].push(int34(&b64));
            cols[2].push(int34(&c.resize::<64>()));
            cols[3].push(int34(&u));
        }

        UairTrace {
            int: cols
                .into_iter()
                .map(|v| v.into_iter().collect::<DenseMultilinearExtension<_>>())
                .collect::<Vec<_>>()
                .into(),
            ..Default::default()
        }
    }
}

/// Wide variant: 8 int columns = 2 modmul constraints per row, so 4096 rows
/// carry 8192 constraint slots (>= 6,209 like Limber's circuit).
#[derive(Clone, Debug)]
pub struct RsaModMulWideUair;

impl Uair for RsaModMulWideUair {
    type Ideal = ImpossibleIdeal;
    type Scalar = DensePolynomial<RsaInt, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(0, 0, 8);
        UairSignature::new(total, PublicColumnLayout::default(), vec![], vec![], vec![])
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        _down: TraceRow<B::Expr>,
        _from_ref: FromR,
        mbs: MulByScalar,
        _ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&Self::Scalar) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &Self::Scalar) -> Option<B::Expr>,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
    {
        let n = n_scalar();
        for k in [0usize, 4] {
            let ab = up.int[k].clone() * &up.int[k + 1];
            let un = mbs(&up.int[k + 3], &n).expect("u·N overflow");
            b.assert_zero(ab - &up.int[k + 2] - &un);
        }
    }
}

impl GenerateRandomTrace<DEGREE_PLUS_ONE> for RsaModMulWideUair {
    type PolyCoeff = RsaInt;
    type Int = RsaInt;

    fn generate_random_trace<Rng: rand::RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, RsaInt, RsaInt, DEGREE_PLUS_ONE> {
        let len = 1usize << num_vars;
        let n32 = modulus_raw();
        let nz32: NonZero<crypto_bigint::Uint<32>> =
            Option::from(NonZero::new(n32)).expect("N != 0");
        let n64 = n32.resize::<64>();
        let nz64: NonZero<crypto_bigint::Uint<64>> =
            Option::from(NonZero::new(n64)).expect("N != 0");

        let mut cols: [Vec<RsaInt>; 8] = core::array::from_fn(|_| Vec::with_capacity(len));

        for _ in 0..len {
            for k in [0usize, 4] {
                let mut rand32 = |rng: &mut Rng| -> crypto_bigint::Uint<32> {
                    crypto_bigint::Uint::<32>::from_words(core::array::from_fn(|_| {
                        rng.next_u64()
                    }))
                };
                let a = rand32(rng).div_rem(&nz32).1;
                let b = rand32(rng).div_rem(&nz32).1;
                let a64 = a.resize::<64>();
                let b64 = b.resize::<64>();
                let (lo, _hi) = a64.widening_mul(&b64);
                let (u, c) = lo.div_rem(&nz64);

                cols[k].push(int34(&a64));
                cols[k + 1].push(int34(&b64));
                cols[k + 2].push(int34(&c.resize::<64>()));
                cols[k + 3].push(int34(&u));
            }
        }

        UairTrace {
            int: cols
                .into_iter()
                .map(|v| v.into_iter().collect::<DenseMultilinearExtension<_>>())
                .collect::<Vec<_>>()
                .into(),
            ..Default::default()
        }
    }
}

/// Wide-16 variant: 16 int columns = 4 modmul constraints per row, so
/// 2048 rows (nvars = 11) carry 8192 constraint slots — the same
/// statement as wide8 @ nvars = 12 in a shape whose folded-4× row
/// length (4·2048 = 8192) still fits the F65537 NTT cap at rate 1/8.
#[derive(Clone, Debug)]
pub struct RsaModMulWide16Uair;

impl Uair for RsaModMulWide16Uair {
    type Ideal = ImpossibleIdeal;
    type Scalar = DensePolynomial<RsaInt, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(0, 0, 16);
        UairSignature::new(total, PublicColumnLayout::default(), vec![], vec![], vec![])
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        _down: TraceRow<B::Expr>,
        _from_ref: FromR,
        mbs: MulByScalar,
        _ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&Self::Scalar) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &Self::Scalar) -> Option<B::Expr>,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
    {
        let n = n_scalar();
        for k in [0usize, 4, 8, 12] {
            let ab = up.int[k].clone() * &up.int[k + 1];
            let un = mbs(&up.int[k + 3], &n).expect("u·N overflow");
            b.assert_zero(ab - &up.int[k + 2] - &un);
        }
    }
}

impl GenerateRandomTrace<DEGREE_PLUS_ONE> for RsaModMulWide16Uair {
    type PolyCoeff = RsaInt;
    type Int = RsaInt;

    fn generate_random_trace<Rng: rand::RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, RsaInt, RsaInt, DEGREE_PLUS_ONE> {
        let len = 1usize << num_vars;
        let n32 = modulus_raw();
        let nz32: NonZero<crypto_bigint::Uint<32>> =
            Option::from(NonZero::new(n32)).expect("N != 0");
        let n64 = n32.resize::<64>();
        let nz64: NonZero<crypto_bigint::Uint<64>> =
            Option::from(NonZero::new(n64)).expect("N != 0");

        let mut cols: [Vec<RsaInt>; 16] = core::array::from_fn(|_| Vec::with_capacity(len));

        for _ in 0..len {
            for k in [0usize, 4, 8, 12] {
                let mut rand32 = |rng: &mut Rng| -> crypto_bigint::Uint<32> {
                    crypto_bigint::Uint::<32>::from_words(core::array::from_fn(|_| {
                        rng.next_u64()
                    }))
                };
                let a = rand32(rng).div_rem(&nz32).1;
                let b = rand32(rng).div_rem(&nz32).1;
                let a64 = a.resize::<64>();
                let b64 = b.resize::<64>();
                let (lo, _hi) = a64.widening_mul(&b64);
                let (u, c) = lo.div_rem(&nz64);

                cols[k].push(int34(&a64));
                cols[k + 1].push(int34(&b64));
                cols[k + 2].push(int34(&c.resize::<64>()));
                cols[k + 3].push(int34(&u));
            }
        }

        UairTrace {
            int: cols
                .into_iter()
                .map(|v| v.into_iter().collect::<DenseMultilinearExtension<_>>())
                .collect::<Vec<_>>()
                .into(),
            ..Default::default()
        }
    }
}

//
// Folded-4× instantiation: Int<34> cells quartered at radix 2^512
// (generalized from the 2^64 ECDSA quartering) into Int<9> quarters
// (512 magnitude bits + one sign-headroom limb).
//

const HALF_DEGREE_PLUS_ONE: usize = DEGREE_PLUS_ONE / 2; // 16
const QUARTER_DEGREE_PLUS_ONE: usize = DEGREE_PLUS_ONE / 4; // 8
const BIG_QUARTER: usize = 9; // quarter Eval limbs: 512-bit + headroom
const BIG_QUARTER_CW: usize = 14; // quarter codeword limbs (mirrors BIG→BIG_CW growth)
const BIG_QUARTER_M: usize = 18; // quarter combination-ring limbs

type RsaQuarterIntZt = GenericBenchZipTypes<
    Int<BIG_QUARTER>,
    Int<BIG_QUARTER_CW>,
    Uint<FIELD_LIMBS>,
    MillerRabin,
    i128,
    i128,
    Int<BIG_QUARTER_M>,
    Int<BIG_QUARTER_M>,
    ScalarProduct,
    ScalarProduct,
    MBSInnerProduct,
>;

#[derive(Clone, Debug)]
struct RsaFolded4xZincTypes;

impl IntFoldedZincTypes4x<DEGREE_PLUS_ONE, QUARTER_DEGREE_PLUS_ONE, BIG, BIG_QUARTER>
    for RsaFolded4xZincTypes
{
    type Chal = i128;
    type Pt = i128;
    type Fmod = Uint<FIELD_LIMBS>;
    type PrimeTest = MillerRabin;

    type BinaryZt = GenericBenchZipTypes<
        BinaryPoly<QUARTER_DEGREE_PLUS_ONE>,
        DensePolynomial<i64, QUARTER_DEGREE_PLUS_ONE>,
        Self::Fmod,
        Self::PrimeTest,
        Self::Chal,
        Self::Pt,
        Int<5>,
        DensePolynomial<Int<5>, QUARTER_DEGREE_PLUS_ONE>,
        BinaryPolyInnerProduct<Self::Chal, QUARTER_DEGREE_PLUS_ONE>,
        DensePolyInnerProduct<
            Int<5>,
            Self::Chal,
            Int<5>,
            MBSInnerProduct,
            QUARTER_DEGREE_PLUS_ONE,
        >,
        MBSInnerProduct,
    >;
    type ArbitraryZt = <RsaZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::ArbitraryZt;
    type IntZt = RsaQuarterIntZt;

    type BinaryLc = IprsCode<Self::BinaryZt, PnttConfigF65537, REP, PERFORM_CHECKS>;
    type ArbitraryLc = <RsaZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::ArbitraryLc;
    type IntLc = IprsCode<Self::IntZt, PnttConfigF65537, REP, PERFORM_CHECKS>;
}

type F4xBinaryZt = <RsaFolded4xZincTypes as IntFoldedZincTypes4x<
    DEGREE_PLUS_ONE,
    QUARTER_DEGREE_PLUS_ONE,
    BIG,
    BIG_QUARTER,
>>::BinaryZt;
type F4xArbitraryZt = <RsaFolded4xZincTypes as IntFoldedZincTypes4x<
    DEGREE_PLUS_ONE,
    QUARTER_DEGREE_PLUS_ONE,
    BIG,
    BIG_QUARTER,
>>::ArbitraryZt;
type F4xIntZt = <RsaFolded4xZincTypes as IntFoldedZincTypes4x<
    DEGREE_PLUS_ONE,
    QUARTER_DEGREE_PLUS_ONE,
    BIG,
    BIG_QUARTER,
>>::IntZt;

//
// Harness
//

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(f64::total_cmp);
    xs[xs.len() / 2]
}

fn ncols_label(n: usize) -> &'static str {
    match n {
        8 => "wide8",
        16 => "wide16",
        _ => "",
    }
}

macro_rules! run_case_for {
    ($ufn:ident, $uty:ty, $ncols:literal) => {
#[allow(clippy::unwrap_used)]
fn $ufn(num_vars: usize, reps: usize) {
    type U = $uty;
    type Zt = RsaZincTypes;
    macro_rules! piop {
        () => {
            ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>
        };
    }

    let mut rng = rand::rng();
    let t0 = Instant::now();
    let trace = U::generate_random_trace(num_vars, &mut rng);
    let witness_gen_s = t0.elapsed().as_secs_f64();

    let poly_size = 1usize << num_vars;
    // Optional rectangular Zip+ geometry (ROWLEN = matrix row length for
    // all three families; default flat, row_len = poly_size).
    let row_len: usize = std::env::var("ROWLEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(poly_size);
    let pp = (
        ZipPlus::<<Zt as ZincTypes<DEGREE_PLUS_ONE>>::BinaryZt, _>::setup(
            row_len,
            IprsCode::new_with_optimal_depth(row_len).unwrap(),
        ),
        ZipPlus::<<Zt as ZincTypes<DEGREE_PLUS_ONE>>::ArbitraryZt, _>::setup(
            row_len,
            IprsCode::new_with_optimal_depth(row_len).unwrap(),
        ),
        ZipPlus::<<Zt as ZincTypes<DEGREE_PLUS_ONE>>::IntZt, _>::setup(
            row_len,
            IprsCode::new_with_optimal_depth(row_len).unwrap(),
        ),
    );

    let proof: Proof<F> =
        <piop!()>::prove::<false, PERFORM_CHECKS>(&pp, &trace, num_vars, zinc_protocol::project_scalar_fn)
            .expect("prove failed");

    let mut prove_times = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        let p: Proof<F> = <piop!()>::prove::<false, PERFORM_CHECKS>(
            &pp,
            &trace,
            num_vars,
            zinc_protocol::project_scalar_fn,
        )
        .expect("prove failed");
        prove_times.push(t0.elapsed().as_secs_f64());
        black_box(p);
    }

    let sig = U::signature();
    let public_trace = trace.public(&sig);

    let proj_ideal = |_: &IdealOrZero<<U as Uair>::Ideal>,
                      _: &<F as PrimeField>::Config|
     -> ImpossibleIdeal { unreachable!("only assert_zero constraints") };

    let verify_once = |p: Proof<F>| {
        <piop!()>::verify::<_, PERFORM_CHECKS>(
            &pp,
            p,
            &public_trace,
            num_vars,
            zinc_protocol::project_scalar_fn,
            proj_ideal,
        )
    };

    match verify_once(proof.clone()) {
        Ok(()) => {
            let mut verify_times = Vec::with_capacity(reps);
            for _ in 0..reps {
                let p = proof.clone();
                let t0 = Instant::now();
                verify_once(p).expect("verify failed");
                verify_times.push(t0.elapsed().as_secs_f64());
            }
            println!(
                "RsaModMul2048[main-beta]{}/rate1-{REP} nvars={num_vars} ({} rows x {} cols, row_len={row_len}): witness-gen {:.3} s | prove median {:.3} s | verify median {:.4} s ({} reps)",
                ncols_label($ncols),
                1usize << num_vars,
                $ncols,
                witness_gen_s,
                median(prove_times),
                median(verify_times),
                reps,
            );
        }
        Err(e) => {
            println!(
                "RsaModMul2048[main-beta]{}/rate1-{REP} nvars={num_vars} ({} rows x {} cols, row_len={row_len}): witness-gen {:.3} s | prove median {:.3} s | VERIFY FAILED: {e:?} ({} reps)",
                ncols_label($ncols),
                1usize << num_vars,
                $ncols,
                witness_gen_s,
                median(prove_times),
                reps,
            );
        }
    }

    eprint_proof_size(
        format!(
            "RsaModMul2048[main-beta]{}/rate1-{REP}/nvars={num_vars}/row_len={row_len}",
            ncols_label($ncols)
        ),
        &proof,
    );
}
    };
}

run_case_for!(run_case, RsaModMulUair, 4);
run_case_for!(run_case_wide8, RsaModMulWideUair, 8);
run_case_for!(run_case_wide16, RsaModMulWide16Uair, 16);

macro_rules! run_case_folded_for {
    ($ufn:ident, $uty:ty, $ncols:literal) => {
#[allow(clippy::unwrap_used)]
fn $ufn(num_vars: usize, reps: usize) {
    type U = $uty;

    let mut rng = rand::rng();
    let t0 = Instant::now();
    let trace = U::generate_random_trace(num_vars, &mut rng);
    let witness_gen_s = t0.elapsed().as_secs_f64();

    // Folded-4× geometry: binary and int lanes commit at 4n.
    let split4_size = 1usize << (num_vars + 2);
    let normal_size = 1usize << num_vars;
    let pp = (
        ZipPlus::<F4xBinaryZt, _>::setup(
            split4_size,
            IprsCode::new_with_optimal_depth(split4_size).unwrap(),
        ),
        ZipPlus::<F4xArbitraryZt, _>::setup(
            normal_size,
            IprsCode::new_with_optimal_depth(normal_size).unwrap(),
        ),
        ZipPlus::<F4xIntZt, _>::setup(
            split4_size,
            IprsCode::new_with_optimal_depth(split4_size).unwrap(),
        ),
    );

    let do_prove = || {
        zinc_protocol::prover::prove_folded_4x::<
            RsaFolded4xZincTypes,
            U,
            F,
            DEGREE_PLUS_ONE,
            HALF_DEGREE_PLUS_ONE,
            QUARTER_DEGREE_PLUS_ONE,
            BIG,
            BIG_QUARTER,
            false,
            PERFORM_CHECKS,
        >(&pp, &trace, num_vars, zinc_protocol::project_scalar_fn)
    };

    let proof: Proof<F> = do_prove().expect("folded prove failed");

    let mut prove_times = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        let p: Proof<F> = do_prove().expect("folded prove failed");
        prove_times.push(t0.elapsed().as_secs_f64());
        black_box(p);
    }

    let sig = U::signature();
    let public_trace = trace.public(&sig);

    let proj_ideal = |_: &IdealOrZero<<U as Uair>::Ideal>,
                      _: &<F as PrimeField>::Config|
     -> ImpossibleIdeal { unreachable!("only assert_zero constraints") };

    let verify_once = |p: Proof<F>| {
        zinc_protocol::verifier::verify_folded_4x::<
            RsaFolded4xZincTypes,
            U,
            F,
            ImpossibleIdeal,
            DEGREE_PLUS_ONE,
            HALF_DEGREE_PLUS_ONE,
            QUARTER_DEGREE_PLUS_ONE,
            BIG,
            BIG_QUARTER,
            PERFORM_CHECKS,
        >(
            &pp,
            p,
            &public_trace,
            num_vars,
            zinc_protocol::project_scalar_fn,
            proj_ideal,
        )
    };

    match verify_once(proof.clone()) {
        Ok(()) => {
            let mut verify_times = Vec::with_capacity(reps);
            for _ in 0..reps {
                let p = proof.clone();
                let t0 = Instant::now();
                verify_once(p).expect("verify failed");
                verify_times.push(t0.elapsed().as_secs_f64());
            }
            println!(
                "RsaModMul2048[main-beta]{}/folded4x/rate1-{REP} nvars={num_vars} ({} rows x {} cols, split4_row_len={split4_size}): witness-gen {:.3} s | prove median {:.3} s | verify median {:.4} s ({} reps)",
                ncols_label($ncols),
                1usize << num_vars,
                $ncols,
                witness_gen_s,
                median(prove_times),
                median(verify_times),
                reps,
            );
        }
        Err(e) => {
            println!(
                "RsaModMul2048[main-beta]{}/folded4x/rate1-{REP} nvars={num_vars} ({} rows x {} cols, split4_row_len={split4_size}): witness-gen {:.3} s | prove median {:.3} s | VERIFY FAILED: {e:?} ({} reps)",
                ncols_label($ncols),
                1usize << num_vars,
                $ncols,
                witness_gen_s,
                median(prove_times),
                reps,
            );
        }
    }

    eprint_proof_size(
        format!(
            "RsaModMul2048[main-beta]{}/folded4x/rate1-{REP}/nvars={num_vars}/split4_row_len={split4_size}",
            ncols_label($ncols)
        ),
        &proof,
    );
}
    };
}

run_case_folded_for!(run_case_folded, RsaModMulUair, 4);
run_case_folded_for!(run_case_folded_wide8, RsaModMulWideUair, 8);
run_case_folded_for!(run_case_folded_wide16, RsaModMulWide16Uair, 16);

fn main() {
    let reps: usize = std::env::var("REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let nvars_list: Vec<usize> = std::env::var("NVARS")
        .ok()
        .map(|s| s.split(',').map(|x| x.parse().expect("bad NVARS")).collect())
        .unwrap_or_else(|| vec![12]);

    eprintln!(
        "single-threaded={} checks={} rate=1/{} openings={} (features: parallel={}, unchecked={})",
        cfg!(not(feature = "parallel")),
        PERFORM_CHECKS,
        REP,
        NUM_COL_OPENINGS_FOR_REP,
        cfg!(feature = "parallel"),
        cfg!(feature = "unchecked"),
    );
    let wide8 = std::env::var("WIDE8").is_ok();
    let wide16 = std::env::var("WIDE16").is_ok();
    let folded = std::env::var("FOLD").is_ok();
    for nv in nvars_list {
        match (folded, wide16, wide8) {
            (true, true, _) => run_case_folded_wide16(nv, reps),
            (true, false, true) => run_case_folded_wide8(nv, reps),
            (true, false, false) => run_case_folded(nv, reps),
            (false, true, _) => run_case_wide16(nv, reps),
            (false, false, true) => run_case_wide8(nv, reps),
            (false, false, false) => run_case(nv, reps),
        }
    }
}
