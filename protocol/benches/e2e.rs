#![allow(clippy::arithmetic_side_effects)]

use ark_ec::{AffineRepr, CurveGroup, PrimeGroup};
use ark_ff::{MontBackend, MontConfig};
use criterion::{
    BatchSize, BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main,
    measurement::WallTime,
};
use crypto_bigint::U64;
use crypto_primitives::{
    ConstIntRing, ConstIntSemiring, ConstSemiring, Field, FixedSemiring, FromPrimitiveWithConfig,
    FromWithConfig, IntRing, PrimeField, ark_ff_fp::Fp as ArkFp, crypto_bigint_int::Int,
    crypto_bigint_monty::MontyField, crypto_bigint_uint::Uint,
};
use num_traits::Zero;
use rand::rng;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::{
    borrow::Cow,
    collections::HashMap,
    fmt::{Debug, Write as _},
    fs,
    hint::black_box,
    marker::PhantomData,
    ops::Neg,
    path::Path,
    sync::{Arc, Mutex, OnceLock},
    time::Instant,
};
use tracing::{
    Id, Subscriber,
    field::{Field as TracingField, Visit},
};
use tracing_subscriber::{layer::Context, prelude::*, registry::LookupSpan};
use zinc_piop::neutron_nova::{
    MleTable, PreparedProductionShaNativeView, ProjectedPublic, ProjectedTrace, SHA_ROW_COUNT,
    SHA_ROW_VARS, SHA_WORD_BITS, ShaBinaryFoldField, ShaBooleanityCatalog, ShaIntCol,
    ShaLinearAccumulatorField, ShaPublicCol, ShaPublicWordCol, ShaSmallFieldDecode,
    ShaSuffixScannerField, ShaWordCol, bit_slice_index,
};
use zinc_poly::mle::DenseMultilinearExtension;
use zinc_poly::univariate::dynamic::over_field::DynamicPolyVecF;
use zinc_poly::{
    ConstCoeffBitWidth, Polynomial,
    univariate::{
        binary::{BinaryPoly, BinaryPolyInnerProduct},
        dense::{DensePolyInnerProduct, DensePolynomial},
        dynamic::over_field::DynamicPolynomialF,
    },
};
use zinc_primality::{MillerRabin, PrimalityTest};
use zinc_protocol::{
    FoldedZincTypes, IntFoldedZincTypes4x, Proof, ZincPlusPiop, ZincTypes,
    fixed_prime::field_cfg_from_curve_scalar,
    pcs::{
        AllHyraxPCSTypes, AllZipPCSTypes, BinaryIntHyraxZipArbitrary, PCSCommitments, PCSParams,
        PCSVerifierParams, ZincPCSTypes,
    },
    production_sha::{
        LinearIdealFoldProverParams, LinearIdealFoldVerifierParams, PACKED_SHA_VALUES_PER_INSTANCE,
        PreparedProductionShaProverInstance, ProductionShaError, ProductionShaMixedHyraxPcs,
        ProductionShaMixedHyraxProof, ProductionShaPackedHyraxProof,
        ProductionShaProjectionAdapter, ProductionShaWitnessPolys, UairShape,
        VerifiedLinearIdealFoldSetup, decide_fold_first_mixed_hyrax,
        fold_prepared_fold_first_mixed_hyrax, packed_sha_layout,
        prepare_fold_first_linear_ideal_fold_witnesses, prepare_linear_ideal_fold_witnesses,
        prove_prepared_fold_first_mixed_hyrax, prove_prepared_linear_ideal_fold_mixed_hyrax,
        prove_prepared_linear_ideal_fold_packed_hyrax, setup_verify_linear_ideal_fold_mixed_hyrax,
        setup_verify_linear_ideal_fold_packed_hyrax,
        verify_fold_first_linear_ideal_fold_mixed_hyrax, verify_linear_ideal_fold_mixed_hyrax,
        verify_linear_ideal_fold_packed_hyrax,
    },
};
use zinc_test_uair::{
    BigLinearUair, BigLinearUairWithPublicInput, BinaryDecompositionUair, EC_FP_INT_LIMBS,
    EcdsaUair, GenerateRandomTrace, SHA256_INITIAL_STATE, Sha256CompressionSliceUair, Sha256Ideal,
    Sha256MessageBlock, ShaEcdsaUair, ShaProxy, TestUairNoMultiplication,
    sha256::{K_CANONICAL, cols as sha256_cols},
    sha256_padded_message_blocks, synthesize_sha256_chain_trace, synthesize_sha256_chain_witnesses,
};
use zinc_transcript::{
    Blake3Transcript,
    traits::{ConstTranscribable, Transcribable},
};
use zinc_uair::{
    ConstraintBuilder, PublicStructureError, TraceRow, Uair, UairSignature, UairTrace,
    degree_counter::count_effective_max_degree,
    ideal::{DegreeOneIdeal, Ideal, IdealCheck, ImpossibleIdeal, rotation::RotationIdeal},
    ideal_collector::IdealOrZero,
};
use zinc_utils::{
    cfg_into_iter, cfg_iter,
    delayed_reduction::{
        BarrettDelayedReduction, DelayedFieldProductSum, DelayedModularReductionAlgorithm,
        MontgomeryLimbs,
    },
    from_ref::FromRef,
    inner_product::{InnerProduct, MBSInnerProduct, ScalarProduct},
    inner_transparent_field::InnerTransparentField,
    mul_by_scalar::MulByScalar,
    named::Named,
    projectable_to_field::ProjectableToField,
};
use zip_plus::{
    code::{
        LinearCode,
        iprs::{IprsCode, PnttConfigF65537},
    },
    pcs::generic::{PCS, ZipPlusPCS},
    pcs::hyrax::{
        BinaryLanes, DensePolyScalarLanes, HyraxBlindingMode, HyraxPCS, IntScalarLane,
        ScalarFieldLane,
    },
    pcs::structs::{ZipPlus, ZipPlusCommitment, ZipPlusParams, ZipTypes},
    utils::{eprint_bytes_size, eprint_bytes_size_breakdown, eprint_proof_size},
};

//
// Type definitions and constants
//

const PERFORM_CHECKS: bool = if cfg!(feature = "unchecked") {
    zinc_utils::UNCHECKED
} else {
    zinc_utils::CHECKED
};

/// Repetition factor for linear code, an inverse rate. Defaults to 4 (rate
/// 1/4); enabling the `iprs-rate-1-8` cargo feature switches every IPRS
/// instance in this file to inverse-rate 8 (rate 1/8), and
/// `iprs-rate-1-16` switches to inverse-rate 16 (rate 1/16).
/// `iprs-rate-1-16` takes precedence if both are enabled.
const REP: usize = if cfg!(feature = "iprs-rate-1-16") {
    16
} else if cfg!(feature = "iprs-rate-1-8") {
    8
} else {
    4
};

/// Number of column openings the PCS performs. Tied to `REP`: rate 1/4
/// uses 150 openings, rate 1/8 uses 100, rate 1/16 uses 75.
const NUM_COL_OPENINGS_FOR_REP: usize = if cfg!(feature = "iprs-rate-1-16") {
    75
} else if cfg!(feature = "iprs-rate-1-8") {
    100
} else {
    150
};

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

/// Type constraints here must exactly match the constraints in `ZipTypes`
/// (except for added  Send + Sync constraints)
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

impl<Int, CwR, Chal, Pt, BinaryCombR, CombR, IntCombR, Fmod, PrimeTest, const D: usize> ZincTypes<D>
    for GenericBenchZincTypes<Int, CwR, Chal, Pt, BinaryCombR, CombR, IntCombR, Fmod, PrimeTest, D>
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
// Constants and concrete types
//

const DEGREE_PLUS_ONE: usize = 32;
const INT_LIMBS: usize = U64::LIMBS;
// 256-bit field modulus (4 × u64 limbs).
const FIELD_LIMBS: usize = U64::LIMBS * 4;

type F = MontyField<FIELD_LIMBS>;

type BenchZincTypes = GenericBenchZincTypes<
    /* Int         = */ i64,
    /* CwR         = */ i128,
    /* Chal        = */ i128,
    /* Pt          = */ i128,
    /* BinaryCombR = */ Int<{ INT_LIMBS * 5 }>,
    /* CombR       = */ Int<{ INT_LIMBS * 6 }>,
    /* IntCombR    = */ Int<{ INT_LIMBS * 4 }>,
    /* Fmod        = */ Uint<FIELD_LIMBS>,
    MillerRabin,
    DEGREE_PLUS_ONE,
>;
type Pp<Zt> = (
    ZipPlusParams<
        <Zt as ZincTypes<DEGREE_PLUS_ONE>>::BinaryZt,
        <Zt as ZincTypes<DEGREE_PLUS_ONE>>::BinaryLc,
    >,
    ZipPlusParams<
        <Zt as ZincTypes<DEGREE_PLUS_ONE>>::ArbitraryZt,
        <Zt as ZincTypes<DEGREE_PLUS_ONE>>::ArbitraryLc,
    >,
    ZipPlusParams<
        <Zt as ZincTypes<DEGREE_PLUS_ONE>>::IntZt,
        <Zt as ZincTypes<DEGREE_PLUS_ONE>>::IntLc,
    >,
);

/// Use row size equal to poly size, resulting in flat single-row matrices
#[allow(clippy::unwrap_used)]
fn setup_pp(num_vars: usize) -> Pp<BenchZincTypes> {
    let poly_size = 1 << num_vars;
    (
        ZipPlus::setup(
            poly_size,
            IprsCode::new_with_optimal_depth(poly_size).unwrap(),
        ),
        ZipPlus::setup(
            poly_size,
            IprsCode::new_with_optimal_depth(poly_size).unwrap(),
        ),
        ZipPlus::setup(
            poly_size,
            IprsCode::new_with_optimal_depth(poly_size).unwrap(),
        ),
    )
}

//
// Real-UAIR bench types — wired for the EcdsaUair / Sha256CompressionSliceUair
// / ShaEcdsaUair ports from main-gamma. Cell type is `Int<EC_FP_INT_LIMBS>`
// (= `Int<5>`, 320-bit); CwR and CombR scale 2× and 4× respectively. F is
// shared with `BenchZincTypes` (256-bit MontyField, holds the secp256k1
// base prime used by `fixed_prime::secp256k1_field_cfg`).
//

type RealEcdsaInt = Int<EC_FP_INT_LIMBS>;

type RealEcdsaBenchZincTypes = GenericBenchZincTypes<
    /* Int         = */ RealEcdsaInt,
    /* CwR         = */ Int<6>,
    /* Chal        = */ i128,
    /* Pt          = */ i128,
    /* BinaryCombR = */ Int<5>,
    /* CombR       = */ Int<{ EC_FP_INT_LIMBS * 4 }>,
    /* IntCombR    = */ Int<8>,
    /* Fmod        = */ Uint<FIELD_LIMBS>,
    MillerRabin,
    DEGREE_PLUS_ONE,
>;

#[allow(clippy::unwrap_used)]
fn setup_pp_real_ecdsa(num_vars: usize) -> Pp<RealEcdsaBenchZincTypes> {
    let poly_size = 1 << num_vars;
    (
        ZipPlus::setup(
            poly_size,
            IprsCode::new_with_optimal_depth(poly_size).unwrap(),
        ),
        ZipPlus::setup(
            poly_size,
            IprsCode::new_with_optimal_depth(poly_size).unwrap(),
        ),
        ZipPlus::setup(
            poly_size,
            IprsCode::new_with_optimal_depth(poly_size).unwrap(),
        ),
    )
}

fn try_setup_pp_real_ecdsa(num_vars: usize) -> Result<Pp<RealEcdsaBenchZincTypes>, String> {
    let poly_size = 1 << num_vars;
    let binary = ZipPlus::setup(
        poly_size,
        IprsCode::new_with_optimal_depth(poly_size).map_err(|error| error.to_string())?,
    );
    let arbitrary = ZipPlus::setup(
        poly_size,
        IprsCode::new_with_optimal_depth(poly_size).map_err(|error| error.to_string())?,
    );
    let int = ZipPlus::setup(
        poly_size,
        IprsCode::new_with_optimal_depth(poly_size).map_err(|error| error.to_string())?,
    );
    Ok((binary, arbitrary, int))
}

/// Project an `IdealOrZero<Sha256Ideal<RealEcdsaInt>>` to `Sha256Ideal<F>`
/// for the verifier. Zero ideals are filtered upstream of this closure (see
/// piop's ideal_check), so the `Zero` arm is unreachable.
fn sha256_real_project_ideal(
    ideal: &IdealOrZero<Sha256Ideal<RealEcdsaInt>>,
    field_cfg: &<F as PrimeField>::Config,
) -> Sha256Ideal<F> {
    match ideal {
        IdealOrZero::NonZero(Sha256Ideal::RotX2(r)) => {
            Sha256Ideal::RotX2(RotationIdeal::from_with_cfg(r, field_cfg))
        }
        IdealOrZero::NonZero(Sha256Ideal::RotXw1) => Sha256Ideal::RotXw1,
        IdealOrZero::Zero => {
            unreachable!("zero ideals are filtered before this closure runs")
        }
    }
}

const REAL_SHA256_CHAIN_BLOCKS: usize = 8;
const REAL_SHA256_CHAIN_NUM_VARS: usize = 10;

fn real_sha256_chain_message() -> String {
    vec!["hello world"; 40].join(" ")
}

#[allow(clippy::unwrap_used)]
fn real_sha256_chain_blocks() -> [Sha256MessageBlock; REAL_SHA256_CHAIN_BLOCKS] {
    let message = real_sha256_chain_message();
    sha256_padded_message_blocks::<REAL_SHA256_CHAIN_BLOCKS>(message.as_bytes())
        .expect("real SHA-256 benchmark fixture should canonically pad to eight blocks")
}

#[allow(clippy::unwrap_used)]
fn exact_sha256_chain_blocks<const N: usize>() -> [Sha256MessageBlock; N] {
    assert!(N > 0, "SHA instance sweep requires at least one block");
    const SHA_BLOCK_BYTES: usize = 64;
    const SHA_PADDING_BYTES: usize = 9;
    let message_len = N
        .checked_mul(SHA_BLOCK_BYTES)
        .and_then(|len| len.checked_sub(SHA_PADDING_BYTES))
        .expect("SHA instance sweep message length fits usize");
    let message = (0..message_len)
        .map(|idx| b'a' + u8::try_from(idx % 26).expect("alphabet index fits u8"))
        .collect::<Vec<_>>();
    sha256_padded_message_blocks::<N>(&message)
        .expect("instance sweep fixture should pad to exactly N SHA-256 blocks")
}

fn log2_power_of_two(value: usize) -> usize {
    assert!(value.is_power_of_two(), "value must be a power of two");
    value.trailing_zeros() as usize
}

#[allow(clippy::unwrap_used)]
fn real_sha256_chain_trace(
    num_vars: usize,
) -> UairTrace<'static, RealEcdsaInt, RealEcdsaInt, DEGREE_PLUS_ONE> {
    let (trace, _final_state) = synthesize_sha256_chain_trace::<
        RealEcdsaInt,
        REAL_SHA256_CHAIN_BLOCKS,
    >(num_vars, SHA256_INITIAL_STATE, real_sha256_chain_blocks())
    .expect("real SHA-256 monolithic chain trace synthesis should succeed");
    trace
}

#[derive(Clone, Debug)]
struct ProjectionShaBenchUair<R>(PhantomData<R>);

impl<R> Uair for ProjectionShaBenchUair<R>
where
    R: ConstSemiring + 'static,
{
    type Ideal = Sha256Ideal<R>;
    type Scalar = DensePolynomial<R, DEGREE_PLUS_ONE>;

    fn signature() -> UairSignature {
        Sha256CompressionSliceUair::<R>::signature()
    }

    fn constrain_general<B, FromR, MulByScalarFn, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        down: TraceRow<B::Expr>,
        from_ref: FromR,
        mbs: MulByScalarFn,
        ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&Self::Scalar) -> B::Expr,
        MulByScalarFn: Fn(&B::Expr, &Self::Scalar) -> Option<B::Expr>,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
    {
        Sha256CompressionSliceUair::<R>::constrain_general(
            b,
            up,
            down,
            from_ref,
            mbs,
            ideal_from_ref,
        );
    }

    fn verify_public_structure<RT, IntT, const D: usize>(
        public_trace: &UairTrace<'_, RT, IntT, D>,
        num_vars: usize,
    ) -> Result<(), PublicStructureError>
    where
        RT: Clone,
        IntT: Clone + num_traits::Zero,
    {
        Sha256CompressionSliceUair::<R>::verify_public_structure(public_trace, num_vars)
    }
}

type ArkFBn254 = ArkFp<MontBackend<ark_bn254::FrConfig, 4>, 4>;
type ArkFSecp256k1 = ArkFp<MontBackend<ark_secp256k1::FrConfig, 4>, 4>;

/// Field-specific hooks for the ProjectionFold bench projection helpers.
///
/// The arkworks-backed const field cannot implement
/// `FromWithConfig<&RealEcdsaInt>` (trait and type are both foreign) and has
/// custom config acquisition and int projection hooks, while four-limb fields
/// share the DMR bit-slice scalarization path.
trait BenchShaField: PrimeField + FromPrimitiveWithConfig {
    fn curve_field_cfg<C: AffineRepr>() -> Self::Config;

    fn from_bench_int(value: &RealEcdsaInt, field_cfg: &Self::Config) -> Self;

    fn scalarize_bit_slices(
        bit_slices: &MleTable<Self>,
        a: &Self,
        field_cfg: &Self::Config,
    ) -> Result<MleTable<Self>, ProductionShaError<Self>>;
}

impl BenchShaField for MontyField<FIELD_LIMBS> {
    fn curve_field_cfg<C: AffineRepr>() -> Self::Config {
        field_cfg_from_curve_scalar::<Self, Uint<FIELD_LIMBS>, C>()
    }

    fn from_bench_int(value: &RealEcdsaInt, field_cfg: &Self::Config) -> Self {
        Self::from_with_cfg(value, field_cfg)
    }

    fn scalarize_bit_slices(
        bit_slices: &MleTable<Self>,
        a: &Self,
        field_cfg: &Self::Config,
    ) -> Result<MleTable<Self>, ProductionShaError<Self>> {
        projection_sha_scalarize_bit_slices_dmr(bit_slices, a, field_cfg)
    }
}

impl<M: MontConfig<4>> BenchShaField for ArkFp<MontBackend<M, 4>, 4> {
    fn curve_field_cfg<C: AffineRepr>() -> Self::Config {}

    fn from_bench_int(value: &RealEcdsaInt, _field_cfg: &Self::Config) -> Self {
        let (abs, is_negative) = if value.is_negative() {
            (
                value.checked_abs().expect("bench int fits absolute value"),
                true,
            )
        } else {
            (*value, false)
        };
        let words = abs.as_uint().as_words();
        let mut bytes = Vec::with_capacity(words.len() * size_of::<u64>());
        for word in words {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        let magnitude = ArkFp::new(ark_ff::PrimeField::from_le_bytes_mod_order(&bytes));
        if is_negative { -magnitude } else { magnitude }
    }

    fn scalarize_bit_slices(
        bit_slices: &MleTable<Self>,
        a: &Self,
        field_cfg: &Self::Config,
    ) -> Result<MleTable<Self>, ProductionShaError<Self>> {
        projection_sha_scalarize_bit_slices_dmr(bit_slices, a, field_cfg)
    }
}

fn projection_sha_binary_col<'a, F: PrimeField>(
    public_trace: &'a UairTrace<'_, RealEcdsaInt, RealEcdsaInt, DEGREE_PLUS_ONE>,
    witness_trace: &'a UairTrace<'_, RealEcdsaInt, RealEcdsaInt, DEGREE_PLUS_ONE>,
    flat_col: usize,
) -> Result<&'a DenseMultilinearExtension<BinaryPoly<DEGREE_PLUS_ONE>>, ProductionShaError<F>> {
    if flat_col < sha256_cols::NUM_BIN_PUB {
        public_trace
            .binary_poly
            .get(flat_col)
            .ok_or(ProductionShaError::LengthMismatch {
                label: "SHA binary public source columns",
                got: public_trace.binary_poly.len(),
                expected: flat_col + 1,
            })
    } else {
        let witness_col = flat_col - sha256_cols::NUM_BIN_PUB;
        witness_trace
            .binary_poly
            .get(witness_col)
            .ok_or(ProductionShaError::LengthMismatch {
                label: "SHA binary witness source columns",
                got: witness_trace.binary_poly.len(),
                expected: witness_col + 1,
            })
    }
}

fn projection_sha_int_col<'a, F: PrimeField>(
    public_trace: &'a UairTrace<'_, RealEcdsaInt, RealEcdsaInt, DEGREE_PLUS_ONE>,
    witness_trace: &'a UairTrace<'_, RealEcdsaInt, RealEcdsaInt, DEGREE_PLUS_ONE>,
    flat_col: usize,
) -> Result<&'a DenseMultilinearExtension<RealEcdsaInt>, ProductionShaError<F>> {
    if flat_col < sha256_cols::NUM_INT_PUB {
        public_trace
            .int
            .get(flat_col)
            .ok_or(ProductionShaError::LengthMismatch {
                label: "SHA int public source columns",
                got: public_trace.int.len(),
                expected: flat_col + 1,
            })
    } else {
        let witness_col = flat_col - sha256_cols::NUM_INT_PUB;
        witness_trace
            .int
            .get(witness_col)
            .ok_or(ProductionShaError::LengthMismatch {
                label: "SHA int witness source columns",
                got: witness_trace.int.len(),
                expected: witness_col + 1,
            })
    }
}

fn projection_sha_project_binary_source<F: PrimeField>(
    col: &DenseMultilinearExtension<BinaryPoly<DEGREE_PLUS_ONE>>,
    field_cfg: &<F as PrimeField>::Config,
) -> Result<Vec<Vec<F>>, ProductionShaError<F>> {
    if col.evaluations.len() < SHA_ROW_COUNT {
        return Err(ProductionShaError::LengthMismatch {
            label: "SHA binary source rows",
            got: col.evaluations.len(),
            expected: SHA_ROW_COUNT,
        });
    }
    let rows = &col.evaluations[..SHA_ROW_COUNT];
    Ok(cfg_iter!(rows)
        .map(|poly| {
            poly.iter()
                .take(SHA_WORD_BITS)
                .map(|bit| {
                    if bit.into_inner() {
                        F::one_with_cfg(field_cfg)
                    } else {
                        F::zero_with_cfg(field_cfg)
                    }
                })
                .collect()
        })
        .collect())
}

fn projection_sha_project_int_source<F: BenchShaField>(
    col: &DenseMultilinearExtension<RealEcdsaInt>,
    field_cfg: &<F as PrimeField>::Config,
) -> Result<Vec<F>, ProductionShaError<F>> {
    if col.evaluations.len() < SHA_ROW_COUNT {
        return Err(ProductionShaError::LengthMismatch {
            label: "SHA int source rows",
            got: col.evaluations.len(),
            expected: SHA_ROW_COUNT,
        });
    }
    let rows = &col.evaluations[..SHA_ROW_COUNT];
    Ok(cfg_iter!(rows)
        .map(|value| F::from_bench_int(value, field_cfg))
        .collect())
}

fn projection_sha_truncate_row_domain<Eval: Clone + Send + Sync, F: PrimeField>(
    col: &DenseMultilinearExtension<Eval>,
    label: &'static str,
) -> Result<DenseMultilinearExtension<Eval>, ProductionShaError<F>> {
    if col.evaluations.len() < SHA_ROW_COUNT {
        return Err(ProductionShaError::LengthMismatch {
            label,
            got: col.evaluations.len(),
            expected: SHA_ROW_COUNT,
        });
    }
    Ok(DenseMultilinearExtension {
        evaluations: cfg_iter!(&col.evaluations[..SHA_ROW_COUNT])
            .cloned()
            .collect(),
        num_vars: SHA_ROW_VARS,
    })
}

fn projection_sha_word_scalar_at_two<F: PrimeField>(
    bits: &[F],
    field_cfg: &<F as PrimeField>::Config,
) -> F {
    let two = F::one_with_cfg(field_cfg) + F::one_with_cfg(field_cfg);
    let mut power = F::one_with_cfg(field_cfg);
    let mut value = F::zero_with_cfg(field_cfg);
    for bit in bits {
        value += bit.clone() * &power;
        power *= &two;
    }
    value
}

fn projection_sha_mle_table_from_columns<T>(columns: Vec<Vec<T>>) -> MleTable<T> {
    columns
        .into_iter()
        .map(|evaluations| DenseMultilinearExtension {
            evaluations,
            num_vars: SHA_ROW_VARS,
        })
        .collect()
}

fn projection_sha_flatten_bit_columns<T: Clone + Send + Sync>(
    columns: Vec<Vec<Vec<T>>>,
) -> MleTable<T> {
    let flattened = cfg_into_iter!(0..columns.len() * SHA_WORD_BITS)
        .map(|flat_idx| {
            let col_idx = flat_idx / SHA_WORD_BITS;
            let bit = flat_idx % SHA_WORD_BITS;
            columns[col_idx]
                .iter()
                .map(|row_bits| row_bits[bit].clone())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    projection_sha_mle_table_from_columns(flattened)
}

fn projection_sha_flatten_bit_column_refs<T: Clone + Send + Sync>(
    columns: &[&[Vec<T>]],
) -> MleTable<T> {
    let flattened = cfg_into_iter!(0..columns.len() * SHA_WORD_BITS)
        .map(|flat_idx| {
            let col_idx = flat_idx / SHA_WORD_BITS;
            let bit = flat_idx % SHA_WORD_BITS;
            columns[col_idx]
                .iter()
                .map(|row_bits| row_bits[bit].clone())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    projection_sha_mle_table_from_columns(flattened)
}

fn projection_sha_mle_bit_slices_from_native_word_bits<F, const COLS: usize>(
    word_bits: &[[u128; SHA_WORD_BITS]; COLS],
    field_cfg: &<F as PrimeField>::Config,
) -> MleTable<F>
where
    F: PrimeField,
{
    let zero = F::zero_with_cfg(field_cfg);
    let one = F::one_with_cfg(field_cfg);
    let mut columns = Vec::with_capacity(COLS * SHA_WORD_BITS);
    for bits in word_bits {
        for &mask in bits {
            let mut evaluations = Vec::with_capacity(SHA_ROW_COUNT);
            for row in 0..SHA_ROW_COUNT {
                evaluations.push(if ((mask >> row) & 1) == 1 {
                    one.clone()
                } else {
                    zero.clone()
                });
            }
            columns.push(evaluations);
        }
    }
    projection_sha_mle_table_from_columns(columns)
}

fn projection_sha_project_binary_sources_to_native_word_bits<F>(
    cols: &[&DenseMultilinearExtension<BinaryPoly<DEGREE_PLUS_ONE>>],
    field_cfg: &<F as PrimeField>::Config,
) -> Result<(MleTable<F>, Vec<[u128; SHA_WORD_BITS]>), ProductionShaError<F>>
where
    F: PrimeField,
{
    let zero = F::zero_with_cfg(field_cfg);
    let one = F::one_with_cfg(field_cfg);
    let mut flat_columns = (0..cols.len() * SHA_WORD_BITS)
        .map(|_| Vec::with_capacity(SHA_ROW_COUNT))
        .collect::<Vec<_>>();
    let mut packed_columns = Vec::with_capacity(cols.len());

    for (col_idx, col) in cols.iter().copied().enumerate() {
        if col.evaluations.len() < SHA_ROW_COUNT {
            return Err(ProductionShaError::LengthMismatch {
                label: "SHA binary source rows",
                got: col.evaluations.len(),
                expected: SHA_ROW_COUNT,
            });
        }

        let mut packed = [0u128; SHA_WORD_BITS];
        for (row, poly) in col.evaluations.iter().take(SHA_ROW_COUNT).enumerate() {
            let mut seen_bits = 0usize;
            for (bit_idx, bit) in poly.iter().take(SHA_WORD_BITS).enumerate() {
                seen_bits += 1;
                let value = bit.into_inner();
                if value {
                    packed[bit_idx] |= 1u128 << row;
                }
                flat_columns[bit_slice_index(col_idx, bit_idx, SHA_WORD_BITS)].push(if value {
                    one.clone()
                } else {
                    zero.clone()
                });
            }
            if seen_bits != SHA_WORD_BITS {
                return Err(ProductionShaError::LengthMismatch {
                    label: "SHA binary source bits",
                    got: seen_bits,
                    expected: SHA_WORD_BITS,
                });
            }
        }
        packed_columns.push(packed);
    }

    Ok((
        projection_sha_mle_table_from_columns(flat_columns),
        packed_columns,
    ))
}

fn projection_sha_scalarize_bit_slices_dmr<G>(
    bit_slices: &MleTable<G>,
    a: &G,
    field_cfg: &G::Config,
) -> Result<MleTable<G>, ProductionShaError<G>>
where
    G: MontgomeryLimbs + Send + Sync,
{
    let powers = zinc_utils::powers(a.clone(), G::one_with_cfg(field_cfg), SHA_WORD_BITS);
    let word_count = bit_slices.len() / SHA_WORD_BITS;
    let one = G::one_with_cfg(field_cfg);
    let reducer = BarrettDelayedReduction::<G>::new(field_cfg);
    let words = cfg_into_iter!(0..word_count)
        .map(|col_idx| {
            let bit_cols = (0..SHA_WORD_BITS)
                .map(|bit| {
                    let bit_col = &bit_slices[bit_slice_index(col_idx, bit, SHA_WORD_BITS)];
                    if bit_col.num_vars != SHA_ROW_VARS
                        || bit_col.evaluations.len() != SHA_ROW_COUNT
                    {
                        return Err(ProductionShaError::LengthMismatch {
                            label: "SHA scalarized bit-slice rows",
                            got: bit_col.evaluations.len(),
                            expected: SHA_ROW_COUNT,
                        });
                    }
                    Ok(bit_col)
                })
                .collect::<Result<Vec<_>, ProductionShaError<G>>>()?;
            let mut out_col = Vec::with_capacity(SHA_ROW_COUNT);
            for row in 0..SHA_ROW_COUNT {
                out_col.push(projection_sha_scalarize_binary_row_dmr(
                    &bit_cols, row, &powers, &one, field_cfg, &reducer,
                ));
            }
            Ok(out_col)
        })
        .collect::<Result<Vec<_>, ProductionShaError<G>>>()?;
    Ok(projection_sha_mle_table_from_columns(words))
}

fn projection_sha_scalarize_binary_row_dmr<G>(
    bit_cols: &[&DenseMultilinearExtension<G>],
    row: usize,
    powers: &[G],
    one: &G,
    field_cfg: &G::Config,
    reducer: &BarrettDelayedReduction<'_, G>,
) -> G
where
    G: MontgomeryLimbs,
{
    let mut bucket = Uint::<5>::zero();
    let mut pending_adds = 0usize;
    let mut acc = G::zero_with_cfg(field_cfg);

    for (bit_col, power) in bit_cols.iter().zip(powers) {
        let bit = &bit_col.evaluations[row];
        if G::is_zero(bit) {
            continue;
        }
        if bit != one {
            return projection_sha_scalarize_row_naive(bit_cols, row, powers, field_cfg);
        }
        reducer.add(&mut bucket, power);
        pending_adds = pending_adds.saturating_add(1);
        if pending_adds >= reducer.flush_adds() {
            let pending = std::mem::replace(&mut bucket, Uint::zero());
            acc += reducer.reduce(pending);
            pending_adds = 0;
        }
    }

    if !bucket.is_zero() {
        acc += reducer.reduce(bucket);
    }
    acc
}

fn projection_sha_scalarize_row_naive<G: PrimeField>(
    bit_cols: &[&DenseMultilinearExtension<G>],
    row: usize,
    powers: &[G],
    field_cfg: &G::Config,
) -> G {
    let mut value = G::zero_with_cfg(field_cfg);
    for (bit_col, power) in bit_cols.iter().zip(powers) {
        value += bit_col.evaluations[row].clone() * power;
    }
    value
}

fn projection_sha_selector_expected<F: PrimeField>(
    selector: ShaPublicCol,
    row: usize,
    field_cfg: &<F as PrimeField>::Config,
) -> F {
    let active = match selector {
        ShaPublicCol::SInit => row < 4,
        ShaPublicCol::SMsg => row < 16,
        ShaPublicCol::SSched => row < 48,
        ShaPublicCol::SUpd => row < 64,
        ShaPublicCol::SFf => (64..68).contains(&row),
        ShaPublicCol::SOut => (68..72).contains(&row),
        _ => false,
    };
    if active {
        F::one_with_cfg(field_cfg)
    } else {
        F::zero_with_cfg(field_cfg)
    }
}

fn projection_sha_k_expected<F: PrimeField + FromPrimitiveWithConfig>(
    row: usize,
    field_cfg: &<F as PrimeField>::Config,
) -> F {
    if (3..67).contains(&row) {
        F::from_with_cfg(K_CANONICAL[row - 3] as u64, field_cfg)
    } else {
        F::zero_with_cfg(field_cfg)
    }
}

fn projection_sha_projected_public_from_sources<F: PrimeField + FromPrimitiveWithConfig>(
    pa_a: &[Vec<F>],
    pa_e: &[Vec<F>],
    message: &[Vec<F>],
    field_cfg: &<F as PrimeField>::Config,
) -> MleTable<F> {
    let mut columns = vec![vec![F::zero_with_cfg(field_cfg); SHA_ROW_COUNT]; ShaPublicCol::COUNT];
    for row in 0..SHA_ROW_COUNT {
        columns[ShaPublicCol::K.index()][row] = projection_sha_k_expected(row, field_cfg);
        columns[ShaPublicCol::PAIn.index()][row] =
            projection_sha_word_scalar_at_two(&pa_a[row], field_cfg);
        columns[ShaPublicCol::PEIn.index()][row] =
            projection_sha_word_scalar_at_two(&pa_e[row], field_cfg);
        columns[ShaPublicCol::PAOut.index()][row] =
            projection_sha_word_scalar_at_two(&pa_a[row], field_cfg);
        columns[ShaPublicCol::PEOut.index()][row] =
            projection_sha_word_scalar_at_two(&pa_e[row], field_cfg);
        columns[ShaPublicCol::Message.index()][row] =
            projection_sha_word_scalar_at_two(&message[row], field_cfg);
    }
    for selector in [
        ShaPublicCol::SInit,
        ShaPublicCol::SMsg,
        ShaPublicCol::SSched,
        ShaPublicCol::SUpd,
        ShaPublicCol::SFf,
        ShaPublicCol::SOut,
    ] {
        for row in 0..SHA_ROW_COUNT {
            columns[selector.index()][row] =
                projection_sha_selector_expected(selector, row, field_cfg);
        }
    }
    projection_sha_mle_table_from_columns(columns)
}

fn projection_sha_int_to_i64<F: PrimeField>(
    value: &RealEcdsaInt,
) -> Result<i64, ProductionShaError<F>> {
    let (abs, is_negative) = if value.is_negative() {
        (
            value
                .checked_abs()
                .ok_or(ProductionShaError::NonCanonicalProofObject(
                    "production SHA native int value does not fit i64",
                ))?,
            true,
        )
    } else {
        (*value, false)
    };
    let mut magnitude = 0u64;
    for (idx, &word) in abs.as_uint().as_words().iter().enumerate() {
        let word = word as u64;
        if idx == 0 {
            magnitude = word;
        } else if word != 0 {
            return Err(ProductionShaError::NonCanonicalProofObject(
                "production SHA native int value does not fit i64",
            ));
        }
    }
    let magnitude = i64::try_from(magnitude).map_err(|_| {
        ProductionShaError::NonCanonicalProofObject(
            "production SHA native int value does not fit i64",
        )
    })?;
    if is_negative {
        magnitude
            .checked_neg()
            .ok_or(ProductionShaError::NonCanonicalProofObject(
                "production SHA native int value does not fit i64",
            ))
    } else {
        Ok(magnitude)
    }
}

fn projection_sha_pack_binary_source_bits<F: PrimeField>(
    col: &DenseMultilinearExtension<BinaryPoly<DEGREE_PLUS_ONE>>,
) -> Result<[u128; SHA_WORD_BITS], ProductionShaError<F>> {
    if col.evaluations.len() < SHA_ROW_COUNT {
        return Err(ProductionShaError::LengthMismatch {
            label: "SHA binary source rows",
            got: col.evaluations.len(),
            expected: SHA_ROW_COUNT,
        });
    }
    let mut packed = [0u128; SHA_WORD_BITS];
    for (row, poly) in col.evaluations.iter().take(SHA_ROW_COUNT).enumerate() {
        let mut seen_bits = 0usize;
        for (bit_idx, bit) in poly.iter().take(SHA_WORD_BITS).enumerate() {
            seen_bits += 1;
            if bit.into_inner() {
                packed[bit_idx] |= 1u128 << row;
            }
        }
        if seen_bits != SHA_WORD_BITS {
            return Err(ProductionShaError::LengthMismatch {
                label: "SHA binary source bits",
                got: seen_bits,
                expected: SHA_WORD_BITS,
            });
        }
    }
    Ok(packed)
}

fn projection_sha_project_int_source_with_values<F: BenchShaField>(
    col: &DenseMultilinearExtension<RealEcdsaInt>,
    field_cfg: &<F as PrimeField>::Config,
) -> Result<(Vec<F>, [i64; SHA_ROW_COUNT]), ProductionShaError<F>> {
    if col.evaluations.len() < SHA_ROW_COUNT {
        return Err(ProductionShaError::LengthMismatch {
            label: "SHA int source rows",
            got: col.evaluations.len(),
            expected: SHA_ROW_COUNT,
        });
    }
    let mut projected = Vec::with_capacity(SHA_ROW_COUNT);
    let mut native = [0i64; SHA_ROW_COUNT];
    for (row, value) in col.evaluations.iter().take(SHA_ROW_COUNT).enumerate() {
        native[row] = projection_sha_int_to_i64::<F>(value)?;
        projected.push(F::from_with_cfg(native[row], field_cfg));
    }
    Ok((projected, native))
}

fn projection_sha_packed_word_row_value(bits: &[u128; SHA_WORD_BITS], row: usize) -> i64 {
    let mut value = 0i64;
    for (bit, mask) in bits.iter().enumerate() {
        if ((mask >> row) & 1) == 1 {
            value |= 1i64 << bit;
        }
    }
    value
}

fn projection_sha_selector_expected_i64(selector: ShaPublicCol, row: usize) -> i64 {
    match selector {
        ShaPublicCol::SInit if row < 4 => 1,
        ShaPublicCol::SMsg if row < 16 => 1,
        ShaPublicCol::SSched if row < 48 => 1,
        ShaPublicCol::SUpd if row < 64 => 1,
        ShaPublicCol::SFf if (64..68).contains(&row) => 1,
        ShaPublicCol::SOut if (68..72).contains(&row) => 1,
        _ => 0,
    }
}

fn projection_sha_native_public_values_from_bits(
    pa_a: &[u128; SHA_WORD_BITS],
    pa_e: &[u128; SHA_WORD_BITS],
    message: &[u128; SHA_WORD_BITS],
) -> [[i64; SHA_ROW_COUNT]; ShaPublicCol::COUNT] {
    let mut columns = [[0i64; SHA_ROW_COUNT]; ShaPublicCol::COUNT];
    for row in 0..SHA_ROW_COUNT {
        columns[ShaPublicCol::K.index()][row] = if (3..67).contains(&row) {
            i64::from(K_CANONICAL[row - 3])
        } else {
            0
        };
        columns[ShaPublicCol::PAIn.index()][row] = projection_sha_packed_word_row_value(pa_a, row);
        columns[ShaPublicCol::PEIn.index()][row] = projection_sha_packed_word_row_value(pa_e, row);
        columns[ShaPublicCol::PAOut.index()][row] = projection_sha_packed_word_row_value(pa_a, row);
        columns[ShaPublicCol::PEOut.index()][row] = projection_sha_packed_word_row_value(pa_e, row);
        columns[ShaPublicCol::Message.index()][row] =
            projection_sha_packed_word_row_value(message, row);
    }
    for selector in [
        ShaPublicCol::SInit,
        ShaPublicCol::SMsg,
        ShaPublicCol::SSched,
        ShaPublicCol::SUpd,
        ShaPublicCol::SFf,
        ShaPublicCol::SOut,
    ] {
        for row in 0..SHA_ROW_COUNT {
            columns[selector.index()][row] = projection_sha_selector_expected_i64(selector, row);
        }
    }
    columns
}

fn projection_sha_mle_table_from_native_i64_columns<F, const COLS: usize>(
    columns: &[[i64; SHA_ROW_COUNT]; COLS],
    field_cfg: &<F as PrimeField>::Config,
) -> MleTable<F>
where
    F: PrimeField + FromPrimitiveWithConfig,
{
    projection_sha_mle_table_from_columns(
        columns
            .iter()
            .map(|column| {
                column
                    .iter()
                    .map(|&value| F::from_with_cfg(value, field_cfg))
                    .collect()
            })
            .collect(),
    )
}

fn projection_sha_scalarized_from_native_word_bits<F>(
    word_bits: &[[u128; SHA_WORD_BITS]; ShaWordCol::COUNT],
    field_cfg: &<F as PrimeField>::Config,
) -> MleTable<F>
where
    F: PrimeField + FromPrimitiveWithConfig,
{
    let columns = word_bits
        .iter()
        .map(|bits| {
            (0..SHA_ROW_COUNT)
                .map(|row| {
                    F::from_with_cfg(projection_sha_packed_word_row_value(bits, row), field_cfg)
                })
                .collect()
        })
        .collect();
    projection_sha_mle_table_from_columns(columns)
}

impl<F: BenchShaField> ProductionShaProjectionAdapter<RealEcdsaBenchZincTypes, F, DEGREE_PLUS_ONE>
    for ProjectionShaBenchUair<RealEcdsaInt>
{
    fn project_production_sha_public(
        _shape: &UairShape<Self>,
        public_trace: &UairTrace<'_, RealEcdsaInt, RealEcdsaInt, DEGREE_PLUS_ONE>,
        field_cfg: &<F as PrimeField>::Config,
    ) -> Result<ProjectedPublic<F>, ProductionShaError<F>> {
        let empty_witness = UairTrace {
            binary_poly: Cow::Borrowed(&[]),
            arbitrary_poly: Cow::Borrowed(&[]),
            int: Cow::Borrowed(&[]),
        };
        let pa_a = projection_sha_project_binary_source(
            projection_sha_binary_col(public_trace, &empty_witness, sha256_cols::PA_A)?,
            field_cfg,
        )?;
        let pa_e = projection_sha_project_binary_source(
            projection_sha_binary_col(public_trace, &empty_witness, sha256_cols::PA_E)?,
            field_cfg,
        )?;
        let message = projection_sha_project_binary_source(
            projection_sha_binary_col(public_trace, &empty_witness, sha256_cols::PA_M)?,
            field_cfg,
        )?;
        let public_columns =
            projection_sha_projected_public_from_sources(&pa_a, &pa_e, &message, field_cfg);
        let public_bit_columns = [
            pa_a.as_slice(),
            pa_e.as_slice(),
            pa_a.as_slice(),
            pa_e.as_slice(),
            message.as_slice(),
        ];
        Ok(ProjectedPublic {
            columns: public_columns,
            bit_slices: Some(projection_sha_flatten_bit_column_refs(&public_bit_columns)),
        })
    }

    fn project_production_sha_witness(
        _shape: &UairShape<Self>,
        public_trace: &UairTrace<'_, RealEcdsaInt, RealEcdsaInt, DEGREE_PLUS_ONE>,
        witness_trace: &UairTrace<'_, RealEcdsaInt, RealEcdsaInt, DEGREE_PLUS_ONE>,
        field_cfg: &<F as PrimeField>::Config,
    ) -> Result<
        (
            ProjectedTrace<F>,
            ProjectedPublic<F>,
            ProductionShaWitnessPolys<RealEcdsaBenchZincTypes, DEGREE_PLUS_ONE>,
        ),
        ProductionShaError<F>,
    > {
        let word_sources = [
            sha256_cols::W_A,
            sha256_cols::W_E,
            sha256_cols::W_SIG0,
            sha256_cols::W_SIG1,
            sha256_cols::W_W,
            sha256_cols::W_LSIG0,
            sha256_cols::W_LSIG1,
            sha256_cols::W_U_EF,
            sha256_cols::W_U_NEG_E_G,
            sha256_cols::W_MAJ,
            sha256_cols::W_MU_PACKED,
            sha256_cols::PA_OV_SIG0,
            sha256_cols::PA_OV_SIG1,
            sha256_cols::PA_OV_LSIG0,
            sha256_cols::PA_OV_LSIG1,
            sha256_cols::PA_R_CH2_COMP,
            sha256_cols::PA_R_MAJ_COMP,
        ];
        let int_sources = [
            sha256_cols::PA_C_C7,
            sha256_cols::PA_C_C8,
            sha256_cols::PA_C_C9,
            sha256_cols::PA_C_FF_A,
            sha256_cols::PA_C_FF_E,
        ];

        let word_cols = cfg_iter!(&word_sources)
            .map(|&col| projection_sha_binary_col(public_trace, witness_trace, col))
            .collect::<Result<Vec<_>, _>>()?;
        let int_cols = cfg_iter!(&int_sources)
            .map(|&col| projection_sha_int_col(public_trace, witness_trace, col))
            .collect::<Result<Vec<_>, _>>()?;

        let bit_columns = cfg_iter!(&word_cols)
            .map(|&col| projection_sha_project_binary_source(col, field_cfg))
            .collect::<Result<Vec<_>, _>>()?;
        let bit_slices = projection_sha_flatten_bit_columns(bit_columns);
        let scalarized =
            F::scalarize_bit_slices(&bit_slices, &F::from_with_cfg(2u64, field_cfg), field_cfg)?;
        let pa_a = projection_sha_project_binary_source(
            projection_sha_binary_col(public_trace, witness_trace, sha256_cols::PA_A)?,
            field_cfg,
        )?;
        let pa_e = projection_sha_project_binary_source(
            projection_sha_binary_col(public_trace, witness_trace, sha256_cols::PA_E)?,
            field_cfg,
        )?;
        let message = projection_sha_project_binary_source(
            projection_sha_binary_col(public_trace, witness_trace, sha256_cols::PA_M)?,
            field_cfg,
        )?;
        let public_columns =
            projection_sha_projected_public_from_sources(&pa_a, &pa_e, &message, field_cfg);
        let int_columns = cfg_iter!(&int_cols)
            .map(|&col| projection_sha_project_int_source(col, field_cfg))
            .collect::<Result<Vec<_>, _>>()?;
        let public_bit_columns = [
            pa_a.as_slice(),
            pa_e.as_slice(),
            pa_a.as_slice(),
            pa_e.as_slice(),
            message.as_slice(),
        ];

        let trace = ProjectedTrace {
            bit_slices,
            scalarized,
            int_columns: projection_sha_mle_table_from_columns(int_columns),
            public_columns: public_columns.clone(),
        };
        let public = ProjectedPublic {
            columns: public_columns,
            bit_slices: Some(projection_sha_flatten_bit_column_refs(&public_bit_columns)),
        };
        Ok((
            trace,
            public,
            ProductionShaWitnessPolys {
                binary: cfg_iter!(&word_cols)
                    .map(|&col| {
                        projection_sha_truncate_row_domain(
                            col,
                            "SHA binary witness row-domain projection",
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?,
                arbitrary: Vec::new(),
                int: cfg_iter!(&int_cols)
                    .map(|&col| {
                        projection_sha_truncate_row_domain(
                            col,
                            "SHA int witness row-domain projection",
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            },
        ))
    }

    fn project_production_sha_witness_with_native_view(
        _shape: &UairShape<Self>,
        public_trace: &UairTrace<'_, RealEcdsaInt, RealEcdsaInt, DEGREE_PLUS_ONE>,
        witness_trace: &UairTrace<'_, RealEcdsaInt, RealEcdsaInt, DEGREE_PLUS_ONE>,
        field_cfg: &<F as PrimeField>::Config,
    ) -> Result<
        (
            ProjectedTrace<F>,
            ProjectedPublic<F>,
            ProductionShaWitnessPolys<RealEcdsaBenchZincTypes, DEGREE_PLUS_ONE>,
            Option<PreparedProductionShaNativeView>,
        ),
        ProductionShaError<F>,
    > {
        let word_sources = [
            sha256_cols::W_A,
            sha256_cols::W_E,
            sha256_cols::W_SIG0,
            sha256_cols::W_SIG1,
            sha256_cols::W_W,
            sha256_cols::W_LSIG0,
            sha256_cols::W_LSIG1,
            sha256_cols::W_U_EF,
            sha256_cols::W_U_NEG_E_G,
            sha256_cols::W_MAJ,
            sha256_cols::W_MU_PACKED,
            sha256_cols::PA_OV_SIG0,
            sha256_cols::PA_OV_SIG1,
            sha256_cols::PA_OV_LSIG0,
            sha256_cols::PA_OV_LSIG1,
            sha256_cols::PA_R_CH2_COMP,
            sha256_cols::PA_R_MAJ_COMP,
        ];
        let int_sources = [
            sha256_cols::PA_C_C7,
            sha256_cols::PA_C_C8,
            sha256_cols::PA_C_C9,
            sha256_cols::PA_C_FF_A,
            sha256_cols::PA_C_FF_E,
        ];

        let word_cols = cfg_iter!(&word_sources)
            .map(|&col| projection_sha_binary_col(public_trace, witness_trace, col))
            .collect::<Result<Vec<_>, _>>()?;
        let int_cols = cfg_iter!(&int_sources)
            .map(|&col| projection_sha_int_col(public_trace, witness_trace, col))
            .collect::<Result<Vec<_>, _>>()?;

        let (bit_slices, word_bit_vec) =
            projection_sha_project_binary_sources_to_native_word_bits::<F>(&word_cols, field_cfg)?;
        let word_bits: [[u128; SHA_WORD_BITS]; ShaWordCol::COUNT] = word_bit_vec
            .try_into()
            .map_err(
                |bits: Vec<[u128; SHA_WORD_BITS]>| ProductionShaError::LengthMismatch {
                    label: "SHA native word columns",
                    got: bits.len(),
                    expected: ShaWordCol::COUNT,
                },
            )?;

        let scalarized =
            projection_sha_scalarized_from_native_word_bits::<F>(&word_bits, field_cfg);
        let pa_a_bits = projection_sha_pack_binary_source_bits::<F>(projection_sha_binary_col(
            public_trace,
            witness_trace,
            sha256_cols::PA_A,
        )?)?;
        let pa_e_bits = projection_sha_pack_binary_source_bits::<F>(projection_sha_binary_col(
            public_trace,
            witness_trace,
            sha256_cols::PA_E,
        )?)?;
        let message_bits = projection_sha_pack_binary_source_bits::<F>(projection_sha_binary_col(
            public_trace,
            witness_trace,
            sha256_cols::PA_M,
        )?)?;
        let public_word_bits = [pa_a_bits, pa_e_bits, pa_a_bits, pa_e_bits, message_bits];
        let public_bit_slices = projection_sha_mle_bit_slices_from_native_word_bits::<
            F,
            { ShaPublicWordCol::COUNT },
        >(&public_word_bits, field_cfg);
        let public_values =
            projection_sha_native_public_values_from_bits(&pa_a_bits, &pa_e_bits, &message_bits);
        let public_columns = projection_sha_mle_table_from_native_i64_columns::<
            F,
            { ShaPublicCol::COUNT },
        >(&public_values, field_cfg);

        let int_projection = cfg_iter!(&int_cols)
            .map(|&col| projection_sha_project_int_source_with_values(col, field_cfg))
            .collect::<Result<Vec<_>, _>>()?;
        let (int_columns, int_value_vec): (Vec<_>, Vec<_>) = int_projection.into_iter().unzip();
        let int_values: [[i64; SHA_ROW_COUNT]; ShaIntCol::COUNT] = int_value_vec
            .try_into()
            .map_err(
                |values: Vec<[i64; SHA_ROW_COUNT]>| ProductionShaError::LengthMismatch {
                    label: "SHA native int columns",
                    got: values.len(),
                    expected: ShaIntCol::COUNT,
                },
            )?;

        let trace = ProjectedTrace {
            bit_slices,
            scalarized,
            int_columns: projection_sha_mle_table_from_columns(int_columns),
            public_columns: public_columns.clone(),
        };
        let public = ProjectedPublic {
            columns: public_columns,
            bit_slices: Some(public_bit_slices),
        };
        let native_view = PreparedProductionShaNativeView {
            word_bits: Box::new(word_bits),
            int_values: Box::new(int_values),
            public_word_bits: Box::new(public_word_bits),
            public_values: Box::new(public_values),
        };
        Ok((
            trace,
            public,
            ProductionShaWitnessPolys {
                binary: cfg_iter!(&word_cols)
                    .map(|&col| {
                        projection_sha_truncate_row_domain(
                            col,
                            "SHA binary witness row-domain projection",
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?,
                arbitrary: Vec::new(),
                int: cfg_iter!(&int_cols)
                    .map(|&col| {
                        projection_sha_truncate_row_domain(
                            col,
                            "SHA int witness row-domain projection",
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            },
            Some(native_view),
        ))
    }
}

fn projection_sha_hyrax_key_pair<C, Lanes>(
    width: usize,
    offset: u64,
) -> (
    zip_plus::pcs::hyrax::HyraxCommitmentKey<C>,
    zip_plus::pcs::hyrax::HyraxVerifierKey<C>,
)
where
    C: AffineRepr,
    Lanes: Clone + Debug + Send + Sync,
{
    let generator = C::Group::generator();
    let bases = (0..width)
        .map(|idx| {
            let scalar = C::ScalarField::from(
                offset + u64::try_from(idx).expect("Hyrax basis index fits u64") + 1,
            );
            (generator * scalar).into_affine()
        })
        .collect::<Vec<_>>();
    let h = generator
        * C::ScalarField::from(offset + u64::try_from(width).expect("Hyrax width fits u64") + 1);
    HyraxPCS::<C, Lanes>::setup_from_bases_with_blinding(
        width,
        bases,
        h,
        HyraxBlindingMode::Unblinded,
    )
    .expect("Hyrax benchmark setup must be valid")
}

fn projection_sha_hyrax_pcs_params<C, F>(
    width: usize,
) -> (
    PCSParams<AllHyraxPCSTypes<C>, RealEcdsaBenchZincTypes, F, DEGREE_PLUS_ONE>,
    PCSVerifierParams<AllHyraxPCSTypes<C>, RealEcdsaBenchZincTypes, F, DEGREE_PLUS_ONE>,
)
where
    C: AffineRepr,
    F: PrimeField + zip_plus::pcs::hyrax::HyraxFieldBridge<C>,
{
    let (shared_ck, shared_vk) = projection_sha_hyrax_key_pair::<C, BinaryLanes>(width, 0);
    let (arbitrary_ck, arbitrary_vk) =
        projection_sha_hyrax_key_pair::<C, DensePolyScalarLanes>(width, 1_000);
    let binary_ck = shared_ck.clone();
    let int_ck = shared_ck;
    let binary_vk = shared_vk.clone();
    let int_vk = shared_vk;

    (
        PCSParams::<AllHyraxPCSTypes<C>, RealEcdsaBenchZincTypes, F, DEGREE_PLUS_ONE> {
            binary: binary_ck,
            arbitrary: arbitrary_ck,
            int: int_ck,
        },
        PCSVerifierParams::<AllHyraxPCSTypes<C>, RealEcdsaBenchZincTypes, F, DEGREE_PLUS_ONE> {
            binary: binary_vk,
            arbitrary: arbitrary_vk,
            int: int_vk,
        },
    )
}

type BenchBn254HyraxParams = (
    PCSParams<
        AllHyraxPCSTypes<ark_bn254::G1Affine>,
        RealEcdsaBenchZincTypes,
        ArkFBn254,
        DEGREE_PLUS_ONE,
    >,
    PCSVerifierParams<
        AllHyraxPCSTypes<ark_bn254::G1Affine>,
        RealEcdsaBenchZincTypes,
        ArkFBn254,
        DEGREE_PLUS_ONE,
    >,
);

type BenchProjectionShaUair = ProjectionShaBenchUair<RealEcdsaInt>;
type BenchBn254MixedHyraxVerifierSetup = VerifiedLinearIdealFoldSetup<
    AllHyraxPCSTypes<ark_bn254::G1Affine>,
    BenchProjectionShaUair,
    RealEcdsaBenchZincTypes,
    ArkFBn254,
    DEGREE_PLUS_ONE,
>;

static PROJECTION_SHA_BN254_HYRAX_PARAMS: OnceLock<Mutex<HashMap<usize, BenchBn254HyraxParams>>> =
    OnceLock::new();
static PROJECTION_SHA_BN254_MIXED_HYRAX_VERIFIER_SETUPS: OnceLock<
    Mutex<HashMap<usize, BenchBn254MixedHyraxVerifierSetup>>,
> = OnceLock::new();

fn projection_sha_bn254_hyrax_pcs_params(width: usize) -> BenchBn254HyraxParams {
    let mut cache = PROJECTION_SHA_BN254_HYRAX_PARAMS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("ProjectionFold Hyrax params cache mutex poisoned");
    cache
        .entry(width)
        .or_insert_with(|| projection_sha_hyrax_pcs_params::<ark_bn254::G1Affine, ArkFBn254>(width))
        .clone()
}

fn projection_sha_bn254_mixed_hyrax_verifier_setup(
    width: usize,
) -> BenchBn254MixedHyraxVerifierSetup {
    let mut cache = PROJECTION_SHA_BN254_MIXED_HYRAX_VERIFIER_SETUPS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("ProjectionFold Hyrax verifier setup cache mutex poisoned");
    cache
        .entry(width)
        .or_insert_with(|| {
            let (_, pcs_verifier_params) = projection_sha_bn254_hyrax_pcs_params(width);
            setup_verify_linear_ideal_fold_mixed_hyrax::<
                ark_bn254::G1Affine,
                BenchProjectionShaUair,
                RealEcdsaBenchZincTypes,
                ArkFBn254,
                DEGREE_PLUS_ONE,
            >(
                LinearIdealFoldVerifierParams::new(
                    pcs_verifier_params,
                    ArkFBn254::curve_field_cfg::<ark_bn254::G1Affine>(),
                ),
                UairShape::<BenchProjectionShaUair>::new(SHA_ROW_VARS),
            )
            .expect("mixed Hyrax verifier setup succeeds")
        })
        .clone()
}

fn projection_sha_packed_hyrax_pcs_params<C, F>(
    width: usize,
) -> (
    PCSParams<AllHyraxPCSTypes<C>, RealEcdsaBenchZincTypes, F, DEGREE_PLUS_ONE>,
    PCSVerifierParams<AllHyraxPCSTypes<C>, RealEcdsaBenchZincTypes, F, DEGREE_PLUS_ONE>,
)
where
    C: AffineRepr,
    F: PrimeField + zip_plus::pcs::hyrax::HyraxFieldBridge<C>,
{
    let (packed_ck, packed_vk) = projection_sha_hyrax_key_pair::<C, ScalarFieldLane>(width, 10_000);
    let (arbitrary_ck, arbitrary_vk) =
        projection_sha_hyrax_key_pair::<C, DensePolyScalarLanes>(SHA_ROW_COUNT, 20_000);
    let (int_ck, int_vk) = projection_sha_hyrax_key_pair::<C, IntScalarLane>(SHA_ROW_COUNT, 30_000);

    (
        PCSParams::<AllHyraxPCSTypes<C>, RealEcdsaBenchZincTypes, F, DEGREE_PLUS_ONE> {
            binary: packed_ck,
            arbitrary: arbitrary_ck,
            int: int_ck,
        },
        PCSVerifierParams::<AllHyraxPCSTypes<C>, RealEcdsaBenchZincTypes, F, DEGREE_PLUS_ONE> {
            binary: packed_vk,
            arbitrary: arbitrary_vk,
            int: int_vk,
        },
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HyraxReportCategory {
    OgHash,
    OgMixedHyrax,
    ProjectionFoldMixedHyrax,
    ProjectionFoldPackedHyrax,
}

impl HyraxReportCategory {
    fn algorithm(self) -> &'static str {
        match self {
            Self::OgHash => "OG Zinc+ hash PCS",
            Self::OgMixedHyrax => "OG Zinc+ with Mixed Hyrax",
            Self::ProjectionFoldMixedHyrax => "ProjectionFold with Mixed Hyrax",
            Self::ProjectionFoldPackedHyrax => "ProjectionFold with Packed Hyrax",
        }
    }

    fn order(self) -> usize {
        match self {
            Self::OgHash => 0,
            Self::OgMixedHyrax => 1,
            Self::ProjectionFoldMixedHyrax => 2,
            Self::ProjectionFoldPackedHyrax => 3,
        }
    }

    fn short_label(self) -> &'static str {
        match self {
            Self::OgHash => "OG hash",
            Self::OgMixedHyrax => "OG mixed",
            Self::ProjectionFoldMixedHyrax => "PF mixed",
            Self::ProjectionFoldPackedHyrax => "PF packed",
        }
    }
}

#[derive(Clone, Debug)]
struct HyraxWidthSweepRow {
    algorithm: &'static str,
    variant: String,
    width: Option<usize>,
    ecc_points_per_commitment: Option<usize>,
    ecc_points_per_proof: Option<usize>,
    /// Headline prover time: median of the recorded warmed samples.
    prover_ms: f64,
    prover_mean_ms: f64,
    prover_min_ms: f64,
    prover_max_ms: f64,
    prover_samples: usize,
    /// Headline verifier time: median of the recorded warmed samples.
    verifier_ms: f64,
    verifier_mean_ms: f64,
    verifier_min_ms: f64,
    verifier_max_ms: f64,
    verifier_samples: usize,
    proof_bytes: usize,
    proof_zstd_bytes: usize,
    category: HyraxReportCategory,
}

#[derive(Clone, Debug)]
struct HyraxInstanceSweepRow {
    algorithm: &'static str,
    variant: String,
    instances: usize,
    ell: usize,
    l0: usize,
    tail_vars: usize,
    width: usize,
    ecc_points_per_commitment: usize,
    ecc_points_per_proof: usize,
    trace_witness_ms: f64,
    setup_ms: f64,
    prepare_sumfold_basis_ms: f64,
    prepare_ms: f64,
    setup_prepare_ms: f64,
    probe_prove_ms: f64,
    probe_commit_ms: f64,
    probe_sumfold_linear_ms: f64,
    probe_sumfold_booleanity_ms: f64,
    probe_sumfold_group_ms: f64,
    probe_sumfold_prove_rounds_ms: f64,
    probe_sumfold_ms: f64,
    probe_fold_ms: f64,
    probe_open_ms: f64,
    probe_open_core_ms: f64,
    /// Headline prover time: median of the recorded warmed samples.
    prover_ms: f64,
    prover_mean_ms: f64,
    prover_min_ms: f64,
    prover_max_ms: f64,
    prover_samples: usize,
    /// Headline verifier time: median of the recorded warmed samples.
    verifier_ms: f64,
    verifier_mean_ms: f64,
    verifier_min_ms: f64,
    verifier_max_ms: f64,
    verifier_samples: usize,
    proof_bytes: usize,
    proof_zstd_bytes: usize,
}

#[derive(Clone, Debug)]
struct OgSha256InstanceSweepRow {
    algorithm: &'static str,
    status: &'static str,
    error: String,
    instances: usize,
    pcs_width: usize,
    active_rows: usize,
    domain_rows: usize,
    num_vars: usize,
    setup_ms: Option<f64>,
    prover_ms: Option<f64>,
    prover_mean_ms: Option<f64>,
    prover_min_ms: Option<f64>,
    prover_max_ms: Option<f64>,
    prover_samples: Option<usize>,
    verifier_ms: Option<f64>,
    verifier_mean_ms: Option<f64>,
    verifier_min_ms: Option<f64>,
    verifier_max_ms: Option<f64>,
    verifier_samples: Option<usize>,
    proof_bytes: Option<usize>,
    proof_zstd_bytes: Option<usize>,
}

#[derive(Clone, Debug)]
struct Sha256CombinedInstanceSweepRow {
    algorithm: String,
    variant: String,
    status: &'static str,
    error: String,
    instances: usize,
    ell: Option<usize>,
    l0: Option<usize>,
    tail_vars: Option<usize>,
    width: Option<usize>,
    active_rows: Option<usize>,
    domain_rows: Option<usize>,
    num_vars: Option<usize>,
    setup_ms: Option<f64>,
    trace_witness_ms: Option<f64>,
    pcs_setup_ms: Option<f64>,
    prepare_sumfold_basis_ms: Option<f64>,
    prepare_ms: Option<f64>,
    setup_prepare_ms: Option<f64>,
    probe_prove_ms: Option<f64>,
    probe_commit_ms: Option<f64>,
    probe_sumfold_linear_ms: Option<f64>,
    probe_sumfold_booleanity_ms: Option<f64>,
    probe_sumfold_group_ms: Option<f64>,
    probe_sumfold_prove_rounds_ms: Option<f64>,
    probe_sumfold_ms: Option<f64>,
    probe_fold_ms: Option<f64>,
    probe_open_ms: Option<f64>,
    probe_open_core_ms: Option<f64>,
    prover_ms: Option<f64>,
    prover_mean_ms: Option<f64>,
    prover_min_ms: Option<f64>,
    prover_max_ms: Option<f64>,
    prover_samples: Option<usize>,
    verifier_ms: Option<f64>,
    verifier_mean_ms: Option<f64>,
    verifier_min_ms: Option<f64>,
    verifier_max_ms: Option<f64>,
    verifier_samples: Option<usize>,
    proof_bytes: Option<usize>,
    proof_zstd_bytes: Option<usize>,
}

#[derive(Clone, Debug)]
struct SkippedPackedWidth {
    width: usize,
    reason: String,
}

#[derive(Clone, Copy, Debug)]
struct TimingStats {
    median_ms: f64,
    mean_ms: f64,
    min_ms: f64,
    max_ms: f64,
    samples: usize,
}

#[derive(Clone, Copy, Debug)]
struct HyraxSetupPrepareTimings {
    trace_witness_ms: f64,
    setup_ms: f64,
    prepare_sumfold_basis_ms: f64,
    prepare_ms: f64,
    setup_prepare_ms: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct HyraxProvePhaseTimings {
    prove_ms: f64,
    commit_ms: f64,
    sumfold_linear_ms: f64,
    sumfold_booleanity_ms: f64,
    sumfold_group_ms: f64,
    sumfold_prove_rounds_ms: f64,
    sumfold_ms: f64,
    fold_ms: f64,
    open_ms: f64,
    open_core_ms: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProvePhase {
    PrepareSumfoldBasis,
    FreshInstances,
    FreshCommitMixedHyraxInstances,
    FreshCommitMixedHyraxInstance,
    SumfoldAccumulators,
    SumfoldLinearAccumulator,
    SumfoldQuadraticPrefixAccumulator,
    SumfoldGroup,
    SumfoldProve,
    FoldAfterSumfold,
    PcsOpening,
    PcsOpenCore,
}

const PROVE_PHASE_COUNT: usize = 12;

impl ProvePhase {
    fn from_name(value: &str) -> Option<Self> {
        Some(match value {
            "prepare_sumfold_basis" => Self::PrepareSumfoldBasis,
            "fresh_instances" => Self::FreshInstances,
            "fresh_commit_mixed_hyrax_instances" => Self::FreshCommitMixedHyraxInstances,
            "fresh_commit_mixed_hyrax_instance" => Self::FreshCommitMixedHyraxInstance,
            "sumfold_accumulators" => Self::SumfoldAccumulators,
            "sumfold_linear_accumulator" => Self::SumfoldLinearAccumulator,
            "sumfold_quadratic_prefix_accumulator" => Self::SumfoldQuadraticPrefixAccumulator,
            "sumfold_group" => Self::SumfoldGroup,
            "sumfold_prove" => Self::SumfoldProve,
            "fold_after_sumfold" => Self::FoldAfterSumfold,
            "pcs_opening" => Self::PcsOpening,
            "pcs_open_core" => Self::PcsOpenCore,
            _ => return None,
        })
    }

    fn index(self) -> usize {
        match self {
            Self::PrepareSumfoldBasis => 0,
            Self::FreshInstances => 11,
            Self::FreshCommitMixedHyraxInstances => 1,
            Self::FreshCommitMixedHyraxInstance => 2,
            Self::SumfoldAccumulators => 3,
            Self::SumfoldLinearAccumulator => 4,
            Self::SumfoldQuadraticPrefixAccumulator => 5,
            Self::SumfoldGroup => 6,
            Self::SumfoldProve => 7,
            Self::FoldAfterSumfold => 8,
            Self::PcsOpening => 9,
            Self::PcsOpenCore => 10,
        }
    }
}

#[derive(Clone, Debug)]
struct PhaseSpanTiming {
    phase: Option<ProvePhase>,
    raw: String,
    entered_at: Option<Instant>,
}

#[derive(Clone, Default)]
struct PhaseTimingLayer {
    totals_ms: Arc<Mutex<[f64; PROVE_PHASE_COUNT]>>,
    // Every span's busy time keyed by its raw `phase` field, so sub-phases not
    // in the fixed `ProvePhase` set (decider/commit/fold internals) can be
    // profiled without extending the enum.
    all_phases_ms: Arc<Mutex<HashMap<String, f64>>>,
}

static PHASE_TIMING_LAYER: OnceLock<PhaseTimingLayer> = OnceLock::new();

#[derive(Default)]
struct PhaseFieldVisitor {
    phase: Option<ProvePhase>,
    raw: Option<String>,
}

const HYRAX_WIDTH_SWEEP_WARMUP_RUNS: usize = 2;
const HYRAX_WIDTH_SWEEP_TUNING_SAMPLES: usize = 5;
const HYRAX_WIDTH_SWEEP_CONFIRMATION_SAMPLES: usize = 15;
const HYRAX_WIDTH_SWEEP_CONFIRMATION_TOP_K: usize = 3;
const HYRAX_INSTANCE_SWEEP_WIDTH: usize = 1024;
const OG_SHA256_SWEEP_DEFAULT_HYRAX_WIDTH: usize = 1024;
const OG_SHA256_SWEEP_DEFAULT_WARMUP_RUNS: usize = 1;
const OG_SHA256_SWEEP_DEFAULT_SAMPLES: usize = 3;
const SHA256_COMBINED_SWEEP_DEFAULT_WARMUP_RUNS: usize = 1;
const SHA256_COMBINED_SWEEP_DEFAULT_SAMPLES: usize = 3;

impl TimingStats {
    fn from_samples(samples: &[f64]) -> Self {
        assert!(!samples.is_empty(), "timing stats require samples");
        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let midpoint = sorted.len() / 2;
        let median_ms = if sorted.len() % 2 == 0 {
            (sorted[midpoint - 1] + sorted[midpoint]) * 0.5
        } else {
            sorted[midpoint]
        };
        let mean_ms = samples.iter().sum::<f64>() / samples.len() as f64;
        let min_ms = samples
            .iter()
            .fold(f64::INFINITY, |min, value| min.min(*value));
        let max_ms = samples
            .iter()
            .fold(f64::NEG_INFINITY, |max, value| max.max(*value));
        Self {
            median_ms,
            mean_ms,
            min_ms,
            max_ms,
            samples: samples.len(),
        }
    }
}

impl Visit for PhaseFieldVisitor {
    fn record_str(&mut self, field: &TracingField, value: &str) {
        if field.name() == "phase" {
            self.phase = ProvePhase::from_name(value);
            self.raw = Some(value.to_owned());
        }
    }

    fn record_debug(&mut self, field: &TracingField, value: &dyn Debug) {
        if field.name() == "phase" {
            let value = format!("{value:?}");
            let value = value
                .strip_prefix('"')
                .and_then(|value| value.strip_suffix('"'))
                .unwrap_or(&value);
            self.phase = ProvePhase::from_name(value);
            self.raw = Some(value.to_owned());
        }
    }
}

impl PhaseTimingLayer {
    fn reset(&self) {
        let mut totals = self.totals_ms.lock().expect("phase timing mutex poisoned");
        *totals = [0.0; PROVE_PHASE_COUNT];
        self.all_phases_ms
            .lock()
            .expect("phase timing mutex poisoned")
            .clear();
    }

    /// Busy-time in ms for every observed `phase`, sorted descending.
    fn all_phases_sorted(&self) -> Vec<(String, f64)> {
        let map = self
            .all_phases_ms
            .lock()
            .expect("phase timing mutex poisoned");
        let mut entries: Vec<(String, f64)> =
            map.iter().map(|(name, ms)| (name.clone(), *ms)).collect();
        entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        entries
    }

    /// Busy-time in ms for a single raw phase name (0.0 if never seen).
    fn raw_phase_ms(&self, name: &str) -> f64 {
        self.all_phases_ms
            .lock()
            .expect("phase timing mutex poisoned")
            .get(name)
            .copied()
            .unwrap_or(0.0)
    }

    fn phase_ms(totals: &[f64; PROVE_PHASE_COUNT], phase: ProvePhase) -> f64 {
        totals[phase.index()]
    }

    fn total_ms(&self, phase: ProvePhase) -> f64 {
        let totals = self.totals_ms.lock().expect("phase timing mutex poisoned");
        Self::phase_ms(&totals, phase)
    }

    fn snapshot(&self, prove_ms: f64) -> HyraxProvePhaseTimings {
        let totals = self.totals_ms.lock().expect("phase timing mutex poisoned");
        let commit_wall_ms = Self::phase_ms(&totals, ProvePhase::FreshCommitMixedHyraxInstances);
        let commit_instance_ms = Self::phase_ms(&totals, ProvePhase::FreshCommitMixedHyraxInstance);
        HyraxProvePhaseTimings {
            prove_ms,
            commit_ms: if commit_wall_ms > 0.0 {
                commit_wall_ms
            } else {
                commit_instance_ms
            },
            sumfold_linear_ms: Self::phase_ms(&totals, ProvePhase::SumfoldLinearAccumulator),
            sumfold_booleanity_ms: Self::phase_ms(
                &totals,
                ProvePhase::SumfoldQuadraticPrefixAccumulator,
            ),
            sumfold_group_ms: Self::phase_ms(&totals, ProvePhase::SumfoldGroup),
            sumfold_prove_rounds_ms: Self::phase_ms(&totals, ProvePhase::SumfoldProve),
            sumfold_ms: Self::phase_ms(&totals, ProvePhase::SumfoldAccumulators)
                + Self::phase_ms(&totals, ProvePhase::SumfoldProve),
            fold_ms: Self::phase_ms(&totals, ProvePhase::FoldAfterSumfold),
            open_ms: Self::phase_ms(&totals, ProvePhase::PcsOpening),
            open_core_ms: Self::phase_ms(&totals, ProvePhase::PcsOpenCore),
        }
    }
}

fn phase_timing_layer() -> PhaseTimingLayer {
    let layer = PHASE_TIMING_LAYER
        .get_or_init(|| {
            let layer = PhaseTimingLayer::default();
            let subscriber = tracing_subscriber::registry().with(layer.clone());
            let _ = tracing::subscriber::set_global_default(subscriber);
            layer
        })
        .clone();
    layer.reset();
    layer
}

impl<S> tracing_subscriber::Layer<S> for PhaseTimingLayer
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    fn on_new_span(&self, attrs: &tracing::span::Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let mut visitor = PhaseFieldVisitor::default();
        attrs.record(&mut visitor);
        let Some(raw) = visitor.raw else {
            return;
        };
        let Some(span) = ctx.span(id) else {
            return;
        };
        span.extensions_mut().insert(PhaseSpanTiming {
            phase: visitor.phase,
            raw,
            entered_at: None,
        });
    }

    fn on_record(&self, id: &Id, values: &tracing::span::Record<'_>, ctx: Context<'_, S>) {
        let mut visitor = PhaseFieldVisitor::default();
        values.record(&mut visitor);
        let Some(raw) = visitor.raw else {
            return;
        };
        let Some(span) = ctx.span(id) else {
            return;
        };
        let mut extensions = span.extensions_mut();
        if let Some(timing) = extensions.get_mut::<PhaseSpanTiming>() {
            timing.phase = visitor.phase;
            timing.raw = raw;
        }
    }

    fn on_enter(&self, id: &Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(id) else {
            return;
        };
        let mut extensions = span.extensions_mut();
        if let Some(timing) = extensions.get_mut::<PhaseSpanTiming>() {
            timing.entered_at = Some(Instant::now());
        }
    }

    fn on_exit(&self, id: &Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(id) else {
            return;
        };
        let mut extensions = span.extensions_mut();
        let Some(timing) = extensions.get_mut::<PhaseSpanTiming>() else {
            return;
        };
        let Some(start) = timing.entered_at.take() else {
            return;
        };
        let elapsed_ms = elapsed_ms(start);
        if let Some(phase) = timing.phase {
            let mut totals = self.totals_ms.lock().expect("phase timing mutex poisoned");
            totals[phase.index()] += elapsed_ms;
        }
        *self
            .all_phases_ms
            .lock()
            .expect("phase timing mutex poisoned")
            .entry(timing.raw.clone())
            .or_insert(0.0) += elapsed_ms;
    }
}

#[allow(clippy::too_many_arguments)]
fn hyrax_width_sweep_row(
    category: HyraxReportCategory,
    variant: String,
    width: Option<usize>,
    ecc_points_per_commitment: Option<usize>,
    ecc_points_per_proof: Option<usize>,
    prover: TimingStats,
    verifier: TimingStats,
    proof_bytes: usize,
    proof_zstd_bytes: usize,
) -> HyraxWidthSweepRow {
    HyraxWidthSweepRow {
        algorithm: category.algorithm(),
        variant,
        width,
        ecc_points_per_commitment,
        ecc_points_per_proof,
        prover_ms: prover.median_ms,
        prover_mean_ms: prover.mean_ms,
        prover_min_ms: prover.min_ms,
        prover_max_ms: prover.max_ms,
        prover_samples: prover.samples,
        verifier_ms: verifier.median_ms,
        verifier_mean_ms: verifier.mean_ms,
        verifier_min_ms: verifier.min_ms,
        verifier_max_ms: verifier.max_ms,
        verifier_samples: verifier.samples,
        proof_bytes,
        proof_zstd_bytes,
        category,
    }
}

#[allow(clippy::too_many_arguments)]
fn hyrax_instance_sweep_row(
    instances: usize,
    ell: usize,
    l0: usize,
    width: usize,
    ecc_points_per_commitment: usize,
    ecc_points_per_proof: usize,
    setup_prepare: HyraxSetupPrepareTimings,
    prove_phases: HyraxProvePhaseTimings,
    prover: TimingStats,
    verifier: TimingStats,
    proof_bytes: usize,
    proof_zstd_bytes: usize,
) -> HyraxInstanceSweepRow {
    HyraxInstanceSweepRow {
        algorithm: HyraxReportCategory::ProjectionFoldMixedHyrax.algorithm(),
        variant: format!("l0 {l0} instances {instances}"),
        instances,
        ell,
        l0,
        tail_vars: ell.saturating_sub(l0),
        width,
        ecc_points_per_commitment,
        ecc_points_per_proof,
        trace_witness_ms: setup_prepare.trace_witness_ms,
        setup_ms: setup_prepare.setup_ms,
        prepare_sumfold_basis_ms: setup_prepare.prepare_sumfold_basis_ms,
        prepare_ms: setup_prepare.prepare_ms,
        setup_prepare_ms: setup_prepare.setup_prepare_ms,
        probe_prove_ms: prove_phases.prove_ms,
        probe_commit_ms: prove_phases.commit_ms,
        probe_sumfold_linear_ms: prove_phases.sumfold_linear_ms,
        probe_sumfold_booleanity_ms: prove_phases.sumfold_booleanity_ms,
        probe_sumfold_group_ms: prove_phases.sumfold_group_ms,
        probe_sumfold_prove_rounds_ms: prove_phases.sumfold_prove_rounds_ms,
        probe_sumfold_ms: prove_phases.sumfold_ms,
        probe_fold_ms: prove_phases.fold_ms,
        probe_open_ms: prove_phases.open_ms,
        probe_open_core_ms: prove_phases.open_core_ms,
        prover_ms: prover.median_ms,
        prover_mean_ms: prover.mean_ms,
        prover_min_ms: prover.min_ms,
        prover_max_ms: prover.max_ms,
        prover_samples: prover.samples,
        verifier_ms: verifier.median_ms,
        verifier_mean_ms: verifier.mean_ms,
        verifier_min_ms: verifier.min_ms,
        verifier_max_ms: verifier.max_ms,
        verifier_samples: verifier.samples,
        proof_bytes,
        proof_zstd_bytes,
    }
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1_000.0
}

fn zstd_len(raw: &[u8]) -> usize {
    zstd::encode_all(raw, zip_plus::utils::ZSTD_LEVEL)
        .expect("zstd compression failed")
        .len()
}

fn append_len_prefixed_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    let len = u64::try_from(bytes.len()).expect("proof component length fits u64");
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(bytes);
}

fn production_packed_hyrax_proof_raw_bytes<C, Fp>(
    proof: &ProductionShaPackedHyraxProof<C, Fp>,
) -> Vec<u8>
where
    C: AffineRepr,
    Fp: PrimeField,
    zinc_piop::combined_poly_resolver::Proof<Fp>: Transcribable,
    zinc_piop::ideal_check::Proof<Fp>: Transcribable,
    zinc_piop::multipoint_eval::Proof<Fp>: Transcribable,
    zinc_piop::sumcheck::multi_degree::MultiDegreeSumcheckProof<Fp>: Transcribable,
    DynamicPolyVecF<Fp>: Transcribable,
{
    let mut out = Vec::new();
    out.extend_from_slice(&(proof.instance_commitments.len() as u64).to_le_bytes());
    for commitment in &proof.instance_commitments {
        commitment.write_bytes(&mut out);
    }
    append_transcribable_bytes(&mut out, &proof.ideal_check);
    append_transcribable_bytes(&mut out, &proof.sumfold_proof);
    append_transcribable_bytes(&mut out, &proof.resolver);
    append_transcribable_bytes(&mut out, &proof.combined_sumcheck);
    append_transcribable_bytes(&mut out, &proof.multipoint_eval);
    append_transcribable_bytes(
        &mut out,
        DynamicPolyVecF::reinterpret(&proof.witness_lifted_evals),
    );
    append_len_prefixed_bytes(&mut out, &proof.opening_proof);
    out
}

fn production_mixed_hyrax_proof_raw_bytes<C, Fp>(
    proof: &ProductionShaMixedHyraxProof<C, Fp>,
) -> Vec<u8>
where
    C: AffineRepr,
    Fp: PrimeField,
    zinc_piop::combined_poly_resolver::Proof<Fp>: Transcribable,
    zinc_piop::ideal_check::Proof<Fp>: Transcribable,
    zinc_piop::multipoint_eval::Proof<Fp>: Transcribable,
    zinc_piop::sumcheck::multi_degree::MultiDegreeSumcheckProof<Fp>: Transcribable,
    DynamicPolyVecF<Fp>: Transcribable,
{
    let mut out = Vec::new();
    out.extend_from_slice(&(proof.instance_commitments.len() as u64).to_le_bytes());
    for commitment in &proof.instance_commitments {
        commitment.write_bytes(&mut out);
    }
    append_transcribable_bytes(&mut out, &proof.ideal_check);
    append_transcribable_bytes(&mut out, &proof.sumfold_proof);
    append_transcribable_bytes(&mut out, &proof.resolver);
    append_transcribable_bytes(&mut out, &proof.combined_sumcheck);
    append_transcribable_bytes(&mut out, &proof.multipoint_eval);
    append_transcribable_bytes(
        &mut out,
        DynamicPolyVecF::reinterpret(&proof.witness_lifted_evals),
    );
    append_len_prefixed_bytes(&mut out, &proof.opening_proof);
    out
}

fn fold_first_mixed_hyrax_proof_raw_bytes<C, Fp>(
    proof: &zinc_protocol::production_sha::FoldFirstShaMixedHyraxProof<C, Fp>,
) -> Vec<u8>
where
    C: AffineRepr,
    Fp: PrimeField,
    zinc_piop::combined_poly_resolver::Proof<Fp>: Transcribable,
    zinc_piop::multipoint_eval::Proof<Fp>: Transcribable,
    zinc_piop::sumcheck::multi_degree::MultiDegreeSumcheckProof<Fp>: Transcribable,
    DynamicPolyVecF<Fp>: Transcribable,
{
    let mut out = Vec::new();
    out.extend_from_slice(&(proof.instance_commitments.len() as u64).to_le_bytes());
    for commitment in &proof.instance_commitments {
        commitment.write_bytes(&mut out);
    }
    let skip_round_values = vec![DynamicPolynomialF {
        coeffs: proof.skip_round.node_values.clone(),
    }];
    append_transcribable_bytes(&mut out, DynamicPolyVecF::reinterpret(&skip_round_values));
    let folded_ideal_polys = proof.folded_ideal_polys.to_vec();
    append_transcribable_bytes(&mut out, DynamicPolyVecF::reinterpret(&folded_ideal_polys));
    append_transcribable_bytes(&mut out, &proof.resolver);
    append_transcribable_bytes(&mut out, &proof.combined_sumcheck);
    append_transcribable_bytes(&mut out, &proof.multipoint_eval);
    append_transcribable_bytes(
        &mut out,
        DynamicPolyVecF::reinterpret(&proof.witness_lifted_evals),
    );
    append_len_prefixed_bytes(&mut out, &proof.opening_proof);
    out
}

fn measure_warmed<T>(
    warmup_runs: usize,
    sample_count: usize,
    mut f: impl FnMut() -> T,
) -> (T, TimingStats) {
    assert!(sample_count > 0, "warmed measurement requires samples");
    for _ in 0..warmup_runs {
        black_box(f());
    }

    let mut samples = Vec::with_capacity(sample_count);
    let mut last = None;
    for _ in 0..sample_count {
        let start = Instant::now();
        let value = black_box(f());
        samples.push(elapsed_ms(start));
        last = Some(value);
    }

    (
        last.expect("sample_count is non-zero"),
        TimingStats::from_samples(&samples),
    )
}

fn measure_mixed_hyrax_prove_phase_timings<C, U, Zt, F, const D: usize>(
    pp: &LinearIdealFoldProverParams<AllHyraxPCSTypes<C>, U, Zt, F, D>,
    shape: &UairShape<U>,
    prepared_instances: &[PreparedProductionShaProverInstance<Zt, F, D>],
) -> HyraxProvePhaseTimings
where
    U: Uair,
    Zt: ZincTypes<D>,
    F: InnerTransparentField
        + DelayedFieldProductSum
        + ShaBinaryFoldField
        + FromPrimitiveWithConfig
        + zip_plus::pcs::hyrax::HyraxFieldBridge<C>
        + Send
        + Sync
        + 'static
        + ShaLinearAccumulatorField
        + ShaSuffixScannerField,
    F::Inner: Transcribable + Default + Send + Sync,
    F::Modulus: Transcribable,
    C: ProductionShaMixedHyraxPcs<Zt, F, D>,
    DensePolyScalarLanes: zip_plus::pcs::hyrax::HyraxLanes<C, DensePolynomial<Zt::Int, D>, D>,
    IntScalarLane: zip_plus::pcs::hyrax::HyraxLanes<C, Zt::Int, D>,
{
    let layer = phase_timing_layer();
    let start = Instant::now();
    let mut transcript = Blake3Transcript::new();
    prove_prepared_linear_ideal_fold_mixed_hyrax::<C, U, Zt, F, D>(
        pp,
        shape,
        prepared_instances,
        v1_booleanity_catalog(),
        &mut transcript,
    )
    .expect("mixed Hyrax phase-timing probe failed");
    layer.snapshot(elapsed_ms(start))
}

trait CsvValue {
    fn write_csv(self, out: &mut String);
}

impl CsvValue for &str {
    fn write_csv(self, out: &mut String) {
        if self.chars().any(|ch| matches!(ch, ',' | '"' | '\n')) {
            out.push('"');
            for ch in self.chars() {
                if ch == '"' {
                    out.push('"');
                }
                out.push(ch);
            }
            out.push('"');
        } else {
            out.push_str(self);
        }
    }
}

impl CsvValue for String {
    fn write_csv(self, out: &mut String) {
        self.as_str().write_csv(out);
    }
}

impl CsvValue for &String {
    fn write_csv(self, out: &mut String) {
        self.as_str().write_csv(out);
    }
}

impl CsvValue for usize {
    fn write_csv(self, out: &mut String) {
        write!(out, "{self}").expect("write to String cannot fail");
    }
}

impl CsvValue for f64 {
    fn write_csv(self, out: &mut String) {
        write!(out, "{self:.6}").expect("write to String cannot fail");
    }
}

impl CsvValue for Option<usize> {
    fn write_csv(self, out: &mut String) {
        match self {
            Some(value) => value.write_csv(out),
            None => out.push_str("N/A"),
        }
    }
}

impl CsvValue for Option<f64> {
    fn write_csv(self, out: &mut String) {
        match self {
            Some(value) => value.write_csv(out),
            None => out.push_str("N/A"),
        }
    }
}

macro_rules! push_csv_row {
    ($csv:expr, $first:expr $(, $rest:expr)* $(,)?) => {{
        ($first).write_csv(&mut $csv);
        $(
            $csv.push(',');
            ($rest).write_csv(&mut $csv);
        )*
        $csv.push('\n');
    }};
}

fn write_hyrax_width_sweep_csv(path: &Path, rows: &[HyraxWidthSweepRow]) {
    let mut csv = String::from(
        "algorithm,variant,width,ecc_points_per_commitment,ecc_points_per_proof,prover_median_ms,prover_mean_ms,prover_min_ms,prover_max_ms,prover_samples,verifier_median_ms,verifier_mean_ms,verifier_min_ms,verifier_max_ms,verifier_samples,proof_bytes,proof_zstd_bytes\n",
    );
    for row in rows {
        push_csv_row!(
            csv,
            row.algorithm,
            row.variant.as_str(),
            row.width,
            row.ecc_points_per_commitment,
            row.ecc_points_per_proof,
            row.prover_ms,
            row.prover_mean_ms,
            row.prover_min_ms,
            row.prover_max_ms,
            row.prover_samples,
            row.verifier_ms,
            row.verifier_mean_ms,
            row.verifier_min_ms,
            row.verifier_max_ms,
            row.verifier_samples,
            row.proof_bytes,
            row.proof_zstd_bytes
        );
    }
    fs::write(path, csv).expect("write hyrax width sweep CSV");
}

fn write_hyrax_instance_sweep_csv(path: &Path, rows: &[HyraxInstanceSweepRow]) {
    let mut csv = String::from(
        "algorithm,variant,instances,ell,l0,tail_vars,width,ecc_points_per_commitment,ecc_points_per_proof,trace_witness_ms,setup_ms,prepare_sumfold_basis_ms,prepare_ms,setup_prepare_ms,probe_prove_ms,probe_commit_ms,probe_sumfold_linear_ms,probe_sumfold_booleanity_ms,probe_sumfold_group_ms,probe_sumfold_prove_rounds_ms,probe_sumfold_ms,probe_fold_ms,probe_open_ms,probe_open_core_ms,prover_median_ms,prover_mean_ms,prover_min_ms,prover_max_ms,prover_samples,verifier_median_ms,verifier_mean_ms,verifier_min_ms,verifier_max_ms,verifier_samples,proof_bytes,proof_zstd_bytes\n",
    );
    for row in rows {
        push_csv_row!(
            csv,
            row.algorithm,
            row.variant.as_str(),
            row.instances,
            row.ell,
            row.l0,
            row.tail_vars,
            row.width,
            row.ecc_points_per_commitment,
            row.ecc_points_per_proof,
            row.trace_witness_ms,
            row.setup_ms,
            row.prepare_sumfold_basis_ms,
            row.prepare_ms,
            row.setup_prepare_ms,
            row.probe_prove_ms,
            row.probe_commit_ms,
            row.probe_sumfold_linear_ms,
            row.probe_sumfold_booleanity_ms,
            row.probe_sumfold_group_ms,
            row.probe_sumfold_prove_rounds_ms,
            row.probe_sumfold_ms,
            row.probe_fold_ms,
            row.probe_open_ms,
            row.probe_open_core_ms,
            row.prover_ms,
            row.prover_mean_ms,
            row.prover_min_ms,
            row.prover_max_ms,
            row.prover_samples,
            row.verifier_ms,
            row.verifier_mean_ms,
            row.verifier_min_ms,
            row.verifier_max_ms,
            row.verifier_samples,
            row.proof_bytes,
            row.proof_zstd_bytes
        );
    }
    fs::write(path, csv).expect("write Hyrax instance sweep CSV");
}

fn write_og_sha256_instance_sweep_csv(path: &Path, rows: &[OgSha256InstanceSweepRow]) {
    let mut csv = String::from(
        "algorithm,status,error,instances,pcs_width,active_rows,domain_rows,num_vars,setup_ms,prover_median_ms,prover_mean_ms,prover_min_ms,prover_max_ms,prover_samples,verifier_median_ms,verifier_mean_ms,verifier_min_ms,verifier_max_ms,verifier_samples,proof_bytes,proof_zstd_bytes\n",
    );
    for row in rows {
        push_csv_row!(
            csv,
            row.algorithm,
            row.status,
            row.error.as_str(),
            row.instances,
            row.pcs_width,
            row.active_rows,
            row.domain_rows,
            row.num_vars,
            row.setup_ms,
            row.prover_ms,
            row.prover_mean_ms,
            row.prover_min_ms,
            row.prover_max_ms,
            row.prover_samples,
            row.verifier_ms,
            row.verifier_mean_ms,
            row.verifier_min_ms,
            row.verifier_max_ms,
            row.verifier_samples,
            row.proof_bytes,
            row.proof_zstd_bytes
        );
    }
    fs::write(path, csv).expect("write OG SHA-256 instance sweep CSV");
}

fn write_sha256_combined_instance_sweep_csv(path: &Path, rows: &[Sha256CombinedInstanceSweepRow]) {
    let mut csv = String::from(
        "algorithm,variant,status,error,instances,ell,l0,tail_vars,width,active_rows,domain_rows,num_vars,setup_ms,trace_witness_ms,pcs_setup_ms,prepare_sumfold_basis_ms,prepare_ms,setup_prepare_ms,probe_prove_ms,probe_commit_ms,probe_sumfold_linear_ms,probe_sumfold_booleanity_ms,probe_sumfold_group_ms,probe_sumfold_prove_rounds_ms,probe_sumfold_ms,probe_fold_ms,probe_open_ms,probe_open_core_ms,prover_median_ms,prover_mean_ms,prover_min_ms,prover_max_ms,prover_samples,verifier_median_ms,verifier_mean_ms,verifier_min_ms,verifier_max_ms,verifier_samples,proof_bytes,proof_zstd_bytes\n",
    );
    for row in rows {
        push_csv_row!(
            csv,
            row.algorithm.as_str(),
            row.variant.as_str(),
            row.status,
            row.error.as_str(),
            row.instances,
            row.ell,
            row.l0,
            row.tail_vars,
            row.width,
            row.active_rows,
            row.domain_rows,
            row.num_vars,
            row.setup_ms,
            row.trace_witness_ms,
            row.pcs_setup_ms,
            row.prepare_sumfold_basis_ms,
            row.prepare_ms,
            row.setup_prepare_ms,
            row.probe_prove_ms,
            row.probe_commit_ms,
            row.probe_sumfold_linear_ms,
            row.probe_sumfold_booleanity_ms,
            row.probe_sumfold_group_ms,
            row.probe_sumfold_prove_rounds_ms,
            row.probe_sumfold_ms,
            row.probe_fold_ms,
            row.probe_open_ms,
            row.probe_open_core_ms,
            row.prover_ms,
            row.prover_mean_ms,
            row.prover_min_ms,
            row.prover_max_ms,
            row.prover_samples,
            row.verifier_ms,
            row.verifier_mean_ms,
            row.verifier_min_ms,
            row.verifier_max_ms,
            row.verifier_samples,
            row.proof_bytes,
            row.proof_zstd_bytes
        );
    }
    fs::write(path, csv).expect("write combined SHA-256 instance sweep CSV");
}

fn write_hyrax_width_sweep_skipped_csv(path: &Path, skipped: &[SkippedPackedWidth]) {
    let mut csv = String::from("width,reason\n");
    for width in skipped {
        csv.push_str(&format!("{},{}\n", width.width, width.reason));
    }
    fs::write(path, csv).expect("write skipped packed widths CSV");
}

fn packed_width_label(width: usize) -> String {
    match width {
        35 => "549/16".to_string(),
        69 => "549/8".to_string(),
        138 => "549/4".to_string(),
        275 => "549/2".to_string(),
        549 => "549".to_string(),
        1098 => "549*2".to_string(),
        2196 => "549*4".to_string(),
        4392 => "549*8".to_string(),
        8784 => "549*16".to_string(),
        17568 => "549*32".to_string(),
        35136 => "549*64".to_string(),
        70272 => "549*128".to_string(),
        _ => width.to_string(),
    }
}

fn packed_width_candidates<F>() -> (Vec<(String, usize)>, Vec<SkippedPackedWidth>)
where
    F: PrimeField,
{
    let mut requested = vec![
        18usize, 35, 36, 69, 70, 96, 137, 138, 139, 183, 274, 275, 276, 549, 1098, 2196, 4392,
        8784, 17568, 35136, 70272,
    ];
    requested.sort_unstable();
    requested.dedup();

    let mut valid = Vec::new();
    let mut skipped = Vec::new();
    for width in requested {
        match packed_sha_layout::<F>(width) {
            Ok(_) => valid.push((packed_width_label(width), width)),
            Err(error) => skipped.push(SkippedPackedWidth {
                width,
                reason: error.to_string(),
            }),
        }
    }
    (valid, skipped)
}

fn mixed_width_candidates() -> Vec<(String, usize)> {
    [8usize, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        .into_iter()
        .map(|width| (format!("width {width}"), width))
        .collect()
}

fn finite_range(values: impl Iterator<Item = f64>) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for value in values.filter(|value| value.is_finite()) {
        min = min.min(value);
        max = max.max(value);
    }
    if !min.is_finite() || !max.is_finite() {
        return (0.0, 1.0);
    }
    if (max - min).abs() < f64::EPSILON {
        (min * 0.9, max * 1.1 + 1.0)
    } else {
        let pad = (max - min) * 0.08;
        (min - pad, max + pad)
    }
}

fn svg_polyline(
    points: &[(f64, f64)],
    color: &str,
    xmap: &dyn Fn(f64) -> f64,
    ymap: &dyn Fn(f64) -> f64,
) -> String {
    let coords = points
        .iter()
        .map(|(x, y)| format!("{:.2},{:.2}", xmap(*x), ymap(*y)))
        .collect::<Vec<_>>()
        .join(" ");
    format!(r#"<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{coords}"/>"#)
}

#[derive(Clone, Copy)]
struct Rgb(u8, u8, u8);

const PNG_WIDTH: usize = 980;
const PNG_HEIGHT: usize = 560;
const PNG_AXIS: Rgb = Rgb(34, 34, 34);
const PNG_PROVER: Rgb = Rgb(15, 118, 110);
const PNG_VERIFIER: Rgb = Rgb(124, 58, 237);
const PNG_PROOF: Rgb = Rgb(37, 99, 235);
const PNG_MIXED: Rgb = Rgb(220, 38, 38);
const PNG_OG: Rgb = Rgb(17, 24, 39);
const PNG_SCATTER: Rgb = Rgb(249, 115, 22);

fn blank_png_canvas() -> Vec<u8> {
    vec![255u8; PNG_WIDTH * PNG_HEIGHT * 3]
}

fn put_png_pixel(pixels: &mut [u8], x: i32, y: i32, color: Rgb) {
    if x < 0 || y < 0 || x >= PNG_WIDTH as i32 || y >= PNG_HEIGHT as i32 {
        return;
    }
    let idx = (y as usize * PNG_WIDTH + x as usize) * 3;
    pixels[idx] = color.0;
    pixels[idx + 1] = color.1;
    pixels[idx + 2] = color.2;
}

fn draw_png_line(pixels: &mut [u8], mut x0: i32, mut y0: i32, x1: i32, y1: i32, color: Rgb) {
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        put_png_pixel(pixels, x0, y0, color);
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

fn draw_png_axes(pixels: &mut [u8]) {
    draw_png_line(pixels, 80, 500, 940, 500, PNG_AXIS);
    draw_png_line(pixels, 80, 70, 80, 500, PNG_AXIS);
}

fn draw_png_dashed_hline(pixels: &mut [u8], y: f64, color: Rgb) {
    let y = y.round() as i32;
    let mut x = 80;
    while x < 940 {
        draw_png_line(pixels, x, y, (x + 14).min(940), y, color);
        x += 24;
    }
}

fn draw_png_filled_circle(pixels: &mut [u8], cx: f64, cy: f64, radius: f64, color: Rgb) {
    let cx = cx.round() as i32;
    let cy = cy.round() as i32;
    let radius = radius.round().max(1.0) as i32;
    for y in -radius..=radius {
        for x in -radius..=radius {
            if x * x + y * y <= radius * radius {
                put_png_pixel(pixels, cx + x, cy + y, color);
            }
        }
    }
}

fn draw_png_polyline(
    pixels: &mut [u8],
    points: &[(f64, f64)],
    color: Rgb,
    xmap: &dyn Fn(f64) -> f64,
    ymap: &dyn Fn(f64) -> f64,
) {
    for pair in points.windows(2) {
        let (x0, y0) = pair[0];
        let (x1, y1) = pair[1];
        draw_png_line(
            pixels,
            xmap(x0).round() as i32,
            ymap(y0).round() as i32,
            xmap(x1).round() as i32,
            ymap(y1).round() as i32,
            color,
        );
    }
    for (x, y) in points {
        draw_png_filled_circle(pixels, xmap(*x), ymap(*y), 3.5, color);
    }
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            let mask = 0u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

fn adler32(bytes: &[u8]) -> u32 {
    const MOD: u32 = 65_521;
    let mut a = 1u32;
    let mut b = 0u32;
    for byte in bytes {
        a = (a + u32::from(*byte)) % MOD;
        b = (b + a) % MOD;
    }
    (b << 16) | a
}

fn push_png_chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let mut crc_input = Vec::with_capacity(kind.len() + data.len());
    crc_input.extend_from_slice(kind);
    crc_input.extend_from_slice(data);
    out.extend_from_slice(&crc32(&crc_input).to_be_bytes());
}

fn zlib_stored_blocks(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + data.len() / 65_535 * 5 + 8);
    out.extend_from_slice(&[0x78, 0x01]);
    let mut remaining = data;
    while !remaining.is_empty() {
        let chunk_len = remaining.len().min(65_535);
        let final_block = chunk_len == remaining.len();
        out.push(if final_block { 0x01 } else { 0x00 });
        let len = chunk_len as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(&remaining[..chunk_len]);
        remaining = &remaining[chunk_len..];
    }
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

fn write_png(path: &Path, pixels: &[u8]) {
    assert_eq!(pixels.len(), PNG_WIDTH * PNG_HEIGHT * 3);
    let mut scanlines = Vec::with_capacity((PNG_WIDTH * 3 + 1) * PNG_HEIGHT);
    for row in pixels.chunks(PNG_WIDTH * 3) {
        scanlines.push(0);
        scanlines.extend_from_slice(row);
    }

    let mut png = Vec::new();
    png.extend_from_slice(b"\x89PNG\r\n\x1a\n");
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&(PNG_WIDTH as u32).to_be_bytes());
    ihdr.extend_from_slice(&(PNG_HEIGHT as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    push_png_chunk(&mut png, b"IHDR", &ihdr);
    push_png_chunk(&mut png, b"IDAT", &zlib_stored_blocks(&scanlines));
    push_png_chunk(&mut png, b"IEND", &[]);
    fs::write(path, png).expect("write hyrax sweep PNG plot");
}

fn write_svg(path: &Path, title: &str, body: &str) {
    let svg = format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="980" height="560" viewBox="0 0 980 560">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="32" y="36" font-family="sans-serif" font-size="22" font-weight="700">{title}</text>
<line x1="80" y1="500" x2="940" y2="500" stroke="#222"/>
<line x1="80" y1="70" x2="80" y2="500" stroke="#222"/>
{body}
</svg>"##
    );
    fs::write(path, svg).expect("write hyrax sweep SVG plot");
}

fn write_hyrax_width_sweep_plots(out_dir: &Path, rows: &[HyraxWidthSweepRow]) {
    fn width_points(
        rows: &[&HyraxWidthSweepRow],
        metric: impl Fn(&HyraxWidthSweepRow) -> f64,
    ) -> Vec<(f64, f64)> {
        rows.iter()
            .filter_map(|row| row.width.map(|width| (width as f64, metric(row))))
            .collect()
    }

    fn category_svg_color(category: HyraxReportCategory) -> &'static str {
        match category {
            HyraxReportCategory::OgHash => "#111827",
            HyraxReportCategory::OgMixedHyrax => "#f97316",
            HyraxReportCategory::ProjectionFoldMixedHyrax => "#dc2626",
            HyraxReportCategory::ProjectionFoldPackedHyrax => "#2563eb",
        }
    }

    fn category_png_color(category: HyraxReportCategory) -> Rgb {
        match category {
            HyraxReportCategory::OgHash => PNG_OG,
            HyraxReportCategory::OgMixedHyrax => PNG_SCATTER,
            HyraxReportCategory::ProjectionFoldMixedHyrax => PNG_MIXED,
            HyraxReportCategory::ProjectionFoldPackedHyrax => PNG_PROOF,
        }
    }

    let packed = rows
        .iter()
        .filter(|row| row.category == HyraxReportCategory::ProjectionFoldPackedHyrax)
        .collect::<Vec<_>>();
    let projection_mixed = rows
        .iter()
        .filter(|row| row.category == HyraxReportCategory::ProjectionFoldMixedHyrax)
        .collect::<Vec<_>>();
    let og_mixed = rows
        .iter()
        .filter(|row| row.category == HyraxReportCategory::OgMixedHyrax)
        .collect::<Vec<_>>();
    let og_hash = rows
        .iter()
        .find(|row| row.category == HyraxReportCategory::OgHash);
    let width_rows = rows
        .iter()
        .filter(|row| row.width.is_some())
        .collect::<Vec<_>>();

    let width_min = width_rows
        .iter()
        .filter_map(|row| row.width)
        .min()
        .unwrap_or(1) as f64;
    let width_max = width_rows
        .iter()
        .filter_map(|row| row.width)
        .max()
        .unwrap_or(2) as f64;
    let xlog =
        |x: f64| 80.0 + ((x.ln() - width_min.ln()) / (width_max.ln() - width_min.ln())) * 860.0;

    let (time_min, time_max) = finite_range(
        width_rows
            .iter()
            .flat_map(|row| [row.prover_ms, row.verifier_ms])
            .chain(
                og_hash
                    .iter()
                    .flat_map(|row| [row.prover_ms, row.verifier_ms]),
            ),
    );
    let ytime = |y: f64| 500.0 - ((y - time_min) / (time_max - time_min)) * 430.0;
    let packed_prover_points = width_points(&packed, |row| row.prover_ms);
    let packed_verifier_points = width_points(&packed, |row| row.verifier_ms);
    let projection_mixed_prover_points = width_points(&projection_mixed, |row| row.prover_ms);
    let projection_mixed_verifier_points = width_points(&projection_mixed, |row| row.verifier_ms);
    let og_mixed_prover_points = width_points(&og_mixed, |row| row.prover_ms);
    let og_mixed_verifier_points = width_points(&og_mixed, |row| row.verifier_ms);
    let mut body = String::new();
    body.push_str(&svg_polyline(
        &packed_prover_points,
        "#2563eb",
        &xlog,
        &ytime,
    ));
    body.push_str(&svg_polyline(
        &packed_verifier_points,
        "#60a5fa",
        &xlog,
        &ytime,
    ));
    body.push_str(&svg_polyline(
        &projection_mixed_prover_points,
        "#dc2626",
        &xlog,
        &ytime,
    ));
    body.push_str(&svg_polyline(
        &projection_mixed_verifier_points,
        "#fca5a5",
        &xlog,
        &ytime,
    ));
    body.push_str(&svg_polyline(
        &og_mixed_prover_points,
        "#f97316",
        &xlog,
        &ytime,
    ));
    body.push_str(&svg_polyline(
        &og_mixed_verifier_points,
        "#facc15",
        &xlog,
        &ytime,
    ));
    let mut png = blank_png_canvas();
    draw_png_axes(&mut png);
    draw_png_polyline(&mut png, &packed_prover_points, PNG_PROOF, &xlog, &ytime);
    draw_png_polyline(
        &mut png,
        &packed_verifier_points,
        Rgb(96, 165, 250),
        &xlog,
        &ytime,
    );
    draw_png_polyline(
        &mut png,
        &projection_mixed_prover_points,
        PNG_MIXED,
        &xlog,
        &ytime,
    );
    draw_png_polyline(
        &mut png,
        &projection_mixed_verifier_points,
        Rgb(252, 165, 165),
        &xlog,
        &ytime,
    );
    draw_png_polyline(
        &mut png,
        &og_mixed_prover_points,
        PNG_SCATTER,
        &xlog,
        &ytime,
    );
    draw_png_polyline(
        &mut png,
        &og_mixed_verifier_points,
        Rgb(250, 204, 21),
        &xlog,
        &ytime,
    );
    if let Some(row) = og_hash {
        body.push_str(&format!(
            r##"<line x1="80" x2="940" y1="{:.2}" y2="{:.2}" stroke="#0f766e" stroke-dasharray="6 4"/><line x1="80" x2="940" y1="{:.2}" y2="{:.2}" stroke="#7c3aed" stroke-dasharray="6 4"/>"##,
            ytime(row.prover_ms),
            ytime(row.prover_ms),
            ytime(row.verifier_ms),
            ytime(row.verifier_ms)
        ));
        draw_png_dashed_hline(&mut png, ytime(row.prover_ms), PNG_PROVER);
        draw_png_dashed_hline(&mut png, ytime(row.verifier_ms), PNG_VERIFIER);
    }
    body.push_str(r##"<text x="650" y="82" font-family="sans-serif" font-size="13" fill="#2563eb">PF packed</text><text x="650" y="102" font-family="sans-serif" font-size="13" fill="#dc2626">PF mixed</text><text x="650" y="122" font-family="sans-serif" font-size="13" fill="#f97316">OG mixed</text><text x="650" y="142" font-family="sans-serif" font-size="13" fill="#111827">OG hash dashed</text>"##);
    write_svg(
        &out_dir.join("plot1_time_vs_width.svg"),
        "Prover/Verifier Time vs Width",
        &body,
    );
    write_png(&out_dir.join("plot1_time_vs_width.png"), &png);

    let (proof_min, proof_max) = finite_range(rows.iter().map(|row| row.proof_bytes as f64));
    let yproof = |y: f64| 500.0 - ((y - proof_min) / (proof_max - proof_min)) * 430.0;
    let packed_proof_points = width_points(&packed, |row| row.proof_bytes as f64);
    let projection_mixed_proof_points =
        width_points(&projection_mixed, |row| row.proof_bytes as f64);
    let og_mixed_proof_points = width_points(&og_mixed, |row| row.proof_bytes as f64);
    let mut body = String::new();
    body.push_str(&svg_polyline(
        &packed_proof_points,
        "#2563eb",
        &xlog,
        &yproof,
    ));
    body.push_str(&svg_polyline(
        &projection_mixed_proof_points,
        "#dc2626",
        &xlog,
        &yproof,
    ));
    body.push_str(&svg_polyline(
        &og_mixed_proof_points,
        "#f97316",
        &xlog,
        &yproof,
    ));
    let mut png = blank_png_canvas();
    draw_png_axes(&mut png);
    draw_png_polyline(&mut png, &packed_proof_points, PNG_PROOF, &xlog, &yproof);
    draw_png_polyline(
        &mut png,
        &projection_mixed_proof_points,
        PNG_MIXED,
        &xlog,
        &yproof,
    );
    draw_png_polyline(
        &mut png,
        &og_mixed_proof_points,
        PNG_SCATTER,
        &xlog,
        &yproof,
    );
    if let Some(row) = og_hash {
        body.push_str(&format!(
            r##"<line x1="80" x2="940" y1="{:.2}" y2="{:.2}" stroke="#111827" stroke-dasharray="3 5"/>"##,
            yproof(row.proof_bytes as f64),
            yproof(row.proof_bytes as f64)
        ));
        draw_png_dashed_hline(&mut png, yproof(row.proof_bytes as f64), PNG_OG);
    }
    write_svg(
        &out_dir.join("plot2_proof_size_vs_width.svg"),
        "Proof Size vs Width",
        &body,
    );
    write_png(&out_dir.join("plot2_proof_size_vs_width.png"), &png);

    let (scatter_x_min, scatter_x_max) =
        finite_range(width_rows.iter().map(|row| row.proof_bytes as f64));
    let (scatter_y_min, scatter_y_max) = finite_range(width_rows.iter().map(|row| row.prover_ms));
    let sx = |x: f64| 80.0 + ((x - scatter_x_min) / (scatter_x_max - scatter_x_min)) * 860.0;
    let sy = |y: f64| 500.0 - ((y - scatter_y_min) / (scatter_y_max - scatter_y_min)) * 430.0;
    let max_ecc = width_rows
        .iter()
        .filter_map(|row| row.ecc_points_per_commitment)
        .max()
        .unwrap_or(1) as f64;
    let mut body = String::new();
    let mut png = blank_png_canvas();
    draw_png_axes(&mut png);
    for row in &width_rows {
        let ecc = row.ecc_points_per_commitment.unwrap_or(1) as f64;
        let radius = 4.0 + 8.0 * (ecc / max_ecc).sqrt();
        let x = sx(row.proof_bytes as f64);
        let y = sy(row.prover_ms);
        let color = category_svg_color(row.category);
        body.push_str(&format!(
            r##"<circle cx="{x:.2}" cy="{y:.2}" r="{radius:.2}" fill="{color}" fill-opacity="0.72"/><text x="{:.2}" y="{:.2}" font-family="sans-serif" font-size="11">{} {}</text>"##,
            x + radius + 2.0,
            y - radius,
            row.category.short_label(),
            row.variant
        ));
        draw_png_filled_circle(&mut png, x, y, radius, category_png_color(row.category));
    }
    write_svg(
        &out_dir.join("plot3_pareto_scatter.svg"),
        "Prover Time vs Proof Size",
        &body,
    );
    write_png(&out_dir.join("plot3_pareto_scatter.png"), &png);

    let ecc_min = width_rows
        .iter()
        .filter_map(|row| row.ecc_points_per_commitment)
        .min()
        .unwrap_or(1) as f64;
    let ecc_max = width_rows
        .iter()
        .filter_map(|row| row.ecc_points_per_commitment)
        .max()
        .unwrap_or(2) as f64;
    let x_ecc = |x: f64| 80.0 + ((x.ln() - ecc_min.ln()) / (ecc_max.ln() - ecc_min.ln())) * 860.0;
    let (verifier_min, verifier_max) = finite_range(width_rows.iter().map(|row| row.verifier_ms));
    let yver = |y: f64| 500.0 - ((y - verifier_min) / (verifier_max - verifier_min)) * 430.0;
    let packed_verifier_ecc_points = packed
        .iter()
        .map(|row| {
            (
                row.ecc_points_per_commitment.unwrap_or_default() as f64,
                row.verifier_ms,
            )
        })
        .collect::<Vec<_>>();
    let projection_mixed_verifier_ecc_points = projection_mixed
        .iter()
        .map(|row| {
            (
                row.ecc_points_per_commitment.unwrap_or_default() as f64,
                row.verifier_ms,
            )
        })
        .collect::<Vec<_>>();
    let og_mixed_verifier_ecc_points = og_mixed
        .iter()
        .map(|row| {
            (
                row.ecc_points_per_commitment.unwrap_or_default() as f64,
                row.verifier_ms,
            )
        })
        .collect::<Vec<_>>();
    let mut body = String::new();
    body.push_str(&svg_polyline(
        &packed_verifier_ecc_points,
        "#2563eb",
        &x_ecc,
        &yver,
    ));
    body.push_str(&svg_polyline(
        &projection_mixed_verifier_ecc_points,
        "#dc2626",
        &x_ecc,
        &yver,
    ));
    body.push_str(&svg_polyline(
        &og_mixed_verifier_ecc_points,
        "#f97316",
        &x_ecc,
        &yver,
    ));
    let mut png = blank_png_canvas();
    draw_png_axes(&mut png);
    draw_png_polyline(
        &mut png,
        &packed_verifier_ecc_points,
        PNG_PROOF,
        &x_ecc,
        &yver,
    );
    draw_png_polyline(
        &mut png,
        &projection_mixed_verifier_ecc_points,
        PNG_MIXED,
        &x_ecc,
        &yver,
    );
    draw_png_polyline(
        &mut png,
        &og_mixed_verifier_ecc_points,
        PNG_SCATTER,
        &x_ecc,
        &yver,
    );
    write_svg(
        &out_dir.join("plot4_verifier_vs_ecc.svg"),
        "Verifier Time vs ECC Points",
        &body,
    );
    write_png(&out_dir.join("plot4_verifier_vs_ecc.png"), &png);
}

fn write_hyrax_instance_sweep_plot(out_dir: &Path, rows: &[HyraxInstanceSweepRow]) {
    if rows.is_empty() {
        return;
    }

    fn l0_svg_color(l0: usize) -> &'static str {
        match l0 {
            3 => "#0f766e",
            4 => "#dc2626",
            5 => "#2563eb",
            _ => "#111827",
        }
    }

    fn l0_png_color(l0: usize) -> Rgb {
        match l0 {
            3 => PNG_PROVER,
            4 => PNG_MIXED,
            5 => PNG_PROOF,
            _ => PNG_OG,
        }
    }

    fn push_svg_y_axis_labels(body: &mut String, min: f64, max: f64, ymap: &dyn Fn(f64) -> f64) {
        const TICKS: usize = 5;
        for idx in 0..=TICKS {
            let value = min + (max - min) * idx as f64 / TICKS as f64;
            let y = ymap(value);
            body.push_str(&format!(
                r##"<line x1="74" y1="{y:.2}" x2="80" y2="{y:.2}" stroke="#222"/><text x="68" y="{:.2}" font-family="sans-serif" font-size="11" text-anchor="end">{value:.0} ms</text>"##,
                y + 3.5
            ));
        }
    }

    let mut l0_values = rows.iter().map(|row| row.l0).collect::<Vec<_>>();
    l0_values.sort_unstable();
    l0_values.dedup();

    let instance_max = rows.iter().map(|row| row.instances).max().unwrap_or(2) as f64;
    let xinstance = |x: f64| 80.0 + (x / instance_max) * 860.0;
    let (time_min, time_max) = (0.0, finite_range(rows.iter().map(|row| row.prover_ms)).1);
    let ytime = |y: f64| 500.0 - ((y - time_min) / (time_max - time_min)) * 430.0;

    let mut body = String::new();
    push_svg_y_axis_labels(&mut body, time_min, time_max, &ytime);
    let mut png = blank_png_canvas();
    draw_png_axes(&mut png);
    for (idx, l0) in l0_values.iter().copied().enumerate() {
        let mut l0_rows = rows.iter().filter(|row| row.l0 == l0).collect::<Vec<_>>();
        l0_rows.sort_by_key(|row| row.instances);
        let prover_points = l0_rows
            .iter()
            .map(|row| (row.instances as f64, row.prover_ms))
            .collect::<Vec<_>>();
        let prover_svg = l0_svg_color(l0);
        let prover_png = l0_png_color(l0);
        body.push_str(&svg_polyline(
            &prover_points,
            prover_svg,
            &xinstance,
            &ytime,
        ));
        draw_png_polyline(&mut png, &prover_points, prover_png, &xinstance, &ytime);
        body.push_str(&format!(
            r##"<text x="700" y="{}" font-family="sans-serif" font-size="13" fill="{prover_svg}">l0 {l0}</text>"##,
            84 + idx * 24
        ));
    }
    let mut instance_labels = rows.iter().map(|row| row.instances).collect::<Vec<_>>();
    instance_labels.sort_unstable();
    instance_labels.dedup();
    for instances in instance_labels {
        body.push_str(&format!(
            r##"<text x="{:.2}" y="522" font-family="sans-serif" font-size="12" text-anchor="middle">{}</text>"##,
            xinstance(instances as f64),
            instances
        ));
    }
    body.push_str(
        r##"<text x="510" y="548" font-family="sans-serif" font-size="13" text-anchor="middle">instances, linear scale</text><text x="18" y="285" font-family="sans-serif" font-size="13" text-anchor="middle" transform="rotate(-90 18 285)">prover time (ms)</text>"##,
    );
    write_svg(
        &out_dir.join("plot5_instance_scaling.svg"),
        "ProjectionFold Mixed Hyrax Prover Time vs Instances by L0 (Linear X)",
        &body,
    );
    write_png(&out_dir.join("plot5_instance_scaling.png"), &png);

    let tail_min = rows.iter().map(|row| row.tail_vars).min().unwrap_or(0) as f64;
    let tail_max = rows.iter().map(|row| row.tail_vars).max().unwrap_or(1) as f64;
    let xtail = |x: f64| {
        if (tail_max - tail_min).abs() < f64::EPSILON {
            510.0
        } else {
            80.0 + ((x - tail_min) / (tail_max - tail_min)) * 860.0
        }
    };
    let mut body = String::new();
    push_svg_y_axis_labels(&mut body, time_min, time_max, &ytime);
    let mut png = blank_png_canvas();
    draw_png_axes(&mut png);
    for (idx, l0) in l0_values.iter().copied().enumerate() {
        let mut l0_rows = rows.iter().filter(|row| row.l0 == l0).collect::<Vec<_>>();
        l0_rows.sort_by_key(|row| row.tail_vars);
        let prover_points = l0_rows
            .iter()
            .map(|row| (row.tail_vars as f64, row.prover_ms))
            .collect::<Vec<_>>();
        let prover_svg = l0_svg_color(l0);
        let prover_png = l0_png_color(l0);
        body.push_str(&svg_polyline(&prover_points, prover_svg, &xtail, &ytime));
        draw_png_polyline(&mut png, &prover_points, prover_png, &xtail, &ytime);
        body.push_str(&format!(
            r##"<text x="700" y="{}" font-family="sans-serif" font-size="13" fill="{prover_svg}">l0 {l0}</text>"##,
            84 + idx * 24
        ));
    }
    for tail in tail_min as usize..=tail_max as usize {
        body.push_str(&format!(
            r##"<text x="{:.2}" y="522" font-family="sans-serif" font-size="12" text-anchor="middle">{}</text>"##,
            xtail(tail as f64),
            tail
        ));
    }
    write_svg(
        &out_dir.join("plot6_tail_scaling.svg"),
        "ProjectionFold Mixed Hyrax Prover Time vs Tail Variables by L0",
        &body,
    );
    write_png(&out_dir.join("plot6_tail_scaling.png"), &png);
}

fn write_hyrax_width_sweep_report(
    out_dir: &Path,
    rows: &[HyraxWidthSweepRow],
    instance_rows: &[HyraxInstanceSweepRow],
    confirmation_rows: &[HyraxWidthSweepRow],
    skipped_widths: &[SkippedPackedWidth],
) {
    fn best_row_by<'a>(
        rows: impl Iterator<Item = &'a HyraxWidthSweepRow>,
        metric: impl Fn(&HyraxWidthSweepRow) -> f64,
    ) -> Option<&'a HyraxWidthSweepRow> {
        rows.min_by(|a, b| {
            metric(a)
                .partial_cmp(&metric(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    fn best_instance_row_by<'a>(
        rows: impl Iterator<Item = &'a HyraxInstanceSweepRow>,
        metric: impl Fn(&HyraxInstanceSweepRow) -> f64,
    ) -> Option<&'a HyraxInstanceSweepRow> {
        rows.min_by(|a, b| {
            metric(a)
                .partial_cmp(&metric(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    fn metric_span(
        rows: &[&HyraxWidthSweepRow],
        metric: impl Fn(&HyraxWidthSweepRow) -> f64,
    ) -> (f64, f64) {
        rows.iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), row| {
                let value = metric(row);
                (min.min(value), max.max(value))
            })
    }

    fn normalize(value: f64, min: f64, max: f64) -> f64 {
        if (max - min).abs() < f64::EPSILON {
            0.0
        } else {
            (value - min) / (max - min)
        }
    }

    fn row_summary(row: &HyraxWidthSweepRow) -> String {
        format!(
            "{} / {}: prover median {:.3} ms, verifier median {:.3} ms, ECC commitment {}, ECC proof {}, proof {} bytes, zstd {} bytes",
            row.algorithm,
            row.variant,
            row.prover_ms,
            row.verifier_ms,
            row.ecc_points_per_commitment
                .map(|value| value.to_string())
                .unwrap_or_else(|| "N/A".to_string()),
            row.ecc_points_per_proof
                .map(|value| value.to_string())
                .unwrap_or_else(|| "N/A".to_string()),
            row.proof_bytes,
            row.proof_zstd_bytes
        )
    }

    fn instance_row_summary(row: &HyraxInstanceSweepRow) -> String {
        format!(
            "{} / {}: instances {}, ell {}, l0 {}, tail {}, width {}, setup+prepare {:.3} ms, prover median {:.3} ms ({:.3} ms/instance), verifier median {:.3} ms ({:.3} ms/instance), proof {} bytes, zstd {} bytes",
            row.algorithm,
            row.variant,
            row.instances,
            row.ell,
            row.l0,
            row.tail_vars,
            row.width,
            row.setup_prepare_ms,
            row.prover_ms,
            row.prover_ms / row.instances as f64,
            row.verifier_ms,
            row.verifier_ms / row.instances as f64,
            row.proof_bytes,
            row.proof_zstd_bytes
        )
    }

    fn optional_usize(value: Option<usize>) -> String {
        value
            .map(|value| value.to_string())
            .unwrap_or_else(|| "N/A".to_string())
    }

    fn html_escape(value: &str) -> String {
        value
            .replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
    }

    fn dominates(a: &HyraxWidthSweepRow, b: &HyraxWidthSweepRow) -> bool {
        let all_no_worse = a.prover_ms <= b.prover_ms
            && a.verifier_ms <= b.verifier_ms
            && a.proof_bytes <= b.proof_bytes
            && a.proof_zstd_bytes <= b.proof_zstd_bytes;
        let at_least_one_better = a.prover_ms < b.prover_ms
            || a.verifier_ms < b.verifier_ms
            || a.proof_bytes < b.proof_bytes
            || a.proof_zstd_bytes < b.proof_zstd_bytes;
        all_no_worse && at_least_one_better
    }

    fn pareto_frontier<'a>(rows: &[&'a HyraxWidthSweepRow]) -> Vec<&'a HyraxWidthSweepRow> {
        let mut frontier = rows
            .iter()
            .copied()
            .filter(|row| {
                !rows.iter().copied().any(|other| {
                    !std::ptr::eq::<HyraxWidthSweepRow>(other, *row) && dominates(other, row)
                })
            })
            .collect::<Vec<_>>();
        frontier.sort_by(|a, b| {
            a.prover_ms
                .partial_cmp(&b.prover_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        frontier
    }

    fn row_table(rows: &[&HyraxWidthSweepRow], winner_rows: &[&HyraxWidthSweepRow]) -> String {
        let mut table = String::from(
            "<table><thead><tr><th>algorithm</th><th>variant</th><th>width</th><th>ECC points/commitment</th><th>ECC points/proof</th><th>prover median ms</th><th>prover mean ms</th><th>prover min ms</th><th>prover max ms</th><th>verifier median ms</th><th>verifier mean ms</th><th>verifier min ms</th><th>verifier max ms</th><th>samples p/v</th><th>proof bytes</th><th>zstd bytes</th></tr></thead><tbody>",
        );
        for row in rows {
            let row = *row;
            let class = if winner_rows
                .iter()
                .any(|winner| std::ptr::eq::<HyraxWidthSweepRow>(*winner, row))
            {
                r#" class="winner""#
            } else {
                ""
            };
            table.push_str(&format!(
                "<tr{class}><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{:.3}</td><td>{:.3}</td><td>{:.3}</td><td>{:.3}</td><td>{:.3}</td><td>{:.3}</td><td>{:.3}</td><td>{:.3}</td><td>{}/{}</td><td>{}</td><td>{}</td></tr>",
                html_escape(row.algorithm),
                html_escape(&row.variant),
                optional_usize(row.width),
                optional_usize(row.ecc_points_per_commitment),
                optional_usize(row.ecc_points_per_proof),
                row.prover_ms,
                row.prover_mean_ms,
                row.prover_min_ms,
                row.prover_max_ms,
                row.verifier_ms,
                row.verifier_mean_ms,
                row.verifier_min_ms,
                row.verifier_max_ms,
                row.prover_samples,
                row.verifier_samples,
                row.proof_bytes,
                row.proof_zstd_bytes
            ));
        }
        table.push_str("</tbody></table>");
        table
    }

    fn instance_row_table(
        rows: &[&HyraxInstanceSweepRow],
        winner_rows: &[&HyraxInstanceSweepRow],
    ) -> String {
        let mut table = String::from(
            "<table><thead><tr><th>algorithm</th><th>variant</th><th>instances</th><th>ell</th><th>l0</th><th>tail vars</th><th>width</th><th>ECC points/commitment</th><th>ECC points/proof</th><th>trace/witness ms</th><th>PCS setup ms</th><th>prepare ms</th><th>setup+prepare ms</th><th>probe prove ms</th><th>probe commit ms</th><th>probe sumfold ms</th><th>probe fold ms</th><th>probe open ms</th><th>probe open core ms</th><th>prover median ms</th><th>prover mean ms</th><th>prover min ms</th><th>prover max ms</th><th>prover ms/instance</th><th>prover scaling vs L0 base</th><th>verifier median ms</th><th>verifier mean ms</th><th>verifier min ms</th><th>verifier max ms</th><th>verifier ms/instance</th><th>verifier scaling vs L0 base</th><th>samples p/v</th><th>proof bytes</th><th>proof growth vs L0 base</th><th>zstd bytes</th><th>zstd growth vs L0 base</th></tr></thead><tbody>",
        );
        for row in rows {
            let row = *row;
            let base = rows
                .iter()
                .filter(|candidate| candidate.l0 == row.l0)
                .min_by_key(|candidate| candidate.instances)
                .copied()
                .expect("each row has a same-l0 base row");
            let class = if winner_rows
                .iter()
                .any(|winner| std::ptr::eq::<HyraxInstanceSweepRow>(*winner, row))
            {
                r#" class="winner""#
            } else {
                ""
            };
            table.push_str(&format!(
                "<tr{class}><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{:.3}</td><td>{:.3}</td><td>{:.3}</td><td>{:.3}</td><td>{:.3}</td><td>{:.3}</td><td>{:.3}</td><td>{:.3}</td><td>{:.3}</td><td>{:.3}</td><td>{:.3}</td><td>{:.3}</td><td>{:.3}</td><td>{:.3}</td><td>{:.3}</td><td>{:.2}x</td><td>{:.3}</td><td>{:.3}</td><td>{:.3}</td><td>{:.3}</td><td>{:.3}</td><td>{:.2}x</td><td>{}/{}</td><td>{}</td><td>{:.2}x</td><td>{}</td><td>{:.2}x</td></tr>",
                html_escape(row.algorithm),
                html_escape(&row.variant),
                row.instances,
                row.ell,
                row.l0,
                row.tail_vars,
                row.width,
                row.ecc_points_per_commitment,
                row.ecc_points_per_proof,
                row.trace_witness_ms,
                row.setup_ms,
                row.prepare_ms,
                row.setup_prepare_ms,
                row.probe_prove_ms,
                row.probe_commit_ms,
                row.probe_sumfold_ms,
                row.probe_fold_ms,
                row.probe_open_ms,
                row.probe_open_core_ms,
                row.prover_ms,
                row.prover_mean_ms,
                row.prover_min_ms,
                row.prover_max_ms,
                row.prover_ms / row.instances as f64,
                row.prover_ms / base.prover_ms,
                row.verifier_ms,
                row.verifier_mean_ms,
                row.verifier_min_ms,
                row.verifier_max_ms,
                row.verifier_ms / row.instances as f64,
                row.verifier_ms / base.verifier_ms,
                row.prover_samples,
                row.verifier_samples,
                row.proof_bytes,
                row.proof_bytes as f64 / base.proof_bytes as f64,
                row.proof_zstd_bytes,
                row.proof_zstd_bytes as f64 / base.proof_zstd_bytes as f64
            ));
        }
        table.push_str("</tbody></table>");
        table
    }

    fn best_l0_by_instance_table(rows: &[&HyraxInstanceSweepRow]) -> String {
        let mut instances = rows.iter().map(|row| row.instances).collect::<Vec<_>>();
        instances.sort_unstable();
        instances.dedup();
        let mut table = String::from(
            "<table><thead><tr><th>instances</th><th>best prover l0</th><th>best prover ms</th><th>best verifier l0</th><th>best verifier ms</th><th>smallest raw l0</th><th>raw bytes</th><th>smallest zstd l0</th><th>zstd bytes</th></tr></thead><tbody>",
        );
        for instance_count in instances {
            let matching = rows
                .iter()
                .copied()
                .filter(|row| row.instances == instance_count)
                .collect::<Vec<_>>();
            let best_prover = best_instance_row_by(matching.iter().copied(), |row| row.prover_ms)
                .expect("instance count has at least one row");
            let best_verifier =
                best_instance_row_by(matching.iter().copied(), |row| row.verifier_ms)
                    .expect("instance count has at least one row");
            let smallest_raw =
                best_instance_row_by(matching.iter().copied(), |row| row.proof_bytes as f64)
                    .expect("instance count has at least one row");
            let smallest_zstd =
                best_instance_row_by(matching.iter().copied(), |row| row.proof_zstd_bytes as f64)
                    .expect("instance count has at least one row");
            table.push_str(&format!(
                "<tr><td>{}</td><td>{}</td><td>{:.3}</td><td>{}</td><td>{:.3}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
                instance_count,
                best_prover.l0,
                best_prover.prover_ms,
                best_verifier.l0,
                best_verifier.verifier_ms,
                smallest_raw.l0,
                smallest_raw.proof_bytes,
                smallest_zstd.l0,
                smallest_zstd.proof_zstd_bytes
            ));
        }
        table.push_str("</tbody></table>");
        table
    }

    let packed_rows = rows
        .iter()
        .filter(|row| row.category == HyraxReportCategory::ProjectionFoldPackedHyrax)
        .collect::<Vec<_>>();
    let projection_mixed_rows = rows
        .iter()
        .filter(|row| row.category == HyraxReportCategory::ProjectionFoldMixedHyrax)
        .collect::<Vec<_>>();
    let og_mixed_rows = rows
        .iter()
        .filter(|row| row.category == HyraxReportCategory::OgMixedHyrax)
        .collect::<Vec<_>>();
    let hyrax_rows = rows
        .iter()
        .filter(|row| row.category != HyraxReportCategory::OgHash)
        .collect::<Vec<_>>();
    let og_zip = rows
        .iter()
        .find(|row| row.category == HyraxReportCategory::OgHash);
    let hyrax_fastest_prover = best_row_by(hyrax_rows.iter().copied(), |row| row.prover_ms);
    let hyrax_fastest_verifier = best_row_by(hyrax_rows.iter().copied(), |row| row.verifier_ms);
    let hyrax_smallest_proof =
        best_row_by(hyrax_rows.iter().copied(), |row| row.proof_bytes as f64);
    let hyrax_smallest_zstd = best_row_by(hyrax_rows.iter().copied(), |row| {
        row.proof_zstd_bytes as f64
    });
    let packed_fastest_prover = best_row_by(packed_rows.iter().copied(), |row| row.prover_ms);
    let packed_fastest_verifier = best_row_by(packed_rows.iter().copied(), |row| row.verifier_ms);
    let packed_smallest_proof =
        best_row_by(packed_rows.iter().copied(), |row| row.proof_bytes as f64);
    let packed_smallest_zstd = best_row_by(packed_rows.iter().copied(), |row| {
        row.proof_zstd_bytes as f64
    });
    let projection_mixed_fastest_prover =
        best_row_by(projection_mixed_rows.iter().copied(), |row| row.prover_ms);
    let projection_mixed_fastest_verifier =
        best_row_by(projection_mixed_rows.iter().copied(), |row| row.verifier_ms);
    let projection_mixed_smallest_proof =
        best_row_by(projection_mixed_rows.iter().copied(), |row| {
            row.proof_bytes as f64
        });
    let projection_mixed_smallest_zstd =
        best_row_by(projection_mixed_rows.iter().copied(), |row| {
            row.proof_zstd_bytes as f64
        });
    let og_mixed_fastest_prover = best_row_by(og_mixed_rows.iter().copied(), |row| row.prover_ms);
    let og_mixed_fastest_verifier =
        best_row_by(og_mixed_rows.iter().copied(), |row| row.verifier_ms);
    let og_mixed_smallest_proof =
        best_row_by(og_mixed_rows.iter().copied(), |row| row.proof_bytes as f64);
    let og_mixed_smallest_zstd = best_row_by(og_mixed_rows.iter().copied(), |row| {
        row.proof_zstd_bytes as f64
    });
    let packed_balanced = if packed_rows.is_empty() {
        None
    } else {
        let (prover_min, prover_max) = metric_span(&packed_rows, |row| row.prover_ms);
        let (verifier_min, verifier_max) = metric_span(&packed_rows, |row| row.verifier_ms);
        let (proof_min, proof_max) = metric_span(&packed_rows, |row| row.proof_bytes as f64);
        let (zstd_min, zstd_max) = metric_span(&packed_rows, |row| row.proof_zstd_bytes as f64);
        packed_rows.iter().copied().min_by(|a, b| {
            let score = |row: &HyraxWidthSweepRow| {
                normalize(row.prover_ms, prover_min, prover_max)
                    + normalize(row.verifier_ms, verifier_min, verifier_max)
                    + normalize(row.proof_bytes as f64, proof_min, proof_max)
                    + normalize(row.proof_zstd_bytes as f64, zstd_min, zstd_max)
            };
            score(a)
                .partial_cmp(&score(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    };
    let winner_rows = [
        hyrax_fastest_prover,
        hyrax_fastest_verifier,
        hyrax_smallest_proof,
        hyrax_smallest_zstd,
        packed_fastest_prover,
        packed_fastest_verifier,
        packed_smallest_proof,
        packed_smallest_zstd,
        projection_mixed_fastest_prover,
        projection_mixed_fastest_verifier,
        projection_mixed_smallest_proof,
        projection_mixed_smallest_zstd,
        og_mixed_fastest_prover,
        og_mixed_fastest_verifier,
        og_mixed_smallest_proof,
        og_mixed_smallest_zstd,
        packed_balanced,
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();

    let mut all_rows = rows.iter().collect::<Vec<_>>();
    all_rows.sort_by_key(|row| {
        (
            row.category.order(),
            row.width.unwrap_or_default(),
            row.variant.clone(),
        )
    });
    let table = row_table(&all_rows, &winner_rows);
    let pareto_rows = pareto_frontier(&hyrax_rows);
    let pareto_table = row_table(&pareto_rows, &[]);
    let confirmation_refs = confirmation_rows.iter().collect::<Vec<_>>();
    let confirmation_table = row_table(&confirmation_refs, &[]);
    let mut instance_refs = instance_rows.iter().collect::<Vec<_>>();
    instance_refs.sort_by_key(|row| (row.l0, row.instances));
    let instance_fastest_prover =
        best_instance_row_by(instance_refs.iter().copied(), |row| row.prover_ms);
    let instance_fastest_verifier =
        best_instance_row_by(instance_refs.iter().copied(), |row| row.verifier_ms);
    let instance_best_amortized_prover =
        best_instance_row_by(instance_refs.iter().copied(), |row| {
            row.prover_ms / row.instances as f64
        });
    let instance_best_amortized_verifier =
        best_instance_row_by(instance_refs.iter().copied(), |row| {
            row.verifier_ms / row.instances as f64
        });
    let instance_winner_rows = [
        instance_fastest_prover,
        instance_fastest_verifier,
        instance_best_amortized_prover,
        instance_best_amortized_verifier,
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();
    let instance_scaling_html = if instance_refs.is_empty() {
        String::new()
    } else {
        let ell8_rows = instance_refs
            .iter()
            .copied()
            .filter(|row| row.ell == 8)
            .collect::<Vec<_>>();
        let ell8_fastest_prover =
            best_instance_row_by(ell8_rows.iter().copied(), |row| row.prover_ms);
        let ell8_fastest_verifier =
            best_instance_row_by(ell8_rows.iter().copied(), |row| row.verifier_ms);
        let best_l0_table = best_l0_by_instance_table(&instance_refs);
        let instance_table = instance_row_table(&instance_refs, &instance_winner_rows);
        format!(
            r#"<h2>ProjectionFold L0 Scaling</h2>
<p>ProjectionFold Mixed Hyrax L0 sweep with width {HYRAX_INSTANCE_SWEEP_WIDTH}, BN254 Hyrax, and ArkFBn254. Rows cover l0 3 with ell 3..8, l0 4 with ell 4..8, and l0 5 with ell 5..8.</p>
<ul>
<li><strong>Fastest prover instance row:</strong> {}</li>
<li><strong>Fastest verifier instance row:</strong> {}</li>
<li><strong>Best amortized prover row:</strong> {}</li>
<li><strong>Best amortized verifier row:</strong> {}</li>
<li><strong>Fastest 256-instance prover L0:</strong> {}</li>
<li><strong>Fastest 256-instance verifier L0:</strong> {}</li>
</ul>
<h3>Best L0 by instance count</h3>
{best_l0_table}
<h3>All L0 rows</h3>
{instance_table}
<img src="plot5_instance_scaling.svg" alt="ProjectionFold L0 instance scaling">"#,
            instance_fastest_prover
                .map(instance_row_summary)
                .unwrap_or_else(|| "N/A".to_string()),
            instance_fastest_verifier
                .map(instance_row_summary)
                .unwrap_or_else(|| "N/A".to_string()),
            instance_best_amortized_prover
                .map(instance_row_summary)
                .unwrap_or_else(|| "N/A".to_string()),
            instance_best_amortized_verifier
                .map(instance_row_summary)
                .unwrap_or_else(|| "N/A".to_string()),
            ell8_fastest_prover
                .map(instance_row_summary)
                .unwrap_or_else(|| "N/A".to_string()),
            ell8_fastest_verifier
                .map(instance_row_summary)
                .unwrap_or_else(|| "N/A".to_string())
        )
    };

    let skipped_html = if skipped_widths.is_empty() {
        String::new()
    } else {
        let mut skipped_table = String::from(
            "<h2>Skipped Requested Widths</h2><table><thead><tr><th>width</th><th>reason</th></tr></thead><tbody>",
        );
        for skipped in skipped_widths {
            skipped_table.push_str(&format!(
                "<tr><td>{}</td><td>{}</td></tr>",
                skipped.width,
                html_escape(&skipped.reason)
            ));
        }
        skipped_table.push_str("</tbody></table>");
        skipped_table
    };

    let most_performant = format!(
        r#"<h2>Most performant</h2>
<ul>
<li><strong>Fastest Hyrax prover recommendation:</strong> {}</li>
<li><strong>Fastest Hyrax verifier:</strong> {}</li>
<li><strong>Smallest Hyrax raw proof:</strong> {}</li>
<li><strong>Smallest Hyrax zstd proof:</strong> {}</li>
<li><strong>Best OG mixed-Hyrax prover width:</strong> {}</li>
<li><strong>Best OG mixed-Hyrax verifier width:</strong> {}</li>
<li><strong>Smallest OG mixed-Hyrax raw proof:</strong> {}</li>
<li><strong>Smallest OG mixed-Hyrax zstd proof:</strong> {}</li>
<li><strong>Best ProjectionFold mixed-Hyrax prover width:</strong> {}</li>
<li><strong>Best ProjectionFold mixed-Hyrax verifier width:</strong> {}</li>
<li><strong>Smallest ProjectionFold mixed-Hyrax raw proof:</strong> {}</li>
<li><strong>Smallest ProjectionFold mixed-Hyrax zstd proof:</strong> {}</li>
<li><strong>Best ProjectionFold packed-Hyrax prover width:</strong> {}</li>
<li><strong>Best ProjectionFold packed-Hyrax verifier width:</strong> {}</li>
<li><strong>Smallest ProjectionFold packed-Hyrax raw proof:</strong> {}</li>
<li><strong>Smallest ProjectionFold packed-Hyrax zstd proof:</strong> {}</li>
<li><strong>Best balanced ProjectionFold packed width:</strong> {} <em>(min normalized prover + verifier + raw proof + zstd proof)</em></li>
<li><strong>OG Zinc+ hash baseline:</strong> {}</li>
</ul>
<p>Rows highlighted in the main table are Hyrax winners for at least one metric. Timing values are warmed medians.</p>"#,
        hyrax_fastest_prover
            .map(row_summary)
            .unwrap_or_else(|| "N/A".to_string()),
        hyrax_fastest_verifier
            .map(row_summary)
            .unwrap_or_else(|| "N/A".to_string()),
        hyrax_smallest_proof
            .map(row_summary)
            .unwrap_or_else(|| "N/A".to_string()),
        hyrax_smallest_zstd
            .map(row_summary)
            .unwrap_or_else(|| "N/A".to_string()),
        og_mixed_fastest_prover
            .map(row_summary)
            .unwrap_or_else(|| "N/A".to_string()),
        og_mixed_fastest_verifier
            .map(row_summary)
            .unwrap_or_else(|| "N/A".to_string()),
        og_mixed_smallest_proof
            .map(row_summary)
            .unwrap_or_else(|| "N/A".to_string()),
        og_mixed_smallest_zstd
            .map(row_summary)
            .unwrap_or_else(|| "N/A".to_string()),
        projection_mixed_fastest_prover
            .map(row_summary)
            .unwrap_or_else(|| "N/A".to_string()),
        projection_mixed_fastest_verifier
            .map(row_summary)
            .unwrap_or_else(|| "N/A".to_string()),
        projection_mixed_smallest_proof
            .map(row_summary)
            .unwrap_or_else(|| "N/A".to_string()),
        projection_mixed_smallest_zstd
            .map(row_summary)
            .unwrap_or_else(|| "N/A".to_string()),
        packed_fastest_prover
            .map(row_summary)
            .unwrap_or_else(|| "N/A".to_string()),
        packed_fastest_verifier
            .map(row_summary)
            .unwrap_or_else(|| "N/A".to_string()),
        packed_smallest_proof
            .map(row_summary)
            .unwrap_or_else(|| "N/A".to_string()),
        packed_smallest_zstd
            .map(row_summary)
            .unwrap_or_else(|| "N/A".to_string()),
        packed_balanced
            .map(row_summary)
            .unwrap_or_else(|| "N/A".to_string()),
        og_zip.map(row_summary).unwrap_or_else(|| "N/A".to_string())
    );

    let html = format!(
        r#"<!doctype html>
<html><head><meta charset="utf-8"><title>Hyrax Width Sweep</title>
<style>body{{font-family:Inter,system-ui,sans-serif;margin:32px;color:#111827}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #d1d5db;padding:6px 8px;text-align:right}}th:first-child,td:first-child{{text-align:left}}tr.winner{{background:#ecfdf5}}tr.winner td:first-child{{font-weight:700}}img{{max-width:100%;border:1px solid #e5e7eb;margin:16px 0}}</style>
</head><body><h1>Hyrax Width Sweep</h1>
<p>Values per instance: {PACKED_SHA_VALUES_PER_INSTANCE}. Widths below 549 use row-local source padding for separable packed openings, so actual ECC points may exceed ceil(values/width).</p>
<p>Mixed Hyrax widths tune the row-domain commitment width for the specialized binary/int lane layout. Widths below 128 increase the number of mixed commitment rows; widths above 128 cannot reduce mixed ECC commitments below one row per source, but they test wider opening vectors and fixed-base MSM behavior.</p>
<p>Timing mode: {HYRAX_WIDTH_SWEEP_WARMUP_RUNS} warmup runs per config, {HYRAX_WIDTH_SWEEP_TUNING_SAMPLES} recorded tuning samples per config, and {HYRAX_WIDTH_SWEEP_CONFIRMATION_SAMPLES} recorded confirmation samples for the top {HYRAX_WIDTH_SWEEP_CONFIRMATION_TOP_K} Hyrax prover candidates. Headline times are medians.</p>
{most_performant}
{table}
{skipped_html}
{instance_scaling_html}
<h2>Confirmation Pass</h2>
{confirmation_table}
<h2>Pareto Frontier</h2>
<p>Non-dominated Hyrax rows across prover median, verifier median, raw proof bytes, and zstd proof bytes.</p>
{pareto_table}
<h2>Plots</h2>
<img src="plot1_time_vs_width.svg" alt="time vs width">
<img src="plot2_proof_size_vs_width.svg" alt="proof size vs width">
<img src="plot3_pareto_scatter.svg" alt="pareto scatter">
<img src="plot4_verifier_vs_ecc.svg" alt="verifier vs ecc">
</body></html>"#
    );
    fs::write(out_dir.join("report.html"), html).expect("write hyrax width sweep HTML report");
}

fn measure_projectionfold_mixed_hyrax_instances_with_samples<const N: usize, const L0: usize>(
    warmup_runs: usize,
    sample_count: usize,
    width: usize,
) -> HyraxInstanceSweepRow {
    type C = ark_bn254::G1Affine;
    type HyraxF = ArkFBn254;
    type P = AllHyraxPCSTypes<C>;
    type U = ProjectionShaBenchUair<RealEcdsaInt>;

    let ell = log2_power_of_two(N);
    assert!(ell >= L0, "instance sweep requires ell >= l0");

    let setup_start = Instant::now();
    let trace_start = Instant::now();
    let message_blocks = exact_sha256_chain_blocks::<N>();
    let num_vars = SHA_ROW_VARS + ell;
    let (_mono_trace, mono_final_state) = synthesize_sha256_chain_trace::<RealEcdsaInt, N>(
        num_vars,
        SHA256_INITIAL_STATE,
        message_blocks,
    )
    .expect("monolithic instance-sweep SHA trace synthesis should succeed");
    let (witnesses, projection_final_state) =
        synthesize_sha256_chain_witnesses::<RealEcdsaInt, N>(SHA256_INITIAL_STATE, message_blocks)
            .expect("ProjectionFold instance-sweep SHA witness synthesis should succeed");
    assert_eq!(mono_final_state, projection_final_state);
    let trace_witness_ms = elapsed_ms(trace_start);

    let setup_phase_start = Instant::now();
    let shape = UairShape::<U>::new(SHA_ROW_VARS);
    let field_cfg = HyraxF::curve_field_cfg::<C>();
    let (pcs_params, _) = projection_sha_bn254_hyrax_pcs_params(width);
    let pp =
        LinearIdealFoldProverParams::<P, U, RealEcdsaBenchZincTypes, HyraxF, DEGREE_PLUS_ONE>::new(
            pcs_params,
            field_cfg.clone(),
            L0,
        );
    let vs = projection_sha_bn254_mixed_hyrax_verifier_setup(width);
    let setup_ms = elapsed_ms(setup_phase_start);

    let prepare_start = Instant::now();
    let prepare_layer = phase_timing_layer();
    let prepared_instances = prepare_linear_ideal_fold_witnesses::<
        U,
        RealEcdsaBenchZincTypes,
        HyraxF,
        DEGREE_PLUS_ONE,
    >(&shape, &witnesses, &pp.field_cfg)
    .expect("ProjectionFold instance-sweep SHA witness preparation should succeed");
    let prepare_ms = elapsed_ms(prepare_start);
    let prepare_sumfold_basis_ms = prepare_layer.total_ms(ProvePhase::PrepareSumfoldBasis);
    let setup_prepare_ms = elapsed_ms(setup_start);
    let setup_prepare = HyraxSetupPrepareTimings {
        trace_witness_ms,
        setup_ms,
        prepare_sumfold_basis_ms,
        prepare_ms,
        setup_prepare_ms,
    };

    let prove_phases = measure_mixed_hyrax_prove_phase_timings::<
        C,
        U,
        RealEcdsaBenchZincTypes,
        HyraxF,
        DEGREE_PLUS_ONE,
    >(&pp, &shape, &prepared_instances);

    if env_bool_or("SHA256_COMBINED_SWEEP_TRACE_ONCE", false) {
        eprintln!("ProjectionFold instance sweep tracing: instances={N}, l0={L0}");
        eprintln!(
            "phase probe: prove={:.3} ms commit={:.3} ms sumfold_linear={:.3} ms sumfold_booleanity={:.3} ms sumfold_group={:.3} ms sumfold_prove_rounds={:.3} ms sumfold={:.3} ms fold={:.3} ms open={:.3} ms open_core={:.3} ms",
            prove_phases.prove_ms,
            prove_phases.commit_ms,
            prove_phases.sumfold_linear_ms,
            prove_phases.sumfold_booleanity_ms,
            prove_phases.sumfold_group_ms,
            prove_phases.sumfold_prove_rounds_ms,
            prove_phases.sumfold_ms,
            prove_phases.fold_ms,
            prove_phases.open_ms,
            prove_phases.open_core_ms
        );
        let subscriber = tracing_subscriber::fmt()
            .with_writer(std::io::stderr)
            .with_target(true)
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE)
            .finish();
        tracing::subscriber::with_default(subscriber, || {
            let mut prover_transcript = Blake3Transcript::new();
            let traced_output = prove_prepared_linear_ideal_fold_mixed_hyrax::<
                C,
                U,
                RealEcdsaBenchZincTypes,
                HyraxF,
                DEGREE_PLUS_ONE,
            >(
                &pp,
                &shape,
                &prepared_instances,
                v1_booleanity_catalog(),
                &mut prover_transcript,
            )
            .expect("traced instance-sweep mixed Hyrax prover failed");

            let mut verifier_transcript = Blake3Transcript::new();
            let traced_verified = verify_linear_ideal_fold_mixed_hyrax::<
                C,
                U,
                RealEcdsaBenchZincTypes,
                HyraxF,
                DEGREE_PLUS_ONE,
            >(
                &vs,
                &traced_output.fresh_instances,
                &traced_output.proof,
                v1_booleanity_catalog(),
                &mut verifier_transcript,
            )
            .expect("traced instance-sweep mixed Hyrax verifier failed");
            assert_eq!(traced_verified.target, traced_output.folded_instance.target);
            assert_eq!(traced_verified.public, traced_output.folded_instance.public);
        });
    }

    let measured_prover_warmups = warmup_runs.saturating_sub(1);
    let (output, prover_stats) = measure_warmed(measured_prover_warmups, sample_count, || {
        let mut transcript = Blake3Transcript::new();
        prove_prepared_linear_ideal_fold_mixed_hyrax::<
            C,
            U,
            RealEcdsaBenchZincTypes,
            HyraxF,
            DEGREE_PLUS_ONE,
        >(
            &pp,
            &shape,
            &prepared_instances,
            v1_booleanity_catalog(),
            &mut transcript,
        )
        .expect("instance-sweep mixed Hyrax prover failed")
    });
    let (_, verifier_stats) = measure_warmed(warmup_runs, sample_count, || {
        let mut transcript = Blake3Transcript::new();
        verify_linear_ideal_fold_mixed_hyrax::<
                C,
                U,
                RealEcdsaBenchZincTypes,
                HyraxF,
                DEGREE_PLUS_ONE,
            >(
                &vs,
                &output.fresh_instances,
                &output.proof,
                v1_booleanity_catalog(),
                &mut transcript,
            )
            .expect("instance-sweep mixed Hyrax verifier failed")
    });
    let raw = production_mixed_hyrax_proof_raw_bytes(&output.proof);
    let ecc_points_per_commitment = output
        .proof
        .instance_commitments
        .first()
        .map(|commitment| {
            commitment.binary.group_point_count() + commitment.int.group_point_count()
        })
        .unwrap_or_default();
    let ecc_points_per_proof = output
        .proof
        .instance_commitments
        .iter()
        .map(|commitment| {
            commitment.binary.group_point_count() + commitment.int.group_point_count()
        })
        .sum();

    hyrax_instance_sweep_row(
        N,
        ell,
        L0,
        width,
        ecc_points_per_commitment,
        ecc_points_per_proof,
        setup_prepare,
        prove_phases,
        prover_stats,
        verifier_stats,
        raw.len(),
        zstd_len(&raw),
    )
}

fn measure_projectionfold_mixed_hyrax_instances<const N: usize, const L0: usize>()
-> HyraxInstanceSweepRow {
    measure_projectionfold_mixed_hyrax_instances_with_samples::<N, L0>(
        HYRAX_WIDTH_SWEEP_WARMUP_RUNS,
        HYRAX_WIDTH_SWEEP_CONFIRMATION_SAMPLES,
        HYRAX_INSTANCE_SWEEP_WIDTH,
    )
}

fn measure_foldfirst_mixed_hyrax_instances_with_samples<const N: usize>(
    warmup_runs: usize,
    sample_count: usize,
    width: usize,
) -> HyraxInstanceSweepRow {
    type C = ark_bn254::G1Affine;
    type HyraxF = ArkFBn254;
    type P = AllHyraxPCSTypes<C>;
    type U = ProjectionShaBenchUair<RealEcdsaInt>;

    let ell = log2_power_of_two(N);

    let setup_start = Instant::now();
    let trace_start = Instant::now();
    let message_blocks = exact_sha256_chain_blocks::<N>();
    let num_vars = SHA_ROW_VARS + ell;
    let (_mono_trace, mono_final_state) = synthesize_sha256_chain_trace::<RealEcdsaInt, N>(
        num_vars,
        SHA256_INITIAL_STATE,
        message_blocks,
    )
    .expect("monolithic fold-first sweep SHA trace synthesis should succeed");
    let (witnesses, projection_final_state) =
        synthesize_sha256_chain_witnesses::<RealEcdsaInt, N>(SHA256_INITIAL_STATE, message_blocks)
            .expect("fold-first sweep SHA witness synthesis should succeed");
    assert_eq!(mono_final_state, projection_final_state);
    let trace_witness_ms = elapsed_ms(trace_start);

    let setup_phase_start = Instant::now();
    let shape = UairShape::<U>::new(SHA_ROW_VARS);
    let field_cfg = HyraxF::curve_field_cfg::<C>();
    let (pcs_params, _) = projection_sha_bn254_hyrax_pcs_params(width);
    let pp =
        LinearIdealFoldProverParams::<P, U, RealEcdsaBenchZincTypes, HyraxF, DEGREE_PLUS_ONE>::new(
            pcs_params,
            field_cfg.clone(),
            0,
        );
    let vs = projection_sha_bn254_mixed_hyrax_verifier_setup(width);
    let setup_ms = elapsed_ms(setup_phase_start);

    let prepare_start = Instant::now();
    let prepared_instances = prepare_fold_first_linear_ideal_fold_witnesses::<
        U,
        RealEcdsaBenchZincTypes,
        HyraxF,
        DEGREE_PLUS_ONE,
    >(&shape, &witnesses, &pp.field_cfg)
    .expect("fold-first sweep SHA witness preparation should succeed");
    let prepare_ms = elapsed_ms(prepare_start);
    let setup_prepare = HyraxSetupPrepareTimings {
        trace_witness_ms,
        setup_ms,
        prepare_sumfold_basis_ms: 0.0,
        prepare_ms,
        setup_prepare_ms: elapsed_ms(setup_start),
    };

    // Per-stage probe: fold stage (with the commit span isolated by the
    // phase-timing layer) and the deferred decider, on one untimed pass.
    let booleanity_catalog = fold_first_booleanity_catalog();
    let probe_layer = phase_timing_layer();
    let (fold_stage_ms, decide_stage_ms) = {
        let mut best = (f64::MAX, f64::MAX);
        for _pass in 0..3 {
            let mut transcript = Blake3Transcript::new();
            let fold_start = Instant::now();
            let (_fresh, fold_output) = fold_prepared_fold_first_mixed_hyrax::<
                C,
                U,
                RealEcdsaBenchZincTypes,
                HyraxF,
                DEGREE_PLUS_ONE,
            >(
                &pp,
                &shape,
                &prepared_instances,
                booleanity_catalog,
                &mut transcript,
            )
            .expect("fold-first sweep fold stage failed");
            let fold_ms = elapsed_ms(fold_start);
            let decide_start = Instant::now();
            decide_fold_first_mixed_hyrax::<C, U, RealEcdsaBenchZincTypes, HyraxF, DEGREE_PLUS_ONE>(
                &pp,
                &fold_output,
                &mut transcript,
            )
            .expect("fold-first sweep decide stage failed");
            best = (best.0.min(fold_ms), best.1.min(elapsed_ms(decide_start)));
        }
        best
    };
    let probe_commit_ms = {
        // Stage A (instance creation): the whole fresh-instances phase —
        // MSM commits plus instance assembly — none of which is folding.
        // Spans accumulate across the three probe passes; average them.
        let stage_a = probe_layer.total_ms(ProvePhase::FreshInstances) / 3.0;
        let wall = probe_layer.total_ms(ProvePhase::FreshCommitMixedHyraxInstances) / 3.0;
        if stage_a > 0.0 {
            stage_a
        } else if wall > 0.0 {
            wall
        } else {
            probe_layer.total_ms(ProvePhase::FreshCommitMixedHyraxInstance) / 3.0
        }
    };
    if env_bool_or("SHA256_COMBINED_SWEEP_TRACE_ONCE", false) {
        eprintln!("fold-first stage tracing: instances={N}");
        let subscriber = tracing_subscriber::fmt()
            .with_writer(std::io::stderr)
            .with_target(false)
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE)
            .finish();
        tracing::subscriber::with_default(subscriber, || {
            let mut transcript = Blake3Transcript::new();
            let (_fresh, fold_output) = fold_prepared_fold_first_mixed_hyrax::<
                C,
                U,
                RealEcdsaBenchZincTypes,
                HyraxF,
                DEGREE_PLUS_ONE,
            >(
                &pp,
                &shape,
                &prepared_instances,
                booleanity_catalog,
                &mut transcript,
            )
            .expect("traced fold-first fold stage failed");
            decide_fold_first_mixed_hyrax::<C, U, RealEcdsaBenchZincTypes, HyraxF, DEGREE_PLUS_ONE>(
                &pp,
                &fold_output,
                &mut transcript,
            )
            .expect("traced fold-first decide stage failed");
        });
    }

    if env_bool_or("SHA256_COMBINED_SWEEP_PROFILE_ONCE", false) {
        // Sub-phase breakdown: warm once, then measure one monolithic prove
        // under the per-`phase` timing layer and dump every span's busy time
        // plus commit / fold-core / decide roll-ups.
        let run_once = || {
            let mut transcript = Blake3Transcript::new();
            let (_fresh, fold_output) = fold_prepared_fold_first_mixed_hyrax::<
                C,
                U,
                RealEcdsaBenchZincTypes,
                HyraxF,
                DEGREE_PLUS_ONE,
            >(
                &pp,
                &shape,
                &prepared_instances,
                booleanity_catalog,
                &mut transcript,
            )
            .expect("profile fold stage failed");
            decide_fold_first_mixed_hyrax::<C, U, RealEcdsaBenchZincTypes, HyraxF, DEGREE_PLUS_ONE>(
                &pp,
                &fold_output,
                &mut transcript,
            )
            .expect("profile decide stage failed");
        };
        run_once();
        probe_layer.reset();
        let prove_start = Instant::now();
        run_once();
        let prove_wall = elapsed_ms(prove_start);

        let ms = |name: &str| probe_layer.raw_phase_ms(name);
        let commit = ms("fresh_instances");
        let precompute = ms("fresh_precompute_pcs");
        let commit_msm = ms("fresh_commit_mixed_hyrax_instances");
        let assembly = commit - precompute - commit_msm;
        let core = ms("fold_first_gram")
            + ms("fold_first_skip_round")
            + ms("fold_projected_traces")
            + ms("fold_first_folded_ideal")
            + ms("fold_prover_data");
        let decide = ms("row_sumcheck") + ms("endpoint_multipoint") + ms("pcs_opening");

        eprintln!(
            "=== fold-first PROFILE_ONCE N={N} tier={booleanity_catalog:?} prove_wall={prove_wall:.2}ms ==="
        );
        eprintln!("all spans (busy ms, parents include children):");
        for (name, span_ms) in probe_layer.all_phases_sorted() {
            eprintln!("  {span_ms:9.3}  {name}");
        }
        eprintln!(
            "roll-up: COMMIT {commit:.2} (precompute {precompute:.2} + msm {commit_msm:.2} + assembly {assembly:.2})  |  FOLD-CORE {core:.2}  |  DECIDE {decide:.2} (row {:.2} + endpoint {:.2} + open {:.2})",
            ms("row_sumcheck"),
            ms("endpoint_multipoint"),
            ms("pcs_opening"),
        );
        eprintln!(
            "decide detail: row_build {:.2} row_core {:.2} | endpoint_resolver {:.2} endpoint_reduce {:.2} endpoint_terminal {:.2} | pcs_open_core {:.2} pcs_lifted_evals {:.2}",
            ms("row_sumcheck_build_group"),
            ms("row_sumcheck_prove_core"),
            ms("endpoint_resolver"),
            ms("endpoint_reduce"),
            ms("endpoint_terminal"),
            ms("pcs_open_core"),
            ms("pcs_lifted_evals"),
        );
    }

    let measured_prover_warmups = warmup_runs.saturating_sub(1);
    let (output, prover_stats) = measure_warmed(measured_prover_warmups, sample_count, || {
        let mut transcript = Blake3Transcript::new();
        prove_prepared_fold_first_mixed_hyrax::<
            C,
            U,
            RealEcdsaBenchZincTypes,
            HyraxF,
            DEGREE_PLUS_ONE,
        >(
            &pp,
            &shape,
            &prepared_instances,
            booleanity_catalog,
            &mut transcript,
        )
        .expect("fold-first sweep mixed Hyrax prover failed")
    });
    let (_, verifier_stats) = measure_warmed(warmup_runs, sample_count, || {
        let mut transcript = Blake3Transcript::new();
        verify_fold_first_linear_ideal_fold_mixed_hyrax::<
            C,
            U,
            RealEcdsaBenchZincTypes,
            HyraxF,
            DEGREE_PLUS_ONE,
        >(
            &vs,
            &output.fresh_instances,
            &output.proof,
            booleanity_catalog,
            &mut transcript,
        )
        .expect("fold-first sweep mixed Hyrax verifier failed")
    });
    let raw = fold_first_mixed_hyrax_proof_raw_bytes(&output.proof);
    let ecc_points_per_commitment = output
        .proof
        .instance_commitments
        .first()
        .map(|commitment| {
            commitment.binary.group_point_count() + commitment.int.group_point_count()
        })
        .unwrap_or_default();
    let ecc_points_per_proof = output
        .proof
        .instance_commitments
        .iter()
        .map(|commitment| {
            commitment.binary.group_point_count() + commitment.int.group_point_count()
        })
        .sum();

    // Column mapping for the fold-first rows: probe_commit = fresh commits,
    // probe_sumfold = folding core (fold stage minus commits),
    // probe_fold = fold-stage wall total, probe_open = deferred decider.
    let prove_phases = HyraxProvePhaseTimings {
        prove_ms: prover_stats.median_ms,
        commit_ms: probe_commit_ms,
        sumfold_ms: (fold_stage_ms - probe_commit_ms).max(0.0),
        fold_ms: fold_stage_ms,
        open_ms: decide_stage_ms,
        ..Default::default()
    };
    let mut row = hyrax_instance_sweep_row(
        N,
        ell,
        0,
        width,
        ecc_points_per_commitment,
        ecc_points_per_proof,
        setup_prepare,
        prove_phases,
        prover_stats,
        verifier_stats,
        raw.len(),
        zstd_len(&raw),
    );
    row.variant = format!("fold-first instances {N}");
    row
}

fn combined_row_from_fold_first(row: HyraxInstanceSweepRow) -> Sha256CombinedInstanceSweepRow {
    let mut combined = combined_row_from_l0(row);
    combined.algorithm = "Fold-First SumFold ProjectionFold Mixed Hyrax".to_string();
    combined.variant = "fold-first".to_string();
    combined.l0 = None;
    combined.tail_vars = None;
    combined
}

fn env_usize_or(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_bool_or(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            _ => panic!("{name} must be a boolean value"),
        })
        .unwrap_or(default)
}

fn env_booleanity_catalog_or(name: &str, default: ShaBooleanityCatalog) -> ShaBooleanityCatalog {
    std::env::var(name)
        .ok()
        .map(|value| match value.trim().to_ascii_lowercase().as_str() {
            "full" => ShaBooleanityCatalog::Full,
            "tier1" => ShaBooleanityCatalog::Tier1DropChMajAux,
            "tier2" => ShaBooleanityCatalog::Tier2DropXorResults,
            _ => panic!("{name} must be one of full, tier1, tier2"),
        })
        .unwrap_or(default)
}

fn fold_first_booleanity_catalog() -> ShaBooleanityCatalog {
    env_booleanity_catalog_or(
        "SHA256_COMBINED_SWEEP_BOOLEANITY_TIER",
        ShaBooleanityCatalog::Tier2DropXorResults,
    )
}

fn v1_booleanity_catalog() -> ShaBooleanityCatalog {
    env_booleanity_catalog_or(
        "SHA256_COMBINED_SWEEP_V1_BOOLEANITY_TIER",
        ShaBooleanityCatalog::Full,
    )
}

fn env_usize_filter(name: &str) -> Option<Vec<usize>> {
    let raw = std::env::var(name).ok()?;
    let values = raw
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| {
            value
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("{name} must contain comma-separated usize values"))
        })
        .collect::<Vec<_>>();
    assert!(!values.is_empty(), "{name} must not be empty");
    Some(values)
}

fn enabled_by_filter(filter: Option<&[usize]>, value: usize) -> bool {
    filter.map_or(true, |values| values.contains(&value))
}

fn og_sha256_num_vars(instances: usize) -> (usize, usize, usize) {
    let active_rows = instances
        .checked_mul(sha256_cols::ROWS_PER_COMP)
        .and_then(|rows| rows.checked_add(4))
        .expect("OG SHA-256 sweep active row count fits usize");
    let domain_rows = active_rows.next_power_of_two();
    let num_vars = log2_power_of_two(domain_rows);
    (active_rows, domain_rows, num_vars)
}

fn skipped_og_sha256_row(
    algorithm: &'static str,
    error: String,
    instances: usize,
    pcs_width: usize,
    active_rows: usize,
    domain_rows: usize,
    num_vars: usize,
    setup_ms: Option<f64>,
) -> OgSha256InstanceSweepRow {
    OgSha256InstanceSweepRow {
        algorithm,
        status: "skipped",
        error,
        instances,
        pcs_width,
        active_rows,
        domain_rows,
        num_vars,
        setup_ms,
        prover_ms: None,
        prover_mean_ms: None,
        prover_min_ms: None,
        prover_max_ms: None,
        prover_samples: None,
        verifier_ms: None,
        verifier_mean_ms: None,
        verifier_min_ms: None,
        verifier_max_ms: None,
        verifier_samples: None,
        proof_bytes: None,
        proof_zstd_bytes: None,
    }
}

fn og_sha256_all_hyrax_pcs_params<C: AffineRepr>(
    width: usize,
) -> (
    PCSParams<AllHyraxPCSTypes<C>, RealEcdsaBenchZincTypes, F, DEGREE_PLUS_ONE>,
    PCSVerifierParams<AllHyraxPCSTypes<C>, RealEcdsaBenchZincTypes, F, DEGREE_PLUS_ONE>,
)
where
    F: zip_plus::pcs::hyrax::HyraxFieldBridge<C>,
    AllHyraxPCSTypes<C>: ZincPCSTypes<
            RealEcdsaBenchZincTypes,
            F,
            DEGREE_PLUS_ONE,
            BinaryPCS = HyraxPCS<C, BinaryLanes>,
            ArbitraryPCS = HyraxPCS<C, DensePolyScalarLanes>,
            IntPCS = HyraxPCS<C, IntScalarLane>,
        >,
{
    let (binary_ck, binary_vk) = HyraxPCS::<C, BinaryLanes>::setup(
        width,
        b"og-zinc-sha256-sweep-binary",
        HyraxBlindingMode::Unblinded,
    )
    .expect("OG SHA-256 all-Hyrax binary setup should be valid");
    let (arbitrary_ck, arbitrary_vk) = HyraxPCS::<C, DensePolyScalarLanes>::setup(
        width,
        b"og-zinc-sha256-sweep-arbitrary",
        HyraxBlindingMode::Unblinded,
    )
    .expect("OG SHA-256 all-Hyrax arbitrary setup should be valid");
    let (int_ck, int_vk) = HyraxPCS::<C, IntScalarLane>::setup(
        width,
        b"og-zinc-sha256-sweep-int",
        HyraxBlindingMode::Unblinded,
    )
    .expect("OG SHA-256 all-Hyrax int setup should be valid");

    (
        PCSParams::<AllHyraxPCSTypes<C>, RealEcdsaBenchZincTypes, F, DEGREE_PLUS_ONE> {
            binary: binary_ck,
            arbitrary: arbitrary_ck,
            int: int_ck,
        },
        PCSVerifierParams::<AllHyraxPCSTypes<C>, RealEcdsaBenchZincTypes, F, DEGREE_PLUS_ONE> {
            binary: binary_vk,
            arbitrary: arbitrary_vk,
            int: int_vk,
        },
    )
}

fn try_zip_pcs_params(
    num_vars: usize,
) -> Result<
    (
        PCSParams<AllZipPCSTypes, RealEcdsaBenchZincTypes, F, DEGREE_PLUS_ONE>,
        PCSVerifierParams<AllZipPCSTypes, RealEcdsaBenchZincTypes, F, DEGREE_PLUS_ONE>,
    ),
    String,
> {
    let pp = try_setup_pp_real_ecdsa(num_vars)?;
    Ok((
        PCSParams::<AllZipPCSTypes, RealEcdsaBenchZincTypes, F, DEGREE_PLUS_ONE> {
            binary: pp.0.clone(),
            arbitrary: pp.1.clone(),
            int: pp.2.clone(),
        },
        PCSVerifierParams::<AllZipPCSTypes, RealEcdsaBenchZincTypes, F, DEGREE_PLUS_ONE> {
            binary: pp.0,
            arbitrary: pp.1,
            int: pp.2,
        },
    ))
}

fn measure_og_zinc_sha256_zip_instances<const N: usize>(
    warmup_runs: usize,
    sample_count: usize,
) -> OgSha256InstanceSweepRow {
    type U = Sha256CompressionSliceUair<RealEcdsaInt>;
    type P = AllZipPCSTypes;

    let (active_rows, domain_rows, num_vars) = og_sha256_num_vars(N);
    eprintln!(
        "measuring OG Zinc+ Zip SHA-256 instances={N}, active_rows={active_rows}, num_vars={num_vars}, warmups={warmup_runs}, samples={sample_count}"
    );

    let setup_start = Instant::now();
    let (pp, vp) = match try_zip_pcs_params(num_vars) {
        Ok(params) => params,
        Err(error) => {
            return skipped_og_sha256_row(
                "OG Zinc+ ZipBn254",
                error,
                N,
                domain_rows,
                active_rows,
                domain_rows,
                num_vars,
                Some(elapsed_ms(setup_start)),
            );
        }
    };
    let message_blocks = exact_sha256_chain_blocks::<N>();
    let (trace, _final_state) = synthesize_sha256_chain_trace::<RealEcdsaInt, N>(
        num_vars,
        SHA256_INITIAL_STATE,
        message_blocks,
    )
    .expect("OG Zinc+ Zip SHA chain trace synthesis should succeed");
    let public_trace = trace.public(&U::signature());
    let field_cfg = field_cfg_from_curve_scalar::<
        F,
        <RealEcdsaBenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::Fmod,
        ark_bn254::G1Affine,
    >();
    let setup_ms = elapsed_ms(setup_start);

    let (proof, prover_stats) = measure_warmed(warmup_runs, sample_count, || {
        ZincPlusPiop::<RealEcdsaBenchZincTypes, U, F, DEGREE_PLUS_ONE>::prove_with_pcs_and_field_cfg::<
            P,
            false,
            PERFORM_CHECKS,
        >(
            &pp,
            &trace,
            num_vars,
            zinc_protocol::project_scalar_fn,
            field_cfg.clone(),
        )
        .expect("OG Zinc+ Zip SHA prover failed")
    });
    let (_, verifier_stats) = measure_warmed(warmup_runs, sample_count, || {
        ZincPlusPiop::<RealEcdsaBenchZincTypes, U, F, DEGREE_PLUS_ONE>::verify_with_pcs_and_field_cfg::<
            P,
            Sha256Ideal<F>,
            PERFORM_CHECKS,
        >(
            &vp,
            proof.clone(),
            &public_trace,
            num_vars,
            zinc_protocol::project_scalar_fn,
            sha256_real_project_ideal,
            field_cfg.clone(),
        )
        .expect("OG Zinc+ Zip SHA verifier failed")
    });
    let raw = generic_pcs_proof_raw_bytes::<P, RealEcdsaBenchZincTypes>(&proof);

    OgSha256InstanceSweepRow {
        algorithm: "OG Zinc+ ZipBn254",
        status: "ok",
        error: String::new(),
        instances: N,
        pcs_width: domain_rows,
        active_rows,
        domain_rows,
        num_vars,
        setup_ms: Some(setup_ms),
        prover_ms: Some(prover_stats.median_ms),
        prover_mean_ms: Some(prover_stats.mean_ms),
        prover_min_ms: Some(prover_stats.min_ms),
        prover_max_ms: Some(prover_stats.max_ms),
        prover_samples: Some(prover_stats.samples),
        verifier_ms: Some(verifier_stats.median_ms),
        verifier_mean_ms: Some(verifier_stats.mean_ms),
        verifier_min_ms: Some(verifier_stats.min_ms),
        verifier_max_ms: Some(verifier_stats.max_ms),
        verifier_samples: Some(verifier_stats.samples),
        proof_bytes: Some(raw.len()),
        proof_zstd_bytes: Some(zstd_len(&raw)),
    }
}

fn measure_og_zinc_sha256_all_hyrax_instances<const N: usize>(
    warmup_runs: usize,
    sample_count: usize,
    pcs_width: usize,
) -> OgSha256InstanceSweepRow {
    type U = Sha256CompressionSliceUair<RealEcdsaInt>;
    type C = ark_bn254::G1Affine;
    type P = AllHyraxPCSTypes<C>;

    let (active_rows, domain_rows, num_vars) = og_sha256_num_vars(N);
    eprintln!(
        "measuring OG Zinc+ SHA-256 instances={N}, active_rows={active_rows}, num_vars={num_vars}, pcs_width={pcs_width}, warmups={warmup_runs}, samples={sample_count}"
    );

    let setup_start = Instant::now();
    let message_blocks = exact_sha256_chain_blocks::<N>();
    let (trace, _final_state) = synthesize_sha256_chain_trace::<RealEcdsaInt, N>(
        num_vars,
        SHA256_INITIAL_STATE,
        message_blocks,
    )
    .expect("OG Zinc+ SHA chain trace synthesis should succeed");
    let public_trace = trace.public(&U::signature());
    let field_cfg = field_cfg_from_curve_scalar::<
        F,
        <RealEcdsaBenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::Fmod,
        ark_bn254::G1Affine,
    >();
    let (pp, vp) = og_sha256_all_hyrax_pcs_params::<C>(pcs_width);
    let setup_ms = elapsed_ms(setup_start);

    let (proof, prover_stats) = measure_warmed(warmup_runs, sample_count, || {
        ZincPlusPiop::<RealEcdsaBenchZincTypes, U, F, DEGREE_PLUS_ONE>::prove_with_pcs_and_field_cfg::<
            P,
            false,
            PERFORM_CHECKS,
        >(
            &pp,
            &trace,
            num_vars,
            zinc_protocol::project_scalar_fn,
            field_cfg.clone(),
        )
        .expect("OG Zinc+ SHA prover failed")
    });
    let (_, verifier_stats) = measure_warmed(warmup_runs, sample_count, || {
        ZincPlusPiop::<RealEcdsaBenchZincTypes, U, F, DEGREE_PLUS_ONE>::verify_with_pcs_and_field_cfg::<
            P,
            Sha256Ideal<F>,
            PERFORM_CHECKS,
        >(
            &vp,
            proof.clone(),
            &public_trace,
            num_vars,
            zinc_protocol::project_scalar_fn,
            sha256_real_project_ideal,
            field_cfg.clone(),
        )
        .expect("OG Zinc+ SHA verifier failed")
    });
    let raw = generic_pcs_proof_raw_bytes::<P, RealEcdsaBenchZincTypes>(&proof);

    OgSha256InstanceSweepRow {
        algorithm: "OG Zinc+ AllHyraxBn254",
        status: "ok",
        error: String::new(),
        instances: N,
        pcs_width,
        active_rows,
        domain_rows,
        num_vars,
        setup_ms: Some(setup_ms),
        prover_ms: Some(prover_stats.median_ms),
        prover_mean_ms: Some(prover_stats.mean_ms),
        prover_min_ms: Some(prover_stats.min_ms),
        prover_max_ms: Some(prover_stats.max_ms),
        prover_samples: Some(prover_stats.samples),
        verifier_ms: Some(verifier_stats.median_ms),
        verifier_mean_ms: Some(verifier_stats.mean_ms),
        verifier_min_ms: Some(verifier_stats.min_ms),
        verifier_max_ms: Some(verifier_stats.max_ms),
        verifier_samples: Some(verifier_stats.samples),
        proof_bytes: Some(raw.len()),
        proof_zstd_bytes: Some(zstd_len(&raw)),
    }
}

macro_rules! for_sha256_instance_sizes {
    ($callback:ident $(, $arg:expr)* $(,)?) => {{
        $callback!(8 $(, $arg)*);
        $callback!(16 $(, $arg)*);
        $callback!(32 $(, $arg)*);
        $callback!(64 $(, $arg)*);
        $callback!(128 $(, $arg)*);
        $callback!(256 $(, $arg)*);
    }};
}

macro_rules! for_sha256_l0s {
    ($callback:ident, $n:tt $(, $arg:expr)* $(,)?) => {{
        $callback!($n, 3 $(, $arg)*);
        $callback!($n, 4 $(, $arg)*);
        $callback!($n, 5 $(, $arg)*);
    }};
}

macro_rules! push_og_sha256_instance_rows {
    ($n:literal, $rows:expr, $warmup_runs:expr, $sample_count:expr, $pcs_width:expr) => {{
        $rows.push(measure_og_zinc_sha256_zip_instances::<$n>(
            $warmup_runs,
            $sample_count,
        ));
        $rows.push(measure_og_zinc_sha256_all_hyrax_instances::<$n>(
            $warmup_runs,
            $sample_count,
            $pcs_width,
        ));
    }};
}

macro_rules! push_sha256_combined_instance_if_enabled {
    (
        $n:literal,
        $rows:expr,
        $warmup_runs:expr,
        $sample_count:expr,
        $pcs_width:expr,
        $include_og:expr,
        $enabled_instances:expr,
        $enabled_l0s:expr
    ) => {{
        if enabled_by_filter($enabled_instances, $n) {
            push_sha256_combined_instance::<$n>(
                &mut $rows,
                $warmup_runs,
                $sample_count,
                $pcs_width,
                $include_og,
                $enabled_l0s,
            );
        }
    }};
}

macro_rules! push_projectionfold_instance_row_for_l0 {
    ($n:literal, $l0:literal, $rows:expr) => {{
        if log2_power_of_two($n) >= $l0 {
            $rows.push(measure_projectionfold_mixed_hyrax_instances::<$n, $l0>());
        }
    }};
}

macro_rules! push_projectionfold_instance_rows {
    ($n:literal, $rows:expr) => {{
        for_sha256_l0s!(push_projectionfold_instance_row_for_l0, $n, $rows);
    }};
}

fn measure_og_zinc_sha256_instance_rows(
    warmup_runs: usize,
    sample_count: usize,
    pcs_width: usize,
) -> Vec<OgSha256InstanceSweepRow> {
    let mut rows = Vec::new();
    for_sha256_instance_sizes!(
        push_og_sha256_instance_rows,
        rows,
        warmup_runs,
        sample_count,
        pcs_width
    );
    rows
}

fn combined_row_from_og(row: OgSha256InstanceSweepRow) -> Sha256CombinedInstanceSweepRow {
    let variant = if row.algorithm.contains("Zip") {
        "zip".to_string()
    } else {
        format!("all-hyrax width {}", row.pcs_width)
    };
    Sha256CombinedInstanceSweepRow {
        algorithm: row.algorithm.to_string(),
        variant,
        status: row.status,
        error: row.error,
        instances: row.instances,
        ell: Some(log2_power_of_two(row.instances)),
        l0: None,
        tail_vars: None,
        width: Some(row.pcs_width),
        active_rows: Some(row.active_rows),
        domain_rows: Some(row.domain_rows),
        num_vars: Some(row.num_vars),
        setup_ms: row.setup_ms,
        trace_witness_ms: None,
        pcs_setup_ms: None,
        prepare_sumfold_basis_ms: None,
        prepare_ms: None,
        setup_prepare_ms: None,
        probe_prove_ms: None,
        probe_commit_ms: None,
        probe_sumfold_linear_ms: None,
        probe_sumfold_booleanity_ms: None,
        probe_sumfold_group_ms: None,
        probe_sumfold_prove_rounds_ms: None,
        probe_sumfold_ms: None,
        probe_fold_ms: None,
        probe_open_ms: None,
        probe_open_core_ms: None,
        prover_ms: row.prover_ms,
        prover_mean_ms: row.prover_mean_ms,
        prover_min_ms: row.prover_min_ms,
        prover_max_ms: row.prover_max_ms,
        prover_samples: row.prover_samples,
        verifier_ms: row.verifier_ms,
        verifier_mean_ms: row.verifier_mean_ms,
        verifier_min_ms: row.verifier_min_ms,
        verifier_max_ms: row.verifier_max_ms,
        verifier_samples: row.verifier_samples,
        proof_bytes: row.proof_bytes,
        proof_zstd_bytes: row.proof_zstd_bytes,
    }
}

fn combined_row_from_l0(row: HyraxInstanceSweepRow) -> Sha256CombinedInstanceSweepRow {
    let num_vars = SHA_ROW_VARS + row.ell;
    Sha256CombinedInstanceSweepRow {
        algorithm: "Small-value SumFold ProjectionFold Mixed Hyrax".to_string(),
        variant: format!("l0 {}", row.l0),
        status: "ok",
        error: String::new(),
        instances: row.instances,
        ell: Some(row.ell),
        l0: Some(row.l0),
        tail_vars: Some(row.tail_vars),
        width: Some(row.width),
        active_rows: Some(row.instances * SHA_ROW_COUNT),
        domain_rows: Some(1usize << num_vars),
        num_vars: Some(num_vars),
        setup_ms: Some(row.setup_ms),
        trace_witness_ms: Some(row.trace_witness_ms),
        pcs_setup_ms: None,
        prepare_sumfold_basis_ms: Some(row.prepare_sumfold_basis_ms),
        prepare_ms: Some(row.prepare_ms),
        setup_prepare_ms: Some(row.setup_prepare_ms),
        probe_prove_ms: Some(row.probe_prove_ms),
        probe_commit_ms: Some(row.probe_commit_ms),
        probe_sumfold_linear_ms: Some(row.probe_sumfold_linear_ms),
        probe_sumfold_booleanity_ms: Some(row.probe_sumfold_booleanity_ms),
        probe_sumfold_group_ms: Some(row.probe_sumfold_group_ms),
        probe_sumfold_prove_rounds_ms: Some(row.probe_sumfold_prove_rounds_ms),
        probe_sumfold_ms: Some(row.probe_sumfold_ms),
        probe_fold_ms: Some(row.probe_fold_ms),
        probe_open_ms: Some(row.probe_open_ms),
        probe_open_core_ms: Some(row.probe_open_core_ms),
        prover_ms: Some(row.prover_ms),
        prover_mean_ms: Some(row.prover_mean_ms),
        prover_min_ms: Some(row.prover_min_ms),
        prover_max_ms: Some(row.prover_max_ms),
        prover_samples: Some(row.prover_samples),
        verifier_ms: Some(row.verifier_ms),
        verifier_mean_ms: Some(row.verifier_mean_ms),
        verifier_min_ms: Some(row.verifier_min_ms),
        verifier_max_ms: Some(row.verifier_max_ms),
        verifier_samples: Some(row.verifier_samples),
        proof_bytes: Some(row.proof_bytes),
        proof_zstd_bytes: Some(row.proof_zstd_bytes),
    }
}

fn skipped_l0_combined_row(
    instances: usize,
    l0: usize,
    width: usize,
) -> Sha256CombinedInstanceSweepRow {
    let ell = log2_power_of_two(instances);
    let num_vars = SHA_ROW_VARS + ell;
    Sha256CombinedInstanceSweepRow {
        algorithm: "Small-value SumFold ProjectionFold Mixed Hyrax".to_string(),
        variant: format!("l0 {l0}"),
        status: "skipped",
        error: format!("ell {ell} is smaller than l0 {l0}"),
        instances,
        ell: Some(ell),
        l0: Some(l0),
        tail_vars: None,
        width: Some(width),
        active_rows: Some(instances * SHA_ROW_COUNT),
        domain_rows: Some(1usize << num_vars),
        num_vars: Some(num_vars),
        setup_ms: None,
        trace_witness_ms: None,
        pcs_setup_ms: None,
        prepare_sumfold_basis_ms: None,
        prepare_ms: None,
        setup_prepare_ms: None,
        probe_prove_ms: None,
        probe_commit_ms: None,
        probe_sumfold_linear_ms: None,
        probe_sumfold_booleanity_ms: None,
        probe_sumfold_group_ms: None,
        probe_sumfold_prove_rounds_ms: None,
        probe_sumfold_ms: None,
        probe_fold_ms: None,
        probe_open_ms: None,
        probe_open_core_ms: None,
        prover_ms: None,
        prover_mean_ms: None,
        prover_min_ms: None,
        prover_max_ms: None,
        prover_samples: None,
        verifier_ms: None,
        verifier_mean_ms: None,
        verifier_min_ms: None,
        verifier_max_ms: None,
        verifier_samples: None,
        proof_bytes: None,
        proof_zstd_bytes: None,
    }
}

fn push_l0_combined_row<const N: usize, const L0: usize>(
    rows: &mut Vec<Sha256CombinedInstanceSweepRow>,
    warmup_runs: usize,
    sample_count: usize,
    pcs_width: usize,
    enabled_l0s: Option<&[usize]>,
) {
    if !enabled_by_filter(enabled_l0s, L0) {
        return;
    }
    if log2_power_of_two(N) < L0 {
        rows.push(skipped_l0_combined_row(N, L0, pcs_width));
        return;
    }
    rows.push(combined_row_from_l0(
        measure_projectionfold_mixed_hyrax_instances_with_samples::<N, L0>(
            warmup_runs,
            sample_count,
            pcs_width,
        ),
    ));
}

macro_rules! push_l0_combined_row_for_l0 {
    (
        $n:tt,
        $l0:literal,
        $rows:expr,
        $warmup_runs:expr,
        $sample_count:expr,
        $pcs_width:expr,
        $enabled_l0s:expr
    ) => {{
        push_l0_combined_row::<$n, $l0>(
            $rows,
            $warmup_runs,
            $sample_count,
            $pcs_width,
            $enabled_l0s,
        );
    }};
}

fn push_sha256_combined_instance<const N: usize>(
    rows: &mut Vec<Sha256CombinedInstanceSweepRow>,
    warmup_runs: usize,
    sample_count: usize,
    pcs_width: usize,
    include_og: bool,
    enabled_l0s: Option<&[usize]>,
) {
    if include_og {
        rows.push(combined_row_from_og(
            measure_og_zinc_sha256_zip_instances::<N>(warmup_runs, sample_count),
        ));
        rows.push(combined_row_from_og(
            measure_og_zinc_sha256_all_hyrax_instances::<N>(warmup_runs, sample_count, pcs_width),
        ));
    }

    for_sha256_l0s!(
        push_l0_combined_row_for_l0,
        N,
        rows,
        warmup_runs,
        sample_count,
        pcs_width,
        enabled_l0s
    );

    if env_bool_or("SHA256_COMBINED_SWEEP_INCLUDE_FOLD_FIRST", true) {
        rows.push(combined_row_from_fold_first(
            measure_foldfirst_mixed_hyrax_instances_with_samples::<N>(
                warmup_runs,
                sample_count,
                pcs_width,
            ),
        ));
    }
}

fn measure_sha256_combined_instance_rows(
    warmup_runs: usize,
    sample_count: usize,
    pcs_width: usize,
) -> Vec<Sha256CombinedInstanceSweepRow> {
    let mut rows = Vec::new();
    drop(projection_sha_bn254_hyrax_pcs_params(pcs_width));
    drop(projection_sha_bn254_mixed_hyrax_verifier_setup(pcs_width));
    let enabled_instances = env_usize_filter("SHA256_COMBINED_SWEEP_INSTANCES");
    let enabled_l0s = env_usize_filter("SHA256_COMBINED_SWEEP_L0S");
    let include_og = env_bool_or("SHA256_COMBINED_SWEEP_INCLUDE_OG", true);
    let enabled_instances = enabled_instances.as_deref();
    let enabled_l0s = enabled_l0s.as_deref();

    for_sha256_instance_sizes!(
        push_sha256_combined_instance_if_enabled,
        rows,
        warmup_runs,
        sample_count,
        pcs_width,
        include_og,
        enabled_instances,
        enabled_l0s
    );
    rows
}

pub fn run_og_sha256_instance_sweep_report() {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("protocol crate should live under the workspace root");
    let out_dir = workspace_root.join("target/og-sha256-instance-sweep");
    fs::create_dir_all(&out_dir).expect("create OG SHA-256 sweep output directory");

    let warmup_runs = env_usize_or(
        "OG_SHA256_SWEEP_WARMUP_RUNS",
        OG_SHA256_SWEEP_DEFAULT_WARMUP_RUNS,
    );
    let sample_count = env_usize_or("OG_SHA256_SWEEP_SAMPLES", OG_SHA256_SWEEP_DEFAULT_SAMPLES);
    let pcs_width = env_usize_or(
        "OG_SHA256_SWEEP_HYRAX_WIDTH",
        OG_SHA256_SWEEP_DEFAULT_HYRAX_WIDTH,
    );
    assert!(
        sample_count > 0,
        "OG SHA-256 sweep requires at least one sample"
    );

    let rows = measure_og_zinc_sha256_instance_rows(warmup_runs, sample_count, pcs_width);
    let csv_path = out_dir.join("results.csv");
    write_og_sha256_instance_sweep_csv(&csv_path, &rows);
    eprintln!("OG SHA-256 instance sweep wrote {}", csv_path.display());
}

pub fn run_sha256_combined_instance_sweep_report() {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("protocol crate should live under the workspace root");
    let out_dir = workspace_root.join("target/sha256-combined-instance-sweep");
    fs::create_dir_all(&out_dir).expect("create combined SHA-256 sweep output directory");

    let warmup_runs = env_usize_or(
        "SHA256_COMBINED_SWEEP_WARMUP_RUNS",
        SHA256_COMBINED_SWEEP_DEFAULT_WARMUP_RUNS,
    );
    let sample_count = env_usize_or(
        "SHA256_COMBINED_SWEEP_SAMPLES",
        SHA256_COMBINED_SWEEP_DEFAULT_SAMPLES,
    );
    let pcs_width = env_usize_or(
        "SHA256_COMBINED_SWEEP_HYRAX_WIDTH",
        OG_SHA256_SWEEP_DEFAULT_HYRAX_WIDTH,
    );
    assert!(
        sample_count > 0,
        "combined SHA-256 sweep requires at least one sample"
    );

    let rows = measure_sha256_combined_instance_rows(warmup_runs, sample_count, pcs_width);
    let csv_path = out_dir.join("results.csv");
    write_sha256_combined_instance_sweep_csv(&csv_path, &rows);
    eprintln!(
        "combined SHA-256 instance sweep wrote {}",
        csv_path.display()
    );
}

pub fn run_hyrax_width_sweep_report() {
    type C = ark_bn254::G1Affine;
    type ZipF = MontyField<FIELD_LIMBS>;
    type HyraxF = ArkFBn254;
    type P = AllHyraxPCSTypes<C>;
    type U = ProjectionShaBenchUair<RealEcdsaInt>;

    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("protocol crate should live under the workspace root");
    let out_dir = workspace_root.join("target/hyrax-width-sweep");
    fs::create_dir_all(&out_dir).expect("create hyrax width sweep output directory");

    let message_blocks = real_sha256_chain_blocks();
    let (_mono_trace, mono_final_state) =
        synthesize_sha256_chain_trace::<RealEcdsaInt, REAL_SHA256_CHAIN_BLOCKS>(
            REAL_SHA256_CHAIN_NUM_VARS,
            SHA256_INITIAL_STATE,
            message_blocks,
        )
        .expect("monolithic N=8 SHA trace synthesis should succeed");
    let (witnesses, projection_final_state) = synthesize_sha256_chain_witnesses::<
        RealEcdsaInt,
        REAL_SHA256_CHAIN_BLOCKS,
    >(SHA256_INITIAL_STATE, message_blocks)
    .expect("ProjectionFold SHA witness synthesis should succeed");
    assert_eq!(mono_final_state, projection_final_state);

    let shape = UairShape::<U>::new(SHA_ROW_VARS);
    let zip_field_cfg = field_cfg_from_curve_scalar::<
        ZipF,
        <RealEcdsaBenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::Fmod,
        C,
    >();
    let hyrax_field_cfg = HyraxF::curve_field_cfg::<C>();
    let prepared_instances = prepare_linear_ideal_fold_witnesses::<
        U,
        RealEcdsaBenchZincTypes,
        HyraxF,
        DEGREE_PLUS_ONE,
    >(&shape, &witnesses, &hyrax_field_cfg)
    .expect("ProjectionFold SHA witness preparation should succeed");

    let mut rows = Vec::new();

    let (zip_pp, zip_vp) = zip_pcs_params(REAL_SHA256_CHAIN_NUM_VARS);
    let trace = real_sha256_chain_trace(REAL_SHA256_CHAIN_NUM_VARS);
    let (zip_proof, zip_prover_stats) = measure_warmed(
        HYRAX_WIDTH_SWEEP_WARMUP_RUNS,
        HYRAX_WIDTH_SWEEP_TUNING_SAMPLES,
        || {
            ZincPlusPiop::<
                RealEcdsaBenchZincTypes,
                Sha256CompressionSliceUair<RealEcdsaInt>,
                ZipF,
                DEGREE_PLUS_ONE,
            >::prove_with_pcs_and_field_cfg::<AllZipPCSTypes, false, PERFORM_CHECKS>(
                &zip_pp,
                &trace,
                REAL_SHA256_CHAIN_NUM_VARS,
                zinc_protocol::project_scalar_fn,
                zip_field_cfg.clone(),
            )
            .expect("OG Zinc+ SHA prover failed")
        },
    );
    let public_trace = trace.public(&Sha256CompressionSliceUair::<RealEcdsaInt>::signature());
    let (_, zip_verifier_stats) = measure_warmed(
        HYRAX_WIDTH_SWEEP_WARMUP_RUNS,
        HYRAX_WIDTH_SWEEP_TUNING_SAMPLES,
        || {
            ZincPlusPiop::<
                RealEcdsaBenchZincTypes,
                Sha256CompressionSliceUair<RealEcdsaInt>,
                ZipF,
                DEGREE_PLUS_ONE,
            >::verify_with_pcs_and_field_cfg::<AllZipPCSTypes, _, PERFORM_CHECKS>(
                &zip_vp,
                zip_proof.clone(),
                &public_trace,
                REAL_SHA256_CHAIN_NUM_VARS,
                zinc_protocol::project_scalar_fn,
                sha256_real_project_ideal,
                zip_field_cfg.clone(),
            )
            .expect("OG Zinc+ SHA verifier failed")
        },
    );
    let zip_raw =
        generic_pcs_proof_raw_bytes::<AllZipPCSTypes, RealEcdsaBenchZincTypes>(&zip_proof);
    rows.push(hyrax_width_sweep_row(
        HyraxReportCategory::OgHash,
        "baseline".to_string(),
        None,
        None,
        None,
        zip_prover_stats,
        zip_verifier_stats,
        zip_raw.len(),
        zstd_len(&zip_raw),
    ));

    let measure_og_mixed_hyrax =
        |variant: String, width: usize, sample_count: usize| -> HyraxWidthSweepRow {
            let (hyrax_pp, hyrax_vp) =
                hyrax_pcs_params_with_width::<C>(REAL_SHA256_CHAIN_NUM_VARS, width);
            let (hyrax_proof, hyrax_prover_stats) =
                measure_warmed(HYRAX_WIDTH_SWEEP_WARMUP_RUNS, sample_count, || {
                    ZincPlusPiop::<
                        RealEcdsaBenchZincTypes,
                        Sha256CompressionSliceUair<RealEcdsaInt>,
                        ZipF,
                        DEGREE_PLUS_ONE,
                    >::prove_with_pcs_and_field_cfg::<
                        BinaryIntHyraxZipArbitrary<C>,
                        false,
                        PERFORM_CHECKS,
                    >(
                        &hyrax_pp,
                        &trace,
                        REAL_SHA256_CHAIN_NUM_VARS,
                        zinc_protocol::project_scalar_fn,
                        zip_field_cfg.clone(),
                    )
                    .expect("OG Zinc+ mixed-Hyrax SHA prover failed")
                });
            let (_, hyrax_verifier_stats) =
                measure_warmed(HYRAX_WIDTH_SWEEP_WARMUP_RUNS, sample_count, || {
                    ZincPlusPiop::<
                        RealEcdsaBenchZincTypes,
                        Sha256CompressionSliceUair<RealEcdsaInt>,
                        ZipF,
                        DEGREE_PLUS_ONE,
                    >::verify_with_pcs_and_field_cfg::<
                        BinaryIntHyraxZipArbitrary<C>,
                        _,
                        PERFORM_CHECKS,
                    >(
                        &hyrax_vp,
                        hyrax_proof.clone(),
                        &public_trace,
                        REAL_SHA256_CHAIN_NUM_VARS,
                        zinc_protocol::project_scalar_fn,
                        sha256_real_project_ideal,
                        zip_field_cfg.clone(),
                    )
                    .expect("OG Zinc+ mixed-Hyrax SHA verifier failed")
                });
            let hyrax_raw = generic_pcs_proof_raw_bytes::<
                BinaryIntHyraxZipArbitrary<C>,
                RealEcdsaBenchZincTypes,
            >(&hyrax_proof);
            let hyrax_ecc = hyrax_proof.commitments.binary.group_point_count()
                + hyrax_proof.commitments.int.group_point_count();
            hyrax_width_sweep_row(
                HyraxReportCategory::OgMixedHyrax,
                variant,
                Some(width),
                Some(hyrax_ecc),
                Some(hyrax_ecc),
                hyrax_prover_stats,
                hyrax_verifier_stats,
                hyrax_raw.len(),
                zstd_len(&hyrax_raw),
            )
        };

    let measure_projection_mixed_hyrax =
        |variant: String, width: usize, sample_count: usize| -> HyraxWidthSweepRow {
            let (mixed_pcs_params, _) = projection_sha_bn254_hyrax_pcs_params(width);
            let mixed_pp = LinearIdealFoldProverParams::<
                P,
                U,
                RealEcdsaBenchZincTypes,
                HyraxF,
                DEGREE_PLUS_ONE,
            >::new(mixed_pcs_params, hyrax_field_cfg.clone(), 3);
            let mixed_vs = projection_sha_bn254_mixed_hyrax_verifier_setup(width);
            let (mixed_output, mixed_prover_stats) =
                measure_warmed(HYRAX_WIDTH_SWEEP_WARMUP_RUNS, sample_count, || {
                    let mut transcript = Blake3Transcript::new();
                    prove_prepared_linear_ideal_fold_mixed_hyrax::<
                        C,
                        U,
                        RealEcdsaBenchZincTypes,
                        HyraxF,
                        DEGREE_PLUS_ONE,
                    >(
                        &mixed_pp,
                        &shape,
                        &prepared_instances,
                        ShaBooleanityCatalog::Full,
                        &mut transcript,
                    )
                    .expect("mixed Hyrax prover failed")
                });
            let (_, mixed_verifier_stats) =
                measure_warmed(HYRAX_WIDTH_SWEEP_WARMUP_RUNS, sample_count, || {
                    let mut transcript = Blake3Transcript::new();
                    verify_linear_ideal_fold_mixed_hyrax::<
                        C,
                        U,
                        RealEcdsaBenchZincTypes,
                        HyraxF,
                        DEGREE_PLUS_ONE,
                    >(
                        &mixed_vs,
                        &mixed_output.fresh_instances,
                        &mixed_output.proof,
                        ShaBooleanityCatalog::Full,
                        &mut transcript,
                    )
                    .expect("mixed Hyrax verifier failed")
                });
            let mixed_raw = production_mixed_hyrax_proof_raw_bytes(&mixed_output.proof);
            let mixed_ecc = mixed_output
                .proof
                .instance_commitments
                .first()
                .map(|commitment| {
                    commitment.binary.group_point_count() + commitment.int.group_point_count()
                })
                .unwrap_or_default();
            let mixed_fresh_ecc = mixed_output
                .proof
                .instance_commitments
                .iter()
                .map(|commitment| {
                    commitment.binary.group_point_count() + commitment.int.group_point_count()
                })
                .sum();
            hyrax_width_sweep_row(
                HyraxReportCategory::ProjectionFoldMixedHyrax,
                variant,
                Some(width),
                Some(mixed_ecc),
                Some(mixed_fresh_ecc),
                mixed_prover_stats,
                mixed_verifier_stats,
                mixed_raw.len(),
                zstd_len(&mixed_raw),
            )
        };

    let mixed_widths = mixed_width_candidates();
    for (variant, width) in &mixed_widths {
        rows.push(measure_og_mixed_hyrax(
            variant.clone(),
            *width,
            HYRAX_WIDTH_SWEEP_TUNING_SAMPLES,
        ));
        rows.push(measure_projection_mixed_hyrax(
            variant.clone(),
            *width,
            HYRAX_WIDTH_SWEEP_TUNING_SAMPLES,
        ));
    }

    let (widths, skipped_widths) = packed_width_candidates::<HyraxF>();
    let measure_packed_hyrax =
        |variant: String, width: usize, sample_count: usize| -> HyraxWidthSweepRow {
            let layout =
                packed_sha_layout::<HyraxF>(width).expect("candidate packed width is valid");
            let (pcs_params, verifier_params) =
                projection_sha_packed_hyrax_pcs_params::<C, HyraxF>(width);
            let pp = LinearIdealFoldProverParams::<
                P,
                U,
                RealEcdsaBenchZincTypes,
                HyraxF,
                DEGREE_PLUS_ONE,
            >::new(pcs_params, hyrax_field_cfg.clone(), 3);
            let vs = setup_verify_linear_ideal_fold_packed_hyrax::<
                C,
                U,
                RealEcdsaBenchZincTypes,
                HyraxF,
                DEGREE_PLUS_ONE,
            >(
                LinearIdealFoldVerifierParams::new(verifier_params, hyrax_field_cfg.clone()),
                shape.clone(),
            )
            .expect("packed Hyrax verifier setup succeeds");
            let (output, prover_stats) =
                measure_warmed(HYRAX_WIDTH_SWEEP_WARMUP_RUNS, sample_count, || {
                    let mut transcript = Blake3Transcript::new();
                    prove_prepared_linear_ideal_fold_packed_hyrax::<
                        C,
                        U,
                        RealEcdsaBenchZincTypes,
                        HyraxF,
                        DEGREE_PLUS_ONE,
                    >(
                        &pp,
                        &shape,
                        &prepared_instances,
                        ShaBooleanityCatalog::Full,
                        &mut transcript,
                    )
                    .expect("packed Hyrax prover failed")
                });
            let (_, verifier_stats) =
                measure_warmed(HYRAX_WIDTH_SWEEP_WARMUP_RUNS, sample_count, || {
                    let mut transcript = Blake3Transcript::new();
                    verify_linear_ideal_fold_packed_hyrax::<
                        C,
                        U,
                        RealEcdsaBenchZincTypes,
                        HyraxF,
                        DEGREE_PLUS_ONE,
                    >(
                        &vs,
                        &output.fresh_instances,
                        &output.proof,
                        ShaBooleanityCatalog::Full,
                        &mut transcript,
                    )
                    .expect("packed Hyrax verifier failed")
                });
            let raw = production_packed_hyrax_proof_raw_bytes(&output.proof);
            let fresh_ecc = output
                .proof
                .instance_commitments
                .iter()
                .map(|commitment| commitment.group_point_count())
                .sum();
            hyrax_width_sweep_row(
                HyraxReportCategory::ProjectionFoldPackedHyrax,
                variant,
                Some(width),
                Some(layout.ecc_points_per_instance()),
                Some(fresh_ecc),
                prover_stats,
                verifier_stats,
                raw.len(),
                zstd_len(&raw),
            )
        };

    for (label, width) in &widths {
        rows.push(measure_packed_hyrax(
            format!("packed {label}"),
            *width,
            HYRAX_WIDTH_SWEEP_TUNING_SAMPLES,
        ));
    }

    let mut confirmation_targets = rows
        .iter()
        .filter(|row| row.category != HyraxReportCategory::OgHash)
        .map(|row| (row.prover_ms, row.category, row.variant.clone(), row.width))
        .collect::<Vec<_>>();
    confirmation_targets.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    confirmation_targets.truncate(HYRAX_WIDTH_SWEEP_CONFIRMATION_TOP_K);
    let confirmation_rows = confirmation_targets
        .into_iter()
        .map(|(_, category, variant, width)| match category {
            HyraxReportCategory::OgMixedHyrax => measure_og_mixed_hyrax(
                variant,
                width.expect("OG mixed confirmation target has a width"),
                HYRAX_WIDTH_SWEEP_CONFIRMATION_SAMPLES,
            ),
            HyraxReportCategory::ProjectionFoldMixedHyrax => measure_projection_mixed_hyrax(
                variant,
                width.expect("ProjectionFold mixed confirmation target has a width"),
                HYRAX_WIDTH_SWEEP_CONFIRMATION_SAMPLES,
            ),
            HyraxReportCategory::ProjectionFoldPackedHyrax => measure_packed_hyrax(
                variant,
                width.expect("packed confirmation target has a width"),
                HYRAX_WIDTH_SWEEP_CONFIRMATION_SAMPLES,
            ),
            HyraxReportCategory::OgHash => unreachable!("OG hash is not selected for confirmation"),
        })
        .collect::<Vec<_>>();

    let include_instance_sweeps = env_bool_or("HYRAX_WIDTH_SWEEP_INCLUDE_INSTANCE_SWEEP", false);
    let instance_rows = if include_instance_sweeps {
        let mut rows = Vec::new();
        for_sha256_instance_sizes!(push_projectionfold_instance_rows, rows);
        rows
    } else {
        Vec::new()
    };

    write_hyrax_width_sweep_csv(&out_dir.join("results.csv"), &rows);
    write_hyrax_width_sweep_csv(&out_dir.join("confirmation.csv"), &confirmation_rows);
    if include_instance_sweeps {
        let og_instance_rows = measure_og_zinc_sha256_instance_rows(
            env_usize_or(
                "OG_SHA256_SWEEP_WARMUP_RUNS",
                OG_SHA256_SWEEP_DEFAULT_WARMUP_RUNS,
            ),
            env_usize_or("OG_SHA256_SWEEP_SAMPLES", OG_SHA256_SWEEP_DEFAULT_SAMPLES),
            env_usize_or(
                "OG_SHA256_SWEEP_HYRAX_WIDTH",
                OG_SHA256_SWEEP_DEFAULT_HYRAX_WIDTH,
            ),
        );
        write_hyrax_instance_sweep_csv(&out_dir.join("instance_results.csv"), &instance_rows);
        write_hyrax_instance_sweep_csv(&out_dir.join("l0_instance_results.csv"), &instance_rows);
        write_og_sha256_instance_sweep_csv(
            &out_dir.join("og_instance_results.csv"),
            &og_instance_rows,
        );
        write_hyrax_instance_sweep_plot(&out_dir, &instance_rows);
    }
    write_hyrax_width_sweep_skipped_csv(&out_dir.join("skipped_widths.csv"), &skipped_widths);
    write_hyrax_width_sweep_plots(&out_dir, &rows);
    write_hyrax_width_sweep_report(
        &out_dir,
        &rows,
        &instance_rows,
        &confirmation_rows,
        &skipped_widths,
    );
    if include_instance_sweeps {
        eprintln!(
            "hyrax width sweep wrote {}, {}, {}, {}, {}, {}, {}, and SVG/PNG plots",
            out_dir.join("results.csv").display(),
            out_dir.join("confirmation.csv").display(),
            out_dir.join("instance_results.csv").display(),
            out_dir.join("l0_instance_results.csv").display(),
            out_dir.join("og_instance_results.csv").display(),
            out_dir.join("skipped_widths.csv").display(),
            out_dir.join("report.html").display()
        );
    } else {
        eprintln!(
            "hyrax width sweep wrote {}, {}, {}, {}, and SVG/PNG plots",
            out_dir.join("results.csv").display(),
            out_dir.join("confirmation.csv").display(),
            out_dir.join("skipped_widths.csv").display(),
            out_dir.join("report.html").display()
        );
    }
}

//
// End-to-end benchmarks (total prove/verify time)
//

#[allow(clippy::too_many_arguments)]
fn do_bench_e2e<Zt, U, IdealOverF>(
    group: &mut BenchmarkGroup<WallTime>,
    label: &str,
    num_vars: usize,
    pp: &Pp<Zt>,
    trace: &UairTrace<'static, Zt::Int, Zt::Int, DEGREE_PLUS_ONE>,
    project_scalar: impl Fn(&U::Scalar, &<F as PrimeField>::Config) -> DynamicPolynomialF<F>
    + Copy
    + Sync,
    project_ideal: impl Fn(&IdealOrZero<U::Ideal>, &<F as PrimeField>::Config) -> IdealOverF + Copy,
) where
    Zt: ZincTypes<DEGREE_PLUS_ONE>,
    Zt::Int: ProjectableToField<F> + num_traits::Zero,
    <Zt::BinaryZt as ZipTypes>::Cw: ProjectableToField<F>,
    <Zt::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<F>,
    <Zt::ArbitraryZt as ZipTypes>::Cw: ProjectableToField<F>,
    <Zt::IntZt as ZipTypes>::Cw: ProjectableToField<F>,
    F: FromWithConfig<Zt::Int>
        + for<'a> FromWithConfig<&'a <Zt::BinaryZt as ZipTypes>::CombR>
        + for<'a> FromWithConfig<&'a <Zt::ArbitraryZt as ZipTypes>::CombR>
        + for<'a> FromWithConfig<&'a <Zt::IntZt as ZipTypes>::CombR>
        + for<'a> FromWithConfig<&'a Zt::Chal>
        + for<'a> FromWithConfig<&'a Zt::Pt>
        + for<'a> MulByScalar<&'a F>
        + FromRef<F>
        + Send
        + Sync
        + 'static,
    F: for<'a> FromWithConfig<&'a Zt::Int>,
    <F as Field>::Modulus: ConstTranscribable + FromRef<Zt::Fmod>,
    U: Uair + 'static,
    IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<F>>,
{
    let params = format!("{label}/nvars={num_vars}");

    macro_rules! zinc_plus {
        () => {
            ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>
        };
    }

    macro_rules! bench_prove {
        ($label:literal, $mle_first:expr) => {
            group.bench_function(BenchmarkId::new($label, &params), |bench| {
                bench.iter(|| {
                    black_box(<zinc_plus!()>::prove::<{ $mle_first }, PERFORM_CHECKS>(
                        pp,
                        trace,
                        num_vars,
                        project_scalar,
                    ))
                    .expect("Prover failed");
                });
            });
        };
    }

    bench_prove!("Prove (Combined)", false);

    // Effective max degree excludes zero-ideal (assert_zero) constraints,
    // which are skipped in the fq_sumcheck fold. So MLE-first is valid for
    // any UAIR whose *non*-zero-ideal constraints are all linear, even if
    // some assert_zero constraints have higher degree (e.g. ShaEcdsa).
    if count_effective_max_degree::<U>() <= 1 {
        bench_prove!("Prove (MLE-first)", true);
    }

    let proof: Proof<F> =
        <zinc_plus!()>::prove::<false, PERFORM_CHECKS>(pp, trace, num_vars, project_scalar)
            .expect("proof generation for verifier bench");

    let sig = U::signature();
    let public_trace = trace.public(&sig);

    group.bench_function(BenchmarkId::new("Verify", &params), |bench| {
        bench.iter_batched(
            || proof.clone(),
            |proof| {
                black_box(<zinc_plus!()>::verify::<_, PERFORM_CHECKS>(
                    pp,
                    proof,
                    &public_trace,
                    num_vars,
                    project_scalar,
                    project_ideal,
                ))
                .expect("Verifier failed");
            },
            BatchSize::SmallInput,
        );
    });

    eprint_proof_size(&params, &proof);
}

//
// Per-step benchmarks: each step is benchmarked in isolation by cloning
// cached intermediate state rather than re-running all preceding steps.
//

#[allow(clippy::too_many_arguments, clippy::unwrap_used)]
fn do_bench_steps<Zt, U, IdealOverF>(
    group: &mut BenchmarkGroup<WallTime>,
    label: &str,
    num_vars: usize,
    pp: &Pp<Zt>,
    trace: &UairTrace<'static, Zt::Int, Zt::Int, DEGREE_PLUS_ONE>,
    project_scalar: fn(&U::Scalar, &<F as PrimeField>::Config) -> DynamicPolynomialF<F>,
    project_ideal: impl Fn(&IdealOrZero<U::Ideal>, &<F as PrimeField>::Config) -> IdealOverF + Copy,
) where
    Zt: ZincTypes<DEGREE_PLUS_ONE>,
    Zt::Int: ProjectableToField<F> + num_traits::Zero,
    <Zt::BinaryZt as ZipTypes>::Cw: ProjectableToField<F>,
    <Zt::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<F>,
    <Zt::ArbitraryZt as ZipTypes>::Cw: ProjectableToField<F>,
    <Zt::IntZt as ZipTypes>::Cw: ProjectableToField<F>,
    F: FromWithConfig<Zt::Int>
        + for<'a> FromWithConfig<&'a <Zt::BinaryZt as ZipTypes>::CombR>
        + for<'a> FromWithConfig<&'a <Zt::ArbitraryZt as ZipTypes>::CombR>
        + for<'a> FromWithConfig<&'a <Zt::IntZt as ZipTypes>::CombR>
        + for<'a> FromWithConfig<&'a Zt::Chal>
        + for<'a> FromWithConfig<&'a Zt::Pt>
        + for<'a> MulByScalar<&'a F>
        + FromRef<F>
        + Send
        + Sync
        + 'static,
    F: for<'a> FromWithConfig<&'a Zt::Int>,
    <F as Field>::Modulus: ConstTranscribable + FromRef<Zt::Fmod>,
    U: Uair + 'static,
    IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<F>>,
{
    let params = format!("{label}/nvars={num_vars}");

    macro_rules! step_bench {
        ($side:literal / $step_name:literal, setup = || $setup:expr, run = |$s:ident| $run:expr $(,)?) => {
            group.bench_function(
                BenchmarkId::new(format!("{}/{}", $side, $step_name), &params),
                |b| {
                    b.iter_batched(
                        || $setup,
                        |$s| {
                            black_box($run).expect("step failed");
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        };
    }

    macro_rules! piop {
        () => {
            ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>
        };
    }

    //
    // Prover per-step benchmarks
    //

    // Build the chain once; each bench clones the cached state.

    let p_committed = <piop!()>::step0_commit(pp, trace, num_vars).unwrap();
    let p_projected = p_committed.clone().step1_combined(project_scalar).unwrap();
    let p_ideal_checked = p_projected.clone().step2_ideal_check().unwrap();
    let p_eval_projected = p_ideal_checked.clone().step3_eval_projection().unwrap();
    let p_sumchecked = p_eval_projected.clone().step4_sumcheck().unwrap();
    let p_mp_evaled = p_sumchecked.clone().step5_multipoint_eval().unwrap();
    let p_lifted = p_mp_evaled.clone().step6_lift_and_project().unwrap();

    step_bench!(
        "Prove" / "0: Commit",
        setup = || {},
        run = |_s| <piop!()>::step0_commit(pp, trace, num_vars),
    );

    step_bench!(
        "Prove" / "1: Prime projection (Combined)",
        setup = || p_committed.clone(),
        run = |s| s.step1_combined(project_scalar),
    );

    if count_effective_max_degree::<U>() <= 1 {
        step_bench!(
            "Prove" / "1: Prime projection (MLE-first)",
            setup = || p_committed.clone(),
            run = |s| s.step1_mle_first(project_scalar),
        );
    }

    step_bench!(
        "Prove" / "2: Ideal check (Combined)",
        setup = || p_projected.clone(),
        run = |s| s.step2_ideal_check(),
    );

    if count_effective_max_degree::<U>() <= 1 {
        let p_projected_mle = p_committed.clone().step1_mle_first(project_scalar).unwrap();
        step_bench!(
            "Prove" / "2: Ideal check (MLE-first)",
            setup = || p_projected_mle.clone(),
            run = |s| s.step2_ideal_check(),
        );
    }

    step_bench!(
        "Prove" / "3: Eval projection",
        setup = || p_ideal_checked.clone(),
        run = |s| s.step3_eval_projection(),
    );

    step_bench!(
        "Prove" / "4: Combined sumcheck",
        setup = || p_eval_projected.clone(),
        run = |s| s.step4_sumcheck(),
    );

    step_bench!(
        "Prove" / "5: Multi-point eval",
        setup = || p_sumchecked.clone(),
        run = |s| s.step5_multipoint_eval(),
    );

    step_bench!(
        "Prove" / "6: Lift-and-project",
        setup = || p_mp_evaled.clone(),
        run = |s| s.step6_lift_and_project(),
    );

    step_bench!(
        "Prove" / "7: PCS open",
        setup = || p_lifted.clone(),
        run = |s| s.step7_pcs_open::<PERFORM_CHECKS>(),
    );

    //
    // Verifier per-step benchmarks
    //

    macro_rules! zinc_plus {
        () => {
            ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>
        };
    }

    let proof: Proof<F> =
        <zinc_plus!()>::prove::<false, PERFORM_CHECKS>(pp, trace, num_vars, project_scalar)
            .expect("proof generation for verifier bench");

    let sig = U::signature();
    let public_trace = trace.public(&sig);

    let v_transcript = ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::step0_reconstruct_transcript::<
        IdealOverF,
    >(pp, proof.clone(), &public_trace, num_vars)
    .unwrap();
    let v_prime_projected = v_transcript.clone().step1_prime_projection().unwrap();
    let v_ideal_checked = v_prime_projected
        .clone()
        .step2_ideal_check(project_ideal)
        .unwrap();
    let v_eval_projected = v_ideal_checked
        .clone()
        .step3_eval_projection(project_scalar)
        .unwrap();
    let v_sumchecked = v_eval_projected.clone().step4_sumcheck_verify().unwrap();
    let v_mp_evaled = v_sumchecked.clone().step5_multipoint_eval::<U>().unwrap();
    let v_lifted = v_mp_evaled.clone().step6_lifted_evals::<U>().unwrap();

    step_bench!(
        "Verify" / "0: Transcript reconstruct",
        setup = || proof.clone(),
        run = |proof| ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::step0_reconstruct_transcript::<
            IdealOverF,
        >(pp, proof, &public_trace, num_vars,),
    );

    step_bench!(
        "Verify" / "1: Prime projection",
        setup = || v_transcript.clone(),
        run = |s| s.step1_prime_projection(),
    );

    step_bench!(
        "Verify" / "2: Ideal check",
        setup = || v_prime_projected.clone(),
        run = |s| s.step2_ideal_check(project_ideal),
    );

    step_bench!(
        "Verify" / "3: Eval projection",
        setup = || v_ideal_checked.clone(),
        run = |s| s.step3_eval_projection(project_scalar),
    );

    step_bench!(
        "Verify" / "4: Sumcheck verify",
        setup = || v_eval_projected.clone(),
        run = |s| s.step4_sumcheck_verify(),
    );

    step_bench!(
        "Verify" / "5: Multi-point eval",
        setup = || v_sumchecked.clone(),
        run = |s| s.step5_multipoint_eval::<U>(),
    );

    step_bench!(
        "Verify" / "6: Lifted evals",
        setup = || v_mp_evaled.clone(),
        run = |s| s.step6_lifted_evals::<U>(),
    );

    step_bench!(
        "Verify" / "7: PCS verify",
        setup = || v_lifted.clone(),
        run = |s| s.step7_pcs_verify::<U, PERFORM_CHECKS>(),
    );
}

fn append_transcribable_bytes<T: Transcribable>(out: &mut Vec<u8>, value: &T) {
    let offset = out.len();
    out.resize(offset + T::LENGTH_NUM_BYTES + value.get_num_bytes(), 0);
    let rest = value.write_transcription_bytes_subset(&mut out[offset..]);
    assert!(rest.is_empty(), "transcription buffer should be exact");
}

fn generic_pcs_proof_raw_bytes<P, Zt>(
    proof: &Proof<F, PCSCommitments<P, Zt, F, DEGREE_PLUS_ONE>>,
) -> Vec<u8>
where
    Zt: ZincTypes<DEGREE_PLUS_ONE>,
    P: ZincPCSTypes<Zt, F, DEGREE_PLUS_ONE>,
{
    let mut out = Vec::new();
    <<P as ZincPCSTypes<Zt, F, DEGREE_PLUS_ONE>>::BinaryPCS as PCS<
        F,
        BinaryPoly<DEGREE_PLUS_ONE>,
        DEGREE_PLUS_ONE,
    >>::write_commitment_bytes(&proof.commitments.binary, &mut out);
    <<P as ZincPCSTypes<Zt, F, DEGREE_PLUS_ONE>>::ArbitraryPCS as PCS<
        F,
        DensePolynomial<Zt::Int, DEGREE_PLUS_ONE>,
        DEGREE_PLUS_ONE,
    >>::write_commitment_bytes(&proof.commitments.arbitrary, &mut out);
    <<P as ZincPCSTypes<Zt, F, DEGREE_PLUS_ONE>>::IntPCS as PCS<
        F,
        Zt::Int,
        DEGREE_PLUS_ONE,
    >>::write_commitment_bytes(&proof.commitments.int, &mut out);

    let zip_len = u32::try_from(proof.zip.len()).expect("zip length must fit into u32");
    out.extend_from_slice(&zip_len.to_le_bytes());
    out.extend_from_slice(&proof.zip);
    append_transcribable_bytes(&mut out, &proof.ideal_check);
    append_transcribable_bytes(&mut out, &proof.resolver);
    append_transcribable_bytes(&mut out, &proof.combined_sumcheck);
    append_transcribable_bytes(&mut out, &proof.multipoint_eval);
    append_transcribable_bytes(
        &mut out,
        DynamicPolyVecF::reinterpret(&proof.witness_lifted_evals),
    );
    out
}

fn eprint_generic_pcs_proof_size<P, Zt>(
    label: &str,
    proof: &Proof<F, PCSCommitments<P, Zt, F, DEGREE_PLUS_ONE>>,
) where
    Zt: ZincTypes<DEGREE_PLUS_ONE>,
    P: ZincPCSTypes<Zt, F, DEGREE_PLUS_ONE>,
{
    let raw = generic_pcs_proof_raw_bytes::<P, Zt>(proof);
    eprint_bytes_size(label, &raw);
}

#[allow(clippy::too_many_arguments)]
fn do_bench_pcs_e2e<Zt, U, IdealOverF, P>(
    group: &mut BenchmarkGroup<WallTime>,
    label: &str,
    num_vars: usize,
    pp: &PCSParams<P, Zt, F, DEGREE_PLUS_ONE>,
    vp: &PCSVerifierParams<P, Zt, F, DEGREE_PLUS_ONE>,
    trace: &UairTrace<'static, Zt::Int, Zt::Int, DEGREE_PLUS_ONE>,
    field_cfg: <F as PrimeField>::Config,
    project_scalar: impl Fn(&U::Scalar, &<F as PrimeField>::Config) -> DynamicPolynomialF<F>
    + Copy
    + Sync,
    project_ideal: impl Fn(&IdealOrZero<U::Ideal>, &<F as PrimeField>::Config) -> IdealOverF + Copy,
) where
    Zt: ZincTypes<DEGREE_PLUS_ONE>,
    Zt::Int: ProjectableToField<F> + num_traits::Zero,
    <Zt::BinaryZt as ZipTypes>::Cw: ProjectableToField<F>,
    <Zt::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<F>,
    <Zt::ArbitraryZt as ZipTypes>::Cw: ProjectableToField<F>,
    <Zt::IntZt as ZipTypes>::Cw: ProjectableToField<F>,
    F: FromWithConfig<Zt::Int>
        + for<'a> FromWithConfig<&'a <Zt::BinaryZt as ZipTypes>::CombR>
        + for<'a> FromWithConfig<&'a <Zt::ArbitraryZt as ZipTypes>::CombR>
        + for<'a> FromWithConfig<&'a <Zt::IntZt as ZipTypes>::CombR>
        + for<'a> FromWithConfig<&'a Zt::Chal>
        + for<'a> FromWithConfig<&'a Zt::Pt>
        + for<'a> MulByScalar<&'a F>
        + FromRef<F>
        + Send
        + Sync
        + 'static,
    F: for<'a> FromWithConfig<&'a Zt::Int>,
    <F as Field>::Modulus: ConstTranscribable + FromRef<Zt::Fmod>,
    U: Uair + 'static,
    IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<F>>,
    P: ZincPCSTypes<Zt, F, DEGREE_PLUS_ONE>,
{
    let params = format!("{label}/nvars={num_vars}");

    macro_rules! zinc_plus {
        () => {
            ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>
        };
    }

    macro_rules! bench_prove {
        ($label:literal, $mle_first:expr) => {
            group.bench_function(BenchmarkId::new($label, &params), |bench| {
                bench.iter(|| {
                    black_box(<zinc_plus!()>::prove_with_pcs_and_field_cfg::<
                        P,
                        { $mle_first },
                        PERFORM_CHECKS,
                    >(
                        pp, trace, num_vars, project_scalar, field_cfg.clone()
                    ))
                    .expect("Prover failed");
                });
            });
        };
    }

    bench_prove!("Prove (Combined)", false);
    if count_effective_max_degree::<U>() <= 1 {
        bench_prove!("Prove (MLE-first)", true);
    }

    let proof = <zinc_plus!()>::prove_with_pcs_and_field_cfg::<P, false, PERFORM_CHECKS>(
        pp,
        trace,
        num_vars,
        project_scalar,
        field_cfg.clone(),
    )
    .expect("proof generation for verifier bench");

    let sig = U::signature();
    let public_trace = trace.public(&sig);

    group.bench_function(BenchmarkId::new("Verify", &params), |bench| {
        bench.iter_batched(
            || proof.clone(),
            |proof| {
                black_box(<zinc_plus!()>::verify_with_pcs_and_field_cfg::<
                    P,
                    IdealOverF,
                    PERFORM_CHECKS,
                >(
                    vp,
                    proof,
                    &public_trace,
                    num_vars,
                    project_scalar,
                    project_ideal,
                    field_cfg.clone(),
                ))
                .expect("Verifier failed");
            },
            BatchSize::SmallInput,
        );
    });

    eprint_generic_pcs_proof_size::<P, Zt>(&params, &proof);
}

#[allow(clippy::too_many_arguments, clippy::unwrap_used)]
fn do_bench_pcs_steps<Zt, U, IdealOverF, P>(
    group: &mut BenchmarkGroup<WallTime>,
    label: &str,
    num_vars: usize,
    pp: &PCSParams<P, Zt, F, DEGREE_PLUS_ONE>,
    vp: &PCSVerifierParams<P, Zt, F, DEGREE_PLUS_ONE>,
    trace: &UairTrace<'static, Zt::Int, Zt::Int, DEGREE_PLUS_ONE>,
    field_cfg: <F as PrimeField>::Config,
    project_scalar: fn(&U::Scalar, &<F as PrimeField>::Config) -> DynamicPolynomialF<F>,
    project_ideal: impl Fn(&IdealOrZero<U::Ideal>, &<F as PrimeField>::Config) -> IdealOverF + Copy,
) where
    Zt: ZincTypes<DEGREE_PLUS_ONE>,
    Zt::Int: ProjectableToField<F> + num_traits::Zero,
    <Zt::BinaryZt as ZipTypes>::Cw: ProjectableToField<F>,
    <Zt::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<F>,
    <Zt::ArbitraryZt as ZipTypes>::Cw: ProjectableToField<F>,
    <Zt::IntZt as ZipTypes>::Cw: ProjectableToField<F>,
    F: FromWithConfig<Zt::Int>
        + for<'a> FromWithConfig<&'a <Zt::BinaryZt as ZipTypes>::CombR>
        + for<'a> FromWithConfig<&'a <Zt::ArbitraryZt as ZipTypes>::CombR>
        + for<'a> FromWithConfig<&'a <Zt::IntZt as ZipTypes>::CombR>
        + for<'a> FromWithConfig<&'a Zt::Chal>
        + for<'a> FromWithConfig<&'a Zt::Pt>
        + for<'a> MulByScalar<&'a F>
        + FromRef<F>
        + Send
        + Sync
        + 'static,
    F: for<'a> FromWithConfig<&'a Zt::Int>,
    <F as Field>::Modulus: ConstTranscribable + FromRef<Zt::Fmod>,
    U: Uair + 'static,
    IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<F>>,
    P: ZincPCSTypes<Zt, F, DEGREE_PLUS_ONE>,
{
    let params = format!("{label}/nvars={num_vars}");

    macro_rules! step_bench {
        ($side:literal / $step_name:literal, setup = || $setup:expr, run = |$s:ident| $run:expr $(,)?) => {
            group.bench_function(
                BenchmarkId::new(format!("{}/{}", $side, $step_name), &params),
                |b| {
                    b.iter_batched(
                        || $setup,
                        |$s| {
                            black_box($run).expect("step failed");
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        };
    }

    macro_rules! piop {
        () => {
            ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>
        };
    }

    let p_committed = <piop!()>::step0_commit_with_pcs::<P>(pp, trace, num_vars).unwrap();
    let p_projected = p_committed
        .clone()
        .step1_combined_with_field_cfg(project_scalar, field_cfg.clone())
        .unwrap();
    let p_ideal_checked = p_projected.clone().step2_ideal_check().unwrap();
    let p_eval_projected = p_ideal_checked.clone().step3_eval_projection().unwrap();
    let p_sumchecked = p_eval_projected.clone().step4_sumcheck().unwrap();
    let p_mp_evaled = p_sumchecked.clone().step5_multipoint_eval().unwrap();
    let p_lifted = p_mp_evaled.clone().step6_lift_and_project().unwrap();

    step_bench!(
        "Prove" / "0: Commit",
        setup = || {},
        run = |_s| <piop!()>::step0_commit_with_pcs::<P>(pp, trace, num_vars),
    );

    step_bench!(
        "Prove" / "7: PCS open",
        setup = || p_lifted.clone(),
        run = |s| s.step7_pcs_open::<PERFORM_CHECKS>(),
    );

    let proof = <piop!()>::prove_with_pcs_and_field_cfg::<P, false, PERFORM_CHECKS>(
        pp,
        trace,
        num_vars,
        project_scalar,
        field_cfg.clone(),
    )
    .expect("proof generation for verifier bench");
    let sig = U::signature();
    let public_trace = trace.public(&sig);
    let v_transcript = <piop!()>::step0_reconstruct_transcript_with_pcs::<IdealOverF, P>(
        vp,
        proof,
        &public_trace,
        num_vars,
    )
    .unwrap();
    let v_prime_projected = v_transcript
        .clone()
        .step1_prime_projection_with_field_cfg(field_cfg.clone())
        .unwrap();
    let v_ideal_checked = v_prime_projected
        .clone()
        .step2_ideal_check(project_ideal)
        .unwrap();
    let v_eval_projected = v_ideal_checked
        .clone()
        .step3_eval_projection(project_scalar)
        .unwrap();
    let v_sumchecked = v_eval_projected.clone().step4_sumcheck_verify().unwrap();
    let v_mp_evaled = v_sumchecked.clone().step5_multipoint_eval::<U>().unwrap();
    let v_lifted = v_mp_evaled.clone().step6_lifted_evals::<U>().unwrap();

    step_bench!(
        "Verify" / "7: PCS verify",
        setup = || v_lifted.clone(),
        run = |s| s.step7_pcs_verify::<U, PERFORM_CHECKS>(),
    );
}

//
// Specific benchmarks for each UAIR
//

fn do_bench_uair<U>(group: &mut BenchmarkGroup<WallTime>, label: &str, num_vars: usize)
where
    U: Uair<
            Ideal = DegreeOneIdeal<<BenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::Int>,
            Scalar = DensePolynomial<<BenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::Int, 32>,
        > + GenerateRandomTrace<
            DEGREE_PLUS_ONE,
            PolyCoeff = <BenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::Int,
            Int = <BenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::Int,
        > + 'static,
    F: for<'a> FromWithConfig<&'a <BenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::Int>,
{
    let mut rng = rng();
    let trace = U::generate_random_trace(num_vars, &mut rng);

    let pp = setup_pp(num_vars);

    let proj_ideal = |ideal: &IdealOrZero<U::Ideal>, field_cfg: &<F as PrimeField>::Config| {
        ideal.map(|i| DegreeOneIdeal::from_with_cfg(i, field_cfg))
    };

    do_bench_e2e::<BenchZincTypes, U, _>(
        group,
        label,
        num_vars,
        &pp,
        &trace,
        zinc_protocol::project_scalar_fn,
        proj_ideal,
    );
}

fn do_bench_steps_uair<U>(group: &mut BenchmarkGroup<WallTime>, label: &str, num_vars: usize)
where
    U: Uair<
            Ideal = DegreeOneIdeal<<BenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::Int>,
            Scalar = DensePolynomial<<BenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::Int, 32>,
        > + GenerateRandomTrace<
            DEGREE_PLUS_ONE,
            PolyCoeff = <BenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::Int,
            Int = <BenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::Int,
        > + 'static,
    F: for<'a> FromWithConfig<&'a <BenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::Int>,
{
    let mut rng = rng();
    let trace = U::generate_random_trace(num_vars, &mut rng);

    let pp = setup_pp(num_vars);

    let proj_ideal = |ideal: &IdealOrZero<U::Ideal>, field_cfg: &<F as PrimeField>::Config| {
        ideal.map(|i| DegreeOneIdeal::from_with_cfg(i, field_cfg))
    };

    do_bench_steps::<BenchZincTypes, U, _>(
        group,
        label,
        num_vars,
        &pp,
        &trace,
        zinc_protocol::project_scalar_fn,
        proj_ideal,
    );
}

//
// Real-UAIR benches (ECDSA / SHA-256 / SHA+ECDSA from main-gamma).
//
// Each pair (`_e2e` and `_steps`) delegates to the generic `do_bench_e2e` /
// `do_bench_steps` helpers above with `RealEcdsaBenchZincTypes` (Int<5>),
// matching the eight-step taxonomy used by every other bench in this file.
//

fn bench_real_ecdsa_e2e(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    type U = EcdsaUair<RealEcdsaInt>;

    let mut rng = rng();
    let trace = U::generate_random_trace(num_vars, &mut rng);
    let pp = setup_pp_real_ecdsa(num_vars);

    let proj_ideal =
        |_: &IdealOrZero<<U as Uair>::Ideal>, _: &<F as PrimeField>::Config| -> ImpossibleIdeal {
            unreachable!("EcdsaUair has only assert_zero constraints")
        };

    do_bench_e2e::<RealEcdsaBenchZincTypes, U, _>(
        group,
        "RealEcdsa",
        num_vars,
        &pp,
        &trace,
        zinc_protocol::project_scalar_fn,
        proj_ideal,
    );
}

fn bench_real_ecdsa_steps(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    type U = EcdsaUair<RealEcdsaInt>;

    let mut rng = rng();
    let trace = U::generate_random_trace(num_vars, &mut rng);
    let pp = setup_pp_real_ecdsa(num_vars);

    let proj_ideal =
        |_: &IdealOrZero<<U as Uair>::Ideal>, _: &<F as PrimeField>::Config| -> ImpossibleIdeal {
            unreachable!("EcdsaUair has only assert_zero constraints")
        };

    do_bench_steps::<RealEcdsaBenchZincTypes, U, _>(
        group,
        "RealEcdsa",
        num_vars,
        &pp,
        &trace,
        zinc_protocol::project_scalar_fn,
        proj_ideal,
    );
}

fn bench_real_sha256_e2e(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    type U = Sha256CompressionSliceUair<RealEcdsaInt>;

    let trace = real_sha256_chain_trace(num_vars);
    let pp = setup_pp_real_ecdsa(num_vars);

    do_bench_e2e::<RealEcdsaBenchZincTypes, U, _>(
        group,
        "RealSha256Chain8",
        num_vars,
        &pp,
        &trace,
        zinc_protocol::project_scalar_fn,
        sha256_real_project_ideal,
    );
}

fn bench_real_sha256_steps(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    type U = Sha256CompressionSliceUair<RealEcdsaInt>;

    let trace = real_sha256_chain_trace(num_vars);
    let pp = setup_pp_real_ecdsa(num_vars);

    do_bench_steps::<RealEcdsaBenchZincTypes, U, _>(
        group,
        "RealSha256Chain8",
        num_vars,
        &pp,
        &trace,
        zinc_protocol::project_scalar_fn,
        sha256_real_project_ideal,
    );
}

fn zip_pcs_params(
    num_vars: usize,
) -> (
    PCSParams<AllZipPCSTypes, RealEcdsaBenchZincTypes, F, DEGREE_PLUS_ONE>,
    PCSVerifierParams<AllZipPCSTypes, RealEcdsaBenchZincTypes, F, DEGREE_PLUS_ONE>,
) {
    let pp = setup_pp_real_ecdsa(num_vars);
    (
        PCSParams::<AllZipPCSTypes, RealEcdsaBenchZincTypes, F, DEGREE_PLUS_ONE> {
            binary: pp.0.clone(),
            arbitrary: pp.1.clone(),
            int: pp.2.clone(),
        },
        PCSVerifierParams::<AllZipPCSTypes, RealEcdsaBenchZincTypes, F, DEGREE_PLUS_ONE> {
            binary: pp.0,
            arbitrary: pp.1,
            int: pp.2,
        },
    )
}

fn hyrax_pcs_params<C: AffineRepr>(
    num_vars: usize,
) -> (
    PCSParams<BinaryIntHyraxZipArbitrary<C>, RealEcdsaBenchZincTypes, F, DEGREE_PLUS_ONE>,
    PCSVerifierParams<BinaryIntHyraxZipArbitrary<C>, RealEcdsaBenchZincTypes, F, DEGREE_PLUS_ONE>,
)
where
    BinaryIntHyraxZipArbitrary<C>: ZincPCSTypes<
            RealEcdsaBenchZincTypes,
            F,
            DEGREE_PLUS_ONE,
            BinaryPCS = HyraxPCS<C, BinaryLanes>,
            ArbitraryPCS = ZipPlusPCS<
                <RealEcdsaBenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::ArbitraryZt,
                <RealEcdsaBenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::ArbitraryLc,
            >,
            IntPCS = HyraxPCS<C, IntScalarLane>,
        >,
{
    let pp = setup_pp_real_ecdsa(num_vars);
    let binary_width = pp.0.linear_code.row_len();
    let int_width = pp.2.linear_code.row_len();
    hyrax_pcs_params_from_zip_pp::<C>(pp, binary_width, int_width, "default")
}

fn hyrax_pcs_params_with_width<C: AffineRepr>(
    num_vars: usize,
    width: usize,
) -> (
    PCSParams<BinaryIntHyraxZipArbitrary<C>, RealEcdsaBenchZincTypes, F, DEGREE_PLUS_ONE>,
    PCSVerifierParams<BinaryIntHyraxZipArbitrary<C>, RealEcdsaBenchZincTypes, F, DEGREE_PLUS_ONE>,
)
where
    BinaryIntHyraxZipArbitrary<C>: ZincPCSTypes<
            RealEcdsaBenchZincTypes,
            F,
            DEGREE_PLUS_ONE,
            BinaryPCS = HyraxPCS<C, BinaryLanes>,
            ArbitraryPCS = ZipPlusPCS<
                <RealEcdsaBenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::ArbitraryZt,
                <RealEcdsaBenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::ArbitraryLc,
            >,
            IntPCS = HyraxPCS<C, IntScalarLane>,
        >,
{
    let pp = setup_pp_real_ecdsa(num_vars);
    let domain_suffix = format!("width-{width}");
    hyrax_pcs_params_from_zip_pp::<C>(pp, width, width, &domain_suffix)
}

fn hyrax_pcs_params_from_zip_pp<C: AffineRepr>(
    pp: Pp<RealEcdsaBenchZincTypes>,
    binary_width: usize,
    int_width: usize,
    domain_suffix: &str,
) -> (
    PCSParams<BinaryIntHyraxZipArbitrary<C>, RealEcdsaBenchZincTypes, F, DEGREE_PLUS_ONE>,
    PCSVerifierParams<BinaryIntHyraxZipArbitrary<C>, RealEcdsaBenchZincTypes, F, DEGREE_PLUS_ONE>,
)
where
    BinaryIntHyraxZipArbitrary<C>: ZincPCSTypes<
            RealEcdsaBenchZincTypes,
            F,
            DEGREE_PLUS_ONE,
            BinaryPCS = HyraxPCS<C, BinaryLanes>,
            ArbitraryPCS = ZipPlusPCS<
                <RealEcdsaBenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::ArbitraryZt,
                <RealEcdsaBenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::ArbitraryLc,
            >,
            IntPCS = HyraxPCS<C, IntScalarLane>,
        >,
{
    let binary_domain = format!("zinc-plus-bench-real-sha256-hyrax-binary-{domain_suffix}");
    let int_domain = format!("zinc-plus-bench-real-sha256-hyrax-int-{domain_suffix}");
    let (binary, binary_vk) = HyraxPCS::<C, BinaryLanes>::setup(
        binary_width,
        binary_domain.as_bytes(),
        HyraxBlindingMode::Unblinded,
    )
    .expect("Hyrax binary benchmark setup must be valid");
    let (int, int_vk) = HyraxPCS::<C, IntScalarLane>::setup(
        int_width,
        int_domain.as_bytes(),
        HyraxBlindingMode::Unblinded,
    )
    .expect("Hyrax int benchmark setup must be valid");
    (
        PCSParams::<BinaryIntHyraxZipArbitrary<C>, RealEcdsaBenchZincTypes, F, DEGREE_PLUS_ONE> {
            binary,
            arbitrary: pp.1.clone(),
            int,
        },
        PCSVerifierParams::<
            BinaryIntHyraxZipArbitrary<C>,
            RealEcdsaBenchZincTypes,
            F,
            DEGREE_PLUS_ONE,
        > {
            binary: binary_vk,
            arbitrary: pp.1,
            int: int_vk,
        },
    )
}

fn bench_real_sha256_pcs_curve_e2e<C: AffineRepr>(
    group: &mut BenchmarkGroup<WallTime>,
    num_vars: usize,
    zip_label: &str,
    hyrax_label: &str,
) where
    BinaryIntHyraxZipArbitrary<C>: ZincPCSTypes<
            RealEcdsaBenchZincTypes,
            F,
            DEGREE_PLUS_ONE,
            BinaryPCS = HyraxPCS<C, BinaryLanes>,
            ArbitraryPCS = ZipPlusPCS<
                <RealEcdsaBenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::ArbitraryZt,
                <RealEcdsaBenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::ArbitraryLc,
            >,
            IntPCS = HyraxPCS<C, IntScalarLane>,
        >,
{
    type U = Sha256CompressionSliceUair<RealEcdsaInt>;

    let trace = real_sha256_chain_trace(num_vars);
    let field_cfg = field_cfg_from_curve_scalar::<
        F,
        <RealEcdsaBenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::Fmod,
        C,
    >();

    let (zip_pp, zip_vp) = zip_pcs_params(num_vars);
    do_bench_pcs_e2e::<RealEcdsaBenchZincTypes, U, _, AllZipPCSTypes>(
        group,
        zip_label,
        num_vars,
        &zip_pp,
        &zip_vp,
        &trace,
        field_cfg.clone(),
        zinc_protocol::project_scalar_fn,
        sha256_real_project_ideal,
    );

    let (hyrax_pp, hyrax_vp) = hyrax_pcs_params::<C>(num_vars);
    do_bench_pcs_e2e::<RealEcdsaBenchZincTypes, U, _, BinaryIntHyraxZipArbitrary<C>>(
        group,
        hyrax_label,
        num_vars,
        &hyrax_pp,
        &hyrax_vp,
        &trace,
        field_cfg,
        zinc_protocol::project_scalar_fn,
        sha256_real_project_ideal,
    );
}

fn bench_real_sha256_pcs_curve_steps<C: AffineRepr>(
    group: &mut BenchmarkGroup<WallTime>,
    num_vars: usize,
    zip_label: &str,
    hyrax_label: &str,
) where
    BinaryIntHyraxZipArbitrary<C>: ZincPCSTypes<
            RealEcdsaBenchZincTypes,
            F,
            DEGREE_PLUS_ONE,
            BinaryPCS = HyraxPCS<C, BinaryLanes>,
            ArbitraryPCS = ZipPlusPCS<
                <RealEcdsaBenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::ArbitraryZt,
                <RealEcdsaBenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::ArbitraryLc,
            >,
            IntPCS = HyraxPCS<C, IntScalarLane>,
        >,
{
    type U = Sha256CompressionSliceUair<RealEcdsaInt>;

    let trace = real_sha256_chain_trace(num_vars);
    let field_cfg = field_cfg_from_curve_scalar::<
        F,
        <RealEcdsaBenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::Fmod,
        C,
    >();

    let (zip_pp, zip_vp) = zip_pcs_params(num_vars);
    do_bench_pcs_steps::<RealEcdsaBenchZincTypes, U, _, AllZipPCSTypes>(
        group,
        zip_label,
        num_vars,
        &zip_pp,
        &zip_vp,
        &trace,
        field_cfg.clone(),
        zinc_protocol::project_scalar_fn,
        sha256_real_project_ideal,
    );

    let (hyrax_pp, hyrax_vp) = hyrax_pcs_params::<C>(num_vars);
    do_bench_pcs_steps::<RealEcdsaBenchZincTypes, U, _, BinaryIntHyraxZipArbitrary<C>>(
        group,
        hyrax_label,
        num_vars,
        &hyrax_pp,
        &hyrax_vp,
        &trace,
        field_cfg,
        zinc_protocol::project_scalar_fn,
        sha256_real_project_ideal,
    );
}

fn bench_real_sha256_pcs_e2e(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    bench_real_sha256_pcs_curve_e2e::<ark_bn254::G1Affine>(
        group,
        num_vars,
        "RealSha256Chain8PCS/ZipBn254Fr",
        "RealSha256Chain8PCS/HyraxBn254Unblinded",
    );
    bench_real_sha256_pcs_curve_e2e::<ark_secp256k1::Affine>(
        group,
        num_vars,
        "RealSha256Chain8PCS/ZipSecp256k1Fr",
        "RealSha256Chain8PCS/HyraxSecp256k1Unblinded",
    );
}

fn bench_real_sha256_pcs_steps(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    bench_real_sha256_pcs_curve_steps::<ark_bn254::G1Affine>(
        group,
        num_vars,
        "RealSha256Chain8PCS/ZipBn254Fr",
        "RealSha256Chain8PCS/HyraxBn254Unblinded",
    );
    bench_real_sha256_pcs_curve_steps::<ark_secp256k1::Affine>(
        group,
        num_vars,
        "RealSha256Chain8PCS/ZipSecp256k1Fr",
        "RealSha256Chain8PCS/HyraxSecp256k1Unblinded",
    );
}

fn bench_og_sha256_zip_compare(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    type U = Sha256CompressionSliceUair<RealEcdsaInt>;

    let trace = real_sha256_chain_trace(num_vars);
    let field_cfg = field_cfg_from_curve_scalar::<
        F,
        <RealEcdsaBenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::Fmod,
        ark_bn254::G1Affine,
    >();
    let (zip_pp, zip_vp) = zip_pcs_params(num_vars);

    do_bench_pcs_e2e::<RealEcdsaBenchZincTypes, U, _, AllZipPCSTypes>(
        group,
        "OG-ZincPlus-ZipBn254/SHA256Chain8",
        num_vars,
        &zip_pp,
        &zip_vp,
        &trace,
        field_cfg,
        zinc_protocol::project_scalar_fn,
        sha256_real_project_ideal,
    );
}

fn bench_projectionfold_sha256_concise_hyrax<C, F>(
    group: &mut BenchmarkGroup<WallTime>,
    label: &str,
) where
    C: ProductionShaMixedHyraxPcs<RealEcdsaBenchZincTypes, F, DEGREE_PLUS_ONE>
        + Send
        + Sync
        + 'static,
    F: BenchShaField
        + InnerTransparentField
        + DelayedFieldProductSum
        + ShaBinaryFoldField
        + ShaLinearAccumulatorField
        + ShaSmallFieldDecode
        + ShaSuffixScannerField
        + zip_plus::pcs::hyrax::HyraxFieldBridge<C>
        + Send
        + Sync
        + 'static,
    F::Inner: Transcribable + Default + Send + Sync,
    F::Modulus: Transcribable,
{
    type P<C> = AllHyraxPCSTypes<C>;
    type U = ProjectionShaBenchUair<RealEcdsaInt>;

    let message_blocks = real_sha256_chain_blocks();
    let (_mono_trace, mono_final_state) =
        synthesize_sha256_chain_trace::<RealEcdsaInt, REAL_SHA256_CHAIN_BLOCKS>(
            REAL_SHA256_CHAIN_NUM_VARS,
            SHA256_INITIAL_STATE,
            message_blocks,
        )
        .expect("monolithic N=8 SHA trace synthesis should succeed");
    let (witnesses, projection_final_state) = synthesize_sha256_chain_witnesses::<
        RealEcdsaInt,
        REAL_SHA256_CHAIN_BLOCKS,
    >(SHA256_INITIAL_STATE, message_blocks)
    .expect("ProjectionFold SHA witness synthesis should succeed");
    assert_eq!(mono_final_state, projection_final_state);

    let shape = UairShape::<U>::new(SHA_ROW_VARS);
    let field_cfg = F::curve_field_cfg::<C>();
    let (pcs_params, pcs_verifier_params) = projection_sha_hyrax_pcs_params::<C, F>(SHA_ROW_COUNT);
    let pp =
        LinearIdealFoldProverParams::<P<C>, U, RealEcdsaBenchZincTypes, F, DEGREE_PLUS_ONE>::new(
            pcs_params,
            field_cfg.clone(),
            3,
        );
    let vs = setup_verify_linear_ideal_fold_mixed_hyrax::<
        C,
        U,
        RealEcdsaBenchZincTypes,
        F,
        DEGREE_PLUS_ONE,
    >(
        LinearIdealFoldVerifierParams::new(pcs_verifier_params, field_cfg),
        shape.clone(),
    )
    .expect("ProjectionFold SHA verifier setup succeeds");

    let params = format!("{label}/SHA256Chain8/row-vars={SHA_ROW_VARS}");
    let prepared_instances = prepare_linear_ideal_fold_witnesses::<
        U,
        RealEcdsaBenchZincTypes,
        F,
        DEGREE_PLUS_ONE,
    >(&shape, &witnesses, &pp.field_cfg)
    .expect("ProjectionFold SHA witness preparation should succeed");

    group.bench_function(BenchmarkId::new("Prove", &params), |bench| {
        bench.iter(|| {
            let mut transcript = Blake3Transcript::new();
            black_box(prove_prepared_linear_ideal_fold_mixed_hyrax::<
                C,
                U,
                RealEcdsaBenchZincTypes,
                F,
                DEGREE_PLUS_ONE,
            >(
                &pp,
                &shape,
                &prepared_instances,
                ShaBooleanityCatalog::Full,
                &mut transcript,
            ))
            .expect("ProjectionFold Concise prover failed");
        });
    });

    let mut prover_transcript = Blake3Transcript::new();
    let output = prove_prepared_linear_ideal_fold_mixed_hyrax::<
        C,
        U,
        RealEcdsaBenchZincTypes,
        F,
        DEGREE_PLUS_ONE,
    >(
        &pp,
        &shape,
        &prepared_instances,
        ShaBooleanityCatalog::Full,
        &mut prover_transcript,
    )
    .expect("proof generation for ProjectionFold verifier bench");

    let mut verifier_transcript = Blake3Transcript::new();
    let verified =
        verify_linear_ideal_fold_mixed_hyrax::<C, U, RealEcdsaBenchZincTypes, F, DEGREE_PLUS_ONE>(
            &vs,
            &output.fresh_instances,
            &output.proof,
            ShaBooleanityCatalog::Full,
            &mut verifier_transcript,
        )
        .expect("ProjectionFold verifier preflight failed");
    assert_eq!(verified.target, output.folded_instance.target);
    assert_eq!(verified.public, output.folded_instance.public);

    eprintln!("    ProjectionFold Concise tracing ({params}):");
    let subscriber = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_target(true)
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE)
        .finish();
    tracing::subscriber::with_default(subscriber, || {
        let mut prover_transcript = Blake3Transcript::new();
        let traced_output = prove_prepared_linear_ideal_fold_mixed_hyrax::<
            C,
            U,
            RealEcdsaBenchZincTypes,
            F,
            DEGREE_PLUS_ONE,
        >(
            &pp,
            &shape,
            &prepared_instances,
            ShaBooleanityCatalog::Full,
            &mut prover_transcript,
        )
        .expect("ProjectionFold traced prover failed");

        let mut verifier_transcript = Blake3Transcript::new();
        let traced_verified = verify_linear_ideal_fold_mixed_hyrax::<
            C,
            U,
            RealEcdsaBenchZincTypes,
            F,
            DEGREE_PLUS_ONE,
        >(
            &vs,
            &traced_output.fresh_instances,
            &traced_output.proof,
            ShaBooleanityCatalog::Full,
            &mut verifier_transcript,
        )
        .expect("ProjectionFold traced verifier failed");
        assert_eq!(traced_verified.target, traced_output.folded_instance.target);
        assert_eq!(traced_verified.public, traced_output.folded_instance.public);
    });

    group.bench_function(BenchmarkId::new("Verify", &params), |bench| {
        bench.iter(|| {
            let mut transcript = Blake3Transcript::new();
            black_box(verify_linear_ideal_fold_mixed_hyrax::<
                C,
                U,
                RealEcdsaBenchZincTypes,
                F,
                DEGREE_PLUS_ONE,
            >(
                &vs,
                &output.fresh_instances,
                &output.proof,
                ShaBooleanityCatalog::Full,
                &mut transcript,
            ))
            .expect("ProjectionFold Concise verifier failed");
        });
    });
}

fn bench_projectionfold_sha256_concise_hyrax_bn254(group: &mut BenchmarkGroup<WallTime>) {
    bench_projectionfold_sha256_concise_hyrax::<ark_bn254::G1Affine, ArkFBn254>(
        group,
        "ProjectionFoldConcise-HyraxBn254",
    );
}

/// Same pipeline on the dynamic Montgomery field, kept as a comparison point
/// for the curve-native arkworks instantiation above.
fn bench_projectionfold_sha256_concise_hyrax_bn254_monty(group: &mut BenchmarkGroup<WallTime>) {
    bench_projectionfold_sha256_concise_hyrax::<ark_bn254::G1Affine, MontyField<FIELD_LIMBS>>(
        group,
        "ProjectionFoldConcise-HyraxBn254-MontyField",
    );
}

fn bench_projectionfold_sha256_concise_hyrax_secp256k1(group: &mut BenchmarkGroup<WallTime>) {
    bench_projectionfold_sha256_concise_hyrax::<ark_secp256k1::Affine, ArkFSecp256k1>(
        group,
        "ProjectionFoldConcise-HyraxSecp256k1",
    );
}

fn bench_real_sha_ecdsa_e2e(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    type U = ShaEcdsaUair<RealEcdsaInt>;

    let mut rng = rng();
    let trace = U::generate_random_trace(num_vars, &mut rng);
    let pp = setup_pp_real_ecdsa(num_vars);

    do_bench_e2e::<RealEcdsaBenchZincTypes, U, _>(
        group,
        "ShaEcdsa",
        num_vars,
        &pp,
        &trace,
        zinc_protocol::project_scalar_fn,
        sha256_real_project_ideal,
    );
}

fn bench_real_sha_ecdsa_steps(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    type U = ShaEcdsaUair<RealEcdsaInt>;

    let mut rng = rng();
    let trace = U::generate_random_trace(num_vars, &mut rng);
    let pp = setup_pp_real_ecdsa(num_vars);

    do_bench_steps::<RealEcdsaBenchZincTypes, U, _>(
        group,
        "ShaEcdsa",
        num_vars,
        &pp,
        &trace,
        zinc_protocol::project_scalar_fn,
        sha256_real_project_ideal,
    );
}

fn bench_no_mult_e2e(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    do_bench_uair::<TestUairNoMultiplication<i64>>(group, "NoMult", num_vars);
}
fn bench_binary_decomposition_e2e(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    do_bench_uair::<BinaryDecompositionUair<i64>>(group, "BinaryDecomposition", num_vars);
}
fn bench_big_linear_e2e(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    do_bench_uair::<BigLinearUair<i64>>(group, "BigLinear", num_vars);
}
fn bench_sha_proxy_e2e(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    do_bench_uair::<ShaProxy<i64>>(group, "ShaProxy", num_vars);
}
fn bench_big_linear_public_input_e2e(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    do_bench_uair::<BigLinearUairWithPublicInput<i64>>(group, "BigLinearPI", num_vars);
}

fn bench_no_mult_steps(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    do_bench_steps_uair::<TestUairNoMultiplication<i64>>(group, "NoMult", num_vars);
}
fn bench_binary_decomposition_steps(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    do_bench_steps_uair::<BinaryDecompositionUair<i64>>(group, "BinaryDecomposition", num_vars);
}
fn bench_big_linear_steps(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    do_bench_steps_uair::<BigLinearUair<i64>>(group, "BigLinear", num_vars);
}
fn bench_sha_proxy_steps(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    do_bench_steps_uair::<ShaProxy<i64>>(group, "ShaProxy", num_vars);
}
fn bench_big_linear_public_input_steps(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    do_bench_steps_uair::<BigLinearUairWithPublicInput<i64>>(group, "BigLinearPI", num_vars);
}

//
// Criterion entry points
//

fn e2e_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zinc+ E2E");

    // bench_no_mult_e2e(&mut group, 8);
    // bench_no_mult_e2e(&mut group, 10);
    // bench_no_mult_e2e(&mut group, 12);
    //
    // bench_binary_decomposition_e2e(&mut group, 8);
    // bench_binary_decomposition_e2e(&mut group, 10);
    // bench_binary_decomposition_e2e(&mut group, 12);
    //
    // bench_big_linear_e2e(&mut group, 8);
    // bench_big_linear_e2e(&mut group, 10);
    // bench_big_linear_e2e(&mut group, 12);
    //
    // bench_big_linear_public_input_e2e(&mut group, 8);
    // bench_big_linear_public_input_e2e(&mut group, 10);
    // bench_big_linear_public_input_e2e(&mut group, 12);
    //
    // bench_sha_proxy_e2e(&mut group, 8);
    // bench_sha_proxy_e2e(&mut group, 10);
    // bench_sha_proxy_e2e(&mut group, 12);

    // Real UAIRs ported from main-gamma. Trace size for ECDSA needs >= 256
    // rows (Shamir loop), so num_vars=9 is the smallest meaningful size.
    // bench_real_ecdsa_e2e(&mut group, 9);
    bench_real_sha256_e2e(&mut group, REAL_SHA256_CHAIN_NUM_VARS);
    bench_real_sha256_pcs_e2e(&mut group, REAL_SHA256_CHAIN_NUM_VARS);
    bench_real_sha_ecdsa_e2e(&mut group, 9);

    group.finish();
}

fn e2e_steps_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zinc+ E2E Steps");

    // bench_no_mult_steps(&mut group, 8);
    // bench_no_mult_steps(&mut group, 10);
    // bench_no_mult_steps(&mut group, 12);
    //
    // bench_binary_decomposition_steps(&mut group, 8);
    // bench_binary_decomposition_steps(&mut group, 10);
    // bench_binary_decomposition_steps(&mut group, 12);
    //
    // bench_big_linear_steps(&mut group, 8);
    // bench_big_linear_steps(&mut group, 10);
    // bench_big_linear_steps(&mut group, 12);
    //
    // bench_big_linear_public_input_steps(&mut group, 8);
    // bench_big_linear_public_input_steps(&mut group, 10);
    // bench_big_linear_public_input_steps(&mut group, 12);
    //
    // bench_sha_proxy_steps(&mut group, 8);
    // bench_sha_proxy_steps(&mut group, 10);
    // bench_sha_proxy_steps(&mut group, 12);

    // Real UAIRs ported from main-gamma. See `e2e_benches` for the
    // num_vars=9 lower-bound rationale.
    bench_real_ecdsa_steps(&mut group, 9);
    bench_real_sha256_steps(&mut group, REAL_SHA256_CHAIN_NUM_VARS);
    bench_real_sha256_pcs_steps(&mut group, REAL_SHA256_CHAIN_NUM_VARS);
    bench_real_sha_ecdsa_steps(&mut group, 9);

    group.finish();
}

fn sha256_proving_system_compare_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("SHA-256 Proving System Comparison");

    bench_og_sha256_zip_compare(&mut group, REAL_SHA256_CHAIN_NUM_VARS);
    bench_projectionfold_sha256_concise_hyrax_bn254(&mut group);
    bench_projectionfold_sha256_concise_hyrax_bn254_monty(&mut group);
    bench_projectionfold_sha256_concise_hyrax_secp256k1(&mut group);

    group.finish();
}

//
// Folded Zip+ (1× fold) — total prove/verify benchmark.
//
// Mirrors the unfolded e2e bench but commits binary witness columns as
// BinaryPoly<HALF_DEGREE_PLUS_ONE> halves of length 2n and verifies
// the binary PCS opening at the extended point (r_0 ‖ γ).
//

const HALF_DEGREE_PLUS_ONE: usize = DEGREE_PLUS_ONE / 2;

type FoldedPp1x<ZtF> = (
    ZipPlusParams<
        <ZtF as FoldedZincTypes<DEGREE_PLUS_ONE, HALF_DEGREE_PLUS_ONE>>::BinaryZt,
        <ZtF as FoldedZincTypes<DEGREE_PLUS_ONE, HALF_DEGREE_PLUS_ONE>>::BinaryLc,
    >,
    ZipPlusParams<
        <ZtF as FoldedZincTypes<DEGREE_PLUS_ONE, HALF_DEGREE_PLUS_ONE>>::ArbitraryZt,
        <ZtF as FoldedZincTypes<DEGREE_PLUS_ONE, HALF_DEGREE_PLUS_ONE>>::ArbitraryLc,
    >,
    ZipPlusParams<
        <ZtF as FoldedZincTypes<DEGREE_PLUS_ONE, HALF_DEGREE_PLUS_ONE>>::IntZt,
        <ZtF as FoldedZincTypes<DEGREE_PLUS_ONE, HALF_DEGREE_PLUS_ONE>>::IntLc,
    >,
);

#[allow(clippy::unwrap_used)]
fn setup_folded_pp_real_ecdsa(num_vars: usize) -> FoldedPp1x<BenchFoldedRealEcdsaZincTypes> {
    let split_size = 1 << (num_vars + 1);
    let normal_size = 1 << num_vars;
    (
        ZipPlus::setup(
            split_size,
            IprsCode::new_with_optimal_depth(split_size).unwrap(),
        ),
        ZipPlus::setup(
            normal_size,
            IprsCode::new_with_optimal_depth(normal_size).unwrap(),
        ),
        ZipPlus::setup(
            normal_size,
            IprsCode::new_with_optimal_depth(normal_size).unwrap(),
        ),
    )
}

#[allow(clippy::too_many_arguments)]
fn do_bench_e2e_folded<ZtF, U, IdealOverF>(
    group: &mut BenchmarkGroup<WallTime>,
    label: &str,
    num_vars: usize,
    pp: &FoldedPp1x<ZtF>,
    trace: &UairTrace<'static, ZtF::Int, ZtF::Int, DEGREE_PLUS_ONE>,
    project_scalar: impl Fn(&U::Scalar, &<F as PrimeField>::Config) -> DynamicPolynomialF<F>
    + Copy
    + Sync,
    project_ideal: impl Fn(&IdealOrZero<U::Ideal>, &<F as PrimeField>::Config) -> IdealOverF + Copy,
) where
    ZtF: FoldedZincTypes<DEGREE_PLUS_ONE, HALF_DEGREE_PLUS_ONE>,
    ZtF::Int: ProjectableToField<F> + num_traits::Zero,
    <ZtF::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<F>,
    <ZtF::BinaryZt as ZipTypes>::Cw: ProjectableToField<F>,
    <ZtF::ArbitraryZt as ZipTypes>::Cw: ProjectableToField<F>,
    <ZtF::IntZt as ZipTypes>::Cw: ProjectableToField<F>,
    F: for<'a> FromWithConfig<&'a ZtF::Int>
        + for<'a> FromWithConfig<&'a <ZtF::BinaryZt as ZipTypes>::CombR>
        + for<'a> FromWithConfig<&'a <ZtF::ArbitraryZt as ZipTypes>::CombR>
        + for<'a> FromWithConfig<&'a <ZtF::IntZt as ZipTypes>::CombR>
        + for<'a> FromWithConfig<&'a ZtF::Chal>
        + for<'a> FromWithConfig<&'a ZtF::Pt>,
    <F as Field>::Modulus: ConstTranscribable + FromRef<ZtF::Fmod>,
    U: Uair + 'static,
    IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<F>>,
{
    let params = format!("{label}/nvars={num_vars}");

    macro_rules! bench_prove_folded {
        ($label:literal, $mle_first:expr) => {
            group.bench_function(BenchmarkId::new($label, &params), |bench| {
                bench.iter(|| {
                    black_box(zinc_protocol::prover::prove_folded::<
                        ZtF,
                        U,
                        F,
                        DEGREE_PLUS_ONE,
                        HALF_DEGREE_PLUS_ONE,
                        { $mle_first },
                        PERFORM_CHECKS,
                    >(pp, trace, num_vars, project_scalar))
                    .expect("Folded prover failed");
                });
            });
        };
    }

    bench_prove_folded!("Prove (folded)", false);

    if count_effective_max_degree::<U>() <= 1 {
        bench_prove_folded!("Prove (folded MLE-first)", true);
    }

    let proof: Proof<F> = zinc_protocol::prover::prove_folded::<
        ZtF,
        U,
        F,
        DEGREE_PLUS_ONE,
        HALF_DEGREE_PLUS_ONE,
        false,
        PERFORM_CHECKS,
    >(pp, trace, num_vars, project_scalar)
    .expect("proof generation for folded verifier bench");

    let sig = U::signature();
    let public_trace = trace.public(&sig);

    group.bench_function(BenchmarkId::new("Verify (folded)", &params), |bench| {
        bench.iter_batched(
            || proof.clone(),
            |proof| {
                black_box(zinc_protocol::verifier::verify_folded::<
                    ZtF,
                    U,
                    F,
                    IdealOverF,
                    DEGREE_PLUS_ONE,
                    HALF_DEGREE_PLUS_ONE,
                    PERFORM_CHECKS,
                >(
                    pp,
                    proof,
                    &public_trace,
                    num_vars,
                    project_scalar,
                    project_ideal,
                ))
                .expect("Folded verifier failed");
            },
            BatchSize::SmallInput,
        );
    });

    eprint_proof_size(&params, &proof);
}

//
// Folded Zip+ (4× fold) — total prove/verify benchmark.
//
// Mirrors the 1× fold bench but commits binary witness columns as
// twice-split BinaryPoly<QUARTER_DEGREE_PLUS_ONE> entries of length 4n
// and verifies the binary PCS opening at the doubly-extended point
// (r_0 ‖ γ₁ ‖ γ₂).
//

const QUARTER_DEGREE_PLUS_ONE: usize = DEGREE_PLUS_ONE / 4;

type FoldedPp4x<ZtF> = (
    ZipPlusParams<
        <ZtF as FoldedZincTypes<DEGREE_PLUS_ONE, QUARTER_DEGREE_PLUS_ONE>>::BinaryZt,
        <ZtF as FoldedZincTypes<DEGREE_PLUS_ONE, QUARTER_DEGREE_PLUS_ONE>>::BinaryLc,
    >,
    ZipPlusParams<
        <ZtF as FoldedZincTypes<DEGREE_PLUS_ONE, QUARTER_DEGREE_PLUS_ONE>>::ArbitraryZt,
        <ZtF as FoldedZincTypes<DEGREE_PLUS_ONE, QUARTER_DEGREE_PLUS_ONE>>::ArbitraryLc,
    >,
    ZipPlusParams<
        <ZtF as FoldedZincTypes<DEGREE_PLUS_ONE, QUARTER_DEGREE_PLUS_ONE>>::IntZt,
        <ZtF as FoldedZincTypes<DEGREE_PLUS_ONE, QUARTER_DEGREE_PLUS_ONE>>::IntLc,
    >,
);

/// 4× folded e2e bench: routes binary AND int through `MultiZip3` for
/// shared-Merkle collapse, then opens at the doubly-extended point
/// `(r_0 ‖ γ₁ ‖ γ₂)`. Calls [`prove_folded_4x`] / [`verify_folded_4x`].
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn do_bench_e2e_folded_4x<ZtF, U, IdealOverF>(
    group: &mut BenchmarkGroup<WallTime>,
    label: &str,
    num_vars: usize,
    pp: &(
        ZipPlusParams<ZtF::BinaryZt, ZtF::BinaryLc>,
        ZipPlusParams<ZtF::ArbitraryZt, ZtF::ArbitraryLc>,
        ZipPlusParams<ZtF::IntZt, ZtF::IntLc>,
    ),
    trace: &UairTrace<'static, Int<EC_FP_INT_LIMBS>, Int<EC_FP_INT_LIMBS>, DEGREE_PLUS_ONE>,
    project_scalar: impl Fn(&U::Scalar, &<F as PrimeField>::Config) -> DynamicPolynomialF<F>
    + Copy
    + Sync,
    project_ideal: impl Fn(&IdealOrZero<U::Ideal>, &<F as PrimeField>::Config) -> IdealOverF + Copy,
) where
    ZtF: IntFoldedZincTypes4x<
            DEGREE_PLUS_ONE,
            QUARTER_DEGREE_PLUS_ONE,
            EC_FP_INT_LIMBS,
            INT_QUARTER_LIMBS_BENCH,
        >,
    Int<EC_FP_INT_LIMBS>: ProjectableToField<F>,
    Int<INT_QUARTER_LIMBS_BENCH>: ProjectableToField<F>,
    <ZtF::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<F>,
    <ZtF::BinaryZt as ZipTypes>::Cw: ProjectableToField<F>,
    <ZtF::ArbitraryZt as ZipTypes>::Cw: ProjectableToField<F>,
    <ZtF::IntZt as ZipTypes>::Cw: ProjectableToField<F>,
    F: for<'a> FromWithConfig<&'a Int<EC_FP_INT_LIMBS>>
        + for<'a> FromWithConfig<&'a Int<INT_QUARTER_LIMBS_BENCH>>
        + for<'a> FromWithConfig<&'a <ZtF::BinaryZt as ZipTypes>::CombR>
        + for<'a> FromWithConfig<&'a <ZtF::ArbitraryZt as ZipTypes>::CombR>
        + for<'a> FromWithConfig<&'a <ZtF::IntZt as ZipTypes>::CombR>
        + for<'a> FromWithConfig<&'a ZtF::Chal>
        + for<'a> FromWithConfig<&'a ZtF::Pt>,
    <F as Field>::Modulus: ConstTranscribable + FromRef<ZtF::Fmod>,
    U: Uair<
            Scalar = zinc_poly::univariate::dense::DensePolynomial<
                Int<EC_FP_INT_LIMBS>,
                DEGREE_PLUS_ONE,
            >,
        > + 'static,
    IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<F>>,
{
    let params = format!("{label}/nvars={num_vars}");

    macro_rules! bench_prove_folded_4x {
        ($label:literal, $mle_first:expr) => {
            group.bench_function(BenchmarkId::new($label, &params), |bench| {
                bench.iter(|| {
                    black_box(zinc_protocol::prover::prove_folded_4x::<
                        ZtF,
                        U,
                        F,
                        DEGREE_PLUS_ONE,
                        HALF_DEGREE_PLUS_ONE,
                        QUARTER_DEGREE_PLUS_ONE,
                        EC_FP_INT_LIMBS,
                        INT_QUARTER_LIMBS_BENCH,
                        { $mle_first },
                        PERFORM_CHECKS,
                    >(pp, trace, num_vars, project_scalar))
                    .expect("Folded 4× prover failed");
                });
            });
        };
    }

    if count_effective_max_degree::<U>() <= 1 {
        bench_prove_folded_4x!("Prove (folded 4× MLE-first)", true);
    }

    let proof: Proof<F> = zinc_protocol::prover::prove_folded_4x::<
        ZtF,
        U,
        F,
        DEGREE_PLUS_ONE,
        HALF_DEGREE_PLUS_ONE,
        QUARTER_DEGREE_PLUS_ONE,
        EC_FP_INT_LIMBS,
        INT_QUARTER_LIMBS_BENCH,
        false,
        PERFORM_CHECKS,
    >(pp, trace, num_vars, project_scalar)
    .expect("proof generation for folded 4× verifier bench");

    let sig = U::signature();
    let public_trace = trace.public(&sig);

    eprintln!("    Folded 4× tracing ({params}):");
    let subscriber = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_target(true)
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE)
        .finish();
    tracing::subscriber::with_default(subscriber, || {
        let (_traced_proof, _traced_timings) =
            zinc_protocol::prover::prove_folded_4x_with_timings::<
                ZtF,
                U,
                F,
                DEGREE_PLUS_ONE,
                HALF_DEGREE_PLUS_ONE,
                QUARTER_DEGREE_PLUS_ONE,
                EC_FP_INT_LIMBS,
                INT_QUARTER_LIMBS_BENCH,
                false,
                PERFORM_CHECKS,
            >(pp, trace, num_vars, project_scalar)
            .expect("Folded 4× traced prover failed");
    });

    group.bench_function(BenchmarkId::new("Verify (folded 4×)", &params), |bench| {
        bench.iter_batched(
            || proof.clone(),
            |proof| {
                black_box(zinc_protocol::verifier::verify_folded_4x::<
                    ZtF,
                    U,
                    F,
                    IdealOverF,
                    DEGREE_PLUS_ONE,
                    HALF_DEGREE_PLUS_ONE,
                    QUARTER_DEGREE_PLUS_ONE,
                    EC_FP_INT_LIMBS,
                    INT_QUARTER_LIMBS_BENCH,
                    PERFORM_CHECKS,
                >(
                    pp,
                    proof,
                    &public_trace,
                    num_vars,
                    project_scalar,
                    project_ideal,
                ))
                .expect("Folded 4× verifier failed");
            },
            BatchSize::SmallInput,
        );
    });

    let label_full = format!("Folded 4×/{params}");
    eprint_proof_size(&label_full, &proof);

    // Re-run the prover once more to harvest the per-domain Zip+ byte
    // breakdown. We discard the proof and only keep the breakdown.
    let (_proof_for_bd, zip_breakdown) =
        zinc_protocol::prover::prove_folded_4x_with_zip_breakdown::<
            ZtF,
            U,
            F,
            DEGREE_PLUS_ONE,
            HALF_DEGREE_PLUS_ONE,
            QUARTER_DEGREE_PLUS_ONE,
            EC_FP_INT_LIMBS,
            INT_QUARTER_LIMBS_BENCH,
            false,
            PERFORM_CHECKS,
        >(pp, trace, num_vars, project_scalar)
        .expect("zip-breakdown prove failed");

    eprint_folded_4x_proof_size_breakdown(&label_full, &proof);
    eprint_folded_4x_zip_substep_breakdown(&label_full, &proof, &zip_breakdown);

    if count_effective_max_degree::<U>() <= 1 {
        eprint_folded_4x_per_region_prove_timings::<ZtF, U, _, true>(
            &label_full,
            "MLE-first",
            pp,
            trace,
            num_vars,
            project_scalar,
        );
    }
    eprint_folded_4x_per_region_verify_timings::<ZtF, U, IdealOverF, _, _>(
        &label_full,
        pp,
        &proof,
        &public_trace,
        num_vars,
        project_scalar,
        project_ideal,
    );
}

/// Per-region prove timings for the int-fold-4× variant. Mirrors
/// [`eprint_folded_4x_per_region_timings`] but routed through
/// [`prove_folded_4x_with_timings`].
#[allow(clippy::too_many_arguments)]
fn eprint_folded_4x_per_region_prove_timings<ZtF, U, S, const MLE_FIRST: bool>(
    params: &str,
    lane: &str,
    pp: &(
        ZipPlusParams<ZtF::BinaryZt, ZtF::BinaryLc>,
        ZipPlusParams<ZtF::ArbitraryZt, ZtF::ArbitraryLc>,
        ZipPlusParams<ZtF::IntZt, ZtF::IntLc>,
    ),
    trace: &UairTrace<'static, Int<EC_FP_INT_LIMBS>, Int<EC_FP_INT_LIMBS>, DEGREE_PLUS_ONE>,
    num_vars: usize,
    project_scalar: S,
) where
    ZtF: IntFoldedZincTypes4x<
            DEGREE_PLUS_ONE,
            QUARTER_DEGREE_PLUS_ONE,
            EC_FP_INT_LIMBS,
            INT_QUARTER_LIMBS_BENCH,
        >,
    Int<EC_FP_INT_LIMBS>: ProjectableToField<F>,
    Int<INT_QUARTER_LIMBS_BENCH>: ProjectableToField<F>,
    <ZtF::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<F>,
    <ZtF::BinaryZt as ZipTypes>::Cw: ProjectableToField<F>,
    <ZtF::ArbitraryZt as ZipTypes>::Cw: ProjectableToField<F>,
    <ZtF::IntZt as ZipTypes>::Cw: ProjectableToField<F>,
    F: for<'a> FromWithConfig<&'a Int<EC_FP_INT_LIMBS>>
        + for<'a> FromWithConfig<&'a Int<INT_QUARTER_LIMBS_BENCH>>
        + for<'a> FromWithConfig<&'a <ZtF::BinaryZt as ZipTypes>::CombR>
        + for<'a> FromWithConfig<&'a <ZtF::ArbitraryZt as ZipTypes>::CombR>
        + for<'a> FromWithConfig<&'a <ZtF::IntZt as ZipTypes>::CombR>
        + for<'a> FromWithConfig<&'a ZtF::Chal>
        + for<'a> FromWithConfig<&'a ZtF::Pt>,
    <F as Field>::Modulus: ConstTranscribable + FromRef<ZtF::Fmod>,
    U: Uair<
            Scalar = zinc_poly::univariate::dense::DensePolynomial<
                Int<EC_FP_INT_LIMBS>,
                DEGREE_PLUS_ONE,
            >,
        > + 'static,
    S: Fn(&U::Scalar, &<F as PrimeField>::Config) -> DynamicPolynomialF<F> + Copy + Sync,
{
    use zinc_protocol::prover::{FoldedProveTimings, prove_folded_4x_with_timings};

    const N: u32 = 100;

    let _ = prove_folded_4x_with_timings::<
        ZtF,
        U,
        F,
        DEGREE_PLUS_ONE,
        HALF_DEGREE_PLUS_ONE,
        QUARTER_DEGREE_PLUS_ONE,
        EC_FP_INT_LIMBS,
        INT_QUARTER_LIMBS_BENCH,
        MLE_FIRST,
        PERFORM_CHECKS,
    >(pp, trace, num_vars, project_scalar)
    .expect("warmup folded-4× prove failed");

    let mut sum = FoldedProveTimings::default();
    for _ in 0..N {
        let (_proof, t) = prove_folded_4x_with_timings::<
            ZtF,
            U,
            F,
            DEGREE_PLUS_ONE,
            HALF_DEGREE_PLUS_ONE,
            QUARTER_DEGREE_PLUS_ONE,
            EC_FP_INT_LIMBS,
            INT_QUARTER_LIMBS_BENCH,
            MLE_FIRST,
            PERFORM_CHECKS,
        >(pp, trace, num_vars, project_scalar)
        .expect("timed folded-4× prove failed");
        sum.add_assign(&t);
    }
    sum.divide_by(N);

    let total = sum.total();
    let pct = |d: std::time::Duration| (d.as_secs_f64() / total.as_secs_f64()) * 100.0;
    eprintln!(
        "    Folded 4× per-region prove timings, {} lane ({}, mean of N={} runs):",
        lane, params, N
    );
    eprintln!(
        "      step 0  commit            {:>9.3} ms ({:>4.1}%)",
        sum.step0_commit.as_secs_f64() * 1e3,
        pct(sum.step0_commit)
    );
    eprintln!(
        "      step 1  prime projection  {:>9.3} ms ({:>4.1}%)",
        sum.step1_prime_projection.as_secs_f64() * 1e3,
        pct(sum.step1_prime_projection)
    );
    eprintln!(
        "      step 2  ideal check       {:>9.3} ms ({:>4.1}%)",
        sum.step2_ideal_check.as_secs_f64() * 1e3,
        pct(sum.step2_ideal_check)
    );
    eprintln!(
        "      step 3  eval projection   {:>9.3} ms ({:>4.1}%)",
        sum.step3_eval_projection.as_secs_f64() * 1e3,
        pct(sum.step3_eval_projection)
    );
    eprintln!(
        "      step 4  sumcheck          {:>9.3} ms ({:>4.1}%)",
        sum.step4_sumcheck.as_secs_f64() * 1e3,
        pct(sum.step4_sumcheck)
    );
    eprintln!(
        "      step 5  multipoint eval   {:>9.3} ms ({:>4.1}%)",
        sum.step5_multipoint_eval.as_secs_f64() * 1e3,
        pct(sum.step5_multipoint_eval)
    );
    eprintln!(
        "      step 6  lift-and-project  {:>9.3} ms ({:>4.1}%)",
        sum.step6_lift_and_project.as_secs_f64() * 1e3,
        pct(sum.step6_lift_and_project)
    );
    eprintln!(
        "      step 7  pcs open          {:>9.3} ms ({:>4.1}%)",
        sum.step7_pcs_open.as_secs_f64() * 1e3,
        pct(sum.step7_pcs_open)
    );
    eprintln!(
        "      step 8  compress (zstd-{}){:>9.3} ms ({:>4.1}%)",
        zip_plus::utils::ZSTD_LEVEL,
        sum.step8_compress.as_secs_f64() * 1e3,
        pct(sum.step8_compress)
    );
    eprintln!(
        "      assembly                  {:>9.3} ms ({:>4.1}%)",
        sum.assembly.as_secs_f64() * 1e3,
        pct(sum.assembly)
    );
    eprintln!(
        "      total                     {:>9.3} ms",
        total.as_secs_f64() * 1e3
    );
}

/// Per-region verify timings for the int-fold-4× variant.
#[allow(clippy::too_many_arguments)]
fn eprint_folded_4x_per_region_verify_timings<ZtF, U, IdealOverF, S, I>(
    params: &str,
    pp: &(
        ZipPlusParams<ZtF::BinaryZt, ZtF::BinaryLc>,
        ZipPlusParams<ZtF::ArbitraryZt, ZtF::ArbitraryLc>,
        ZipPlusParams<ZtF::IntZt, ZtF::IntLc>,
    ),
    proof: &Proof<F>,
    public_trace: &UairTrace<'_, Int<EC_FP_INT_LIMBS>, Int<EC_FP_INT_LIMBS>, DEGREE_PLUS_ONE>,
    num_vars: usize,
    project_scalar: S,
    project_ideal: I,
) where
    ZtF: IntFoldedZincTypes4x<
            DEGREE_PLUS_ONE,
            QUARTER_DEGREE_PLUS_ONE,
            EC_FP_INT_LIMBS,
            INT_QUARTER_LIMBS_BENCH,
        >,
    Int<EC_FP_INT_LIMBS>: ProjectableToField<F>,
    Int<INT_QUARTER_LIMBS_BENCH>: ProjectableToField<F>,
    <ZtF::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<F>,
    <ZtF::BinaryZt as ZipTypes>::Cw: ProjectableToField<F>,
    <ZtF::ArbitraryZt as ZipTypes>::Cw: ProjectableToField<F>,
    <ZtF::IntZt as ZipTypes>::Cw: ProjectableToField<F>,
    F: for<'a> FromWithConfig<&'a Int<EC_FP_INT_LIMBS>>
        + for<'a> FromWithConfig<&'a Int<INT_QUARTER_LIMBS_BENCH>>
        + for<'a> FromWithConfig<&'a <ZtF::BinaryZt as ZipTypes>::CombR>
        + for<'a> FromWithConfig<&'a <ZtF::ArbitraryZt as ZipTypes>::CombR>
        + for<'a> FromWithConfig<&'a <ZtF::IntZt as ZipTypes>::CombR>
        + for<'a> FromWithConfig<&'a ZtF::Chal>,
    <F as Field>::Modulus: ConstTranscribable + FromRef<ZtF::Fmod>,
    U: Uair<
            Scalar = zinc_poly::univariate::dense::DensePolynomial<
                Int<EC_FP_INT_LIMBS>,
                DEGREE_PLUS_ONE,
            >,
        > + 'static,
    IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<F>>,
    S: Fn(&U::Scalar, &<F as PrimeField>::Config) -> DynamicPolynomialF<F> + Copy + Sync,
    I: Fn(&IdealOrZero<U::Ideal>, &<F as PrimeField>::Config) -> IdealOverF + Copy,
{
    use zinc_protocol::verifier::{FoldedVerifyTimings, verify_folded_4x_with_timings};

    const N: u32 = 100;

    let _ = verify_folded_4x_with_timings::<
        ZtF,
        U,
        F,
        IdealOverF,
        DEGREE_PLUS_ONE,
        HALF_DEGREE_PLUS_ONE,
        QUARTER_DEGREE_PLUS_ONE,
        EC_FP_INT_LIMBS,
        INT_QUARTER_LIMBS_BENCH,
        PERFORM_CHECKS,
    >(
        pp,
        proof.clone(),
        public_trace,
        num_vars,
        project_scalar,
        project_ideal,
    )
    .expect("warmup folded-4× verify failed");

    let mut sum = FoldedVerifyTimings::default();
    for _ in 0..N {
        let t = verify_folded_4x_with_timings::<
            ZtF,
            U,
            F,
            IdealOverF,
            DEGREE_PLUS_ONE,
            HALF_DEGREE_PLUS_ONE,
            QUARTER_DEGREE_PLUS_ONE,
            EC_FP_INT_LIMBS,
            INT_QUARTER_LIMBS_BENCH,
            PERFORM_CHECKS,
        >(
            pp,
            proof.clone(),
            public_trace,
            num_vars,
            project_scalar,
            project_ideal,
        )
        .expect("timed folded-4× verify failed");
        sum.add_assign(&t);
    }
    sum.divide_by(N);

    let total = sum.total();
    let pct = |d: std::time::Duration| (d.as_secs_f64() / total.as_secs_f64()) * 100.0;
    eprintln!(
        "    Folded 4× per-region verify timings ({}, mean of N={} runs):",
        params, N
    );
    eprintln!(
        "      step 0  reconstruct trans {:>9.3} ms ({:>4.1}%)",
        sum.step0_reconstruct_transcript.as_secs_f64() * 1e3,
        pct(sum.step0_reconstruct_transcript)
    );
    eprintln!(
        "      step 1  prime projection  {:>9.3} ms ({:>4.1}%)",
        sum.step1_prime_projection.as_secs_f64() * 1e3,
        pct(sum.step1_prime_projection)
    );
    eprintln!(
        "      step 2  ideal check       {:>9.3} ms ({:>4.1}%)",
        sum.step2_ideal_check.as_secs_f64() * 1e3,
        pct(sum.step2_ideal_check)
    );
    eprintln!(
        "      step 3  eval projection   {:>9.3} ms ({:>4.1}%)",
        sum.step3_eval_projection.as_secs_f64() * 1e3,
        pct(sum.step3_eval_projection)
    );
    eprintln!(
        "      step 4  sumcheck verify   {:>9.3} ms ({:>4.1}%)",
        sum.step4_sumcheck_verify.as_secs_f64() * 1e3,
        pct(sum.step4_sumcheck_verify)
    );
    eprintln!(
        "      step 5  multipoint eval   {:>9.3} ms ({:>4.1}%)",
        sum.step5_multipoint_eval.as_secs_f64() * 1e3,
        pct(sum.step5_multipoint_eval)
    );
    eprintln!(
        "      step 6  lifted evals      {:>9.3} ms ({:>4.1}%)",
        sum.step6_lifted_evals.as_secs_f64() * 1e3,
        pct(sum.step6_lifted_evals)
    );
    eprintln!(
        "      step 7  pcs verify        {:>9.3} ms ({:>4.1}%)",
        sum.step7_pcs_verify.as_secs_f64() * 1e3,
        pct(sum.step7_pcs_verify)
    );
    eprintln!(
        "      total                     {:>9.3} ms",
        total.as_secs_f64() * 1e3
    );
}

/// Serialize each `Proof<F>` component into its own byte buffer and report
/// per-part raw + zstd-compressed sizes, so we can see how much each part
/// of the proof contributes to the total size. Sizes match the per-field
/// encoding used in `Proof::write_transcription_bytes_exact` (no extra
/// length prefixes).
fn eprint_folded_4x_proof_size_breakdown<F>(label: &str, proof: &Proof<F>)
where
    F: PrimeField,
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    fn to_bytes<T: Transcribable>(t: &T) -> Vec<u8> {
        let n = t.get_num_bytes();
        let mut buf = vec![0_u8; n];
        t.write_transcription_bytes_exact(&mut buf);
        buf
    }

    // 3 commitments concatenated (each ConstTranscribable, no length prefix).
    let mut commits = Vec::with_capacity(
        3_usize.saturating_mul(<ZipPlusCommitment as ConstTranscribable>::NUM_BYTES),
    );
    commits.extend_from_slice(&to_bytes(&proof.commitments.0));
    commits.extend_from_slice(&to_bytes(&proof.commitments.1));
    commits.extend_from_slice(&to_bytes(&proof.commitments.2));

    let ideal = to_bytes(&proof.ideal_check);
    let resolver = to_bytes(&proof.resolver);
    let combined_sumcheck = to_bytes(&proof.combined_sumcheck);
    let multipoint_eval = to_bytes(&proof.multipoint_eval);
    let witness_evals = to_bytes(DynamicPolyVecF::reinterpret(&proof.witness_lifted_evals));

    eprint_bytes_size_breakdown(
        label,
        &[
            ("commitments (3x)", &commits),
            ("zip (PCS bytes)", &proof.zip),
            ("ideal_check", &ideal),
            ("resolver", &resolver),
            ("combined_sumcheck", &combined_sumcheck),
            ("multipoint_eval", &multipoint_eval),
            ("witness_lifted_evals", &witness_evals),
        ],
    );
}

/// Per-domain (binary / arbitrary / integer) × per-substep breakdown of
/// the bytes inside `proof.zip`. Substeps mirror the four distinct
/// writes inside `ZipPlus::prove_f`: row evaluation vector `b`, the
/// combined row, opened column values across all `cw_matrices`, and
/// the Merkle authentication paths. The trailing total should match
/// the raw size of `proof.zip` exactly (the only u32 length prefix
/// outside this block belongs to the outer `Proof` envelope).
fn eprint_folded_4x_zip_substep_breakdown<F>(
    label: &str,
    proof: &Proof<F>,
    breakdown: &zinc_protocol::prover::FoldedProveZipBreakdown,
) where
    F: PrimeField,
{
    fn fmt_thousands(n: usize) -> String {
        let s = n.to_string();
        s.as_bytes()
            .rchunks(3)
            .rev()
            .map(|c| std::str::from_utf8(c).expect("ascii digits"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    let zip_total = proof.zip.len();
    let zip_total_f = (zip_total.max(1)) as f64;

    let domains: [(&str, &zip_plus::pcs::ZipPlusProveByteBreakdown); 3] = [
        ("bin (split2)", &breakdown.bin),
        ("arbitrary", &breakdown.arb),
        ("int", &breakdown.int),
    ];

    let zstd_len = |raw: &[u8]| -> usize {
        zstd::encode_all(raw, zip_plus::utils::ZSTD_LEVEL)
            .expect("zstd compression failed")
            .len()
    };

    eprintln!("    Zip+ PCS-byte substep breakdown ({label}):");
    eprintln!(
        "      {:<22} {:>14} {:>14} {:>14} {:>14} {:>14} {:>7}",
        "domain (raw)", "b", "combined_row", "column_values", "merkle_paths", "total", "of zip%",
    );
    let mut sum_b = 0_usize;
    let mut sum_cr = 0_usize;
    let mut sum_cv = 0_usize;
    let mut sum_mp = 0_usize;
    for (name, bd) in &domains {
        let total = bd.total();
        sum_b = sum_b.saturating_add(bd.b.len());
        sum_cr = sum_cr.saturating_add(bd.combined_row.len());
        sum_cv = sum_cv.saturating_add(bd.column_values.len());
        sum_mp = sum_mp.saturating_add(bd.merkle_proofs.len());
        eprintln!(
            "      {:<22} {:>14} {:>14} {:>14} {:>14} {:>14} {:>6.1}%",
            name,
            fmt_thousands(bd.b.len()),
            fmt_thousands(bd.combined_row.len()),
            fmt_thousands(bd.column_values.len()),
            fmt_thousands(bd.merkle_proofs.len()),
            fmt_thousands(total),
            100.0 * (total as f64) / zip_total_f,
        );
    }
    let sum_total = sum_b
        .saturating_add(sum_cr)
        .saturating_add(sum_cv)
        .saturating_add(sum_mp);
    eprintln!(
        "      {:<22} {:>14} {:>14} {:>14} {:>14} {:>14} {:>6.1}%",
        "TOTAL (raw)",
        fmt_thousands(sum_b),
        fmt_thousands(sum_cr),
        fmt_thousands(sum_cv),
        fmt_thousands(sum_mp),
        fmt_thousands(sum_total),
        100.0 * (sum_total as f64) / zip_total_f,
    );

    // Per-substep zstd-compressed sizes. Each (domain, substep) cell is
    // compressed independently, so per-row totals are the sum of the four
    // independently-compressed substeps — they slightly overshoot the
    // size of compressing the per-domain concatenation, and even more
    // so vs. compressing the whole proof.zip (cross-section redundancy
    // is lost when split). The trailing rows give those reference points.
    let zstd_label = format!("zstd-{}", zip_plus::utils::ZSTD_LEVEL);
    eprintln!(
        "      {:<22} {:>14} {:>14} {:>14} {:>14} {:>14} {:>7}",
        format!("domain ({zstd_label})"),
        "b",
        "combined_row",
        "column_values",
        "merkle_paths",
        "total",
        "of zip%",
    );
    let zip_zstd_total = zstd_len(&proof.zip);
    let zip_zstd_total_f = (zip_zstd_total.max(1)) as f64;
    let mut zstd_sum_b = 0_usize;
    let mut zstd_sum_cr = 0_usize;
    let mut zstd_sum_cv = 0_usize;
    let mut zstd_sum_mp = 0_usize;
    for (name, bd) in &domains {
        let zb = zstd_len(&bd.b);
        let zcr = zstd_len(&bd.combined_row);
        let zcv = zstd_len(&bd.column_values);
        let zmp = zstd_len(&bd.merkle_proofs);
        let row_total = zb
            .saturating_add(zcr)
            .saturating_add(zcv)
            .saturating_add(zmp);
        zstd_sum_b = zstd_sum_b.saturating_add(zb);
        zstd_sum_cr = zstd_sum_cr.saturating_add(zcr);
        zstd_sum_cv = zstd_sum_cv.saturating_add(zcv);
        zstd_sum_mp = zstd_sum_mp.saturating_add(zmp);
        eprintln!(
            "      {:<22} {:>14} {:>14} {:>14} {:>14} {:>14} {:>6.1}%",
            name,
            fmt_thousands(zb),
            fmt_thousands(zcr),
            fmt_thousands(zcv),
            fmt_thousands(zmp),
            fmt_thousands(row_total),
            100.0 * (row_total as f64) / zip_zstd_total_f,
        );
    }
    let zstd_sum_total = zstd_sum_b
        .saturating_add(zstd_sum_cr)
        .saturating_add(zstd_sum_cv)
        .saturating_add(zstd_sum_mp);
    eprintln!(
        "      {:<22} {:>14} {:>14} {:>14} {:>14} {:>14} {:>6.1}%",
        format!("TOTAL ({zstd_label})"),
        fmt_thousands(zstd_sum_b),
        fmt_thousands(zstd_sum_cr),
        fmt_thousands(zstd_sum_cv),
        fmt_thousands(zstd_sum_mp),
        fmt_thousands(zstd_sum_total),
        100.0 * (zstd_sum_total as f64) / zip_zstd_total_f,
    );
    eprintln!(
        "      (proof.zip raw = {} bytes; {zstd_label} whole-blob = {} bytes; substeps cover step-7 PCS writes only)",
        fmt_thousands(zip_total),
        fmt_thousands(zip_zstd_total),
    );
}

//
// Real-UAIR folded benches (1× and 4×). These reuse the generic
// `do_bench_e2e_folded` / `do_bench_e2e_folded_4x` helpers above with
// folded Zinc-types instances that pin `Int = RealEcdsaInt` (Int<5>) and
// reuse the arbitrary/int Zip-types from `RealEcdsaBenchZincTypes`.
//

#[derive(Clone, Debug)]
struct BenchFoldedRealEcdsaZincTypes;

impl FoldedZincTypes<DEGREE_PLUS_ONE, HALF_DEGREE_PLUS_ONE> for BenchFoldedRealEcdsaZincTypes {
    type Int = RealEcdsaInt;
    type Chal = i128;
    type Pt = i128;
    type Fmod = Uint<FIELD_LIMBS>;
    type PrimeTest = MillerRabin;

    type BinaryZt = GenericBenchZipTypes<
        BinaryPoly<HALF_DEGREE_PLUS_ONE>,
        DensePolynomial<i64, HALF_DEGREE_PLUS_ONE>,
        Self::Fmod,
        Self::PrimeTest,
        Self::Chal,
        Self::Pt,
        Int<5>,
        DensePolynomial<Int<5>, HALF_DEGREE_PLUS_ONE>,
        BinaryPolyInnerProduct<Self::Chal, HALF_DEGREE_PLUS_ONE>,
        DensePolyInnerProduct<Int<5>, Self::Chal, Int<5>, MBSInnerProduct, HALF_DEGREE_PLUS_ONE>,
        MBSInnerProduct,
    >;

    type ArbitraryZt = <RealEcdsaBenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::ArbitraryZt;
    type IntZt = <RealEcdsaBenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::IntZt;

    type BinaryLc = IprsCode<Self::BinaryZt, PnttConfigF65537, REP, PERFORM_CHECKS>;
    type ArbitraryLc = <RealEcdsaBenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::ArbitraryLc;
    type IntLc = <RealEcdsaBenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::IntLc;
}

//
// 4× int-fold variant of the bench Zinc-types. Implements
// `IntFoldedZincTypes4x` so that `prove_folded_4x` /
// `verify_folded_4x` route the int Zip+ commitments
// through Int<2> quarters (alongside the binary BinaryPoly<8>
// quarters) — and the shared-Merkle dispatch then collapses the two
// codeword-length-matching commitments into a single Merkle tree
// (one path per opening instead of two).
//
// `INT_QUARTER_LIMBS_BENCH = 2`: the 64-bit quarter values
// `q_0..q_3` from the decomposition `v = q_0 + 2^64·q_1 + 2^128·q_2
// + 2^192·q_3` get stored in `Int<2>` (one limb of sign-bit
// headroom over the 64 magnitude bits per quarter).
//

const INT_QUARTER_LIMBS_BENCH: usize = 2;

/// Quarter-int Zip+ types: `Eval = Int<2>`, `Cw = Int<3>`. `CombR` is
/// sized to match the regular (unfolded) int section's `CombR = Int<8>`
/// — Eval=Int<2> means each entry has only ~128 magnitude bits so the
/// linear-combination domain doesn't need the full 1024-bit `Int<16>`
/// the unfolded ECDSA Eval=Int<4> path uses. Picking `Int<8>` keeps
/// `validate_input`'s bit-width check satisfied while halving NTT cost
/// in `linear_code.encode_wide`, which dominates the int-fold-4x
/// verifier (the 4× row-len already makes the int instance the
/// long pole; doubling per-element cost is what made it 5× slower
/// than the regular folded-4× verifier).
type RealEcdsaIntQuarterZt = GenericBenchZipTypes<
    Int<INT_QUARTER_LIMBS_BENCH>,
    Int<{ INT_QUARTER_LIMBS_BENCH + 1 }>,
    Uint<FIELD_LIMBS>,
    MillerRabin,
    i128,
    i128,
    Int<6>,
    Int<6>,
    ScalarProduct,
    ScalarProduct,
    MBSInnerProduct,
>;

#[derive(Clone, Debug)]
struct BenchFoldedRealEcdsaZincTypes4x;

impl
    IntFoldedZincTypes4x<
        DEGREE_PLUS_ONE,
        QUARTER_DEGREE_PLUS_ONE,
        EC_FP_INT_LIMBS,
        INT_QUARTER_LIMBS_BENCH,
    > for BenchFoldedRealEcdsaZincTypes4x
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
        DensePolyInnerProduct<Int<5>, Self::Chal, Int<5>, MBSInnerProduct, QUARTER_DEGREE_PLUS_ONE>,
        MBSInnerProduct,
    >;
    type ArbitraryZt = <RealEcdsaBenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::ArbitraryZt;
    type IntZt = RealEcdsaIntQuarterZt;

    type BinaryLc = IprsCode<Self::BinaryZt, PnttConfigF65537, REP, PERFORM_CHECKS>;
    type ArbitraryLc = <RealEcdsaBenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::ArbitraryLc;
    type IntLc = IprsCode<Self::IntZt, PnttConfigF65537, REP, PERFORM_CHECKS>;
}

/// Setup PCS params for the 4× int-fold path: binary AND int both
/// commit at split4_size = 4n; arb stays at normal_size.
#[allow(clippy::type_complexity, clippy::unwrap_used)]
fn setup_folded_4x_pp_real_ecdsa(
    num_vars: usize,
) -> (
    ZipPlusParams<
        <BenchFoldedRealEcdsaZincTypes4x as IntFoldedZincTypes4x<
            DEGREE_PLUS_ONE,
            QUARTER_DEGREE_PLUS_ONE,
            EC_FP_INT_LIMBS,
            INT_QUARTER_LIMBS_BENCH,
        >>::BinaryZt,
        <BenchFoldedRealEcdsaZincTypes4x as IntFoldedZincTypes4x<
            DEGREE_PLUS_ONE,
            QUARTER_DEGREE_PLUS_ONE,
            EC_FP_INT_LIMBS,
            INT_QUARTER_LIMBS_BENCH,
        >>::BinaryLc,
    >,
    ZipPlusParams<
        <BenchFoldedRealEcdsaZincTypes4x as IntFoldedZincTypes4x<
            DEGREE_PLUS_ONE,
            QUARTER_DEGREE_PLUS_ONE,
            EC_FP_INT_LIMBS,
            INT_QUARTER_LIMBS_BENCH,
        >>::ArbitraryZt,
        <BenchFoldedRealEcdsaZincTypes4x as IntFoldedZincTypes4x<
            DEGREE_PLUS_ONE,
            QUARTER_DEGREE_PLUS_ONE,
            EC_FP_INT_LIMBS,
            INT_QUARTER_LIMBS_BENCH,
        >>::ArbitraryLc,
    >,
    ZipPlusParams<
        <BenchFoldedRealEcdsaZincTypes4x as IntFoldedZincTypes4x<
            DEGREE_PLUS_ONE,
            QUARTER_DEGREE_PLUS_ONE,
            EC_FP_INT_LIMBS,
            INT_QUARTER_LIMBS_BENCH,
        >>::IntZt,
        <BenchFoldedRealEcdsaZincTypes4x as IntFoldedZincTypes4x<
            DEGREE_PLUS_ONE,
            QUARTER_DEGREE_PLUS_ONE,
            EC_FP_INT_LIMBS,
            INT_QUARTER_LIMBS_BENCH,
        >>::IntLc,
    >,
) {
    let split4_size = 1 << (num_vars + 2);
    let normal_size = 1 << num_vars;
    (
        ZipPlus::setup(
            split4_size,
            IprsCode::new_with_optimal_depth(split4_size).unwrap(),
        ),
        ZipPlus::setup(
            normal_size,
            IprsCode::new_with_optimal_depth(normal_size).unwrap(),
        ),
        ZipPlus::setup(
            split4_size,
            IprsCode::new_with_optimal_depth(split4_size).unwrap(),
        ),
    )
}

fn bench_real_ecdsa_e2e_folded(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    type U = EcdsaUair<RealEcdsaInt>;

    let mut rng = rng();
    let trace = U::generate_random_trace(num_vars, &mut rng);
    let pp = setup_folded_pp_real_ecdsa(num_vars);

    let proj_ideal =
        |_: &IdealOrZero<<U as Uair>::Ideal>, _: &<F as PrimeField>::Config| -> ImpossibleIdeal {
            unreachable!("EcdsaUair has only assert_zero constraints")
        };

    do_bench_e2e_folded::<BenchFoldedRealEcdsaZincTypes, U, _>(
        group,
        "RealEcdsa",
        num_vars,
        &pp,
        &trace,
        zinc_protocol::project_scalar_fn,
        proj_ideal,
    );
}

fn bench_real_sha256_e2e_folded(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    type U = Sha256CompressionSliceUair<RealEcdsaInt>;

    let trace = real_sha256_chain_trace(num_vars);
    let pp = setup_folded_pp_real_ecdsa(num_vars);

    do_bench_e2e_folded::<BenchFoldedRealEcdsaZincTypes, U, _>(
        group,
        "RealSha256Chain8",
        num_vars,
        &pp,
        &trace,
        zinc_protocol::project_scalar_fn,
        sha256_real_project_ideal,
    );
}

fn bench_real_sha_ecdsa_e2e_folded(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    type U = ShaEcdsaUair<RealEcdsaInt>;

    let mut rng = rng();
    let trace = U::generate_random_trace(num_vars, &mut rng);
    let pp = setup_folded_pp_real_ecdsa(num_vars);

    do_bench_e2e_folded::<BenchFoldedRealEcdsaZincTypes, U, _>(
        group,
        "ShaEcdsa",
        num_vars,
        &pp,
        &trace,
        zinc_protocol::project_scalar_fn,
        sha256_real_project_ideal,
    );
}

/// ShaEcdsa 4× folded: binary AND int both quartered
/// (BinaryPoly<8> / Int<2>) and committed under one Merkle tree
/// via `MultiZip3`. One Merkle path per opening instead of three.
fn bench_real_sha_ecdsa_e2e_folded_4x(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    type U = ShaEcdsaUair<RealEcdsaInt>;

    let mut rng = rng();
    let trace = U::generate_random_trace(num_vars, &mut rng);
    let pp = setup_folded_4x_pp_real_ecdsa(num_vars);

    let witness_params = format!("ShaEcdsa/nvars={num_vars}");
    group.bench_function(
        BenchmarkId::new("Witness gen (folded 4×)", &witness_params),
        |bench| {
            bench.iter(|| {
                black_box(U::generate_random_trace(num_vars, &mut rng));
            });
        },
    );

    do_bench_e2e_folded_4x::<BenchFoldedRealEcdsaZincTypes4x, U, _>(
        group,
        "ShaEcdsa",
        num_vars,
        &pp,
        &trace,
        zinc_protocol::project_scalar_fn,
        sha256_real_project_ideal,
    );
}

fn e2e_folded_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zinc+ E2E Folded");

    // bench_sha_proxy_e2e_folded(&mut group, 8);
    // bench_sha_proxy_e2e_folded(&mut group, 10);
    // bench_sha_proxy_e2e_folded(&mut group, 12);

    bench_real_ecdsa_e2e_folded(&mut group, 9);
    bench_real_sha256_e2e_folded(&mut group, REAL_SHA256_CHAIN_NUM_VARS);
    bench_real_sha_ecdsa_e2e_folded(&mut group, 9);

    group.finish();
}

fn e2e_folded_4x_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zinc+ E2E Folded 4x");

    bench_real_sha_ecdsa_e2e_folded_4x(&mut group, 9);

    group.finish();
    print_peak_rss("Zinc+ E2E Folded 4x");
}

/// Prints peak resident-set-size of the bench process via `getrusage(RUSAGE_SELF)`.
/// `ru_maxrss` is monotonic across the process lifetime, so the value reflects
/// the peak observed across every bench that has run so far in this process.
fn print_peak_rss(label: &str) {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: getrusage writes a fully-initialized rusage on success.
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if rc != 0 {
        eprintln!(
            "[{label}] getrusage failed: {}",
            std::io::Error::last_os_error()
        );
        return;
    }
    let usage = unsafe { usage.assume_init() };

    // macOS reports `ru_maxrss` in bytes; Linux/BSD report kilobytes.
    #[cfg(target_os = "macos")]
    let bytes = usage.ru_maxrss as u64;
    #[cfg(not(target_os = "macos"))]
    let bytes = (usage.ru_maxrss as u64).saturating_mul(1024);

    let mib = bytes as f64 / (1024.0 * 1024.0);
    let gib = mib / 1024.0;
    eprintln!("[{label}] peak RSS: {bytes} B ({mib:.2} MiB / {gib:.3} GiB)");
}

criterion_group! {
    name = e2e;
    config = Criterion::default().sample_size(500);
    targets = e2e_benches
}
criterion_group! {
    name = e2e_steps;
    config = Criterion::default().sample_size(100);
    targets = e2e_steps_benches
}
criterion_group! {
    name = sha256_compare;
    config = Criterion::default().sample_size(20);
    targets = sha256_proving_system_compare_benches
}
criterion_group! {
    name = e2e_folded;
    config = Criterion::default().sample_size(500);
    targets = e2e_folded_benches
}
criterion_group! {
    name = e2e_folded_4x;
    config = Criterion::default().sample_size(500);
    targets = e2e_folded_4x_benches
}
criterion_main!(e2e, e2e_steps, sha256_compare, e2e_folded, e2e_folded_4x);
