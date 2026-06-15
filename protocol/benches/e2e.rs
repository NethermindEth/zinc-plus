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
    borrow::Cow, fmt::Debug, fs, hint::black_box, marker::PhantomData, ops::Neg, path::Path,
    time::Instant,
};
use zinc_piop::neutron_nova::{
    MleTable, ProjectedPublic, ProjectedTrace, SHA_ROW_COUNT, SHA_ROW_VARS, SHA_WORD_BITS,
    ShaBinaryFoldField, ShaLinearAccumulatorField, ShaPublicCol, bit_slice_index,
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
        ProductionShaError, ProductionShaMixedHyraxPcs, ProductionShaMixedHyraxProof,
        ProductionShaPackedHyraxProof, ProductionShaProjectionAdapter, ProductionShaWitnessPolys,
        UairShape, packed_sha_layout, prepare_linear_ideal_fold_witnesses,
        prove_prepared_linear_ideal_fold_mixed_hyrax,
        prove_prepared_linear_ideal_fold_packed_hyrax,
        setup_verify_linear_ideal_fold_mixed_hyrax,
        setup_verify_linear_ideal_fold_packed_hyrax, verify_linear_ideal_fold_mixed_hyrax,
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
/// no Montgomery-limb Barrett reducer, so config acquisition, int projection,
/// and bit-slice scalarization route through this seam.
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
        projection_sha_scalarize_bit_slices_generic(bit_slices, a, field_cfg)
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

fn projection_sha_scalarize_bit_slices_dmr(
    bit_slices: &MleTable<F>,
    a: &F,
    field_cfg: &<F as PrimeField>::Config,
) -> Result<MleTable<F>, ProductionShaError<F>> {
    let powers = zinc_utils::powers(a.clone(), F::one_with_cfg(field_cfg), SHA_WORD_BITS);
    let word_count = bit_slices.len() / SHA_WORD_BITS;
    let one = F::one_with_cfg(field_cfg);
    let reducer = BarrettDelayedReduction::<F>::new(field_cfg);
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
                .collect::<Result<Vec<_>, ProductionShaError<F>>>()?;
            let mut out_col = Vec::with_capacity(SHA_ROW_COUNT);
            for row in 0..SHA_ROW_COUNT {
                out_col.push(projection_sha_scalarize_binary_row_dmr(
                    &bit_cols, row, &powers, &one, field_cfg, &reducer,
                ));
            }
            Ok(out_col)
        })
        .collect::<Result<Vec<_>, ProductionShaError<F>>>()?;
    Ok(projection_sha_mle_table_from_columns(words))
}

fn projection_sha_scalarize_bit_slices_generic<F: PrimeField>(
    bit_slices: &MleTable<F>,
    a: &F,
    field_cfg: &F::Config,
) -> Result<MleTable<F>, ProductionShaError<F>> {
    let powers = zinc_utils::powers(a.clone(), F::one_with_cfg(field_cfg), SHA_WORD_BITS);
    let word_count = bit_slices.len() / SHA_WORD_BITS;
    let one = F::one_with_cfg(field_cfg);
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
                .collect::<Result<Vec<_>, ProductionShaError<F>>>()?;
            let mut out_col = Vec::with_capacity(SHA_ROW_COUNT);
            for row in 0..SHA_ROW_COUNT {
                let mut value = F::zero_with_cfg(field_cfg);
                for (bit_col, power) in bit_cols.iter().zip(&powers) {
                    let bit = &bit_col.evaluations[row];
                    if F::is_zero(bit) {
                        continue;
                    }
                    if bit == &one {
                        value += power.clone();
                    } else {
                        value += bit.clone() * power;
                    }
                }
                out_col.push(value);
            }
            Ok(out_col)
        })
        .collect::<Result<Vec<_>, ProductionShaError<F>>>()?;
    Ok(projection_sha_mle_table_from_columns(words))
}

fn projection_sha_scalarize_binary_row_dmr(
    bit_cols: &[&DenseMultilinearExtension<F>],
    row: usize,
    powers: &[F],
    one: &F,
    field_cfg: &<F as PrimeField>::Config,
    reducer: &BarrettDelayedReduction<'_, F>,
) -> F {
    let mut bucket = Uint::<5>::zero();
    let mut pending_adds = 0usize;
    let mut acc = F::zero_with_cfg(field_cfg);

    for (bit_col, power) in bit_cols.iter().zip(powers) {
        let bit = &bit_col.evaluations[row];
        if F::is_zero(bit) {
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

fn projection_sha_scalarize_row_naive(
    bit_cols: &[&DenseMultilinearExtension<F>],
    row: usize,
    powers: &[F],
    field_cfg: &<F as PrimeField>::Config,
) -> F {
    let mut value = F::zero_with_cfg(field_cfg);
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
        let scalarized = F::scalarize_bit_slices(
            &bit_slices,
            &F::from_with_cfg(2u64, field_cfg),
            field_cfg,
        )?;
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

#[derive(Clone, Debug)]
struct HyraxWidthSweepRow {
    label: String,
    width: Option<usize>,
    ecc_points_per_instance: Option<usize>,
    fresh_ecc_points: Option<usize>,
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
    kind: &'static str,
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

const HYRAX_WIDTH_SWEEP_WARMUP_RUNS: usize = 2;
const HYRAX_WIDTH_SWEEP_TUNING_SAMPLES: usize = 5;
const HYRAX_WIDTH_SWEEP_CONFIRMATION_SAMPLES: usize = 15;
const HYRAX_WIDTH_SWEEP_CONFIRMATION_TOP_K: usize = 3;

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

#[allow(clippy::too_many_arguments)]
fn hyrax_width_sweep_row(
    label: String,
    width: Option<usize>,
    ecc_points_per_instance: Option<usize>,
    fresh_ecc_points: Option<usize>,
    prover: TimingStats,
    verifier: TimingStats,
    proof_bytes: usize,
    proof_zstd_bytes: usize,
    kind: &'static str,
) -> HyraxWidthSweepRow {
    HyraxWidthSweepRow {
        label,
        width,
        ecc_points_per_instance,
        fresh_ecc_points,
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
        kind,
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

fn write_hyrax_width_sweep_csv(path: &Path, rows: &[HyraxWidthSweepRow]) {
    let mut csv = String::from(
        "label,kind,width,ecc_points_per_instance,fresh_ecc_points,prover_median_ms,prover_mean_ms,prover_min_ms,prover_max_ms,prover_samples,verifier_median_ms,verifier_mean_ms,verifier_min_ms,verifier_max_ms,verifier_samples,proof_bytes,proof_zstd_bytes\n",
    );
    for row in rows {
        csv.push_str(&format!(
            "{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{},{:.6},{:.6},{:.6},{:.6},{},{},{}\n",
            row.label,
            row.kind,
            row.width
                .map(|value| value.to_string())
                .unwrap_or_else(|| "N/A".to_string()),
            row.ecc_points_per_instance
                .map(|value| value.to_string())
                .unwrap_or_else(|| "N/A".to_string()),
            row.fresh_ecc_points
                .map(|value| value.to_string())
                .unwrap_or_else(|| "N/A".to_string()),
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
        ));
    }
    fs::write(path, csv).expect("write hyrax width sweep CSV");
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
    [128usize, 256, 512, 1024, 2048, 4096]
        .into_iter()
        .map(|width| (format!("mixed {width}"), width))
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
    let packed = rows
        .iter()
        .filter(|row| row.kind == "packed")
        .collect::<Vec<_>>();
    let og = rows.iter().find(|row| row.kind == "og_zip");
    let fastest_mixed = rows
        .iter()
        .filter(|row| row.kind == "mixed_hyrax")
        .min_by(|a, b| {
            a.prover_ms
                .partial_cmp(&b.prover_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

    let width_min = packed.iter().filter_map(|row| row.width).min().unwrap_or(1) as f64;
    let width_max = packed.iter().filter_map(|row| row.width).max().unwrap_or(2) as f64;
    let xlog =
        |x: f64| 80.0 + ((x.ln() - width_min.ln()) / (width_max.ln() - width_min.ln())) * 860.0;

    let (time_min, time_max) = finite_range(
        packed
            .iter()
            .flat_map(|row| [row.prover_ms, row.verifier_ms])
            .chain(og.iter().flat_map(|row| [row.prover_ms, row.verifier_ms]))
            .chain(fastest_mixed.iter().flat_map(|row| [row.prover_ms, row.verifier_ms])),
    );
    let ytime = |y: f64| 500.0 - ((y - time_min) / (time_max - time_min)) * 430.0;
    let prover_points = packed
        .iter()
        .map(|row| (row.width.unwrap_or_default() as f64, row.prover_ms))
        .collect::<Vec<_>>();
    let verifier_points = packed
        .iter()
        .map(|row| (row.width.unwrap_or_default() as f64, row.verifier_ms))
        .collect::<Vec<_>>();
    let mut body = String::new();
    body.push_str(&svg_polyline(&prover_points, "#0f766e", &xlog, &ytime));
    body.push_str(&svg_polyline(&verifier_points, "#7c3aed", &xlog, &ytime));
    let mut png = blank_png_canvas();
    draw_png_axes(&mut png);
    draw_png_polyline(&mut png, &prover_points, PNG_PROVER, &xlog, &ytime);
    draw_png_polyline(&mut png, &verifier_points, PNG_VERIFIER, &xlog, &ytime);
    if let Some(row) = og {
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
    if let Some(row) = fastest_mixed {
        body.push_str(&format!(
            r##"<line x1="80" x2="940" y1="{:.2}" y2="{:.2}" stroke="#dc2626" stroke-dasharray="8 5"/><line x1="80" x2="940" y1="{:.2}" y2="{:.2}" stroke="#dc2626" stroke-dasharray="2 5"/>"##,
            ytime(row.prover_ms),
            ytime(row.prover_ms),
            ytime(row.verifier_ms),
            ytime(row.verifier_ms)
        ));
        draw_png_dashed_hline(&mut png, ytime(row.prover_ms), PNG_MIXED);
        draw_png_dashed_hline(&mut png, ytime(row.verifier_ms), PNG_MIXED);
    }
    body.push_str(r##"<text x="720" y="92" font-family="sans-serif" font-size="14" fill="#0f766e">prover</text><text x="720" y="112" font-family="sans-serif" font-size="14" fill="#7c3aed">verifier</text>"##);
    write_svg(
        &out_dir.join("plot1_time_vs_width.svg"),
        "Prover/Verifier Time vs Width",
        &body,
    );
    write_png(&out_dir.join("plot1_time_vs_width.png"), &png);

    let (proof_min, proof_max) = finite_range(rows.iter().map(|row| row.proof_bytes as f64));
    let yproof = |y: f64| 500.0 - ((y - proof_min) / (proof_max - proof_min)) * 430.0;
    let proof_points = packed
        .iter()
        .map(|row| (row.width.unwrap_or_default() as f64, row.proof_bytes as f64))
        .collect::<Vec<_>>();
    let mut body = svg_polyline(&proof_points, "#2563eb", &xlog, &yproof);
    let mut png = blank_png_canvas();
    draw_png_axes(&mut png);
    draw_png_polyline(&mut png, &proof_points, PNG_PROOF, &xlog, &yproof);
    if let Some(row) = fastest_mixed {
        body.push_str(&format!(
            r##"<line x1="80" x2="940" y1="{:.2}" y2="{:.2}" stroke="#dc2626" stroke-dasharray="6 4"/>"##,
            yproof(row.proof_bytes as f64),
            yproof(row.proof_bytes as f64)
        ));
        draw_png_dashed_hline(&mut png, yproof(row.proof_bytes as f64), PNG_MIXED);
    }
    if let Some(row) = og {
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
        finite_range(packed.iter().map(|row| row.proof_bytes as f64));
    let (scatter_y_min, scatter_y_max) = finite_range(packed.iter().map(|row| row.prover_ms));
    let sx = |x: f64| 80.0 + ((x - scatter_x_min) / (scatter_x_max - scatter_x_min)) * 860.0;
    let sy = |y: f64| 500.0 - ((y - scatter_y_min) / (scatter_y_max - scatter_y_min)) * 430.0;
    let max_ecc = packed
        .iter()
        .filter_map(|row| row.ecc_points_per_instance)
        .max()
        .unwrap_or(1) as f64;
    let mut body = String::new();
    let mut png = blank_png_canvas();
    draw_png_axes(&mut png);
    for row in &packed {
        let ecc = row.ecc_points_per_instance.unwrap_or(1) as f64;
        let radius = 4.0 + 8.0 * (ecc / max_ecc).sqrt();
        let x = sx(row.proof_bytes as f64);
        let y = sy(row.prover_ms);
        body.push_str(&format!(
            r##"<circle cx="{x:.2}" cy="{y:.2}" r="{radius:.2}" fill="#f97316" fill-opacity="0.72"/><text x="{:.2}" y="{:.2}" font-family="sans-serif" font-size="11">{}</text>"##,
            x + radius + 2.0,
            y - radius,
            row.label
        ));
        draw_png_filled_circle(&mut png, x, y, radius, PNG_SCATTER);
    }
    write_svg(
        &out_dir.join("plot3_pareto_scatter.svg"),
        "Prover Time vs Proof Size",
        &body,
    );
    write_png(&out_dir.join("plot3_pareto_scatter.png"), &png);

    let ecc_min = packed
        .iter()
        .filter_map(|row| row.ecc_points_per_instance)
        .min()
        .unwrap_or(1) as f64;
    let ecc_max = packed
        .iter()
        .filter_map(|row| row.ecc_points_per_instance)
        .max()
        .unwrap_or(2) as f64;
    let x_ecc = |x: f64| 80.0 + ((x.ln() - ecc_min.ln()) / (ecc_max.ln() - ecc_min.ln())) * 860.0;
    let (verifier_min, verifier_max) = finite_range(packed.iter().map(|row| row.verifier_ms));
    let yver = |y: f64| 500.0 - ((y - verifier_min) / (verifier_max - verifier_min)) * 430.0;
    let verifier_ecc_points = packed
        .iter()
        .map(|row| {
            (
                row.ecc_points_per_instance.unwrap_or_default() as f64,
                row.verifier_ms,
            )
        })
        .collect::<Vec<_>>();
    let body = svg_polyline(&verifier_ecc_points, "#9333ea", &x_ecc, &yver);
    let mut png = blank_png_canvas();
    draw_png_axes(&mut png);
    draw_png_polyline(&mut png, &verifier_ecc_points, PNG_VERIFIER, &x_ecc, &yver);
    write_svg(
        &out_dir.join("plot4_verifier_vs_ecc.svg"),
        "Verifier Time vs ECC Points",
        &body,
    );
    write_png(&out_dir.join("plot4_verifier_vs_ecc.png"), &png);
}

fn write_hyrax_width_sweep_report(
    out_dir: &Path,
    rows: &[HyraxWidthSweepRow],
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
            "{}{}: prover median {:.3} ms, verifier median {:.3} ms, proof {} bytes, zstd {} bytes",
            row.label,
            row.width
                .map(|width| format!(
                    " (width {width}, ECC/inst {}, fresh ECC {})",
                    row.ecc_points_per_instance
                        .map(|value| value.to_string())
                        .unwrap_or_else(|| "N/A".to_string()),
                    row.fresh_ecc_points
                        .map(|value| value.to_string())
                        .unwrap_or_else(|| "N/A".to_string())
                ))
                .unwrap_or_default(),
            row.prover_ms,
            row.verifier_ms,
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
            "<table><thead><tr><th>label</th><th>kind</th><th>width</th><th>ECC points/inst</th><th>fresh ECC points</th><th>prover median ms</th><th>prover mean ms</th><th>prover min ms</th><th>prover max ms</th><th>verifier median ms</th><th>verifier mean ms</th><th>verifier min ms</th><th>verifier max ms</th><th>samples p/v</th><th>proof bytes</th><th>zstd bytes</th></tr></thead><tbody>",
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
                html_escape(&row.label),
                row.kind,
                optional_usize(row.width),
                optional_usize(row.ecc_points_per_instance),
                optional_usize(row.fresh_ecc_points),
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

    let packed_rows = rows
        .iter()
        .filter(|row| row.kind == "packed")
        .collect::<Vec<_>>();
    let mixed_rows = rows
        .iter()
        .filter(|row| row.kind == "mixed_hyrax")
        .collect::<Vec<_>>();
    let hyrax_rows = rows
        .iter()
        .filter(|row| row.kind != "og_zip")
        .collect::<Vec<_>>();
    let og_zip = rows.iter().find(|row| row.kind == "og_zip");
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
    let mixed_fastest_prover = best_row_by(mixed_rows.iter().copied(), |row| row.prover_ms);
    let mixed_fastest_verifier = best_row_by(mixed_rows.iter().copied(), |row| row.verifier_ms);
    let mixed_smallest_proof =
        best_row_by(mixed_rows.iter().copied(), |row| row.proof_bytes as f64);
    let mixed_smallest_zstd =
        best_row_by(mixed_rows.iter().copied(), |row| row.proof_zstd_bytes as f64);
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
        mixed_fastest_prover,
        mixed_fastest_verifier,
        mixed_smallest_proof,
        mixed_smallest_zstd,
        packed_balanced,
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();

    let all_rows = rows.iter().collect::<Vec<_>>();
    let table = row_table(&all_rows, &winner_rows);
    let pareto_rows = pareto_frontier(&hyrax_rows);
    let pareto_table = row_table(&pareto_rows, &[]);
    let confirmation_refs = confirmation_rows.iter().collect::<Vec<_>>();
    let confirmation_table = row_table(&confirmation_refs, &[]);

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
<li><strong>Fastest mixed prover:</strong> {}</li>
<li><strong>Fastest mixed verifier:</strong> {}</li>
<li><strong>Smallest mixed raw proof:</strong> {}</li>
<li><strong>Smallest mixed zstd proof:</strong> {}</li>
<li><strong>Fastest packed prover:</strong> {}</li>
<li><strong>Fastest packed verifier:</strong> {}</li>
<li><strong>Smallest packed raw proof:</strong> {}</li>
<li><strong>Smallest packed zstd proof:</strong> {}</li>
<li><strong>Best balanced packed width:</strong> {} <em>(min normalized prover + verifier + raw proof + zstd proof)</em></li>
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
        mixed_fastest_prover
            .map(row_summary)
            .unwrap_or_else(|| "N/A".to_string()),
        mixed_fastest_verifier
            .map(row_summary)
            .unwrap_or_else(|| "N/A".to_string()),
        mixed_smallest_proof
            .map(row_summary)
            .unwrap_or_else(|| "N/A".to_string()),
        mixed_smallest_zstd
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
<p>Mixed Hyrax widths tune the row-domain commitment width for the specialized binary/int lane layout. The current mixed folded-opening path is single-row only, so this sweep starts at 128. Widths above 128 cannot reduce mixed ECC commitments below one row per source, but they test wider opening vectors and fixed-base MSM behavior.</p>
<p>Timing mode: {HYRAX_WIDTH_SWEEP_WARMUP_RUNS} warmup runs per config, {HYRAX_WIDTH_SWEEP_TUNING_SAMPLES} recorded tuning samples per config, and {HYRAX_WIDTH_SWEEP_CONFIRMATION_SAMPLES} recorded confirmation samples for the top {HYRAX_WIDTH_SWEEP_CONFIRMATION_TOP_K} Hyrax prover candidates. Headline times are medians.</p>
{most_performant}
{table}
{skipped_html}
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
        "OG Zinc+ hash PCS".to_string(),
        None,
        None,
        None,
        zip_prover_stats,
        zip_verifier_stats,
        zip_raw.len(),
        zstd_len(&zip_raw),
        "og_zip",
    ));

    let measure_mixed_hyrax =
        |label: String, width: usize, sample_count: usize| -> HyraxWidthSweepRow {
        let (mixed_pcs_params, mixed_verifier_params) =
            projection_sha_hyrax_pcs_params::<C, HyraxF>(width);
        let mixed_pp = LinearIdealFoldProverParams::<
            P,
            U,
            RealEcdsaBenchZincTypes,
            HyraxF,
            DEGREE_PLUS_ONE,
        >::new(mixed_pcs_params, hyrax_field_cfg.clone(), 3);
        let mixed_vs = setup_verify_linear_ideal_fold_mixed_hyrax::<
            C,
            U,
            RealEcdsaBenchZincTypes,
            HyraxF,
            DEGREE_PLUS_ONE,
        >(
            LinearIdealFoldVerifierParams::new(mixed_verifier_params, hyrax_field_cfg.clone()),
            shape.clone(),
        )
        .expect("mixed Hyrax verifier setup succeeds");
        let (mixed_output, mixed_prover_stats) =
            measure_warmed(HYRAX_WIDTH_SWEEP_WARMUP_RUNS, sample_count, || {
                let mut transcript = Blake3Transcript::new();
                prove_prepared_linear_ideal_fold_mixed_hyrax::<
                    C,
                    U,
                    RealEcdsaBenchZincTypes,
                    HyraxF,
                    DEGREE_PLUS_ONE,
                >(&mixed_pp, &shape, &prepared_instances, &mut transcript)
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
            label,
            Some(width),
            Some(mixed_ecc),
            Some(mixed_fresh_ecc),
            mixed_prover_stats,
            mixed_verifier_stats,
            mixed_raw.len(),
            zstd_len(&mixed_raw),
            "mixed_hyrax",
        )
    };

    let mixed_widths = mixed_width_candidates();
    for (label, width) in &mixed_widths {
        rows.push(measure_mixed_hyrax(
            label.clone(),
            *width,
            HYRAX_WIDTH_SWEEP_TUNING_SAMPLES,
        ));
    }

    let (widths, skipped_widths) = packed_width_candidates::<HyraxF>();
    let measure_packed_hyrax = |label: String,
                                width: usize,
                                sample_count: usize|
     -> HyraxWidthSweepRow {
        let layout = packed_sha_layout::<HyraxF>(width).expect("candidate packed width is valid");
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
                >(&pp, &shape, &prepared_instances, &mut transcript)
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
                >(&vs, &output.fresh_instances, &output.proof, &mut transcript)
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
            label,
            Some(width),
            Some(layout.ecc_points_per_instance()),
            Some(fresh_ecc),
            prover_stats,
            verifier_stats,
            raw.len(),
            zstd_len(&raw),
            "packed",
        )
    };

    for (label, width) in &widths {
        rows.push(measure_packed_hyrax(
            label.clone(),
            *width,
            HYRAX_WIDTH_SWEEP_TUNING_SAMPLES,
        ));
    }

    let mut confirmation_targets = rows
        .iter()
        .filter(|row| row.kind != "og_zip")
        .map(|row| (row.prover_ms, row.kind, row.label.clone(), row.width))
        .collect::<Vec<_>>();
    confirmation_targets.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    confirmation_targets.truncate(HYRAX_WIDTH_SWEEP_CONFIRMATION_TOP_K);
    let confirmation_rows = confirmation_targets
        .into_iter()
        .map(|(_, kind, label, width)| match kind {
            "mixed_hyrax" => measure_mixed_hyrax(
                label,
                width.expect("mixed confirmation target has a width"),
                HYRAX_WIDTH_SWEEP_CONFIRMATION_SAMPLES,
            ),
            "packed" => measure_packed_hyrax(
                label,
                width.expect("packed confirmation target has a width"),
                HYRAX_WIDTH_SWEEP_CONFIRMATION_SAMPLES,
            ),
            _ => unreachable!("only Hyrax rows are selected for confirmation"),
        })
        .collect::<Vec<_>>();

    write_hyrax_width_sweep_csv(&out_dir.join("results.csv"), &rows);
    write_hyrax_width_sweep_csv(&out_dir.join("confirmation.csv"), &confirmation_rows);
    write_hyrax_width_sweep_skipped_csv(&out_dir.join("skipped_widths.csv"), &skipped_widths);
    write_hyrax_width_sweep_plots(&out_dir, &rows);
    write_hyrax_width_sweep_report(&out_dir, &rows, &confirmation_rows, &skipped_widths);
    eprintln!(
        "hyrax width sweep wrote {}, {}, {}, {}, and SVG/PNG plots",
        out_dir.join("results.csv").display(),
        out_dir.join("confirmation.csv").display(),
        out_dir.join("skipped_widths.csv").display(),
        out_dir.join("report.html").display()
    );
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
    let (binary, binary_vk) = HyraxPCS::<C, BinaryLanes>::setup(
        binary_width,
        b"zinc-plus-bench-real-sha256-hyrax-binary",
        HyraxBlindingMode::Unblinded,
    )
    .expect("Hyrax binary benchmark setup must be valid");
    let (int, int_vk) = HyraxPCS::<C, IntScalarLane>::setup(
        int_width,
        b"zinc-plus-bench-real-sha256-hyrax-int",
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
    let (pcs_params, pcs_verifier_params) =
        projection_sha_hyrax_pcs_params::<C, F>(SHA_ROW_COUNT);
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
                &pp, &shape, &prepared_instances, &mut transcript
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
    >(&pp, &shape, &prepared_instances, &mut prover_transcript)
    .expect("proof generation for ProjectionFold verifier bench");

    let mut verifier_transcript = Blake3Transcript::new();
    let verified =
        verify_linear_ideal_fold_mixed_hyrax::<C, U, RealEcdsaBenchZincTypes, F, DEGREE_PLUS_ONE>(
            &vs,
            &output.fresh_instances,
            &output.proof,
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
        >(&pp, &shape, &prepared_instances, &mut prover_transcript)
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
