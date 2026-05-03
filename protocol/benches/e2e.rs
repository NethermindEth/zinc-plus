#![allow(clippy::arithmetic_side_effects)]

use criterion::{
    BatchSize, BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main,
    measurement::WallTime,
};
use crypto_bigint::U64;
use crypto_primitives::{
    ConstIntRing, ConstIntSemiring, Field, FixedSemiring, FromWithConfig, PrimeField,
    crypto_bigint_int::Int, crypto_bigint_monty::MontyField, crypto_bigint_uint::Uint,
};
use rand::rng;
use std::{fmt::Debug, hint::black_box, marker::PhantomData, ops::Neg};
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
};
use zinc_test_uair::{
    BigLinearUair, BigLinearUairWithPublicInput, BinaryDecompositionUair, EC_FP_INT_LIMBS,
    EcdsaUair, GenerateRandomTrace, Sha256CompressionSliceUair, Sha256Ideal, ShaEcdsaUair,
    ShaProxy, TestUairNoMultiplication,
};
use zinc_poly::univariate::dynamic::over_field::DynamicPolyVecF;
use zinc_transcript::traits::{ConstTranscribable, Transcribable};
use zinc_uair::{
    Uair, UairTrace,
    degree_counter::count_effective_max_degree,
    ideal::{DegreeOneIdeal, Ideal, IdealCheck, ImpossibleIdeal, rotation::RotationIdeal},
    ideal_collector::IdealOrZero,
};
use zinc_utils::{
    from_ref::FromRef,
    inner_product::{InnerProduct, MBSInnerProduct, ScalarProduct},
    mul_by_scalar::MulByScalar,
    named::Named,
    projectable_to_field::ProjectableToField,
};
use zip_plus::{
    code::iprs::{IprsCode, PnttConfigF65537},
    pcs::structs::{ZipPlus, ZipPlusCommitment, ZipPlusParams, ZipTypes},
    utils::{eprint_bytes_size_breakdown, eprint_proof_size},
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
// Constants and concrete types
//

const DEGREE_PLUS_ONE: usize = 32;
const INT_LIMBS: usize = U64::LIMBS;
// `fixed-prime` branch: 256-bit field modulus (4 × u64 limbs) so that the
// fixed secp256k1 base prime fits in `Fmod = Uint<FIELD_LIMBS>`.
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

    let proj_ideal = |_: &IdealOrZero<<U as Uair>::Ideal>,
                      _: &<F as PrimeField>::Config|
     -> ImpossibleIdeal {
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

    let proj_ideal = |_: &IdealOrZero<<U as Uair>::Ideal>,
                      _: &<F as PrimeField>::Config|
     -> ImpossibleIdeal {
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

    let mut rng = rng();
    let trace = U::generate_random_trace(num_vars, &mut rng);
    let pp = setup_pp_real_ecdsa(num_vars);

    do_bench_e2e::<RealEcdsaBenchZincTypes, U, _>(
        group,
        "RealSha256",
        num_vars,
        &pp,
        &trace,
        zinc_protocol::project_scalar_fn,
        sha256_real_project_ideal,
    );
}

fn bench_real_sha256_steps(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    type U = Sha256CompressionSliceUair<RealEcdsaInt>;

    let mut rng = rng();
    let trace = U::generate_random_trace(num_vars, &mut rng);
    let pp = setup_pp_real_ecdsa(num_vars);

    do_bench_steps::<RealEcdsaBenchZincTypes, U, _>(
        group,
        "RealSha256",
        num_vars,
        &pp,
        &trace,
        zinc_protocol::project_scalar_fn,
        sha256_real_project_ideal,
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
    bench_real_sha256_e2e(&mut group, 9);
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
    bench_real_sha256_steps(&mut group, 9);
    bench_real_sha_ecdsa_steps(&mut group, 9);

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

    group.bench_function(
        BenchmarkId::new("Verify (folded 4×)", &params),
        |bench| {
            bench.iter_batched(
                || proof.clone(),
                |proof| {
                    black_box(
                        zinc_protocol::verifier::verify_folded_4x::<
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
                        ),
                    )
                    .expect("Folded 4× verifier failed");
                },
                BatchSize::SmallInput,
            );
        },
    );

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
    use zinc_protocol::prover::{prove_folded_4x_with_timings, FoldedProveTimings};

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
    use zinc_protocol::verifier::{
        verify_folded_4x_with_timings, FoldedVerifyTimings,
    };

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
    let mut commits =
        Vec::with_capacity(3_usize.saturating_mul(<ZipPlusCommitment as ConstTranscribable>::NUM_BYTES));
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
        DensePolyInnerProduct<
            Int<5>,
            Self::Chal,
            Int<5>,
            MBSInnerProduct,
            HALF_DEGREE_PLUS_ONE,
        >,
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
    Int<{ EC_FP_INT_LIMBS * 2 }>,
    Int<{ EC_FP_INT_LIMBS * 2 }>,
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
        DensePolyInnerProduct<
            Int<5>,
            Self::Chal,
            Int<5>,
            MBSInnerProduct,
            QUARTER_DEGREE_PLUS_ONE,
        >,
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

    let proj_ideal = |_: &IdealOrZero<<U as Uair>::Ideal>,
                      _: &<F as PrimeField>::Config|
     -> ImpossibleIdeal {
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

    let mut rng = rng();
    let trace = U::generate_random_trace(num_vars, &mut rng);
    let pp = setup_folded_pp_real_ecdsa(num_vars);

    do_bench_e2e_folded::<BenchFoldedRealEcdsaZincTypes, U, _>(
        group,
        "RealSha256",
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
fn bench_real_sha_ecdsa_e2e_folded_4x(
    group: &mut BenchmarkGroup<WallTime>,
    num_vars: usize,
) {
    type U = ShaEcdsaUair<RealEcdsaInt>;

    let mut rng = rng();
    let trace = U::generate_random_trace(num_vars, &mut rng);
    let pp = setup_folded_4x_pp_real_ecdsa(num_vars);

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
    bench_real_sha256_e2e_folded(&mut group, 9);
    bench_real_sha_ecdsa_e2e_folded(&mut group, 9);

    group.finish();
}

fn e2e_folded_4x_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zinc+ E2E Folded 4x");

    bench_real_sha_ecdsa_e2e_folded_4x(&mut group, 9);

    group.finish();
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
    name = e2e_folded;
    config = Criterion::default().sample_size(500);
    targets = e2e_folded_benches
}
criterion_group! {
    name = e2e_folded_4x;
    config = Criterion::default().sample_size(500);
    targets = e2e_folded_4x_benches
}
criterion_main!(e2e, e2e_steps, e2e_folded, e2e_folded_4x);
