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
use zinc_protocol::{FoldedZincTypes, Proof, ZincPlusPiop, ZincTypes};
use zinc_test_uair::{
    BigLinearUair, BigLinearUairWithPublicInput, BinaryDecompositionUair, ECDSA_INT_LIMBS,
    EcdsaScalarSliceUair, GenerateRandomTrace, Sha256CompressionSliceUair, Sha256Ideal,
    ShaEcdsaLinearizedUair, ShaEcdsaUair, ShaProxy, TestUairNoMultiplication,
};
use zinc_transcript::traits::ConstTranscribable;
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
    code::iprs::{IprsCode, PnttConfigF12289},
    pcs::structs::{ZipPlus, ZipPlusParams, ZipTypes},
    utils::eprint_proof_size,
};

//
// Type definitions and constants
//

const PERFORM_CHECKS: bool = if cfg!(feature = "unchecked") {
    zinc_utils::UNCHECKED
} else {
    zinc_utils::CHECKED
};

/// Repetition factor for linear code, an inverse rate.
/// Enable `bench-rate-1-8` to switch from the default rate 1/4 to rate 1/8.
#[cfg(feature = "bench-rate-1-8")]
const REP: usize = 8;
#[cfg(not(feature = "bench-rate-1-8"))]
const REP: usize = 4;

const RATE_TAG: &str = match REP {
    8 => "/rate=1_8",
    _ => "/rate=1_4",
};

/// At rate 1/8 we bump the IPRS tree depth by one beyond
/// `new_with_optimal_depth`'s default heuristic.
const EXTRA_DEPTH: usize = if REP >= 8 { 1 } else { 0 };

/// Mirrors `IprsCode::new_with_optimal_depth`'s formula, then adds
/// `EXTRA_DEPTH`. Kept in sync with zip-plus/src/code/iprs.rs.
fn iprs_depth(row_len: usize) -> usize {
    const MAX_BASE_COLS_LOG2: usize = 7;
    let target_base_len = 1usize << MAX_BASE_COLS_LOG2;
    let base = 1.max(((1.max(row_len / target_base_len)).ilog2() as usize).div_ceil(3));
    base + EXTRA_DEPTH
}

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
    const NUM_COLUMN_OPENINGS: usize = match REP {
        8 => 96,
        _ => 144,
    };
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
struct GenericBenchZincTypes<Int, CwR, Chal, Pt, CombR, Fmod, PrimeTest, const D: usize>(
    PhantomData<(Int, CwR, Chal, Pt, CombR, Fmod, PrimeTest)>,
);

impl<Int, CwR, Chal, Pt, CombR, Fmod, PrimeTest, const D: usize> ZincTypes<D>
    for GenericBenchZincTypes<Int, CwR, Chal, Pt, CombR, Fmod, PrimeTest, D>
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
    Fmod: ConstIntSemiring + ConstTranscribable + Named,
    PrimeTest: PrimalityTest<Fmod> + Debug + Send + Sync,
{
    type Int = Int;
    type Chal = Chal;
    type Pt = Pt;
    type CombR = CombR;
    type Fmod = Fmod;
    type PrimeTest = PrimeTest;

    type BinaryZt = GenericBenchZipTypes<
        BinaryPoly<D>,
        DensePolynomial<i64, D>,
        Fmod,
        PrimeTest,
        Chal,
        Pt,
        CombR,
        DensePolynomial<CombR, D>,
        BinaryPolyInnerProduct<Chal, D>,
        DensePolyInnerProduct<CombR, Chal, CombR, MBSInnerProduct, D>,
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
        CombR,
        CombR,
        ScalarProduct,
        ScalarProduct,
        MBSInnerProduct,
    >;

    type BinaryLc = IprsCode<Self::BinaryZt, PnttConfigF12289, REP, PERFORM_CHECKS>;
    type ArbitraryLc = IprsCode<Self::ArbitraryZt, PnttConfigF12289, REP, PERFORM_CHECKS>;
    type IntLc = IprsCode<Self::IntZt, PnttConfigF12289, REP, PERFORM_CHECKS>;
}

//
// Constants and concrete types
//

const DEGREE_PLUS_ONE: usize = 32;
const INT_LIMBS: usize = U64::LIMBS;
const FIELD_LIMBS: usize = U64::LIMBS * 3;

type F = MontyField<FIELD_LIMBS>;

type BenchZincTypes = GenericBenchZincTypes<
    /* Int = */ i64,
    /* CwR = */ i128,
    /* Chal = */ i128,
    /* Pt = */ i128,
    /* CombR = */ Int<{ INT_LIMBS * 6 }>,
    /* Fmod = */ Uint<FIELD_LIMBS>,
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
            IprsCode::new(poly_size, iprs_depth(poly_size)).unwrap(),
        ),
        ZipPlus::setup(
            poly_size,
            IprsCode::new(poly_size, iprs_depth(poly_size)).unwrap(),
        ),
        ZipPlus::setup(
            poly_size,
            IprsCode::new(poly_size, iprs_depth(poly_size)).unwrap(),
        ),
    )
}

// 192-bit (3 limbs) random prime field — matches the module-level `F` used
// by the Sha256Slice bench. Every ECDSA constraint is `assert_zero` and is
// skipped in the fq_sumcheck fold by the `ConstraintFolder::assert_zero`
// no-op, so the large intermediates from (e.g.) the scalar inverse
// `s · w − 1 − q_sw · n` never enter the combined polynomial. The
// non-zero-ideal constraints that *do* enter fq_sumcheck are SHA-side
// rotation/schedule constraints with small-integer coefficients, for
// which a narrower field is sufficient.
const ECDSA_BENCH_FIELD_LIMBS: usize = 3;
const ECDSA_BENCH_K: usize = ECDSA_INT_LIMBS * 2;
const ECDSA_BENCH_M: usize = ECDSA_INT_LIMBS * 4;

type EcdsaBenchInt = Int<ECDSA_INT_LIMBS>;
type EcdsaBenchF = MontyField<ECDSA_BENCH_FIELD_LIMBS>;

type EcdsaBenchZincTypes = GenericBenchZincTypes<
    /* Int = */ EcdsaBenchInt,
    /* CwR = */ Int<ECDSA_BENCH_K>,
    /* Chal = */ i128,
    /* Pt = */ i128,
    /* CombR = */ Int<ECDSA_BENCH_M>,
    /* Fmod = */ Uint<ECDSA_BENCH_FIELD_LIMBS>,
    MillerRabin,
    DEGREE_PLUS_ONE,
>;

#[allow(clippy::unwrap_used)]
fn setup_ecdsa_pp(num_vars: usize) -> Pp<EcdsaBenchZincTypes> {
    let poly_size = 1 << num_vars;
    (
        ZipPlus::setup(
            poly_size,
            IprsCode::new(poly_size, iprs_depth(poly_size)).unwrap(),
        ),
        ZipPlus::setup(
            poly_size,
            IprsCode::new(poly_size, iprs_depth(poly_size)).unwrap(),
        ),
        ZipPlus::setup(
            poly_size,
            IprsCode::new(poly_size, iprs_depth(poly_size)).unwrap(),
        ),
    )
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
    project_scalar: impl Fn(&U::Scalar, &<F as PrimeField>::Config) -> DynamicPolynomialF<F> + Copy,
    project_ideal: impl Fn(&IdealOrZero<U::Ideal>, &<F as PrimeField>::Config) -> IdealOverF + Copy,
) where
    Zt: ZincTypes<DEGREE_PLUS_ONE>,
    Zt::Int: ProjectableToField<F>,
    <Zt::BinaryZt as ZipTypes>::Cw: ProjectableToField<F>,
    <Zt::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<F>,
    <Zt::ArbitraryZt as ZipTypes>::Cw: ProjectableToField<F>,
    <Zt::IntZt as ZipTypes>::Cw: ProjectableToField<F>,
    F: FromWithConfig<Zt::Int>
        + for<'a> FromWithConfig<&'a Zt::CombR>
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
    let params = format!("{label}/nvars={num_vars}{RATE_TAG}");

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
    Zt::Int: ProjectableToField<F>,
    <Zt::BinaryZt as ZipTypes>::Cw: ProjectableToField<F>,
    <Zt::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<F>,
    <Zt::ArbitraryZt as ZipTypes>::Cw: ProjectableToField<F>,
    <Zt::IntZt as ZipTypes>::Cw: ProjectableToField<F>,
    F: FromWithConfig<Zt::Int>
        + for<'a> FromWithConfig<&'a Zt::CombR>
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
    let params = format!("{label}/nvars={num_vars}{RATE_TAG}");

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

// ---------------------------------------------------------------------------
// SHA-256 compression-slice benches. `Sha256CompressionSliceUair` uses the
// custom `Sha256Ideal` enum rather than `DegreeOneIdeal` that
// `do_bench_uair`/`do_bench_steps_uair` hard-code, so we wire the projection
// closure by hand and call `do_bench_e2e`/`do_bench_steps` directly.
// ---------------------------------------------------------------------------

fn sha256_slice_project_ideal(
    ideal: &IdealOrZero<Sha256Ideal<<BenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::Int>>,
    field_cfg: &<F as PrimeField>::Config,
) -> Sha256Ideal<F> {
    // Zero ideals are filtered out upstream of this closure (see
    // piop/src/ideal_check.rs), so we only receive NonZero.
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

fn bench_sha256_slice_e2e(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    type U = Sha256CompressionSliceUair<<BenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::Int>;

    let mut rng = rng();
    let trace = U::generate_random_trace(num_vars, &mut rng);
    let pp = setup_pp(num_vars);

    do_bench_e2e::<BenchZincTypes, U, _>(
        group,
        "Sha256Slice",
        num_vars,
        &pp,
        &trace,
        zinc_protocol::project_scalar_fn,
        sha256_slice_project_ideal,
    );
}

fn bench_sha256_slice_steps(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    type U = Sha256CompressionSliceUair<<BenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::Int>;

    let mut rng = rng();
    let trace = U::generate_random_trace(num_vars, &mut rng);
    let pp = setup_pp(num_vars);

    do_bench_steps::<BenchZincTypes, U, _>(
        group,
        "Sha256Slice",
        num_vars,
        &pp,
        &trace,
        zinc_protocol::project_scalar_fn,
        sha256_slice_project_ideal,
    );
}

// ---------------------------------------------------------------------------
// ECDSA scalar-slice benches (standalone — does not reuse do_bench_e2e /
// do_bench_steps_uair because `EcdsaScalarSliceUair` requires
// `EcdsaBenchZincTypes` with `Int<ECDSA_INT_LIMBS>` rather than `i64`).
// ---------------------------------------------------------------------------

#[allow(clippy::unwrap_used)]
fn bench_ecdsa_slice_e2e(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    type U = EcdsaScalarSliceUair<EcdsaBenchInt>;

    let mut rng = rng();
    let trace = U::generate_random_trace(num_vars, &mut rng);
    let pp = setup_ecdsa_pp(num_vars);
    let sig = U::signature();
    let public_trace = trace.public(&sig);

    let proj_ideal = |_ideal: &IdealOrZero<ImpossibleIdeal>,
                      _field_cfg: &<EcdsaBenchF as PrimeField>::Config|
     -> ImpossibleIdeal {
        unreachable!("ECDSA scalar slice uses only assert_zero; no non-trivial ideals")
    };

    let params = format!("EcdsaSlice/nvars={num_vars}{RATE_TAG}");

    group.bench_function(BenchmarkId::new("Prove (Combined)", &params), |bench| {
        bench.iter(|| {
            black_box(
                ZincPlusPiop::<EcdsaBenchZincTypes, U, EcdsaBenchF, DEGREE_PLUS_ONE>::prove::<
                    false,
                    PERFORM_CHECKS,
                >(&pp, &trace, num_vars, zinc_protocol::project_scalar_fn),
            )
            .expect("Prover failed");
        });
    });

    let proof: Proof<EcdsaBenchF> =
        ZincPlusPiop::<EcdsaBenchZincTypes, U, EcdsaBenchF, DEGREE_PLUS_ONE>::prove::<
            false,
            PERFORM_CHECKS,
        >(&pp, &trace, num_vars, zinc_protocol::project_scalar_fn)
        .expect("proof generation for verifier bench");

    group.bench_function(BenchmarkId::new("Verify", &params), |bench| {
        bench.iter_batched(
            || proof.clone(),
            |proof| {
                black_box(
                    ZincPlusPiop::<EcdsaBenchZincTypes, U, EcdsaBenchF, DEGREE_PLUS_ONE>::verify::<
                        _,
                        PERFORM_CHECKS,
                    >(
                        &pp,
                        proof,
                        &public_trace,
                        num_vars,
                        zinc_protocol::project_scalar_fn,
                        proj_ideal,
                    ),
                )
                .expect("Verifier failed");
            },
            BatchSize::SmallInput,
        );
    });

    eprint_proof_size(&params, &proof);
}

/// Per-step bench for ECDSA scalar slice. Mirrors `do_bench_steps` but with
/// `EcdsaBenchF` / `EcdsaBenchZincTypes` wired by hand. ECDSA max constraint
/// degree is 3 (from `b · (b − 1)` and `k · (k − 1)`), so the MLE-first
/// variants are skipped — they only apply when max degree ≤ 1.
#[allow(clippy::unwrap_used)]
fn bench_ecdsa_slice_steps(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    type U = EcdsaScalarSliceUair<EcdsaBenchInt>;

    let mut rng = rng();
    let trace = U::generate_random_trace(num_vars, &mut rng);
    let pp = setup_ecdsa_pp(num_vars);
    let sig = U::signature();
    let public_trace = trace.public(&sig);

    let proj_ideal = |_ideal: &IdealOrZero<ImpossibleIdeal>,
                      _field_cfg: &<EcdsaBenchF as PrimeField>::Config|
     -> ImpossibleIdeal {
        unreachable!("ECDSA scalar slice uses only assert_zero; no non-trivial ideals")
    };
    let project_scalar = zinc_protocol::project_scalar_fn;

    let params = format!("EcdsaSlice/nvars={num_vars}{RATE_TAG}");

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
            ZincPlusPiop::<EcdsaBenchZincTypes, U, EcdsaBenchF, DEGREE_PLUS_ONE>
        };
    }

    let p_committed = <piop!()>::step0_commit(&pp, &trace, num_vars).unwrap();
    let p_projected = p_committed.clone().step1_combined(project_scalar).unwrap();
    let p_ideal_checked = p_projected.clone().step2_ideal_check().unwrap();
    let p_eval_projected = p_ideal_checked.clone().step3_eval_projection().unwrap();
    let p_sumchecked = p_eval_projected.clone().step4_sumcheck().unwrap();
    let p_mp_evaled = p_sumchecked.clone().step5_multipoint_eval().unwrap();
    let p_lifted = p_mp_evaled.clone().step6_lift_and_project().unwrap();

    step_bench!(
        "Prove" / "0: Commit",
        setup = || {},
        run = |_s| <piop!()>::step0_commit(&pp, &trace, num_vars),
    );
    step_bench!(
        "Prove" / "1: Prime projection (Combined)",
        setup = || p_committed.clone(),
        run = |s| s.step1_combined(project_scalar),
    );
    step_bench!(
        "Prove" / "2: Ideal check (Combined)",
        setup = || p_projected.clone(),
        run = |s| s.step2_ideal_check(),
    );
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

    let proof: Proof<EcdsaBenchF> =
        <piop!()>::prove::<false, PERFORM_CHECKS>(&pp, &trace, num_vars, project_scalar)
            .expect("proof generation for verifier bench");

    let v_transcript = <piop!()>::step0_reconstruct_transcript::<ImpossibleIdeal>(
        &pp,
        proof.clone(),
        &public_trace,
        num_vars,
    )
    .unwrap();
    let v_prime_projected = v_transcript.clone().step1_prime_projection().unwrap();
    let v_ideal_checked = v_prime_projected.clone().step2_ideal_check(proj_ideal).unwrap();
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
        run = |proof| <piop!()>::step0_reconstruct_transcript::<ImpossibleIdeal>(
            &pp,
            proof,
            &public_trace,
            num_vars,
        ),
    );
    step_bench!(
        "Verify" / "1: Prime projection",
        setup = || v_transcript.clone(),
        run = |s| s.step1_prime_projection(),
    );
    step_bench!(
        "Verify" / "2: Ideal check",
        setup = || v_prime_projected.clone(),
        run = |s| s.step2_ideal_check(proj_ideal),
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

// ---------------------------------------------------------------------------
// Merged SHA-256 + ECDSA UAIR benches. Reuses `EcdsaBenchZincTypes` /
// `EcdsaBenchF` — same Int width and F width as the ECDSA-only slice.
// ---------------------------------------------------------------------------

/// Ideal projector for the merged UAIR. `ShaEcdsaUair` reuses
/// `Sha256Ideal<R>` as its ideal (ECDSA emits only `assert_zero`, so SHA's
/// ideal set is a superset).
fn sha_ecdsa_project_ideal(
    ideal: &IdealOrZero<Sha256Ideal<EcdsaBenchInt>>,
    field_cfg: &<EcdsaBenchF as PrimeField>::Config,
) -> Sha256Ideal<EcdsaBenchF> {
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

#[allow(clippy::unwrap_used)]
fn bench_sha_ecdsa_e2e(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    type U = ShaEcdsaUair<EcdsaBenchInt>;

    let mut rng = rng();
    let trace = U::generate_random_trace(num_vars, &mut rng);
    let pp = setup_ecdsa_pp(num_vars);
    let sig = U::signature();
    let public_trace = trace.public(&sig);

    let params = format!("ShaEcdsa/nvars={num_vars}{RATE_TAG}");

    group.bench_function(BenchmarkId::new("Prove (Combined)", &params), |bench| {
        bench.iter(|| {
            black_box(
                ZincPlusPiop::<EcdsaBenchZincTypes, U, EcdsaBenchF, DEGREE_PLUS_ONE>::prove::<
                    false,
                    PERFORM_CHECKS,
                >(&pp, &trace, num_vars, zinc_protocol::project_scalar_fn),
            )
            .expect("Prover failed");
        });
    });

    let proof: Proof<EcdsaBenchF> =
        ZincPlusPiop::<EcdsaBenchZincTypes, U, EcdsaBenchF, DEGREE_PLUS_ONE>::prove::<
            false,
            PERFORM_CHECKS,
        >(&pp, &trace, num_vars, zinc_protocol::project_scalar_fn)
        .expect("proof generation for verifier bench");

    group.bench_function(BenchmarkId::new("Verify", &params), |bench| {
        bench.iter_batched(
            || proof.clone(),
            |proof| {
                black_box(
                    ZincPlusPiop::<
                        EcdsaBenchZincTypes,
                        U,
                        EcdsaBenchF,
                        DEGREE_PLUS_ONE,
                    >::verify::<_, PERFORM_CHECKS>(
                        &pp,
                        proof,
                        &public_trace,
                        num_vars,
                        zinc_protocol::project_scalar_fn,
                        sha_ecdsa_project_ideal,
                    ),
                )
                .expect("Verifier failed");
            },
            BatchSize::SmallInput,
        );
    });

    eprint_proof_size(&params, &proof);
}

/// Per-step bench for the merged SHA+ECDSA UAIR. Mirrors
/// `bench_ecdsa_slice_steps` with `ShaEcdsaUair` in place of the scalar
/// slice and the `Sha256Ideal` projector wired in.
#[allow(clippy::unwrap_used)]
fn bench_sha_ecdsa_steps(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    type U = ShaEcdsaUair<EcdsaBenchInt>;

    let mut rng = rng();
    let trace = U::generate_random_trace(num_vars, &mut rng);
    let pp = setup_ecdsa_pp(num_vars);
    let sig = U::signature();
    let public_trace = trace.public(&sig);

    let project_scalar = zinc_protocol::project_scalar_fn;

    let params = format!("ShaEcdsa/nvars={num_vars}{RATE_TAG}");

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
            ZincPlusPiop::<EcdsaBenchZincTypes, U, EcdsaBenchF, DEGREE_PLUS_ONE>
        };
    }

    let p_committed = <piop!()>::step0_commit(&pp, &trace, num_vars).unwrap();
    let p_projected = p_committed.clone().step1_combined(project_scalar).unwrap();
    let p_ideal_checked = p_projected.clone().step2_ideal_check().unwrap();
    let p_eval_projected = p_ideal_checked.clone().step3_eval_projection().unwrap();
    let p_sumchecked = p_eval_projected.clone().step4_sumcheck().unwrap();
    let p_mp_evaled = p_sumchecked.clone().step5_multipoint_eval().unwrap();
    let p_lifted = p_mp_evaled.clone().step6_lift_and_project().unwrap();

    step_bench!(
        "Prove" / "0: Commit",
        setup = || {},
        run = |_s| <piop!()>::step0_commit(&pp, &trace, num_vars),
    );
    step_bench!(
        "Prove" / "1: Prime projection (Combined)",
        setup = || p_committed.clone(),
        run = |s| s.step1_combined(project_scalar),
    );
    step_bench!(
        "Prove" / "2: Ideal check (Combined)",
        setup = || p_projected.clone(),
        run = |s| s.step2_ideal_check(),
    );
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

    let proof: Proof<EcdsaBenchF> =
        <piop!()>::prove::<false, PERFORM_CHECKS>(&pp, &trace, num_vars, project_scalar)
            .expect("proof generation for verifier bench");

    let v_transcript = <piop!()>::step0_reconstruct_transcript::<Sha256Ideal<EcdsaBenchF>>(
        &pp,
        proof.clone(),
        &public_trace,
        num_vars,
    )
    .unwrap();
    let v_prime_projected = v_transcript.clone().step1_prime_projection().unwrap();
    let v_ideal_checked = v_prime_projected
        .clone()
        .step2_ideal_check(sha_ecdsa_project_ideal)
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
        run = |proof| <piop!()>::step0_reconstruct_transcript::<Sha256Ideal<EcdsaBenchF>>(
            &pp,
            proof,
            &public_trace,
            num_vars,
        ),
    );
    step_bench!(
        "Verify" / "1: Prime projection",
        setup = || v_transcript.clone(),
        run = |s| s.step1_prime_projection(),
    );
    step_bench!(
        "Verify" / "2: Ideal check",
        setup = || v_prime_projected.clone(),
        run = |s| s.step2_ideal_check(sha_ecdsa_project_ideal),
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

// TEMPORARY: prover-only bench of the selector-free `ShaEcdsaLinearizedUair`.
//
// Drops SHA's row-dependent selectors (s_sched_anch, s_upd_anch, s_init,
// s_final). This makes C7–C12 degree 1 instead of 2, effective max degree
// drops to 1, and the MLE-first path unlocks. The proof produced here
// does NOT verify — the dropped constraints fail on their formerly-inactive
// rows — so we only time the prover.
fn bench_sha_ecdsa_linearized_prover_only(
    group: &mut BenchmarkGroup<WallTime>,
    num_vars: usize,
) {
    type U = ShaEcdsaLinearizedUair<EcdsaBenchInt>;

    let mut rng = rng();
    let trace = U::generate_random_trace(num_vars, &mut rng);
    let pp = setup_ecdsa_pp(num_vars);

    let params = format!("ShaEcdsaLinearized/nvars={num_vars}{RATE_TAG}");

    group.bench_function(BenchmarkId::new("Prove (Combined)", &params), |bench| {
        bench.iter(|| {
            black_box(
                ZincPlusPiop::<EcdsaBenchZincTypes, U, EcdsaBenchF, DEGREE_PLUS_ONE>::prove::<
                    false,
                    PERFORM_CHECKS,
                >(&pp, &trace, num_vars, zinc_protocol::project_scalar_fn),
            )
            .expect("Prover failed");
        });
    });

    group.bench_function(BenchmarkId::new("Prove (MLE-first)", &params), |bench| {
        bench.iter(|| {
            black_box(
                ZincPlusPiop::<EcdsaBenchZincTypes, U, EcdsaBenchF, DEGREE_PLUS_ONE>::prove::<
                    true,
                    PERFORM_CHECKS,
                >(&pp, &trace, num_vars, zinc_protocol::project_scalar_fn),
            )
            .expect("Prover failed");
        });
    });
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

    bench_sha256_slice_e2e(&mut group, 9);
    bench_ecdsa_slice_e2e(&mut group, 9);
    bench_sha_ecdsa_e2e(&mut group, 9);
    bench_sha_ecdsa_linearized_prover_only(&mut group, 9);

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

    bench_sha256_slice_steps(&mut group, 9);
    bench_ecdsa_slice_steps(&mut group, 9);
    bench_sha_ecdsa_steps(&mut group, 9);

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

#[derive(Clone, Debug)]
struct BenchFoldedZincTypes;

impl FoldedZincTypes<DEGREE_PLUS_ONE, HALF_DEGREE_PLUS_ONE> for BenchFoldedZincTypes {
    type Int = i64;
    type Chal = i128;
    type Pt = i128;
    type CombR = Int<{ INT_LIMBS * 6 }>;
    type Fmod = Uint<FIELD_LIMBS>;
    type PrimeTest = MillerRabin;

    type BinaryZt = GenericBenchZipTypes<
        BinaryPoly<HALF_DEGREE_PLUS_ONE>,
        DensePolynomial<i64, HALF_DEGREE_PLUS_ONE>,
        Self::Fmod,
        Self::PrimeTest,
        Self::Chal,
        Self::Pt,
        Self::CombR,
        DensePolynomial<Self::CombR, HALF_DEGREE_PLUS_ONE>,
        BinaryPolyInnerProduct<Self::Chal, HALF_DEGREE_PLUS_ONE>,
        DensePolyInnerProduct<
            Self::CombR,
            Self::Chal,
            Self::CombR,
            MBSInnerProduct,
            HALF_DEGREE_PLUS_ONE,
        >,
        MBSInnerProduct,
    >;

    type ArbitraryZt = <BenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::ArbitraryZt;
    type IntZt = <BenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::IntZt;

    type BinaryLc = IprsCode<Self::BinaryZt, PnttConfigF12289, REP, PERFORM_CHECKS>;
    type ArbitraryLc = <BenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::ArbitraryLc;
    type IntLc = <BenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::IntLc;
}

type FoldedPp = (
    ZipPlusParams<
        <BenchFoldedZincTypes as FoldedZincTypes<
            DEGREE_PLUS_ONE,
            HALF_DEGREE_PLUS_ONE,
        >>::BinaryZt,
        <BenchFoldedZincTypes as FoldedZincTypes<
            DEGREE_PLUS_ONE,
            HALF_DEGREE_PLUS_ONE,
        >>::BinaryLc,
    >,
    ZipPlusParams<
        <BenchFoldedZincTypes as FoldedZincTypes<
            DEGREE_PLUS_ONE,
            HALF_DEGREE_PLUS_ONE,
        >>::ArbitraryZt,
        <BenchFoldedZincTypes as FoldedZincTypes<
            DEGREE_PLUS_ONE,
            HALF_DEGREE_PLUS_ONE,
        >>::ArbitraryLc,
    >,
    ZipPlusParams<
        <BenchFoldedZincTypes as FoldedZincTypes<
            DEGREE_PLUS_ONE,
            HALF_DEGREE_PLUS_ONE,
        >>::IntZt,
        <BenchFoldedZincTypes as FoldedZincTypes<
            DEGREE_PLUS_ONE,
            HALF_DEGREE_PLUS_ONE,
        >>::IntLc,
    >,
);

#[allow(clippy::unwrap_used)]
fn setup_folded_pp(num_vars: usize) -> FoldedPp {
    // Binary commitment is over the split column (length 2n with
    // BinaryPoly<HALF_D> entries), so its num_vars is num_vars + 1.
    let split_size = 1 << (num_vars + 1);
    let normal_size = 1 << num_vars;
    (
        ZipPlus::setup(
            split_size,
            IprsCode::new(split_size, iprs_depth(split_size)).unwrap(),
        ),
        ZipPlus::setup(
            normal_size,
            IprsCode::new(normal_size, iprs_depth(normal_size)).unwrap(),
        ),
        ZipPlus::setup(
            normal_size,
            IprsCode::new(normal_size, iprs_depth(normal_size)).unwrap(),
        ),
    )
}

#[allow(clippy::too_many_arguments)]
fn do_bench_e2e_folded<U, IdealOverF>(
    group: &mut BenchmarkGroup<WallTime>,
    label: &str,
    num_vars: usize,
    pp: &FoldedPp,
    trace: &UairTrace<'static, i64, i64, DEGREE_PLUS_ONE>,
    project_scalar: impl Fn(&U::Scalar, &<F as PrimeField>::Config) -> DynamicPolynomialF<F>
    + Copy
    + Sync,
    project_ideal: impl Fn(&IdealOrZero<U::Ideal>, &<F as PrimeField>::Config) -> IdealOverF + Copy,
) where
    U: Uair + 'static,
    IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<F>>,
{
    let params = format!("{label}/nvars={num_vars}{RATE_TAG}");

    group.bench_function(BenchmarkId::new("Prove (folded)", &params), |bench| {
        bench.iter(|| {
            black_box(zinc_protocol::prover::prove_folded::<
                BenchFoldedZincTypes,
                U,
                F,
                DEGREE_PLUS_ONE,
                HALF_DEGREE_PLUS_ONE,
                PERFORM_CHECKS,
            >(pp, trace, num_vars, project_scalar))
            .expect("Folded prover failed");
        });
    });

    let proof: Proof<F> = zinc_protocol::prover::prove_folded::<
        BenchFoldedZincTypes,
        U,
        F,
        DEGREE_PLUS_ONE,
        HALF_DEGREE_PLUS_ONE,
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
                    BenchFoldedZincTypes,
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

#[allow(clippy::unwrap_used)]
fn do_bench_uair_folded<U>(group: &mut BenchmarkGroup<WallTime>, label: &str, num_vars: usize)
where
    U: Uair<
            Ideal = DegreeOneIdeal<i64>,
            Scalar = DensePolynomial<i64, DEGREE_PLUS_ONE>,
        > + GenerateRandomTrace<DEGREE_PLUS_ONE, PolyCoeff = i64, Int = i64>
        + 'static,
{
    let mut rng = rng();
    let trace = U::generate_random_trace(num_vars, &mut rng);
    let pp = setup_folded_pp(num_vars);

    let proj_ideal = |ideal: &IdealOrZero<U::Ideal>, field_cfg: &<F as PrimeField>::Config| {
        ideal.map(|i| DegreeOneIdeal::from_with_cfg(i, field_cfg))
    };

    do_bench_e2e_folded::<U, _>(
        group,
        label,
        num_vars,
        &pp,
        &trace,
        zinc_protocol::project_scalar_fn,
        proj_ideal,
    );
}

fn bench_sha_proxy_e2e_folded(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    do_bench_uair_folded::<ShaProxy<i64>>(group, "ShaProxy", num_vars);
}

// ---------------------------------------------------------------------------
// SHA-256 compression-slice folded bench. Reuses `BenchFoldedZincTypes`
// (i64-based) — same field as `bench_sha256_slice_e2e`.
// ---------------------------------------------------------------------------

fn bench_sha256_slice_e2e_folded(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    type U = Sha256CompressionSliceUair<i64>;

    let mut rng = rng();
    let trace = U::generate_random_trace(num_vars, &mut rng);
    let pp = setup_folded_pp(num_vars);

    do_bench_e2e_folded::<U, _>(
        group,
        "Sha256Slice",
        num_vars,
        &pp,
        &trace,
        zinc_protocol::project_scalar_fn,
        sha256_slice_project_ideal,
    );
}

// ---------------------------------------------------------------------------
// ECDSA + SHA+ECDSA folded benches. ECDSA uses `EcdsaBenchZincTypes` with
// `Int<ECDSA_INT_LIMBS>` instead of `i64`, so we need a parallel
// `FoldedZincTypes` impl whose ArbitraryZt/IntZt match `EcdsaBenchZincTypes`.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct EcdsaBenchFoldedZincTypes;

impl FoldedZincTypes<DEGREE_PLUS_ONE, HALF_DEGREE_PLUS_ONE> for EcdsaBenchFoldedZincTypes {
    type Int = EcdsaBenchInt;
    type Chal = i128;
    type Pt = i128;
    type CombR = Int<ECDSA_BENCH_M>;
    type Fmod = Uint<ECDSA_BENCH_FIELD_LIMBS>;
    type PrimeTest = MillerRabin;

    type BinaryZt = GenericBenchZipTypes<
        BinaryPoly<HALF_DEGREE_PLUS_ONE>,
        DensePolynomial<i64, HALF_DEGREE_PLUS_ONE>,
        Self::Fmod,
        Self::PrimeTest,
        Self::Chal,
        Self::Pt,
        Self::CombR,
        DensePolynomial<Self::CombR, HALF_DEGREE_PLUS_ONE>,
        BinaryPolyInnerProduct<Self::Chal, HALF_DEGREE_PLUS_ONE>,
        DensePolyInnerProduct<
            Self::CombR,
            Self::Chal,
            Self::CombR,
            MBSInnerProduct,
            HALF_DEGREE_PLUS_ONE,
        >,
        MBSInnerProduct,
    >;

    type ArbitraryZt = <EcdsaBenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::ArbitraryZt;
    type IntZt = <EcdsaBenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::IntZt;

    type BinaryLc = IprsCode<Self::BinaryZt, PnttConfigF12289, REP, PERFORM_CHECKS>;
    type ArbitraryLc = <EcdsaBenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::ArbitraryLc;
    type IntLc = <EcdsaBenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::IntLc;
}

type EcdsaFoldedPp = (
    ZipPlusParams<
        <EcdsaBenchFoldedZincTypes as FoldedZincTypes<
            DEGREE_PLUS_ONE,
            HALF_DEGREE_PLUS_ONE,
        >>::BinaryZt,
        <EcdsaBenchFoldedZincTypes as FoldedZincTypes<
            DEGREE_PLUS_ONE,
            HALF_DEGREE_PLUS_ONE,
        >>::BinaryLc,
    >,
    ZipPlusParams<
        <EcdsaBenchFoldedZincTypes as FoldedZincTypes<
            DEGREE_PLUS_ONE,
            HALF_DEGREE_PLUS_ONE,
        >>::ArbitraryZt,
        <EcdsaBenchFoldedZincTypes as FoldedZincTypes<
            DEGREE_PLUS_ONE,
            HALF_DEGREE_PLUS_ONE,
        >>::ArbitraryLc,
    >,
    ZipPlusParams<
        <EcdsaBenchFoldedZincTypes as FoldedZincTypes<
            DEGREE_PLUS_ONE,
            HALF_DEGREE_PLUS_ONE,
        >>::IntZt,
        <EcdsaBenchFoldedZincTypes as FoldedZincTypes<
            DEGREE_PLUS_ONE,
            HALF_DEGREE_PLUS_ONE,
        >>::IntLc,
    >,
);

#[allow(clippy::unwrap_used)]
fn setup_ecdsa_folded_pp(num_vars: usize) -> EcdsaFoldedPp {
    let split_size = 1 << (num_vars + 1);
    let normal_size = 1 << num_vars;
    (
        ZipPlus::setup(
            split_size,
            IprsCode::new(split_size, iprs_depth(split_size)).unwrap(),
        ),
        ZipPlus::setup(
            normal_size,
            IprsCode::new(normal_size, iprs_depth(normal_size)).unwrap(),
        ),
        ZipPlus::setup(
            normal_size,
            IprsCode::new(normal_size, iprs_depth(normal_size)).unwrap(),
        ),
    )
}

#[allow(clippy::unwrap_used)]
fn bench_ecdsa_slice_e2e_folded(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    type U = EcdsaScalarSliceUair<EcdsaBenchInt>;

    let mut rng = rng();
    let trace = U::generate_random_trace(num_vars, &mut rng);
    let pp = setup_ecdsa_folded_pp(num_vars);
    let sig = U::signature();
    let public_trace = trace.public(&sig);

    let proj_ideal = |_ideal: &IdealOrZero<ImpossibleIdeal>,
                      _field_cfg: &<EcdsaBenchF as PrimeField>::Config|
     -> ImpossibleIdeal {
        unreachable!("ECDSA scalar slice uses only assert_zero; no non-trivial ideals")
    };

    let params = format!("EcdsaSlice/nvars={num_vars}{RATE_TAG}");

    group.bench_function(BenchmarkId::new("Prove (folded)", &params), |bench| {
        bench.iter(|| {
            black_box(zinc_protocol::prover::prove_folded::<
                EcdsaBenchFoldedZincTypes,
                U,
                EcdsaBenchF,
                DEGREE_PLUS_ONE,
                HALF_DEGREE_PLUS_ONE,
                PERFORM_CHECKS,
            >(&pp, &trace, num_vars, zinc_protocol::project_scalar_fn))
            .expect("Folded prover failed");
        });
    });

    let proof: Proof<EcdsaBenchF> = zinc_protocol::prover::prove_folded::<
        EcdsaBenchFoldedZincTypes,
        U,
        EcdsaBenchF,
        DEGREE_PLUS_ONE,
        HALF_DEGREE_PLUS_ONE,
        PERFORM_CHECKS,
    >(&pp, &trace, num_vars, zinc_protocol::project_scalar_fn)
    .expect("proof generation for folded verifier bench");

    group.bench_function(BenchmarkId::new("Verify (folded)", &params), |bench| {
        bench.iter_batched(
            || proof.clone(),
            |proof| {
                black_box(zinc_protocol::verifier::verify_folded::<
                    EcdsaBenchFoldedZincTypes,
                    U,
                    EcdsaBenchF,
                    ImpossibleIdeal,
                    DEGREE_PLUS_ONE,
                    HALF_DEGREE_PLUS_ONE,
                    PERFORM_CHECKS,
                >(
                    &pp,
                    proof,
                    &public_trace,
                    num_vars,
                    zinc_protocol::project_scalar_fn,
                    proj_ideal,
                ))
                .expect("Folded verifier failed");
            },
            BatchSize::SmallInput,
        );
    });

    eprint_proof_size(&params, &proof);
}

#[allow(clippy::unwrap_used)]
fn bench_sha_ecdsa_e2e_folded(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    type U = ShaEcdsaUair<EcdsaBenchInt>;

    let mut rng = rng();
    let trace = U::generate_random_trace(num_vars, &mut rng);
    let pp = setup_ecdsa_folded_pp(num_vars);
    let sig = U::signature();
    let public_trace = trace.public(&sig);

    let params = format!("ShaEcdsa/nvars={num_vars}{RATE_TAG}");

    group.bench_function(BenchmarkId::new("Prove (folded)", &params), |bench| {
        bench.iter(|| {
            black_box(zinc_protocol::prover::prove_folded::<
                EcdsaBenchFoldedZincTypes,
                U,
                EcdsaBenchF,
                DEGREE_PLUS_ONE,
                HALF_DEGREE_PLUS_ONE,
                PERFORM_CHECKS,
            >(&pp, &trace, num_vars, zinc_protocol::project_scalar_fn))
            .expect("Folded prover failed");
        });
    });

    let proof: Proof<EcdsaBenchF> = zinc_protocol::prover::prove_folded::<
        EcdsaBenchFoldedZincTypes,
        U,
        EcdsaBenchF,
        DEGREE_PLUS_ONE,
        HALF_DEGREE_PLUS_ONE,
        PERFORM_CHECKS,
    >(&pp, &trace, num_vars, zinc_protocol::project_scalar_fn)
    .expect("proof generation for folded verifier bench");

    group.bench_function(BenchmarkId::new("Verify (folded)", &params), |bench| {
        bench.iter_batched(
            || proof.clone(),
            |proof| {
                black_box(zinc_protocol::verifier::verify_folded::<
                    EcdsaBenchFoldedZincTypes,
                    U,
                    EcdsaBenchF,
                    Sha256Ideal<EcdsaBenchF>,
                    DEGREE_PLUS_ONE,
                    HALF_DEGREE_PLUS_ONE,
                    PERFORM_CHECKS,
                >(
                    &pp,
                    proof,
                    &public_trace,
                    num_vars,
                    zinc_protocol::project_scalar_fn,
                    sha_ecdsa_project_ideal,
                ))
                .expect("Folded verifier failed");
            },
            BatchSize::SmallInput,
        );
    });

    eprint_proof_size(&params, &proof);
}

fn e2e_folded_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zinc+ E2E Folded");

    bench_sha256_slice_e2e_folded(&mut group, 9);
    bench_ecdsa_slice_e2e_folded(&mut group, 9);
    bench_sha_ecdsa_e2e_folded(&mut group, 9);

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
criterion_main!(e2e, e2e_steps, e2e_folded);
