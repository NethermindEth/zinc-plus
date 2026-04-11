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
use std::{hint::black_box, marker::PhantomData, ops::Neg, time::Duration};
use zinc_poly::{
    ConstCoeffBitWidth, Polynomial,
    univariate::{
        binary::{BinaryPoly, BinaryPolyInnerProduct},
        dense::{DensePolyInnerProduct, DensePolynomial},
        dynamic::over_field::DynamicPolynomialF,
    },
};
use zinc_primality::{MillerRabin, PrimalityTest};
use zinc_protocol::{Proof, StepTimings, ZincPlusPiop, ZincTypes};
use zinc_test_uair::{
    BigLinearUair, SHAProxy, BigLinearUairWithPublicInput, BinaryDecompositionUair, EcdsaUair,
    EcdsaUairLimbs, GenerateRandomTrace, ShaProxyEcdsaUair, TestAirNoMultiplication,
};
use zinc_transcript::traits::ConstTranscribable;
use zinc_uair::{
    Uair, UairTrace,
    degree_counter::count_effective_max_degree,
    ideal::{Ideal, IdealCheck, degree_one::DegreeOneIdeal, mixed::MixedDegreeOneOrXnMinusOne},
    ideal_collector::IdealOrZero,
};
use zinc_utils::{
    from_ref::FromRef,
    inner_product::{InnerProduct, MBSInnerProduct, ScalarProduct},
    mul_by_scalar::MulByScalar,
    named::Named,
    projectable_to_field::ProjectableToField,
};
#[cfg(feature = "bench-rate-1-16")]
use zip_plus::code::iprs::PnttConfigF65537;
#[cfg(not(feature = "bench-rate-1-16"))]
use zip_plus::code::iprs::PnttConfigF12289;
use zip_plus::{
    code::iprs::IprsCode,
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
/// Enable `bench-rate-1-8` or `bench-rate-1-16` to switch from the default
/// rate 1/4 to rate 1/8 or rate 1/16. `bench-rate-1-16` takes precedence
/// if both are set.
#[cfg(feature = "bench-rate-1-16")]
const REP: usize = 16;
#[cfg(all(feature = "bench-rate-1-8", not(feature = "bench-rate-1-16")))]
const REP: usize = 8;
#[cfg(not(any(feature = "bench-rate-1-8", feature = "bench-rate-1-16")))]
const REP: usize = 4;

const RATE_TAG: &str = match REP {
    16 => "/rate=1_16",
    8 => "/rate=1_8",
    _ => "/rate=1_4",
};

/// For higher inverse rates (1/8, 1/16) we bump the IPRS tree depth by one
/// beyond `new_with_optimal_depth`'s default heuristic. Enable
/// `bench-extra-depth-rate-1-4` to force the same +1 on rate 1/4.
#[cfg(feature = "bench-extra-depth-rate-1-4")]
const EXTRA_DEPTH: usize = 1;
#[cfg(not(feature = "bench-extra-depth-rate-1-4"))]
const EXTRA_DEPTH: usize = if REP >= 8 { 1 } else { 0 };

/// Base field for the IPRS PNTT. Rates 1/4 and 1/8 use F12289; rate 1/16
/// stays on F65537 since its codewords would overflow F12289 for bench
/// polynomial sizes.
#[cfg(not(feature = "bench-rate-1-16"))]
type BenchPnttConfig = PnttConfigF12289;
#[cfg(feature = "bench-rate-1-16")]
type BenchPnttConfig = PnttConfigF65537;

/// Mirrors `IprsCode::new_with_optimal_depth`'s formula, then adds
/// `EXTRA_DEPTH`. Kept in sync with zip-plus/src/code/iprs.rs.
fn iprs_depth(row_len: usize) -> usize {
    const MAX_BASE_COLS_LOG2: usize = 7;
    let target_base_len = 1usize << MAX_BASE_COLS_LOG2;
    let base = 1.max(((1.max(row_len / target_base_len)).ilog2() as usize).div_ceil(3));
    base + EXTRA_DEPTH
}

/// Names and selectors for the eight per-step prover sub-benches used by all
/// "Prove (Combined)" benches in this file. Each entry is
/// `(bench_label, |&StepTimings| -> Duration)`; the label is passed to
/// `BenchmarkId::new` and the closure picks that step's field out of the
/// `StepTimings` returned by `prove_with_step_timings`.
const STEP_PICKS: [(&str, fn(&StepTimings) -> Duration); 8] = [
    ("Step 0: Commit", |t| t.commit),
    ("Step 1: Prime Projection", |t| t.prime_projection),
    ("Step 2: Ideal Check", |t| t.ideal_check),
    ("Step 3: Eval Projection", |t| t.eval_projection),
    ("Step 4: Fq Sumcheck", |t| t.fq_sumcheck),
    ("Step 5: Multipoint Eval", |t| t.multipoint_eval),
    ("Step 6: Lift-and-Project", |t| t.lift_and_project),
    ("Step 7: PCS Open", |t| t.pcs_open),
];

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
    Eval: Named + ConstCoeffBitWidth + Default + Clone + Send + Sync,
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
    EvalDotChal: InnerProduct<Eval, Chal, CombR> + Send + Sync,
    CombDotChal: InnerProduct<Comb, Chal, CombR> + Send + Sync,
    ArrCombRDotChal: InnerProduct<[CombR], Chal, CombR> + Send + Sync,
{
    const NUM_COLUMN_OPENINGS: usize = match REP {
        16 => 72,
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
    PrimeTest: PrimalityTest<Fmod> + Send + Sync,
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

    type BinaryLc = IprsCode<Self::BinaryZt, BenchPnttConfig, REP, PERFORM_CHECKS>;
    type ArbitraryLc = IprsCode<Self::ArbitraryZt, BenchPnttConfig, REP, PERFORM_CHECKS>;
    type IntLc = IprsCode<Self::IntZt, BenchPnttConfig, REP, PERFORM_CHECKS>;
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

//
// Main benchmarking routine
//

#[allow(clippy::too_many_arguments)]
fn do_bench<Zt, U, IdealOverF>(
    group: &mut BenchmarkGroup<WallTime>,
    label: &str,
    num_vars: usize,
    setup: impl Fn(usize) -> Pp<Zt>,
    trace: UairTrace<'static, Zt::Int, Zt::Int, DEGREE_PLUS_ONE>,
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
    let pp = setup(num_vars);
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
                        &pp,
                        &trace,
                        num_vars,
                        project_scalar,
                    ))
                    .expect("Prover failed");
                });
            });
        };
    }

    // Replaces the single "Prove (Combined)" bench with eight per-step
    // sub-benches. Each sample still runs the full prover once; only the
    // targeted step's duration is accumulated.
    //
    // Effective max degree excludes zero-ideal (assert_zero) constraints,
    // which are skipped in the fq_sumcheck fold and discarded in
    // prove_linear. So MLE-first is valid for any UAIR whose *non*-zero-
    // ideal constraints are all linear, even if some assert_zero
    // constraints have higher degree (e.g. ShaEcdsa).
    let use_mle_first = count_effective_max_degree::<U>() <= 1;
    for (name, pick) in STEP_PICKS {
        group.bench_function(BenchmarkId::new(name, &params), |bench| {
            bench.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let (proof, timings) = if use_mle_first {
                        <zinc_plus!()>::prove_with_step_timings::<true, PERFORM_CHECKS>(
                            &pp,
                            &trace,
                            num_vars,
                            project_scalar,
                        )
                    } else {
                        <zinc_plus!()>::prove_with_step_timings::<false, PERFORM_CHECKS>(
                            &pp,
                            &trace,
                            num_vars,
                            project_scalar,
                        )
                    }
                    .expect("Prover failed");
                    total += pick(&timings);
                    black_box(proof);
                }
                total
            });
        });
    }

    if use_mle_first {
        bench_prove!("Prove (E2E)", true);
    } else {
        bench_prove!("Prove (E2E)", false);
    }

    let proof: Proof<F> =
        <zinc_plus!()>::prove::<false, PERFORM_CHECKS>(&pp, &trace, num_vars, project_scalar)
            .expect("proof generation for verifier bench");

    let sig = U::signature();
    let public_trace = trace.public(&sig);

    group.bench_function(BenchmarkId::new("Verify", &params), |bench| {
        bench.iter_batched(
            || proof.clone(),
            |proof| {
                black_box(<zinc_plus!()>::verify::<_, PERFORM_CHECKS>(
                    &pp,
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
// Specific benchmarks for each AIR
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

    let proj_ideal = |ideal: &IdealOrZero<U::Ideal>, field_cfg: &<F as PrimeField>::Config| {
        ideal.map(|i| DegreeOneIdeal::from_with_cfg(i, field_cfg))
    };

    do_bench::<BenchZincTypes, U, _>(
        group,
        label,
        num_vars,
        setup_pp,
        trace,
        zinc_protocol::project_scalar_fn,
        proj_ideal,
    );
}

fn bench_no_mult(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    do_bench_uair::<TestAirNoMultiplication<i64>>(group, "NoMult", num_vars);
}

fn bench_binary_decomposition(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    do_bench_uair::<BinaryDecompositionUair<i64>>(group, "BinaryDecomposition", num_vars);
}

fn bench_big_linear(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    do_bench_uair::<BigLinearUair<i64>>(group, "BigLinear", num_vars);
}

/// SHAProxy uses the mixed `(X-2) | (X^32 - 1)` ideal type, so it can't go
/// through `do_bench_uair` (which is hardcoded to `DegreeOneIdeal`). This
/// helper mirrors `do_bench_uair` but with the mixed-ideal projector.
fn do_bench_sha_proxy<U>(group: &mut BenchmarkGroup<WallTime>, label: &str, num_vars: usize)
where
    U: Uair<
            Ideal = MixedDegreeOneOrXnMinusOne<<BenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::Int, 32>,
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

    let proj_ideal = |ideal: &IdealOrZero<U::Ideal>, field_cfg: &<F as PrimeField>::Config| {
        ideal.map(|i| MixedDegreeOneOrXnMinusOne::<F, 32>::from_with_cfg(i, field_cfg))
    };

    do_bench::<BenchZincTypes, U, _>(
        group,
        label,
        num_vars,
        setup_pp,
        trace,
        zinc_protocol::project_scalar_fn,
        proj_ideal,
    );
}

fn bench_sha_proxy(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    do_bench_sha_proxy::<SHAProxy<i64>>(group, "SHAProxy", num_vars);
}

fn bench_big_linear_public_input(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    do_bench_uair::<BigLinearUairWithPublicInput<i64>>(group, "BigLinearPI", num_vars);
}

//
// ECDSA prover-only bench
//
// `EcdsaUair`'s trace cells are `Int<4>` (256-bit signed). The witness is a
// real Shamir's-trick walk over secp256k1, but it intentionally does not
// satisfy the integer-equality constraints (the field arithmetic is mod p).
// Therefore we **only** time the prover here — running the verifier on this
// trace would correctly reject it. This bench exists to measure prover
// throughput on the realistic ECDSA constraint shape (9 constraints, max
// degree 13).
//

type EcdsaInt = Int<{ INT_LIMBS * 4 }>;

type EcdsaBenchZincTypes = GenericBenchZincTypes<
    /* Int = */ EcdsaInt,
    /* CwR = */ Int<{ INT_LIMBS * 8 }>,
    /* Chal = */ i128,
    /* Pt = */ i128,
    /* CombR = */ Int<{ INT_LIMBS * 16 }>,
    /* Fmod = */ Uint<FIELD_LIMBS>,
    MillerRabin,
    DEGREE_PLUS_ONE,
>;

#[allow(clippy::unwrap_used)]
fn setup_pp_ecdsa(num_vars: usize) -> Pp<EcdsaBenchZincTypes> {
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

fn bench_ecdsa(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    let mut rng = rng();
    let trace = EcdsaUair::generate_random_trace(num_vars, &mut rng);
    let pp = setup_pp_ecdsa(num_vars);
    let params = format!("ECDSA/nvars={num_vars}{RATE_TAG}");

    // Every ECDSA constraint is asserted via `assert_zero`, so its
    // effective max degree (over non-zero-ideal constraints) is 0, and
    // the MLE-first path is valid. See `do_bench` for the same gating.
    let use_mle_first = count_effective_max_degree::<EcdsaUair>() <= 1;

    macro_rules! bench_prove_ecdsa {
        ($label:literal, $mle_first:expr) => {
            group.bench_function(BenchmarkId::new($label, &params), |bench| {
                bench.iter(|| {
                    black_box(ZincPlusPiop::<
                        EcdsaBenchZincTypes,
                        EcdsaUair,
                        F,
                        DEGREE_PLUS_ONE,
                    >::prove::<{ $mle_first }, PERFORM_CHECKS>(
                        &pp,
                        &trace,
                        num_vars,
                        zinc_protocol::project_scalar_fn,
                    ))
                    .expect("Prover failed");
                });
            });
        };
    }
    if use_mle_first {
        bench_prove_ecdsa!("Prove (E2E)", true);
    } else {
        bench_prove_ecdsa!("Prove (E2E)", false);
    }

    let proof: Proof<F> =
        ZincPlusPiop::<EcdsaBenchZincTypes, EcdsaUair, F, DEGREE_PLUS_ONE>::prove::<
            false,
            PERFORM_CHECKS,
        >(&pp, &trace, num_vars, zinc_protocol::project_scalar_fn)
        .expect("proof generation for size logging");
    eprint_proof_size(&params, &proof);

    // Eight per-step sub-benches replacing the single "Prove (Combined)"
    // bench. See STEP_PICKS for the step layout and trade-off notes.
    for (name, pick) in STEP_PICKS {
        group.bench_function(BenchmarkId::new(name, &params), |bench| {
            bench.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let (proof, timings) = if use_mle_first {
                        ZincPlusPiop::<
                            EcdsaBenchZincTypes,
                            EcdsaUair,
                            F,
                            DEGREE_PLUS_ONE,
                        >::prove_with_step_timings::<true, PERFORM_CHECKS>(
                            &pp,
                            &trace,
                            num_vars,
                            zinc_protocol::project_scalar_fn,
                        )
                    } else {
                        ZincPlusPiop::<
                            EcdsaBenchZincTypes,
                            EcdsaUair,
                            F,
                            DEGREE_PLUS_ONE,
                        >::prove_with_step_timings::<false, PERFORM_CHECKS>(
                            &pp,
                            &trace,
                            num_vars,
                            zinc_protocol::project_scalar_fn,
                        )
                    }
                    .expect("Prover failed");
                    total += pick(&timings);
                    black_box(proof);
                }
                total
            });
        });
    }
}

fn bench_sha_ecdsa(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    let mut rng = rng();
    let trace = ShaProxyEcdsaUair::generate_random_trace(num_vars, &mut rng);
    let pp = setup_pp_ecdsa(num_vars);
    let params = format!("ShaEcdsa/nvars={num_vars}{RATE_TAG}");

    // Raw max_degree = 5 (from ECDSA's deepest assert_zero), but every
    // ECDSA constraint is asserted via `assert_zero` and so skipped in
    // the fq_sumcheck fold. The only surviving constraints are SHAProxy's
    // linear ideal-check ones, so effective max degree = 1 and MLE-first
    // is valid.
    let use_mle_first = count_effective_max_degree::<ShaProxyEcdsaUair>() <= 1;

    macro_rules! bench_prove_sha_ecdsa {
        ($label:literal, $mle_first:expr) => {
            group.bench_function(BenchmarkId::new($label, &params), |bench| {
                bench.iter(|| {
                    black_box(ZincPlusPiop::<
                        EcdsaBenchZincTypes,
                        ShaProxyEcdsaUair,
                        F,
                        DEGREE_PLUS_ONE,
                    >::prove::<{ $mle_first }, PERFORM_CHECKS>(
                        &pp,
                        &trace,
                        num_vars,
                        zinc_protocol::project_scalar_fn,
                    ))
                    .expect("Prover failed");
                });
            });
        };
    }
    if use_mle_first {
        bench_prove_sha_ecdsa!("Prove (E2E)", true);
    } else {
        bench_prove_sha_ecdsa!("Prove (E2E)", false);
    }

    let proof: Proof<F> =
        ZincPlusPiop::<EcdsaBenchZincTypes, ShaProxyEcdsaUair, F, DEGREE_PLUS_ONE>::prove::<
            false,
            PERFORM_CHECKS,
        >(&pp, &trace, num_vars, zinc_protocol::project_scalar_fn)
        .expect("proof generation for size logging");
    eprint_proof_size(&params, &proof);

    // Every sample runs the full prover once; only the targeted step's
    // duration is accumulated into Criterion's measurement, so this bench is
    // ~8x more expensive than a single monolithic "Prove (Combined)" sample.
    for (name, pick) in STEP_PICKS {
        group.bench_function(BenchmarkId::new(name, &params), |bench| {
            bench.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let (_proof, timings) = if use_mle_first {
                        ZincPlusPiop::<
                            EcdsaBenchZincTypes,
                            ShaProxyEcdsaUair,
                            F,
                            DEGREE_PLUS_ONE,
                        >::prove_with_step_timings::<true, PERFORM_CHECKS>(
                            &pp,
                            &trace,
                            num_vars,
                            zinc_protocol::project_scalar_fn,
                        )
                    } else {
                        ZincPlusPiop::<
                            EcdsaBenchZincTypes,
                            ShaProxyEcdsaUair,
                            F,
                            DEGREE_PLUS_ONE,
                        >::prove_with_step_timings::<false, PERFORM_CHECKS>(
                            &pp,
                            &trace,
                            num_vars,
                            zinc_protocol::project_scalar_fn,
                        )
                    }
                    .expect("Prover failed");
                    total += pick(&timings);
                    black_box(_proof);
                }
                total
            });
        });
    }
}

// Prover-only bench for `EcdsaUairLimbs`. Cell type is i64, matching the
// existing `BenchZincTypes`, so we reuse `setup_pp` directly. The witness
// is non-satisfying (inherits the same caveat as `EcdsaUair`), so the
// verifier is intentionally not run.
fn bench_ecdsa_limbs(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    let mut rng = rng();
    let trace = EcdsaUairLimbs::generate_random_trace(num_vars, &mut rng);
    let pp = setup_pp(num_vars);
    let params = format!("EcdsaLimbs/nvars={num_vars}{RATE_TAG}");

    group.bench_function(BenchmarkId::new("Prove (Combined)", &params), |bench| {
        bench.iter(|| {
            black_box(
                ZincPlusPiop::<BenchZincTypes, EcdsaUairLimbs, F, DEGREE_PLUS_ONE>::prove::<
                    false,
                    PERFORM_CHECKS,
                >(
                    &pp,
                    &trace,
                    num_vars,
                    zinc_protocol::project_scalar_fn,
                ),
            )
            .expect("Prover failed");
        });
    });
}

//
// Criterion entry point
//

fn e2e_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zinc+ E2E");

    // bench_no_mult(&mut group, 8);
    // bench_no_mult(&mut group, 10);
    // bench_no_mult(&mut group, 12);

    // bench_binary_decomposition(&mut group, 8);
    // bench_binary_decomposition(&mut group, 10);
    // bench_binary_decomposition(&mut group, 12);

    // bench_big_linear(&mut group, 8);
    // bench_big_linear(&mut group, 10);
    // bench_big_linear(&mut group, 12);
    bench_big_linear(&mut group, 9);

    // bench_sha_proxy(&mut group, 8);
    // bench_sha_proxy(&mut group, 10);
    // bench_sha_proxy(&mut group, 12);
    bench_sha_proxy(&mut group, 9);

    bench_ecdsa(&mut group, 9);

    bench_sha_ecdsa(&mut group, 9);

    bench_ecdsa_limbs(&mut group, 9);

    // bench_big_linear_public_input(&mut group, 8);
    // bench_big_linear_public_input(&mut group, 10);
    // bench_big_linear_public_input(&mut group, 12);

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(500);
    targets = e2e_benches
}
criterion_main!(benches);
