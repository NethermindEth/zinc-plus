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
use std::{hint::black_box, marker::PhantomData, ops::Neg};
use zinc_poly::{
    ConstCoeffBitWidth, Polynomial,
    univariate::{
        binary::{BinaryPoly, BinaryPolyInnerProduct},
        dense::{DensePolyInnerProduct, DensePolynomial},
        dynamic::over_field::DynamicPolynomialF,
    },
};
use zinc_primality::{MillerRabin, PrimalityTest};
use zinc_protocol::{Proof, ZincPlusPiop, ZincTypes};
use zinc_test_uair::{
    BigLinearUair, BigLinearUairWithPublicInput, BinaryDecompositionUair, GenerateRandomTrace,
    TestAirNoMultiplication,
};
use zinc_transcript::traits::ConstTranscribable;
use zinc_uair::{
    Uair, UairTrace,
    degree_counter::count_max_degree,
    ideal::{Ideal, IdealCheck, degree_one::DegreeOneIdeal},
    ideal_collector::IdealOrZero,
};
use zinc_utils::{
    CHECKED,
    from_ref::FromRef,
    inner_product::{InnerProduct, MBSInnerProduct, ScalarProduct},
    mul_by_scalar::MulByScalar,
    named::Named,
    projectable_to_field::ProjectableToField,
};
use zip_plus::{
    code::{
        LinearCode,
        iprs::{IprsCode, PnttConfigF2_16_1},
    },
    pcs::structs::{ZipPlus, ZipPlusParams, ZipTypes},
};

//
// Type definitions and constants
//

const IPRS_DEPTH: usize = 1;

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
    const NUM_COLUMN_OPENINGS: usize = 200;
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

    type BinaryLc = IprsCode<Self::BinaryZt, PnttConfigF2_16_1<IPRS_DEPTH>, CHECKED>;
    type ArbitraryLc = IprsCode<Self::ArbitraryZt, PnttConfigF2_16_1<IPRS_DEPTH>, CHECKED>;
    type IntLc = IprsCode<Self::IntZt, PnttConfigF2_16_1<IPRS_DEPTH>, CHECKED>;
}

//
// Constants and concrete types
//

const DEGREE_PLUS_ONE: usize = 32;
const INT_LIMBS: usize = U64::LIMBS;
const FIELD_LIMBS: usize = 3;

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

fn setup_pp(num_vars: usize) -> Pp<BenchZincTypes> {
    let poly_size = 1 << num_vars;
    (
        ZipPlus::setup(
            poly_size,
            <BenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::BinaryLc::new(poly_size),
        ),
        ZipPlus::setup(
            poly_size,
            <BenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::ArbitraryLc::new(poly_size),
        ),
        ZipPlus::setup(
            poly_size,
            <BenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::IntLc::new(poly_size),
        ),
    )
}

//
// Main benchmarking routine
//

#[allow(clippy::too_many_arguments)]
fn bench_prove_verify<Zt, U, IdealOverF>(
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
                    black_box(<zinc_plus!()>::prove::<{ $mle_first }, CHECKED>(
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

    bench_prove!("Prove (Combined)", false);

    if count_max_degree::<U>() <= 1 {
        bench_prove!("Prove (MLE-first)", true);
    }

    let proof: Proof<F> =
        <zinc_plus!()>::prove::<false, CHECKED>(&pp, &trace, num_vars, project_scalar)
            .expect("proof generation for verifier bench");

    let sig = U::signature();
    let public_trace = trace.public(&sig);

    group.bench_function(BenchmarkId::new("Verify", &params), |bench| {
        bench.iter_batched(
            || proof.clone(),
            |proof| {
                black_box(<zinc_plus!()>::verify::<_, CHECKED>(
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

    bench_prove_verify::<BenchZincTypes, U, _>(
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

fn bench_big_linear_public_input(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    do_bench_uair::<BigLinearUairWithPublicInput<i64>>(group, "BigLinearPI", num_vars);
}

//
// Criterion entry point
//

fn e2e_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zinc+ E2E");

    bench_no_mult(&mut group, 9);

    bench_binary_decomposition(&mut group, 9);

    bench_big_linear(&mut group, 9);

    bench_big_linear_public_input(&mut group, 9);

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(500);
    targets = e2e_benches
}
criterion_main!(benches);
