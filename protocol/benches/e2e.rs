#![allow(clippy::arithmetic_side_effects)]

use std::hint::black_box;

use criterion::{
    AxisScale, BatchSize, BenchmarkGroup, BenchmarkId, Criterion, PlotConfiguration,
    criterion_group, criterion_main, measurement::WallTime,
};
use crypto_bigint::U64;
use crypto_primitives::{
    crypto_bigint_int::Int, crypto_bigint_monty::MontyField,
    crypto_bigint_uint::Uint,
};
use rand::rng;
use zinc_poly::{
    mle::DenseMultilinearExtension,
    univariate::{
        binary::{BinaryPoly, BinaryPolyInnerProduct},
        dense::{DensePolynomial, DensePolyInnerProduct},
        ideal::DegreeOneIdeal,
    },
};
use zinc_primality::MillerRabin;
use zinc_protocol::{project_scalar_fn, Proof, ZincPlusPiop, ZincTypes};
use zinc_test_uair::{
    BigLinearUair, BinaryDecompositionUair, GenerateMultiTypeWitness, GenerateSingleTypeWitness,
    TestAirNoMultiplication,
};
use zinc_uair::Uair;
use zinc_utils::{
    UNCHECKED,
    inner_product::{MBSInnerProduct, ScalarProduct},
};
use zip_plus::{
    code::{
        LinearCode,
        iprs::{IprsCode, PnttConfigF2_16_1},
    },
    pcs::structs::{ZipPlus, ZipPlusParams, ZipTypes},
};

//
// Types shared across all benchmarks
//

const INT_LIMBS: usize = U64::LIMBS;
const FIELD_LIMBS: usize = 4;
const DEGREE_PLUS_ONE: usize = 32;

const K: usize = INT_LIMBS * 4;
const M: usize = INT_LIMBS * 8;
const IPRS_DEPTH: usize = 1;

type F = MontyField<FIELD_LIMBS>;

pub struct BinPolyZipTypes;
impl ZipTypes for BinPolyZipTypes {
    const NUM_COLUMN_OPENINGS: usize = 200;
    type Eval = BinaryPoly<DEGREE_PLUS_ONE>;
    type Cw = DensePolynomial<i64, DEGREE_PLUS_ONE>;
    type Fmod = Uint<K>;
    type PrimeTest = MillerRabin;
    type Chal = i128;
    type Pt = i128;
    type CombR = Int<M>;
    type Comb = DensePolynomial<Self::CombR, DEGREE_PLUS_ONE>;
    type EvalDotChal = BinaryPolyInnerProduct<Self::Chal, DEGREE_PLUS_ONE>;
    type CombDotChal = DensePolyInnerProduct<
        Self::CombR,
        Self::Chal,
        Self::CombR,
        MBSInnerProduct,
        DEGREE_PLUS_ONE,
    >;
    type ArrCombRDotChal = MBSInnerProduct;
}

pub struct ArbitraryPolyZipTypesIprs;
impl ZipTypes for ArbitraryPolyZipTypesIprs {
    const NUM_COLUMN_OPENINGS: usize = 200;
    type Eval = DensePolynomial<i64, DEGREE_PLUS_ONE>;
    type Cw = DensePolynomial<i64, DEGREE_PLUS_ONE>;
    type Fmod = Uint<K>;
    type PrimeTest = MillerRabin;
    type Chal = i128;
    type Pt = i128;
    type CombR = Int<M>;
    type Comb = DensePolynomial<Self::CombR, DEGREE_PLUS_ONE>;
    type EvalDotChal =
        DensePolyInnerProduct<i64, Self::Chal, Self::CombR, MBSInnerProduct, DEGREE_PLUS_ONE>;
    type CombDotChal = DensePolyInnerProduct<
        Self::CombR,
        Self::Chal,
        Self::CombR,
        MBSInnerProduct,
        DEGREE_PLUS_ONE,
    >;
    type ArrCombRDotChal = MBSInnerProduct;
}

pub struct IntZipTypes;
impl ZipTypes for IntZipTypes {
    const NUM_COLUMN_OPENINGS: usize = 200;
    type Eval = i64;
    type Cw = i128;
    type Fmod = Uint<K>;
    type PrimeTest = MillerRabin;
    type Chal = i128;
    type Pt = i128;
    type CombR = Int<M>;
    type Comb = Self::CombR;
    type EvalDotChal = ScalarProduct;
    type CombDotChal = ScalarProduct;
    type ArrCombRDotChal = MBSInnerProduct;
}

struct BenchZincTypes;

impl ZincTypes<DEGREE_PLUS_ONE> for BenchZincTypes {
    type Int = i64;
    type Chal = i128;
    type Pt = i128;
    type CombR = Int<M>;
    type Fmod = Uint<K>;
    type PrimeTest = MillerRabin;

    type BinaryZt = BinPolyZipTypes;
    type ArbitraryZt = ArbitraryPolyZipTypesIprs;
    type IntZt = IntZipTypes;

    type BinaryLc = IprsCode<Self::BinaryZt, PnttConfigF2_16_1<IPRS_DEPTH>, UNCHECKED>;
    type ArbitraryLc = IprsCode<Self::ArbitraryZt, PnttConfigF2_16_1<IPRS_DEPTH>, UNCHECKED>;
    type IntLc = IprsCode<Self::IntZt, PnttConfigF2_16_1<IPRS_DEPTH>, UNCHECKED>;
}

type Pp = (
    ZipPlusParams<BinPolyZipTypes, <BenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::BinaryLc>,
    ZipPlusParams<
        ArbitraryPolyZipTypesIprs,
        <BenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::ArbitraryLc,
    >,
    ZipPlusParams<IntZipTypes, <BenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::IntLc>,
);

fn setup_pp(num_vars: usize) -> Pp {
    let poly_size = 1 << num_vars;
    (
        ZipPlus::<BinPolyZipTypes, <BenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::BinaryLc>::setup(
            poly_size,
            <BenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::BinaryLc::new(poly_size),
        ),
        ZipPlus::<
            ArbitraryPolyZipTypesIprs,
            <BenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::ArbitraryLc,
        >::setup(
            poly_size,
            <BenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::ArbitraryLc::new(poly_size),
        ),
        ZipPlus::<IntZipTypes, <BenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::IntLc>::setup(
            poly_size,
            <BenchZincTypes as ZincTypes<DEGREE_PLUS_ONE>>::IntLc::new(poly_size),
        ),
    )
}

fn bench_prove_verify<U>(
    group: &mut BenchmarkGroup<WallTime>,
    label: &str,
    num_vars: usize,
    bin: &[DenseMultilinearExtension<BinaryPoly<DEGREE_PLUS_ONE>>],
    arb: &[DenseMultilinearExtension<DensePolynomial<i64, DEGREE_PLUS_ONE>>],
    int: &[DenseMultilinearExtension<i64>],
) where
    U: Uair<Scalar = DensePolynomial<i64, DEGREE_PLUS_ONE>, Ideal = DegreeOneIdeal<i64>> + 'static,
{
    let pp = setup_pp(num_vars);
    let params = format!("{label}/nvars={num_vars}");

    group.bench_function(BenchmarkId::new("Prove", &params), |bench| {
        bench.iter(|| {
            black_box(
                ZincPlusPiop::<BenchZincTypes, U, F, DEGREE_PLUS_ONE>::prove::<UNCHECKED>(
                    &pp,
                    bin,
                    arb,
                    int,
                    num_vars,
                    project_scalar_fn,
                ),
            )
            .expect("Prover failed");
        });
    });

    let proof: Proof<F> =
        ZincPlusPiop::<BenchZincTypes, U, F, DEGREE_PLUS_ONE>::prove::<UNCHECKED>(
            &pp,
            bin,
            arb,
            int,
            num_vars,
            project_scalar_fn,
        )
        .expect("proof generation for verifier bench");

    group.bench_function(BenchmarkId::new("Verify", &params), |bench| {
        bench.iter_batched(
            || proof.clone(),
            |proof| {
                black_box(
                    ZincPlusPiop::<BenchZincTypes, U, F, DEGREE_PLUS_ONE>::verify::<_, UNCHECKED>(
                        &pp,
                        proof,
                        num_vars,
                        project_scalar_fn,
                        |ideal, field_cfg| {
                            ideal.map(|i| DegreeOneIdeal::from_with_cfg(i, field_cfg))
                        },
                    ),
                )
                .expect("Verifier failed");
            },
            BatchSize::SmallInput,
        );
    });
}

// ── Per-UAIR witness generation + benchmark dispatch ─────────────────────────

fn bench_no_mult(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    let mut rng = rng();
    let trace = TestAirNoMultiplication::<i64>::generate_witness(num_vars, &mut rng);
    bench_prove_verify::<TestAirNoMultiplication<i64>>(group, "NoMult", num_vars, &[], &trace, &[]);
}

fn bench_binary_decomposition(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    let mut rng = rng();
    let (bin, arb, int) = BinaryDecompositionUair::<i64>::generate_witness(num_vars, &mut rng);
    bench_prove_verify::<BinaryDecompositionUair<i64>>(
        group,
        "BinaryDecomposition",
        num_vars,
        &bin,
        &arb,
        &int,
    );
}

fn bench_big_linear(group: &mut BenchmarkGroup<WallTime>, num_vars: usize) {
    let mut rng = rng();
    let (bin, arb, int) = BigLinearUair::<i64>::generate_witness(num_vars, &mut rng);
    bench_prove_verify::<BigLinearUair<i64>>(group, "BigLinear", num_vars, &bin, &arb, &int);
}

// ── Criterion entry point ────────────────────────────────────────────────────

fn e2e_benches(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    let mut group = c.benchmark_group("Zinc+ E2E");
    group.plot_config(plot_config);

    for &nv in &[8, 10, 12] {
        bench_no_mult(&mut group, nv);
        bench_binary_decomposition(&mut group, nv);
        bench_big_linear(&mut group, nv);
    }

    group.finish();
}

criterion_group!(benches, e2e_benches);
criterion_main!(benches);
