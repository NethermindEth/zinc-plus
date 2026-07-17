use std::hint::black_box;

use criterion::{
    AxisScale, BatchSize, BenchmarkGroup, BenchmarkId, Criterion, PlotConfiguration,
    criterion_group, criterion_main, measurement::WallTime,
};
use crypto_primitives::{
    ConstIntSemiring, ProjectElementWithConfig, WithAssociatedInteger,
    crypto_bigint_int::Int,
    crypto_bigint_monty::{MontyField, MontyFieldElement},
};
use rand::rng;
use zinc_piop::{
    ideal_check::{IdealCheckProtocol, Proof},
    projections::{
        project_scalars, project_trace_coeffs_column_major, project_trace_coeffs_row_major,
    },
};
use zinc_poly::univariate::dense::DensePolynomial;
use zinc_primality::{MillerRabin, PrimalityTest};
use zinc_test_uair::{
    BigLinearUair, BinaryDecompositionUair, GenerateRandomTrace, TestUairNoMultiplication,
    TestUairSimpleMultiplication,
};
use zinc_transcript::{
    Blake3Transcript,
    traits::{ConstTranscribable, Transcript},
};
use zinc_uair::{
    Uair, UairTrace, constraint_counter::count_constraints, ideal::DegreeOneIdeal,
    ideal_collector::IdealOrZero,
};

const DEGREE_PLUS_ONE: usize = 32;

type Witness<const INT_LIMBS: usize> = DensePolynomial<Int<INT_LIMBS>, DEGREE_PLUS_ONE>;
type F<const FIELD_LIMBS: usize> = MontyField<FIELD_LIMBS>;

#[allow(clippy::arithmetic_side_effects)]
fn bench_no_mult<const INT_LIMBS: usize, const FIELD_LIMBS: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    witness_size: usize,
) where
    TestUairNoMultiplication<Int<INT_LIMBS>, <F<FIELD_LIMBS> as WithAssociatedInteger>::Integer>:
        Uair<Scalar = Witness<INT_LIMBS>, Ideal = DegreeOneIdeal<Int<INT_LIMBS>>>
            + GenerateRandomTrace<DEGREE_PLUS_ONE, PolyCoeff = Int<INT_LIMBS>, Int = Int<INT_LIMBS>>,
    MillerRabin: PrimalityTest<<F<FIELD_LIMBS> as WithAssociatedInteger>::Integer>,
{
    let mut rng = rng();
    let num_vars = zinc_utils::log2(witness_size) as usize;
    let trace = TestUairNoMultiplication::generate_random_trace(num_vars, &mut rng);

    let params = format!("NoMult/LIMBS={}/nvars={}", FIELD_LIMBS, num_vars);

    let num_constraints = count_constraints::<TestUairNoMultiplication<_, _>>();

    let prove = |field_cfg: &F<FIELD_LIMBS>,
                 trace: &UairTrace<_, _, _, _>,
                 transcript: &mut Blake3Transcript|
     -> Proof<MontyFieldElement<FIELD_LIMBS>> {
        let trace = project_trace_coeffs_row_major(trace, field_cfg);

        let projected_scalars = project_scalars::<F<FIELD_LIMBS>, TestUairNoMultiplication<_, _>>(
            field_cfg,
            |scalar| {
                scalar
                    .iter()
                    .map(|coeff| field_cfg.project(coeff))
                    .collect()
            },
        );

        // Even though this UAIR is linear, using prove_combined yields much better
        // prover performance for it.
        let evaluation_point = transcript.get_field_challenges(num_vars, field_cfg);
        IdealCheckProtocol::<TestUairNoMultiplication<_, _>>::prove_combined::<_, DEGREE_PLUS_ONE>(
            transcript,
            &trace,
            &projected_scalars,
            /* family_idx = */ 0,
            num_constraints.q,
            &evaluation_point,
            field_cfg,
        )
        .expect("Prover failed")
    };

    group.bench_function(BenchmarkId::new("Ideal Check Prover", &params), |bench| {
        let mut transcript = Blake3Transcript::new();
        let field_cfg = transcript.get_random_field_cfg::<F<FIELD_LIMBS>, _, MillerRabin>();
        bench.iter_batched(
            || transcript.clone(),
            |mut transcript| {
                let _ = black_box(prove(&field_cfg, &trace, &mut transcript));
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function(BenchmarkId::new("Ideal Check Verifier", &params), |bench| {
        let mut transcript = Blake3Transcript::new();
        let field_cfg = transcript.get_random_field_cfg::<F<FIELD_LIMBS>, _, MillerRabin>();
        let proof = prove(&field_cfg, &trace, &mut transcript);

        bench.iter_batched(
            || (proof.clone(), transcript.clone()),
            |(proof, mut transcript)| {
                let evaluation_point = transcript.get_field_challenges(num_vars, &field_cfg);
                let _ = black_box(
                    IdealCheckProtocol::<TestUairNoMultiplication<_, _>>::verify_as_subprotocol(
                        &mut transcript,
                        proof,
                        /* family_idx = */ 0,
                        num_constraints.q,
                        &evaluation_point,
                        |ideal_over_ring| {
                            ideal_over_ring.map(|i| DegreeOneIdeal::project(&field_cfg, i))
                        },
                        |_| unreachable!("not used here"),
                        &field_cfg,
                    ),
                )
                .expect("Failed to verify");
            },
            BatchSize::SmallInput,
        );
    });
}

pub fn bench_no_mult_3(group: &mut BenchmarkGroup<WallTime>, witness_size: usize) {
    bench_no_mult::<3, 4>(group, witness_size)
}

pub fn bench_no_mult_4(group: &mut BenchmarkGroup<WallTime>, witness_size: usize) {
    bench_no_mult::<4, 5>(group, witness_size)
}

#[allow(clippy::arithmetic_side_effects)]
fn bench_simple_mult<const INT_LIMBS: usize, const FIELD_LIMBS: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    witness_size: usize,
) where
    TestUairSimpleMultiplication<
        Int<INT_LIMBS>,
        <F<FIELD_LIMBS> as WithAssociatedInteger>::Integer,
    >: Uair<Scalar = Witness<INT_LIMBS>>
        + GenerateRandomTrace<DEGREE_PLUS_ONE, PolyCoeff = Int<INT_LIMBS>, Int = Int<INT_LIMBS>>,
    MillerRabin: PrimalityTest<<F<FIELD_LIMBS> as WithAssociatedInteger>::Integer>,
{
    let mut rng = rng();
    let num_vars = zinc_utils::log2(witness_size) as usize;
    let trace = TestUairSimpleMultiplication::generate_random_trace(num_vars, &mut rng);

    let params = format!("SimpleMult/LIMBS={}/nvars={}", FIELD_LIMBS, num_vars);

    let num_constraints = count_constraints::<TestUairSimpleMultiplication<_, _>>();

    let prove = |field_cfg: &F<FIELD_LIMBS>,
                 trace: &UairTrace<_, _, _, _>,
                 transcript: &mut Blake3Transcript|
     -> Proof<MontyFieldElement<FIELD_LIMBS>> {
        let trace = project_trace_coeffs_row_major(trace, field_cfg);

        let projected_scalars = project_scalars::<F<FIELD_LIMBS>, TestUairSimpleMultiplication<_, _>>(
            field_cfg,
            |scalar| {
                scalar
                    .iter()
                    .map(|coeff| field_cfg.project(coeff))
                    .collect()
            },
        );

        let evaluation_point = transcript.get_field_challenges(num_vars, field_cfg);
        IdealCheckProtocol::<TestUairSimpleMultiplication<_, _>>::prove_combined::<_, DEGREE_PLUS_ONE>(
            transcript,
            &trace,
            &projected_scalars,
            /* family_idx = */ 0,
            num_constraints.q,
            &evaluation_point,
            field_cfg,
        )
        .expect("Prover failed")
    };

    group.bench_function(BenchmarkId::new("Ideal Check Prover", &params), |bench| {
        let mut transcript = Blake3Transcript::new();
        let field_cfg = transcript.get_random_field_cfg::<F<FIELD_LIMBS>, _, MillerRabin>();
        bench.iter_batched(
            || transcript.clone(),
            |mut transcript| {
                let _ = black_box(prove(&field_cfg, &trace, &mut transcript));
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_with_input(
        BenchmarkId::new("Ideal Check Verifier", &params),
        &trace,
        |bench, trace| {
            let mut transcript = Blake3Transcript::new();
            let field_cfg = transcript.get_random_field_cfg::<F<FIELD_LIMBS>, _, MillerRabin>();
            let proof = prove(&field_cfg, trace, &mut transcript);

            bench.iter_batched(
                || (proof.clone(), transcript.clone()),
                |(proof, mut transcript)| {
                    let evaluation_point =
                        transcript.get_field_challenges(num_vars, &field_cfg);
                    let _ = black_box(IdealCheckProtocol::<TestUairSimpleMultiplication<_, _>>::verify_as_subprotocol(
                        &mut transcript,
                        proof,
                        /* family_idx = */ 0,
                        num_constraints.q,
                        &evaluation_point,
                        |_ideal_over_ring| IdealOrZero::<DegreeOneIdeal<_>>::zero(),
                        |_| unreachable!("not used here"),
                        &field_cfg,
                    ))
                    .expect("Failed to verify");
                },
                BatchSize::SmallInput,
            );
        },
    );
}

pub fn bench_simple_mult_3(group: &mut BenchmarkGroup<WallTime>, witness_size: usize) {
    bench_simple_mult::<3, 4>(group, witness_size)
}

pub fn bench_simple_mult_4(group: &mut BenchmarkGroup<WallTime>, witness_size: usize) {
    bench_simple_mult::<4, 5>(group, witness_size)
}

#[allow(clippy::arithmetic_side_effects)]
fn bench_binary_decomposition<const FIELD_LIMBS: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    witness_size: usize,
) where
    <F<FIELD_LIMBS> as WithAssociatedInteger>::Integer: ConstIntSemiring + ConstTranscribable,
    MillerRabin: PrimalityTest<<F<FIELD_LIMBS> as WithAssociatedInteger>::Integer>,
{
    macro_rules! uair_type {
        () => {
            BinaryDecompositionUair::<u32, <F<FIELD_LIMBS> as WithAssociatedInteger>::Integer>
        };
    }

    let mut rng = rng();
    let num_vars = zinc_utils::log2(witness_size) as usize;
    let trace = <uair_type!()>::generate_random_trace(num_vars, &mut rng);

    let params = format!(
        "BinaryDecomposition/LIMBS={}/nvars={}",
        FIELD_LIMBS, num_vars
    );

    let num_constraints = count_constraints::<uair_type!()>();

    let prove = |field_cfg: &F<FIELD_LIMBS>,
                 trace: &UairTrace<_, _, _, _>,
                 transcript: &mut Blake3Transcript|
     -> Proof<MontyFieldElement<FIELD_LIMBS>> {
        let trace = project_trace_coeffs_row_major(trace, field_cfg);

        let projected_scalars =
            project_scalars::<F<FIELD_LIMBS>, uair_type!()>(field_cfg, |scalar| {
                scalar
                    .iter()
                    .map(|coeff| field_cfg.project(coeff))
                    .collect()
            });

        let evaluation_point = transcript.get_field_challenges(num_vars, field_cfg);
        IdealCheckProtocol::<uair_type!()>::prove_combined::<_, DEGREE_PLUS_ONE>(
            transcript,
            &trace,
            &projected_scalars,
            /* family_idx = */ 0,
            num_constraints.q,
            &evaluation_point,
            field_cfg,
        )
        .expect("Prover failed")
    };

    group.bench_function(BenchmarkId::new("Ideal Check Prover", &params), |bench| {
        let mut transcript = Blake3Transcript::new();
        let field_cfg = transcript.get_random_field_cfg::<F<FIELD_LIMBS>, _, MillerRabin>();
        bench.iter_batched(
            || transcript.clone(),
            |mut transcript| {
                let _ = black_box(prove(&field_cfg, &trace, &mut transcript));
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function(BenchmarkId::new("Ideal Check Verifier", &params), |bench| {
        let mut transcript = Blake3Transcript::new();
        let field_cfg = transcript.get_random_field_cfg::<F<FIELD_LIMBS>, _, MillerRabin>();
        let proof = prove(&field_cfg, &trace, &mut transcript);

        bench.iter_batched(
            || (proof.clone(), transcript.clone()),
            |(proof, mut transcript)| {
                let evaluation_point = transcript.get_field_challenges(num_vars, &field_cfg);
                let _ = black_box(IdealCheckProtocol::<uair_type!()>::verify_as_subprotocol(
                    &mut transcript,
                    proof,
                    /* family_idx = */ 0,
                    num_constraints.q,
                    &evaluation_point,
                    |ideal_over_ring| {
                        ideal_over_ring.map(|ideal_over_ring| {
                            DegreeOneIdeal::project(&field_cfg, ideal_over_ring)
                        })
                    },
                    |_| unreachable!("not used here"),
                    &field_cfg,
                ))
                .expect("Failed to verify");
            },
            BatchSize::SmallInput,
        );
    });
}

#[allow(clippy::arithmetic_side_effects)]
fn bench_big_linear_uair<const FIELD_LIMBS: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    witness_size: usize,
) where
    <F<FIELD_LIMBS> as WithAssociatedInteger>::Integer: ConstIntSemiring + ConstTranscribable,
    MillerRabin: PrimalityTest<<F<FIELD_LIMBS> as WithAssociatedInteger>::Integer>,
{
    macro_rules! uair_type {
        () => {
            BigLinearUair::<u32, <F<FIELD_LIMBS> as WithAssociatedInteger>::Integer>
        };
    }

    let mut rng = rng();
    let num_vars = zinc_utils::log2(witness_size) as usize;
    let trace = <uair_type!()>::generate_random_trace(num_vars, &mut rng);

    let params = format!("BigLinearUair/LIMBS={}/nvars={}", FIELD_LIMBS, num_vars);

    let num_constraints = count_constraints::<uair_type!()>();

    macro_rules! prove {
        ($transcript:expr, $field_cfg:expr, $gen_trace:ident, $prove_fn:ident) => {{
            let trace = $gen_trace::<_, u32, u32, _, _>(&trace, $field_cfg);

            let projected_scalars =
                project_scalars::<F<FIELD_LIMBS>, uair_type!()>($field_cfg, |scalar| {
                    scalar
                        .iter()
                        .map(|coeff| $field_cfg.project(coeff))
                        .collect()
                });

            let evaluation_point = $transcript.get_field_challenges(num_vars, $field_cfg);
            IdealCheckProtocol::<uair_type!()>::$prove_fn::<_, DEGREE_PLUS_ONE>(
                $transcript,
                &trace,
                &projected_scalars,
                /* family_idx = */ 0,
                num_constraints.q,
                &evaluation_point,
                $field_cfg,
            )
            .expect("Prover failed")
        }};
    }

    group.bench_function(
        BenchmarkId::new("Ideal Check Prover (MLE-first)", &params),
        |bench| {
            let mut transcript = Blake3Transcript::new();
            let field_cfg = transcript.get_random_field_cfg::<F<FIELD_LIMBS>, _, MillerRabin>();
            bench.iter_batched(
                || (&field_cfg, transcript.clone()),
                |(field_cfg, mut transcript)| {
                    let proof = prove!(
                        &mut transcript,
                        field_cfg,
                        project_trace_coeffs_column_major,
                        prove_mle_first
                    );
                    black_box(proof);
                },
                BatchSize::SmallInput,
            );
        },
    );

    group.bench_function(
        BenchmarkId::new("Ideal Check Prover (Combined)", &params),
        |bench| {
            let mut transcript = Blake3Transcript::new();
            let field_cfg = transcript.get_random_field_cfg::<F<FIELD_LIMBS>, _, MillerRabin>();
            bench.iter_batched(
                || (&field_cfg, transcript.clone()),
                |(field_cfg, mut transcript)| {
                    let proof = prove!(
                        &mut transcript,
                        field_cfg,
                        project_trace_coeffs_row_major,
                        prove_combined
                    );
                    black_box(proof);
                },
                BatchSize::SmallInput,
            );
        },
    );

    let mut transcript = Blake3Transcript::new();
    let field_cfg = transcript.get_random_field_cfg::<F<FIELD_LIMBS>, _, MillerRabin>();
    let proof = prove!(
        &mut transcript,
        &field_cfg,
        project_trace_coeffs_column_major,
        prove_mle_first
    );

    group.bench_function(BenchmarkId::new("Ideal Check Verifier", &params), |bench| {
        bench.iter_batched(
            || (proof.clone(), transcript.clone()),
            |(proof, mut transcript)| {
                let evaluation_point = transcript.get_field_challenges(num_vars, &field_cfg);
                let _ = black_box(IdealCheckProtocol::<uair_type!()>::verify_as_subprotocol(
                    &mut transcript,
                    proof,
                    /* family_idx = */ 0,
                    num_constraints.q,
                    &evaluation_point,
                    |ideal_over_ring| {
                        ideal_over_ring.map(|ideal_over_ring| {
                            DegreeOneIdeal::project(&field_cfg, ideal_over_ring)
                        })
                    },
                    |_| unreachable!("not used here"),
                    &field_cfg,
                ))
                .expect("Failed to verify");
            },
            BatchSize::SmallInput,
        );
    });
}

/// Before/after diff for combined_poly_builder (parallel vs sequential):
///   1. cargo bench -p zinc-piop --bench ideal_check -- "Ideal Check Prover"
///      --save-baseline sequential
///   2. cargo bench -p zinc-piop --bench ideal_check --features parallel --
///      "Ideal Check Prover" --baseline sequential
pub fn ideal_check_benches(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("Ideal check benchmarks");
    group.plot_config(plot_config);

    bench_no_mult_3(&mut group, 1 << 14);
    bench_no_mult_4(&mut group, 1 << 14);
    bench_no_mult_3(&mut group, 1 << 15);
    bench_no_mult_4(&mut group, 1 << 15);
    bench_no_mult_3(&mut group, 1 << 16);
    bench_no_mult_4(&mut group, 1 << 16);
    bench_no_mult_3(&mut group, 1 << 17);
    bench_no_mult_4(&mut group, 1 << 17);

    bench_simple_mult_3(&mut group, 1 << 2);
    bench_simple_mult_4(&mut group, 1 << 2);

    bench_binary_decomposition::<3>(&mut group, 1 << 12);
    bench_binary_decomposition::<4>(&mut group, 1 << 12);
    bench_binary_decomposition::<3>(&mut group, 1 << 13);
    bench_binary_decomposition::<4>(&mut group, 1 << 13);
    bench_binary_decomposition::<3>(&mut group, 1 << 14);
    bench_binary_decomposition::<4>(&mut group, 1 << 14);
    bench_binary_decomposition::<3>(&mut group, 1 << 15);
    bench_binary_decomposition::<4>(&mut group, 1 << 15);
    bench_binary_decomposition::<3>(&mut group, 1 << 16);
    bench_binary_decomposition::<4>(&mut group, 1 << 16);
    bench_binary_decomposition::<3>(&mut group, 1 << 17);
    bench_binary_decomposition::<4>(&mut group, 1 << 17);

    bench_big_linear_uair::<3>(&mut group, 1 << 12);
    bench_big_linear_uair::<4>(&mut group, 1 << 12);
    bench_big_linear_uair::<3>(&mut group, 1 << 13);
    bench_big_linear_uair::<4>(&mut group, 1 << 13);
    bench_big_linear_uair::<3>(&mut group, 1 << 14);
    bench_big_linear_uair::<4>(&mut group, 1 << 14);
    bench_big_linear_uair::<3>(&mut group, 1 << 15);
    bench_big_linear_uair::<4>(&mut group, 1 << 15);
    bench_big_linear_uair::<3>(&mut group, 1 << 16);
    bench_big_linear_uair::<4>(&mut group, 1 << 16);
    bench_big_linear_uair::<3>(&mut group, 1 << 17);
    bench_big_linear_uair::<4>(&mut group, 1 << 17);

    group.finish();
}

criterion_group!(benches, ideal_check_benches);
criterion_main!(benches);
