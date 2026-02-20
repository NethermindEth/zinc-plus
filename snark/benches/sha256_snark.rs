//! SNARK benchmark on the SHA-256 compression UAIR.
//!
//! Breaks the prover pipeline into individual steps so that the cost
//! contribution of each stage is visible.
//!
//! Uses IPRS codes for the PCS layer. The IPRS configuration per `num_vars`
//! matches the batched PCS pipeline table's rate-1/4 configs (F65537):
//!
//!   nvars=9  → R4B32 D=1  (poly_size=512,  row_len=256)
//!   nvars=10 → R4B64 D=1  (poly_size=1024, row_len=512)
//!   nvars=11 → R4B16 D=2  (poly_size=2048, row_len=1024)

#![allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::unwrap_used,
    clippy::type_complexity,
)]

use std::collections::HashMap;
use std::hint::black_box;

use criterion::{
    BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main, measurement::WallTime,
};
use crypto_bigint::U64;
use crypto_primitives::{
    Field, FromWithConfig, PrimeField, crypto_bigint_int::Int, crypto_bigint_monty::MontyField,
    crypto_bigint_uint::Uint,
};
use rand::rng;
use zinc_piop::{
    combined_poly_resolver::CombinedPolyResolver,
    ideal_check::IdealCheckProtocol,
    projections::{
        project_scalars, project_scalars_to_field, project_trace_coeffs, project_trace_to_field,
    },
};
use zinc_poly::{
    mle::DenseMultilinearExtension,
    univariate::{
        binary::{BinaryPoly, BinaryPolyInnerProduct, BinaryPolyWideningMulByScalar},
        dense::{DensePolyInnerProduct, DensePolynomial},
        dynamic::over_field::DynamicPolynomialF,
    },
};
use zinc_primality::MillerRabin;
use zinc_test_uair::{GenerateMultiTypeWitness, sha256_compression::Sha256CompressionUair};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_uair::{
    constraint_counter::count_constraints,
    degree_counter::count_max_degree,
};
use zinc_utils::inner_product::{MBSInnerProduct, ScalarProduct};
use zinc_utils::mul_by_scalar::ScalarWideningMulByScalar;
use zip_plus::{
    batched_pcs::structs::BatchedZipPlus,
    code::{
        LinearCode,
        iprs::{
            IprsCode,
            PnttConfigF2_16R4B16, PnttConfigF2_16R4B32, PnttConfigF2_16R4B64,
        },
    },
    merkle::HASH_OUT_LEN,
    pcs::structs::{ZipPlus, ZipTypes},
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const INT_LIMBS: usize = U64::LIMBS; // 1 on 64-bit
const FIELD_LIMBS: usize = INT_LIMBS * 4; // = 4

// ---------------------------------------------------------------------------
// ZipTypes for binary polynomial columns (BinaryPoly<32>) — IPRS
// ---------------------------------------------------------------------------

struct BpZt;
impl ZipTypes for BpZt {
    const NUM_COLUMN_OPENINGS: usize = 200;
    type Eval = BinaryPoly<32>;
    type Cw = DensePolynomial<i64, 32>;
    type Fmod = Uint<FIELD_LIMBS>;
    type PrimeTest = MillerRabin;
    type Chal = i128;
    type Pt = i128;
    // Extra limb (6 vs 5) to accommodate batch accumulation in IPRS.
    type CombR = Int<{ INT_LIMBS * 6 }>;
    type Comb = DensePolynomial<Self::CombR, 32>;
    type EvalDotChal = BinaryPolyInnerProduct<Self::Chal, 32>;
    type CombDotChal =
        DensePolyInnerProduct<Self::CombR, Self::Chal, Self::CombR, MBSInnerProduct, 32>;
    type ArrCombRDotChal = MBSInnerProduct;
}

// ---------------------------------------------------------------------------
// ZipTypes for integer columns (i64 eval, i128 codeword) — IPRS
// ---------------------------------------------------------------------------

struct IntZt;
impl ZipTypes for IntZt {
    const NUM_COLUMN_OPENINGS: usize = 200;
    type Eval = i64;
    type Cw = i128;
    type Fmod = Uint<FIELD_LIMBS>;
    type PrimeTest = MillerRabin;
    type Chal = i128;
    type Pt = i128;
    // 201 bits needed for linear combination; Int<4> = 256 bits.
    type CombR = Int<4>;
    type Comb = Int<4>;
    type EvalDotChal = ScalarProduct;
    type CombDotChal = ScalarProduct;
    type ArrCombRDotChal = MBSInnerProduct;
}

// ---------------------------------------------------------------------------
// IPRS code aliases per num_vars
// ---------------------------------------------------------------------------

// nvars=9  → R4B32 D=1 (row_len=256, poly_size=512)
type BpIprs9 = IprsCode<BpZt, PnttConfigF2_16R4B32<1>, BinaryPolyWideningMulByScalar<i64>, true>;
type IntIprs9 = IprsCode<IntZt, PnttConfigF2_16R4B32<1>, ScalarWideningMulByScalar<i128>, true>;

// nvars=10 → R4B64 D=1 (row_len=512, poly_size=1024)
type BpIprs10 = IprsCode<BpZt, PnttConfigF2_16R4B64<1>, BinaryPolyWideningMulByScalar<i64>, true>;
type IntIprs10 = IprsCode<IntZt, PnttConfigF2_16R4B64<1>, ScalarWideningMulByScalar<i128>, true>;

// nvars=11 → R4B16 D=2 (row_len=1024, poly_size=2048)
type BpIprs11 = IprsCode<BpZt, PnttConfigF2_16R4B16<2>, BinaryPolyWideningMulByScalar<i64>, true>;
type IntIprs11 = IprsCode<IntZt, PnttConfigF2_16R4B16<2>, ScalarWideningMulByScalar<i128>, true>;

// ---------------------------------------------------------------------------
// Field type
// ---------------------------------------------------------------------------

type F = MontyField<FIELD_LIMBS>;

// ---------------------------------------------------------------------------
// SHA-256 SNARK benchmark
// ---------------------------------------------------------------------------

fn bench_sha256_snark(c: &mut Criterion) {
    let mut group = c.benchmark_group("SHA256 SNARK steps");
    run_sha256_bench::<BpIprs9, IntIprs9>(&mut group, 9);
    run_sha256_bench::<BpIprs10, IntIprs10>(&mut group, 10);
    run_sha256_bench::<BpIprs11, IntIprs11>(&mut group, 11);
    group.finish();
}

/// Helper: build the Fiat-Shamir transcript up to field_cfg sampling.
/// Absorbs the two commitment roots (binary-poly and integer) and then
/// samples the random field configuration.
fn build_transcript_and_field_cfg(
    bp_comm_root: &zip_plus::merkle::MtHash,
    int_comm_root: &zip_plus::merkle::MtHash,
) -> (zinc_transcript::KeccakTranscript, <F as PrimeField>::Config) {
    let mut t = zinc_transcript::KeccakTranscript::new();
    let mut buf = [0u8; HASH_OUT_LEN];
    for root in [bp_comm_root, int_comm_root] {
        ConstTranscribable::write_transcription_bytes(root, &mut buf);
        t.absorb(&buf);
    }
    let field_cfg: <F as PrimeField>::Config =
        t.get_random_field_cfg::<F, Uint<FIELD_LIMBS>, MillerRabin>();
    (t, field_cfg)
}


fn run_sha256_bench<BpCode, IntCode>(
    group: &mut criterion::BenchmarkGroup<WallTime>,
    num_vars: usize,
) where
    BpCode: LinearCode<BpZt>,
    IntCode: LinearCode<IntZt>,
{
    let mut rng = rng();
    let params = format!("nvars={num_vars}");

    // ---------- Shared setup (done once, outside the benchmarks) ----------
    let (binary_poly_trace, _, int_trace_u32) =
        Sha256CompressionUair::<INT_LIMBS>::generate_witness(num_vars, &mut rng);

    // Int<INT_LIMBS> trace — used by the PIOP (ideal check, combined resolver).
    let int_trace: Vec<DenseMultilinearExtension<Int<INT_LIMBS>>> = int_trace_u32
        .iter()
        .map(|col| col.iter().map(|&v| Int::from(v as i64)).collect())
        .collect();

    // i64 trace — used by the PCS (IntZt::Eval = i64).
    let int_trace_i64: Vec<DenseMultilinearExtension<i64>> = int_trace_u32
        .iter()
        .map(|col| col.iter().map(|&v| v as i64).collect())
        .collect();

    let poly_size = 1usize << num_vars;
    let bp_pcs_params = ZipPlus::<BpZt, BpCode>::setup(poly_size, BpCode::new(poly_size));
    let int_pcs_params = ZipPlus::<IntZt, IntCode>::setup(poly_size, IntCode::new(poly_size));

    let n_bp = binary_poly_trace.len();

    let project_scalar =
        |scalar: &DensePolynomial<Int<INT_LIMBS>, 64>, cfg: &<F as PrimeField>::Config| {
            DynamicPolynomialF {
                coeffs: scalar
                    .iter()
                    .map(|coeff| F::from_with_cfg(coeff.clone(), cfg))
                    .collect(),
            }
        };

    // Pre-compute commits (needed by test, transcript, and later steps)
    let (bp_hint, bp_comm) =
        BatchedZipPlus::<BpZt, BpCode>::commit(&bp_pcs_params, &binary_poly_trace).unwrap();
    let (int_hint, int_comm) =
        BatchedZipPlus::<IntZt, IntCode>::commit(&int_pcs_params, &int_trace_i64).unwrap();

    // Pre-compute transcript + field_cfg (needed by projection / PIOP steps)
    let (_, field_cfg) = build_transcript_and_field_cfg(
        &bp_comm.root,
        &int_comm.root,
    );

    // Pre-compute projected trace (needed by ideal check and later)
    let projected_trace: Vec<DenseMultilinearExtension<DynamicPolynomialF<F>>> =
        project_trace_coeffs::<F, Int<INT_LIMBS>, Int<INT_LIMBS>, 32>(
            &binary_poly_trace,
            &[],
            &int_trace,
            &field_cfg,
        );

    let projected_scalars: HashMap<DensePolynomial<Int<INT_LIMBS>, 64>, DynamicPolynomialF<F>> =
        project_scalars::<F, Sha256CompressionUair<INT_LIMBS>>(|scalar| {
            project_scalar(scalar, &field_cfg)
        });

    let num_constraints = count_constraints::<Sha256CompressionUair<INT_LIMBS>>();
    let max_degree = count_max_degree::<Sha256CompressionUair<INT_LIMBS>>();

    // =====================================================================
    // Step 1: Batched PCS commit (2 batches: bp + int)
    // =====================================================================
    group.bench_with_input(
        BenchmarkId::new("1_pcs_commit", &params),
        &num_vars,
        |bench, _| {
            bench.iter(|| {
                let c1 = BatchedZipPlus::<BpZt, BpCode>::commit(
                    &bp_pcs_params, &binary_poly_trace,
                ).unwrap();
                let c2 = BatchedZipPlus::<IntZt, IntCode>::commit(
                    &int_pcs_params, &int_trace_i64,
                ).unwrap();
                black_box((c1, c2));
            });
        },
    );

    // =====================================================================
    // Step 2: Batched PCS proximity test (2 batches: bp + int)
    // =====================================================================
    group.bench_with_input(
        BenchmarkId::new("2_pcs_test", &params),
        &num_vars,
        |bench, _| {
            bench.iter(|| {
                let t1 = BatchedZipPlus::<BpZt, BpCode>::test::<true>(
                    &bp_pcs_params, &binary_poly_trace, &bp_hint,
                ).unwrap();
                let t2 = BatchedZipPlus::<IntZt, IntCode>::test::<true>(
                    &int_pcs_params, &int_trace_i64, &int_hint,
                ).unwrap();
                black_box((t1, t2));
            });
        },
    );

    // =====================================================================
    // Step 3: Transcript + field config sampling
    // =====================================================================
    group.bench_with_input(
        BenchmarkId::new("3_transcript_field_cfg", &params),
        &num_vars,
        |bench, _| {
            bench.iter(|| {
                black_box(build_transcript_and_field_cfg(
                    &bp_comm.root,
                    &int_comm.root,
                ));
            });
        },
    );

    // =====================================================================
    // Step 4: Project trace to F[X]
    // =====================================================================
    group.bench_with_input(
        BenchmarkId::new("4_project_trace_fx", &params),
        &num_vars,
        |bench, _| {
            bench.iter(|| {
                black_box(
                    project_trace_coeffs::<F, Int<INT_LIMBS>, Int<INT_LIMBS>, 32>(
                        &binary_poly_trace,
                        &[],
                        &int_trace,
                        &field_cfg,
                    ),
                );
            });
        },
    );

    // =====================================================================
    // Step 5: Project scalars to F[X]
    // =====================================================================
    group.bench_with_input(
        BenchmarkId::new("5_project_scalars_fx", &params),
        &num_vars,
        |bench, _| {
            bench.iter(|| {
                black_box(
                    project_scalars::<F, Sha256CompressionUair<INT_LIMBS>>(|scalar| {
                        project_scalar(scalar, &field_cfg)
                    }),
                );
            });
        },
    );

    // =====================================================================
    // Step 6: Ideal check prover
    // =====================================================================
    group.bench_with_input(
        BenchmarkId::new("6_ideal_check", &params),
        &num_vars,
        |bench, _| {
            bench.iter_batched(
                || {
                    // Rebuild transcript to the same state the prover would have
                    let (t, _) = build_transcript_and_field_cfg(
                        &bp_comm.root,
                        &int_comm.root,
                    );
                    t
                },
                |mut prover_transcript| {
                    black_box(
                        Sha256CompressionUair::<INT_LIMBS>::prove_as_subprotocol(
                            &mut prover_transcript,
                            &projected_trace,
                            &projected_scalars,
                            num_constraints,
                            num_vars,
                            true,
                            &field_cfg,
                        )
                        .unwrap(),
                    );
                },
                BatchSize::SmallInput,
            );
        },
    );

    // =====================================================================
    // Step 7: Project trace & scalars F[X] -> F
    // =====================================================================
    group.bench_with_input(
        BenchmarkId::new("7_project_to_f", &params),
        &num_vars,
        |bench, _| {
            // We need a projecting element; run the ideal check once to advance
            // the transcript to the right state.
            let (mut t, _) = build_transcript_and_field_cfg(
                &bp_comm.root,
                &int_comm.root,
            );
            let _ = Sha256CompressionUair::<INT_LIMBS>::prove_as_subprotocol(
                &mut t,
                &projected_trace,
                &projected_scalars,
                num_constraints,
                num_vars,
                true,
                &field_cfg,
            )
            .unwrap();
            let projecting_element: F = t.get_field_challenge(&field_cfg);

            bench.iter(|| {
                let tf: Vec<DenseMultilinearExtension<<F as Field>::Inner>> =
                    project_trace_to_field::<F, 32>(
                        &binary_poly_trace,
                        &[],
                        &projected_trace[n_bp..],
                        &projecting_element,
                    );
                let sf: HashMap<DensePolynomial<Int<INT_LIMBS>, 64>, F> =
                    project_scalars_to_field(projected_scalars.clone(), &projecting_element)
                        .unwrap();
                black_box((tf, sf));
            });
        },
    );

    // =====================================================================
    // Step 8: Combined polynomial resolver
    // =====================================================================
    group.bench_with_input(
        BenchmarkId::new("8_combined_resolver", &params),
        &num_vars,
        |bench, _| {
            // Run through ideal check once to get ic_prover_state
            let (mut t, _) = build_transcript_and_field_cfg(
                &bp_comm.root,
                &int_comm.root,
            );
            let (_, ic_prover_state) =
                Sha256CompressionUair::<INT_LIMBS>::prove_as_subprotocol(
                    &mut t,
                    &projected_trace,
                    &projected_scalars,
                    num_constraints,
                    num_vars,
                    true,
                    &field_cfg,
                )
                .unwrap();
            let projecting_element: F = t.get_field_challenge(&field_cfg);

            let trace_f: Vec<DenseMultilinearExtension<<F as Field>::Inner>> =
                project_trace_to_field::<F, 32>(
                    &binary_poly_trace,
                    &[],
                    &projected_trace[n_bp..],
                    &projecting_element,
                );
            let scalars_f: HashMap<DensePolynomial<Int<INT_LIMBS>, 64>, F> =
                project_scalars_to_field(projected_scalars.clone(), &projecting_element).unwrap();

            bench.iter_batched(
                || {
                    // Clone transcript at this point
                    (t.clone(), trace_f.clone())
                },
                |(mut transcript, tf)| {
                    black_box(
                        CombinedPolyResolver::<F>::prove_as_subprotocol::<
                            Sha256CompressionUair<INT_LIMBS>,
                        >(
                            &mut transcript,
                            tf,
                            &ic_prover_state.evaluation_point,
                            &scalars_f,
                            num_constraints,
                            num_vars,
                            max_degree,
                            &field_cfg,
                        )
                        .unwrap(),
                    );
                },
                BatchSize::SmallInput,
            );
        },
    );
}

criterion_group!(benches, bench_sha256_snark);
criterion_main!(benches);
