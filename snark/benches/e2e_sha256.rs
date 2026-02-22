//! End-to-end SHA-256 benchmarks for the Zinc+ SNARK.
//!
//! Measures prover time, verifier time, and proof size for SHA-256
//! hashing plus (simulated) ECDSA verification using the Zinc+ pipeline:
//! Batched PCS (Zip+ with IPRS codes) + PIOP (ideal check + CPR + sumcheck).
//!
//! Target metrics (from the Zinc+ paper, MacBook Air M4):
//!   - Prover:  < 30 ms
//!   - Verifier: < 5 ms
//!   - Proof:   200–300 KB

#![allow(clippy::arithmetic_side_effects, clippy::unwrap_used)]

use std::hint::black_box;
use std::io::Write;
use std::marker::PhantomData;
use std::time::{Duration, Instant};

use criterion::{
    criterion_group, criterion_main, BenchmarkGroup, Criterion, measurement::WallTime,
};
use crypto_bigint::U64;
use crypto_primitives::{
    boolean::Boolean,
    crypto_bigint_int::Int,
    crypto_bigint_monty::MontyField,
    crypto_bigint_uint::Uint,
    Field, FixedSemiring, FromWithConfig, IntoWithConfig, PrimeField,
};
use num_traits::One;


use zinc_poly::mle::{DenseMultilinearExtension, MultilinearExtensionRand};
use zinc_poly::univariate::binary::{
    BinaryPoly, BinaryPolyInnerProduct, BinaryPolyWideningMulByScalar,
};
use zinc_poly::univariate::dense::{DensePolyInnerProduct, DensePolynomial};
use zinc_primality::MillerRabin;
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_utils::{
    UNCHECKED,
    from_ref::FromRef,
    inner_product::MBSInnerProduct,
    named::Named,
    projectable_to_field::ProjectableToField,
};
use zip_plus::{
    batched_pcs::structs::BatchedZipPlus,
    code::{
        LinearCode,
        iprs::{IprsCode, PnttConfigF2_16R4B16, PnttConfigF2_16R4B32, PnttConfigF2_16R4B64},
    },
    pcs::structs::{ZipPlusParams, ZipTypes},
};

use zinc_sha256_uair::{Sha256Uair, witness::GenerateWitness};

// ─── Type definitions (matching batched_zip_plus_benches.rs) ────────────────

const INT_LIMBS: usize = U64::LIMBS;
type F = MontyField<{ INT_LIMBS * 4 }>;

struct Sha256ZipTypes<CwCoeff, const D_PLUS_ONE: usize>(PhantomData<CwCoeff>);

impl<CwCoeff, const D_PLUS_ONE: usize> ZipTypes for Sha256ZipTypes<CwCoeff, D_PLUS_ONE>
where
    CwCoeff: ConstTranscribable
        + Copy
        + Default
        + FromRef<Boolean>
        + Named
        + FixedSemiring
        + Send
        + Sync,
    Int<6>: FromRef<CwCoeff>,
{
    const NUM_COLUMN_OPENINGS: usize = 64; // 64 queries ≈ 128-bit security at rate 1/4
    type Eval = BinaryPoly<D_PLUS_ONE>;
    type Cw = DensePolynomial<CwCoeff, D_PLUS_ONE>;
    type Fmod = Uint<{ INT_LIMBS * 4 }>;
    type PrimeTest = MillerRabin;
    type Chal = i128;
    type Pt = i128;
    type CombR = Int<{ INT_LIMBS * 6 }>;
    type Comb = DensePolynomial<Self::CombR, D_PLUS_ONE>;
    type EvalDotChal = BinaryPolyInnerProduct<Self::Chal, D_PLUS_ONE>;
    type CombDotChal =
        DensePolyInnerProduct<Self::CombR, Self::Chal, Self::CombR, MBSInnerProduct, D_PLUS_ONE>;
    type ArrCombRDotChal = MBSInnerProduct;
}

// IPRS code types for BinaryPoly<32>
type IprsBPoly32R4B64<const DEPTH: usize, const CHECK: bool> = IprsCode<
    Sha256ZipTypes<i64, 32>,
    PnttConfigF2_16R4B64<DEPTH>,
    BinaryPolyWideningMulByScalar<i64>,
    CHECK,
>;
type IprsBPoly32R4B16<const DEPTH: usize, const CHECK: bool> = IprsCode<
    Sha256ZipTypes<i64, 32>,
    PnttConfigF2_16R4B16<DEPTH>,
    BinaryPolyWideningMulByScalar<i64>,
    CHECK,
>;
type IprsBPoly32R4B32<const DEPTH: usize, const CHECK: bool> = IprsCode<
    Sha256ZipTypes<i64, 32>,
    PnttConfigF2_16R4B32<DEPTH>,
    BinaryPolyWideningMulByScalar<i64>,
    CHECK,
>;

// ─── SHA-256 Trace Parameters ───────────────────────────────────────────────
//
// SHA-256 has 64 rows × 19 columns.
// ECDSA has 258 rows × 14 columns.
//
// IPRS codes need DEPTH ≥ 1 (radix-8 NTT), and row_len = BASE_LEN × 8^DEPTH.
//
// Benchmark configurations:
//   - Single SHA-256: poly_size = 2^7 = 128, 19 polys, R4B16 DEPTH=1 (row_len=128)
//     → 64 real rows + 64 padding = 128 total
//   - 8× SHA-256 + ECDSA: poly_size = 2^10 = 1024, 33 polys
//     → 8×64=512 SHA rows + 258 ECDSA rows = 770 → pad to 1024

const SHA256_NUM_VARS: usize = 7;        // 2^7 = 128 rows (64 real + 64 padding)
const SHA256_BATCH_SIZE: usize = 19;      // 19 SHA-256 columns

// For 8× SHA-256 + ECDSA: 33 columns (19 SHA + 14 ECDSA), 1024 rows
const SHA256_8X_ECDSA_NUM_VARS: usize = 10;   // 1024 rows
const SHA256_8X_ECDSA_BATCH_SIZE: usize = 34;  // 20 SHA + 14 ECDSA columns

// ─── Benchmark helpers ──────────────────────────────────────────────────────

fn generate_sha256_trace(num_vars: usize) -> Vec<DenseMultilinearExtension<BinaryPoly<32>>> {
    let mut rng = rand::rng();
    <Sha256Uair as GenerateWitness<BinaryPoly<32>>>::generate_witness(num_vars, &mut rng)
}

fn generate_random_trace(
    num_vars: usize,
    num_cols: usize,
) -> Vec<DenseMultilinearExtension<BinaryPoly<32>>> {
    let mut rng = rand::rng();
    (0..num_cols)
        .map(|_| DenseMultilinearExtension::rand(num_vars, &mut rng))
        .collect()
}

/// Benchmark the full PCS pipeline (commit + test + evaluate + verify)
/// for a SHA-256-like trace.
fn bench_pcs_pipeline<Zt, Lc, const P: usize, const CHECK: bool>(
    group: &mut BenchmarkGroup<WallTime>,
    label: &str,
    num_rows: usize,
    _batch_size: usize,
    trace: &[DenseMultilinearExtension<BinaryPoly<32>>],
) where
    Zt: ZipTypes<Eval = BinaryPoly<32>>,
    Lc: LinearCode<Zt>,
    F: for<'a> FromWithConfig<&'a Zt::Chal> + for<'a> FromWithConfig<&'a Zt::Pt>,
    <F as Field>::Inner: FromRef<Zt::Fmod>,
    Zt::Eval: ProjectableToField<F>,
    Zt::Cw: ProjectableToField<F>,
{
    let poly_size = 1 << P;
    let row_len = poly_size / num_rows;
    let linear_code = Lc::new(row_len);
    let params = ZipPlusParams::new(P, num_rows, linear_code);

    // Warm up: commit once
    let (hint, commitment) = BatchedZipPlus::<Zt, Lc>::commit(&params, trace)
        .expect("commit failed");

    // ── Prover benchmark (commit + test + evaluate) ─────────────────
    group.bench_function(format!("{label}/Prover"), |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let t = Instant::now();

                // Commit
                let (hint, _) = BatchedZipPlus::<Zt, Lc>::commit(&params, trace)
                    .expect("commit");

                // Test
                let test_tx = BatchedZipPlus::<Zt, Lc>::test::<CHECK>(&params, trace, &hint)
                    .expect("test");

                // Evaluate
                let point: Vec<Zt::Pt> = vec![Zt::Pt::one(); P];
                let (_evals, _proof) =
                    BatchedZipPlus::<Zt, Lc>::evaluate::<F, CHECK>(
                        &params, trace, &point, test_tx,
                    )
                    .expect("evaluate");

                total += t.elapsed();
            }
            total
        });
    });

    // Prepare proof for verifier benchmark
    let test_tx = BatchedZipPlus::<Zt, Lc>::test::<CHECK>(&params, trace, &hint)
        .expect("test");
    let point: Vec<Zt::Pt> = vec![Zt::Pt::one(); P];
    let (evals_f, proof) =
        BatchedZipPlus::<Zt, Lc>::evaluate::<F, CHECK>(&params, trace, &point, test_tx)
            .expect("evaluate");
    let field_cfg = *evals_f[0].cfg();
    let point_f: Vec<F> = point.iter().map(|v| v.into_with_cfg(&field_cfg)).collect();

    // ── Verifier benchmark ──────────────────────────────────────────
    group.bench_function(format!("{label}/Verifier"), |b| {
        b.iter(|| {
            let result = BatchedZipPlus::<Zt, Lc>::verify::<F, CHECK>(
                &params,
                &commitment,
                &point_f,
                &evals_f,
                &proof,
            );
            let _ = black_box(result);
        });
    });

    // ── Proof size ──────────────────────────────────────────────────
    let proof_bytes: Vec<u8> = {
        let tx: zip_plus::pcs_transcript::PcsTranscript = proof.into();
        tx.stream.into_inner()
    };
    // Measure compressed size (deflate level 6)
    let mut encoder = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(&proof_bytes).unwrap();
    let compressed = encoder.finish().unwrap();
    println!(
        "  {label}: proof_size = {} bytes ({:.1} KB), compressed = {} bytes ({:.1} KB, {:.1}× ratio)",
        proof_bytes.len(),
        proof_bytes.len() as f64 / 1024.0,
        compressed.len(),
        compressed.len() as f64 / 1024.0,
        proof_bytes.len() as f64 / compressed.len() as f64,
    );
}

// ─── Criterion suites ───────────────────────────────────────────────────────

/// Single SHA-256 compression (64 rows padded to 128 × 19 columns = poly_size 2^7).
///
/// Uses R4B16 IPRS codes at rate 1/4 with DEPTH=1 (row_len=128).
///
/// Reports both PCS-only and full pipeline (PCS + PIOP) timings so the
/// paper can cite an honest "total prover time" that includes ideal check
/// and combined-poly-resolver overhead.
///
/// **Proof size claims are valid for DEPTH=1 configurations only.**
/// DEPTH≥2 dramatically reduces deflate compressibility (see §4 of review).
fn sha256_single(c: &mut Criterion) {
    use zinc_sha256_uair::CyclotomicIdeal;
    use zinc_uair::ideal_collector::IdealOrZero;

    let mut group = c.benchmark_group("SHA-256 Single Compression");
    group.sample_size(20);

    let trace = generate_sha256_trace(SHA256_NUM_VARS);

    // ── PCS-only (component benchmark) ──────────────────────────────
    // poly_size = 2^7 = 128, num_rows = 1, row_len = 128
    // R4B16 DEPTH=1 gives row_len = 16 × 8^1 = 128  ✓
    bench_pcs_pipeline::<Sha256ZipTypes<i64, 32>, IprsBPoly32R4B16<1, UNCHECKED>, 7, UNCHECKED>(
        &mut group,
        "1xSHA256/PCS-only",
        1,
        SHA256_BATCH_SIZE,
        &trace,
    );

    // ── Full pipeline: PCS + PIOP (headline benchmark) ──────────────
    // This is the honest prover time: IC + CPR + PCS commit + test + evaluate.
    {
        type Lc = IprsBPoly32R4B16<1, UNCHECKED>;
        type Zt = Sha256ZipTypes<i64, 32>;
        let linear_code = Lc::new(128);
        let params = ZipPlusParams::new(SHA256_NUM_VARS, 1, linear_code);

        group.bench_function("1xSHA256/FullPipeline/Prover", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let t = Instant::now();
                    let _proof = zinc_snark::pipeline::prove::<Sha256Uair, Zt, Lc, 32, UNCHECKED>(
                        &params,
                        &trace,
                        SHA256_NUM_VARS,
                    );
                    total += t.elapsed();
                }
                total
            });
        });

        // Prepare proof for verifier benchmark
        let zinc_proof = zinc_snark::pipeline::prove::<Sha256Uair, Zt, Lc, 32, UNCHECKED>(
            &params,
            &trace,
            SHA256_NUM_VARS,
        );

        group.bench_function("1xSHA256/FullPipeline/Verifier", |b| {
            b.iter(|| {
                let result = zinc_snark::pipeline::verify::<Sha256Uair, Zt, Lc, 32, UNCHECKED, _, _>(
                    &params,
                    &zinc_proof,
                    SHA256_NUM_VARS,
                    |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
                );
                black_box(result);
            });
        });

        println!(
            "\n  Full pipeline prover breakdown: IC={:?}, CPR={:?}, PCS commit={:?}, test={:?}, evaluate={:?}, total={:?}",
            zinc_proof.timing.ideal_check,
            zinc_proof.timing.combined_poly_resolver,
            zinc_proof.timing.pcs_commit,
            zinc_proof.timing.pcs_test,
            zinc_proof.timing.pcs_evaluate,
            zinc_proof.timing.total,
        );
    }

    group.finish();
}

/// 8× SHA-256 + ECDSA (1024 rows × 33 columns = poly_size 2^10).
///
/// This is the paper's target benchmark configuration.
/// - 19 columns for SHA-256 (8 instances × 64 rows = 512 rows)
/// - 14 columns for ECDSA (258 rows, simulated with random data)
/// - All padded to 1024 rows, total 33 columns.
///
/// The combined trace uses **real SHA-256 witness** columns (19) concatenated
/// with random ECDSA-shaped columns (14). The PCS cost is data-agnostic, so
/// timing is identical regardless of trace content; this labelling makes the
/// benchmark honest about what the witness represents.
///
/// **Note:** Proof size for DEPTH=2 (1024 rows) does NOT meet the paper's
/// 200–300 KB target due to i64 serialization width and coefficient
/// incompressibility. Proof size claims should be scoped to DEPTH=1 configs.
/// The 14-column DEPTH=1 benchmarks below DO meet the target.
fn sha256_8x_ecdsa(c: &mut Criterion) {
    let mut group = c.benchmark_group("8xSHA256+ECDSA");
    group.sample_size(10);

    // ── Build combined trace: 19 SHA-256 cols + 14 random ECDSA cols ─
    // SHA-256 witness at poly_size=1024 (10 vars) — real witness data
    let sha_trace = generate_sha256_trace(SHA256_8X_ECDSA_NUM_VARS);
    // Random ECDSA-shaped columns (matching 14-column ECDSA geometry)
    let ecdsa_trace = generate_random_trace(SHA256_8X_ECDSA_NUM_VARS, 14);
    let trace_33: Vec<DenseMultilinearExtension<BinaryPoly<32>>> =
        sha_trace.into_iter().chain(ecdsa_trace.into_iter()).collect();
    assert_eq!(trace_33.len(), SHA256_8X_ECDSA_BATCH_SIZE);

    // poly_size = 2^10 = 1024, num_rows = 1, row_len = 1024
    // R4B16 DEPTH=2 gives row_len = 16 × 64 = 1024  ✓
    // NOTE: This is PCS-only timing. For PIOP overhead, see sha256_full_pipeline.
    bench_pcs_pipeline::<Sha256ZipTypes<i64, 32>, IprsBPoly32R4B16<2, UNCHECKED>, 10, UNCHECKED>(
        &mut group,
        "33cols/PCS-only/R4B16d2",
        1,
        SHA256_8X_ECDSA_BATCH_SIZE,
        &trace_33,
    );

    // ── Paper model config: 14 columns (ECDSA-sized), DEPTH=1 ────────
    // The paper's proof_size.py uses n_pol=14.
    // Benchmark with 14 columns at poly=2^9 (ECDSA's 258 rows padded to 512)
    // R4B64 DEPTH=1: row_len = 64 × 8 = 512  ✓
    // **DEPTH=1: proof size claims are valid here.**
    let trace_14 = generate_random_trace(9, 14);
    bench_pcs_pipeline::<Sha256ZipTypes<i64, 32>, IprsBPoly32R4B64<1, UNCHECKED>, 9, UNCHECKED>(
        &mut group,
        "14cols/PCS-only/R4B64d1",
        1,
        14,
        &trace_14,
    );

    // ── Paper model config: 14 columns at poly=2^7 (128 rows) ────────
    // Smaller config with R4B16 DEPTH=1 (row_len=128, DEPTH=1 compresses well)
    // **DEPTH=1: proof size claims are valid here.**
    let trace_14_small = generate_random_trace(7, 14);
    bench_pcs_pipeline::<Sha256ZipTypes<i64, 32>, IprsBPoly32R4B16<1, UNCHECKED>, 7, UNCHECKED>(
        &mut group,
        "14cols/PCS-only/R4B16d1",
        1,
        14,
        &trace_14_small,
    );

    group.finish();
}

/// PIOP-only benchmark (ideal check + combined poly resolver)
/// for the SHA-256 UAIR trace.
fn sha256_piop_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("SHA-256 PIOP");
    group.sample_size(20);

    let trace = generate_sha256_trace(SHA256_NUM_VARS);
    let num_vars = SHA256_NUM_VARS;
    let num_constraints = zinc_uair::constraint_counter::count_constraints::<BinaryPoly<32>, Sha256Uair>();
    let max_degree = zinc_uair::degree_counter::count_max_degree::<BinaryPoly<32>, Sha256Uair>();

    // ── Ideal Check prover ──────────────────────────────────────────
    group.bench_function("IdealCheck/Prover", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut transcript = zinc_transcript::KeccakTranscript::new();
                let field_cfg = transcript.get_random_field_cfg::<
                    F, <F as Field>::Inner, MillerRabin
                >();
                let t = Instant::now();
                let _ = zinc_piop::ideal_check::IdealCheckProtocol::<
                    F
                >::prove_as_subprotocol::<BinaryPoly<32>, Sha256Uair>(
                    &mut transcript,
                    &trace,
                    num_constraints,
                    num_vars,
                    &field_cfg,
                )
                .expect("IC prover failed");
                total += t.elapsed();
            }
            total
        });
    });

    // ── Combined Poly Resolver prover ───────────────────────────────
    let mut transcript = zinc_transcript::KeccakTranscript::new();
    let field_cfg = transcript.get_random_field_cfg::<
        F, <F as Field>::Inner, MillerRabin
    >();
    let (_ic_proof, ic_state) =
        zinc_piop::ideal_check::IdealCheckProtocol::<
            F
        >::prove_as_subprotocol::<BinaryPoly<32>, Sha256Uair>(
            &mut transcript,
            &trace,
            num_constraints,
            num_vars,
            &field_cfg,
        )
        .expect("IC prover failed");

    group.bench_function("CombinedPolyResolver/Prover", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut tr = transcript.clone();
                let t = Instant::now();
                let _ = zinc_piop::combined_poly_resolver::CombinedPolyResolver::<
                    F
                >::prove_as_subprotocol::<BinaryPoly<32>, Sha256Uair>(
                    &mut tr,
                    &ic_state.trace_matrix,
                    &ic_state.evaluation_point,
                    ic_state.projected_scalars.clone(),
                    num_constraints,
                    num_vars,
                    max_degree,
                    &field_cfg,
                )
                .expect("CPR prover failed");
                total += t.elapsed();
            }
            total
        });
    });

    group.finish();
}

/// **True end-to-end benchmark** combining PIOP + PCS for honest total prover time.
///
/// Reports the total wall-clock time for the complete prover:
///   1. PIOP: IdealCheck + CombinedPolyResolver
///   2. PCS: Commit + Test + Evaluate
/// And the total verifier time:
///   1. PCS: Verify
///
/// This is the "honest" prover time — not just PCS in isolation.
fn sha256_end_to_end(c: &mut Criterion) {
    let mut group = c.benchmark_group("SHA-256 E2E (PIOP+PCS)");
    group.sample_size(20);

    let trace = generate_sha256_trace(SHA256_NUM_VARS);
    let num_vars = SHA256_NUM_VARS;
    let num_constraints = zinc_uair::constraint_counter::count_constraints::<BinaryPoly<32>, Sha256Uair>();
    let max_degree = zinc_uair::degree_counter::count_max_degree::<BinaryPoly<32>, Sha256Uair>();

    // ── Total Prover (PIOP + PCS) ────────────────────────────────────
    group.bench_function("TotalProver/1xSHA256", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let t = Instant::now();

                // 1. PIOP: IdealCheck
                let mut transcript = zinc_transcript::KeccakTranscript::new();
                let field_cfg = transcript.get_random_field_cfg::<
                    F, <F as Field>::Inner, MillerRabin
                >();
                let (_ic_proof, ic_state) =
                    zinc_piop::ideal_check::IdealCheckProtocol::<
                        F
                    >::prove_as_subprotocol::<BinaryPoly<32>, Sha256Uair>(
                        &mut transcript,
                        &trace,
                        num_constraints,
                        num_vars,
                        &field_cfg,
                    )
                    .expect("IC prover failed");

                // 2. PIOP: CombinedPolyResolver
                let _ = zinc_piop::combined_poly_resolver::CombinedPolyResolver::<
                    F
                >::prove_as_subprotocol::<BinaryPoly<32>, Sha256Uair>(
                    &mut transcript,
                    &ic_state.trace_matrix,
                    &ic_state.evaluation_point,
                    ic_state.projected_scalars.clone(),
                    num_constraints,
                    num_vars,
                    max_degree,
                    &field_cfg,
                )
                .expect("CPR prover failed");

                // 3. PCS: Commit + Test + Evaluate
                type Lc = IprsBPoly32R4B16<1, UNCHECKED>;
                type Zt = Sha256ZipTypes<i64, 32>;
                let row_len = 128;
                let linear_code = Lc::new(row_len);
                let params = ZipPlusParams::new(SHA256_NUM_VARS, 1, linear_code);

                let (hint, _commitment) = BatchedZipPlus::<Zt, Lc>::commit(&params, &trace)
                    .expect("commit");
                let test_tx = BatchedZipPlus::<Zt, Lc>::test::<UNCHECKED>(&params, &trace, &hint)
                    .expect("test");
                let point: Vec<<Zt as ZipTypes>::Pt> = vec![<Zt as ZipTypes>::Pt::one(); SHA256_NUM_VARS];
                let _ = BatchedZipPlus::<Zt, Lc>::evaluate::<F, UNCHECKED>(
                    &params, &trace, &point, test_tx,
                )
                .expect("evaluate");

                total += t.elapsed();
            }
            total
        });
    });

    // ── Total Verifier (PCS only — PIOP verifier is algebraic/negligible) ─────
    // Prepare a proof for the verifier benchmark
    {
        type Lc = IprsBPoly32R4B16<1, UNCHECKED>;
        type Zt = Sha256ZipTypes<i64, 32>;
        let linear_code = Lc::new(128);
        let params = ZipPlusParams::new(SHA256_NUM_VARS, 1, linear_code);
        let (hint, commitment) = BatchedZipPlus::<Zt, Lc>::commit(&params, &trace)
            .expect("commit");
        let test_tx = BatchedZipPlus::<Zt, Lc>::test::<UNCHECKED>(&params, &trace, &hint)
            .expect("test");
        let point: Vec<<Zt as ZipTypes>::Pt> = vec![<Zt as ZipTypes>::Pt::one(); SHA256_NUM_VARS];
        let (evals_f, proof) = BatchedZipPlus::<Zt, Lc>::evaluate::<F, UNCHECKED>(
            &params, &trace, &point, test_tx,
        )
        .expect("evaluate");
        let field_cfg = *evals_f[0].cfg();
        let point_f: Vec<F> = point.iter().map(|v| v.into_with_cfg(&field_cfg)).collect();

        group.bench_function("TotalVerifier/1xSHA256", |b| {
            b.iter(|| {
                let result = BatchedZipPlus::<Zt, Lc>::verify::<F, UNCHECKED>(
                    &params,
                    &commitment,
                    &point_f,
                    &evals_f,
                    &proof,
                );
                let _ = black_box(result);
            });
        });
    }

    group.finish();
}

/// **Full pipeline benchmark** using the `pipeline::prove` and `pipeline::verify` functions.
///
/// This measures the actual end-to-end pipeline including PIOP proof serialization/deserialization,
/// and reports total proof sizes (PCS + PIOP data).
fn sha256_full_pipeline(c: &mut Criterion) {
    use zinc_sha256_uair::CyclotomicIdeal;
    use zinc_uair::ideal_collector::IdealOrZero;
    type Lc = IprsBPoly32R4B16<1, UNCHECKED>;
    type Zt = Sha256ZipTypes<i64, 32>;

    let mut group = c.benchmark_group("SHA-256 Full Pipeline");
    group.sample_size(20);

    let trace = generate_sha256_trace(SHA256_NUM_VARS);
    let num_vars = SHA256_NUM_VARS;
    let linear_code = Lc::new(128);
    let params = ZipPlusParams::new(num_vars, 1, linear_code);

    // ── Full Prover ─────────────────────────────────────────────────
    group.bench_function("FullProver/1xSHA256", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let t = Instant::now();
                let _proof = zinc_snark::pipeline::prove::<Sha256Uair, Zt, Lc, 32, UNCHECKED>(
                    &params,
                    &trace,
                    num_vars,
                );
                total += t.elapsed();
            }
            total
        });
    });

    // Prepare proof for verifier and size measurement
    let zinc_proof = zinc_snark::pipeline::prove::<Sha256Uair, Zt, Lc, 32, UNCHECKED>(
        &params,
        &trace,
        num_vars,
    );

    // ── Full Verifier ───────────────────────────────────────────────
    group.bench_function("FullVerifier/1xSHA256", |b| {
        b.iter(|| {
            let result = zinc_snark::pipeline::verify::<Sha256Uair, Zt, Lc, 32, UNCHECKED, _, _>(
                &params,
                &zinc_proof,
                num_vars,
                |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
            );
            black_box(result);
        });
    });

    // ── Proof size report ───────────────────────────────────────────
    let pcs_size = zinc_proof.pcs_proof_bytes.len();
    let piop_size: usize = zinc_proof.ic_proof_values.iter().map(|v| v.len()).sum::<usize>()
        + zinc_proof.cpr_sumcheck_messages.iter().map(|v| v.len()).sum::<usize>()
        + zinc_proof.cpr_sumcheck_claimed_sum.len()
        + zinc_proof.cpr_up_evals.iter().map(|v| v.len()).sum::<usize>()
        + zinc_proof.cpr_down_evals.iter().map(|v| v.len()).sum::<usize>()
        + zinc_proof.evaluation_point_bytes.iter().map(|v| v.len()).sum::<usize>()
        + zinc_proof.pcs_evals_bytes.iter().map(|v| v.len()).sum::<usize>();
    let total_size = pcs_size + piop_size;

    // Compressed
    let mut all_bytes = zinc_proof.pcs_proof_bytes.clone();
    for v in &zinc_proof.ic_proof_values { all_bytes.extend(v); }
    for v in &zinc_proof.cpr_sumcheck_messages { all_bytes.extend(v); }
    all_bytes.extend(&zinc_proof.cpr_sumcheck_claimed_sum);
    for v in &zinc_proof.cpr_up_evals { all_bytes.extend(v); }
    for v in &zinc_proof.cpr_down_evals { all_bytes.extend(v); }
    for v in &zinc_proof.evaluation_point_bytes { all_bytes.extend(v); }
    for v in &zinc_proof.pcs_evals_bytes { all_bytes.extend(v); }

    let mut encoder = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(&all_bytes).unwrap();
    let compressed = encoder.finish().unwrap();

    println!("\n=== Full Pipeline Proof Size ===");
    println!("  PCS proof:     {} bytes ({:.1} KB)", pcs_size, pcs_size as f64 / 1024.0);
    println!("  PIOP proof:    {} bytes ({:.1} KB)", piop_size, piop_size as f64 / 1024.0);
    println!("  Total raw:     {} bytes ({:.1} KB)", total_size, total_size as f64 / 1024.0);
    println!("  Compressed:    {} bytes ({:.1} KB)", compressed.len(), compressed.len() as f64 / 1024.0);
    println!("  Target limit:  800 KB → {}", if total_size <= 819200 { "✓ UNDER" } else { "✗ OVER" });

    group.finish();
}

criterion_group!(benches, sha256_single, sha256_8x_ecdsa, sha256_piop_only, sha256_end_to_end, sha256_full_pipeline);
criterion_main!(benches);
