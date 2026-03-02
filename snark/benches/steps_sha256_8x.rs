//! Per-step breakdown benchmarks for the 8×SHA-256 proving stack (no ECDSA).
//!
//! Run with:
//!   cargo bench --bench steps_sha256_8x -p zinc-snark --features "parallel simd asm"

#![allow(clippy::arithmetic_side_effects, clippy::unwrap_used)]

use std::hint::black_box;
use std::marker::PhantomData;
use std::time::{Duration, Instant};

use zinc_utils::peak_mem::MemoryTracker;

use criterion::{criterion_group, criterion_main, Criterion};
use crypto_bigint::U64;
use crypto_primitives::{
    boolean::Boolean,
    crypto_bigint_int::Int,
    crypto_bigint_monty::MontyField,
    crypto_bigint_uint::Uint,
    Field, FixedSemiring, PrimeField,
};
use zinc_poly::mle::DenseMultilinearExtension;
use zinc_poly::univariate::binary::{
    BinaryPoly, BinaryPolyInnerProduct, BinaryPolyWideningMulByScalar,
};
use zinc_poly::univariate::dense::{DensePolyInnerProduct, DensePolynomial};
use zinc_poly::univariate::dynamic::over_field::DynamicPolynomialF;
use zinc_primality::MillerRabin;
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_utils::{
    UNCHECKED,
    from_ref::FromRef,
    inner_product::MBSInnerProduct,
    named::Named,
};
use zip_plus::{
    code::{
        LinearCode,
        iprs::{IprsCode, PnttConfigF2_16R4B64},
    },
    pcs::structs::{ZipPlus, ZipPlusParams, ZipTypes},
};

use zinc_sha256_uair::{Sha256Uair, witness::GenerateWitness};
use zinc_uair::Uair;
use zinc_piop::projections::{
    project_trace_coeffs, project_trace_to_field,
    project_scalars, project_scalars_to_field,
};
use zinc_piop::lookup::{LookupColumnSpec, LookupTableType, LookupWitnessSource, AffineLookupSpec};
use zinc_piop::lookup::{
    BatchedDecompLogupProtocol, group_lookup_specs,
};
use zinc_piop::lookup::pipeline::build_lookup_instance_from_indices_pub;
use zinc_piop::lookup::pipeline::generate_table_and_shifts;
use zinc_piop::sumcheck::multi_degree::{MultiDegreeSumcheck, MultiDegreeSumcheckProof};
use zinc_piop::sumcheck::prover::{NatEvaluatedPolyWithoutConstant, ProverMsg};
use zinc_piop::ideal_check::IdealCheckProtocol;
use zinc_piop::combined_poly_resolver::CombinedPolyResolver;
use zinc_utils::projectable_to_field::ProjectableToField;
use zip_plus::pcs::ZipPlusProof;

// ─── Type definitions ───────────────────────────────────────────────────────

const INT_LIMBS: usize = U64::LIMBS;
type F = MontyField<{ INT_LIMBS * 3 }>;

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
    const NUM_COLUMN_OPENINGS: usize = 147;
    type Eval = BinaryPoly<D_PLUS_ONE>;
    type Cw = DensePolynomial<CwCoeff, D_PLUS_ONE>;
    type Fmod = Uint<{ INT_LIMBS * 3 }>;
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

type IprsBPoly32R4B64<const DEPTH: usize, const CHECK: bool> = IprsCode<
    Sha256ZipTypes<i64, 32>,
    PnttConfigF2_16R4B64<DEPTH>,
    BinaryPolyWideningMulByScalar<i64>,
    CHECK,
>;

// ─── Parameters ─────────────────────────────────────────────────────────────

const SHA256_8X_NUM_VARS: usize = 9;      // 2^9 = 512 rows (8 × 64 SHA rounds)
const SHA256_BATCH_SIZE: usize = 30;       // 30 SHA-256 columns (27 bitpoly + 3 int)
const SHA256_LOOKUP_COL_COUNT: usize = 10; // 10 Q[X] columns need lookup

fn sha256_lookup_specs() -> Vec<LookupColumnSpec> {
    (0..SHA256_LOOKUP_COL_COUNT)
        .map(|i| LookupColumnSpec {
            column_index: i,
            table_type: LookupTableType::BitPoly { width: 32 },
        })
        .collect()
}

fn generate_sha256_trace(num_vars: usize) -> Vec<DenseMultilinearExtension<BinaryPoly<32>>> {
    let mut rng = rand::rng();
    <Sha256Uair as GenerateWitness<BinaryPoly<32>>>::generate_witness(num_vars, &mut rng)
}

// ─── Benchmark ──────────────────────────────────────────────────────────────

/// Measures each main step of the E2E proving stack individually for
/// 8×SHA-256 compressions (**no** ECDSA).
///
/// Steps benchmarked:
///   1. WitnessGen — generate the 30-column BinaryPoly trace (512 rows)
///   2. PCS/Commit — Zip+ commit (Merkle tree construction)
///   3. PIOP/IdealCheck — Ideal Check prover
///   4. PIOP/CPR — Combined Poly Resolver prover
///   5. PIOP/Lookup — GKR batched decomposed LogUp prover
///   6. PCS/Prove — Zip+ prove (test + evaluation combined)
///   7. E2E/Prover — total (pipeline::prove)
///   8. E2E/Verifier — total (pipeline::verify)
fn sha256_8x_stepwise(c: &mut Criterion) {
    use zinc_sha256_uair::CyclotomicIdeal;
    use zinc_uair::ideal_collector::IdealOrZero;
    use zinc_piop::lookup::prove_batched_lookup_with_indices;

    let mem_tracker = MemoryTracker::start();

    let mut group = c.benchmark_group("8xSHA256 Steps");
    group.sample_size(100);

    type ShaZt = Sha256ZipTypes<i64, 32>;
    type ShaLc = IprsBPoly32R4B64<1, UNCHECKED>;

    let sha_lc = ShaLc::new(512);
    let sha_params = ZipPlusParams::<ShaZt, ShaLc>::new(SHA256_8X_NUM_VARS, 1, sha_lc);

    let sha_lookup_specs = sha256_lookup_specs();

    // ── 1. Witness Generation ───────────────────────────────────────
    group.bench_function("WitnessGen", |b| {
        b.iter(|| {
            let trace = generate_sha256_trace(SHA256_8X_NUM_VARS);
            black_box(trace);
        });
    });

    // Pre-generate the trace used by all subsequent steps.
    let sha_trace = generate_sha256_trace(SHA256_8X_NUM_VARS);
    assert_eq!(sha_trace.len(), SHA256_BATCH_SIZE);

    // Build private trace (exclude public columns) for PCS.
    let sha_sig = Sha256Uair::signature();
    let sha_pcs_trace: Vec<_> = sha_trace.iter().enumerate()
        .filter(|(i, _)| !sha_sig.public_columns.contains(i))
        .map(|(_, c)| c.clone()).collect();

    let num_constraints = zinc_uair::constraint_counter::count_constraints::<Sha256Uair>();
    let max_degree = zinc_uair::degree_counter::count_max_degree::<Sha256Uair>();

    // ── 2. PCS Commit ───────────────────────────────────────────────
    group.bench_function("PCS/Commit", |b| {
        b.iter(|| {
            let r = ZipPlus::<ShaZt, ShaLc>::commit(&sha_params, &sha_pcs_trace);
            let _ = black_box(r);
        });
    });

    // ── 3. PIOP / Ideal Check ───────────────────────────────────────
    group.bench_function("PIOP/IdealCheck", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut transcript = zinc_transcript::KeccakTranscript::new();
                let field_cfg = transcript.get_random_field_cfg::<
                    F, <F as Field>::Inner, MillerRabin
                >();
                let projected_trace = project_trace_coeffs::<F, bool, bool, 32>(
                    &sha_trace, &[], &[], &field_cfg,
                );
                let projected_scalars = project_scalars::<F, Sha256Uair>(|scalar| {
                    let one = F::one_with_cfg(&field_cfg);
                    let zero = F::zero_with_cfg(&field_cfg);
                    DynamicPolynomialF::new(
                        scalar.iter().map(|coeff| {
                            if coeff.into_inner() { one.clone() } else { zero.clone() }
                        }).collect::<Vec<_>>()
                    )
                });
                let t = Instant::now();
                let _ = zinc_piop::ideal_check::IdealCheckProtocol::<F>::prove_as_subprotocol::<Sha256Uair>(
                    &mut transcript,
                    &projected_trace,
                    &projected_scalars,
                    num_constraints,
                    SHA256_8X_NUM_VARS,
                    &field_cfg,
                ).expect("IC prover failed");
                total += t.elapsed();
            }
            total
        });
    });

    // ── 4. PIOP / Combined Poly Resolver ────────────────────────────
    let mut transcript_for_cpr = zinc_transcript::KeccakTranscript::new();
    let field_cfg_cpr = transcript_for_cpr.get_random_field_cfg::<
        F, <F as Field>::Inner, MillerRabin
    >();
    let projected_trace_cpr = project_trace_coeffs::<F, bool, bool, 32>(
        &sha_trace, &[], &[], &field_cfg_cpr,
    );
    let projected_scalars_cpr = project_scalars::<F, Sha256Uair>(|scalar| {
        let one = F::one_with_cfg(&field_cfg_cpr);
        let zero = F::zero_with_cfg(&field_cfg_cpr);
        DynamicPolynomialF::new(
            scalar.iter().map(|coeff| {
                if coeff.into_inner() { one.clone() } else { zero.clone() }
            }).collect::<Vec<_>>()
        )
    });
    let (_ic_proof_cpr, ic_state_cpr) =
        zinc_piop::ideal_check::IdealCheckProtocol::<F>::prove_as_subprotocol::<Sha256Uair>(
            &mut transcript_for_cpr,
            &projected_trace_cpr,
            &projected_scalars_cpr,
            num_constraints,
            SHA256_8X_NUM_VARS,
            &field_cfg_cpr,
        ).expect("IC prover failed");

    let projecting_elem_cpr: F = transcript_for_cpr.get_field_challenge(&field_cfg_cpr);
    let field_trace_cpr = project_trace_to_field::<F, 32>(
        &sha_trace, &[], &[], &projecting_elem_cpr,
    );
    let field_projected_scalars_cpr =
        project_scalars_to_field(projected_scalars_cpr.clone(), &projecting_elem_cpr)
            .expect("scalar projection failed");

    group.bench_function("PIOP/CPR", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut tr = transcript_for_cpr.clone();
                let t = Instant::now();
                let _ = zinc_piop::combined_poly_resolver::CombinedPolyResolver::<F>::prove_as_subprotocol::<Sha256Uair>(
                    &mut tr,
                    field_trace_cpr.clone(),
                    &ic_state_cpr.evaluation_point,
                    &field_projected_scalars_cpr,
                    num_constraints,
                    SHA256_8X_NUM_VARS,
                    max_degree,
                    &field_cfg_cpr,
                ).expect("CPR prover failed");
                total += t.elapsed();
            }
            total
        });
    });

    // ── 5. PIOP / Lookup ────────────────────────────────────────────
    {
        let mut transcript_lk = transcript_for_cpr.clone();
        let (_cpr_proof_lk, _cpr_state_lk) =
            zinc_piop::combined_poly_resolver::CombinedPolyResolver::<F>::prove_as_subprotocol::<Sha256Uair>(
                &mut transcript_lk,
                field_trace_cpr.clone(),
                &ic_state_cpr.evaluation_point,
                &field_projected_scalars_cpr,
                num_constraints,
                SHA256_8X_NUM_VARS,
                max_degree,
                &field_cfg_cpr,
            ).expect("CPR prover failed");

        let mut needed: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
        for spec in &sha_lookup_specs {
            let next_id = needed.len();
            needed.entry(spec.column_index).or_insert(next_id);
        }
        let mut columns: Vec<Vec<F>> = Vec::with_capacity(needed.len());
        let mut raw_indices: Vec<Vec<usize>> = Vec::with_capacity(needed.len());
        for &orig_idx in needed.keys() {
            let col_f: Vec<F> = field_trace_cpr[orig_idx]
                .iter()
                .map(|inner| F::new_unchecked_with_cfg(inner.clone(), &field_cfg_cpr))
                .collect();
            columns.push(col_f);
            let col_idx: Vec<usize> = sha_trace[orig_idx]
                .iter()
                .map(|bp| {
                    let mut idx = 0usize;
                    for (j, coeff) in bp.iter().enumerate() {
                        if coeff.into_inner() { idx |= 1usize << j; }
                    }
                    idx
                })
                .collect();
            raw_indices.push(col_idx);
        }
        let index_map: std::collections::BTreeMap<usize, usize> = needed.keys()
            .enumerate().map(|(new, &orig)| (orig, new)).collect();
        let remapped_specs: Vec<LookupColumnSpec> = sha_lookup_specs.iter()
            .map(|s| LookupColumnSpec { column_index: index_map[&s.column_index], table_type: s.table_type.clone() })
            .collect();

        group.bench_function("PIOP/Lookup", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let mut tr = transcript_lk.clone();
                    let t = Instant::now();
                    let _ = prove_batched_lookup_with_indices(
                        &mut tr, &columns, &raw_indices, &remapped_specs,
                        &projecting_elem_cpr, &field_cfg_cpr,
                    ).expect("lookup prover failed");
                    total += t.elapsed();
                }
                total
            });
        });
    }

    // ── 6. PCS Prove ────────────────────────────────────────────────
    // Use the CPR evaluation point directly as the PCS evaluation point (r_PCS = r_CPR).
    let sha_pcs_point: Vec<F> = {
        let mut tr = transcript_for_cpr.clone();
        let (_, cpr_state) =
            zinc_piop::combined_poly_resolver::CombinedPolyResolver::<F>::prove_as_subprotocol::<Sha256Uair>(
                &mut tr,
                field_trace_cpr.clone(),
                &ic_state_cpr.evaluation_point,
                &field_projected_scalars_cpr,
                num_constraints,
                SHA256_8X_NUM_VARS,
                max_degree,
                &field_cfg_cpr,
            ).expect("CPR prover failed");
        cpr_state.evaluation_point
    };

    let (sha_hint, _sha_comm) = ZipPlus::<ShaZt, ShaLc>::commit(&sha_params, &sha_pcs_trace)
        .expect("commit");

    group.bench_function("PCS/Prove", |b| {
        b.iter(|| {
            let r = ZipPlus::<ShaZt, ShaLc>::prove::<F, UNCHECKED>(
                &sha_params, &sha_pcs_trace, &sha_pcs_point, &sha_hint,
            );
            let _ = black_box(r);
        });
    });

    // ── 7. E2E Total Prover ─────────────────────────────────────────
    group.bench_function("E2E/Prover", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let t = Instant::now();
                let _ = zinc_snark::pipeline::prove::<Sha256Uair, ShaZt, ShaLc, 32, UNCHECKED>(
                    &sha_params, &sha_trace, SHA256_8X_NUM_VARS, &sha_lookup_specs,
                );
                total += t.elapsed();
            }
            total
        });
    });

    // ── 8. E2E Total Verifier ───────────────────────────────────────
    let sha_proof = zinc_snark::pipeline::prove::<Sha256Uair, ShaZt, ShaLc, 32, UNCHECKED>(
        &sha_params, &sha_trace, SHA256_8X_NUM_VARS, &sha_lookup_specs,
    );

    let sha_sig_pub = Sha256Uair::signature();
    let sha_public_cols: Vec<_> = sha_sig_pub.public_columns.iter()
        .map(|&i| sha_trace[i].clone()).collect();

    group.bench_function("E2E/Verifier", |b| {
        b.iter(|| {
            let r = zinc_snark::pipeline::verify::<Sha256Uair, ShaZt, ShaLc, 32, UNCHECKED, _, _>(
                &sha_params, &sha_proof, SHA256_8X_NUM_VARS,
                |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
                &sha_public_cols,
            );
            black_box(r);
        });
    });

    // ── Timing breakdown summary ────────────────────────────────────
    eprintln!("\n=== 8xSHA256 (no ECDSA) Pipeline Timing ===");
    eprintln!("  IC={:?}, CPR={:?}, Lookup={:?}, PCS(commit={:?}, prove={:?}), total={:?}",
        sha_proof.timing.ideal_check,
        sha_proof.timing.combined_poly_resolver,
        sha_proof.timing.lookup,
        sha_proof.timing.pcs_commit,
        sha_proof.timing.pcs_prove,
        sha_proof.timing.total,
    );

    let mem_snapshot = mem_tracker.stop();
    eprintln!("  {mem_snapshot}");

    group.finish();
}

criterion_group!(benches, sha256_8x_stepwise);
criterion_main!(benches);
