//! Per-step breakdown benchmarks for 8×SHA-256 using the **dual-ring** pipeline.
//!
//! Exercises both `Sha256UairBp` (16 F₂[X] constraints, max_degree=1) and
//! `Sha256UairQx` (3 Q[X] carry constraints, max_degree=2) through the
//! shared-challenge dual-ring pipeline (`prove_dual_ring` / `verify_dual_ring`).
//!
//! The dual-ring pipeline does NOT use lookup. It performs:
//! - Shared PCS commit over the BinaryPoly trace
//! - IC₁ (BinaryPoly) + IC₂ (Q[X]) at a shared evaluation point
//! - Batched multi-degree sumcheck (group 0 = BP CPR, group 1 = QX CPR)
//! - QX shift sumcheck
//! - PCS prove at the CPR evaluation point
//!
//! Also benchmarks single-ring `prove`/`verify` for comparison.
//!
//! Run with:
//!   cargo bench --bench steps_sha256_8x_dual_ring -p zinc-snark --features "parallel simd asm"

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

use zinc_sha256_uair::{
    Sha256Uair, Sha256UairQx, CyclotomicIdeal,
    convert_trace_to_qx,
    witness::GenerateWitness,
};
use zinc_uair::Uair;
use zinc_uair::ideal_collector::IdealOrZero;
use zinc_piop::projections::{
    project_trace_to_field,
    project_scalars, project_scalars_to_field,
};
use zinc_piop::lookup::{LookupColumnSpec, LookupTableType};
use zinc_piop::ideal_check::IdealCheckProtocol;
use zinc_piop::combined_poly_resolver::CombinedPolyResolver;


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
    const NUM_COLUMN_OPENINGS: usize = 131;
    const GRINDING_BITS: usize = 8;
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
const SHA256_LOOKUP_COL_COUNT: usize = 10; // 10 Q[X] columns need lookup (for single-ring comparison)

fn sha256_lookup_specs() -> Vec<LookupColumnSpec> {
    (0..SHA256_LOOKUP_COL_COUNT)
        .map(|i| LookupColumnSpec {
            column_index: i,
            table_type: LookupTableType::BitPoly { width: 32, chunk_width: None },
        })
        .collect()
}

fn generate_sha256_trace(num_vars: usize) -> Vec<DenseMultilinearExtension<BinaryPoly<32>>> {
    let mut rng = rand::rng();
    <Sha256Uair as GenerateWitness<BinaryPoly<32>>>::generate_witness(num_vars, &mut rng)
}


// ─── Benchmark ──────────────────────────────────────────────────────────────

/// Measures the dual-ring pipeline (both F₂[X] and Q[X] constraints) versus
/// the single-ring pipeline (F₂[X] only).
///
/// Steps benchmarked:
///   1.  WitnessGen — generate the 30-column BinaryPoly trace (512 rows)
///   2.  TraceConvert — convert BinaryPoly<32> → DensePolynomial<i64, 64>
///   3.  PCS/Commit — Zip+ commit (Merkle tree construction)
///   4.  PIOP/FieldSetup — transcript init + random field config
///   5.  PIOP/BP IdealCheck — BinaryPoly Ideal Check prover (MLE-first)
///   6.  PIOP/BP Project CPR — project scalars and trace for BP CPR
///   7.  PIOP/BP CPR — BinaryPoly Combined Poly Resolver
///   8.  PCS/Prove — Zip+ prove (test + evaluation)
///   9.  E2E/DualRingProver — full dual-ring pipeline prover
///  10.  E2E/DualRingVerifier — full dual-ring pipeline verifier
///  11.  E2E/SingleRingProver — single-ring pipeline prover (comparison)
///  12.  E2E/SingleRingVerifier — single-ring pipeline verifier (comparison)
///
/// Also reports:
///  - Dual-ring timing breakdown (IC, CPR, PCS)
///  - Dual-ring proof size breakdown (raw + deflate-compressed)
///  - Single-ring proof size (for comparison)
///  - Peak memory usage
fn sha256_8x_dual_ring_stepwise(c: &mut Criterion) {

    let mem_tracker = MemoryTracker::start();

    let mut group = c.benchmark_group("8xSHA256 Dual-Ring Steps");
    group.sample_size(10);

    type ShaZt = Sha256ZipTypes<i64, 32>;
    type ShaLc = IprsBPoly32R4B64<1, UNCHECKED>;

    let sha_lc = ShaLc::new(512);
    let sha_params = ZipPlusParams::<ShaZt, ShaLc>::new(SHA256_8X_NUM_VARS, 1, sha_lc);

    let sha_lookup_specs = sha256_lookup_specs();

    let num_vars = SHA256_8X_NUM_VARS;

    let bp_num_constraints = zinc_uair::constraint_counter::count_constraints::<Sha256Uair>();
    let bp_max_degree = zinc_uair::degree_counter::count_max_degree::<Sha256Uair>();
    let _qx_num_constraints = zinc_uair::constraint_counter::count_constraints::<Sha256UairQx>();
    let _qx_max_degree = zinc_uair::degree_counter::count_max_degree::<Sha256UairQx>();

    // ── 1. Witness Generation ───────────────────────────────────────
    group.bench_function("WitnessGen", |b| {
        b.iter(|| {
            let trace = generate_sha256_trace(num_vars);
            black_box(trace);
        });
    });

    // Pre-generate the trace used by all subsequent steps.
    let sha_trace = generate_sha256_trace(num_vars);
    assert_eq!(sha_trace.len(), SHA256_BATCH_SIZE);

    // ── 2. Trace Conversion (BinaryPoly<32> → DensePolynomial<i64, 64>) ──
    group.bench_function("TraceConvert", |b| {
        b.iter(|| {
            let qx = convert_trace_to_qx(&sha_trace);
            black_box(qx);
        });
    });

    // Build private trace (exclude public columns) for PCS.
    let sha_sig = Sha256Uair::signature();
    let sha_excluded = sha_sig.pcs_excluded_columns();
    let sha_pcs_trace: Vec<_> = sha_trace.iter().enumerate()
        .filter(|(i, _)| !sha_excluded.contains(i))
        .map(|(_, c)| c.clone()).collect();

    // ── 3. PCS Commit ───────────────────────────────────────────────
    group.bench_function("PCS/Commit", |b| {
        b.iter(|| {
            let r = ZipPlus::<ShaZt, ShaLc>::commit(&sha_params, &sha_pcs_trace);
            let _ = black_box(r);
        });
    });

    // ── 4. PIOP / Field Setup ───────────────────────────────────────
    group.bench_function("PIOP/FieldSetup", |b| {
        b.iter(|| {
            let mut transcript = zinc_transcript::KeccakTranscript::new();
            let field_cfg = transcript.get_random_field_cfg::<
                F, <F as Field>::Inner, MillerRabin
            >();
            black_box(field_cfg);
        });
    });

    // ── 5. PIOP / BP Ideal Check (MLE-first, max_degree == 1) ────────
    assert_eq!(bp_max_degree, 1, "Sha256UairBp should have max_degree == 1 (MLE-first path)");
    group.bench_function("PIOP/BP IdealCheck", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut transcript = zinc_transcript::KeccakTranscript::new();
                let field_cfg = transcript.get_random_field_cfg::<
                    F, <F as Field>::Inner, MillerRabin
                >();
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
                let _ = IdealCheckProtocol::<F>::prove_mle_first::<Sha256Uair, 32>(
                    &mut transcript,
                    &sha_trace,
                    &projected_scalars,
                    bp_num_constraints,
                    num_vars,
                    &field_cfg,
                ).expect("BP Ideal Check prover failed");
                total += t.elapsed();
            }
            total
        });
    });

    // ── 6–7. PIOP / BP CPR (project + sumcheck) ─────────────────────
    // Set up CPR state for BP side (used by both the individual benchmark
    // and the PCS/Prove step).
    let mut transcript_for_cpr = zinc_transcript::KeccakTranscript::new();
    let field_cfg_cpr = transcript_for_cpr.get_random_field_cfg::<
        F, <F as Field>::Inner, MillerRabin
    >();
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
        IdealCheckProtocol::<F>::prove_mle_first::<Sha256Uair, 32>(
            &mut transcript_for_cpr,
            &sha_trace,
            &projected_scalars_cpr,
            bp_num_constraints,
            num_vars,
            &field_cfg_cpr,
        ).expect("BP IC failed");

    let projecting_elem_cpr: F = transcript_for_cpr.get_field_challenge(&field_cfg_cpr);

    group.bench_function("PIOP/BP Project CPR", |b| {
        b.iter(|| {
            let fproj_scalars =
                project_scalars_to_field(projected_scalars_cpr.clone(), &projecting_elem_cpr)
                    .expect("scalar projection");
            let field_trace = project_trace_to_field::<F, 32>(
                &sha_trace, &[], &[], &projecting_elem_cpr,
            );
            black_box((&fproj_scalars, &field_trace));
        });
    });

    let field_trace_cpr = project_trace_to_field::<F, 32>(
        &sha_trace, &[], &[], &projecting_elem_cpr,
    );
    let field_projected_scalars_cpr =
        project_scalars_to_field(projected_scalars_cpr.clone(), &projecting_elem_cpr)
            .expect("scalar projection failed");

    group.bench_function("PIOP/BP CPR", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut tr = transcript_for_cpr.clone();
                let t = Instant::now();
                let _ = CombinedPolyResolver::<F>::prove_as_subprotocol::<Sha256Uair>(
                    &mut tr,
                    field_trace_cpr.clone(),
                    &ic_state_cpr.evaluation_point,
                    &field_projected_scalars_cpr,
                    bp_num_constraints,
                    num_vars,
                    bp_max_degree,
                    &field_cfg_cpr,
                ).expect("BP CPR prover failed");
                total += t.elapsed();
            }
            total
        });
    });

    // ── 8. PCS Prove ────────────────────────────────────────────────
    let sha_pcs_point: Vec<F> = {
        let mut tr = transcript_for_cpr.clone();
        let (_, cpr_state) =
            CombinedPolyResolver::<F>::prove_as_subprotocol::<Sha256Uair>(
                &mut tr,
                field_trace_cpr.clone(),
                &ic_state_cpr.evaluation_point,
                &field_projected_scalars_cpr,
                bp_num_constraints,
                num_vars,
                bp_max_degree,
                &field_cfg_cpr,
            ).expect("BP CPR failed");
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

    // ══════════════════════════════════════════════════════════════════
    // ── E2E Prover / Verifier Benchmarks ────────────────────────────
    // ══════════════════════════════════════════════════════════════════

    // ── 9. E2E Dual-Ring Prover ─────────────────────────────────────
    group.bench_function("E2E/DualRingProver", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let t = Instant::now();
                let _ = zinc_snark::pipeline::prove_dual_ring::<
                    Sha256Uair,
                    Sha256UairQx,
                    ShaZt, ShaLc,
                    32, 64,
                    UNCHECKED,
                    _,
                >(
                    &sha_params,
                    &sha_trace,
                    num_vars,
                    convert_trace_to_qx,
                );
                total += t.elapsed();
            }
            total
        });
    });

    // ── 10. E2E Dual-Ring Verifier ──────────────────────────────────
    let dual_ring_proof = zinc_snark::pipeline::prove_dual_ring::<
        Sha256Uair,
        Sha256UairQx,
        ShaZt, ShaLc,
        32, 64,
        UNCHECKED,
        _,
    >(
        &sha_params,
        &sha_trace,
        num_vars,
        convert_trace_to_qx,
    );

    group.bench_function("E2E/DualRingVerifier", |b| {
        b.iter(|| {
            let r = zinc_snark::pipeline::verify_dual_ring::<
                Sha256Uair,
                Sha256UairQx,
                ShaZt, ShaLc,
                32, 64,
                UNCHECKED,
                zinc_snark::pipeline::TrivialIdeal,
                _,
            >(
                &sha_params,
                &dual_ring_proof,
                num_vars,
                |_ideal: &IdealOrZero<_>| zinc_snark::pipeline::TrivialIdeal,
            );
            black_box(r);
        });
    });

    // ── 11. E2E Single-Ring Prover (comparison) ─────────────────────
    group.bench_function("E2E/SingleRingProver", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let t = Instant::now();
                let _ = zinc_snark::pipeline::prove::<Sha256Uair, ShaZt, ShaLc, 32, UNCHECKED>(
                    &sha_params, &sha_trace, num_vars, &sha_lookup_specs,
                );
                total += t.elapsed();
            }
            total
        });
    });

    // ── 12. E2E Single-Ring Verifier (comparison) ───────────────────
    let single_ring_proof = zinc_snark::pipeline::prove::<Sha256Uair, ShaZt, ShaLc, 32, UNCHECKED>(
        &sha_params, &sha_trace, num_vars, &sha_lookup_specs,
    );

    let sha_public_cols: Vec<_> = sha_sig.public_columns.iter()
        .map(|&i| sha_trace[i].clone()).collect();

    group.bench_function("E2E/SingleRingVerifier", |b| {
        b.iter(|| {
            let r = zinc_snark::pipeline::verify::<
                Sha256Uair, ShaZt, ShaLc, 32, UNCHECKED, _, _,
            >(
                &sha_params, &single_ring_proof, num_vars,
                |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
                &sha_public_cols,
            );
            black_box(r);
        });
    });

    // ══════════════════════════════════════════════════════════════════
    // ── Timing Breakdown ────────────────────────────────────────────
    // ══════════════════════════════════════════════════════════════════

    eprintln!("\n=== Dual-Ring Pipeline Timing ===");
    eprintln!("  IC (BP+QX):      {:?}", dual_ring_proof.timing.ideal_check);
    eprintln!("  CPR (batched):   {:?}", dual_ring_proof.timing.combined_poly_resolver);
    eprintln!("  PCS commit:      {:?}", dual_ring_proof.timing.pcs_commit);
    eprintln!("  PCS prove:       {:?}", dual_ring_proof.timing.pcs_prove);
    eprintln!("  Total:           {:?}", dual_ring_proof.timing.total);

    eprintln!("\n=== Single-Ring Pipeline Timing ===");
    eprintln!("  IC:              {:?}", single_ring_proof.timing.ideal_check);
    eprintln!("  CPR:             {:?}", single_ring_proof.timing.combined_poly_resolver);
    eprintln!("  Lookup:          {:?}", single_ring_proof.timing.lookup);
    eprintln!("  PCS commit:      {:?}", single_ring_proof.timing.pcs_commit);
    eprintln!("  PCS prove:       {:?}", single_ring_proof.timing.pcs_prove);
    eprintln!("  Total:           {:?}", single_ring_proof.timing.total);

    // ══════════════════════════════════════════════════════════════════
    // ── Proof Size Breakdown ────────────────────────────────────────
    // ══════════════════════════════════════════════════════════════════

    {
        use zinc_snark::pipeline::FIELD_LIMBS;

        let fe_bytes = <Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;

        fn write_fe(buf: &mut Vec<u8>, f: &zinc_snark::pipeline::PiopField) {
            use zinc_snark::pipeline::FIELD_LIMBS;
            let sz = <Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;
            let start = buf.len();
            buf.resize(start + sz, 0);
            f.inner().write_transcription_bytes(&mut buf[start..]);
        }

        // ── Dual-ring proof size decomposition ──────────────────────

        // PCS
        let dr_pcs = dual_ring_proof.pcs_proof_bytes.len();

        // IC (both UAIRs)
        let dr_bp_ic: usize = dual_ring_proof.bp_ic_proof_values.iter().map(|v| v.len()).sum();
        let dr_qx_ic: usize = dual_ring_proof.qx_ic_proof_values.iter().map(|v| v.len()).sum();
        let dr_ic_total = dr_bp_ic + dr_qx_ic;

        // Multi-degree sumcheck messages + claimed sums
        let dr_md_msgs: usize = dual_ring_proof.md_group_messages.iter()
            .flat_map(|group| group.iter())
            .map(|msg| msg.len())
            .sum();
        let dr_md_sums: usize = dual_ring_proof.md_claimed_sums.iter().map(|v| v.len()).sum();
        let dr_md_total = dr_md_msgs + dr_md_sums;

        // CPR up/down evals (both UAIRs)
        let dr_bp_cpr_up: usize = dual_ring_proof.bp_cpr_up_evals.iter().map(|v| v.len()).sum();
        let dr_bp_cpr_dn: usize = dual_ring_proof.bp_cpr_down_evals.iter().map(|v| v.len()).sum();
        let dr_qx_cpr_up: usize = dual_ring_proof.qx_cpr_up_evals.iter().map(|v| v.len()).sum();
        let dr_qx_cpr_dn: usize = dual_ring_proof.qx_cpr_down_evals.iter().map(|v| v.len()).sum();
        let dr_cpr_evals = dr_bp_cpr_up + dr_bp_cpr_dn + dr_qx_cpr_up + dr_qx_cpr_dn;

        // QX shift sumcheck
        let dr_shift_sc: usize = dual_ring_proof.qx_shift_sumcheck.as_ref().map_or(0, |sc| {
            let rounds: usize = sc.rounds.iter().map(|v| v.len()).sum();
            let finals: usize = sc.v_finals.iter().map(|v| v.len()).sum();
            rounds + finals
        });

        // Evaluation point + PCS evals
        let dr_eval_pt: usize = dual_ring_proof.evaluation_point_bytes.iter().map(|v| v.len()).sum();
        let dr_pcs_eval: usize = dual_ring_proof.pcs_evals_bytes.iter().map(|v| v.len()).sum();

        let dr_piop_total = dr_ic_total + dr_md_total + dr_cpr_evals
            + dr_shift_sc + dr_eval_pt + dr_pcs_eval;
        let dr_total_raw = dr_pcs + dr_piop_total;

        // Build serialized byte buffer for compression
        let mut dr_all_bytes: Vec<u8> = Vec::with_capacity(dr_total_raw);
        dr_all_bytes.extend(&dual_ring_proof.pcs_proof_bytes);
        for v in &dual_ring_proof.bp_ic_proof_values { dr_all_bytes.extend(v); }
        for v in &dual_ring_proof.qx_ic_proof_values { dr_all_bytes.extend(v); }
        for group in &dual_ring_proof.md_group_messages {
            for msg in group { dr_all_bytes.extend(msg); }
        }
        for v in &dual_ring_proof.md_claimed_sums { dr_all_bytes.extend(v); }
        for v in &dual_ring_proof.bp_cpr_up_evals   { dr_all_bytes.extend(v); }
        for v in &dual_ring_proof.bp_cpr_down_evals  { dr_all_bytes.extend(v); }
        for v in &dual_ring_proof.qx_cpr_up_evals   { dr_all_bytes.extend(v); }
        for v in &dual_ring_proof.qx_cpr_down_evals  { dr_all_bytes.extend(v); }
        if let Some(ref sc) = dual_ring_proof.qx_shift_sumcheck {
            for v in &sc.rounds  { dr_all_bytes.extend(v); }
            for v in &sc.v_finals { dr_all_bytes.extend(v); }
        }
        for v in &dual_ring_proof.evaluation_point_bytes { dr_all_bytes.extend(v); }
        for v in &dual_ring_proof.pcs_evals_bytes { dr_all_bytes.extend(v); }

        let dr_compressed = {
            use std::io::Write;
            let mut enc = flate2::write::DeflateEncoder::new(
                Vec::new(), flate2::Compression::default(),
            );
            enc.write_all(&dr_all_bytes).unwrap();
            enc.finish().unwrap()
        };

        eprintln!("\n=== Dual-Ring Proof Size ===");
        eprintln!("  PCS:              {:>6} B  ({:.1} KB)", dr_pcs, dr_pcs as f64 / 1024.0);
        eprintln!("  BP IC:            {:>6} B", dr_bp_ic);
        eprintln!("  QX IC:            {:>6} B", dr_qx_ic);
        eprintln!("  IC total:         {:>6} B", dr_ic_total);
        eprintln!("  MD sumcheck:      {:>6} B  (msgs={dr_md_msgs}, sums={dr_md_sums})", dr_md_total);
        eprintln!("  MD degrees:       {:?}", dual_ring_proof.md_degrees);
        eprintln!("  BP CPR evals:     {:>6} B  (up={dr_bp_cpr_up}, down={dr_bp_cpr_dn})",
            dr_bp_cpr_up + dr_bp_cpr_dn);
        eprintln!("  QX CPR evals:     {:>6} B  (up={dr_qx_cpr_up}, down={dr_qx_cpr_dn})",
            dr_qx_cpr_up + dr_qx_cpr_dn);
        eprintln!("  QX Shift SC:      {:>6} B", dr_shift_sc);
        eprintln!("  Eval point:       {:>6} B", dr_eval_pt);
        eprintln!("  PCS evals:        {:>6} B", dr_pcs_eval);
        eprintln!("  ─────────────────────────");
        eprintln!("  PIOP total:       {:>6} B  ({:.1} KB)", dr_piop_total, dr_piop_total as f64 / 1024.0);
        eprintln!("  Total raw:        {:>6} B  ({:.1} KB)", dr_total_raw, dr_total_raw as f64 / 1024.0);
        eprintln!("  Compressed:       {:>6} B  ({:.1} KB, {:.1}x ratio)",
            dr_compressed.len(), dr_compressed.len() as f64 / 1024.0,
            dr_all_bytes.len() as f64 / dr_compressed.len() as f64);

        // ── Single-ring proof size (for comparison) ─────────────────
        use zinc_snark::pipeline::LookupProofData;

        let sr_pcs = single_ring_proof.pcs_proof_bytes.len();
        let sr_ic: usize = single_ring_proof.ic_proof_values.iter().map(|v| v.len()).sum();
        let sr_cpr_msg: usize = single_ring_proof.cpr_sumcheck_messages.iter().map(|v| v.len()).sum();
        let sr_cpr_sum = single_ring_proof.cpr_sumcheck_claimed_sum.len();
        let sr_cpr_sc = sr_cpr_msg + sr_cpr_sum;
        let sr_cpr_up: usize = single_ring_proof.cpr_up_evals.iter().map(|v| v.len()).sum();
        let sr_cpr_dn: usize = single_ring_proof.cpr_down_evals.iter().map(|v| v.len()).sum();
        let sr_lookup: usize = match &single_ring_proof.lookup_proof {
            Some(LookupProofData::Classic(proof)) => {
                let mut t = 0usize;
                for gp in &proof.group_proofs {
                    let m: usize = gp.aggregated_multiplicities.iter().map(|v| v.len()).sum();
                    let w: usize = gp.chunk_inverse_witnesses.iter()
                        .flat_map(|o| o.iter()).map(|i| i.len()).sum();
                    t += (m + w + gp.inverse_table.len()) * fe_bytes;
                }
                t
            }
            _ => 0,
        };
        let sr_shift_sc: usize = single_ring_proof.shift_sumcheck.as_ref().map_or(0, |sc| {
            let rounds: usize = sc.rounds.iter().map(|v| v.len()).sum();
            let finals: usize = sc.v_finals.iter().map(|v| v.len()).sum();
            rounds + finals
        });
        let sr_eval_pt: usize = single_ring_proof.evaluation_point_bytes.iter().map(|v| v.len()).sum();
        let sr_pcs_eval: usize = single_ring_proof.pcs_evals_bytes.iter().map(|v| v.len()).sum();
        let sr_piop_total = sr_ic + sr_cpr_sc + (sr_cpr_up + sr_cpr_dn)
            + sr_lookup + sr_shift_sc + sr_eval_pt + sr_pcs_eval;
        let sr_total_raw = sr_pcs + sr_piop_total;

        let mut sr_all_bytes = Vec::with_capacity(sr_total_raw);
        sr_all_bytes.extend(&single_ring_proof.pcs_proof_bytes);
        for v in &single_ring_proof.ic_proof_values { sr_all_bytes.extend(v); }
        for v in &single_ring_proof.cpr_sumcheck_messages { sr_all_bytes.extend(v); }
        sr_all_bytes.extend(&single_ring_proof.cpr_sumcheck_claimed_sum);
        for v in &single_ring_proof.cpr_up_evals { sr_all_bytes.extend(v); }
        for v in &single_ring_proof.cpr_down_evals { sr_all_bytes.extend(v); }
        if let Some(LookupProofData::Classic(ref proof)) = single_ring_proof.lookup_proof {
            for gp in &proof.group_proofs {
                for v in &gp.aggregated_multiplicities {
                    for f in v { write_fe(&mut sr_all_bytes, f); }
                }
                for outer in &gp.chunk_inverse_witnesses {
                    for inner in outer {
                        for f in inner { write_fe(&mut sr_all_bytes, f); }
                    }
                }
                for f in &gp.inverse_table { write_fe(&mut sr_all_bytes, f); }
            }
        }
        if let Some(ref sc) = single_ring_proof.shift_sumcheck {
            for v in &sc.rounds { sr_all_bytes.extend(v); }
            for v in &sc.v_finals { sr_all_bytes.extend(v); }
        }
        for v in &single_ring_proof.evaluation_point_bytes { sr_all_bytes.extend(v); }
        for v in &single_ring_proof.pcs_evals_bytes { sr_all_bytes.extend(v); }

        let sr_compressed = {
            use std::io::Write;
            let mut enc = flate2::write::DeflateEncoder::new(
                Vec::new(), flate2::Compression::default(),
            );
            enc.write_all(&sr_all_bytes).unwrap();
            enc.finish().unwrap()
        };

        eprintln!("\n=== Single-Ring Proof Size ===");
        eprintln!("  PCS:              {:>6} B  ({:.1} KB)", sr_pcs, sr_pcs as f64 / 1024.0);
        eprintln!("  IC:               {:>6} B", sr_ic);
        eprintln!("  CPR sumcheck:     {:>6} B  (msgs={sr_cpr_msg}, sum={sr_cpr_sum})", sr_cpr_sc);
        eprintln!("  CPR evals:        {:>6} B  (up={sr_cpr_up}, down={sr_cpr_dn})", sr_cpr_up + sr_cpr_dn);
        eprintln!("  Lookup:           {:>6} B  ({:.1} KB)", sr_lookup, sr_lookup as f64 / 1024.0);
        eprintln!("  Shift SC:         {:>6} B", sr_shift_sc);
        eprintln!("  Eval point:       {:>6} B", sr_eval_pt);
        eprintln!("  PCS evals:        {:>6} B", sr_pcs_eval);
        eprintln!("  ─────────────────────────");
        eprintln!("  PIOP total:       {:>6} B  ({:.1} KB)", sr_piop_total, sr_piop_total as f64 / 1024.0);
        eprintln!("  Total raw:        {:>6} B  ({:.1} KB)", sr_total_raw, sr_total_raw as f64 / 1024.0);
        eprintln!("  Compressed:       {:>6} B  ({:.1} KB, {:.1}x ratio)",
            sr_compressed.len(), sr_compressed.len() as f64 / 1024.0,
            sr_all_bytes.len() as f64 / sr_compressed.len() as f64);

        // ── Comparison ──────────────────────────────────────────────
        eprintln!("\n=== Dual-Ring vs Single-Ring Comparison ===");
        eprintln!("  {:30}  {:>8}  {:>8}", "Component", "DualRing", "SingleRing");
        eprintln!("  {}", "─".repeat(50));
        eprintln!("  {:30}  {:>8}  {:>8}", "PCS", dr_pcs, sr_pcs);
        eprintln!("  {:30}  {:>8}  {:>8}", "IC", dr_ic_total, sr_ic);
        eprintln!("  {:30}  {:>8}  {:>8}", "CPR/MD sumcheck", dr_md_total, sr_cpr_sc);
        eprintln!("  {:30}  {:>8}  {:>8}", "CPR evals", dr_cpr_evals, sr_cpr_up + sr_cpr_dn);
        eprintln!("  {:30}  {:>8}  {:>8}", "Lookup", 0, sr_lookup);
        eprintln!("  {:30}  {:>8}  {:>8}", "Shift SC", dr_shift_sc, sr_shift_sc);
        eprintln!("  {:30}  {:>8}  {:>8}", "Eval point + PCS evals",
            dr_eval_pt + dr_pcs_eval, sr_eval_pt + sr_pcs_eval);
        eprintln!("  {}", "─".repeat(50));
        eprintln!("  {:30}  {:>8}  {:>8}", "Total raw (B)", dr_total_raw, sr_total_raw);
        eprintln!("  {:30}  {:>7.1}K  {:>7.1}K", "Total raw (KB)",
            dr_total_raw as f64 / 1024.0, sr_total_raw as f64 / 1024.0);
        let raw_diff = dr_total_raw as i64 - sr_total_raw as i64;
        eprintln!("  Dual-ring overhead (raw):     {:+} B  ({:+.1} KB)",
            raw_diff, raw_diff as f64 / 1024.0);
        eprintln!("  {:30}  {:>8}  {:>8}", "Compressed (B)",
            dr_compressed.len(), sr_compressed.len());
        let compr_diff = dr_compressed.len() as i64 - sr_compressed.len() as i64;
        eprintln!("  Dual-ring overhead (compr):   {:+} B  ({:+.1} KB)",
            compr_diff, compr_diff as f64 / 1024.0);
    }

    let mem_snapshot = mem_tracker.stop();
    eprintln!("\n=== Peak Memory ===");
    eprintln!("  {mem_snapshot}");

    group.finish();
}

criterion_group!(benches, sha256_8x_dual_ring_stepwise);
criterion_main!(benches);
