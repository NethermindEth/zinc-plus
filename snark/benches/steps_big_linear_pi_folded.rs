//! Per-step breakdown benchmarks for the BigLinearUairWithPublicInput
//! proving stack with **4x-folded** BinaryPoly<32> → BinaryPoly<16> →
//! BinaryPoly<8> columns.
//!
//! This UAIR has 17 columns (16 binary polynomial + 1 integer), with the
//! first 4 binary polynomial columns designated as public inputs.
//! It uses degree-1 constraints and legacy (blanket shift-by-1) trace
//! shifts, but no lookups.
//!
//! Run with:
//!   cargo bench --bench steps_big_linear_pi_folded -p zinc-snark --features "parallel simd asm"

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
    Field, PrimeField,
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
    inner_product::MBSInnerProduct,
};
use zip_plus::{
    code::{
        LinearCode,
        iprs::{
            IprsCode,
            PnttConfigF2_16R4B32,
        },
    },
    pcs::structs::{ZipPlus, ZipPlusParams, ZipTypes},
    pcs::folding::{split_columns, fold_claims_prove},
};

use zinc_test_uair::{BigLinearUairWithPublicInput, GenerateMultiTypeWitness};
use zinc_uair::{ConstraintBuilder, TraceRow, Uair, UairSignature};

use zinc_piop::projections::{
    project_trace_to_field,
    project_scalars, project_scalars_to_field,
};

use zinc_piop::ideal_check::IdealCheckProtocol;
use zinc_piop::combined_poly_resolver::CombinedPolyResolver;

// ─── Dummy Q[X] UAIR for U2 type parameter ─────────────────────────────────
// The pipeline function requires U2::Scalar: Deref<Target = [i64]>.
// This is only exercised when the `qx-constraints` feature is enabled,
// which is not used for this bench.

struct DummyQxUair;

impl Uair for DummyQxUair {
    type Ideal = zinc_uair::ideal::ImpossibleIdeal;
    type Scalar = DensePolynomial<i64, 32>;

    fn signature() -> UairSignature {
        UairSignature {
            binary_poly_cols: 0,
            arbitrary_poly_cols: 0,
            int_cols: 0,
            shifts: vec![],
            public_columns: vec![],
        }
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        _b: &mut B,
        _up: TraceRow<B::Expr>,
        _down: TraceRow<B::Expr>,
        _from_ref: FromR,
        _mbs: MulByScalar,
        _ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
    {
    }
}

// ─── Type definitions ───────────────────────────────────────────────────────

const INT_LIMBS: usize = U64::LIMBS;
type F = MontyField<{ INT_LIMBS * 3 }>;

struct BigLinearPIZipTypes(PhantomData<()>);

impl ZipTypes for BigLinearPIZipTypes {
    const NUM_COLUMN_OPENINGS: usize = 118; // matches SHA-256 bench security level
    const GRINDING_BITS: usize = 16;
    type Eval = BinaryPoly<8>;
    type Cw = DensePolynomial<i64, 8>;
    type Fmod = Uint<{ INT_LIMBS * 3 }>;
    type PrimeTest = MillerRabin;
    type Chal = i128;
    type Pt = i128;
    type CombR = Int<{ INT_LIMBS * 6 }>;
    type Comb = DensePolynomial<Self::CombR, 8>;
    type EvalDotChal = BinaryPolyInnerProduct<Self::Chal, 8>;
    type CombDotChal =
        DensePolyInnerProduct<Self::CombR, Self::Chal, Self::CombR, MBSInnerProduct, 8>;
    type ArrCombRDotChal = MBSInnerProduct;
}

type Zt = BigLinearPIZipTypes;
type Lc = IprsCode<
    Zt,
    PnttConfigF2_16R4B32<2>,
    BinaryPolyWideningMulByScalar<i64>,
    UNCHECKED,
>;

// ─── Parameters ─────────────────────────────────────────────────────────────

const NUM_VARS: usize = 9;           // 2^9 = 512 rows
const TOTAL_COLS: usize = 17;        // 16 binary poly + 1 int

/// Generate the BigLinearUairWithPublicInput trace.
fn generate_trace(num_vars: usize) -> (
    Vec<DenseMultilinearExtension<BinaryPoly<32>>>,
    Vec<DenseMultilinearExtension<DensePolynomial<u32, 32>>>,
    Vec<DenseMultilinearExtension<u32>>,
) {
    let mut rng = rand::rng();
    BigLinearUairWithPublicInput::generate_witness(num_vars, &mut rng)
}

/// Flatten the multi-type trace into a single BinaryPoly<32> trace for the
/// pipeline (binary poly columns ++ int columns projected to BinaryPoly).
fn flatten_trace(
    bp_cols: &[DenseMultilinearExtension<BinaryPoly<32>>],
    _arb_cols: &[DenseMultilinearExtension<DensePolynomial<u32, 32>>],
    int_cols: &[DenseMultilinearExtension<u32>],
) -> Vec<DenseMultilinearExtension<BinaryPoly<32>>> {
    let mut flat: Vec<DenseMultilinearExtension<BinaryPoly<32>>> = bp_cols.to_vec();
    for int_col in int_cols {
        let bp_col: DenseMultilinearExtension<BinaryPoly<32>> = int_col
            .iter()
            .map(|v| BinaryPoly::from(*v))
            .collect();
        flat.push(bp_col);
    }
    flat
}

// ─── Benchmark ────────────────────────────────────────────────────────────

/// Measures each main step of the **4x-folded** proving stack for the
/// BigLinearUairWithPublicInput circuit.
///
/// ## Individual bench steps
///
///   1.  WitnessGen — generate the 17-column trace (512 rows)
///   2.  Folding/SplitColumns — double-split BinaryPoly<32> → BinaryPoly<16> → BinaryPoly<8>
///   3.  PCS/Commit (4x-folded) — Zip+ commit over BinaryPoly<8> split columns
///   4.  PIOP/FieldSetup — transcript init + random field config
///   5.  PIOP/Project Ideal Check — project_scalars for Ideal Check
///   6.  PIOP/IdealCheck — Ideal Check prover (MLE-first, degree 1)
///   7.  PIOP/Project Main field sumcheck — project_scalars_to_field + project_trace_to_field
///   8.  PIOP/Main field sumcheck — Combined Poly Resolver prover
///   9.  Folding/FoldClaims (2-round) — two-round column folding protocol
///  10.  PCS/Prove (4x-folded) — Zip+ prove over BinaryPoly<8> split columns
///  11.  E2E/Prover — total (prove_hybrid_gkr_logup_4x_folded, no lookups)
///  12.  E2E/Verifier — total (verify_classic_logup_4x_folded)
fn big_linear_pi_folded_stepwise(c: &mut Criterion) {
    use zinc_poly::univariate::ideal::DegreeOneIdeal;
    use zinc_uair::ideal_collector::IdealOrZero;

    let mem_tracker = MemoryTracker::start();

    let mut group = c.benchmark_group(format!(
        "BigLinearPI 4x-Folded Steps (grind={})",
        <Zt as ZipTypes>::GRINDING_BITS,
    ));
    group.sample_size(100);

    // -- Folded PCS params -------------------------------------------
    let folded_extra_vars: usize = 2; // Two splits → num_vars + 2
    let folded_num_vars = NUM_VARS + folded_extra_vars;
    let folded_row_len = 1usize << folded_num_vars;
    let folded_lc = Lc::new(folded_row_len);
    let folded_params = ZipPlusParams::<Zt, Lc>::new(
        folded_num_vars, 1, folded_lc,
    );

    let num_constraints = zinc_uair::constraint_counter::count_constraints::<BigLinearUairWithPublicInput>();
    let max_degree = zinc_uair::degree_counter::count_max_degree::<BigLinearUairWithPublicInput>();

    // ── 1. Witness Generation ───────────────────────────────────────
    group.bench_function("WitnessGen", |b| {
        b.iter(|| {
            let trace = generate_trace(NUM_VARS);
            black_box(trace);
        });
    });

    // Pre-generate the trace used by all subsequent steps.
    let (bp_cols, arb_cols, int_cols) = generate_trace(NUM_VARS);
    let flat_trace = flatten_trace(&bp_cols, &arb_cols, &int_cols);
    assert_eq!(flat_trace.len(), TOTAL_COLS);

    // Build private (PCS-committed) trace — exclude public columns.
    let sig = BigLinearUairWithPublicInput::signature();
    let excluded = sig.pcs_excluded_columns();
    let pcs_trace: Vec<_> = flat_trace.iter().enumerate()
        .filter(|(i, _)| !excluded.contains(i))
        .map(|(_, c)| c.clone()).collect();

    // ── 2. Folding / Split Columns ───────────────────────────────────
    group.bench_function("Folding/SplitColumns", |b| {
        b.iter(|| {
            let half = split_columns::<32, 16>(&pcs_trace);
            let quarter = split_columns::<16, 8>(&half);
            black_box(quarter);
        });
    });

    let half_trace: Vec<DenseMultilinearExtension<BinaryPoly<16>>> =
        split_columns::<32, 16>(&pcs_trace);
    let split_trace: Vec<DenseMultilinearExtension<BinaryPoly<8>>> =
        split_columns::<16, 8>(&half_trace);

    // ── 3. PCS Commit ───────────────────────────────────────────────
    group.bench_function("PCS/Commit (4x-folded)", |b| {
        b.iter(|| {
            let r = ZipPlus::<Zt, Lc>::commit(&folded_params, &split_trace);
            let _ = black_box(r);
        });
    });

    // ── 4. PIOP / Field Setup ────────────────────────────────────────
    group.bench_function("PIOP/FieldSetup", |b| {
        b.iter(|| {
            let mut transcript = zinc_transcript::KeccakTranscript::new();
            let field_cfg = transcript.get_random_field_cfg::<
                F, <F as Field>::Inner, MillerRabin
            >();
            black_box(field_cfg);
        });
    });

    // ── 5. PIOP / Project Trace for Ideal Check ─────────────────────
    assert_eq!(max_degree, 1, "BigLinearUairWithPublicInput max_degree should be 1, got {max_degree}");
    group.bench_function("PIOP/Project Ideal Check", |b| {
        let mut tr_setup = zinc_transcript::KeccakTranscript::new();
        let fcfg = tr_setup.get_random_field_cfg::<F, <F as Field>::Inner, MillerRabin>();
        b.iter(|| {
            let projected_scalars = project_scalars::<F, BigLinearUairWithPublicInput>(|scalar| {
                let one = F::one_with_cfg(&fcfg);
                let zero = F::zero_with_cfg(&fcfg);
                DynamicPolynomialF::new(
                    scalar.iter().map(|coeff| {
                        if coeff.into_inner() { one.clone() } else { zero.clone() }
                    }).collect::<Vec<_>>()
                )
            });
            black_box(&projected_scalars);
        });
    });

    // ── 6. PIOP / Ideal Check (MLE-first, on original BinaryPoly<32> trace)
    group.bench_function("PIOP/IdealCheck", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut transcript = zinc_transcript::KeccakTranscript::new();
                let field_cfg = transcript.get_random_field_cfg::<
                    F, <F as Field>::Inner, MillerRabin
                >();
                let projected_scalars = project_scalars::<F, BigLinearUairWithPublicInput>(|scalar| {
                    let one = F::one_with_cfg(&field_cfg);
                    let zero = F::zero_with_cfg(&field_cfg);
                    DynamicPolynomialF::new(
                        scalar.iter().map(|coeff| {
                            if coeff.into_inner() { one.clone() } else { zero.clone() }
                        }).collect::<Vec<_>>()
                    )
                });
                let t = Instant::now();
                let _ = IdealCheckProtocol::<F>::prove_mle_first::<BigLinearUairWithPublicInput, 32>(
                    &mut transcript,
                    &flat_trace,
                    &projected_scalars,
                    num_constraints,
                    NUM_VARS,
                    &field_cfg,
                ).expect("Ideal Check prover failed");
                total += t.elapsed();
            }
            total
        });
    });

    // ── 7. PIOP / Project Trace for Main field sumcheck ─────────────
    let mut transcript_for_cpr = zinc_transcript::KeccakTranscript::new();
    let field_cfg_cpr = transcript_for_cpr.get_random_field_cfg::<
        F, <F as Field>::Inner, MillerRabin
    >();
    let projected_scalars_cpr = project_scalars::<F, BigLinearUairWithPublicInput>(|scalar| {
        let one = F::one_with_cfg(&field_cfg_cpr);
        let zero = F::zero_with_cfg(&field_cfg_cpr);
        DynamicPolynomialF::new(
            scalar.iter().map(|coeff| {
                if coeff.into_inner() { one.clone() } else { zero.clone() }
            }).collect::<Vec<_>>()
        )
    });
    let (_ic_proof_cpr, ic_state_cpr) =
        IdealCheckProtocol::<F>::prove_mle_first::<BigLinearUairWithPublicInput, 32>(
            &mut transcript_for_cpr,
            &flat_trace,
            &projected_scalars_cpr,
            num_constraints,
            NUM_VARS,
            &field_cfg_cpr,
        ).expect("Ideal Check prover failed");

    let projecting_elem_cpr: F = transcript_for_cpr.get_field_challenge(&field_cfg_cpr);

    group.bench_function("PIOP/Project Main field sumcheck", |b| {
        b.iter(|| {
            let fproj_scalars =
                project_scalars_to_field(projected_scalars_cpr.clone(), &projecting_elem_cpr)
                    .expect("scalar projection");
            let field_trace = project_trace_to_field::<F, 32>(
                &flat_trace, &[], &[], &projecting_elem_cpr,
            );
            black_box((&fproj_scalars, &field_trace));
        });
    });

    let field_trace_cpr = project_trace_to_field::<F, 32>(
        &flat_trace, &[], &[], &projecting_elem_cpr,
    );
    let field_projected_scalars_cpr =
        project_scalars_to_field(projected_scalars_cpr.clone(), &projecting_elem_cpr)
            .expect("scalar projection failed");

    // ── 8. PIOP / Combined Poly Resolver ────────────────────────────
    group.bench_function("PIOP/Main field sumcheck", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut tr = transcript_for_cpr.clone();
                let t = Instant::now();
                let _ = CombinedPolyResolver::<F>::prove_as_subprotocol::<BigLinearUairWithPublicInput>(
                    &mut tr,
                    field_trace_cpr.clone(),
                    &ic_state_cpr.evaluation_point,
                    &field_projected_scalars_cpr,
                    num_constraints,
                    NUM_VARS,
                    max_degree,
                    &field_cfg_cpr,
                ).expect("Main field sumcheck prover failed");
                total += t.elapsed();
            }
            total
        });
    });

    // ── 9. Folding Protocol (two-round fold: D→HALF_D→QUARTER_D) ──
    {
        let mut transcript_fold = transcript_for_cpr.clone();
        let (_cpr_proof_fold, cpr_state_fold) =
            CombinedPolyResolver::<F>::prove_as_subprotocol::<BigLinearUairWithPublicInput>(
                &mut transcript_fold,
                field_trace_cpr.clone(),
                &ic_state_cpr.evaluation_point,
                &field_projected_scalars_cpr,
                num_constraints,
                NUM_VARS,
                max_degree,
                &field_cfg_cpr,
            )
            .expect("CPR prove failed for folding bench");
        let fold_piop_point = &cpr_state_fold.evaluation_point[..NUM_VARS];

        group.bench_function("Folding/FoldClaims (2-round)", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let mut tr = transcript_fold.clone();
                    let t = Instant::now();
                    let fold1 = fold_claims_prove::<F, _, 16>(
                        &mut tr, &half_trace, fold_piop_point,
                        &projecting_elem_cpr, &field_cfg_cpr,
                    ).expect("fold1 failed");
                    let _fold2 = fold_claims_prove::<F, _, 8>(
                        &mut tr, &split_trace, &fold1.new_point,
                        &projecting_elem_cpr, &field_cfg_cpr,
                    ).expect("fold2 failed");
                    total += t.elapsed();
                }
                total
            });
        });
    }

    // ── 10. PCS Prove ──────────────────────────────────────────────
    let folded_pcs_point: Vec<F> = {
        let mut tr = transcript_for_cpr.clone();
        let (_, cpr_state) = CombinedPolyResolver::<F>::prove_as_subprotocol::<BigLinearUairWithPublicInput>(
            &mut tr,
            field_trace_cpr.clone(),
            &ic_state_cpr.evaluation_point,
            &field_projected_scalars_cpr,
            num_constraints,
            NUM_VARS,
            max_degree,
            &field_cfg_cpr,
        ).expect("Main field sumcheck prover failed");
        let mut pt = cpr_state.evaluation_point;
        for _ in 0..folded_extra_vars {
            pt.push(F::one_with_cfg(&field_cfg_cpr)); // placeholder γ
        }
        pt
    };

    let (folded_hint, _) = ZipPlus::<Zt, Lc>::commit(&folded_params, &split_trace)
        .expect("commit");

    group.bench_function("Lifting/RingEvals (4x-folded)", |b| {
        b.iter(|| {
            let r = zinc_snark::pipeline::compute_ring_evals::<8>(
                &split_trace, &folded_pcs_point, &field_cfg_cpr,
            );
            black_box(r);
        });
    });

    group.bench_function("PCS/Prove (4x-folded)", |b| {
        b.iter(|| {
            let r = ZipPlus::<Zt, Lc>::prove::<F, UNCHECKED>(
                &folded_params, &split_trace, &folded_pcs_point, &folded_hint,
            );
            let _ = black_box(r);
        });
    });

    // ── 11. E2E Total Prover ─────────────────────────────────────────
    // Public columns for verifier
    let public_cols: Vec<_> = sig.public_columns.iter()
        .map(|&i| flat_trace[i].clone()).collect();

    // Use the pipeline with empty lookup specs
    let empty_lookup_specs: Vec<zinc_piop::lookup::LookupColumnSpec> = vec![];
    let empty_affine_specs: Vec<zinc_piop::lookup::AffineLookupSpec> = vec![];

    // Run the E2E prover for timing and proof size analysis
    let proof_4x = zinc_snark::pipeline::prove_hybrid_gkr_logup_4x_folded::<
        BigLinearUairWithPublicInput, DummyQxUair,
        Zt, Lc, 32, 16, 8, UNCHECKED,
    >(
        &folded_params, &flat_trace, NUM_VARS,
        &empty_lookup_specs, &empty_affine_specs, 2,
    );

    // Print prover pipeline timing breakdown.
    {
        let t = &proof_4x.timing;
        eprintln!("\n── BigLinearPI 4x-Folded Prover Pipeline Timing ────────");
        eprintln!("  Split columns: {:>8.3} ms", t.split_columns.as_secs_f64() * 1000.0);
        eprintln!("  PCS commit:    {:>8.3} ms", t.pcs_commit.as_secs_f64() * 1000.0);
        eprintln!("  Ideal Check:   {:>8.3} ms", t.ideal_check.as_secs_f64() * 1000.0);
        eprintln!("  Field proj:    {:>8.3} ms", t.field_projection.as_secs_f64() * 1000.0);
        eprintln!("  Col eval:      {:>8.3} ms", (t.combined_poly_resolver - t.field_projection).as_secs_f64() * 1000.0);
        eprintln!("  Proj+Eval:     {:>8.3} ms", t.combined_poly_resolver.as_secs_f64() * 1000.0);
        eprintln!("  Shift SC:      {:>8.3} ms", t.shift_sumcheck.as_secs_f64() * 1000.0);
        eprintln!("  Folding:       {:>8.3} ms", t.folding.as_secs_f64() * 1000.0);
        eprintln!("  Lifting:       {:>8.3} ms", t.lifting.as_secs_f64() * 1000.0);
        eprintln!("  PCS prove:     {:>8.3} ms", t.pcs_prove.as_secs_f64() * 1000.0);
        eprintln!("  Total:         {:>8.3} ms", t.total.as_secs_f64() * 1000.0);
        let accounted = t.split_columns + t.pcs_commit + t.ideal_check
            + t.combined_poly_resolver + t.shift_sumcheck + t.folding + t.lifting + t.pcs_prove;
        let unaccounted = t.total.saturating_sub(accounted);
        eprintln!("  Unaccounted:   {:>8.3} ms (serialize only)", unaccounted.as_secs_f64() * 1000.0);
        eprintln!("────────────────────────────────────────────────────────\n");
    }

    // ── Proof size breakdown ────────────────────────────────────────
    {
        use zinc_snark::pipeline::FIELD_LIMBS;

        let pcs_bytes = proof_4x.pcs_proof_bytes.len();
        let ic_bytes: usize = proof_4x.ic_proof_values.iter().map(|v| v.len()).sum();
        let cpr_msg_bytes: usize = proof_4x.cpr_sumcheck_messages.iter().map(|v| v.len()).sum();
        let cpr_sum_bytes = proof_4x.cpr_sumcheck_claimed_sum.len();
        let cpr_sc_total = cpr_msg_bytes + cpr_sum_bytes;
        let cpr_up: usize = proof_4x.cpr_up_evals.iter().map(|v| v.len()).sum();
        let cpr_dn: usize = proof_4x.cpr_down_evals.iter().map(|v| v.len()).sum();
        let cpr_eval_total = cpr_up + cpr_dn;

        let shift_sc_bytes: usize = proof_4x.shift_sumcheck.as_ref().map_or(0, |sc| {
            let rounds: usize = sc.rounds.iter().map(|v| v.len()).sum();
            let finals: usize = sc.v_finals.iter().map(|v| v.len()).sum();
            rounds + finals
        });

        let eval_pt_bytes: usize = proof_4x.evaluation_point_bytes.iter().map(|v| v.len()).sum();
        let pcs_eval_bytes: usize = proof_4x.pcs_evals_bytes.iter().map(|v| v.len()).sum();

        let fold_c1: usize = proof_4x.folding_c1s_bytes.iter().map(|v| v.len()).sum();
        let fold_c2: usize = proof_4x.folding_c2s_bytes.iter().map(|v| v.len()).sum();
        let fold_c3: usize = proof_4x.folding_c3s_bytes.iter().map(|v| v.len()).sum();
        let fold_c4: usize = proof_4x.folding_c4s_bytes.iter().map(|v| v.len()).sum();
        let folding_total = fold_c1 + fold_c2 + fold_c3 + fold_c4;

        let piop_total = ic_bytes + cpr_sc_total + cpr_eval_total
            + shift_sc_bytes + eval_pt_bytes + pcs_eval_bytes + folding_total;
        let total_raw = pcs_bytes + piop_total;

        let mut all_bytes = Vec::with_capacity(total_raw);
        all_bytes.extend(&proof_4x.pcs_proof_bytes);
        for v in &proof_4x.ic_proof_values { all_bytes.extend(v); }
        for v in &proof_4x.cpr_sumcheck_messages { all_bytes.extend(v); }
        all_bytes.extend(&proof_4x.cpr_sumcheck_claimed_sum);
        for v in &proof_4x.cpr_up_evals { all_bytes.extend(v); }
        for v in &proof_4x.cpr_down_evals { all_bytes.extend(v); }
        if let Some(ref sc) = proof_4x.shift_sumcheck {
            for v in &sc.rounds { all_bytes.extend(v); }
            for v in &sc.v_finals { all_bytes.extend(v); }
        }
        for v in &proof_4x.evaluation_point_bytes { all_bytes.extend(v); }
        for v in &proof_4x.pcs_evals_bytes { all_bytes.extend(v); }
        for v in &proof_4x.folding_c1s_bytes { all_bytes.extend(v); }
        for v in &proof_4x.folding_c2s_bytes { all_bytes.extend(v); }
        for v in &proof_4x.folding_c3s_bytes { all_bytes.extend(v); }
        for v in &proof_4x.folding_c4s_bytes { all_bytes.extend(v); }

        let compressed = {
            use std::io::Write;
            let mut encoder = flate2::write::DeflateEncoder::new(
                Vec::new(), flate2::Compression::default(),
            );
            encoder.write_all(&all_bytes).unwrap();
            encoder.finish().unwrap()
        };

        eprintln!("\n=== BigLinearPI 4x-Folded Proof Size ===");
        eprintln!("  PCS:              {:>6} B  ({:.1} KB)", pcs_bytes, pcs_bytes as f64 / 1024.0);
        eprintln!("  IC:               {:>6} B", ic_bytes);
        if cpr_sc_total > 0 {
            eprintln!("  CPR sumcheck:     {:>6} B  (msgs={cpr_msg_bytes}, sum={cpr_sum_bytes})", cpr_sc_total);
        }
        eprintln!("  Col evals:        {:>6} B  (up={cpr_up}, down={cpr_dn})", cpr_eval_total);
        eprintln!("  Shift SC:         {:>6} B", shift_sc_bytes);
        eprintln!("  Eval point:       {:>6} B", eval_pt_bytes);
        eprintln!("  PCS evals:        {:>6} B", pcs_eval_bytes);
        eprintln!("  Folding:          {:>6} B  (c1={fold_c1}, c2={fold_c2}, c3={fold_c3}, c4={fold_c4})", folding_total);
        eprintln!("  ─────────────────────────────");
        eprintln!("  PIOP total:       {:>6} B  ({:.1} KB)", piop_total, piop_total as f64 / 1024.0);
        eprintln!("  Total raw:        {:>6} B  ({:.1} KB)", total_raw, total_raw as f64 / 1024.0);
        eprintln!("  Compressed:       {:>6} B  ({:.1} KB, {:.1}x ratio)",
            compressed.len(), compressed.len() as f64 / 1024.0,
            all_bytes.len() as f64 / compressed.len() as f64);
        eprintln!("═══════════════════════════════════════════\n");
    }

    group.bench_function("E2E/Prover (4x-folded)", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let (bp, arb, int) = generate_trace(NUM_VARS);
                let trace = flatten_trace(&bp, &arb, &int);
                let t = Instant::now();
                let _ = zinc_snark::pipeline::prove_hybrid_gkr_logup_4x_folded::<
                    BigLinearUairWithPublicInput, DummyQxUair,
                    Zt, Lc, 32, 16, 8, UNCHECKED,
                >(
                    &folded_params, &trace, NUM_VARS,
                    &empty_lookup_specs, &empty_affine_specs, 2,
                );
                total += t.elapsed();
            }
            total
        });
    });

    // ── 12. E2E Total Verifier ─────────────────────────────────────
    {
        let r = zinc_snark::pipeline::verify_classic_logup_4x_folded::<
            BigLinearUairWithPublicInput, DummyQxUair,
            Zt, Lc, 32, 16, 8, UNCHECKED, _, _, _, _,
        >(
            &folded_params, &proof_4x, NUM_VARS,
            |_: &IdealOrZero<DegreeOneIdeal<u32>>| zinc_snark::pipeline::TrivialIdeal,
            |_: &IdealOrZero<zinc_uair::ideal::ImpossibleIdeal>| zinc_snark::pipeline::TrivialIdeal,
            &public_cols,
            &[], // no boundary public columns
        );
        assert!(r.accepted, "Verifier rejected the proof");
        let t = &r.timing;
        println!("\n── Verifier step timing (BigLinearPI 4x-folded) ────────");
        println!("  IC verify:           {:>8.3} ms", t.ideal_check_verify.as_secs_f64() * 1000.0);
        println!("  CPR+Lookup verify:   {:>8.3} ms", t.combined_poly_resolver_verify.as_secs_f64() * 1000.0);
        println!("  Lookup verify:       {:>8.3} ms", t.lookup_verify.as_secs_f64() * 1000.0);
        println!("  PCS verify:          {:>8.3} ms", t.pcs_verify.as_secs_f64() * 1000.0);
        println!("  Total:               {:>8.3} ms", t.total.as_secs_f64() * 1000.0);
        println!("─────────────────────────────────────────────────────────\n");
    }

    group.bench_function("E2E/Verifier (4x-folded)", |b| {
        b.iter(|| {
            let r = zinc_snark::pipeline::verify_classic_logup_4x_folded::<
                BigLinearUairWithPublicInput, DummyQxUair,
                Zt, Lc, 32, 16, 8, UNCHECKED, _, _, _, _,
            >(
                &folded_params, &proof_4x, NUM_VARS,
                |_: &IdealOrZero<DegreeOneIdeal<u32>>| zinc_snark::pipeline::TrivialIdeal,
                |_: &IdealOrZero<zinc_uair::ideal::ImpossibleIdeal>| zinc_snark::pipeline::TrivialIdeal,
                &public_cols,
                &[],
            );
            black_box(r);
        });
    });

    let mem_snapshot = mem_tracker.stop();
    eprintln!("\n=== Peak Memory ===");
    eprintln!("  {mem_snapshot}");

    group.finish();
}

criterion_group!(benches, big_linear_pi_folded_stepwise);
criterion_main!(benches);
