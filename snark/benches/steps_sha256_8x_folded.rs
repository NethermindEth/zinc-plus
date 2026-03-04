//! Per-step breakdown benchmarks for the 8×SHA-256 proving stack with
//! **folded** BinaryPoly<32> → BinaryPoly<16> columns.
//!
//! This benchmark mirrors [`steps_sha256_8x`] but uses the folded pipeline
//! (`prove_classic_logup_folded` / `verify_classic_logup_folded`) which:
//!   1. Splits each PCS-committed BinaryPoly<32> column into two BinaryPoly<16>
//!      halves (u = lower 16 coeffs, w = upper 16 coeffs).
//!   2. Commits and proves the PCS over the smaller BinaryPoly<16> codewords.
//!   3. Runs a folding protocol to reduce the original BinaryPoly<32> eval
//!      claims to the committed BinaryPoly<16> columns.
//!
//! Run with:
//!   cargo bench --bench steps_sha256_8x_folded -p zinc-snark --features "parallel simd asm"

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
        iprs::{
            IprsCode,
            PnttConfigF12289R4B64, PnttConfigF12289R4B16,
            PnttConfigF2_16R4B32, PnttConfigF2_16R4B4,
        },
    },
    pcs::structs::{ZipPlus, ZipPlusParams, ZipTypes},
    pcs::folding::split_columns,
};

use zinc_sha256_uair::{Sha256Uair, witness::GenerateWitness};
use zinc_uair::Uair;
use zinc_piop::projections::{
    project_trace_to_field,
    project_scalars, project_scalars_to_field,
};
use zinc_piop::lookup::{AffineLookupSpec, LookupColumnSpec, LookupTableType};

use zinc_piop::ideal_check::IdealCheckProtocol;
use zinc_piop::combined_poly_resolver::CombinedPolyResolver;
use zinc_piop::shift_sumcheck::{
    shift_sumcheck_prove,
    shift_sumcheck_verify, shift_sumcheck_verify_pre, shift_sumcheck_verify_finalize,
    ShiftClaim, ShiftSumcheckProof, ShiftRoundPoly,
};
use zinc_utils::projectable_to_field::ProjectableToField;


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

/// Original (non-folded) PCS types — used only for the side-by-side
/// comparison benchmarks.
/// Uses F12289 (p = 12289, p-1 = 2^12 × 3) with rate 1/4.
type OrigZt = Sha256ZipTypes<i64, 32>;
type OrigLc = IprsCode<
    OrigZt,
    PnttConfigF12289R4B64<1>,
    BinaryPolyWideningMulByScalar<i64>,
    UNCHECKED,
>;

/// Folded PCS types — BinaryPoly<16> codewords.
/// Uses PnttConfigF12289R4B16<2>: INPUT_LEN = 16 × 8² = 1024 (doubled row_len),
/// OUTPUT_LEN = 64 × 8² = 4096 (rate 1/4, same as original).
type FoldedZt = Sha256ZipTypes<i64, 16>;
type FoldedLc = IprsCode<
    FoldedZt,
    PnttConfigF12289R4B16<2>,
    BinaryPolyWideningMulByScalar<i64>,
    UNCHECKED,
>;

/// 4x-Folded PCS types — BinaryPoly<8> codewords.
/// Uses PnttConfigF2_16R4B32<2> (BASE_LEN=32, BASE_DIM=128, DEPTH=2):
///   INPUT_LEN = 32 × 64 = 2048 rows,
///   OUTPUT_LEN = 128 × 64 = 8192 (rate 1/4).
/// Field: F65537 (2^16+1, TWO_ADICITY=16), which has small twiddles (max 4096)
/// giving max codeword coefficient ~2^50, vs ~2^62 at depth 3.
type FoldedZt4x = Sha256ZipTypes<i64, 8>;
type FoldedLc4x = IprsCode<
    FoldedZt4x,
    PnttConfigF2_16R4B32<2>,
    BinaryPolyWideningMulByScalar<i64>,
    UNCHECKED,
>;

/// 8x-Folded PCS types — BinaryPoly<4> codewords.
/// Uses PnttConfigF2_16R4B4<3> (BASE_LEN=4, DEPTH=3):
///   INPUT_LEN = 2048 rows,
///   OUTPUT_LEN = 8192 (rate 1/4).
/// Field: F65537 (2^16+1), same as the 4x variant.
type FoldedZt8x = Sha256ZipTypes<i64, 4>;
type FoldedLc8x = IprsCode<
    FoldedZt8x,
    PnttConfigF2_16R4B4<3>,
    BinaryPolyWideningMulByScalar<i64>,
    UNCHECKED,
>;

// ─── Parameters ─────────────────────────────────────────────────────────────

const SHA256_8X_NUM_VARS: usize = 9;      // 2^9 = 512 rows (8 × 64 SHA rounds)
const SHA256_BATCH_SIZE: usize = 30;       // 30 SHA-256 columns (27 bitpoly + 3 int)
const SHA256_LOOKUP_COL_COUNT: usize = 10; // 10 Q[X] columns need lookup

/// Default lookup specs: 8 chunks of 2^4 each (chunk_width=4, total_width=32).
///
/// 32 / 4 = 8 chunks, each with a subtable of size 2^4 = 16 entries.
/// This gives the smallest compressed proof size.
fn sha256_lookup_specs() -> Vec<LookupColumnSpec> {
    (0..SHA256_LOOKUP_COL_COUNT)
        .map(|i| LookupColumnSpec {
            column_index: i,
            table_type: LookupTableType::BitPoly { width: 32, chunk_width: Some(4) },
        })
        .collect()
}

fn sha256_affine_lookup_specs() -> Vec<AffineLookupSpec> {
    use zinc_sha256_uair::{
        COL_E_HAT, COL_E_TM1, COL_CH_EF_HAT, COL_E_TM2, COL_CH_NEG_EG_HAT,
        COL_A_HAT, COL_A_TM1, COL_A_TM2, COL_MAJ_HAT,
    };
    let bp32 = LookupTableType::BitPoly { width: 32, chunk_width: Some(4) };
    vec![
        AffineLookupSpec {
            terms: vec![(COL_E_HAT, 1), (COL_E_TM1, 1), (COL_CH_EF_HAT, -2)],
            constant_offset_bits: 0,
            table_type: bp32.clone(),
        },
        AffineLookupSpec {
            terms: vec![(COL_E_HAT, -1), (COL_E_TM2, 1), (COL_CH_NEG_EG_HAT, -2)],
            constant_offset_bits: 0xFFFF_FFFF,
            table_type: bp32.clone(),
        },
        AffineLookupSpec {
            terms: vec![(COL_A_HAT, 1), (COL_A_TM1, 1), (COL_A_TM2, 1), (COL_MAJ_HAT, -2)],
            constant_offset_bits: 0,
            table_type: bp32,
        },
    ]
}

fn generate_sha256_trace(num_vars: usize) -> Vec<DenseMultilinearExtension<BinaryPoly<32>>> {
    let mut rng = rand::rng();
    <Sha256Uair as GenerateWitness<BinaryPoly<32>>>::generate_witness(num_vars, &mut rng)
}

/// Lookup specs with 4 chunks of 2^8 each (chunk_width=8, total_width=32).
fn sha256_lookup_specs_4chunks() -> Vec<LookupColumnSpec> {
    (0..SHA256_LOOKUP_COL_COUNT)
        .map(|i| LookupColumnSpec {
            column_index: i,
            table_type: LookupTableType::BitPoly { width: 32, chunk_width: Some(8) },
        })
        .collect()
}

fn sha256_affine_lookup_specs_4chunks() -> Vec<AffineLookupSpec> {
    use zinc_sha256_uair::{
        COL_E_HAT, COL_E_TM1, COL_CH_EF_HAT, COL_E_TM2, COL_CH_NEG_EG_HAT,
        COL_A_HAT, COL_A_TM1, COL_A_TM2, COL_MAJ_HAT,
    };
    let bp32 = LookupTableType::BitPoly { width: 32, chunk_width: Some(8) };
    vec![
        AffineLookupSpec {
            terms: vec![(COL_E_HAT, 1), (COL_E_TM1, 1), (COL_CH_EF_HAT, -2)],
            constant_offset_bits: 0,
            table_type: bp32.clone(),
        },
        AffineLookupSpec {
            terms: vec![(COL_E_HAT, -1), (COL_E_TM2, 1), (COL_CH_NEG_EG_HAT, -2)],
            constant_offset_bits: 0xFFFF_FFFF,
            table_type: bp32.clone(),
        },
        AffineLookupSpec {
            terms: vec![(COL_A_HAT, 1), (COL_A_TM1, 1), (COL_A_TM2, 1), (COL_MAJ_HAT, -2)],
            constant_offset_bits: 0,
            table_type: bp32,
        },
    ]
}

// ─── Benchmark ──────────────────────────────────────────────────────────────

/// Measures each main step of the **folded** E2E proving stack individually
/// for 8×SHA-256 compressions (**no** ECDSA).
///
/// Prover steps benchmarked:
///   1.  WitnessGen — generate the 30-column BinaryPoly<32> trace (512 rows)
///   2.  Folding/SplitColumns — split BinaryPoly<32> to BinaryPoly<16> halves
///   3.  PCS/Commit (folded) — Zip+ commit over BinaryPoly<16> split columns
///   4.  PCS/Commit (original) — Zip+ commit over BinaryPoly<32> (comparison)
///   5.  PIOP/FieldSetup — transcript init + random field config
///   6.  PIOP/Project Ideal Check — project_scalars for Ideal Check
///   7.  PIOP/IdealCheck — Ideal Check prover (MLE-first, on original trace)
///   8.  PIOP/Project Main field sumcheck — project_scalars_to_field + project_trace_to_field
///   9.  PIOP/Main field sumcheck — Combined Poly Resolver prover
///  10.  PIOP/LookupExtract — extract lookup columns from field trace
///  11.  PIOP/Lookup — classic batched decomposed LogUp prover
///  12.  PCS/Prove (folded) — Zip+ prove over BinaryPoly<16> split columns
///  13.  PCS/Prove (original) — Zip+ prove over BinaryPoly<32> (comparison)
///  14.  E2E/Prover (folded) — total (prove_classic_logup_folded)
///  15.  E2E/Prover (original) — total (pipeline::prove)
///  16.  E2E/Verifier (folded)
///  17.  E2E/Verifier (original)
///
/// Also reports:
///  - Proof size breakdown (folded vs original, raw + compressed)
///  - Peak memory usage
fn sha256_8x_folded_stepwise(c: &mut Criterion) {
    use zinc_sha256_uair::CyclotomicIdeal;
    use zinc_uair::ideal_collector::IdealOrZero;
    use zinc_piop::lookup::prove_batched_lookup_with_indices;

    let mem_tracker = MemoryTracker::start();

    let mut group = c.benchmark_group("8xSHA256 Folded Steps");
    group.sample_size(100);

    // ── Original PCS params (for comparison) ────────────────────────
    let orig_lc = OrigLc::new(512);
    let orig_params = ZipPlusParams::<OrigZt, OrigLc>::new(SHA256_8X_NUM_VARS, 1, orig_lc);

    // ── Folded PCS params ───────────────────────────────────────────
    //
    // Split columns have num_vars + 1 variables (2^{nv+1} evaluations).
    // Double row_len (1024) so that num_rows = 2^10 / 1024 = 1, matching the
    // original. Each codeword element is BinaryPoly<16> (128 B) instead of
    // BinaryPoly<32> (256 B), so column openings are 2x cheaper.
    let folded_num_vars = SHA256_8X_NUM_VARS + 1; // 10
    let row_len = 1024;
    let folded_num_rows = (1usize << folded_num_vars) / row_len; // 1
    let folded_lc = FoldedLc::new(row_len);
    let folded_params = ZipPlusParams::<FoldedZt, FoldedLc>::new(
        folded_num_vars, folded_num_rows, folded_lc,
    );

    // 4x-folded params: two splits → num_vars + 2, row_len = 2048, num_rows = 1
    let folded_4x_num_vars = SHA256_8X_NUM_VARS + 2; // 11
    let row_len_4x = 2048;
    let folded_4x_lc = FoldedLc4x::new(row_len_4x);
    let folded_4x_params = ZipPlusParams::<FoldedZt4x, FoldedLc4x>::new(
        folded_4x_num_vars, 1, folded_4x_lc,
    );

    // 8x-folded params: three splits → num_vars + 3, row_len = 2048, num_rows = 2
    let folded_8x_num_vars = SHA256_8X_NUM_VARS + 3; // 12
    let row_len_8x = 2048;
    let folded_8x_num_rows = (1usize << folded_8x_num_vars) / row_len_8x; // 2
    let folded_8x_lc = FoldedLc8x::new(row_len_8x);
    let folded_8x_params = ZipPlusParams::<FoldedZt8x, FoldedLc8x>::new(
        folded_8x_num_vars, folded_8x_num_rows, folded_8x_lc,
    );

    let sha_lookup_specs = sha256_lookup_specs();
    let sha_affine_specs = sha256_affine_lookup_specs();

    let num_constraints = zinc_uair::constraint_counter::count_constraints::<Sha256Uair>();
    let max_degree = zinc_uair::degree_counter::count_max_degree::<Sha256Uair>();
    let num_vars = SHA256_8X_NUM_VARS;

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

    // Build private (PCS-committed) trace — exclude public + shift-source columns.
    let sha_sig = Sha256Uair::signature();
    let sha_excluded = sha_sig.pcs_excluded_columns();
    let sha_pcs_trace: Vec<_> = sha_trace.iter().enumerate()
        .filter(|(i, _)| !sha_excluded.contains(i))
        .map(|(_, c)| c.clone()).collect();

    // ── 2. Folding / Split Columns ──────────────────────────────────
    group.bench_function("Folding/SplitColumns", |b| {
        b.iter(|| {
            let split = split_columns::<32, 16>(&sha_pcs_trace);
            black_box(split);
        });
    });

    let split_trace: Vec<DenseMultilinearExtension<BinaryPoly<16>>> =
        split_columns::<32, 16>(&sha_pcs_trace);

    // ── 3. PCS Commit (folded — BinaryPoly<16>) ─────────────────────
    group.bench_function("PCS/Commit (folded)", |b| {
        b.iter(|| {
            let r = ZipPlus::<FoldedZt, FoldedLc>::commit(&folded_params, &split_trace);
            let _ = black_box(r);
        });
    });

    // ── 4. PCS Commit (original — BinaryPoly<32>, for comparison) ───
    group.bench_function("PCS/Commit (original)", |b| {
        b.iter(|| {
            let r = ZipPlus::<OrigZt, OrigLc>::commit(&orig_params, &sha_pcs_trace);
            let _ = black_box(r);
        });
    });

    // ── 5. PIOP / Field Setup ────────────────────────────────────────
    group.bench_function("PIOP/FieldSetup", |b| {
        b.iter(|| {
            let mut transcript = zinc_transcript::KeccakTranscript::new();
            let field_cfg = transcript.get_random_field_cfg::<
                F, <F as Field>::Inner, MillerRabin
            >();
            black_box(field_cfg);
        });
    });

    // ── 6. PIOP / Project Trace for Ideal Check ─────────────────────
    assert!(max_degree >= 1 && max_degree <= 2, "SHA-256 UAIR max_degree should be 1 or 2, got {max_degree}");
    group.bench_function("PIOP/Project Ideal Check", |b| {
        let mut tr_setup = zinc_transcript::KeccakTranscript::new();
        let fcfg = tr_setup.get_random_field_cfg::<F, <F as Field>::Inner, MillerRabin>();
        b.iter(|| {
            let projected_scalars = project_scalars::<F, Sha256Uair>(|scalar| {
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

    // ── 7. PIOP / Ideal Check (MLE-first, on original BinaryPoly<32> trace)
    group.bench_function("PIOP/IdealCheck", |b| {
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
                    num_constraints,
                    SHA256_8X_NUM_VARS,
                    &field_cfg,
                ).expect("Ideal Check prover failed");
                total += t.elapsed();
            }
            total
        });
    });

    // ── 8. PIOP / Project Trace for Main field sumcheck ─────────────────────────────
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
            num_constraints,
            SHA256_8X_NUM_VARS,
            &field_cfg_cpr,
        ).expect("Ideal Check prover failed");

    let projecting_elem_cpr: F = transcript_for_cpr.get_field_challenge(&field_cfg_cpr);

    group.bench_function("PIOP/Project Main field sumcheck", |b| {
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

    // ── 9. PIOP / Combined Poly Resolver ────────────────────────────
    group.bench_function("PIOP/Main field sumcheck", |b| {
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
                    num_constraints,
                    SHA256_8X_NUM_VARS,
                    max_degree,
                    &field_cfg_cpr,
                ).expect("Main field sumcheck prover failed");
                total += t.elapsed();
            }
            total
        });
    });

    // ── 10. PIOP / Lookup Extract ───────────────────────────────────
    group.bench_function("PIOP/LookupExtract", |b| {
        b.iter(|| {
            let mut needed_b: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
            for spec in &sha_lookup_specs {
                let next = needed_b.len();
                needed_b.entry(spec.column_index).or_insert(next);
            }
            let mut columns: Vec<Vec<F>> = Vec::with_capacity(needed_b.len());
            let mut raw_indices: Vec<Vec<usize>> = Vec::with_capacity(needed_b.len());
            for &orig_idx in needed_b.keys() {
                let col_f: Vec<F> = field_trace_cpr[orig_idx].iter()
                    .map(|inner| F::new_unchecked_with_cfg(inner.clone(), &field_cfg_cpr)).collect();
                columns.push(col_f);
                let col_idx: Vec<usize> = sha_trace[orig_idx].iter().map(|bp| {
                    let mut idx = 0usize;
                    for (j, coeff) in bp.iter().enumerate() {
                        if coeff.into_inner() { idx |= 1usize << j; }
                    }
                    idx
                }).collect();
                raw_indices.push(col_idx);
            }
            let index_map: std::collections::BTreeMap<usize, usize> = needed_b.keys()
                .enumerate().map(|(n, &o)| (o, n)).collect();
            let remapped: Vec<LookupColumnSpec> = sha_lookup_specs.iter()
                .map(|s| LookupColumnSpec { column_index: index_map[&s.column_index], table_type: s.table_type.clone() })
                .collect();
            black_box((&columns, &raw_indices, &remapped));
        });
    });

    // ── 11. PIOP / Lookup (classic) ─────────────────────────────────
    {
        let mut transcript_lk = transcript_for_cpr.clone();
        let _ = CombinedPolyResolver::<F>::prove_as_subprotocol::<Sha256Uair>(
            &mut transcript_lk,
            field_trace_cpr.clone(),
            &ic_state_cpr.evaluation_point,
            &field_projected_scalars_cpr,
            num_constraints,
            SHA256_8X_NUM_VARS,
            max_degree,
            &field_cfg_cpr,
        ).expect("Main field sumcheck prover failed");

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

    // ── 11b. PIOP / Shift Sumcheck ──────────────────────────────────
    //
    // Benchmarks the shift sumcheck prover which reduces shifted-column
    // evaluation claims from the CPR down_evals to plain MLE claims on
    // the unshifted source columns at a new random point.
    {
        // Re-run CPR to get the state we need for the shift sumcheck.
        let mut transcript_ss = transcript_for_cpr.clone();
        let (cpr_proof_ss, cpr_state_ss) =
            CombinedPolyResolver::<F>::prove_as_subprotocol::<Sha256Uair>(
                &mut transcript_ss,
                field_trace_cpr.clone(),
                &ic_state_cpr.evaluation_point,
                &field_projected_scalars_cpr,
                num_constraints,
                SHA256_8X_NUM_VARS,
                max_degree,
                &field_cfg_cpr,
            )
            .expect("CPR prove failed for shift sumcheck bench");

        // Extract source columns for the shift claims.
        let sha_sig_ss = Sha256Uair::signature();
        let shift_trace_columns: Vec<DenseMultilinearExtension<<F as Field>::Inner>> =
            sha_sig_ss.shifts.iter()
                .map(|spec| field_trace_cpr[spec.source_col].clone())
                .collect();

        // Build shift claims from CPR down_evals.
        let shift_claims: Vec<ShiftClaim<F>> = sha_sig_ss.shifts
            .iter()
            .enumerate()
            .map(|(i, spec)| ShiftClaim {
                source_col: i,
                shift_amount: spec.shift_amount,
                eval_point: cpr_state_ss.evaluation_point.clone(),
                claimed_eval: cpr_proof_ss.down_evals[i].clone(),
            })
            .collect();

        group.bench_function("PIOP/ShiftSumcheck", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let mut tr = transcript_ss.clone();
                    let t = Instant::now();
                    let _ = shift_sumcheck_prove(
                        &mut tr,
                        &shift_claims,
                        &shift_trace_columns,
                        SHA256_8X_NUM_VARS,
                        &field_cfg_cpr,
                    );
                    total += t.elapsed();
                }
                total
            });
        });
    }

    // ── 12. PCS Prove (folded — BinaryPoly<16> split columns) ───────
    let folded_pcs_point: Vec<F> = {
        let mut tr = transcript_for_cpr.clone();
        let (_, cpr_state) = CombinedPolyResolver::<F>::prove_as_subprotocol::<Sha256Uair>(
            &mut tr,
            field_trace_cpr.clone(),
            &ic_state_cpr.evaluation_point,
            &field_projected_scalars_cpr,
            num_constraints,
            SHA256_8X_NUM_VARS,
            max_degree,
            &field_cfg_cpr,
        ).expect("Main field sumcheck prover failed");
        // The folded PCS point extends the main field sumcheck eval point by one extra
        // coordinate (γ from the folding protocol).  For this step bench
        // we add a dummy extra coordinate.
        let mut pt = cpr_state.evaluation_point;
        pt.push(F::one_with_cfg(&field_cfg_cpr)); // placeholder γ
        pt
    };

    let (folded_hint, _) = ZipPlus::<FoldedZt, FoldedLc>::commit(&folded_params, &split_trace)
        .expect("commit");

    group.bench_function("PCS/Prove (folded)", |b| {
        b.iter(|| {
            let r = ZipPlus::<FoldedZt, FoldedLc>::prove::<F, UNCHECKED>(
                &folded_params, &split_trace, &folded_pcs_point, &folded_hint,
            );
            let _ = black_box(r);
        });
    });

    // ── 13. PCS Prove (original — for comparison) ───────────────────
    let orig_pcs_point: Vec<F> = folded_pcs_point[..num_vars].to_vec();
    let (orig_hint, _) = ZipPlus::<OrigZt, OrigLc>::commit(&orig_params, &sha_pcs_trace)
        .expect("commit");

    group.bench_function("PCS/Prove (original)", |b| {
        b.iter(|| {
            let r = ZipPlus::<OrigZt, OrigLc>::prove::<F, UNCHECKED>(
                &orig_params, &sha_pcs_trace, &orig_pcs_point, &orig_hint,
            );
            let _ = black_box(r);
        });
    });

    // ── 14. E2E Total Prover (folded pipeline) ──────────────────────
    group.bench_function("E2E/Prover (folded)", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let t = Instant::now();
                let _ = zinc_snark::pipeline::prove_classic_logup_folded::<
                    Sha256Uair, FoldedZt, FoldedLc, 32, 16, UNCHECKED,
                >(
                    &folded_params, &sha_trace, SHA256_8X_NUM_VARS,
                    &sha_lookup_specs, &sha_affine_specs,
                );
                total += t.elapsed();
            }
            total
        });
    });

    // ── 15. E2E Total Prover (original pipeline, for comparison) ────
    group.bench_function("E2E/Prover (original)", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let t = Instant::now();
                let _ = zinc_snark::pipeline::prove::<Sha256Uair, OrigZt, OrigLc, 32, UNCHECKED>(
                    &orig_params, &sha_trace, SHA256_8X_NUM_VARS, &sha_lookup_specs,
                );
                total += t.elapsed();
            }
            total
        });
    });

    // ── 16. E2E Total Verifier (folded) ─────────────────────────────
    let folded_proof = zinc_snark::pipeline::prove_classic_logup_folded::<
        Sha256Uair, FoldedZt, FoldedLc, 32, 16, UNCHECKED,
    >(
        &folded_params, &sha_trace, SHA256_8X_NUM_VARS,
        &sha_lookup_specs, &sha_affine_specs,
    );

    let sha_sig_pub = Sha256Uair::signature();
    let sha_public_cols: Vec<_> = sha_sig_pub.public_columns.iter()
        .map(|&i| sha_trace[i].clone()).collect();

    {
        let r = zinc_snark::pipeline::verify_classic_logup_folded::<
            Sha256Uair, FoldedZt, FoldedLc, 32, 16, UNCHECKED, _, _,
        >(
            &folded_params, &folded_proof, SHA256_8X_NUM_VARS,
            |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
            &sha_public_cols,
        );
        let t = &r.timing;
        println!("\n── Verifier step timing (folded) ────────────────────────");
        println!("  IC verify:           {:>8.3} ms", t.ideal_check_verify.as_secs_f64() * 1000.0);
        println!("  CPR+Lookup verify:   {:>8.3} ms", t.combined_poly_resolver_verify.as_secs_f64() * 1000.0);
        println!("  Lookup verify:       {:>8.3} ms", t.lookup_verify.as_secs_f64() * 1000.0);
        println!("  PCS verify:          {:>8.3} ms", t.pcs_verify.as_secs_f64() * 1000.0);
        println!("  Total:               {:>8.3} ms", t.total.as_secs_f64() * 1000.0);
        println!("─────────────────────────────────────────────────────────\n");
    }

    group.bench_function("E2E/Verifier (folded)", |b| {
        b.iter(|| {
            let r = zinc_snark::pipeline::verify_classic_logup_folded::<
                Sha256Uair, FoldedZt, FoldedLc, 32, 16, UNCHECKED, _, _,
            >(
                &folded_params, &folded_proof, SHA256_8X_NUM_VARS,
                |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
                &sha_public_cols,
            );
            black_box(r);
        });
    });

    // ── 17. E2E Total Verifier (original, for comparison) ───────────
    let orig_proof = zinc_snark::pipeline::prove::<Sha256Uair, OrigZt, OrigLc, 32, UNCHECKED>(
        &orig_params, &sha_trace, SHA256_8X_NUM_VARS, &sha_lookup_specs,
    );

    {
        let r = zinc_snark::pipeline::verify::<Sha256Uair, OrigZt, OrigLc, 32, UNCHECKED, _, _>(
            &orig_params, &orig_proof, SHA256_8X_NUM_VARS,
            |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
            &sha_public_cols,
        );
        let t = &r.timing;
        println!("\n── Verifier step timing (original) ──────────────────────");
        println!("  IC verify:           {:>8.3} ms", t.ideal_check_verify.as_secs_f64() * 1000.0);
        println!("  CPR+Lookup verify:   {:>8.3} ms", t.combined_poly_resolver_verify.as_secs_f64() * 1000.0);
        println!("  Lookup verify:       {:>8.3} ms", t.lookup_verify.as_secs_f64() * 1000.0);
        println!("  PCS verify:          {:>8.3} ms", t.pcs_verify.as_secs_f64() * 1000.0);
        println!("  Total:               {:>8.3} ms", t.total.as_secs_f64() * 1000.0);
        println!("─────────────────────────────────────────────────────────\n");
    }

    group.bench_function("E2E/Verifier (original)", |b| {
        b.iter(|| {
            let r = zinc_snark::pipeline::verify::<Sha256Uair, OrigZt, OrigLc, 32, UNCHECKED, _, _>(
                &orig_params, &orig_proof, SHA256_8X_NUM_VARS,
                |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
                &sha_public_cols,
            );
            black_box(r);
        });
    });

    // ── 18. E2E Total Prover (GKR folded pipeline) ──────────────────
    group.bench_function("E2E/Prover (GKR folded)", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let t = Instant::now();
                let _ = zinc_snark::pipeline::prove_gkr_logup_folded::<
                    Sha256Uair, FoldedZt, FoldedLc, 32, 16, UNCHECKED,
                >(
                    &folded_params, &sha_trace, SHA256_8X_NUM_VARS,
                    &sha_lookup_specs, &sha_affine_specs,
                );
                total += t.elapsed();
            }
            total
        });
    });

    // ── 19. E2E Total Verifier (GKR folded) ─────────────────────────
    let gkr_folded_proof = zinc_snark::pipeline::prove_gkr_logup_folded::<
        Sha256Uair, FoldedZt, FoldedLc, 32, 16, UNCHECKED,
    >(
        &folded_params, &sha_trace, SHA256_8X_NUM_VARS,
        &sha_lookup_specs, &sha_affine_specs,
    );

    {
        let r = zinc_snark::pipeline::verify_classic_logup_folded::<
            Sha256Uair, FoldedZt, FoldedLc, 32, 16, UNCHECKED, _, _,
        >(
            &folded_params, &gkr_folded_proof, SHA256_8X_NUM_VARS,
            |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
            &sha_public_cols,
        );
        let t = &r.timing;
        println!("\n── Verifier step timing (GKR folded) ────────────────────");
        println!("  IC verify:           {:>8.3} ms", t.ideal_check_verify.as_secs_f64() * 1000.0);
        println!("  CPR+Lookup verify:   {:>8.3} ms", t.combined_poly_resolver_verify.as_secs_f64() * 1000.0);
        println!("  Lookup verify:       {:>8.3} ms", t.lookup_verify.as_secs_f64() * 1000.0);
        println!("  PCS verify:          {:>8.3} ms", t.pcs_verify.as_secs_f64() * 1000.0);
        println!("  Total:               {:>8.3} ms", t.total.as_secs_f64() * 1000.0);
        println!("─────────────────────────────────────────────────────────\n");
    }

    group.bench_function("E2E/Verifier (GKR folded)", |b| {
        b.iter(|| {
            let r = zinc_snark::pipeline::verify_classic_logup_folded::<
                Sha256Uair, FoldedZt, FoldedLc, 32, 16, UNCHECKED, _, _,
            >(
                &folded_params, &gkr_folded_proof, SHA256_8X_NUM_VARS,
                |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
                &sha_public_cols,
            );
            black_box(r);
        });
    });

    // ── 19b. E2E Prover/Verifier (GKR folded, 4-chunk: 4×2^8) ──────
    //
    // Same GKR-based prover but with chunk_width=8 → 4 chunks of 256
    // entries each (vs. the default 8 chunks of 16).
    let sha_lookup_specs_4c_gkr = sha256_lookup_specs_4chunks();
    let sha_affine_specs_4c_gkr = sha256_affine_lookup_specs_4chunks();

    let gkr_folded_proof_4c = zinc_snark::pipeline::prove_gkr_logup_folded::<
        Sha256Uair, FoldedZt, FoldedLc, 32, 16, UNCHECKED,
    >(
        &folded_params, &sha_trace, SHA256_8X_NUM_VARS,
        &sha_lookup_specs_4c_gkr, &sha_affine_specs_4c_gkr,
    );

    {
        let r = zinc_snark::pipeline::verify_classic_logup_folded::<
            Sha256Uair, FoldedZt, FoldedLc, 32, 16, UNCHECKED, _, _,
        >(
            &folded_params, &gkr_folded_proof_4c, SHA256_8X_NUM_VARS,
            |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
            &sha_public_cols,
        );
        let t = &r.timing;
        println!("\n── Verifier step timing (GKR folded 4-chunk) ────────────");
        println!("  IC verify:           {:>8.3} ms", t.ideal_check_verify.as_secs_f64() * 1000.0);
        println!("  CPR+Lookup verify:   {:>8.3} ms", t.combined_poly_resolver_verify.as_secs_f64() * 1000.0);
        println!("  Lookup verify:       {:>8.3} ms", t.lookup_verify.as_secs_f64() * 1000.0);
        println!("  PCS verify:          {:>8.3} ms", t.pcs_verify.as_secs_f64() * 1000.0);
        println!("  Total:               {:>8.3} ms", t.total.as_secs_f64() * 1000.0);
        println!("─────────────────────────────────────────────────────────\n");
    }

    group.bench_function("E2E/Prover (GKR folded 4-chunk)", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let t = Instant::now();
                let _ = zinc_snark::pipeline::prove_gkr_logup_folded::<
                    Sha256Uair, FoldedZt, FoldedLc, 32, 16, UNCHECKED,
                >(
                    &folded_params, &sha_trace, SHA256_8X_NUM_VARS,
                    &sha_lookup_specs_4c_gkr, &sha_affine_specs_4c_gkr,
                );
                total += t.elapsed();
            }
            total
        });
    });

    group.bench_function("E2E/Verifier (GKR folded 4-chunk)", |b| {
        b.iter(|| {
            let r = zinc_snark::pipeline::verify_classic_logup_folded::<
                Sha256Uair, FoldedZt, FoldedLc, 32, 16, UNCHECKED, _, _,
            >(
                &folded_params, &gkr_folded_proof_4c, SHA256_8X_NUM_VARS,
                |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
                &sha_public_cols,
            );
            black_box(r);
        });
    });


    // ── 20. E2E Total Prover (4x folded pipeline) ───────────────────
    group.bench_function("E2E/Prover (4x folded)", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let t = Instant::now();
                let _ = zinc_snark::pipeline::prove_classic_logup_4x_folded::<
                    Sha256Uair, FoldedZt4x, FoldedLc4x, 32, 16, 8, UNCHECKED,
                >(
                    &folded_4x_params, &sha_trace, SHA256_8X_NUM_VARS,
                    &sha_lookup_specs, &sha_affine_specs,
                );
                total += t.elapsed();
            }
            total
        });
    });

    // ── 21. E2E Total Verifier (4x folded) ──────────────────────────
    let folded_4x_proof = zinc_snark::pipeline::prove_classic_logup_4x_folded::<
        Sha256Uair, FoldedZt4x, FoldedLc4x, 32, 16, 8, UNCHECKED,
    >(
        &folded_4x_params, &sha_trace, SHA256_8X_NUM_VARS,
        &sha_lookup_specs, &sha_affine_specs,
    );

    {
        let r = zinc_snark::pipeline::verify_classic_logup_4x_folded::<
            Sha256Uair, FoldedZt4x, FoldedLc4x, 32, 16, 8, UNCHECKED, _, _,
        >(
            &folded_4x_params, &folded_4x_proof, SHA256_8X_NUM_VARS,
            |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
            &sha_public_cols,
        );
        let t = &r.timing;
        println!("\n── Verifier step timing (4x folded) ─────────────────────");
        println!("  IC verify:           {:>8.3} ms", t.ideal_check_verify.as_secs_f64() * 1000.0);
        println!("  CPR+Lookup verify:   {:>8.3} ms", t.combined_poly_resolver_verify.as_secs_f64() * 1000.0);
        println!("  Lookup verify:       {:>8.3} ms", t.lookup_verify.as_secs_f64() * 1000.0);
        println!("  PCS verify:          {:>8.3} ms", t.pcs_verify.as_secs_f64() * 1000.0);
        println!("  Total:               {:>8.3} ms", t.total.as_secs_f64() * 1000.0);
        println!("─────────────────────────────────────────────────────────\n");
    }

    group.bench_function("E2E/Verifier (4x folded)", |b| {
        b.iter(|| {
            let r = zinc_snark::pipeline::verify_classic_logup_4x_folded::<
                Sha256Uair, FoldedZt4x, FoldedLc4x, 32, 16, 8, UNCHECKED, _, _,
            >(
                &folded_4x_params, &folded_4x_proof, SHA256_8X_NUM_VARS,
                |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
                &sha_public_cols,
            );
            black_box(r);
        });
    });

    // ── 22. E2E Prover/Verifier (4x folded, 4-chunk lookup: 4×2^8) ─
    //
    // Uses chunk_width=8, width=32 → 4 chunks of 2^8 = 256 entries each.
    // The default (steps 20/21) now uses 8 chunks; this variant benchmarks
    // the previous 4-chunk configuration for comparison.
    let sha_lookup_specs_4c = sha256_lookup_specs_4chunks();
    let sha_affine_specs_4c = sha256_affine_lookup_specs_4chunks();

    let folded_4x_proof_4c = zinc_snark::pipeline::prove_classic_logup_4x_folded::<
        Sha256Uair, FoldedZt4x, FoldedLc4x, 32, 16, 8, UNCHECKED,
    >(
        &folded_4x_params, &sha_trace, SHA256_8X_NUM_VARS,
        &sha_lookup_specs_4c, &sha_affine_specs_4c,
    );

    group.bench_function("E2E/Prover (4x folded 4-chunk)", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let t = Instant::now();
                let _ = zinc_snark::pipeline::prove_classic_logup_4x_folded::<
                    Sha256Uair, FoldedZt4x, FoldedLc4x, 32, 16, 8, UNCHECKED,
                >(
                    &folded_4x_params, &sha_trace, SHA256_8X_NUM_VARS,
                    &sha_lookup_specs_4c, &sha_affine_specs_4c,
                );
                total += t.elapsed();
            }
            total
        });
    });

    group.bench_function("E2E/Verifier (4x folded 4-chunk)", |b| {
        b.iter(|| {
            let r = zinc_snark::pipeline::verify_classic_logup_4x_folded::<
                Sha256Uair, FoldedZt4x, FoldedLc4x, 32, 16, 8, UNCHECKED, _, _,
            >(
                &folded_4x_params, &folded_4x_proof_4c, SHA256_8X_NUM_VARS,
                |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
                &sha_public_cols,
            );
            black_box(r);
        });
    });

    // ── 22b. E2E Prover/Verifier (4x folded, 4-chunk, Hybrid GKR c=2) ─
    let hybrid_4x_proof = zinc_snark::pipeline::prove_hybrid_gkr_logup_4x_folded::<
        Sha256Uair, FoldedZt4x, FoldedLc4x, 32, 16, 8, UNCHECKED,
    >(
        &folded_4x_params, &sha_trace, SHA256_8X_NUM_VARS,
        &sha_lookup_specs_4c, &sha_affine_specs_4c, 2,
    );

    // Print prover pipeline timing breakdown for hybrid 4x.
    {
        let t = &hybrid_4x_proof.timing;
        eprintln!("\n── Hybrid 4x GKR c=2 Prover Pipeline Timing ──────────");
        eprintln!("  PCS commit:    {:>8.3} ms", t.pcs_commit.as_secs_f64() * 1000.0);
        eprintln!("  Ideal Check:   {:>8.3} ms", t.ideal_check.as_secs_f64() * 1000.0);
        eprintln!("  CPR (incl proj+extract): {:>8.3} ms", t.combined_poly_resolver.as_secs_f64() * 1000.0);
        eprintln!("  Lookup:        {:>8.3} ms", t.lookup.as_secs_f64() * 1000.0);
        eprintln!("  PCS prove:     {:>8.3} ms", t.pcs_prove.as_secs_f64() * 1000.0);
        eprintln!("  Total:         {:>8.3} ms", t.total.as_secs_f64() * 1000.0);
        let accounted = t.pcs_commit + t.ideal_check + t.combined_poly_resolver + t.lookup + t.pcs_prove;
        let unaccounted = t.total.saturating_sub(accounted);
        eprintln!("  Unaccounted:   {:>8.3} ms (split+fold+shift+serialize)", unaccounted.as_secs_f64() * 1000.0);
        eprintln!("────────────────────────────────────────────────────────\n");
    }

    group.bench_function("E2E/Prover (4x Hybrid GKR c=2 4-chunk)", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let t = Instant::now();
                let _ = zinc_snark::pipeline::prove_hybrid_gkr_logup_4x_folded::<
                    Sha256Uair, FoldedZt4x, FoldedLc4x, 32, 16, 8, UNCHECKED,
                >(
                    &folded_4x_params, &sha_trace, SHA256_8X_NUM_VARS,
                    &sha_lookup_specs_4c, &sha_affine_specs_4c, 2,
                );
                total += t.elapsed();
            }
            total
        });
    });

    {
        let r = zinc_snark::pipeline::verify_classic_logup_4x_folded::<
            Sha256Uair, FoldedZt4x, FoldedLc4x, 32, 16, 8, UNCHECKED, _, _,
        >(
            &folded_4x_params, &hybrid_4x_proof, SHA256_8X_NUM_VARS,
            |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
            &sha_public_cols,
        );
        let t = &r.timing;
        println!("\n── Verifier step timing (4x Hybrid GKR c=2 4-chunk) ────");
        println!("  IC verify:           {:>8.3} ms", t.ideal_check_verify.as_secs_f64() * 1000.0);
        println!("  CPR+Lookup verify:   {:>8.3} ms", t.combined_poly_resolver_verify.as_secs_f64() * 1000.0);
        println!("  Lookup verify:       {:>8.3} ms", t.lookup_verify.as_secs_f64() * 1000.0);
        println!("  PCS verify:          {:>8.3} ms", t.pcs_verify.as_secs_f64() * 1000.0);
        println!("  Total:               {:>8.3} ms", t.total.as_secs_f64() * 1000.0);
        println!("─────────────────────────────────────────────────────────\n");
    }

    group.bench_function("E2E/Verifier (4x Hybrid GKR c=2 4-chunk)", |b| {
        b.iter(|| {
            let r = zinc_snark::pipeline::verify_classic_logup_4x_folded::<
                Sha256Uair, FoldedZt4x, FoldedLc4x, 32, 16, 8, UNCHECKED, _, _,
            >(
                &folded_4x_params, &hybrid_4x_proof, SHA256_8X_NUM_VARS,
                |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
                &sha_public_cols,
            );
            black_box(r);
        });
    });


    // ── Proof size breakdown (4x folded) — raw + compressed ─────────
    {
        use zinc_snark::pipeline::{FIELD_LIMBS, LookupProofData, PiopField};

        let fe_bytes = <Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;

        // Helper to serialise a single PiopField into a byte buffer.
        fn write_fe(buf: &mut Vec<u8>, f: &PiopField) {
            use zinc_snark::pipeline::FIELD_LIMBS;
            let sz = <Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;
            let start = buf.len();
            buf.resize(start + sz, 0);
            f.inner().write_transcription_bytes(&mut buf[start..]);
        }

        // ── 4x raw sizes ─────────────────────────────────────────────
        let pcs_4x     = folded_4x_proof.pcs_proof_bytes.len();
        let ic_4x: usize  = folded_4x_proof.ic_proof_values.iter().map(|v| v.len()).sum();
        let fold_c1_4x: usize = folded_4x_proof.folding_c1s_bytes.iter().map(|v| v.len()).sum();
        let fold_c2_4x: usize = folded_4x_proof.folding_c2s_bytes.iter().map(|v| v.len()).sum();
        let fold_c3_4x: usize = folded_4x_proof.folding_c3s_bytes.iter().map(|v| v.len()).sum();
        let fold_c4_4x: usize = folded_4x_proof.folding_c4s_bytes.iter().map(|v| v.len()).sum();
        let folding_4x = fold_c1_4x + fold_c2_4x + fold_c3_4x + fold_c4_4x;
        let cpr_up_4x: usize   = folded_4x_proof.cpr_up_evals.iter().map(|v| v.len()).sum();
        let cpr_dn_4x: usize   = folded_4x_proof.cpr_down_evals.iter().map(|v| v.len()).sum();
        let eval_pt_4x: usize  = folded_4x_proof.evaluation_point_bytes.iter().map(|v| v.len()).sum();
        let pcs_eval_4x: usize = folded_4x_proof.pcs_evals_bytes.iter().map(|v| v.len()).sum();
        let cpr_sc_4x: usize   = folded_4x_proof.cpr_sumcheck_messages.iter().map(|v| v.len()).sum::<usize>()
            + folded_4x_proof.cpr_sumcheck_claimed_sum.len();

        let lk_4x: usize = match &folded_4x_proof.lookup_proof {
            Some(LookupProofData::BatchedClassic(bp)) => {
                let mut t = 0usize;
                for gp in &bp.lookup_group_proofs {
                    let m: usize = gp.aggregated_multiplicities.iter().map(|v| v.len()).sum();
                    let w: usize = gp.chunk_inverse_witnesses.iter()
                        .flat_map(|o| o.iter()).map(|i| i.len()).sum();
                    t += (m + w + gp.inverse_table.len()) * fe_bytes;
                }
                // MD sumcheck messages and claimed sums.
                t += bp.md_proof.group_messages.iter()
                    .flat_map(|g| g.iter())
                    .map(|m| m.0.tail_evaluations.len() * fe_bytes)
                    .sum::<usize>();
                t += bp.md_proof.claimed_sums.len() * fe_bytes;
                t
            }
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

        let shift_4x: usize = folded_4x_proof.shift_sumcheck.as_ref().map_or(0, |sc| {
            sc.rounds.iter().map(|v| v.len()).sum::<usize>()
                + sc.v_finals.iter().map(|v| v.len()).sum::<usize>()
        });

        let total_4x = pcs_4x + ic_4x + cpr_sc_4x + cpr_up_4x + cpr_dn_4x
            + lk_4x + shift_4x + folding_4x + eval_pt_4x + pcs_eval_4x;

        // ── Serialise all bytes for compression ───────────────────────
        let mut all_bytes_4x: Vec<u8> = Vec::with_capacity(total_4x);
        all_bytes_4x.extend(&folded_4x_proof.pcs_proof_bytes);
        for v in &folded_4x_proof.ic_proof_values { all_bytes_4x.extend(v); }
        for v in &folded_4x_proof.cpr_sumcheck_messages { all_bytes_4x.extend(v); }
        all_bytes_4x.extend(&folded_4x_proof.cpr_sumcheck_claimed_sum);
        for v in &folded_4x_proof.cpr_up_evals   { all_bytes_4x.extend(v); }
        for v in &folded_4x_proof.cpr_down_evals  { all_bytes_4x.extend(v); }
        // Serialise lookup proof (BatchedClassic stores PiopField values).
        if let Some(LookupProofData::BatchedClassic(bp)) = &folded_4x_proof.lookup_proof {
            for sum in &bp.md_proof.claimed_sums { write_fe(&mut all_bytes_4x, sum); }
            for group_msgs in &bp.md_proof.group_messages {
                for msg in group_msgs {
                    for e in &msg.0.tail_evaluations { write_fe(&mut all_bytes_4x, e); }
                }
            }
            for gp in &bp.lookup_group_proofs {
                for v in &gp.aggregated_multiplicities {
                    for f in v { write_fe(&mut all_bytes_4x, f); }
                }
                for outer in &gp.chunk_inverse_witnesses {
                    for inner in outer { for f in inner { write_fe(&mut all_bytes_4x, f); } }
                }
                for f in &gp.inverse_table { write_fe(&mut all_bytes_4x, f); }
            }
        }
        if let Some(ref sc) = folded_4x_proof.shift_sumcheck {
            for v in &sc.rounds  { all_bytes_4x.extend(v); }
            for v in &sc.v_finals { all_bytes_4x.extend(v); }
        }
        for v in &folded_4x_proof.folding_c1s_bytes { all_bytes_4x.extend(v); }
        for v in &folded_4x_proof.folding_c2s_bytes { all_bytes_4x.extend(v); }
        for v in &folded_4x_proof.folding_c3s_bytes { all_bytes_4x.extend(v); }
        for v in &folded_4x_proof.folding_c4s_bytes { all_bytes_4x.extend(v); }
        for v in &folded_4x_proof.evaluation_point_bytes { all_bytes_4x.extend(v); }
        for v in &folded_4x_proof.pcs_evals_bytes { all_bytes_4x.extend(v); }

        let compressed_4x = {
            use std::io::Write;
            let mut enc = flate2::write::DeflateEncoder::new(
                Vec::new(), flate2::Compression::default(),
            );
            enc.write_all(&all_bytes_4x).unwrap();
            enc.finish().unwrap()
        };

        // ── 2x folded raw sizes (for comparison) ─────────────────────
        let pcs_2x     = folded_proof.pcs_proof_bytes.len();
        let ic_2x: usize  = folded_proof.ic_proof_values.iter().map(|v| v.len()).sum();
        let fold_c1_2x: usize = folded_proof.folding_c1s_bytes.iter().map(|v| v.len()).sum();
        let fold_c2_2x: usize = folded_proof.folding_c2s_bytes.iter().map(|v| v.len()).sum();
        let cpr_up_2x: usize   = folded_proof.cpr_up_evals.iter().map(|v| v.len()).sum();
        let cpr_dn_2x: usize   = folded_proof.cpr_down_evals.iter().map(|v| v.len()).sum();
        let eval_pt_2x: usize  = folded_proof.evaluation_point_bytes.iter().map(|v| v.len()).sum();
        let pcs_eval_2x: usize = folded_proof.pcs_evals_bytes.iter().map(|v| v.len()).sum();
        let cpr_sc_2x: usize   = folded_proof.cpr_sumcheck_messages.iter().map(|v| v.len()).sum::<usize>()
            + folded_proof.cpr_sumcheck_claimed_sum.len();
        let lk_2x: usize = match &folded_proof.lookup_proof {
            Some(LookupProofData::BatchedClassic(bp)) => {
                let mut t = 0usize;
                for gp in &bp.lookup_group_proofs {
                    let m: usize = gp.aggregated_multiplicities.iter().map(|v| v.len()).sum();
                    let w: usize = gp.chunk_inverse_witnesses.iter()
                        .flat_map(|o| o.iter()).map(|i| i.len()).sum();
                    t += (m + w + gp.inverse_table.len()) * fe_bytes;
                }
                t += bp.md_proof.group_messages.iter().flat_map(|g| g.iter())
                    .map(|m| m.0.tail_evaluations.len() * fe_bytes).sum::<usize>();
                t += bp.md_proof.claimed_sums.len() * fe_bytes;
                t
            }
            Some(LookupProofData::Classic(p)) => {
                p.group_proofs.iter().map(|gp| {
                    let m: usize = gp.aggregated_multiplicities.iter().map(|v| v.len()).sum();
                    let w: usize = gp.chunk_inverse_witnesses.iter()
                        .flat_map(|o| o.iter()).map(|i| i.len()).sum();
                    (m + w + gp.inverse_table.len()) * fe_bytes
                }).sum()
            }
            _ => 0,
        };
        let shift_2x: usize = folded_proof.shift_sumcheck.as_ref().map_or(0, |sc| {
            sc.rounds.iter().map(|v| v.len()).sum::<usize>()
                + sc.v_finals.iter().map(|v| v.len()).sum::<usize>()
        });
        let total_2x = pcs_2x + ic_2x + cpr_sc_2x + cpr_up_2x + cpr_dn_2x
            + lk_2x + shift_2x + fold_c1_2x + fold_c2_2x + eval_pt_2x + pcs_eval_2x;

        let mut all_bytes_2x: Vec<u8> = Vec::with_capacity(total_2x);
        all_bytes_2x.extend(&folded_proof.pcs_proof_bytes);
        for v in &folded_proof.ic_proof_values { all_bytes_2x.extend(v); }
        for v in &folded_proof.cpr_sumcheck_messages { all_bytes_2x.extend(v); }
        all_bytes_2x.extend(&folded_proof.cpr_sumcheck_claimed_sum);
        for v in &folded_proof.cpr_up_evals  { all_bytes_2x.extend(v); }
        for v in &folded_proof.cpr_down_evals { all_bytes_2x.extend(v); }
        if let Some(LookupProofData::BatchedClassic(bp)) = &folded_proof.lookup_proof {
            for sum in &bp.md_proof.claimed_sums { write_fe(&mut all_bytes_2x, sum); }
            for group_msgs in &bp.md_proof.group_messages {
                for msg in group_msgs {
                    for e in &msg.0.tail_evaluations { write_fe(&mut all_bytes_2x, e); }
                }
            }
            for gp in &bp.lookup_group_proofs {
                for v in &gp.aggregated_multiplicities {
                    for f in v { write_fe(&mut all_bytes_2x, f); }
                }
                for outer in &gp.chunk_inverse_witnesses {
                    for inner in outer { for f in inner { write_fe(&mut all_bytes_2x, f); } }
                }
                for f in &gp.inverse_table { write_fe(&mut all_bytes_2x, f); }
            }
        }
        if let Some(ref sc) = folded_proof.shift_sumcheck {
            for v in &sc.rounds  { all_bytes_2x.extend(v); }
            for v in &sc.v_finals { all_bytes_2x.extend(v); }
        }
        for v in &folded_proof.folding_c1s_bytes { all_bytes_2x.extend(v); }
        for v in &folded_proof.folding_c2s_bytes { all_bytes_2x.extend(v); }
        for v in &folded_proof.evaluation_point_bytes { all_bytes_2x.extend(v); }
        for v in &folded_proof.pcs_evals_bytes { all_bytes_2x.extend(v); }

        let compressed_2x = {
            use std::io::Write;
            let mut enc = flate2::write::DeflateEncoder::new(
                Vec::new(), flate2::Compression::default(),
            );
            enc.write_all(&all_bytes_2x).unwrap();
            enc.finish().unwrap()
        };

        // ── Print ─────────────────────────────────────────────────────
        eprintln!("\n=== 8xSHA256 4x Folded Proof Size ===");
        eprintln!("  PCS:            {:>6} B  ({:.1} KB)", pcs_4x, pcs_4x as f64 / 1024.0);
        eprintln!("  IC:             {:>6} B", ic_4x);
        eprintln!("  CPR sumcheck:   {:>6} B", cpr_sc_4x);
        eprintln!("  CPR evals:      {:>6} B  (up={cpr_up_4x}, down={cpr_dn_4x})",
            cpr_up_4x + cpr_dn_4x);
        eprintln!("  Lookup:         {:>6} B", lk_4x);
        eprintln!("  Shift SC:       {:>6} B", shift_4x);
        eprintln!("  Folding:        {:>6} B  (c1={fold_c1_4x}, c2={fold_c2_4x}, c3={fold_c3_4x}, c4={fold_c4_4x})", folding_4x);
        eprintln!("  Eval point:     {:>6} B", eval_pt_4x);
        eprintln!("  PCS evals:      {:>6} B", pcs_eval_4x);
        eprintln!("  ─────────────────────────");
        eprintln!("  Total raw:      {:>6} B  ({:.1} KB)", total_4x, total_4x as f64 / 1024.0);
        eprintln!("  Compressed:     {:>6} B  ({:.1} KB, {:.1}x ratio)",
            compressed_4x.len(), compressed_4x.len() as f64 / 1024.0,
            all_bytes_4x.len() as f64 / compressed_4x.len() as f64);

        eprintln!("\n=== 4x Folded vs 2x Folded Comparison ===");
        eprintln!("  2x PCS:         {:>6} B  ({:.1} KB)", pcs_2x, pcs_2x as f64 / 1024.0);
        eprintln!("  4x PCS:         {:>6} B  ({:.1} KB)", pcs_4x, pcs_4x as f64 / 1024.0);
        let pcs_diff = pcs_2x as i64 - pcs_4x as i64;
        eprintln!("  PCS savings:    {:>6} B  ({:+.1} KB)", pcs_diff, pcs_diff as f64 / 1024.0);
        eprintln!("  ─────────────────────────");
        eprintln!("  2x Folding:     {:>6} B  (c1={fold_c1_2x}, c2={fold_c2_2x})",
            fold_c1_2x + fold_c2_2x);
        eprintln!("  4x Folding:     {:>6} B  (c1={fold_c1_4x}, c2={fold_c2_4x}, c3={fold_c3_4x}, c4={fold_c4_4x})",
            folding_4x);
        eprintln!("  ─────────────────────────");
        eprintln!("  2x total raw:   {:>6} B  ({:.1} KB)", total_2x, total_2x as f64 / 1024.0);
        eprintln!("  4x total raw:   {:>6} B  ({:.1} KB)", total_4x, total_4x as f64 / 1024.0);
        let raw_diff = total_2x as i64 - total_4x as i64;
        eprintln!("  Raw savings:    {:>6} B  ({:+.1} KB, {:.2}x)",
            raw_diff, raw_diff as f64 / 1024.0,
            total_2x as f64 / total_4x as f64);
        eprintln!("  ─────────────────────────");
        eprintln!("  2x compressed:  {:>6} B  ({:.1} KB)", compressed_2x.len(), compressed_2x.len() as f64 / 1024.0);
        eprintln!("  4x compressed:  {:>6} B  ({:.1} KB)", compressed_4x.len(), compressed_4x.len() as f64 / 1024.0);
        let compr_diff = compressed_2x.len() as i64 - compressed_4x.len() as i64;
        eprintln!("  Compr savings:  {:>6} B  ({:+.1} KB, {:.2}x)",
            compr_diff, compr_diff as f64 / 1024.0,
            compressed_2x.len() as f64 / compressed_4x.len() as f64);
    }

    // ── Proof size breakdown for chunk variants ─────────────────────
    //
    // Compute raw + compressed proof size for each chunk configuration
    // across both 2x and 4x folding.
    {
        use zinc_snark::pipeline::{FIELD_LIMBS, PiopField};

        let fe_bytes = <Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;

        fn write_fe_cv(buf: &mut Vec<u8>, f: &PiopField) {
            use zinc_snark::pipeline::FIELD_LIMBS;
            let sz = <Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;
            let start = buf.len();
            buf.resize(start + sz, 0);
            f.inner().write_transcription_bytes(&mut buf[start..]);
        }

        /// Compute raw and compressed proof size for a 2x FoldedZincProof.
        /// Returns (raw_total, compressed_len).
        fn proof_size_2x(
            proof: &zinc_snark::pipeline::FoldedZincProof,
            label: &str,
            fe_bytes: usize,
        ) -> (usize, usize) {
            use zinc_snark::pipeline::LookupProofData;

            let pcs     = proof.pcs_proof_bytes.len();
            let ic: usize  = proof.ic_proof_values.iter().map(|v| v.len()).sum();
            let fold_c1: usize = proof.folding_c1s_bytes.iter().map(|v| v.len()).sum();
            let fold_c2: usize = proof.folding_c2s_bytes.iter().map(|v| v.len()).sum();
            let folding = fold_c1 + fold_c2;
            let cpr_up: usize   = proof.cpr_up_evals.iter().map(|v| v.len()).sum();
            let cpr_dn: usize   = proof.cpr_down_evals.iter().map(|v| v.len()).sum();
            let eval_pt: usize  = proof.evaluation_point_bytes.iter().map(|v| v.len()).sum();
            let pcs_eval: usize = proof.pcs_evals_bytes.iter().map(|v| v.len()).sum();
            let cpr_sc: usize   = proof.cpr_sumcheck_messages.iter().map(|v| v.len()).sum::<usize>()
                + proof.cpr_sumcheck_claimed_sum.len();

            let lk: usize = match &proof.lookup_proof {
                Some(LookupProofData::BatchedClassic(bp)) => {
                    let mut t = 0usize;
                    for gp in &bp.lookup_group_proofs {
                        let m: usize = gp.aggregated_multiplicities.iter().map(|v| v.len()).sum();
                        let w: usize = gp.chunk_inverse_witnesses.iter()
                            .flat_map(|o| o.iter()).map(|i| i.len()).sum();
                        t += (m + w + gp.inverse_table.len()) * fe_bytes;
                    }
                    t += bp.md_proof.group_messages.iter()
                        .flat_map(|g| g.iter())
                        .map(|m| m.0.tail_evaluations.len() * fe_bytes)
                        .sum::<usize>();
                    t += bp.md_proof.claimed_sums.len() * fe_bytes;
                    t
                }
                _ => 0,
            };
            let shift: usize = proof.shift_sumcheck.as_ref().map_or(0, |sc| {
                sc.rounds.iter().map(|v| v.len()).sum::<usize>()
                    + sc.v_finals.iter().map(|v| v.len()).sum::<usize>()
            });
            let total = pcs + ic + cpr_sc + cpr_up + cpr_dn + lk + shift + folding + eval_pt + pcs_eval;

            let mut all_bytes: Vec<u8> = Vec::with_capacity(total);
            all_bytes.extend(&proof.pcs_proof_bytes);
            for v in &proof.ic_proof_values { all_bytes.extend(v); }
            for v in &proof.cpr_sumcheck_messages { all_bytes.extend(v); }
            all_bytes.extend(&proof.cpr_sumcheck_claimed_sum);
            for v in &proof.cpr_up_evals  { all_bytes.extend(v); }
            for v in &proof.cpr_down_evals { all_bytes.extend(v); }
            if let Some(LookupProofData::BatchedClassic(bp)) = &proof.lookup_proof {
                for sum in &bp.md_proof.claimed_sums { write_fe_cv(&mut all_bytes, sum); }
                for group_msgs in &bp.md_proof.group_messages {
                    for msg in group_msgs {
                        for e in &msg.0.tail_evaluations { write_fe_cv(&mut all_bytes, e); }
                    }
                }
                for gp in &bp.lookup_group_proofs {
                    for v in &gp.aggregated_multiplicities {
                        for f in v { write_fe_cv(&mut all_bytes, f); }
                    }
                    for outer in &gp.chunk_inverse_witnesses {
                        for inner in outer { for f in inner { write_fe_cv(&mut all_bytes, f); } }
                    }
                    for f in &gp.inverse_table { write_fe_cv(&mut all_bytes, f); }
                }
            }
            if let Some(ref sc) = proof.shift_sumcheck {
                for v in &sc.rounds  { all_bytes.extend(v); }
                for v in &sc.v_finals { all_bytes.extend(v); }
            }
            for v in &proof.folding_c1s_bytes { all_bytes.extend(v); }
            for v in &proof.folding_c2s_bytes { all_bytes.extend(v); }
            for v in &proof.evaluation_point_bytes { all_bytes.extend(v); }
            for v in &proof.pcs_evals_bytes { all_bytes.extend(v); }

            let compressed = {
                use std::io::Write;
                let mut enc = flate2::write::DeflateEncoder::new(
                    Vec::new(), flate2::Compression::default(),
                );
                enc.write_all(&all_bytes).unwrap();
                enc.finish().unwrap()
            };

            eprintln!("\n=== {label} Proof Size ===");
            eprintln!("  PCS:            {:>6} B  ({:.1} KB)", pcs, pcs as f64 / 1024.0);
            eprintln!("  IC:             {:>6} B", ic);
            eprintln!("  CPR sumcheck:   {:>6} B", cpr_sc);
            eprintln!("  CPR evals:      {:>6} B  (up={cpr_up}, down={cpr_dn})", cpr_up + cpr_dn);
            eprintln!("  Lookup:         {:>6} B  ({:.1} KB)", lk, lk as f64 / 1024.0);
            eprintln!("  Shift SC:       {:>6} B", shift);
            eprintln!("  Folding:        {:>6} B  (c1={fold_c1}, c2={fold_c2})", folding);
            eprintln!("  Eval point:     {:>6} B", eval_pt);
            eprintln!("  PCS evals:      {:>6} B", pcs_eval);
            eprintln!("  ─────────────────────────");
            eprintln!("  Total raw:      {:>6} B  ({:.1} KB)", total, total as f64 / 1024.0);
            eprintln!("  Compressed:     {:>6} B  ({:.1} KB, {:.1}x ratio)",
                compressed.len(), compressed.len() as f64 / 1024.0,
                all_bytes.len() as f64 / compressed.len() as f64);

            (total, compressed.len())
        }

        /// Compute raw and compressed proof size for a 4x Folded4xZincProof.
        /// Returns (raw_total, compressed_len).
        fn proof_size_4x(
            proof: &zinc_snark::pipeline::Folded4xZincProof,
            label: &str,
            fe_bytes: usize,
        ) -> (usize, usize) {
            use zinc_snark::pipeline::LookupProofData;

            let pcs     = proof.pcs_proof_bytes.len();
            let ic: usize  = proof.ic_proof_values.iter().map(|v| v.len()).sum();
            let fold_c1: usize = proof.folding_c1s_bytes.iter().map(|v| v.len()).sum();
            let fold_c2: usize = proof.folding_c2s_bytes.iter().map(|v| v.len()).sum();
            let fold_c3: usize = proof.folding_c3s_bytes.iter().map(|v| v.len()).sum();
            let fold_c4: usize = proof.folding_c4s_bytes.iter().map(|v| v.len()).sum();
            let folding = fold_c1 + fold_c2 + fold_c3 + fold_c4;
            let cpr_up: usize   = proof.cpr_up_evals.iter().map(|v| v.len()).sum();
            let cpr_dn: usize   = proof.cpr_down_evals.iter().map(|v| v.len()).sum();
            let eval_pt: usize  = proof.evaluation_point_bytes.iter().map(|v| v.len()).sum();
            let pcs_eval: usize = proof.pcs_evals_bytes.iter().map(|v| v.len()).sum();
            let cpr_sc: usize   = proof.cpr_sumcheck_messages.iter().map(|v| v.len()).sum::<usize>()
                + proof.cpr_sumcheck_claimed_sum.len();

            let lk: usize = match &proof.lookup_proof {
                Some(LookupProofData::BatchedClassic(bp)) => {
                    let mut t = 0usize;
                    for gp in &bp.lookup_group_proofs {
                        let m: usize = gp.aggregated_multiplicities.iter().map(|v| v.len()).sum();
                        let w: usize = gp.chunk_inverse_witnesses.iter()
                            .flat_map(|o| o.iter()).map(|i| i.len()).sum();
                        t += (m + w + gp.inverse_table.len()) * fe_bytes;
                    }
                    t += bp.md_proof.group_messages.iter()
                        .flat_map(|g| g.iter())
                        .map(|m| m.0.tail_evaluations.len() * fe_bytes)
                        .sum::<usize>();
                    t += bp.md_proof.claimed_sums.len() * fe_bytes;
                    t
                }
                Some(LookupProofData::HybridGkr(hp)) => {
                    let mut t = 0usize;
                    for group in &hp.group_proofs {
                        let m: usize = group.aggregated_multiplicities.iter().map(|v| v.len()).sum();
                        t += m * fe_bytes;
                        let wg = &group.witness_gkr;
                        t += (wg.roots_p.len() + wg.roots_q.len()) * fe_bytes;
                        for lp in &wg.layer_proofs {
                            if let Some(ref sc) = lp.sumcheck_proof {
                                t += sc.messages.iter().map(|m| m.0.tail_evaluations.len()).sum::<usize>() * fe_bytes;
                                t += fe_bytes;
                            }
                            t += (lp.p_lefts.len() + lp.p_rights.len() + lp.q_lefts.len() + lp.q_rights.len()) * fe_bytes;
                        }
                        let sent_p: usize = wg.sent_p.iter().map(|v| v.len()).sum();
                        let sent_q: usize = wg.sent_q.iter().map(|v| v.len()).sum();
                        t += (sent_p + sent_q) * fe_bytes;
                        let tg = &group.table_gkr;
                        t += 2 * fe_bytes;
                        for lp in &tg.layer_proofs {
                            if let Some(ref sc) = lp.sumcheck_proof {
                                t += sc.messages.iter().map(|m| m.0.tail_evaluations.len()).sum::<usize>() * fe_bytes;
                                t += fe_bytes;
                            }
                            t += 4 * fe_bytes;
                        }
                    }
                    t
                }
                _ => 0,
            };
            let shift: usize = proof.shift_sumcheck.as_ref().map_or(0, |sc| {
                sc.rounds.iter().map(|v| v.len()).sum::<usize>()
                    + sc.v_finals.iter().map(|v| v.len()).sum::<usize>()
            });
            let total = pcs + ic + cpr_sc + cpr_up + cpr_dn + lk + shift + folding + eval_pt + pcs_eval;

            let mut all_bytes: Vec<u8> = Vec::with_capacity(total);
            all_bytes.extend(&proof.pcs_proof_bytes);
            for v in &proof.ic_proof_values { all_bytes.extend(v); }
            for v in &proof.cpr_sumcheck_messages { all_bytes.extend(v); }
            all_bytes.extend(&proof.cpr_sumcheck_claimed_sum);
            for v in &proof.cpr_up_evals  { all_bytes.extend(v); }
            for v in &proof.cpr_down_evals { all_bytes.extend(v); }
            if let Some(LookupProofData::BatchedClassic(bp)) = &proof.lookup_proof {
                for sum in &bp.md_proof.claimed_sums { write_fe_cv(&mut all_bytes, sum); }
                for group_msgs in &bp.md_proof.group_messages {
                    for msg in group_msgs {
                        for e in &msg.0.tail_evaluations { write_fe_cv(&mut all_bytes, e); }
                    }
                }
                for gp in &bp.lookup_group_proofs {
                    for v in &gp.aggregated_multiplicities {
                        for f in v { write_fe_cv(&mut all_bytes, f); }
                    }
                    for outer in &gp.chunk_inverse_witnesses {
                        for inner in outer { for f in inner { write_fe_cv(&mut all_bytes, f); } }
                    }
                    for f in &gp.inverse_table { write_fe_cv(&mut all_bytes, f); }
                }
            } else if let Some(LookupProofData::HybridGkr(hp)) = &proof.lookup_proof {
                for group in &hp.group_proofs {
                    for v in &group.aggregated_multiplicities {
                        for f in v { write_fe_cv(&mut all_bytes, f); }
                    }
                    let wg = &group.witness_gkr;
                    for f in &wg.roots_p { write_fe_cv(&mut all_bytes, f); }
                    for f in &wg.roots_q { write_fe_cv(&mut all_bytes, f); }
                    for lp in &wg.layer_proofs {
                        if let Some(ref sc) = lp.sumcheck_proof {
                            write_fe_cv(&mut all_bytes, &sc.claimed_sum);
                            for m in &sc.messages { for e in &m.0.tail_evaluations { write_fe_cv(&mut all_bytes, e); } }
                        }
                        for f in &lp.p_lefts { write_fe_cv(&mut all_bytes, f); }
                        for f in &lp.p_rights { write_fe_cv(&mut all_bytes, f); }
                        for f in &lp.q_lefts { write_fe_cv(&mut all_bytes, f); }
                        for f in &lp.q_rights { write_fe_cv(&mut all_bytes, f); }
                    }
                    for v in &wg.sent_p { for f in v { write_fe_cv(&mut all_bytes, f); } }
                    for v in &wg.sent_q { for f in v { write_fe_cv(&mut all_bytes, f); } }
                    let tg = &group.table_gkr;
                    write_fe_cv(&mut all_bytes, &tg.root_p);
                    write_fe_cv(&mut all_bytes, &tg.root_q);
                    for lp in &tg.layer_proofs {
                        if let Some(ref sc) = lp.sumcheck_proof {
                            write_fe_cv(&mut all_bytes, &sc.claimed_sum);
                            for m in &sc.messages { for e in &m.0.tail_evaluations { write_fe_cv(&mut all_bytes, e); } }
                        }
                        write_fe_cv(&mut all_bytes, &lp.p_left);
                        write_fe_cv(&mut all_bytes, &lp.p_right);
                        write_fe_cv(&mut all_bytes, &lp.q_left);
                        write_fe_cv(&mut all_bytes, &lp.q_right);
                    }
                }
            }
            if let Some(ref sc) = proof.shift_sumcheck {
                for v in &sc.rounds  { all_bytes.extend(v); }
                for v in &sc.v_finals { all_bytes.extend(v); }
            }
            for v in &proof.folding_c1s_bytes { all_bytes.extend(v); }
            for v in &proof.folding_c2s_bytes { all_bytes.extend(v); }
            for v in &proof.folding_c3s_bytes { all_bytes.extend(v); }
            for v in &proof.folding_c4s_bytes { all_bytes.extend(v); }
            for v in &proof.evaluation_point_bytes { all_bytes.extend(v); }
            for v in &proof.pcs_evals_bytes { all_bytes.extend(v); }

            let compressed = {
                use std::io::Write;
                let mut enc = flate2::write::DeflateEncoder::new(
                    Vec::new(), flate2::Compression::default(),
                );
                enc.write_all(&all_bytes).unwrap();
                enc.finish().unwrap()
            };

            eprintln!("\n=== {label} Proof Size ===");
            eprintln!("  PCS:            {:>6} B  ({:.1} KB)", pcs, pcs as f64 / 1024.0);
            eprintln!("  IC:             {:>6} B", ic);
            eprintln!("  CPR sumcheck:   {:>6} B", cpr_sc);
            eprintln!("  CPR evals:      {:>6} B  (up={cpr_up}, down={cpr_dn})", cpr_up + cpr_dn);
            eprintln!("  Lookup:         {:>6} B  ({:.1} KB)", lk, lk as f64 / 1024.0);
            eprintln!("  Shift SC:       {:>6} B", shift);
            eprintln!("  Folding:        {:>6} B  (c1={fold_c1}, c2={fold_c2}, c3={fold_c3}, c4={fold_c4})", folding);
            eprintln!("  Eval point:     {:>6} B", eval_pt);
            eprintln!("  PCS evals:      {:>6} B", pcs_eval);
            eprintln!("  ─────────────────────────");
            eprintln!("  Total raw:      {:>6} B  ({:.1} KB)", total, total as f64 / 1024.0);
            eprintln!("  Compressed:     {:>6} B  ({:.1} KB, {:.1}x ratio)",
                compressed.len(), compressed.len() as f64 / 1024.0,
                all_bytes.len() as f64 / compressed.len() as f64);

            (total, compressed.len())
        }

        // ── 2x folded, 4-chunk ────────────────────────────────────────
        let folded_proof_4c = zinc_snark::pipeline::prove_classic_logup_folded::<
            Sha256Uair, FoldedZt, FoldedLc, 32, 16, UNCHECKED,
        >(
            &folded_params, &sha_trace, SHA256_8X_NUM_VARS,
            &sha_lookup_specs_4c, &sha_affine_specs_4c,
        );
        let (raw_2x_4c, compr_2x_4c) = proof_size_2x(&folded_proof_4c, "2x Folded 4-chunk (4×2^8)", fe_bytes);

        // ── 2x folded, 8-chunk (default) ──────────────────────────────
        let (raw_2x_8c, compr_2x_8c) = proof_size_2x(&folded_proof, "2x Folded 8-chunk (8×2^4)", fe_bytes);

        // ── 4x folded, 4-chunk ────────────────────────────────────────
        let (raw_4x_4c, compr_4x_4c) = proof_size_4x(&folded_4x_proof_4c, "4x Folded 4-chunk (4×2^8)", fe_bytes);

        // ── 4x folded, 8-chunk (default) ──────────────────────────────
        let (raw_4x_8c, compr_4x_8c) = proof_size_4x(&folded_4x_proof, "4x Folded 8-chunk (8×2^4)", fe_bytes);

        // ── 4x Hybrid GKR c=2, 4-chunk ──────────────────────────────
        let (raw_hybrid_4x, compr_hybrid_4x) = proof_size_4x(&hybrid_4x_proof, "4x Hybrid GKR c=2 4-chunk", fe_bytes);



        // ── Summary comparison table ──────────────────────────────────
        eprintln!("\n=== Chunk Variant Proof Size Comparison ===");
        eprintln!("  {:35}  {:>8}  {:>8}", "Configuration", "Raw (B)", "Compr (B)");
        eprintln!("  {}", "─".repeat(55));
        eprintln!("  {:35}  {:>8}  {:>8}", "2x folded, 4-chunk (4×2^8)",  raw_2x_4c, compr_2x_4c);
        eprintln!("  {:35}  {:>8}  {:>8}", "2x folded, 8-chunk (8×2^4) *",  raw_2x_8c, compr_2x_8c);
        eprintln!("  {:35}  {:>8}  {:>8}", "4x folded, 4-chunk (4×2^8)",  raw_4x_4c, compr_4x_4c);
        eprintln!("  {:35}  {:>8}  {:>8}", "4x folded, 8-chunk (8×2^4) *",  raw_4x_8c, compr_4x_8c);
        eprintln!("  {:35}  {:>8}  {:>8}", "4x Hybrid GKR c=2, 4-chunk",  raw_hybrid_4x, compr_hybrid_4x);
        eprintln!("  (* = default configuration)");
    }

    // ══════════════════════════════════════════════════════════════════
    // ── Verifier Step-by-Step Breakdown (folded) ────────────────────
    // ══════════════════════════════════════════════════════════════════

    {
        use zinc_snark::pipeline::{
            FIELD_LIMBS, PiopField, LookupProofData, BatchedCprLookupProof,
            field_from_bytes, reconstruct_up_evals, reconstruct_shift_v_finals,
            TrivialIdeal, eval_affine_parent,
        };
        use zinc_piop::sumcheck::multi_degree::MultiDegreeSumcheck;
        use zinc_piop::lookup::BatchedDecompLogupProtocol;
        use zinc_piop::lookup::pipeline::generate_table_and_shifts;
        use zinc_piop::lookup::LookupWitnessSource;
        use zinc_poly::mle::MultilinearExtensionWithConfig;
        use zip_plus::pcs::folding::{fold_claims_verify, compute_alpha_power};

        let field_elem_size =
            <Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;

        // Extract the batched proof (prove_classic_logup_folded always
        // produces BatchedClassic, even without lookups).
        let batched_proof: &BatchedCprLookupProof = match &folded_proof.lookup_proof {
            Some(LookupProofData::BatchedClassic(bp)) => bp,
            other => panic!(
                "Expected BatchedClassic proof from prove_classic_logup_folded, got {:?}",
                other.as_ref().map(|_| "other variant"),
            ),
        };

        // Helper: deserialize IC proof values.
        let deser_ic = |proof_values: &[Vec<u8>], fcfg: &<PiopField as PrimeField>::Config| -> Vec<DynamicPolynomialF<PiopField>> {
            proof_values.iter().map(|bytes| {
                let n = bytes.len() / field_elem_size;
                DynamicPolynomialF::new(
                    (0..n).map(|i| field_from_bytes(
                        &bytes[i * field_elem_size..(i + 1) * field_elem_size], fcfg,
                    )).collect::<Vec<_>>()
                )
            }).collect()
        };

        // Helper: replay transcript through IC verify.
        let replay_through_ic = |tr: &mut zinc_transcript::KeccakTranscript, fcfg: &<PiopField as PrimeField>::Config|
            -> zinc_piop::ideal_check::VerifierSubClaim<PiopField> {
            IdealCheckProtocol::<PiopField>::verify_as_subprotocol::<Sha256Uair, _, _>(
                tr,
                zinc_piop::ideal_check::Proof { combined_mle_values: deser_ic(&folded_proof.ic_proof_values, fcfg) },
                num_constraints, num_vars,
                |_: &IdealOrZero<CyclotomicIdeal>| TrivialIdeal,
                fcfg,
            ).unwrap()
        };

        // Helper: project scalars for verifier-side main field sumcheck.
        let verifier_proj_scalars = |fcfg: &<PiopField as PrimeField>::Config, proj_elem: &PiopField| -> std::collections::HashMap<BinaryPoly<32>, PiopField> {
            let psc = project_scalars::<PiopField, Sha256Uair>(|scalar| {
                let one = PiopField::one_with_cfg(fcfg);
                let zero = PiopField::zero_with_cfg(fcfg);
                DynamicPolynomialF::new(
                    scalar.iter().map(|coeff| {
                        if coeff.into_inner() { one.clone() } else { zero.clone() }
                    }).collect::<Vec<_>>()
                )
            });
            project_scalars_to_field(psc, proj_elem).unwrap()
        };

        // Helper: replay main field sumcheck pre-sumcheck using the batched claimed sum.
        let replay_cpr_pre = |tr: &mut zinc_transcript::KeccakTranscript,
                              fcfg: &<PiopField as PrimeField>::Config,
                              ic_subclaim: &zinc_piop::ideal_check::VerifierSubClaim<PiopField>|
            -> (PiopField, std::collections::HashMap<BinaryPoly<32>, PiopField>,
                zinc_piop::combined_poly_resolver::CprVerifierPreSumcheck<PiopField>)
        {
            let projecting_element: PiopField = tr.get_field_challenge(fcfg);
            let fps = verifier_proj_scalars(fcfg, &projecting_element);
            let cpr_pre = CombinedPolyResolver::<PiopField>::build_verifier_pre_sumcheck::<Sha256Uair>(
                tr, &batched_proof.md_proof.claimed_sums[0], num_constraints,
                &projecting_element, &fps, ic_subclaim, fcfg,
            ).expect("Main field sumcheck pre-sumcheck");
            (projecting_element, fps, cpr_pre)
        };

        // Helper: replay lookup pre-sumcheck for all groups
        // (absorbs multiplicities / inverses, draws β / γ / r).
        let replay_lookup_pre = |tr: &mut zinc_transcript::KeccakTranscript,
                                 fcfg: &<PiopField as PrimeField>::Config,
                                 projecting_element: &PiopField|
            -> (Vec<zinc_piop::lookup::LookupVerifierPreSumcheck<PiopField>>, usize)
        {
            let mut lookup_pres = Vec::new();
            for (group_proof, meta) in batched_proof.lookup_group_proofs.iter()
                .zip(batched_proof.lookup_group_meta.iter())
            {
                let (subtable, shifts) = generate_table_and_shifts(
                    &meta.table_type, projecting_element, fcfg,
                );
                let pre = BatchedDecompLogupProtocol::<PiopField>::build_verifier_pre_sumcheck(
                    tr, group_proof, &subtable, &shifts,
                    meta.num_columns, meta.witness_len, fcfg,
                ).expect("Lookup pre-sumcheck");
                lookup_pres.push(pre);
            }
            let shared_nv = lookup_pres.iter()
                .map(|p| p.num_vars)
                .max()
                .map_or(num_vars, |m| m.max(num_vars));
            (lookup_pres, shared_nv)
        };

        // ── V1. Verifier / Field Setup ──────────────────────────────
        group.bench_function("V/FieldSetup", |b| {
            b.iter(|| {
                let mut transcript = zinc_transcript::KeccakTranscript::new();
                let field_cfg = transcript.get_random_field_cfg::<
                    PiopField, <PiopField as Field>::Inner, MillerRabin
                >();
                black_box(&field_cfg);
            });
        });

        // ── V2. Verifier / Ideal Check ──────────────────────────────
        group.bench_function("V/Ideal Check", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let mut tr = zinc_transcript::KeccakTranscript::new();
                    let fcfg = tr.get_random_field_cfg::<
                        PiopField, <PiopField as Field>::Inner, MillerRabin
                    >();
                    let t = Instant::now();
                    let _ = replay_through_ic(&mut tr, &fcfg);
                    total += t.elapsed();
                }
                total
            });
        });

        // ── V3. Verifier / Main field sumcheck Pre-Sumcheck ─────────────────────────
        // Uses batched_proof.md_proof.claimed_sums[0] as the main field sumcheck claimed
        // sum (prove_classic_logup_folded uses multi-degree sumcheck).
        group.bench_function("V/Main field sumcheck Pre", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let mut tr = zinc_transcript::KeccakTranscript::new();
                    let fcfg = tr.get_random_field_cfg::<
                        PiopField, <PiopField as Field>::Inner, MillerRabin
                    >();
                    let ic_subclaim = replay_through_ic(&mut tr, &fcfg);
                    let t = Instant::now();
                    let _ = replay_cpr_pre(&mut tr, &fcfg, &ic_subclaim);
                    total += t.elapsed();
                }
                total
            });
        });

        // ── V4. Verifier / Multi-Degree Sumcheck ────────────────────
        // The folded pipeline batches main field sumcheck + lookup into one multi-degree
        // sumcheck with shared challenges.
        group.bench_function("V/MDSumcheck", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let mut tr = zinc_transcript::KeccakTranscript::new();
                    let fcfg = tr.get_random_field_cfg::<
                        PiopField, <PiopField as Field>::Inner, MillerRabin
                    >();
                    let ic_subclaim = replay_through_ic(&mut tr, &fcfg);
                    let (projecting_element, _, _) = replay_cpr_pre(&mut tr, &fcfg, &ic_subclaim);
                    let (_, shared_nv) = replay_lookup_pre(&mut tr, &fcfg, &projecting_element);
                    let t = Instant::now();
                    let _ = MultiDegreeSumcheck::<PiopField>::verify_as_subprotocol(
                        &mut tr, shared_nv, &batched_proof.md_proof, &fcfg,
                    ).expect("Multi-degree sumcheck verify");
                    total += t.elapsed();
                }
                total
            });
        });

        // ── V5. Verifier / Main field sumcheck Finalize ─────────────────────────────
        group.bench_function("V/Main field sumcheck Finalize", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let mut tr = zinc_transcript::KeccakTranscript::new();
                    let fcfg = tr.get_random_field_cfg::<
                        PiopField, <PiopField as Field>::Inner, MillerRabin
                    >();
                    let ic_subclaim = replay_through_ic(&mut tr, &fcfg);
                    let (projecting_element, fps, cpr_pre) = replay_cpr_pre(&mut tr, &fcfg, &ic_subclaim);
                    let (_, shared_nv) = replay_lookup_pre(&mut tr, &fcfg, &projecting_element);
                    let md_subclaims = MultiDegreeSumcheck::<PiopField>::verify_as_subprotocol(
                        &mut tr, shared_nv, &batched_proof.md_proof, &fcfg,
                    ).unwrap();

                    let t = Instant::now();
                    let sig_v = Sha256Uair::signature();
                    let full_up_evals = if sig_v.public_columns.is_empty() {
                        batched_proof.cpr_up_evals.clone()
                    } else {
                        let bin_proj = BinaryPoly::<32>::prepare_projection(&projecting_element);
                        let public_evals: Vec<PiopField> = sha_public_cols.iter().map(|col| {
                            let mut mle: DenseMultilinearExtension<<PiopField as Field>::Inner> =
                                col.iter().map(|bp| bin_proj(bp).inner().clone()).collect();
                            let target_nv = md_subclaims.point.len();
                            if target_nv > mle.num_vars {
                                mle.evaluations.resize(1 << target_nv, Default::default());
                                mle.num_vars = target_nv;
                            }
                            mle.evaluate_with_config(&md_subclaims.point, &fcfg).unwrap()
                        }).collect();
                        reconstruct_up_evals(
                            &batched_proof.cpr_up_evals, &public_evals,
                            &sig_v.public_columns, sig_v.total_cols(),
                        )
                    };
                    let _ = CombinedPolyResolver::<PiopField>::finalize_verifier::<Sha256Uair>(
                        &mut tr,
                        md_subclaims.point.clone(),
                        md_subclaims.expected_evaluations[0].clone(),
                        &cpr_pre, full_up_evals,
                        batched_proof.cpr_down_evals.clone(),
                        num_vars, &fps, &fcfg,
                    ).expect("Main field sumcheck finalize");
                    total += t.elapsed();
                }
                total
            });
        });

        // ── V5b. Verifier / Shift Sumcheck Verify ───────────────────
        //
        // Verifies the batched shift-evaluation claims produced by the
        // shift sumcheck prover.  For UAIRs with public shift columns,
        // uses the split API (verify_pre + finalize); otherwise calls
        // the monolithic shift_sumcheck_verify.
        group.bench_function("V/ShiftSumcheckVerify", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let mut tr = zinc_transcript::KeccakTranscript::new();
                    let fcfg = tr.get_random_field_cfg::<
                        PiopField, <PiopField as Field>::Inner, MillerRabin
                    >();
                    let ic_subclaim = replay_through_ic(&mut tr, &fcfg);
                    let (projecting_element, fps, cpr_pre) = replay_cpr_pre(&mut tr, &fcfg, &ic_subclaim);
                    let (lookup_pres, shared_nv) = replay_lookup_pre(&mut tr, &fcfg, &projecting_element);
                    let md_subclaims = MultiDegreeSumcheck::<PiopField>::verify_as_subprotocol(
                        &mut tr, shared_nv, &batched_proof.md_proof, &fcfg,
                    ).unwrap();
                    // Replay CPR finalize.
                    let sig_v = Sha256Uair::signature();
                    let full_up_evals = if sig_v.public_columns.is_empty() {
                        batched_proof.cpr_up_evals.clone()
                    } else {
                        let bin_proj = BinaryPoly::<32>::prepare_projection(&projecting_element);
                        let public_evals: Vec<PiopField> = sha_public_cols.iter().map(|col| {
                            let mut mle: DenseMultilinearExtension<<PiopField as Field>::Inner> =
                                col.iter().map(|bp| bin_proj(bp).inner().clone()).collect();
                            let target_nv = md_subclaims.point.len();
                            if target_nv > mle.num_vars {
                                mle.evaluations.resize(1 << target_nv, Default::default());
                                mle.num_vars = target_nv;
                            }
                            mle.evaluate_with_config(&md_subclaims.point, &fcfg).unwrap()
                        }).collect();
                        reconstruct_up_evals(
                            &batched_proof.cpr_up_evals, &public_evals,
                            &sig_v.public_columns, sig_v.total_cols(),
                        )
                    };
                    let _ = CombinedPolyResolver::<PiopField>::finalize_verifier::<Sha256Uair>(
                        &mut tr,
                        md_subclaims.point.clone(),
                        md_subclaims.expected_evaluations[0].clone(),
                        &cpr_pre, full_up_evals.clone(),
                        batched_proof.cpr_down_evals.clone(),
                        num_vars, &fps, &fcfg,
                    ).unwrap();
                    // Replay lookup finalize.
                    for (g, (lk_pre, (group_proof, meta))) in lookup_pres.iter()
                        .zip(batched_proof.lookup_group_proofs.iter()
                            .zip(batched_proof.lookup_group_meta.iter()))
                        .enumerate()
                    {
                        let one = PiopField::one_with_cfg(&fcfg);
                        let w_nv = zinc_utils::log2(meta.witness_len.next_power_of_two()) as usize;
                        let mut eq_sum_w = one.clone();
                        for i in w_nv..md_subclaims.point.len() {
                            eq_sum_w *= one.clone() - &md_subclaims.point[i];
                        }
                        let parent_evals: Vec<PiopField> = meta.witness_sources.iter()
                            .map(|ws| match ws {
                                LookupWitnessSource::Column { column_index } =>
                                    full_up_evals[*column_index].clone(),
                                LookupWitnessSource::Affine { terms, constant_offset_bits } =>
                                    eval_affine_parent::<32>(
                                        terms, *constant_offset_bits,
                                        &full_up_evals,
                                        &projecting_element, &eq_sum_w,
                                        &fcfg,
                                    ),
                            })
                            .collect();
                        BatchedDecompLogupProtocol::<PiopField>::finalize_verifier(
                            lk_pre, group_proof,
                            &md_subclaims.point,
                            &md_subclaims.expected_evaluations[g + 1],
                            &parent_evals,
                            &fcfg,
                        ).unwrap();
                    }

                    // ── Timed section: shift sumcheck verify ──
                    let t = Instant::now();

                    if let Some(ref ss_proof_data) = folded_proof.shift_sumcheck {
                        let ss_down_evals: Vec<PiopField> = folded_proof.cpr_down_evals.iter()
                            .map(|b| field_from_bytes(b, &fcfg)).collect();
                        let claims: Vec<ShiftClaim<PiopField>> = sig_v.shifts
                            .iter()
                            .enumerate()
                            .map(|(i, spec)| ShiftClaim {
                                source_col: i,
                                shift_amount: spec.shift_amount,
                                eval_point: md_subclaims.point.clone(),
                                claimed_eval: ss_down_evals[i].clone(),
                            })
                            .collect();
                        let rounds: Vec<ShiftRoundPoly<PiopField>> = ss_proof_data.rounds.iter().map(|bytes| {
                            ShiftRoundPoly {
                                evals: [
                                    field_from_bytes(&bytes[0..field_elem_size], &fcfg),
                                    field_from_bytes(&bytes[field_elem_size..2 * field_elem_size], &fcfg),
                                    field_from_bytes(&bytes[2 * field_elem_size..3 * field_elem_size], &fcfg),
                                ],
                            }
                        }).collect();
                        let ss_proof = ShiftSumcheckProof { rounds };
                        let private_v_finals: Vec<PiopField> = ss_proof_data.v_finals.iter()
                            .map(|b| field_from_bytes(b, &fcfg)).collect();

                        let has_public_shifts = sig_v.shifts.iter()
                            .any(|spec| sig_v.is_public_column(spec.source_col));
                        if has_public_shifts {
                            let ss_pre = shift_sumcheck_verify_pre(
                                &mut tr, &ss_proof, &claims, num_vars, &fcfg,
                            ).expect("shift sumcheck pre-verify");
                            let binary_poly_projection =
                                BinaryPoly::<32>::prepare_projection(&projecting_element);
                            let challenge_point_le: Vec<PiopField> =
                                ss_pre.challenge_point.iter().rev().cloned().collect();
                            let public_shift_specs: Vec<&zinc_uair::ShiftSpec> = sig_v.shifts.iter()
                                .filter(|spec| sig_v.is_public_column(spec.source_col))
                                .collect();
                            let public_v_finals: Vec<PiopField> = public_shift_specs.iter()
                                .map(|spec| {
                                    let pcd_idx = sig_v.public_columns.iter()
                                        .position(|&c| c == spec.source_col)
                                        .expect("public shift source_col not found");
                                    let col = &sha_public_cols[pcd_idx];
                                    let mle: DenseMultilinearExtension<<PiopField as Field>::Inner> =
                                        col.iter()
                                            .map(|bp| binary_poly_projection(bp).inner().clone())
                                            .collect();
                                    mle.evaluate_with_config(&challenge_point_le, &fcfg)
                                        .expect("public shift MLE eval")
                                })
                                .collect();
                            let full_v_finals = reconstruct_shift_v_finals(
                                &private_v_finals,
                                &public_v_finals,
                                sig_v.shifts.len(),
                                |i| sig_v.is_public_shift(i),
                            );
                            shift_sumcheck_verify_finalize(
                                &mut tr, &ss_pre, &claims, &full_v_finals, &fcfg,
                            ).expect("shift sumcheck finalize");
                        } else {
                            shift_sumcheck_verify(
                                &mut tr, &ss_proof, &claims, &private_v_finals, num_vars, &fcfg,
                            ).expect("shift sumcheck verify");
                        }
                    }

                    total += t.elapsed();
                }
                total
            });
        });

        // ── V6. Verifier / Folding Verify ───────────────────────────
        //
        // Checks c₁[j] + α^{16} · c₂[j] == original_eval[j] for each
        // committed column, then squeezes (β, γ) to build the extended
        // PCS evaluation point.
        group.bench_function("V/FoldingVerify", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let mut tr = zinc_transcript::KeccakTranscript::new();
                    let fcfg = tr.get_random_field_cfg::<
                        PiopField, <PiopField as Field>::Inner, MillerRabin
                    >();
                    let ic_subclaim = replay_through_ic(&mut tr, &fcfg);
                    let (projecting_element, fps, cpr_pre) = replay_cpr_pre(&mut tr, &fcfg, &ic_subclaim);
                    let (lookup_pres, shared_nv) = replay_lookup_pre(&mut tr, &fcfg, &projecting_element);
                    let md_subclaims = MultiDegreeSumcheck::<PiopField>::verify_as_subprotocol(
                        &mut tr, shared_nv, &batched_proof.md_proof, &fcfg,
                    ).unwrap();

                    let sig_v = Sha256Uair::signature();
                    let full_up_evals = if sig_v.public_columns.is_empty() {
                        batched_proof.cpr_up_evals.clone()
                    } else {
                        let bin_proj = BinaryPoly::<32>::prepare_projection(&projecting_element);
                        let public_evals: Vec<PiopField> = sha_public_cols.iter().map(|col| {
                            let mut mle: DenseMultilinearExtension<<PiopField as Field>::Inner> =
                                col.iter().map(|bp| bin_proj(bp).inner().clone()).collect();
                            let target_nv = md_subclaims.point.len();
                            if target_nv > mle.num_vars {
                                mle.evaluations.resize(1 << target_nv, Default::default());
                                mle.num_vars = target_nv;
                            }
                            mle.evaluate_with_config(&md_subclaims.point, &fcfg).unwrap()
                        }).collect();
                        reconstruct_up_evals(
                            &batched_proof.cpr_up_evals, &public_evals,
                            &sig_v.public_columns, sig_v.total_cols(),
                        )
                    };
                    let full_up_evals_saved = full_up_evals.clone();
                    let _ = CombinedPolyResolver::<PiopField>::finalize_verifier::<Sha256Uair>(
                        &mut tr,
                        md_subclaims.point.clone(),
                        md_subclaims.expected_evaluations[0].clone(),
                        &cpr_pre, full_up_evals,
                        batched_proof.cpr_down_evals.clone(),
                        num_vars, &fps, &fcfg,
                    ).unwrap();

                    // Replay lookup finalize (transcript state needed for folding).
                    for (g, (lk_pre, (group_proof, meta))) in lookup_pres.iter()
                        .zip(batched_proof.lookup_group_proofs.iter()
                            .zip(batched_proof.lookup_group_meta.iter()))
                        .enumerate()
                    {
                        let one = PiopField::one_with_cfg(&fcfg);
                        let w_nv = zinc_utils::log2(meta.witness_len.next_power_of_two()) as usize;
                        let mut eq_sum_w = one.clone();
                        for i in w_nv..md_subclaims.point.len() {
                            eq_sum_w *= one.clone() - &md_subclaims.point[i];
                        }
                        let parent_evals: Vec<PiopField> = meta.witness_sources.iter()
                            .map(|ws| match ws {
                                LookupWitnessSource::Column { column_index } =>
                                    full_up_evals_saved[*column_index].clone(),
                                LookupWitnessSource::Affine { terms, constant_offset_bits } =>
                                    eval_affine_parent::<32>(
                                        terms, *constant_offset_bits,
                                        &full_up_evals_saved,
                                        &projecting_element, &eq_sum_w,
                                        &fcfg,
                                    ),
                            })
                            .collect();
                        BatchedDecompLogupProtocol::<PiopField>::finalize_verifier(
                            lk_pre, group_proof,
                            &md_subclaims.point,
                            &md_subclaims.expected_evaluations[g + 1],
                            &parent_evals,
                            &fcfg,
                        ).unwrap_or_else(|e| panic!("Lookup finalize failed (group {g}): {e:?}"));
                    }

                    // Replay shift sumcheck verify (transcript state needed for folding).
                    if let Some(ref ss_proof_data) = folded_proof.shift_sumcheck {
                        let sig_ss = Sha256Uair::signature();
                        let ss_down_evals: Vec<PiopField> = folded_proof.cpr_down_evals.iter()
                            .map(|b| field_from_bytes(b, &fcfg)).collect();
                        let claims: Vec<ShiftClaim<PiopField>> = sig_ss.shifts
                            .iter()
                            .enumerate()
                            .map(|(i, spec)| ShiftClaim {
                                source_col: i,
                                shift_amount: spec.shift_amount,
                                eval_point: md_subclaims.point.clone(),
                                claimed_eval: ss_down_evals[i].clone(),
                            })
                            .collect();
                        let rounds: Vec<ShiftRoundPoly<PiopField>> = ss_proof_data.rounds.iter().map(|bytes| {
                            ShiftRoundPoly {
                                evals: [
                                    field_from_bytes(&bytes[0..field_elem_size], &fcfg),
                                    field_from_bytes(&bytes[field_elem_size..2 * field_elem_size], &fcfg),
                                    field_from_bytes(&bytes[2 * field_elem_size..3 * field_elem_size], &fcfg),
                                ],
                            }
                        }).collect();
                        let ss_proof = ShiftSumcheckProof { rounds };
                        let private_v_finals: Vec<PiopField> = ss_proof_data.v_finals.iter()
                            .map(|b| field_from_bytes(b, &fcfg)).collect();
                        let has_public_shifts = sig_ss.shifts.iter()
                            .any(|spec| sig_ss.is_public_column(spec.source_col));
                        if has_public_shifts {
                            let ss_pre = shift_sumcheck_verify_pre(
                                &mut tr, &ss_proof, &claims, num_vars, &fcfg,
                            ).unwrap();
                            let binary_poly_projection =
                                BinaryPoly::<32>::prepare_projection(&projecting_element);
                            let challenge_point_le: Vec<PiopField> =
                                ss_pre.challenge_point.iter().rev().cloned().collect();
                            let public_shift_specs: Vec<&zinc_uair::ShiftSpec> = sig_ss.shifts.iter()
                                .filter(|spec| sig_ss.is_public_column(spec.source_col))
                                .collect();
                            let public_v_finals: Vec<PiopField> = public_shift_specs.iter()
                                .map(|spec| {
                                    let pcd_idx = sig_ss.public_columns.iter()
                                        .position(|&c| c == spec.source_col)
                                        .expect("public shift source_col not found");
                                    let col = &sha_public_cols[pcd_idx];
                                    let mle: DenseMultilinearExtension<<PiopField as Field>::Inner> =
                                        col.iter()
                                            .map(|bp| binary_poly_projection(bp).inner().clone())
                                            .collect();
                                    mle.evaluate_with_config(&challenge_point_le, &fcfg)
                                        .expect("public shift MLE eval")
                                })
                                .collect();
                            let full_v_finals = reconstruct_shift_v_finals(
                                &private_v_finals,
                                &public_v_finals,
                                sig_ss.shifts.len(),
                                |i| sig_ss.is_public_shift(i),
                            );
                            shift_sumcheck_verify_finalize(
                                &mut tr, &ss_pre, &claims, &full_v_finals, &fcfg,
                            ).unwrap();
                        } else {
                            shift_sumcheck_verify(
                                &mut tr, &ss_proof, &claims, &private_v_finals, num_vars, &fcfg,
                            ).unwrap();
                        }
                    }

                    // ── Timed section: folding verify ──
                    let t = Instant::now();

                    // Deserialize c1s and c2s.
                    let c1s: Vec<PiopField> = folded_proof.folding_c1s_bytes.iter()
                        .map(|b| field_from_bytes(b, &fcfg)).collect();
                    let c2s: Vec<PiopField> = folded_proof.folding_c2s_bytes.iter()
                        .map(|b| field_from_bytes(b, &fcfg)).collect();

                    // Filter original evals to PCS-committed columns only.
                    let pcs_excluded_v = sig_v.pcs_excluded_columns();
                    let original_evals: Vec<PiopField> = full_up_evals_saved.iter().enumerate()
                        .filter(|(i, _)| !pcs_excluded_v.contains(i))
                        .map(|(_, e)| e.clone()).collect();

                    let alpha_power = compute_alpha_power(&projecting_element, 16);

                    // The main field sumcheck point is the shared multi-degree sumcheck point
                    // truncated to num_vars.
                    let cpr_point = &md_subclaims.point[..num_vars];

                    let _ = fold_claims_verify(
                        &mut tr,
                        &c1s,
                        &c2s,
                        &original_evals,
                        &alpha_power,
                        cpr_point,
                        &fcfg,
                    ).expect("folding verify");

                    total += t.elapsed();
                }
                total
            });
        });

        // ── V7. Verifier / PCS Verify (folded) ─────────────────────
        // Pre-compute the correct PCS point by replaying through the
        // folding verification once (the PCS point = r || γ, derived
        // from fold_claims_verify).
        let pcs_verify_point: Vec<PiopField> = {
            let mut tr = zinc_transcript::KeccakTranscript::new();
            let fcfg = tr.get_random_field_cfg::<
                PiopField, <PiopField as Field>::Inner, MillerRabin
            >();
            let ic_subclaim = replay_through_ic(&mut tr, &fcfg);
            let (projecting_element, fps, cpr_pre) = replay_cpr_pre(&mut tr, &fcfg, &ic_subclaim);
            let (lookup_pres, shared_nv) = replay_lookup_pre(&mut tr, &fcfg, &projecting_element);
            let md_subclaims = MultiDegreeSumcheck::<PiopField>::verify_as_subprotocol(
                &mut tr, shared_nv, &batched_proof.md_proof, &fcfg,
            ).unwrap();
            let sig_v = Sha256Uair::signature();
            let full_up_evals = if sig_v.public_columns.is_empty() {
                batched_proof.cpr_up_evals.clone()
            } else {
                let bin_proj = BinaryPoly::<32>::prepare_projection(&projecting_element);
                let public_evals: Vec<PiopField> = sha_public_cols.iter().map(|col| {
                    let mut mle: DenseMultilinearExtension<<PiopField as Field>::Inner> =
                        col.iter().map(|bp| bin_proj(bp).inner().clone()).collect();
                    let target_nv = md_subclaims.point.len();
                    if target_nv > mle.num_vars {
                        mle.evaluations.resize(1 << target_nv, Default::default());
                        mle.num_vars = target_nv;
                    }
                    mle.evaluate_with_config(&md_subclaims.point, &fcfg).unwrap()
                }).collect();
                reconstruct_up_evals(
                    &batched_proof.cpr_up_evals, &public_evals,
                    &sig_v.public_columns, sig_v.total_cols(),
                )
            };
            let _ = CombinedPolyResolver::<PiopField>::finalize_verifier::<Sha256Uair>(
                &mut tr,
                md_subclaims.point.clone(),
                md_subclaims.expected_evaluations[0].clone(),
                &cpr_pre, full_up_evals.clone(),
                batched_proof.cpr_down_evals.clone(),
                num_vars, &fps, &fcfg,
            ).unwrap();
            // Replay lookup finalize.
            for (g, (lk_pre, (group_proof, meta))) in lookup_pres.iter()
                .zip(batched_proof.lookup_group_proofs.iter()
                    .zip(batched_proof.lookup_group_meta.iter()))
                .enumerate()
            {
                let one = PiopField::one_with_cfg(&fcfg);
                let w_nv = zinc_utils::log2(meta.witness_len.next_power_of_two()) as usize;
                let mut eq_sum_w = one.clone();
                for i in w_nv..md_subclaims.point.len() {
                    eq_sum_w *= one.clone() - &md_subclaims.point[i];
                }
                let parent_evals: Vec<PiopField> = meta.witness_sources.iter()
                    .map(|ws| match ws {
                        LookupWitnessSource::Column { column_index } =>
                            full_up_evals[*column_index].clone(),
                        LookupWitnessSource::Affine { terms, constant_offset_bits } =>
                            eval_affine_parent::<32>(
                                terms, *constant_offset_bits,
                                &full_up_evals,
                                &projecting_element, &eq_sum_w,
                                &fcfg,
                            ),
                    })
                    .collect();
                BatchedDecompLogupProtocol::<PiopField>::finalize_verifier(
                    lk_pre, group_proof,
                    &md_subclaims.point,
                    &md_subclaims.expected_evaluations[g + 1],
                    &parent_evals,
                    &fcfg,
                ).unwrap();
            }
            // Replay shift sumcheck verify.
            if let Some(ref ss_proof_data) = folded_proof.shift_sumcheck {
                let sig_ss = Sha256Uair::signature();
                let ss_down_evals: Vec<PiopField> = folded_proof.cpr_down_evals.iter()
                    .map(|b| field_from_bytes(b, &fcfg)).collect();
                let claims: Vec<ShiftClaim<PiopField>> = sig_ss.shifts
                    .iter()
                    .enumerate()
                    .map(|(i, spec)| ShiftClaim {
                        source_col: i,
                        shift_amount: spec.shift_amount,
                        eval_point: md_subclaims.point.clone(),
                        claimed_eval: ss_down_evals[i].clone(),
                    })
                    .collect();
                let rounds: Vec<ShiftRoundPoly<PiopField>> = ss_proof_data.rounds.iter().map(|bytes| {
                    ShiftRoundPoly {
                        evals: [
                            field_from_bytes(&bytes[0..field_elem_size], &fcfg),
                            field_from_bytes(&bytes[field_elem_size..2 * field_elem_size], &fcfg),
                            field_from_bytes(&bytes[2 * field_elem_size..3 * field_elem_size], &fcfg),
                        ],
                    }
                }).collect();
                let ss_proof = ShiftSumcheckProof { rounds };
                let private_v_finals: Vec<PiopField> = ss_proof_data.v_finals.iter()
                    .map(|b| field_from_bytes(b, &fcfg)).collect();
                let has_public_shifts = sig_ss.shifts.iter()
                    .any(|spec| sig_ss.is_public_column(spec.source_col));
                if has_public_shifts {
                    let ss_pre = shift_sumcheck_verify_pre(
                        &mut tr, &ss_proof, &claims, num_vars, &fcfg,
                    ).unwrap();
                    let binary_poly_projection =
                        BinaryPoly::<32>::prepare_projection(&projecting_element);
                    let challenge_point_le: Vec<PiopField> =
                        ss_pre.challenge_point.iter().rev().cloned().collect();
                    let public_shift_specs: Vec<&zinc_uair::ShiftSpec> = sig_ss.shifts.iter()
                        .filter(|spec| sig_ss.is_public_column(spec.source_col))
                        .collect();
                    let public_v_finals: Vec<PiopField> = public_shift_specs.iter()
                        .map(|spec| {
                            let pcd_idx = sig_ss.public_columns.iter()
                                .position(|&c| c == spec.source_col)
                                .expect("public shift source_col not found");
                            let col = &sha_public_cols[pcd_idx];
                            let mle: DenseMultilinearExtension<<PiopField as Field>::Inner> =
                                col.iter()
                                    .map(|bp| binary_poly_projection(bp).inner().clone())
                                    .collect();
                            mle.evaluate_with_config(&challenge_point_le, &fcfg)
                                .expect("public shift MLE eval")
                        })
                        .collect();
                    let full_v_finals = reconstruct_shift_v_finals(
                        &private_v_finals,
                        &public_v_finals,
                        sig_ss.shifts.len(),
                        |i| sig_ss.is_public_shift(i),
                    );
                    shift_sumcheck_verify_finalize(
                        &mut tr, &ss_pre, &claims, &full_v_finals, &fcfg,
                    ).unwrap();
                } else {
                    shift_sumcheck_verify(
                        &mut tr, &ss_proof, &claims, &private_v_finals, num_vars, &fcfg,
                    ).unwrap();
                }
            }
            // Run folding verify to get the PCS point (r || γ).
            let pcs_excluded_v = sig_v.pcs_excluded_columns();
            let original_evals: Vec<PiopField> = full_up_evals.iter().enumerate()
                .filter(|(i, _)| !pcs_excluded_v.contains(i))
                .map(|(_, e)| e.clone()).collect();
            let alpha_power = compute_alpha_power(&projecting_element, 16);
            let cpr_point = &md_subclaims.point[..num_vars];
            let c1s: Vec<PiopField> = folded_proof.folding_c1s_bytes.iter()
                .map(|b| field_from_bytes(b, &fcfg)).collect();
            let c2s: Vec<PiopField> = folded_proof.folding_c2s_bytes.iter()
                .map(|b| field_from_bytes(b, &fcfg)).collect();
            let folding_out = fold_claims_verify(
                &mut tr, &c1s, &c2s, &original_evals,
                &alpha_power, cpr_point, &fcfg,
            ).expect("folding verify for PCS point derivation");
            folding_out.new_point
        };

        group.bench_function("V/PCSVerify (folded)", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    // ── Timed section ──
                    let t = Instant::now();
                    let mut pcs_transcript = zip_plus::pcs_transcript::PcsTranscript {
                        fs_transcript: zinc_transcript::KeccakTranscript::default(),
                        stream: std::io::Cursor::new(folded_proof.pcs_proof_bytes.clone()),
                    };
                    let pcs_field_cfg = pcs_transcript.fs_transcript
                        .get_random_field_cfg::<PiopField, <FoldedZt as ZipTypes>::Fmod, <FoldedZt as ZipTypes>::PrimeTest>();
                    let eval_f: PiopField = PiopField::new_unchecked_with_cfg(
                        <Uint<{INT_LIMBS * 3}> as ConstTranscribable>::read_transcription_bytes(&folded_proof.pcs_evals_bytes[0]),
                        &pcs_field_cfg,
                    );
                    // The folded PCS point is (r || γ) derived from the folding
                    // verification. Re-create it in the PCS field config.
                    let point_f: Vec<PiopField> = pcs_verify_point.iter()
                        .map(|p| PiopField::new_unchecked_with_cfg(p.inner().clone(), &pcs_field_cfg))
                        .collect();
                    ZipPlus::<FoldedZt, FoldedLc>::verify_with_field_cfg::<PiopField, UNCHECKED>(
                        &folded_params, &folded_proof.commitment, &point_f, &eval_f,
                        pcs_transcript, &pcs_field_cfg,
                    ).expect("PCS verify (folded)");
                    total += t.elapsed();
                }
                total
            });
        });
    }

    // ── Timing breakdown summary ────────────────────────────────────
    eprintln!("\n=== 8xSHA256 Folded Pipeline Timing (Classic Lookup) ===");
    eprintln!("  IC={:?}, CPR={:?}, Lookup={:?}, PCS(commit={:?}, prove={:?}), total={:?}",
        folded_proof.timing.ideal_check,
        folded_proof.timing.combined_poly_resolver,
        folded_proof.timing.lookup,
        folded_proof.timing.pcs_commit,
        folded_proof.timing.pcs_prove,
        folded_proof.timing.total,
    );
    eprintln!("  GKR Lookup pipeline timing:");
    eprintln!("  IC={:?}, CPR={:?}, Lookup={:?}, PCS(commit={:?}, prove={:?}), total={:?}",
        gkr_folded_proof.timing.ideal_check,
        gkr_folded_proof.timing.combined_poly_resolver,
        gkr_folded_proof.timing.lookup,
        gkr_folded_proof.timing.pcs_commit,
        gkr_folded_proof.timing.pcs_prove,
        gkr_folded_proof.timing.total,
    );
    eprintln!("  Original pipeline timing:");
    eprintln!("  IC={:?}, CPR={:?}, Lookup={:?}, PCS(commit={:?}, prove={:?}), total={:?}",
        orig_proof.timing.ideal_check,
        orig_proof.timing.combined_poly_resolver,
        orig_proof.timing.lookup,
        orig_proof.timing.pcs_commit,
        orig_proof.timing.pcs_prove,
        orig_proof.timing.total,
    );

    // ── Proof size breakdown (folded, classic lookup) ──────────────
    // Helper: compute byte size of a GKR lookup proof.
    fn gkr_lookup_proof_bytes(proof: &zinc_piop::lookup::GkrPipelineLookupProof<zinc_snark::pipeline::PiopField>, fe_bytes: usize) -> usize {
        let mut total = 0usize;
        for gp in &proof.group_proofs {
            // aggregated_multiplicities: L × T field elements
            let mults: usize = gp.aggregated_multiplicities.iter().map(|v| v.len()).sum();
            total += mults * fe_bytes;
            // witness_gkr: BatchedGkrFractionProof
            total += gp.witness_gkr.roots_p.len() * fe_bytes; // roots_p
            total += gp.witness_gkr.roots_q.len() * fe_bytes; // roots_q
            for layer in &gp.witness_gkr.layer_proofs {
                // sumcheck_proof messages (if present)
                if let Some(ref sc) = layer.sumcheck_proof {
                    total += fe_bytes; // claimed_sum
                    for msg in &sc.messages {
                        total += msg.0.tail_evaluations.len() * fe_bytes;
                    }
                }
                // 4L evaluations per layer (p_lefts, p_rights, q_lefts, q_rights)
                total += layer.p_lefts.len() * fe_bytes;
                total += layer.p_rights.len() * fe_bytes;
                total += layer.q_lefts.len() * fe_bytes;
                total += layer.q_rights.len() * fe_bytes;
            }
            // table_gkr: GkrFractionProof
            total += 2 * fe_bytes; // root_p, root_q
            for layer in &gp.table_gkr.layer_proofs {
                if let Some(ref sc) = layer.sumcheck_proof {
                    total += fe_bytes; // claimed_sum
                    for msg in &sc.messages {
                        total += msg.0.tail_evaluations.len() * fe_bytes;
                    }
                }
                total += 4 * fe_bytes; // p_left, p_right, q_left, q_right
            }
        }
        total
    }

    {
        use zinc_snark::pipeline::{FIELD_LIMBS, LookupProofData};
        let fe_bytes = <Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;

        // PCS proof bytes.
        let pcs_bytes = folded_proof.pcs_proof_bytes.len();

        // IC proof values.
        let ic_bytes: usize = folded_proof.ic_proof_values.iter().map(|v| v.len()).sum();

        // Main field sumcheck messages + claimed sum.
        let cpr_msg_bytes: usize = folded_proof.cpr_sumcheck_messages.iter().map(|v| v.len()).sum();
        let cpr_sum_bytes = folded_proof.cpr_sumcheck_claimed_sum.len();
        let cpr_sc_total = cpr_msg_bytes + cpr_sum_bytes;

        // Main field sumcheck up/down evaluations.
        let cpr_up: usize = folded_proof.cpr_up_evals.iter().map(|v| v.len()).sum();
        let cpr_dn: usize = folded_proof.cpr_down_evals.iter().map(|v| v.len()).sum();
        let cpr_eval_total = cpr_up + cpr_dn;

        // Lookup data.
        let lookup_bytes: usize = match &folded_proof.lookup_proof {
            Some(LookupProofData::Classic(proof)) => {
                let mut total_lk = 0usize;
                for gp in &proof.group_proofs {
                    let mults: usize = gp.aggregated_multiplicities.iter().map(|v| v.len()).sum();
                    let inv_w: usize = gp.chunk_inverse_witnesses.iter()
                        .flat_map(|outer| outer.iter())
                        .map(|inner| inner.len())
                        .sum();
                    let inv_t = gp.inverse_table.len();
                    total_lk += (mults + inv_w + inv_t) * fe_bytes;
                }
                total_lk
            }
            Some(LookupProofData::BatchedClassic(bp)) => {
                let mut total_lk = 0usize;
                for gp in &bp.lookup_group_proofs {
                    let mults: usize = gp.aggregated_multiplicities.iter().map(|v| v.len()).sum();
                    let inv_w: usize = gp.chunk_inverse_witnesses.iter()
                        .flat_map(|outer| outer.iter())
                        .map(|inner| inner.len())
                        .sum();
                    let inv_t = gp.inverse_table.len();
                    total_lk += (mults + inv_w + inv_t) * fe_bytes;
                }
                total_lk
            }
            Some(LookupProofData::Gkr(gkr_proof)) => {
                gkr_lookup_proof_bytes(gkr_proof, fe_bytes)
            }
            Some(LookupProofData::HybridGkr(hp)) => {
                let mut t = 0usize;
                for group in &hp.group_proofs {
                    let m: usize = group.aggregated_multiplicities.iter().map(|v| v.len()).sum();
                    t += m * fe_bytes;
                    let wg = &group.witness_gkr;
                    t += (wg.roots_p.len() + wg.roots_q.len()) * fe_bytes;
                    for lp in &wg.layer_proofs {
                        if let Some(ref sc) = lp.sumcheck_proof {
                            t += sc.messages.iter().map(|m| m.0.tail_evaluations.len()).sum::<usize>() * fe_bytes;
                            t += fe_bytes;
                        }
                        t += (lp.p_lefts.len() + lp.p_rights.len() + lp.q_lefts.len() + lp.q_rights.len()) * fe_bytes;
                    }
                    let sent_p: usize = wg.sent_p.iter().map(|v| v.len()).sum();
                    let sent_q: usize = wg.sent_q.iter().map(|v| v.len()).sum();
                    t += (sent_p + sent_q) * fe_bytes;
                    let tg = &group.table_gkr;
                    t += 2 * fe_bytes;
                    for lp in &tg.layer_proofs {
                        if let Some(ref sc) = lp.sumcheck_proof {
                            t += sc.messages.iter().map(|m| m.0.tail_evaluations.len()).sum::<usize>() * fe_bytes;
                            t += fe_bytes;
                        }
                        t += 4 * fe_bytes;
                    }
                }
                t
            }
            None => 0,
        };

        // Shift sumcheck data.
        let shift_sc_bytes: usize = folded_proof.shift_sumcheck.as_ref().map_or(0, |sc| {
            let rounds: usize = sc.rounds.iter().map(|v| v.len()).sum();
            let finals: usize = sc.v_finals.iter().map(|v| v.len()).sum();
            rounds + finals
        });

        // Folding data (c₁s and c₂s).
        let folding_c1s_bytes: usize = folded_proof.folding_c1s_bytes.iter().map(|v| v.len()).sum();
        let folding_c2s_bytes: usize = folded_proof.folding_c2s_bytes.iter().map(|v| v.len()).sum();
        let folding_total = folding_c1s_bytes + folding_c2s_bytes;

        // Evaluation point + PCS evals.
        let eval_pt_bytes: usize = folded_proof.evaluation_point_bytes.iter().map(|v| v.len()).sum();
        let pcs_eval_bytes: usize = folded_proof.pcs_evals_bytes.iter().map(|v| v.len()).sum();

        let piop_total = ic_bytes + cpr_sc_total + cpr_eval_total
            + lookup_bytes + shift_sc_bytes + folding_total + eval_pt_bytes + pcs_eval_bytes;
        let total_raw = pcs_bytes + piop_total;

        // Build serialized byte buffer and compress with deflate.
        let mut all_bytes = Vec::with_capacity(total_raw);
        all_bytes.extend(&folded_proof.pcs_proof_bytes);
        for v in &folded_proof.ic_proof_values { all_bytes.extend(v); }
        for v in &folded_proof.cpr_sumcheck_messages { all_bytes.extend(v); }
        all_bytes.extend(&folded_proof.cpr_sumcheck_claimed_sum);
        for v in &folded_proof.cpr_up_evals { all_bytes.extend(v); }
        for v in &folded_proof.cpr_down_evals { all_bytes.extend(v); }
        if let Some(ref sc) = folded_proof.shift_sumcheck {
            for v in &sc.rounds { all_bytes.extend(v); }
            for v in &sc.v_finals { all_bytes.extend(v); }
        }
        for v in &folded_proof.folding_c1s_bytes { all_bytes.extend(v); }
        for v in &folded_proof.folding_c2s_bytes { all_bytes.extend(v); }
        for v in &folded_proof.evaluation_point_bytes { all_bytes.extend(v); }
        for v in &folded_proof.pcs_evals_bytes { all_bytes.extend(v); }

        let compressed = {
            use std::io::Write;
            let mut encoder = flate2::write::DeflateEncoder::new(
                Vec::new(), flate2::Compression::default(),
            );
            encoder.write_all(&all_bytes).unwrap();
            encoder.finish().unwrap()
        };

        // ── Original proof size (for comparison) ────────────────────
        let orig_pcs_bytes = orig_proof.pcs_proof_bytes.len();
        let orig_ic_bytes: usize = orig_proof.ic_proof_values.iter().map(|v| v.len()).sum();
        let orig_cpr_msg: usize = orig_proof.cpr_sumcheck_messages.iter().map(|v| v.len()).sum();
        let orig_cpr_sum = orig_proof.cpr_sumcheck_claimed_sum.len();
        let orig_cpr_up: usize = orig_proof.cpr_up_evals.iter().map(|v| v.len()).sum();
        let orig_cpr_dn: usize = orig_proof.cpr_down_evals.iter().map(|v| v.len()).sum();
        let orig_lookup_bytes: usize = match &orig_proof.lookup_proof {
            Some(LookupProofData::Classic(proof)) => {
                let mut t = 0usize;
                for gp in &proof.group_proofs {
                    let m: usize = gp.aggregated_multiplicities.iter().map(|v| v.len()).sum();
                    let w: usize = gp.chunk_inverse_witnesses.iter().flat_map(|o| o.iter()).map(|i| i.len()).sum();
                    let i = gp.inverse_table.len();
                    t += (m + w + i) * fe_bytes;
                }
                t
            }
            _ => 0,
        };
        let orig_shift_sc: usize = orig_proof.shift_sumcheck.as_ref().map_or(0, |sc| {
            let r: usize = sc.rounds.iter().map(|v| v.len()).sum();
            let f: usize = sc.v_finals.iter().map(|v| v.len()).sum();
            r + f
        });
        let orig_eval_pt: usize = orig_proof.evaluation_point_bytes.iter().map(|v| v.len()).sum();
        let orig_pcs_eval: usize = orig_proof.pcs_evals_bytes.iter().map(|v| v.len()).sum();
        let orig_piop_total = orig_ic_bytes + (orig_cpr_msg + orig_cpr_sum) + (orig_cpr_up + orig_cpr_dn)
            + orig_lookup_bytes + orig_shift_sc + orig_eval_pt + orig_pcs_eval;
        let orig_total_raw = orig_pcs_bytes + orig_piop_total;

        let mut orig_all_bytes = Vec::with_capacity(orig_total_raw);
        orig_all_bytes.extend(&orig_proof.pcs_proof_bytes);
        for v in &orig_proof.ic_proof_values { orig_all_bytes.extend(v); }
        for v in &orig_proof.cpr_sumcheck_messages { orig_all_bytes.extend(v); }
        orig_all_bytes.extend(&orig_proof.cpr_sumcheck_claimed_sum);
        for v in &orig_proof.cpr_up_evals { orig_all_bytes.extend(v); }
        for v in &orig_proof.cpr_down_evals { orig_all_bytes.extend(v); }
        if let Some(LookupProofData::Classic(proof)) = &orig_proof.lookup_proof {
            fn write_fe(buf: &mut Vec<u8>, f: &zinc_snark::pipeline::PiopField) {
                use zinc_snark::pipeline::FIELD_LIMBS;
                let sz = <Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;
                let start = buf.len();
                buf.resize(start + sz, 0);
                f.inner().write_transcription_bytes(&mut buf[start..]);
            }
            for gp in &proof.group_proofs {
                for v in &gp.aggregated_multiplicities {
                    for f in v { write_fe(&mut orig_all_bytes, f); }
                }
                for outer in &gp.chunk_inverse_witnesses {
                    for inner in outer {
                        for f in inner { write_fe(&mut orig_all_bytes, f); }
                    }
                }
                for f in &gp.inverse_table { write_fe(&mut orig_all_bytes, f); }
            }
        }
        if let Some(ref sc) = orig_proof.shift_sumcheck {
            for v in &sc.rounds { orig_all_bytes.extend(v); }
            for v in &sc.v_finals { orig_all_bytes.extend(v); }
        }
        for v in &orig_proof.evaluation_point_bytes { orig_all_bytes.extend(v); }
        for v in &orig_proof.pcs_evals_bytes { orig_all_bytes.extend(v); }

        let orig_compressed = {
            use std::io::Write;
            let mut encoder = flate2::write::DeflateEncoder::new(
                Vec::new(), flate2::Compression::default(),
            );
            encoder.write_all(&orig_all_bytes).unwrap();
            encoder.finish().unwrap()
        };

        eprintln!("\n=== 8xSHA256 Folded Proof Size ===");
        eprintln!("  PCS:            {:>6} B  ({:.1} KB)", pcs_bytes, pcs_bytes as f64 / 1024.0);
        eprintln!("  IC:             {:>6} B", ic_bytes);
        eprintln!("  CPR sumcheck:   {:>6} B  (msgs={}, sum={})", cpr_sc_total, cpr_msg_bytes, cpr_sum_bytes);
        eprintln!("  CPR evals:      {:>6} B  (up={}, down={})", cpr_eval_total, cpr_up, cpr_dn);
        eprintln!("  Lookup:         {:>6} B", lookup_bytes);
        eprintln!("  Shift SC:       {:>6} B", shift_sc_bytes);
        eprintln!("  Folding:        {:>6} B  (c1s={}, c2s={})", folding_total, folding_c1s_bytes, folding_c2s_bytes);
        eprintln!("  Eval point:     {:>6} B", eval_pt_bytes);
        eprintln!("  PCS evals:      {:>6} B", pcs_eval_bytes);
        eprintln!("  ─────────────────────────");
        eprintln!("  PIOP total:     {:>6} B  ({:.1} KB)", piop_total, piop_total as f64 / 1024.0);
        eprintln!("  Total raw:      {:>6} B  ({:.1} KB)", total_raw, total_raw as f64 / 1024.0);
        eprintln!("  Compressed:     {:>6} B  ({:.1} KB, {:.1}x ratio)",
            compressed.len(), compressed.len() as f64 / 1024.0,
            all_bytes.len() as f64 / compressed.len() as f64);

        eprintln!("\n=== Folded vs Original Comparison ===");
        eprintln!("  Original PCS:     {:>6} B  ({:.1} KB)", orig_pcs_bytes, orig_pcs_bytes as f64 / 1024.0);
        eprintln!("  Folded PCS:       {:>6} B  ({:.1} KB)", pcs_bytes, pcs_bytes as f64 / 1024.0);
        let pcs_diff = orig_pcs_bytes as i64 - pcs_bytes as i64;
        eprintln!("  PCS savings:      {:>6} B  ({:+.1} KB)", pcs_diff, pcs_diff as f64 / 1024.0);
        eprintln!("  ─────────────────────────");
        eprintln!("  Original total:   {:>6} B  ({:.1} KB)", orig_total_raw, orig_total_raw as f64 / 1024.0);
        eprintln!("  Folded total:     {:>6} B  ({:.1} KB)", total_raw, total_raw as f64 / 1024.0);
        let total_diff = orig_total_raw as i64 - total_raw as i64;
        eprintln!("  Total savings:    {:>6} B  ({:+.1} KB, {:.1}x)",
            total_diff, total_diff as f64 / 1024.0,
            orig_total_raw as f64 / total_raw as f64);
        eprintln!("  ─────────────────────────");
        eprintln!("  Original compr:   {:>6} B  ({:.1} KB)", orig_compressed.len(), orig_compressed.len() as f64 / 1024.0);
        eprintln!("  Folded compr:     {:>6} B  ({:.1} KB)", compressed.len(), compressed.len() as f64 / 1024.0);
        let compr_diff = orig_compressed.len() as i64 - compressed.len() as i64;
        eprintln!("  Compr savings:    {:>6} B  ({:+.1} KB, {:.1}x)",
            compr_diff, compr_diff as f64 / 1024.0,
            orig_compressed.len() as f64 / compressed.len() as f64);

        // ── GKR folded proof size breakdown ─────────────────────────
        let gkr_pcs_bytes = gkr_folded_proof.pcs_proof_bytes.len();
        let gkr_ic_bytes: usize = gkr_folded_proof.ic_proof_values.iter().map(|v| v.len()).sum();
        let gkr_cpr_msg: usize = gkr_folded_proof.cpr_sumcheck_messages.iter().map(|v| v.len()).sum();
        let gkr_cpr_sum = gkr_folded_proof.cpr_sumcheck_claimed_sum.len();
        let gkr_cpr_sc_total = gkr_cpr_msg + gkr_cpr_sum;
        let gkr_cpr_up: usize = gkr_folded_proof.cpr_up_evals.iter().map(|v| v.len()).sum();
        let gkr_cpr_dn: usize = gkr_folded_proof.cpr_down_evals.iter().map(|v| v.len()).sum();
        let gkr_cpr_eval_total = gkr_cpr_up + gkr_cpr_dn;
        let gkr_lookup_bytes: usize = match &gkr_folded_proof.lookup_proof {
            Some(LookupProofData::Gkr(gkr_proof)) =>
                gkr_lookup_proof_bytes(gkr_proof, fe_bytes),
            Some(LookupProofData::BatchedClassic(bp)) => {
                let mut t = 0usize;
                for gp in &bp.lookup_group_proofs {
                    let m: usize = gp.aggregated_multiplicities.iter().map(|v| v.len()).sum();
                    let w: usize = gp.chunk_inverse_witnesses.iter().flat_map(|o| o.iter()).map(|i| i.len()).sum();
                    t += (m + w + gp.inverse_table.len()) * fe_bytes;
                }
                t
            }
            Some(LookupProofData::Classic(proof)) => {
                let mut t = 0usize;
                for gp in &proof.group_proofs {
                    let m: usize = gp.aggregated_multiplicities.iter().map(|v| v.len()).sum();
                    let w: usize = gp.chunk_inverse_witnesses.iter().flat_map(|o| o.iter()).map(|i| i.len()).sum();
                    t += (m + w + gp.inverse_table.len()) * fe_bytes;
                }
                t
            }
            Some(LookupProofData::HybridGkr(hp)) => {
                let mut t = 0usize;
                for group in &hp.group_proofs {
                    let m: usize = group.aggregated_multiplicities.iter().map(|v| v.len()).sum();
                    t += m * fe_bytes;
                    let wg = &group.witness_gkr;
                    t += (wg.roots_p.len() + wg.roots_q.len()) * fe_bytes;
                    for lp in &wg.layer_proofs {
                        if let Some(ref sc) = lp.sumcheck_proof {
                            t += sc.messages.iter().map(|m| m.0.tail_evaluations.len()).sum::<usize>() * fe_bytes;
                            t += fe_bytes;
                        }
                        t += (lp.p_lefts.len() + lp.p_rights.len() + lp.q_lefts.len() + lp.q_rights.len()) * fe_bytes;
                    }
                    let sent_p: usize = wg.sent_p.iter().map(|v| v.len()).sum();
                    let sent_q: usize = wg.sent_q.iter().map(|v| v.len()).sum();
                    t += (sent_p + sent_q) * fe_bytes;
                    let tg = &group.table_gkr;
                    t += 2 * fe_bytes;
                    for lp in &tg.layer_proofs {
                        if let Some(ref sc) = lp.sumcheck_proof {
                            t += sc.messages.iter().map(|m| m.0.tail_evaluations.len()).sum::<usize>() * fe_bytes;
                            t += fe_bytes;
                        }
                        t += 4 * fe_bytes;
                    }
                }
                t
            }
            None => 0,
        };
        let gkr_shift_sc: usize = gkr_folded_proof.shift_sumcheck.as_ref().map_or(0, |sc| {
            let r: usize = sc.rounds.iter().map(|v| v.len()).sum();
            let f: usize = sc.v_finals.iter().map(|v| v.len()).sum();
            r + f
        });
        let gkr_folding_c1s: usize = gkr_folded_proof.folding_c1s_bytes.iter().map(|v| v.len()).sum();
        let gkr_folding_c2s: usize = gkr_folded_proof.folding_c2s_bytes.iter().map(|v| v.len()).sum();
        let gkr_folding_total = gkr_folding_c1s + gkr_folding_c2s;
        let gkr_eval_pt: usize = gkr_folded_proof.evaluation_point_bytes.iter().map(|v| v.len()).sum();
        let gkr_pcs_eval: usize = gkr_folded_proof.pcs_evals_bytes.iter().map(|v| v.len()).sum();

        let gkr_piop_total = gkr_ic_bytes + gkr_cpr_sc_total + gkr_cpr_eval_total
            + gkr_lookup_bytes + gkr_shift_sc + gkr_folding_total + gkr_eval_pt + gkr_pcs_eval;
        let gkr_total_raw = gkr_pcs_bytes + gkr_piop_total;

        // Build GKR serialized byte buffer and compress.
        let mut gkr_all_bytes = Vec::with_capacity(gkr_total_raw);
        gkr_all_bytes.extend(&gkr_folded_proof.pcs_proof_bytes);
        for v in &gkr_folded_proof.ic_proof_values { gkr_all_bytes.extend(v); }
        for v in &gkr_folded_proof.cpr_sumcheck_messages { gkr_all_bytes.extend(v); }
        gkr_all_bytes.extend(&gkr_folded_proof.cpr_sumcheck_claimed_sum);
        for v in &gkr_folded_proof.cpr_up_evals { gkr_all_bytes.extend(v); }
        for v in &gkr_folded_proof.cpr_down_evals { gkr_all_bytes.extend(v); }
        // Serialize GKR lookup proof fields.
        if let Some(LookupProofData::Gkr(gkr_proof)) = &gkr_folded_proof.lookup_proof {
            fn write_fe(buf: &mut Vec<u8>, f: &zinc_snark::pipeline::PiopField) {
                use zinc_snark::pipeline::FIELD_LIMBS;
                let sz = <Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;
                let start = buf.len();
                buf.resize(start + sz, 0);
                f.inner().write_transcription_bytes(&mut buf[start..]);
            }
            for gp in &gkr_proof.group_proofs {
                for v in &gp.aggregated_multiplicities {
                    for f in v { write_fe(&mut gkr_all_bytes, f); }
                }
                for r in &gp.witness_gkr.roots_p { write_fe(&mut gkr_all_bytes, r); }
                for r in &gp.witness_gkr.roots_q { write_fe(&mut gkr_all_bytes, r); }
                for layer in &gp.witness_gkr.layer_proofs {
                    if let Some(ref sc) = layer.sumcheck_proof {
                        write_fe(&mut gkr_all_bytes, &sc.claimed_sum);
                        for msg in &sc.messages {
                            for e in &msg.0.tail_evaluations { write_fe(&mut gkr_all_bytes, e); }
                        }
                    }
                    for e in &layer.p_lefts { write_fe(&mut gkr_all_bytes, e); }
                    for e in &layer.p_rights { write_fe(&mut gkr_all_bytes, e); }
                    for e in &layer.q_lefts { write_fe(&mut gkr_all_bytes, e); }
                    for e in &layer.q_rights { write_fe(&mut gkr_all_bytes, e); }
                }
                write_fe(&mut gkr_all_bytes, &gp.table_gkr.root_p);
                write_fe(&mut gkr_all_bytes, &gp.table_gkr.root_q);
                for layer in &gp.table_gkr.layer_proofs {
                    if let Some(ref sc) = layer.sumcheck_proof {
                        write_fe(&mut gkr_all_bytes, &sc.claimed_sum);
                        for msg in &sc.messages {
                            for e in &msg.0.tail_evaluations { write_fe(&mut gkr_all_bytes, e); }
                        }
                    }
                    write_fe(&mut gkr_all_bytes, &layer.p_left);
                    write_fe(&mut gkr_all_bytes, &layer.p_right);
                    write_fe(&mut gkr_all_bytes, &layer.q_left);
                    write_fe(&mut gkr_all_bytes, &layer.q_right);
                }
            }
        }
        if let Some(ref sc) = gkr_folded_proof.shift_sumcheck {
            for v in &sc.rounds { gkr_all_bytes.extend(v); }
            for v in &sc.v_finals { gkr_all_bytes.extend(v); }
        }
        for v in &gkr_folded_proof.folding_c1s_bytes { gkr_all_bytes.extend(v); }
        for v in &gkr_folded_proof.folding_c2s_bytes { gkr_all_bytes.extend(v); }
        for v in &gkr_folded_proof.evaluation_point_bytes { gkr_all_bytes.extend(v); }
        for v in &gkr_folded_proof.pcs_evals_bytes { gkr_all_bytes.extend(v); }

        let gkr_compressed = {
            use std::io::Write;
            let mut encoder = flate2::write::DeflateEncoder::new(
                Vec::new(), flate2::Compression::default(),
            );
            encoder.write_all(&gkr_all_bytes).unwrap();
            encoder.finish().unwrap()
        };

        eprintln!("\n=== 8xSHA256 GKR Folded Proof Size ===");
        eprintln!("  PCS:            {:>6} B  ({:.1} KB)", gkr_pcs_bytes, gkr_pcs_bytes as f64 / 1024.0);
        eprintln!("  IC:             {:>6} B", gkr_ic_bytes);
        eprintln!("  CPR sumcheck:   {:>6} B  (msgs={}, sum={})", gkr_cpr_sc_total, gkr_cpr_msg, gkr_cpr_sum);
        eprintln!("  CPR evals:      {:>6} B  (up={}, down={})", gkr_cpr_eval_total, gkr_cpr_up, gkr_cpr_dn);
        eprintln!("  Lookup (GKR):   {:>6} B", gkr_lookup_bytes);
        eprintln!("  Shift SC:       {:>6} B", gkr_shift_sc);
        eprintln!("  Folding:        {:>6} B  (c1s={}, c2s={})", gkr_folding_total, gkr_folding_c1s, gkr_folding_c2s);
        eprintln!("  Eval point:     {:>6} B", gkr_eval_pt);
        eprintln!("  PCS evals:      {:>6} B", gkr_pcs_eval);
        eprintln!("  ─────────────────────────");
        eprintln!("  PIOP total:     {:>6} B  ({:.1} KB)", gkr_piop_total, gkr_piop_total as f64 / 1024.0);
        eprintln!("  Total raw:      {:>6} B  ({:.1} KB)", gkr_total_raw, gkr_total_raw as f64 / 1024.0);
        eprintln!("  Compressed:     {:>6} B  ({:.1} KB, {:.1}x ratio)",
            gkr_compressed.len(), gkr_compressed.len() as f64 / 1024.0,
            gkr_all_bytes.len() as f64 / gkr_compressed.len() as f64);

        eprintln!("\n=== Lookup Variant Comparison (Folded) ===");
        eprintln!("  Classic lookup:   {:>6} B  ({:.1} KB)", lookup_bytes, lookup_bytes as f64 / 1024.0);
        eprintln!("  GKR lookup:       {:>6} B  ({:.1} KB)", gkr_lookup_bytes, gkr_lookup_bytes as f64 / 1024.0);
        let lk_diff = lookup_bytes as i64 - gkr_lookup_bytes as i64;
        eprintln!("  Lookup savings:   {:>6} B  ({:+.1} KB, {:.1}x)",
            lk_diff, lk_diff as f64 / 1024.0,
            lookup_bytes as f64 / gkr_lookup_bytes.max(1) as f64);
        eprintln!("  ─────────────────────────");
        eprintln!("  Classic total:    {:>6} B  ({:.1} KB)", total_raw, total_raw as f64 / 1024.0);
        eprintln!("  GKR total:        {:>6} B  ({:.1} KB)", gkr_total_raw, gkr_total_raw as f64 / 1024.0);
        let total_lk_diff = total_raw as i64 - gkr_total_raw as i64;
        eprintln!("  Total savings:    {:>6} B  ({:+.1} KB, {:.1}x)",
            total_lk_diff, total_lk_diff as f64 / 1024.0,
            total_raw as f64 / gkr_total_raw.max(1) as f64);
        eprintln!("  ─────────────────────────");
        eprintln!("  Classic compr:    {:>6} B  ({:.1} KB)", compressed.len(), compressed.len() as f64 / 1024.0);
        eprintln!("  GKR compr:        {:>6} B  ({:.1} KB)", gkr_compressed.len(), gkr_compressed.len() as f64 / 1024.0);
        let compr_lk_diff = compressed.len() as i64 - gkr_compressed.len() as i64;
        eprintln!("  Compr savings:    {:>6} B  ({:+.1} KB, {:.1}x)",
            compr_lk_diff, compr_lk_diff as f64 / 1024.0,
            compressed.len() as f64 / gkr_compressed.len().max(1) as f64);
    }

    // ── Hybrid GKR LogUp: cost analysis + verifier timing ─────────
    //
    // Explores the tradeoff: run c layers of GKR from root, then send
    // the intermediate fraction values in the clear. The bottom half
    // runs as a fresh GKR of depth d-c (fewer sumcheck rounds).
    {
        use zinc_piop::lookup::{
            HybridGkrBatchedDecompLogupProtocol,
            analyze_hybrid_costs,
        };
        use zinc_piop::lookup::structs::BatchedDecompLookupInstance;
        use zinc_piop::lookup::tables::{
            bitpoly_shift, generate_bitpoly_table,
            decompose_raw_indices_to_chunks,
        };
        use zinc_snark::pipeline::FIELD_LIMBS;

        let fe_bytes = <Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;

        // ── Print cost analysis table ────────────────────────────────
        let metrics = analyze_hybrid_costs(
            SHA256_LOOKUP_COL_COUNT + 3, // 10 column lookups + 3 affine
            8,                           // K=8 chunks
            1usize << SHA256_8X_NUM_VARS, // W=512
            16,                          // T=16 (subtable for chunk_width=4)
            fe_bytes,
        );

        eprintln!("\n=== Hybrid GKR LogUp Cost Analysis ===");
        eprintln!("  Parameters: L={}, K=8, W={}, T=16, d_w={}, fe={}B",
            SHA256_LOOKUP_COL_COUNT + 3,
            1usize << SHA256_8X_NUM_VARS,
            metrics[0].tree_depth,
            fe_bytes);
        eprintln!("  Full GKR witness: {} B, Classic witness: {} B",
            metrics[0].full_gkr_witness_proof_bytes,
            metrics[0].classic_witness_proof_bytes);
        eprintln!("  Full GKR sumcheck rounds: {}\n", metrics[0].full_sc_rounds);
        eprintln!("  {:>2} | {:>8} {:>8} | {:>5} {:>5} {:>5} {:>5} | {:>6} | {:>6}",
            "c", "TopOnly", "Top+Bot", "TopSC", "BotSC", "Total", "Saved", "MLE", "SentFE");
        for m in &metrics {
            eprintln!("  {:>2} | {:>6}B {:>6}B | {:>5} {:>5} {:>5} {:>5} | {:>6} | {:>6}",
                m.cutoff,
                m.hybrid_top_only_proof_bytes,
                m.hybrid_full_proof_bytes,
                m.top_sc_rounds,
                m.bottom_sc_rounds,
                m.hybrid_total_sc_rounds,
                m.sc_rounds_saved,
                m.mle_eval_ops,
                m.sent_intermediate_fe,
            );
        }

        // ── Benchmark hybrid verifier at various cutoffs ─────────────
        //
        // Build a lookup instance using projected trace columns and
        // benchmark the hybrid protocol's prove + verify.

        let mut h_transcript = zinc_transcript::KeccakTranscript::new();
        let h_field_cfg = h_transcript.get_random_field_cfg::<
            F, <F as Field>::Inner, MillerRabin
        >();

        let projecting_element: F = h_transcript.get_field_challenge(&h_field_cfg);
        let bin_proj = BinaryPoly::<32>::prepare_projection(&projecting_element);

        // Project the first 10 lookup columns to field elements.
        let projected_cols: Vec<Vec<F>> = (0..SHA256_LOOKUP_COL_COUNT)
            .map(|col_idx| {
                sha_trace[col_idx].iter()
                    .map(|bp| bin_proj(bp))
                    .collect()
            })
            .collect();

        // Generate subtable and shifts.
        let chunk_width = 4usize;
        let num_chunks = 32 / chunk_width; // 8
        let subtable = generate_bitpoly_table(chunk_width, &projecting_element, &h_field_cfg);
        let shifts: Vec<F> = (0..num_chunks)
            .map(|k| bitpoly_shift(k * chunk_width, &projecting_element))
            .collect();

        // Build raw indices from the trace for decomposition.
        let raw_indices: Vec<Vec<usize>> = (0..SHA256_LOOKUP_COL_COUNT)
            .map(|col_idx| {
                sha_trace[col_idx].iter()
                    .map(|bp| {
                        let mut idx = 0usize;
                        for (j, coeff) in bp.iter().enumerate() {
                            if coeff.into_inner() { idx |= 1usize << j; }
                        }
                        idx
                    })
                    .collect()
            })
            .collect();

        // Build chunk decompositions.
        let all_chunks: Vec<Vec<Vec<F>>> = raw_indices.iter()
            .map(|indices| {
                decompose_raw_indices_to_chunks(
                    indices, 32, chunk_width, &subtable,
                )
            })
            .collect();

        let lookup_instance = BatchedDecompLookupInstance {
            witnesses: projected_cols,
            subtable: subtable.clone(),
            shifts: shifts.clone(),
            chunks: all_chunks,
        };

        // Benchmark hybrid prove+verify at cutoffs 1..8
        let cutoffs_to_bench = [1, 2, 3, 4, 5, 6, 8, 10];
        eprintln!("\n=== Hybrid GKR Prover + Verifier Timing ===");

        for &cutoff in &cutoffs_to_bench {
            // Prove once for correctness + warm up.
            let mut pt = zinc_transcript::KeccakTranscript::new();
            let (hybrid_proof, _state) =
                HybridGkrBatchedDecompLogupProtocol::<F>::prove_as_subprotocol(
                    &mut pt, &lookup_instance, cutoff, &h_field_cfg,
                )
                .expect("hybrid prove should succeed");

            // Verify once for correctness.
            let mut vt = zinc_transcript::KeccakTranscript::new();
            let _ = HybridGkrBatchedDecompLogupProtocol::<F>::verify_as_subprotocol(
                &mut vt, &hybrid_proof, &subtable, &shifts,
                SHA256_LOOKUP_COL_COUNT, 1usize << SHA256_8X_NUM_VARS, &h_field_cfg,
            ).expect("hybrid verify should succeed");

            // Time the prover (N iterations).
            let n_iters = 100;
            let start = Instant::now();
            for _ in 0..n_iters {
                let mut pt2 = zinc_transcript::KeccakTranscript::new();
                let _ = black_box(
                    HybridGkrBatchedDecompLogupProtocol::<F>::prove_as_subprotocol(
                        &mut pt2, &lookup_instance, cutoff, &h_field_cfg,
                    )
                );
            }
            let prove_elapsed = start.elapsed();
            let prove_per_iter = prove_elapsed / n_iters as u32;

            // Time the verifier (N iterations).
            let start = Instant::now();
            for _ in 0..n_iters {
                let mut vt2 = zinc_transcript::KeccakTranscript::new();
                let _ = black_box(
                    HybridGkrBatchedDecompLogupProtocol::<F>::verify_as_subprotocol(
                        &mut vt2, &hybrid_proof, &subtable, &shifts,
                        SHA256_LOOKUP_COL_COUNT, 1usize << SHA256_8X_NUM_VARS, &h_field_cfg,
                    )
                );
            }
            let elapsed = start.elapsed();
            let per_iter = elapsed / n_iters as u32;

            // Compute proof size.
            let mut hybrid_proof_fe = 0usize;
            // Aggregated multiplicities.
            for agg in &hybrid_proof.aggregated_multiplicities {
                hybrid_proof_fe += agg.len();
            }
            // Witness GKR (hybrid).
            let wg = &hybrid_proof.witness_gkr;
            hybrid_proof_fe += wg.roots_p.len() + wg.roots_q.len();
            for layer in &wg.layer_proofs {
                if let Some(ref sc) = layer.sumcheck_proof {
                    hybrid_proof_fe += 1; // claimed_sum
                    for msg in &sc.messages {
                        hybrid_proof_fe += msg.0.tail_evaluations.len();
                    }
                }
                hybrid_proof_fe += layer.p_lefts.len() + layer.p_rights.len()
                    + layer.q_lefts.len() + layer.q_rights.len();
            }
            // Sent intermediate values.
            for v in &wg.sent_p { hybrid_proof_fe += v.len(); }
            for v in &wg.sent_q { hybrid_proof_fe += v.len(); }
            // Table GKR.
            let tg = &hybrid_proof.table_gkr;
            hybrid_proof_fe += 2; // root_p, root_q
            for layer in &tg.layer_proofs {
                if let Some(ref sc) = layer.sumcheck_proof {
                    hybrid_proof_fe += 1;
                    for msg in &sc.messages {
                        hybrid_proof_fe += msg.0.tail_evaluations.len();
                    }
                }
                hybrid_proof_fe += 4; // p_left, p_right, q_left, q_right
            }

            let hybrid_proof_bytes = hybrid_proof_fe * fe_bytes;

            eprintln!("  cutoff={:>2}: prove={:>7.3}ms  verify={:>7.3}ms  proof={:>6}B ({:>4}FE)  sent={:>5}FE",
                cutoff,
                prove_per_iter.as_secs_f64() * 1000.0,
                per_iter.as_secs_f64() * 1000.0,
                hybrid_proof_bytes,
                hybrid_proof_fe,
                wg.sent_p.iter().map(|v| v.len()).sum::<usize>()
                    + wg.sent_q.iter().map(|v| v.len()).sum::<usize>(),
            );
        }

        // Also benchmark the full GKR verifier for comparison.
        {
            use zinc_piop::lookup::GkrBatchedDecompLogupProtocol;

            let mut pt = zinc_transcript::KeccakTranscript::new();
            let (full_proof, _state) =
                GkrBatchedDecompLogupProtocol::<F>::prove_as_subprotocol(
                    &mut pt, &lookup_instance, &h_field_cfg,
                )
                .expect("full GKR prove should succeed");

            // Time the full GKR prover.
            let n_iters = 100;
            let start = Instant::now();
            for _ in 0..n_iters {
                let mut pt2 = zinc_transcript::KeccakTranscript::new();
                let _ = black_box(
                    GkrBatchedDecompLogupProtocol::<F>::prove_as_subprotocol(
                        &mut pt2, &lookup_instance, &h_field_cfg,
                    )
                );
            }
            let prove_elapsed = start.elapsed();
            let prove_per_iter = prove_elapsed / n_iters as u32;

            // Time the full GKR verifier.
            let start = Instant::now();
            for _ in 0..n_iters {
                let mut vt2 = zinc_transcript::KeccakTranscript::new();
                let _ = black_box(
                    GkrBatchedDecompLogupProtocol::<F>::verify_as_subprotocol(
                        &mut vt2, &full_proof, &subtable, &shifts,
                        SHA256_LOOKUP_COL_COUNT, 1usize << SHA256_8X_NUM_VARS, &h_field_cfg,
                    )
                );
            }
            let elapsed = start.elapsed();
            let per_iter = elapsed / n_iters as u32;

            eprintln!("  full GKR: prove={:>7.3}ms  verify={:>7.3}ms  (baseline)",
                prove_per_iter.as_secs_f64() * 1000.0,
                per_iter.as_secs_f64() * 1000.0);
        }
    }

    let mem_snapshot = mem_tracker.stop();
    eprintln!("\n=== Peak Memory ===");
    eprintln!("  {mem_snapshot}");

    group.finish();
}

criterion_group!(benches, sha256_8x_folded_stepwise);
criterion_main!(benches);
