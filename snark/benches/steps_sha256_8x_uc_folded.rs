//! Per-step breakdown benchmarks for the 8×SHA-256 proving stack with
//! **underconstrained** UAIR (F₂[X] columns removed) and **folded**
//! BinaryPoly<32> → BinaryPoly<16> columns.
//!
//! This benchmark mirrors [`steps_sha256_8x_folded`] but uses
//! `Sha256UairBpUnderconstrained` which drops the 4 F₂[X] columns (S₀, S₁,
//! R₀, R₁) and the 4 constraints referencing them, resulting in 26 trace
//! columns (23 bitpoly + 3 int) instead of 30 (27 + 3).
//!
//! Run with:
//!   cargo bench --bench steps_sha256_8x_uc_folded -p zinc-snark --features "parallel simd asm"

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
            PnttConfigF2_16R4B32,
        },
    },
    pcs::structs::{ZipPlus, ZipPlusParams, ZipTypes},
    pcs::folding::split_columns,
};

use zinc_sha256_uair::underconstrained::{
    Sha256UairBpUnderconstrained,
    UC_NUM_COLS,
    UC_COL_A_HAT, UC_COL_E_HAT, UC_COL_E_TM1, UC_COL_E_TM2,
    UC_COL_CH_EF_HAT, UC_COL_CH_NEG_EG_HAT,
    UC_COL_A_TM1, UC_COL_A_TM2, UC_COL_MAJ_HAT,
};
use zinc_sha256_uair::Sha256Ideal;
use zinc_sha256_uair::Sha256UairQx;
use zinc_sha256_uair::witness::GenerateWitness;
use zinc_uair::Uair;
use zinc_piop::projections::{
    project_trace_to_field,
    project_scalars, project_scalars_to_field,
};
use zinc_piop::lookup::{AffineLookupSpec, LookupColumnSpec, LookupTableType};

use zinc_piop::ideal_check::IdealCheckProtocol;
use zinc_piop::combined_poly_resolver::CombinedPolyResolver;



// ─── Type definitions ───────────────────────────────────────────────────────

const INT_LIMBS: usize = U64::LIMBS;
type F = MontyField<{ INT_LIMBS * 3 }>;

struct UcSha256ZipTypes<CwCoeff, const D_PLUS_ONE: usize>(PhantomData<CwCoeff>);

impl<CwCoeff, const D_PLUS_ONE: usize> ZipTypes for UcSha256ZipTypes<CwCoeff, D_PLUS_ONE>
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
    const NUM_COLUMN_OPENINGS: usize = 118;
    const GRINDING_BITS: usize = 16;
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

/// Original (non-folded) PCS types — used only for side-by-side comparison.
type OrigZt = UcSha256ZipTypes<i64, 32>;
type OrigLc = IprsCode<
    OrigZt,
    PnttConfigF12289R4B64<1>,
    BinaryPolyWideningMulByScalar<i64>,
    UNCHECKED,
>;

/// Folded PCS types — BinaryPoly<16> codewords.
type FoldedZt = UcSha256ZipTypes<i64, 16>;
type FoldedLc = IprsCode<
    FoldedZt,
    PnttConfigF12289R4B16<2>,
    BinaryPolyWideningMulByScalar<i64>,
    UNCHECKED,
>;

/// 4x-Folded PCS types — BinaryPoly<8> codewords.
type FoldedZt4x = UcSha256ZipTypes<i64, 8>;
type FoldedLc4x = IprsCode<
    FoldedZt4x,
    PnttConfigF2_16R4B32<2>,
    BinaryPolyWideningMulByScalar<i64>,
    UNCHECKED,
>;

// ─── Parameters ─────────────────────────────────────────────────────────────

const SHA256_8X_NUM_VARS: usize = 9;      // 2^9 = 512 rows (8 × 64 SHA rounds)
const SHA256_UC_BATCH_SIZE: usize = UC_NUM_COLS; // 26 columns (23 bitpoly + 3 int)
const SHA256_LOOKUP_COL_COUNT: usize = 10; // 10 Q[X] columns need lookup

/// Default lookup specs: 8 chunks of 2^4 each (chunk_width=4, total_width=32).
fn uc_sha256_lookup_specs() -> Vec<LookupColumnSpec> {
    (0..SHA256_LOOKUP_COL_COUNT)
        .map(|i| LookupColumnSpec {
            column_index: i,
            table_type: LookupTableType::BitPoly { width: 32, chunk_width: Some(4) },
        })
        .collect()
}

fn uc_sha256_affine_lookup_specs() -> Vec<AffineLookupSpec> {
    let bp32 = LookupTableType::BitPoly { width: 32, chunk_width: Some(4) };
    vec![
        AffineLookupSpec {
            terms: vec![(UC_COL_E_HAT, 1), (UC_COL_E_TM1, 1), (UC_COL_CH_EF_HAT, -2)],
            constant_offset_bits: 0,
            table_type: bp32.clone(),
        },
        AffineLookupSpec {
            terms: vec![(UC_COL_E_HAT, -1), (UC_COL_E_TM2, 1), (UC_COL_CH_NEG_EG_HAT, -2)],
            constant_offset_bits: 0xFFFF_FFFF,
            table_type: bp32.clone(),
        },
        AffineLookupSpec {
            terms: vec![(UC_COL_A_HAT, 1), (UC_COL_A_TM1, 1), (UC_COL_A_TM2, 1), (UC_COL_MAJ_HAT, -2)],
            constant_offset_bits: 0,
            table_type: bp32,
        },
    ]
}

fn generate_uc_sha256_trace(num_vars: usize) -> Vec<DenseMultilinearExtension<BinaryPoly<32>>> {
    let mut rng = rand::rng();
    <Sha256UairBpUnderconstrained as GenerateWitness<BinaryPoly<32>>>::generate_witness(num_vars, &mut rng)
}

/// Lookup specs with 2 chunks of 2^16 each.
#[allow(dead_code)]
fn uc_sha256_lookup_specs_2chunks() -> Vec<LookupColumnSpec> {
    (0..SHA256_LOOKUP_COL_COUNT)
        .map(|i| LookupColumnSpec {
            column_index: i,
            table_type: LookupTableType::BitPoly { width: 32, chunk_width: Some(16) },
        })
        .collect()
}

#[allow(dead_code)]
fn uc_sha256_affine_lookup_specs_2chunks() -> Vec<AffineLookupSpec> {
    let bp32 = LookupTableType::BitPoly { width: 32, chunk_width: Some(16) };
    vec![
        AffineLookupSpec {
            terms: vec![(UC_COL_E_HAT, 1), (UC_COL_E_TM1, 1), (UC_COL_CH_EF_HAT, -2)],
            constant_offset_bits: 0,
            table_type: bp32.clone(),
        },
        AffineLookupSpec {
            terms: vec![(UC_COL_E_HAT, -1), (UC_COL_E_TM2, 1), (UC_COL_CH_NEG_EG_HAT, -2)],
            constant_offset_bits: 0xFFFF_FFFF,
            table_type: bp32.clone(),
        },
        AffineLookupSpec {
            terms: vec![(UC_COL_A_HAT, 1), (UC_COL_A_TM1, 1), (UC_COL_A_TM2, 1), (UC_COL_MAJ_HAT, -2)],
            constant_offset_bits: 0,
            table_type: bp32,
        },
    ]
}

/// Lookup specs with 4 chunks of 2^8 each.
fn uc_sha256_lookup_specs_4chunks() -> Vec<LookupColumnSpec> {
    (0..SHA256_LOOKUP_COL_COUNT)
        .map(|i| LookupColumnSpec {
            column_index: i,
            table_type: LookupTableType::BitPoly { width: 32, chunk_width: Some(8) },
        })
        .collect()
}

fn uc_sha256_affine_lookup_specs_4chunks() -> Vec<AffineLookupSpec> {
    let bp32 = LookupTableType::BitPoly { width: 32, chunk_width: Some(8) };
    vec![
        AffineLookupSpec {
            terms: vec![(UC_COL_E_HAT, 1), (UC_COL_E_TM1, 1), (UC_COL_CH_EF_HAT, -2)],
            constant_offset_bits: 0,
            table_type: bp32.clone(),
        },
        AffineLookupSpec {
            terms: vec![(UC_COL_E_HAT, -1), (UC_COL_E_TM2, 1), (UC_COL_CH_NEG_EG_HAT, -2)],
            constant_offset_bits: 0xFFFF_FFFF,
            table_type: bp32.clone(),
        },
        AffineLookupSpec {
            terms: vec![(UC_COL_A_HAT, 1), (UC_COL_A_TM1, 1), (UC_COL_A_TM2, 1), (UC_COL_MAJ_HAT, -2)],
            constant_offset_bits: 0,
            table_type: bp32,
        },
    ]
}

/// Lookup specs with 3 chunks of 2^11 each (chunk_width=11, total_width=33).
#[allow(dead_code)]
fn uc_sha256_lookup_specs_3chunks() -> Vec<LookupColumnSpec> {
    (0..SHA256_LOOKUP_COL_COUNT)
        .map(|i| LookupColumnSpec {
            column_index: i,
            table_type: LookupTableType::BitPoly { width: 33, chunk_width: Some(11) },
        })
        .collect()
}

#[allow(dead_code)]
fn uc_sha256_affine_lookup_specs_3chunks() -> Vec<AffineLookupSpec> {
    let bp33 = LookupTableType::BitPoly { width: 33, chunk_width: Some(11) };
    vec![
        AffineLookupSpec {
            terms: vec![(UC_COL_E_HAT, 1), (UC_COL_E_TM1, 1), (UC_COL_CH_EF_HAT, -2)],
            constant_offset_bits: 0,
            table_type: bp33.clone(),
        },
        AffineLookupSpec {
            terms: vec![(UC_COL_E_HAT, -1), (UC_COL_E_TM2, 1), (UC_COL_CH_NEG_EG_HAT, -2)],
            constant_offset_bits: 0xFFFF_FFFF,
            table_type: bp33.clone(),
        },
        AffineLookupSpec {
            terms: vec![(UC_COL_A_HAT, 1), (UC_COL_A_TM1, 1), (UC_COL_A_TM2, 1), (UC_COL_MAJ_HAT, -2)],
            constant_offset_bits: 0,
            table_type: bp33,
        },
    ]
}

// ─── Benchmark ──────────────────────────────────────────────────────────────

/// Measures each main step of the **folded** E2E proving stack for the
/// **underconstrained** 8×SHA-256 (no F₂[X] columns).
///
/// Prover steps benchmarked:
///   1.  WitnessGen — generate the 26-column BinaryPoly<32> trace (512 rows)
///   2.  Folding/SplitColumns — split BinaryPoly<32> to BinaryPoly<16> halves
///   3.  PCS/Commit (folded) — Zip+ commit over BinaryPoly<16> split columns
///   4.  PCS/Commit (original) — Zip+ commit over BinaryPoly<32> (comparison)
///   5.  PIOP/FieldSetup — transcript init + random field config
///   6.  PIOP/Project Ideal Check — project_scalars for Ideal Check
///   7.  PIOP/IdealCheck — Ideal Check prover (MLE-first)
///   8.  PIOP/Project Main field sumcheck — project_scalars_to_field + project_trace_to_field
///   9.  PIOP/Main field sumcheck — Combined Poly Resolver prover
///  10.  PIOP/LookupExtract — extract lookup columns from field trace
///  11.  PIOP/Lookup — classic batched decomposed LogUp prover
///  12.  PCS/Prove (folded)
///  13.  PCS/Prove (original)
///  14.  E2E/Prover (folded)
///  15.  E2E/Prover (original)
///  16.  E2E/Verifier (folded)
///  17.  E2E/Verifier (original)
fn uc_sha256_8x_folded_stepwise(c: &mut Criterion) {
    use zinc_uair::ideal_collector::IdealOrZero;
    use zinc_piop::lookup::prove_batched_lookup_with_indices;

    let mem_tracker = MemoryTracker::start();

    let mut group = c.benchmark_group("8xSHA256 UC Folded Steps");
    group.sample_size(100);

    // ── Original PCS params (for comparison) ────────────────────────
    let orig_lc = OrigLc::new(512);
    let orig_params = ZipPlusParams::<OrigZt, OrigLc>::new(SHA256_8X_NUM_VARS, 1, orig_lc);

    // ── Folded PCS params ───────────────────────────────────────────
    let folded_num_vars = SHA256_8X_NUM_VARS + 1; // 10
    let row_len = 1024;
    let folded_num_rows = (1usize << folded_num_vars) / row_len; // 1
    let folded_lc = FoldedLc::new(row_len);
    let folded_params = ZipPlusParams::<FoldedZt, FoldedLc>::new(
        folded_num_vars, folded_num_rows, folded_lc,
    );

    // 4x-folded params
    let folded_4x_num_vars = SHA256_8X_NUM_VARS + 2; // 11
    let row_len_4x = 2048;
    let folded_4x_lc = FoldedLc4x::new(row_len_4x);
    let folded_4x_params = ZipPlusParams::<FoldedZt4x, FoldedLc4x>::new(
        folded_4x_num_vars, 1, folded_4x_lc,
    );

    let sha_lookup_specs = uc_sha256_lookup_specs();
    let sha_affine_specs = uc_sha256_affine_lookup_specs();

    let num_constraints = zinc_uair::constraint_counter::count_constraints::<Sha256UairBpUnderconstrained>();
    let max_degree = zinc_uair::degree_counter::count_max_degree::<Sha256UairBpUnderconstrained>();
    let num_vars = SHA256_8X_NUM_VARS;

    // ── 1. Witness Generation ───────────────────────────────────────
    group.bench_function("WitnessGen", |b| {
        b.iter(|| {
            let trace = generate_uc_sha256_trace(SHA256_8X_NUM_VARS);
            black_box(trace);
        });
    });

    // Pre-generate the trace used by all subsequent steps.
    let sha_trace = generate_uc_sha256_trace(SHA256_8X_NUM_VARS);
    assert_eq!(sha_trace.len(), SHA256_UC_BATCH_SIZE);

    // Build private (PCS-committed) trace — exclude public + shift-source columns.
    let sha_sig = Sha256UairBpUnderconstrained::signature();
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
    assert_eq!(max_degree, 1, "UC SHA-256 UAIR should have max_degree == 1 (MLE-first path)");
    group.bench_function("PIOP/Project Ideal Check", |b| {
        let mut tr_setup = zinc_transcript::KeccakTranscript::new();
        let fcfg = tr_setup.get_random_field_cfg::<F, <F as Field>::Inner, MillerRabin>();
        b.iter(|| {
            let projected_scalars = project_scalars::<F, Sha256UairBpUnderconstrained>(|scalar| {
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

    // ── 7. PIOP / Ideal Check (MLE-first)
    group.bench_function("PIOP/IdealCheck", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut transcript = zinc_transcript::KeccakTranscript::new();
                let field_cfg = transcript.get_random_field_cfg::<
                    F, <F as Field>::Inner, MillerRabin
                >();
                let projected_scalars = project_scalars::<F, Sha256UairBpUnderconstrained>(|scalar| {
                    let one = F::one_with_cfg(&field_cfg);
                    let zero = F::zero_with_cfg(&field_cfg);
                    DynamicPolynomialF::new(
                        scalar.iter().map(|coeff| {
                            if coeff.into_inner() { one.clone() } else { zero.clone() }
                        }).collect::<Vec<_>>()
                    )
                });
                let t = Instant::now();
                let _ = IdealCheckProtocol::<F>::prove_mle_first::<Sha256UairBpUnderconstrained, 32>(
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
    let projected_scalars_cpr = project_scalars::<F, Sha256UairBpUnderconstrained>(|scalar| {
        let one = F::one_with_cfg(&field_cfg_cpr);
        let zero = F::zero_with_cfg(&field_cfg_cpr);
        DynamicPolynomialF::new(
            scalar.iter().map(|coeff| {
                if coeff.into_inner() { one.clone() } else { zero.clone() }
            }).collect::<Vec<_>>()
        )
    });
    let (_ic_proof_cpr, ic_state_cpr) =
        IdealCheckProtocol::<F>::prove_mle_first::<Sha256UairBpUnderconstrained, 32>(
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
                let _ = CombinedPolyResolver::<F>::prove_as_subprotocol::<Sha256UairBpUnderconstrained>(
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
        let _ = CombinedPolyResolver::<F>::prove_as_subprotocol::<Sha256UairBpUnderconstrained>(
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

    // ── 12. PCS Prove (folded — BinaryPoly<16> split columns) ───────
    let folded_pcs_point: Vec<F> = {
        let mut tr = transcript_for_cpr.clone();
        let (_, cpr_state) = CombinedPolyResolver::<F>::prove_as_subprotocol::<Sha256UairBpUnderconstrained>(
            &mut tr,
            field_trace_cpr.clone(),
            &ic_state_cpr.evaluation_point,
            &field_projected_scalars_cpr,
            num_constraints,
            SHA256_8X_NUM_VARS,
            max_degree,
            &field_cfg_cpr,
        ).expect("Main field sumcheck prover failed");
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

    // ── E2E benchmarks: only Hybrid GKR c=2 regime is active.
    // ── Classic logup / non-hybrid GKR / original pipeline commented out.
    /*
    // ── 14. E2E Total Prover (folded pipeline) ──────────────────────
    group.bench_function("E2E/Prover (folded)", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let t = Instant::now();
                let _ = zinc_snark::pipeline::prove_classic_logup_folded::<
                    Sha256UairBpUnderconstrained, FoldedZt, FoldedLc, 32, 16, UNCHECKED,
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
                let _ = zinc_snark::pipeline::prove::<Sha256UairBpUnderconstrained, OrigZt, OrigLc, 32, UNCHECKED>(
                    &orig_params, &sha_trace, SHA256_8X_NUM_VARS, &sha_lookup_specs,
                );
                total += t.elapsed();
            }
            total
        });
    });

    // ── 16. E2E Total Verifier (folded) ─────────────────────────────
    let folded_proof = zinc_snark::pipeline::prove_classic_logup_folded::<
        Sha256UairBpUnderconstrained, FoldedZt, FoldedLc, 32, 16, UNCHECKED,
    >(
        &folded_params, &sha_trace, SHA256_8X_NUM_VARS,
        &sha_lookup_specs, &sha_affine_specs,
    );
    */

    let sha_sig_pub = Sha256UairBpUnderconstrained::signature();
    let sha_public_cols: Vec<_> = sha_sig_pub.public_columns.iter()
        .map(|&i| sha_trace[i].clone()).collect();

    /*
    {
        let r = zinc_snark::pipeline::verify_classic_logup_folded::<
            Sha256UairBpUnderconstrained, FoldedZt, FoldedLc, 32, 16, UNCHECKED, _, _,
        >(
            &folded_params, &folded_proof, SHA256_8X_NUM_VARS,
            |_: &IdealOrZero<Sha256Ideal>| zinc_snark::pipeline::TrivialIdeal,
            &sha_public_cols,
        );
        let t = &r.timing;
        println!("\n── Verifier step timing (UC folded) ─────────────────────");
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
                Sha256UairBpUnderconstrained, FoldedZt, FoldedLc, 32, 16, UNCHECKED, _, _,
            >(
                &folded_params, &folded_proof, SHA256_8X_NUM_VARS,
                |_: &IdealOrZero<Sha256Ideal>| zinc_snark::pipeline::TrivialIdeal,
                &sha_public_cols,
            );
            black_box(r);
        });
    });

    // ── 17. E2E Total Verifier (original, for comparison) ───────────
    let orig_proof = zinc_snark::pipeline::prove::<Sha256UairBpUnderconstrained, OrigZt, OrigLc, 32, UNCHECKED>(
        &orig_params, &sha_trace, SHA256_8X_NUM_VARS, &sha_lookup_specs,
    );

    {
        let r = zinc_snark::pipeline::verify::<Sha256UairBpUnderconstrained, OrigZt, OrigLc, 32, UNCHECKED, _, _>(
            &orig_params, &orig_proof, SHA256_8X_NUM_VARS,
            |_: &IdealOrZero<Sha256Ideal>| zinc_snark::pipeline::TrivialIdeal,
            &sha_public_cols,
        );
        let t = &r.timing;
        println!("\n── Verifier step timing (UC original) ───────────────────");
        println!("  IC verify:           {:>8.3} ms", t.ideal_check_verify.as_secs_f64() * 1000.0);
        println!("  CPR+Lookup verify:   {:>8.3} ms", t.combined_poly_resolver_verify.as_secs_f64() * 1000.0);
        println!("  Lookup verify:       {:>8.3} ms", t.lookup_verify.as_secs_f64() * 1000.0);
        println!("  PCS verify:          {:>8.3} ms", t.pcs_verify.as_secs_f64() * 1000.0);
        println!("  Total:               {:>8.3} ms", t.total.as_secs_f64() * 1000.0);
        println!("─────────────────────────────────────────────────────────\n");
    }

    group.bench_function("E2E/Verifier (original)", |b| {
        b.iter(|| {
            let r = zinc_snark::pipeline::verify::<Sha256UairBpUnderconstrained, OrigZt, OrigLc, 32, UNCHECKED, _, _>(
                &orig_params, &orig_proof, SHA256_8X_NUM_VARS,
                |_: &IdealOrZero<Sha256Ideal>| zinc_snark::pipeline::TrivialIdeal,
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
                    Sha256UairBpUnderconstrained, FoldedZt, FoldedLc, 32, 16, UNCHECKED,
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
        Sha256UairBpUnderconstrained, FoldedZt, FoldedLc, 32, 16, UNCHECKED,
    >(
        &folded_params, &sha_trace, SHA256_8X_NUM_VARS,
        &sha_lookup_specs, &sha_affine_specs,
    );

    {
        let r = zinc_snark::pipeline::verify_classic_logup_folded::<
            Sha256UairBpUnderconstrained, FoldedZt, FoldedLc, 32, 16, UNCHECKED, _, _,
        >(
            &folded_params, &gkr_folded_proof, SHA256_8X_NUM_VARS,
            |_: &IdealOrZero<Sha256Ideal>| zinc_snark::pipeline::TrivialIdeal,
            &sha_public_cols,
        );
        let t = &r.timing;
        println!("\n── Verifier step timing (UC GKR folded) ─────────────────");
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
                Sha256UairBpUnderconstrained, FoldedZt, FoldedLc, 32, 16, UNCHECKED, _, _,
            >(
                &folded_params, &gkr_folded_proof, SHA256_8X_NUM_VARS,
                |_: &IdealOrZero<Sha256Ideal>| zinc_snark::pipeline::TrivialIdeal,
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
                    Sha256UairBpUnderconstrained, Sha256UairQx, FoldedZt4x, FoldedLc4x, 32, 16, 8, UNCHECKED,
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
        Sha256UairBpUnderconstrained, Sha256UairQx, FoldedZt4x, FoldedLc4x, 32, 16, 8, UNCHECKED,
    >(
        &folded_4x_params, &sha_trace, SHA256_8X_NUM_VARS,
        &sha_lookup_specs, &sha_affine_specs,
    );

    {
        let r = zinc_snark::pipeline::verify_classic_logup_4x_folded::<
            Sha256UairBpUnderconstrained, Sha256UairQx, FoldedZt4x, FoldedLc4x, 32, 16, 8, UNCHECKED, _, _, _, _,
        >(
            &folded_4x_params, &folded_4x_proof, SHA256_8X_NUM_VARS,
            |_: &IdealOrZero<Sha256Ideal>| zinc_snark::pipeline::TrivialIdeal,
            |_| zinc_snark::pipeline::TrivialIdeal,
            &sha_public_cols,
        );
        let t = &r.timing;
        println!("\n── Verifier step timing (UC 4x folded) ──────────────────");
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
                Sha256UairBpUnderconstrained, Sha256UairQx, FoldedZt4x, FoldedLc4x, 32, 16, 8, UNCHECKED, _, _, _, _,
            >(
                &folded_4x_params, &folded_4x_proof, SHA256_8X_NUM_VARS,
                |_: &IdealOrZero<Sha256Ideal>| zinc_snark::pipeline::TrivialIdeal,
                |_| zinc_snark::pipeline::TrivialIdeal,
                &sha_public_cols,
            );
            black_box(r);
        });
    });

    // ── 22. E2E Total Prover (4x folded 4-chunk) ────────────────────
    // Uses chunk_width=8, width=32 → 4 chunks of 2^8 = 256 entries each.
    */
    let sha_lookup_specs_4c = uc_sha256_lookup_specs_4chunks();
    let sha_affine_specs_4c = uc_sha256_affine_lookup_specs_4chunks();
    /*

    let folded_4x_proof_4c = zinc_snark::pipeline::prove_classic_logup_4x_folded::<
        Sha256UairBpUnderconstrained, Sha256UairQx, FoldedZt4x, FoldedLc4x, 32, 16, 8, UNCHECKED,
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
                    Sha256UairBpUnderconstrained, Sha256UairQx, FoldedZt4x, FoldedLc4x, 32, 16, 8, UNCHECKED,
                >(
                    &folded_4x_params, &sha_trace, SHA256_8X_NUM_VARS,
                    &sha_lookup_specs_4c, &sha_affine_specs_4c,
                );
                total += t.elapsed();
            }
            total
        });
    });

    // ── 23. E2E Total Verifier (4x folded 4-chunk) ──────────────────
    {
        let r = zinc_snark::pipeline::verify_classic_logup_4x_folded::<
            Sha256UairBpUnderconstrained, Sha256UairQx, FoldedZt4x, FoldedLc4x, 32, 16, 8, UNCHECKED, _, _, _, _,
        >(
            &folded_4x_params, &folded_4x_proof_4c, SHA256_8X_NUM_VARS,
            |_: &IdealOrZero<Sha256Ideal>| zinc_snark::pipeline::TrivialIdeal,
            |_| zinc_snark::pipeline::TrivialIdeal,
            &sha_public_cols,
        );
        let t = &r.timing;
        println!("\n── Verifier step timing (UC 4x folded 4-chunk) ──────────");
        println!("  IC verify:           {:>8.3} ms", t.ideal_check_verify.as_secs_f64() * 1000.0);
        println!("  CPR+Lookup verify:   {:>8.3} ms", t.combined_poly_resolver_verify.as_secs_f64() * 1000.0);
        println!("  Lookup verify:       {:>8.3} ms", t.lookup_verify.as_secs_f64() * 1000.0);
        println!("  PCS verify:          {:>8.3} ms", t.pcs_verify.as_secs_f64() * 1000.0);
        println!("  Total:               {:>8.3} ms", t.total.as_secs_f64() * 1000.0);
        println!("─────────────────────────────────────────────────────────\n");
    }

    group.bench_function("E2E/Verifier (4x folded 4-chunk)", |b| {
        b.iter(|| {
            let r = zinc_snark::pipeline::verify_classic_logup_4x_folded::<
                Sha256UairBpUnderconstrained, Sha256UairQx, FoldedZt4x, FoldedLc4x, 32, 16, 8, UNCHECKED, _, _, _, _,
            >(
                &folded_4x_params, &folded_4x_proof_4c, SHA256_8X_NUM_VARS,
                |_: &IdealOrZero<Sha256Ideal>| zinc_snark::pipeline::TrivialIdeal,
                |_| zinc_snark::pipeline::TrivialIdeal,
                &sha_public_cols,
            );
            black_box(r);
        });
    });
    */ // end of commented-out classic logup / non-hybrid GKR benchmarks

    // ── 26. E2E Prover/Verifier (4x folded, 4-chunk, Hybrid GKR c=2) ─
    let hybrid_4x_proof = zinc_snark::pipeline::prove_hybrid_gkr_logup_4x_folded::<
        Sha256UairBpUnderconstrained, Sha256UairQx, FoldedZt4x, FoldedLc4x, 32, 16, 8, UNCHECKED,
    >(
        &folded_4x_params, &sha_trace, SHA256_8X_NUM_VARS,
        &sha_lookup_specs_4c, &sha_affine_specs_4c, 2,
    );

    group.bench_function("E2E/Prover (4x Hybrid GKR c=2 4-chunk)", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let t = Instant::now();
                let _ = zinc_snark::pipeline::prove_hybrid_gkr_logup_4x_folded::<
                    Sha256UairBpUnderconstrained, Sha256UairQx, FoldedZt4x, FoldedLc4x, 32, 16, 8, UNCHECKED,
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
            Sha256UairBpUnderconstrained, Sha256UairQx, FoldedZt4x, FoldedLc4x, 32, 16, 8, UNCHECKED, _, _, _, _,
        >(
            &folded_4x_params, &hybrid_4x_proof, SHA256_8X_NUM_VARS,
            |_: &IdealOrZero<Sha256Ideal>| zinc_snark::pipeline::TrivialIdeal,
            |_| zinc_snark::pipeline::TrivialIdeal,
            &sha_public_cols,
        );
        let t = &r.timing;
        println!("\n── Verifier step timing (UC 4x Hybrid GKR c=2 4-chunk) ──");
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
                Sha256UairBpUnderconstrained, Sha256UairQx, FoldedZt4x, FoldedLc4x, 32, 16, 8, UNCHECKED, _, _, _, _,
            >(
                &folded_4x_params, &hybrid_4x_proof, SHA256_8X_NUM_VARS,
                |_: &IdealOrZero<Sha256Ideal>| zinc_snark::pipeline::TrivialIdeal,
                |_| zinc_snark::pipeline::TrivialIdeal,
                &sha_public_cols,
            );
            black_box(r);
        });
    });

    // ── Timing breakdown + proof-size sections commented out:
    // ── they reference folded_proof / gkr_folded_proof / orig_proof
    // ── which are no longer generated.
    /*
    // ── Timing breakdown summary ────────────────────────────────────
    eprintln!("\n=== 8xSHA256 UC Folded Pipeline Timing (Classic Lookup) ===");
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
    eprintln!("  4x Hybrid GKR c=2 4-chunk pipeline timing:");
    eprintln!("  IC={:?}, CPR={:?}, Lookup={:?}, PCS(commit={:?}, prove={:?}), total={:?}",
        hybrid_4x_proof.timing.ideal_check,
        hybrid_4x_proof.timing.combined_poly_resolver,
        hybrid_4x_proof.timing.lookup,
        hybrid_4x_proof.timing.pcs_commit,
        hybrid_4x_proof.timing.pcs_prove,
        hybrid_4x_proof.timing.total,
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

    // ── Proof size breakdown ────────────────────────────────────────
    //
    // Compute raw + compressed proof size for each configuration:
    //   2x folded (8-chunk default), 4x folded (8-chunk), 4x folded (4-chunk),
    //   and the original (non-folded) pipeline for comparison.
    {
        use zinc_snark::pipeline::{FIELD_LIMBS, LookupProofData, PiopField};
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
                Some(LookupProofData::Gkr(gp)) => {
                    let mut t = 0usize;
                    for group in &gp.group_proofs {
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
            } else if let Some(LookupProofData::Gkr(gp)) = &proof.lookup_proof {
                for group in &gp.group_proofs {
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

        // ── 2x folded, 8-chunk (default) ──────────────────────────────
        let (raw_2x_8c, compr_2x_8c) = proof_size_2x(&folded_proof, "UC 2x Folded 8-chunk (8×2^4)", fe_bytes);

        // ── GKR folded, 8-chunk ───────────────────────────────────────
        let (raw_gkr_2x, compr_gkr_2x) = proof_size_2x(&gkr_folded_proof, "UC GKR Folded 8-chunk (8×2^4)", fe_bytes);

        // ── 4x folded, 8-chunk (default) ──────────────────────────────
        let (raw_4x_8c, compr_4x_8c) = proof_size_4x(&folded_4x_proof, "UC 4x Folded 8-chunk (8×2^4)", fe_bytes);

        // ── 4x folded, 4-chunk ────────────────────────────────────────
        let (raw_4x_4c, compr_4x_4c) = proof_size_4x(&folded_4x_proof_4c, "UC 4x Folded 4-chunk (4×2^8)", fe_bytes);

        // ── 4x Hybrid GKR c=2, 4-chunk ──────────────────────────────
        let (raw_hybrid_4x, compr_hybrid_4x) = proof_size_4x(&hybrid_4x_proof, "UC 4x Hybrid GKR c=2 4-chunk", fe_bytes);

        // ── Original (non-folded) ─────────────────────────────────────
        let orig_total_raw = {
            let pcs = orig_proof.pcs_proof_bytes.len();
            let ic: usize = orig_proof.ic_proof_values.iter().map(|v| v.len()).sum();
            let cpr_sc: usize = orig_proof.cpr_sumcheck_messages.iter().map(|v| v.len()).sum::<usize>()
                + orig_proof.cpr_sumcheck_claimed_sum.len();
            let cpr_up: usize = orig_proof.cpr_up_evals.iter().map(|v| v.len()).sum();
            let cpr_dn: usize = orig_proof.cpr_down_evals.iter().map(|v| v.len()).sum();
            let lk: usize = match &orig_proof.lookup_proof {
                Some(LookupProofData::Classic(proof)) => {
                    let mut t = 0usize;
                    for gp in &proof.group_proofs {
                        let m: usize = gp.aggregated_multiplicities.iter().map(|v| v.len()).sum();
                        let w: usize = gp.chunk_inverse_witnesses.iter().flat_map(|o| o.iter()).map(|i| i.len()).sum();
                        t += (m + w + gp.inverse_table.len()) * fe_bytes;
                    }
                    t
                }
                _ => 0,
            };
            let shift: usize = orig_proof.shift_sumcheck.as_ref().map_or(0, |sc| {
                sc.rounds.iter().map(|v| v.len()).sum::<usize>()
                    + sc.v_finals.iter().map(|v| v.len()).sum::<usize>()
            });
            let eval_pt: usize = orig_proof.evaluation_point_bytes.iter().map(|v| v.len()).sum();
            let pcs_eval: usize = orig_proof.pcs_evals_bytes.iter().map(|v| v.len()).sum();
            pcs + ic + cpr_sc + cpr_up + cpr_dn + lk + shift + eval_pt + pcs_eval
        };
        let orig_compressed_len = {
            let mut buf = Vec::new();
            buf.extend(&orig_proof.pcs_proof_bytes);
            for v in &orig_proof.ic_proof_values { buf.extend(v); }
            for v in &orig_proof.cpr_sumcheck_messages { buf.extend(v); }
            buf.extend(&orig_proof.cpr_sumcheck_claimed_sum);
            for v in &orig_proof.cpr_up_evals { buf.extend(v); }
            for v in &orig_proof.cpr_down_evals { buf.extend(v); }
            if let Some(LookupProofData::Classic(proof)) = &orig_proof.lookup_proof {
                for gp in &proof.group_proofs {
                    for v in &gp.aggregated_multiplicities {
                        for f in v { write_fe_cv(&mut buf, f); }
                    }
                    for outer in &gp.chunk_inverse_witnesses {
                        for inner in outer { for f in inner { write_fe_cv(&mut buf, f); } }
                    }
                    for f in &gp.inverse_table { write_fe_cv(&mut buf, f); }
                }
            }
            if let Some(ref sc) = orig_proof.shift_sumcheck {
                for v in &sc.rounds { buf.extend(v); }
                for v in &sc.v_finals { buf.extend(v); }
            }
            for v in &orig_proof.evaluation_point_bytes { buf.extend(v); }
            for v in &orig_proof.pcs_evals_bytes { buf.extend(v); }
            use std::io::Write;
            let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
            enc.write_all(&buf).unwrap();
            enc.finish().unwrap().len()
        };

        eprintln!("\n=== UC Proof Size Comparison ===");
        eprintln!("  {:35}  {:>8}  {:>8}  {:>8}", "Configuration", "Raw (B)", "Raw (KB)", "Compr (KB)");
        eprintln!("  {}", "─".repeat(70));
        eprintln!("  {:35}  {:>8}  {:>8.1}  {:>8.1}", "Original (non-folded)",
            orig_total_raw, orig_total_raw as f64 / 1024.0, orig_compressed_len as f64 / 1024.0);
        eprintln!("  {:35}  {:>8}  {:>8.1}  {:>8.1}", "2x folded, 8-chunk (8×2^4) *",
            raw_2x_8c, raw_2x_8c as f64 / 1024.0, compr_2x_8c as f64 / 1024.0);
        eprintln!("  {:35}  {:>8}  {:>8.1}  {:>8.1}", "GKR folded, 8-chunk (8×2^4)",
            raw_gkr_2x, raw_gkr_2x as f64 / 1024.0, compr_gkr_2x as f64 / 1024.0);
        eprintln!("  {:35}  {:>8}  {:>8.1}  {:>8.1}", "4x folded, 8-chunk (8×2^4)",
            raw_4x_8c, raw_4x_8c as f64 / 1024.0, compr_4x_8c as f64 / 1024.0);
        eprintln!("  {:35}  {:>8}  {:>8.1}  {:>8.1}", "4x folded, 4-chunk (4×2^8)",
            raw_4x_4c, raw_4x_4c as f64 / 1024.0, compr_4x_4c as f64 / 1024.0);
        eprintln!("  {:35}  {:>8}  {:>8.1}  {:>8.1}", "4x Hybrid GKR c=2, 4-chunk",
            raw_hybrid_4x, raw_hybrid_4x as f64 / 1024.0, compr_hybrid_4x as f64 / 1024.0);
        eprintln!("  (* = default configuration)");
        let best_raw = raw_4x_4c.min(raw_4x_8c).min(raw_2x_8c).min(raw_gkr_2x).min(raw_hybrid_4x);
        let best_compr = compr_4x_4c.min(compr_4x_8c).min(compr_2x_8c).min(compr_gkr_2x).min(compr_hybrid_4x);
        let savings_raw = orig_total_raw as i64 - best_raw as i64;
        let savings_compr = orig_compressed_len as i64 - best_compr as i64;
        eprintln!("  Best raw savings vs original:     {:>+6} B ({:+.1} KB, {:.2}x)",
            savings_raw, savings_raw as f64 / 1024.0, orig_total_raw as f64 / best_raw as f64);
        eprintln!("  Best compr savings vs original:   {:>+6} B ({:+.1} KB, {:.2}x)",
            savings_compr, savings_compr as f64 / 1024.0, orig_compressed_len as f64 / best_compr as f64);
    }
    */ // end of commented-out timing / proof-size sections

    // ── Hybrid GKR timing ───────────────────────────────────────────
    eprintln!("\n=== 8xSHA256 UC 4x Hybrid GKR c=2 Pipeline Timing ===");
    eprintln!("  IC={:?}, CPR={:?}, Lookup={:?}, PCS(commit={:?}, prove={:?}), total={:?}",
        hybrid_4x_proof.timing.ideal_check,
        hybrid_4x_proof.timing.combined_poly_resolver,
        hybrid_4x_proof.timing.lookup,
        hybrid_4x_proof.timing.pcs_commit,
        hybrid_4x_proof.timing.pcs_prove,
        hybrid_4x_proof.timing.total,
    );

    let mem_snapshot = mem_tracker.stop();
    eprintln!("\n=== Peak Memory ===");
    eprintln!("  {mem_snapshot}");

    group.finish();
}

criterion_group!(benches, uc_sha256_8x_folded_stepwise);
criterion_main!(benches);
