//! Per-step breakdown benchmarks for the 8x SHA-256 proving stack with
//! **4x-folded** BinaryPoly<32> -> BinaryPoly<16> -> BinaryPoly<8> columns,
//! using the Hybrid GKR c=2 lookup protocol with 4-chunk decomposition.
//!
//! This benchmark uses the 4x-folded pipeline
//! (`prove_hybrid_gkr_logup_4x_folded` / `verify_classic_logup_4x_folded`)
//! which double-splits columns and uses GKR-based lookup verification.
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
            PnttConfigF2_16R4B32,
        },
    },
    pcs::structs::{ZipPlus, ZipPlusParams, ZipTypes},
    pcs::folding::{split_columns, fold_claims_prove},
};
#[allow(unused_imports)]
use zinc_sha256_uair::witness::GenerateWitness;
use zinc_uair::Uair;

// ─── Feature-gated UAIR type selection ──────────────────────────────────────

#[cfg(not(feature = "no-f2x"))]
use zinc_sha256_uair::{Sha256Uair, Sha256UairQx};

#[cfg(not(feature = "no-f2x"))]
type BenchBpUair = Sha256Uair;
#[cfg(not(feature = "no-f2x"))]
type BenchQxUair = Sha256UairQx;
#[cfg(not(feature = "no-f2x"))]
const SHA256_BATCH_SIZE: usize = 30; // 27 bitpoly + 3 int

#[cfg(feature = "no-f2x")]
use zinc_sha256_uair::no_f2x::{
    Sha256UairBpNoF2x, Sha256UairQxNoF2x,
    Sha256QxNoF2xIdeal,
    NO_F2X_NUM_COLS,
};
#[cfg(all(feature = "no-f2x", feature = "true-ideal"))]
use zinc_sha256_uair::no_f2x::Sha256QxNoF2xIdealOverF;

#[cfg(feature = "no-f2x")]
type BenchBpUair = Sha256UairBpNoF2x;
#[cfg(feature = "no-f2x")]
type BenchQxUair = Sha256UairQxNoF2x;
#[cfg(feature = "no-f2x")]
const SHA256_BATCH_SIZE: usize = NO_F2X_NUM_COLS;
use zinc_piop::projections::{
    project_trace_to_field,
    project_scalars, project_scalars_to_field,
};
use zinc_piop::lookup::{AffineLookupSpec, LookupColumnSpec, LookupTableType};

use zinc_piop::ideal_check::IdealCheckProtocol;
use zinc_piop::combined_poly_resolver::CombinedPolyResolver;
use zinc_piop::shift_sumcheck::{shift_sumcheck_prove, ShiftClaim};

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

/// 4x-Folded PCS types — BinaryPoly<8> codewords.
/// Uses PnttConfigF2_16R4B32<2> (BASE_LEN=32, BASE_DIM=128, DEPTH=2):
///   INPUT_LEN = 32 × 64 = 2048 rows,
///   OUTPUT_LEN = 128 × 64 = 8192 (rate 1/4).
/// Field: F65537 (2^16+1, TWO_ADICITY=16), which has small twiddles (max 4096)
/// giving max codeword coefficient ~2^50, vs ~2^62 at depth 3.
#[cfg(not(feature = "full-fold"))]
type FoldedZt4x = Sha256ZipTypes<i64, 8>;
#[cfg(not(feature = "full-fold"))]
type FoldedLc4x = IprsCode<
    FoldedZt4x,
    PnttConfigF2_16R4B32<2>,
    BinaryPolyWideningMulByScalar<i64>,
    UNCHECKED,
>;

/// 32x Full-folded PCS types — BinaryPoly<1> codewords.
/// Uses PnttConfigF2_16R4B32<3> (BASE_LEN=32, DEPTH=3):
///   INPUT_LEN = 32 × 512 = 16384 rows,
///   OUTPUT_LEN = 128 × 512 = 65536 (rate 1/4).
#[cfg(feature = "full-fold")]
type FoldedZt4x = Sha256ZipTypes<i64, 1>;
#[cfg(feature = "full-fold")]
type FoldedLc4x = IprsCode<
    FoldedZt4x,
    PnttConfigF2_16R4B32<3>,
    BinaryPolyWideningMulByScalar<i64>,
    UNCHECKED,
>;

// ─── Parameters ─────────────────────────────────────────────────────────────

const SHA256_8X_NUM_VARS: usize = 9;      // 2^9 = 512 rows (8 × 64 SHA rounds)
// SHA256_BATCH_SIZE is defined above based on the no-f2x feature flag.
const SHA256_LOOKUP_COL_COUNT: usize = 10; // 10 Q[X] columns need lookup

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

fn generate_sha256_trace(num_vars: usize) -> Vec<DenseMultilinearExtension<BinaryPoly<32>>> {
    let mut rng = rand::rng();
    #[cfg(not(feature = "no-f2x"))]
    {
        <zinc_sha256_uair::Sha256Uair as GenerateWitness<BinaryPoly<32>>>::generate_witness(num_vars, &mut rng)
    }
    #[cfg(feature = "no-f2x")]
    {
        zinc_sha256_uair::no_f2x::generate_no_f2x_witness(num_vars, &mut rng)
    }
}

// ─── Benchmark ────────────────────────────────────────────────────────────

/// Measures each main step of the **4x-folded Hybrid GKR c=2** E2E proving
/// stack individually for 8x SHA-256 compressions (**no** ECDSA).
///
/// Uses the 4x-folded pipeline with 4-chunk lookups (chunk_width=8).
///
/// ## E2E prover pipeline (`prove_hybrid_gkr_logup_4x_folded`)
///
/// The E2E pipeline executes the following steps in order:
///   0.  Double-split columns: BinaryPoly<32> -> BinaryPoly<16> -> BinaryPoly<8>
///   1.  PCS/Commit (4x-folded) over BinaryPoly<8>
///   2.  Transcript init + absorb PCS commitment root + derive random field config
///   2b. IC (BinaryPoly, Sha256Uair) — MLE-first path for max_degree==1
///   2c. QX IC (Sha256UairQx, trivial ideal) — prove_from_binary_poly_at_point
///   3.  Draw projecting element α; project scalars + trace to field
///       Pre-extract lookup columns + append affine virtual columns (13 total)
///       Extract shift trace source columns
///   3a. CPR (Sha256Uair) — standalone MLSumcheck, degree 3
///   3b. QX CPR (Sha256UairQx) — standalone MLSumcheck, degree 3
///   3c. Hybrid GKR batched lookup (cutoff=2) on 13 columns (10 base + 3 affine)
///   3d. Shift sumcheck — reduce shifted-column claims to unshifted MLE claims
///   4.  Two-round folding: fold_claims_prove D→HALF_D then HALF_D→QUARTER_D
///   5.  PCS/Prove at (r ‖ γ₁ ‖ γ₂) over BinaryPoly<8> columns
///
/// ## Individual bench steps
///
/// The following steps are benchmarked individually. They exercise the same
/// algorithms as the E2E pipeline but with independent transcript state
/// (see fidelity notes in the documentation).
///
///   1.  WitnessGen -- generate the 30-column BinaryPoly<32> trace (512 rows)
///   2.  Folding/SplitColumns -- double split BinaryPoly<32> -> BinaryPoly<16> -> BinaryPoly<8>
///   3.  PCS/Commit (4x-folded) -- Zip+ commit over BinaryPoly<8> split columns
///   4.  PIOP/FieldSetup -- transcript init + random field config (no root absorption)
///   5.  PIOP/Project Ideal Check -- project_scalars for Ideal Check
///   6.  PIOP/IdealCheck -- Ideal Check prover (MLE-first, on original trace)
///   7.  PIOP/Project Main field sumcheck -- project_scalars_to_field + project_trace_to_field
///   8.  PIOP/Main field sumcheck -- Combined Poly Resolver prover
///   9.  PIOP/LookupExtract -- extract lookup columns from field trace (10 base only)
///  10.  PIOP/Lookup -- classic batched decomposed LogUp prover (10 columns, 4-chunk)
///  11.  PIOP/ShiftSumcheck -- shift sumcheck prover
///  12.  Folding/FoldClaims (2-round) -- two-round column folding protocol
///  13.  PCS/Prove (4x-folded) -- Zip+ prove over BinaryPoly<8> split columns
///  14.  E2E/Prover -- total (prove_hybrid_gkr_logup_4x_folded, Hybrid GKR c=2)
///  15.  E2E/Verifier -- total (verify_classic_logup_4x_folded)
///
/// ## Notable differences between individual steps and E2E
///
/// - Step 4: The E2E absorbs the PCS commitment root before deriving the
///   field config; the individual step does not. This means all Fiat-Shamir
///   challenges diverge from the E2E (but timing is statistically equivalent).
/// - Step 10: The individual step uses **classic** batched LogUp on 10 base
///   columns; the E2E uses **Hybrid GKR** LogUp on 13 columns (10 base + 3
///   affine). The classic step is kept as a comparison baseline.
/// - Steps 9-10: The individual steps do NOT include affine virtual column
///   construction (`append_affine_virtual_columns`) — only the 10 base
///   lookups are extracted.
/// - Step 13: The individual step uses `prove()` (no seed); the E2E uses
///   `prove_with_seed(&root_buf)` which absorbs the root into the PCS
///   transcript.
/// - No individual steps exist for: QX IC, QX CPR, or affine column
///   construction. These are only measured via the E2E prover (step 14).
///
/// Also reports:
///  - Proof size breakdown (raw + compressed)
///  - Peak memory usage
fn sha256_8x_folded_stepwise(c: &mut Criterion) {
    use zinc_sha256_uair::CyclotomicIdeal;
    use zinc_uair::ideal_collector::IdealOrZero;
    use zinc_piop::lookup::prove_batched_lookup_with_indices;

    let mem_tracker = MemoryTracker::start();

    #[cfg(not(feature = "full-fold"))]
    let fold_label = "4x-Folded";
    #[cfg(feature = "full-fold")]
    let fold_label = "32x Full-Folded";

    let mut group = c.benchmark_group(format!(
        "8xSHA256 {} Hybrid GKR c=2 Steps (grind={})",
        fold_label,
        <FoldedZt4x as ZipTypes>::GRINDING_BITS,
    ));
    group.sample_size(100);

    // -- Folded PCS params -------------------------------------------
    #[cfg(not(feature = "full-fold"))]
    let folded_extra_vars: usize = 2; // Two splits -> num_vars + 2
    #[cfg(feature = "full-fold")]
    let folded_extra_vars: usize = 5; // Five splits -> num_vars + 5

    let folded_num_vars = SHA256_8X_NUM_VARS + folded_extra_vars;
    let folded_row_len = 1usize << folded_num_vars;
    let folded_lc = FoldedLc4x::new(folded_row_len);
    let folded_params = ZipPlusParams::<FoldedZt4x, FoldedLc4x>::new(
        folded_num_vars, 1, folded_lc,
    );

    let sha_lookup_specs = sha256_lookup_specs_4chunks();
    let sha_affine_specs = sha256_affine_lookup_specs_4chunks();

    let num_constraints = zinc_uair::constraint_counter::count_constraints::<BenchBpUair>();
    let max_degree = zinc_uair::degree_counter::count_max_degree::<BenchBpUair>();

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
    let sha_sig = BenchBpUair::signature();
    let sha_excluded = sha_sig.pcs_excluded_columns();
    let sha_pcs_trace: Vec<_> = sha_trace.iter().enumerate()
        .filter(|(i, _)| !sha_excluded.contains(i))
        .map(|(_, c)| c.clone()).collect();

    // ── 2. Folding / Split Columns ───────────────────────────────────
    #[cfg(not(feature = "full-fold"))]
    group.bench_function("Folding/SplitColumns", |b| {
        b.iter(|| {
            let half = split_columns::<32, 16>(&sha_pcs_trace);
            let quarter = split_columns::<16, 8>(&half);
            black_box(quarter);
        });
    });

    #[cfg(feature = "full-fold")]
    group.bench_function("Folding/SplitColumns (5-round)", |b| {
        b.iter(|| {
            let s16 = split_columns::<32, 16>(&sha_pcs_trace);
            let s8 = split_columns::<16, 8>(&s16);
            let s4 = split_columns::<8, 4>(&s8);
            let s2 = split_columns::<4, 2>(&s4);
            let s1 = split_columns::<2, 1>(&s2);
            black_box(s1);
        });
    });

    #[cfg(not(feature = "full-fold"))]
    let half_trace: Vec<DenseMultilinearExtension<BinaryPoly<16>>> =
        split_columns::<32, 16>(&sha_pcs_trace);
    #[cfg(not(feature = "full-fold"))]
    let split_trace: Vec<DenseMultilinearExtension<BinaryPoly<8>>> =
        split_columns::<16, 8>(&half_trace);

    #[cfg(feature = "full-fold")]
    let split16: Vec<DenseMultilinearExtension<BinaryPoly<16>>> =
        split_columns::<32, 16>(&sha_pcs_trace);
    #[cfg(feature = "full-fold")]
    let split8: Vec<DenseMultilinearExtension<BinaryPoly<8>>> =
        split_columns::<16, 8>(&split16);
    #[cfg(feature = "full-fold")]
    let split4: Vec<DenseMultilinearExtension<BinaryPoly<4>>> =
        split_columns::<8, 4>(&split8);
    #[cfg(feature = "full-fold")]
    let split2: Vec<DenseMultilinearExtension<BinaryPoly<2>>> =
        split_columns::<4, 2>(&split4);
    #[cfg(feature = "full-fold")]
    let split1: Vec<DenseMultilinearExtension<BinaryPoly<1>>> =
        split_columns::<2, 1>(&split2);

    // ── 3. PCS Commit ───────────────────────────────────────────────
    #[cfg(not(feature = "full-fold"))]
    group.bench_function("PCS/Commit (4x-folded)", |b| {
        b.iter(|| {
            let r = ZipPlus::<FoldedZt4x, FoldedLc4x>::commit(&folded_params, &split_trace);
            let _ = black_box(r);
        });
    });

    #[cfg(feature = "full-fold")]
    group.bench_function("PCS/Commit (32x full-folded)", |b| {
        b.iter(|| {
            let r = ZipPlus::<FoldedZt4x, FoldedLc4x>::commit(&folded_params, &split1);
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
    assert!(max_degree >= 1 && max_degree <= 2, "SHA-256 UAIR max_degree should be 1 or 2, got {max_degree}");
    group.bench_function("PIOP/Project Ideal Check", |b| {
        let mut tr_setup = zinc_transcript::KeccakTranscript::new();
        let fcfg = tr_setup.get_random_field_cfg::<F, <F as Field>::Inner, MillerRabin>();
        b.iter(|| {
            let projected_scalars = project_scalars::<F, BenchBpUair>(|scalar| {
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
                let projected_scalars = project_scalars::<F, BenchBpUair>(|scalar| {
                    let one = F::one_with_cfg(&field_cfg);
                    let zero = F::zero_with_cfg(&field_cfg);
                    DynamicPolynomialF::new(
                        scalar.iter().map(|coeff| {
                            if coeff.into_inner() { one.clone() } else { zero.clone() }
                        }).collect::<Vec<_>>()
                    )
                });
                let t = Instant::now();
                let _ = IdealCheckProtocol::<F>::prove_mle_first::<BenchBpUair, 32>(
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

    // ── 7. PIOP / Project Trace for Main field sumcheck ─────────────────────────────
    let mut transcript_for_cpr = zinc_transcript::KeccakTranscript::new();
    let field_cfg_cpr = transcript_for_cpr.get_random_field_cfg::<
        F, <F as Field>::Inner, MillerRabin
    >();
    let projected_scalars_cpr = project_scalars::<F, BenchBpUair>(|scalar| {
        let one = F::one_with_cfg(&field_cfg_cpr);
        let zero = F::zero_with_cfg(&field_cfg_cpr);
        DynamicPolynomialF::new(
            scalar.iter().map(|coeff| {
                if coeff.into_inner() { one.clone() } else { zero.clone() }
            }).collect::<Vec<_>>()
        )
    });
    let (_ic_proof_cpr, ic_state_cpr) =
        IdealCheckProtocol::<F>::prove_mle_first::<BenchBpUair, 32>(
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

    // ── 8. PIOP / Combined Poly Resolver ────────────────────────────
    group.bench_function("PIOP/Main field sumcheck", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut tr = transcript_for_cpr.clone();
                let t = Instant::now();
                let _ = CombinedPolyResolver::<F>::prove_as_subprotocol::<BenchBpUair>(
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

    // ── 9. PIOP / Lookup Extract ───────────────────────────────────
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

    // ── 10. PIOP / Lookup (classic) ─────────────────────────────────
    {
        let mut transcript_lk = transcript_for_cpr.clone();
        let _ = CombinedPolyResolver::<F>::prove_as_subprotocol::<BenchBpUair>(
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

    // ── 11. PIOP / Shift Sumcheck ──────────────────────────────────
    //
    // Benchmarks the shift sumcheck prover which reduces shifted-column
    // evaluation claims from the CPR down_evals to plain MLE claims on
    // the unshifted source columns at a new random point.
    {
        // Re-run CPR to get the state we need for the shift sumcheck.
        let mut transcript_ss = transcript_for_cpr.clone();
        let (cpr_proof_ss, cpr_state_ss) =
            CombinedPolyResolver::<F>::prove_as_subprotocol::<BenchBpUair>(
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
        let sha_sig_ss = BenchBpUair::signature();
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

    // ── 12. Folding Protocol (two-round fold: D→HALF_D→QUARTER_D) ──
    //
    // Benchmarks the two-round column folding (fold_claims_prove × 2)
    // that reduces BinaryPoly<32> commitment claims to BinaryPoly<8>.
    {
        let mut transcript_fold = transcript_for_cpr.clone();
        let (_cpr_proof_fold, cpr_state_fold) =
            CombinedPolyResolver::<F>::prove_as_subprotocol::<BenchBpUair>(
                &mut transcript_fold,
                field_trace_cpr.clone(),
                &ic_state_cpr.evaluation_point,
                &field_projected_scalars_cpr,
                num_constraints,
                SHA256_8X_NUM_VARS,
                max_degree,
                &field_cfg_cpr,
            )
            .expect("CPR prove failed for folding bench");
        let fold_piop_point = &cpr_state_fold.evaluation_point[..SHA256_8X_NUM_VARS];

        #[cfg(not(feature = "full-fold"))]
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

        #[cfg(feature = "full-fold")]
        group.bench_function("Folding/FoldClaims (5-round)", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let mut tr = transcript_fold.clone();
                    let t = Instant::now();
                    let f1 = fold_claims_prove::<F, _, 16>(
                        &mut tr, &split16, fold_piop_point,
                        &projecting_elem_cpr, &field_cfg_cpr,
                    ).expect("fold1 failed");
                    let f2 = fold_claims_prove::<F, _, 8>(
                        &mut tr, &split8, &f1.new_point,
                        &projecting_elem_cpr, &field_cfg_cpr,
                    ).expect("fold2 failed");
                    let f3 = fold_claims_prove::<F, _, 4>(
                        &mut tr, &split4, &f2.new_point,
                        &projecting_elem_cpr, &field_cfg_cpr,
                    ).expect("fold3 failed");
                    let f4 = fold_claims_prove::<F, _, 2>(
                        &mut tr, &split2, &f3.new_point,
                        &projecting_elem_cpr, &field_cfg_cpr,
                    ).expect("fold4 failed");
                    let _f5 = fold_claims_prove::<F, _, 1>(
                        &mut tr, &split1, &f4.new_point,
                        &projecting_elem_cpr, &field_cfg_cpr,
                    ).expect("fold5 failed");
                    total += t.elapsed();
                }
                total
            });
        });
    }

    // ── 13. PCS Prove ──────────────────────────────────────────────
    let folded_pcs_point: Vec<F> = {
        let mut tr = transcript_for_cpr.clone();
        let (_, cpr_state) = CombinedPolyResolver::<F>::prove_as_subprotocol::<BenchBpUair>(
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
        for _ in 0..folded_extra_vars {
            pt.push(F::one_with_cfg(&field_cfg_cpr)); // placeholder γ
        }
        pt
    };

    #[cfg(not(feature = "full-fold"))]
    let (folded_hint, _) = ZipPlus::<FoldedZt4x, FoldedLc4x>::commit(&folded_params, &split_trace)
        .expect("commit");
    #[cfg(feature = "full-fold")]
    let (folded_hint, _) = ZipPlus::<FoldedZt4x, FoldedLc4x>::commit(&folded_params, &split1)
        .expect("commit");

    // ── 12b. Lifting: Ring-valued MLE evaluations ─────────────────
    #[cfg(not(feature = "full-fold"))]
    group.bench_function("Lifting/RingEvals (4x-folded)", |b| {
        b.iter(|| {
            let r = zinc_snark::pipeline::compute_ring_evals::<8>(
                &split_trace, &folded_pcs_point, &field_cfg_cpr,
            );
            black_box(r);
        });
    });

    #[cfg(feature = "full-fold")]
    group.bench_function("Lifting/RingEvals (32x full-folded)", |b| {
        b.iter(|| {
            let r = zinc_snark::pipeline::compute_ring_evals::<1>(
                &split1, &folded_pcs_point, &field_cfg_cpr,
            );
            black_box(r);
        });
    });

    #[cfg(not(feature = "full-fold"))]
    group.bench_function("PCS/Prove (4x-folded)", |b| {
        b.iter(|| {
            let r = ZipPlus::<FoldedZt4x, FoldedLc4x>::prove::<F, UNCHECKED>(
                &folded_params, &split_trace, &folded_pcs_point, &folded_hint,
            );
            let _ = black_box(r);
        });
    });

    #[cfg(feature = "full-fold")]
    group.bench_function("PCS/Prove (32x full-folded)", |b| {
        b.iter(|| {
            let r = ZipPlus::<FoldedZt4x, FoldedLc4x>::prove::<F, UNCHECKED>(
                &folded_params, &split1, &folded_pcs_point, &folded_hint,
            );
            let _ = black_box(r);
        });
    });

    // ── 14. E2E Total Prover ─────────────────────────────────────────

    // Shared: public columns for verifier
    let sha_sig_pub = BenchBpUair::signature();
    let sha_public_cols: Vec<_> = sha_sig_pub.public_columns.iter()
        .map(|&i| sha_trace[i].clone()).collect();

    #[cfg(feature = "boundary")]
    let bdry_public_cols: Vec<_> = {
        let bdry_trace = zinc_sha256_uair::boundary::generate_boundary_witness::<32>(&sha_trace, SHA256_8X_NUM_VARS);
        let bdry_sig = zinc_sha256_uair::boundary::Sha256UairBoundaryNoF2x::signature();
        bdry_sig.public_columns.iter()
            .map(|&i| bdry_trace[i].clone()).collect()
    };
    #[cfg(not(feature = "boundary"))]
    let bdry_public_cols: Vec<zinc_poly::mle::DenseMultilinearExtension<BinaryPoly<32>>> = vec![];

    #[cfg(not(feature = "full-fold"))]
    {
    let hybrid_4x_proof = zinc_snark::pipeline::prove_hybrid_gkr_logup_4x_folded::<
        BenchBpUair, BenchQxUair, FoldedZt4x, FoldedLc4x, 32, 16, 8, UNCHECKED,
    >(
        &folded_params, &sha_trace, SHA256_8X_NUM_VARS,
        &sha_lookup_specs, &sha_affine_specs, 2,
    );

    // Print prover pipeline timing breakdown.
    {
        let t = &hybrid_4x_proof.timing;
        eprintln!("\n── Hybrid 4x GKR c=2 Prover Pipeline Timing ──────────");
        eprintln!("  Split columns: {:>8.3} ms", t.split_columns.as_secs_f64() * 1000.0);
        eprintln!("  PCS commit:    {:>8.3} ms", t.pcs_commit.as_secs_f64() * 1000.0);
        eprintln!("  Ideal Check:   {:>8.3} ms  (QX IC: {:.3} ms)", t.ideal_check.as_secs_f64() * 1000.0, t.qx_ideal_check.as_secs_f64() * 1000.0);
        eprintln!("  Field proj:    {:>8.3} ms", t.field_projection.as_secs_f64() * 1000.0);
        eprintln!("  Lookup extract:{:>8.3} ms", t.lookup_extract.as_secs_f64() * 1000.0);
        eprintln!("  Col eval:      {:>8.3} ms", (t.combined_poly_resolver - t.field_projection - t.lookup_extract).as_secs_f64() * 1000.0);
        eprintln!("  Proj+Eval:     {:>8.3} ms", t.combined_poly_resolver.as_secs_f64() * 1000.0);
        eprintln!("  Lookup:        {:>8.3} ms", t.lookup.as_secs_f64() * 1000.0);
        eprintln!("  Shift SC:      {:>8.3} ms", t.shift_sumcheck.as_secs_f64() * 1000.0);
        eprintln!("  Folding:       {:>8.3} ms", t.folding.as_secs_f64() * 1000.0);
        eprintln!("  Lifting:       {:>8.3} ms", t.lifting.as_secs_f64() * 1000.0);
        eprintln!("  PCS prove:     {:>8.3} ms", t.pcs_prove.as_secs_f64() * 1000.0);
        eprintln!("  Total:         {:>8.3} ms", t.total.as_secs_f64() * 1000.0);
        let accounted = t.split_columns + t.pcs_commit + t.ideal_check
            + t.combined_poly_resolver + t.lookup + t.shift_sumcheck + t.folding + t.lifting + t.pcs_prove;
        let unaccounted = t.total.saturating_sub(accounted);
        eprintln!("  Unaccounted:   {:>8.3} ms (serialize only)", unaccounted.as_secs_f64() * 1000.0);
        eprintln!("────────────────────────────────────────────────────────\n");
    }

    // ── Proof size breakdown ────────────────────────────────────────
    {
        use zinc_snark::pipeline::{FIELD_LIMBS, LookupProofData};
        let fe_bytes = <Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;

        fn write_fe(buf: &mut Vec<u8>, f: &zinc_snark::pipeline::PiopField) {
            use zinc_snark::pipeline::FIELD_LIMBS;
            let sz = <Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;
            let start = buf.len();
            buf.resize(start + sz, 0);
            f.inner().write_transcription_bytes(&mut buf[start..]);
        }

        let pcs_bytes = hybrid_4x_proof.pcs_proof_bytes.len();
        let ic_bytes: usize = hybrid_4x_proof.ic_proof_values.iter().map(|v| v.len()).sum();
        let qx_ic_bytes: usize = hybrid_4x_proof.qx_ic_proof_values.iter().map(|v| v.len()).sum();
        let cpr_msg_bytes: usize = hybrid_4x_proof.cpr_sumcheck_messages.iter().map(|v| v.len()).sum();
        let cpr_sum_bytes = hybrid_4x_proof.cpr_sumcheck_claimed_sum.len();
        let cpr_sc_total = cpr_msg_bytes + cpr_sum_bytes;
        let cpr_up: usize = hybrid_4x_proof.cpr_up_evals.iter().map(|v| v.len()).sum();
        let cpr_dn: usize = hybrid_4x_proof.cpr_down_evals.iter().map(|v| v.len()).sum();
        let cpr_eval_total = cpr_up + cpr_dn;
        let qx_cpr_msg: usize = hybrid_4x_proof.qx_cpr_sumcheck_messages.iter().map(|v| v.len()).sum();
        let qx_cpr_sum = hybrid_4x_proof.qx_cpr_sumcheck_claimed_sum.len();
        let qx_cpr_up: usize = hybrid_4x_proof.qx_cpr_up_evals.iter().map(|v| v.len()).sum();
        let qx_cpr_dn: usize = hybrid_4x_proof.qx_cpr_down_evals.iter().map(|v| v.len()).sum();
        let qx_cpr_total = qx_cpr_msg + qx_cpr_sum + qx_cpr_up + qx_cpr_dn;

        let lookup_bytes: usize = match &hybrid_4x_proof.lookup_proof {
            Some(LookupProofData::HybridGkr(proof)) => {
                let mut t = 0usize;
                for gp in &proof.group_proofs {
                    let m: usize = gp.aggregated_multiplicities.iter()
                        .map(|v| v.len()).sum::<usize>() * fe_bytes;
                    let w_roots = (gp.witness_gkr.roots_p.len()
                        + gp.witness_gkr.roots_q.len()) * fe_bytes;
                    let mut w_layers = 0usize;
                    for lp in &gp.witness_gkr.layer_proofs {
                        let sc_fe: usize = lp.sumcheck_proof.as_ref().map_or(0, |sc| {
                            sc.messages.iter().map(|msg| msg.0.tail_evaluations.len()).sum::<usize>() + 1
                        });
                        let child_fe = lp.p_lefts.len() + lp.p_rights.len()
                            + lp.q_lefts.len() + lp.q_rights.len();
                        w_layers += (sc_fe + child_fe) * fe_bytes;
                    }
                    let w_sent: usize = (gp.witness_gkr.sent_p.iter()
                        .chain(gp.witness_gkr.sent_q.iter())
                        .map(|v| v.len()).sum::<usize>()) * fe_bytes;
                    let t_root = 2 * fe_bytes;
                    let mut t_layers = 0usize;
                    for lp in &gp.table_gkr.layer_proofs {
                        let sc_fe: usize = lp.sumcheck_proof.as_ref().map_or(0, |sc| {
                            sc.messages.iter().map(|msg| msg.0.tail_evaluations.len()).sum::<usize>() + 1
                        });
                        let child_fe = 4;
                        t_layers += (sc_fe + child_fe) * fe_bytes;
                    }
                    t += m + w_roots + w_layers + w_sent + t_root + t_layers;
                }
                t
            }
            Some(LookupProofData::Classic(proof)) => {
                let mut t = 0usize;
                for gp in &proof.group_proofs {
                    let m: usize = gp.aggregated_multiplicities.iter().map(|v| v.len()).sum::<usize>();
                    let w: usize = gp.chunk_inverse_witnesses.iter()
                        .flat_map(|o| o.iter()).map(|i| i.len()).sum::<usize>();
                    t += (m + w + gp.inverse_table.len()) * fe_bytes;
                }
                t
            }
            _ => 0,
        };

        let shift_sc_bytes: usize = hybrid_4x_proof.shift_sumcheck.as_ref().map_or(0, |sc| {
            let rounds: usize = sc.rounds.iter().map(|v| v.len()).sum();
            let finals: usize = sc.v_finals.iter().map(|v| v.len()).sum();
            rounds + finals
        });
        let qx_shift_sc_bytes: usize = hybrid_4x_proof.qx_shift_sumcheck.as_ref().map_or(0, |sc| {
            let rounds: usize = sc.rounds.iter().map(|v| v.len()).sum();
            let finals: usize = sc.v_finals.iter().map(|v| v.len()).sum();
            rounds + finals
        });

        let eval_pt_bytes: usize = hybrid_4x_proof.evaluation_point_bytes.iter().map(|v| v.len()).sum();
        let pcs_eval_bytes: usize = hybrid_4x_proof.pcs_evals_bytes.iter().map(|v| v.len()).sum();

        let fold_c1: usize = hybrid_4x_proof.folding_c1s_bytes.iter().map(|v| v.len()).sum();
        let fold_c2: usize = hybrid_4x_proof.folding_c2s_bytes.iter().map(|v| v.len()).sum();
        let fold_c3: usize = hybrid_4x_proof.folding_c3s_bytes.iter().map(|v| v.len()).sum();
        let fold_c4: usize = hybrid_4x_proof.folding_c4s_bytes.iter().map(|v| v.len()).sum();
        let folding_total = fold_c1 + fold_c2 + fold_c3 + fold_c4;

        let piop_total = ic_bytes + qx_ic_bytes + cpr_sc_total + cpr_eval_total
            + qx_cpr_total + lookup_bytes + shift_sc_bytes + qx_shift_sc_bytes
            + eval_pt_bytes + pcs_eval_bytes + folding_total;
        let total_raw = pcs_bytes + piop_total;

        let mut all_bytes = Vec::with_capacity(total_raw);
        all_bytes.extend(&hybrid_4x_proof.pcs_proof_bytes);
        for v in &hybrid_4x_proof.ic_proof_values { all_bytes.extend(v); }
        for v in &hybrid_4x_proof.qx_ic_proof_values { all_bytes.extend(v); }
        for v in &hybrid_4x_proof.cpr_sumcheck_messages { all_bytes.extend(v); }
        all_bytes.extend(&hybrid_4x_proof.cpr_sumcheck_claimed_sum);
        for v in &hybrid_4x_proof.cpr_up_evals { all_bytes.extend(v); }
        for v in &hybrid_4x_proof.cpr_down_evals { all_bytes.extend(v); }
        for v in &hybrid_4x_proof.qx_cpr_sumcheck_messages { all_bytes.extend(v); }
        all_bytes.extend(&hybrid_4x_proof.qx_cpr_sumcheck_claimed_sum);
        for v in &hybrid_4x_proof.qx_cpr_up_evals { all_bytes.extend(v); }
        for v in &hybrid_4x_proof.qx_cpr_down_evals { all_bytes.extend(v); }
        match &hybrid_4x_proof.lookup_proof {
            Some(LookupProofData::HybridGkr(proof)) => {
                for gp in &proof.group_proofs {
                    for v in &gp.aggregated_multiplicities {
                        for f in v { write_fe(&mut all_bytes, f); }
                    }
                    for f in &gp.witness_gkr.roots_p { write_fe(&mut all_bytes, f); }
                    for f in &gp.witness_gkr.roots_q { write_fe(&mut all_bytes, f); }
                    for lp in &gp.witness_gkr.layer_proofs {
                        if let Some(ref sc) = lp.sumcheck_proof {
                            write_fe(&mut all_bytes, &sc.claimed_sum);
                            for msg in &sc.messages {
                                for f in &msg.0.tail_evaluations { write_fe(&mut all_bytes, f); }
                            }
                        }
                        for f in &lp.p_lefts { write_fe(&mut all_bytes, f); }
                        for f in &lp.p_rights { write_fe(&mut all_bytes, f); }
                        for f in &lp.q_lefts { write_fe(&mut all_bytes, f); }
                        for f in &lp.q_rights { write_fe(&mut all_bytes, f); }
                    }
                    for v in &gp.witness_gkr.sent_p { for f in v { write_fe(&mut all_bytes, f); } }
                    for v in &gp.witness_gkr.sent_q { for f in v { write_fe(&mut all_bytes, f); } }
                    write_fe(&mut all_bytes, &gp.table_gkr.root_p);
                    write_fe(&mut all_bytes, &gp.table_gkr.root_q);
                    for lp in &gp.table_gkr.layer_proofs {
                        if let Some(ref sc) = lp.sumcheck_proof {
                            write_fe(&mut all_bytes, &sc.claimed_sum);
                            for msg in &sc.messages {
                                for f in &msg.0.tail_evaluations { write_fe(&mut all_bytes, f); }
                            }
                        }
                        write_fe(&mut all_bytes, &lp.p_left);
                        write_fe(&mut all_bytes, &lp.p_right);
                        write_fe(&mut all_bytes, &lp.q_left);
                        write_fe(&mut all_bytes, &lp.q_right);
                    }
                }
            }
            Some(LookupProofData::Classic(proof)) => {
                for gp in &proof.group_proofs {
                    for v in &gp.aggregated_multiplicities {
                        for f in v { write_fe(&mut all_bytes, f); }
                    }
                    for outer in &gp.chunk_inverse_witnesses {
                        for inner in outer {
                            for f in inner { write_fe(&mut all_bytes, f); }
                        }
                    }
                    for f in &gp.inverse_table { write_fe(&mut all_bytes, f); }
                }
            }
            _ => {}
        }
        if let Some(ref sc) = hybrid_4x_proof.shift_sumcheck {
            for v in &sc.rounds { all_bytes.extend(v); }
            for v in &sc.v_finals { all_bytes.extend(v); }
        }
        if let Some(ref sc) = hybrid_4x_proof.qx_shift_sumcheck {
            for v in &sc.rounds { all_bytes.extend(v); }
            for v in &sc.v_finals { all_bytes.extend(v); }
        }
        for v in &hybrid_4x_proof.evaluation_point_bytes { all_bytes.extend(v); }
        for v in &hybrid_4x_proof.pcs_evals_bytes { all_bytes.extend(v); }
        for v in &hybrid_4x_proof.folding_c1s_bytes { all_bytes.extend(v); }
        for v in &hybrid_4x_proof.folding_c2s_bytes { all_bytes.extend(v); }
        for v in &hybrid_4x_proof.folding_c3s_bytes { all_bytes.extend(v); }
        for v in &hybrid_4x_proof.folding_c4s_bytes { all_bytes.extend(v); }

        let compressed = {
            use std::io::Write;
            let mut encoder = flate2::write::DeflateEncoder::new(
                Vec::new(), flate2::Compression::default(),
            );
            encoder.write_all(&all_bytes).unwrap();
            encoder.finish().unwrap()
        };

        eprintln!("\n=== 4x-Folded Hybrid GKR c=2 Proof Size ===");
        eprintln!("  PCS:              {:>6} B  ({:.1} KB)", pcs_bytes, pcs_bytes as f64 / 1024.0);
        eprintln!("  IC:               {:>6} B", ic_bytes);
        if qx_ic_bytes > 0 {
            eprintln!("  QX IC:            {:>6} B", qx_ic_bytes);
        }
        if cpr_sc_total > 0 {
            eprintln!("  CPR sumcheck:     {:>6} B  (msgs={cpr_msg_bytes}, sum={cpr_sum_bytes})", cpr_sc_total);
        }
        eprintln!("  Col evals:        {:>6} B  (up={cpr_up}, down={cpr_dn})", cpr_eval_total);
        if qx_cpr_total > 0 {
            eprintln!("  QX CPR:           {:>6} B", qx_cpr_total);
        }
        eprintln!("  Lookup:           {:>6} B  ({:.1} KB)", lookup_bytes, lookup_bytes as f64 / 1024.0);
        eprintln!("  Shift SC:         {:>6} B", shift_sc_bytes);
        if qx_shift_sc_bytes > 0 {
            eprintln!("  QX Shift SC:      {:>6} B", qx_shift_sc_bytes);
        }
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

    group.bench_function("E2E/Prover (4x Hybrid GKR c=2 4-chunk)", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let trace = generate_sha256_trace(SHA256_8X_NUM_VARS);
                let t = Instant::now();
                let _ = zinc_snark::pipeline::prove_hybrid_gkr_logup_4x_folded::<
                    BenchBpUair, BenchQxUair, FoldedZt4x, FoldedLc4x, 32, 16, 8, UNCHECKED,
                >(
                    &folded_params, &trace, SHA256_8X_NUM_VARS,
                    &sha_lookup_specs, &sha_affine_specs, 2,
                );
                total += t.elapsed();
            }
            total
        });
    });

    // ── 15. E2E Total Verifier (4x Hybrid GKR c=2 4-chunk) ─────────
    {
        #[cfg(not(feature = "true-ideal"))]
        fn qx_ideal_closure(_: &IdealOrZero<Sha256QxNoF2xIdeal>) -> zinc_snark::pipeline::TrivialIdeal {
            zinc_snark::pipeline::TrivialIdeal
        }
        #[cfg(feature = "true-ideal")]
        let qx_ideal_closure = |ideal: &IdealOrZero<Sha256QxNoF2xIdeal>| -> Sha256QxNoF2xIdealOverF {
            match ideal {
                IdealOrZero::Zero => Sha256QxNoF2xIdealOverF::Cyclotomic,
                IdealOrZero::Ideal(Sha256QxNoF2xIdeal::Cyclotomic) => Sha256QxNoF2xIdealOverF::Cyclotomic,
                #[cfg(feature = "true-ideal")]
                IdealOrZero::Ideal(Sha256QxNoF2xIdeal::DegreeOne) => Sha256QxNoF2xIdealOverF::DegreeOne,
                IdealOrZero::Ideal(Sha256QxNoF2xIdeal::Trivial) => Sha256QxNoF2xIdealOverF::Trivial,
            }
        };
        let r = zinc_snark::pipeline::verify_classic_logup_4x_folded::<
            BenchBpUair, BenchQxUair, FoldedZt4x, FoldedLc4x, 32, 16, 8, UNCHECKED, _, _, _, _,
        >(
            &folded_params, &hybrid_4x_proof, SHA256_8X_NUM_VARS,
            |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
            qx_ideal_closure,
            &sha_public_cols,
            &bdry_public_cols,
        );
        assert!(r.accepted, "Verifier rejected the proof (4x Hybrid GKR c=2 4-chunk)");
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
            #[cfg(not(feature = "true-ideal"))]
            fn qx_ideal_closure(_: &IdealOrZero<Sha256QxNoF2xIdeal>) -> zinc_snark::pipeline::TrivialIdeal {
                zinc_snark::pipeline::TrivialIdeal
            }
            #[cfg(feature = "true-ideal")]
            let qx_ideal_closure = |ideal: &IdealOrZero<Sha256QxNoF2xIdeal>| -> Sha256QxNoF2xIdealOverF {
                match ideal {
                    IdealOrZero::Zero => Sha256QxNoF2xIdealOverF::Cyclotomic,
                    IdealOrZero::Ideal(Sha256QxNoF2xIdeal::Cyclotomic) => Sha256QxNoF2xIdealOverF::Cyclotomic,
                    #[cfg(feature = "true-ideal")]
                    IdealOrZero::Ideal(Sha256QxNoF2xIdeal::DegreeOne) => Sha256QxNoF2xIdealOverF::DegreeOne,
                    IdealOrZero::Ideal(Sha256QxNoF2xIdeal::Trivial) => Sha256QxNoF2xIdealOverF::Trivial,
                }
            };
            let r = zinc_snark::pipeline::verify_classic_logup_4x_folded::<
                BenchBpUair, BenchQxUair, FoldedZt4x, FoldedLc4x, 32, 16, 8, UNCHECKED, _, _, _, _,
            >(
                &folded_params, &hybrid_4x_proof, SHA256_8X_NUM_VARS,
                |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
                qx_ideal_closure,
                &sha_public_cols,
                &bdry_public_cols,
            );
            black_box(r);
        });
    });
    } // end #[cfg(not(feature = "full-fold"))]

    #[cfg(feature = "full-fold")]
    {
    let full_proof = zinc_snark::pipeline::prove_hybrid_gkr_logup_full_folded::<
        BenchBpUair, BenchQxUair, FoldedZt4x, FoldedLc4x, 32, 16, 8, 4, 2, 1, UNCHECKED,
    >(
        &folded_params, &sha_trace, SHA256_8X_NUM_VARS,
        &sha_lookup_specs, &sha_affine_specs, 2,
    );

    // Print prover pipeline timing breakdown.
    {
        let t = &full_proof.timing;
        eprintln!("\n── Hybrid 32x Full-Folded GKR c=2 Prover Pipeline Timing ──");
        eprintln!("  Split columns: {:>8.3} ms", t.split_columns.as_secs_f64() * 1000.0);
        eprintln!("  PCS commit:    {:>8.3} ms", t.pcs_commit.as_secs_f64() * 1000.0);
        eprintln!("  Ideal Check:   {:>8.3} ms  (QX IC: {:.3} ms)", t.ideal_check.as_secs_f64() * 1000.0, t.qx_ideal_check.as_secs_f64() * 1000.0);
        eprintln!("  Field proj:    {:>8.3} ms", t.field_projection.as_secs_f64() * 1000.0);
        eprintln!("  Lookup extract:{:>8.3} ms", t.lookup_extract.as_secs_f64() * 1000.0);
        eprintln!("  Col eval:      {:>8.3} ms", (t.combined_poly_resolver - t.field_projection - t.lookup_extract).as_secs_f64() * 1000.0);
        eprintln!("  Proj+Eval:     {:>8.3} ms", t.combined_poly_resolver.as_secs_f64() * 1000.0);
        eprintln!("  Lookup:        {:>8.3} ms", t.lookup.as_secs_f64() * 1000.0);
        eprintln!("  Shift SC:      {:>8.3} ms", t.shift_sumcheck.as_secs_f64() * 1000.0);
        eprintln!("  Folding:       {:>8.3} ms", t.folding.as_secs_f64() * 1000.0);
        eprintln!("  Lifting:       {:>8.3} ms", t.lifting.as_secs_f64() * 1000.0);
        eprintln!("  PCS prove:     {:>8.3} ms", t.pcs_prove.as_secs_f64() * 1000.0);
        eprintln!("  Total:         {:>8.3} ms", t.total.as_secs_f64() * 1000.0);
        let accounted = t.split_columns + t.pcs_commit + t.ideal_check
            + t.combined_poly_resolver + t.lookup + t.shift_sumcheck + t.folding + t.lifting + t.pcs_prove;
        let unaccounted = t.total.saturating_sub(accounted);
        eprintln!("  Unaccounted:   {:>8.3} ms (serialize only)", unaccounted.as_secs_f64() * 1000.0);
        eprintln!("────────────────────────────────────────────────────────────\n");
    }

    // ── Proof size (simplified for full-fold) ───────────────────────
    {
        let pcs_bytes = full_proof.pcs_proof_bytes.len();
        let piop_bytes: usize = full_proof.ic_proof_values.iter().map(|v| v.len()).sum::<usize>()
            + full_proof.qx_ic_proof_values.iter().map(|v| v.len()).sum::<usize>()
            + full_proof.cpr_up_evals.iter().map(|v| v.len()).sum::<usize>()
            + full_proof.cpr_down_evals.iter().map(|v| v.len()).sum::<usize>()
            + full_proof.evaluation_point_bytes.iter().map(|v| v.len()).sum::<usize>()
            + full_proof.pcs_evals_bytes.iter().map(|v| v.len()).sum::<usize>()
            + full_proof.ring_evals_bytes.iter().map(|v| v.len()).sum::<usize>();
        let folding_bytes: usize = full_proof.folding_rounds.iter()
            .map(|fr| {
                fr.c1s_bytes.iter().map(|v| v.len()).sum::<usize>()
                + fr.c2s_bytes.iter().map(|v| v.len()).sum::<usize>()
            })
            .sum();
        let shift_bytes: usize = full_proof.shift_sumcheck.as_ref().map_or(0, |sc| {
            sc.rounds.iter().map(|v| v.len()).sum::<usize>()
            + sc.v_finals.iter().map(|v| v.len()).sum::<usize>()
        });
        let total_raw = pcs_bytes + piop_bytes + folding_bytes + shift_bytes;
        eprintln!("\n=== 32x Full-Folded Hybrid GKR c=2 Proof Size ===");
        eprintln!("  PCS:     {:>6} B  ({:.1} KB)", pcs_bytes, pcs_bytes as f64 / 1024.0);
        eprintln!("  PIOP:    {:>6} B  ({:.1} KB)", piop_bytes, piop_bytes as f64 / 1024.0);
        eprintln!("  Folding: {:>6} B  ({} rounds)", folding_bytes, full_proof.folding_rounds.len());
        eprintln!("  Shift:   {:>6} B", shift_bytes);
        eprintln!("  Total:   {:>6} B  ({:.1} KB)", total_raw, total_raw as f64 / 1024.0);
        eprintln!("═══════════════════════════════════════════════════\n");
    }

    group.bench_function("E2E/Prover (32x Full-Folded Hybrid GKR c=2 4-chunk)", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let trace = generate_sha256_trace(SHA256_8X_NUM_VARS);
                let t = Instant::now();
                let _ = zinc_snark::pipeline::prove_hybrid_gkr_logup_full_folded::<
                    BenchBpUair, BenchQxUair, FoldedZt4x, FoldedLc4x, 32, 16, 8, 4, 2, 1, UNCHECKED,
                >(
                    &folded_params, &trace, SHA256_8X_NUM_VARS,
                    &sha_lookup_specs, &sha_affine_specs, 2,
                );
                total += t.elapsed();
            }
            total
        });
    });

    // ── 15. E2E Total Verifier (32x Full-Folded) ────────────────────
    {
        #[cfg(not(feature = "true-ideal"))]
        fn qx_ideal_closure(_: &IdealOrZero<Sha256QxNoF2xIdeal>) -> zinc_snark::pipeline::TrivialIdeal {
            zinc_snark::pipeline::TrivialIdeal
        }
        #[cfg(feature = "true-ideal")]
        let qx_ideal_closure = |ideal: &IdealOrZero<Sha256QxNoF2xIdeal>| -> Sha256QxNoF2xIdealOverF {
            match ideal {
                IdealOrZero::Zero => Sha256QxNoF2xIdealOverF::Cyclotomic,
                IdealOrZero::Ideal(Sha256QxNoF2xIdeal::Cyclotomic) => Sha256QxNoF2xIdealOverF::Cyclotomic,
                #[cfg(feature = "true-ideal")]
                IdealOrZero::Ideal(Sha256QxNoF2xIdeal::DegreeOne) => Sha256QxNoF2xIdealOverF::DegreeOne,
                IdealOrZero::Ideal(Sha256QxNoF2xIdeal::Trivial) => Sha256QxNoF2xIdealOverF::Trivial,
            }
        };
        // fold_half_sizes: after each split, the "half" size.
        // D=32→16→8→4→2→1
        let fold_half_sizes = &[16usize, 8, 4, 2, 1];
        let r = zinc_snark::pipeline::verify_classic_logup_full_folded::<
            BenchBpUair, BenchQxUair, FoldedZt4x, FoldedLc4x, 32, 1, UNCHECKED, _, _, _, _,
        >(
            &folded_params, &full_proof, SHA256_8X_NUM_VARS,
            fold_half_sizes,
            |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
            qx_ideal_closure,
            &sha_public_cols,
            &bdry_public_cols,
        );
        let t = &r.timing;
        println!("\n── Verifier step timing (32x Full-Folded Hybrid GKR c=2) ──");
        println!("  IC verify:           {:>8.3} ms", t.ideal_check_verify.as_secs_f64() * 1000.0);
        println!("  CPR+Lookup verify:   {:>8.3} ms", t.combined_poly_resolver_verify.as_secs_f64() * 1000.0);
        println!("  Lookup verify:       {:>8.3} ms", t.lookup_verify.as_secs_f64() * 1000.0);
        println!("  PCS verify:          {:>8.3} ms", t.pcs_verify.as_secs_f64() * 1000.0);
        println!("  Total:               {:>8.3} ms", t.total.as_secs_f64() * 1000.0);
        println!("──────────────────────────────────────────────────────────────\n");
    }

    group.bench_function("E2E/Verifier (32x Full-Folded Hybrid GKR c=2 4-chunk)", |b| {
        b.iter(|| {
            #[cfg(not(feature = "true-ideal"))]
            fn qx_ideal_closure(_: &IdealOrZero<Sha256QxNoF2xIdeal>) -> zinc_snark::pipeline::TrivialIdeal {
                zinc_snark::pipeline::TrivialIdeal
            }
            #[cfg(feature = "true-ideal")]
            let qx_ideal_closure = |ideal: &IdealOrZero<Sha256QxNoF2xIdeal>| -> Sha256QxNoF2xIdealOverF {
                match ideal {
                    IdealOrZero::Zero => Sha256QxNoF2xIdealOverF::Cyclotomic,
                    IdealOrZero::Ideal(Sha256QxNoF2xIdeal::Cyclotomic) => Sha256QxNoF2xIdealOverF::Cyclotomic,
                    #[cfg(feature = "true-ideal")]
                    IdealOrZero::Ideal(Sha256QxNoF2xIdeal::DegreeOne) => Sha256QxNoF2xIdealOverF::DegreeOne,
                    IdealOrZero::Ideal(Sha256QxNoF2xIdeal::Trivial) => Sha256QxNoF2xIdealOverF::Trivial,
                }
            };
            let fold_half_sizes = &[16usize, 8, 4, 2, 1];
            let r = zinc_snark::pipeline::verify_classic_logup_full_folded::<
                BenchBpUair, BenchQxUair, FoldedZt4x, FoldedLc4x, 32, 1, UNCHECKED, _, _, _, _,
            >(
                &folded_params, &full_proof, SHA256_8X_NUM_VARS,
                fold_half_sizes,
                |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
                qx_ideal_closure,
                &sha_public_cols,
                &bdry_public_cols,
            );
            black_box(r);
        });
    });
    } // end #[cfg(feature = "full-fold")]

    let mem_snapshot = mem_tracker.stop();
    eprintln!("\n=== Peak Memory ===");
    eprintln!("  {mem_snapshot}");

    group.finish();
}

criterion_group!(benches, sha256_8x_folded_stepwise);
criterion_main!(benches);
