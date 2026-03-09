//! Per-step breakdown benchmarks for the 8×SHA-256 + ECDSA proving stack
//! with **4x-folded** BinaryPoly<32> → BinaryPoly<16> → BinaryPoly<8> columns,
//! using the Hybrid GKR c=2 lookup protocol with 4-chunk decomposition.
//!
//! This benchmark extends `steps_sha256_8x_folded` by adding ECDSA signature
//! verification as a **second circuit** in the **dual-circuit** pipeline.
//!
//! - SHA-256 uses the 4x-folded PCS (BinaryPoly<32>→<16>→<8>).
//! - ECDSA uses a separate PCS over `Int<4>` (256-bit integers), with its own
//!   `EcdsaScalarZipTypes` and a 256-bit PCS field (`FScalar`).
//! - **CPR is batched**: SHA and ECDSA CPR groups are combined into a single
//!   multi-degree sumcheck.
//! - **Shift/Eval sumcheck is unified**: eq-claims for all SHA and ECDSA
//!   columns, plus ECDSA's 3 genuine shifts (X+1, Y+1, Z+1).
//! - **Lookup is SHA only**: ECDSA has no lookup columns.
//! - **Folding is SHA only**: BinaryPoly requires two folding rounds; ECDSA
//!   `Int<4>` columns are not folded.
//!
//! The verifier additionally performs:
//!   - SHA-256 feed-forward: adding the initial hash values H₀…H₇ to the
//!     final working variables.
//!
//! Run with:
//!   cargo bench --bench steps_sha256_8x_ecdsa_folded -p zinc-snark --features "parallel simd asm"

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
    Field, FixedSemiring, FromWithConfig, IntoWithConfig, PrimeField,
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
    inner_product::{MBSInnerProduct, ScalarProduct},
    mul_by_scalar::ScalarWideningMulByScalar,
    named::Named,
};
use zip_plus::{
    code::{
        LinearCode,
        iprs::{
            IprsCode,
            PnttConfigF2_16R4B4,
            PnttConfigF2_16R4B64,
        },
    },
    pcs::structs::{ZipPlus, ZipPlusParams, ZipTypes},
    pcs::folding::{split_columns, fold_claims_prove},
};
#[cfg(not(feature = "no-f2x"))]
use zinc_sha256_uair::witness::GenerateWitness;
use zinc_uair::Uair;

// ─── Feature-gated UAIR type selection ──────────────────────────────────────

#[cfg(not(feature = "no-f2x"))]
use zinc_sha256_uair::Sha256Uair;

#[cfg(not(feature = "no-f2x"))]
type BenchShaUair = Sha256Uair;
#[cfg(not(feature = "no-f2x"))]
const SHA256_BATCH_SIZE: usize = 30; // 27 bitpoly + 3 int

#[cfg(feature = "no-f2x")]
use zinc_sha256_uair::no_f2x::{Sha256UairBpNoF2x, NO_F2X_NUM_COLS};

#[cfg(feature = "no-f2x")]
type BenchShaUair = Sha256UairBpNoF2x;
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
use zinc_piop::sumcheck::multi_degree::MultiDegreeSumcheck;

// ECDSA imports
use zinc_ecdsa_uair::EcdsaUairInt;
use zinc_ecdsa_uair::witness::GenerateWitness as EcdsaGenerateWitness;

/// Lightweight wrapper for CPR prover results.
struct CprProofResult<F> {
    up_evals: Vec<F>,
    down_evals: Vec<F>,
}

// ─── Type definitions ───────────────────────────────────────────────────────

const INT_LIMBS: usize = U64::LIMBS;
/// 192-bit PIOP field (shared by SHA and ECDSA CPR/shift/lookup).
type F = MontyField<{ INT_LIMBS * 3 }>;
/// 256-bit PCS field for ECDSA (matches secp256k1 word size).
type FScalar = MontyField<{ INT_LIMBS * 4 }>;

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

struct EcdsaScalarZipTypes;

impl ZipTypes for EcdsaScalarZipTypes {
    const NUM_COLUMN_OPENINGS: usize = 118;
    const GRINDING_BITS: usize = 16;
    type Eval = Int<{ INT_LIMBS * 4 }>;
    type Cw = Int<{ INT_LIMBS * 5 }>;
    type Fmod = Uint<{ INT_LIMBS * 4 }>;
    type PrimeTest = MillerRabin;
    type Chal = i128;
    type Pt = i128;
    type CombR = Int<{ INT_LIMBS * 8 }>;
    type Comb = Self::CombR;
    type EvalDotChal = ScalarProduct;
    type CombDotChal = ScalarProduct;
    type ArrCombRDotChal = MBSInnerProduct;
}

/// 4x-Folded PCS types — BinaryPoly<8> codewords.
/// Uses PnttConfigF2_16R4B4<3> (BASE_LEN=4, DEPTH=3) which gives
///   INPUT_LEN = 4 × 2^9 = 2048 rows,
///   OUTPUT_LEN proportional (rate ~1/4).
type FoldedZt4x = Sha256ZipTypes<i64, 8>;
type FoldedLc4x = IprsCode<
    FoldedZt4x,
    PnttConfigF2_16R4B4<3>,
    BinaryPolyWideningMulByScalar<i64>,
    UNCHECKED,
>;

/// ECDSA PCS types — Int<4> (256-bit) scalars, no folding.
type EcZt = EcdsaScalarZipTypes;
type EcLc = IprsCode<
    EcdsaScalarZipTypes,
    PnttConfigF2_16R4B64<1>,
    ScalarWideningMulByScalar<Int<{ INT_LIMBS * 5 }>>,
    UNCHECKED,
>;

// ─── Parameters ─────────────────────────────────────────────────────────────

const SHA256_8X_NUM_VARS: usize = 9;      // 2^9 = 512 rows (8 × 64 SHA rounds)
const SHA256_LOOKUP_COL_COUNT: usize = 10; // 10 Q[X] columns need lookup
const ECDSA_NUM_VARS: usize = 9;           // 2^9 = 512 rows (matches SHA for shared eval point)

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

/// Generate a valid ECDSA trace with real secp256k1 Int<4> witness.
fn generate_ecdsa_trace(
    num_vars: usize,
) -> Vec<DenseMultilinearExtension<Int<{ INT_LIMBS * 4 }>>> {
    let mut rng = rand::rng();
    <EcdsaUairInt as EcdsaGenerateWitness<Int<4>>>::generate_witness(num_vars, &mut rng)
}

/// Interpret a `BinaryPoly<32>` as a 32-bit unsigned integer (polynomial
/// evaluated at X = 2, equivalently the little-endian bit interpretation).
fn bp_to_u32(bp: &BinaryPoly<32>) -> u32 {
    let mut val: u32 = 0;
    for (i, coeff) in bp.iter().enumerate() {
        if coeff.into_inner() {
            val |= 1u32 << i;
        }
    }
    val
}

// ─── Benchmark ────────────────────────────────────────────────────────────

/// Measures each main step of the **4x-folded Hybrid GKR c=2 Dual-Circuit**
/// E2E proving stack for 8×SHA-256 compressions + ECDSA verification.
///
/// ## Pipeline overview
///
/// Uses `prove_dual_circuit_hybrid_gkr_4x_folded` which handles:
///   - SHA PCS: 4x-folded (BinaryPoly<32>→<16>→<8>), committed + proven
///   - ECDSA PCS: unfolded `Int<4>` PCS, committed + proven
///   - Shared IC evaluation point for both UAIRs
///   - Batched multi-degree sumcheck: SHA CPR (degree 1) + ECDSA CPR (degree ≤13)
///   - Unified evaluation sumcheck: eq-claims for all SHA + ECDSA columns,
///     plus ECDSA's 3 genuine shift claims (X+1, Y+1, Z+1)
///   - Hybrid GKR c=2 lookup (SHA columns only)
///   - 2-round SHA column folding
///
/// ## Individual bench steps
///
///   1.  SHA/WitnessGen — SHA-256 BinaryPoly<32> trace (512 rows, 30 cols)
///   2.  ECDSA/WitnessGen — ECDSA Int<4> trace (512 rows, 11 cols)
///   3.  Folding/SplitColumns — double split BinaryPoly<32> → <16> → <8>
///   4.  SHA PCS/Commit (4x-folded) — Zip+ commit over BinaryPoly<8>
///   5.  ECDSA PCS/Commit — Zip+ commit over Int<4>
///   6.  PIOP/FieldSetup — transcript + random field config
///   7.  PIOP/SHA IdealCheck — SHA BinaryPoly Ideal Check (MLE-first)
///   8.  PIOP/ECDSA IdealCheck — ECDSA Ideal Check at shared point
///   9.  PIOP/Project Main field sumcheck — project both traces to field
///  10.  PIOP/Batched CPR — multi-degree sumcheck (SHA + ECDSA)
///  11.  PIOP/Lookup — Hybrid GKR c=2 lookup (SHA only)
///  12.  PIOP/Unified Eval Sumcheck — eq + shift claims (SHA + ECDSA)
///  13.  Folding/FoldClaims (2-round) — SHA column folding
///  14.  SHA PCS/Prove (4x-folded)
///  15.  ECDSA PCS/Prove
///  16.  E2E/Prover — total dual-circuit pipeline
///  17.  E2E/Verifier — total (dual-circuit verify + feed-forward)
fn sha256_8x_ecdsa_folded_stepwise(c: &mut Criterion) {
    use zinc_sha256_uair::CyclotomicIdeal;
    use zinc_uair::ideal::ImpossibleIdeal;
    use zinc_uair::ideal_collector::IdealOrZero;
    use zinc_piop::lookup::pipeline::prove_hybrid_gkr_batched_lookup_with_indices;

    let mem_tracker = MemoryTracker::start();

    let mut group = c.benchmark_group(format!(
        "8xSHA256+ECDSA 4x-Folded Dual-Circuit Hybrid GKR c=2 Steps (grind={})",
        <FoldedZt4x as ZipTypes>::GRINDING_BITS,
    ));
    group.sample_size(100);

    // -- PCS params ---------------------------------------------------
    let folded_4x_num_vars = SHA256_8X_NUM_VARS + 2; // 11
    let row_len_4x = 2048;
    let folded_4x_lc = FoldedLc4x::new(row_len_4x);
    let sha_pcs_params = ZipPlusParams::<FoldedZt4x, FoldedLc4x>::new(
        folded_4x_num_vars, 1, folded_4x_lc,
    );
    let ec_pcs_params = ZipPlusParams::<EcZt, EcLc>::new(
        ECDSA_NUM_VARS, 1, EcLc::new(512),
    );

    let sha_lookup_specs = sha256_lookup_specs_4chunks();
    let sha_affine_specs = sha256_affine_lookup_specs_4chunks();

    let sha_num_constraints = zinc_uair::constraint_counter::count_constraints::<BenchShaUair>();
    let sha_max_degree = zinc_uair::degree_counter::count_max_degree::<BenchShaUair>();
    let ecdsa_num_constraints = zinc_uair::constraint_counter::count_constraints::<EcdsaUairInt>();
    let ecdsa_max_degree = zinc_uair::degree_counter::count_max_degree::<EcdsaUairInt>();
    let num_vars = SHA256_8X_NUM_VARS;

    assert_eq!(
        SHA256_8X_NUM_VARS, ECDSA_NUM_VARS,
        "SHA and ECDSA traces must share the same num_vars for a shared IC evaluation point"
    );

    // ── 1. SHA Witness Generation ───────────────────────────────────
    group.bench_function("SHA/WitnessGen", |b| {
        b.iter(|| black_box(generate_sha256_trace(SHA256_8X_NUM_VARS)));
    });

    // ── 2. ECDSA Witness Generation ─────────────────────────────────
    group.bench_function("ECDSA/WitnessGen", |b| {
        b.iter(|| black_box(generate_ecdsa_trace(ECDSA_NUM_VARS)));
    });

    // Pre-generate traces for subsequent steps.
    let sha_trace = generate_sha256_trace(SHA256_8X_NUM_VARS);
    assert_eq!(sha_trace.len(), SHA256_BATCH_SIZE);

    let ecdsa_trace = generate_ecdsa_trace(ECDSA_NUM_VARS);
    assert_eq!(ecdsa_trace.len(), zinc_ecdsa_uair::NUM_COLS);

    // Build PCS traces — exclude public + shift-source columns.
    let sha_sig = BenchShaUair::signature();
    let ec_sig_int = EcdsaUairInt::signature();
    let sha_excluded = sha_sig.pcs_excluded_columns();
    let ec_excluded = ec_sig_int.pcs_excluded_columns();
    let sha_pcs_trace: Vec<_> = sha_trace.iter().enumerate()
        .filter(|(i, _)| !sha_excluded.contains(i))
        .map(|(_, c)| c.clone()).collect();
    let ecdsa_pcs_trace: Vec<_> = ecdsa_trace.iter().enumerate()
        .filter(|(i, _)| !ec_excluded.contains(i))
        .map(|(_, c)| c.clone()).collect();

    // ── 3. Folding / Double Split Columns (SHA only) ────────────────
    group.bench_function("Folding/SplitColumns", |b| {
        b.iter(|| {
            let half = split_columns::<32, 16>(&sha_pcs_trace);
            let quarter = split_columns::<16, 8>(&half);
            black_box(quarter);
        });
    });

    let half_trace: Vec<DenseMultilinearExtension<BinaryPoly<16>>> =
        split_columns::<32, 16>(&sha_pcs_trace);
    let split_trace: Vec<DenseMultilinearExtension<BinaryPoly<8>>> =
        split_columns::<16, 8>(&half_trace);

    // ── 4. SHA PCS Commit (4x-folded — BinaryPoly<8>) ──────────────
    group.bench_function("SHA PCS/Commit (4x-folded)", |b| {
        b.iter(|| {
            let r = ZipPlus::<FoldedZt4x, FoldedLc4x>::commit(&sha_pcs_params, &split_trace);
            let _ = black_box(r);
        });
    });

    // ── 5. ECDSA PCS/Commit (Int<4>) ────────────────────────────────
    group.bench_function("ECDSA PCS/Commit", |b| {
        b.iter(|| {
            let r = ZipPlus::<EcZt, EcLc>::commit(&ec_pcs_params, &ecdsa_pcs_trace);
            let _ = black_box(r);
        });
    });

    // ── 6. PIOP / Field Setup ────────────────────────────────────────
    group.bench_function("PIOP/FieldSetup", |b| {
        b.iter(|| {
            let mut transcript = zinc_transcript::KeccakTranscript::new();
            let field_cfg = transcript.get_random_field_cfg::<
                F, <F as Field>::Inner, MillerRabin
            >();
            black_box(field_cfg);
        });
    });

    // ── 7. PIOP / SHA Ideal Check (MLE-first, on original BinaryPoly<32> trace)
    group.bench_function("PIOP/SHA IdealCheck", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut transcript = zinc_transcript::KeccakTranscript::new();
                let field_cfg = transcript.get_random_field_cfg::<
                    F, <F as Field>::Inner, MillerRabin
                >();
                let projected_scalars = project_scalars::<F, BenchShaUair>(|scalar| {
                    let one = F::one_with_cfg(&field_cfg);
                    let zero = F::zero_with_cfg(&field_cfg);
                    DynamicPolynomialF::new(
                        scalar.iter().map(|coeff| {
                            if coeff.into_inner() { one.clone() } else { zero.clone() }
                        }).collect::<Vec<_>>()
                    )
                });
                let t = Instant::now();
                let _ = IdealCheckProtocol::<F>::prove_mle_first::<BenchShaUair, 32>(
                    &mut transcript,
                    &sha_trace,
                    &projected_scalars,
                    sha_num_constraints,
                    SHA256_8X_NUM_VARS,
                    &field_cfg,
                ).expect("SHA Ideal Check prover failed");
                total += t.elapsed();
            }
            total
        });
    });

    // ── 8. PIOP / ECDSA Ideal Check ─────────────────────────────────
    //
    // ECDSA constraints are evaluated at the shared IC evaluation point.
    // The Int<4> trace is projected to DynamicPolynomialF<F> (degree 0).
    group.bench_function("PIOP/ECDSA IdealCheck", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut transcript = zinc_transcript::KeccakTranscript::new();
                let field_cfg = transcript.get_random_field_cfg::<
                    F, <F as Field>::Inner, MillerRabin
                >();
                let ic_eval_point: Vec<F> =
                    transcript.get_field_challenges(ECDSA_NUM_VARS, &field_cfg);

                let ec_projected_trace: Vec<DenseMultilinearExtension<DynamicPolynomialF<F>>> =
                    ecdsa_trace.iter().map(|col_mle| {
                        col_mle.iter().map(|elem|
                            DynamicPolynomialF { coeffs: vec![F::from_with_cfg(elem.clone(), &field_cfg)] }
                        ).collect()
                    }).collect();

                let ec_projected_scalars = project_scalars::<F, EcdsaUairInt>(|scalar| {
                    DynamicPolynomialF { coeffs: vec![F::from_with_cfg(scalar.clone(), &field_cfg)] }
                });

                let t = Instant::now();
                let _ = IdealCheckProtocol::<F>::prove_at_point::<EcdsaUairInt>(
                    &mut transcript,
                    &ec_projected_trace,
                    &ec_projected_scalars,
                    ecdsa_num_constraints,
                    &ic_eval_point,
                    &field_cfg,
                ).expect("ECDSA Ideal Check prover failed");
                total += t.elapsed();
            }
            total
        });
    });

    // ── Pre-compute unified transcript state for subsequent steps ────
    //
    // Follow the same flow as the dual-circuit pipeline:
    //   1. Transcript ← field config
    //   2. Shared IC evaluation point
    //   3. SHA IC (MLE-first at shared point)
    //   4. ECDSA IC (at shared point)
    //   5. Shared projecting element
    //   6. Project both traces to field
    //   7. Multi-degree sumcheck (batched CPR)
    //   8. Finalize CPR for both circuits

    let mut unified_tr = zinc_transcript::KeccakTranscript::new();
    let sha_fcfg = unified_tr.get_random_field_cfg::<F, <F as Field>::Inner, MillerRabin>();

    let ic_evaluation_point: Vec<F> =
        unified_tr.get_field_challenges(num_vars, &sha_fcfg);

    // SHA IC (at shared point) — MLE-first path.
    let sha_proj_scalars = project_scalars::<F, BenchShaUair>(|scalar| {
        let one = F::one_with_cfg(&sha_fcfg);
        let zero = F::zero_with_cfg(&sha_fcfg);
        DynamicPolynomialF::new(
            scalar.iter().map(|c| if c.into_inner() { one.clone() } else { zero.clone() }).collect::<Vec<_>>()
        )
    });
    let _ = IdealCheckProtocol::<F>::prove_mle_first_at_point::<BenchShaUair, 32>(
        &mut unified_tr, &sha_trace, &sha_proj_scalars,
        sha_num_constraints, &ic_evaluation_point, &sha_fcfg,
    ).expect("SHA IC");

    // ECDSA IC (at shared point).
    let ec_proj_trace: Vec<DenseMultilinearExtension<DynamicPolynomialF<F>>> =
        ecdsa_trace.iter().map(|col_mle| {
            col_mle.iter().map(|elem|
                DynamicPolynomialF { coeffs: vec![F::from_with_cfg(elem.clone(), &sha_fcfg)] }
            ).collect()
        }).collect();
    let ec_proj_scalars = project_scalars::<F, EcdsaUairInt>(|scalar| {
        DynamicPolynomialF { coeffs: vec![F::from_with_cfg(scalar.clone(), &sha_fcfg)] }
    });
    let _ = IdealCheckProtocol::<F>::prove_at_point::<EcdsaUairInt>(
        &mut unified_tr, &ec_proj_trace, &ec_proj_scalars,
        ecdsa_num_constraints, &ic_evaluation_point, &sha_fcfg,
    ).expect("ECDSA IC");

    // Shared projecting element.
    let projecting_elem: F = unified_tr.get_field_challenge(&sha_fcfg);

    // Project both traces to field.
    let sha_field_trace = project_trace_to_field::<F, 32>(
        &sha_trace, &[], &[], &projecting_elem,
    );
    let sha_fproj_scalars = project_scalars_to_field(sha_proj_scalars.clone(), &projecting_elem)
        .expect("SHA scalar projection failed");
    let ec_field_trace = project_trace_to_field::<F, 1>(
        &[], &[], &ec_proj_trace, &projecting_elem,
    );
    let ec_fproj_scalars = project_scalars_to_field(ec_proj_scalars.clone(), &projecting_elem)
        .expect("ECDSA scalar projection failed");

    // ── 9. PIOP / Project Main field sumcheck (both traces) ─────────
    group.bench_function("PIOP/Project Main field sumcheck", |b| {
        b.iter(|| {
            let sha_fs = project_scalars_to_field(sha_proj_scalars.clone(), &projecting_elem)
                .expect("sha scalar projection");
            let sha_ft = project_trace_to_field::<F, 32>(
                &sha_trace, &[], &[], &projecting_elem,
            );
            let ec_fs = project_scalars_to_field(ec_proj_scalars.clone(), &projecting_elem)
                .expect("ecdsa scalar projection");
            let ec_ft = project_trace_to_field::<F, 1>(
                &[], &[], &ec_proj_trace, &projecting_elem,
            );
            black_box((&sha_fs, &sha_ft, &ec_fs, &ec_ft));
        });
    });

    // ── 10. PIOP / Batched CPR (multi-degree sumcheck: SHA + ECDSA) ─
    group.bench_function("PIOP/Batched CPR", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut tr = unified_tr.clone();
                let t = Instant::now();
                let sha_group = CombinedPolyResolver::<F>::build_prover_group::<BenchShaUair>(
                    &mut tr, sha_field_trace.clone(), &ic_evaluation_point,
                    &sha_fproj_scalars, sha_num_constraints, num_vars, sha_max_degree, &sha_fcfg,
                ).expect("SHA build_prover_group");
                let sha_num_cols = sha_group.num_cols;
                let ec_group = CombinedPolyResolver::<F>::build_prover_group::<EcdsaUairInt>(
                    &mut tr, ec_field_trace.clone(), &ic_evaluation_point,
                    &ec_fproj_scalars, ecdsa_num_constraints, num_vars, ecdsa_max_degree, &sha_fcfg,
                ).expect("ECDSA build_prover_group");
                let ec_num_cols = ec_group.num_cols;
                let groups = vec![
                    (sha_group.degree, sha_group.mles, sha_group.comb_fn),
                    (ec_group.degree, ec_group.mles, ec_group.comb_fn),
                ];
                let (_, mut prover_states) =
                    MultiDegreeSumcheck::<F>::prove_as_subprotocol(
                        &mut tr, groups, num_vars, &sha_fcfg,
                    );
                let _ = CombinedPolyResolver::<F>::finalize_prover(
                    &mut tr, prover_states.remove(0), sha_num_cols, &sha_fcfg,
                ).expect("SHA finalize_prover");
                let _ = CombinedPolyResolver::<F>::finalize_prover(
                    &mut tr, prover_states.remove(0), ec_num_cols, &sha_fcfg,
                ).expect("ECDSA finalize_prover");
                total += t.elapsed();
            }
            total
        });
    });

    // Pre-compute the post-CPR transcript state for lookup/shift benchmarks.
    let (unified_tr_post_cpr, cpr_eval_point, sha_cpr_proof, ec_cpr_proof) = {
        let mut tr = unified_tr.clone();
        let sg = CombinedPolyResolver::<F>::build_prover_group::<BenchShaUair>(
            &mut tr, sha_field_trace.clone(), &ic_evaluation_point,
            &sha_fproj_scalars, sha_num_constraints, num_vars, sha_max_degree, &sha_fcfg,
        ).expect("SHA build_prover_group");
        let snc = sg.num_cols;
        let eg = CombinedPolyResolver::<F>::build_prover_group::<EcdsaUairInt>(
            &mut tr, ec_field_trace.clone(), &ic_evaluation_point,
            &ec_fproj_scalars, ecdsa_num_constraints, num_vars, ecdsa_max_degree, &sha_fcfg,
        ).expect("ECDSA build_prover_group");
        let enc = eg.num_cols;
        let groups = vec![
            (sg.degree, sg.mles, sg.comb_fn),
            (eg.degree, eg.mles, eg.comb_fn),
        ];
        let (_, mut ps) = MultiDegreeSumcheck::<F>::prove_as_subprotocol(
            &mut tr, groups, num_vars, &sha_fcfg,
        );
        let (sha_up, sha_dn, sha_ps) = CombinedPolyResolver::<F>::finalize_prover(
            &mut tr, ps.remove(0), snc, &sha_fcfg,
        ).expect("SHA finalize_prover");
        let (ec_up, ec_dn, _ec_ps) = CombinedPolyResolver::<F>::finalize_prover(
            &mut tr, ps.remove(0), enc, &sha_fcfg,
        ).expect("ECDSA finalize_prover");
        let sha_cpr_result = CprProofResult { up_evals: sha_up, down_evals: sha_dn };
        let ec_cpr_result = CprProofResult { up_evals: ec_up, down_evals: ec_dn };
        (tr, sha_ps.evaluation_point, sha_cpr_result, ec_cpr_result)
    };

    // ── 11. PIOP / Lookup (SHA only, Hybrid GKR c=2) ───────────────
    {
        let mut needed: std::collections::BTreeMap<usize, usize> =
            std::collections::BTreeMap::new();
        for spec in &sha_lookup_specs {
            let next_id = needed.len();
            needed.entry(spec.column_index).or_insert(next_id);
        }
        let mut columns: Vec<Vec<F>> = Vec::with_capacity(needed.len());
        let mut raw_indices: Vec<Vec<usize>> = Vec::with_capacity(needed.len());
        for &orig_idx in needed.keys() {
            let col_f: Vec<F> = sha_field_trace[orig_idx]
                .iter()
                .map(|inner| F::new_unchecked_with_cfg(inner.clone(), &sha_fcfg))
                .collect();
            columns.push(col_f);
            let col_idx: Vec<usize> = sha_trace[orig_idx]
                .iter()
                .map(|bp| {
                    let mut idx = 0usize;
                    for (j, coeff) in bp.iter().enumerate() {
                        if coeff.into_inner() {
                            idx |= 1usize << j;
                        }
                    }
                    idx
                })
                .collect();
            raw_indices.push(col_idx);
        }
        let index_map: std::collections::BTreeMap<usize, usize> = needed
            .keys()
            .enumerate()
            .map(|(new, &orig)| (orig, new))
            .collect();
        let remapped_specs: Vec<LookupColumnSpec> = sha_lookup_specs
            .iter()
            .map(|s| LookupColumnSpec {
                column_index: index_map[&s.column_index],
                table_type: s.table_type.clone(),
            })
            .collect();

        group.bench_function("PIOP/Lookup (Hybrid GKR c=2)", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let mut tr = unified_tr_post_cpr.clone();
                    let t = Instant::now();
                    let _ = prove_hybrid_gkr_batched_lookup_with_indices(
                        &mut tr,
                        &columns,
                        &raw_indices,
                        &remapped_specs,
                        &projecting_elem,
                        2, // cutoff c=2
                        &sha_fcfg,
                    )
                    .expect("lookup prover failed");
                    total += t.elapsed();
                }
                total
            });
        });
    }

    // ── 12. PIOP / Unified Eval Sumcheck (eq + shift) ───────────────
    //
    // Batches: eq-claims for all SHA up-eval columns, eq-claims for all
    // ECDSA up-eval columns, and ECDSA's 3 genuine shifts (X+1, Y+1, Z+1).
    {
        let c1_num_up = sha_cpr_proof.up_evals.len();
        let c2_num_up = ec_cpr_proof.up_evals.len();

        let mut shift_trace_columns: Vec<DenseMultilinearExtension<<F as Field>::Inner>> =
            Vec::new();
        // SHA non-shift columns for eq claims.
        for i in 0..c1_num_up {
            shift_trace_columns.push(sha_field_trace[i].clone());
        }
        // ECDSA non-shift columns for eq claims.
        for j in 0..c2_num_up {
            shift_trace_columns.push(ec_field_trace[j].clone());
        }
        // ECDSA shift-source columns.
        for spec in &ec_sig_int.shifts {
            shift_trace_columns.push(ec_field_trace[spec.source_col].clone());
        }

        let mut claims: Vec<ShiftClaim<F>> = Vec::new();
        // Eq claims for SHA columns (shift_amount = 0).
        for i in 0..c1_num_up {
            claims.push(ShiftClaim {
                source_col: i,
                shift_amount: 0,
                eval_point: cpr_eval_point.clone(),
                claimed_eval: sha_cpr_proof.up_evals[i].clone(),
            });
        }
        // Eq claims for ECDSA columns (shift_amount = 0).
        for j in 0..c2_num_up {
            claims.push(ShiftClaim {
                source_col: c1_num_up + j,
                shift_amount: 0,
                eval_point: cpr_eval_point.clone(),
                claimed_eval: ec_cpr_proof.up_evals[j].clone(),
            });
        }
        // ECDSA genuine shift claims.
        for (k, spec) in ec_sig_int.shifts.iter().enumerate() {
            claims.push(ShiftClaim {
                source_col: c1_num_up + c2_num_up + k,
                shift_amount: spec.shift_amount,
                eval_point: cpr_eval_point.clone(),
                claimed_eval: ec_cpr_proof.down_evals[k].clone(),
            });
        }

        group.bench_function("PIOP/Unified Eval Sumcheck", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let mut tr = unified_tr_post_cpr.clone();
                    let t = Instant::now();
                    let _ = shift_sumcheck_prove(
                        &mut tr,
                        &claims,
                        &shift_trace_columns,
                        num_vars,
                        &sha_fcfg,
                    );
                    total += t.elapsed();
                }
                total
            });
        });
    }

    // ── 13. Folding Protocol (two-round fold: D→HALF_D→QUARTER_D) ──
    //
    // Only SHA columns are folded (BinaryPoly<32>→<16>→<8>).
    // ECDSA Int<4> columns are NOT folded.
    {
        let fold_piop_point = &cpr_eval_point[..SHA256_8X_NUM_VARS];

        group.bench_function("Folding/FoldClaims (2-round)", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let mut tr = unified_tr_post_cpr.clone();
                    let t = Instant::now();
                    let fold1 = fold_claims_prove::<F, _, 16>(
                        &mut tr,
                        &half_trace,
                        fold_piop_point,
                        &projecting_elem,
                        &sha_fcfg,
                    )
                    .expect("fold1 failed");
                    let _fold2 = fold_claims_prove::<F, _, 8>(
                        &mut tr,
                        &split_trace,
                        &fold1.new_point,
                        &projecting_elem,
                        &sha_fcfg,
                    )
                    .expect("fold2 failed");
                    total += t.elapsed();
                }
                total
            });
        });
    }

    // ── 14. SHA PCS Prove (4x-folded — BinaryPoly<8>) ──────────────
    //
    // Use prove_with_seed with the Merkle commitment root so the step
    // benchmark reproduces the same transcript state (and grinding
    // nonce) as the real pipeline.
    let (sha_pcs_hint, sha_commitment) =
        ZipPlus::<FoldedZt4x, FoldedLc4x>::commit(&sha_pcs_params, &split_trace).expect("SHA commit");
    let mut sha_root_buf = [0u8; zip_plus::merkle::HASH_OUT_LEN];
    sha_commitment.root.write_transcription_bytes(&mut sha_root_buf);

    let folded_4x_pcs_point: Vec<F> = {
        // Derive the PCS field config by absorbing the commitment root,
        // matching the pipeline's transcript state.
        let mut t = zinc_transcript::KeccakTranscript::default();
        t.absorb(&sha_root_buf);
        let pcs1_cfg = t.get_random_field_cfg::<F, <F as Field>::Inner, MillerRabin>();
        let mut pt: Vec<F> = cpr_eval_point[..SHA256_8X_NUM_VARS]
            .iter()
            .map(|p| p.retrieve().into_with_cfg(&pcs1_cfg))
            .collect();
        pt.push(F::one_with_cfg(&pcs1_cfg)); // placeholder γ₁
        pt.push(F::one_with_cfg(&pcs1_cfg)); // placeholder γ₂
        pt
    };

    group.bench_function("SHA PCS/Prove (4x-folded)", |b| {
        b.iter(|| {
            let r = ZipPlus::<FoldedZt4x, FoldedLc4x>::prove_with_seed::<F, UNCHECKED>(
                &sha_pcs_params,
                &split_trace,
                &folded_4x_pcs_point,
                &sha_pcs_hint,
                &sha_root_buf,
            );
            let _ = black_box(r);
        });
    });

    // ── 15. ECDSA PCS Prove (Int<4>) ────────────────────────────────
    //
    // The ECDSA PCS operates over FScalar (256-bit MontyField<4>), not the
    // 192-bit PiopField F.  Convert the shared eval point via the same
    // path used by the dual-circuit pipeline (piop_point_to_pcs_field),
    // and use prove_with_seed with the ECDSA commitment root.
    let (ec_pcs_hint, ec_commitment) =
        ZipPlus::<EcZt, EcLc>::commit(&ec_pcs_params, &ecdsa_pcs_trace).expect("ECDSA commit");
    let mut ec_root_buf = [0u8; zip_plus::merkle::HASH_OUT_LEN];
    ec_commitment.root.write_transcription_bytes(&mut ec_root_buf);

    let ecdsa_pcs_point: Vec<FScalar> = {
        let mut t = zinc_transcript::KeccakTranscript::default();
        t.absorb(&ec_root_buf);
        let pcs2_fcfg = t.get_random_field_cfg::<
            FScalar,
            <EcdsaScalarZipTypes as ZipTypes>::Fmod,
            <EcdsaScalarZipTypes as ZipTypes>::PrimeTest,
        >();
        zinc_snark::pipeline::piop_point_to_pcs_field(
            &cpr_eval_point[..ECDSA_NUM_VARS],
            &pcs2_fcfg,
        )
    };

    group.bench_function("ECDSA PCS/Prove", |b| {
        b.iter(|| {
            let r = ZipPlus::<EcZt, EcLc>::prove_with_seed::<FScalar, UNCHECKED>(
                &ec_pcs_params,
                &ecdsa_pcs_trace,
                &ecdsa_pcs_point,
                &ec_pcs_hint,
                &ec_root_buf,
            );
            let _ = black_box(r);
        });
    });

    // ── 16. E2E Total Prover ────────────────────────────────────────

    let dual_proof = zinc_snark::pipeline::prove_dual_circuit_hybrid_gkr_4x_folded::<
        BenchShaUair,
        EcdsaUairInt,
        Int<{ INT_LIMBS * 4 }>,
        FoldedZt4x,
        FoldedLc4x,
        EcZt,
        EcLc,
        FScalar,
        32,
        16,
        8,
        UNCHECKED,
    >(
        &sha_pcs_params,
        &sha_trace,
        &ec_pcs_params,
        &ecdsa_trace,
        SHA256_8X_NUM_VARS,
        &sha_lookup_specs,
        &sha_affine_specs,
        2,
    );

    // Print prover pipeline timing breakdown.
    {
        let t = &dual_proof.timing;
        eprintln!("\n── Dual-Circuit 4x Hybrid GKR c=2 Prover Pipeline Timing ──");
        eprintln!("  PCS commit:    {:>8.3} ms", t.pcs_commit.as_secs_f64() * 1000.0);
        eprintln!("  Ideal Check:   {:>8.3} ms", t.ideal_check.as_secs_f64() * 1000.0);
        eprintln!("  CPR (batched): {:>8.3} ms", t.combined_poly_resolver.as_secs_f64() * 1000.0);
        eprintln!("  Lookup:        {:>8.3} ms", t.lookup.as_secs_f64() * 1000.0);
        eprintln!("  PCS prove:     {:>8.3} ms", t.pcs_prove.as_secs_f64() * 1000.0);
        eprintln!("  Total:         {:>8.3} ms", t.total.as_secs_f64() * 1000.0);
        let accounted = t.pcs_commit + t.ideal_check + t.combined_poly_resolver + t.lookup + t.pcs_prove;
        let unaccounted = t.total.saturating_sub(accounted);
        eprintln!("  Unaccounted:   {:>8.3} ms (split+fold+shift+serialize)", unaccounted.as_secs_f64() * 1000.0);
        eprintln!("────────────────────────────────────────────────────────\n");
    }

    group.bench_function("E2E/Prover", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let t = Instant::now();
                let _ = zinc_snark::pipeline::prove_dual_circuit_hybrid_gkr_4x_folded::<
                    BenchShaUair,
                    EcdsaUairInt,
                    Int<{ INT_LIMBS * 4 }>,
                    FoldedZt4x,
                    FoldedLc4x,
                    EcZt,
                    EcLc,
                    FScalar,
                    32,
                    16,
                    8,
                    UNCHECKED,
                >(
                    &sha_pcs_params,
                    &sha_trace,
                    &ec_pcs_params,
                    &ecdsa_trace,
                    SHA256_8X_NUM_VARS,
                    &sha_lookup_specs,
                    &sha_affine_specs,
                    2,
                );
                total += t.elapsed();
            }
            total
        });
    });

    // ── 17. E2E Total Verifier ──────────────────────────────────────

    // Build public column data for the verifier.
    let sha_public_column_data: Vec<_> = sha_sig.public_columns.iter()
        .map(|&i| sha_trace[i].clone())
        .collect();
    let ecdsa_public_column_data: Vec<_> = ec_sig_int.public_columns.iter()
        .map(|&i| ecdsa_trace[i].clone())
        .collect();

    // SHA-256 feed-forward for one instance starting at `base` row.
    let feed_forward = |trace: &[DenseMultilinearExtension<BinaryPoly<32>>],
                         base: usize|
     -> [u32; 8] {
        use zinc_sha256_uair::constants::H as SHA_H;
        [
            SHA_H[0].wrapping_add(bp_to_u32(&trace[0].evaluations[base + 64])),
            SHA_H[1].wrapping_add(bp_to_u32(&trace[0].evaluations[base + 63])),
            SHA_H[2].wrapping_add(bp_to_u32(&trace[0].evaluations[base + 62])),
            SHA_H[3].wrapping_add(bp_to_u32(&trace[0].evaluations[base + 61])),
            SHA_H[4].wrapping_add(bp_to_u32(&trace[1].evaluations[base + 64])),
            SHA_H[5].wrapping_add(bp_to_u32(&trace[1].evaluations[base + 63])),
            SHA_H[6].wrapping_add(bp_to_u32(&trace[1].evaluations[base + 62])),
            SHA_H[7].wrapping_add(bp_to_u32(&trace[1].evaluations[base + 61])),
        ]
    };

    // Print one verifier run timing.
    {
        let r = zinc_snark::pipeline::verify_dual_circuit_hybrid_gkr_4x_folded::<
            BenchShaUair,
            EcdsaUairInt,
            Int<{ INT_LIMBS * 4 }>,
            FoldedZt4x,
            FoldedLc4x,
            EcZt,
            EcLc,
            FScalar,
            32,
            16,
            8,
            UNCHECKED,
            zinc_snark::pipeline::TrivialIdeal,
            _,
            zinc_snark::pipeline::TrivialIdeal,
            _,
        >(
            &sha_pcs_params,
            &ec_pcs_params,
            &dual_proof,
            SHA256_8X_NUM_VARS,
            |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
            |_: &IdealOrZero<ImpossibleIdeal>| zinc_snark::pipeline::TrivialIdeal,
            &sha_public_column_data,
            &ecdsa_public_column_data,
        );

        let t = &r.timing;
        println!("\n── Verifier step timing (Dual-Circuit 4x Hybrid GKR c=2) ────");
        println!("  IC verify:           {:>8.3} ms", t.ideal_check_verify.as_secs_f64() * 1000.0);
        println!("  CPR+Lookup verify:   {:>8.3} ms", t.combined_poly_resolver_verify.as_secs_f64() * 1000.0);
        println!("  Lookup verify:       {:>8.3} ms", t.lookup_verify.as_secs_f64() * 1000.0);
        println!("  PCS verify:          {:>8.3} ms", t.pcs_verify.as_secs_f64() * 1000.0);
        println!("  Total:               {:>8.3} ms", t.total.as_secs_f64() * 1000.0);
        println!("─────────────────────────────────────────────────────────\n");

        // Feed-forward sanity check.
        let sha_digest = feed_forward(&sha_trace, 0);
        eprintln!("  SHA-256 digest (instance 0): {:08x?}", sha_digest);
    }

    group.bench_function("E2E/Verifier", |b| {
        b.iter(|| {
            let r = zinc_snark::pipeline::verify_dual_circuit_hybrid_gkr_4x_folded::<
                BenchShaUair,
                EcdsaUairInt,
                Int<{ INT_LIMBS * 4 }>,
                FoldedZt4x,
                FoldedLc4x,
                EcZt,
                EcLc,
                FScalar,
                32,
                16,
                8,
                UNCHECKED,
                zinc_snark::pipeline::TrivialIdeal,
                _,
                zinc_snark::pipeline::TrivialIdeal,
                _,
            >(
                &sha_pcs_params,
                &ec_pcs_params,
                &dual_proof,
                SHA256_8X_NUM_VARS,
                |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
                |_: &IdealOrZero<ImpossibleIdeal>| zinc_snark::pipeline::TrivialIdeal,
                &sha_public_column_data,
                &ecdsa_public_column_data,
            );
            black_box(&r);

            // Feed-forward: compute SHA-256 digest from trace.
            let _digest = black_box(feed_forward(&sha_trace, 0));
        });
    });

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

        // PCS proof bytes.
        let pcs1_bytes = dual_proof.pcs1_proof_bytes.len();
        let pcs2_bytes = dual_proof.pcs2_proof_bytes.len();

        // IC proof values (serialized).
        let ic1_bytes: usize = dual_proof.ic1_proof_values.iter().map(|v| v.len()).sum();
        let ic2_bytes: usize = dual_proof.ic2_proof_values.iter().map(|v| v.len()).sum();

        // Batched multi-degree sumcheck.
        let md_msg_bytes: usize = dual_proof
            .md_group_messages
            .iter()
            .flat_map(|grp| grp.iter())
            .map(|v| v.len())
            .sum();
        let md_sum_bytes: usize = dual_proof.md_claimed_sums.iter().map(|v| v.len()).sum();
        let md_total = md_msg_bytes + md_sum_bytes;

        // CPR up/down evaluations.
        let cpr1_up: usize = dual_proof.cpr1_up_evals.iter().map(|v| v.len()).sum();
        let cpr1_dn: usize = dual_proof.cpr1_down_evals.iter().map(|v| v.len()).sum();
        let cpr2_up: usize = dual_proof.cpr2_up_evals.iter().map(|v| v.len()).sum();
        let cpr2_dn: usize = dual_proof.cpr2_down_evals.iter().map(|v| v.len()).sum();
        let cpr_total = cpr1_up + cpr1_dn + cpr2_up + cpr2_dn;

        // Unified evaluation sumcheck.
        let eval_sc_bytes: usize = dual_proof.unified_eval_sumcheck.as_ref().map_or(0, |sc| {
            sc.rounds.iter().map(|v| v.len()).sum::<usize>()
                + sc.v_finals.iter().map(|v| v.len()).sum::<usize>()
        });

        // Lookup data (hybrid GKR).
        let lookup_bytes: usize = match &dual_proof.lookup_proof {
            Some(LookupProofData::HybridGkr(proof)) => {
                let mut t = 0usize;
                for gp in &proof.group_proofs {
                    let m: usize = gp.aggregated_multiplicities.iter()
                        .map(|v| v.len())
                        .sum::<usize>()
                        * fe_bytes;
                    let w_roots =
                        (gp.witness_gkr.roots_p.len() + gp.witness_gkr.roots_q.len()) * fe_bytes;
                    let mut w_layers = 0usize;
                    for lp in &gp.witness_gkr.layer_proofs {
                        let sc_fe: usize = lp.sumcheck_proof.as_ref().map_or(0, |sc| {
                            sc.messages
                                .iter()
                                .map(|msg| msg.0.tail_evaluations.len())
                                .sum::<usize>()
                                + 1
                        });
                        let child_fe = lp.p_lefts.len()
                            + lp.p_rights.len()
                            + lp.q_lefts.len()
                            + lp.q_rights.len();
                        w_layers += (sc_fe + child_fe) * fe_bytes;
                    }
                    let w_sent: usize = (gp
                        .witness_gkr
                        .sent_p
                        .iter()
                        .chain(gp.witness_gkr.sent_q.iter())
                        .map(|v| v.len())
                        .sum::<usize>())
                        * fe_bytes;
                    let t_root = 2 * fe_bytes;
                    let mut t_layers = 0usize;
                    for lp in &gp.table_gkr.layer_proofs {
                        let sc_fe: usize = lp.sumcheck_proof.as_ref().map_or(0, |sc| {
                            sc.messages
                                .iter()
                                .map(|msg| msg.0.tail_evaluations.len())
                                .sum::<usize>()
                                + 1
                        });
                        let child_fe = 4;
                        t_layers += (sc_fe + child_fe) * fe_bytes;
                    }
                    t += m + w_roots + w_layers + w_sent + t_root + t_layers;
                }
                t
            }
            _ => {
                dual_proof
                    .lookup_group_proofs
                    .iter()
                    .map(|gp| {
                        let mults: usize =
                            gp.aggregated_multiplicities.iter().map(|v| v.len()).sum();
                        let inv_w: usize = gp
                            .chunk_inverse_witnesses
                            .iter()
                            .flat_map(|o| o.iter())
                            .map(|i| i.len())
                            .sum();
                        (mults + inv_w + gp.inverse_table.len()) * fe_bytes
                    })
                    .sum()
            }
        };

        // Evaluation point + PCS evals.
        let eval_pt_bytes: usize = dual_proof.evaluation_point_bytes.iter().map(|v| v.len()).sum();
        let pcs1_eval_bytes: usize = dual_proof.pcs1_evals_bytes.iter().map(|v| v.len()).sum();
        let pcs2_eval_bytes: usize = dual_proof.pcs2_evals_bytes.iter().map(|v| v.len()).sum();

        // Folding evals (SHA only).
        let fold_c1: usize = dual_proof.pcs1_folding_c1s_bytes.iter().map(|v| v.len()).sum();
        let fold_c2: usize = dual_proof.pcs1_folding_c2s_bytes.iter().map(|v| v.len()).sum();
        let fold_c3: usize = dual_proof.pcs1_folding_c3s_bytes.iter().map(|v| v.len()).sum();
        let fold_c4: usize = dual_proof.pcs1_folding_c4s_bytes.iter().map(|v| v.len()).sum();
        let folding_total = fold_c1 + fold_c2 + fold_c3 + fold_c4;

        let piop_total = ic1_bytes
            + ic2_bytes
            + md_total
            + cpr_total
            + eval_sc_bytes
            + lookup_bytes
            + eval_pt_bytes
            + pcs1_eval_bytes
            + pcs2_eval_bytes
            + folding_total;
        let total_raw = pcs1_bytes + pcs2_bytes + piop_total;

        // Build serialized byte buffer and compress with deflate.
        let mut all_bytes = Vec::with_capacity(total_raw);
        all_bytes.extend(&dual_proof.pcs1_proof_bytes);
        all_bytes.extend(&dual_proof.pcs2_proof_bytes);
        for v in &dual_proof.ic1_proof_values {
            all_bytes.extend(v);
        }
        for v in &dual_proof.ic2_proof_values {
            all_bytes.extend(v);
        }
        for grp in &dual_proof.md_group_messages {
            for v in grp {
                all_bytes.extend(v);
            }
        }
        for v in &dual_proof.md_claimed_sums {
            all_bytes.extend(v);
        }
        for v in &dual_proof.cpr1_up_evals {
            all_bytes.extend(v);
        }
        for v in &dual_proof.cpr1_down_evals {
            all_bytes.extend(v);
        }
        for v in &dual_proof.cpr2_up_evals {
            all_bytes.extend(v);
        }
        for v in &dual_proof.cpr2_down_evals {
            all_bytes.extend(v);
        }
        if let Some(ref sc) = dual_proof.unified_eval_sumcheck {
            for v in &sc.rounds {
                all_bytes.extend(v);
            }
            for v in &sc.v_finals {
                all_bytes.extend(v);
            }
        }
        match &dual_proof.lookup_proof {
            Some(LookupProofData::HybridGkr(proof)) => {
                for gp in &proof.group_proofs {
                    for v in &gp.aggregated_multiplicities {
                        for f in v {
                            write_fe(&mut all_bytes, f);
                        }
                    }
                    for f in &gp.witness_gkr.roots_p {
                        write_fe(&mut all_bytes, f);
                    }
                    for f in &gp.witness_gkr.roots_q {
                        write_fe(&mut all_bytes, f);
                    }
                    for lp in &gp.witness_gkr.layer_proofs {
                        if let Some(ref sc) = lp.sumcheck_proof {
                            write_fe(&mut all_bytes, &sc.claimed_sum);
                            for msg in &sc.messages {
                                for f in &msg.0.tail_evaluations {
                                    write_fe(&mut all_bytes, f);
                                }
                            }
                        }
                        for f in &lp.p_lefts {
                            write_fe(&mut all_bytes, f);
                        }
                        for f in &lp.p_rights {
                            write_fe(&mut all_bytes, f);
                        }
                        for f in &lp.q_lefts {
                            write_fe(&mut all_bytes, f);
                        }
                        for f in &lp.q_rights {
                            write_fe(&mut all_bytes, f);
                        }
                    }
                    for v in &gp.witness_gkr.sent_p {
                        for f in v {
                            write_fe(&mut all_bytes, f);
                        }
                    }
                    for v in &gp.witness_gkr.sent_q {
                        for f in v {
                            write_fe(&mut all_bytes, f);
                        }
                    }
                    write_fe(&mut all_bytes, &gp.table_gkr.root_p);
                    write_fe(&mut all_bytes, &gp.table_gkr.root_q);
                    for lp in &gp.table_gkr.layer_proofs {
                        if let Some(ref sc) = lp.sumcheck_proof {
                            write_fe(&mut all_bytes, &sc.claimed_sum);
                            for msg in &sc.messages {
                                for f in &msg.0.tail_evaluations {
                                    write_fe(&mut all_bytes, f);
                                }
                            }
                        }
                        write_fe(&mut all_bytes, &lp.p_left);
                        write_fe(&mut all_bytes, &lp.p_right);
                        write_fe(&mut all_bytes, &lp.q_left);
                        write_fe(&mut all_bytes, &lp.q_right);
                    }
                }
            }
            _ => {
                for gp in &dual_proof.lookup_group_proofs {
                    for v in &gp.aggregated_multiplicities {
                        for f in v {
                            write_fe(&mut all_bytes, f);
                        }
                    }
                    for outer in &gp.chunk_inverse_witnesses {
                        for inner in outer {
                            for f in inner {
                                write_fe(&mut all_bytes, f);
                            }
                        }
                    }
                    for f in &gp.inverse_table {
                        write_fe(&mut all_bytes, f);
                    }
                }
            }
        }
        for v in &dual_proof.evaluation_point_bytes {
            all_bytes.extend(v);
        }
        for v in &dual_proof.pcs1_evals_bytes {
            all_bytes.extend(v);
        }
        for v in &dual_proof.pcs2_evals_bytes {
            all_bytes.extend(v);
        }
        for v in &dual_proof.pcs1_folding_c1s_bytes {
            all_bytes.extend(v);
        }
        for v in &dual_proof.pcs1_folding_c2s_bytes {
            all_bytes.extend(v);
        }
        for v in &dual_proof.pcs1_folding_c3s_bytes {
            all_bytes.extend(v);
        }
        for v in &dual_proof.pcs1_folding_c4s_bytes {
            all_bytes.extend(v);
        }

        let compressed = {
            use std::io::Write;
            let mut encoder =
                flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
            encoder.write_all(&all_bytes).unwrap();
            encoder.finish().unwrap()
        };

        eprintln!("\n=== 4x-Folded Dual-Circuit Hybrid GKR c=2 Proof Size ===");
        eprintln!(
            "  PCS (SHA 4x-folded): {:>7} B  ({:.1} KB)",
            pcs1_bytes,
            pcs1_bytes as f64 / 1024.0
        );
        eprintln!(
            "  PCS (ECDSA):         {:>7} B  ({:.1} KB)",
            pcs2_bytes,
            pcs2_bytes as f64 / 1024.0
        );
        eprintln!("  IC (SHA):            {:>7} B", ic1_bytes);
        eprintln!("  IC (ECDSA):          {:>7} B", ic2_bytes);
        eprintln!(
            "  MD sumcheck:         {:>7} B  (msgs={md_msg_bytes}, sums={md_sum_bytes})",
            md_total
        );
        eprintln!(
            "  CPR evals:           {:>7} B  (c1_up={cpr1_up}, c1_dn={cpr1_dn}, c2_up={cpr2_up}, c2_dn={cpr2_dn})",
            cpr_total
        );
        eprintln!("  Eval sumcheck:       {:>7} B", eval_sc_bytes);
        eprintln!(
            "  Lookup:              {:>7} B  ({:.1} KB)",
            lookup_bytes,
            lookup_bytes as f64 / 1024.0
        );
        eprintln!("  Eval point:          {:>7} B", eval_pt_bytes);
        eprintln!(
            "  PCS evals:           {:>7} B  (SHA={pcs1_eval_bytes}, ECDSA={pcs2_eval_bytes})",
            pcs1_eval_bytes + pcs2_eval_bytes
        );
        eprintln!(
            "  Folding:             {:>7} B  (c1={fold_c1}, c2={fold_c2}, c3={fold_c3}, c4={fold_c4})",
            folding_total
        );
        eprintln!("  ─────────────────────────────────────────");
        eprintln!(
            "  PIOP total:          {:>7} B  ({:.1} KB)",
            piop_total,
            piop_total as f64 / 1024.0
        );
        eprintln!(
            "  Total raw:           {:>7} B  ({:.1} KB)",
            total_raw,
            total_raw as f64 / 1024.0
        );
        eprintln!(
            "  Compressed:          {:>7} B  ({:.1} KB, {:.1}x ratio)",
            compressed.len(),
            compressed.len() as f64 / 1024.0,
            all_bytes.len() as f64 / compressed.len() as f64
        );
        eprintln!("═══════════════════════════════════════════\n");
    }

    let mem_snapshot = mem_tracker.stop();
    eprintln!("\n=== Peak Memory ===");
    eprintln!("  {mem_snapshot}");

    group.finish();
}

criterion_group!(benches, sha256_8x_ecdsa_folded_stepwise);
criterion_main!(benches);
