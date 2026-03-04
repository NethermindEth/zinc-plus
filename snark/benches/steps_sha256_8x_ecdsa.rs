//! Per-step breakdown benchmarks for 8×SHA-256 + ECDSA proving stack.
//!
//! Run with:
//!   cargo bench --bench steps_sha256_8x_ecdsa -p zinc-snark --features "parallel simd asm"

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
    Field, FixedSemiring, FromWithConfig, PrimeField,
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
        iprs::{IprsCode, PnttConfigF2_16R4B4, PnttConfigF2_16R4B16, PnttConfigF2_16R4B64},
    },
    pcs::structs::{ZipPlus, ZipPlusParams, ZipTypes},
    pcs::folding::split_columns,
};

use zinc_ecdsa_uair::EcdsaUairInt;
use zinc_ecdsa_uair::witness::GenerateWitness as EcdsaGenerateWitness;
use zinc_sha256_uair::{Sha256Uair, witness::GenerateWitness};
use zinc_piop::projections::{
    project_trace_coeffs, project_trace_to_field,
    project_scalars, project_scalars_to_field,
};
use zinc_piop::lookup::{LookupColumnSpec, LookupTableType, LookupWitnessSource, AffineLookupSpec};
use zinc_piop::lookup::{
    BatchedDecompLogupProtocol, group_lookup_specs,
};
use zinc_piop::lookup::pipeline::build_lookup_instance_from_indices_pub;
use zinc_piop::shift_sumcheck::{
    ShiftClaim, ShiftSumcheckProof, ShiftRoundPoly,
    shift_sumcheck_prove, shift_sumcheck_verify,
    shift_sumcheck_verify_pre, shift_sumcheck_verify_finalize,
};
use zinc_piop::sumcheck::multi_degree::{MultiDegreeSumcheck, MultiDegreeSumcheckProof};
use zinc_piop::sumcheck::prover::{NatEvaluatedPolyWithoutConstant, ProverMsg};
use zinc_piop::ideal_check::IdealCheckProtocol;
use zinc_piop::combined_poly_resolver::CombinedPolyResolver;
use zinc_piop::lookup::pipeline::generate_table_and_shifts;
use zinc_utils::projectable_to_field::ProjectableToField;

use zinc_uair::Uair;

// ─── Type definitions ───────────────────────────────────────────────────────

const INT_LIMBS: usize = U64::LIMBS;
type F = MontyField<{ INT_LIMBS * 3 }>;
/// 256-bit PCS field for ECDSA — matches the secp256k1 PIOP field size.
type FScalar = MontyField<{ INT_LIMBS * 4 }>;

/// 256-bit Montgomery field for ECDSA-specific PIOP steps.
///
/// The ECDSA constraints hold mod secp256k1's base field prime p,
/// so the PIOP field for ECDSA uses p directly (no random prime).
type EcdsaField = MontyField<{ INT_LIMBS * 4 }>;

/// Field config for ECDSA PIOP steps: secp256k1 base field p.
fn secp256k1_field_config() -> crypto_bigint::modular::MontyParams<{ U64::LIMBS * 4 }> {
    let modulus = crypto_bigint::Uint::<{ U64::LIMBS * 4 }>::from_be_hex(
        "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f",
    );
    let modulus = crypto_bigint::Odd::new(modulus).expect("secp256k1 p is odd");
    crypto_bigint::modular::MontyParams::new(modulus)
}

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

struct EcdsaScalarZipTypes;

impl ZipTypes for EcdsaScalarZipTypes {
    const NUM_COLUMN_OPENINGS: usize = 131;
    const GRINDING_BITS: usize = 8;
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

type IprsBPoly32R4B64<const DEPTH: usize, const CHECK: bool> = IprsCode<
    Sha256ZipTypes<i64, 32>,
    PnttConfigF2_16R4B64<DEPTH>,
    BinaryPolyWideningMulByScalar<i64>,
    CHECK,
>;

type IprsInt4R4B64<const DEPTH: usize, const CHECK: bool> = IprsCode<
    EcdsaScalarZipTypes,
    PnttConfigF2_16R4B64<DEPTH>,
    ScalarWideningMulByScalar<Int<{ INT_LIMBS * 5 }>>,
    CHECK,
>;

// ── Folded SHA-256 PCS type (BinaryPoly<16>, half the width of BinaryPoly<32>) ──
//
// Uses PnttConfigF2_16R4B16<2> (DEPTH=2) which gives INPUT_LEN = 16 × 8² = 1024,
// matching a row_len of 1024.  Each split column has 2× the original column length
// so the commitment row fits in a single BinaryPoly<16> opening instead of two.
type FoldedShaZt = Sha256ZipTypes<i64, 16>;
type FoldedShaLc = IprsCode<FoldedShaZt, PnttConfigF2_16R4B16<2>, BinaryPolyWideningMulByScalar<i64>, UNCHECKED>;

// ── 4x Folded SHA-256 PCS type (BinaryPoly<8>, a quarter the width of BinaryPoly<32>) ──
//
// Uses PnttConfigF2_16R4B4<3> (BASE_LEN=4, DEPTH=3) which gives
// INPUT_LEN = 4 × 2^9 = 2048, matching a row_len of 2048.  Each 4x-split
// column has 4× the original column length, so the commitment row fits in a
// single BinaryPoly<8> opening instead of four.
type FoldedSha4xZt = Sha256ZipTypes<i64, 8>;
type FoldedSha4xLc = IprsCode<FoldedSha4xZt, PnttConfigF2_16R4B4<3>, BinaryPolyWideningMulByScalar<i64>, UNCHECKED>;

// ─── Parameters ─────────────────────────────────────────────────────────────

const SHA256_8X_NUM_VARS: usize = 9;
const SHA256_BATCH_SIZE: usize = 30;       // 30 SHA-256 columns (27 bitpoly + 3 int)
const ECDSA_NUM_VARS: usize = 9;
const SHA256_LOOKUP_COL_COUNT: usize = 10;

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

fn generate_sha256_trace(num_vars: usize) -> Vec<DenseMultilinearExtension<BinaryPoly<32>>> {
    let mut rng = rand::rng();
    <Sha256Uair as GenerateWitness<BinaryPoly<32>>>::generate_witness(num_vars, &mut rng)
}

/// Generate a valid ECDSA trace with real F_p witness (secp256k1).
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

// ─── Benchmark ──────────────────────────────────────────────────────────────

/// Measures each main step of the E2E proving stack individually for
/// 8×SHA-256 compressions **plus** ECDSA verification.
///
/// Steps benchmarked (for each sub-protocol and combined):
///   1.  SHA/WitnessGen        — SHA-256 trace (26 cols × 512 rows)
///   2.  ECDSA/WitnessGen      — ECDSA trace (11 cols × 512 rows)
///   3.  SHA/PCS/Commit        — Zip+ commit for SHA batch
///   4.  ECDSA/PCS/Commit      — Zip+ commit for ECDSA batch
///   5.  SHA/PIOP/FieldSetup   — transcript + random field config
///   6.  SHA/PIOP/Project Ideal Check    — project_trace_coeffs + project_scalars
///   7.  ECDSA/PIOP/FieldSetup — transcript + random field config
///   8.  ECDSA/PIOP/Project Ideal Check  — project ECDSA trace to DynamicPolynomialF
///   9.  SHA/PIOP/IdealCheck
///  10.  ECDSA/PIOP/IdealCheck
///  11.  SHA/PIOP/Project Main field sumcheck   — project_scalars_to_field + project_trace_to_field
///  12.  ECDSA/PIOP/Project Main field sumcheck — same for ECDSA
///  13.  PIOP/Main field sumcheck              — unified multi-degree sumcheck
///  14.  SHA/PIOP/LookupExtract — extract lookup columns from field trace
///  15.  SHA/PIOP/Lookup        — lookup sumcheck (standalone)
///  16.  PIOP/Batched Main field sumcheck+Lookup — all main field sumchecks + SHA Lookup in one
///                                multi-degree sumcheck (A/B comparison with 13+15)
///  17.  SHA/PCS/Prove
///  18.  ECDSA/PCS/Prove
///  19.  E2E/Prover  (unified dual-circuit pipeline)
///  20.  E2E/Verifier (unified dual-circuit pipeline)
fn sha256_8x_ecdsa_stepwise(c: &mut Criterion) {
    use zinc_sha256_uair::CyclotomicIdeal;
    use zinc_uair::ideal::ImpossibleIdeal;
    use zinc_uair::ideal_collector::IdealOrZero;
    use zinc_piop::lookup::prove_batched_lookup_with_indices;
    use zinc_piop::lookup::prove_gkr_batched_lookup_with_indices;

    let mem_tracker = MemoryTracker::start();

    let mut group = c.benchmark_group("8xSHA256+ECDSA Steps");
    group.sample_size(100);

    type ShaZt = Sha256ZipTypes<i64, 32>;
    type ShaLc = IprsBPoly32R4B64<1, UNCHECKED>;
    type EcZt = EcdsaScalarZipTypes;
    type EcLc = IprsInt4R4B64<1, UNCHECKED>;

    let sha_params = ZipPlusParams::<ShaZt, ShaLc>::new(SHA256_8X_NUM_VARS, 1, ShaLc::new(512));
    let ec_params  = ZipPlusParams::<EcZt, EcLc>::new(ECDSA_NUM_VARS, 1, EcLc::new(512));

    // Folded SHA params: split columns are BinaryPoly<16> with 2× original row length
    // (1024 entries per column after splitting), so row_len = 1024 and num_rows = 1.
    let folded_sha_params = ZipPlusParams::<FoldedShaZt, FoldedShaLc>::new(
        SHA256_8X_NUM_VARS + 1, // folded columns have num_vars + 1
        1,
        FoldedShaLc::new(1024),
    );

    // 4x-Folded SHA params: split columns are BinaryPoly<8> with 4× original row length
    // (2048 entries per column after two splits), so row_len = 2048 and num_rows = 1.
    let folded_4x_sha_params = ZipPlusParams::<FoldedSha4xZt, FoldedSha4xLc>::new(
        SHA256_8X_NUM_VARS + 2, // 4x-folded columns have num_vars + 2
        1,
        FoldedSha4xLc::new(2048),
    );

    let sha_lookup_specs = sha256_lookup_specs();
    let sha_affine_specs = sha256_affine_lookup_specs();

    let sha_num_constraints  = zinc_uair::constraint_counter::count_constraints::<Sha256Uair>();
    let sha_max_degree       = zinc_uair::degree_counter::count_max_degree::<Sha256Uair>();
    let ecdsa_num_constraints = zinc_uair::constraint_counter::count_constraints::<EcdsaUairInt>();
    let ecdsa_max_degree      = zinc_uair::degree_counter::count_max_degree::<EcdsaUairInt>();

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
    // Real F_p witness for ECDSA-only steps (constraints hold mod secp256k1 p).
    let ecdsa_real_trace = generate_ecdsa_trace(ECDSA_NUM_VARS);
    // Real ECDSA witness for combined SHA+ECDSA steps.  The values are
    // the same Int<4> elements as ecdsa_real_trace; constraint checks
    // in the 192-bit PIOP field won't hold (they require secp256k1 p),
    // but the PCS proof size is realistic.
    let ecdsa_trace = generate_ecdsa_trace(ECDSA_NUM_VARS);

    // Build PCS traces — exclude public columns AND shift source
    // columns (whose evaluation claims are resolved by the shift
    // sumcheck, not the PCS).
    let sha_sig = Sha256Uair::signature();
    let ec_sig_int = EcdsaUairInt::signature();
    let sha_excluded = sha_sig.pcs_excluded_columns();
    let ec_excluded = ec_sig_int.pcs_excluded_columns();
    let sha_pcs_trace: Vec<_> = sha_trace.iter().enumerate()
        .filter(|(i, _)| !sha_excluded.contains(i))
        .map(|(_, c)| c.clone()).collect();
    let ecdsa_pcs_trace: Vec<_> = ecdsa_trace.iter().enumerate()
        .filter(|(i, _)| !ec_excluded.contains(i))
        .map(|(_, c)| c.clone()).collect();

    // ── 2b. SHA Folding / Split Columns ─────────────────────────────
    //
    // Measures the cost of splitting BinaryPoly<32> → BinaryPoly<16>
    // (2x folding) for the SHA PCS-committed columns.
    group.bench_function("SHA/Folding/SplitColumns", |b| {
        b.iter(|| {
            let split = split_columns::<32, 16>(&sha_pcs_trace);
            black_box(split);
        });
    });

    let sha_split_trace: Vec<DenseMultilinearExtension<BinaryPoly<16>>> =
        split_columns::<32, 16>(&sha_pcs_trace);

    // ── 3. SHA PCS Commit (original — BinaryPoly<32>) ───────────────
    group.bench_function("SHA/PCS/Commit", |b| {
        b.iter(|| black_box(
            ZipPlus::<ShaZt, ShaLc>::commit(&sha_params, &sha_pcs_trace).expect("commit")
        ));
    });

    // ── 3b. SHA PCS Commit (folded — BinaryPoly<16>) ────────────────
    group.bench_function("SHA/PCS/Commit (folded)", |b| {
        b.iter(|| black_box(
            ZipPlus::<FoldedShaZt, FoldedShaLc>::commit(&folded_sha_params, &sha_split_trace).expect("commit")
        ));
    });

    // ── 4. ECDSA PCS Commit ─────────────────────────────────────────
    group.bench_function("ECDSA/PCS/Commit", |b| {
        b.iter(|| black_box(
            ZipPlus::<EcZt, EcLc>::commit(&ec_params, &ecdsa_pcs_trace).expect("commit")
        ));
    });

    // ── 5. SHA PIOP / Field Setup ────────────────────────────────────
    //
    // In pipeline::prove, ideal_check_time includes transcript creation
    // and field config generation. Measure these separately.
    group.bench_function("SHA/PIOP/FieldSetup", |b| {
        b.iter(|| {
            let mut transcript = zinc_transcript::KeccakTranscript::new();
            let field_cfg = transcript.get_random_field_cfg::<
                F, <F as Field>::Inner, MillerRabin
            >();
            black_box(field_cfg);
        });
    });

    // ── 6. SHA PIOP / Project Trace for Ideal Check ─────────────────
    //
    // With MLE-first (max_degree == 1), the pipeline skips the full
    // project_trace_coeffs and only calls project_scalars.  Measure
    // the actual cost seen in the E2E pipeline.
    group.bench_function("SHA/PIOP/Project Ideal Check", |b| {
        let mut tr_setup = zinc_transcript::KeccakTranscript::new();
        let fcfg = tr_setup.get_random_field_cfg::<F, <F as Field>::Inner, MillerRabin>();
        assert_eq!(sha_max_degree, 1, "SHA-256 UAIR should have max_degree == 1 (MLE-first path)");
        b.iter(|| {
            // MLE-first path: only project scalars; no trace projection needed.
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

    // ── 7. ECDSA PIOP / Field Setup ─────────────────────────────────
    //
    // For ECDSA the PIOP field is secp256k1's base-field p (fixed, 256-bit).
    // No random prime is sampled from the transcript.
    group.bench_function("ECDSA/PIOP/FieldSetup", |b| {
        b.iter(|| {
            let field_cfg = secp256k1_field_config();
            black_box(field_cfg);
        });
    });

    // ── 8. ECDSA PIOP / Project Trace for Ideal Check ───────────────
    //
    // Convert Int<4> trace to DynamicPolynomialF<EcdsaField> + project_scalars.
    // Uses secp256k1 p so that Int<4> → field reduction preserves F_p arithmetic.
    group.bench_function("ECDSA/PIOP/Project Ideal Check", |b| {
        let fcfg = secp256k1_field_config();
        b.iter(|| {
            let projected_trace: Vec<DenseMultilinearExtension<DynamicPolynomialF<EcdsaField>>> =
                ecdsa_real_trace.iter().map(|col_mle| {
                    col_mle.iter().map(|elem|
                        DynamicPolynomialF { coeffs: vec![EcdsaField::from_with_cfg(elem.clone(), &fcfg)] }
                    ).collect()
                }).collect();
            let projected_scalars = project_scalars::<EcdsaField, EcdsaUairInt>(|scalar| {
                DynamicPolynomialF { coeffs: vec![EcdsaField::from_with_cfg(scalar.clone(), &fcfg)] }
            });
            black_box((&projected_trace, &projected_scalars));
        });
    });

    // ── 9. SHA PIOP / Ideal Check ───────────────────────────────────
    group.bench_function("SHA/PIOP/IdealCheck", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut transcript = zinc_transcript::KeccakTranscript::new();
                let field_cfg = transcript.get_random_field_cfg::<
                    F, <F as Field>::Inner, MillerRabin
                >();
                // MLE-first path: only project scalars, not the trace.
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
                let _ = zinc_piop::ideal_check::IdealCheckProtocol::<F>::prove_mle_first::<Sha256Uair, 32>(
                    &mut transcript, &sha_trace, &projected_scalars,
                    sha_num_constraints, SHA256_8X_NUM_VARS, &field_cfg,
                ).expect("Ideal Check");
                total += t.elapsed();
            }
            total
        });
    });

    // ── 10. ECDSA PIOP / Ideal Check ───────────────────────────────
    //
    // Runs over EcdsaField (secp256k1 p, 256-bit) so that Int<4>
    // constraint identities hold after field reduction.
    group.bench_function("ECDSA/PIOP/IdealCheck", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut transcript = zinc_transcript::KeccakTranscript::new();
                let field_cfg = secp256k1_field_config();
                let projected_trace: Vec<DenseMultilinearExtension<DynamicPolynomialF<EcdsaField>>> =
                    ecdsa_real_trace.iter().map(|col_mle| {
                        col_mle.iter().map(|elem|
                            DynamicPolynomialF { coeffs: vec![EcdsaField::from_with_cfg(elem.clone(), &field_cfg)] }
                        ).collect()
                    }).collect();
                let projected_scalars = project_scalars::<EcdsaField, EcdsaUairInt>(|scalar| {
                    DynamicPolynomialF { coeffs: vec![EcdsaField::from_with_cfg(scalar.clone(), &field_cfg)] }
                });
                let t = Instant::now();
                let _ = zinc_piop::ideal_check::IdealCheckProtocol::<EcdsaField>::prove_as_subprotocol::<EcdsaUairInt>(
                    &mut transcript, &projected_trace, &projected_scalars,
                    ecdsa_num_constraints, ECDSA_NUM_VARS, &field_cfg,
                ).expect("Ideal Check");
                total += t.elapsed();
            }
            total
        });
    });

    // ── 11. SHA PIOP / Project Trace for Main field sumcheck ────────────────────────
    //
    // project_scalars_to_field + project_trace_to_field — the F[X]→F
    // specialisation that pipeline::prove includes in main_field_sumcheck_time.
    group.bench_function("SHA/PIOP/Project Main field sumcheck", |b| {
        let mut tr_setup = zinc_transcript::KeccakTranscript::new();
        let fcfg = tr_setup.get_random_field_cfg::<F, <F as Field>::Inner, MillerRabin>();
        let proj_elem: F = tr_setup.get_field_challenge(&fcfg);
        let proj_scalars_setup = project_scalars::<F, Sha256Uair>(|scalar| {
            let one = F::one_with_cfg(&fcfg);
            let zero = F::zero_with_cfg(&fcfg);
            DynamicPolynomialF::new(
                scalar.iter().map(|c| if c.into_inner() { one.clone() } else { zero.clone() })
                    .collect::<Vec<_>>()
            )
        });
        b.iter(|| {
            let fproj_scalars =
                project_scalars_to_field(proj_scalars_setup.clone(), &proj_elem)
                    .expect("scalar projection");
            let field_trace = project_trace_to_field::<F, 32>(
                &sha_trace, &[], &[], &proj_elem,
            );
            black_box((&fproj_scalars, &field_trace));
        });
    });

    // ── 12. ECDSA PIOP / Project Trace for Main field sumcheck ──────────────────────
    //
    // EcdsaField (secp256k1 p). Degree-0 polynomials, so evaluation
    // at any projection element is trivially the constant.
    group.bench_function("ECDSA/PIOP/Project Main field sumcheck", |b| {
        let fcfg = secp256k1_field_config();
        let proj_elem: EcdsaField = {
            let mut tr_setup = zinc_transcript::KeccakTranscript::new();
            tr_setup.get_field_challenge(&fcfg)
        };
        let ec_proj_trace_bench: Vec<DenseMultilinearExtension<DynamicPolynomialF<EcdsaField>>> =
            ecdsa_real_trace.iter().map(|col_mle| {
                col_mle.iter().map(|elem|
                    DynamicPolynomialF { coeffs: vec![EcdsaField::from_with_cfg(elem.clone(), &fcfg)] }
                ).collect()
            }).collect();
        let proj_scalars_setup = project_scalars::<EcdsaField, EcdsaUairInt>(|scalar| {
            DynamicPolynomialF { coeffs: vec![EcdsaField::from_with_cfg(scalar.clone(), &fcfg)] }
        });
        b.iter(|| {
            let fproj_scalars =
                project_scalars_to_field(proj_scalars_setup.clone(), &proj_elem)
                    .expect("scalar projection");
            let field_trace = project_trace_to_field::<EcdsaField, 1>(
                &[], &[], &ec_proj_trace_bench, &proj_elem,
            );
            black_box((&fproj_scalars, &field_trace));
        });
    });

    // ── 13. Unified PIOP / Main field sumcheck (SHA + ECDSA) ────────────────────────
    //
    // Both main field sumcheck groups are batched into a single multi-degree
    // sumcheck.  One shared transcript, IC evaluation point, and
    // projecting element — matching the unified pipeline.
    assert_eq!(SHA256_8X_NUM_VARS, ECDSA_NUM_VARS,
        "multi-degree sumcheck requires both protocols to have the same num_vars");
    let num_vars = SHA256_8X_NUM_VARS;

    let mut unified_tr = zinc_transcript::KeccakTranscript::new();
    let sha_fcfg = unified_tr.get_random_field_cfg::<F, <F as Field>::Inner, MillerRabin>();

    // Shared IC evaluation point.
    let ic_evaluation_point: Vec<F> =
        unified_tr.get_field_challenges(num_vars, &sha_fcfg);

    // SHA IC (at shared point) — MLE-first path for max_degree == 1.
    let sha_proj_scalars = project_scalars::<F, Sha256Uair>(|scalar| {
        let one = F::one_with_cfg(&sha_fcfg);
        let zero = F::zero_with_cfg(&sha_fcfg);
        DynamicPolynomialF::new(
            scalar.iter().map(|c| if c.into_inner() { one.clone() } else { zero.clone() }).collect::<Vec<_>>()
        )
    });
    let _ = zinc_piop::ideal_check::IdealCheckProtocol::<F>::prove_mle_first_at_point::<Sha256Uair, 32>(
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
    let _ = zinc_piop::ideal_check::IdealCheckProtocol::<F>::prove_at_point::<EcdsaUairInt>(
        &mut unified_tr, &ec_proj_trace, &ec_proj_scalars,
        ecdsa_num_constraints, &ic_evaluation_point, &sha_fcfg,
    ).expect("ECDSA IC");

    // Shared projecting element.
    let sha_proj_elem: F = unified_tr.get_field_challenge(&sha_fcfg);

    // Project both traces to field.
    let sha_field_trace = project_trace_to_field::<F, 32>(
        &sha_trace, &[], &[], &sha_proj_elem,
    );
    let sha_fproj_scalars = project_scalars_to_field(sha_proj_scalars.clone(), &sha_proj_elem)
        .expect("SHA scalar projection failed");
    let ec_field_trace = project_trace_to_field::<F, 1>(
        &[], &[], &ec_proj_trace, &sha_proj_elem,
    );
    let ec_fproj_scalars = project_scalars_to_field(ec_proj_scalars.clone(), &sha_proj_elem)
        .expect("ECDSA scalar projection failed");

    // Pre-compute the post-main-field-sumcheck transcript state (needed by lookup benchmarks).
    let unified_tr_post_cpr = {
        let mut tr = unified_tr.clone();
        let sg = zinc_piop::combined_poly_resolver::CombinedPolyResolver::<F>::build_prover_group::<Sha256Uair>(
            &mut tr, sha_field_trace.clone(), &ic_evaluation_point,
            &sha_fproj_scalars, sha_num_constraints, num_vars, sha_max_degree, &sha_fcfg,
        ).expect("SHA build_prover_group");
        let snc = sg.num_cols;
        let eg = zinc_piop::combined_poly_resolver::CombinedPolyResolver::<F>::build_prover_group::<EcdsaUairInt>(
            &mut tr, ec_field_trace.clone(), &ic_evaluation_point,
            &ec_fproj_scalars, ecdsa_num_constraints, num_vars, ecdsa_max_degree, &sha_fcfg,
        ).expect("ECDSA build_prover_group");
        let enc = eg.num_cols;
        let groups = vec![
            (sg.degree, sg.mles, sg.comb_fn),
            (eg.degree, eg.mles, eg.comb_fn),
        ];
        let (_, mut ps) = zinc_piop::sumcheck::multi_degree::MultiDegreeSumcheck::<F>::prove_as_subprotocol(
            &mut tr, groups, num_vars, &sha_fcfg,
        );
        let _ = zinc_piop::combined_poly_resolver::CombinedPolyResolver::<F>::finalize_prover(
            &mut tr, ps.remove(0), snc, &sha_fcfg,
        ).expect("SHA finalize_prover");
        let _ = zinc_piop::combined_poly_resolver::CombinedPolyResolver::<F>::finalize_prover(
            &mut tr, ps.remove(0), enc, &sha_fcfg,
        ).expect("ECDSA finalize_prover");
        tr
    };

    group.bench_function("PIOP/Main field sumcheck", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut tr = unified_tr.clone();
                let t = Instant::now();
                let sha_group = zinc_piop::combined_poly_resolver::CombinedPolyResolver::<F>::build_prover_group::<Sha256Uair>(
                    &mut tr, sha_field_trace.clone(), &ic_evaluation_point,
                    &sha_fproj_scalars, sha_num_constraints, num_vars, sha_max_degree, &sha_fcfg,
                ).expect("SHA build_prover_group");
                let sha_num_cols = sha_group.num_cols;
                let ec_group = zinc_piop::combined_poly_resolver::CombinedPolyResolver::<F>::build_prover_group::<EcdsaUairInt>(
                    &mut tr, ec_field_trace.clone(), &ic_evaluation_point,
                    &ec_fproj_scalars, ecdsa_num_constraints, num_vars, ecdsa_max_degree, &sha_fcfg,
                ).expect("ECDSA build_prover_group");
                let ec_num_cols = ec_group.num_cols;
                let groups = vec![
                    (sha_group.degree, sha_group.mles, sha_group.comb_fn),
                    (ec_group.degree, ec_group.mles, ec_group.comb_fn),
                ];
                let (_, mut prover_states) =
                    zinc_piop::sumcheck::multi_degree::MultiDegreeSumcheck::<F>::prove_as_subprotocol(
                        &mut tr, groups, num_vars, &sha_fcfg,
                    );
                let sha_ps = prover_states.remove(0);
                let _ = zinc_piop::combined_poly_resolver::CombinedPolyResolver::<F>::finalize_prover(
                    &mut tr, sha_ps, sha_num_cols, &sha_fcfg,
                ).expect("SHA finalize_prover");
                let ec_ps = prover_states.remove(0);
                let _ = zinc_piop::combined_poly_resolver::CombinedPolyResolver::<F>::finalize_prover(
                    &mut tr, ec_ps, ec_num_cols, &sha_fcfg,
                ).expect("ECDSA finalize_prover");
                total += t.elapsed();
            }
            total
        });
    });

    // ── 13a. Unified Eval Sumcheck (eq + shift) ──────────────────────
    //
    // After the main field sumcheck multi-degree sumcheck finalizes, a unified batched
    // sumcheck reduces ALL column evaluation claims to a single random
    // point ("point unification"):
    //   • eq-based claims for every SHA-256 up-eval column (shift=0)
    //   • eq-based claims for every ECDSA up-eval column   (shift=0)
    //   • genuine shift claims for ECDSA shifted columns   (shift=1)
    //
    // Pre-compute main field sumcheck state (eval point, up/down evals) and field
    // trace columns.  Only the shift_sumcheck_prove call is timed.
    let ec_sig = EcdsaUairInt::signature();

    // Run a fresh main field sumcheck to obtain the evaluation point, up_evals and down_evals.
    let (cpr_eval_point, sha_up_evals, ec_up_evals, ec_down_evals) = {
        let mut tr = unified_tr.clone();
        let sha_g = zinc_piop::combined_poly_resolver::CombinedPolyResolver::<F>::build_prover_group::<Sha256Uair>(
            &mut tr, sha_field_trace.clone(), &ic_evaluation_point,
            &sha_fproj_scalars, sha_num_constraints, num_vars, sha_max_degree, &sha_fcfg,
        ).expect("SHA build_prover_group");
        let snc = sha_g.num_cols;
        let ec_g = zinc_piop::combined_poly_resolver::CombinedPolyResolver::<F>::build_prover_group::<EcdsaUairInt>(
            &mut tr, ec_field_trace.clone(), &ic_evaluation_point,
            &ec_fproj_scalars, ecdsa_num_constraints, num_vars, ecdsa_max_degree, &sha_fcfg,
        ).expect("ECDSA build_prover_group");
        let enc = ec_g.num_cols;
        let groups = vec![
            (sha_g.degree, sha_g.mles, sha_g.comb_fn),
            (ec_g.degree, ec_g.mles, ec_g.comb_fn),
        ];
        let (_, mut ps) = zinc_piop::sumcheck::multi_degree::MultiDegreeSumcheck::<F>::prove_as_subprotocol(
            &mut tr, groups, num_vars, &sha_fcfg,
        );
        let (sha_up, _, sha_st) = zinc_piop::combined_poly_resolver::CombinedPolyResolver::<F>::finalize_prover(
            &mut tr, ps.remove(0), snc, &sha_fcfg,
        ).expect("SHA finalize_prover");
        let (ec_up, ec_down, _) = zinc_piop::combined_poly_resolver::CombinedPolyResolver::<F>::finalize_prover(
            &mut tr, ps.remove(0), enc, &sha_fcfg,
        ).expect("ECDSA finalize_prover");
        (sha_st.evaluation_point, sha_up, ec_up, ec_down)
    };

    // Capture the post-main-field-sumcheck-finalize transcript state.
    let unified_eval_pre_transcript = {
        let mut tr = unified_tr.clone();
        let sha_g = zinc_piop::combined_poly_resolver::CombinedPolyResolver::<F>::build_prover_group::<Sha256Uair>(
            &mut tr, sha_field_trace.clone(), &ic_evaluation_point,
            &sha_fproj_scalars, sha_num_constraints, num_vars, sha_max_degree, &sha_fcfg,
        ).expect("SHA build_prover_group");
        let snc = sha_g.num_cols;
        let ec_g = zinc_piop::combined_poly_resolver::CombinedPolyResolver::<F>::build_prover_group::<EcdsaUairInt>(
            &mut tr, ec_field_trace.clone(), &ic_evaluation_point,
            &ec_fproj_scalars, ecdsa_num_constraints, num_vars, ecdsa_max_degree, &sha_fcfg,
        ).expect("ECDSA build_prover_group");
        let enc = ec_g.num_cols;
        let groups = vec![
            (sha_g.degree, sha_g.mles, sha_g.comb_fn),
            (ec_g.degree, ec_g.mles, ec_g.comb_fn),
        ];
        let (_, mut ps) = zinc_piop::sumcheck::multi_degree::MultiDegreeSumcheck::<F>::prove_as_subprotocol(
            &mut tr, groups, num_vars, &sha_fcfg,
        );
        let _ = zinc_piop::combined_poly_resolver::CombinedPolyResolver::<F>::finalize_prover(
            &mut tr, ps.remove(0), snc, &sha_fcfg,
        ).expect("SHA finalize_prover");
        let _ = zinc_piop::combined_poly_resolver::CombinedPolyResolver::<F>::finalize_prover(
            &mut tr, ps.remove(0), enc, &sha_fcfg,
        ).expect("ECDSA finalize_prover");
        tr
    };

    // Pre-build the unified trace columns array:
    //   [sha_col_0 … sha_col_{m1-1}, ec_col_0 … ec_col_{m2-1}]
    let c1_num_up = sha_up_evals.len();
    let c2_num_up = ec_up_evals.len();
    let unified_trace_columns: Vec<DenseMultilinearExtension<<F as Field>::Inner>> = {
        let mut v = Vec::with_capacity(c1_num_up + c2_num_up);
        v.extend(sha_field_trace.iter().cloned());
        v.extend(ec_field_trace.iter().cloned());
        v
    };

    // Pre-build the unified claims.
    let unified_claims: Vec<ShiftClaim<F>> = {
        let mut claims = Vec::with_capacity(c1_num_up + c2_num_up + ec_sig.shifts.len());
        // Eq claims for SHA-256 columns.
        for i in 0..c1_num_up {
            claims.push(ShiftClaim {
                source_col: i,
                shift_amount: 0,
                eval_point: cpr_eval_point.clone(),
                claimed_eval: sha_up_evals[i].clone(),
            });
        }
        // Eq claims for ECDSA columns.
        for j in 0..c2_num_up {
            claims.push(ShiftClaim {
                source_col: c1_num_up + j,
                shift_amount: 0,
                eval_point: cpr_eval_point.clone(),
                claimed_eval: ec_up_evals[j].clone(),
            });
        }
        // Shift claims for ECDSA shifted columns.
        for (k, spec) in ec_sig.shifts.iter().enumerate() {
            claims.push(ShiftClaim {
                source_col: c1_num_up + spec.source_col,
                shift_amount: spec.shift_amount,
                eval_point: cpr_eval_point.clone(),
                claimed_eval: ec_down_evals[k].clone(),
            });
        }
        claims
    };

    group.bench_function("PIOP/UnifiedEvalSumcheck", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut tr = unified_eval_pre_transcript.clone();
                let t = Instant::now();
                let _ = black_box(shift_sumcheck_prove(
                    &mut tr,
                    &unified_claims,
                    &unified_trace_columns,
                    num_vars,
                    &sha_fcfg,
                ));
                total += t.elapsed();
            }
            total
        });
    });

    // ── 14. SHA PIOP / Lookup Extract ────────────────────────────────
    //
    // Extract lookup columns from the projected field trace. In
    // the pipeline this cost is included in main_field_sumcheck_time.
    {
        group.bench_function("SHA/PIOP/LookupExtract", |b| {
            b.iter(|| {
                let mut needed: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
                for spec in &sha_lookup_specs {
                    let next = needed.len();
                    needed.entry(spec.column_index).or_insert(next);
                }
                let mut columns: Vec<Vec<F>> = Vec::with_capacity(needed.len());
                let mut raw_indices: Vec<Vec<usize>> = Vec::with_capacity(needed.len());
                for &orig_idx in needed.keys() {
                    let col_f: Vec<F> = sha_field_trace[orig_idx].iter()
                        .map(|inner| F::new_unchecked_with_cfg(inner.clone(), &sha_fcfg)).collect();
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
                let index_map: std::collections::BTreeMap<usize, usize> = needed.keys()
                    .enumerate().map(|(n, &o)| (o, n)).collect();
                let remapped: Vec<LookupColumnSpec> = sha_lookup_specs.iter()
                    .map(|s| LookupColumnSpec { column_index: index_map[&s.column_index], table_type: s.table_type.clone() })
                    .collect();
                black_box((&columns, &raw_indices, &remapped));
            });
        });
    }

    // ── 15. SHA PIOP / Lookup ───────────────────────────────────────
    {
        let sha_tr_lk = unified_tr_post_cpr.clone();

        let mut needed: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
        for spec in &sha_lookup_specs {
            let next = needed.len();
            needed.entry(spec.column_index).or_insert(next);
        }
        let mut columns: Vec<Vec<F>> = Vec::with_capacity(needed.len());
        let mut raw_indices: Vec<Vec<usize>> = Vec::with_capacity(needed.len());
        for &orig_idx in needed.keys() {
            let col_f: Vec<F> = sha_field_trace[orig_idx].iter()
                .map(|inner| F::new_unchecked_with_cfg(inner.clone(), &sha_fcfg)).collect();
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
        let index_map: std::collections::BTreeMap<usize, usize> = needed.keys()
            .enumerate().map(|(n, &o)| (o, n)).collect();
        let remapped: Vec<LookupColumnSpec> = sha_lookup_specs.iter()
            .map(|s| LookupColumnSpec { column_index: index_map[&s.column_index], table_type: s.table_type.clone() })
            .collect();

        group.bench_function("SHA/PIOP/Lookup", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let mut tr = sha_tr_lk.clone();
                    let t = Instant::now();
                    let _ = prove_batched_lookup_with_indices(
                        &mut tr, &columns, &raw_indices, &remapped,
                        &sha_proj_elem, &sha_fcfg,
                    ).expect("lookup");
                    total += t.elapsed();
                }
                total
            });
        });

        // GKR variant for comparison.
        group.bench_function("SHA/PIOP/GkrLookup", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let mut tr = sha_tr_lk.clone();
                    let t = Instant::now();
                    let _ = prove_gkr_batched_lookup_with_indices(
                        &mut tr, &columns, &raw_indices, &remapped,
                        &sha_proj_elem, &sha_fcfg,
                    ).expect("gkr lookup");
                    total += t.elapsed();
                }
                total
            });
        });
    }

    // ── 16. PIOP / Batched Main field sumcheck+Lookup (multi-degree sumcheck A/B) ──
    //
    // Direct comparison: batches SHA CPR + ECDSA CPR + SHA Lookup into
    // a single multi-degree sumcheck (matching prove_classic_logup
    // behaviour). Compare this step against step 13 + step 15 to see
    // whether batching the lookup with the CPR saves prover time.
    //
    // Expected: roughly the same or slightly slower because the lookup
    // MLEs are zero-padded from 2^8 to 2^9 (doubling per-round work
    // for the lookup group and adding one extra round).
    {
        // Pre-compute lookup data outside the timing loop.
        let mut needed_b: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
        for spec in &sha_lookup_specs {
            let next = needed_b.len();
            needed_b.entry(spec.column_index).or_insert(next);
        }
        let mut columns_b: Vec<Vec<F>> = Vec::with_capacity(needed_b.len());
        let mut raw_indices_b: Vec<Vec<usize>> = Vec::with_capacity(needed_b.len());
        for &orig_idx in needed_b.keys() {
            let col_f: Vec<F> = sha_field_trace[orig_idx].iter()
                .map(|inner| F::new_unchecked_with_cfg(inner.clone(), &sha_fcfg)).collect();
            columns_b.push(col_f);
            let col_idx: Vec<usize> = sha_trace[orig_idx].iter().map(|bp| {
                let mut idx = 0usize;
                for (j, coeff) in bp.iter().enumerate() {
                    if coeff.into_inner() { idx |= 1usize << j; }
                }
                idx
            }).collect();
            raw_indices_b.push(col_idx);
        }
        let index_map_b: std::collections::BTreeMap<usize, usize> = needed_b.keys()
            .enumerate().map(|(n, &o)| (o, n)).collect();
        let remapped_b: Vec<LookupColumnSpec> = sha_lookup_specs.iter()
            .map(|s| LookupColumnSpec { column_index: index_map_b[&s.column_index], table_type: s.table_type.clone() })
            .collect();
        let lk_groups_def = group_lookup_specs(&remapped_b);

        group.bench_function("PIOP/Batched Main field sumcheck+Lookup", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let mut tr = unified_tr.clone();
                    let t = Instant::now();

                    // Build SHA CPR group.
                    let sha_cpr = zinc_piop::combined_poly_resolver::CombinedPolyResolver::<F>::build_prover_group::<Sha256Uair>(
                        &mut tr, sha_field_trace.clone(), &ic_evaluation_point,
                        &sha_fproj_scalars, sha_num_constraints, num_vars, sha_max_degree, &sha_fcfg,
                    ).expect("SHA build_prover_group");
                    let sha_nc = sha_cpr.num_cols;

                    // Build ECDSA CPR group.
                    let ec_cpr = zinc_piop::combined_poly_resolver::CombinedPolyResolver::<F>::build_prover_group::<EcdsaUairInt>(
                        &mut tr, ec_field_trace.clone(), &ic_evaluation_point,
                        &ec_fproj_scalars, ecdsa_num_constraints, num_vars, ecdsa_max_degree, &sha_fcfg,
                    ).expect("ECDSA build_prover_group");
                    let ec_nc = ec_cpr.num_cols;

                    // Build lookup group(s).
                    let mut lk_pre: Vec<(usize, Vec<DenseMultilinearExtension<<F as Field>::Inner>>, Box<dyn Fn(&[F]) -> F + Send + Sync>, usize)> = Vec::new();
                    let mut shared_num_vars = num_vars;
                    for grp in &lk_groups_def {
                        let instance = build_lookup_instance_from_indices_pub(
                            &columns_b, &raw_indices_b, grp, &sha_proj_elem, &sha_fcfg,
                        ).expect("lookup instance build");
                        let lk_group = BatchedDecompLogupProtocol::<F>::build_prover_group(
                            &mut tr, &instance, &sha_fcfg,
                        ).expect("lookup build_prover_group");
                        if lk_group.num_vars > shared_num_vars {
                            shared_num_vars = lk_group.num_vars;
                        }
                        lk_pre.push((lk_group.degree, lk_group.mles, lk_group.comb_fn, lk_group.num_vars));
                    }

                    // Assemble multi-degree sumcheck groups.
                    let mut sumcheck_groups: Vec<(
                        usize,
                        Vec<DenseMultilinearExtension<<F as Field>::Inner>>,
                        Box<dyn Fn(&[F]) -> F + Send + Sync>,
                    )> = Vec::with_capacity(2 + lk_pre.len());

                    // Pad CPR MLEs to shared_num_vars if needed.
                    let mut sha_mles = sha_cpr.mles;
                    let mut ec_mles = ec_cpr.mles;
                    if shared_num_vars > num_vars {
                        let target = 1usize << shared_num_vars;
                        for mle in &mut sha_mles {
                            mle.evaluations.resize(target, Default::default());
                            mle.num_vars = shared_num_vars;
                        }
                        for mle in &mut ec_mles {
                            mle.evaluations.resize(target, Default::default());
                            mle.num_vars = shared_num_vars;
                        }
                    }
                    sumcheck_groups.push((sha_cpr.degree, sha_mles, sha_cpr.comb_fn));
                    sumcheck_groups.push((ec_cpr.degree, ec_mles, ec_cpr.comb_fn));

                    // Add lookup groups (pad if needed).
                    for (deg, mut mles, cfn, lk_nv) in lk_pre {
                        if lk_nv < shared_num_vars {
                            let target = 1usize << shared_num_vars;
                            for mle in &mut mles {
                                mle.evaluations.resize(target, Default::default());
                                mle.num_vars = shared_num_vars;
                            }
                        }
                        sumcheck_groups.push((deg, mles, cfn));
                    }

                    // Run the combined multi-degree sumcheck.
                    let (_, mut ps) = MultiDegreeSumcheck::<F>::prove_as_subprotocol(
                        &mut tr, sumcheck_groups, shared_num_vars, &sha_fcfg,
                    );

                    // Finalize CPR groups.
                    let sha_ps = ps.remove(0);
                    let _ = zinc_piop::combined_poly_resolver::CombinedPolyResolver::<F>::finalize_prover(
                        &mut tr, sha_ps, sha_nc, &sha_fcfg,
                    ).expect("SHA finalize");
                    let ec_ps = ps.remove(0);
                    let _ = zinc_piop::combined_poly_resolver::CombinedPolyResolver::<F>::finalize_prover(
                        &mut tr, ec_ps, ec_nc, &sha_fcfg,
                    ).expect("ECDSA finalize");
                    // (lookup finalize omitted — it's proof assembly only)

                    total += t.elapsed();
                }
                total
            });
        });
    }

    // ── 17. SHA PCS Prove ────────────────────────────────────────────
    // r_PCS = r_CPR (truncated to num_vars).
    let sha_pcs_point: Vec<F> = cpr_eval_point[..SHA256_8X_NUM_VARS].to_vec();
    // ECDSA shares the same CPR eval point (multi-degree sumcheck) but
    // may have a different num_vars.
    let _ecdsa_pcs_point: Vec<F> = cpr_eval_point[..ECDSA_NUM_VARS].to_vec();

    let (sha_hint, _) = ZipPlus::<ShaZt, ShaLc>::commit(&sha_params, &sha_pcs_trace).expect("commit");
    group.bench_function("SHA/PCS/Prove", |b| {
        b.iter(|| {
            black_box(
                ZipPlus::<ShaZt, ShaLc>::prove::<F, UNCHECKED>(&sha_params, &sha_pcs_trace, &sha_pcs_point, &sha_hint)
            )
        });
    });

    // ── 18. ECDSA PCS Prove ──────────────────────────────────────────
    // The ECDSA PCS operates over FScalar (256-bit MontyField<4>), not the
    // 192-bit PiopField F.  Convert the shared eval point via the same
    // path used by the dual-circuit pipeline (piop_point_to_pcs_field).
    let ecdsa_pcs_point_f2: Vec<FScalar> = {
        let pcs2_fcfg = zinc_transcript::KeccakTranscript::default()
            .get_random_field_cfg::<FScalar, <EcdsaScalarZipTypes as ZipTypes>::Fmod, <EcdsaScalarZipTypes as ZipTypes>::PrimeTest>();
        zinc_snark::pipeline::piop_point_to_pcs_field(&cpr_eval_point[..ECDSA_NUM_VARS], &pcs2_fcfg)
    };
    let (ec_hint, _) = ZipPlus::<EcZt, EcLc>::commit(&ec_params, &ecdsa_pcs_trace).expect("commit");
    group.bench_function("ECDSA/PCS/Prove", |b| {
        b.iter(|| {
            black_box(
                ZipPlus::<EcZt, EcLc>::prove::<FScalar, UNCHECKED>(&ec_params, &ecdsa_pcs_trace, &ecdsa_pcs_point_f2, &ec_hint)
            )
        });
    });

    // ── 18b. SHA PCS Prove (folded — BinaryPoly<16>) ──────────────
    //
    // For the folded PCS, the evaluation point has one extra coordinate
    // (γ from the folding protocol). We append a dummy extra coordinate.
    let folded_sha_pcs_point: Vec<F> = {
        let mut pt = sha_pcs_point.clone();
        pt.push(F::one_with_cfg(&sha_fcfg)); // placeholder γ
        pt
    };

    let (folded_sha_hint, _) = ZipPlus::<FoldedShaZt, FoldedShaLc>::commit(
        &folded_sha_params, &sha_split_trace,
    ).expect("commit");

    group.bench_function("SHA/PCS/Prove (folded)", |b| {
        b.iter(|| {
            black_box(
                ZipPlus::<FoldedShaZt, FoldedShaLc>::prove::<F, UNCHECKED>(
                    &folded_sha_params, &sha_split_trace, &folded_sha_pcs_point, &folded_sha_hint,
                )
            )
        });
    });

    // ── 19. E2E Total Prover (unified dual-circuit pipeline) ─────
    //
    // Uses prove_dual_circuit which shares one transcript, one IC eval
    // point, one projecting element, and one multi-degree sumcheck
    // (SHA CPR + ECDSA CPR + SHA lookup) across both circuits.
    group.bench_function("E2E/Prover", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let t = Instant::now();
                let _ = zinc_snark::pipeline::prove_dual_circuit::<
                    Sha256Uair,
                    EcdsaUairInt,
                    Int<{ INT_LIMBS * 4 }>,
                    ShaZt, ShaLc,
                    EcZt, EcLc,
                    FScalar,
                    32,
                    UNCHECKED,
                >(
                    &sha_params, &sha_trace,
                    &ec_params, &ecdsa_trace,
                    SHA256_8X_NUM_VARS,
                    &sha_lookup_specs,
                    &sha_affine_specs,
                );
                total += t.elapsed();
            }
            total
        });
    });

    // ── 20. E2E Total Verifier (unified dual-circuit pipeline) ──────
    let dual_proof = zinc_snark::pipeline::prove_dual_circuit::<
        Sha256Uair,
        EcdsaUairInt,
        Int<{ INT_LIMBS * 4 }>,
        ShaZt, ShaLc,
        EcZt, EcLc,
        FScalar,
        32,
        UNCHECKED,
    >(
        &sha_params, &sha_trace,
        &ec_params, &ecdsa_trace,
        SHA256_8X_NUM_VARS,
        &sha_lookup_specs,
        &sha_affine_specs,
    );

    // Build public column data for the verifier.
    let sha_public_column_data: Vec<_> = sha_sig.public_columns.iter()
        .map(|&i| sha_trace[i].clone())
        .collect();
    let ecdsa_public_column_data: Vec<_> = ec_sig_int.public_columns.iter()
        .map(|&i| ecdsa_trace[i].clone())
        .collect();

    // ── Verifier-side booleanity check on public columns b1, b2 ────
    //
    // The ECDSA UAIR declares b1 and b2 as public columns.  Instead of
    // burning two algebraic constraints (degree-2 each) to enforce
    // booleanity inside the constraint system, the verifier checks it
    // directly on the raw column data.  This is sound because the
    // columns are public: the verifier already possesses every
    // evaluation, so a per-element {0,1} check is O(N) field
    // comparisons -- negligible compared to PCS verification.
    let check_boolean_column = |col: &DenseMultilinearExtension<Int<{ INT_LIMBS * 4 }>>| {
        let zero = Int::<{ INT_LIMBS * 4 }>::default();
        let one  = Int::<{ INT_LIMBS * 4 }>::from_ref(&1i64);
        for (i, v) in col.evaluations.iter().enumerate() {
            assert!(
                *v == zero || *v == one,
                "ECDSA public column not boolean at row {i}: {v:?}"
            );
        }
    };
    // public_columns = [COL_B1, COL_B2, COL_SEL_INIT, COL_SEL_FINAL]
    check_boolean_column(&ecdsa_public_column_data[0]);
    check_boolean_column(&ecdsa_public_column_data[1]);

    // ── SHA-256 Feed-Forward & Hash-to-ECDSA Connection ─────────────
    //
    // Two verifier-side computations that close the gap between the
    // proved SHA-256 compression and the proved ECDSA verification:
    //
    //  (A) **Feed-forward**: the SHA-256 UAIR proves the 64-round state
    //      update.  The final hash digest additionally requires:
    //        digest[i] = H_init[i] + state_final[i]  (wrapping mod 2^32)
    //      The verifier extracts the 8 final working variables from the
    //      trace and performs 8 wrapping additions per SHA instance.
    //
    //  (B) **Hash-to-ECDSA connection**: the ECDSA UAIR uses scalars
    //      u1 = e * s^{-1} mod n and u2 = r * s^{-1} mod n, where e is
    //      the message hash output by SHA-256.  The verifier:
    //        1. Reconstructs u1 from the public b1 column bits.
    //        2. Computes expected u1 = digest * s^{-1} mod n.
    //        3. Asserts equality.
    //
    // In a production deployment the SHA trace rows storing the final
    // working variables would be made public or opened via additional
    // PCS queries.  Here we benchmark the raw computational cost.

    /// SHA-256 feed-forward for one instance starting at `base` row.
    ///
    /// Final working variables after 64 rounds:
    ///   a = col_a[base+64], b = col_a[base+63], c = col_a[base+62], d = col_a[base+61]
    ///   e = col_e[base+64], f = col_e[base+63], g = col_e[base+62], h = col_e[base+61]
    let feed_forward = |trace: &[DenseMultilinearExtension<BinaryPoly<32>>],
                         base: usize| -> [u32; 8] {
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

    /// Reconstruct u1 scalar from ECDSA b1 column bits as 32 big-endian bytes.
    ///
    /// The ECDSA scalar-mul loop rows (0-indexed 1..256) process bits
    /// 256..1 of u1.  Bit-index at row t is (257 - t).
    let reconstruct_u1 = |ec_trace: &[DenseMultilinearExtension<Int<{ INT_LIMBS * 4 }>>]| -> [u8; 32] {
        let zero_int = Int::<{ INT_LIMBS * 4 }>::default();
        let mut scalar = [0u8; 32];
        let n = ec_trace[zinc_ecdsa_uair::COL_B1].evaluations.len();
        for t in 1..257.min(n) {
            if ec_trace[zinc_ecdsa_uair::COL_B1].evaluations[t] != zero_int {
                let bit_idx = 257 - t; // 256 down to 1
                if bit_idx > 0 && bit_idx <= 256 {
                    let byte_pos = (bit_idx - 1) / 8;
                    let bit_pos  = (bit_idx - 1) % 8;
                    scalar[31 - byte_pos] |= 1u8 << bit_pos;
                }
            }
        }
        scalar
    };

    // Pre-compute feed-forward for instance 0.
    let sha_digest = feed_forward(&sha_trace, 0);

    // Sanity check: SHA-256("") = e3b0c442 98fc1c14 9afbf4c8 996fb924
    //                              27ae41e4 649b934c a495991b 7852b855
    {
        let expected: [u32; 8] = [
            0xe3b0c442, 0x98fc1c14, 0x9afbf4c8, 0x996fb924,
            0x27ae41e4, 0x649b934c, 0xa495991b, 0x7852b855,
        ];
        assert_eq!(
            sha_digest, expected,
            "SHA-256 feed-forward sanity check failed"
        );
    }

    // Reconstruct ECDSA u1 from b1 bits.
    let ecdsa_u1 = reconstruct_u1(&ecdsa_trace);

    // In a real deployment the verifier would also check:
    //   u1 == sha_digest_as_scalar * s^{-1}  (mod secp256k1 order)
    // using the public signature parameter s.  We skip the modular-
    // arithmetic assertion and only measure the extraction +
    // reconstruction cost.
    eprintln!("  SHA-256 digest (instance 0): {:08x?}", sha_digest);
    eprintln!("  ECDSA u1 (from b1 bits):    {}",
        ecdsa_u1.iter().map(|b| format!("{b:02x}")).collect::<String>());

    group.bench_function("E2E/Verifier", |b| {
        b.iter(|| {
            let r = zinc_snark::pipeline::verify_dual_circuit::<
                Sha256Uair,
                EcdsaUairInt,
                Int<{ INT_LIMBS * 4 }>,
                ShaZt, ShaLc,
                EcZt, EcLc,
                FScalar,
                32,
                UNCHECKED,
                zinc_snark::pipeline::TrivialIdeal, _,
                zinc_snark::pipeline::TrivialIdeal, _,
            >(
                &sha_params, &ec_params,
                &dual_proof,
                SHA256_8X_NUM_VARS,
                |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
                |_: &IdealOrZero<ImpossibleIdeal>| zinc_snark::pipeline::TrivialIdeal,
                &sha_public_column_data,
                &ecdsa_public_column_data,
            );
            let _ = black_box(r);

            // Feed-forward: compute SHA-256 digest from trace.
            let digest = black_box(feed_forward(&sha_trace, 0));

            // Hash→ECDSA connection: reconstruct u₁ from b₁ bits.
            let u1 = black_box(reconstruct_u1(&ecdsa_trace));
        });
    });

    // ── 21. E2E Total Prover (folded SHA PCS) ────────────────────
    //
    // Same as E2E/Prover but uses prove_dual_circuit_folded, which
    // commits the SHA trace as split BinaryPoly<16> columns (folding
    // BinaryPoly<32>→BinaryPoly<16>) so each PCS opening is ~2× smaller.
    group.bench_function("E2E/Prover (folded SHA)", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let t = Instant::now();
                let _ = zinc_snark::pipeline::prove_dual_circuit_folded::<
                    Sha256Uair,
                    EcdsaUairInt,
                    Int<{ INT_LIMBS * 4 }>,
                    FoldedShaZt, FoldedShaLc,
                    EcZt, EcLc,
                    FScalar,
                    32, 16,
                    UNCHECKED,
                >(
                    &folded_sha_params, &sha_trace,
                    &ec_params, &ecdsa_trace,
                    SHA256_8X_NUM_VARS,
                    &sha_lookup_specs,
                    &sha_affine_specs,
                );
                total += t.elapsed();
            }
            total
        });
    });

    // ── 22. E2E Total Verifier (folded SHA PCS) ───────────────────
    let folded_dual_proof = zinc_snark::pipeline::prove_dual_circuit_folded::<
        Sha256Uair,
        EcdsaUairInt,
        Int<{ INT_LIMBS * 4 }>,
        FoldedShaZt, FoldedShaLc,
        EcZt, EcLc,
        FScalar,
        32, 16,
        UNCHECKED,
    >(
        &folded_sha_params, &sha_trace,
        &ec_params, &ecdsa_trace,
        SHA256_8X_NUM_VARS,
        &sha_lookup_specs,
        &sha_affine_specs,
    );

    group.bench_function("E2E/Verifier (folded SHA)", |b| {
        b.iter(|| {
            let r = zinc_snark::pipeline::verify_dual_circuit_folded::<
                Sha256Uair,
                EcdsaUairInt,
                Int<{ INT_LIMBS * 4 }>,
                FoldedShaZt, FoldedShaLc,
                EcZt, EcLc,
                FScalar,
                32, 16,
                UNCHECKED,
                zinc_snark::pipeline::TrivialIdeal, _,
                zinc_snark::pipeline::TrivialIdeal, _,
            >(
                &folded_sha_params, &ec_params,
                &folded_dual_proof,
                SHA256_8X_NUM_VARS,
                |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
                |_: &IdealOrZero<ImpossibleIdeal>| zinc_snark::pipeline::TrivialIdeal,
                &sha_public_column_data,
                &ecdsa_public_column_data,
            );
            // IC₂ absorb-only: ECDSA constraints hold mod secp256k1 p,
            // not the 192-bit PIOP prime, so we absorb but skip the
            // ideal membership check.  All other verification steps
            // (PCS, CPR, lookup) still execute normally.
            let _ = black_box(r);

            // Feed-forward: compute SHA-256 digest from trace.
            let digest = black_box(feed_forward(&sha_trace, 0));

            // Hash→ECDSA connection: reconstruct u₁ from b₁ bits.
            let u1 = black_box(reconstruct_u1(&ecdsa_trace));
        });
    });

    // ── Folded SHA proof size breakdown + comparison ─────────────
    {
        use zinc_snark::pipeline::FIELD_LIMBS;
        use crypto_primitives::crypto_bigint_uint::Uint;
        let fe_bytes = <Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;

        fn write_fe_folded(buf: &mut Vec<u8>, f: &zinc_snark::pipeline::PiopField) {
            use zinc_snark::pipeline::FIELD_LIMBS;
            use crypto_primitives::crypto_bigint_uint::Uint;
            use zinc_transcript::traits::ConstTranscribable;
            let sz = <Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;
            let start = buf.len();
            buf.resize(start + sz, 0);
            f.inner().write_transcription_bytes(&mut buf[start..]);
        }

        // PCS bytes.
        let orig_pcs1  = dual_proof.pcs1_proof_bytes.len();
        let folded_pcs1 = folded_dual_proof.pcs1_proof_bytes.len();
        let pcs2_bytes  = folded_dual_proof.pcs2_proof_bytes.len(); // unchanged

        // Folding overhead (c1s + c2s).
        let folding_bytes: usize =
            folded_dual_proof.pcs1_folding_c1s_bytes.iter().map(|v| v.len()).sum::<usize>()
            + folded_dual_proof.pcs1_folding_c2s_bytes.iter().map(|v| v.len()).sum::<usize>();

        // IC bytes (same PIOP, identical to original).
        let ic1_bytes: usize = folded_dual_proof.ic1_proof_values.iter().map(|v| v.len()).sum();
        let ic2_bytes: usize = folded_dual_proof.ic2_proof_values.iter().map(|v| v.len()).sum();

        // MD sumcheck (same).
        let md_msg_bytes: usize = folded_dual_proof.md_group_messages.iter()
            .flat_map(|grp| grp.iter()).map(|v| v.len()).sum();
        let md_sum_bytes: usize = folded_dual_proof.md_claimed_sums.iter().map(|v| v.len()).sum();
        let md_total = md_msg_bytes + md_sum_bytes;

        // CPR evals (same).
        let cpr1_up: usize = folded_dual_proof.cpr1_up_evals.iter().map(|v| v.len()).sum();
        let cpr1_dn: usize = folded_dual_proof.cpr1_down_evals.iter().map(|v| v.len()).sum();
        let cpr2_up: usize = folded_dual_proof.cpr2_up_evals.iter().map(|v| v.len()).sum();
        let cpr2_dn: usize = folded_dual_proof.cpr2_down_evals.iter().map(|v| v.len()).sum();
        let cpr_total = cpr1_up + cpr1_dn + cpr2_up + cpr2_dn;

        // Unified eval sumcheck (same).
        let eval_sc_bytes: usize = folded_dual_proof.unified_eval_sumcheck.as_ref().map_or(0, |sc| {
            sc.rounds.iter().map(|v| v.len()).sum::<usize>()
                + sc.v_finals.iter().map(|v| v.len()).sum::<usize>()
        });

        // Lookup (same).
        let lookup_bytes: usize = folded_dual_proof.lookup_group_proofs.iter().map(|gp| {
            let mults: usize = gp.aggregated_multiplicities.iter().map(|v| v.len()).sum();
            let inv_w: usize = gp.chunk_inverse_witnesses.iter()
                .flat_map(|outer| outer.iter()).map(|inner| inner.len()).sum();
            let inv_t = gp.inverse_table.len();
            (mults + inv_w + inv_t) * fe_bytes
        }).sum();

        // Eval point + PCS evals.
        let eval_pt_bytes: usize = folded_dual_proof.evaluation_point_bytes.iter().map(|v| v.len()).sum();
        let pcs1_eval_bytes: usize = folded_dual_proof.pcs1_evals_bytes.iter().map(|v| v.len()).sum();
        let pcs2_eval_bytes: usize = folded_dual_proof.pcs2_evals_bytes.iter().map(|v| v.len()).sum();

        let piop_total = ic1_bytes + ic2_bytes + md_total + cpr_total
            + eval_sc_bytes + lookup_bytes + eval_pt_bytes
            + pcs1_eval_bytes + pcs2_eval_bytes + folding_bytes;
        let total_raw = folded_pcs1 + pcs2_bytes + piop_total;

        // Serialize all folded proof bytes for compression.
        let mut folded_all_bytes = Vec::with_capacity(total_raw);
        folded_all_bytes.extend(&folded_dual_proof.pcs1_proof_bytes);
        folded_all_bytes.extend(&folded_dual_proof.pcs2_proof_bytes);
        for v in &folded_dual_proof.ic1_proof_values  { folded_all_bytes.extend(v); }
        for v in &folded_dual_proof.ic2_proof_values  { folded_all_bytes.extend(v); }
        for grp in &folded_dual_proof.md_group_messages {
            for v in grp { folded_all_bytes.extend(v); }
        }
        for v in &folded_dual_proof.md_claimed_sums { folded_all_bytes.extend(v); }
        for v in &folded_dual_proof.cpr1_up_evals   { folded_all_bytes.extend(v); }
        for v in &folded_dual_proof.cpr1_down_evals  { folded_all_bytes.extend(v); }
        for v in &folded_dual_proof.cpr2_up_evals   { folded_all_bytes.extend(v); }
        for v in &folded_dual_proof.cpr2_down_evals  { folded_all_bytes.extend(v); }
        if let Some(ref sc) = folded_dual_proof.unified_eval_sumcheck {
            for v in &sc.rounds   { folded_all_bytes.extend(v); }
            for v in &sc.v_finals { folded_all_bytes.extend(v); }
        }
        for gp in &folded_dual_proof.lookup_group_proofs {
            for v in &gp.aggregated_multiplicities {
                for f in v { write_fe_folded(&mut folded_all_bytes, f); }
            }
            for outer in &gp.chunk_inverse_witnesses {
                for inner in outer {
                    for f in inner { write_fe_folded(&mut folded_all_bytes, f); }
                }
            }
            for f in &gp.inverse_table { write_fe_folded(&mut folded_all_bytes, f); }
        }
        for v in &folded_dual_proof.evaluation_point_bytes { folded_all_bytes.extend(v); }
        for v in &folded_dual_proof.pcs1_evals_bytes  { folded_all_bytes.extend(v); }
        for v in &folded_dual_proof.pcs2_evals_bytes  { folded_all_bytes.extend(v); }
        for v in &folded_dual_proof.pcs1_folding_c1s_bytes { folded_all_bytes.extend(v); }
        for v in &folded_dual_proof.pcs1_folding_c2s_bytes { folded_all_bytes.extend(v); }

        let folded_compressed = {
            use std::io::Write;
            let mut enc = flate2::write::DeflateEncoder::new(
                Vec::new(), flate2::Compression::default(),
            );
            enc.write_all(&folded_all_bytes).unwrap();
            enc.finish().unwrap()
        };

        // We need the original totals too — fetch from the outer scope.
        // Recompute them here from dual_proof to keep the block self-contained.
        let orig_pcs2 = dual_proof.pcs2_proof_bytes.len();
        let orig_ic1: usize  = dual_proof.ic1_proof_values.iter().map(|v| v.len()).sum();
        let orig_ic2: usize  = dual_proof.ic2_proof_values.iter().map(|v| v.len()).sum();
        let orig_md: usize   = dual_proof.md_group_messages.iter().flat_map(|g| g.iter()).map(|v| v.len()).sum::<usize>()
                              + dual_proof.md_claimed_sums.iter().map(|v| v.len()).sum::<usize>();
        let orig_cpr: usize  = dual_proof.cpr1_up_evals.iter().map(|v| v.len()).sum::<usize>()
                              + dual_proof.cpr1_down_evals.iter().map(|v| v.len()).sum::<usize>()
                              + dual_proof.cpr2_up_evals.iter().map(|v| v.len()).sum::<usize>()
                              + dual_proof.cpr2_down_evals.iter().map(|v| v.len()).sum::<usize>();
        let orig_eval_sc: usize = dual_proof.unified_eval_sumcheck.as_ref().map_or(0, |sc| {
            sc.rounds.iter().map(|v| v.len()).sum::<usize>()
                + sc.v_finals.iter().map(|v| v.len()).sum::<usize>()
        });
        let orig_lookup: usize = dual_proof.lookup_group_proofs.iter().map(|gp| {
            let mults: usize = gp.aggregated_multiplicities.iter().map(|v| v.len()).sum();
            let inv_w: usize = gp.chunk_inverse_witnesses.iter()
                .flat_map(|outer| outer.iter()).map(|inner| inner.len()).sum();
            (mults + inv_w + gp.inverse_table.len()) * fe_bytes
        }).sum();
        let orig_eval_pt: usize = dual_proof.evaluation_point_bytes.iter().map(|v| v.len()).sum();
        let orig_pcs_evals: usize = dual_proof.pcs1_evals_bytes.iter().map(|v| v.len()).sum::<usize>()
                                  + dual_proof.pcs2_evals_bytes.iter().map(|v| v.len()).sum::<usize>();
        let orig_piop = orig_ic1 + orig_ic2 + orig_md + orig_cpr + orig_eval_sc
                      + orig_lookup + orig_eval_pt + orig_pcs_evals;
        let orig_total_raw = orig_pcs1 + orig_pcs2 + orig_piop;

        eprintln!("\n=== Folded SHA Dual-Circuit Proof Size ===");
        eprintln!("  PCS (SHA folded):  {:>7} B  ({:.1} KB)   [was {} B / {:.1} KB]",
            folded_pcs1, folded_pcs1 as f64 / 1024.0, orig_pcs1, orig_pcs1 as f64 / 1024.0);
        eprintln!("  PCS (ECDSA):       {:>7} B  ({:.1} KB)", pcs2_bytes, pcs2_bytes as f64 / 1024.0);
        eprintln!("  Folding overhead:  {:>7} B  (c1s+c2s)", folding_bytes);
        eprintln!("  IC (SHA):          {:>7} B", ic1_bytes);
        eprintln!("  IC (ECDSA):        {:>7} B", ic2_bytes);
        eprintln!("  MD sumcheck:       {:>7} B  (msgs={}, sums={})", md_total, md_msg_bytes, md_sum_bytes);
        eprintln!("  CPR evals:         {:>7} B  (c1_up={}, c1_dn={}, c2_up={}, c2_dn={})",
            cpr_total, cpr1_up, cpr1_dn, cpr2_up, cpr2_dn);
        eprintln!("  Eval sumcheck:     {:>7} B", eval_sc_bytes);
        eprintln!("  Lookup:            {:>7} B  ({} groups)", lookup_bytes, folded_dual_proof.lookup_group_proofs.len());
        eprintln!("  Eval point:        {:>7} B", eval_pt_bytes);
        eprintln!("  PCS evals:         {:>7} B  (c1={}, c2={})", pcs1_eval_bytes + pcs2_eval_bytes, pcs1_eval_bytes, pcs2_eval_bytes);
        eprintln!("  ─────────────────────────────────────────");
        eprintln!("  PIOP total:        {:>7} B  ({:.1} KB)", piop_total, piop_total as f64 / 1024.0);
        eprintln!("  Total raw:         {:>7} B  ({:.1} KB)   [was {} B / {:.1} KB]",
            total_raw, total_raw as f64 / 1024.0, orig_total_raw, orig_total_raw as f64 / 1024.0);
        eprintln!("  Compressed:        {:>7} B  ({:.1} KB, {:.1}x ratio)",
            folded_compressed.len(), folded_compressed.len() as f64 / 1024.0,
            folded_all_bytes.len() as f64 / folded_compressed.len() as f64);
        let raw_savings    = orig_total_raw as i64 - total_raw as i64;
        eprintln!("  ─────────────────────────────────────────");
        eprintln!("  Raw savings:       {:>+7} B  ({:+.1} KB, {:.2}x ratio)",
            raw_savings, raw_savings as f64 / 1024.0,
            orig_total_raw as f64 / total_raw as f64);
    }

    // ── 23. E2E Total Prover (4x-folded SHA PCS) ─────────────────
    //
    // Same as E2E/Prover (folded SHA) but applies two rounds of
    // fold_claims_prove and commits the SHA trace as BinaryPoly<8>
    // columns (BinaryPoly<32>→BinaryPoly<16>→BinaryPoly<8>).
    group.bench_function("E2E/Prover (4x-folded SHA)", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let t = Instant::now();
                let _ = zinc_snark::pipeline::prove_dual_circuit_4x_folded::<
                    Sha256Uair,
                    EcdsaUairInt,
                    Int<{ INT_LIMBS * 4 }>,
                    FoldedSha4xZt, FoldedSha4xLc,
                    EcZt, EcLc,
                    FScalar,
                    32, 16, 8,
                    UNCHECKED,
                >(
                    &folded_4x_sha_params, &sha_trace,
                    &ec_params, &ecdsa_trace,
                    SHA256_8X_NUM_VARS,
                    &sha_lookup_specs,
                    &sha_affine_specs,
                );
                total += t.elapsed();
            }
            total
        });
    });

    // ── 24. E2E Total Verifier (4x-folded SHA PCS) ───────────────
    let folded_4x_dual_proof = zinc_snark::pipeline::prove_dual_circuit_4x_folded::<
        Sha256Uair,
        EcdsaUairInt,
        Int<{ INT_LIMBS * 4 }>,
        FoldedSha4xZt, FoldedSha4xLc,
        EcZt, EcLc,
        FScalar,
        32, 16, 8,
        UNCHECKED,
    >(
        &folded_4x_sha_params, &sha_trace,
        &ec_params, &ecdsa_trace,
        SHA256_8X_NUM_VARS,
        &sha_lookup_specs,
        &sha_affine_specs,
    );

    group.bench_function("E2E/Verifier (4x-folded SHA)", |b| {
        b.iter(|| {
            let r = zinc_snark::pipeline::verify_dual_circuit_4x_folded::<
                Sha256Uair,
                EcdsaUairInt,
                Int<{ INT_LIMBS * 4 }>,
                FoldedSha4xZt, FoldedSha4xLc,
                EcZt, EcLc,
                FScalar,
                32, 16, 8,
                UNCHECKED,
                zinc_snark::pipeline::TrivialIdeal, _,
                zinc_snark::pipeline::TrivialIdeal, _,
            >(
                &folded_4x_sha_params, &ec_params,
                &folded_4x_dual_proof,
                SHA256_8X_NUM_VARS,
                |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
                |_: &IdealOrZero<ImpossibleIdeal>| zinc_snark::pipeline::TrivialIdeal,
                &sha_public_column_data,
                &ecdsa_public_column_data,
            );
            let _ = black_box(r);

            // Feed-forward: compute SHA-256 digest from trace.
            let digest = black_box(feed_forward(&sha_trace, 0));

            // Hash→ECDSA connection: reconstruct u₁ from b₁ bits.
            let u1 = black_box(reconstruct_u1(&ecdsa_trace));
        });
    });

    // ── 24b. E2E Prover/Verifier (folded SHA, 4-chunk lookup) ──────
    //
    // Same as E2E/Prover (folded SHA) but with chunk_width=8 → 4 chunks
    // of 256 entries each (vs. the default 8 chunks of 16).
    let sha_lookup_specs_4c = sha256_lookup_specs_4chunks();
    let sha_affine_specs_4c = sha256_affine_lookup_specs_4chunks();

    group.bench_function("E2E/Prover (folded SHA 4-chunk)", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let t = Instant::now();
                let _ = zinc_snark::pipeline::prove_dual_circuit_folded::<
                    Sha256Uair,
                    EcdsaUairInt,
                    Int<{ INT_LIMBS * 4 }>,
                    FoldedShaZt, FoldedShaLc,
                    EcZt, EcLc,
                    FScalar,
                    32, 16,
                    UNCHECKED,
                >(
                    &folded_sha_params, &sha_trace,
                    &ec_params, &ecdsa_trace,
                    SHA256_8X_NUM_VARS,
                    &sha_lookup_specs_4c,
                    &sha_affine_specs_4c,
                );
                total += t.elapsed();
            }
            total
        });
    });

    let folded_dual_proof_4c = zinc_snark::pipeline::prove_dual_circuit_folded::<
        Sha256Uair,
        EcdsaUairInt,
        Int<{ INT_LIMBS * 4 }>,
        FoldedShaZt, FoldedShaLc,
        EcZt, EcLc,
        FScalar,
        32, 16,
        UNCHECKED,
    >(
        &folded_sha_params, &sha_trace,
        &ec_params, &ecdsa_trace,
        SHA256_8X_NUM_VARS,
        &sha_lookup_specs_4c,
        &sha_affine_specs_4c,
    );

    group.bench_function("E2E/Verifier (folded SHA 4-chunk)", |b| {
        b.iter(|| {
            let r = zinc_snark::pipeline::verify_dual_circuit_folded::<
                Sha256Uair,
                EcdsaUairInt,
                Int<{ INT_LIMBS * 4 }>,
                FoldedShaZt, FoldedShaLc,
                EcZt, EcLc,
                FScalar,
                32, 16,
                UNCHECKED,
                zinc_snark::pipeline::TrivialIdeal, _,
                zinc_snark::pipeline::TrivialIdeal, _,
            >(
                &folded_sha_params, &ec_params,
                &folded_dual_proof_4c,
                SHA256_8X_NUM_VARS,
                |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
                |_: &IdealOrZero<ImpossibleIdeal>| zinc_snark::pipeline::TrivialIdeal,
                &sha_public_column_data,
                &ecdsa_public_column_data,
            );
            let _ = black_box(r);

            // Feed-forward: compute SHA-256 digest from trace.
            let digest = black_box(feed_forward(&sha_trace, 0));

            // Hash→ECDSA connection: reconstruct u₁ from b₁ bits.
            let u1 = black_box(reconstruct_u1(&ecdsa_trace));
        });
    });

    // ── 24c. E2E Prover/Verifier (4x-folded SHA, 4-chunk lookup) ───
    group.bench_function("E2E/Prover (4x-folded SHA 4-chunk)", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let t = Instant::now();
                let _ = zinc_snark::pipeline::prove_dual_circuit_4x_folded::<
                    Sha256Uair,
                    EcdsaUairInt,
                    Int<{ INT_LIMBS * 4 }>,
                    FoldedSha4xZt, FoldedSha4xLc,
                    EcZt, EcLc,
                    FScalar,
                    32, 16, 8,
                    UNCHECKED,
                >(
                    &folded_4x_sha_params, &sha_trace,
                    &ec_params, &ecdsa_trace,
                    SHA256_8X_NUM_VARS,
                    &sha_lookup_specs_4c,
                    &sha_affine_specs_4c,
                );
                total += t.elapsed();
            }
            total
        });
    });

    let folded_4x_dual_proof_4c = zinc_snark::pipeline::prove_dual_circuit_4x_folded::<
        Sha256Uair,
        EcdsaUairInt,
        Int<{ INT_LIMBS * 4 }>,
        FoldedSha4xZt, FoldedSha4xLc,
        EcZt, EcLc,
        FScalar,
        32, 16, 8,
        UNCHECKED,
    >(
        &folded_4x_sha_params, &sha_trace,
        &ec_params, &ecdsa_trace,
        SHA256_8X_NUM_VARS,
        &sha_lookup_specs_4c,
        &sha_affine_specs_4c,
    );

    group.bench_function("E2E/Verifier (4x-folded SHA 4-chunk)", |b| {
        b.iter(|| {
            let r = zinc_snark::pipeline::verify_dual_circuit_4x_folded::<
                Sha256Uair,
                EcdsaUairInt,
                Int<{ INT_LIMBS * 4 }>,
                FoldedSha4xZt, FoldedSha4xLc,
                EcZt, EcLc,
                FScalar,
                32, 16, 8,
                UNCHECKED,
                zinc_snark::pipeline::TrivialIdeal, _,
                zinc_snark::pipeline::TrivialIdeal, _,
            >(
                &folded_4x_sha_params, &ec_params,
                &folded_4x_dual_proof_4c,
                SHA256_8X_NUM_VARS,
                |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
                |_: &IdealOrZero<ImpossibleIdeal>| zinc_snark::pipeline::TrivialIdeal,
                &sha_public_column_data,
                &ecdsa_public_column_data,
            );
            let _ = black_box(r);

            // Feed-forward: compute SHA-256 digest from trace.
            let digest = black_box(feed_forward(&sha_trace, 0));

            // Hash→ECDSA connection: reconstruct u₁ from b₁ bits.
            let u1 = black_box(reconstruct_u1(&ecdsa_trace));
        });
    });

    // ── 24d. E2E Prover/Verifier (4x folded, 4-chunk, Hybrid GKR c=2)
    //
    // SHA-256 single-circuit prover benchmark + dual-circuit (SHA + ECDSA)
    // verifier benchmark, both using the hybrid GKR lookup protocol with
    // cutoff c=2 and 4-chunk decomposition.
    let hybrid_4x_proof = zinc_snark::pipeline::prove_hybrid_gkr_logup_4x_folded::<
        Sha256Uair, FoldedSha4xZt, FoldedSha4xLc, 32, 16, 8, UNCHECKED,
    >(
        &folded_4x_sha_params, &sha_trace, SHA256_8X_NUM_VARS,
        &sha_lookup_specs_4c, &sha_affine_specs_4c, 2,
    );

    group.bench_function("E2E/Prover (4x Hybrid GKR c=2 4-chunk)", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let t = Instant::now();
                let _ = zinc_snark::pipeline::prove_hybrid_gkr_logup_4x_folded::<
                    Sha256Uair, FoldedSha4xZt, FoldedSha4xLc, 32, 16, 8, UNCHECKED,
                >(
                    &folded_4x_sha_params, &sha_trace, SHA256_8X_NUM_VARS,
                    &sha_lookup_specs_4c, &sha_affine_specs_4c, 2,
                );
                total += t.elapsed();
            }
            total
        });
    });

    // ── 24e. E2E Prover/Verifier (4x folded, 4-chunk, Hybrid GKR c=2, Dual-Circuit SHA+ECDSA)
    //
    // Full dual-circuit prover + verifier using the hybrid GKR lookup
    // protocol with cutoff c=2 and 4-chunk decomposition over both
    // the SHA-256 and ECDSA UAIRs.
    group.bench_function("E2E/Prover (4x Hybrid GKR c=2 4-chunk Dual)", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let t = Instant::now();
                let _ = zinc_snark::pipeline::prove_dual_circuit_hybrid_gkr_4x_folded::<
                    Sha256Uair,
                    EcdsaUairInt,
                    Int<{ INT_LIMBS * 4 }>,
                    FoldedSha4xZt, FoldedSha4xLc,
                    EcZt, EcLc,
                    FScalar,
                    32, 16, 8,
                    UNCHECKED,
                >(
                    &folded_4x_sha_params, &sha_trace,
                    &ec_params, &ecdsa_trace,
                    SHA256_8X_NUM_VARS,
                    &sha_lookup_specs_4c,
                    &sha_affine_specs_4c,
                    2,
                );
                total += t.elapsed();
            }
            total
        });
    });

    let hybrid_4x_dual_proof = zinc_snark::pipeline::prove_dual_circuit_hybrid_gkr_4x_folded::<
        Sha256Uair,
        EcdsaUairInt,
        Int<{ INT_LIMBS * 4 }>,
        FoldedSha4xZt, FoldedSha4xLc,
        EcZt, EcLc,
        FScalar,
        32, 16, 8,
        UNCHECKED,
    >(
        &folded_4x_sha_params, &sha_trace,
        &ec_params, &ecdsa_trace,
        SHA256_8X_NUM_VARS,
        &sha_lookup_specs_4c,
        &sha_affine_specs_4c,
        2,
    );

    group.bench_function("E2E/Verifier (4x Hybrid GKR c=2 4-chunk Dual)", |b| {
        b.iter(|| {
            let r = zinc_snark::pipeline::verify_dual_circuit_hybrid_gkr_4x_folded::<
                Sha256Uair,
                EcdsaUairInt,
                Int<{ INT_LIMBS * 4 }>,
                FoldedSha4xZt, FoldedSha4xLc,
                EcZt, EcLc,
                FScalar,
                32, 16, 8,
                UNCHECKED,
                zinc_snark::pipeline::TrivialIdeal, _,
                zinc_snark::pipeline::TrivialIdeal, _,
            >(
                &folded_4x_sha_params, &ec_params,
                &hybrid_4x_dual_proof,
                SHA256_8X_NUM_VARS,
                |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
                |_: &IdealOrZero<ImpossibleIdeal>| zinc_snark::pipeline::TrivialIdeal,
                &sha_public_column_data,
                &ecdsa_public_column_data,
            );
            let _ = black_box(r);

            // Feed-forward: compute SHA-256 digest from trace.
            let digest = black_box(feed_forward(&sha_trace, 0));

            // Hash→ECDSA connection: reconstruct u₁ from b₁ bits.
            let u1 = black_box(reconstruct_u1(&ecdsa_trace));
        });
    });

    // ── 4x-Folded SHA proof size breakdown + comparison ──────────
    {
        use zinc_snark::pipeline::FIELD_LIMBS;
        use crypto_primitives::crypto_bigint_uint::Uint;
        let fe_bytes = <Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;

        fn write_fe_4x(buf: &mut Vec<u8>, f: &zinc_snark::pipeline::PiopField) {
            use zinc_snark::pipeline::FIELD_LIMBS;
            use crypto_primitives::crypto_bigint_uint::Uint;
            use zinc_transcript::traits::ConstTranscribable;
            let sz = <Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;
            let start = buf.len();
            buf.resize(start + sz, 0);
            f.inner().write_transcription_bytes(&mut buf[start..]);
        }

        let orig_pcs1_4x = dual_proof.pcs1_proof_bytes.len();
        let pcs1_4x      = folded_4x_dual_proof.pcs1_proof_bytes.len();
        let pcs2_4x      = folded_4x_dual_proof.pcs2_proof_bytes.len();

        // Folding overhead: all four folding byte vecs.
        let folding_bytes_4x: usize =
            folded_4x_dual_proof.pcs1_folding_c1s_bytes.iter().map(|v| v.len()).sum::<usize>()
            + folded_4x_dual_proof.pcs1_folding_c2s_bytes.iter().map(|v| v.len()).sum::<usize>()
            + folded_4x_dual_proof.pcs1_folding_c3s_bytes.iter().map(|v| v.len()).sum::<usize>()
            + folded_4x_dual_proof.pcs1_folding_c4s_bytes.iter().map(|v| v.len()).sum::<usize>();

        let ic1_4x: usize = folded_4x_dual_proof.ic1_proof_values.iter().map(|v| v.len()).sum();
        let ic2_4x: usize = folded_4x_dual_proof.ic2_proof_values.iter().map(|v| v.len()).sum();

        let md_msg_4x: usize = folded_4x_dual_proof.md_group_messages.iter()
            .flat_map(|grp| grp.iter()).map(|v| v.len()).sum();
        let md_sum_4x: usize = folded_4x_dual_proof.md_claimed_sums.iter().map(|v| v.len()).sum();
        let md_total_4x = md_msg_4x + md_sum_4x;

        let cpr1_up_4x: usize = folded_4x_dual_proof.cpr1_up_evals.iter().map(|v| v.len()).sum();
        let cpr1_dn_4x: usize = folded_4x_dual_proof.cpr1_down_evals.iter().map(|v| v.len()).sum();
        let cpr2_up_4x: usize = folded_4x_dual_proof.cpr2_up_evals.iter().map(|v| v.len()).sum();
        let cpr2_dn_4x: usize = folded_4x_dual_proof.cpr2_down_evals.iter().map(|v| v.len()).sum();
        let cpr_total_4x = cpr1_up_4x + cpr1_dn_4x + cpr2_up_4x + cpr2_dn_4x;

        let eval_sc_4x: usize = folded_4x_dual_proof.unified_eval_sumcheck.as_ref().map_or(0, |sc| {
            sc.rounds.iter().map(|v| v.len()).sum::<usize>()
                + sc.v_finals.iter().map(|v| v.len()).sum::<usize>()
        });

        let lookup_4x: usize = folded_4x_dual_proof.lookup_group_proofs.iter().map(|gp| {
            let mults: usize = gp.aggregated_multiplicities.iter().map(|v| v.len()).sum();
            let inv_w: usize = gp.chunk_inverse_witnesses.iter()
                .flat_map(|outer| outer.iter()).map(|inner| inner.len()).sum();
            let inv_t = gp.inverse_table.len();
            (mults + inv_w + inv_t) * fe_bytes
        }).sum();

        let eval_pt_4x: usize = folded_4x_dual_proof.evaluation_point_bytes.iter().map(|v| v.len()).sum();
        let pcs1_eval_4x: usize = folded_4x_dual_proof.pcs1_evals_bytes.iter().map(|v| v.len()).sum();
        let pcs2_eval_4x: usize = folded_4x_dual_proof.pcs2_evals_bytes.iter().map(|v| v.len()).sum();

        let piop_total_4x = ic1_4x + ic2_4x + md_total_4x + cpr_total_4x
            + eval_sc_4x + lookup_4x + eval_pt_4x
            + pcs1_eval_4x + pcs2_eval_4x + folding_bytes_4x;
        let total_raw_4x = pcs1_4x + pcs2_4x + piop_total_4x;

        // Serialize all 4x-folded proof bytes for compression.
        let mut all_bytes_4x = Vec::with_capacity(total_raw_4x);
        all_bytes_4x.extend(&folded_4x_dual_proof.pcs1_proof_bytes);
        all_bytes_4x.extend(&folded_4x_dual_proof.pcs2_proof_bytes);
        for v in &folded_4x_dual_proof.ic1_proof_values  { all_bytes_4x.extend(v); }
        for v in &folded_4x_dual_proof.ic2_proof_values  { all_bytes_4x.extend(v); }
        for grp in &folded_4x_dual_proof.md_group_messages {
            for v in grp { all_bytes_4x.extend(v); }
        }
        for v in &folded_4x_dual_proof.md_claimed_sums { all_bytes_4x.extend(v); }
        for v in &folded_4x_dual_proof.cpr1_up_evals   { all_bytes_4x.extend(v); }
        for v in &folded_4x_dual_proof.cpr1_down_evals  { all_bytes_4x.extend(v); }
        for v in &folded_4x_dual_proof.cpr2_up_evals   { all_bytes_4x.extend(v); }
        for v in &folded_4x_dual_proof.cpr2_down_evals  { all_bytes_4x.extend(v); }
        if let Some(ref sc) = folded_4x_dual_proof.unified_eval_sumcheck {
            for v in &sc.rounds   { all_bytes_4x.extend(v); }
            for v in &sc.v_finals { all_bytes_4x.extend(v); }
        }
        for gp in &folded_4x_dual_proof.lookup_group_proofs {
            for v in &gp.aggregated_multiplicities {
                for f in v { write_fe_4x(&mut all_bytes_4x, f); }
            }
            for outer in &gp.chunk_inverse_witnesses {
                for inner in outer {
                    for f in inner { write_fe_4x(&mut all_bytes_4x, f); }
                }
            }
            for f in &gp.inverse_table { write_fe_4x(&mut all_bytes_4x, f); }
        }
        for v in &folded_4x_dual_proof.evaluation_point_bytes { all_bytes_4x.extend(v); }
        for v in &folded_4x_dual_proof.pcs1_evals_bytes  { all_bytes_4x.extend(v); }
        for v in &folded_4x_dual_proof.pcs2_evals_bytes  { all_bytes_4x.extend(v); }
        for v in &folded_4x_dual_proof.pcs1_folding_c1s_bytes { all_bytes_4x.extend(v); }
        for v in &folded_4x_dual_proof.pcs1_folding_c2s_bytes { all_bytes_4x.extend(v); }
        for v in &folded_4x_dual_proof.pcs1_folding_c3s_bytes { all_bytes_4x.extend(v); }
        for v in &folded_4x_dual_proof.pcs1_folding_c4s_bytes { all_bytes_4x.extend(v); }

        let compressed_4x = {
            use std::io::Write;
            let mut enc = flate2::write::DeflateEncoder::new(
                Vec::new(), flate2::Compression::default(),
            );
            enc.write_all(&all_bytes_4x).unwrap();
            enc.finish().unwrap()
        };

        // Original totals for comparison (recomputed from dual_proof).
        let orig_pcs2 = dual_proof.pcs2_proof_bytes.len();
        let orig_ic1_4x: usize  = dual_proof.ic1_proof_values.iter().map(|v| v.len()).sum();
        let orig_ic2_4x: usize  = dual_proof.ic2_proof_values.iter().map(|v| v.len()).sum();
        let orig_md_4x: usize   = dual_proof.md_group_messages.iter().flat_map(|g| g.iter()).map(|v| v.len()).sum::<usize>()
                                + dual_proof.md_claimed_sums.iter().map(|v| v.len()).sum::<usize>();
        let orig_cpr_4x: usize  = dual_proof.cpr1_up_evals.iter().map(|v| v.len()).sum::<usize>()
                                + dual_proof.cpr1_down_evals.iter().map(|v| v.len()).sum::<usize>()
                                + dual_proof.cpr2_up_evals.iter().map(|v| v.len()).sum::<usize>()
                                + dual_proof.cpr2_down_evals.iter().map(|v| v.len()).sum::<usize>();
        let orig_eval_sc_4x: usize = dual_proof.unified_eval_sumcheck.as_ref().map_or(0, |sc| {
            sc.rounds.iter().map(|v| v.len()).sum::<usize>()
                + sc.v_finals.iter().map(|v| v.len()).sum::<usize>()
        });
        let orig_lookup_4x: usize = dual_proof.lookup_group_proofs.iter().map(|gp| {
            let mults: usize = gp.aggregated_multiplicities.iter().map(|v| v.len()).sum();
            let inv_w: usize = gp.chunk_inverse_witnesses.iter()
                .flat_map(|outer| outer.iter()).map(|inner| inner.len()).sum();
            (mults + inv_w + gp.inverse_table.len()) * fe_bytes
        }).sum();
        let orig_eval_pt_4x: usize = dual_proof.evaluation_point_bytes.iter().map(|v| v.len()).sum();
        let orig_pcs_evals_4x: usize = dual_proof.pcs1_evals_bytes.iter().map(|v| v.len()).sum::<usize>()
                                     + dual_proof.pcs2_evals_bytes.iter().map(|v| v.len()).sum::<usize>();
        let orig_piop_4x = orig_ic1_4x + orig_ic2_4x + orig_md_4x + orig_cpr_4x + orig_eval_sc_4x
                         + orig_lookup_4x + orig_eval_pt_4x + orig_pcs_evals_4x;
        let orig_total_raw_4x = orig_pcs1_4x + orig_pcs2 + orig_piop_4x;

        eprintln!("\n=== 4x-Folded SHA Dual-Circuit Proof Size ===");
        eprintln!("  PCS (SHA 4x-folded): {:>7} B  ({:.1} KB)   [was {} B / {:.1} KB]",
            pcs1_4x, pcs1_4x as f64 / 1024.0, orig_pcs1_4x, orig_pcs1_4x as f64 / 1024.0);
        eprintln!("  PCS (ECDSA):         {:>7} B  ({:.1} KB)", pcs2_4x, pcs2_4x as f64 / 1024.0);
        eprintln!("  Folding overhead:    {:>7} B  (c1s+c2s+c3s+c4s)", folding_bytes_4x);
        eprintln!("  IC (SHA):            {:>7} B", ic1_4x);
        eprintln!("  IC (ECDSA):          {:>7} B", ic2_4x);
        eprintln!("  MD sumcheck:         {:>7} B  (msgs={}, sums={})", md_total_4x, md_msg_4x, md_sum_4x);
        eprintln!("  CPR evals:           {:>7} B  (c1_up={}, c1_dn={}, c2_up={}, c2_dn={})",
            cpr_total_4x, cpr1_up_4x, cpr1_dn_4x, cpr2_up_4x, cpr2_dn_4x);
        eprintln!("  Eval sumcheck:       {:>7} B", eval_sc_4x);
        eprintln!("  Lookup:              {:>7} B  ({} groups)", lookup_4x, folded_4x_dual_proof.lookup_group_proofs.len());
        eprintln!("  Eval point:          {:>7} B", eval_pt_4x);
        eprintln!("  PCS evals:           {:>7} B  (c1={}, c2={})", pcs1_eval_4x + pcs2_eval_4x, pcs1_eval_4x, pcs2_eval_4x);
        eprintln!("  ─────────────────────────────────────────");
        eprintln!("  PIOP total:          {:>7} B  ({:.1} KB)", piop_total_4x, piop_total_4x as f64 / 1024.0);
        eprintln!("  Total raw:           {:>7} B  ({:.1} KB)   [was {} B / {:.1} KB]",
            total_raw_4x, total_raw_4x as f64 / 1024.0, orig_total_raw_4x, orig_total_raw_4x as f64 / 1024.0);
        eprintln!("  Compressed:          {:>7} B  ({:.1} KB, {:.1}x ratio)",
            compressed_4x.len(), compressed_4x.len() as f64 / 1024.0,
            all_bytes_4x.len() as f64 / compressed_4x.len() as f64);
        let raw_savings_4x = orig_total_raw_4x as i64 - total_raw_4x as i64;
        eprintln!("  ─────────────────────────────────────────");
        eprintln!("  Raw savings (vs orig): {:>+7} B  ({:+.1} KB, {:.2}x ratio)",
            raw_savings_4x, raw_savings_4x as f64 / 1024.0,
            orig_total_raw_4x as f64 / total_raw_4x as f64);
    }

    // ══════════════════════════════════════════════════════════════════
    // ── Fixed-C2 Dual-Circuit Pipeline (ECDSA over secp256k1 field) ─
    // ══════════════════════════════════════════════════════════════════
    //
    // NOTE: prove_dual_circuit_fixed_c2 / verify_dual_circuit_fixed_c2
    // are not yet implemented in the pipeline.  The benchmarks below are
    // commented out until those functions are added.
    //
    // type ShaZtFC2 = Sha256ZipTypes<i64, 32>;
    // type ShaLcFC2 = IprsBPoly32R4B64<1, UNCHECKED>;
    // type EcZtFC2 = EcdsaScalarZipTypes;
    // type EcLcFC2 = IprsInt4R4B64<1, UNCHECKED>;
    // let c2_field_cfg = secp256k1_field_config();
    /*

    // ── 25. E2E Total Prover (fixed C2) ──────────────────────────────
    group.bench_function("E2E/Prover (fixed C2)", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let t = Instant::now();
                let _ = zinc_snark::pipeline::prove_dual_circuit_fixed_c2::<
                    Sha256Uair,
                    EcdsaUairInt,
                    Int<{ INT_LIMBS * 4 }>,
                    ShaZtFC2, ShaLcFC2,
                    EcZtFC2, EcLcFC2,
                    EcdsaField,
                    32,
                    UNCHECKED,
                >(
                    &sha_params, &sha_trace,
                    &ec_params, &ecdsa_trace,
                    SHA256_8X_NUM_VARS,
                    &sha_lookup_specs,
                    &sha_affine_specs,
                    &c2_field_cfg,
                );
                total += t.elapsed();
            }
            total
        });
    });

    // ── 26. E2E Total Verifier (fixed C2) ────────────────────────────
    let fixed_c2_proof = zinc_snark::pipeline::prove_dual_circuit_fixed_c2::<
        Sha256Uair,
        EcdsaUairInt,
        Int<{ INT_LIMBS * 4 }>,
        ShaZtFC2, ShaLcFC2,
        EcZtFC2, EcLcFC2,
        EcdsaField,
        32,
        UNCHECKED,
    >(
        &sha_params, &sha_trace,
        &ec_params, &ecdsa_trace,
        SHA256_8X_NUM_VARS,
        &sha_lookup_specs,
        &sha_affine_specs,
        &c2_field_cfg,
    );

    group.bench_function("E2E/Verifier (fixed C2)", |b| {
        b.iter(|| {
            let r = zinc_snark::pipeline::verify_dual_circuit_fixed_c2::<
                Sha256Uair,
                EcdsaUairInt,
                Int<{ INT_LIMBS * 4 }>,
                ShaZtFC2, ShaLcFC2,
                EcZtFC2, EcLcFC2,
                EcdsaField,
                32,
                UNCHECKED,
                zinc_snark::pipeline::TrivialIdeal, _,
                zinc_snark::pipeline::TrivialIdeal, _,
            >(
                &sha_params, &ec_params,
                &fixed_c2_proof,
                SHA256_8X_NUM_VARS,
                |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
                |_: &IdealOrZero<ImpossibleIdeal>| zinc_snark::pipeline::TrivialIdeal,
                &sha_public_column_data,
                &ecdsa_public_column_data,
                &c2_field_cfg,
            );
            assert!(r.accepted, "Fixed-C2 dual-circuit verification must succeed");
            let _ = black_box(r);

            // Feed-forward: compute SHA-256 digest from trace.
            let digest = black_box(feed_forward(&sha_trace, 0));

            // Hash→ECDSA connection: reconstruct u₁ from b₁ bits.
            let u1 = black_box(reconstruct_u1(&ecdsa_trace));
        });
    });

    // ── Fixed-C2 proof size breakdown ────────────────────────────────
    {
        use zinc_snark::pipeline::FIELD_LIMBS;
        use crypto_primitives::crypto_bigint_uint::Uint;
        let piop_fe_bytes = <Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;
        let _c2f_fe_bytes = <Uint<{ INT_LIMBS * 4 }> as ConstTranscribable>::NUM_BYTES;

        let pcs1_bytes = fixed_c2_proof.pcs1_proof_bytes.len();
        let pcs2_bytes = fixed_c2_proof.pcs2_proof_bytes.len();

        let ic1_bytes: usize = fixed_c2_proof.ic1_proof_values.iter().map(|v| v.len()).sum();
        let ic2_bytes: usize = fixed_c2_proof.ic2_proof_values.iter().map(|v| v.len()).sum();

        // C1 MD sumcheck (PiopField).
        let c1_md_msg: usize = fixed_c2_proof.c1_md_group_messages.iter()
            .flat_map(|g| g.iter()).map(|v| v.len()).sum();
        let c1_md_sum: usize = fixed_c2_proof.c1_md_claimed_sums.iter().map(|v| v.len()).sum();
        let c1_md_total = c1_md_msg + c1_md_sum;

        // C2 MD sumcheck (C2F).
        let c2_md_msg: usize = fixed_c2_proof.c2_md_group_messages.iter()
            .flat_map(|g| g.iter()).map(|v| v.len()).sum();
        let c2_md_sum: usize = fixed_c2_proof.c2_md_claimed_sums.iter().map(|v| v.len()).sum();
        let c2_md_total = c2_md_msg + c2_md_sum;

        // CPR evals.
        let cpr1_up: usize = fixed_c2_proof.cpr1_up_evals.iter().map(|v| v.len()).sum();
        let cpr1_dn: usize = fixed_c2_proof.cpr1_down_evals.iter().map(|v| v.len()).sum();
        let cpr2_up: usize = fixed_c2_proof.cpr2_up_evals.iter().map(|v| v.len()).sum();
        let cpr2_dn: usize = fixed_c2_proof.cpr2_down_evals.iter().map(|v| v.len()).sum();
        let cpr_total = cpr1_up + cpr1_dn + cpr2_up + cpr2_dn;

        // Eval sumchecks.
        let c1_eval_sc: usize = fixed_c2_proof.c1_eval_sumcheck.as_ref().map_or(0, |sc| {
            sc.rounds.iter().map(|v| v.len()).sum::<usize>()
                + sc.v_finals.iter().map(|v| v.len()).sum::<usize>()
        });
        let c2_eval_sc: usize = fixed_c2_proof.c2_eval_sumcheck.as_ref().map_or(0, |sc| {
            sc.rounds.iter().map(|v| v.len()).sum::<usize>()
                + sc.v_finals.iter().map(|v| v.len()).sum::<usize>()
        });

        // Lookup.
        let lookup_bytes: usize = fixed_c2_proof.lookup_group_proofs.iter().map(|gp| {
            let mults: usize = gp.aggregated_multiplicities.iter().map(|v| v.len()).sum();
            let inv_w: usize = gp.chunk_inverse_witnesses.iter()
                .flat_map(|o| o.iter()).map(|i| i.len()).sum();
            let inv_t = gp.inverse_table.len();
            (mults + inv_w + inv_t) * piop_fe_bytes
        }).sum();

        // Eval points.
        let c1_eval_pt: usize = fixed_c2_proof.c1_evaluation_point_bytes.iter().map(|v| v.len()).sum();
        let c2_eval_pt: usize = fixed_c2_proof.c2_evaluation_point_bytes.iter().map(|v| v.len()).sum();

        // PCS evals.
        let pcs1_eval: usize = fixed_c2_proof.pcs1_evals_bytes.iter().map(|v| v.len()).sum();
        let pcs2_eval: usize = fixed_c2_proof.pcs2_evals_bytes.iter().map(|v| v.len()).sum();

        let piop_total = ic1_bytes + ic2_bytes + c1_md_total + c2_md_total
            + cpr_total + c1_eval_sc + c2_eval_sc + lookup_bytes
            + c1_eval_pt + c2_eval_pt + pcs1_eval + pcs2_eval;
        let total_raw = pcs1_bytes + pcs2_bytes + piop_total;

        // Compressed.
        let mut all_bytes = Vec::with_capacity(total_raw);
        all_bytes.extend(&fixed_c2_proof.pcs1_proof_bytes);
        all_bytes.extend(&fixed_c2_proof.pcs2_proof_bytes);
        for v in &fixed_c2_proof.ic1_proof_values { all_bytes.extend(v); }
        for v in &fixed_c2_proof.ic2_proof_values { all_bytes.extend(v); }
        for grp in &fixed_c2_proof.c1_md_group_messages { for v in grp { all_bytes.extend(v); } }
        for v in &fixed_c2_proof.c1_md_claimed_sums { all_bytes.extend(v); }
        for grp in &fixed_c2_proof.c2_md_group_messages { for v in grp { all_bytes.extend(v); } }
        for v in &fixed_c2_proof.c2_md_claimed_sums { all_bytes.extend(v); }
        for v in &fixed_c2_proof.cpr1_up_evals { all_bytes.extend(v); }
        for v in &fixed_c2_proof.cpr1_down_evals { all_bytes.extend(v); }
        for v in &fixed_c2_proof.cpr2_up_evals { all_bytes.extend(v); }
        for v in &fixed_c2_proof.cpr2_down_evals { all_bytes.extend(v); }
        if let Some(ref sc) = fixed_c2_proof.c1_eval_sumcheck {
            for v in &sc.rounds { all_bytes.extend(v); }
            for v in &sc.v_finals { all_bytes.extend(v); }
        }
        if let Some(ref sc) = fixed_c2_proof.c2_eval_sumcheck {
            for v in &sc.rounds { all_bytes.extend(v); }
            for v in &sc.v_finals { all_bytes.extend(v); }
        }
        // Lookup field elements.
        {
            fn write_fe_fc2(buf: &mut Vec<u8>, f: &zinc_snark::pipeline::PiopField) {
                use zinc_snark::pipeline::FIELD_LIMBS;
                use crypto_primitives::crypto_bigint_uint::Uint;
                use zinc_transcript::traits::ConstTranscribable;
                let sz = <Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;
                let start = buf.len();
                buf.resize(start + sz, 0);
                f.inner().write_transcription_bytes(&mut buf[start..]);
            }
            for gp in &fixed_c2_proof.lookup_group_proofs {
                for v in &gp.aggregated_multiplicities {
                    for f in v { write_fe_fc2(&mut all_bytes, f); }
                }
                for outer in &gp.chunk_inverse_witnesses {
                    for inner in outer {
                        for f in inner { write_fe_fc2(&mut all_bytes, f); }
                    }
                }
                for f in &gp.inverse_table { write_fe_fc2(&mut all_bytes, f); }
            }
        }
        for v in &fixed_c2_proof.c1_evaluation_point_bytes { all_bytes.extend(v); }
        for v in &fixed_c2_proof.c2_evaluation_point_bytes { all_bytes.extend(v); }
        for v in &fixed_c2_proof.pcs1_evals_bytes { all_bytes.extend(v); }
        for v in &fixed_c2_proof.pcs2_evals_bytes { all_bytes.extend(v); }

        let compressed = {
            use std::io::Write;
            let mut enc = flate2::write::DeflateEncoder::new(
                Vec::new(), flate2::Compression::default(),
            );
            enc.write_all(&all_bytes).unwrap();
            enc.finish().unwrap()
        };

        eprintln!("\n=== Fixed-C2 Dual-Circuit Proof Size (ECDSA over secp256k1) ===");
        eprintln!("  PCS (SHA):           {:>7} B  ({:.1} KB)", pcs1_bytes, pcs1_bytes as f64 / 1024.0);
        eprintln!("  PCS (ECDSA/C2F):     {:>7} B  ({:.1} KB)", pcs2_bytes, pcs2_bytes as f64 / 1024.0);
        eprintln!("  IC (SHA/PiopField):  {:>7} B", ic1_bytes);
        eprintln!("  IC (ECDSA/C2F):      {:>7} B", ic2_bytes);
        eprintln!("  C1 MD sumcheck:      {:>7} B  (msgs={}, sums={})", c1_md_total, c1_md_msg, c1_md_sum);
        eprintln!("  C2 MD sumcheck:      {:>7} B  (msgs={}, sums={})", c2_md_total, c2_md_msg, c2_md_sum);
        eprintln!("  CPR evals:           {:>7} B  (c1_up={}, c1_dn={}, c2_up={}, c2_dn={})",
            cpr_total, cpr1_up, cpr1_dn, cpr2_up, cpr2_dn);
        eprintln!("  C1 eval sumcheck:    {:>7} B", c1_eval_sc);
        eprintln!("  C2 eval sumcheck:    {:>7} B", c2_eval_sc);
        eprintln!("  Lookup:              {:>7} B  ({} groups)", lookup_bytes, fixed_c2_proof.lookup_group_proofs.len());
        eprintln!("  C1 eval point:       {:>7} B", c1_eval_pt);
        eprintln!("  C2 eval point:       {:>7} B", c2_eval_pt);
        eprintln!("  PCS evals:           {:>7} B  (c1={}, c2={})", pcs1_eval + pcs2_eval, pcs1_eval, pcs2_eval);
        eprintln!("  ─────────────────────────────────────────");
        eprintln!("  PIOP total:          {:>7} B  ({:.1} KB)", piop_total, piop_total as f64 / 1024.0);
        eprintln!("  Total raw:           {:>7} B  ({:.1} KB)", total_raw, total_raw as f64 / 1024.0);
        eprintln!("  Compressed:          {:>7} B  ({:.1} KB, {:.1}x ratio)",
            compressed.len(), compressed.len() as f64 / 1024.0,
            all_bytes.len() as f64 / compressed.len() as f64);

        // Timing breakdown.
        eprintln!("\n  Fixed-C2 timing: PCS commit={:?}, IC={:?}, CPR+Lookup={:?}, PCS prove={:?}, total={:?}",
            fixed_c2_proof.timing.pcs_commit,
            fixed_c2_proof.timing.ideal_check,
            fixed_c2_proof.timing.combined_poly_resolver,
            fixed_c2_proof.timing.pcs_prove,
            fixed_c2_proof.timing.total,
        );
    }
    */ // end of commented-out Fixed-C2 section

    // ══════════════════════════════════════════════════════════════════
    // ── Chunk Variant Proof Size Comparison (folded dual-circuit) ────
    // ══════════════════════════════════════════════════════════════════
    //
    // Computes raw + compressed proof sizes for every (folding x chunk)
    // combination and prints a summary table.
    {
        use zinc_snark::pipeline::FIELD_LIMBS;
        use crypto_primitives::crypto_bigint_uint::Uint;
        let fe_bytes = <Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;

        fn write_fe_cv(buf: &mut Vec<u8>, f: &zinc_snark::pipeline::PiopField) {
            use zinc_snark::pipeline::FIELD_LIMBS;
            use crypto_primitives::crypto_bigint_uint::Uint;
            use zinc_transcript::traits::ConstTranscribable;
            let sz = <Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;
            let start = buf.len();
            buf.resize(start + sz, 0);
            f.inner().write_transcription_bytes(&mut buf[start..]);
        }

        // Helper: serialise & compress a folded (2x) dual-circuit proof.
        let proof_size_folded = |proof: &zinc_snark::pipeline::FoldedDualCircuitZincProof,
                                  label: &str,
                                  _fe_bytes: usize|
            -> (usize, usize)
        {
            let mut all_bytes = Vec::new();
            all_bytes.extend(&proof.pcs1_proof_bytes);
            all_bytes.extend(&proof.pcs2_proof_bytes);
            for v in &proof.ic1_proof_values  { all_bytes.extend(v); }
            for v in &proof.ic2_proof_values  { all_bytes.extend(v); }
            for grp in &proof.md_group_messages { for v in grp { all_bytes.extend(v); } }
            for v in &proof.md_claimed_sums   { all_bytes.extend(v); }
            for v in &proof.cpr1_up_evals     { all_bytes.extend(v); }
            for v in &proof.cpr1_down_evals   { all_bytes.extend(v); }
            for v in &proof.cpr2_up_evals     { all_bytes.extend(v); }
            for v in &proof.cpr2_down_evals   { all_bytes.extend(v); }
            if let Some(ref sc) = proof.unified_eval_sumcheck {
                for v in &sc.rounds   { all_bytes.extend(v); }
                for v in &sc.v_finals { all_bytes.extend(v); }
            }
            for gp in &proof.lookup_group_proofs {
                for v in &gp.aggregated_multiplicities {
                    for f in v { write_fe_cv(&mut all_bytes, f); }
                }
                for outer in &gp.chunk_inverse_witnesses {
                    for inner in outer { for f in inner { write_fe_cv(&mut all_bytes, f); } }
                }
                for f in &gp.inverse_table { write_fe_cv(&mut all_bytes, f); }
            }
            for v in &proof.evaluation_point_bytes { all_bytes.extend(v); }
            for v in &proof.pcs1_evals_bytes       { all_bytes.extend(v); }
            for v in &proof.pcs2_evals_bytes       { all_bytes.extend(v); }
            for v in &proof.pcs1_folding_c1s_bytes { all_bytes.extend(v); }
            for v in &proof.pcs1_folding_c2s_bytes { all_bytes.extend(v); }

            let compressed = {
                use std::io::Write;
                let mut enc = flate2::write::DeflateEncoder::new(
                    Vec::new(), flate2::Compression::default(),
                );
                enc.write_all(&all_bytes).unwrap();
                enc.finish().unwrap()
            };

            eprintln!("\n=== {label} Proof Size ===");
            eprintln!("  Total raw:      {:>7} B  ({:.1} KB)", all_bytes.len(), all_bytes.len() as f64 / 1024.0);
            eprintln!("  Compressed:     {:>7} B  ({:.1} KB, {:.1}x ratio)",
                compressed.len(), compressed.len() as f64 / 1024.0,
                all_bytes.len() as f64 / compressed.len() as f64);

            (all_bytes.len(), compressed.len())
        };

        // Helper: serialise & compress a 4x-folded dual-circuit proof.
        let proof_size_4x = |proof: &zinc_snark::pipeline::FoldedDualCircuit4xZincProof,
                              label: &str,
                              _fe_bytes: usize|
            -> (usize, usize)
        {
            let mut all_bytes = Vec::new();
            all_bytes.extend(&proof.pcs1_proof_bytes);
            all_bytes.extend(&proof.pcs2_proof_bytes);
            for v in &proof.ic1_proof_values  { all_bytes.extend(v); }
            for v in &proof.ic2_proof_values  { all_bytes.extend(v); }
            for grp in &proof.md_group_messages { for v in grp { all_bytes.extend(v); } }
            for v in &proof.md_claimed_sums   { all_bytes.extend(v); }
            for v in &proof.cpr1_up_evals     { all_bytes.extend(v); }
            for v in &proof.cpr1_down_evals   { all_bytes.extend(v); }
            for v in &proof.cpr2_up_evals     { all_bytes.extend(v); }
            for v in &proof.cpr2_down_evals   { all_bytes.extend(v); }
            if let Some(ref sc) = proof.unified_eval_sumcheck {
                for v in &sc.rounds   { all_bytes.extend(v); }
                for v in &sc.v_finals { all_bytes.extend(v); }
            }
            for gp in &proof.lookup_group_proofs {
                for v in &gp.aggregated_multiplicities {
                    for f in v { write_fe_cv(&mut all_bytes, f); }
                }
                for outer in &gp.chunk_inverse_witnesses {
                    for inner in outer { for f in inner { write_fe_cv(&mut all_bytes, f); } }
                }
                for f in &gp.inverse_table { write_fe_cv(&mut all_bytes, f); }
            }
            for v in &proof.evaluation_point_bytes     { all_bytes.extend(v); }
            for v in &proof.pcs1_evals_bytes           { all_bytes.extend(v); }
            for v in &proof.pcs2_evals_bytes           { all_bytes.extend(v); }
            for v in &proof.pcs1_folding_c1s_bytes     { all_bytes.extend(v); }
            for v in &proof.pcs1_folding_c2s_bytes     { all_bytes.extend(v); }
            for v in &proof.pcs1_folding_c3s_bytes     { all_bytes.extend(v); }
            for v in &proof.pcs1_folding_c4s_bytes     { all_bytes.extend(v); }

            let compressed = {
                use std::io::Write;
                let mut enc = flate2::write::DeflateEncoder::new(
                    Vec::new(), flate2::Compression::default(),
                );
                enc.write_all(&all_bytes).unwrap();
                enc.finish().unwrap()
            };

            eprintln!("\n=== {label} Proof Size ===");
            eprintln!("  Total raw:      {:>7} B  ({:.1} KB)", all_bytes.len(), all_bytes.len() as f64 / 1024.0);
            eprintln!("  Compressed:     {:>7} B  ({:.1} KB, {:.1}x ratio)",
                compressed.len(), compressed.len() as f64 / 1024.0,
                all_bytes.len() as f64 / compressed.len() as f64);

            (all_bytes.len(), compressed.len())
        };

        // Helper: serialise & compress a single-circuit 4x-folded proof
        // (Folded4xZincProof), including HybridGkr lookup data.
        let proof_size_hybrid_4x = |proof: &zinc_snark::pipeline::Folded4xZincProof,
                                     label: &str,
                                     fe_bytes: usize|
            -> (usize, usize)
        {
            use zinc_snark::pipeline::LookupProofData;

            let pcs      = proof.pcs_proof_bytes.len();
            let ic: usize  = proof.ic_proof_values.iter().map(|v| v.len()).sum();
            let fold_c1: usize = proof.folding_c1s_bytes.iter().map(|v| v.len()).sum();
            let fold_c2: usize = proof.folding_c2s_bytes.iter().map(|v| v.len()).sum();
            let fold_c3: usize = proof.folding_c3s_bytes.iter().map(|v| v.len()).sum();
            let fold_c4: usize = proof.folding_c4s_bytes.iter().map(|v| v.len()).sum();
            let folding  = fold_c1 + fold_c2 + fold_c3 + fold_c4;
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
                                t += fe_bytes; // claimed_sum
                            }
                            t += (lp.p_lefts.len() + lp.p_rights.len() + lp.q_lefts.len() + lp.q_rights.len()) * fe_bytes;
                        }
                        let sent_p: usize = wg.sent_p.iter().map(|v| v.len()).sum();
                        let sent_q: usize = wg.sent_q.iter().map(|v| v.len()).sum();
                        t += (sent_p + sent_q) * fe_bytes;
                        let tg = &group.table_gkr;
                        t += 2 * fe_bytes; // root_p, root_q
                        for lp in &tg.layer_proofs {
                            if let Some(ref sc) = lp.sumcheck_proof {
                                t += sc.messages.iter().map(|m| m.0.tail_evaluations.len()).sum::<usize>() * fe_bytes;
                                t += fe_bytes; // claimed_sum
                            }
                            t += 4 * fe_bytes; // p_left, p_right, q_left, q_right
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

            // Serialise all fields for compression measurement.
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
            eprintln!("  PCS:            {:>7} B  ({:.1} KB)", pcs, pcs as f64 / 1024.0);
            eprintln!("  IC:             {:>7} B", ic);
            eprintln!("  CPR sumcheck:   {:>7} B", cpr_sc);
            eprintln!("  CPR evals:      {:>7} B  (up={cpr_up}, down={cpr_dn})", cpr_up + cpr_dn);
            eprintln!("  Lookup:         {:>7} B  ({:.1} KB)", lk, lk as f64 / 1024.0);
            eprintln!("  Shift SC:       {:>7} B", shift);
            eprintln!("  Folding:        {:>7} B  (c1={fold_c1}, c2={fold_c2}, c3={fold_c3}, c4={fold_c4})", folding);
            eprintln!("  Eval point:     {:>7} B", eval_pt);
            eprintln!("  PCS evals:      {:>7} B", pcs_eval);
            eprintln!("  ─────────────────────────────");
            eprintln!("  Total raw:      {:>7} B  ({:.1} KB)", total, total as f64 / 1024.0);
            eprintln!("  Compressed:     {:>7} B  ({:.1} KB, {:.1}x ratio)",
                compressed.len(), compressed.len() as f64 / 1024.0,
                all_bytes.len() as f64 / compressed.len() as f64);

            (total, compressed.len())
        };

        // ── Compute all configurations ────────────────────────────────
        let (raw_2x_8c, compr_2x_8c) = proof_size_folded(&folded_dual_proof,    "2x Folded 8-chunk (8×2^4)", fe_bytes);
        let (raw_2x_4c, compr_2x_4c) = proof_size_folded(&folded_dual_proof_4c, "2x Folded 4-chunk (4×2^8)", fe_bytes);
        let (raw_4x_8c, compr_4x_8c) = proof_size_4x(&folded_4x_dual_proof,    "4x Folded 8-chunk (8×2^4)", fe_bytes);
        let (raw_4x_4c, compr_4x_4c) = proof_size_4x(&folded_4x_dual_proof_4c, "4x Folded 4-chunk (4×2^8)", fe_bytes);

        // ── Hybrid GKR c=2 (single-circuit SHA only) ────────────────
        let (raw_hybrid_4x, compr_hybrid_4x) = proof_size_hybrid_4x(&hybrid_4x_proof, "4x Hybrid GKR c=2 4-chunk (SHA)", fe_bytes);

        // ── Hybrid GKR c=2 (dual-circuit SHA+ECDSA) ─────────────────

        // ── Summary comparison table ──────────────────────────────────
        eprintln!("\n=== Chunk Variant Proof Size Comparison (Dual-Circuit + Hybrid) ===");
        eprintln!("  {:42}  {:>8}  {:>8}", "Configuration", "Raw (B)", "Compr (B)");
        eprintln!("  {}", "─".repeat(62));
        eprintln!("  {:42}  {:>8}  {:>8}", "2x folded, 8-chunk (8×2^4) *",  raw_2x_8c, compr_2x_8c);
        eprintln!("  {:42}  {:>8}  {:>8}", "2x folded, 4-chunk (4×2^8)",    raw_2x_4c, compr_2x_4c);
        eprintln!("  {:42}  {:>8}  {:>8}", "4x folded, 8-chunk (8×2^4) *",  raw_4x_8c, compr_4x_8c);
        eprintln!("  {:42}  {:>8}  {:>8}", "4x folded, 4-chunk (4×2^8)",    raw_4x_4c, compr_4x_4c);
        eprintln!("  {:42}  {:>8}  {:>8}", "4x Hybrid GKR c=2, 4-chunk (SHA only)", raw_hybrid_4x, compr_hybrid_4x);
        eprintln!("  (* = default configuration)");
    }

    // ══════════════════════════════════════════════════════════════════
    // ── Verifier Step-by-Step Breakdown ──────────────────────────────
    // ══════════════════════════════════════════════════════════════════
    //
    // Replays the verify_dual_circuit logic piece by piece so that each
    // verifier sub-cost can be measured independently via Criterion.
    //
    // Each benchmark function replays the Fiat-Shamir transcript from
    // scratch to the relevant stage, then times only the target step.
    // This matches what the real verifier does (all sequential, all
    // deterministic).

    {
        use zinc_snark::pipeline::{
            FIELD_LIMBS, PiopField,
            field_from_bytes, reconstruct_up_evals, TrivialIdeal,
        };
        use zinc_uair::ideal::ImpossibleIdeal;
        use zinc_poly::mle::MultilinearExtensionWithConfig;

        let field_elem_size =
            <Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;

        // Shorthand: deserialize a vector of IC proof values.
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

        // Shorthand: deserialize multi-degree sumcheck proof.
        let deser_md = |fcfg: &<PiopField as PrimeField>::Config| -> MultiDegreeSumcheckProof<PiopField> {
            let md_msgs: Vec<Vec<ProverMsg<PiopField>>> = dual_proof
                .md_group_messages.iter().map(|grp| {
                    grp.iter().map(|bytes| {
                        let n = bytes.len() / field_elem_size;
                        ProverMsg(NatEvaluatedPolyWithoutConstant::new(
                            (0..n).map(|i| field_from_bytes(
                                &bytes[i * field_elem_size..(i + 1) * field_elem_size], fcfg,
                            )).collect()
                        ))
                    }).collect()
                }).collect();
            let md_sums: Vec<PiopField> = dual_proof
                .md_claimed_sums.iter().map(|b| field_from_bytes(b, fcfg)).collect();
            MultiDegreeSumcheckProof {
                group_messages: md_msgs,
                claimed_sums: md_sums,
                degrees: dual_proof.md_degrees.clone(),
            }
        };

        // ── V1. Verifier / Field Setup ──────────────────────────────
        //
        // Transcript init + random field config + IC evaluation point.
        group.bench_function("V/FieldSetup", |b| {
            b.iter(|| {
                let mut transcript = zinc_transcript::KeccakTranscript::new();
                let field_cfg = transcript.get_random_field_cfg::<
                    PiopField, <PiopField as Field>::Inner, MillerRabin
                >();
                let ic_pt: Vec<PiopField> =
                    transcript.get_field_challenges(num_vars, &field_cfg);
                black_box((&field_cfg, ic_pt));
            });
        });

        // ── V2. Verifier / Ideal Check ──────────────────────────────
        //
        // Deserialize IC proof values + verify both IC₁ (SHA) and IC₂
        // (ECDSA). Transcript setup overhead is excluded by using
        // iter_custom with manual timing.
        group.bench_function("V/Ideal Check", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let mut tr = zinc_transcript::KeccakTranscript::new();
                    let fcfg = tr.get_random_field_cfg::<
                        PiopField, <PiopField as Field>::Inner, MillerRabin
                    >();
                    let ic_pt: Vec<PiopField> = tr.get_field_challenges(num_vars, &fcfg);

                    let t = Instant::now();

                    // IC₁ verify.
                    let c1_proof = zinc_piop::ideal_check::Proof::<PiopField> {
                        combined_mle_values: deser_ic(&dual_proof.ic1_proof_values, &fcfg),
                    };
                    let _ = IdealCheckProtocol::<PiopField>::verify_at_point::<Sha256Uair, _, _>(
                        &mut tr, c1_proof, sha_num_constraints, ic_pt.clone(),
                        &|_: &IdealOrZero<CyclotomicIdeal>| TrivialIdeal,
                        &fcfg,
                    ).expect("C1 IC verify");

                    // IC₂ verify.
                    let c2_proof = zinc_piop::ideal_check::Proof::<PiopField> {
                        combined_mle_values: deser_ic(&dual_proof.ic2_proof_values, &fcfg),
                    };
                    let _ = IdealCheckProtocol::<PiopField>::verify_at_point::<EcdsaUairInt, _, _>(
                        &mut tr, c2_proof, ecdsa_num_constraints, ic_pt,
                        &|_: &IdealOrZero<ImpossibleIdeal>| TrivialIdeal,
                        &fcfg,
                    ).expect("C2 IC verify");

                    total += t.elapsed();
                }
                total
            });
        });

        // ── V3. Verifier / CPR + Lookup Pre-Sumcheck ────────────────
        //
        // Projecting element + CPR₁ pre + lookup pre + CPR₂ pre.
        group.bench_function("V/Main field sumcheck+LookupPre", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    // Replay transcript state to end of IC.
                    let mut tr = zinc_transcript::KeccakTranscript::new();
                    let fcfg = tr.get_random_field_cfg::<
                        PiopField, <PiopField as Field>::Inner, MillerRabin
                    >();
                    let ic_pt: Vec<PiopField> = tr.get_field_challenges(num_vars, &fcfg);
                    let c1_proof = zinc_piop::ideal_check::Proof::<PiopField> {
                        combined_mle_values: deser_ic(&dual_proof.ic1_proof_values, &fcfg),
                    };
                    let c1_sub = IdealCheckProtocol::<PiopField>::verify_at_point::<Sha256Uair, _, _>(
                        &mut tr, c1_proof, sha_num_constraints, ic_pt.clone(),
                        &|_: &IdealOrZero<CyclotomicIdeal>| TrivialIdeal, &fcfg,
                    ).expect("C1 IC");
                    let c2_proof = zinc_piop::ideal_check::Proof::<PiopField> {
                        combined_mle_values: deser_ic(&dual_proof.ic2_proof_values, &fcfg),
                    };
                    let c2_sub = IdealCheckProtocol::<PiopField>::verify_at_point::<EcdsaUairInt, _, _>(
                        &mut tr, c2_proof, ecdsa_num_constraints, ic_pt,
                        &|_: &IdealOrZero<ImpossibleIdeal>| TrivialIdeal, &fcfg,
                    ).expect("C2 IC");

                    // ── Timed section ──
                    let t = Instant::now();

                    let proj_elem: PiopField = tr.get_field_challenge(&fcfg);

                    // CPR₁ pre-sumcheck.
                    let c1_psc = project_scalars::<PiopField, Sha256Uair>(|scalar| {
                        let one = PiopField::one_with_cfg(&fcfg);
                        let zero = PiopField::zero_with_cfg(&fcfg);
                        DynamicPolynomialF::new(
                            scalar.iter().map(|c| if c.into_inner() { one.clone() } else { zero.clone() }).collect::<Vec<_>>()
                        )
                    });
                    let c1_fps = project_scalars_to_field(c1_psc, &proj_elem).unwrap();

                    let _ = CombinedPolyResolver::<PiopField>::build_verifier_pre_sumcheck::<Sha256Uair>(
                        &mut tr,
                        &field_from_bytes(&dual_proof.md_claimed_sums[0], &fcfg),
                        sha_num_constraints, &proj_elem, &c1_fps, &c1_sub, &fcfg,
                    ).expect("C1 CPR pre");

                    // Lookup pre-sumcheck.
                    for (gp, meta) in dual_proof.lookup_group_proofs.iter()
                        .zip(dual_proof.lookup_group_meta.iter())
                    {
                        let (subtable, shifts) = generate_table_and_shifts(
                            &meta.table_type, &proj_elem, &fcfg,
                        );
                        let _ = BatchedDecompLogupProtocol::<PiopField>::build_verifier_pre_sumcheck(
                            &mut tr, gp, &subtable, &shifts,
                            meta.num_columns, meta.witness_len, &fcfg,
                        ).expect("lookup pre");
                    }

                    // CPR₂ pre-sumcheck.
                    let c2_psc = project_scalars::<PiopField, EcdsaUairInt>(|scalar| {
                        DynamicPolynomialF { coeffs: vec![PiopField::from_with_cfg(scalar.clone(), &fcfg)] }
                    });
                    let c2_fps = project_scalars_to_field(c2_psc, &proj_elem).unwrap();

                    let _ = CombinedPolyResolver::<PiopField>::build_verifier_pre_sumcheck::<EcdsaUairInt>(
                        &mut tr,
                        &field_from_bytes(&dual_proof.md_claimed_sums[1], &fcfg),
                        ecdsa_num_constraints, &proj_elem, &c2_fps, &c2_sub, &fcfg,
                    ).expect("C2 CPR pre");

                    total += t.elapsed();
                }
                total
            });
        });

        // ── V4. Verifier / Multi-Degree Sumcheck ────────────────────
        //
        // Deserialize + verify the batched sumcheck proof.
        group.bench_function("V/MDSumcheck", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    // Replay to end of pre-sumcheck stage.
                    let mut tr = zinc_transcript::KeccakTranscript::new();
                    let fcfg = tr.get_random_field_cfg::<
                        PiopField, <PiopField as Field>::Inner, MillerRabin
                    >();
                    let ic_pt: Vec<PiopField> = tr.get_field_challenges(num_vars, &fcfg);
                    let c1_sub = IdealCheckProtocol::<PiopField>::verify_at_point::<Sha256Uair, _, _>(
                        &mut tr,
                        zinc_piop::ideal_check::Proof { combined_mle_values: deser_ic(&dual_proof.ic1_proof_values, &fcfg) },
                        sha_num_constraints, ic_pt.clone(),
                        &|_: &IdealOrZero<CyclotomicIdeal>| TrivialIdeal, &fcfg,
                    ).unwrap();
                    let c2_sub = IdealCheckProtocol::<PiopField>::verify_at_point::<EcdsaUairInt, _, _>(
                        &mut tr,
                        zinc_piop::ideal_check::Proof { combined_mle_values: deser_ic(&dual_proof.ic2_proof_values, &fcfg) },
                        ecdsa_num_constraints, ic_pt,
                        &|_: &IdealOrZero<ImpossibleIdeal>| TrivialIdeal, &fcfg,
                    ).unwrap();
                    let proj_elem: PiopField = tr.get_field_challenge(&fcfg);
                    let c1_psc = project_scalars::<PiopField, Sha256Uair>(|s| {
                        let one = PiopField::one_with_cfg(&fcfg); let zero = PiopField::zero_with_cfg(&fcfg);
                        DynamicPolynomialF::new(s.iter().map(|c| if c.into_inner() { one.clone() } else { zero.clone() }).collect::<Vec<_>>())
                    });
                    let c1_fps = project_scalars_to_field(c1_psc, &proj_elem).unwrap();
                    let _ = CombinedPolyResolver::<PiopField>::build_verifier_pre_sumcheck::<Sha256Uair>(
                        &mut tr, &field_from_bytes(&dual_proof.md_claimed_sums[0], &fcfg),
                        sha_num_constraints, &proj_elem, &c1_fps, &c1_sub, &fcfg,
                    ).unwrap();
                    let mut lk_pres = Vec::new();
                    for (gp, meta) in dual_proof.lookup_group_proofs.iter().zip(dual_proof.lookup_group_meta.iter()) {
                        let (subtable, shifts) = generate_table_and_shifts(&meta.table_type, &proj_elem, &fcfg);
                        lk_pres.push(BatchedDecompLogupProtocol::<PiopField>::build_verifier_pre_sumcheck(
                            &mut tr, gp, &subtable, &shifts, meta.num_columns, meta.witness_len, &fcfg,
                        ).unwrap());
                    }
                    let c2_psc = project_scalars::<PiopField, EcdsaUairInt>(|s| {
                        DynamicPolynomialF { coeffs: vec![PiopField::from_with_cfg(s.clone(), &fcfg)] }
                    });
                    let c2_fps = project_scalars_to_field(c2_psc, &proj_elem).unwrap();
                    let _ = CombinedPolyResolver::<PiopField>::build_verifier_pre_sumcheck::<EcdsaUairInt>(
                        &mut tr, &field_from_bytes(&dual_proof.md_claimed_sums[1], &fcfg),
                        ecdsa_num_constraints, &proj_elem, &c2_fps, &c2_sub, &fcfg,
                    ).unwrap();

                    let shared_nv = lk_pres.iter().map(|p| p.num_vars).max()
                        .map_or(num_vars, |m| m.max(num_vars));

                    // ── Timed section ──
                    let md_proof = deser_md(&fcfg);
                    let t = Instant::now();
                    let _ = MultiDegreeSumcheck::<PiopField>::verify_as_subprotocol(
                        &mut tr, shared_nv, &md_proof, &fcfg,
                    ).expect("MD sumcheck verify");
                    total += t.elapsed();
                }
                total
            });
        });

        // ── V5. Verifier / Main field sumcheck Finalize ─────────────────────────────
        //
        // Finalize CPR for both circuits (includes public column MLE eval).
        group.bench_function("V/Main field sumcheck Finalize", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    // Replay through MD sumcheck.
                    let mut tr = zinc_transcript::KeccakTranscript::new();
                    let fcfg = tr.get_random_field_cfg::<
                        PiopField, <PiopField as Field>::Inner, MillerRabin
                    >();
                    let ic_pt: Vec<PiopField> = tr.get_field_challenges(num_vars, &fcfg);
                    let c1_sub = IdealCheckProtocol::<PiopField>::verify_at_point::<Sha256Uair, _, _>(
                        &mut tr,
                        zinc_piop::ideal_check::Proof { combined_mle_values: deser_ic(&dual_proof.ic1_proof_values, &fcfg) },
                        sha_num_constraints, ic_pt.clone(),
                        &|_: &IdealOrZero<CyclotomicIdeal>| TrivialIdeal, &fcfg,
                    ).unwrap();
                    let c2_sub = IdealCheckProtocol::<PiopField>::verify_at_point::<EcdsaUairInt, _, _>(
                        &mut tr,
                        zinc_piop::ideal_check::Proof { combined_mle_values: deser_ic(&dual_proof.ic2_proof_values, &fcfg) },
                        ecdsa_num_constraints, ic_pt,
                        &|_: &IdealOrZero<ImpossibleIdeal>| TrivialIdeal, &fcfg,
                    ).unwrap();
                    let proj_elem: PiopField = tr.get_field_challenge(&fcfg);
                    let c1_psc = project_scalars::<PiopField, Sha256Uair>(|s| {
                        let one = PiopField::one_with_cfg(&fcfg); let zero = PiopField::zero_with_cfg(&fcfg);
                        DynamicPolynomialF::new(s.iter().map(|c| if c.into_inner() { one.clone() } else { zero.clone() }).collect::<Vec<_>>())
                    });
                    let c1_fps = project_scalars_to_field(c1_psc, &proj_elem).unwrap();
                    let c1_cpr_pre = CombinedPolyResolver::<PiopField>::build_verifier_pre_sumcheck::<Sha256Uair>(
                        &mut tr, &field_from_bytes(&dual_proof.md_claimed_sums[0], &fcfg),
                        sha_num_constraints, &proj_elem, &c1_fps, &c1_sub, &fcfg,
                    ).unwrap();
                    let mut lk_pres = Vec::new();
                    for (gp, meta) in dual_proof.lookup_group_proofs.iter().zip(dual_proof.lookup_group_meta.iter()) {
                        let (subtable, shifts) = generate_table_and_shifts(&meta.table_type, &proj_elem, &fcfg);
                        lk_pres.push(BatchedDecompLogupProtocol::<PiopField>::build_verifier_pre_sumcheck(
                            &mut tr, gp, &subtable, &shifts, meta.num_columns, meta.witness_len, &fcfg,
                        ).unwrap());
                    }
                    let c2_psc = project_scalars::<PiopField, EcdsaUairInt>(|s| {
                        DynamicPolynomialF { coeffs: vec![PiopField::from_with_cfg(s.clone(), &fcfg)] }
                    });
                    let c2_fps = project_scalars_to_field(c2_psc, &proj_elem).unwrap();
                    let c2_cpr_pre = CombinedPolyResolver::<PiopField>::build_verifier_pre_sumcheck::<EcdsaUairInt>(
                        &mut tr, &field_from_bytes(&dual_proof.md_claimed_sums[1], &fcfg),
                        ecdsa_num_constraints, &proj_elem, &c2_fps, &c2_sub, &fcfg,
                    ).unwrap();
                    let shared_nv = lk_pres.iter().map(|p| p.num_vars).max()
                        .map_or(num_vars, |m| m.max(num_vars));
                    let md_proof = deser_md(&fcfg);
                    let md_sub = MultiDegreeSumcheck::<PiopField>::verify_as_subprotocol(
                        &mut tr, shared_nv, &md_proof, &fcfg,
                    ).unwrap();

                    // ── Timed section ──
                    let t = Instant::now();

                    // CPR₁ finalize.
                    let c1_priv_up: Vec<PiopField> = dual_proof.cpr1_up_evals.iter()
                        .map(|b| field_from_bytes(b, &fcfg)).collect();
                    let c1_down: Vec<PiopField> = dual_proof.cpr1_down_evals.iter()
                        .map(|b| field_from_bytes(b, &fcfg)).collect();
                    let c1_sig_v = Sha256Uair::signature();
                    let c1_up = if c1_sig_v.public_columns.is_empty() {
                        c1_priv_up
                    } else {
                        let bin_proj = BinaryPoly::<32>::prepare_projection(&proj_elem);
                        let c1_pub: Vec<PiopField> = sha_public_column_data.iter().map(|col| {
                            let mut mle: DenseMultilinearExtension<<PiopField as Field>::Inner> =
                                col.iter().map(|bp| bin_proj(bp).inner().clone()).collect();
                            if md_sub.point.len() > mle.num_vars {
                                mle.evaluations.resize(1 << md_sub.point.len(), Default::default());
                                mle.num_vars = md_sub.point.len();
                            }
                            mle.evaluate_with_config(&md_sub.point, &fcfg).unwrap()
                        }).collect();
                        reconstruct_up_evals(&c1_priv_up, &c1_pub, &c1_sig_v.public_columns, c1_sig_v.total_cols())
                    };
                    let _ = CombinedPolyResolver::<PiopField>::finalize_verifier::<Sha256Uair>(
                        &mut tr, md_sub.point.clone(), md_sub.expected_evaluations[0].clone(),
                        &c1_cpr_pre, c1_up, c1_down, num_vars, &c1_fps, &fcfg,
                    ).expect("C1 CPR finalize");

                    // CPR₂ finalize.
                    let c2_priv_up: Vec<PiopField> = dual_proof.cpr2_up_evals.iter()
                        .map(|b| field_from_bytes(b, &fcfg)).collect();
                    let c2_down: Vec<PiopField> = dual_proof.cpr2_down_evals.iter()
                        .map(|b| field_from_bytes(b, &fcfg)).collect();
                    let c2_sig_v = EcdsaUairInt::signature();
                    let c2_up = if c2_sig_v.public_columns.is_empty() {
                        c2_priv_up
                    } else {
                        let c2_pub: Vec<PiopField> = ecdsa_public_column_data.iter().map(|col| {
                            let proj_col: DenseMultilinearExtension<<PiopField as Field>::Inner> =
                                col.iter().map(|v| PiopField::from_with_cfg(v.clone(), &fcfg).inner().clone()).collect();
                            proj_col.evaluate_with_config(&md_sub.point, &fcfg).unwrap()
                        }).collect();
                        reconstruct_up_evals(&c2_priv_up, &c2_pub, &c2_sig_v.public_columns, c2_sig_v.total_cols())
                    };
                    let _ = CombinedPolyResolver::<PiopField>::finalize_verifier::<EcdsaUairInt>(
                        &mut tr, md_sub.point.clone(), md_sub.expected_evaluations[1].clone(),
                        &c2_cpr_pre, c2_up, c2_down, num_vars, &c2_fps, &fcfg,
                    ).expect("C2 CPR finalize");

                    total += t.elapsed();
                }
                total
            });
        });

        // ── V6. Verifier / Unified Eval Sumcheck ────────────────────
        group.bench_function("V/UnifiedEvalSC", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    // Replay through CPR finalize.
                    let mut tr = zinc_transcript::KeccakTranscript::new();
                    let fcfg = tr.get_random_field_cfg::<
                        PiopField, <PiopField as Field>::Inner, MillerRabin
                    >();
                    let ic_pt: Vec<PiopField> = tr.get_field_challenges(num_vars, &fcfg);
                    let c1_sub = IdealCheckProtocol::<PiopField>::verify_at_point::<Sha256Uair, _, _>(
                        &mut tr,
                        zinc_piop::ideal_check::Proof { combined_mle_values: deser_ic(&dual_proof.ic1_proof_values, &fcfg) },
                        sha_num_constraints, ic_pt.clone(),
                        &|_: &IdealOrZero<CyclotomicIdeal>| TrivialIdeal, &fcfg,
                    ).unwrap();
                    let c2_sub = IdealCheckProtocol::<PiopField>::verify_at_point::<EcdsaUairInt, _, _>(
                        &mut tr,
                        zinc_piop::ideal_check::Proof { combined_mle_values: deser_ic(&dual_proof.ic2_proof_values, &fcfg) },
                        ecdsa_num_constraints, ic_pt,
                        &|_: &IdealOrZero<ImpossibleIdeal>| TrivialIdeal, &fcfg,
                    ).unwrap();
                    let proj_elem: PiopField = tr.get_field_challenge(&fcfg);
                    let c1_psc = project_scalars::<PiopField, Sha256Uair>(|s| {
                        let one = PiopField::one_with_cfg(&fcfg); let zero = PiopField::zero_with_cfg(&fcfg);
                        DynamicPolynomialF::new(s.iter().map(|c| if c.into_inner() { one.clone() } else { zero.clone() }).collect::<Vec<_>>())
                    });
                    let c1_fps = project_scalars_to_field(c1_psc, &proj_elem).unwrap();
                    let c1_cpr_pre = CombinedPolyResolver::<PiopField>::build_verifier_pre_sumcheck::<Sha256Uair>(
                        &mut tr, &field_from_bytes(&dual_proof.md_claimed_sums[0], &fcfg),
                        sha_num_constraints, &proj_elem, &c1_fps, &c1_sub, &fcfg,
                    ).unwrap();
                    let mut lk_pres = Vec::new();
                    for (gp, meta) in dual_proof.lookup_group_proofs.iter().zip(dual_proof.lookup_group_meta.iter()) {
                        let (subtable, shifts) = generate_table_and_shifts(&meta.table_type, &proj_elem, &fcfg);
                        lk_pres.push(BatchedDecompLogupProtocol::<PiopField>::build_verifier_pre_sumcheck(
                            &mut tr, gp, &subtable, &shifts, meta.num_columns, meta.witness_len, &fcfg,
                        ).unwrap());
                    }
                    let c2_psc = project_scalars::<PiopField, EcdsaUairInt>(|s| {
                        DynamicPolynomialF { coeffs: vec![PiopField::from_with_cfg(s.clone(), &fcfg)] }
                    });
                    let c2_fps = project_scalars_to_field(c2_psc, &proj_elem).unwrap();
                    let c2_cpr_pre = CombinedPolyResolver::<PiopField>::build_verifier_pre_sumcheck::<EcdsaUairInt>(
                        &mut tr, &field_from_bytes(&dual_proof.md_claimed_sums[1], &fcfg),
                        ecdsa_num_constraints, &proj_elem, &c2_fps, &c2_sub, &fcfg,
                    ).unwrap();
                    let shared_nv = lk_pres.iter().map(|p| p.num_vars).max().map_or(num_vars, |m| m.max(num_vars));
                    let md_proof = deser_md(&fcfg);
                    let md_sub = MultiDegreeSumcheck::<PiopField>::verify_as_subprotocol(
                        &mut tr, shared_nv, &md_proof, &fcfg,
                    ).unwrap();

                    // CPR₁ finalize.
                    let c1_priv_up: Vec<PiopField> = dual_proof.cpr1_up_evals.iter().map(|b| field_from_bytes(b, &fcfg)).collect();
                    let c1_down: Vec<PiopField> = dual_proof.cpr1_down_evals.iter().map(|b| field_from_bytes(b, &fcfg)).collect();
                    let c1_sig_v = Sha256Uair::signature();
                    let c1_up = if c1_sig_v.public_columns.is_empty() { c1_priv_up.clone() } else {
                        let bp = BinaryPoly::<32>::prepare_projection(&proj_elem);
                        let pe: Vec<PiopField> = sha_public_column_data.iter().map(|col| {
                            let mut mle: DenseMultilinearExtension<<PiopField as Field>::Inner> = col.iter().map(|v| bp(v).inner().clone()).collect();
                            if md_sub.point.len() > mle.num_vars { mle.evaluations.resize(1 << md_sub.point.len(), Default::default()); mle.num_vars = md_sub.point.len(); }
                            mle.evaluate_with_config(&md_sub.point, &fcfg).unwrap()
                        }).collect();
                        reconstruct_up_evals(&c1_priv_up, &pe, &c1_sig_v.public_columns, c1_sig_v.total_cols())
                    };
                    let c1_up_saved = c1_up.clone();
                    let c1_cpr_subclaim = CombinedPolyResolver::<PiopField>::finalize_verifier::<Sha256Uair>(
                        &mut tr, md_sub.point.clone(), md_sub.expected_evaluations[0].clone(),
                        &c1_cpr_pre, c1_up, c1_down, num_vars, &c1_fps, &fcfg,
                    ).unwrap();

                    let c2_priv_up: Vec<PiopField> = dual_proof.cpr2_up_evals.iter().map(|b| field_from_bytes(b, &fcfg)).collect();
                    let c2_down: Vec<PiopField> = dual_proof.cpr2_down_evals.iter().map(|b| field_from_bytes(b, &fcfg)).collect();
                    let c2_sig_v = EcdsaUairInt::signature();
                    let c2_up = if c2_sig_v.public_columns.is_empty() { c2_priv_up.clone() } else {
                        let pe: Vec<PiopField> = ecdsa_public_column_data.iter().map(|col| {
                            let pc: DenseMultilinearExtension<<PiopField as Field>::Inner> = col.iter().map(|v| PiopField::from_with_cfg(v.clone(), &fcfg).inner().clone()).collect();
                            pc.evaluate_with_config(&md_sub.point, &fcfg).unwrap()
                        }).collect();
                        reconstruct_up_evals(&c2_priv_up, &pe, &c2_sig_v.public_columns, c2_sig_v.total_cols())
                    };
                    let c2_up_saved = c2_up.clone();
                    let c2_down_saved = c2_down.clone();
                    let _ = CombinedPolyResolver::<PiopField>::finalize_verifier::<EcdsaUairInt>(
                        &mut tr, md_sub.point.clone(), md_sub.expected_evaluations[1].clone(),
                        &c2_cpr_pre, c2_up, c2_down, num_vars, &c2_fps, &fcfg,
                    ).unwrap();

                    // ── Timed section ──
                    let t = Instant::now();

                    if let Some(ref ss_data) = dual_proof.unified_eval_sumcheck {
                        let c1_num_up = c1_up_saved.len();
                        let c2_num_up = c2_up_saved.len();
                        let mut claims: Vec<ShiftClaim<PiopField>> = Vec::new();
                        for i in 0..c1_num_up {
                            claims.push(ShiftClaim {
                                source_col: i, shift_amount: 0,
                                eval_point: c1_cpr_subclaim.evaluation_point.clone(),
                                claimed_eval: c1_up_saved[i].clone(),
                            });
                        }
                        for j in 0..c2_num_up {
                            claims.push(ShiftClaim {
                                source_col: c1_num_up + j, shift_amount: 0,
                                eval_point: c1_cpr_subclaim.evaluation_point.clone(),
                                claimed_eval: c2_up_saved[j].clone(),
                            });
                        }
                        for (k, spec) in c2_sig_v.shifts.iter().enumerate() {
                            claims.push(ShiftClaim {
                                source_col: c1_num_up + spec.source_col,
                                shift_amount: spec.shift_amount,
                                eval_point: c1_cpr_subclaim.evaluation_point.clone(),
                                claimed_eval: c2_down_saved[k].clone(),
                            });
                        }
                        let rounds: Vec<ShiftRoundPoly<PiopField>> = ss_data.rounds.iter().map(|bytes| {
                            ShiftRoundPoly {
                                evals: [
                                    field_from_bytes(&bytes[0..field_elem_size], &fcfg),
                                    field_from_bytes(&bytes[field_elem_size..2*field_elem_size], &fcfg),
                                    field_from_bytes(&bytes[2*field_elem_size..3*field_elem_size], &fcfg),
                                ],
                            }
                        }).collect();
                        let ss_proof = ShiftSumcheckProof { rounds };

                        // Determine public unified claims.
                        let is_pub_unified = |idx: usize| -> bool {
                            if idx < c1_num_up {
                                c1_sig_v.is_public_column(idx)
                            } else if idx < c1_num_up + c2_num_up {
                                c2_sig_v.is_public_column(idx - c1_num_up)
                            } else {
                                c2_sig_v.is_public_shift(idx - c1_num_up - c2_num_up)
                            }
                        };
                        let has_public = (0..claims.len()).any(&is_pub_unified);

                        if has_public {
                            let ss_pre = shift_sumcheck_verify_pre(
                                &mut tr, &ss_proof, &claims, num_vars, &fcfg,
                            ).expect("eval SC pre-verify");
                            let challenge_point_le: Vec<PiopField> =
                                ss_pre.challenge_point.iter().rev().cloned().collect();
                            let bp = BinaryPoly::<32>::prepare_projection(&proj_elem);
                            let private_v_finals: Vec<PiopField> = ss_data.v_finals.iter()
                                .map(|b| field_from_bytes(b, &fcfg)).collect();
                            let total_claims = claims.len();
                            let mut full_v_finals = Vec::with_capacity(total_claims);
                            let mut priv_idx = 0usize;
                            for idx in 0..total_claims {
                                if is_pub_unified(idx) {
                                    let v = if idx < c1_num_up {
                                        let pcd_idx = c1_sig_v.public_columns.iter()
                                            .position(|&c| c == idx).unwrap();
                                        let mle: DenseMultilinearExtension<<PiopField as Field>::Inner> =
                                            sha_public_column_data[pcd_idx].iter()
                                                .map(|v| bp(v).inner().clone()).collect();
                                        mle.evaluate_with_config(&challenge_point_le, &fcfg).unwrap()
                                    } else if idx < c1_num_up + c2_num_up {
                                        let col_idx = idx - c1_num_up;
                                        let pcd_idx = c2_sig_v.public_columns.iter()
                                            .position(|&c| c == col_idx).unwrap();
                                        let pc: DenseMultilinearExtension<<PiopField as Field>::Inner> =
                                            ecdsa_public_column_data[pcd_idx].iter()
                                                .map(|v| PiopField::from_with_cfg(v.clone(), &fcfg).inner().clone()).collect();
                                        pc.evaluate_with_config(&challenge_point_le, &fcfg).unwrap()
                                    } else {
                                        let shift_idx = idx - c1_num_up - c2_num_up;
                                        let spec = &c2_sig_v.shifts[shift_idx];
                                        let pcd_idx = c2_sig_v.public_columns.iter()
                                            .position(|&c| c == spec.source_col).unwrap();
                                        let pc: DenseMultilinearExtension<<PiopField as Field>::Inner> =
                                            ecdsa_public_column_data[pcd_idx].iter()
                                                .map(|v| PiopField::from_with_cfg(v.clone(), &fcfg).inner().clone()).collect();
                                        pc.evaluate_with_config(&challenge_point_le, &fcfg).unwrap()
                                    };
                                    full_v_finals.push(v);
                                } else {
                                    full_v_finals.push(private_v_finals[priv_idx].clone());
                                    priv_idx += 1;
                                }
                            }
                            let _ = shift_sumcheck_verify_finalize(
                                &mut tr, &ss_pre, &claims, &full_v_finals, &fcfg,
                            ).expect("eval SC verify");
                        } else {
                            let v_finals: Vec<PiopField> = ss_data.v_finals.iter()
                                .map(|b| field_from_bytes(b, &fcfg)).collect();
                            let _ = shift_sumcheck_verify(
                                &mut tr, &ss_proof, &claims, &v_finals, num_vars, &fcfg,
                            ).expect("eval SC verify");
                        }
                    }

                    total += t.elapsed();
                }
                total
            });
        });

        // ── V7. Verifier / Lookup Finalize ──────────────────────────
        group.bench_function("V/LookupFinalize", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    // Full replay through unified eval sumcheck.
                    let mut tr = zinc_transcript::KeccakTranscript::new();
                    let fcfg = tr.get_random_field_cfg::<PiopField, <PiopField as Field>::Inner, MillerRabin>();
                    let ic_pt: Vec<PiopField> = tr.get_field_challenges(num_vars, &fcfg);
                    let c1_sub = IdealCheckProtocol::<PiopField>::verify_at_point::<Sha256Uair, _, _>(
                        &mut tr, zinc_piop::ideal_check::Proof { combined_mle_values: deser_ic(&dual_proof.ic1_proof_values, &fcfg) },
                        sha_num_constraints, ic_pt.clone(), &|_: &IdealOrZero<CyclotomicIdeal>| TrivialIdeal, &fcfg,
                    ).unwrap();
                    let c2_sub = IdealCheckProtocol::<PiopField>::verify_at_point::<EcdsaUairInt, _, _>(
                        &mut tr, zinc_piop::ideal_check::Proof { combined_mle_values: deser_ic(&dual_proof.ic2_proof_values, &fcfg) },
                        ecdsa_num_constraints, ic_pt,
                        &|_: &IdealOrZero<ImpossibleIdeal>| TrivialIdeal, &fcfg,
                    ).unwrap();
                    let proj_elem: PiopField = tr.get_field_challenge(&fcfg);
                    let c1_fps = project_scalars_to_field(project_scalars::<PiopField, Sha256Uair>(|s| {
                        let one = PiopField::one_with_cfg(&fcfg); let zero = PiopField::zero_with_cfg(&fcfg);
                        DynamicPolynomialF::new(s.iter().map(|c| if c.into_inner() { one.clone() } else { zero.clone() }).collect::<Vec<_>>())
                    }), &proj_elem).unwrap();
                    let c1_cpr_pre = CombinedPolyResolver::<PiopField>::build_verifier_pre_sumcheck::<Sha256Uair>(
                        &mut tr, &field_from_bytes(&dual_proof.md_claimed_sums[0], &fcfg),
                        sha_num_constraints, &proj_elem, &c1_fps, &c1_sub, &fcfg,
                    ).unwrap();
                    let mut lk_pres = Vec::new();
                    for (gp, meta) in dual_proof.lookup_group_proofs.iter().zip(dual_proof.lookup_group_meta.iter()) {
                        let (st, sh) = generate_table_and_shifts(&meta.table_type, &proj_elem, &fcfg);
                        lk_pres.push(BatchedDecompLogupProtocol::<PiopField>::build_verifier_pre_sumcheck(
                            &mut tr, gp, &st, &sh, meta.num_columns, meta.witness_len, &fcfg,
                        ).unwrap());
                    }
                    let c2_fps = project_scalars_to_field(project_scalars::<PiopField, EcdsaUairInt>(|s| {
                        DynamicPolynomialF { coeffs: vec![PiopField::from_with_cfg(s.clone(), &fcfg)] }
                    }), &proj_elem).unwrap();
                    let c2_cpr_pre = CombinedPolyResolver::<PiopField>::build_verifier_pre_sumcheck::<EcdsaUairInt>(
                        &mut tr, &field_from_bytes(&dual_proof.md_claimed_sums[1], &fcfg),
                        ecdsa_num_constraints, &proj_elem, &c2_fps, &c2_sub, &fcfg,
                    ).unwrap();
                    let shared_nv = lk_pres.iter().map(|p| p.num_vars).max().map_or(num_vars, |m| m.max(num_vars));
                    let md_sub = MultiDegreeSumcheck::<PiopField>::verify_as_subprotocol(
                        &mut tr, shared_nv, &deser_md(&fcfg), &fcfg,
                    ).unwrap();

                    // CPR finalize (both).
                    let c1_priv_up: Vec<PiopField> = dual_proof.cpr1_up_evals.iter().map(|b| field_from_bytes(b, &fcfg)).collect();
                    let c1_down: Vec<PiopField> = dual_proof.cpr1_down_evals.iter().map(|b| field_from_bytes(b, &fcfg)).collect();
                    let c1_sig_v = Sha256Uair::signature();
                    let c1_up = if c1_sig_v.public_columns.is_empty() { c1_priv_up } else {
                        let bp = BinaryPoly::<32>::prepare_projection(&proj_elem);
                        let pe: Vec<PiopField> = sha_public_column_data.iter().map(|col| {
                            let mut mle: DenseMultilinearExtension<<PiopField as Field>::Inner> = col.iter().map(|v| bp(v).inner().clone()).collect();
                            if md_sub.point.len() > mle.num_vars { mle.evaluations.resize(1 << md_sub.point.len(), Default::default()); mle.num_vars = md_sub.point.len(); }
                            mle.evaluate_with_config(&md_sub.point, &fcfg).unwrap()
                        }).collect();
                        reconstruct_up_evals(&c1_priv_up, &pe, &c1_sig_v.public_columns, c1_sig_v.total_cols())
                    };
                    let c1_up_saved = c1_up.clone();
                    let c1_cpr_subclaim = CombinedPolyResolver::<PiopField>::finalize_verifier::<Sha256Uair>(
                        &mut tr, md_sub.point.clone(), md_sub.expected_evaluations[0].clone(),
                        &c1_cpr_pre, c1_up, c1_down, num_vars, &c1_fps, &fcfg,
                    ).unwrap();
                    let c2_priv_up: Vec<PiopField> = dual_proof.cpr2_up_evals.iter().map(|b| field_from_bytes(b, &fcfg)).collect();
                    let c2_down: Vec<PiopField> = dual_proof.cpr2_down_evals.iter().map(|b| field_from_bytes(b, &fcfg)).collect();
                    let c2_sig_v = EcdsaUairInt::signature();
                    let c2_up = if c2_sig_v.public_columns.is_empty() { c2_priv_up } else {
                        let pe: Vec<PiopField> = ecdsa_public_column_data.iter().map(|col| {
                            let pc: DenseMultilinearExtension<<PiopField as Field>::Inner> = col.iter().map(|v| PiopField::from_with_cfg(v.clone(), &fcfg).inner().clone()).collect();
                            pc.evaluate_with_config(&md_sub.point, &fcfg).unwrap()
                        }).collect();
                        reconstruct_up_evals(&c2_priv_up, &pe, &c2_sig_v.public_columns, c2_sig_v.total_cols())
                    };
                    let c2_up_saved = c2_up.clone();
                    let c2_down_saved = c2_down.clone();
                    let _ = CombinedPolyResolver::<PiopField>::finalize_verifier::<EcdsaUairInt>(
                        &mut tr, md_sub.point.clone(), md_sub.expected_evaluations[1].clone(),
                        &c2_cpr_pre, c2_up, c2_down, num_vars, &c2_fps, &fcfg,
                    ).unwrap();

                    // Unified eval sumcheck.
                    if let Some(ref ss_data) = dual_proof.unified_eval_sumcheck {
                        let c1_num_up = c1_up_saved.len();
                        let c2_num_up = c2_up_saved.len();
                        let mut claims: Vec<ShiftClaim<PiopField>> = Vec::new();
                        for i in 0..c1_num_up { claims.push(ShiftClaim { source_col: i, shift_amount: 0, eval_point: c1_cpr_subclaim.evaluation_point.clone(), claimed_eval: c1_up_saved[i].clone() }); }
                        for j in 0..c2_num_up { claims.push(ShiftClaim { source_col: c1_num_up + j, shift_amount: 0, eval_point: c1_cpr_subclaim.evaluation_point.clone(), claimed_eval: c2_up_saved[j].clone() }); }
                        for (k, spec) in c2_sig_v.shifts.iter().enumerate() { claims.push(ShiftClaim { source_col: c1_num_up + spec.source_col, shift_amount: spec.shift_amount, eval_point: c1_cpr_subclaim.evaluation_point.clone(), claimed_eval: c2_down_saved[k].clone() }); }
                        let rounds: Vec<ShiftRoundPoly<PiopField>> = ss_data.rounds.iter().map(|bytes| ShiftRoundPoly { evals: [
                            field_from_bytes(&bytes[0..field_elem_size], &fcfg),
                            field_from_bytes(&bytes[field_elem_size..2*field_elem_size], &fcfg),
                            field_from_bytes(&bytes[2*field_elem_size..3*field_elem_size], &fcfg),
                        ]}).collect();
                        let is_pub_unified = |idx: usize| -> bool {
                            if idx < c1_num_up {
                                c1_sig_v.is_public_column(idx)
                            } else if idx < c1_num_up + c2_num_up {
                                c2_sig_v.is_public_column(idx - c1_num_up)
                            } else {
                                c2_sig_v.is_public_shift(idx - c1_num_up - c2_num_up)
                            }
                        };
                        let has_public = (0..claims.len()).any(&is_pub_unified);
                        let ss_proof = ShiftSumcheckProof { rounds };

                        if has_public {
                            let ss_pre = shift_sumcheck_verify_pre(
                                &mut tr, &ss_proof, &claims, num_vars, &fcfg,
                            ).unwrap();
                            let challenge_point_le: Vec<PiopField> =
                                ss_pre.challenge_point.iter().rev().cloned().collect();
                            let bp = BinaryPoly::<32>::prepare_projection(&proj_elem);
                            let private_v_finals: Vec<PiopField> = ss_data.v_finals.iter()
                                .map(|b| field_from_bytes(b, &fcfg)).collect();
                            let total_claims = claims.len();
                            let mut full_v_finals = Vec::with_capacity(total_claims);
                            let mut priv_idx = 0usize;
                            for idx in 0..total_claims {
                                if is_pub_unified(idx) {
                                    let v = if idx < c1_num_up {
                                        let pcd_idx = c1_sig_v.public_columns.iter()
                                            .position(|&c| c == idx).unwrap();
                                        let mle: DenseMultilinearExtension<<PiopField as Field>::Inner> =
                                            sha_public_column_data[pcd_idx].iter()
                                                .map(|v| bp(v).inner().clone()).collect();
                                        mle.evaluate_with_config(&challenge_point_le, &fcfg).unwrap()
                                    } else if idx < c1_num_up + c2_num_up {
                                        let col_idx = idx - c1_num_up;
                                        let pcd_idx = c2_sig_v.public_columns.iter()
                                            .position(|&c| c == col_idx).unwrap();
                                        let pc: DenseMultilinearExtension<<PiopField as Field>::Inner> =
                                            ecdsa_public_column_data[pcd_idx].iter()
                                                .map(|v| PiopField::from_with_cfg(v.clone(), &fcfg).inner().clone()).collect();
                                        pc.evaluate_with_config(&challenge_point_le, &fcfg).unwrap()
                                    } else {
                                        let shift_idx = idx - c1_num_up - c2_num_up;
                                        let spec = &c2_sig_v.shifts[shift_idx];
                                        let pcd_idx = c2_sig_v.public_columns.iter()
                                            .position(|&c| c == spec.source_col).unwrap();
                                        let pc: DenseMultilinearExtension<<PiopField as Field>::Inner> =
                                            ecdsa_public_column_data[pcd_idx].iter()
                                                .map(|v| PiopField::from_with_cfg(v.clone(), &fcfg).inner().clone()).collect();
                                        pc.evaluate_with_config(&challenge_point_le, &fcfg).unwrap()
                                    };
                                    full_v_finals.push(v);
                                } else {
                                    full_v_finals.push(private_v_finals[priv_idx].clone());
                                    priv_idx += 1;
                                }
                            }
                            let _ = shift_sumcheck_verify_finalize(
                                &mut tr, &ss_pre, &claims, &full_v_finals, &fcfg,
                            ).unwrap();
                        } else {
                            let v_finals: Vec<PiopField> = ss_data.v_finals.iter()
                                .map(|b| field_from_bytes(b, &fcfg)).collect();
                            let _ = shift_sumcheck_verify(
                                &mut tr, &ss_proof, &claims, &v_finals, num_vars, &fcfg,
                            ).unwrap();
                        }
                    }

                    // ── Timed section ──
                    let t = Instant::now();
                    for (g, (lk_pre, gp)) in lk_pres.iter()
                        .zip(dual_proof.lookup_group_proofs.iter()).enumerate()
                    {
                        let meta = &dual_proof.lookup_group_meta[g];
                        // eq_sum_w for domain-scaling of affine constant offsets.
                        let eq_sum_w = {
                            let one = PiopField::one_with_cfg(&fcfg);
                            let w_nv = zinc_utils::log2(meta.witness_len.next_power_of_two()) as usize;
                            let mut prod = one.clone();
                            for i in w_nv..md_sub.point.len() {
                                prod *= one.clone() - &md_sub.point[i];
                            }
                            prod
                        };
                        let parent_evals: Vec<PiopField> = meta
                            .witness_sources.iter()
                            .map(|ws| match ws {
                                LookupWitnessSource::Column { column_index } =>
                                    c1_up_saved[*column_index].clone(),
                                LookupWitnessSource::Affine { terms, constant_offset_bits } =>
                                    zinc_snark::pipeline::eval_affine_parent::<32>(
                                        terms, *constant_offset_bits,
                                        &c1_up_saved,
                                        &proj_elem, &eq_sum_w,
                                        &fcfg,
                                    ),
                            })
                            .collect();
                        BatchedDecompLogupProtocol::<PiopField>::finalize_verifier(
                            lk_pre, gp, &md_sub.point,
                            &md_sub.expected_evaluations[g + 2],
                            &parent_evals,
                            &fcfg,
                        ).expect("lookup finalize");
                    }
                    total += t.elapsed();
                }
                total
            });
        });

        // ── V8. Verifier / PCS Verify ───────────────────────────────
        //
        // Verify both PCS proofs. This does NOT require replaying the
        // PIOP transcript — the PCS verifier uses its own independent
        // transcript and the committed evaluation point.
        group.bench_function("V/PCSVerify", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    // We only need the evaluation point to verify PCS.
                    // Replay just enough to derive it.
                    let mut tr = zinc_transcript::KeccakTranscript::new();
                    let fcfg = tr.get_random_field_cfg::<PiopField, <PiopField as Field>::Inner, MillerRabin>();
                    let ic_pt: Vec<PiopField> = tr.get_field_challenges(num_vars, &fcfg);
                    let c1_sub = IdealCheckProtocol::<PiopField>::verify_at_point::<Sha256Uair, _, _>(
                        &mut tr, zinc_piop::ideal_check::Proof { combined_mle_values: deser_ic(&dual_proof.ic1_proof_values, &fcfg) },
                        sha_num_constraints, ic_pt.clone(), &|_: &IdealOrZero<CyclotomicIdeal>| TrivialIdeal, &fcfg,
                    ).unwrap();
                    let c2_sub = IdealCheckProtocol::<PiopField>::verify_at_point::<EcdsaUairInt, _, _>(
                        &mut tr, zinc_piop::ideal_check::Proof { combined_mle_values: deser_ic(&dual_proof.ic2_proof_values, &fcfg) },
                        ecdsa_num_constraints, ic_pt,
                        &|_: &IdealOrZero<ImpossibleIdeal>| TrivialIdeal, &fcfg,
                    ).unwrap();
                    let proj_elem: PiopField = tr.get_field_challenge(&fcfg);
                    let c1_fps = project_scalars_to_field(project_scalars::<PiopField, Sha256Uair>(|s| {
                        let one = PiopField::one_with_cfg(&fcfg); let zero = PiopField::zero_with_cfg(&fcfg);
                        DynamicPolynomialF::new(s.iter().map(|c| if c.into_inner() { one.clone() } else { zero.clone() }).collect::<Vec<_>>())
                    }), &proj_elem).unwrap();
                    let _c1_cpr_pre = CombinedPolyResolver::<PiopField>::build_verifier_pre_sumcheck::<Sha256Uair>(
                        &mut tr, &field_from_bytes(&dual_proof.md_claimed_sums[0], &fcfg),
                        sha_num_constraints, &proj_elem, &c1_fps, &c1_sub, &fcfg,
                    ).unwrap();
                    for (gp, meta) in dual_proof.lookup_group_proofs.iter().zip(dual_proof.lookup_group_meta.iter()) {
                        let (st, sh) = generate_table_and_shifts(&meta.table_type, &proj_elem, &fcfg);
                        let _ = BatchedDecompLogupProtocol::<PiopField>::build_verifier_pre_sumcheck(
                            &mut tr, gp, &st, &sh, meta.num_columns, meta.witness_len, &fcfg,
                        ).unwrap();
                    }
                    let c2_fps = project_scalars_to_field(project_scalars::<PiopField, EcdsaUairInt>(|s| {
                        DynamicPolynomialF { coeffs: vec![PiopField::from_with_cfg(s.clone(), &fcfg)] }
                    }), &proj_elem).unwrap();
                    let _ = CombinedPolyResolver::<PiopField>::build_verifier_pre_sumcheck::<EcdsaUairInt>(
                        &mut tr, &field_from_bytes(&dual_proof.md_claimed_sums[1], &fcfg),
                        ecdsa_num_constraints, &proj_elem, &c2_fps, &c2_sub, &fcfg,
                    ).unwrap();
                    // Derive PCS eval point from the proof's stored evaluation point.
                    let cpr_eval_pt: Vec<PiopField> = dual_proof.evaluation_point_bytes.iter()
                        .map(|b| field_from_bytes(b, &fcfg)).collect();

                    // ── Timed section ──
                    let t = Instant::now();

                    // PCS₁ + PCS₂ verify in parallel (matches verify_dual_circuit).
                    let verify_pcs1 = || {
                        let mut pcs1_tr = zip_plus::pcs_transcript::PcsTranscript {
                            fs_transcript: zinc_transcript::KeccakTranscript::default(),
                            stream: std::io::Cursor::new(dual_proof.pcs1_proof_bytes.clone()),
                        };
                        let pcs1_fcfg = pcs1_tr.fs_transcript
                            .get_random_field_cfg::<PiopField, <Sha256ZipTypes<i64, 32> as ZipTypes>::Fmod, <Sha256ZipTypes<i64, 32> as ZipTypes>::PrimeTest>();
                        let pt1_f: Vec<PiopField> = cpr_eval_pt[..num_vars].to_vec();
                        let eval1_f: PiopField = PiopField::new_unchecked_with_cfg(
                            <Uint<{INT_LIMBS * 3}> as ConstTranscribable>::read_transcription_bytes(&dual_proof.pcs1_evals_bytes[0]),
                            &pcs1_fcfg,
                        );
                        ZipPlus::<ShaZt, ShaLc>::verify_with_field_cfg::<PiopField, UNCHECKED>(
                            &sha_params, &dual_proof.pcs1_commitment, &pt1_f, &eval1_f, pcs1_tr, &pcs1_fcfg,
                        ).expect("PCS1 verify");
                    };

                    let verify_pcs2 = || {
                        let mut pcs2_tr = zip_plus::pcs_transcript::PcsTranscript {
                            fs_transcript: zinc_transcript::KeccakTranscript::default(),
                            stream: std::io::Cursor::new(dual_proof.pcs2_proof_bytes.clone()),
                        };
                        let pcs2_fcfg = pcs2_tr.fs_transcript
                            .get_random_field_cfg::<FScalar, <EcdsaScalarZipTypes as ZipTypes>::Fmod, <EcdsaScalarZipTypes as ZipTypes>::PrimeTest>();
                        let pt2_f: Vec<FScalar> = zinc_snark::pipeline::piop_point_to_pcs_field(&cpr_eval_pt[..num_vars], &pcs2_fcfg);
                        let eval2_f: FScalar = FScalar::new_unchecked_with_cfg(
                            <Uint<{INT_LIMBS * 4}> as ConstTranscribable>::read_transcription_bytes(&dual_proof.pcs2_evals_bytes[0]),
                            &pcs2_fcfg,
                        );
                        ZipPlus::<EcZt, EcLc>::verify_with_field_cfg::<FScalar, UNCHECKED>(
                            &ec_params, &dual_proof.pcs2_commitment, &pt2_f, &eval2_f, pcs2_tr, &pcs2_fcfg,
                        ).expect("PCS2 verify");
                    };

                    rayon::join(verify_pcs1, verify_pcs2);

                    total += t.elapsed();
                }
                total
            });
        });

        // ── V9. Verifier / Feed-Forward + Hash→ECDSA Connection ──────
        //
        // Measures the verifier-side computation that bridges the proved
        // SHA-256 compression with the proved ECDSA verification:
        //   (A) SHA-256 feed-forward: extract 8 final working variables,
        //       add initial hash values H[0..7]  →  8 wrapping adds.
        //   (B) Reconstruct scalar u₁ from ECDSA public column b₁ bits.
        //   (C) (In production) compare u₁ with hash · s⁻¹ mod n.
        //
        // Cost is O(N_sha + N_ecdsa) field-element reads + negligible
        // arithmetic, dominated by the b₁ bit scan (256 iterations).
        group.bench_function("V/FeedFwd+Connect", |b| {
            b.iter(|| {
                // (A) Feed-forward for SHA instance 0.
                let digest = black_box(feed_forward(&sha_trace, 0));
                // (B) Reconstruct u₁ from ECDSA b₁ bits.
                let u1 = black_box(reconstruct_u1(&ecdsa_trace));
            });
        });
    }

    // ── Timing breakdown summary ────────────────────────────────────
    eprintln!("\n=== 8xSHA256+ECDSA Per-Step Timing (Unified Dual-Circuit) ===");
    eprintln!("  Dual-circuit: PCS commit={:?}, IC={:?}, CPR+Lookup={:?}, PCS prove={:?}, serialize={:?}, total={:?}",
        dual_proof.timing.pcs_commit,
        dual_proof.timing.ideal_check,
        dual_proof.timing.combined_poly_resolver,
        dual_proof.timing.pcs_prove,
        dual_proof.timing.serialize,
        dual_proof.timing.total,
    );
    eprintln!("  Multi-degree sumcheck: {} groups, degrees {:?}",
        dual_proof.md_degrees.len(),
        dual_proof.md_degrees,
    );

    // ── Proof size breakdown ────────────────────────────────────────
    {
        use zinc_snark::pipeline::FIELD_LIMBS;
        use crypto_primitives::crypto_bigint_uint::Uint;
        let fe_bytes = <Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;

        // PCS proof bytes (raw serialized).
        let pcs1_bytes = dual_proof.pcs1_proof_bytes.len();
        let pcs2_bytes = dual_proof.pcs2_proof_bytes.len();

        // IC proof values.
        let ic1_bytes: usize = dual_proof.ic1_proof_values.iter().map(|v| v.len()).sum();
        let ic2_bytes: usize = dual_proof.ic2_proof_values.iter().map(|v| v.len()).sum();

        // Multi-degree sumcheck messages + claimed sums.
        let md_msg_bytes: usize = dual_proof.md_group_messages.iter()
            .flat_map(|grp| grp.iter())
            .map(|v| v.len())
            .sum();
        let md_sum_bytes: usize = dual_proof.md_claimed_sums.iter().map(|v| v.len()).sum();
        let md_total = md_msg_bytes + md_sum_bytes;

        // Main field sumcheck up/down evaluations.
        let cpr1_up: usize = dual_proof.cpr1_up_evals.iter().map(|v| v.len()).sum();
        let cpr1_dn: usize = dual_proof.cpr1_down_evals.iter().map(|v| v.len()).sum();
        let cpr2_up: usize = dual_proof.cpr2_up_evals.iter().map(|v| v.len()).sum();
        let cpr2_dn: usize = dual_proof.cpr2_down_evals.iter().map(|v| v.len()).sum();
        let cpr_total = cpr1_up + cpr1_dn + cpr2_up + cpr2_dn;

        // Unified evaluation sumcheck (eq + shift claims).
        let eval_sc_bytes: usize = dual_proof.unified_eval_sumcheck.as_ref().map_or(0, |sc| {
            let rounds: usize = sc.rounds.iter().map(|v| v.len()).sum();
            let finals: usize = sc.v_finals.iter().map(|v| v.len()).sum();
            rounds + finals
        });

        // Lookup data.
        let lookup_bytes: usize = {
            let _meta: usize = dual_proof.lookup_group_meta.len() * std::mem::size_of::<zinc_piop::lookup::LookupGroupMeta>();
            let proofs: usize = dual_proof.lookup_group_proofs.iter().map(|gp| {
                let mults: usize = gp.aggregated_multiplicities.iter().map(|v| v.len()).sum();
                let inv_w: usize = gp.chunk_inverse_witnesses.iter()
                    .flat_map(|outer| outer.iter())
                    .map(|inner| inner.len())
                    .sum();
                let inv_t = gp.inverse_table.len();
                (mults + inv_w + inv_t) * fe_bytes
            }).sum();
            proofs
        };

        // Evaluation point + PCS evals.
        let eval_pt_bytes: usize = dual_proof.evaluation_point_bytes.iter().map(|v| v.len()).sum();
        let pcs1_eval_bytes: usize = dual_proof.pcs1_evals_bytes.iter().map(|v| v.len()).sum();
        let pcs2_eval_bytes: usize = dual_proof.pcs2_evals_bytes.iter().map(|v| v.len()).sum();

        let piop_total = ic1_bytes + ic2_bytes + md_total + cpr_total
            + eval_sc_bytes + lookup_bytes + eval_pt_bytes
            + pcs1_eval_bytes + pcs2_eval_bytes;
        let total_raw = pcs1_bytes + pcs2_bytes + piop_total;

        // Compressed size (deflate).
        let mut all_bytes = Vec::with_capacity(total_raw);
        all_bytes.extend(&dual_proof.pcs1_proof_bytes);
        all_bytes.extend(&dual_proof.pcs2_proof_bytes);
        for v in &dual_proof.ic1_proof_values { all_bytes.extend(v); }
        for v in &dual_proof.ic2_proof_values { all_bytes.extend(v); }
        for grp in &dual_proof.md_group_messages {
            for v in grp { all_bytes.extend(v); }
        }
        for v in &dual_proof.md_claimed_sums { all_bytes.extend(v); }
        for v in &dual_proof.cpr1_up_evals { all_bytes.extend(v); }
        for v in &dual_proof.cpr1_down_evals { all_bytes.extend(v); }
        for v in &dual_proof.cpr2_up_evals { all_bytes.extend(v); }
        for v in &dual_proof.cpr2_down_evals { all_bytes.extend(v); }
        if let Some(ref sc) = dual_proof.unified_eval_sumcheck {
            for v in &sc.rounds { all_bytes.extend(v); }
            for v in &sc.v_finals { all_bytes.extend(v); }
        }
        // Lookup field elements (serialize inline).
        for gp in &dual_proof.lookup_group_proofs {
            fn write_fe(buf: &mut Vec<u8>, f: &zinc_snark::pipeline::PiopField) {
                use zinc_snark::pipeline::FIELD_LIMBS;
                let sz = <Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;
                let start = buf.len();
                buf.resize(start + sz, 0);
                f.inner().write_transcription_bytes(&mut buf[start..]);
            }
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
        for v in &dual_proof.evaluation_point_bytes { all_bytes.extend(v); }
        for v in &dual_proof.pcs1_evals_bytes { all_bytes.extend(v); }
        for v in &dual_proof.pcs2_evals_bytes { all_bytes.extend(v); }

        let compressed = {
            use std::io::Write;
            let mut encoder = flate2::write::DeflateEncoder::new(
                Vec::new(), flate2::Compression::default(),
            );
            encoder.write_all(&all_bytes).unwrap();
            encoder.finish().unwrap()
        };

        // Compress PCS (SHA) and PCS (ECDSA) individually.
        let pcs1_compressed = {
            use std::io::Write;
            let mut encoder = flate2::write::DeflateEncoder::new(
                Vec::new(), flate2::Compression::default(),
            );
            encoder.write_all(&dual_proof.pcs1_proof_bytes).unwrap();
            encoder.finish().unwrap()
        };
        let pcs2_compressed = {
            use std::io::Write;
            let mut encoder = flate2::write::DeflateEncoder::new(
                Vec::new(), flate2::Compression::default(),
            );
            encoder.write_all(&dual_proof.pcs2_proof_bytes).unwrap();
            encoder.finish().unwrap()
        };

        eprintln!("\n=== 8xSHA256+ECDSA Dual-Circuit Proof Size ===");
        eprintln!("  PCS (SHA):      {:>6} B  ({:.1} KB)  compressed: {} B ({:.1} KB, {:.1}x)",
            pcs1_bytes, pcs1_bytes as f64 / 1024.0,
            pcs1_compressed.len(), pcs1_compressed.len() as f64 / 1024.0,
            pcs1_bytes as f64 / pcs1_compressed.len() as f64);
        eprintln!("  PCS (ECDSA):    {:>6} B  ({:.1} KB)  compressed: {} B ({:.1} KB, {:.1}x)",
            pcs2_bytes, pcs2_bytes as f64 / 1024.0,
            pcs2_compressed.len(), pcs2_compressed.len() as f64 / 1024.0,
            pcs2_bytes as f64 / pcs2_compressed.len() as f64);
        eprintln!("  IC  (SHA):      {:>6} B", ic1_bytes);
        eprintln!("  IC  (ECDSA):    {:>6} B", ic2_bytes);
        eprintln!("  MD sumcheck:    {:>6} B  (msgs={}, sums={})", md_total, md_msg_bytes, md_sum_bytes);
        eprintln!("  CPR evals:      {:>6} B  (c1_up={}, c1_dn={}, c2_up={}, c2_dn={})",
            cpr_total, cpr1_up, cpr1_dn, cpr2_up, cpr2_dn);
        eprintln!("  Eval sumcheck:  {:>6} B", eval_sc_bytes);
        eprintln!("  Lookup:         {:>6} B  ({} groups)", lookup_bytes, dual_proof.lookup_group_proofs.len());
        eprintln!("  Eval point:     {:>6} B", eval_pt_bytes);
        eprintln!("  PCS evals:      {:>6} B  (c1={}, c2={})", pcs1_eval_bytes + pcs2_eval_bytes, pcs1_eval_bytes, pcs2_eval_bytes);
        eprintln!("  ─────────────────────────");
        eprintln!("  PIOP total:     {:>6} B  ({:.1} KB)", piop_total, piop_total as f64 / 1024.0);
        eprintln!("  Total raw:      {:>6} B  ({:.1} KB)", total_raw, total_raw as f64 / 1024.0);
        eprintln!("  Compressed:     {:>6} B  ({:.1} KB, {:.1}x ratio)",
            compressed.len(), compressed.len() as f64 / 1024.0,
            all_bytes.len() as f64 / compressed.len() as f64);

        // ── GKR lookup proof size comparison ────────────────────────
        //
        // Run the GKR lookup prover on the same data and compare proof
        // sizes with the classic lookup proof produced by prove_dual_circuit.
        {
            // Re-extract lookup columns (same as step 15).
            let mut needed: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
            for spec in &sha_lookup_specs {
                let next = needed.len();
                needed.entry(spec.column_index).or_insert(next);
            }
            let mut lk_columns: Vec<Vec<F>> = Vec::with_capacity(needed.len());
            let mut lk_raw_indices: Vec<Vec<usize>> = Vec::with_capacity(needed.len());
            for &orig_idx in needed.keys() {
                let col_f: Vec<F> = sha_field_trace[orig_idx].iter()
                    .map(|inner| F::new_unchecked_with_cfg(inner.clone(), &sha_fcfg)).collect();
                lk_columns.push(col_f);
                let col_idx: Vec<usize> = sha_trace[orig_idx].iter().map(|bp| {
                    let mut idx = 0usize;
                    for (j, coeff) in bp.iter().enumerate() {
                        if coeff.into_inner() { idx |= 1usize << j; }
                    }
                    idx
                }).collect();
                lk_raw_indices.push(col_idx);
            }
            let index_map: std::collections::BTreeMap<usize, usize> = needed.keys()
                .enumerate().map(|(n, &o)| (o, n)).collect();
            let lk_remapped: Vec<LookupColumnSpec> = sha_lookup_specs.iter()
                .map(|s| LookupColumnSpec { column_index: index_map[&s.column_index], table_type: s.table_type.clone() })
                .collect();

            // Run GKR lookup prover.
            let mut gkr_tr = unified_tr_post_cpr.clone();
            let (gkr_proof, _gkr_state) = prove_gkr_batched_lookup_with_indices(
                &mut gkr_tr,
                &lk_columns,
                &lk_raw_indices,
                &lk_remapped,
                &sha_proj_elem,
                &sha_fcfg,
            ).expect("GKR lookup proof failed");

            // Serialize GKR lookup proof to bytes.
            fn write_fe_gkr(buf: &mut Vec<u8>, f: &zinc_snark::pipeline::PiopField) {
                use zinc_snark::pipeline::FIELD_LIMBS;
                let sz = <Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;
                let start = buf.len();
                buf.resize(start + sz, 0);
                f.inner().write_transcription_bytes(&mut buf[start..]);
            }
            let mut gkr_bytes = Vec::new();
            for gp in &gkr_proof.group_proofs {
                // Aggregated multiplicities (same as classic).
                for v in &gp.aggregated_multiplicities {
                    for f in v { write_fe_gkr(&mut gkr_bytes, f); }
                }
                // Batched witness GKR proof.
                for f in &gp.witness_gkr.roots_p { write_fe_gkr(&mut gkr_bytes, f); }
                for f in &gp.witness_gkr.roots_q { write_fe_gkr(&mut gkr_bytes, f); }
                for lp in &gp.witness_gkr.layer_proofs {
                    if let Some(ref sc) = lp.sumcheck_proof {
                        write_fe_gkr(&mut gkr_bytes, &sc.claimed_sum);
                        for msg in &sc.messages {
                            for f in &msg.0.tail_evaluations { write_fe_gkr(&mut gkr_bytes, f); }
                        }
                    }
                    for f in &lp.p_lefts { write_fe_gkr(&mut gkr_bytes, f); }
                    for f in &lp.p_rights { write_fe_gkr(&mut gkr_bytes, f); }
                    for f in &lp.q_lefts { write_fe_gkr(&mut gkr_bytes, f); }
                    for f in &lp.q_rights { write_fe_gkr(&mut gkr_bytes, f); }
                }
                // Table GKR proof.
                write_fe_gkr(&mut gkr_bytes, &gp.table_gkr.root_p);
                write_fe_gkr(&mut gkr_bytes, &gp.table_gkr.root_q);
                for lp in &gp.table_gkr.layer_proofs {
                    if let Some(ref sc) = lp.sumcheck_proof {
                        write_fe_gkr(&mut gkr_bytes, &sc.claimed_sum);
                        for msg in &sc.messages {
                            for f in &msg.0.tail_evaluations { write_fe_gkr(&mut gkr_bytes, f); }
                        }
                    }
                    write_fe_gkr(&mut gkr_bytes, &lp.p_left);
                    write_fe_gkr(&mut gkr_bytes, &lp.p_right);
                    write_fe_gkr(&mut gkr_bytes, &lp.q_left);
                    write_fe_gkr(&mut gkr_bytes, &lp.q_right);
                }
            }
            let gkr_lookup_bytes = gkr_bytes.len();

            // The classic lookup also includes a sumcheck proof (messages) that
            // is part of the MD sumcheck in the dual pipeline. Compute the
            // classic lookup-only bytes by adding the MD lookup group messages.
            let classic_lookup_md_bytes: usize = if dual_proof.md_group_messages.len() > 2 {
                dual_proof.md_group_messages[2..].iter()
                    .flat_map(|grp| grp.iter())
                    .map(|v| v.len())
                    .sum::<usize>()
                + dual_proof.md_claimed_sums[2..].iter()
                    .map(|v| v.len())
                    .sum::<usize>()
            } else {
                0
            };
            let classic_lookup_total = lookup_bytes + classic_lookup_md_bytes;

            // Build GKR version of all_bytes: replace classic lookup data
            // with GKR lookup data.
            let mut gkr_all_bytes = Vec::with_capacity(total_raw);
            gkr_all_bytes.extend(&dual_proof.pcs1_proof_bytes);
            gkr_all_bytes.extend(&dual_proof.pcs2_proof_bytes);
            for v in &dual_proof.ic1_proof_values { gkr_all_bytes.extend(v); }
            for v in &dual_proof.ic2_proof_values { gkr_all_bytes.extend(v); }
            // MD sumcheck: only include CPR groups (0, 1), skip lookup groups.
            for (i, grp) in dual_proof.md_group_messages.iter().enumerate() {
                if i < 2 { for v in grp { gkr_all_bytes.extend(v); } }
            }
            for (i, v) in dual_proof.md_claimed_sums.iter().enumerate() {
                if i < 2 { gkr_all_bytes.extend(v); }
            }
            for v in &dual_proof.cpr1_up_evals { gkr_all_bytes.extend(v); }
            for v in &dual_proof.cpr1_down_evals { gkr_all_bytes.extend(v); }
            for v in &dual_proof.cpr2_up_evals { gkr_all_bytes.extend(v); }
            for v in &dual_proof.cpr2_down_evals { gkr_all_bytes.extend(v); }
            if let Some(ref sc) = dual_proof.unified_eval_sumcheck {
                for v in &sc.rounds { gkr_all_bytes.extend(v); }
                for v in &sc.v_finals { gkr_all_bytes.extend(v); }
            }
            // GKR lookup bytes instead of classic.
            gkr_all_bytes.extend(&gkr_bytes);
            for v in &dual_proof.evaluation_point_bytes { gkr_all_bytes.extend(v); }
            for v in &dual_proof.pcs1_evals_bytes { gkr_all_bytes.extend(v); }
            for v in &dual_proof.pcs2_evals_bytes { gkr_all_bytes.extend(v); }

            let gkr_total_raw = gkr_all_bytes.len();

            let gkr_compressed = {
                use std::io::Write;
                let mut encoder = flate2::write::DeflateEncoder::new(
                    Vec::new(), flate2::Compression::default(),
                );
                encoder.write_all(&gkr_all_bytes).unwrap();
                encoder.finish().unwrap()
            };

            eprintln!("\n=== Classic vs GKR Lookup Proof Size ===");
            eprintln!("  Classic lookup: {:>6} B  ({:.1} KB)  [data={}, MD sumcheck={}]",
                classic_lookup_total, classic_lookup_total as f64 / 1024.0,
                lookup_bytes, classic_lookup_md_bytes);
            eprintln!("  GKR lookup:     {:>6} B  ({:.1} KB)",
                gkr_lookup_bytes, gkr_lookup_bytes as f64 / 1024.0);
            eprintln!("  Lookup savings: {:>6} B  ({:.1} KB, {:.1}x smaller)",
                classic_lookup_total - gkr_lookup_bytes,
                (classic_lookup_total - gkr_lookup_bytes) as f64 / 1024.0,
                classic_lookup_total as f64 / gkr_lookup_bytes as f64);
            eprintln!("  ─────────────────────────");
            eprintln!("  Classic total raw:  {:>7} B  ({:.1} KB)", total_raw, total_raw as f64 / 1024.0);
            eprintln!("  GKR total raw:      {:>7} B  ({:.1} KB)", gkr_total_raw, gkr_total_raw as f64 / 1024.0);
            eprintln!("  Classic compressed: {:>7} B  ({:.1} KB)", compressed.len(), compressed.len() as f64 / 1024.0);
            eprintln!("  GKR compressed:     {:>7} B  ({:.1} KB)", gkr_compressed.len(), gkr_compressed.len() as f64 / 1024.0);
            eprintln!("  Compressed savings: {:>7} B  ({:.1} KB, {:.1}x smaller)",
                compressed.len() - gkr_compressed.len(),
                (compressed.len() - gkr_compressed.len()) as f64 / 1024.0,
                compressed.len() as f64 / gkr_compressed.len() as f64);
        }
    }

    let mem_snapshot = mem_tracker.stop();
    eprintln!("  {mem_snapshot}");

    group.finish();
}

criterion_group!(benches, sha256_8x_ecdsa_stepwise);
criterion_main!(benches);
