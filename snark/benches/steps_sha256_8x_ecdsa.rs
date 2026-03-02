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
        iprs::{IprsCode, PnttConfigF2_16R4B64},
    },
    pcs::structs::{ZipPlus, ZipPlusParams, ZipTypes},
};

use zinc_ecdsa_uair::EcdsaUairInt;
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
type FScalar = MontyField<{ INT_LIMBS * 3 }>;

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

struct EcdsaScalarZipTypes;

impl ZipTypes for EcdsaScalarZipTypes {
    const NUM_COLUMN_OPENINGS: usize = 147;
    type Eval = Int<{ INT_LIMBS * 4 }>;
    type Cw = Int<{ INT_LIMBS * 5 }>;
    type Fmod = Uint<{ INT_LIMBS * 3 }>;
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

// ─── Parameters ─────────────────────────────────────────────────────────────

const SHA256_8X_NUM_VARS: usize = 9;
const SHA256_BATCH_SIZE: usize = 30;       // 30 SHA-256 columns (27 bitpoly + 3 int)
const ECDSA_NUM_VARS: usize = 9;
const ECDSA_BATCH_SIZE: usize = zinc_ecdsa_uair::NUM_COLS;
const SHA256_LOOKUP_COL_COUNT: usize = 10;

fn sha256_lookup_specs() -> Vec<LookupColumnSpec> {
    (0..SHA256_LOOKUP_COL_COUNT)
        .map(|i| LookupColumnSpec {
            column_index: i,
            table_type: LookupTableType::BitPoly { width: 32 },
        })
        .collect()
}

fn sha256_affine_lookup_specs() -> Vec<AffineLookupSpec> {
    use zinc_sha256_uair::{
        COL_E_HAT, COL_E_TM1, COL_CH_EF_HAT, COL_E_TM2, COL_CH_NEG_EG_HAT,
        COL_A_HAT, COL_A_TM1, COL_A_TM2, COL_MAJ_HAT,
    };
    let bp32 = LookupTableType::BitPoly { width: 32 };
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

fn generate_zero_scalar_trace(
    num_vars: usize,
    num_cols: usize,
) -> Vec<DenseMultilinearExtension<Int<{ INT_LIMBS * 4 }>>> {
    let zero = Int::<{ INT_LIMBS * 4 }>::default();
    let rows = 1usize << num_vars;
    (0..num_cols)
        .map(|_| DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            vec![zero; rows],
            zero,
        ))
        .collect()
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
///   6.  SHA/PIOP/ProjectIC    — project_trace_coeffs + project_scalars
///   7.  ECDSA/PIOP/FieldSetup — transcript + random field config
///   8.  ECDSA/PIOP/ProjectIC  — project ECDSA trace to DynamicPolynomialF
///   9.  SHA/PIOP/IdealCheck
///  10.  ECDSA/PIOP/IdealCheck
///  11.  SHA/PIOP/ProjectCPR   — project_scalars_to_field + project_trace_to_field
///  12.  ECDSA/PIOP/ProjectCPR — same for ECDSA
///  13.  PIOP/CPR              — unified multi-degree sumcheck (SHA + ECDSA CPR only)
///  14.  SHA/PIOP/LookupExtract — extract lookup columns from field trace
///  15.  SHA/PIOP/Lookup        — lookup sumcheck (standalone, separate from CPR)
///  16.  PIOP/BatchedCPR+Lookup — SHA CPR + ECDSA CPR + SHA Lookup in one
///                                multi-degree sumcheck (A/B comparison with 13+15)
///  17.  SHA/PCS/Prove
///  18.  ECDSA/PCS/Prove
///  19.  E2E/Prover  (unified dual-circuit pipeline)
///  20.  E2E/Verifier (unified dual-circuit pipeline)
fn sha256_8x_ecdsa_stepwise(c: &mut Criterion) {
    use zinc_sha256_uair::CyclotomicIdeal;
    use zinc_ecdsa_uair::EcdsaIdealOverF;
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
        b.iter(|| black_box(generate_zero_scalar_trace(ECDSA_NUM_VARS, ECDSA_BATCH_SIZE)));
    });

    // Pre-generate traces for subsequent steps.
    let sha_trace = generate_sha256_trace(SHA256_8X_NUM_VARS);
    let ecdsa_trace = generate_zero_scalar_trace(ECDSA_NUM_VARS, ECDSA_BATCH_SIZE);

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

    // ── 3. SHA PCS Commit ───────────────────────────────────────────
    group.bench_function("SHA/PCS/Commit", |b| {
        b.iter(|| black_box(
            ZipPlus::<ShaZt, ShaLc>::commit(&sha_params, &sha_pcs_trace).expect("commit")
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
    group.bench_function("SHA/PIOP/ProjectIC", |b| {
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
    group.bench_function("ECDSA/PIOP/FieldSetup", |b| {
        b.iter(|| {
            let mut transcript = zinc_transcript::KeccakTranscript::new();
            let field_cfg = transcript.get_random_field_cfg::<
                F, <F as Field>::Inner, MillerRabin
            >();
            black_box(field_cfg);
        });
    });

    // ── 8. ECDSA PIOP / Project Trace for Ideal Check ───────────────
    //
    // Convert Int<4> trace to DynamicPolynomialF + project_scalars.
    group.bench_function("ECDSA/PIOP/ProjectIC", |b| {
        let mut tr_setup = zinc_transcript::KeccakTranscript::new();
        let fcfg = tr_setup.get_random_field_cfg::<F, <F as Field>::Inner, MillerRabin>();
        b.iter(|| {
            let projected_trace: Vec<DenseMultilinearExtension<DynamicPolynomialF<F>>> =
                ecdsa_trace.iter().map(|col_mle| {
                    col_mle.iter().map(|elem|
                        DynamicPolynomialF { coeffs: vec![F::from_with_cfg(elem.clone(), &fcfg)] }
                    ).collect()
                }).collect();
            let projected_scalars = project_scalars::<F, EcdsaUairInt>(|scalar| {
                DynamicPolynomialF { coeffs: vec![F::from_with_cfg(scalar.clone(), &fcfg)] }
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
                ).expect("IC");
                total += t.elapsed();
            }
            total
        });
    });

    // ── 10. ECDSA PIOP / Ideal Check ───────────────────────────────
    group.bench_function("ECDSA/PIOP/IdealCheck", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut transcript = zinc_transcript::KeccakTranscript::new();
                let field_cfg = transcript.get_random_field_cfg::<
                    F, <F as Field>::Inner, MillerRabin
                >();
                let projected_trace: Vec<DenseMultilinearExtension<DynamicPolynomialF<F>>> =
                    ecdsa_trace.iter().map(|col_mle| {
                        col_mle.iter().map(|elem|
                            DynamicPolynomialF { coeffs: vec![F::from_with_cfg(elem.clone(), &field_cfg)] }
                        ).collect()
                    }).collect();
                let projected_scalars = project_scalars::<F, EcdsaUairInt>(|scalar| {
                    DynamicPolynomialF { coeffs: vec![F::from_with_cfg(scalar.clone(), &field_cfg)] }
                });
                let t = Instant::now();
                let _ = zinc_piop::ideal_check::IdealCheckProtocol::<F>::prove_as_subprotocol::<EcdsaUairInt>(
                    &mut transcript, &projected_trace, &projected_scalars,
                    ecdsa_num_constraints, ECDSA_NUM_VARS, &field_cfg,
                ).expect("IC");
                total += t.elapsed();
            }
            total
        });
    });

    // ── 11. SHA PIOP / Project Trace for CPR ────────────────────────
    //
    // project_scalars_to_field + project_trace_to_field — the F[X]→F
    // specialisation that pipeline::prove includes in cpr_time.
    group.bench_function("SHA/PIOP/ProjectCPR", |b| {
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

    // ── 12. ECDSA PIOP / Project Trace for CPR ──────────────────────
    group.bench_function("ECDSA/PIOP/ProjectCPR", |b| {
        let mut tr_setup = zinc_transcript::KeccakTranscript::new();
        let fcfg = tr_setup.get_random_field_cfg::<F, <F as Field>::Inner, MillerRabin>();
        let proj_elem: F = tr_setup.get_field_challenge(&fcfg);
        let ec_proj_trace_bench: Vec<DenseMultilinearExtension<DynamicPolynomialF<F>>> =
            ecdsa_trace.iter().map(|col_mle| {
                col_mle.iter().map(|elem|
                    DynamicPolynomialF { coeffs: vec![F::from_with_cfg(elem.clone(), &fcfg)] }
                ).collect()
            }).collect();
        let proj_scalars_setup = project_scalars::<F, EcdsaUairInt>(|scalar| {
            DynamicPolynomialF { coeffs: vec![F::from_with_cfg(scalar.clone(), &fcfg)] }
        });
        b.iter(|| {
            let fproj_scalars =
                project_scalars_to_field(proj_scalars_setup.clone(), &proj_elem)
                    .expect("scalar projection");
            let field_trace = project_trace_to_field::<F, 1>(
                &[], &[], &ec_proj_trace_bench, &proj_elem,
            );
            black_box((&fproj_scalars, &field_trace));
        });
    });

    // ── 13. Unified PIOP / CPR (SHA + ECDSA) ────────────────────────
    //
    // Both CPR sumcheck groups are batched into a single multi-degree
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

    // Pre-compute the post-CPR transcript state (needed by lookup benchmarks).
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

    group.bench_function("PIOP/CPR", |b| {
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
    // After the CPR multi-degree sumcheck finalizes, a unified batched
    // sumcheck reduces ALL column evaluation claims to a single random
    // point ("point unification"):
    //   • eq-based claims for every SHA-256 up-eval column (shift=0)
    //   • eq-based claims for every ECDSA up-eval column   (shift=0)
    //   • genuine shift claims for ECDSA shifted columns   (shift=1)
    //
    // Pre-compute CPR state (eval point, up/down evals) and field
    // trace columns.  Only the shift_sumcheck_prove call is timed.
    let ec_sig = EcdsaUairInt::signature();

    // Run a fresh CPR to obtain the evaluation point, up_evals and down_evals.
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

    // Capture the post-CPR-finalize transcript state.
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
    // the pipeline this cost is included in cpr_time.
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

    // ── 16. PIOP / Batched CPR+Lookup (multi-degree sumcheck A/B) ──
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

        group.bench_function("PIOP/BatchedCPR+Lookup", |b| {
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
    let ecdsa_pcs_point: Vec<F> = cpr_eval_point[..ECDSA_NUM_VARS].to_vec();

    let (sha_hint, _) = ZipPlus::<ShaZt, ShaLc>::commit(&sha_params, &sha_pcs_trace).expect("commit");
    group.bench_function("SHA/PCS/Prove", |b| {
        b.iter(|| {
            black_box(
                ZipPlus::<ShaZt, ShaLc>::prove::<F, UNCHECKED>(&sha_params, &sha_pcs_trace, &sha_pcs_point, &sha_hint)
            )
        });
    });

    // ── 18. ECDSA PCS Prove ──────────────────────────────────────────
    let (ec_hint, _) = ZipPlus::<EcZt, EcLc>::commit(&ec_params, &ecdsa_pcs_trace).expect("commit");
    group.bench_function("ECDSA/PCS/Prove", |b| {
        b.iter(|| {
            black_box(
                ZipPlus::<EcZt, EcLc>::prove::<FScalar, UNCHECKED>(&ec_params, &ecdsa_pcs_trace, &ecdsa_pcs_point, &ec_hint)
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
    // using the public signature parameter s.  The current benchmark
    // uses a zero ECDSA trace, so we skip the modular-arithmetic
    // assertion and only measure the extraction + reconstruction cost.
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
                EcdsaIdealOverF, _,
            >(
                &sha_params, &ec_params,
                &dual_proof,
                SHA256_8X_NUM_VARS,
                |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
                |ideal: &IdealOrZero<ImpossibleIdeal>| match ideal {
                    IdealOrZero::Zero => EcdsaIdealOverF,
                    IdealOrZero::Ideal(_) => panic!("ECDSA has no non-zero ideal constraints"),
                },
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
        group.bench_function("V/IC", |b| {
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
                        &|ideal: &IdealOrZero<ImpossibleIdeal>| match ideal {
                            IdealOrZero::Zero => EcdsaIdealOverF,
                            IdealOrZero::Ideal(_) => panic!("no non-zero ideal"),
                        },
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
        group.bench_function("V/CPR+LookupPre", |b| {
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
                        &|ideal: &IdealOrZero<ImpossibleIdeal>| match ideal {
                            IdealOrZero::Zero => EcdsaIdealOverF,
                            IdealOrZero::Ideal(_) => panic!("no non-zero ideal"),
                        }, &fcfg,
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
                        &|ideal: &IdealOrZero<ImpossibleIdeal>| match ideal {
                            IdealOrZero::Zero => EcdsaIdealOverF, IdealOrZero::Ideal(_) => panic!("no non-zero ideal"),
                        }, &fcfg,
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

        // ── V5. Verifier / CPR Finalize ─────────────────────────────
        //
        // Finalize CPR for both circuits (includes public column MLE eval).
        group.bench_function("V/CPRFinalize", |b| {
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
                        &|ideal: &IdealOrZero<ImpossibleIdeal>| match ideal {
                            IdealOrZero::Zero => EcdsaIdealOverF, IdealOrZero::Ideal(_) => panic!("no non-zero ideal"),
                        }, &fcfg,
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
                        &|ideal: &IdealOrZero<ImpossibleIdeal>| match ideal {
                            IdealOrZero::Zero => EcdsaIdealOverF, IdealOrZero::Ideal(_) => panic!("no non-zero ideal"),
                        }, &fcfg,
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
                        ecdsa_num_constraints, ic_pt, &|ideal: &IdealOrZero<ImpossibleIdeal>| match ideal {
                            IdealOrZero::Zero => EcdsaIdealOverF, IdealOrZero::Ideal(_) => panic!("no non-zero ideal"),
                        }, &fcfg,
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
                        ecdsa_num_constraints, ic_pt, &|ideal: &IdealOrZero<ImpossibleIdeal>| match ideal {
                            IdealOrZero::Zero => EcdsaIdealOverF, IdealOrZero::Ideal(_) => panic!("no non-zero ideal"),
                        }, &fcfg,
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
                            <Uint<{INT_LIMBS * 3}> as ConstTranscribable>::read_transcription_bytes(&dual_proof.pcs2_evals_bytes[0]),
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

        // CPR up/down evaluations.
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

        eprintln!("\n=== 8xSHA256+ECDSA Dual-Circuit Proof Size ===");
        eprintln!("  PCS (SHA):      {:>6} B  ({:.1} KB)", pcs1_bytes, pcs1_bytes as f64 / 1024.0);
        eprintln!("  PCS (ECDSA):    {:>6} B  ({:.1} KB)", pcs2_bytes, pcs2_bytes as f64 / 1024.0);
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
