//! Per-step breakdown benchmarks for 8×SHA-256 + ECDSA proving stack.
//!
//! Run with:
//!   cargo bench --bench steps_sha256_8x_ecdsa -p zinc-snark --features "parallel simd asm"

#![allow(clippy::arithmetic_side_effects, clippy::unwrap_used)]

use std::hint::black_box;
use std::marker::PhantomData;
use std::time::{Duration, Instant};

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
    batched_pcs::structs::BatchedZipPlus,
    code::{
        LinearCode,
        iprs::{IprsCode, PnttConfigF2_16R4B64},
    },
    pcs::structs::{ZipPlusParams, ZipTypes},
};

use zinc_ecdsa_uair::EcdsaUairInt;
use zinc_sha256_uair::{Sha256Uair, witness::GenerateWitness};
use zinc_piop::projections::{
    project_trace_coeffs, project_trace_to_field,
    project_scalars, project_scalars_to_field,
};
use zinc_piop::lookup::{LookupColumnSpec, LookupTableType};
use zinc_piop::lookup::{
    BatchedDecompLogupProtocol, group_lookup_specs,
};
use zinc_piop::lookup::pipeline::build_lookup_instance_from_indices_pub;
use zinc_piop::sumcheck::multi_degree::MultiDegreeSumcheck;

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
const SHA256_BATCH_SIZE: usize = 17;
const ECDSA_NUM_VARS: usize = 9;
const ECDSA_BATCH_SIZE: usize = 9;
const SHA256_LOOKUP_COL_COUNT: usize = 10;

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

// ─── Benchmark ──────────────────────────────────────────────────────────────

/// Measures each main step of the E2E proving stack individually for
/// 8×SHA-256 compressions **plus** ECDSA verification.
///
/// Steps benchmarked (for each sub-protocol and combined):
///   1.  SHA/WitnessGen        — SHA-256 trace (17 cols × 512 rows)
///   2.  ECDSA/WitnessGen      — ECDSA trace (9 cols × 512 rows)
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
///  17.  SHA/PCS/Test
///  18.  ECDSA/PCS/Test
///  19.  SHA/PCS/Evaluate
///  20.  ECDSA/PCS/Evaluate
///  21.  E2E/Prover  (combined total)
///  22.  E2E/Verifier (combined total)
fn sha256_8x_ecdsa_stepwise(c: &mut Criterion) {
    use zinc_sha256_uair::CyclotomicIdeal;
    use zinc_ecdsa_uair::EcdsaIdealOverF;
    use zinc_uair::ideal::ImpossibleIdeal;
    use zinc_uair::ideal_collector::IdealOrZero;
    use zinc_piop::lookup::prove_batched_lookup_with_indices;

    let mut group = c.benchmark_group("8xSHA256+ECDSA Steps");
    group.sample_size(10);

    type ShaZt = Sha256ZipTypes<i64, 32>;
    type ShaLc = IprsBPoly32R4B64<1, UNCHECKED>;
    type EcZt = EcdsaScalarZipTypes;
    type EcLc = IprsInt4R4B64<1, UNCHECKED>;

    let sha_params = ZipPlusParams::<ShaZt, ShaLc>::new(SHA256_8X_NUM_VARS, 1, ShaLc::new(512));
    let ec_params  = ZipPlusParams::<EcZt, EcLc>::new(ECDSA_NUM_VARS, 1, EcLc::new(512));

    let sha_lookup_specs = sha256_lookup_specs();

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

    // ── 3. SHA PCS Commit ───────────────────────────────────────────
    group.bench_function("SHA/PCS/Commit", |b| {
        b.iter(|| black_box(
            BatchedZipPlus::<ShaZt, ShaLc>::commit(&sha_params, &sha_trace).expect("commit")
        ));
    });

    // ── 4. ECDSA PCS Commit ─────────────────────────────────────────
    group.bench_function("ECDSA/PCS/Commit", |b| {
        b.iter(|| black_box(
            BatchedZipPlus::<EcZt, EcLc>::commit(&ec_params, &ecdsa_trace).expect("commit")
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
    // project_trace_coeffs + project_scalars — transforms BinaryPoly
    // witness into DynamicPolynomialF over the PIOP field.
    group.bench_function("SHA/PIOP/ProjectIC", |b| {
        let mut tr_setup = zinc_transcript::KeccakTranscript::new();
        let fcfg = tr_setup.get_random_field_cfg::<F, <F as Field>::Inner, MillerRabin>();
        b.iter(|| {
            let projected_trace = project_trace_coeffs::<F, bool, bool, 32>(
                &sha_trace, &[], &[], &fcfg,
            );
            let projected_scalars = project_scalars::<F, Sha256Uair>(|scalar| {
                let one = F::one_with_cfg(&fcfg);
                let zero = F::zero_with_cfg(&fcfg);
                DynamicPolynomialF::new(
                    scalar.iter().map(|coeff| {
                        if coeff.into_inner() { one.clone() } else { zero.clone() }
                    }).collect::<Vec<_>>()
                )
            });
            black_box((&projected_trace, &projected_scalars));
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
                    &mut transcript, &projected_trace, &projected_scalars,
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

    // SHA IC (at shared point).
    let sha_proj_trace = project_trace_coeffs::<F, bool, bool, 32>(
        &sha_trace, &[], &[], &sha_fcfg,
    );
    let sha_proj_scalars = project_scalars::<F, Sha256Uair>(|scalar| {
        let one = F::one_with_cfg(&sha_fcfg);
        let zero = F::zero_with_cfg(&sha_fcfg);
        DynamicPolynomialF::new(
            scalar.iter().map(|c| if c.into_inner() { one.clone() } else { zero.clone() }).collect::<Vec<_>>()
        )
    });
    let _ = zinc_piop::ideal_check::IdealCheckProtocol::<F>::prove_at_point::<Sha256Uair>(
        &mut unified_tr, &sha_proj_trace, &sha_proj_scalars,
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

    // ── 17. SHA PCS Test ────────────────────────────────────────────
    let (sha_hint, _) = BatchedZipPlus::<ShaZt, ShaLc>::commit(&sha_params, &sha_trace).expect("commit");
    group.bench_function("SHA/PCS/Test", |b| {
        b.iter(|| black_box(
            BatchedZipPlus::<ShaZt, ShaLc>::test::<UNCHECKED>(&sha_params, &sha_trace, &sha_hint)
        ));
    });

    // ── 18. ECDSA PCS Test ──────────────────────────────────────────
    let (ec_hint, _) = BatchedZipPlus::<EcZt, EcLc>::commit(&ec_params, &ecdsa_trace).expect("commit");
    group.bench_function("ECDSA/PCS/Test", |b| {
        b.iter(|| black_box(
            BatchedZipPlus::<EcZt, EcLc>::test::<UNCHECKED>(&ec_params, &ecdsa_trace, &ec_hint)
        ));
    });

    // ── 19. SHA PCS Evaluate ────────────────────────────────────────
    group.bench_function("SHA/PCS/Evaluate", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let tx = BatchedZipPlus::<ShaZt, ShaLc>::test::<UNCHECKED>(
                    &sha_params, &sha_trace, &sha_hint,
                ).expect("test");
                let pt: Vec<i128> = vec![1i128; SHA256_8X_NUM_VARS];
                let t = Instant::now();
                let _ = BatchedZipPlus::<ShaZt, ShaLc>::evaluate::<F, UNCHECKED>(
                    &sha_params, &sha_trace, &pt, tx,
                ).expect("evaluate");
                total += t.elapsed();
            }
            total
        });
    });

    // ── 20. ECDSA PCS Evaluate ──────────────────────────────────────
    group.bench_function("ECDSA/PCS/Evaluate", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let tx = BatchedZipPlus::<EcZt, EcLc>::test::<UNCHECKED>(
                    &ec_params, &ecdsa_trace, &ec_hint,
                ).expect("test");
                let pt: Vec<i128> = vec![1i128; ECDSA_NUM_VARS];
                let t = Instant::now();
                let _ = BatchedZipPlus::<EcZt, EcLc>::evaluate::<FScalar, UNCHECKED>(
                    &ec_params, &ecdsa_trace, &pt, tx,
                ).expect("evaluate");
                total += t.elapsed();
            }
            total
        });
    });

    // ── 21. E2E Total Prover ────────────────────────────────────────
    //
    // Uses prove_classic_logup which batches CPR + lookup into a single
    // multi-degree sumcheck (matching the paper's proving strategy).
    group.bench_function("E2E/Prover", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let t = Instant::now();
                let _ = zinc_snark::pipeline::prove_classic_logup::<Sha256Uair, ShaZt, ShaLc, 32, UNCHECKED>(
                    &sha_params, &sha_trace, SHA256_8X_NUM_VARS, &sha_lookup_specs,
                );
                let _ = zinc_snark::pipeline::prove_generic::<
                    EcdsaUairInt, Int<{ INT_LIMBS * 4 }>, EcZt, EcLc, FScalar, UNCHECKED,
                >(&ec_params, &ecdsa_trace, ECDSA_NUM_VARS, &[]);
                total += t.elapsed();
            }
            total
        });
    });

    // ── 22. E2E Total Verifier ──────────────────────────────────────
    let sha_proof = zinc_snark::pipeline::prove_classic_logup::<Sha256Uair, ShaZt, ShaLc, 32, UNCHECKED>(
        &sha_params, &sha_trace, SHA256_8X_NUM_VARS, &sha_lookup_specs,
    );
    let ec_proof = zinc_snark::pipeline::prove_generic::<
        EcdsaUairInt, Int<{ INT_LIMBS * 4 }>, EcZt, EcLc, FScalar, UNCHECKED,
    >(&ec_params, &ecdsa_trace, ECDSA_NUM_VARS, &[]);

    group.bench_function("E2E/Verifier", |b| {
        b.iter(|| {
            let r1 = zinc_snark::pipeline::verify::<Sha256Uair, ShaZt, ShaLc, 32, UNCHECKED, _, _>(
                &sha_params, &sha_proof, SHA256_8X_NUM_VARS,
                |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
            );
            let _ = black_box(r1);
            let r2 = zinc_snark::pipeline::verify_generic::<
                EcdsaUairInt, Int<{ INT_LIMBS * 4 }>, EcZt, EcLc, FScalar, UNCHECKED, EcdsaIdealOverF, _,
            >(
                &ec_params, &ec_proof, ECDSA_NUM_VARS,
                |ideal: &IdealOrZero<ImpossibleIdeal>| match ideal {
                    IdealOrZero::Zero => EcdsaIdealOverF,
                    IdealOrZero::Ideal(_) => panic!("ECDSA has no non-zero ideal constraints"),
                },
            );
            let _ = black_box(r2);
        });
    });

    // ── Timing breakdown summary ────────────────────────────────────
    eprintln!("\n=== 8xSHA256+ECDSA Per-Step Timing ===");
    eprintln!("  SHA (batched CPR+Lookup): IC={:?}, CPR+Lookup={:?}, PCS(commit={:?}, test={:?}, eval={:?}), total={:?}",
        sha_proof.timing.ideal_check, sha_proof.timing.combined_poly_resolver,
        sha_proof.timing.pcs_commit,
        sha_proof.timing.pcs_test, sha_proof.timing.pcs_evaluate, sha_proof.timing.total,
    );
    eprintln!("  ECDSA: IC={:?}, CPR={:?}, Lookup={:?}, PCS(commit={:?}, test={:?}, eval={:?}), total={:?}",
        ec_proof.timing.ideal_check, ec_proof.timing.combined_poly_resolver,
        ec_proof.timing.lookup, ec_proof.timing.pcs_commit,
        ec_proof.timing.pcs_test, ec_proof.timing.pcs_evaluate, ec_proof.timing.total,
    );

    group.finish();
}

criterion_group!(benches, sha256_8x_ecdsa_stepwise);
criterion_main!(benches);
