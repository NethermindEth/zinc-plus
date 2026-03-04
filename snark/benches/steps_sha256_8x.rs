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
    project_trace_to_field,
    project_scalars, project_scalars_to_field,
};
use zinc_piop::lookup::{LookupColumnSpec, LookupTableType};

use zinc_piop::sumcheck::prover::{NatEvaluatedPolyWithoutConstant, ProverMsg};
use zinc_piop::ideal_check::IdealCheckProtocol;
use zinc_piop::combined_poly_resolver::CombinedPolyResolver;
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
            table_type: LookupTableType::BitPoly { width: 32, chunk_width: None },
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
/// Prover steps benchmarked:
///   1.  WitnessGen — generate the 30-column BinaryPoly trace (512 rows)
///   2.  PCS/Commit — Zip+ commit (Merkle tree construction)
///   3.  PIOP/FieldSetup — transcript init + random field config
///   4.  PIOP/Project Ideal Check — project_scalars for Ideal Check
///   5.  PIOP/IdealCheck — Ideal Check prover (MLE-first)
///   6.  PIOP/Project Main field sumcheck — project_scalars_to_field + project_trace_to_field
///   7.  PIOP/Main field sumcheck — Combined Poly Resolver prover
///   8.  PIOP/LookupExtract — extract lookup columns from field trace
///   9.  PIOP/Lookup — classic batched decomposed LogUp prover
///  10.  PIOP/GkrLookup — GKR batched decomposed LogUp prover
///  11.  PCS/Prove — Zip+ prove (test + evaluation combined)
///  12.  E2E/Prover — total (pipeline::prove)
///  13.  E2E/Verifier — total (pipeline::verify)
///
/// Verifier steps benchmarked:
///  V1. V/FieldSetup — transcript init + field config
///  V2. V/Ideal Check — Ideal Check verification
///  V3. V/Main field sumcheck Pre — projecting element + main field sumcheck pre-sumcheck
///  V4. V/Main field sumcheck Verify — main field sumcheck verification
///  V5. V/Main field sumcheck Finalize — main field sumcheck finalize (includes public column MLE eval)
///  V6. V/LookupVerify — lookup verification
///  V7. V/PCSVerify — PCS verification
///
/// Also reports:
///  - Proof size breakdown (raw + deflate-compressed)
///  - Peak memory usage
fn sha256_8x_stepwise(c: &mut Criterion) {
    use zinc_sha256_uair::CyclotomicIdeal;
    use zinc_uair::ideal_collector::IdealOrZero;
    use zinc_piop::lookup::prove_batched_lookup_with_indices;
    use zinc_piop::lookup::prove_gkr_batched_lookup_with_indices;

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
    let sha_excluded = sha_sig.pcs_excluded_columns();
    let sha_pcs_trace: Vec<_> = sha_trace.iter().enumerate()
        .filter(|(i, _)| !sha_excluded.contains(i))
        .map(|(_, c)| c.clone()).collect();

    let num_constraints = zinc_uair::constraint_counter::count_constraints::<Sha256Uair>();
    let max_degree = zinc_uair::degree_counter::count_max_degree::<Sha256Uair>();
    let num_vars = SHA256_8X_NUM_VARS;

    // ── 2. PCS Commit ───────────────────────────────────────────────
    group.bench_function("PCS/Commit", |b| {
        b.iter(|| {
            let r = ZipPlus::<ShaZt, ShaLc>::commit(&sha_params, &sha_pcs_trace);
            let _ = black_box(r);
        });
    });

    // ── 3. PIOP / Field Setup ────────────────────────────────────────
    group.bench_function("PIOP/FieldSetup", |b| {
        b.iter(|| {
            let mut transcript = zinc_transcript::KeccakTranscript::new();
            let field_cfg = transcript.get_random_field_cfg::<
                F, <F as Field>::Inner, MillerRabin
            >();
            black_box(field_cfg);
        });
    });

    // ── 4. PIOP / Project Trace for Ideal Check ─────────────────────
    assert_eq!(max_degree, 1, "SHA-256 UAIR should have max_degree == 1 (MLE-first path)");
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

    // ── 5. PIOP / Ideal Check (MLE-first for max_degree == 1) ────────
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
                let _ = zinc_piop::ideal_check::IdealCheckProtocol::<F>::prove_mle_first::<Sha256Uair, 32>(
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

    // ── 6. PIOP / Project Trace for Main field sumcheck ─────────────────────────────
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
        zinc_piop::ideal_check::IdealCheckProtocol::<F>::prove_mle_first::<Sha256Uair, 32>(
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

    // ── 7. PIOP / Combined Poly Resolver ────────────────────────────
    group.bench_function("PIOP/Main field sumcheck", |b| {
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
                ).expect("Main field sumcheck prover failed");
                total += t.elapsed();
            }
            total
        });
    });

    // ── 8. PIOP / Lookup Extract ────────────────────────────────────
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

    // ── 9. PIOP / Lookup (classic) ──────────────────────────────────
    // ── 10. PIOP / GKR Lookup ───────────────────────────────────────
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

        group.bench_function("PIOP/GkrLookup", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let mut tr = transcript_lk.clone();
                    let t = Instant::now();
                    let _ = prove_gkr_batched_lookup_with_indices(
                        &mut tr, &columns, &raw_indices, &remapped_specs,
                        &projecting_elem_cpr, &field_cfg_cpr,
                    ).expect("gkr lookup prover failed");
                    total += t.elapsed();
                }
                total
            });
        });
    }

    // ── 11. PCS Prove ───────────────────────────────────────────────
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
            ).expect("Main field sumcheck prover failed");
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

    // ── 12. E2E Total Prover ────────────────────────────────────────
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

    // ── 13. E2E Total Verifier ──────────────────────────────────────
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

    // ══════════════════════════════════════════════════════════════════
    // ── Verifier Step-by-Step Breakdown ──────────────────────────────
    // ══════════════════════════════════════════════════════════════════
    //
    // Replays the verify logic piece by piece so that each verifier
    // sub-cost can be measured independently via Criterion.

    {
        use zinc_snark::pipeline::{
            FIELD_LIMBS, PiopField,
            field_from_bytes, reconstruct_up_evals, TrivialIdeal,
            LookupProofData,
        };
        use zinc_piop::lookup::{verify_batched_lookup, verify_gkr_batched_lookup};
        use zinc_piop::sumcheck::{MLSumcheck, SumcheckProof};
        use zinc_piop::shift_sumcheck::{
            ShiftClaim, ShiftSumcheckProof, ShiftRoundPoly,
            shift_sumcheck_verify, shift_sumcheck_verify_pre, shift_sumcheck_verify_finalize,
        };
        use zinc_poly::mle::MultilinearExtensionWithConfig;

        let field_elem_size =
            <Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;

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

        // Helper: deserialize main field sumcheck proof.
        let deser_cpr_sumcheck = |fcfg: &<PiopField as PrimeField>::Config| -> SumcheckProof<PiopField> {
            let messages: Vec<ProverMsg<PiopField>> = sha_proof
                .cpr_sumcheck_messages.iter().map(|bytes| {
                    let n = bytes.len() / field_elem_size;
                    ProverMsg(NatEvaluatedPolyWithoutConstant::new(
                        (0..n).map(|i| field_from_bytes(
                            &bytes[i * field_elem_size..(i + 1) * field_elem_size], fcfg,
                        )).collect()
                    ))
                }).collect();
            let claimed_sum = field_from_bytes(&sha_proof.cpr_sumcheck_claimed_sum, fcfg);
            SumcheckProof { messages, claimed_sum }
        };

        // Helper: replay transcript through IC verify.
        let replay_through_ic = |tr: &mut zinc_transcript::KeccakTranscript, fcfg: &<PiopField as PrimeField>::Config|
            -> zinc_piop::ideal_check::VerifierSubClaim<PiopField> {
            IdealCheckProtocol::<PiopField>::verify_as_subprotocol::<Sha256Uair, _, _>(
                tr,
                zinc_piop::ideal_check::Proof { combined_mle_values: deser_ic(&sha_proof.ic_proof_values, fcfg) },
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

        // Helper: replay shift sumcheck verify on the transcript.
        // Must be called after main field sumcheck finalize to keep transcript in sync.
        let replay_shift_sumcheck = |tr: &mut zinc_transcript::KeccakTranscript,
                                      fcfg: &<PiopField as PrimeField>::Config,
                                      projecting_element: &PiopField,
                                      cpr_eval_point: &[PiopField],
                                      cpr_down_evals: &[PiopField]| {
            let sig_ss = Sha256Uair::signature();
            if sig_ss.shifts.is_empty() { return; }
            let ss_data = sha_proof.shift_sumcheck.as_ref()
                .expect("SHA256 has shifts but proof has no shift_sumcheck");

            let claims: Vec<ShiftClaim<PiopField>> = sig_ss.shifts.iter()
                .enumerate()
                .map(|(i, spec)| ShiftClaim {
                    source_col: i,
                    shift_amount: spec.shift_amount,
                    eval_point: cpr_eval_point.to_vec(),
                    claimed_eval: cpr_down_evals[i].clone(),
                })
                .collect();

            let rounds: Vec<ShiftRoundPoly<PiopField>> = ss_data.rounds.iter().map(|bytes| {
                ShiftRoundPoly {
                    evals: [
                        field_from_bytes(&bytes[0..field_elem_size], fcfg),
                        field_from_bytes(&bytes[field_elem_size..2 * field_elem_size], fcfg),
                        field_from_bytes(&bytes[2 * field_elem_size..3 * field_elem_size], fcfg),
                    ],
                }
            }).collect();
            let ss_proof = ShiftSumcheckProof { rounds };

            let private_v_finals: Vec<PiopField> = ss_data.v_finals.iter()
                .map(|b| field_from_bytes(b, fcfg)).collect();

            let has_public_shifts = sig_ss.shifts.iter()
                .any(|spec| sig_ss.is_public_column(spec.source_col));

            if has_public_shifts {
                let ss_pre = shift_sumcheck_verify_pre(
                    tr, &ss_proof, &claims, num_vars, fcfg,
                ).expect("shift sumcheck pre-verify");

                let bin_proj = BinaryPoly::<32>::prepare_projection(projecting_element);
                let challenge_point_le: Vec<PiopField> =
                    ss_pre.challenge_point.iter().rev().cloned().collect();

                let public_shift_specs: Vec<&zinc_uair::ShiftSpec> = sig_ss.shifts.iter()
                    .filter(|spec| sig_ss.is_public_column(spec.source_col))
                    .collect();
                let public_v_finals: Vec<PiopField> = public_shift_specs.iter()
                    .map(|spec| {
                        let pcd_idx = sig_ss.public_columns.iter()
                            .position(|&c| c == spec.source_col)
                            .expect("public shift source_col not in public_columns");
                        let col = &sha_public_cols[pcd_idx];
                        let mle: DenseMultilinearExtension<<PiopField as Field>::Inner> =
                            col.iter().map(|bp| bin_proj(bp).inner().clone()).collect();
                        mle.evaluate_with_config(&challenge_point_le, fcfg).unwrap()
                    })
                    .collect();

                // Reconstruct full v_finals: interleave private and public.
                let mut full_v_finals = Vec::with_capacity(sig_ss.shifts.len());
                let mut priv_idx = 0usize;
                let mut pub_idx = 0usize;
                for i in 0..sig_ss.shifts.len() {
                    if sig_ss.is_public_shift(i) {
                        full_v_finals.push(public_v_finals[pub_idx].clone());
                        pub_idx += 1;
                    } else {
                        full_v_finals.push(private_v_finals[priv_idx].clone());
                        priv_idx += 1;
                    }
                }

                shift_sumcheck_verify_finalize(
                    tr, &ss_pre, &claims, &full_v_finals, fcfg,
                ).expect("shift sumcheck finalize");
            } else {
                shift_sumcheck_verify(
                    tr, &ss_proof, &claims, &private_v_finals, num_vars, fcfg,
                ).expect("shift sumcheck verify");
            }
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
                    let projecting_element: PiopField = tr.get_field_challenge(&fcfg);
                    let fps = verifier_proj_scalars(&fcfg, &projecting_element);
                    let cpr_sc = deser_cpr_sumcheck(&fcfg);
                    let _ = CombinedPolyResolver::<PiopField>::build_verifier_pre_sumcheck::<Sha256Uair>(
                        &mut tr, &cpr_sc.claimed_sum, num_constraints,
                        &projecting_element, &fps, &ic_subclaim, &fcfg,
                    ).expect("Main field sumcheck pre-sumcheck");
                    total += t.elapsed();
                }
                total
            });
        });

        // ── V4. Verifier / Main field sumcheck Verify ─────────────────────────────
        group.bench_function("V/Main field sumcheck Verify", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let mut tr = zinc_transcript::KeccakTranscript::new();
                    let fcfg = tr.get_random_field_cfg::<
                        PiopField, <PiopField as Field>::Inner, MillerRabin
                    >();
                    let ic_subclaim = replay_through_ic(&mut tr, &fcfg);
                    let projecting_element: PiopField = tr.get_field_challenge(&fcfg);
                    let fps = verifier_proj_scalars(&fcfg, &projecting_element);
                    let cpr_sc = deser_cpr_sumcheck(&fcfg);
                    let _ = CombinedPolyResolver::<PiopField>::build_verifier_pre_sumcheck::<Sha256Uair>(
                        &mut tr, &cpr_sc.claimed_sum, num_constraints,
                        &projecting_element, &fps, &ic_subclaim, &fcfg,
                    ).unwrap();

                    let t = Instant::now();
                    let _ = MLSumcheck::<PiopField>::verify_as_subprotocol(
                        &mut tr, num_vars, max_degree + 2, &cpr_sc, &fcfg,
                    ).expect("Main field sumcheck verify");
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
                    let projecting_element: PiopField = tr.get_field_challenge(&fcfg);
                    let fps = verifier_proj_scalars(&fcfg, &projecting_element);
                    let cpr_sc = deser_cpr_sumcheck(&fcfg);
                    let cpr_pre = CombinedPolyResolver::<PiopField>::build_verifier_pre_sumcheck::<Sha256Uair>(
                        &mut tr, &cpr_sc.claimed_sum, num_constraints,
                        &projecting_element, &fps, &ic_subclaim, &fcfg,
                    ).unwrap();
                    let subclaim = MLSumcheck::<PiopField>::verify_as_subprotocol(
                        &mut tr, num_vars, max_degree + 2, &cpr_sc, &fcfg,
                    ).unwrap();

                    let t = Instant::now();
                    let private_up_evals: Vec<PiopField> = sha_proof.cpr_up_evals.iter()
                        .map(|b| field_from_bytes(b, &fcfg)).collect();
                    let cpr_down_evals: Vec<PiopField> = sha_proof.cpr_down_evals.iter()
                        .map(|b| field_from_bytes(b, &fcfg)).collect();
                    let sig_v = Sha256Uair::signature();
                    let full_up_evals = if sig_v.public_columns.is_empty() {
                        private_up_evals
                    } else {
                        let bin_proj = BinaryPoly::<32>::prepare_projection(&projecting_element);
                        let public_evals: Vec<PiopField> = sha_public_cols.iter().map(|col| {
                            let mle: DenseMultilinearExtension<<PiopField as Field>::Inner> =
                                col.iter().map(|bp| bin_proj(bp).inner().clone()).collect();
                            mle.evaluate_with_config(&subclaim.point, &fcfg).unwrap()
                        }).collect();
                        reconstruct_up_evals(
                            &private_up_evals, &public_evals,
                            &sig_v.public_columns, sig_v.total_cols(),
                        )
                    };
                    let _ = CombinedPolyResolver::<PiopField>::finalize_verifier::<Sha256Uair>(
                        &mut tr, subclaim.point, subclaim.expected_evaluation,
                        &cpr_pre, full_up_evals, cpr_down_evals, num_vars, &fps, &fcfg,
                    ).expect("Main field sumcheck finalize");
                    total += t.elapsed();
                }
                total
            });
        });

        // ── V5b. Verifier / Shift Sumcheck ────────────────────────
        group.bench_function("V/ShiftSumcheck", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let mut tr = zinc_transcript::KeccakTranscript::new();
                    let fcfg = tr.get_random_field_cfg::<
                        PiopField, <PiopField as Field>::Inner, MillerRabin
                    >();
                    let ic_subclaim = replay_through_ic(&mut tr, &fcfg);
                    let projecting_element: PiopField = tr.get_field_challenge(&fcfg);
                    let fps = verifier_proj_scalars(&fcfg, &projecting_element);
                    let cpr_sc = deser_cpr_sumcheck(&fcfg);
                    let cpr_pre = CombinedPolyResolver::<PiopField>::build_verifier_pre_sumcheck::<Sha256Uair>(
                        &mut tr, &cpr_sc.claimed_sum, num_constraints,
                        &projecting_element, &fps, &ic_subclaim, &fcfg,
                    ).unwrap();
                    let subclaim = MLSumcheck::<PiopField>::verify_as_subprotocol(
                        &mut tr, num_vars, max_degree + 2, &cpr_sc, &fcfg,
                    ).unwrap();
                    let private_up_evals: Vec<PiopField> = sha_proof.cpr_up_evals.iter()
                        .map(|b| field_from_bytes(b, &fcfg)).collect();
                    let cpr_down_evals: Vec<PiopField> = sha_proof.cpr_down_evals.iter()
                        .map(|b| field_from_bytes(b, &fcfg)).collect();
                    let sig_v = Sha256Uair::signature();
                    let full_up_evals = if sig_v.public_columns.is_empty() {
                        private_up_evals
                    } else {
                        let bin_proj = BinaryPoly::<32>::prepare_projection(&projecting_element);
                        let public_evals: Vec<PiopField> = sha_public_cols.iter().map(|col| {
                            let mle: DenseMultilinearExtension<<PiopField as Field>::Inner> =
                                col.iter().map(|bp| bin_proj(bp).inner().clone()).collect();
                            mle.evaluate_with_config(&subclaim.point, &fcfg).unwrap()
                        }).collect();
                        reconstruct_up_evals(&private_up_evals, &public_evals,
                            &sig_v.public_columns, sig_v.total_cols())
                    };
                    let cpr_subclaim = CombinedPolyResolver::<PiopField>::finalize_verifier::<Sha256Uair>(
                        &mut tr, subclaim.point, subclaim.expected_evaluation,
                        &cpr_pre, full_up_evals, cpr_down_evals.clone(), num_vars, &fps, &fcfg,
                    ).unwrap();

                    let t = Instant::now();
                    replay_shift_sumcheck(
                        &mut tr, &fcfg, &projecting_element,
                        &cpr_subclaim.evaluation_point, &cpr_down_evals,
                    );
                    total += t.elapsed();
                }
                total
            });
        });

        // ── V6. Verifier / Lookup Verify ────────────────────────────
        group.bench_function("V/LookupVerify", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    // Full replay through main field sumcheck finalize + shift sumcheck.
                    let mut tr = zinc_transcript::KeccakTranscript::new();
                    let fcfg = tr.get_random_field_cfg::<
                        PiopField, <PiopField as Field>::Inner, MillerRabin
                    >();
                    let ic_subclaim = replay_through_ic(&mut tr, &fcfg);
                    let projecting_element: PiopField = tr.get_field_challenge(&fcfg);
                    let fps = verifier_proj_scalars(&fcfg, &projecting_element);
                    let cpr_sc = deser_cpr_sumcheck(&fcfg);
                    let cpr_pre = CombinedPolyResolver::<PiopField>::build_verifier_pre_sumcheck::<Sha256Uair>(
                        &mut tr, &cpr_sc.claimed_sum, num_constraints,
                        &projecting_element, &fps, &ic_subclaim, &fcfg,
                    ).unwrap();
                    let subclaim = MLSumcheck::<PiopField>::verify_as_subprotocol(
                        &mut tr, num_vars, max_degree + 2, &cpr_sc, &fcfg,
                    ).unwrap();
                    let private_up_evals: Vec<PiopField> = sha_proof.cpr_up_evals.iter()
                        .map(|b| field_from_bytes(b, &fcfg)).collect();
                    let cpr_down_evals: Vec<PiopField> = sha_proof.cpr_down_evals.iter()
                        .map(|b| field_from_bytes(b, &fcfg)).collect();
                    let sig_v = Sha256Uair::signature();
                    let full_up_evals = if sig_v.public_columns.is_empty() {
                        private_up_evals
                    } else {
                        let bin_proj = BinaryPoly::<32>::prepare_projection(&projecting_element);
                        let public_evals: Vec<PiopField> = sha_public_cols.iter().map(|col| {
                            let mle: DenseMultilinearExtension<<PiopField as Field>::Inner> =
                                col.iter().map(|bp| bin_proj(bp).inner().clone()).collect();
                            mle.evaluate_with_config(&subclaim.point, &fcfg).unwrap()
                        }).collect();
                        reconstruct_up_evals(&private_up_evals, &public_evals,
                            &sig_v.public_columns, sig_v.total_cols())
                    };
                    let cpr_subclaim = CombinedPolyResolver::<PiopField>::finalize_verifier::<Sha256Uair>(
                        &mut tr, subclaim.point, subclaim.expected_evaluation,
                        &cpr_pre, full_up_evals, cpr_down_evals.clone(), num_vars, &fps, &fcfg,
                    ).unwrap();

                    // Replay shift sumcheck to keep transcript in sync.
                    replay_shift_sumcheck(
                        &mut tr, &fcfg, &projecting_element,
                        &cpr_subclaim.evaluation_point, &cpr_down_evals,
                    );

                    // ── Timed section ──
                    let t = Instant::now();
                    if let Some(ref lookup_data) = sha_proof.lookup_proof {
                        match lookup_data {
                            LookupProofData::Gkr(proof) => {
                                let _ = verify_gkr_batched_lookup(
                                    &mut tr, proof, &projecting_element, &fcfg,
                                ).expect("GKR lookup verify");
                            }
                            LookupProofData::Classic(proof) => {
                                let _ = verify_batched_lookup(
                                    &mut tr, proof, &projecting_element, &fcfg,
                                ).expect("classic lookup verify");
                            }
                            LookupProofData::BatchedClassic(_) => {
                                panic!("single-circuit prove should not produce BatchedClassic");
                            }
                            LookupProofData::HybridGkr(_) => {
                                panic!("single-circuit prove should not produce HybridGkr");
                            }
                        }
                    }
                    total += t.elapsed();
                }
                total
            });
        });

        // ── V7. Verifier / PCS Verify ───────────────────────────────
        group.bench_function("V/PCSVerify", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    // Derive evaluation point from proof.
                    let mut pcs_tr_setup = zinc_transcript::KeccakTranscript::new();
                    let pcs_fcfg = pcs_tr_setup.get_random_field_cfg::<
                        PiopField, <PiopField as Field>::Inner, MillerRabin
                    >();
                    let cpr_eval_pt: Vec<PiopField> = sha_proof.evaluation_point_bytes.iter()
                        .map(|b| field_from_bytes(b, &pcs_fcfg)).collect();

                    // ── Timed section ──
                    let t = Instant::now();
                    let mut pcs_transcript = zip_plus::pcs_transcript::PcsTranscript {
                        fs_transcript: zinc_transcript::KeccakTranscript::default(),
                        stream: std::io::Cursor::new(sha_proof.pcs_proof_bytes.clone()),
                    };
                    let pcs_field_cfg = pcs_transcript.fs_transcript
                        .get_random_field_cfg::<PiopField, <ShaZt as ZipTypes>::Fmod, <ShaZt as ZipTypes>::PrimeTest>();
                    let eval_f: PiopField = PiopField::new_unchecked_with_cfg(
                        <Uint<{INT_LIMBS * 3}> as ConstTranscribable>::read_transcription_bytes(&sha_proof.pcs_evals_bytes[0]),
                        &pcs_field_cfg,
                    );
                    let point_f: Vec<PiopField> = cpr_eval_pt[..num_vars].to_vec();
                    ZipPlus::<ShaZt, ShaLc>::verify_with_field_cfg::<PiopField, UNCHECKED>(
                        &sha_params, &sha_proof.commitment, &point_f, &eval_f,
                        pcs_transcript, &pcs_field_cfg,
                    ).expect("PCS verify");
                    total += t.elapsed();
                }
                total
            });
        });
    }

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

    // ── Proof size breakdown ────────────────────────────────────────
    {
        use zinc_snark::pipeline::{FIELD_LIMBS, LookupProofData};
        let fe_bytes = <Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;

        // PCS proof bytes.
        let pcs_bytes = sha_proof.pcs_proof_bytes.len();

        // IC proof values.
        let ic_bytes: usize = sha_proof.ic_proof_values.iter().map(|v| v.len()).sum();

        // Main field sumcheck messages + claimed sum.
        let cpr_msg_bytes: usize = sha_proof.cpr_sumcheck_messages.iter().map(|v| v.len()).sum();
        let cpr_sum_bytes = sha_proof.cpr_sumcheck_claimed_sum.len();
        let cpr_sc_total = cpr_msg_bytes + cpr_sum_bytes;

        // Main field sumcheck up/down evaluations.
        let cpr_up: usize = sha_proof.cpr_up_evals.iter().map(|v| v.len()).sum();
        let cpr_dn: usize = sha_proof.cpr_down_evals.iter().map(|v| v.len()).sum();
        let cpr_eval_total = cpr_up + cpr_dn;

        // Lookup data.
        let lookup_bytes: usize = match &sha_proof.lookup_proof {
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
            Some(LookupProofData::Gkr(_)) | Some(LookupProofData::BatchedClassic(_)) | Some(LookupProofData::HybridGkr(_)) => 0,
            None => 0,
        };

        // Shift sumcheck data.
        let shift_sc_bytes: usize = sha_proof.shift_sumcheck.as_ref().map_or(0, |sc| {
            let rounds: usize = sc.rounds.iter().map(|v| v.len()).sum();
            let finals: usize = sc.v_finals.iter().map(|v| v.len()).sum();
            rounds + finals
        });

        // Evaluation point + PCS evals.
        let eval_pt_bytes: usize = sha_proof.evaluation_point_bytes.iter().map(|v| v.len()).sum();
        let pcs_eval_bytes: usize = sha_proof.pcs_evals_bytes.iter().map(|v| v.len()).sum();

        let piop_total = ic_bytes + cpr_sc_total + cpr_eval_total
            + lookup_bytes + shift_sc_bytes + eval_pt_bytes + pcs_eval_bytes;
        let total_raw = pcs_bytes + piop_total;

        // Build serialized byte buffer and compress with deflate.
        let mut all_bytes = Vec::with_capacity(total_raw);
        all_bytes.extend(&sha_proof.pcs_proof_bytes);
        for v in &sha_proof.ic_proof_values { all_bytes.extend(v); }
        for v in &sha_proof.cpr_sumcheck_messages { all_bytes.extend(v); }
        all_bytes.extend(&sha_proof.cpr_sumcheck_claimed_sum);
        for v in &sha_proof.cpr_up_evals { all_bytes.extend(v); }
        for v in &sha_proof.cpr_down_evals { all_bytes.extend(v); }
        if let Some(LookupProofData::Classic(ref proof)) = sha_proof.lookup_proof {
            fn write_fe(buf: &mut Vec<u8>, f: &zinc_snark::pipeline::PiopField) {
                use zinc_snark::pipeline::FIELD_LIMBS;
                let sz = <Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;
                let start = buf.len();
                buf.resize(start + sz, 0);
                f.inner().write_transcription_bytes(&mut buf[start..]);
            }
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
        if let Some(ref sc) = sha_proof.shift_sumcheck {
            for v in &sc.rounds { all_bytes.extend(v); }
            for v in &sc.v_finals { all_bytes.extend(v); }
        }
        for v in &sha_proof.evaluation_point_bytes { all_bytes.extend(v); }
        for v in &sha_proof.pcs_evals_bytes { all_bytes.extend(v); }

        let compressed = {
            use std::io::Write;
            let mut encoder = flate2::write::DeflateEncoder::new(
                Vec::new(), flate2::Compression::default(),
            );
            encoder.write_all(&all_bytes).unwrap();
            encoder.finish().unwrap()
        };

        eprintln!("\n=== 8xSHA256 (no ECDSA) Proof Size ===");
        eprintln!("  PCS:            {:>6} B  ({:.1} KB)", pcs_bytes, pcs_bytes as f64 / 1024.0);
        eprintln!("  IC:             {:>6} B", ic_bytes);
        eprintln!("  CPR sumcheck:   {:>6} B  (msgs={}, sum={})", cpr_sc_total, cpr_msg_bytes, cpr_sum_bytes);
        eprintln!("  CPR evals:      {:>6} B  (up={}, down={})", cpr_eval_total, cpr_up, cpr_dn);
        eprintln!("  Lookup:         {:>6} B", lookup_bytes);
        eprintln!("  Shift SC:       {:>6} B", shift_sc_bytes);
        eprintln!("  Eval point:     {:>6} B", eval_pt_bytes);
        eprintln!("  PCS evals:      {:>6} B", pcs_eval_bytes);
        eprintln!("  ─────────────────────────");
        eprintln!("  PIOP total:     {:>6} B  ({:.1} KB)", piop_total, piop_total as f64 / 1024.0);
        eprintln!("  Total raw:      {:>6} B  ({:.1} KB)", total_raw, total_raw as f64 / 1024.0);
        eprintln!("  Compressed:     {:>6} B  ({:.1} KB, {:.1}x ratio)",
            compressed.len(), compressed.len() as f64 / 1024.0,
            all_bytes.len() as f64 / compressed.len() as f64);

        // ── GKR lookup proof size comparison ────────────────────────
        {
            use zinc_piop::lookup::prove_gkr_batched_lookup_with_indices;

            let mut needed: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
            for spec in &sha_lookup_specs {
                let next = needed.len();
                needed.entry(spec.column_index).or_insert(next);
            }
            let mut lk_columns: Vec<Vec<F>> = Vec::with_capacity(needed.len());
            let mut lk_raw_indices: Vec<Vec<usize>> = Vec::with_capacity(needed.len());
            for &orig_idx in needed.keys() {
                let col_f: Vec<F> = field_trace_cpr[orig_idx].iter()
                    .map(|inner| F::new_unchecked_with_cfg(inner.clone(), &field_cfg_cpr)).collect();
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

            let mut gkr_tr = transcript_for_cpr.clone();
            let _ = zinc_piop::combined_poly_resolver::CombinedPolyResolver::<F>::prove_as_subprotocol::<Sha256Uair>(
                &mut gkr_tr,
                field_trace_cpr.clone(),
                &ic_state_cpr.evaluation_point,
                &field_projected_scalars_cpr,
                num_constraints,
                SHA256_8X_NUM_VARS,
                max_degree,
                &field_cfg_cpr,
            ).expect("Main field sumcheck prover failed");

            let (gkr_proof, _) = prove_gkr_batched_lookup_with_indices(
                &mut gkr_tr,
                &lk_columns, &lk_raw_indices, &lk_remapped,
                &projecting_elem_cpr, &field_cfg_cpr,
            ).expect("GKR lookup proof failed");

            fn write_fe_gkr(buf: &mut Vec<u8>, f: &zinc_snark::pipeline::PiopField) {
                use zinc_snark::pipeline::FIELD_LIMBS;
                let sz = <Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;
                let start = buf.len();
                buf.resize(start + sz, 0);
                f.inner().write_transcription_bytes(&mut buf[start..]);
            }
            let mut gkr_bytes = Vec::new();
            for gp in &gkr_proof.group_proofs {
                for v in &gp.aggregated_multiplicities {
                    for f in v { write_fe_gkr(&mut gkr_bytes, f); }
                }
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

            eprintln!("\n=== Classic vs GKR Lookup Proof Size ===");
            eprintln!("  Classic lookup: {:>6} B  ({:.1} KB)", lookup_bytes, lookup_bytes as f64 / 1024.0);
            eprintln!("  GKR lookup:     {:>6} B  ({:.1} KB)", gkr_lookup_bytes, gkr_lookup_bytes as f64 / 1024.0);
            if lookup_bytes > gkr_lookup_bytes {
                eprintln!("  Savings:        {:>6} B  ({:.1}x smaller)",
                    lookup_bytes - gkr_lookup_bytes,
                    lookup_bytes as f64 / gkr_lookup_bytes as f64);
            } else {
                eprintln!("  GKR is larger by {} B", gkr_lookup_bytes - lookup_bytes);
            }
        }
    }

    let mem_snapshot = mem_tracker.stop();
    eprintln!("\n=== Peak Memory ===");
    eprintln!("  {mem_snapshot}");

    group.finish();
}

criterion_group!(benches, sha256_8x_stepwise);
criterion_main!(benches);
