//! Criterion benchmark for the SHA-256 `F_2[X]` UAIR
//! ([`zinc_test_uair::Sha256F2Uair`]) routed through
//! [`ZincPlusPiopF2`].
//!
//! Mirrors the integer-pipeline `bench_real_sha256_e2e` in
//! `benches/e2e.rs` but commits all witness columns as bit-polynomials
//! in `F_2[X]` (no overflow witnesses, no booleanity sumcheck).
//!
//! Three bench groups are reported:
//!
//! 1. `Zinc+ F_2 SHA-256` — end-to-end (WitnessGen / Prove / Verify).
//! 2. `Zinc+ F_2 SHA-256 Steps` — top-level prover and verifier step
//!    breakdown (Commit / UAIR / Open).
//! 3. `Zinc+ F_2 SHA-256 Micro` — fine-grained timings *inside* the
//!    UAIR and Open phases. Identifies the hot paths inside
//!    `prove_f2_uair_with_groups`, `verify_f2_uair`, and
//!    `verify_f2_open` by replicating their inner sub-steps and
//!    timing each.
//!
//! Trace shape: 7 chained compressions, `num_vars = 9` → 2^9 = 512
//! rows (480 active + 32 slack).
//!
//! **Required features**: `parallel`, `simd`, `unchecked`. The bench
//! is gated by `required-features` in `Cargo.toml`; running without
//! those features is rejected by cargo.

#![allow(clippy::arithmetic_side_effects)]

use std::hint::black_box;
use std::marker::PhantomData;

use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main,
    measurement::WallTime,
};
use crypto_bigint::U64;
use crypto_primitives::{Field, FromWithConfig, crypto_bigint_int::Int, crypto_bigint_uint::Uint};
use rand::rng;
use zinc_poly::{
    mle::DenseMultilinearExtension,
    univariate::{
        binary::{BinaryPoly, BinaryPolyInnerProduct},
        binary_gf192::BinaryFieldGF192,
        dense::{DensePolyInnerProduct, DensePolynomial},
    },
};
use zinc_primality::MillerRabin;
use zinc_protocol::f2_prove::{
    F2VirtualBpSpec, F2ZincTypes, ZincPlusPiopF2, eq_dot_column_groups,
    extract_column_evals_eq_dot_col,
};
use zinc_test_uair::{
    GenerateRandomTrace, Sha256F2Ideal, Sha256F2Uair, sha256_f2_project_ideal,
    sha256_f2_project_scalar,
};
use zinc_transcript::{Blake3Transcript, traits::Transcript};
use zinc_uair::{ideal_collector::IdealOrZero, Uair, constraint_counter::count_constraints};
use zinc_utils::inner_product::MBSInnerProduct;
use zip_plus::{
    code::{
        raa::RaaConfig,
        raa_f2::{RaaF2Code, recommended_num_column_openings},
    },
    pcs::structs::{ZipPlusParams, ZipTypes},
};

use zinc_piop::{
    ideal_check::IdealCheckProtocol,
    projections::project_f2_trace_row_major,
    sumcheck::multi_degree::MultiDegreeSumcheck,
};

// ---------------------------------------------------------------------------
// F_2 ZipTypes / F2ZincTypes infrastructure.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct BenchF2ZipTypes<const D: usize> {}

impl<const D: usize> ZipTypes for BenchF2ZipTypes<D> {
    const NUM_COLUMN_OPENINGS: usize = recommended_num_column_openings(REP);
    type Eval = BinaryPoly<D>;
    type Cw = BinaryPoly<D>;
    type Fmod = Uint<{ U64::LIMBS * 4 }>;
    type PrimeTest = MillerRabin;
    type Chal = i128;
    type Pt = i128;
    type CombR = Int<{ U64::LIMBS * 8 }>;
    type Comb = DensePolynomial<Self::CombR, D>;
    type EvalDotChal = BinaryPolyInnerProduct<Self::Chal, D>;
    type CombDotChal =
        DensePolyInnerProduct<Self::CombR, Self::Chal, Self::CombR, MBSInnerProduct, D>;
    type ArrCombRDotChal = MBSInnerProduct;
}

#[derive(Copy, Clone)]
struct BenchRaaConfig;
impl RaaConfig for BenchRaaConfig {
    const PERMUTE_IN_PLACE: bool = false;
    const CHECK_FOR_OVERFLOWS: bool = false;
}

#[derive(Clone, Debug)]
struct BenchF2Types<const D: usize>(PhantomData<()>);

const REP: usize = 4;

impl<const D: usize> F2ZincTypes<D> for BenchF2Types<D> {
    type BinaryZt = BenchF2ZipTypes<D>;
    type BinaryLc = RaaF2Code<Self::BinaryZt, BenchRaaConfig, REP>;
}

// ---------------------------------------------------------------------------
// Bench setup.
// ---------------------------------------------------------------------------

type R = Int<4>;
type U = Sha256F2Uair<R>;
const D: usize = 32;

struct ProverFixture {
    trace: zinc_uair::UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D>,
    pp: ZipPlusParams<
        <BenchF2Types<D> as F2ZincTypes<D>>::BinaryZt,
        <BenchF2Types<D> as F2ZincTypes<D>>::BinaryLc,
    >,
    num_vars: usize,
    num_primary: usize,
}

fn setup_prover(num_vars: usize) -> ProverFixture {
    let mut rng_local = rng();
    let row_len: usize = 32;
    let poly_size = 1usize << num_vars;
    let num_rows = poly_size / row_len;
    assert_eq!(num_rows * row_len, poly_size);

    let trace = U::generate_random_trace(num_vars, &mut rng_local);
    let lc = <BenchF2Types<D> as F2ZincTypes<D>>::BinaryLc::new(row_len);
    let pp = ZipPlusParams::new(num_vars, num_rows, lc);

    ProverFixture {
        trace,
        pp,
        num_vars,
        num_primary: zinc_test_uair::sha256_f2::cols::NUM_BIN,
    }
}

// ---------------------------------------------------------------------------
// Per-region: top-level prover & verifier steps.
// ---------------------------------------------------------------------------

fn bench_prover_steps(group: &mut BenchmarkGroup<WallTime>, id: &str, fx: &ProverFixture) {
    group.bench_function(BenchmarkId::new("1-Commit", id), |bench| {
        bench.iter(|| {
            let mut transcript = Blake3Transcript::new();
            let (hint, comm) =
                ZincPlusPiopF2::<BenchF2Types<D>, U, D>::commit_and_absorb_f2_trace(
                    &mut transcript,
                    &fx.pp,
                    &fx.trace.binary_poly,
                )
                .expect("commit should succeed");
            black_box((hint, comm));
        });
    });

    group.bench_function(BenchmarkId::new("2-UAIR", id), |bench| {
        bench.iter_batched(
            || {
                let mut transcript = Blake3Transcript::new();
                let _ =
                    ZincPlusPiopF2::<BenchF2Types<D>, U, D>::commit_and_absorb_f2_trace(
                        &mut transcript,
                        &fx.pp,
                        &fx.trace.binary_poly,
                    )
                    .expect("commit should succeed");
                transcript
            },
            |mut transcript| {
                let (proof, subclaim) = ZincPlusPiopF2::<BenchF2Types<D>, U, D>
                    ::prove_f2_uair_with_groups(
                        &mut transcript,
                        &fx.trace,
                        &[] as &[F2VirtualBpSpec],
                        fx.num_vars,
                        sha256_f2_project_scalar::<R>,
                        eq_dot_column_groups,
                    )
                    .expect("UAIR prove should succeed");
                black_box((proof, subclaim));
            },
            criterion::BatchSize::PerIteration,
        );
    });

    group.bench_function(BenchmarkId::new("3-Open", id), |bench| {
        bench.iter_batched(
            || {
                let mut transcript = Blake3Transcript::new();
                let (hint, _comm) =
                    ZincPlusPiopF2::<BenchF2Types<D>, U, D>::commit_and_absorb_f2_trace(
                        &mut transcript,
                        &fx.pp,
                        &fx.trace.binary_poly,
                    )
                    .expect("commit should succeed");
                let (_uair_proof, subclaim) = ZincPlusPiopF2::<BenchF2Types<D>, U, D>
                    ::prove_f2_uair_with_groups(
                        &mut transcript,
                        &fx.trace,
                        &[],
                        fx.num_vars,
                        sha256_f2_project_scalar::<R>,
                        eq_dot_column_groups,
                    )
                    .expect("UAIR prove should succeed");
                (hint, subclaim, transcript)
            },
            |(hint, subclaim, mut transcript)| {
                let open_proof = ZincPlusPiopF2::<BenchF2Types<D>, U, D>::prove_f2_open(
                    &mut transcript,
                    &fx.pp,
                    &hint,
                    &fx.trace.binary_poly,
                    &subclaim.sumcheck_point,
                    &subclaim.alpha,
                    recommended_num_column_openings(REP),
                );
                black_box(open_proof);
            },
            criterion::BatchSize::PerIteration,
        );
    });
}

fn bench_verifier_steps(group: &mut BenchmarkGroup<WallTime>, id: &str, fx: &ProverFixture) {
    let proof = {
        let mut transcript = Blake3Transcript::new();
        ZincPlusPiopF2::<BenchF2Types<D>, U, D>::prove_f2_full(
            &mut transcript,
            &fx.pp,
            &fx.trace,
            &[],
            fx.num_vars,
            sha256_f2_project_scalar::<R>,
            recommended_num_column_openings(REP),
        )
        .expect("prove for verifier bench should succeed")
    };

    group.bench_function(BenchmarkId::new("1-VerifyUAIR", id), |bench| {
        bench.iter_batched(
            || {
                let mut transcript = Blake3Transcript::new();
                ZincPlusPiopF2::<BenchF2Types<D>, U, D>::absorb_commitment(
                    &mut transcript,
                    &proof.commitment,
                );
                transcript
            },
            |mut transcript| {
                let subclaim = ZincPlusPiopF2::<BenchF2Types<D>, U, D>::verify_f2_uair(
                    &mut transcript,
                    &proof.uair,
                    &[],
                    fx.num_vars,
                    fx.num_primary,
                    |ideal: &IdealOrZero<Sha256F2Ideal>| sha256_f2_project_ideal(ideal),
                )
                .expect("UAIR verify should succeed");
                black_box(subclaim);
            },
            criterion::BatchSize::PerIteration,
        );
    });

    group.bench_function(BenchmarkId::new("2-VerifyOpen", id), |bench| {
        bench.iter_batched(
            || {
                let mut transcript = Blake3Transcript::new();
                ZincPlusPiopF2::<BenchF2Types<D>, U, D>::absorb_commitment(
                    &mut transcript,
                    &proof.commitment,
                );
                let subclaim =
                    ZincPlusPiopF2::<BenchF2Types<D>, U, D>::verify_f2_uair(
                        &mut transcript,
                        &proof.uair,
                        &[],
                        fx.num_vars,
                        fx.num_primary,
                        |ideal: &IdealOrZero<Sha256F2Ideal>| sha256_f2_project_ideal(ideal),
                    )
                    .expect("UAIR verify should succeed");
                (transcript, subclaim)
            },
            |(mut transcript, subclaim)| {
                ZincPlusPiopF2::<BenchF2Types<D>, U, D>::verify_f2_open(
                    &mut transcript,
                    &fx.pp,
                    &proof.commitment,
                    &proof.open,
                    &subclaim,
                )
                .expect("open verify should succeed");
                black_box(());
            },
            criterion::BatchSize::PerIteration,
        );
    });
}

// ---------------------------------------------------------------------------
// MICRO breakdown: timing each sub-step inside prove_f2_uair_with_groups.
//
// The five sub-steps replicate the body of `prove_f2_uair_with_groups`
// in `protocol/src/f2_prove.rs`:
//   a. RowMajorProject  — `project_f2_trace_row_major`
//   b. ProjectScalars   — `zinc_piop::projections::project_scalars`
//   c. ICProveCombined  — `<U as IdealCheckProtocol>::prove_combined`
//   d. AlphaProject     — α draw + per-cell eval at α to GF(2^192)
//   e. Sumcheck         — `MultiDegreeSumcheck::prove_as_subprotocol`
// ---------------------------------------------------------------------------

fn bench_micro_prover_uair(
    group: &mut BenchmarkGroup<WallTime>,
    id: &str,
    fx: &ProverFixture,
) {
    use zinc_piop::projections::project_scalars;

    let num_constraints = count_constraints::<U>();
    let field_cfg = ();

    // a) RowMajorProject
    group.bench_function(BenchmarkId::new("UAIR-a-RowMajorProject", id), |bench| {
        bench.iter(|| {
            let row_major =
                project_f2_trace_row_major::<BinaryFieldGF192, _, _, D>(&fx.trace, &field_cfg);
            black_box(row_major);
        });
    });

    // b) ProjectScalars
    group.bench_function(BenchmarkId::new("UAIR-b-ProjectScalars", id), |bench| {
        bench.iter(|| {
            let scalars = project_scalars::<BinaryFieldGF192, U>(|s| {
                sha256_f2_project_scalar::<R>(s)
            });
            black_box(scalars);
        });
    });

    // c) ICProveCombined
    group.bench_function(BenchmarkId::new("UAIR-c-ICProveCombined", id), |bench| {
        let row_major =
            project_f2_trace_row_major::<BinaryFieldGF192, _, _, D>(&fx.trace, &field_cfg);
        let scalars = project_scalars::<BinaryFieldGF192, U>(|s| {
            sha256_f2_project_scalar::<R>(s)
        });
        bench.iter_batched(
            || {
                // Pre-prime transcript with the commit-absorb so the
                // IC's challenge draws are realistic.
                let mut transcript = Blake3Transcript::new();
                let _ = ZincPlusPiopF2::<BenchF2Types<D>, U, D>
                    ::commit_and_absorb_f2_trace(
                        &mut transcript, &fx.pp, &fx.trace.binary_poly,
                    )
                    .expect("commit");
                transcript
            },
            |mut transcript| {
                let (proof, state) = <U as IdealCheckProtocol>
                    ::prove_combined::<BinaryFieldGF192>(
                        &mut transcript,
                        &row_major,
                        &scalars,
                        num_constraints,
                        fx.num_vars,
                        &field_cfg,
                    )
                    .expect("IC prove");
                black_box((proof, state));
            },
            criterion::BatchSize::PerIteration,
        );
    });

    // d) AlphaProject
    group.bench_function(BenchmarkId::new("UAIR-d-AlphaProject", id), |bench| {
        bench.iter_batched(
            || {
                // Draw a fresh α each iteration.
                let mut transcript = Blake3Transcript::new();
                let alpha: BinaryFieldGF192 = transcript.get_field_challenge(&field_cfg);
                alpha
            },
            |alpha| {
                let projected: Vec<DenseMultilinearExtension<BinaryFieldGF192>> = fx
                    .trace
                    .binary_poly
                    .iter()
                    .map(|col| {
                        let evals_at_alpha: Vec<BinaryFieldGF192> = col
                            .evaluations
                            .iter()
                            .map(|cell| {
                                zinc_poly::univariate::binary_gf192::eval_f2_poly_d_at::<D>(
                                    cell, &alpha,
                                )
                            })
                            .collect();
                        DenseMultilinearExtension::from_evaluations_vec(
                            col.num_vars,
                            evals_at_alpha,
                            BinaryFieldGF192::zero(),
                        )
                    })
                    .collect();
                black_box(projected);
            },
            criterion::BatchSize::PerIteration,
        );
    });

    // e) Sumcheck
    group.bench_function(BenchmarkId::new("UAIR-e-Sumcheck", id), |bench| {
        // Build the projected trace + groups once outside the timed loop.
        let alpha: BinaryFieldGF192 = {
            let mut t = Blake3Transcript::new();
            t.get_field_challenge(&field_cfg)
        };
        let projected: Vec<DenseMultilinearExtension<BinaryFieldGF192>> = fx
            .trace
            .binary_poly
            .iter()
            .map(|col| {
                let evals_at_alpha: Vec<BinaryFieldGF192> = col
                    .evaluations
                    .iter()
                    .map(|cell| {
                        zinc_poly::univariate::binary_gf192::eval_f2_poly_d_at::<D>(
                            cell, &alpha,
                        )
                    })
                    .collect();
                DenseMultilinearExtension::from_evaluations_vec(
                    col.num_vars,
                    evals_at_alpha,
                    BinaryFieldGF192::zero(),
                )
            })
            .collect();
        let ic_eval_point: Vec<BinaryFieldGF192> = (0..fx.num_vars)
            .map(|i| BinaryFieldGF192::from_with_cfg(i as u64 + 1, &field_cfg))
            .collect();

        bench.iter_batched(
            || {
                let groups = eq_dot_column_groups(&ic_eval_point, &projected, &field_cfg);
                let transcript = Blake3Transcript::new();
                (groups, transcript)
            },
            |(groups, mut transcript)| {
                let (proof, states) =
                    MultiDegreeSumcheck::<BinaryFieldGF192>::prove_as_subprotocol(
                        &mut transcript,
                        groups,
                        fx.num_vars,
                        &field_cfg,
                    );
                black_box((proof, states));
            },
            criterion::BatchSize::PerIteration,
        );
    });
}

// ---------------------------------------------------------------------------
// MICRO breakdown: timing each sub-step inside `verify_f2_uair`.
//   a. ICVerify      — `<U as IdealCheckProtocol>::verify_as_subprotocol`
//   b. SumcheckVerify — `MultiDegreeSumcheck::verify_as_subprotocol`
//   c. ExtractEval   — `extract_column_evals_eq_dot_col` (eq inv + 41×mul)
// ---------------------------------------------------------------------------

fn bench_micro_verifier_uair(
    group: &mut BenchmarkGroup<WallTime>,
    id: &str,
    fx: &ProverFixture,
) {
    let proof = {
        let mut transcript = Blake3Transcript::new();
        ZincPlusPiopF2::<BenchF2Types<D>, U, D>::prove_f2_full(
            &mut transcript,
            &fx.pp,
            &fx.trace,
            &[],
            fx.num_vars,
            sha256_f2_project_scalar::<R>,
            recommended_num_column_openings(REP),
        )
        .expect("prove")
    };

    let num_constraints = count_constraints::<U>();
    let field_cfg = ();

    // a) IC verify
    group.bench_function(BenchmarkId::new("VerifyUAIR-a-IC", id), |bench| {
        bench.iter_batched(
            || {
                let mut t = Blake3Transcript::new();
                ZincPlusPiopF2::<BenchF2Types<D>, U, D>::absorb_commitment(&mut t, &proof.commitment);
                t
            },
            |mut transcript| {
                let ic_subclaim = <U as IdealCheckProtocol>
                    ::verify_as_subprotocol::<_, Sha256F2Ideal, _>(
                        &mut transcript,
                        proof.uair.ic_proof.clone(),
                        num_constraints,
                        fx.num_vars,
                        |ideal: &IdealOrZero<Sha256F2Ideal>| sha256_f2_project_ideal(ideal),
                        &field_cfg,
                    )
                    .expect("IC verify");
                black_box(ic_subclaim);
            },
            criterion::BatchSize::PerIteration,
        );
    });

    // b) Sumcheck verify
    group.bench_function(BenchmarkId::new("VerifyUAIR-b-Sumcheck", id), |bench| {
        bench.iter_batched(
            || {
                let mut t = Blake3Transcript::new();
                ZincPlusPiopF2::<BenchF2Types<D>, U, D>::absorb_commitment(&mut t, &proof.commitment);
                let _ = <U as IdealCheckProtocol>
                    ::verify_as_subprotocol::<_, Sha256F2Ideal, _>(
                        &mut t,
                        proof.uair.ic_proof.clone(),
                        num_constraints,
                        fx.num_vars,
                        |ideal: &IdealOrZero<Sha256F2Ideal>| sha256_f2_project_ideal(ideal),
                        &field_cfg,
                    )
                    .expect("IC verify");
                let _: BinaryFieldGF192 = t.get_field_challenge(&field_cfg);
                t
            },
            |mut transcript| {
                let sc = MultiDegreeSumcheck::<BinaryFieldGF192>::verify_as_subprotocol(
                    &mut transcript,
                    fx.num_vars,
                    &proof.uair.sumcheck_proof,
                    &field_cfg,
                )
                .expect("sumcheck verify");
                black_box(sc);
            },
            criterion::BatchSize::PerIteration,
        );
    });

    // c) Extract eval claims (eq invert + per-column division).
    group.bench_function(BenchmarkId::new("VerifyUAIR-c-Extract", id), |bench| {
        // Precompute (ic_eval_point, md_subclaims) outside the loop.
        let (ic_eval_point, md_subclaims) = {
            let mut t = Blake3Transcript::new();
            ZincPlusPiopF2::<BenchF2Types<D>, U, D>::absorb_commitment(&mut t, &proof.commitment);
            let ic_subclaim = <U as IdealCheckProtocol>
                ::verify_as_subprotocol::<_, Sha256F2Ideal, _>(
                    &mut t,
                    proof.uair.ic_proof.clone(),
                    num_constraints,
                    fx.num_vars,
                    |ideal: &IdealOrZero<Sha256F2Ideal>| sha256_f2_project_ideal(ideal),
                    &field_cfg,
                )
                .expect("IC");
            let _: BinaryFieldGF192 = t.get_field_challenge(&field_cfg);
            let md = MultiDegreeSumcheck::<BinaryFieldGF192>::verify_as_subprotocol(
                &mut t,
                fx.num_vars,
                &proof.uair.sumcheck_proof,
                &field_cfg,
            )
            .expect("sumcheck");
            (ic_subclaim.evaluation_point, md)
        };

        bench.iter(|| {
            let evals =
                extract_column_evals_eq_dot_col(&ic_eval_point, &md_subclaims).expect("eq_inv");
            black_box(evals);
        });
    });
}

// ---------------------------------------------------------------------------
// Criterion entry points.
// ---------------------------------------------------------------------------

fn e2e_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zinc+ F_2 SHA-256");
    let fx = setup_prover(9);
    let id = format!("nvars={}", fx.num_vars);

    group.bench_function(BenchmarkId::new("WitnessGen", &id), |bench| {
        let mut rng_local = rng();
        bench.iter(|| {
            black_box(U::generate_random_trace(fx.num_vars, &mut rng_local));
        });
    });

    group.bench_function(BenchmarkId::new("Prove", &id), |bench| {
        bench.iter(|| {
            let mut transcript = Blake3Transcript::new();
            let proof = ZincPlusPiopF2::<BenchF2Types<D>, U, D>::prove_f2_full(
                &mut transcript,
                &fx.pp,
                &fx.trace,
                &[],
                fx.num_vars,
                sha256_f2_project_scalar::<R>,
                recommended_num_column_openings(REP),
            )
            .expect("prove_f2_full should succeed");
            black_box(proof);
        });
    });

    let proof = {
        let mut transcript = Blake3Transcript::new();
        ZincPlusPiopF2::<BenchF2Types<D>, U, D>::prove_f2_full(
            &mut transcript,
            &fx.pp,
            &fx.trace,
            &[],
            fx.num_vars,
            sha256_f2_project_scalar::<R>,
            recommended_num_column_openings(REP),
        )
        .expect("prove for verifier bench should succeed")
    };

    group.bench_function(BenchmarkId::new("Verify", &id), |bench| {
        bench.iter(|| {
            let mut transcript = Blake3Transcript::new();
            let subclaim = ZincPlusPiopF2::<BenchF2Types<D>, U, D>::verify_f2_full(
                &mut transcript,
                &fx.pp,
                &proof,
                &[],
                fx.num_vars,
                fx.num_primary,
                |ideal: &IdealOrZero<Sha256F2Ideal>| sha256_f2_project_ideal(ideal),
            )
            .expect("verify_f2_full should succeed");
            black_box(subclaim);
        });
    });

    group.finish();
}

fn step_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zinc+ F_2 SHA-256 Steps");
    let fx = setup_prover(9);
    let id = format!("nvars={}", fx.num_vars);
    bench_prover_steps(&mut group, &id, &fx);
    bench_verifier_steps(&mut group, &id, &fx);
    group.finish();
}

fn micro_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zinc+ F_2 SHA-256 Micro");
    let fx = setup_prover(9);
    let id = format!("nvars={}", fx.num_vars);
    bench_micro_prover_uair(&mut group, &id, &fx);
    bench_micro_verifier_uair(&mut group, &id, &fx);
    group.finish();
}

criterion_group! {
    name = e2e;
    config = Criterion::default().sample_size(10);
    targets = e2e_benches
}
criterion_group! {
    name = steps;
    config = Criterion::default().sample_size(10);
    targets = step_benches
}
criterion_group! {
    name = micro;
    config = Criterion::default().sample_size(10);
    targets = micro_benches
}
criterion_main!(e2e, steps, micro);
