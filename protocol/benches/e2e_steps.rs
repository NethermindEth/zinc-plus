#![allow(clippy::arithmetic_side_effects)]

//! Per-step prover benchmark for BigLinearUairWithPublicInput, num_vars=9, 3 limbs.

use std::hint::black_box;
use std::marker::PhantomData;
use std::ops::Neg;

use criterion::{
    BatchSize, Criterion, criterion_group, criterion_main,
};
use crypto_bigint::U64;
use crypto_primitives::{
    ConstIntRing, ConstIntSemiring, Field, FixedSemiring, FromPrimitiveWithConfig, FromWithConfig,
    PrimeField, crypto_bigint_int::Int, crypto_bigint_monty::MontyField,
    crypto_bigint_uint::Uint,
};
use num_traits::Zero;
use rand::rng;
use zinc_piop::{
    combined_poly_resolver::CombinedPolyResolver,
    ideal_check::IdealCheckProtocol,
    multipoint_eval::MultipointEval,
    projections::{
        evaluate_trace_to_column_mles, project_scalars, project_scalars_to_field,
        project_trace_coeffs_row_major,
    },
};
use zinc_poly::{
    ConstCoeffBitWidth, Polynomial,
    mle::DenseMultilinearExtension,
    univariate::{
        binary::{BinaryPoly, BinaryPolyInnerProduct},
        dense::{DensePolyInnerProduct, DensePolynomial},
    },
};
use zinc_primality::{MillerRabin, PrimalityTest};
use zinc_protocol::{absorb_public_columns, compute_lifted_evals};
use zinc_test_uair::{BigLinearUairWithPublicInput, GenerateMultiTypeWitness};
use zinc_transcript::{
    traits::{ConstTranscribable, Transcript},
};
use zinc_uair::{
    Uair, constraint_counter::count_constraints, degree_counter::count_max_degree,
};
use zinc_utils::{
    CHECKED,
    from_ref::FromRef,
    inner_product::{InnerProduct, MBSInnerProduct, ScalarProduct},
    mul_by_scalar::MulByScalar,
    named::Named,
};
use zip_plus::{
    code::{
        LinearCode,
        iprs::{IprsCode, PnttConfigF2_16_1},
    },
    pcs::structs::{ZipPlus, ZipPlusCommitment, ZipPlusParams, ZipTypes},
    pcs_transcript::PcsProverTranscript,
};

use zinc_protocol::ZincTypes;

// ── Type scaffolding (mirrors e2e.rs) ──────────────────────────────────

const IPRS_DEPTH: usize = 1;
const DEGREE_PLUS_ONE: usize = 32;
const INT_LIMBS: usize = U64::LIMBS;
const FIELD_LIMBS: usize = 3;

type F = MontyField<FIELD_LIMBS>;

#[allow(clippy::type_complexity)]
#[derive(Debug, Clone, Copy)]
pub struct GenericBenchZipTypes<
    Eval,
    Cw,
    Fmod,
    PrimeTest,
    Chal,
    Pt,
    CombR,
    Comb,
    EvalDotChal,
    CombDotChal,
    ArrCombRDotChal,
>(
    PhantomData<(
        Eval,
        Cw,
        Fmod,
        PrimeTest,
        Chal,
        Pt,
        CombR,
        Comb,
        EvalDotChal,
        CombDotChal,
        ArrCombRDotChal,
    )>,
);

impl<Eval, Cw, Fmod, PrimeTest, Chal, Pt, CombR, Comb, EvalDotChal, CombDotChal, ArrCombRDotChal>
    ZipTypes
    for GenericBenchZipTypes<
        Eval,
        Cw,
        Fmod,
        PrimeTest,
        Chal,
        Pt,
        CombR,
        Comb,
        EvalDotChal,
        CombDotChal,
        ArrCombRDotChal,
    >
where
    Eval: Named + ConstCoeffBitWidth + Default + Clone + Send + Sync,
    Cw: FixedSemiring + ConstCoeffBitWidth + ConstTranscribable + FromRef<Eval> + Named + Copy,
    Fmod: ConstIntSemiring + ConstTranscribable + Named,
    PrimeTest: PrimalityTest<Fmod> + Send + Sync,
    Chal: ConstIntRing + ConstTranscribable + Named,
    Pt: ConstIntRing,
    CombR: ConstIntRing
        + Neg<Output = CombR>
        + ConstTranscribable
        + FromRef<CombR>
        + for<'a> MulByScalar<&'a Chal>,
    Comb: FixedSemiring + Polynomial<CombR> + FromRef<Eval> + FromRef<Cw> + Named,
    EvalDotChal: InnerProduct<Eval, Chal, CombR> + Send + Sync,
    CombDotChal: InnerProduct<Comb, Chal, CombR> + Send + Sync,
    ArrCombRDotChal: InnerProduct<[CombR], Chal, CombR> + Send + Sync,
{
    const NUM_COLUMN_OPENINGS: usize = 200;
    type Eval = Eval;
    type Cw = Cw;
    type Fmod = Fmod;
    type PrimeTest = PrimeTest;
    type Chal = Chal;
    type Pt = Pt;
    type CombR = CombR;
    type Comb = Comb;
    type EvalDotChal = EvalDotChal;
    type CombDotChal = CombDotChal;
    type ArrCombRDotChal = ArrCombRDotChal;
}

struct GenericBenchZincTypes<Int, CwR, Chal, Pt, CombR, Fmod, PrimeTest, const D: usize>(
    PhantomData<(Int, CwR, Chal, Pt, CombR, Fmod, PrimeTest)>,
);

impl<Int, CwR, Chal, Pt, CombR, Fmod, PrimeTest, const D: usize> ZincTypes<D>
    for GenericBenchZincTypes<Int, CwR, Chal, Pt, CombR, Fmod, PrimeTest, D>
where
    Int: ConstIntSemiring
        + for<'a> MulByScalar<&'a i64, CwR>
        + Named
        + ConstCoeffBitWidth
        + ConstTranscribable
        + Default
        + Clone
        + Send
        + Sync,
    CwR: FixedSemiring
        + for<'a> MulByScalar<&'a i64>
        + ConstCoeffBitWidth
        + ConstTranscribable
        + Named
        + FromRef<Int>
        + FromRef<CwR>
        + Copy,
    Chal: ConstIntRing + ConstTranscribable + Named,
    Pt: ConstIntRing,
    CombR: ConstIntRing
        + Polynomial<CombR>
        + Neg<Output = CombR>
        + for<'a> MulByScalar<&'a i64>
        + for<'a> MulByScalar<&'a Chal>
        + ConstTranscribable
        + Named
        + FromRef<i64>
        + FromRef<Int>
        + FromRef<CwR>
        + FromRef<Chal>
        + FromRef<CombR>,
    Fmod: ConstIntSemiring + ConstTranscribable + Named,
    PrimeTest: PrimalityTest<Fmod> + Send + Sync,
{
    type Int = Int;
    type Chal = Chal;
    type Pt = Pt;
    type CombR = CombR;
    type Fmod = Fmod;
    type PrimeTest = PrimeTest;

    type BinaryZt = GenericBenchZipTypes<
        BinaryPoly<D>,
        DensePolynomial<i64, D>,
        Fmod,
        PrimeTest,
        Chal,
        Pt,
        CombR,
        DensePolynomial<CombR, D>,
        BinaryPolyInnerProduct<Chal, D>,
        DensePolyInnerProduct<CombR, Chal, CombR, MBSInnerProduct, D>,
        MBSInnerProduct,
    >;
    type ArbitraryZt = GenericBenchZipTypes<
        DensePolynomial<Int, D>,
        DensePolynomial<CwR, D>,
        Fmod,
        PrimeTest,
        Chal,
        Pt,
        CombR,
        DensePolynomial<CombR, D>,
        DensePolyInnerProduct<Int, Chal, CombR, MBSInnerProduct, D>,
        DensePolyInnerProduct<CombR, Chal, CombR, MBSInnerProduct, D>,
        MBSInnerProduct,
    >;
    type IntZt = GenericBenchZipTypes<
        Int,
        CwR,
        Fmod,
        PrimeTest,
        Chal,
        Pt,
        CombR,
        CombR,
        ScalarProduct,
        ScalarProduct,
        MBSInnerProduct,
    >;

    type BinaryLc = IprsCode<Self::BinaryZt, PnttConfigF2_16_1<IPRS_DEPTH>, CHECKED>;
    type ArbitraryLc = IprsCode<Self::ArbitraryZt, PnttConfigF2_16_1<IPRS_DEPTH>, CHECKED>;
    type IntLc = IprsCode<Self::IntZt, PnttConfigF2_16_1<IPRS_DEPTH>, CHECKED>;
}

type BenchZincTypes = GenericBenchZincTypes<
    i64,
    i128,
    i128,
    i128,
    Int<{ INT_LIMBS * 6 }>,
    Uint<FIELD_LIMBS>,
    MillerRabin,
    DEGREE_PLUS_ONE,
>;

type Zt = BenchZincTypes;
type U = BigLinearUairWithPublicInput<i64>;

type BinaryZt = <Zt as ZincTypes<DEGREE_PLUS_ONE>>::BinaryZt;
type ArbitraryZt = <Zt as ZincTypes<DEGREE_PLUS_ONE>>::ArbitraryZt;
type IntZt = <Zt as ZincTypes<DEGREE_PLUS_ONE>>::IntZt;

type Pp = (
    ZipPlusParams<BinaryZt, <Zt as ZincTypes<DEGREE_PLUS_ONE>>::BinaryLc>,
    ZipPlusParams<ArbitraryZt, <Zt as ZincTypes<DEGREE_PLUS_ONE>>::ArbitraryLc>,
    ZipPlusParams<IntZt, <Zt as ZincTypes<DEGREE_PLUS_ONE>>::IntLc>,
);

fn setup_pp(num_vars: usize) -> Pp {
    let poly_size = 1 << num_vars;
    (
        ZipPlus::setup(
            poly_size,
            <Zt as ZincTypes<DEGREE_PLUS_ONE>>::BinaryLc::new(poly_size),
        ),
        ZipPlus::setup(
            poly_size,
            <Zt as ZincTypes<DEGREE_PLUS_ONE>>::ArbitraryLc::new(poly_size),
        ),
        ZipPlus::setup(
            poly_size,
            <Zt as ZincTypes<DEGREE_PLUS_ONE>>::IntLc::new(poly_size),
        ),
    )
}

// ── Benchmark ──────────────────────────────────────────────────────────

const NUM_VARS: usize = 9;

fn e2e_steps_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("BigLinearPI Prover Steps");

    // Generate witness once
    let mut rng = rng();
    let (trace_bin, _trace_arb, trace_int) =
        <U as GenerateMultiTypeWitness>::generate_witness(NUM_VARS, &mut rng);
    let trace_arb: Vec<DenseMultilinearExtension<<ArbitraryZt as ZipTypes>::Eval>> = vec![];

    let pp = setup_pp(NUM_VARS);
    let (pp_bin, pp_arb, pp_int) = &pp;

    let sig = U::signature();
    let num_pub_bin = sig.public_binary_poly_cols;
    let num_pub_arb = sig.public_arbitrary_poly_cols;
    let num_pub_int = sig.public_int_cols;

    let witness_bin = &trace_bin[num_pub_bin..];
    let witness_arb = &trace_arb[num_pub_arb..];
    let witness_int = &trace_int[num_pub_int..];

    // ── Step 0: Commit ──

    group.bench_function("Step0_Commit", |bench| {
        bench.iter(|| {
            macro_rules! commit_optionally {
                ($pp:expr, $trace:expr) => {
                    if $trace.is_empty() {
                        (
                            None,
                            ZipPlusCommitment {
                                root: Default::default(),
                                batch_size: 0,
                            },
                        )
                    } else {
                        let (hint, commitment) =
                            ZipPlus::commit($pp, $trace).expect("commit failed");
                        (Some(hint), commitment)
                    }
                };
            }
            let (hint_bin, commitment_bin) = commit_optionally!(pp_bin, witness_bin);
            let (hint_arb, commitment_arb) = commit_optionally!(pp_arb, witness_arb);
            let (hint_int, commitment_int) = commit_optionally!(pp_int, witness_int);

            let mut pcs_transcript = PcsProverTranscript::new_from_commitments(
                [&commitment_bin, &commitment_arb, &commitment_int].into_iter(),
            );

            absorb_public_columns(
                &mut pcs_transcript.fs_transcript,
                &trace_bin[..num_pub_bin],
            );
            absorb_public_columns(
                &mut pcs_transcript.fs_transcript,
                &trace_arb[..num_pub_arb],
            );
            absorb_public_columns(
                &mut pcs_transcript.fs_transcript,
                &trace_int[..num_pub_int],
            );

            black_box((hint_bin, hint_arb, hint_int, pcs_transcript));
        });
    });

    // For subsequent steps, we need the transcript state from step 0.
    // Run step 0 once to get materialized state.
    macro_rules! commit_optionally {
        ($pp:expr, $trace:expr) => {
            if $trace.is_empty() {
                (
                    None,
                    ZipPlusCommitment {
                        root: Default::default(),
                        batch_size: 0,
                    },
                )
            } else {
                let (hint, commitment) = ZipPlus::commit($pp, $trace).expect("commit failed");
                (Some(hint), commitment)
            }
        };
    }
    let (hint_bin, commitment_bin) = commit_optionally!(pp_bin, witness_bin);
    let (hint_arb, commitment_arb) = commit_optionally!(pp_arb, witness_arb);
    let (hint_int, commitment_int) = commit_optionally!(pp_int, witness_int);

    let mut pcs_transcript = PcsProverTranscript::new_from_commitments(
        [&commitment_bin, &commitment_arb, &commitment_int].into_iter(),
    );
    absorb_public_columns(
        &mut pcs_transcript.fs_transcript,
        &trace_bin[..num_pub_bin],
    );
    absorb_public_columns(
        &mut pcs_transcript.fs_transcript,
        &trace_arb[..num_pub_arb],
    );
    absorb_public_columns(
        &mut pcs_transcript.fs_transcript,
        &trace_int[..num_pub_int],
    );

    // ── Step 1: Prime projection ──

    let transcript_after_step0 = pcs_transcript.fs_transcript.clone();

    group.bench_function("Step1_PrimeProjection", |bench| {
        bench.iter_batched(
            || transcript_after_step0.clone(),
            |mut transcript| {
                let field_cfg =
                    transcript.get_random_field_cfg::<F, Uint<FIELD_LIMBS>, MillerRabin>();
                let projected_scalars =
                    project_scalars::<F, U>(|s| zinc_protocol::project_scalar_fn(s, &field_cfg));
                let projected_trace =
                    project_trace_coeffs_row_major::<F, i64, i64, DEGREE_PLUS_ONE>(
                        &trace_bin,
                        &trace_arb,
                        &trace_int,
                        &field_cfg,
                    );
                black_box((projected_scalars, projected_trace, field_cfg));
            },
            BatchSize::SmallInput,
        );
    });

    // Materialize step 1 output
    let field_cfg = pcs_transcript
        .fs_transcript
        .get_random_field_cfg::<F, Uint<FIELD_LIMBS>, MillerRabin>();
    let projected_scalars_fx =
        project_scalars::<F, U>(|s| zinc_protocol::project_scalar_fn(s, &field_cfg));
    let num_constraints = count_constraints::<U>();
    let projected_trace = project_trace_coeffs_row_major::<F, i64, i64, DEGREE_PLUS_ONE>(
        &trace_bin,
        &trace_arb,
        &trace_int,
        &field_cfg,
    );

    // ── Step 2: Ideal check ──

    let transcript_after_step1 = pcs_transcript.fs_transcript.clone();

    group.bench_function("Step2_IdealCheck", |bench| {
        bench.iter_batched(
            || transcript_after_step1.clone(),
            |mut transcript| {
                let (ic_proof, ic_state) = U::prove_combined(
                    &mut transcript,
                    &projected_trace,
                    &projected_scalars_fx,
                    num_constraints,
                    NUM_VARS,
                    &field_cfg,
                )
                .expect("ideal check failed");
                black_box((ic_proof, ic_state));
            },
            BatchSize::SmallInput,
        );
    });

    // Materialize step 2
    let (ic_proof, ic_prover_state) = U::prove_combined(
        &mut pcs_transcript.fs_transcript,
        &projected_trace,
        &projected_scalars_fx,
        num_constraints,
        NUM_VARS,
        &field_cfg,
    )
    .expect("ideal check failed");

    // ── Step 3: Evaluation projection ──

    let transcript_after_step2 = pcs_transcript.fs_transcript.clone();

    group.bench_function("Step3_EvalProjection", |bench| {
        bench.iter_batched(
            || transcript_after_step2.clone(),
            |mut transcript| {
                let projecting_element: i128 = transcript.get_challenge();
                let projecting_element_f: F =
                    F::from_with_cfg(&projecting_element, &field_cfg);
                let projected_trace_f =
                    evaluate_trace_to_column_mles(&projected_trace, &projecting_element_f);
                let projected_scalars_f = project_scalars_to_field(
                    projected_scalars_fx.clone(),
                    &projecting_element_f,
                )
                .expect("scalar projection failed");
                black_box((projected_trace_f, projected_scalars_f));
            },
            BatchSize::SmallInput,
        );
    });

    // Materialize step 3
    let projecting_element: i128 = pcs_transcript.fs_transcript.get_challenge();
    let projecting_element_f: F = F::from_with_cfg(&projecting_element, &field_cfg);
    let projected_trace_f =
        evaluate_trace_to_column_mles(&projected_trace, &projecting_element_f);
    let projected_scalars_f =
        project_scalars_to_field(projected_scalars_fx.clone(), &projecting_element_f)
            .expect("scalar projection failed");
    let max_degree = count_max_degree::<U>();

    // ── Step 4: F_q Sumcheck ──

    let transcript_after_step3 = pcs_transcript.fs_transcript.clone();

    group.bench_function("Step4_Sumcheck", |bench| {
        bench.iter_batched(
            || transcript_after_step3.clone(),
            |mut transcript| {
                let (cpr_proof, cpr_state) =
                    CombinedPolyResolver::prove_as_subprotocol::<U>(
                        &mut transcript,
                        projected_trace_f.clone(),
                        &ic_prover_state.evaluation_point,
                        &projected_scalars_f,
                        num_constraints,
                        NUM_VARS,
                        max_degree,
                        &field_cfg,
                    )
                    .expect("sumcheck failed");
                black_box((cpr_proof, cpr_state));
            },
            BatchSize::SmallInput,
        );
    });

    // Materialize step 4
    let (cpr_proof, cpr_prover_state) = CombinedPolyResolver::prove_as_subprotocol::<U>(
        &mut pcs_transcript.fs_transcript,
        projected_trace_f.clone(),
        &ic_prover_state.evaluation_point,
        &projected_scalars_f,
        num_constraints,
        NUM_VARS,
        max_degree,
        &field_cfg,
    )
    .expect("sumcheck failed");

    // ── Step 5: Multi-point evaluation sumcheck ──

    let transcript_after_step4 = pcs_transcript.fs_transcript.clone();

    group.bench_function("Step5_MultipointEval", |bench| {
        bench.iter_batched(
            || transcript_after_step4.clone(),
            |mut transcript| {
                let (mp_proof, mp_state) = MultipointEval::prove_as_subprotocol(
                    &mut transcript,
                    &projected_trace_f,
                    &cpr_prover_state.evaluation_point,
                    &cpr_proof.up_evals,
                    &cpr_proof.down_evals,
                    &field_cfg,
                )
                .expect("multipoint eval failed");
                black_box((mp_proof, mp_state));
            },
            BatchSize::SmallInput,
        );
    });

    // Materialize step 5
    let (_mp_proof, mp_prover_state) = MultipointEval::prove_as_subprotocol(
        &mut pcs_transcript.fs_transcript,
        &projected_trace_f,
        &cpr_prover_state.evaluation_point,
        &cpr_proof.up_evals,
        &cpr_proof.down_evals,
        &field_cfg,
    )
    .expect("multipoint eval failed");

    let r_0 = &mp_prover_state.eval_point;

    // ── Step 6: Lift-and-project ──

    group.bench_function("Step6_LiftAndProject", |bench| {
        bench.iter(|| {
            let lifted_evals = compute_lifted_evals::<F, DEGREE_PLUS_ONE>(
                r_0,
                &trace_bin,
                &projected_trace,
                &field_cfg,
            );
            black_box(lifted_evals);
        });
    });

    // Materialize step 6
    let lifted_evals = compute_lifted_evals::<F, DEGREE_PLUS_ONE>(
        r_0,
        &trace_bin,
        &projected_trace,
        &field_cfg,
    );

    let mut transcription_buf: Vec<u8> = vec![0; <F as Field>::Inner::NUM_BYTES];
    for bar_u in &lifted_evals {
        pcs_transcript
            .fs_transcript
            .absorb_random_field_slice(&bar_u.coeffs, &mut transcription_buf);
    }

    // ── Step 7: PCS open ──

    let transcript_after_step6 = pcs_transcript.clone();

    group.bench_function("Step7_PcsOpen", |bench| {
        bench.iter_batched(
            || transcript_after_step6.clone(),
            |mut pcs_t| {
                if let Some(hint_bin) = &hint_bin {
                    let _ = ZipPlus::<BinaryZt, <Zt as ZincTypes<DEGREE_PLUS_ONE>>::BinaryLc>::prove_f::<
                        _,
                        CHECKED,
                    >(
                        &mut pcs_t, pp_bin, witness_bin, r_0, hint_bin, &field_cfg,
                    )
                    .expect("pcs prove bin failed");
                }
                if let Some(hint_arb) = &hint_arb {
                    let _ = ZipPlus::<
                        ArbitraryZt,
                        <Zt as ZincTypes<DEGREE_PLUS_ONE>>::ArbitraryLc,
                    >::prove_f::<_, CHECKED>(
                        &mut pcs_t, pp_arb, witness_arb, r_0, hint_arb, &field_cfg,
                    )
                    .expect("pcs prove arb failed");
                }
                if let Some(hint_int) = &hint_int {
                    let _ =
                        ZipPlus::<IntZt, <Zt as ZincTypes<DEGREE_PLUS_ONE>>::IntLc>::prove_f::<
                            _,
                            CHECKED,
                        >(
                            &mut pcs_t, pp_int, witness_int, r_0, hint_int, &field_cfg,
                        )
                        .expect("pcs prove int failed");
                }
                black_box(pcs_t.stream.into_inner());
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(500);
    targets = e2e_steps_benches
}
criterion_main!(benches);
