//! XMSS multi-validator SHA-256 benchmark using the 4x-folded pipeline.
//!
//! This benchmark extends `steps_sha256_8x_folded` to prove **16 SHA-256
//! compressions** (1024 rows, `num_vars=10`) representing XMSS signature
//! verification for multiple validators. Each validator requires ~6–8
//! compression blocks for message hashing, WOTS chain segments, public
//! key hashing, and Merkle tree path verification.
//!
//! Uses `PnttConfigF2_16R4B64<2>` (BASE_LEN=64, DEPTH=2, INPUT_LEN=4096)
//! with the 4x-folded pipeline (BinaryPoly<32> → 16 → 8).
//!
//! Run with:
//!   cargo bench --bench steps_sha256_xmss_folded -p zinc-snark --features "parallel simd asm qx-constraints no-f2x true-ideal"

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
            PnttConfigF2_16R4B64,
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
const XMSS_BATCH_SIZE: usize = 30;

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
const XMSS_BATCH_SIZE: usize = NO_F2X_NUM_COLS;
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

struct XmssZipTypes<CwCoeff, const D_PLUS_ONE: usize>(PhantomData<CwCoeff>);

impl<CwCoeff, const D_PLUS_ONE: usize> ZipTypes for XmssZipTypes<CwCoeff, D_PLUS_ONE>
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

/// PCS types for XMSS benchmark: BinaryPoly<8> codewords.
/// Uses PnttConfigF2_16R4B64<2> (BASE_LEN=64, BASE_DIM=256, DEPTH=2):
///   INPUT_LEN = 64 × 64 = 4096,
///   OUTPUT_LEN = 256 × 64 = 16384 (rate 1/4).
/// Supports num_vars=10 (1024 rows = 16 SHA-256 compressions).
type FoldedZt = XmssZipTypes<i64, 8>;
type FoldedLc = IprsCode<
    FoldedZt,
    PnttConfigF2_16R4B64<2>,
    BinaryPolyWideningMulByScalar<i64>,
    UNCHECKED,
>;

// ─── Parameters ─────────────────────────────────────────────────────────────

/// 2^10 = 1024 rows = 16 × 64 SHA-256 compressions.
/// With ~6 blocks/validator, this fits 2 XMSS validators (12 active blocks)
/// plus 4 blocks of tail-safety padding.
const XMSS_NUM_VARS: usize = 10;

const XMSS_LOOKUP_COL_COUNT: usize = 10;

fn xmss_lookup_specs_4chunks() -> Vec<LookupColumnSpec> {
    (0..XMSS_LOOKUP_COL_COUNT)
        .map(|i| LookupColumnSpec {
            column_index: i,
            table_type: LookupTableType::BitPoly { width: 32, chunk_width: Some(8) },
        })
        .collect()
}

fn xmss_affine_lookup_specs_4chunks() -> Vec<AffineLookupSpec> {
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

fn generate_xmss_trace(num_vars: usize) -> Vec<DenseMultilinearExtension<BinaryPoly<32>>> {
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

/// XMSS multi-validator benchmark using the 4x-folded Hybrid GKR c=2 pipeline.
///
/// Proves 16 SHA-256 compressions (1024 rows) representing XMSS signature
/// verification. Uses PnttConfigF2_16R4B64<2> for the PCS.
///
/// Benchmarks:
///   1. WitnessGen — generate the trace (1024 rows)
///   2. E2E/Prover — total proving time (4x-Folded Hybrid GKR c=2)
///   3. E2E/Verifier — total verification time
fn sha256_xmss_folded_stepwise(c: &mut Criterion) {
    use zinc_sha256_uair::CyclotomicIdeal;
    use zinc_uair::ideal_collector::IdealOrZero;

    let mem_tracker = MemoryTracker::start();

    let mut group = c.benchmark_group(format!(
        "XMSS 4x-Folded Hybrid GKR c=2 Steps (grind={})",
        <FoldedZt as ZipTypes>::GRINDING_BITS,
    ));
    group.sample_size(30);

    // -- Folded PCS params -------------------------------------------
    let folded_extra_vars: usize = 2; // Two splits: 32→16→8
    let folded_num_vars = XMSS_NUM_VARS + folded_extra_vars;
    let folded_row_len = 1usize << folded_num_vars;
    let folded_lc = FoldedLc::new(folded_row_len);
    let folded_params = ZipPlusParams::<FoldedZt, FoldedLc>::new(
        folded_num_vars, 1, folded_lc,
    );

    let xmss_lookup_specs = xmss_lookup_specs_4chunks();
    let xmss_affine_specs = xmss_affine_lookup_specs_4chunks();

    // ── 1. Witness Generation ───────────────────────────────────────
    group.bench_function("WitnessGen", |b| {
        b.iter(|| {
            let trace = generate_xmss_trace(XMSS_NUM_VARS);
            black_box(trace);
        });
    });

    // Pre-generate the trace for subsequent steps.
    let xmss_trace = generate_xmss_trace(XMSS_NUM_VARS);
    assert_eq!(xmss_trace.len(), XMSS_BATCH_SIZE);

    let num_compressions = (1usize << XMSS_NUM_VARS) / 64;
    eprintln!(
        "\n── XMSS Trace Info ──────────────────────────────────────",
    );
    eprintln!(
        "  num_vars={}, rows={}, compressions={}, columns={}",
        XMSS_NUM_VARS,
        1usize << XMSS_NUM_VARS,
        num_compressions,
        XMSS_BATCH_SIZE,
    );
    eprintln!("─────────────────────────────────────────────────────────\n");

    // Shared: public columns for verifier
    let sha_sig = BenchBpUair::signature();
    let sha_public_cols: Vec<_> = sha_sig.public_columns.iter()
        .map(|&i| xmss_trace[i].clone()).collect();

    #[cfg(feature = "boundary")]
    let bdry_public_cols: Vec<_> = {
        let bdry_trace = zinc_sha256_uair::boundary::generate_boundary_witness::<32>(&xmss_trace, XMSS_NUM_VARS);
        let bdry_sig = zinc_sha256_uair::boundary::Sha256UairBoundaryNoF2x::signature();
        bdry_sig.public_columns.iter()
            .map(|&i| bdry_trace[i].clone()).collect()
    };
    #[cfg(not(feature = "boundary"))]
    let bdry_public_cols: Vec<zinc_poly::mle::DenseMultilinearExtension<BinaryPoly<32>>> = vec![];

    // ── 2. E2E Total Prover ─────────────────────────────────────────
    // Warmup run to prime thread pools and caches.
    let _warmup_proof = zinc_snark::pipeline::prove_hybrid_gkr_logup_4x_folded::<
        BenchBpUair, BenchQxUair, FoldedZt, FoldedLc, 32, 16, 8, UNCHECKED,
    >(
        &folded_params, &xmss_trace, XMSS_NUM_VARS,
        &xmss_lookup_specs, &xmss_affine_specs, 2,
    );

    let hybrid_proof = zinc_snark::pipeline::prove_hybrid_gkr_logup_4x_folded::<
        BenchBpUair, BenchQxUair, FoldedZt, FoldedLc, 32, 16, 8, UNCHECKED,
    >(
        &folded_params, &xmss_trace, XMSS_NUM_VARS,
        &xmss_lookup_specs, &xmss_affine_specs, 2,
    );

    // Print prover pipeline timing breakdown.
    {
        let t = &hybrid_proof.timing;
        eprintln!("\n── XMSS 4x-Folded Hybrid GKR c=2 Prover Pipeline Timing ──");
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
        eprintln!("──────────────────────────────────────────────────────────\n");
    }

    // ── Proof size ──────────────────────────────────────────────────
    {
        use zinc_snark::pipeline::LookupProofData;
        let pcs_bytes = hybrid_proof.pcs_proof_bytes.len();
        let ic_bytes: usize = hybrid_proof.ic_proof_values.iter().map(|v| v.len()).sum();
        let qx_ic_bytes: usize = hybrid_proof.qx_ic_proof_values.iter().map(|v| v.len()).sum();
        let cpr_msg_bytes: usize = hybrid_proof.cpr_sumcheck_messages.iter().map(|v| v.len()).sum();
        let cpr_sum_bytes = hybrid_proof.cpr_sumcheck_claimed_sum.len();
        let cpr_up: usize = hybrid_proof.cpr_up_evals.iter().map(|v| v.len()).sum();
        let cpr_dn: usize = hybrid_proof.cpr_down_evals.iter().map(|v| v.len()).sum();
        let qx_cpr_msg: usize = hybrid_proof.qx_cpr_sumcheck_messages.iter().map(|v| v.len()).sum();
        let qx_cpr_sum = hybrid_proof.qx_cpr_sumcheck_claimed_sum.len();
        let qx_cpr_up: usize = hybrid_proof.qx_cpr_up_evals.iter().map(|v| v.len()).sum();
        let qx_cpr_dn: usize = hybrid_proof.qx_cpr_down_evals.iter().map(|v| v.len()).sum();
        let shift_sc_bytes: usize = hybrid_proof.shift_sumcheck.as_ref().map_or(0, |sc| {
            let rounds: usize = sc.rounds.iter().map(|v| v.len()).sum();
            let finals: usize = sc.v_finals.iter().map(|v| v.len()).sum();
            rounds + finals
        });
        let qx_shift_sc_bytes: usize = hybrid_proof.qx_shift_sumcheck.as_ref().map_or(0, |sc| {
            let rounds: usize = sc.rounds.iter().map(|v| v.len()).sum();
            let finals: usize = sc.v_finals.iter().map(|v| v.len()).sum();
            rounds + finals
        });
        let eval_pt_bytes: usize = hybrid_proof.evaluation_point_bytes.iter().map(|v| v.len()).sum();
        let pcs_eval_bytes: usize = hybrid_proof.pcs_evals_bytes.iter().map(|v| v.len()).sum();
        let fold_c1: usize = hybrid_proof.folding_c1s_bytes.iter().map(|v| v.len()).sum();
        let fold_c2: usize = hybrid_proof.folding_c2s_bytes.iter().map(|v| v.len()).sum();
        let fold_c3: usize = hybrid_proof.folding_c3s_bytes.iter().map(|v| v.len()).sum();
        let fold_c4: usize = hybrid_proof.folding_c4s_bytes.iter().map(|v| v.len()).sum();
        let folding_total = fold_c1 + fold_c2 + fold_c3 + fold_c4;

        let lookup_bytes: usize = match &hybrid_proof.lookup_proof {
            Some(LookupProofData::HybridGkr(proof)) => {
                let fe_bytes = <Uint<{ INT_LIMBS * 3 }> as ConstTranscribable>::NUM_BYTES;
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
            _ => 0,
        };

        let piop_total = ic_bytes + qx_ic_bytes
            + cpr_msg_bytes + cpr_sum_bytes + cpr_up + cpr_dn
            + qx_cpr_msg + qx_cpr_sum + qx_cpr_up + qx_cpr_dn
            + lookup_bytes + shift_sc_bytes + qx_shift_sc_bytes
            + eval_pt_bytes + pcs_eval_bytes + folding_total;
        let total_raw = pcs_bytes + piop_total;

        eprintln!("\n=== XMSS 4x-Folded Hybrid GKR c=2 Proof Size ===");
        eprintln!("  PCS:              {:>6} B  ({:.1} KB)", pcs_bytes, pcs_bytes as f64 / 1024.0);
        eprintln!("  PIOP:             {:>6} B  ({:.1} KB)", piop_total, piop_total as f64 / 1024.0);
        eprintln!("  Total raw:        {:>6} B  ({:.1} KB)", total_raw, total_raw as f64 / 1024.0);
        eprintln!("═══════════════════════════════════════════════════\n");
    }

    group.bench_function("E2E/Prover (4x-Folded Hybrid GKR c=2)", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let trace = generate_xmss_trace(XMSS_NUM_VARS);
                let t = Instant::now();
                let _ = zinc_snark::pipeline::prove_hybrid_gkr_logup_4x_folded::<
                    BenchBpUair, BenchQxUair, FoldedZt, FoldedLc, 32, 16, 8, UNCHECKED,
                >(
                    &folded_params, &trace, XMSS_NUM_VARS,
                    &xmss_lookup_specs, &xmss_affine_specs, 2,
                );
                total += t.elapsed();
            }
            total
        });
    });

    // ── 3. E2E Total Verifier ───────────────────────────────────────
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
            BenchBpUair, BenchQxUair, FoldedZt, FoldedLc, 32, 16, 8, UNCHECKED, _, _, _, _,
        >(
            &folded_params, &hybrid_proof, XMSS_NUM_VARS,
            |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
            qx_ideal_closure,
            &sha_public_cols,
            &bdry_public_cols,
        );
        assert!(r.accepted, "Verifier rejected the proof");
        let t = &r.timing;
        eprintln!("\n── XMSS Verifier step timing ────────────────────────────");
        eprintln!("  IC verify:           {:>8.3} ms", t.ideal_check_verify.as_secs_f64() * 1000.0);
        eprintln!("  CPR+Lookup verify:   {:>8.3} ms", t.combined_poly_resolver_verify.as_secs_f64() * 1000.0);
        eprintln!("  Lookup verify:       {:>8.3} ms", t.lookup_verify.as_secs_f64() * 1000.0);
        eprintln!("  PCS verify:          {:>8.3} ms", t.pcs_verify.as_secs_f64() * 1000.0);
        eprintln!("  Total:               {:>8.3} ms", t.total.as_secs_f64() * 1000.0);
        eprintln!("─────────────────────────────────────────────────────────\n");
    }

    group.bench_function("E2E/Verifier (4x-Folded Hybrid GKR c=2)", |b| {
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
                BenchBpUair, BenchQxUair, FoldedZt, FoldedLc, 32, 16, 8, UNCHECKED, _, _, _, _,
            >(
                &folded_params, &hybrid_proof, XMSS_NUM_VARS,
                |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
                qx_ideal_closure,
                &sha_public_cols,
                &bdry_public_cols,
            );
            black_box(r);
        });
    });

    let mem_snapshot = mem_tracker.stop();
    eprintln!("\n=== Peak Memory ===");
    eprintln!("  {mem_snapshot}");

    group.finish();
}

criterion_group!(benches, sha256_xmss_folded_stepwise);
criterion_main!(benches);
