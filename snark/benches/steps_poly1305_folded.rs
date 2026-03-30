//! Poly1305 end-to-end proving and verification benchmark using the
//! 4x-folded pipeline.
//!
//! Proves **16 chained Poly1305 iterations** (num_vars=4, 16 rows):
//!   acc_out = (acc + padded_msg) * r  mod (2^130 - 5)
//!
//! Uses `PnttConfigF2_16R4B1<2>` (BASE_LEN=1, DEPTH=2, INPUT_LEN=64)
//! with the 4x-folded pipeline (BinaryPoly<32> → 16 → 8).
//!
//! Displays proof size (raw and DEFLATE-compressed).
//!
//! Run with:
//!   cargo bench --bench steps_poly1305_folded -p zinc-snark --features "parallel simd asm qx-constraints true-ideal"

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
    crypto_bigint_uint::Uint,
    FixedSemiring,
};
use zinc_poly::mle::DenseMultilinearExtension;
use zinc_poly::univariate::binary::{
    BinaryPoly, BinaryPolyInnerProduct, BinaryPolyWideningMulByScalar,
};
use zinc_poly::univariate::dense::{DensePolyInnerProduct, DensePolynomial};
use zinc_primality::MillerRabin;
use zinc_transcript::traits::ConstTranscribable;
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
            PnttConfigF2_16R4B1,
        },
    },
    pcs::structs::{ZipPlusParams, ZipTypes},
};
use zinc_poly1305_uair::{
    Poly1305UairBp, Poly1305UairQx,
    Poly1305QxIdeal,
    NUM_COLS,
};

use zinc_uair::Uair;
use zinc_uair::ideal_collector::IdealOrZero;

use zinc_piop::lookup::{LookupColumnSpec, LookupTableType};

// ─── Type definitions ───────────────────────────────────────────────────────

const INT_LIMBS: usize = U64::LIMBS;

struct Poly1305ZipTypes<CwCoeff, const D_PLUS_ONE: usize>(PhantomData<CwCoeff>);

impl<CwCoeff, const D_PLUS_ONE: usize> ZipTypes for Poly1305ZipTypes<CwCoeff, D_PLUS_ONE>
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

/// PCS types for Poly1305 benchmark: BinaryPoly<8> codewords.
/// Uses PnttConfigF2_16R4B1<2> (BASE_LEN=1, DEPTH=2):
///   INPUT_LEN = 1 × 2^6 = 64.
/// With num_rows=1: 1 × 64 = 64 = 2^6, supporting folded_num_vars=6.
type FoldedZt = Poly1305ZipTypes<i64, 8>;
type FoldedLc = IprsCode<
    FoldedZt,
    PnttConfigF2_16R4B1<2>,
    BinaryPolyWideningMulByScalar<i64>,
    UNCHECKED,
>;

// ─── Parameters ─────────────────────────────────────────────────────────────

/// 2^4 = 16 rows = 16 chained Poly1305 iterations.
const POLY1305_NUM_VARS: usize = 4;

const POLY1305_BATCH_SIZE: usize = NUM_COLS;

const POLY1305_LOOKUP_COL_COUNT: usize = zinc_poly1305_uair::LOOKUP_COL_COUNT;

fn poly1305_lookup_specs() -> Vec<LookupColumnSpec> {
    (0..POLY1305_LOOKUP_COL_COUNT)
        .map(|i| LookupColumnSpec {
            column_index: i,
            table_type: LookupTableType::BitPoly { width: 32, chunk_width: Some(8) },
        })
        .collect()
}

fn generate_poly1305_trace(num_vars: usize) -> Vec<DenseMultilinearExtension<BinaryPoly<32>>> {
    zinc_poly1305_uair::witness::generate_poly1305_witness(num_vars)
}

// ─── Benchmark ────────────────────────────────────────────────────────────

fn poly1305_folded_stepwise(c: &mut Criterion) {
    use zinc_poly1305_uair::CyclotomicIdeal;

    let mem_tracker = MemoryTracker::start();

    let mut group = c.benchmark_group(format!(
        "Poly1305 4x-Folded Hybrid GKR c=2 Steps (grind={})",
        <FoldedZt as ZipTypes>::GRINDING_BITS,
    ));
    group.sample_size(30);

    // -- Folded PCS params -------------------------------------------
    let folded_extra_vars: usize = 2;
    let folded_num_vars = POLY1305_NUM_VARS + folded_extra_vars;
    let folded_row_len = 1usize << folded_num_vars;
    let folded_lc = FoldedLc::new(folded_row_len);
    let folded_params = ZipPlusParams::<FoldedZt, FoldedLc>::new(
        folded_num_vars, 1, folded_lc,
    );

    let lookup_specs = poly1305_lookup_specs();
    let affine_specs: Vec<zinc_piop::lookup::AffineLookupSpec> = vec![];

    // ── 1. Witness Generation ───────────────────────────────────────
    group.bench_function("WitnessGen", |b| {
        b.iter(|| {
            let trace = generate_poly1305_trace(POLY1305_NUM_VARS);
            black_box(trace);
        });
    });

    // Pre-generate the trace for subsequent steps.
    let trace = generate_poly1305_trace(POLY1305_NUM_VARS);
    assert_eq!(trace.len(), POLY1305_BATCH_SIZE);

    eprintln!(
        "\n── Poly1305 Trace Info ──────────────────────────────────",
    );
    eprintln!(
        "  num_vars={}, rows={}, iterations={}, columns={}",
        POLY1305_NUM_VARS,
        1usize << POLY1305_NUM_VARS,
        1usize << POLY1305_NUM_VARS,
        POLY1305_BATCH_SIZE,
    );
    eprintln!(
        "  Each iteration: acc = (acc + padded_msg) * r mod (2^130 - 5)",
    );
    eprintln!(
        "  Message data: {} bytes ({} × 16-byte blocks)",
        (1usize << POLY1305_NUM_VARS) * 16,
        1usize << POLY1305_NUM_VARS,
    );
    eprintln!("─────────────────────────────────────────────────────────\n");

    // Shared: public columns for verifier
    let bp_sig = Poly1305UairBp::signature();
    let public_cols: Vec<_> = bp_sig.public_columns.iter()
        .map(|&i| trace[i].clone()).collect();

    let bdry_public_cols: Vec<DenseMultilinearExtension<BinaryPoly<32>>> = vec![];

    // ── 2. E2E Total Prover ─────────────────────────────────────────
    // Warmup run.
    let _warmup_proof = zinc_snark::pipeline::prove_hybrid_gkr_logup_4x_folded::<
        Poly1305UairBp, Poly1305UairQx, FoldedZt, FoldedLc, 32, 16, 8, UNCHECKED,
    >(
        &folded_params, &trace, POLY1305_NUM_VARS,
        &lookup_specs, &affine_specs, 2,
    );

    let hybrid_proof = zinc_snark::pipeline::prove_hybrid_gkr_logup_4x_folded::<
        Poly1305UairBp, Poly1305UairQx, FoldedZt, FoldedLc, 32, 16, 8, UNCHECKED,
    >(
        &folded_params, &trace, POLY1305_NUM_VARS,
        &lookup_specs, &affine_specs, 2,
    );

    // Print prover pipeline timing breakdown.
    {
        let t = &hybrid_proof.timing;
        eprintln!("\n── Poly1305 4x-Folded Hybrid GKR c=2 Prover Pipeline Timing ──");
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

    // ── Proof size (raw + compressed) ───────────────────────────────
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

        // Gather all proof bytes for compression.
        let mut all_bytes = Vec::with_capacity(total_raw);
        all_bytes.extend(&hybrid_proof.pcs_proof_bytes);
        for v in &hybrid_proof.ic_proof_values { all_bytes.extend(v); }
        for v in &hybrid_proof.qx_ic_proof_values { all_bytes.extend(v); }
        for v in &hybrid_proof.cpr_sumcheck_messages { all_bytes.extend(v); }
        all_bytes.extend(&hybrid_proof.cpr_sumcheck_claimed_sum);
        for v in &hybrid_proof.cpr_up_evals { all_bytes.extend(v); }
        for v in &hybrid_proof.cpr_down_evals { all_bytes.extend(v); }
        for v in &hybrid_proof.qx_cpr_sumcheck_messages { all_bytes.extend(v); }
        all_bytes.extend(&hybrid_proof.qx_cpr_sumcheck_claimed_sum);
        for v in &hybrid_proof.qx_cpr_up_evals { all_bytes.extend(v); }
        for v in &hybrid_proof.qx_cpr_down_evals { all_bytes.extend(v); }
        if let Some(sc) = &hybrid_proof.shift_sumcheck {
            for v in &sc.rounds { all_bytes.extend(v); }
            for v in &sc.v_finals { all_bytes.extend(v); }
        }
        if let Some(sc) = &hybrid_proof.qx_shift_sumcheck {
            for v in &sc.rounds { all_bytes.extend(v); }
            for v in &sc.v_finals { all_bytes.extend(v); }
        }
        for v in &hybrid_proof.evaluation_point_bytes { all_bytes.extend(v); }
        for v in &hybrid_proof.pcs_evals_bytes { all_bytes.extend(v); }
        for v in &hybrid_proof.folding_c1s_bytes { all_bytes.extend(v); }
        for v in &hybrid_proof.folding_c2s_bytes { all_bytes.extend(v); }
        for v in &hybrid_proof.folding_c3s_bytes { all_bytes.extend(v); }
        for v in &hybrid_proof.folding_c4s_bytes { all_bytes.extend(v); }
        for v in &hybrid_proof.ring_evals_bytes { all_bytes.extend(v); }

        let compressed = {
            use std::io::Write;
            let mut encoder = flate2::write::DeflateEncoder::new(
                Vec::new(), flate2::Compression::default(),
            );
            encoder.write_all(&all_bytes).unwrap();
            encoder.finish().unwrap()
        };

        let ring_evals_bytes: usize = hybrid_proof.ring_evals_bytes.iter().map(|v| v.len()).sum();
        let total_with_ring = total_raw + ring_evals_bytes;

        eprintln!("\n=== Poly1305 4x-Folded Hybrid GKR c=2 Proof Size ===");
        eprintln!("  PCS:              {:>6} B  ({:.1} KB)", pcs_bytes, pcs_bytes as f64 / 1024.0);
        eprintln!("  IC (Bp):          {:>6} B", ic_bytes);
        eprintln!("  IC (Qx):          {:>6} B", qx_ic_bytes);
        let cpr_total = cpr_msg_bytes + cpr_sum_bytes + cpr_up + cpr_dn;
        let qx_cpr_total = qx_cpr_msg + qx_cpr_sum + qx_cpr_up + qx_cpr_dn;
        eprintln!("  CPR (Bp):         {:>6} B  (msg={cpr_msg_bytes} sum={cpr_sum_bytes} up={cpr_up} dn={cpr_dn})", cpr_total);
        eprintln!("  CPR (Qx):         {:>6} B  (msg={qx_cpr_msg} sum={qx_cpr_sum} up={qx_cpr_up} dn={qx_cpr_dn})", qx_cpr_total);
        eprintln!("  Lookup:           {:>6} B", lookup_bytes);
        eprintln!("  Shift SC (Bp):    {:>6} B", shift_sc_bytes);
        eprintln!("  Shift SC (Qx):    {:>6} B", qx_shift_sc_bytes);
        eprintln!("  Eval point:       {:>6} B", eval_pt_bytes);
        eprintln!("  PCS evals:        {:>6} B", pcs_eval_bytes);
        eprintln!("  Folding:          {:>6} B  (c1={fold_c1} c2={fold_c2} c3={fold_c3} c4={fold_c4})", folding_total);
        eprintln!("  Ring evals:       {:>6} B", ring_evals_bytes);
        eprintln!("  ─────────────────────────────────────────────────────");
        eprintln!("  PIOP total:       {:>6} B  ({:.1} KB)", piop_total, piop_total as f64 / 1024.0);
        eprintln!("  Total raw:        {:>6} B  ({:.1} KB)", total_raw, total_raw as f64 / 1024.0);
        eprintln!("  Total+ring:       {:>6} B  ({:.1} KB)", total_with_ring, total_with_ring as f64 / 1024.0);
        eprintln!("  Compressed:       {:>6} B  ({:.1} KB, {:.1}x ratio)",
            compressed.len(), compressed.len() as f64 / 1024.0,
            all_bytes.len() as f64 / compressed.len() as f64);
        eprintln!("═══════════════════════════════════════════════════════\n");
    }

    group.bench_function("E2E/Prover (4x-Folded Hybrid GKR c=2)", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let trace = generate_poly1305_trace(POLY1305_NUM_VARS);
                let t = Instant::now();
                let _ = zinc_snark::pipeline::prove_hybrid_gkr_logup_4x_folded::<
                    Poly1305UairBp, Poly1305UairQx, FoldedZt, FoldedLc, 32, 16, 8, UNCHECKED,
                >(
                    &folded_params, &trace, POLY1305_NUM_VARS,
                    &lookup_specs, &affine_specs, 2,
                );
                total += t.elapsed();
            }
            total
        });
    });

    // ── 3. E2E Total Verifier ───────────────────────────────────────
    {
        let r = zinc_snark::pipeline::verify_classic_logup_4x_folded::<
            Poly1305UairBp, Poly1305UairQx, FoldedZt, FoldedLc, 32, 16, 8, UNCHECKED, _, _, _, _,
        >(
            &folded_params, &hybrid_proof, POLY1305_NUM_VARS,
            |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
            |_: &IdealOrZero<Poly1305QxIdeal>| zinc_snark::pipeline::TrivialIdeal,
            &public_cols,
            &bdry_public_cols,
        );
        assert!(r.accepted, "Verifier rejected the proof");
        let t = &r.timing;
        eprintln!("\n── Poly1305 Verifier step timing ────────────────────────");
        eprintln!("  IC verify:           {:>8.3} ms", t.ideal_check_verify.as_secs_f64() * 1000.0);
        eprintln!("  CPR+Lookup verify:   {:>8.3} ms", t.combined_poly_resolver_verify.as_secs_f64() * 1000.0);
        eprintln!("  Lookup verify:       {:>8.3} ms", t.lookup_verify.as_secs_f64() * 1000.0);
        eprintln!("  PCS verify:          {:>8.3} ms", t.pcs_verify.as_secs_f64() * 1000.0);
        eprintln!("  Total:               {:>8.3} ms", t.total.as_secs_f64() * 1000.0);
        eprintln!("─────────────────────────────────────────────────────────\n");
    }

    group.bench_function("E2E/Verifier (4x-Folded Hybrid GKR c=2)", |b| {
        b.iter(|| {
            let r = zinc_snark::pipeline::verify_classic_logup_4x_folded::<
                Poly1305UairBp, Poly1305UairQx, FoldedZt, FoldedLc, 32, 16, 8, UNCHECKED, _, _, _, _,
            >(
                &folded_params, &hybrid_proof, POLY1305_NUM_VARS,
                |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
                |_: &IdealOrZero<Poly1305QxIdeal>| zinc_snark::pipeline::TrivialIdeal,
                &public_cols,
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

criterion_group!(benches, poly1305_folded_stepwise);
criterion_main!(benches);
