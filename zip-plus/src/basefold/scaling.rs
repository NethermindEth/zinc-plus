//! Scaling study: Zip+ (Brakedown-style) vs Zip++ (basefold arity-8) at the
//! PCS layer, on identical witnesses, points, and field configs.
//!
//! Run with:
//! `cargo test -p zip-plus --release scaling_study -- --ignored --nocapture`

#![cfg(test)]
#![allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::unwrap_used
)]

use crate::{
    basefold::{
        arity8::{commit_batch, prove_batch, verify_batch},
        protocol_glue::{
            BF_NCU, BF_ND, BF_NK, BF_NW, read_arity8_proof, write_arity8_proof,
        },
    },
    code::iprs::{IprsCode, PnttConfigF65537},
    pcs::{structs::ZipPlus, test_utils::TestZipTypes},
    pcs_transcript::PcsProverTranscript,
};
use crypto_primitives::{PrimeField, crypto_bigint_int::Int, crypto_bigint_monty::MontyField};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::time::Instant;
use zinc_poly::mle::DenseMultilinearExtension;
use zinc_primality::MillerRabin;
use zinc_transcript::{Blake3Transcript, traits::Transcript};

type F = MontyField<4>;

/// Rate-1/8 Zip+ instantiation: production query count 100.
#[derive(Debug, Clone)]
struct StudyZt8 {}
impl crate::pcs::structs::ZipTypes for StudyZt8 {
    const NUM_COLUMN_OPENINGS: usize = 100;
    type Eval = Int<2>;
    type Cw = Int<3>;
    type Fmod = crypto_primitives::crypto_bigint_uint::Uint<3>;
    type PrimeTest = MillerRabin;
    type Chal = Int<2>;
    type Pt = Int<2>;
    type CombR = Int<5>;
    type Comb = Self::CombR;
    type EvalDotChal = zinc_utils::inner_product::ScalarProduct;
    type CombDotChal = zinc_utils::inner_product::ScalarProduct;
    type ArrCombRDotChal = zinc_utils::inner_product::MBSInnerProduct;
}
// Eval = Int<2> (the int-lane shape: 64-bit values), Cw = Int<3>,
// CombR = Int<5> (the lc-width check needs 268 bits). NUM_COLUMN_OPENINGS =
// 147 (rate 1/4 production count).
type Zt = TestZipTypes<2, 3, 5>;

fn zstd_len(bytes: &[u8]) -> usize {
    zstd::stream::encode_all(bytes, 3).unwrap().len()
}

fn random_witness(rng: &mut StdRng, len: usize) -> Vec<Int<2>> {
    (0..len)
        .map(|_| {
            let mut words = [0u64; 2];
            words[0] = rng.random();
            let v = Int::<2>::new(crypto_bigint::Int::from_words(words));
            if rng.random::<bool>() { -v } else { v }
        })
        .collect()
}

/// Shared field config and evaluation point, independent of either scheme's
/// transcript (so both prove the *same* claim and must return equal evals).
fn shared_cfg_point(num_vars: usize) -> (<F as PrimeField>::Config, Vec<F>) {
    let mut t = Blake3Transcript::new();
    t.absorb_slice(b"scaling-study");
    let cfg = t.get_random_field_cfg::<F, crypto_primitives::crypto_bigint_uint::Uint<4>, MillerRabin>();
    let point: Vec<F> = t.get_field_challenges(num_vars, &cfg);
    (cfg, point)
}

struct Row {
    raw: usize,
    zstd: usize,
    commit_ms: f64,
    open_ms: f64,
    verify_ms: f64,
    eval: F,
}

fn run_zip<ZtX, const REPX: usize>(
    polys: &[DenseMultilinearExtension<Int<2>>],
    num_vars: usize,
    row_len: usize,
    cfg: &<F as PrimeField>::Config,
    point: &[F],
) -> Row
where
    ZtX: crate::pcs::structs::ZipTypes<
            Eval = Int<2>,
            Cw = Int<3>,
            Chal = Int<2>,
            CombR = Int<5>,
            Comb = Int<5>,
        >,
    <F as crypto_primitives::Field>::Modulus: zinc_utils::from_ref::FromRef<ZtX::Fmod>,
{
    let m = 1usize << num_vars;
    let code = IprsCode::<ZtX, PnttConfigF65537, REPX, true>::new_with_optimal_depth(row_len)
        .expect("zip code");
    let pp = ZipPlus::<ZtX, _>::setup(m, code);

    let t0 = Instant::now();
    let (hint, comm) = ZipPlus::commit(&pp, polys).expect("zip commit");
    let commit_ms = t0.elapsed().as_secs_f64() * 1e3;

    let mut pt = PcsProverTranscript::new_from_commitment(&comm);
    let t0 = Instant::now();
    let eval = ZipPlus::prove_f::<F, true>(&mut pt, &pp, polys, point, &hint, cfg)
        .expect("zip prove");
    let open_ms = t0.elapsed().as_secs_f64() * 1e3;

    let bytes = pt.stream.get_ref().clone();
    let (raw, zs) = (bytes.len(), zstd_len(&bytes));

    let mut vt = pt.into_verification_transcript();
    vt.fs_transcript.absorb_slice(&comm.root.0);
    let t0 = Instant::now();
    let alphas = ZipPlus::<ZtX, IprsCode<ZtX, PnttConfigF65537, REPX, true>>::sample_alphas(
        &mut vt.fs_transcript,
        polys.len(),
    );
    ZipPlus::<ZtX, _>::verify_with_alphas::<F, true>(
        &mut vt, &pp, &comm, cfg, point, &eval, &alphas,
    )
    .expect("zip verify");
    let verify_ms = t0.elapsed().as_secs_f64() * 1e3;

    Row {
        raw,
        zstd: zs,
        commit_ms,
        open_ms,
        verify_ms,
        eval,
    }
}

fn run_basefold(
    witnesses: &[Vec<Int<BF_ND>>],
    num_vars: usize,
    rep: usize,
    q: usize,
    num_rounds: usize,
    cfg: &<F as PrimeField>::Config,
    point: &[F],
) -> Row {
    use crate::basefold::{arity8::Arity8Params, chain8::Radix8Chain, protocol_glue::BF_P};
    let chain = crate::basefold::chain::ChainConfigF167772161;
    let _ = chain;
    // Build the chain at FULL depth regardless of how many rounds fold:
    // the tail then encodes through butterflies (O(n log n)) instead of a
    // dense base case (m*n/8^R), making shallow-R configs viable.
    let full_levels = (num_vars / 3).max(1);
    let chain =
        Radix8Chain::<crate::basefold::chain::ChainConfigF167772161>::new(
            1usize << num_vars,
            rep,
            full_levels,
        )
        .expect("bf chain");
    let params = Arity8Params::new(chain, BF_P, num_rounds, q).expect("bf params");
    let batch = witnesses.len();

    let t0 = Instant::now();
    let (comm, hint) = commit_batch::<_, BF_ND, BF_NK, true>(&params, witnesses).unwrap();
    let commit_ms = t0.elapsed().as_secs_f64() * 1e3;

    let mut pt = PcsProverTranscript::new_from_commitments(std::iter::empty());
    pt.fs_transcript.absorb_slice(&comm.root.0);
    let weights: Vec<F> = (0..batch).map(|_| F::one_with_cfg(cfg)).collect();
    let t0 = Instant::now();
    let (proof, eval) = prove_batch::<_, F, BF_ND, BF_NK, BF_NW, BF_NCU, true>(
        &params,
        witnesses,
        &weights,
        None,
        &hint,
        point,
        cfg,
        &mut pt.fs_transcript,
    )
    .unwrap();
    write_arity8_proof(&mut pt, &params, 64, &proof).unwrap();
    let open_ms = t0.elapsed().as_secs_f64() * 1e3;

    let bytes = pt.stream.get_ref().clone();
    let (raw, zs) = (bytes.len(), zstd_len(&bytes));

    let mut vt = pt.into_verification_transcript();
    vt.fs_transcript.absorb_slice(&comm.root.0);
    let t0 = Instant::now();
    let proof_v = read_arity8_proof::<F, _>(&mut vt, &params, batch, 64, cfg).unwrap();
    verify_batch::<_, F, BF_ND, BF_NK, BF_NW, BF_NCU, true>(
        &params,
        &comm,
        &proof_v,
        &weights,
        None,
        point,
        &eval,
        cfg,
        &mut vt.fs_transcript,
    )
    .unwrap();
    let verify_ms = t0.elapsed().as_secs_f64() * 1e3;

    Row {
        raw,
        zstd: zs,
        commit_ms,
        open_ms,
        verify_ms,
        eval,
    }
}

/// Commit once (R-independent with the full-depth chain), then sweep the
/// fold-round count and return the zstd-smallest configuration.
fn run_basefold_sweep(
    witnesses: &[Vec<Int<BF_ND>>],
    num_vars: usize,
    rep: usize,
    q: usize,
    r_lo: usize,
    r_hi: usize,
    cfg: &<F as PrimeField>::Config,
    point: &[F],
) -> (usize, Row, String) {
    use crate::basefold::{arity8::Arity8Params, chain8::Radix8Chain, protocol_glue::BF_P};
    let full_levels = (num_vars / 3).max(1);
    let chain = Radix8Chain::<crate::basefold::chain::ChainConfigF167772161>::new(
        1usize << num_vars,
        rep,
        full_levels,
    )
    .expect("bf chain");
    let mut params = Arity8Params::new(chain, BF_P, 1, q).expect("bf params");
    let batch = witnesses.len();
    let weights: Vec<F> = (0..batch).map(|_| F::one_with_cfg(cfg)).collect();

    let t0 = Instant::now();
    let (comm, hint) = commit_batch::<_, BF_ND, BF_NK, true>(&params, witnesses).unwrap();
    let commit_ms = t0.elapsed().as_secs_f64() * 1e3;

    let mut best: Option<(usize, Row)> = None;
    for r in r_lo.max(1)..=r_hi.min(full_levels) {
        params.num_rounds = r;

        let mut pt = PcsProverTranscript::new_from_commitments(std::iter::empty());
        pt.fs_transcript.absorb_slice(&comm.root.0);
        let t0 = Instant::now();
        let (proof, eval) = prove_batch::<_, F, BF_ND, BF_NK, BF_NW, BF_NCU, true>(
            &params,
            witnesses,
            &weights,
            None,
            &hint,
            point,
            cfg,
            &mut pt.fs_transcript,
        )
        .unwrap();
        write_arity8_proof(&mut pt, &params, 64, &proof).unwrap();
        let open_ms = t0.elapsed().as_secs_f64() * 1e3;

        let bytes = pt.stream.get_ref().clone();
        let (raw, zs) = (bytes.len(), zstd_len(&bytes));

        let mut vt = pt.into_verification_transcript();
        vt.fs_transcript.absorb_slice(&comm.root.0);
        let t0 = Instant::now();
        let proof_v = read_arity8_proof::<F, _>(&mut vt, &params, batch, 64, cfg).unwrap();
        verify_batch::<_, F, BF_ND, BF_NK, BF_NW, BF_NCU, true>(
            &params,
            &comm,
            &proof_v,
            &weights,
            None,
            point,
            &eval,
            cfg,
            &mut vt.fs_transcript,
        )
        .unwrap();
        let verify_ms = t0.elapsed().as_secs_f64() * 1e3;

        let row = Row {
            raw,
            zstd: zs,
            commit_ms,
            open_ms,
            verify_ms,
            eval,
        };
        if best.as_ref().is_none_or(|(_, b)| row.zstd < b.zstd) {
            best = Some((r, row));
        }
    }
    let (best_r, row) = best.expect("no feasible basefold config");
    params.num_rounds = best_r;
    (best_r, row, profile_string(&params, batch, q))
}

/// Raw-byte budget of the proof stream by component, for the best config.
fn profile_string(
    params: &crate::basefold::arity8::Arity8Params<
        crate::basefold::chain::ChainConfigF167772161,
    >,
    batch: usize,
    q: usize,
) -> String {
    use crate::basefold::{limbs::next_limb_count_arity, protocol_glue::tight_widths};
    let r_max = params.num_rounds;
    let w = tight_widths(params, 64);
    let mut ks = vec![batch];
    for _ in 0..r_max {
        ks.push(next_limb_count_arity(128, params.p, *ks.last().unwrap(), 3));
    }
    let fb = 32usize; // F::Inner bytes per claim
    let claims = fb
        * (batch
            + (1..=r_max).map(|r| ks[r - 1] * 8).sum::<usize>()
            + ks[1..r_max].iter().sum::<usize>());
    let roots = (r_max - 1) * 32;
    let tail = ks[r_max] * params.chain.msg_len_at(r_max) * w.tail;
    let leaves = q
        * (0..r_max)
            .map(|r| ks[r] * 8 * w.leaf[r])
            .sum::<usize>();
    let paths = q
        * (0..r_max)
            .map(|r| {
                4 + 32 * params.chain.codeword_len_at(r + 1).trailing_zeros() as usize
            })
            .sum::<usize>();
    format!(
        "claims {claims} roots {roots} tail {tail} leaves {leaves} paths {paths} \
         (leaf B/round: {:?})",
        w.leaf
    )
}

/// Consolidation study: the Zip++ analogue of "N-x Folded Zip". A fixed
/// amount of witness data is presented as B columns of T/B entries each;
/// consolidating means fewer, longer columns (smaller B). For the int lane
/// (all-ones weights) consolidation needs no machinery at all: the
/// column-index bits fold as plain MLE variables at 1/2-coordinates (see
/// `arity8::tests::consolidation_all_ones_is_point_extension`), so B=1 is
/// the fully consolidated shape of the same data.
#[test]
#[ignore = "consolidation study; run with --release --ignored --nocapture"]
fn consolidation_study() {
    eprintln!(
        "consolidation study: fixed total data T, B columns of m = T/B 64-bit entries.\n\
         both schemes at rate 1/8, Q=100 (the winning dial everywhere in the main study);\n\
         zip+ row_len swept, zip++ fold rounds R swept (one commit per config)."
    );
    for log_t in [19usize, 21] {
        eprintln!("--- total data 2^{log_t} ---");
        for log_b in [5usize, 4, 3, 2, 1, 0] {
            let batch = 1usize << log_b;
            let num_vars = log_t - log_b;
            let m = 1usize << num_vars;
            let mut rng = StdRng::seed_from_u64(7 + log_t as u64);
            let polys: Vec<DenseMultilinearExtension<Int<2>>> = (0..batch)
                .map(|_| DenseMultilinearExtension {
                    num_vars,
                    evaluations: random_witness(&mut rng, m),
                })
                .collect();
            let witnesses: Vec<Vec<Int<BF_ND>>> = polys
                .iter()
                .map(|p| p.evaluations.iter().map(|x| x.resize::<BF_ND>()).collect())
                .collect();
            let (cfg, point) = shared_cfg_point(num_vars);

            // Zip+ reference at the same (B, m) shape.
            let half = num_vars.div_ceil(2);
            let mut zip_best: Option<(usize, Row)> = None;
            for rl_log in half.saturating_sub(1)..=(half + 5).min(13) {
                let row_len = 1usize << rl_log;
                if !(64..=(1 << 13)).contains(&row_len)
                    || !m.is_multiple_of(row_len)
                    || row_len > m
                {
                    continue;
                }
                let row = run_zip::<StudyZt8, 8>(&polys, num_vars, row_len, &cfg, &point);
                if zip_best.as_ref().is_none_or(|(_, b)| row.zstd < b.zstd) {
                    zip_best = Some((rl_log, row));
                }
            }
            let (zip_rl, zip) = zip_best.expect("no feasible zip config");

            // Zip++ R window around the known optimum (R ~ (nv - 9)/3).
            let r_lo = num_vars.saturating_sub(12) / 3;
            let r_hi = (num_vars / 4).max(2);
            let (bf_r, bf, bf_profile) =
                run_basefold_sweep(&witnesses, num_vars, 8, 100, r_lo, r_hi, &cfg, &point);
            assert_eq!(bf.eval, zip.eval, "cross-validation: schemes disagree");

            eprintln!(
                "B={batch:2} m=2^{num_vars}: zip+  raw {:8} zstd {:8} | commit {:8.1} open {:8.1} verify {:7.1} ms (rl=2^{zip_rl})",
                zip.raw, zip.zstd, zip.commit_ms, zip.open_ms, zip.verify_ms,
            );
            eprintln!(
                "              zip++ raw {:8} zstd {:8} | commit {:8.1} open {:8.1} verify {:7.1} ms (R={bf_r})",
                bf.raw, bf.zstd, bf.commit_ms, bf.open_ms, bf.verify_ms,
            );
            eprintln!("              zip++ bytes: {bf_profile}");
        }
    }
}

#[test]
#[ignore = "scaling study; run with --release --ignored --nocapture"]
fn scaling_study_zip_vs_basefold() {
    eprintln!(
        "scaling study (best config vs best config), 64-bit witness entries.\n\
         zip+  : F65537 IPRS, optimal depth, row_len swept; rate 1/4 (Q=147) and 1/8 (Q=100)\n\
         zip++ : arity-8 chain over 5*2^25+1; R swept; rate 1/4 (Q=147) and 1/8 (Q=100)"
    );
    let cases: &[(usize, usize)] = &[(12, 1), (14, 1), (16, 1), (18, 1), (20, 1), (16, 8)];
    for &(num_vars, batch) in cases {
        let m = 1usize << num_vars;
        let mut rng = StdRng::seed_from_u64(7 + num_vars as u64);
        let polys: Vec<DenseMultilinearExtension<Int<2>>> = (0..batch)
            .map(|_| DenseMultilinearExtension {
                num_vars,
                evaluations: random_witness(&mut rng, m),
            })
            .collect();
        let witnesses: Vec<Vec<Int<BF_ND>>> = polys
            .iter()
            .map(|p| p.evaluations.iter().map(|x| x.resize::<BF_ND>()).collect())
            .collect();
        let (cfg, point) = shared_cfg_point(num_vars);

        // ---- Zip+ at its best (row_len x rate). ----
        let half = num_vars.div_ceil(2);
        let mut zip_best: Option<(String, Row)> = None;
        for rl_log in half.saturating_sub(1)..=(half + 5).min(14) {
            let row_len = 1usize << rl_log;
            if !(64..=(1 << 14)).contains(&row_len) || !m.is_multiple_of(row_len) || row_len > m
            {
                continue;
            }
            let row = run_zip::<Zt, 4>(&polys, num_vars, row_len, &cfg, &point);
            let tag = format!("rate 1/4 Q=147 rl=2^{rl_log}");
            if zip_best.as_ref().is_none_or(|(_, b)| row.zstd < b.zstd) {
                zip_best = Some((tag, row));
            }
            // Rate 1/8: codeword_len = row_len * 8 must stay under 65537.
            if row_len <= (1 << 13) {
                let row = run_zip::<StudyZt8, 8>(&polys, num_vars, row_len, &cfg, &point);
                let tag = format!("rate 1/8 Q=100 rl=2^{rl_log}");
                if zip_best.as_ref().is_none_or(|(_, b)| row.zstd < b.zstd) {
                    zip_best = Some((tag, row));
                }
            }
        }
        let (zip_tag, zip) = zip_best.expect("no feasible zip config");

        // ---- Zip++ at its best (R x rate). Lower-bound R so the dense
        // base case stays tractable (m * n / 8^R <= ~2^30). ----
        let mut bf_best: Option<(String, Row)> = None;
        for &(rep, q) in &[(4usize, 147usize), (8, 100)] {
            let full = (num_vars / 3).max(1);
            for r in 1..=full {
                let row = run_basefold(&witnesses, num_vars, rep, q, r, &cfg, &point);
                assert_eq!(row.eval, zip.eval, "cross-validation: schemes disagree");
                let tag = format!("rate 1/{rep} Q={q} R={r}");
                if bf_best.as_ref().is_none_or(|(_, b)| row.zstd < b.zstd) {
                    bf_best = Some((tag, row));
                }
            }
        }
        let (bf_tag, bf) = bf_best.expect("no feasible basefold config");

        eprintln!(
            "m=2^{num_vars} B={batch}: zip+  raw {:8} zstd {:8} | commit {:8.1} open {:8.1} verify {:7.1} ms ({zip_tag})",
            zip.raw, zip.zstd, zip.commit_ms, zip.open_ms, zip.verify_ms,
        );
        eprintln!(
            "            zip++ raw {:8} zstd {:8} | commit {:8.1} open {:8.1} verify {:7.1} ms ({bf_tag})",
            bf.raw, bf.zstd, bf.commit_ms, bf.open_ms, bf.verify_ms,
        );
    }
}
