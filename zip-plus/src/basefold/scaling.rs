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
            BF_NCU, BF_ND, BF_NK, BF_NW, int_lane_params, read_arity8_proof, write_arity8_proof,
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
// Eval = Int<2> (the int-lane shape: 64-bit values), Cw = Int<3>,
// CombR = Int<5> (the lc-width check needs 268 bits). NUM_COLUMN_OPENINGS =
// 147 (rate 1/4 production count).
type Zt = TestZipTypes<2, 3, 5>;
type ZipLc = IprsCode<Zt, PnttConfigF65537, REP, true>;
const REP: usize = 4;
const Q: usize = 147;

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

fn run_zip(
    polys: &[DenseMultilinearExtension<Int<2>>],
    num_vars: usize,
    row_len: usize,
    cfg: &<F as PrimeField>::Config,
    point: &[F],
) -> Row {
    let m = 1usize << num_vars;
    let code = ZipLc::new_with_optimal_depth(row_len).expect("zip code");
    let pp = ZipPlus::<Zt, _>::setup(m, code);

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
    let alphas = ZipPlus::<Zt, ZipLc>::sample_alphas(&mut vt.fs_transcript, polys.len());
    ZipPlus::<Zt, _>::verify_with_alphas::<F, true>(
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
    cfg: &<F as PrimeField>::Config,
    point: &[F],
) -> Row {
    let params = int_lane_params(num_vars, REP, Q).expect("bf params");
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
    write_arity8_proof(&mut pt, &proof).unwrap();
    let open_ms = t0.elapsed().as_secs_f64() * 1e3;

    let bytes = pt.stream.get_ref().clone();
    let (raw, zs) = (bytes.len(), zstd_len(&bytes));

    let mut vt = pt.into_verification_transcript();
    vt.fs_transcript.absorb_slice(&comm.root.0);
    let t0 = Instant::now();
    let proof_v = read_arity8_proof::<F, _>(&mut vt, &params, batch, cfg).unwrap();
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

#[test]
#[ignore = "scaling study; run with --release --ignored --nocapture"]
fn scaling_study_zip_vs_basefold() {
    eprintln!(
        "scaling study: rate 1/4, Q = {Q}, 64-bit witness entries, \
         zip = F65537 IPRS (optimal depth, row_len swept), \
         zip++ = arity-8 chain over 5*2^25+1 at full depth"
    );
    let cases: &[(usize, usize)] = &[
        (12, 1),
        (14, 1),
        (16, 1),
        (18, 1),
        (20, 1),
        (12, 8),
        (14, 8),
        (16, 8),
    ];
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

        // Zip+ at its best row length. The size optimum sits well above
        // sqrt(m) because Q multiplies the column side
        // (row_len* ~ sqrt(Q * m * cw_bytes / comb_bytes)), so sweep wide.
        let half = num_vars.div_ceil(2);
        let mut best: Option<(usize, Row)> = None;
        for rl_log in half.saturating_sub(1)..=(half + 5).min(14) {
            let row_len = 1usize << rl_log;
            if !(64..=(1 << 14)).contains(&row_len) || !m.is_multiple_of(row_len) || row_len > m {
                continue;
            }
            let row = run_zip(&polys, num_vars, row_len, &cfg, &point);
            if best.as_ref().is_none_or(|(_, b)| row.raw < b.raw) {
                best = Some((row_len, row));
            }
        }
        let (best_rl, zip) = best.expect("no feasible zip row length");

        let bf = run_basefold(&witnesses, num_vars, &cfg, &point);
        assert_eq!(zip.eval, bf.eval, "cross-validation: schemes disagree");

        eprintln!(
            "m=2^{num_vars} B={batch}: zip+  raw {:8} zstd {:8} | commit {:8.1} open {:8.1} verify {:7.1} ms (row_len 2^{})",
            zip.raw,
            zip.zstd,
            zip.commit_ms,
            zip.open_ms,
            zip.verify_ms,
            best_rl.trailing_zeros(),
        );
        eprintln!(
            "            zip++ raw {:8} zstd {:8} | commit {:8.1} open {:8.1} verify {:7.1} ms (R={})",
            bf.raw,
            bf.zstd,
            bf.commit_ms,
            bf.open_ms,
            bf.verify_ms,
            num_vars / 3,
        );
    }
}
