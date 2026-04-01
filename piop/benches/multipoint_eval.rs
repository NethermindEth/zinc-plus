#![allow(clippy::arithmetic_side_effects)]

use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use crypto_primitives::{
    FromWithConfig, crypto_bigint_monty::MontyField, crypto_bigint_uint::Uint,
};
use rand::{Rng, rng};
use zinc_piop::multipoint_eval::MultipointEval;
use zinc_poly::mle::{DenseMultilinearExtension, MultilinearExtensionWithConfig};
use zinc_primality::MillerRabin;
use zinc_transcript::{Blake3Transcript, traits::Transcript};
use zinc_uair::ShiftSpec;

const FIELD_LIMBS: usize = 4;
type F = MontyField<FIELD_LIMBS>;

fn bench_multipoint_eval(c: &mut Criterion, num_vars: usize, num_cols: usize) {
    let mut rng = rng();
    let n = 1usize << num_vars;
    let params = format!("nvars={}/cols={}", num_vars, num_cols);

    let mut transcript = Blake3Transcript::new();
    let field_cfg = transcript.get_random_field_cfg::<F, Uint<FIELD_LIMBS>, MillerRabin>();
    let zero_inner = F::from_with_cfg(0u32, &field_cfg).into_inner();

    // Build random trace MLEs.
    let trace_mles: Vec<DenseMultilinearExtension<_>> = (0..num_cols)
        .map(|_| {
            let evals: Vec<_> = (0..n)
                .map(|_| F::from_with_cfg(rng.random::<u32>(), &field_cfg).into_inner())
                .collect();
            DenseMultilinearExtension::from_evaluations_vec(num_vars, evals, zero_inner)
        })
        .collect();

    // Random evaluation point.
    let eval_point: Vec<F> = (0..num_vars)
        .map(|_| F::from_with_cfg(rng.random::<u32>(), &field_cfg))
        .collect();

    // Compute up_evals = v_j(r').
    let up_evals: Vec<F> = trace_mles
        .iter()
        .map(|mle| {
            mle.clone()
                .evaluate_with_config(&eval_point, &field_cfg)
                .expect("up_eval evaluation failed")
        })
        .collect();

    // All columns shift by 1.
    let shifts: Vec<ShiftSpec> = (0..num_cols).map(|i| ShiftSpec::new(i, 1)).collect();

    // Compute down_evals = v_j^{down}(r') (shift by one position).
    let down_evals: Vec<F> = shifts
        .iter()
        .map(|spec| {
            let mle = &trace_mles[spec.source_col()];
            let c = spec.shift_amount();
            let mut shifted = mle.evaluations[c..].to_vec();
            shifted.extend(vec![zero_inner; c]);
            let shifted_mle =
                DenseMultilinearExtension::from_evaluations_vec(num_vars, shifted, zero_inner);
            shifted_mle
                .evaluate_with_config(&eval_point, &field_cfg)
                .expect("down_eval evaluation failed")
        })
        .collect();

    let mut group = c.benchmark_group("Multipoint eval");

    // Prepare a transcript with the seed absorbed once.
    let mut base_transcript = Blake3Transcript::new();
    base_transcript.absorb_slice(b"bench");

    // --- Bench prover ---
    group.bench_function(BenchmarkId::new("Prover", &params), |bench| {
        bench.iter_batched(
            || base_transcript.clone(),
            |mut t| {
                let _ = black_box(
                    MultipointEval::<F>::prove_as_subprotocol(
                        &mut t,
                        &trace_mles,
                        &eval_point,
                        &up_evals,
                        &down_evals,
                        &shifts,
                        &field_cfg,
                    )
                    .expect("prover failed"),
                );
            },
            BatchSize::SmallInput,
        );
    });

    // --- Bench verifier ---
    // First produce a valid proof.
    let mut prover_transcript = Blake3Transcript::new();
    prover_transcript.absorb_slice(b"bench");
    let (proof, prover_state) = MultipointEval::<F>::prove_as_subprotocol(
        &mut prover_transcript,
        &trace_mles,
        &eval_point,
        &up_evals,
        &down_evals,
        &shifts,
        &field_cfg,
    )
    .expect("prover failed");

    let open_evals: Vec<F> = trace_mles
        .iter()
        .map(|mle| {
            mle.clone()
                .evaluate_with_config(&prover_state.eval_point, &field_cfg)
                .expect("open_eval evaluation failed")
        })
        .collect();

    group.bench_function(BenchmarkId::new("Verifier", &params), |bench| {
        bench.iter_batched(
            || (base_transcript.clone(), proof.clone()),
            |(mut t, proof)| {
                let subclaim = MultipointEval::<F>::verify_as_subprotocol(
                    &mut t,
                    proof,
                    &eval_point,
                    &up_evals,
                    &down_evals,
                    &shifts,
                    num_vars,
                    &field_cfg,
                )
                .expect("verifier failed");
                MultipointEval::<F>::verify_subclaim(&subclaim, &open_evals, &shifts, &field_cfg)
                    .expect("subclaim check failed");
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

pub fn multipoint_eval_benches(c: &mut Criterion) {
    // Vary trace size with fixed 3 columns.
    for num_vars in [13, 14, 15, 16, 17] {
        bench_multipoint_eval(c, num_vars, 3);
    }

    // Vary column count at fixed size — this is the key axis for the
    // precombine optimisation: costs become constant (4 MLEs) after precombine
    // instead of scaling with J
    for num_cols in [1, 3, 10, 25, 50, 100] {
        bench_multipoint_eval(c, 14, num_cols);
    }
}

criterion_group!(benches, multipoint_eval_benches);
criterion_main!(benches);
