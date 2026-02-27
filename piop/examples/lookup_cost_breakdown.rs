//! Cost breakdown of the batched decomposition + LogUp prover.
//!
//! Run with:
//!   cargo run -p zinc-piop --example lookup_cost_breakdown --features parallel --release

#![allow(clippy::arithmetic_side_effects)]

use crypto_bigint::{U192, const_monty_params};
use crypto_primitives::{
    FromPrimitiveWithConfig, PrimeField, crypto_bigint_const_monty::ConstMontyField,
};
use num_traits::Zero;
use std::time::Instant;
use zinc_piop::lookup::{
    BatchedDecompLogupProtocol, BatchedDecompLookupInstance,
    tables::{
        batch_inverse_shifted, bitpoly_shift, build_table_index,
        compute_multiplicities_with_index, generate_bitpoly_table,
    },
};
use zinc_poly::{
    mle::DenseMultilinearExtension,
    utils::{build_eq_x_r_inner, build_eq_x_r_vec},
};
use zinc_transcript::{KeccakTranscript, traits::Transcript};
use zinc_utils::inner_transparent_field::InnerTransparentField;

const_monty_params!(BenchModulus, U192, "fffffffffffffffffffffffffffffffeffffffffffffffff");
type F = ConstMontyField<BenchModulus, 3>;

fn main() {
    let chunk_width: usize = 8;
    let num_chunks: usize = 4;
    let num_lookups: usize = 10;
    let witness_size: usize = 1 << 10;
    let field_cfg = &();

    println!("=== Batched LogUp Prover Cost Breakdown ===");
    println!(
        "  chunk_width={}, num_chunks={}, L={}, witness_size={}",
        chunk_width, num_chunks, num_lookups, witness_size
    );
    println!();

    // --- Setup: build instance ---
    let a = F::from(3u32);
    let subtable = generate_bitpoly_table(chunk_width, &a, field_cfg);
    let shifts: Vec<F> = (0..num_chunks)
        .map(|k| bitpoly_shift(k * chunk_width, &a))
        .collect();

    let mut rng = rand::rng();
    let mut witnesses = Vec::with_capacity(num_lookups);
    let mut all_chunks = Vec::with_capacity(num_lookups);
    for _ in 0..num_lookups {
        let chunks: Vec<Vec<F>> = (0..num_chunks)
            .map(|_| {
                (0..witness_size)
                    .map(|_| subtable[rand::Rng::random_range(&mut rng, 0..subtable.len())])
                    .collect()
            })
            .collect();
        let witness: Vec<F> = (0..witness_size)
            .map(|i| {
                let mut val = F::from(0u32);
                for k in 0..num_chunks {
                    val += &(shifts[k] * &chunks[k][i]);
                }
                val
            })
            .collect();
        witnesses.push(witness);
        all_chunks.push(chunks);
    }

    let instance = BatchedDecompLookupInstance {
        witnesses: witnesses.clone(),
        subtable: subtable.clone(),
        shifts: shifts.clone(),
        chunks: all_chunks.clone(),
    };

    // --- Warmup ---
    {
        let mut t = KeccakTranscript::new();
        let _ = BatchedDecompLogupProtocol::<F>::prove_as_subprotocol(&mut t, &instance, field_cfg);
    }

    // --- Full end-to-end timing ---
    let e2e_start = Instant::now();
    {
        let mut t = KeccakTranscript::new();
        let _ = BatchedDecompLogupProtocol::<F>::prove_as_subprotocol(&mut t, &instance, field_cfg)
            .expect("prover failed");
    }
    let e2e_elapsed = e2e_start.elapsed();
    println!("  End-to-end prover:        {:>10.3} ms", e2e_elapsed.as_secs_f64() * 1000.0);
    println!();

    // --- Step-by-step profiling (mirrors prove_as_subprotocol) ---
    let zero = F::from(0u32);
    let one = F::from(1u32);
    let mut buf = vec![0u8; 24]; // 3 limbs × 8 bytes = 24 bytes for Uint<3>
    let mut transcript = KeccakTranscript::new();

    // Step 1: Absorb chunks
    let t0 = Instant::now();
    for lookup_chunks in &all_chunks {
        for chunk in lookup_chunks {
            transcript.absorb_random_field_slice(chunk, &mut buf);
        }
    }
    let step1 = t0.elapsed();

    // Step 2: Multiplicities
    let t0 = Instant::now();
    let table_index = build_table_index(&subtable);
    let all_chunk_multiplicities: Vec<Vec<Vec<F>>> = all_chunks
        .iter()
        .map(|lookup_chunks| {
            lookup_chunks
                .iter()
                .map(|chunk| {
                    compute_multiplicities_with_index(chunk, &table_index, subtable.len(), field_cfg)
                        .unwrap()
                })
                .collect()
        })
        .collect();

    let all_aggregated_multiplicities: Vec<Vec<F>> = all_chunk_multiplicities
        .iter()
        .map(|lookup_mults| {
            let mut agg = vec![zero; subtable.len()];
            for m in lookup_mults {
                for (a, mk) in agg.iter_mut().zip(m.iter()) {
                    *a += mk;
                }
            }
            agg
        })
        .collect();

    for agg in &all_aggregated_multiplicities {
        transcript.absorb_random_field_slice(agg, &mut buf);
    }
    let step2 = t0.elapsed();

    // Step 3: β challenge
    let t0 = Instant::now();
    let beta: F = transcript.get_field_challenge(field_cfg);
    let step3 = t0.elapsed();

    // Step 4: Inverse vectors
    let t0 = Instant::now();
    let all_inverse_witnesses: Vec<Vec<Vec<F>>> = all_chunks
        .iter()
        .map(|lookup_chunks| {
            lookup_chunks
                .iter()
                .map(|chunk| batch_inverse_shifted(&beta, chunk))
                .collect()
        })
        .collect();
    let v_table = batch_inverse_shifted(&beta, &subtable);

    for lookup_invs in &all_inverse_witnesses {
        for u in lookup_invs {
            transcript.absorb_random_field_slice(u, &mut buf);
        }
    }
    transcript.absorb_random_field_slice(&v_table, &mut buf);
    let step4 = t0.elapsed();

    // Step 5: γ challenge
    let t0 = Instant::now();
    let gamma: F = transcript.get_field_challenge(field_cfg);
    let step5 = t0.elapsed();

    // Step 6: γ powers
    let t0 = Instant::now();
    let num_identities = num_lookups * (num_chunks + 1);
    let mut gamma_powers = Vec::with_capacity(num_identities);
    let mut gp = one;
    for _ in 0..num_identities {
        gamma_powers.push(gp);
        gp *= &gamma;
    }
    let step6 = t0.elapsed();

    // Step 7: Precompute H
    let t0 = Instant::now();
    let witness_len = witness_size;
    let w_num_vars = zinc_utils::log2(witness_len.next_power_of_two()) as usize;
    let t_num_vars = zinc_utils::log2(subtable.len().next_power_of_two()) as usize;
    let num_vars = w_num_vars.max(t_num_vars);
    let n = 1usize << num_vars;
    let subtable_len = subtable.len();

    let h_evaluations: Vec<F> = (0..n)
        .map(|j| {
            let mut acc = zero;
            let v_j = if j < subtable_len { &v_table[j] } else { &zero };

            for ell in 0..num_lookups {
                let base_id = ell * (num_chunks + 1);
                for k_idx in 0..num_chunks {
                    let c_j = if j < witness_len { &all_chunks[ell][k_idx][j] } else { &zero };
                    let u_j = if j < witness_len { &all_inverse_witnesses[ell][k_idx][j] } else { &zero };
                    let id = (beta - c_j) * u_j - &one;
                    acc += &(id * &gamma_powers[base_id + k_idx]);
                }
                let m_agg_j = if j < subtable_len { &all_aggregated_multiplicities[ell][j] } else { &zero };
                let mut u_sum = if j < witness_len { all_inverse_witnesses[ell][0][j] } else { zero };
                for k_idx in 1..num_chunks {
                    if j < witness_len {
                        u_sum += &all_inverse_witnesses[ell][k_idx][j];
                    }
                }
                let balance = u_sum - &(*m_agg_j * v_j);
                acc += &(balance * &gamma_powers[base_id + num_chunks]);
            }
            acc
        })
        .collect();
    let step7 = t0.elapsed();

    // Step 8: Build MLEs
    let t0 = Instant::now();
    let r: Vec<F> = transcript.get_field_challenges(num_vars, field_cfg);
    let eq_r = build_eq_x_r_inner::<F>(&r, field_cfg).unwrap();
    let inner_zero = zero.inner().clone();
    let h_mle = DenseMultilinearExtension::from_evaluations_vec(
        num_vars,
        h_evaluations.iter().map(|x| x.inner().clone()).collect(),
        inner_zero,
    );
    let step8 = t0.elapsed();

    // Step 9: Sumcheck
    let t0 = Instant::now();
    let mles = vec![eq_r, h_mle];
    let comb_fn = move |vals: &[F]| -> F { vals[0] * &vals[1] };
    let degree = 2;
    let _ = zinc_piop::sumcheck::MLSumcheck::prove_as_subprotocol(
        &mut transcript,
        mles,
        num_vars,
        degree,
        comb_fn,
        field_cfg,
    );
    let step9 = t0.elapsed();

    let total = step1 + step2 + step3 + step4 + step5 + step6 + step7 + step8 + step9;

    println!("  Step 1  Absorb chunks:    {:>10.3} ms  ({:>5.1}%)", step1.as_secs_f64() * 1000.0, step1.as_secs_f64() / total.as_secs_f64() * 100.0);
    println!("  Step 2  Multiplicities:   {:>10.3} ms  ({:>5.1}%)", step2.as_secs_f64() * 1000.0, step2.as_secs_f64() / total.as_secs_f64() * 100.0);
    println!("  Step 3  β challenge:      {:>10.3} ms  ({:>5.1}%)", step3.as_secs_f64() * 1000.0, step3.as_secs_f64() / total.as_secs_f64() * 100.0);
    println!("  Step 4  Batch inverses:   {:>10.3} ms  ({:>5.1}%)", step4.as_secs_f64() * 1000.0, step4.as_secs_f64() / total.as_secs_f64() * 100.0);
    println!("  Step 5  γ challenge:      {:>10.3} ms  ({:>5.1}%)", step5.as_secs_f64() * 1000.0, step5.as_secs_f64() / total.as_secs_f64() * 100.0);
    println!("  Step 6  γ powers:         {:>10.3} ms  ({:>5.1}%)", step6.as_secs_f64() * 1000.0, step6.as_secs_f64() / total.as_secs_f64() * 100.0);
    println!("  Step 7  Precompute H:     {:>10.3} ms  ({:>5.1}%)", step7.as_secs_f64() * 1000.0, step7.as_secs_f64() / total.as_secs_f64() * 100.0);
    println!("  Step 8  Build MLEs:       {:>10.3} ms  ({:>5.1}%)", step8.as_secs_f64() * 1000.0, step8.as_secs_f64() / total.as_secs_f64() * 100.0);
    println!("  Step 9  Sumcheck:         {:>10.3} ms  ({:>5.1}%)", step9.as_secs_f64() * 1000.0, step9.as_secs_f64() / total.as_secs_f64() * 100.0);
    println!("  ─────────────────────────────────────────────");
    println!("  Total (steps):            {:>10.3} ms", total.as_secs_f64() * 1000.0);
}
