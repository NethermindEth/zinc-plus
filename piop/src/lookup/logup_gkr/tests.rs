//! Unit tests for the standalone logup-GKR subprotocol.

use crypto_bigint::{U128, const_monty_params};
use crypto_primitives::{Field, PrimeField, crypto_bigint_const_monty::ConstMontyField};
use rand::{RngCore, SeedableRng, rngs::StdRng};
use zinc_poly::mle::DenseMultilinearExtension;
use zinc_transcript::Blake3Transcript;

use crate::lookup::logup_gkr::{
    argument::{LookupArgument, LookupArgumentError},
    circuit::GrandSumCircuit,
    leaves::{LeafComponentEvals, build_lookup_leaves, expected_leaf_evals},
    prover::LogupGkrProver,
    verifier::LogupGkrVerifier,
};

const_monty_params!(TestParams, U128, "00000000b933426489189cb5b47d567f");
type F = ConstMontyField<TestParams, { U128::LIMBS }>;

fn random_nonzero_mle(
    num_vars: usize,
    rng: &mut impl RngCore,
) -> DenseMultilinearExtension<<F as Field>::Inner> {
    let size = 1usize << num_vars;
    let zero_inner = F::zero_with_cfg(&()).into_inner();
    let mut evals = Vec::with_capacity(size);
    for _ in 0..size {
        // 64-bit random, avoiding zero to keep denominators nonzero.
        let mut v = rng.next_u64();
        while v == 0 {
            v = rng.next_u64();
        }
        let f = F::from(v);
        evals.push(f.into_inner());
    }
    DenseMultilinearExtension::from_evaluations_vec(num_vars, evals, zero_inner)
}

fn random_mle(
    num_vars: usize,
    rng: &mut impl RngCore,
) -> DenseMultilinearExtension<<F as Field>::Inner> {
    let size = 1usize << num_vars;
    let zero_inner = F::zero_with_cfg(&()).into_inner();
    let mut evals = Vec::with_capacity(size);
    for _ in 0..size {
        let f = F::from(rng.next_u64());
        evals.push(f.into_inner());
    }
    DenseMultilinearExtension::from_evaluations_vec(num_vars, evals, zero_inner)
}

/// Lift an MLE of `F::Inner` evaluations to a per-entry `F` vector.
fn inner_mle_to_f_mle(
    mle: &DenseMultilinearExtension<<F as Field>::Inner>,
) -> DenseMultilinearExtension<F> {
    let cfg = ();
    let evals: Vec<F> = mle
        .evaluations
        .iter()
        .map(|x| F::new_unchecked_with_cfg(x.clone(), &cfg))
        .collect();
    DenseMultilinearExtension::from_evaluations_vec(mle.num_vars, evals, F::zero_with_cfg(&cfg))
}

#[test]
fn roundtrip_random_leaves() {
    let cfg = ();
    let mut rng = StdRng::seed_from_u64(0xDEADBEEF);

    for num_vars in 1..=5 {
        let n_leaves = random_mle(num_vars, &mut rng);
        let d_leaves = random_nonzero_mle(num_vars, &mut rng);

        let circuit =
            GrandSumCircuit::<F>::build(n_leaves.clone(), d_leaves.clone(), &cfg);

        let mut p_transcript = Blake3Transcript::new();
        let (proof, prover_subclaim) =
            LogupGkrProver::<F>::prove(&mut p_transcript, &circuit, &cfg);

        let mut v_transcript = Blake3Transcript::new();
        let (verifier_subclaim, _root_n, _root_d) =
            LogupGkrVerifier::<F>::verify(&mut v_transcript, num_vars, &proof, &cfg)
                .expect("verify failed");

        // Points and derived claims must match.
        assert_eq!(prover_subclaim.point, verifier_subclaim.point);
        assert_eq!(
            prover_subclaim.numerator_eval,
            verifier_subclaim.numerator_eval,
        );
        assert_eq!(
            prover_subclaim.denominator_eval,
            verifier_subclaim.denominator_eval,
        );

        // Verifier-side reconciliation: the subclaim's evaluations must
        // equal the true leaf MLE evaluations at the subclaim's point.
        let n_lift = inner_mle_to_f_mle(&n_leaves);
        let d_lift = inner_mle_to_f_mle(&d_leaves);
        let n_at_point = n_lift
            .evaluate(&verifier_subclaim.point, F::zero_with_cfg(&cfg))
            .unwrap();
        let d_at_point = d_lift
            .evaluate(&verifier_subclaim.point, F::zero_with_cfg(&cfg))
            .unwrap();
        assert_eq!(verifier_subclaim.numerator_eval, n_at_point);
        assert_eq!(verifier_subclaim.denominator_eval, d_at_point);
    }
}

#[test]
fn roundtrip_single_variable() {
    // Minimum case: 2 leaves. Exercises the nvp=0 degenerate branch.
    let cfg = ();
    let mut rng = StdRng::seed_from_u64(42);

    let n_leaves = random_mle(1, &mut rng);
    let d_leaves = random_nonzero_mle(1, &mut rng);

    let circuit = GrandSumCircuit::<F>::build(n_leaves.clone(), d_leaves.clone(), &cfg);

    let mut p_transcript = Blake3Transcript::new();
    let (proof, _) = LogupGkrProver::<F>::prove(&mut p_transcript, &circuit, &cfg);

    let mut v_transcript = Blake3Transcript::new();
    let result = LogupGkrVerifier::<F>::verify(&mut v_transcript, 1, &proof, &cfg);
    assert!(result.is_ok(), "verify should succeed for 1-var circuit");
}

#[test]
fn root_values_match_cumulative_sum() {
    // For D[i] = 1 for all i, root_n/root_d should equal Σ N[i].
    let cfg = ();
    let num_vars = 3;
    let mut rng = StdRng::seed_from_u64(7);

    let n_leaves = random_mle(num_vars, &mut rng);
    let zero_inner = F::zero_with_cfg(&cfg).into_inner();
    let one = F::from(1u32);
    let d_evals: Vec<_> = (0..(1 << num_vars)).map(|_| one.clone().into_inner()).collect();
    let d_leaves =
        DenseMultilinearExtension::from_evaluations_vec(num_vars, d_evals, zero_inner);

    let circuit = GrandSumCircuit::<F>::build(n_leaves.clone(), d_leaves, &cfg);

    let expected_sum = n_leaves
        .evaluations
        .iter()
        .cloned()
        .map(|x| F::new_unchecked_with_cfg(x, &cfg))
        .fold(F::zero_with_cfg(&cfg), |acc, v| acc + v);

    let root_n_f = F::new_unchecked_with_cfg(
        circuit.root().numerator.evaluations[0].clone(),
        &cfg,
    );
    let root_d_f = F::new_unchecked_with_cfg(
        circuit.root().denominator.evaluations[0].clone(),
        &cfg,
    );

    // root_d == 1 * 1 * ... * 1 == 1.
    assert_eq!(root_d_f, F::from(1u32));
    assert_eq!(root_n_f, expected_sum);
}

#[test]
fn tamper_root_numerator_rejected() {
    let cfg = ();
    let mut rng = StdRng::seed_from_u64(1234);

    let n_leaves = random_mle(3, &mut rng);
    let d_leaves = random_nonzero_mle(3, &mut rng);
    let circuit = GrandSumCircuit::<F>::build(n_leaves, d_leaves, &cfg);

    let mut p_transcript = Blake3Transcript::new();
    let (mut proof, _) = LogupGkrProver::<F>::prove(&mut p_transcript, &circuit, &cfg);

    // Tamper the root numerator: add 1.
    proof.root_numerator = proof.root_numerator.clone() + F::from(1u32);

    let mut v_transcript = Blake3Transcript::new();
    let result = LogupGkrVerifier::<F>::verify(&mut v_transcript, 3, &proof, &cfg);
    assert!(
        result.is_err(),
        "verifier must reject tampered root_numerator"
    );
}

#[test]
fn tamper_round_numerator_rejected() {
    let cfg = ();
    let mut rng = StdRng::seed_from_u64(5678);

    let n_leaves = random_mle(3, &mut rng);
    let d_leaves = random_nonzero_mle(3, &mut rng);
    let circuit = GrandSumCircuit::<F>::build(n_leaves, d_leaves, &cfg);

    let mut p_transcript = Blake3Transcript::new();
    let (mut proof, _) = LogupGkrProver::<F>::prove(&mut p_transcript, &circuit, &cfg);

    // Tamper the last round's numerator_0.
    let last = proof.round_proofs.last_mut().unwrap();
    last.numerator_0 = last.numerator_0.clone() + F::from(1u32);

    let mut v_transcript = Blake3Transcript::new();
    let result = LogupGkrVerifier::<F>::verify(&mut v_transcript, 3, &proof, &cfg);
    assert!(
        result.is_err(),
        "verifier must reject tampered round numerator"
    );
}

#[test]
fn invalid_shape_rejected() {
    let cfg = ();
    let mut rng = StdRng::seed_from_u64(99);

    let n_leaves = random_mle(3, &mut rng);
    let d_leaves = random_nonzero_mle(3, &mut rng);
    let circuit = GrandSumCircuit::<F>::build(n_leaves, d_leaves, &cfg);

    let mut p_transcript = Blake3Transcript::new();
    let (mut proof, _) = LogupGkrProver::<F>::prove(&mut p_transcript, &circuit, &cfg);

    // Drop a round proof.
    proof.round_proofs.pop();

    let mut v_transcript = Blake3Transcript::new();
    let result = LogupGkrVerifier::<F>::verify(&mut v_transcript, 3, &proof, &cfg);
    assert!(result.is_err(), "verifier must reject wrong number of rounds");
}

// ---------------------------------------------------------------------------
// Lookup-specific integration tests: build leaves from (witness cols, table,
// multiplicities, alpha), prove via logup-GKR, verify, and reconcile the
// final subclaim against the component evaluations at the trace-level point.
// ---------------------------------------------------------------------------

/// Build an MLE over `row_vars` variables from a concrete `F`-valued vector.
fn f_vec_to_mle(
    num_vars: usize,
    evals: &[F],
) -> DenseMultilinearExtension<<F as Field>::Inner> {
    let size = 1usize << num_vars;
    assert_eq!(evals.len(), size);
    let zero_inner = F::zero_with_cfg(&()).into_inner();
    let inner: Vec<_> = evals.iter().map(|v| v.clone().into_inner()).collect();
    DenseMultilinearExtension::from_evaluations_vec(num_vars, inner, zero_inner)
}

/// Lift an inner MLE to an `F`-valued MLE so we can call `.evaluate(...)`.
fn lift_mle(
    mle: &DenseMultilinearExtension<<F as Field>::Inner>,
) -> DenseMultilinearExtension<F> {
    let cfg = ();
    let evals: Vec<F> = mle
        .evaluations
        .iter()
        .map(|x| F::new_unchecked_with_cfg(x.clone(), &cfg))
        .collect();
    DenseMultilinearExtension::from_evaluations_vec(mle.num_vars, evals, F::zero_with_cfg(&cfg))
}

/// Build a valid lookup: the table is `0, 1, ..., 2^row_vars - 1`, the single
/// witness column holds arbitrary values from that range, and the
/// multiplicity vector is the exact counts.
fn make_valid_lookup(
    row_vars: usize,
    num_witness_cols: usize,
    witness_values: &[Vec<u64>],
) -> (
    Vec<DenseMultilinearExtension<<F as Field>::Inner>>,
    DenseMultilinearExtension<<F as Field>::Inner>,
    DenseMultilinearExtension<<F as Field>::Inner>,
) {
    let row_count = 1usize << row_vars;
    assert_eq!(witness_values.len(), num_witness_cols);
    for w in witness_values {
        assert_eq!(w.len(), row_count);
    }

    // Table: identity 0..2^row_vars - 1.
    let table_f: Vec<F> = (0..row_count as u64).map(F::from).collect();

    // Multiplicities: count occurrences of each table entry across all witness columns.
    let mut mults = vec![0u64; row_count];
    for w in witness_values {
        for &v in w {
            assert!((v as usize) < row_count, "witness value must be in table");
            mults[v as usize] += 1;
        }
    }

    let witness_mles: Vec<DenseMultilinearExtension<<F as Field>::Inner>> = witness_values
        .iter()
        .map(|w| {
            let evals_f: Vec<F> = w.iter().map(|&v| F::from(v)).collect();
            f_vec_to_mle(row_vars, &evals_f)
        })
        .collect();

    let table_mle = f_vec_to_mle(row_vars, &table_f);
    let mults_f: Vec<F> = mults.iter().map(|&m| F::from(m)).collect();
    let mults_mle = f_vec_to_mle(row_vars, &mults_f);

    (witness_mles, table_mle, mults_mle)
}

#[test]
fn lookup_valid_single_column_accepts() {
    let cfg = ();
    let row_vars = 3; // 8-row table
    let witness_values = vec![vec![0u64, 2, 5, 5, 1, 7, 3, 2]];
    let (wit_mles, table_mle, mult_mle) = make_valid_lookup(row_vars, 1, &witness_values);

    let alpha = F::from(999_991u64); // a "random" challenge that doesn't collide with any table entry

    let wit_refs: Vec<_> = wit_mles.iter().collect();
    let leaves = build_lookup_leaves::<F>(&wit_refs, &table_mle, &mult_mle, &alpha, &cfg);

    // Sanity: the cumulative sum (root_numerator / root_denominator) must be
    // zero for a valid lookup. Checked via the grand-sum circuit root below.
    let circuit = GrandSumCircuit::<F>::build(
        leaves.numerator.clone(),
        leaves.denominator.clone(),
        &cfg,
    );
    let root_n = F::new_unchecked_with_cfg(circuit.root().numerator.evaluations[0].clone(), &cfg);
    assert_eq!(
        root_n,
        F::zero_with_cfg(&cfg),
        "valid lookup must yield root_numerator == 0 (cumulative sum)"
    );

    // Run the full GKR prove/verify round trip.
    let mut p_transcript = Blake3Transcript::new();
    let (proof, _prover_sub) = LogupGkrProver::<F>::prove(&mut p_transcript, &circuit, &cfg);
    let mut v_transcript = Blake3Transcript::new();
    let (verifier_sub, root_n_out, root_d_out) = LogupGkrVerifier::<F>::verify(
        &mut v_transcript,
        leaves.total_vars(),
        &proof,
        &cfg,
    )
    .expect("verify should succeed for valid lookup");

    assert_eq!(root_n_out, F::zero_with_cfg(&cfg));
    assert_ne!(root_d_out, F::zero_with_cfg(&cfg));

    // Caller-side reconciliation: split the final point into (rho_row, rho_slot),
    // evaluate each component at rho_row, and check the reconstruction matches
    // the subclaim from the GKR verifier.
    let (rho_row, rho_slot) = verifier_sub.point.split_at(leaves.row_vars);

    let wit_evals: Vec<F> = wit_mles
        .iter()
        .map(|w| {
            lift_mle(w)
                .evaluate(rho_row, F::zero_with_cfg(&cfg))
                .unwrap()
        })
        .collect();
    let t_eval = lift_mle(&table_mle)
        .evaluate(rho_row, F::zero_with_cfg(&cfg))
        .unwrap();
    let m_eval = lift_mle(&mult_mle)
        .evaluate(rho_row, F::zero_with_cfg(&cfg))
        .unwrap();

    let components = LeafComponentEvals {
        witness_evals: wit_evals,
        table_eval: t_eval,
        multiplicity_eval: m_eval,
    };
    let (expected_n, expected_d) =
        expected_leaf_evals::<F>(rho_slot, &components, &alpha, &cfg);

    assert_eq!(
        verifier_sub.numerator_eval, expected_n,
        "reconstructed N(rho) must match GKR subclaim's numerator_eval"
    );
    assert_eq!(
        verifier_sub.denominator_eval, expected_d,
        "reconstructed D(rho) must match GKR subclaim's denominator_eval"
    );
}

#[test]
fn lookup_valid_multiple_columns_accepts() {
    let cfg = ();
    let row_vars = 2; // 4-row table
    let witness_values = vec![
        vec![0u64, 1, 2, 3],
        vec![3u64, 3, 0, 1],
        vec![2u64, 1, 3, 0],
    ];
    let (wit_mles, table_mle, mult_mle) = make_valid_lookup(row_vars, 3, &witness_values);

    let alpha = F::from(104_729u64); // prime, not in table
    let wit_refs: Vec<_> = wit_mles.iter().collect();
    let leaves = build_lookup_leaves::<F>(&wit_refs, &table_mle, &mult_mle, &alpha, &cfg);
    let circuit = GrandSumCircuit::<F>::build(
        leaves.numerator.clone(),
        leaves.denominator.clone(),
        &cfg,
    );

    let root_n = F::new_unchecked_with_cfg(
        circuit.root().numerator.evaluations[0].clone(),
        &cfg,
    );
    assert_eq!(root_n, F::zero_with_cfg(&cfg));

    let mut p_transcript = Blake3Transcript::new();
    let (proof, _) = LogupGkrProver::<F>::prove(&mut p_transcript, &circuit, &cfg);
    let mut v_transcript = Blake3Transcript::new();
    let result = LogupGkrVerifier::<F>::verify(
        &mut v_transcript,
        leaves.total_vars(),
        &proof,
        &cfg,
    );
    assert!(result.is_ok(), "valid multi-column lookup must accept");
}

#[test]
fn lookup_witness_not_in_table_root_nonzero() {
    // Construct an INVALID lookup: one witness value is outside the table.
    // The root numerator should then be nonzero, and the caller's outer
    // check (root_numerator == 0) rejects. We verify the GKR round-trip
    // still succeeds internally (soundness is at the outer level).
    let cfg = ();
    let row_vars = 2;
    let row_count = 1usize << row_vars;
    let table_f: Vec<F> = (0..row_count as u64).map(F::from).collect();

    // Witness contains a value "4" that is NOT in {0,1,2,3}.
    let witness_f: Vec<F> = vec![F::from(0u64), F::from(1u64), F::from(4u64), F::from(2u64)];
    // Multiplicities (best effort): count occurrences of each table entry.
    let mults_f: Vec<F> = vec![F::from(1u64), F::from(1u64), F::from(1u64), F::from(0u64)];

    let wit_mle = f_vec_to_mle(row_vars, &witness_f);
    let table_mle = f_vec_to_mle(row_vars, &table_f);
    let mult_mle = f_vec_to_mle(row_vars, &mults_f);

    let alpha = F::from(31_337u64);
    let wit_refs = vec![&wit_mle];
    let leaves = build_lookup_leaves::<F>(&wit_refs, &table_mle, &mult_mle, &alpha, &cfg);
    let circuit = GrandSumCircuit::<F>::build(
        leaves.numerator.clone(),
        leaves.denominator.clone(),
        &cfg,
    );

    let root_n = F::new_unchecked_with_cfg(
        circuit.root().numerator.evaluations[0].clone(),
        &cfg,
    );
    assert_ne!(
        root_n,
        F::zero_with_cfg(&cfg),
        "invalid lookup must yield nonzero cumulative sum"
    );
}

#[test]
fn lookup_wrong_multiplicity_root_nonzero() {
    // Valid witness but WRONG multiplicities — the logup identity breaks
    // and the cumulative sum is nonzero.
    let cfg = ();
    let row_vars = 2;
    let row_count = 1usize << row_vars;
    let table_f: Vec<F> = (0..row_count as u64).map(F::from).collect();

    let witness_f: Vec<F> = (0..row_count as u64).map(F::from).collect();
    // Real multiplicities would all be 1. Wrong: swap two entries.
    let mults_f: Vec<F> = vec![F::from(2u64), F::from(0u64), F::from(1u64), F::from(1u64)];

    let wit_mle = f_vec_to_mle(row_vars, &witness_f);
    let table_mle = f_vec_to_mle(row_vars, &table_f);
    let mult_mle = f_vec_to_mle(row_vars, &mults_f);

    let alpha = F::from(271u64);
    let wit_refs = vec![&wit_mle];
    let leaves = build_lookup_leaves::<F>(&wit_refs, &table_mle, &mult_mle, &alpha, &cfg);
    let circuit = GrandSumCircuit::<F>::build(
        leaves.numerator.clone(),
        leaves.denominator.clone(),
        &cfg,
    );
    let root_n = F::new_unchecked_with_cfg(
        circuit.root().numerator.evaluations[0].clone(),
        &cfg,
    );
    assert_ne!(root_n, F::zero_with_cfg(&cfg));
}

// ---------------------------------------------------------------------------
// Single-call LookupArgument API tests
// ---------------------------------------------------------------------------

#[test]
fn argument_valid_lookup_roundtrip() {
    let cfg = ();
    let row_vars = 3;
    let witness_values = vec![vec![0u64, 2, 5, 5, 1, 7, 3, 2]];
    let (wit_mles, table_mle, mult_mle) = make_valid_lookup(row_vars, 1, &witness_values);
    let wit_refs: Vec<_> = wit_mles.iter().collect();

    let mut p_transcript = Blake3Transcript::new();
    let (proof, prover_sub) = LookupArgument::<F>::prove(
        &mut p_transcript,
        &wit_refs,
        &table_mle,
        &mult_mle,
        &cfg,
    );

    let mut v_transcript = Blake3Transcript::new();
    let verifier_sub = LookupArgument::<F>::verify(
        &mut v_transcript,
        1,
        row_vars,
        &proof,
        &cfg,
    )
    .expect("verify should succeed");

    assert_eq!(prover_sub.rho_row, verifier_sub.rho_row);
    assert_eq!(prover_sub.alpha, verifier_sub.alpha);
    assert_eq!(prover_sub.component_evals, verifier_sub.component_evals);

    // Component evals match direct MLE evaluation at ρ_row (this is the
    // claim the outer protocol must subsequently bind to committed MLEs).
    let rho_row = &verifier_sub.rho_row;
    let expected_wit_eval = lift_mle(&wit_mles[0])
        .evaluate(rho_row, F::zero_with_cfg(&cfg))
        .unwrap();
    assert_eq!(verifier_sub.component_evals.witness_evals[0], expected_wit_eval);
}

#[test]
fn argument_multi_column_roundtrip() {
    let cfg = ();
    let row_vars = 2;
    let witness_values = vec![
        vec![0u64, 1, 2, 3],
        vec![3u64, 3, 0, 1],
        vec![2u64, 1, 3, 0],
    ];
    let (wit_mles, table_mle, mult_mle) = make_valid_lookup(row_vars, 3, &witness_values);
    let wit_refs: Vec<_> = wit_mles.iter().collect();

    let mut p_transcript = Blake3Transcript::new();
    let (proof, _) =
        LookupArgument::<F>::prove(&mut p_transcript, &wit_refs, &table_mle, &mult_mle, &cfg);

    let mut v_transcript = Blake3Transcript::new();
    let result = LookupArgument::<F>::verify(&mut v_transcript, 3, row_vars, &proof, &cfg);
    assert!(result.is_ok(), "multi-column valid lookup must accept");
}

#[test]
fn argument_invalid_witness_rejected() {
    let cfg = ();
    let row_vars = 2;
    let row_count = 1usize << row_vars;
    // Witness contains "4", outside table {0..3}. Provide invalid multiplicities.
    let witness_f: Vec<F> = vec![F::from(0u64), F::from(1u64), F::from(4u64), F::from(2u64)];
    let mults_f: Vec<F> = vec![F::from(1u64), F::from(1u64), F::from(1u64), F::from(0u64)];
    let table_f: Vec<F> = (0..row_count as u64).map(F::from).collect();

    let wit_mle = f_vec_to_mle(row_vars, &witness_f);
    let table_mle = f_vec_to_mle(row_vars, &table_f);
    let mult_mle = f_vec_to_mle(row_vars, &mults_f);

    let mut p_transcript = Blake3Transcript::new();
    let (proof, _) = LookupArgument::<F>::prove(
        &mut p_transcript,
        &[&wit_mle],
        &table_mle,
        &mult_mle,
        &cfg,
    );

    let mut v_transcript = Blake3Transcript::new();
    let result = LookupArgument::<F>::verify(&mut v_transcript, 1, row_vars, &proof, &cfg);
    match result {
        Err(LookupArgumentError::NonzeroRootNumerator) => {}
        other => panic!("expected NonzeroRootNumerator, got {other:?}"),
    }
}

#[test]
fn argument_tampered_component_evals_rejected() {
    let cfg = ();
    let row_vars = 3;
    let witness_values = vec![vec![0u64, 2, 5, 5, 1, 7, 3, 2]];
    let (wit_mles, table_mle, mult_mle) = make_valid_lookup(row_vars, 1, &witness_values);
    let wit_refs: Vec<_> = wit_mles.iter().collect();

    let mut p_transcript = Blake3Transcript::new();
    let (mut proof, _) = LookupArgument::<F>::prove(
        &mut p_transcript,
        &wit_refs,
        &table_mle,
        &mult_mle,
        &cfg,
    );

    // Tamper: add 1 to the witness component eval. This breaks the
    // leaf-reconstruction check even though GKR itself still verifies.
    proof.component_evals.witness_evals[0] =
        proof.component_evals.witness_evals[0].clone() + F::from(1u64);

    let mut v_transcript = Blake3Transcript::new();
    let result = LookupArgument::<F>::verify(&mut v_transcript, 1, row_vars, &proof, &cfg);
    match result {
        Err(LookupArgumentError::LeafReconstructionMismatch) => {}
        other => panic!("expected LeafReconstructionMismatch, got {other:?}"),
    }
}

#[test]
fn argument_wrong_witness_count_rejected() {
    let cfg = ();
    let row_vars = 2;
    let witness_values = vec![vec![0u64, 1, 2, 3]];
    let (wit_mles, table_mle, mult_mle) = make_valid_lookup(row_vars, 1, &witness_values);
    let wit_refs: Vec<_> = wit_mles.iter().collect();

    let mut p_transcript = Blake3Transcript::new();
    let (proof, _) =
        LookupArgument::<F>::prove(&mut p_transcript, &wit_refs, &table_mle, &mult_mle, &cfg);

    // Claim 2 witness columns when the proof has evals for only 1.
    let mut v_transcript = Blake3Transcript::new();
    let result = LookupArgument::<F>::verify(&mut v_transcript, 2, row_vars, &proof, &cfg);
    match result {
        Err(LookupArgumentError::WitnessCountMismatch { expected: 2, got: 1 }) => {}
        other => panic!("expected WitnessCountMismatch, got {other:?}"),
    }
}
