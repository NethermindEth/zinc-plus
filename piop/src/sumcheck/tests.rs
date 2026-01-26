mod utils;

use crypto_bigint::{U128, const_monty_params};
use crypto_primitives::{Field, crypto_bigint_const_monty::ConstMontyField};
use num_traits::{ConstOne, ConstZero, Zero};
use rand::RngCore;
use zinc_poly::mle::{DenseMultilinearExtension, MultilinearExtensionWithConfig};
use zinc_transcript::{KeccakTranscript, traits::Transcript};
use zinc_utils::inner_transparent_field::InnerTransparentField;

use crate::sumcheck::{
    prover::ProverState,
    tests::utils::{rand_poly, rand_poly_comb_fn},
};

use super::{MLSumcheck, SumcheckProof};

const N: usize = 2;

const_monty_params!(Params, U128, "00000000b933426489189cb5b47d567f");

type F = ConstMontyField<Params, N>;

fn generate_sumcheck_proof<Rn: RngCore>(
    num_vars: usize,
    mut rng: &mut Rn,
) -> (usize, F, SumcheckProof<F>) {
    let mut transcript = KeccakTranscript::default();

    let ((poly_mles, poly_degree), products, sum) =
        rand_poly(num_vars, 2..5, 7, &mut rng, &()).unwrap();

    let comb_fn = |vals: &[F]| -> F { rand_poly_comb_fn(vals, &products, ()) };

    let (proof, _) = MLSumcheck::prove_as_subprotocol(
        &mut transcript,
        poly_mles,
        num_vars,
        poly_degree,
        comb_fn,
        (),
    );
    (poly_degree, sum, proof)
}

#[test]
fn full_sumcheck_protocol_works_correctly() {
    let mut rng = rand::rng();
    let num_vars = 3;

    for _ in 0..20 {
        let (poly_degree, sum, proof) = generate_sumcheck_proof(num_vars, &mut rng);

        let mut transcript = KeccakTranscript::default();
        let res = MLSumcheck::verify_as_subprotocol(
            &mut transcript,
            num_vars,
            poly_degree,
            sum,
            &proof,
            (),
        );
        assert!(res.is_ok())
    }
}

#[test]
fn prover_message_omits_constant_term() {
    let mut rng = rand::rng();
    let num_vars = 3;

    let (poly_degree, _sum, proof) = generate_sumcheck_proof(num_vars, &mut rng);

    assert_eq!(proof.messages[0].0.len(), poly_degree);
}

#[test]
fn subclaim_differs_with_incorrect_claimed_sum() {
    let mut rng = rand::rng();
    let num_vars = 3;

    let mut transcript = KeccakTranscript::default();
    let ((poly_mles, poly_degree), products, sum) =
        rand_poly(num_vars, 2..5, 7, &mut rng, &()).unwrap();

    let comb_fn = move |vals: &[F]| -> F { rand_poly_comb_fn(vals, &products, ()) };

    let (proof, _) = MLSumcheck::prove_as_subprotocol(
        &mut transcript,
        poly_mles,
        num_vars,
        poly_degree,
        comb_fn,
        (),
    );

    let one = F::from(1u32);
    let incorrect_sum = sum + one;

    let mut clean_verifier_transcript = KeccakTranscript::default();
    let clean_subclaim = MLSumcheck::verify_as_subprotocol(
        &mut clean_verifier_transcript,
        num_vars,
        poly_degree,
        sum,
        &proof,
        (),
    )
    .unwrap();

    let mut verifier_transcript = KeccakTranscript::default();
    let res = MLSumcheck::verify_as_subprotocol(
        &mut verifier_transcript,
        num_vars,
        poly_degree,
        incorrect_sum,
        &proof,
        (),
    )
    .unwrap();

    assert_eq!(clean_subclaim.point, res.point);
    assert_ne!(clean_subclaim.expected_evaluation, res.expected_evaluation);
}

#[test]
fn subclaim_changes_when_prover_message_is_tampered() {
    let mut rng = rand::rng();
    let num_vars = 3;

    let mut transcript = KeccakTranscript::default();
    let ((poly_mles, poly_degree), products, sum) =
        rand_poly(num_vars, 2..5, 7, &mut rng, &()).unwrap();

    let comb_fn = move |vals: &[F]| -> F { rand_poly_comb_fn(vals, &products, ()) };

    let (proof, _) = MLSumcheck::prove_as_subprotocol(
        &mut transcript,
        poly_mles,
        num_vars,
        poly_degree,
        comb_fn,
        (),
    );

    let mut clean_verifier_transcript = KeccakTranscript::default();
    let clean_subclaim = MLSumcheck::verify_as_subprotocol(
        &mut clean_verifier_transcript,
        num_vars,
        poly_degree,
        sum,
        &proof,
        (),
    )
    .unwrap();

    let mut tampered_proof = proof.clone();
    let one: F = F::from(1u32);
    tampered_proof.messages[0].0[0] += one;

    let mut verifier_transcript = KeccakTranscript::default();
    let res = MLSumcheck::verify_as_subprotocol(
        &mut verifier_transcript,
        num_vars,
        poly_degree,
        sum,
        &tampered_proof,
        (),
    )
    .unwrap();

    assert_ne!(clean_subclaim.point, res.point);
    assert_ne!(clean_subclaim.expected_evaluation, res.expected_evaluation);
}

#[test]
fn verifier_rejects_proof_with_wrong_degree() {
    let mut rng = rand::rng();
    let num_vars = 3;

    let mut transcript = KeccakTranscript::default();
    let ((poly_mles, poly_degree), products, sum) =
        rand_poly(num_vars, 2..5, 7, &mut rng, &()).unwrap();

    let comb_fn = move |vals: &[F]| -> F { rand_poly_comb_fn(vals, &products, ()) };

    let (proof, _) = MLSumcheck::prove_as_subprotocol(
        &mut transcript,
        poly_mles,
        num_vars,
        poly_degree,
        comb_fn,
        (),
    );

    let incorrect_degree = poly_degree - 1;

    let mut verifier_transcript = KeccakTranscript::default();
    let res = MLSumcheck::verify_as_subprotocol(
        &mut verifier_transcript,
        num_vars,
        incorrect_degree,
        sum,
        &proof,
        (),
    );

    assert!(res.is_err());
}

#[test]
fn protocol_is_deterministic_with_same_transcript() {
    let mut rng = rand::rng();
    let num_vars = 3;

    let ((poly_mles, poly_degree), products, _) =
        rand_poly(num_vars, 2..5, 7, &mut rng, &()).unwrap();

    let comb_fn = move |vals: &[F]| -> F { rand_poly_comb_fn(vals, &products, ()) };

    let mut transcript1 = KeccakTranscript::default();
    let (proof1, _) = MLSumcheck::prove_as_subprotocol(
        &mut transcript1,
        poly_mles.clone(),
        num_vars,
        poly_degree,
        comb_fn.clone(),
        (),
    );

    let mut transcript2 = KeccakTranscript::default();
    let (proof2, _) = MLSumcheck::prove_as_subprotocol(
        &mut transcript2,
        poly_mles,
        num_vars,
        poly_degree,
        comb_fn,
        (),
    );

    assert_eq!(proof1, proof2);
}

#[test]
fn different_polynomials_produce_different_proofs() {
    let mut rng = rand::rng();
    let num_vars = 3;

    let ((poly_mles1, poly_degree1), products1, _) =
        rand_poly(num_vars, 2..5, 7, &mut rng, &()).unwrap();

    let comb_fn1 = {
        let products = products1.clone();
        move |vals: &[F]| -> F { rand_poly_comb_fn(vals, &products, ()) }
    };

    let mut transcript1 = KeccakTranscript::default();
    let (proof1, _) = MLSumcheck::prove_as_subprotocol(
        &mut transcript1,
        poly_mles1.clone(),
        num_vars,
        poly_degree1,
        comb_fn1,
        (),
    );

    let mut poly_mles2 = poly_mles1;
    let one: F = F::ONE;
    poly_mles2[0][0] = F::add_inner(&poly_mles2[0].evaluations[0], one.inner(), &());

    let comb_fn2 = move |vals: &[F]| -> F { rand_poly_comb_fn(vals, &products1, ()) };

    let mut transcript2 = KeccakTranscript::default();
    let (proof2, _) = MLSumcheck::prove_as_subprotocol(
        &mut transcript2,
        poly_mles2,
        num_vars,
        poly_degree1,
        comb_fn2,
        (),
    );

    assert_ne!(proof1, proof2);
}

#[test]
fn sumcheck_with_zero_polynomial() {
    let num_vars = 3;

    let poly_degree = 2;
    let num_mles = 2;
    let zero_evals = vec![<F as Field>::Inner::ZERO; 1 << num_vars];
    let poly_mles: Vec<DenseMultilinearExtension<<F as Field>::Inner>> = (0..num_mles)
        .map(|_| {
            DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                zero_evals.clone(),
                <F as Field>::Inner::ZERO,
            )
        })
        .collect();

    let sum = F::ZERO;

    let comb_fn = |vals: &[F]| -> F { vals.iter().product() };

    let mut transcript = KeccakTranscript::default();
    let (proof, _) = MLSumcheck::prove_as_subprotocol(
        &mut transcript,
        poly_mles,
        num_vars,
        poly_degree,
        comb_fn,
        (),
    );

    assert!(proof.claimed_sum.is_zero());

    let mut verifier_transcript = KeccakTranscript::default();
    let res = MLSumcheck::verify_as_subprotocol(
        &mut verifier_transcript,
        num_vars,
        poly_degree,
        sum,
        &proof,
        (),
    );

    assert!(res.is_ok());
}

#[test]
fn sumcheck_with_constant_polynomial() {
    let num_vars = 3;

    let poly_degree = 2;
    let num_mles = 2;
    let one: F = F::ONE;
    let const_evals = vec![*one.inner(); 1 << num_vars];
    let poly_mles: Vec<DenseMultilinearExtension<<F as Field>::Inner>> = (0..num_mles)
        .map(|_| {
            DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                const_evals.clone(),
                <F as Field>::Inner::ZERO,
            )
        })
        .collect();

    let num_evals = 1 << num_vars;
    let sum = F::from(num_evals);

    let comb_fn = |vals: &[F]| -> F { vals.iter().product() };

    let mut transcript = KeccakTranscript::default();
    let (proof, _) = MLSumcheck::prove_as_subprotocol(
        &mut transcript,
        poly_mles,
        num_vars,
        poly_degree,
        comb_fn,
        (),
    );

    let mut verifier_transcript = KeccakTranscript::default();
    let res = MLSumcheck::verify_as_subprotocol(
        &mut verifier_transcript,
        num_vars,
        poly_degree,
        sum,
        &proof,
        (),
    );

    assert!(res.is_ok());
}

#[test]
fn sumcheck_with_single_variable() {
    let mut rng = rand::rng();
    let num_vars = 1;

    let mut transcript = KeccakTranscript::default();
    let ((poly_mles, poly_degree), products, sum) =
        rand_poly(num_vars, 2..5, 7, &mut rng, &()).unwrap();

    let comb_fn = move |vals: &[F]| -> F { rand_poly_comb_fn(vals, &products, ()) };

    let (proof, _) = MLSumcheck::prove_as_subprotocol(
        &mut transcript,
        poly_mles,
        num_vars,
        poly_degree,
        comb_fn,
        (),
    );

    let mut verifier_transcript = KeccakTranscript::default();
    let res = MLSumcheck::verify_as_subprotocol(
        &mut verifier_transcript,
        num_vars,
        poly_degree,
        sum,
        &proof,
        (),
    );

    assert!(res.is_ok());
}

#[test]
fn subclaim_changes_if_transcript_is_tampered() {
    let mut rng = rand::rng();
    let num_vars = 3;

    let mut prover_transcript = KeccakTranscript::default();
    let ((poly_mles, poly_degree), products, sum) =
        rand_poly(num_vars, 2..5, 7, &mut rng, &()).unwrap();

    let comb_fn = move |vals: &[F]| -> F { rand_poly_comb_fn(vals, &products, ()) };

    let (proof, _) = MLSumcheck::prove_as_subprotocol(
        &mut prover_transcript,
        poly_mles,
        num_vars,
        poly_degree,
        comb_fn,
        (),
    );

    let mut clean_transcript = KeccakTranscript::default();
    let clean_res = MLSumcheck::verify_as_subprotocol(
        &mut clean_transcript,
        num_vars,
        poly_degree,
        sum,
        &proof,
        (),
    )
    .unwrap();

    let mut tampered_transcript = KeccakTranscript::default();
    tampered_transcript.absorb(b"tampering the transcript");
    let tampered_res = MLSumcheck::verify_as_subprotocol(
        &mut tampered_transcript,
        num_vars,
        poly_degree,
        sum,
        &proof,
        (),
    )
    .unwrap();

    assert_ne!(clean_res.point, tampered_res.point);
    assert_ne!(
        clean_res.expected_evaluation,
        tampered_res.expected_evaluation
    );
}

#[test]
#[should_panic(expected = "Prover is not active")]
fn prover_panics_if_round_exceeds_num_vars() {
    let num_vars = 3;

    let mut prover_state = ProverState {
        randomness: vec![F::ZERO; num_vars],
        mles: Vec::new(),
        num_vars,
        max_degree: 2,
        round: num_vars, // Set to the last valid round
        asserted_sum: None,
    };

    let comb_fn = |_vals: &[F]| F::ZERO;

    let verifier_msg = Some(F::ZERO);

    prover_state.prove_round(&verifier_msg, comb_fn, &());
}

#[test]
fn verifier_errors_on_incomplete_proof() {
    let mut rng = rand::rng();
    let num_vars = 3;

    let mut transcript = KeccakTranscript::default();
    let ((poly_mles, poly_degree), products, sum) =
        rand_poly(num_vars, 2..5, 7, &mut rng, &()).unwrap();

    let comb_fn = move |vals: &[F]| -> F { rand_poly_comb_fn(vals, &products, ()) };

    let (proof, _) = MLSumcheck::prove_as_subprotocol(
        &mut transcript,
        poly_mles,
        num_vars,
        poly_degree,
        comb_fn,
        (),
    );

    let mut incomplete_proof = proof.clone();
    incomplete_proof.messages.pop(); // Remove last prover message

    let mut verifier_transcript = KeccakTranscript::default();

    let res = MLSumcheck::verify_as_subprotocol(
        &mut verifier_transcript,
        num_vars,
        poly_degree,
        sum,
        &incomplete_proof,
        (),
    );

    assert!(
        matches!(res, Err(super::SumCheckError::InvalidProofLength { expected, got }) if expected == num_vars && got == num_vars - 1),
        "expected IncorrectRoundCount error"
    );
}

#[test]
fn prover_handles_empty_mle_list() {
    let num_vars = 3;

    let poly_mles: Vec<DenseMultilinearExtension<<F as Field>::Inner>> = Vec::new();
    let poly_degree = 0;
    let sum = F::ZERO;

    let comb_fn = |_vals: &[F]| -> F { F::ZERO };

    let mut transcript = KeccakTranscript::default();
    let (proof, _) = MLSumcheck::prove_as_subprotocol(
        &mut transcript,
        poly_mles,
        num_vars,
        poly_degree,
        comb_fn,
        (),
    );

    let mut verifier_transcript = KeccakTranscript::default();
    let res = MLSumcheck::verify_as_subprotocol(
        &mut verifier_transcript,
        num_vars,
        poly_degree,
        sum,
        &proof,
        (),
    );

    assert!(res.is_ok());
}

#[test]
fn verifier_errors_on_mismatched_nvars() {
    let mut rng = rand::rng();
    let nvars_prover = 3;
    let nvars_verifier = 4;

    let (poly_degree, sum, proof) = generate_sumcheck_proof(nvars_prover, &mut rng);

    let mut transcript = KeccakTranscript::default();
    let res = MLSumcheck::verify_as_subprotocol(
        &mut transcript,
        nvars_verifier, // verifier expects more rounds than the proof contains
        poly_degree,
        sum,
        &proof,
        (),
    );

    assert!(
        matches!(res, Err(super::SumCheckError::InvalidProofLength { expected, got })
            if expected == nvars_verifier && got == nvars_prover),
        "expected IncorrectRoundCount: expected {nvars_verifier}, got {res:?}"
    );
}

#[test]
fn verifier_produces_correct_subclaim() {
    let mut rng = rand::rng();
    let nvars = 3;

    let mut prover_transcript = KeccakTranscript::default();
    let ((poly_mles, poly_degree), products, sum) =
        rand_poly(nvars, 2..5, 7, &mut rng, &()).unwrap();

    let original_mles = poly_mles.clone();
    let products_for_verification = products.clone();

    let comb_fn = move |vals: &[F]| -> F { rand_poly_comb_fn(vals, &products, ()) };

    let (proof, _) = MLSumcheck::prove_as_subprotocol(
        &mut prover_transcript,
        poly_mles,
        nvars,
        poly_degree,
        comb_fn,
        (),
    );

    let mut verifier_transcript = KeccakTranscript::default();
    let subclaim = MLSumcheck::verify_as_subprotocol(
        &mut verifier_transcript,
        nvars,
        poly_degree,
        sum,
        &proof,
        (),
    )
    .unwrap();

    let mle_evals_at_point: Vec<F> = original_mles
        .iter()
        .map(|mle| mle.evaluate_with_config(&subclaim.point, &()).unwrap())
        .collect();

    let manual_eval = rand_poly_comb_fn(&mle_evals_at_point, &products_for_verification, ());

    assert_eq!(manual_eval, subclaim.expected_evaluation);
}

#[test]
#[should_panic(expected = "Attempt to verify a sumcheck claim for 0 variables")]
fn zero_variable_case_returns_correct_subclaim() {
    let num_vars = 0;
    let degree = 2;

    // Let's pick some arbitrary "claimed sum"
    let claimed_sum: F = F::from(42u32);

    // No prover rounds for zero-variable case
    let proof = SumcheckProof::<F> { messages: vec![], claimed_sum };

    let mut transcript = KeccakTranscript::default();
    let _subclaim = MLSumcheck::verify_as_subprotocol(
        &mut transcript,
        num_vars,
        degree,
        claimed_sum,
        &proof,
        (),
    );
}
