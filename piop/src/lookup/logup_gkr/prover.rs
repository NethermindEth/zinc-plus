//! Logup-GKR prover.

use crypto_primitives::{FromPrimitiveWithConfig, PrimeField};
use num_traits::Zero;
use std::marker::PhantomData;
use zinc_poly::{mle::MultilinearExtensionWithConfig, utils::build_eq_x_r_inner};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_utils::inner_transparent_field::InnerTransparentField;

use crate::{CombFn, sumcheck::MLSumcheck};

use super::{
    circuit::{GrandSumCircuit, GrandSumLayer, split_last_variable},
    proof::{LogupGkrProof, LogupGkrRoundProof},
};

/// Result returned by the prover for the caller to reconcile the leaf
/// evaluations against the real leaf MLEs.
#[derive(Clone, Debug)]
pub struct LogupGkrSubclaim<F> {
    /// Evaluation point at the leaf level (length = `num_leaf_vars`).
    pub point: Vec<F>,
    /// `N_leaves(point)` as derived from the protocol.
    pub numerator_eval: F,
    /// `D_leaves(point)` as derived from the protocol.
    pub denominator_eval: F,
}

/// Logup-GKR prover entry point.
pub struct LogupGkrProver<F>(PhantomData<F>);

impl<F> LogupGkrProver<F>
where
    F: PrimeField + InnerTransparentField + FromPrimitiveWithConfig + Send + Sync + 'static,
    F::Inner: ConstTranscribable + Zero + Send + Sync,
    F::Modulus: ConstTranscribable,
{
    /// Prove `sum_{x in {0,1}^n} N(x) / D(x) = root_n / root_d`.
    ///
    /// The caller is responsible for subsequently checking that the
    /// returned subclaim's `(numerator_eval, denominator_eval)` agree
    /// with the leaf MLEs at `point`.
    pub fn prove(
        transcript: &mut impl Transcript,
        circuit: &GrandSumCircuit<F>,
        cfg: &F::Config,
    ) -> (LogupGkrProof<F>, LogupGkrSubclaim<F>) {
        let n_leaves = circuit.num_leaf_vars();
        assert!(n_leaves >= 1, "logup-GKR requires at least 1 leaf variable");

        // Absorb circuit shape into transcript.
        let mut buf = vec![0u8; F::Inner::NUM_BYTES];
        let nvars_f = F::from_with_cfg(n_leaves as u64, cfg);
        transcript.absorb_random_field(&nvars_f, &mut buf);

        // Root is a scalar (0 vars). Expose (root_n, root_d) to the verifier.
        let root = circuit.root();
        let root_numerator =
            F::new_unchecked_with_cfg(root.numerator.evaluations[0].clone(), cfg);
        let root_denominator =
            F::new_unchecked_with_cfg(root.denominator.evaluations[0].clone(), cfg);
        transcript.absorb_random_field(&root_numerator, &mut buf);
        transcript.absorb_random_field(&root_denominator, &mut buf);

        // Descent: for each layer from the one just below the root down
        // to the leaves, sample lambda, run sumcheck, send tail values, sample beta.
        let mut round_proofs = Vec::with_capacity(n_leaves);
        let mut current_point: Vec<F> = Vec::new();
        let mut current_n = root_numerator.clone();
        let mut current_d = root_denominator.clone();

        for layer_idx in 1..=n_leaves {
            // layer_idx 1 means the layer just below root. Parent is layer_idx - 1.
            // Child at this layer has layer_idx variables. Its parent (current) has
            // layer_idx - 1 = nvp variables.
            let child = &circuit.layers[layer_idx];
            let nvp = layer_idx - 1;

            let (round_proof, next_point, next_n, next_d) = prove_one_layer::<F>(
                transcript,
                nvp,
                &current_point,
                current_n.clone(),
                current_d.clone(),
                child,
                cfg,
                &mut buf,
            );

            round_proofs.push(round_proof);
            current_point = next_point;
            current_n = next_n;
            current_d = next_d;
        }

        let subclaim = LogupGkrSubclaim {
            point: current_point,
            numerator_eval: current_n,
            denominator_eval: current_d,
        };

        (
            LogupGkrProof {
                root_numerator,
                root_denominator,
                round_proofs,
            },
            subclaim,
        )
    }
}

#[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
fn prove_one_layer<F>(
    transcript: &mut impl Transcript,
    nvp: usize,
    current_point: &[F],
    current_n: F,
    current_d: F,
    child: &GrandSumLayer<F::Inner>,
    cfg: &F::Config,
    buf: &mut Vec<u8>,
) -> (LogupGkrRoundProof<F>, Vec<F>, F, F)
where
    F: PrimeField + InnerTransparentField + FromPrimitiveWithConfig + Send + Sync + 'static,
    F::Inner: ConstTranscribable + Zero + Send + Sync,
    F::Modulus: ConstTranscribable,
{
    // Sample the batching challenge lambda.
    let lambda: F = transcript.get_field_challenge(cfg);
    transcript.absorb_random_field(&lambda, buf);

    // Split the child MLE by its highest variable.
    let zero_inner = F::zero_with_cfg(cfg).into_inner();
    let (n0_mle, n1_mle) = split_last_variable(&child.numerator, &zero_inner);
    let (d0_mle, d1_mle) = split_last_variable(&child.denominator, &zero_inner);

    let (sumcheck_point, n0_eval, n1_eval, d0_eval, d1_eval, sumcheck_proof) = if nvp == 0 {
        // Degenerate case: no sumcheck, values are direct MLE entries.
        let n0 = F::new_unchecked_with_cfg(n0_mle.evaluations[0].clone(), cfg);
        let n1 = F::new_unchecked_with_cfg(n1_mle.evaluations[0].clone(), cfg);
        let d0 = F::new_unchecked_with_cfg(d0_mle.evaluations[0].clone(), cfg);
        let d1 = F::new_unchecked_with_cfg(d1_mle.evaluations[0].clone(), cfg);
        (Vec::<F>::new(), n0, n1, d0, d1, None)
    } else {
        // Build eq(x, current_point) MLE (nvp vars).
        let eq_mle = build_eq_x_r_inner::<F>(current_point, cfg)
            .expect("build_eq_x_r: non-empty current_point when nvp > 0");

        // Keep clones to evaluate after the sumcheck consumes the MLEs.
        let n0_for_sc = n0_mle.clone();
        let n1_for_sc = n1_mle.clone();
        let d0_for_sc = d0_mle.clone();
        let d1_for_sc = d1_mle.clone();

        // MLE order inside the sumcheck combiner: [eq, n0, n1, d0, d1].
        let mles = vec![eq_mle, n0_for_sc, n1_for_sc, d0_for_sc, d1_for_sc];
        let lambda_for_comb = lambda.clone();
        let comb_fn: CombFn<F> = Box::new(move |v: &[F]| {
            // eq * [lambda * (n0 * d1 + n1 * d0) + d0 * d1]
            let num_part = v[1].clone() * &v[4] + v[2].clone() * &v[3];
            let den_part = v[3].clone() * &v[4];
            v[0].clone() * (lambda_for_comb.clone() * &num_part + den_part)
        });

        let (proof, prover_state) =
            MLSumcheck::<F>::prove_as_subprotocol(transcript, mles, nvp, 3, comb_fn, cfg);

        // Extract x_star from the sumcheck's final randomness.
        let x_star: Vec<F> = prover_state.randomness.clone();
        debug_assert_eq!(x_star.len(), nvp);

        // Evaluate the original (unconsumed) split MLEs at x_star.
        let n0 = n0_mle
            .evaluate_with_config(&x_star, cfg)
            .expect("split_last_variable + evaluate_with_config: dimensions match");
        let n1 = n1_mle
            .evaluate_with_config(&x_star, cfg)
            .expect("split_last_variable + evaluate_with_config: dimensions match");
        let d0 = d0_mle
            .evaluate_with_config(&x_star, cfg)
            .expect("split_last_variable + evaluate_with_config: dimensions match");
        let d1 = d1_mle
            .evaluate_with_config(&x_star, cfg)
            .expect("split_last_variable + evaluate_with_config: dimensions match");

        (x_star, n0, n1, d0, d1, Some(proof))
    };

    // Sanity (nvp == 0 only): the root fold identity
    //   lambda * parent_n + parent_d == lambda * (n0*d1 + n1*d0) + d0*d1
    // must hold directly since there's no sumcheck to mediate.
    #[cfg(debug_assertions)]
    if nvp == 0 {
        let num_part = n0_eval.clone() * &d1_eval + n1_eval.clone() * &d0_eval;
        let den_part = d0_eval.clone() * &d1_eval;
        let rhs = lambda.clone() * &num_part + den_part;
        let lhs = lambda.clone() * &current_n + current_d.clone();
        debug_assert_eq!(lhs, rhs, "logup-GKR root fold identity mismatch");
    }

    // Absorb tail values (4 field elements).
    transcript.absorb_random_field(&n0_eval, buf);
    transcript.absorb_random_field(&n1_eval, buf);
    transcript.absorb_random_field(&d0_eval, buf);
    transcript.absorb_random_field(&d1_eval, buf);

    // Sample beta for the new last variable of the child.
    let beta: F = transcript.get_field_challenge(cfg);
    transcript.absorb_random_field(&beta, buf);

    // Child claim via linear interpolation along the last variable:
    // N_c(x*, beta) = (1 - beta) * n0 + beta * n1.
    let one_minus_beta = {
        let one = F::from_with_cfg(1u64, cfg);
        one - &beta
    };
    let next_n = one_minus_beta.clone() * &n0_eval + beta.clone() * &n1_eval;
    let next_d = one_minus_beta * &d0_eval + beta.clone() * &d1_eval;

    // New point: append beta as the new last coordinate.
    let mut next_point = sumcheck_point;
    next_point.push(beta);

    (
        LogupGkrRoundProof {
            numerator_0: n0_eval,
            numerator_1: n1_eval,
            denominator_0: d0_eval,
            denominator_1: d1_eval,
            sumcheck_proof,
        },
        next_point,
        next_n,
        next_d,
    )
}
