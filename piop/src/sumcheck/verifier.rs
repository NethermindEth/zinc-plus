//! Verifier
use std::convert::identity;

use ark_std::{boxed::Box, cfg_into_iter, vec::Vec};
use crypto_primitives::{FromPrimitiveWithConfig, PrimeField, Semiring};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use zinc_transcript::traits::{ConstTranscribable, Transcript};

use super::{IPForMLSumcheck, SumCheckError, prover::ProverMsg};

pub const SQUEEZE_NATIVE_ELEMENTS_NUM: usize = 1;

/// Verifier State
pub struct VerifierState<F: PrimeField> {
    pub round: usize,
    pub nv: usize,
    pub max_multiplicands: usize,
    pub finished: bool,
    /// a list storing the univariate polynomial in evaluation form sent by the
    /// prover at each round
    pub polynomials_received: Vec<Vec<F>>,
    /// a list storing the randomness sampled by the verifier at each round
    pub randomness: Vec<F>,
    pub config: F::Config,
}

impl<F: PrimeField> VerifierState<F> {
    pub fn new(nvars: usize, degree: usize, config: &F::Config) -> Self {
        Self {
            round: 1,
            nv: nvars,
            max_multiplicands: degree,
            finished: false,
            polynomials_received: Vec::with_capacity(nvars),
            randomness: Vec::with_capacity(nvars),
            config: config.clone(),
        }
    }
}

/// Subclaim when verifier is convinced
#[derive(Debug)]
pub struct SubClaim<F> {
    /// the multi-dimensional point that this multilinear extension is evaluated
    /// to
    pub point: Vec<F>,
    /// the expected evaluation
    pub expected_evaluation: F,
}

impl<F: FromPrimitiveWithConfig> IPForMLSumcheck<F> {
    /// Run verifier at current round, given prover message
    ///
    /// Normally, this function should perform actual verification. Instead,
    /// `verify_round` only samples and stores randomness and perform
    /// verifications altogether in `check_and_generate_subclaim` at
    /// the last step.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn verify_round(
        prover_msg: &ProverMsg<F>,
        verifier_state: &mut VerifierState<F>,
        transcript: &mut impl Transcript,
    ) -> F
    where
        F::Inner: ConstTranscribable,
    {
        if verifier_state.finished {
            panic!("Incorrect verifier state: Verifier is already finished.");
        }

        // Now, verifier should check if the received P(0) + P(1) = expected. The check
        // is moved to `check_and_generate_subclaim`, and will be done after the
        // last round.

        let msg: F = transcript.get_field_challenge(&verifier_state.config);
        verifier_state.randomness.push(msg.clone());
        verifier_state
            .polynomials_received
            .push(prover_msg.evaluations.clone());

        // Now, verifier should set `expected` to P(r).
        // This operation is also moved to `check_and_generate_subclaim`,
        // and will be done after the last round.

        if verifier_state.round == verifier_state.nv {
            // accept and close
            verifier_state.finished = true;
        } else {
            verifier_state.round += 1;
        }
        msg
    }

    /// Verify the sumcheck phase, and generate the subclaim.
    ///
    /// If the asserted sum is correct, then the multilinear polynomial
    /// evaluated at `subclaim.point` is `subclaim.expected_evaluation`.
    /// Otherwise, it is highly unlikely that those two will be equal.
    /// Larger field size guarantees smaller soundness error.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn check_and_generate_subclaim(
        verifier_state: VerifierState<F>,
        asserted_sum: F,
        config: &F::Config,
    ) -> Result<SubClaim<F>, SumCheckError<F>> {
        if !verifier_state.finished {
            panic!("Verifier has not finished.");
        }

        let mut expected = asserted_sum;
        if verifier_state.polynomials_received.len() != verifier_state.nv {
            panic!("insufficient rounds");
        }
        for i in 0..verifier_state.nv {
            let evaluations = &verifier_state.polynomials_received[i];
            if evaluations.len() != verifier_state.max_multiplicands + 1 {
                return Err(SumCheckError::MaxDegreeExceeded);
            }

            let p0 = &evaluations[0];
            if verifier_state.max_multiplicands > 0 {
                let p1 = &evaluations[1];
                if p0.clone() + p1.clone() != expected {
                    return Err(SumCheckError::SumCheckFailed(
                        Box::new(p0.clone() + p1.clone()),
                        Box::new(expected),
                    ));
                }
            } else {
                // Degree 0, constant polynomial
                if p0.clone() != expected {
                    return Err(SumCheckError::SumCheckFailed(
                        Box::new(p0.clone()),
                        Box::new(expected),
                    ));
                }
            }

            expected =
                interpolate_uni_poly(evaluations, verifier_state.randomness[i].clone(), config);
        }

        Ok(SubClaim {
            point: verifier_state.randomness,
            expected_evaluation: expected,
        })
    }
}

/// interpolate the *unique* univariate polynomial of degree *at most*
/// p_i.len()-1 passing through the y-values in p_i at x = 0,..., p_i.len()-1
/// and evaluate this  polynomial at `eval_at`. In other words, efficiently
/// compute  \sum_{i=0}^{len p_i - 1} p_i[i] * (\prod_{j!=i} (eval_at -
/// j)/(i-j))
// All the arithmetic ops in the function
// are made sure to not overflow.
#[allow(clippy::arithmetic_side_effects)]
#[allow(clippy::cast_possible_wrap)]
pub(crate) fn interpolate_uni_poly<F: FromPrimitiveWithConfig>(
    p_i: &[F],
    x: F,
    config: &F::Config,
) -> F {
    // TODO(Alex): Once we have benches, it's worth checking
    //             if we're even winning anything
    //             with specialized branches above.

    // We will need these a few times
    let zero = F::zero_with_cfg(config);
    let one = F::one_with_cfg(config);

    let len = p_i.len();

    let mut evals = vec![];

    let mut prod = x.clone();
    evals.push(x.clone());

    //`prod = \prod_{j} (x - j)`
    // we return early if 0 <= x < len, i.e. if the desired value has been passed
    let mut j = zero.clone();
    for i in 1..len {
        if x == j {
            return p_i[i - 1].clone();
        }
        j += &one;

        let tmp = x.clone() - j.clone();
        evals.push(tmp.clone());
        prod *= tmp;
    }

    if x == j {
        return p_i[len - 1].clone();
    }

    let mut res = zero;
    // we want to compute \prod (j!=i) (i-j) for a given i
    //
    // we start from the last step, which is
    //  denom[len-1] = (len-1) * (len-2) *... * 2 * 1
    // the step before that is
    //  denom[len-2] = (len-2) * (len-3) * ... * 2 * 1 * -1
    // and the step before that is
    //  denom[len-3] = (len-3) * (len-4) * ... * 2 * 1 * -1 * -2
    //
    // i.e., for any i, the one before this will be derived from
    //  denom[i-1] = - denom[i] * (len-i) / i
    //
    // that is, we only need to store
    // - the last denom for i = len-1, and
    // - the ratio between the current step and the last step, which is the product
    //   of -(len-i) / i from all previous steps and we store this product as a
    //   fraction number to reduce field divisions.

    // We know
    //  - 2^61 < factorial(20) < 2^62
    //  - 2^122 < factorial(33) < 2^123
    // so we will be able to compute the ratio
    //  - for len <= 20 with i64
    //  - for len <= 33 with i128
    //  - for len >  33 with BigInt
    if p_i.len() <= 20 {
        let last_denom: F = F::from_with_cfg(factorial(len - 1, identity), config);

        let mut ratio_numerator = 1i64;
        let mut ratio_denominator = 1u64;

        for i in (0..len).rev() {
            let ratio_numerator_f = F::from_with_cfg(ratio_numerator, config);

            let ratio_denominator_f = F::from_with_cfg(ratio_denominator, config);

            let x = prod.clone() * ratio_denominator_f
                / (last_denom.clone() * ratio_numerator_f * &evals[i]);

            res += &(p_i[i].clone() * x);

            // compute ratio for the next step which is current_ratio * -(len-i)/i
            if i != 0 {
                // Using intentionally, overflow isn't possible
                ratio_numerator *= -(len as i64 - i as i64);
                ratio_denominator *= i as u64;
            }
        }
    } else if p_i.len() <= 33 {
        let last_denom = F::from_with_cfg(factorial(len - 1, u128::from), config);
        let mut ratio_numerator = 1i128;
        let mut ratio_denominator = 1u128;

        for i in (0..len).rev() {
            let ratio_numerator_f = F::from_with_cfg(ratio_numerator, config);

            let ratio_denominator_f = F::from_with_cfg(ratio_denominator, config);

            let x: F = prod.clone() * ratio_denominator_f
                / (last_denom.clone() * ratio_numerator_f * &evals[i]);
            res += &(p_i[i].clone() * x);

            // compute ratio for the next step which is current_ratio * -(len-i)/i
            if i != 0 {
                ratio_numerator *= -(len as i128 - i as i128);
                ratio_denominator *= i as u128;
            }
        }
    } else {
        // since we are using field operations, we can merge
        // `last_denom` and `ratio_numerator` into a single field element.
        let mut denom_up = factorial(len - 1, |u| F::from_with_cfg(u, config));
        let mut denom_down = one;

        for i in (0..len).rev() {
            let x = prod.clone() * &denom_down / (denom_up.clone() * &evals[i]);
            res += &(p_i[i].clone() * x);

            // compute denom for the next step is -current_denom * (len-i)/i
            if i != 0 {
                let denom_up_factor = F::from_with_cfg((len - i) as u64, config);
                denom_up *= -denom_up_factor;

                let denom_down_factor = F::from_with_cfg(i as u64, config);
                denom_down *= denom_down_factor;
            }
        }
    }

    res
}

/// Compute the factorial(a) = 1 * 2 * ... * a.
fn factorial<R, F>(a: usize, from_u64: F) -> R
where
    R: Semiring,
    F: Fn(u64) -> R + Send + Sync,
{
    cfg_into_iter!((1..=(a as u64))).map(from_u64).product()
}
