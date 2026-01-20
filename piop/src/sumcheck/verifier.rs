//! Verifier

use ark_std::vec::Vec;
use crypto_primitives::{FromPrimitiveWithConfig, PrimeField};
use zinc_poly::{EvaluatablePolynomial, univariate::nat_evaluation::NatEvaluatedPoly};
use zinc_transcript::traits::{ConstTranscribable, Transcript};

use crate::sumcheck::prover::{NatEvaluatedPolyWithoutConstant, ProverMsg};

use super::SumCheckError;

pub const SQUEEZE_NATIVE_ELEMENTS_NUM: usize = 1;

/// Sumcheck Verifier State.
pub struct VerifierState<F: PrimeField> {
    /// The current round number.
    pub round: usize,
    /// The number of variables the sumcheck polynomial
    /// is in.
    pub nv: usize,
    /// The degree of the polynomial.
    pub max_multiplicands: usize,
    /// `true` if the protocol has finished.
    pub finished: bool,
    /// A list storing the univariate polynomial in evaluation form sent by the
    /// prover at each round so far.
    pub polynomials_received: Vec<NatEvaluatedPolyWithoutConstant<F>>,
    /// A list storing the randomness sampled by the verifier at each round so
    /// far.
    pub randomness: Vec<F>,
    /// The field configuration to which
    /// all the field elements belong to.
    pub config: F::Config,
}

impl<F: PrimeField> VerifierState<F> {
    /// Initialize the verifier state.
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
    /// The multi-dimensional point that this multilinear extension is evaluated
    /// at.
    pub point: Vec<F>,
    /// The expected evaluation.
    pub expected_evaluation: F,
}

impl<F: FromPrimitiveWithConfig> VerifierState<F> {
    /// Run verifier at current round, given prover message.
    ///
    /// Normally, this function should perform actual verification. Instead,
    /// `verify_round` only samples and stores randomness and perform
    /// verifications altogether in `check_and_generate_subclaim` at
    /// the last step.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn verify_round(&mut self, prover_msg: &ProverMsg<F>, transcript: &mut impl Transcript) -> F
    where
        F::Inner: ConstTranscribable,
    {
        if self.finished {
            panic!("Incorrect verifier state: Verifier is already finished.");
        }

        // The constant term is omitted from prover messages. The verifier stores the
        // provided evaluations and will reconstruct the missing value in
        // `check_and_generate_subclaim`.

        let msg: F = transcript.get_field_challenge(&self.config);
        self.randomness.push(msg.clone());
        self.polynomials_received.push(prover_msg.0.clone());

        // Now, verifier should set `expected` to P(r).
        // This operation is also moved to `check_and_generate_subclaim`,
        // and will be done after the last round.

        if self.round == self.nv {
            // accept and close
            self.finished = true;
        } else {
            self.round += 1;
        }
        msg
    }

    /// Verify the sumcheck phase, and generate the subclaim.
    ///
    /// The verifier reconstructs the missing constant term under the
    /// assumption that `P(0) + P(1) == expected`. If the asserted sum is
    /// correct, then the multilinear polynomial evaluated at `subclaim.point`
    /// is `subclaim.expected_evaluation`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn check_and_generate_subclaim(
        self,
        asserted_sum: F,
    ) -> Result<SubClaim<F>, SumCheckError<F>> {
        if !self.finished {
            panic!("Verifier has not finished.");
        }

        let mut expected = asserted_sum;
        if self.polynomials_received.len() != self.nv {
            panic!("insufficient rounds");
        }
        for (i, evaluations_without_constant) in self.polynomials_received.iter().enumerate() {
            let expected_len = if self.max_multiplicands == 0 {
                0
            } else {
                self.max_multiplicands
            };
            if evaluations_without_constant.len() != expected_len {
                return Err(SumCheckError::MaxDegreeExceeded);
            }

            let constant_term = if self.max_multiplicands == 0 {
                expected.clone()
            } else {
                let p1 = evaluations_without_constant
                    .first()
                    .expect("degree > 0 implies the polynomial has an evaluation at 1");
                expected.clone() - p1.clone()
            };
            let mut reconstructed_evaluations =
                Vec::with_capacity(evaluations_without_constant.len() + 1);
            reconstructed_evaluations.push(constant_term);
            reconstructed_evaluations.extend_from_slice(evaluations_without_constant);

            let reconstructed_poly = NatEvaluatedPoly::new(reconstructed_evaluations);
            expected = reconstructed_poly.evaluate_at_point(&self.randomness[i])?;
        }

        Ok(SubClaim {
            point: self.randomness,
            expected_evaluation: expected,
        })
    }
}
