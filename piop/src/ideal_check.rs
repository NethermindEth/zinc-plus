use std::marker::PhantomData;

use ark_std::cfg_into_iter;
use crypto_primitives::{FixedSemiring, Semiring};
use itertools::Itertools;
use thiserror::Error;
use zinc_poly::{EvaluationError, mle::DenseMultilinearExtension};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_uair::{
    ConstraintBuilder, Uair,
    dummy_semiring::DummySemiring,
    ideal::{DummyIdeal, Ideal},
};
use zinc_utils::mul_by_scalar::MulByScalar;

#[derive(Clone, Debug)]
pub struct IdealCheckProof<R> {
    pub combined_mle_values: Vec<R>,
}

#[derive(Clone, Debug)]
pub struct IdealCheckProverState<R, C> {
    pub evaluation_points: Vec<Vec<C>>,
    pub combined_mles: Vec<DenseMultilinearExtension<R>>,
}

pub struct IdealCheckProtocol<R, C>(PhantomData<(R, C)>);

impl<R, C> IdealCheckProtocol<R, C>
where
    R: FixedSemiring + for<'a> MulByScalar<&'a C> + ConstTranscribable,
    C: ConstTranscribable,
{
    pub fn prove_as_subprotocol<U: Uair<R>>(
        transcript: &mut impl Transcript,
        cs_up: &[DenseMultilinearExtension<R>],
        cs_down: &[DenseMultilinearExtension<R>],
        num_constraints: usize,
        num_vars: usize,
    ) -> Result<(IdealCheckProof<R>, IdealCheckProverState<R, C>), IdealCheckError> {
        let combined_mles =
            Self::get_combined_poly_mles::<U>(cs_up, cs_down, num_constraints, num_vars);
        let mut transcription_buf: Vec<u8> = vec![0; R::NUM_BYTES];

        let mut evaluation_points: Vec<Vec<C>> = Vec::with_capacity(num_constraints);
        let mut combined_mle_values: Vec<R> = Vec::with_capacity(num_constraints);

        for combined_mle in &combined_mles {
            let challenge: Vec<C> = transcript.get_challenges(num_vars);

            let mle_value = combined_mle.evaluate(&challenge, R::zero())?;

            mle_value.write_transcription_bytes(&mut transcription_buf);
            transcript.absorb(&transcription_buf);

            evaluation_points.push(challenge);
            combined_mle_values.push(mle_value);
        }

        Ok((
            IdealCheckProof {
                combined_mle_values,
            },
            IdealCheckProverState {
                evaluation_points,
                combined_mles,
            },
        ))
    }

    pub fn verify_as_subprotocol<U: Uair<R>>(
        transcript: &mut impl Transcript,
        proof: IdealCheckProof<R>,
        num_constraints: usize,
        num_vars: usize,
    ) -> Result<(), IdealCheckError> {
        let mut transcription_buf: Vec<u8> = vec![0; R::NUM_BYTES];

        let mut evaluation_points: Vec<Vec<C>> = Vec::with_capacity(num_constraints);
        let mut combined_mle_values: Vec<R> = Vec::with_capacity(num_constraints);

        for mle_value in proof.combined_mle_values {
            let challenge: Vec<C> = transcript.get_challenges(num_vars);

            mle_value.write_transcription_bytes(&mut transcription_buf);
            transcript.absorb(&transcription_buf);

            evaluation_points.push(challenge);
            combined_mle_values.push(mle_value);
        }

        Ok(())
    }

    fn get_combined_poly_mles<U: Uair<R>>(
        cs_up: &[DenseMultilinearExtension<R>],
        cs_down: &[DenseMultilinearExtension<R>],
        num_constraints: usize,
        num_vars: usize,
    ) -> Vec<DenseMultilinearExtension<R>> {
        // Collect h MLEs.
        let len = cs_up[0].evaluations.len();

        let mut h_evals: Vec<Vec<R>> = (0..num_constraints)
            .map(|_| Vec::with_capacity(len))
            .collect_vec();

        let pointers: Vec<*mut R> = h_evals.iter_mut().map(|col| col.as_mut_ptr()).collect_vec();

        cfg_into_iter!(0..len).for_each(|i| {
            let mut builder = IdealCheckConstraintBuilder::new(num_constraints);

            let up = cs_up
                .iter()
                .map(|up| up.evaluations[i].clone())
                .collect_vec();
            let down = cs_down
                .iter()
                .map(|down| down.evaluations[i].clone())
                .collect_vec();

            U::constrain(&mut builder, &up, &down);

            pointers
                .iter()
                .zip(builder.uair_poly_mles_coeffs)
                .for_each(|(ptr, eval)| unsafe {
                    *ptr.add(i) = eval;
                });
        });

        h_evals
            .into_iter()
            .map(|mut evaluations| {
                unsafe {
                    evaluations.set_len(len);
                }

                DenseMultilinearExtension {
                    evaluations,
                    num_vars,
                }
            })
            .collect()
    }
}

pub(crate) struct IdealCheckConstraintBuilder<R: Semiring> {
    pub uair_poly_mles_coeffs: Vec<R>,
}

impl<R: Semiring> IdealCheckConstraintBuilder<R> {
    pub fn new(num_constraints: usize) -> Self {
        Self {
            uair_poly_mles_coeffs: Vec::with_capacity(num_constraints),
        }
    }
}

impl<R: FixedSemiring> ConstraintBuilder for IdealCheckConstraintBuilder<R> {
    type Expr = R;
    // Ignore all ideal business on the side of the prover.
    type Ideal = DummyIdeal<Self::Expr>;

    #[allow(clippy::arithmetic_side_effects)]
    fn assert_in_ideal(&mut self, expr: Self::Expr, _ideal: &Self::Ideal) {
        self.uair_poly_mles_coeffs.push(expr);
    }
}

pub(crate) struct IdealCollector<R: Semiring> {
    pub ideals: Vec<R>,
}

impl<R: Semiring> IdealCollector<R> {
    pub fn new(num_constraints: usize) -> Self {
        Self {
            ideals: Vec::with_capacity(num_constraints),
        }
    }
}

impl<R> ConstraintBuilder for IdealCollector<R>
where
    R: FixedSemiring,
{
    type Expr = DummySemiring;
    type RingForIdeals = R;

    fn assert_in_ideal(&mut self, _expr: Self::Expr, ideal_generator: &Self::Expr) {
        self.ideals.push(ideal_generator.clone());
    }
}

#[derive(Clone, Debug, Error)]
pub enum IdealCheckError {
    #[error("ideal check prover failed to evaluate an mle: {0}")]
    MleEvaluationError(EvaluationError),
}

impl From<EvaluationError> for IdealCheckError {
    fn from(error: EvaluationError) -> Self {
        Self::MleEvaluationError(error)
    }
}

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;

    use crypto_primitives::{FixedSemiring, crypto_bigint_int::Int};
    use itertools::Itertools;
    use num_traits::{ConstZero, One, Zero};
    use zinc_poly::mle::DenseMultilinearExtension;
    use zinc_uair::{Uair, constraint_counter::count_constraints};

    use crate::ideal_check::IdealCheckProtocol;

    struct TestUair<R>(PhantomData<R>);

    impl<R: FixedSemiring> Uair<R> for TestUair<R> {
        fn num_cols() -> usize {
            3
        }

        #[allow(clippy::arithmetic_side_effects)]
        fn constrain<B: zinc_uair::ConstraintBuilder>(b: &mut B, up: &[B::Expr], down: &[B::Expr]) {
            b.assert_in_ideal(up[0].clone() * &down[1] - &up[1], &B::Expr::one());
            b.assert_in_ideal(up[2].clone(), &B::Expr::zero());
        }
    }

    #[test]
    fn test_get_ideals_and_combined_poly_mles() {
        let up: Vec<DenseMultilinearExtension<Int<4>>> = vec![
            DenseMultilinearExtension::from_evaluations_slice(
                4,
                &(0..4).map(Int::from_i64).collect_vec(),
                Int::ZERO,
            ),
            DenseMultilinearExtension::from_evaluations_slice(
                4,
                &(4..8).map(Int::from_i64).collect_vec(),
                Int::ZERO,
            ),
            DenseMultilinearExtension::from_evaluations_slice(
                4,
                &(8..12).map(Int::from_i64).collect_vec(),
                Int::ZERO,
            ),
        ];
        let down: Vec<DenseMultilinearExtension<Int<4>>> = vec![
            DenseMultilinearExtension::from_evaluations_slice(
                4,
                &(12..16).map(Int::from_i64).collect_vec(),
                Int::ZERO,
            ),
            DenseMultilinearExtension::from_evaluations_slice(
                4,
                &(16..20).map(Int::from_i64).collect_vec(),
                Int::ZERO,
            ),
            DenseMultilinearExtension::from_evaluations_slice(
                4,
                &(20..24).map(Int::from_i64).collect_vec(),
                Int::ZERO,
            ),
        ];

        let mles = IdealCheckProtocol::<_, i128>::get_combined_poly_mles::<TestUair<_>>(
            &up,
            &down,
            count_constraints::<Int<4>, TestUair<_>>(),
            4,
        );

        assert_eq!(mles.len(), count_constraints::<Int<4>, TestUair<_>>());

        assert_eq!(&mles[0], &(up[0].clone() * &down[1] - &up[1]));
        assert_eq!(&mles[1], &up[2]);
    }
}
