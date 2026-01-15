use crypto_primitives::{FixedSemiring, Semiring};
use itertools::Itertools;
use std::marker::PhantomData;
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
pub struct Proof<R> {
    pub combined_mle_values: Vec<R>,
}

#[derive(Clone, Debug)]
pub struct ProverState<R, C> {
    pub evaluation_points: Vec<Vec<C>>,
    pub combined_mles: Vec<DenseMultilinearExtension<R>>,
}

pub struct SubClaim<R, C> {
    pub point: Vec<C>,
    pub value: R,
}

pub type Result<T, R, I> = std::result::Result<T, IdealCheckError<R, I>>;

pub struct IdealCheckProtocol<R, C>(PhantomData<(R, C)>);

impl<R, C> IdealCheckProtocol<R, C>
where
    R: FixedSemiring + for<'a> MulByScalar<&'a C> + ConstTranscribable,
    C: ConstTranscribable,
{
    #[allow(clippy::type_complexity)]
    pub fn prove_as_subprotocol<U: Uair<R>>(
        transcript: &mut impl Transcript,
        cs_up: &[DenseMultilinearExtension<R>],
        cs_down: &[DenseMultilinearExtension<R>],
        num_constraints: usize,
        num_vars: usize,
    ) -> Result<(Proof<R>, ProverState<R, C>), R, U::Ideal> {
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
            Proof {
                combined_mle_values,
            },
            ProverState {
                evaluation_points,
                combined_mles,
            },
        ))
    }

    #[allow(clippy::type_complexity)]
    pub fn verify_as_subprotocol<U: Uair<R>>(
        transcript: &mut impl Transcript,
        proof: Proof<R>,
        num_constraints: usize,
        num_vars: usize,
    ) -> Result<Vec<SubClaim<R, C>>, R, U::Ideal> {
        let mut transcription_buf: Vec<u8> = vec![0; R::NUM_BYTES];

        let mut evaluation_points: Vec<Vec<C>> = Vec::with_capacity(num_constraints);
        let combined_mle_values = proof.combined_mle_values;

        for mle_value in &combined_mle_values {
            let challenge: Vec<C> = transcript.get_challenges(num_vars);

            mle_value.write_transcription_bytes(&mut transcription_buf);
            transcript.absorb(&transcription_buf);

            evaluation_points.push(challenge);
        }

        let mut ideal_collector = IdealCollector::<_, U::Ideal>::new(num_constraints);

        let dummy_up_and_down: Vec<DummySemiring> = vec![DummySemiring; num_constraints];

        U::constrain(&mut ideal_collector, &dummy_up_and_down, &dummy_up_and_down);

        ideal_collector
            .ideals
            .iter()
            .zip(combined_mle_values.iter())
            .try_for_each(|(ideal, mle_value)| {
                if !ideal.contains(mle_value) {
                    return Err(IdealCheckError::IdealCheckFailed(
                        mle_value.clone(),
                        ideal.clone(),
                    ));
                }

                Ok(())
            })?;

        Ok(evaluation_points
            .into_iter()
            .zip(combined_mle_values)
            .map(|(point, value)| SubClaim { point, value })
            .collect())
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

        (0..len).for_each(|i| {
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

impl<R: FixedSemiring> ConstraintBuilder<R> for IdealCheckConstraintBuilder<R> {
    type Expr = R;
    // Ignore all ideal business on the side of the prover.
    type Ideal = DummyIdeal<Self::Expr>;

    #[allow(clippy::arithmetic_side_effects)]
    fn assert_in_ideal(&mut self, expr: Self::Expr, _ideal: &Self::Ideal) {
        self.uair_poly_mles_coeffs.push(expr);
    }
}

pub(crate) struct IdealCollector<R: Semiring, I: Ideal<R>> {
    pub ideals: Vec<I>,
    _phantom: PhantomData<R>,
}

impl<R: Semiring, I: Ideal<R>> IdealCollector<R, I> {
    pub fn new(num_constraints: usize) -> Self {
        Self {
            ideals: Vec::with_capacity(num_constraints),
            _phantom: Default::default(),
        }
    }
}

impl<R: FixedSemiring, I: Ideal<R>> ConstraintBuilder<R> for IdealCollector<R, I> {
    type Expr = DummySemiring;
    type Ideal = I;

    fn assert_in_ideal(&mut self, _expr: Self::Expr, ideal: &Self::Ideal) {
        self.ideals.push(ideal.clone());
    }
}

#[derive(Clone, Debug, Error)]
pub enum IdealCheckError<R, I> {
    #[error("ideal check prover failed to evaluate an mle: {0}")]
    MleEvaluationError(EvaluationError),
    #[error("the combined mle evaluation {0} does not belong to the ideal {1}")]
    IdealCheckFailed(R, I),
}

impl<R, I> From<EvaluationError> for IdealCheckError<R, I> {
    fn from(error: EvaluationError) -> Self {
        Self::MleEvaluationError(error)
    }
}

#[cfg(test)]
mod tests {
    use crypto_primitives::{FixedSemiring, crypto_bigint_int::Int};
    use itertools::Itertools;
    use num_traits::ConstZero;
    use zinc_poly::mle::DenseMultilinearExtension;
    use zinc_uair::{
        ConstraintBuilder, Uair,
        constraint_counter::count_constraints,
        ideal::{Ideal, ZeroIdeal},
    };

    use crate::ideal_check::IdealCheckProtocol;

    struct TestUair;

    impl<R: FixedSemiring> Uair<R> for TestUair {
        type Ideal = ZeroIdeal<R>;

        fn num_cols() -> usize {
            3
        }

        #[allow(clippy::arithmetic_side_effects)]
        fn constrain<B: ConstraintBuilder<R>>(b: &mut B, up: &[B::Expr], down: &[B::Expr]) {
            b.assert_in_ideal(up[0].clone() * &down[1] - &up[1], &B::Ideal::zero_ideal());
            b.assert_in_ideal(up[2].clone(), &B::Ideal::zero_ideal());
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

        let mles = IdealCheckProtocol::<_, i128>::get_combined_poly_mles::<TestUair>(
            &up,
            &down,
            count_constraints::<Int<4>, TestUair>(),
            4,
        );

        assert_eq!(mles.len(), count_constraints::<Int<4>, TestUair>());

        assert_eq!(&mles[0], &(up[0].clone() * &down[1] - &up[1]));
        assert_eq!(&mles[1], &up[2]);
    }
}
