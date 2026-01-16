use ark_std::{cfg_into_iter, cfg_iter};
use crypto_primitives::{FixedSemiring, PrimeField, Semiring};
use itertools::Itertools;
use std::marker::PhantomData;
use thiserror::Error;
use zinc_poly::{
    EvaluationError,
    mle::{DenseMultilinearExtension, MultilinearExtensionWithConfig, dense::project_coeffs},
};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_uair::{
    ConstraintBuilder, Uair,
    dummy_semiring::DummySemiring,
    ideal::{DummyIdeal, Ideal},
};
use zinc_utils::{
    inner_transparent_field::InnerTransparentField, mul_by_scalar::MulByScalar,
    projectable_to_field::ProjectableToField,
};

#[derive(Clone, Debug)]
pub struct Proof<R> {
    pub combined_mle_values: Vec<R>,
}

#[derive(Clone, Debug)]
pub struct ProverState<F: PrimeField> {
    pub evaluation_points: Vec<Vec<F>>,
    pub combined_mles: Vec<DenseMultilinearExtension<F::Inner>>,
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
    pub fn prove_as_subprotocol<U, F>(
        transcript: &mut impl Transcript,
        cs_up: Vec<DenseMultilinearExtension<R>>,
        cs_down: Vec<DenseMultilinearExtension<R>>,
        num_constraints: usize,
        num_vars: usize,
        field_cfg: &F::Config,
    ) -> Result<(Proof<F>, ProverState<F>), R, U::Ideal>
    where
        U: Uair<R>,
        R: ProjectableToField<F>,
        F: InnerTransparentField,
        F::Inner: ConstTranscribable,
    {
        let projecting_element: F = transcript.get_field_challenge(field_cfg);

        let cs_up = cfg_into_iter!(cs_up)
            .map(|mle| project_coeffs(mle, &projecting_element))
            .collect_vec();

        let cs_down = cfg_into_iter!(cs_down)
            .map(|mle| project_coeffs(mle, &projecting_element))
            .collect_vec();

        let combined_mles =
            Self::get_combined_poly_mles::<U, F>(&cs_up, &cs_down, num_constraints, num_vars);
        let mut transcription_buf: Vec<u8> = vec![0; R::NUM_BYTES];

        let mut evaluation_points: Vec<Vec<F>> = Vec::with_capacity(num_constraints);
        let mut combined_mle_values: Vec<F> = Vec::with_capacity(num_constraints);

        for combined_mle in &combined_mles {
            let challenge: Vec<F> = transcript.get_field_challenges(num_vars, field_cfg);

            let mle_value = combined_mle.evaluate_with_config(&challenge, field_cfg)?;

            transcript.absorb_random_field(&mle_value, &mut transcription_buf);

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
    pub fn verify_as_subprotocol<U, F>(
        transcript: &mut impl Transcript,
        proof: Proof<F>,
        num_constraints: usize,
        num_vars: usize,
        field_cfg: &F::Config,
    ) -> Result<Vec<SubClaim<R, C>>, R, U::Ideal>
    where
        U: Uair<R>,
        R: ProjectableToField<F>,
        F: InnerTransparentField,
        F::Inner: ConstTranscribable,
    {
        let mut transcription_buf: Vec<u8> = vec![0; R::NUM_BYTES];

        let mut evaluation_points: Vec<Vec<F>> = Vec::with_capacity(num_constraints);
        let combined_mle_values = proof.combined_mle_values;

        for mle_value in &combined_mle_values {
            let challenge: Vec<F> = transcript.get_field_challenges(num_vars, field_cfg);

            transcript.absorb_random_field(mle_value, &mut transcription_buf);

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

    fn get_combined_poly_mles<U, F>(
        cs_up: &[DenseMultilinearExtension<F::Inner>],
        cs_down: &[DenseMultilinearExtension<F::Inner>],
        num_constraints: usize,
        num_vars: usize,
    ) -> Vec<DenseMultilinearExtension<F::Inner>>
    where
        U: Uair<R>,
        R: ProjectableToField<F>,
        F: PrimeField,
        F::Inner: ConstTranscribable,
    {
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
    use num_traits::{ConstZero, Zero};
    use rand::{Rng, rng};
    use zinc_poly::{
        mle::DenseMultilinearExtension,
        univariate::{binary::BinaryPoly, dense::DensePolynomial, ideal::DegreeOneIdeal},
    };
    use zinc_transcript::KeccakTranscript;
    use zinc_uair::{
        ConstraintBuilder, Uair,
        constraint_counter::count_constraints,
        ideal::{Ideal, ZeroIdeal},
    };
    use zinc_utils::from_ref::FromRef;

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

    struct TestAirNoMultiplication;

    impl<const LIMBS: usize> Uair<DensePolynomial<Int<LIMBS>, 32>> for TestAirNoMultiplication {
        type Ideal = DegreeOneIdeal<Int<LIMBS>, ZeroIdeal<DensePolynomial<Int<LIMBS>, 32>>, 32>;

        fn num_cols() -> usize {
            3
        }

        #[allow(clippy::arithmetic_side_effects)]
        fn constrain<B>(b: &mut B, up: &[B::Expr], _down: &[B::Expr])
        where
            B: ConstraintBuilder<DensePolynomial<Int<LIMBS>, 32>>,
            B::Ideal: FromRef<Self::Ideal>,
        {
            b.assert_in_ideal(
                up[0].clone() + &up[1] - &up[2],
                &B::Ideal::from_ref(&DegreeOneIdeal::new(Int::from(2))),
            );
        }
    }

    #[test]
    fn test_successful_verification() {
        let mut rng = rng();

        type Poly = DensePolynomial<Int<5>, 32>;

        let up: Vec<DenseMultilinearExtension<Poly>> = vec![
            DenseMultilinearExtension::from_evaluations_slice(
                4,
                &(0..4)
                    .map(|_| Poly::from_ref(&rng.random::<BinaryPoly<32>>()))
                    .collect_vec(),
                Poly::zero(),
            ),
            DenseMultilinearExtension::from_evaluations_slice(
                4,
                &(0..4)
                    .map(|_| Poly::from_ref(&rng.random::<BinaryPoly<32>>()))
                    .collect_vec(),
                Poly::zero(),
            ),
            DenseMultilinearExtension::from_evaluations_slice(
                4,
                &(0..4)
                    .map(|_| Poly::from_ref(&rng.random::<BinaryPoly<32>>()))
                    .collect_vec(),
                Poly::zero(),
            ),
        ];

        let down: Vec<DenseMultilinearExtension<Poly>> = vec![
            DenseMultilinearExtension::from_evaluations_slice(
                4,
                &(0..4)
                    .map(|_| Poly::from_ref(&rng.random::<BinaryPoly<32>>()))
                    .collect_vec(),
                Poly::zero(),
            ),
            DenseMultilinearExtension::from_evaluations_slice(
                4,
                &(0..4)
                    .map(|_| Poly::from_ref(&rng.random::<BinaryPoly<32>>()))
                    .collect_vec(),
                Poly::zero(),
            ),
            DenseMultilinearExtension::from_evaluations_slice(
                4,
                &(0..4)
                    .map(|_| Poly::from_ref(&rng.random::<BinaryPoly<32>>()))
                    .collect_vec(),
                Poly::zero(),
            ),
        ];

        println!("{:?}", &up);

        let transcript = KeccakTranscript::new();

        let (proof, _) =
            IdealCheckProtocol::<_, Int<2>>::prove_as_subprotocol::<TestAirNoMultiplication>(
                &mut transcript.clone(),
                &up,
                &down,
                count_constraints::<Poly, TestAirNoMultiplication>(),
                4,
            )
            .unwrap();

        assert!(
            IdealCheckProtocol::<_, Int<2>>::verify_as_subprotocol::<TestAirNoMultiplication>(
                &mut transcript.clone(),
                proof,
                count_constraints::<Poly, TestAirNoMultiplication>(),
                4,
            )
            .is_ok()
        );
    }
}
