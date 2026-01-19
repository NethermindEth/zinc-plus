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
    ideal::{DummyIdeal, Ideal, IdealCheck},
};
use zinc_utils::{
    from_ref::FromRef, inner_transparent_field::InnerTransparentField, mul_by_scalar::MulByScalar,
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

pub struct SubClaim<F: PrimeField> {
    pub point: Vec<F>,
    pub value: F,
}

pub type Result<T, R, I> = std::result::Result<T, IdealCheckError<R, I>>;

pub struct IdealCheckProtocol<R, C>(PhantomData<(R, C)>);

impl<R, C> IdealCheckProtocol<R, C>
where
    R: FixedSemiring + for<'a> MulByScalar<&'a C> + ConstTranscribable + 'static,
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

        let combined_mles = Self::get_combined_poly_mles::<U, F>(
            &cs_up,
            &cs_down,
            &projecting_element,
            num_constraints,
            num_vars,
            field_cfg,
        );
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
    ) -> Result<Vec<SubClaim<F>>, F, U::Ideal>
    where
        U: Uair<R>,
        R: ProjectableToField<F>,
        F: InnerTransparentField + IdealCheck<U::Ideal>,
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

        let mut ideal_collector = IdealCollector::new(num_constraints);

        let dummy_up_and_down: Vec<DummySemiring> = vec![DummySemiring; num_constraints];

        U::constrain(&mut ideal_collector, &dummy_up_and_down, &dummy_up_and_down);

        ideal_collector
            .ideals
            .iter()
            .zip(combined_mle_values.iter())
            .try_for_each(|(ideal, mle_value)| {
                if !mle_value.is_contained_in(ideal) {
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
        projecting_element: &F,
        num_constraints: usize,
        num_vars: usize,
        field_cfg: &F::Config,
    ) -> Vec<DenseMultilinearExtension<F::Inner>>
    where
        U: Uair<R>,
        R: ProjectableToField<F>,
        F: PrimeField,
        F::Inner: ConstTranscribable,
    {
        let projection = R::prepare_projection(&projecting_element);
        // Collect h MLEs.
        let len = cs_up[0].evaluations.len();

        let mut h_evals: Vec<Vec<F::Inner>> = (0..num_constraints)
            .map(|_| Vec::with_capacity(len))
            .collect_vec();

        let pointers: Vec<*mut F::Inner> =
            h_evals.iter_mut().map(|col| col.as_mut_ptr()).collect_vec();

        (0..len).for_each(|i| {
            let mut builder = IdealCheckConstraintBuilder::<F>::new(num_constraints);

            let up: Vec<F> = cs_up
                .iter()
                .map(|up| F::new_unchecked_with_cfg(up.evaluations[i].clone(), field_cfg))
                .collect_vec();
            let down: Vec<F> = cs_down
                .iter()
                .map(|down| F::new_unchecked_with_cfg(down.evaluations[i].clone(), field_cfg))
                .collect_vec();

            U::constrain_general(
                &mut builder,
                &up,
                &down,
                |x| projection(x),
                |x, y| Some(projection(y) * x),
            );

            pointers
                .iter()
                .zip(builder.uair_poly_mles_coeffs)
                .for_each(|(ptr, eval)| unsafe {
                    *ptr.add(i) = eval.inner().clone();
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

impl<R: Semiring> ConstraintBuilder for IdealCheckConstraintBuilder<R> {
    type Expr = R;
    // Ignore all ideal business on the side of the prover.
    type Ideal = DummyIdeal;

    #[allow(clippy::arithmetic_side_effects)]
    fn assert_in_ideal(&mut self, expr: Self::Expr, _ideal: &Self::Ideal) {
        self.uair_poly_mles_coeffs.push(expr);
    }
}

pub(crate) struct IdealCollector<I: Ideal> {
    pub ideals: Vec<I>,
}

impl<I: Ideal> IdealCollector<I> {
    pub fn new(num_constraints: usize) -> Self {
        Self {
            ideals: Vec::with_capacity(num_constraints),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CollectedIdeal<I: Ideal>(I);

impl<I: Ideal> Ideal for CollectedIdeal<I> {
    fn zero_ideal() -> Self {
        Self(I::zero_ideal())
    }
}

impl<I: Ideal> FromRef<CollectedIdeal<I>> for CollectedIdeal<I> {
    fn from_ref(value: &CollectedIdeal<I>) -> Self {
        value.clone()
    }
}

impl<I: Ideal> FromRef<I> for CollectedIdeal<I> {
    fn from_ref(value: &I) -> Self {
        Self(value.clone())
    }
}

impl<I: Ideal> IdealCheck<CollectedIdeal<I>> for DummySemiring {
    fn is_contained_in(&self, _ideal: &CollectedIdeal<I>) -> bool {
        true
    }
}

impl<I> ConstraintBuilder for IdealCollector<I>
where
    I: Ideal,
{
    type Expr = DummySemiring;
    type Ideal = CollectedIdeal<I>;

    fn assert_in_ideal(&mut self, _expr: Self::Expr, ideal: &Self::Ideal) {
        self.ideals.push(ideal.0.clone());
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
    use crypto_bigint::{Odd, modular::MontyParams};
    use crypto_primitives::{
        FixedSemiring, PrimeField, crypto_bigint_int::Int, crypto_bigint_monty::MontyField,
    };
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
        ideal::{Ideal, IdealCheck, ZeroIdeal},
    };
    use zinc_utils::from_ref::FromRef;

    use crate::{
        ideal_check::IdealCheckProtocol,
        tests::test_airs::{TestAirNoMultiplication, TestUair},
    };

    const LIMBS: usize = 4;
    type F = MontyField<LIMBS>;

    fn test_config() -> MontyParams<LIMBS> {
        let modulus = crypto_bigint::Uint::<LIMBS>::from_be_hex(
            "0000000000000000000000000000000000860995AE68FC80E1B1BD1E39D54B33",
        );
        let modulus = Odd::new(modulus).expect("modulus should be odd");
        MontyParams::new(modulus)
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

        let field_cfg = test_config();

        let transcript = KeccakTranscript::new();

        let (proof, _) =
            IdealCheckProtocol::<_, Int<2>>::prove_as_subprotocol::<TestAirNoMultiplication, F>(
                &mut transcript.clone(),
                up,
                down,
                count_constraints::<Poly, TestAirNoMultiplication>(),
                4,
                &field_cfg,
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
