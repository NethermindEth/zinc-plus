mod batched_ideal_check;
mod combined_poly_builder;

use batched_ideal_check::*;
use crypto_primitives::{FixedSemiring, FromWithConfig, PrimeField, Semiring};
use derive_more::From;
use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::marker::PhantomData;
use thiserror::Error;
use zinc_poly::{
    CoefficientProjectable, EvaluationError,
    mle::{DenseMultilinearExtension, MultilinearExtensionWithConfig},
    univariate::dynamic::DynamicPolynomial,
};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_uair::{
    ConstraintBuilder, Uair,
    ideal::{DummyIdeal, Ideal, IdealCheck},
    ideal_collector::collect_ideals,
};
use zinc_utils::cfg_iter;
use zinc_utils::{
    inner_transparent_field::InnerTransparentField,
    projectable_to_field::ProjectableToField,
};

#[derive(Clone, Debug)]
pub struct Proof<R> {
    pub combined_mle_values: Vec<R>,
}

#[derive(Clone, Debug)]
pub struct ProverState<F: PrimeField> {
    pub evaluation_points: Vec<Vec<F>>,
    pub combined_mles: Vec<Vec<DenseMultilinearExtension<F::Inner>>>,
}

pub struct SubClaim<F: PrimeField> {
    pub point: Vec<F>,
    pub value: DynamicPolynomial<F>,
}

pub type Result<T, R, I> = std::result::Result<T, IdealCheckError<R, I>>;

pub struct IdealCheckProtocol<R, Rcoeff, const DEGREE_PLUS_ONE: usize>(PhantomData<(R, Rcoeff)>);

impl<R, Rcoeff, const DEGREE_PLUS_ONE: usize> IdealCheckProtocol<R, Rcoeff, DEGREE_PLUS_ONE>
where
    R: CoefficientProjectable<Rcoeff, DEGREE_PLUS_ONE>
        + FixedSemiring
        + ConstTranscribable
        + Send
        + Sync
        + 'static,
{
    #[allow(clippy::type_complexity)]
    pub fn prove_as_subprotocol<U, F>(
        transcript: &mut impl Transcript,
        trace: &[DenseMultilinearExtension<R>],
        num_constraints: usize,
        num_vars: usize,
        field_cfg: &F::Config,
    ) -> Result<(Proof<DynamicPolynomial<F>>, ProverState<F>), R, U::Ideal>
    where
        U: Uair<R>,
        R: ProjectableToField<F>,
        F: InnerTransparentField + FromWithConfig<Rcoeff> + Send + Sync + 'static,
        F::Inner: ConstTranscribable,
    {
        let projecting_element: F = transcript.get_field_challenge(field_cfg);

        let combined_mles = combined_poly_builder::compute_combined_polynomials::<F, _, _, U, _>(
            trace,
            &projecting_element,
            num_constraints,
        );
        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];

        let mut evaluation_points: Vec<Vec<F>> = Vec::with_capacity(num_constraints);
        let mut combined_mle_values: Vec<DynamicPolynomial<F>> =
            Vec::with_capacity(num_constraints);

        for combined_mle in &combined_mles {
            let challenge: Vec<F> = transcript.get_field_challenges(num_vars, field_cfg);

            let mle_coeffs_values = cfg_iter!(combined_mle)
                .map(|coeff_mle| coeff_mle.evaluate_with_config(&challenge, field_cfg))
                .collect::<std::result::Result<Vec<_>, _>>()?;

            transcript.absorb_random_field_slice(&mle_coeffs_values, &mut transcription_buf);

            evaluation_points.push(challenge);
            combined_mle_values.push(DynamicPolynomial::new_trimmed_with_zero(
                mle_coeffs_values,
                &F::zero_with_cfg(field_cfg),
            ));
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
    pub fn verify_as_subprotocol<U, F, IdealOverF, IdealOverFFromRef>(
        transcript: &mut impl Transcript,
        proof: Proof<DynamicPolynomial<F>>,
        num_constraints: usize,
        num_vars: usize,
        ideal_over_f_from_ref: IdealOverFFromRef,
        field_cfg: &F::Config,
    ) -> Result<Vec<SubClaim<F>>, DynamicPolynomial<F>, IdealOverF>
    where
        U: Uair<R>,
        R: ProjectableToField<F>,
        F: PrimeField,
        F::Inner: ConstTranscribable,
        IdealOverF: Ideal,
        DynamicPolynomial<F>: IdealCheck<IdealOverF>,
        IdealOverFFromRef: Fn(&U::Ideal) -> IdealOverF,
    {
        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];

        let mut evaluation_points: Vec<Vec<F>> = Vec::with_capacity(num_constraints);
        let combined_mle_values = proof.combined_mle_values;

        for mle_value in &combined_mle_values {
            let challenge: Vec<F> = transcript.get_field_challenges(num_vars, field_cfg);

            transcript.absorb_random_field_slice(&mle_value.coeffs, &mut transcription_buf);

            evaluation_points.push(challenge);
        }

        let ideal_collector = collect_ideals::<_, U>(num_constraints);

        batched_ideal_check::<_, _>(
            &ideal_collector
                .ideals
                .iter()
                .map(ideal_over_f_from_ref)
                .collect_vec(),
            &combined_mle_values,
        )?;

        Ok(evaluation_points
            .into_iter()
            .zip(combined_mle_values)
            .map(|(point, value)| SubClaim { point, value })
            .collect())
    }
}

pub(crate) struct IdealCheckConstraintBuilder<P: Semiring> {
    pub uair_poly_mles_coeffs: Vec<P>,
}

impl<P: Semiring> IdealCheckConstraintBuilder<P> {
    pub fn new(num_constraints: usize) -> Self {
        Self {
            uair_poly_mles_coeffs: Vec::with_capacity(num_constraints),
        }
    }
}

impl<P: Semiring> ConstraintBuilder for IdealCheckConstraintBuilder<P> {
    type Expr = P;
    // Ignore all ideal business on the side of the prover.
    type Ideal = DummyIdeal;

    #[allow(clippy::arithmetic_side_effects)]
    fn assert_in_ideal(&mut self, expr: Self::Expr, _ideal: &Self::Ideal) {
        self.uair_poly_mles_coeffs.push(expr);
    }
}

#[derive(Clone, Debug, From, Error)]
pub enum IdealCheckError<R, I> {
    #[error("ideal check prover failed to evaluate an mle: {0}")]
    MleEvaluationError(EvaluationError),
    #[error("mle evaluation ideal check failure: {0}")]
    IdealCollectorError(BatchedIdealCheckError<R, I>),
}

#[cfg(test)]
mod tests {
    use crypto_bigint::{Odd, modular::MontyParams};
    use crypto_primitives::{crypto_bigint_int::Int, crypto_bigint_monty::MontyField};
    use itertools::Itertools;
    use num_traits::Zero;
    
    use zinc_poly::{
        mle::DenseMultilinearExtension,
        univariate::{dense::DensePolynomial, ideal::DegreeOneIdeal},
    };
    use zinc_transcript::KeccakTranscript;
    use zinc_uair::constraint_counter::count_constraints;
    

    use crate::{ideal_check::IdealCheckProtocol, tests::test_airs::TestAirNoMultiplication};

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
        type Poly = DensePolynomial<Int<5>, 32>;

        let trace: Vec<DenseMultilinearExtension<Poly>> = vec![
            DenseMultilinearExtension::from_evaluations_slice(
                4,
                &(0..4).map(|i| Poly::from(Int::from_i8(i))).collect_vec(),
                Poly::zero(),
            ),
            DenseMultilinearExtension::from_evaluations_slice(
                4,
                &(0..4).map(|_| Poly::from(Int::from_i8(1))).collect_vec(),
                Poly::zero(),
            ),
            DenseMultilinearExtension::from_evaluations_slice(
                4,
                &(0..4)
                    .map(|i| Poly::from(Int::from_i8(i + 1)))
                    .collect_vec(),
                Poly::zero(),
            ),
        ];

        let field_cfg = test_config();

        let transcript = KeccakTranscript::new();

        let (proof, _) =
            IdealCheckProtocol::<_, _, _>::prove_as_subprotocol::<TestAirNoMultiplication, F>(
                &mut transcript.clone(),
                &trace,
                count_constraints::<Poly, TestAirNoMultiplication>(),
                4,
                &field_cfg,
            )
            .unwrap();

        assert!(
            IdealCheckProtocol::<_, Int<5>, _>::verify_as_subprotocol::<
                TestAirNoMultiplication,
                F,
                _,
                _,
            >(
                &mut transcript.clone(),
                proof,
                count_constraints::<Poly, TestAirNoMultiplication>(),
                4,
                |ideal_over_ring| { DegreeOneIdeal::from_with_cfg(ideal_over_ring, &field_cfg) },
                &field_cfg,
            )
            .is_ok()
        );
    }
}
