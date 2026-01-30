mod batched_ideal_check;
mod combined_poly_builder;
mod structs;

use batched_ideal_check::*;
use crypto_primitives::{Field, FixedSemiring, FromWithConfig, PrimeField, Semiring};
use derive_more::From;
use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::marker::PhantomData;
use structs::*;
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
    inner_transparent_field::InnerTransparentField, projectable_to_field::ProjectableToField,
};

pub type Result<T, R, I> = std::result::Result<T, IdealCheckError<R, I>>;

pub struct IdealCheckProtocol<
    IcTypes: IdealCheckTypes<DEGREE_PLUS_ONE>,
    const DEGREE_PLUS_ONE: usize,
>(PhantomData<IcTypes>);

impl<IcTypes: IdealCheckTypes<DEGREE_PLUS_ONE>, const DEGREE_PLUS_ONE: usize>
    IdealCheckProtocol<IcTypes, DEGREE_PLUS_ONE>
{
    #[allow(clippy::type_complexity)]
    pub fn prove_as_subprotocol<U>(
        transcript: &mut impl Transcript,
        trace: &[DenseMultilinearExtension<IcTypes::Witness>],
        num_constraints: usize,
        num_vars: usize,
        field_cfg: &<IcTypes::F as PrimeField>::Config,
    ) -> Result<
        (
            Proof<IcTypes, DEGREE_PLUS_ONE>,
            ProverState<IcTypes, DEGREE_PLUS_ONE>,
        ),
        IcTypes::Witness,
        U::Ideal,
    >
    where
        U: Uair<IcTypes::Witness>,
        <IcTypes::F as Field>::Inner: ConstTranscribable,
    {
        let projecting_element = transcript.get_field_challenge(field_cfg);

        let combined_mles = combined_poly_builder::compute_combined_polynomials::<IcTypes, U, _>(
            trace,
            &projecting_element,
            num_constraints,
        );
        let mut transcription_buf: Vec<u8> = vec![0; <IcTypes::F as Field>::Inner::NUM_BYTES];

        let mut evaluation_points: Vec<Vec<IcTypes::F>> = Vec::with_capacity(num_constraints);
        let mut combined_mle_values: Vec<DynamicPolynomial<IcTypes::F>> =
            Vec::with_capacity(num_constraints);

        for combined_mle in &combined_mles {
            let challenge = transcript.get_field_challenges(num_vars, field_cfg);

            let mle_coeffs_values = cfg_iter!(combined_mle)
                .map(|coeff_mle| coeff_mle.evaluate_with_config(&challenge, field_cfg))
                .collect::<std::result::Result<Vec<_>, _>>()?;

            transcript.absorb_random_field_slice(&mle_coeffs_values, &mut transcription_buf);

            evaluation_points.push(challenge);
            combined_mle_values.push(DynamicPolynomial::new_trimmed_with_zero(
                mle_coeffs_values,
                &IcTypes::F::zero_with_cfg(field_cfg),
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
    pub fn verify_as_subprotocol<U, IdealOverF, IdealOverFFromRef>(
        transcript: &mut impl Transcript,
        proof: Proof<IcTypes, DEGREE_PLUS_ONE>,
        num_constraints: usize,
        num_vars: usize,
        ideal_over_f_from_ref: IdealOverFFromRef,
        field_cfg: &<IcTypes::F as PrimeField>::Config,
    ) -> Result<
        Vec<VerifierSubClaim<IcTypes, DEGREE_PLUS_ONE>>,
        DynamicPolynomial<IcTypes::F>,
        IdealOverF,
    >
    where
        U: Uair<IcTypes::Witness>,
        <IcTypes::F as Field>::Inner: ConstTranscribable,
        IdealOverF: Ideal,
        DynamicPolynomial<IcTypes::F>: IdealCheck<IdealOverF>,
        IdealOverFFromRef: Fn(&U::Ideal) -> IdealOverF,
    {
        let mut transcription_buf: Vec<u8> = vec![0; <IcTypes::F as Field>::Inner::NUM_BYTES];

        let mut evaluation_points: Vec<Vec<IcTypes::F>> = Vec::with_capacity(num_constraints);
        let combined_mle_values = proof.combined_mle_values;

        for mle_value in &combined_mle_values {
            let challenge = transcript.get_field_challenges(num_vars, field_cfg);

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
            .map(|(point, value)| VerifierSubClaim { point, value })
            .collect())
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

    use crate::{
        ideal_check::{IdealCheckProtocol, structs::IdealCheckTypes},
        tests::test_airs::TestAirNoMultiplication,
    };

    const LIMBS: usize = 4;

    fn test_config() -> MontyParams<LIMBS> {
        let modulus = crypto_bigint::Uint::<LIMBS>::from_be_hex(
            "0000000000000000000000000000000000860995AE68FC80E1B1BD1E39D54B33",
        );
        let modulus = Odd::new(modulus).expect("modulus should be odd");
        MontyParams::new(modulus)
    }

    struct TestIcTypes;

    impl<const DEGREE_PLUS_ONE: usize> IdealCheckTypes<DEGREE_PLUS_ONE> for TestIcTypes {
        type WitnessCoeff = Int<5>;
        type Witness = DensePolynomial<Int<5>, DEGREE_PLUS_ONE>;

        type F = MontyField<4>;
    }

    #[test]
    fn test_successful_verification() {
        let trace: Vec<DenseMultilinearExtension<<TestIcTypes as IdealCheckTypes<_>>::Witness>> = vec![
            (0..4).map(|i| (Int::from_i8(i).into())).collect(),
            (0..4).map(|_| (Int::from_i8(1).into())).collect(),
            (0..4).map(|i| (Int::from_i8(i + 1).into())).collect(),
        ];

        let field_cfg = test_config();

        let transcript = KeccakTranscript::new();

        let (proof, _) =
            IdealCheckProtocol::<TestIcTypes, _>::prove_as_subprotocol::<TestAirNoMultiplication>(
                &mut transcript.clone(),
                &trace,
                count_constraints::<
                    <TestIcTypes as IdealCheckTypes<_>>::Witness,
                    TestAirNoMultiplication,
                >(),
                2,
                &field_cfg,
            )
            .unwrap();

        assert!(
            IdealCheckProtocol::<TestIcTypes, _>::verify_as_subprotocol::<
                TestAirNoMultiplication,
                _,
                _,
            >(
                &mut transcript.clone(),
                proof,
                count_constraints::<
                    <TestIcTypes as IdealCheckTypes<_>>::Witness,
                    TestAirNoMultiplication,
                >(),
                4,
                |ideal_over_ring| { DegreeOneIdeal::from_with_cfg(ideal_over_ring, &field_cfg) },
                &field_cfg,
            )
            .is_ok()
        );
    }
}
