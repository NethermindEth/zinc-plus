//! Ideal-check subprotocol.
mod batched_ideal_check;
mod combined_poly_builder;
mod structs;

use batched_ideal_check::*;
use crypto_primitives::{Field, PrimeField};
use derive_more::From;
use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::marker::PhantomData;
use structs::*;
use thiserror::Error;
use zinc_poly::{
    EvaluationError,
    mle::{DenseMultilinearExtension, MultilinearExtensionWithConfig},
    univariate::dynamic::over_field::DynamicPolynomialF,
};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_uair::{
    Uair,
    ideal::{Ideal, IdealCheck},
    ideal_collector::{CollectedIdeal, collect_ideals},
};
use zinc_utils::cfg_iter;

pub type Result<T, R, I> = std::result::Result<T, IdealCheckError<R, I>>;

/// Ideal-check subprotocol.
pub struct IdealCheckProtocol<
    IcTypes: IdealCheckTypes<DEGREE_PLUS_ONE>,
    const DEGREE_PLUS_ONE: usize,
>(PhantomData<IcTypes>);

impl<IcTypes: IdealCheckTypes<DEGREE_PLUS_ONE>, const DEGREE_PLUS_ONE: usize>
    IdealCheckProtocol<IcTypes, DEGREE_PLUS_ONE>
{
    /// The prover part of the ideal-check subprotocol.
    ///
    /// The prover samples a random field element
    /// and projects the coefficients of coefficients
    /// of the input MLEs. Then it computes the combined polynomials
    /// encoded by the UAIR `U`, samples a random evaluation point
    /// and sends the evaluations of the combined polynomials
    /// to the verifier.
    ///
    /// # Parameters
    /// - `transcript`: the Fiat-Shamir transcript.
    /// - `trace`: the input trace for the UAIR `U`.
    /// - `num_constraints`: the number of constraints the UAIR `U` encodes.
    /// - `num_vars`: the number of variables the trace row MLEs have.
    /// - `field_cfg`: random field configuration sampled on the previous steps
    ///   of the overall protocol.
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

        let evaluation_point = transcript.get_field_challenges(num_vars, field_cfg);

        let combined_mle_values = cfg_iter!(combined_mles)
            .map(|combined_mle| {
                Ok(DynamicPolynomialF::new_trimmed(
                    cfg_iter!(combined_mle)
                        .map(|coeff_mle| {
                            coeff_mle.evaluate_with_config(&evaluation_point, field_cfg)
                        })
                        .collect::<std::result::Result<Vec<_>, _>>()?,
                ))
            })
            .collect::<std::result::Result<Vec<_>, IdealCheckError<_, _>>>()?;

        combined_mle_values.iter().for_each(|combined_mle_value| {
            transcript
                .absorb_random_field_slice(&combined_mle_value.coeffs, &mut transcription_buf);
        });

        Ok((
            Proof {
                combined_mle_values,
            },
            ProverState {
                evaluation_point,
                combined_mles,
            },
        ))
    }

    /// The verifier part of the ideal-check subprotocol.
    ///
    /// The verifier samples a random field element
    /// the same way the prover sampled a random field
    /// element for projecting coefficients but disregards it
    /// as the verifier does not need to project anything.
    /// Then it computes the ideals encoded by the UAIR `U`,
    /// samples a random evaluation point and receives
    /// the evaluations of the combined polynomials sent by the prover
    /// and checks they belong to the corresponding ideals defined
    /// by the UAIR `U`.
    ///
    /// # Parameters
    /// - `transcript`: the Fiat-Shamir transcript.
    /// - `proof`: a purported proof produced by the prover.
    /// - `num_constraints`: the number of constraints the UAIR `U` encodes.
    /// - `num_vars`: the number of variables the trace row MLEs have.
    /// - `ideal_over_f_from_ref`: since the UAIR `U` is not aware of the field
    ///   the ideal check is operating on it defines ideals over the ring
    ///   `IcTypes::Witness`. `ideal_over_f_from_ref` allows to convert the
    ///   ideals over `IcTypes::Witness` into ideals over the field
    ///   `IcTypes::F`. Think of this as a projection for ideals.
    /// - `field_cfg`: random field configuration sampled on the previous steps
    ///   of the overall protocol.
    #[allow(clippy::type_complexity)]
    pub fn verify_as_subprotocol<U, IdealOverF, IdealOverFFromRef>(
        transcript: &mut impl Transcript,
        proof: Proof<IcTypes, DEGREE_PLUS_ONE>,
        num_constraints: usize,
        num_vars: usize,
        ideal_over_f_from_ref: IdealOverFFromRef,
        field_cfg: &<IcTypes::F as PrimeField>::Config,
    ) -> Result<
        VerifierSubClaim<IcTypes, DEGREE_PLUS_ONE>,
        DynamicPolynomialF<IcTypes::F>,
        IdealOverF,
    >
    where
        U: Uair<IcTypes::Witness>,
        <IcTypes::F as Field>::Inner: ConstTranscribable,
        IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<IcTypes::F>>,
        IdealOverFFromRef: Fn(&CollectedIdeal<U::Ideal>) -> IdealOverF,
    {
        // Sample a field element to maintain FS symmetry with
        // the prover.
        let _: IcTypes::F = transcript.get_field_challenge(field_cfg);

        let mut transcription_buf: Vec<u8> = vec![0; <IcTypes::F as Field>::Inner::NUM_BYTES];

        let combined_mle_values = proof.combined_mle_values;

        let evaluation_point = transcript.get_field_challenges(num_vars, field_cfg);

        for mle_value in &combined_mle_values {
            transcript.absorb_random_field_slice(&mle_value.coeffs, &mut transcription_buf);
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

        Ok(VerifierSubClaim {
            evaluation_point,
            values: combined_mle_values,
        })
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

    use rand::rng;
    use zinc_poly::univariate::{
        dense::DensePolynomial, dynamic::over_field::DynamicPolynomialF, ideal::DegreeOneIdeal,
    };
    use zinc_test_uair::{GenerateWitness, TestAirNoMultiplication, TestUairSimpleMultiplication};
    use zinc_transcript::KeccakTranscript;
    use zinc_uair::{
        constraint_counter::count_constraints,
        ideal::{Ideal, IdealCheck},
    };

    use super::*;

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

    fn run_prover<U, const DEGREE_PLUS_ONE: usize>(
        num_vars: usize,
        transcript: &mut impl Transcript,
    ) -> (
        Proof<TestIcTypes, DEGREE_PLUS_ONE>,
        ProverState<TestIcTypes, DEGREE_PLUS_ONE>,
    )
    where
        U: GenerateWitness<DensePolynomial<Int<5>, DEGREE_PLUS_ONE>>,
    {
        let mut rng = rng();

        let trace = U::generate_witness(num_vars, &mut rng);

        let field_cfg = test_config();

        let num_constraints =
            count_constraints::<<TestIcTypes as IdealCheckTypes<_>>::Witness, U>();

        IdealCheckProtocol::<TestIcTypes, _>::prove_as_subprotocol::<U>(
            transcript,
            &trace,
            num_constraints,
            num_vars,
            &field_cfg,
        )
        .unwrap()
    }

    fn test_successful_verification_generic<
        U,
        IdealOverF,
        IdealOverFFromRef,
        const DEGREE_PLUS_ONE: usize,
    >(
        num_vars: usize,
        ideal_over_f_from_ref: IdealOverFFromRef,
    ) where
        U: GenerateWitness<DensePolynomial<Int<5>, DEGREE_PLUS_ONE>>,
        IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<MontyField<LIMBS>>>,
        IdealOverFFromRef: Fn(&CollectedIdeal<U::Ideal>) -> IdealOverF,
    {
        let transcript = KeccakTranscript::new();

        let (proof, prover_state) =
            run_prover::<U, DEGREE_PLUS_ONE>(num_vars, &mut transcript.clone());

        let num_constraints =
            count_constraints::<<TestIcTypes as IdealCheckTypes<_>>::Witness, U>();

        let verifier_result =
            IdealCheckProtocol::<TestIcTypes, _>::verify_as_subprotocol::<U, _, _>(
                &mut transcript.clone(),
                proof,
                num_constraints,
                num_vars,
                ideal_over_f_from_ref,
                &test_config(),
            )
            .expect("Verification failed");

        assert_eq!(
            prover_state.evaluation_point,
            verifier_result.evaluation_point
        );
    }

    #[test]
    fn test_successful_verification() {
        let field_cfg = test_config();

        let num_vars = 2;

        test_successful_verification_generic::<TestAirNoMultiplication, _, _, 32>(
            num_vars,
            |ideal_over_ring| ideal_over_ring.map(|i| DegreeOneIdeal::from_with_cfg(i, &field_cfg)),
        );
        test_successful_verification_generic::<TestUairSimpleMultiplication, _, _, 32>(
            num_vars,
            |_ideal_over_ring| CollectedIdeal::zero(),
        );
    }
}
