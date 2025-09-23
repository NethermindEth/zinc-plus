use std::borrow::Cow;

use crate::{
    ZipError,
    code::LinearCode,
    pcs::{
        structs::{
            ChallengeRing, CodewordRing, EvaluationRing, LinearCombinationRing, MulByScalar,
            MultilinearZip, MultilinearZipData, MultilinearZipParams,
        },
        utils::{ColumnOpening, left_point_to_tensor, validate_input},
    },
    pcs_transcript::PcsTranscript,
    poly::mle::DenseMultilinearExtension,
    traits::{FromRef, Transcribable, Transcript},
    utils::combine_rows,
};
use crypto_primitives::PrimeField;
use itertools::{Itertools, izip};

impl<
    Eval: EvaluationRing,
    Cw: CodewordRing,
    Chal: ChallengeRing,
    Comb: LinearCombinationRing<Eval, Cw, Chal> + for<'z> MulByScalar<&'z Chal>,
    C: LinearCode<Eval, Cw, Comb>,
> MultilinearZip<Eval, Cw, Chal, Comb, C>
{
    pub fn open<F>(
        pp: &MultilinearZipParams<Eval, Cw, Comb, C>,
        poly: &DenseMultilinearExtension<Eval>,
        commit_data: &MultilinearZipData<Cw>,
        point: &[F],
        transcript: &mut PcsTranscript,
    ) -> Result<(), ZipError>
    where
        F: PrimeField + FromRef<Eval> + for<'a> MulByScalar<&'a F>,
        F::Inner: Transcribable,
    {
        validate_input("open", pp.num_vars, [poly], [point])?;

        Self::prove_testing_phase(pp, poly, commit_data, transcript)?;

        Self::prove_evaluation_phase(pp, transcript, point, poly)?;

        Ok(())
    }

    // TODO Apply 2022/1355 https://eprint.iacr.org/2022/1355.pdf#page=30
    pub fn batch_open<F>(
        pp: &MultilinearZipParams<Eval, Cw, Comb, C>,
        polys: &[DenseMultilinearExtension<Eval>],
        comms: &[MultilinearZipData<Cw>],
        points: &[Vec<F>],
        transcript: &mut PcsTranscript,
    ) -> Result<(), ZipError>
    where
        F: PrimeField + FromRef<Eval> + for<'a> MulByScalar<&'a F>,
        F::Inner: Transcribable,
    {
        for (poly, comm, point) in izip!(polys.iter(), comms.iter(), points.iter()) {
            Self::open(pp, poly, comm, point, transcript)?;
        }
        Ok(())
    }

    // Subprotocol functions

    fn prove_evaluation_phase<F>(
        pp: &MultilinearZipParams<Eval, Cw, Comb, C>,
        transcript: &mut PcsTranscript,
        point: &[F],
        poly: &DenseMultilinearExtension<Eval>,
    ) -> Result<(), ZipError>
    where
        F: PrimeField + FromRef<Eval> + for<'a> MulByScalar<&'a F>,
        F::Inner: Transcribable,
    {
        let num_rows = pp.num_rows;
        let row_len = pp.linear_code.row_len();

        // We prove evaluations over the field, so integers need to be mapped to field
        // elements first
        let q_0 = left_point_to_tensor(num_rows, point)?;

        let evaluations = poly.evaluations.iter().map(F::from_ref).collect_vec();

        let q_0_combined_row = if num_rows > 1 {
            // Return the evaluation row combination
            let combined_row = combine_rows(q_0, evaluations, row_len);
            Cow::<Vec<F>>::Owned(combined_row)
        } else {
            // If there is only one row, we have no need to take linear combinations
            // We just return the evaluation row combination
            Cow::Borrowed(&evaluations)
        };

        transcript.write_field_elements(&q_0_combined_row)
    }

    pub(super) fn prove_testing_phase(
        pp: &MultilinearZipParams<Eval, Cw, Comb, C>,
        poly: &DenseMultilinearExtension<Eval>,
        commit_data: &MultilinearZipData<Cw>,
        transcript: &mut PcsTranscript,
    ) -> Result<(), ZipError> {
        if pp.num_rows > 1 {
            // If we can take linear combinations
            // perform the proximity test an arbitrary number of times
            for _ in 0..pp.linear_code.num_proximity_testing() {
                let coeffs = transcript.fs_transcript.get_challenges::<Chal>(pp.num_rows);

                let evals = poly.evaluations.iter().map(Comb::from_ref);

                // u' in the Zinc paper
                let combined_row =
                    combine_rows(coeffs.into_iter(), evals, pp.linear_code.row_len());

                transcript.write_many(&combined_row)?;
            }
        }

        // Open merkle tree for each column drawn
        for _ in 0..pp.linear_code.num_column_opening() {
            let column = transcript.squeeze_challenge_idx(pp.linear_code.codeword_len());
            Self::open_merkle_trees_for_column(pp, commit_data, column, transcript)?;
        }
        Ok(())
    }

    pub(super) fn open_merkle_trees_for_column(
        pp: &MultilinearZipParams<Eval, Cw, Comb, C>,
        commit_data: &MultilinearZipData<Cw>,
        column: usize,
        transcript: &mut PcsTranscript,
    ) -> Result<(), ZipError> {
        let column_values = commit_data
            .rows
            .iter()
            .skip(column)
            .step_by(pp.linear_code.codeword_len());

        // Write the elements in the squeezed column to the shared transcript
        transcript.write_many(column_values)?;

        ColumnOpening::open_at_column(column, commit_data, transcript)
            .map_err(|_| ZipError::InvalidPcsOpen("Failed to open merkle tree".into()))?;

        Ok(())
    }
}

#[cfg(test)]
#[allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
mod tests {
    use crypto_bigint::{Random, U256, const_monty_params};
    use crypto_primitives::crypto_bigint_int::Int;
    use num_traits::{ConstOne, ConstZero, One};
    use rand::prelude::*;

    use crate::{
        code::{LinearCode, raa::RaaCode},
        field::ConstMontyField,
        merkle::MerkleTree,
        pcs::{
            structs::{MultilinearZip, MultilinearZipData},
            test_utils::*,
        },
        pcs_transcript::PcsTranscript,
        poly::{dense::DensePolynomial, mle::DenseMultilinearExtension},
        utils::WORD_FACTOR,
    };

    const INT_LIMBS: usize = WORD_FACTOR;
    const FIELD_LIMBS: usize = 4 * WORD_FACTOR;

    const N: usize = INT_LIMBS;
    const K: usize = INT_LIMBS * 4;
    const M: usize = INT_LIMBS * 8;
    const DEGREE: usize = 2;

    type C = RaaCode<Int<N>, Int<K>, Int<M>>;
    type PolyC = RaaCode<
        DensePolynomial<Int<N>, DEGREE>,
        DensePolynomial<Int<K>, DEGREE>,
        DensePolynomial<Int<M>, DEGREE>,
    >;

    const_monty_params!(
        ModP,
        U256,
        "0000000000000000000000000000000000000000B933426489189CB5B47D567F"
    );
    type F = ConstMontyField<ModP, FIELD_LIMBS>;

    type TestZip = MultilinearZip<Int<N>, Int<K>, Int<N>, Int<M>, C>;
    type TestPolyZip = MultilinearZip<
        DensePolynomial<Int<N>, DEGREE>,
        DensePolynomial<Int<K>, DEGREE>,
        Int<N>,
        DensePolynomial<Int<M>, DEGREE>,
        PolyC,
    >;

    fn random_point<const I: usize>(num_vars: usize, rng: &mut impl RngCore) -> Vec<Int<I>> {
        (0..num_vars).map(|_| Int::random(rng)).collect()
    }

    #[test]
    fn successful_opening_with_correct_polynomial_and_hint() {
        let num_vars = 4;
        let (pp, poly) = setup_test_params(num_vars);

        let (data, _) = TestZip::commit(&pp, &poly).unwrap();

        let mut rng = ThreadRng::default();
        let point_int = random_point::<INT_LIMBS>(num_vars, &mut rng);
        let point_f: Vec<F> = point_int.iter().map(F::from).collect();
        let mut prover_transcript = PcsTranscript::new();

        let result = TestZip::open(&pp, &poly, &data, &point_f, &mut prover_transcript);

        assert!(result.is_ok());
    }

    #[test]
    fn successful_opening_with_correct_polynomial_and_hint_poly() {
        let num_vars = 4;
        let (pp, poly) = setup_poly_test_params(num_vars);

        let (data, _) = TestPolyZip::commit(&pp, &poly).unwrap();

        let mut rng = ThreadRng::default();
        let point_int = random_point::<INT_LIMBS>(num_vars, &mut rng);
        let point_f: Vec<F> = point_int.iter().map(F::from).collect();
        let mut prover_transcript = PcsTranscript::new();

        let result = TestPolyZip::open(&pp, &poly, &data, &point_f, &mut prover_transcript);

        assert!(result.is_ok());
    }

    #[test]
    fn successful_opening_with_a_close_codeword() {
        let num_vars = 4;
        let (pp, poly) = setup_test_params(num_vars);

        let (original_data, _) = TestZip::commit(&pp, &poly).unwrap();

        let mut corrupted_rows = original_data.rows.clone();
        if !corrupted_rows.is_empty() {
            corrupted_rows[0] += Int::ONE;
        }

        let codeword_len = pp.linear_code.codeword_len();
        let corrupted_merkle_tree = MerkleTree::new(&corrupted_rows, codeword_len);
        let corrupted_data = MultilinearZipData::new(corrupted_rows, corrupted_merkle_tree);

        let mut rng = ThreadRng::default();
        let point_int = random_point::<INT_LIMBS>(num_vars, &mut rng);
        let point_f: Vec<F> = point_int.iter().map(F::from).collect();
        let mut prover_transcript = PcsTranscript::new();

        let result = TestZip::open(
            &pp,
            &poly,
            &corrupted_data,
            &point_f,
            &mut prover_transcript,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn failed_opening_due_to_incorrect_polynomial() {
        let num_vars = 4;
        let (pp, poly1) = setup_test_params(num_vars);

        let (data, comm) = TestZip::commit(&pp, &poly1).unwrap();

        let different_evals: Vec<_> = (20..=35).map(Int::from).collect();
        let poly2 = DenseMultilinearExtension::from_evaluations_vec(num_vars, different_evals);

        let point_int: Vec<Int<INT_LIMBS>> =
            (0..num_vars).map(|i| Int::from(i as i32 + 2)).collect();
        let point_f: Vec<F> = point_int.iter().map(F::from).collect();
        let mut prover_transcript = PcsTranscript::new();
        let open_result = TestZip::open(&pp, &poly2, &data, &point_f, &mut prover_transcript);
        assert!(open_result.is_ok());
        let proof = prover_transcript.into_proof();

        let mut verifier_transcript = PcsTranscript::from_proof(&proof);
        let eval = poly1
            .evaluate(&point_int)
            .expect("Failed to evaluate polynomial")
            .into();

        let verification_result =
            TestZip::verify(&pp, &comm, &point_f, &eval, &mut verifier_transcript);

        assert!(verification_result.is_err());
    }

    #[test]
    fn failed_opening_due_to_a_hint_that_is_not_close() {
        let num_vars = 4;
        let (pp, poly) = setup_test_params(num_vars);

        let (original_data, comm) = TestZip::commit(&pp, &poly).unwrap();

        let mut corrupted_rows = original_data.rows.clone();
        let codeword_len = pp.linear_code.codeword_len();
        // Proximity distance is half the codeword length for the default spec.
        // We corrupt more than half of the first row to ensure it's not close.
        let corruption_count = codeword_len / 2 + 1;
        for i in 0..corruption_count {
            if i < corrupted_rows.len() {
                corrupted_rows[i] += Int::ONE;
            }
        }

        let corrupted_merkle_tree = MerkleTree::new(&corrupted_rows, codeword_len);
        let corrupted_data = MultilinearZipData::new(corrupted_rows, corrupted_merkle_tree);

        let point_int: Vec<Int<INT_LIMBS>> =
            (0..num_vars).map(|i| Int::from(i as i32 + 2)).collect();
        let point_f: Vec<F> = point_int.iter().map(F::from).collect();
        let mut prover_transcript = PcsTranscript::new();
        let open_result = TestZip::open(
            &pp,
            &poly,
            &corrupted_data,
            &point_f,
            &mut prover_transcript,
        );
        assert!(open_result.is_ok());
        let proof = prover_transcript.into_proof();

        let mut verifier_transcript = PcsTranscript::from_proof(&proof);
        let eval = poly
            .evaluate(&point_int)
            .expect("Failed to evaluate polynomial")
            .into();

        let verification_result =
            TestZip::verify(&pp, &comm, &point_f, &eval, &mut verifier_transcript);

        assert!(verification_result.is_err());
    }

    #[test]
    fn failed_opening_due_to_oversized_polynomial_coefficients() {
        let num_vars = 4;
        let (pp, _) = setup_test_params(num_vars);

        let oversized_num_vars = 5;
        let oversized_evals: Vec<_> = (0..1 << oversized_num_vars).map(Int::from).collect();
        let oversized_poly =
            DenseMultilinearExtension::from_evaluations_vec(oversized_num_vars, oversized_evals);

        // This data is for a 4-variable poly, but we need it as a placeholder.
        let (data, _) = TestZip::commit(&pp, &setup_test_params::<N, K, M>(num_vars).1).unwrap();

        let mut rng = ThreadRng::default();
        let point_int = random_point::<INT_LIMBS>(oversized_num_vars, &mut rng);
        let point_f: Vec<F> = point_int.iter().map(F::from).collect();
        let mut prover_transcript = PcsTranscript::new();

        let result = TestZip::open(
            &pp,
            &oversized_poly,
            &data,
            &point_f,
            &mut prover_transcript,
        );

        assert!(result.is_err());
    }

    #[test]
    fn successful_testing_phase_with_strong_witness() {
        let num_vars = 4;
        let (pp, poly) = setup_test_params(num_vars);

        let (data, _) = TestZip::commit(&pp, &poly).unwrap();

        let mut prover_transcript = PcsTranscript::new();

        let result = TestZip::prove_testing_phase(&pp, &poly, &data, &mut prover_transcript);

        assert!(result.is_ok());
    }

    #[test]
    fn failed_testing_phase_with_inconsistent_codeword() {
        let num_vars = 4;
        let (pp, poly1) = setup_test_params(num_vars);

        let (_, comm) = TestZip::commit(&pp, &poly1).unwrap();

        let different_evals: Vec<_> = (20..=35).map(Int::from).collect();
        let poly2 = DenseMultilinearExtension::from_evaluations_vec(num_vars, different_evals);
        let (inconsistent_data, _) = TestZip::commit(&pp, &poly2).unwrap();

        let point_int: Vec<Int<INT_LIMBS>> =
            (0..num_vars).map(|i| Int::from(i as i32 + 2)).collect();
        let point_f: Vec<F> = point_int.iter().map(F::from).collect();
        let mut prover_transcript = PcsTranscript::new();
        let open_result = TestZip::open(
            &pp,
            &poly1,
            &inconsistent_data,
            &point_f,
            &mut prover_transcript,
        );
        assert!(open_result.is_ok());
        let proof = prover_transcript.into_proof();

        //    will not match the roots in the original public commitment.
        let mut verifier_transcript = PcsTranscript::from_proof(&proof);
        let eval = poly1
            .evaluate(&point_int)
            .expect("Failed to evaluate polynomial")
            .into();

        let verification_result =
            TestZip::verify(&pp, &comm, &point_f, &eval, &mut verifier_transcript);

        assert!(verification_result.is_err());
    }

    #[test]
    fn successful_evaluation_phase_with_correct_evaluation() {
        let num_vars = 4;
        let (pp, poly) = setup_test_params(num_vars);

        let mut prover_transcript = PcsTranscript::new();
        let point_int: Vec<Int<INT_LIMBS>> =
            (0..num_vars).map(|i| Int::from(i as i32 + 2)).collect();
        let point_f: Vec<F> = point_int.iter().map(F::from).collect();

        let result = TestZip::prove_evaluation_phase(&pp, &mut prover_transcript, &point_f, &poly);

        assert!(result.is_ok());
    }

    #[test]
    fn failed_evaluation_phase_with_incorrect_evaluation() {
        let num_vars = 4;
        let (pp, poly) = setup_test_params(num_vars);

        let (data, comm) = TestZip::commit(&pp, &poly).unwrap();

        let point_int: Vec<Int<INT_LIMBS>> =
            (0..num_vars).map(|i| Int::from(i as i32 + 2)).collect();
        let point_f: Vec<F> = point_int.iter().map(F::from).collect();
        let mut prover_transcript = PcsTranscript::new();
        let open_result = TestZip::open(&pp, &poly, &data, &point_f, &mut prover_transcript);
        assert!(open_result.is_ok());
        let proof = prover_transcript.into_proof();

        let correct_eval: F = poly
            .evaluate(&point_int)
            .expect("Failed to evaluate polynomial")
            .into();

        let incorrect_eval = correct_eval + F::one();

        let mut verifier_transcript = PcsTranscript::from_proof(&proof);
        let verification_result = TestZip::verify(
            &pp,
            &comm,
            &point_f,
            &incorrect_eval, // Use the wrong evaluation here
            &mut verifier_transcript,
        );

        assert!(verification_result.is_err());
    }

    #[test]
    fn opening_and_evaluation_of_the_zero_polynomial() {
        let num_vars = 4;
        let (pp, _) = setup_test_params(num_vars);

        let zero_evals: Vec<_> = (0..1 << num_vars).map(|_| Int::from(0)).collect();
        let zero_poly = DenseMultilinearExtension::from_evaluations_vec(num_vars, zero_evals);

        let (data, comm) = TestZip::commit(&pp, &zero_poly).unwrap();

        let point_int: Vec<Int<INT_LIMBS>> =
            (0..num_vars).map(|i| Int::from(i as i32 + 2)).collect();
        let point_f: Vec<F> = point_int.iter().map(F::from).collect();
        let mut prover_transcript = PcsTranscript::new();
        let open_result = TestZip::open(&pp, &zero_poly, &data, &point_f, &mut prover_transcript);
        assert!(open_result.is_ok());
        let proof = prover_transcript.into_proof();

        let mut verifier_transcript = PcsTranscript::from_proof(&proof);
        let eval = F::ZERO;
        let verification_result =
            TestZip::verify(&pp, &comm, &point_f, &eval, &mut verifier_transcript);

        assert!(verification_result.is_ok());
    }

    #[test]
    fn evaluation_at_the_zero_point() {
        let num_vars = 4;
        let (pp, poly) = setup_test_params(num_vars);

        let (data, comm) = TestZip::commit(&pp, &poly).unwrap();

        let point_int: Vec<Int<INT_LIMBS>> = (0..num_vars).map(|_| Int::from(0)).collect();
        let point_f: Vec<F> = point_int.iter().map(F::from).collect();

        let mut prover_transcript = PcsTranscript::new();
        let open_result = TestZip::open(&pp, &poly, &data, &point_f, &mut prover_transcript);
        assert!(open_result.is_ok());
        let proof = prover_transcript.into_proof();

        let mut verifier_transcript = PcsTranscript::from_proof(&proof);
        let eval: F = poly
            .evaluate(&point_int)
            .expect("Failed to evaluate polynomial")
            .into();

        let verification_result =
            TestZip::verify(&pp, &comm, &point_f, &eval, &mut verifier_transcript);

        assert!(
            verification_result.is_ok(),
            "Verification failed: {verification_result:?}"
        );
    }

    #[test]
    fn polynomial_coefficients_at_maximum_bit_size_boundary() {
        let num_vars = 4;
        let (pp, _) = setup_test_params(num_vars);

        let mut evals: Vec<Int<INT_LIMBS>> = (0..1 << num_vars as i32).map(Int::from).collect();
        evals[1] = Int::from(i64::MAX);
        let poly = DenseMultilinearExtension::from_evaluations_vec(num_vars, evals);

        let (data, comm) = TestZip::commit(&pp, &poly).unwrap();

        // A point of [1, 0, 0, 0] will evaluate to poly.evaluations[1].
        let mut point_coords = vec![Int::from(0); num_vars];
        point_coords[0] = Int::from(1);
        let point_int = point_coords;
        let point_f: Vec<F> = point_int.iter().map(F::from).collect();

        let mut prover_transcript = PcsTranscript::new();
        let open_result = TestZip::open(&pp, &poly, &data, &point_f, &mut prover_transcript);
        assert!(open_result.is_ok());
        let proof = prover_transcript.into_proof();

        let mut verifier_transcript = PcsTranscript::from_proof(&proof);
        let eval: F = poly
            .evaluate(&point_int)
            .expect("failed to evaluate polynomial")
            .into();

        let verification_result =
            TestZip::verify(&pp, &comm, &point_f, &eval, &mut verifier_transcript);

        assert!(
            verification_result.is_ok(),
            "Verification failed: {verification_result:?}",
        );
    }

    #[test]
    fn evaluation_succeeds_with_minimal_polynomial_size_mu_is_2() {
        let num_vars = 2;
        let (pp, poly) = setup_test_params(num_vars);

        let (data, comm) = TestZip::commit(&pp, &poly).unwrap();

        let point_int: Vec<Int<INT_LIMBS>> = vec![Int::from(1), Int::from(2)];
        let point_f: Vec<F> = point_int.iter().map(F::from).collect();
        let mut prover_transcript = PcsTranscript::new();
        let open_result = TestZip::open(&pp, &poly, &data, &point_f, &mut prover_transcript);
        assert!(open_result.is_ok());
        let proof = prover_transcript.into_proof();

        let mut verifier_transcript = PcsTranscript::from_proof(&proof);
        let eval: F = poly
            .evaluate(&point_int)
            .expect("failed to evaluate polynomial")
            .into();

        let verification_result =
            TestZip::verify(&pp, &comm, &point_f, &eval, &mut verifier_transcript);

        assert!(verification_result.is_ok());
    }
}
