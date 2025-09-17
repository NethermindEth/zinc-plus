use std::borrow::Cow;

use crypto_primitives::crypto_bigint_int::Int;
use itertools::{izip, Itertools};
use crypto_primitives::PrimeField;
use crate::{
    poly::dense::DenseMultilinearExtension,
    Error,
    code::LinearCode,
    pcs::{
        structs::{MultilinearZip, MultilinearZipData, MultilinearZipParams},
        utils::{ColumnOpening, left_point_to_tensor, validate_input},
    },
    pcs_transcript::PcsTranscript,
    utils::{combine_rows, expand},
};
use crate::traits::Transcribable;

impl<const N: usize, const L: usize, const K: usize, const M: usize, LC: LinearCode<N, L, K, M>>
    MultilinearZip<N, L, K, M, LC>
{
    pub fn open<F>(
        pp: &MultilinearZipParams<N, L, K, M, LC>,
        poly: &DenseMultilinearExtension<Int<N>>,
        commit_data: &MultilinearZipData<K>,
        point: &[F],
        transcript: &mut PcsTranscript,
    ) -> Result<(), Error>
    where
        F: PrimeField + for<'a> From<&'a Int<N>>,
        F::Inner: Transcribable,
    {
        validate_input("open", pp.num_vars, [poly], [point])?;

        Self::prove_testing_phase(pp, poly, commit_data, transcript)?;

        Self::prove_evaluation_phase(pp, transcript, point, poly)?;

        Ok(())
    }

    // TODO Apply 2022/1355 https://eprint.iacr.org/2022/1355.pdf#page=30
    pub fn batch_open<F>(
        pp: &MultilinearZipParams<N, L, K, M, LC>,
        polys: &[DenseMultilinearExtension<Int<N>>],
        comms: &[MultilinearZipData<K>],
        points: &[Vec<F>],
        transcript: &mut PcsTranscript,
    ) -> Result<(), Error>
    where
        F: PrimeField + for<'a> From<&'a Int<N>>,
        F::Inner: Transcribable,
    {
        for (poly, comm, point) in izip!(polys.iter(), comms.iter(), points.iter()) {
            Self::open(pp, poly, comm, point, transcript)?;
        }
        Ok(())
    }

    // Subprotocol functions

    fn prove_evaluation_phase<F>(
        pp: &MultilinearZipParams<N, L, K, M, LC>,
        transcript: &mut PcsTranscript,
        point: &[F],
        poly: &DenseMultilinearExtension<Int<N>>,
    ) -> Result<(), Error>
    where
        F: PrimeField + for<'a> From<&'a Int<N>>,
        F::Inner: Transcribable,
    {
        let num_rows = pp.num_rows;
        let row_len = pp.linear_code.row_len();

        // We prove evaluations over the field, so integers need to be mapped to field
        // elements first
        let q_0 = left_point_to_tensor(num_rows, point)?;

        let evaluations = poly.evaluations.iter().map(F::from).collect_vec();

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
        pp: &MultilinearZipParams<N, L, K, M, LC>,
        poly: &DenseMultilinearExtension<Int<N>>,
        commit_data: &MultilinearZipData<K>,
        transcript: &mut PcsTranscript,
    ) -> Result<(), Error> {
        if pp.num_rows > 1 {
            // If we can take linear combinations
            // perform the proximity test an arbitrary number of times
            for _ in 0..pp.linear_code.num_proximity_testing() {
                let coeffs = transcript.fs_transcript.get_integer_challenges(pp.num_rows);
                let coeffs = coeffs.iter().map(expand::<N, M>);

                let evals = poly.evaluations.iter().map(expand::<N, M>);

                // u' in the Zinc paper
                let combined_row = combine_rows(coeffs, evals, pp.linear_code.row_len());

                transcript.write_integers(combined_row.iter())?;
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
        pp: &MultilinearZipParams<N, L, K, M, LC>,
        commit_data: &MultilinearZipData<K>,
        column: usize,
        transcript: &mut PcsTranscript,
    ) -> Result<(), Error> {
        let column_values = commit_data
            .rows
            .iter()
            .skip(column)
            .step_by(pp.linear_code.codeword_len());

        // Write the elements in the squeezed column to the shared transcript
        transcript.write_integers(column_values)?;

        ColumnOpening::open_at_column(column, commit_data, transcript)
            .map_err(|_| Error::InvalidPcsOpen("Failed to open merkle tree".into()))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crypto_bigint::{Random, U256, const_monty_params};
    use num_traits::{ConstOne, ConstZero, One};
    use rand::rng;
    use rand_core::RngCore;
    use crypto_primitives::crypto_bigint_int::Int;

    use crate::{
        field::{ConstMontyField},
        poly::dense::DenseMultilinearExtension,
        code::{DefaultLinearCodeSpec, LinearCode},
        code_raa::RaaCode,
        pcs::{
            MerkleTree,
            structs::{MultilinearZip, MultilinearZipData, MultilinearZipParams},
            tests::MockTranscript,
        },
        pcs_transcript::PcsTranscript,
    };
    use crate::utils::WORD_FACTOR;

    const INT_LIMBS: usize = WORD_FACTOR;
    const FIELD_LIMBS: usize = 4 * WORD_FACTOR;

    const N: usize = INT_LIMBS;
    const L: usize = INT_LIMBS * 2;
    const K: usize = INT_LIMBS * 4;
    const M: usize = INT_LIMBS * 8;

    type LC = RaaCode<N, L, K, M>;

    const_monty_params!(
        ModP,
        U256,
        "0000000000000000000000000000000000000000B933426489189CB5B47D567F"
    );

    type F = ConstMontyField<ModP, FIELD_LIMBS>;
    type TestZip = MultilinearZip<N, L, K, M, LC>;

    /// Helper function to set up common parameters for tests.
    fn setup_test_params(
        num_vars: usize,
    ) -> (
        MultilinearZipParams<N, L, K, M, RaaCode<N, L, K, M>>,
        DenseMultilinearExtension<Int<INT_LIMBS>>,
    ) {
        let poly_size = 1 << num_vars;
        let num_rows = 1 << num_vars.div_ceil(2);

        let mut transcript = MockTranscript::default();
        let code = LC::new(&DefaultLinearCodeSpec, poly_size, &mut transcript);
        let pp = MultilinearZipParams::new(num_vars, num_rows, code);

        let evaluations: Vec<_> = (1..=poly_size as i32).map(Int::from).collect();
        let poly = DenseMultilinearExtension::from_evaluations_vec(num_vars, evaluations);

        (pp, poly)
    }

    fn random_point<const I: usize>(num_vars: usize, rng: &mut impl RngCore) -> Vec<Int<I>> {
        (0..num_vars).map(|_| Int::random(rng)).collect()
    }

    #[test]
    fn successful_opening_with_correct_polynomial_and_hint() {
        let num_vars = 4;
        let (pp, poly) = setup_test_params(num_vars);

        let (data, _) = TestZip::commit(&pp, &poly).unwrap();

        let mut rng = rng();
        let point_int = random_point::<INT_LIMBS>(num_vars, &mut rng);
        let point_f: Vec<F> = point_int.iter().map(F::from).collect();
        let mut prover_transcript = PcsTranscript::new();

        let result = TestZip::open(&pp, &poly, &data, &point_f, &mut prover_transcript);

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

        let mut rng = rng();
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
        let (data, _) =
            TestZip::commit(&pp, &setup_test_params(num_vars).1).unwrap();

        let mut rng = rng();
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

        assert!(verification_result.is_ok(), "Verification failed: {verification_result:?}");
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
