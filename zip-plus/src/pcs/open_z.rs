use std::borrow::Cow;

use crate::{
    ZipError,
    code::LinearCode,
    pcs::{
        structs::{MulByScalar, ProjectableToField, ZipPlus, ZipPlusHint, ZipPlusParams, ZipTypes},
        utils::{ColumnOpening, point_to_tensor, validate_input},
    },
    pcs_transcript::PcsTranscript,
    poly::{Polynomial, mle::DenseMultilinearExtension},
    traits::{FromRef, Transcribable, Transcript},
    utils::{combine_rows, inner_product},
};
use crypto_primitives::PrimeField;
use itertools::{Itertools, izip};

// TODO: Split onto test and open(aka eval)
impl<Zt: ZipTypes> ZipPlus<Zt> {
    pub fn open<F>(
        pp: &ZipPlusParams<Zt>,
        poly: &DenseMultilinearExtension<Zt::Eval>,
        commit_hint: &ZipPlusHint<Zt::Cw>,
        point: &[Zt::Pt],
        transcript: &mut PcsTranscript,
    ) -> Result<F, ZipError>
    where
        F: PrimeField + FromRef<Zt::Chal> + FromRef<Zt::Pt> + for<'a> MulByScalar<&'a F>,
        F::Inner: Transcribable,
        Zt::Eval: ProjectableToField<F>,
    {
        validate_input("open", pp.num_vars, [poly], [point])?;

        Self::prove_testing_phase(pp, poly, commit_hint, transcript)?;

        let projecting_element: Zt::Chal = transcript.fs_transcript.get_challenge();
        let projecting_element: F = F::from_ref(&projecting_element);

        Self::prove_evaluation_phase(pp, transcript, point, poly, projecting_element)
    }

    // TODO Apply 2022/1355 https://eprint.iacr.org/2022/1355.pdf#page=30
    pub fn batch_open<F>(
        pp: &ZipPlusParams<Zt>,
        polys: &[DenseMultilinearExtension<Zt::Eval>],
        comms: &[ZipPlusHint<Zt::Cw>],
        points: &[Vec<Zt::Pt>],
        transcript: &mut PcsTranscript,
    ) -> Result<(), ZipError>
    where
        F: PrimeField + FromRef<Zt::Chal> + FromRef<Zt::Pt> + for<'a> MulByScalar<&'a F>,
        F::Inner: Transcribable,
        Zt::Eval: ProjectableToField<F>,
    {
        for (poly, comm, point) in izip!(polys.iter(), comms.iter(), points.iter()) {
            Self::open(pp, poly, comm, point, transcript)?;
        }
        Ok(())
    }

    // Subprotocol functions

    fn prove_evaluation_phase<F>(
        pp: &ZipPlusParams<Zt>,
        transcript: &mut PcsTranscript,
        point: &[Zt::Pt],
        poly: &DenseMultilinearExtension<Zt::Eval>,
        projecting_element: F,
    ) -> Result<F, ZipError>
    where
        F: PrimeField + FromRef<Zt::Pt> + for<'a> MulByScalar<&'a F>,
        F::Inner: Transcribable,
        Zt::Eval: ProjectableToField<F>,
    {
        let num_rows = pp.num_rows;
        let row_len = pp.linear_code.row_len();

        // We prove evaluations over the field, so integers need to be mapped to field
        // elements first
        let point = point.iter().map(F::from_ref).collect_vec();
        let (q_0, q_1) = point_to_tensor(num_rows, &point)?;

        let project = Zt::Eval::prepare_projection(&projecting_element);
        let evaluations: Vec<F> = poly.evaluations.iter().map(project).collect_vec();

        let q_0_combined_row = if num_rows > 1 {
            // Return the evaluation row combination
            let combined_row = combine_rows(q_0, evaluations, row_len);
            Cow::<Vec<F>>::Owned(combined_row)
        } else {
            // If there is only one row, we have no need to take linear combinations
            // We just return the evaluation row combination
            Cow::Borrowed(&evaluations)
        };

        transcript.write_field_elements(&q_0_combined_row)?;
        Ok(inner_product(&q_0_combined_row[..], &q_1))
    }

    pub(super) fn prove_testing_phase(
        pp: &ZipPlusParams<Zt>,
        poly: &DenseMultilinearExtension<Zt::Eval>,
        commit_hint: &ZipPlusHint<Zt::Cw>,
        transcript: &mut PcsTranscript,
    ) -> Result<(), ZipError> {
        if pp.num_rows > 1 {
            // If we can take linear combinations
            // perform the proximity test an arbitrary number of times
            for _ in 0..pp.linear_code.num_proximity_testing() {
                // Values to evaluate the coefficients at
                let alphas = transcript
                    .fs_transcript
                    .get_challenges::<Zt::Chal>(Zt::Comb::DEGREE_BOUND);

                // Coefficients for the linear combination of polynomial with evaluated
                // coefficients
                let coeffs = transcript
                    .fs_transcript
                    .get_challenges::<Zt::Chal>(pp.num_rows);

                let evals = poly
                    .evaluations
                    .iter()
                    .map(Zt::Comb::from_ref)
                    .map(|p| {
                        p.evaluate_at_point(&alphas)
                            .expect("Failed to evaluate polynomial")
                    })
                    .collect_vec();

                // u' in the Zinc paper
                let combined_row = combine_rows(
                    coeffs.into_iter(),
                    evals.into_iter(),
                    pp.linear_code.row_len(),
                );

                transcript.write_many(&combined_row)?;
            }
        }

        // Open merkle tree for each column drawn
        for _ in 0..pp.linear_code.num_column_opening() {
            let column = transcript.squeeze_challenge_idx(pp.linear_code.codeword_len());
            Self::open_merkle_trees_for_column(pp, commit_hint, column, transcript)?;
        }
        Ok(())
    }

    pub(super) fn open_merkle_trees_for_column(
        pp: &ZipPlusParams<Zt>,
        commit_hint: &ZipPlusHint<Zt::Cw>,
        column: usize,
        transcript: &mut PcsTranscript,
    ) -> Result<(), ZipError> {
        let column_values = commit_hint
            .rows
            .iter()
            .skip(column)
            .step_by(pp.linear_code.codeword_len());

        // Write the elements in the squeezed column to the shared transcript
        transcript.write_many(column_values)?;

        ColumnOpening::open_at_column(column, commit_hint, transcript)
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
    use crate::{
        code::LinearCode,
        field::ConstMontyField,
        merkle::MerkleTree,
        pcs::{
            structs::{ZipPlus, ZipPlusHint, ZipTypes},
            test_utils::*,
        },
        pcs_transcript::PcsTranscript,
        poly::mle::DenseMultilinearExtension,
        utils::WORD_FACTOR,
    };
    use crypto_bigint::{U256, const_monty_params};
    use crypto_primitives::{Ring, crypto_bigint_int::Int};
    use num_traits::{ConstOne, ConstZero};
    use rand::{distr::StandardUniform, prelude::*};

    const INT_LIMBS: usize = WORD_FACTOR;
    const FIELD_LIMBS: usize = 4 * WORD_FACTOR;

    const N: usize = INT_LIMBS;
    const K: usize = INT_LIMBS * 4;
    const M: usize = INT_LIMBS * 8;
    const DEGREE: usize = 2;

    const_monty_params!(
        ModP,
        U256,
        "0000000000000000000000000000000000000000B933426489189CB5B47D567F"
    );
    type F = ConstMontyField<ModP, FIELD_LIMBS>;

    type Zt = TestZipTypes<N, K, M>;
    type PolyZt = TestPolyZipTypes<N, K, M, DEGREE>;

    type TestZip = ZipPlus<Zt>;
    type TestPolyZip = ZipPlus<PolyZt>;

    fn random_point<R: Ring>(num_vars: usize, rng: &mut impl RngCore) -> Vec<R>
    where
        StandardUniform: Distribution<R>,
    {
        (0..num_vars).map(|_| rng.random()).collect()
    }

    #[test]
    fn successful_opening_with_correct_polynomial_and_hint() {
        let num_vars = 4;
        let (pp, poly) = setup_test_params(num_vars);

        let (data, _) = TestZip::commit(&pp, &poly).unwrap();

        let mut rng = ThreadRng::default();
        let point = random_point(num_vars, &mut rng);
        let mut prover_transcript = PcsTranscript::new();

        let result = TestZip::open::<F>(&pp, &poly, &data, &point, &mut prover_transcript);

        assert!(result.is_ok());
    }

    #[test]
    fn successful_opening_with_correct_polynomial_and_hint_poly() {
        let num_vars = 4;
        let (pp, poly) = setup_poly_test_params(num_vars);

        let (data, _) = TestPolyZip::commit(&pp, &poly).unwrap();

        let mut rng = ThreadRng::default();
        let point = random_point(num_vars, &mut rng);
        let mut prover_transcript = PcsTranscript::new();

        let result = TestPolyZip::open::<F>(&pp, &poly, &data, &point, &mut prover_transcript);

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
        let corrupted_data = ZipPlusHint::new(corrupted_rows, corrupted_merkle_tree);

        let mut rng = ThreadRng::default();
        let point = random_point(num_vars, &mut rng);
        let mut prover_transcript = PcsTranscript::new();

        let result =
            TestZip::open::<F>(&pp, &poly, &corrupted_data, &point, &mut prover_transcript);

        assert!(result.is_ok());
    }

    #[test]
    fn failed_opening_due_to_incorrect_polynomial() {
        let num_vars = 4;
        let (pp, poly1) = setup_test_params(num_vars);

        let (data, comm) = TestZip::commit(&pp, &poly1).unwrap();

        let different_evals: Vec<_> = (20..=35).map(Int::from).collect();
        let poly2 = DenseMultilinearExtension::from_evaluations_vec(num_vars, different_evals);

        let point: Vec<<Zt as ZipTypes>::Pt> =
            (0..num_vars).map(|i| Int::from(i as i32 + 2)).collect();
        let point_f: Vec<F> = point.iter().map(F::from).collect();
        let mut prover_transcript = PcsTranscript::new();
        let open_result = TestZip::open::<F>(&pp, &poly2, &data, &point, &mut prover_transcript);
        assert!(open_result.is_ok());
        let proof = prover_transcript.into_proof();

        let mut verifier_transcript = PcsTranscript::from_proof(&proof);
        let eval = poly1
            .evaluate(&point)
            .expect("Failed to evaluate polynomial");

        let verification_result =
            TestZip::verify(&pp, &comm, &point_f, &eval.into(), &mut verifier_transcript);

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
        let corrupted_data = ZipPlusHint::new(corrupted_rows, corrupted_merkle_tree);

        let point: Vec<<Zt as ZipTypes>::Pt> =
            (0..num_vars).map(|i| Int::from(i as i32 + 2)).collect();
        let point_f: Vec<F> = point.iter().map(F::from).collect();
        let mut prover_transcript = PcsTranscript::new();
        let open_result =
            TestZip::open::<F>(&pp, &poly, &corrupted_data, &point, &mut prover_transcript);
        assert!(open_result.is_ok());
        let proof = prover_transcript.into_proof();

        let eval_f = open_result.unwrap();
        let mut verifier_transcript = PcsTranscript::from_proof(&proof);
        let expected_eval = poly
            .evaluate(&point)
            .expect("Failed to evaluate polynomial");
        assert_eq!(eval_f, expected_eval.into());

        let verification_result =
            TestZip::verify(&pp, &comm, &point_f, &eval_f, &mut verifier_transcript);

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
        let point = random_point(oversized_num_vars, &mut rng);
        let mut prover_transcript = PcsTranscript::new();

        let result =
            TestZip::open::<F>(&pp, &oversized_poly, &data, &point, &mut prover_transcript);

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

        let point: Vec<<Zt as ZipTypes>::Pt> =
            (0..num_vars).map(|i| Int::from(i as i32 + 2)).collect();
        let point_f: Vec<F> = point.iter().map(F::from).collect();
        let mut prover_transcript = PcsTranscript::new();
        let open_result = TestZip::open::<F>(
            &pp,
            &poly1,
            &inconsistent_data,
            &point,
            &mut prover_transcript,
        );
        assert!(open_result.is_ok());
        let proof = prover_transcript.into_proof();
        let eval_f = open_result.unwrap();

        // will not match the roots in the original public commitment.
        let mut verifier_transcript = PcsTranscript::from_proof(&proof);
        let expected_eval = poly1
            .evaluate(&point)
            .expect("Failed to evaluate polynomial");
        assert_eq!(eval_f, expected_eval.into());

        let verification_result =
            TestZip::verify(&pp, &comm, &point_f, &eval_f, &mut verifier_transcript);

        assert!(verification_result.is_err());
    }

    #[test]
    fn successful_evaluation_phase_with_correct_evaluation() {
        let num_vars = 4;
        let (pp, poly) = setup_test_params(num_vars);

        let mut prover_transcript = PcsTranscript::new();
        let point: Vec<<Zt as ZipTypes>::Pt> =
            (0..num_vars).map(|i| Int::from(i as i32 + 2)).collect();

        // Not really used
        let projecting_element: F = F::ZERO;

        let result = TestZip::prove_evaluation_phase(
            &pp,
            &mut prover_transcript,
            &point,
            &poly,
            projecting_element,
        );

        assert!(result.is_ok());

        let eval_f = result.unwrap();

        let expected_eval = poly
            .evaluate(&point)
            .expect("Failed to evaluate polynomial");
        assert_eq!(eval_f, expected_eval.into());
    }

    #[test]
    fn failed_evaluation_phase_with_incorrect_evaluation() {
        let num_vars = 4;
        let (pp, poly) = setup_test_params(num_vars);

        let (data, comm) = TestZip::commit(&pp, &poly).unwrap();

        let point: Vec<<Zt as ZipTypes>::Pt> =
            (0..num_vars).map(|i| Int::from(i as i32 + 2)).collect();
        let point_f: Vec<F> = point.iter().map(F::from).collect();
        let mut prover_transcript = PcsTranscript::new();
        let open_result = TestZip::open::<F>(&pp, &poly, &data, &point, &mut prover_transcript);
        assert!(open_result.is_ok());
        let proof = prover_transcript.into_proof();

        let correct_eval_f: F = open_result.unwrap();
        let incorrect_eval_f = correct_eval_f + F::ONE;

        let mut verifier_transcript = PcsTranscript::from_proof(&proof);
        let verification_result = TestZip::verify(
            &pp,
            &comm,
            &point_f,
            &incorrect_eval_f, // Use the wrong evaluation here
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

        let point: Vec<<Zt as ZipTypes>::Pt> =
            (0..num_vars).map(|i| Int::from(i as i32 + 2)).collect();
        let point_f: Vec<F> = point.iter().map(F::from).collect();
        let mut prover_transcript = PcsTranscript::new();
        let open_result =
            TestZip::open::<F>(&pp, &zero_poly, &data, &point, &mut prover_transcript);
        assert!(open_result.is_ok());
        let proof = prover_transcript.into_proof();
        let eval_f = F::ZERO;

        let mut verifier_transcript = PcsTranscript::from_proof(&proof);
        let verification_result =
            TestZip::verify(&pp, &comm, &point_f, &eval_f, &mut verifier_transcript);

        assert!(verification_result.is_ok());
    }

    #[test]
    fn evaluation_at_the_zero_point() {
        let num_vars = 4;
        let (pp, poly) = setup_test_params(num_vars);

        let (data, comm) = TestZip::commit(&pp, &poly).unwrap();

        let point: Vec<<Zt as ZipTypes>::Pt> = (0..num_vars).map(|_| Int::from(0)).collect();
        let point_f: Vec<F> = point.iter().map(F::from).collect();

        let mut prover_transcript = PcsTranscript::new();
        let open_result = TestZip::open::<F>(&pp, &poly, &data, &point, &mut prover_transcript);
        assert!(open_result.is_ok());
        let proof = prover_transcript.into_proof();
        let eval_f: F = open_result.unwrap();

        let mut verifier_transcript = PcsTranscript::from_proof(&proof);

        let verification_result =
            TestZip::verify(&pp, &comm, &point_f, &eval_f, &mut verifier_transcript);

        assert!(
            verification_result.is_ok(),
            "Verification failed: {verification_result:?}"
        );
    }

    #[test]
    fn polynomial_coefficients_at_maximum_bit_size_boundary() {
        let num_vars = 4;
        let (pp, _) = setup_test_params(num_vars);

        let mut evals: Vec<<Zt as ZipTypes>::Eval> =
            (0..1 << num_vars as i32).map(Int::from).collect();
        evals[1] = Int::from(i64::MAX);
        let poly = DenseMultilinearExtension::from_evaluations_vec(num_vars, evals);

        let (data, comm) = TestZip::commit(&pp, &poly).unwrap();

        // A point of [1, 0, 0, 0] will evaluate to poly.evaluations[1].
        let mut point = vec![<Zt as ZipTypes>::Pt::ZERO; num_vars];
        point[0] = <Zt as ZipTypes>::Pt::ONE;
        let point_f: Vec<F> = point.iter().map(F::from).collect();

        let mut prover_transcript = PcsTranscript::new();
        let open_result = TestZip::open::<F>(&pp, &poly, &data, &point, &mut prover_transcript);
        assert!(open_result.is_ok());
        let proof = prover_transcript.into_proof();
        let eval_f: F = open_result.unwrap();

        let mut verifier_transcript = PcsTranscript::from_proof(&proof);
        let expected_eval = poly
            .evaluate(&point)
            .expect("failed to evaluate polynomial");
        assert_eq!(eval_f, expected_eval.into());

        let verification_result =
            TestZip::verify(&pp, &comm, &point_f, &eval_f, &mut verifier_transcript);

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

        let point: Vec<<Zt as ZipTypes>::Pt> = vec![Int::from(1), Int::from(2)];
        let point_f: Vec<F> = point.iter().map(F::from).collect();
        let mut prover_transcript = PcsTranscript::new();
        let open_result = TestZip::open::<F>(&pp, &poly, &data, &point, &mut prover_transcript);
        assert!(open_result.is_ok());
        let proof = prover_transcript.into_proof();
        let eval_f: F = open_result.unwrap();

        let mut verifier_transcript = PcsTranscript::from_proof(&proof);
        let expected_eval = poly
            .evaluate(&point)
            .expect("failed to evaluate polynomial");
        assert_eq!(eval_f, expected_eval.into());

        let verification_result =
            TestZip::verify(&pp, &comm, &point_f, &eval_f, &mut verifier_transcript);

        assert!(verification_result.is_ok());
    }
}
