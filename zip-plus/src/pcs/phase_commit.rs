use crate::{
    ZipError,
    code::LinearCode,
    merkle::MerkleTree,
    pcs::{
        structs::{ZipPlus, ZipPlusCommitment, ZipPlusHint, ZipPlusParams, ZipTypes},
        utils::validate_input,
    },
    poly::mle::DenseMultilinearExtension,
};
use ark_std::{cfg_chunks, cfg_chunks_mut};
use crypto_primitives::DenseRowMatrix;
use uninit::out_ref::Out;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

impl<Zt: ZipTypes<DEGREE>, Lc: LinearCode<Zt, DEGREE>, const DEGREE: usize>
    ZipPlus<Zt, Lc, DEGREE>
{
    /// Creates a commitment to a multilinear polynomial using the ZIP PCS
    /// scheme.
    ///
    /// This function implements the commitment phase of the ZIP polynomial
    /// commitment scheme. It encodes the polynomial's evaluations using a
    /// linear error-correcting code and then creates Merkle tree
    /// commitments to the encoded rows.
    ///
    /// # Algorithm
    /// 1. Validates that the polynomial's number of variables matches the
    ///    parameters.
    /// 2. Arranges the polynomial's evaluations into a matrix with
    ///    `pp.num_rows` rows.
    /// 3. Encodes each row using the specified linear code, expanding its
    ///    length from `row_len` to `codeword_len`.
    /// 4. Constructs a Merkle tree for each encoded row to produce a root hash.
    /// 5. Returns the full commitment data (for the prover) and a compact
    ///    commitment (for the verifier).
    ///
    /// # Type Parameters
    /// - `F`: A generic [`Field`] type used for validation purposes, though not
    ///   for the commitment itself, which operates on integers.
    ///
    /// # Parameters
    /// - `pp`: Public parameters (`ZipPlusParams`) containing the configuration
    ///   for the commitment scheme.
    /// - `poly`: The multilinear polynomial (`DenseMultilinearExtension`) to be
    ///   committed to.
    ///
    /// # Returns
    /// A `Result` containing a tuple of:
    /// - `ZipPlusHint`: Full data including encoded rows and Merkle tree, kept
    ///   by the prover for the opening phase.
    /// - `ZipPlusCommitment`: The compact commitment, consisting of only the
    ///   Merkle roots, to be sent to the verifier.
    ///
    /// # Errors
    /// - Returns `Error::InvalidPcsParam` if the polynomial has more variables
    ///   than the parameters support.
    ///
    /// # Panics
    /// - Panics if the number of polynomial evaluations does not perfectly
    ///   match the expected matrix size (`pp.num_rows *
    ///   pp.linear_code.row_len()`).
    /// - Panics if the number of generated Merkle trees does not match
    ///   `pp.num_rows`, indicating an internal logic error.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn commit(
        pp: &ZipPlusParams<Zt, Lc, DEGREE>,
        poly: &DenseMultilinearExtension<Zt::Eval>,
    ) -> Result<(ZipPlusHint<Zt::Cw>, ZipPlusCommitment), ZipError> {
        validate_input::<Zt, Lc, bool, DEGREE>("commit", pp.num_vars, [poly], None)?;

        let expected_num_evals = pp.num_rows * pp.linear_code.row_len();
        assert_eq!(
            poly.evaluations.len(),
            expected_num_evals,
            "Polynomial has an incorrect number of evaluations ({}) for the expected matrix size ({})",
            poly.evaluations.len(),
            expected_num_evals
        );

        let row_len = pp.linear_code.row_len();

        let cw_matrix = Self::encode_rows(pp, row_len, &poly.evaluations);

        let merkle_tree = MerkleTree::new(&cw_matrix.to_rows_slices());
        let root = merkle_tree.root();

        Ok((
            ZipPlusHint::new(cw_matrix, merkle_tree),
            ZipPlusCommitment { root },
        ))
    }

    /// Creates a commitment without constructing Merkle trees.
    ///
    /// This function performs the encoding step of the commitment phase but
    /// deliberately skips the computationally intensive step of building
    /// Merkle trees. It is intended **for testing and benchmarking purposes
    /// only**, where the full commitment structure is not required.
    ///
    /// # Parameters
    /// - `pp`: Public parameters (`ZipPlusParams`).
    /// - `poly`: The multilinear polynomial to commit to.
    ///
    /// # Returns
    /// A `Result` containing `ZipPlusHint` with the encoded rows but
    /// empty Merkle trees, and a `ZipPlusCommitment` with an empty
    /// vector of roots.
    #[allow(dead_code)]
    pub fn commit_no_merkle(
        pp: &ZipPlusParams<Zt, Lc, DEGREE>,
        poly: &DenseMultilinearExtension<Zt::Eval>,
    ) -> Result<DenseRowMatrix<Zt::Cw>, ZipError> {
        validate_input::<Zt, Lc, bool, DEGREE>("commit", pp.num_vars, [poly], None)?;

        let row_len = pp.linear_code.row_len();

        let rows = Self::encode_rows(pp, row_len, &poly.evaluations);
        Ok(rows)
    }

    /// Commits to a batch of multilinear polynomials.
    ///
    /// This function iterates through a slice of polynomials and calls
    /// [`commit`] on each one, collecting the results into a vector.
    ///
    /// # Parameters
    /// - `pp`: Public parameters (`ZipPlusParams`) shared across all
    ///   commitments.
    /// - `polys`: A slice of `DenseMultilinearExtension` polynomials to commit
    ///   to.
    ///
    /// # Returns
    /// A `Result` containing a `Vec` of commitment tuples, where each tuple
    /// corresponds to a polynomial in the input slice.
    #[allow(clippy::type_complexity)]
    pub fn batch_commit(
        pp: &ZipPlusParams<Zt, Lc, DEGREE>,
        polys: &[DenseMultilinearExtension<Zt::Eval>],
    ) -> Result<Vec<(ZipPlusHint<Zt::Cw>, ZipPlusCommitment)>, ZipError> {
        polys.iter().map(|poly| Self::commit(pp, poly)).collect()
    }

    /// Encodes the evaluations of a polynomial by arranging them into rows and
    /// applying a linear code.
    ///
    /// This function treats the polynomial's flat evaluation vector as a matrix
    /// with `pp.num_rows` and encodes each row individually. The resulting
    /// encoded rows are concatenated into a single flat vector. This
    /// operation can be parallelized if the `parallel` feature is enabled.
    ///
    /// # Parameters
    /// - `pp`: Public parameters containing matrix dimensions and the linear
    ///   code.
    /// - `codeword_len`: The length of an encoded row.
    /// - `row_len`: The length of a row before encoding.
    /// - `poly`: The polynomial whose evaluations are to be encoded.
    ///
    /// # Returns
    /// A `Vec<Int<K>>` containing all the encoded rows concatenated together.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn encode_rows(
        pp: &ZipPlusParams<Zt, Lc, DEGREE>,
        row_len: usize,
        evals: &[Zt::Eval],
    ) -> DenseRowMatrix<Zt::Cw> {
        let codeword_len = pp.linear_code.codeword_len();

        // Performance: Using DenseRowMatrix's linearized row in an uninit form
        // is much more performant that using Vec<Vec<_>>.
        let mut encoded_matrix = DenseRowMatrix::<Zt::Cw>::uninit(pp.num_rows, codeword_len);

        cfg_chunks_mut!(encoded_matrix.data, codeword_len)
            .zip(cfg_chunks!(evals, row_len))
            .for_each(|(row, evals)| {
                let encoded: Vec<Zt::Cw> = pp.linear_code.encode(evals);
                Out::from(row).copy_from_slice(encoded.as_slice());
            });

        // Safe because we have just initialized all elements.
        unsafe { encoded_matrix.init() }
    }
}

#[cfg(test)]
#[allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
mod tests {
    use std::slice::from_ref;

    use crate::{
        code::{LinearCode, raa::RaaCode, raa_sign_flip::RaaSignFlippingCode},
        merkle::{MerkleTree, MtHash},
        pcs::{
            structs::{ZipPlus, ZipPlusParams, ZipTypes},
            test_utils::*,
        },
        poly::{dense::DensePolynomial, mle::DenseMultilinearExtension},
    };
    use crypto_bigint::{Random, U64, U192, Word};
    use crypto_primitives::{
        Matrix, boolean::Boolean, crypto_bigint_int::Int, crypto_bigint_monty::MontyField,
    };
    use itertools::Itertools;
    use num_traits::Zero;
    use rand::{Rng, rng};

    const INT_LIMBS: usize = U64::LIMBS;

    const N: usize = INT_LIMBS;
    const K: usize = INT_LIMBS * 3;
    const M: usize = INT_LIMBS * 6;
    const DEGREE: usize = 2;

    type Zt = TestZipTypes<N, K, M>;
    type C = RaaSignFlippingCode<Zt, 4, 0>;

    type PolyZt = TestPolyZipTypes<K, M, DEGREE>;
    type PolyC = RaaCode<PolyZt, 4, DEGREE>;

    type TestZip = ZipPlus<Zt, C, 0>;
    type TestPolyZip = ZipPlus<PolyZt, PolyC, DEGREE>;

    #[test]
    fn commit_rejects_too_many_variables() {
        let (pp, _) = setup_test_params(3); // Setup for 3 variables

        // Create polynomial with 4 variables (which is > 3)
        let evaluations = (1..=16).map(Int::from).collect();
        let poly = DenseMultilinearExtension::from_evaluations_vec(4, evaluations, Zero::zero());

        let result = TestZip::commit(&pp, &poly);
        assert!(result.is_err());
    }

    #[test]
    fn commit_is_deterministic() {
        let (pp, poly) = setup_test_params(3);

        let result1 = TestZip::commit(&pp, &poly).unwrap();
        let result2 = TestZip::commit(&pp, &poly).unwrap();

        assert_eq!(result1.1.root, result2.1.root);
    }

    #[test]
    fn different_polynomials_produce_different_commitments() {
        let (pp, _) = setup_test_params(3);

        let poly1 =
            DenseMultilinearExtension::from_evaluations_vec(3, vec![Int::from(1); 8], Zero::zero());
        let poly2 =
            DenseMultilinearExtension::from_evaluations_vec(3, vec![Int::from(2); 8], Zero::zero());

        let (_, commitment1) = TestZip::commit(&pp, &poly1).unwrap();
        let (_, commitment2) = TestZip::commit(&pp, &poly2).unwrap();

        assert_ne!(commitment1.root, commitment2.root);
    }

    #[test]
    fn commit_succeeds_for_small_polynomial() {
        let code = C::new(16, RAA_CFG);
        let pp = ZipPlusParams::new(4, 4, code);

        let evaluations = vec![Int::from(42); 16];
        let poly = DenseMultilinearExtension::from_evaluations_vec(4, evaluations, Zero::zero());

        let result = TestZip::commit(&pp, &poly);
        assert!(result.is_ok());
    }

    #[test]
    fn commit_succeeds_for_two_variables() {
        let code = C::new(4, RAA_CFG);
        let pp = ZipPlusParams::new(2, 2, code);

        let evaluations = vec![Int::from(1), Int::from(2), Int::from(3), Int::from(4)];
        let poly = DenseMultilinearExtension::from_evaluations_vec(2, evaluations, Zero::zero());

        let result = TestZip::commit(&pp, &poly);
        assert!(result.is_ok());
    }

    #[test]
    fn batch_commit_succeeds() {
        let (pp, _) = setup_test_params(3);

        let polys = vec![
            DenseMultilinearExtension::from_evaluations_vec(
                3,
                (1..=8).map(Int::from).collect(),
                Zero::zero(),
            ),
            DenseMultilinearExtension::from_evaluations_vec(
                3,
                (9..=16).map(Int::from).collect(),
                Zero::zero(),
            ),
        ];

        let results = TestZip::batch_commit(&pp, &polys);
        assert!(results.is_ok());

        let outputs = results.unwrap();
        assert_eq!(outputs.len(), 2);
        assert_ne!(outputs[0].1.root, outputs[1].1.root);
    }

    #[test]
    fn encode_rows_produces_correct_size() {
        let (pp, poly) = setup_test_params(3);
        let encoded = TestZip::encode_rows(&pp, pp.linear_code.row_len(), &poly.evaluations);

        assert_eq!(encoded.num_rows, pp.num_rows);
        assert_eq!(encoded.num_cols, pp.linear_code.codeword_len());
    }

    /// Verifies that the output of `encode_rows` is semantically correct by
    /// comparing it to a direct, row-by-row encoding.
    #[test]
    fn encoded_rows_match_linear_code_definition() {
        let (pp, poly) = setup_test_params(3);
        let encoded = TestZip::encode_rows(&pp, pp.linear_code.row_len(), &poly.evaluations);

        for (i, row_chunk) in encoded.as_rows().enumerate() {
            let start = i * pp.linear_code.row_len();
            let end = start + pp.linear_code.row_len();
            let row_evals = &poly.evaluations[start..end];
            let expected_encoding = pp.linear_code.encode(row_evals);
            assert_eq!(
                row_chunk,
                expected_encoding.as_slice(),
                "Row {i} encoding mismatch",
            );
        }
    }

    /// Verifies that corrupting the encoded data after commitment results in a
    /// different Merkle root.
    #[test]
    fn corrupted_encoding_changes_merkle_root() {
        let (pp, poly) = setup_test_params(3);
        let (data, commitment) = TestZip::commit(&pp, &poly).unwrap();

        assert!(!data.cw_matrix.is_empty());
        let mut cw_matrix = data.cw_matrix.to_rows();
        cw_matrix[0][0] = Int::from(999999);
        let corrupted_row = cw_matrix[0].clone();
        let new_tree = MerkleTree::new(&[corrupted_row.as_slice()]);
        assert_ne!(
            new_tree.root(),
            commitment.root,
            "Corruption should change Merkle root"
        );
    }

    #[test]
    fn batch_commit_with_single_polynomial_is_consistent() {
        let (pp, poly) = setup_test_params(3);

        let batch_result = TestZip::batch_commit(&pp, from_ref(&poly));
        let mut batch_outputs = batch_result.unwrap();
        let (batch_data, batch_commitment) = batch_outputs.remove(0);

        let single_result = TestZip::commit(&pp, &poly);
        let (single_data, single_commitment) = single_result.unwrap();

        assert_eq!(batch_commitment.root, single_commitment.root);
        assert_eq!(batch_data.cw_matrix, single_data.cw_matrix);
    }

    #[test]
    fn encoded_rows_are_nonzero_for_nonzero_input() {
        let (pp, poly) = setup_test_params(3);
        let encoded = TestZip::encode_rows(&pp, pp.linear_code.row_len(), &poly.evaluations);

        assert_eq!(encoded.num_rows, pp.num_rows);
        assert_eq!(encoded.num_cols, pp.linear_code.codeword_len());

        let non_zero_count = encoded
            .as_rows()
            .flatten()
            .filter(|&&x| x != Int::from(0))
            .count();
        assert!(non_zero_count > 0);
    }

    #[test]
    fn commit_produces_correct_merkle_tree_count() {
        let (pp, poly) = setup_test_params(3);
        let (hint, _) = TestZip::commit(&pp, &poly).unwrap();

        assert_eq!(hint.cw_matrix.num_rows, pp.num_rows);
        assert_eq!(hint.cw_matrix.num_cols, pp.linear_code.codeword_len());
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn encoding_is_consistent_across_threads() {
        use rayon::prelude::*;

        let num_vars = 6;
        let poly_size = 1 << num_vars;
        let evaluations = (1..=poly_size).map(|v| Int::from(v as i32)).collect();
        let poly =
            DenseMultilinearExtension::from_evaluations_vec(num_vars, evaluations, Zero::zero());

        let results: Vec<Vec<Vec<Int<4>>>> = (0..10)
            .into_par_iter()
            .map(|_| {
                let code = C::new(poly_size, RAA_CFG);
                let pp = ZipPlusParams::new(num_vars, 8, code);

                let rows = TestZip::encode_rows(&pp, pp.linear_code.row_len(), &poly.evaluations);
                let rows: Vec<Vec<_>> = rows
                    .data
                    .chunks_exact(pp.linear_code.codeword_len())
                    .map(|chunk| chunk.to_vec())
                    .collect();
                rows
            })
            .collect();

        assert!(
            results.windows(2).all(|w| w[0] == w[1]),
            "Parallel encoding runs produced inconsistent results"
        );
    }

    #[test]
    fn commit_succeeds_for_zero_polynomial() {
        let (pp, _) = setup_test_params(3);
        let zero_poly =
            DenseMultilinearExtension::from_evaluations_vec(3, vec![Int::from(0); 8], Zero::zero());
        let result = TestZip::commit(&pp, &zero_poly);
        assert!(result.is_ok());
    }

    #[test]
    fn commit_succeeds_for_alternating_values() {
        let (pp, _) = setup_test_params(3);
        let alternating = (0..8)
            .map(|i| Int::from(if i % 2 == 0 { 1 } else { -1 }))
            .collect();
        let poly = DenseMultilinearExtension::from_evaluations_vec(3, alternating, Zero::zero());
        let result = TestZip::commit(&pp, &poly);
        assert!(result.is_ok());
    }

    #[test]
    fn batch_commit_on_empty_slice_is_ok() {
        let (pp, _) = setup_test_params(3);
        let empty_polys: Vec<DenseMultilinearExtension<Int<INT_LIMBS>>> = vec![];
        let results = TestZip::batch_commit(&pp, &empty_polys);
        assert!(results.is_ok());
        assert!(results.unwrap().is_empty());
    }

    #[test]
    fn encode_rows_succeeds_for_single_row() {
        let code = C::new(4, RAA_CFG);
        let pp = ZipPlusParams::new(2, 1, code);

        // Create a polynomial with 2 variables and 4 evaluations
        let evaluations = vec![Int::from(5); 4];
        let poly = DenseMultilinearExtension::from_evaluations_vec(2, evaluations, Zero::zero());
        let encoded = TestZip::encode_rows(&pp, pp.linear_code.row_len(), &poly.evaluations);
        assert_eq!(encoded.num_rows, 1);
        assert_eq!(encoded.num_cols, pp.linear_code.codeword_len());
    }

    #[test]
    fn encode_rows_succeeds_for_single_poly_row() {
        let code = PolyC::new(4, RAA_CFG);
        let pp = ZipPlusParams::new(2, 1, code);

        // Create a polynomial with 2 variables and 4 evaluations
        let evaluations = vec![
            DensePolynomial::new(vec![Boolean::FALSE, Boolean::FALSE]),
            DensePolynomial::new(vec![Boolean::FALSE, Boolean::TRUE]),
            DensePolynomial::new(vec![Boolean::TRUE, Boolean::FALSE]),
        ];
        let poly = DenseMultilinearExtension::from_evaluations_vec(2, evaluations, Zero::zero());
        let encoded = TestPolyZip::encode_rows(&pp, pp.linear_code.row_len(), &poly.evaluations);
        assert_eq!(encoded.num_rows, 1);
        assert_eq!(encoded.num_cols, pp.linear_code.codeword_len());
    }

    #[test]
    fn matrix_dimensions_are_invariant() {
        let test_cases = vec![(2, 2), (4, 4), (6, 8)];
        for (num_vars, expected_rows) in test_cases {
            assert_eq!(1 << (num_vars / 2), expected_rows);

            let (pp, poly) = setup_test_params(num_vars);
            assert_eq!(pp.num_rows, expected_rows);
            let result = TestZip::commit(&pp, &poly);
            assert!(result.is_ok());
        }
    }

    #[test]
    #[should_panic]
    fn reject_incompatible_dimensions() {
        let (pp, poly) = setup_test_params(3);
        let incompatible_pp = ZipPlusParams::new(3, 3, pp.linear_code);
        let _ = TestZip::commit(&incompatible_pp, &poly);
    }

    #[test]
    fn linear_code_preserves_linearity() {
        let (pp, poly) = setup_test_params(4);
        let encoded = TestZip::encode_rows(&pp, pp.linear_code.row_len(), &poly.evaluations);
        let row_len = pp.linear_code.row_len();
        let codeword_len = pp.linear_code.codeword_len();
        let row1_evals = &poly.evaluations[0..row_len];
        let row2_evals = &poly.evaluations[row_len..2 * row_len];
        let a = Int::from(3);
        let b = Int::<4>::from(5);
        let combined_evals: Vec<_> = (0..row_len)
            .map(|i| a * row1_evals[i] + b.resize() * row2_evals[i])
            .collect();
        let combined_encoded = pp.linear_code.encode(&combined_evals);
        let rows = encoded.as_rows().collect_vec();
        let row1_encoded = rows[0];
        let row2_encoded = rows[1];
        let expected_combined: Vec<_> = (0..codeword_len)
            .map(|i| a.resize() * row1_encoded[i] + b.resize() * row2_encoded[i])
            .collect();
        assert_eq!(combined_encoded, expected_combined);
    }

    #[test]
    #[should_panic]
    fn commit_panics_if_evaluations_not_multiple_of_row_len() {
        let (pp, mut poly) = setup_test_params(4);
        poly.evaluations.truncate(15);
        assert_eq!(poly.evaluations.len(), 15);
        let _ = TestZip::commit(&pp, &poly);
    }

    #[test]
    fn commit_with_many_variables() {
        let num_vars = 16;
        let (pp, poly) = setup_test_params(num_vars);
        assert_eq!(pp.num_vars, num_vars);
        let result = TestZip::commit(&pp, &poly);
        assert!(result.is_ok());
    }

    #[test]
    fn commit_with_smallest_matrix_arrangement() {
        let (pp, poly) = setup_test_params(2);
        assert_eq!(pp.num_rows, 2);
        assert_eq!(pp.linear_code.row_len(), 2);
        let result = TestZip::commit(&pp, &poly);
        assert!(result.is_ok());
    }

    #[test]
    fn encode_rows_handles_large_integer_values() {
        let (pp, _) = setup_test_params(3);
        let max_val = Int::<INT_LIMBS>::from(i64::MAX);
        let poly =
            DenseMultilinearExtension::from_evaluations_vec(3, vec![max_val; 8], Zero::zero());
        let encoded_rows = TestZip::encode_rows(&pp, pp.linear_code.row_len(), &poly.evaluations);
        assert_eq!(encoded_rows.num_rows, pp.num_rows);
        assert_eq!(encoded_rows.num_cols, pp.linear_code.codeword_len());
    }

    #[test]
    #[should_panic(expected = "row_width.is_power_of_two()")]
    fn merkle_tree_new_panics_on_non_power_of_two_leaves() {
        let leaves_data: Vec<Int<INT_LIMBS>> = (0..7).map(Int::from).collect();
        let _ = MerkleTree::new(&[leaves_data.as_slice()]);
    }

    #[test]
    fn proof_sizeis_correct_for_parameters() {
        fn calculate_expected_test_transcript_size_bytes(pp: &ZipPlusParams<Zt, C, 0>) -> usize {
            let size_of_zt_k = K * size_of::<Word>();
            let size_of_zt_m = M * size_of::<Word>();
            let size_of_path_len = size_of::<u64>();
            let size_of_path_elem = size_of::<MtHash>();
            let size_of_dimension = size_of::<u64>();

            let codeword_len = pp.linear_code.codeword_len();
            let merkle_depth = codeword_len.next_power_of_two().ilog2() as usize;

            let proximity_phase_size = pp.linear_code.row_len() * size_of_zt_m;

            let column_values_size = pp.num_rows * size_of_zt_k;
            let single_merkle_proof_size =
                size_of_dimension * 2 + size_of_path_len + merkle_depth * size_of_path_elem;
            let column_opening_phase_size =
                Zt::NUM_COLUMN_OPENINGS * (column_values_size + single_merkle_proof_size);

            proximity_phase_size + column_opening_phase_size
        }

        fn calculate_expected_proof_size_bytes(pp: &ZipPlusParams<Zt, C, 0>) -> usize {
            // F stored as `(value, modulus)` where both `value` and `modulus` are of
            // length `len`.
            // `len` itself is not stored as its known in compile time.
            let size_of_f = 2 * U192::LIMBS * size_of::<Word>();
            let evaluation_phase_size = pp.linear_code.row_len() * size_of_f;

            calculate_expected_test_transcript_size_bytes(pp) + evaluation_phase_size
        }

        const LIMBS: usize = 3;
        type F = MontyField<LIMBS>;

        let mut rng = rng();
        let num_vars = 4;
        let poly_size = 1 << num_vars;
        let linear_code = C::new(poly_size, RAA_CFG);
        let param = TestZip::setup(poly_size, linear_code);
        let evaluations: Vec<_> = (0..poly_size)
            .map(|_| <Zt as ZipTypes<_>>::Eval::from(rng.random::<i8>()))
            .collect();
        let mle =
            DenseMultilinearExtension::from_evaluations_slice(num_vars, &evaluations, Zero::zero());
        let point: Vec<_> = (0..num_vars)
            .map(|_| <Zt as ZipTypes<_>>::Pt::random(&mut rng))
            .collect();

        let (hint, _) = TestZip::commit(&param, &mle).unwrap();

        let test_transcript = TestZip::test(&param, &mle, &hint).unwrap();
        let actual_test_transcript_size_bytes = test_transcript.0.stream.get_ref().len();
        let expected_test_transcript_size_bytes =
            calculate_expected_test_transcript_size_bytes(&param);
        assert_eq!(
            actual_test_transcript_size_bytes,
            expected_test_transcript_size_bytes
        );

        let (_, proof) = TestZip::evaluate::<F>(&param, &mle, &point, test_transcript).unwrap();
        let actual_proof_size_bytes = proof.0.len();
        let expected_proof_size_bytes = calculate_expected_proof_size_bytes(&param);
        assert_eq!(actual_proof_size_bytes, expected_proof_size_bytes);
    }
}
