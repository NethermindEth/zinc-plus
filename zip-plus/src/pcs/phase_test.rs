use crate::{
    ZipError,
    code::LinearCode,
    pcs::{
        ZipPlusTestTranscript,
        structs::{ZipPlus, ZipPlusHint, ZipPlusParams, ZipTypes},
        utils::{ColumnOpening, validate_input},
    },
    pcs_transcript::PcsTranscript,
    poly::{Polynomial, mle::DenseMultilinearExtension},
    traits::{FromRef, Transcript},
    utils::combine_rows,
};
use itertools::Itertools;
use num_traits::ConstZero;

impl<Zt: ZipTypes, Lc: LinearCode<Zt>> ZipPlus<Zt, Lc> {
    pub fn test(
        pp: &ZipPlusParams<Zt, Lc>,
        poly: &DenseMultilinearExtension<Zt::Eval>,
        commit_hint: &ZipPlusHint<Zt::Cw>,
    ) -> Result<ZipPlusTestTranscript, ZipError> {
        validate_input::<Zt, Lc, bool>("test", pp.num_vars, [poly], None)?;

        let mut transcript = PcsTranscript::new();

        // If we can take linear combinations, perform the proximity test
        if pp.num_rows > 1 {
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
            let combined_row =
                combine_rows(&coeffs, &evals, pp.linear_code.row_len(), Zt::CombR::ZERO);

            transcript.write_const_many(&combined_row)?;
        }

        // Open merkle tree for each column drawn
        for _ in 0..Zt::NUM_COLUMN_OPENINGS {
            let column = transcript.squeeze_challenge_idx(pp.linear_code.codeword_len());
            Self::open_merkle_trees_for_column(commit_hint, column, &mut transcript)?;
        }

        Ok(transcript.into())
    }

    pub(super) fn open_merkle_trees_for_column(
        commit_hint: &ZipPlusHint<Zt::Cw>,
        column: usize,
        transcript: &mut PcsTranscript,
    ) -> Result<(), ZipError> {
        let column_values = commit_hint.cw_matrix.as_rows().map(|row| &row[column]);

        // Write the elements in the squeezed column to the shared transcript
        transcript.write_const_many(column_values)?;

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
        code::{raa::RaaCode, raa_sign_flip::RaaSignFlippingCode},
        merkle::MerkleTree,
        pcs::{
            structs::{ZipPlus, ZipPlusHint},
            test_utils::*,
        },
        poly::mle::DenseMultilinearExtension,
    };
    use crypto_bigint::U64;
    use crypto_primitives::crypto_bigint_int::Int;
    use num_traits::{ConstOne, Zero};

    const INT_LIMBS: usize = U64::LIMBS;

    const N: usize = INT_LIMBS;
    const K: usize = INT_LIMBS * 4;
    const M: usize = INT_LIMBS * 8;
    const DEGREE: usize = 2;

    type Zt = TestZipTypes<N, K, M>;
    type C = RaaSignFlippingCode<Zt, 4>;

    type PolyZt = TestPolyZipTypes<N, K, M, DEGREE>;
    type PolyC = RaaCode<PolyZt, 4>;

    type TestZip = ZipPlus<Zt, C>;
    type TestPolyZip = ZipPlus<PolyZt, PolyC>;

    #[test]
    fn successful_testing_with_correct_polynomial_and_hint() {
        let num_vars = 4;
        let (pp, poly) = setup_test_params(num_vars);
        let (hint, _) = TestZip::commit(&pp, &poly).unwrap();
        let result = TestZip::test(&pp, &poly, &hint);
        assert!(result.is_ok());
    }

    #[test]
    fn successful_testing_with_correct_polynomial_and_hint_poly() {
        let num_vars = 4;
        let (pp, poly) = setup_poly_test_params(num_vars);
        let (hint, _) = TestPolyZip::commit(&pp, &poly).unwrap();
        let result = TestPolyZip::test(&pp, &poly, &hint);
        assert!(result.is_ok());
    }

    #[test]
    fn successful_testing_with_a_close_codeword() {
        let num_vars = 4;
        let (pp, poly) = setup_test_params(num_vars);

        let (mut original_hint, _) = TestZip::commit(&pp, &poly).unwrap();

        let mut corrupted_rows = original_hint.cw_matrix.to_rows_slices_mut();
        assert!(!corrupted_rows.is_empty());
        corrupted_rows[0][0] += Int::ONE;

        let corrupted_merkle_tree = MerkleTree::new(&original_hint.cw_matrix.to_rows_slices());
        let corrupted_rows_hint = ZipPlusHint::new(original_hint.cw_matrix, corrupted_merkle_tree);

        let result = TestZip::test(&pp, &poly, &corrupted_rows_hint);

        assert!(result.is_ok());
    }

    #[test]
    fn failed_opening_due_to_oversized_polynomial_coefficients() {
        let num_vars = 4;
        let (pp, _) = setup_test_params(num_vars);

        let oversized_num_vars = 5;
        let oversized_evals: Vec<_> = (0..1 << oversized_num_vars).map(Int::from).collect();
        let oversized_poly = DenseMultilinearExtension::from_evaluations_vec(
            oversized_num_vars,
            oversized_evals,
            Zero::zero(),
        );

        // This hint is for a 4-variable poly, but we need it as a placeholder.
        let (hint, _) = TestZip::commit(&pp, &setup_test_params::<N, K, M>(num_vars).1).unwrap();

        let result = TestZip::test(&pp, &oversized_poly, &hint);

        assert!(result.is_err());
    }
}
