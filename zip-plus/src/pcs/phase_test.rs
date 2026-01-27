use crate::{
    ZipError,
    code::LinearCode,
    combine_rows,
    pcs::{
        ZipPlusTestTranscript,
        structs::{ZipPlus, ZipPlusHint, ZipPlusParams, ZipTypes},
        utils::{ColumnOpening, validate_input},
    },
    pcs_transcript::PcsTranscript,
};
use num_traits::{CheckedAdd, ConstOne, ConstZero, Zero};
use zinc_poly::{Polynomial, mle::DenseMultilinearExtension};
use zinc_transcript::traits::Transcript;
use zinc_utils::{add, inner_product::InnerProduct, mul_by_scalar::MulByScalar};
use zinc_utils::cfg_iter_mut;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// TODO(alex): Return overflow checks!
impl<Zt: ZipTypes, Lc: LinearCode<Zt>> ZipPlus<Zt, Lc> {
    #[allow(clippy::arithmetic_side_effects)]
    pub fn test(
        pp: &ZipPlusParams<Zt, Lc>,
        poly: &DenseMultilinearExtension<Zt::Eval>,
        commit_hint: &ZipPlusHint<Zt::Cw>,
    ) -> Result<ZipPlusTestTranscript, ZipError> {
        validate_input::<Zt, Lc, bool>("test", pp.num_vars, &[poly], &[])?;

        let mut transcript = PcsTranscript::new();

        // If we can take linear combinations, perform the proximity test
        if pp.num_rows > 1 {
            // Values to evaluate the coefficients at
            let alphas = if Zt::Comb::DEGREE_BOUND.is_zero() {
                // If we have just one coefficient
                // we don't take an RLC.
                vec![Zt::Chal::ONE]
            } else {
                transcript
                    .fs_transcript
                    // NB: To take an inner product of coeffs
                    // of a polynomial with the non-strict degree bound B
                    // with a slice of challenges
                    // we need to sample B + 1 challenges.
                    .get_challenges::<Zt::Chal>(Zt::Comb::DEGREE_BOUND + 1)
            };

            // Coefficients for the linear combination of polynomial with evaluated
            // coefficients
            let coeffs = transcript
                .fs_transcript
                .get_challenges::<Zt::Chal>(pp.num_rows);

            // u' in the Zinc paper
            let combined_row = combine_rows!(
                &coeffs,
                poly.evaluations.iter(),
                |eval| Zt::EvalDotChal::inner_product(eval, &alphas, Zt::CombR::ZERO),
                |acc: Zt::CombR, scaled| acc + &scaled,
                pp.linear_code.row_len(),
                Zt::CombR::ZERO
            );

            // transcript.write_const_many(&combined_row)?;
            std::hint::black_box(combined_row);
        }

        // // Open merkle tree for each column drawn
        // for _ in 0..Zt::NUM_COLUMN_OPENINGS {
        //     let column = transcript.squeeze_challenge_idx(pp.linear_code.codeword_len());
        //     Self::open_merkle_trees_for_column(commit_hint, column, &mut transcript)?;
        // }

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
    };
    use crypto_bigint::U64;
    use crypto_primitives::crypto_bigint_int::Int;
    use num_traits::{ConstOne, Zero};
    use zinc_poly::mle::DenseMultilinearExtension;

    const INT_LIMBS: usize = U64::LIMBS;

    const N: usize = INT_LIMBS;
    const K: usize = INT_LIMBS * 4;
    const M: usize = INT_LIMBS * 8;
    const DEGREE_PLUS_ONE: usize = 3;

    type Zt = TestZipTypes<N, K, M>;
    type C = RaaSignFlippingCode<Zt, TestRaaConfig, 4>;

    type PolyZt = TestPolyZipTypes<K, M, DEGREE_PLUS_ONE>;
    type PolyC = RaaCode<PolyZt, TestRaaConfig, 4>;

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
