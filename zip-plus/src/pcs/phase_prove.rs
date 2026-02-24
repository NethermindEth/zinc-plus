use std::vec;

use crate::{
    ZipError, code::LinearCode, combine_rows, pcs::{
        ZipPlusProof,
        structs::{ZipPlus, ZipPlusHint, ZipPlusParams, ZipTypes},
        utils::{point_to_tensor, validate_input},
    }, pcs_transcript::PcsTranscript
};
use crypto_primitives::{FromWithConfig, IntoWithConfig, PrimeField};
use num_traits::{ConstOne, ConstZero, Zero};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use zinc_poly::{Polynomial, mle::DenseMultilinearExtension};
use zinc_transcript::traits::{Transcribable, Transcript};
use zinc_utils::{
    UNCHECKED, cfg_iter_mut,
    from_ref::FromRef,
    inner_product::{InnerProduct, MBSInnerProduct},
    mul_by_scalar::MulByScalar,
    projectable_to_field::ProjectableToField,
};

impl<Zt: ZipTypes, Lc: LinearCode<Zt>> ZipPlus<Zt, Lc> {
    #[allow(clippy::arithmetic_side_effects)]
    pub fn prove<F, const CHECK_FOR_OVERFLOW: bool>(
        pp: &ZipPlusParams<Zt, Lc>,
        polys: &[DenseMultilinearExtension<Zt::Eval>],
        point: &[Zt::Pt],
        commit_hint: &ZipPlusHint<Zt::Cw>,
    ) -> Result<(Vec<F>, ZipPlusProof), ZipError>
    where
        F: PrimeField
            + for<'a> FromWithConfig<&'a Zt::CombR>
            + for<'a> FromWithConfig<&'a Zt::Chal>
            + for<'a> FromWithConfig<&'a Zt::Pt>
            + for<'a> MulByScalar<&'a F>
            + FromRef<F>,
        F::Inner: FromRef<Zt::Fmod> + Transcribable,
        Zt::Eval: ProjectableToField<F>,
    {
        let batch_size = polys.len();
        validate_input::<Zt, Lc, _>(
            "prove", pp.num_vars, batch_size,
            &polys.iter().collect::<Vec<_>>(), &[point],
        )?;

        let num_rows = pp.num_rows;
        let row_len = pp.linear_code.row_len();
        let mut transcript = PcsTranscript::new();
        let field_cfg = transcript.fs_transcript.get_random_field_cfg::<F, Zt::Fmod, Zt::PrimeTest>();
        
        // We prove evaluations over the field, so integers need to be mapped to field elements first
        let point = point.iter().map(|v| v.into_with_cfg(&field_cfg)).collect::<Vec<F>>();
        let (q0, q1) = point_to_tensor(num_rows, &point, &field_cfg)?;

        let degree_bound = Zt::Comb::DEGREE_BOUND;
        let polys_as_comb_r: Vec<Vec<Zt::CombR>> = polys.iter().map(|poly| {
            let alphas = if degree_bound.is_zero() {
                vec![Zt::Chal::ONE]
            } else {
                transcript.fs_transcript.get_challenges(degree_bound + 1)
            };

            poly.evaluations.iter().map(|eval| Zt::EvalDotChal::inner_product::<CHECK_FOR_OVERFLOW>(
                eval,
                &alphas,
                Zt::CombR::ZERO
            )).collect()
        }).collect::<Result<_, ZipError>>()?;
        
        // TODO. b_per_poly can be fused with for loop below. Compute b_i, write to transcript and compute eval_i
        let zero_f = F::zero_with_cfg(&field_cfg);
        let b_per_poly: Vec<F> = polys_as_comb_r.iter().flat_map(|poly_comb_r| {
            (0..num_rows).map(|j| {
                let row_f: Vec<F> = poly_comb_r[j*row_len..(j+1)*row_len].iter().map(|int| int.into_with_cfg(&field_cfg)).collect();
                // b_j = <row_j, q1> (inner product in field)
                MBSInnerProduct::inner_product::<UNCHECKED>(
                    &row_f, 
                    &q1, 
                    zero_f.clone()
                )
            })
        }).collect::<Result<_, _>>()?;

        let mut evals = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let b_i = &b_per_poly[i*num_rows..(i+1) * num_rows];
            transcript.write_field_elements(b_i)?;
            // eval_i = <q_0, b_i> (inner product in field), <q_2, b_i> in paper
            let eval_i = MBSInnerProduct::inner_product::<UNCHECKED>(
                &q0, 
                b_i, 
                zero_f.clone()
            )?;
            evals.push(eval_i)
        }

        // combined_row_i[col] = sum_j(s_j * int_rows_i[j][col]) for each column
        // combined_row: Vec<CombR> (length row_len), s_j in paper
        let coeffs = if pp.num_rows == 1 {
            vec![Zt::Chal::ONE]
        } else {
            transcript.fs_transcript.get_challenges::<Zt::Chal>(num_rows)
        };

        let combined_row: Vec<Zt::CombR> = polys_as_comb_r.iter()
            .try_fold(
                vec![Zt::CombR::ZERO; row_len],
                |mut acc, poly| -> Result<_, ZipError> {
                    let row = combine_rows!(
                        CHECK_FOR_OVERFLOW,
                        &coeffs,
                        poly.iter(),
                        |eval: &Zt::CombR| Ok::<_, ZipError>(eval.clone()),
                        row_len,
                        Zt::CombR::ZERO
                    );

                    acc.iter_mut().zip(row.iter()).for_each(|(a, r)| {
                        if CHECK_FOR_OVERFLOW {
                            *a = zinc_utils::add!(*a, r, "Addition overflow while summing combined rows across polys");
                        } else {
                            *a += r;
                        }
                    });

                    Ok(acc)
                }
            )?;

        transcript.write_const_many(&combined_row)?;
        for _ in 0..Zt::NUM_COLUMN_OPENINGS{
            let column_idx = transcript.squeeze_challenge_idx(pp.linear_code.codeword_len());
            Self::open_merkle_trees_for_column(commit_hint, column_idx, &mut transcript)?;
        }

        Ok((evals, transcript.into()))
    }

    //TODO check this since we optmised merkle by concatenating
    pub(super) fn open_merkle_trees_for_column(
        commit_hint: &ZipPlusHint<Zt::Cw>,
        column_idx: usize,
        transcript: &mut PcsTranscript,
    ) -> Result<(), ZipError> {
        for cw_matrix in &commit_hint.cw_matrices {
            let column_values = cw_matrix.as_rows().map(|row| &row[column_idx]);
            transcript.write_const_many_iter(column_values, cw_matrix.num_rows)?;
        }

        let merkle_proof = commit_hint
            .merkle_tree
            .prove(column_idx)
            .map_err(|_| ZipError::InvalidPcsOpen("Failed to open merkle tree".into()))?;
        transcript
            .write_merkle_proof(&merkle_proof)
            .map_err(|_| ZipError::InvalidPcsOpen("Failed to write a merkle tree proof".into()))?;

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
            structs::{ZipPlus, ZipPlusHint, ZipTypes},
            test_utils::*,
        },
        pcs_transcript::PcsTranscript,
    };
    use crypto_bigint::U64;
    use crypto_primitives::{
        IntoWithConfig, crypto_bigint_boxed_monty::BoxedMontyField, crypto_bigint_int::Int,
    };
    use num_traits::{ConstOne, Zero};
    use zinc_poly::mle::DenseMultilinearExtension;
    use zinc_transcript::traits::Transcript;
    use zinc_utils::{CHECKED, from_ref::FromRef};

    const INT_LIMBS: usize = U64::LIMBS;

    const N: usize = INT_LIMBS;
    const K: usize = INT_LIMBS * 4;
    const M: usize = INT_LIMBS * 8;
    const DEGREE_PLUS_ONE: usize = 3;

    type F = BoxedMontyField;

    type Zt = TestZipTypes<N, K, M>;
    type C = RaaSignFlippingCode<Zt, TestRaaConfig, 4>;

    type PolyZt = TestPolyZipTypes<K, M, DEGREE_PLUS_ONE>;
    type PolyC = RaaCode<PolyZt, TestRaaConfig, 4>;

    type TestZip = ZipPlus<Zt, C>;
    type TestPolyZip = ZipPlus<PolyZt, PolyC>;

    fn test_point(num_vars: usize) -> Vec<Int<INT_LIMBS>> {
        (0..num_vars).map(|i| Int::from(i as i32 + 2)).collect()
    }

    #[test]
    fn prove_succeeds_for_single_poly() {
        let num_vars = 4;
        let (pp, poly) = setup_test_params(num_vars);
        let (hint, _) = TestZip::commit_single(&pp, &poly).unwrap();
        let point = test_point(num_vars);

        let (evals, _proof) =
            TestZip::prove::<F, CHECKED>(&pp, &[poly], &point, &hint).unwrap();
        assert_eq!(evals.len(), 1);
    }

    #[test]
    fn prove_succeeds_for_poly_type() {
        let num_vars = 4;
        let (pp, poly) = setup_poly_test_params(num_vars);
        let (hint, _) = TestPolyZip::commit_single(&pp, &poly).unwrap();
        let point: Vec<i128> = (0..num_vars).map(|i| i as i128 + 2).collect();

        let (evals, _proof) =
            TestPolyZip::prove::<F, CHECKED>(&pp, &[poly], &point, &hint).unwrap();
        assert_eq!(evals.len(), 1);
    }

    #[test]
    fn prove_succeeds_with_corrupted_codeword() {
        let num_vars = 4;
        let (pp, poly) = setup_test_params(num_vars);
        let (mut hint, _) = TestZip::commit_single(&pp, &poly).unwrap();

        {
            let mut rows = hint.cw_matrices[0].to_rows_slices_mut();
            assert!(!rows.is_empty());
            rows[0][0] += Int::ONE;
        }

        let corrupted_tree = {
            let all_rows: Vec<&[_]> = hint.cw_matrices.iter()
                .flat_map(|m| m.as_rows())
                .collect();
            MerkleTree::new(&all_rows)
        };
        let corrupted_hint = ZipPlusHint::new(hint.cw_matrices, corrupted_tree);

        let point = test_point(num_vars);
        let result =
            TestZip::prove::<F, CHECKED>(&pp, &[poly], &point, &corrupted_hint);
        assert!(result.is_ok());
    }

    #[test]
    fn prove_rejects_oversized_polynomial() {
        let num_vars = 4;
        let (pp, _) = setup_test_params(num_vars);
        let oversized_poly: DenseMultilinearExtension<_> =
            (0..1 << 5).map(Int::from).collect();

        let (hint, _) =
            TestZip::commit_single(&pp, &setup_test_params::<N, K, M>(num_vars).1).unwrap();

        let point = test_point(num_vars);
        let result =
            TestZip::prove::<F, CHECKED>(&pp, &[oversized_poly], &point, &hint);
        assert!(result.is_err());
    }

    /// For TestZipTypes (degree_bound = 0), alphas = [1] so prove eval
    /// equals poly(point) lifted to F.
    #[test]
    fn prove_returns_correct_evaluation() {
        let num_vars = 4;
        let (pp, poly) = setup_test_params(num_vars);
        let (hint, _) = TestZip::commit_single(&pp, &poly).unwrap();
        let point = test_point(num_vars);

        let (evals, _) =
            TestZip::prove::<F, CHECKED>(&pp, &[poly.clone()], &point, &hint).unwrap();

        let field_cfg = {
            let mut t = PcsTranscript::new();
            t.fs_transcript
                .get_random_field_cfg::<F, <Zt as ZipTypes>::Fmod, <Zt as ZipTypes>::PrimeTest>()
        };

        let poly_wide: DenseMultilinearExtension<Int<M>> =
            poly.evaluations.iter().map(|e| Int::from_ref(e)).collect();
        let expected_int = poly_wide.evaluate(&point, Zero::zero()).unwrap();
        let expected_f: F = (&expected_int).into_with_cfg(&field_cfg);

        assert_eq!(evals.len(), 1);
        assert_eq!(evals[0], expected_f);
    }

    #[test]
    fn prove_succeeds_for_batch() {
        let num_vars = 4;
        let (pp, _) = setup_test_params(num_vars);

        let polys: Vec<DenseMultilinearExtension<_>> = vec![
            (1..=16).map(Int::from).collect(),
            (17..=32).map(Int::from).collect(),
        ];

        let (hint, _) = TestZip::commit(&pp, &polys).unwrap();
        let point = test_point(num_vars);

        let (evals, _) =
            TestZip::prove::<F, CHECKED>(&pp, &polys, &point, &hint).unwrap();
        assert_eq!(evals.len(), 2);
    }

    /// For TestZipTypes, each batched eval should equal the corresponding
    /// poly(point) in F.
    #[test]
    fn prove_returns_correct_evaluations_for_batch() {
        let num_vars = 4;
        let (pp, _) = setup_test_params(num_vars);

        let polys: Vec<DenseMultilinearExtension<_>> = vec![
            (1..=16).map(Int::from).collect(),
            (17..=32).map(Int::from).collect(),
        ];

        let (hint, _) = TestZip::commit(&pp, &polys).unwrap();
        let point = test_point(num_vars);

        let (evals, _) =
            TestZip::prove::<F, CHECKED>(&pp, &polys, &point, &hint).unwrap();

        let field_cfg = {
            let mut t = PcsTranscript::new();
            t.fs_transcript
                .get_random_field_cfg::<F, <Zt as ZipTypes>::Fmod, <Zt as ZipTypes>::PrimeTest>()
        };

        for (i, poly) in polys.iter().enumerate() {
            let poly_wide: DenseMultilinearExtension<Int<M>> =
                poly.evaluations.iter().map(|e| Int::from_ref(e)).collect();
            let expected_int = poly_wide.evaluate(&point, Zero::zero()).unwrap();
            let expected_f: F = (&expected_int).into_with_cfg(&field_cfg);
            assert_eq!(evals[i], expected_f, "Eval mismatch for poly {i}");
        }
    }
}
