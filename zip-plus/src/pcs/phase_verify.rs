use crate::{
    ZipError,
    code::LinearCode,
    merkle::MtHash,
    pcs::{
        structs::{ZipPlus, ZipPlusCommitment, ZipPlusParams, ZipTypes},
        utils::{point_to_tensor, validate_input},
    },
    pcs_transcript::PcsVerifierTranscript,
};
use crypto_primitives::{FromPrimitiveWithConfig, FromWithConfig, IntoWithConfig, PrimeField};
use itertools::Itertools;
use num_traits::{ConstOne, ConstZero, Zero};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use zinc_poly::Polynomial;
use zinc_transcript::traits::{Transcribable, Transcript};
use zinc_utils::{
    UNCHECKED, cfg_into_iter, cfg_iter,
    from_ref::FromRef,
    inner_product::{InnerProduct, MBSInnerProduct},
    mul_by_scalar::MulByScalar,
    projectable_to_field::ProjectableToField,
};

impl<Zt: ZipTypes, Lc: LinearCode<Zt>> ZipPlus<Zt, Lc> {
    pub fn verify<F, const CHECK_FOR_OVERFLOW: bool>(
        transcript: &mut PcsVerifierTranscript,
        vp: &ZipPlusParams<Zt, Lc>,
        comm: &ZipPlusCommitment,
        field_cfg: &F::Config,
        projecting_element: &Zt::Chal,
        point_f: &[F],
        eval_f: &F,
    ) -> Result<(), ZipError>
    where
        F: FromPrimitiveWithConfig
            + FromRef<F>
            + for<'a> FromWithConfig<&'a Zt::Chal>
            + for<'a> MulByScalar<&'a F>,
        F::Inner: FromRef<Zt::Fmod> + Transcribable,
        Zt::Cw: ProjectableToField<F>,
    {
        validate_input::<Zt, Lc, _>("verify", vp.num_vars, &[], &[point_f])?;

        let columns_opened =
            Self::verify_testing::<CHECK_FOR_OVERFLOW>(vp, &comm.root, transcript)?;

        let projecting_element: F = projecting_element.into_with_cfg(field_cfg);

        Self::verify_evaluation(
            vp,
            point_f,
            eval_f,
            &columns_opened,
            transcript,
            projecting_element,
            field_cfg,
        )?;

        Ok(())
    }

    #[allow(clippy::arithmetic_side_effects, clippy::type_complexity)]
    pub(super) fn verify_testing<const CHECK_FOR_OVERFLOW: bool>(
        vp: &ZipPlusParams<Zt, Lc>,
        root: &MtHash,
        transcript: &mut PcsVerifierTranscript,
    ) -> Result<Vec<(usize, Vec<Zt::Cw>)>, ZipError> {
        // Gather the coeffs and encoded combined rows per proximity test
        let encoded_combined_rows: Option<(Vec<Zt::Chal>, Vec<Zt::Chal>, Vec<Zt::CombR>)> = {
            if vp.num_rows > 1 {
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
                let coeffs = transcript.fs_transcript.get_challenges(vp.num_rows);

                let combined_row: Vec<Zt::CombR> =
                    transcript.read_const_many(vp.linear_code.row_len())?;

                let encoded_combined_row: Vec<Zt::CombR> =
                    vp.linear_code.encode_wide(&combined_row);
                Some((alphas, coeffs, encoded_combined_row))
            } else {
                None
            }
        };

        // Read the transcript sequentially
        let columns_and_proofs: Vec<_> = (0..Zt::NUM_COLUMN_OPENINGS)
            .map(|_| -> Result<_, ZipError> {
                let column_idx = transcript.squeeze_challenge_idx(vp.linear_code.codeword_len());
                let column_values = transcript.read_const_many(vp.num_rows)?;
                let proof = transcript.read_merkle_proof().map_err(|e| {
                    ZipError::InvalidPcsOpen(format!("Failed to read a Merkle proof: {e}"))
                })?;
                Ok((column_idx, column_values, proof))
            })
            .try_collect()?;

        let columns_opened: Vec<(usize, Vec<Zt::Cw>)> = cfg_into_iter!(columns_and_proofs)
            .map(
                |(column_idx, column_values, proof)| -> Result<_, ZipError> {
                    if let Some((ref alphas, ref coeffs, ref encoded_combined_row)) =
                        encoded_combined_rows
                    {
                        Self::verify_column_testing::<CHECK_FOR_OVERFLOW>(
                            alphas,
                            coeffs,
                            encoded_combined_row,
                            &column_values,
                            column_idx,
                            vp.num_rows,
                        )?;
                    }

                    proof
                        .verify(root, &column_values, column_idx)
                        .map_err(|e| {
                            ZipError::InvalidPcsOpen(format!(
                                "Column opening verification failed: {e}"
                            ))
                        })?;

                    Ok((column_idx, column_values))
                },
            )
            .collect::<Result<_, _>>()?;
        Ok(columns_opened)
    }

    pub(super) fn verify_column_testing<const CHECK_FOR_OVERFLOW: bool>(
        alphas: &[Zt::Chal],
        coeffs: &[Zt::Chal],
        encoded_combined_row: &[Zt::CombR],
        column_entries: &[Zt::Cw],
        column: usize,
        num_rows: usize,
    ) -> Result<(), ZipError> {
        let column_entries_comb: Zt::CombR = if num_rows > 1 {
            let column_entries: Vec<_> = column_entries
                .iter()
                .map(Zt::Comb::from_ref)
                .map(|p| {
                    Zt::CombDotChal::inner_product::<CHECK_FOR_OVERFLOW>(
                        &p,
                        alphas,
                        Zt::CombR::ZERO,
                    )
                })
                .try_collect()?;
            Zt::ArrCombRDotChal::inner_product::<CHECK_FOR_OVERFLOW>(
                &column_entries,
                coeffs,
                Zt::CombR::ZERO,
            )?
        } else {
            Zt::CombDotChal::inner_product::<CHECK_FOR_OVERFLOW>(
                &Zt::Comb::from_ref(&column_entries[0]),
                alphas,
                Zt::CombR::ZERO,
            )?
        };

        if column_entries_comb != encoded_combined_row[column] {
            return Err(ZipError::InvalidPcsOpen("Proximity failure".into()));
        }
        Ok(())
    }

    fn verify_evaluation<F>(
        vp: &ZipPlusParams<Zt, Lc>,
        point_f: &[F],
        eval_f: &F,
        columns_opened: &[(usize, Vec<Zt::Cw>)],
        transcript: &mut PcsVerifierTranscript,
        projecting_element: F,
        field_cfg: &F::Config,
    ) -> Result<(), ZipError>
    where
        F: FromPrimitiveWithConfig + FromRef<F> + for<'a> MulByScalar<&'a F>,
        F::Inner: FromRef<Zt::Fmod> + Transcribable,
        Zt::Cw: ProjectableToField<F>,
    {
        let q_0_combined_row = transcript.read_field_elements(vp.linear_code.row_len())?;
        let encoded_combined_row = vp.linear_code.encode_f(&q_0_combined_row);

        let (q_0, q_1) = point_to_tensor(vp.num_rows, point_f, field_cfg)?;

        // It is safe to use inner_product_unchecked because we're in a field.
        if MBSInnerProduct::inner_product::<UNCHECKED>(
            &q_0_combined_row,
            &q_1,
            F::zero_with_cfg(field_cfg),
        )? != *eval_f
        {
            return Err(ZipError::InvalidPcsOpen(
                "Evaluation consistency failure".into(),
            ));
        }
        let project = Zt::Cw::prepare_projection(&projecting_element);
        cfg_iter!(columns_opened).try_for_each(|(column_idx, column_values)| {
            Self::verify_proximity_q_0(
                &q_0,
                &encoded_combined_row,
                column_values,
                *column_idx,
                vp.num_rows,
                &project,
                field_cfg,
            )
        })?;

        Ok(())
    }

    fn verify_proximity_q_0<F>(
        q_0: &[F],
        encoded_q_0_combined_row: &[F],
        column_entries: &[Zt::Cw],
        column: usize,
        num_rows: usize,
        project: &impl Fn(&<Zt as ZipTypes>::Cw) -> F,
        field_cfg: &F::Config,
    ) -> Result<(), ZipError>
    where
        F: PrimeField + for<'a> MulByScalar<&'a F> + FromRef<F>,
    {
        let column_entries_comb = if num_rows > 1 {
            let column_entries = column_entries.iter().map(project).collect_vec();
            // It is safe to use inner_product_unchecked because we're in a field.
            MBSInnerProduct::inner_product::<UNCHECKED>(
                q_0,
                &column_entries,
                F::zero_with_cfg(field_cfg),
            )?
        } else {
            project(column_entries.first().expect("No column entries"))
        };
        if column_entries_comb != encoded_q_0_combined_row[column] {
            return Err(ZipError::InvalidPcsOpen("Proximity failure".into()));
        }

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
        ZipError,
        code::{LinearCode, raa::RaaCode, raa_sign_flip::RaaSignFlippingCode},
        merkle::MerkleTree,
        pcs::{
            structs::{ZipPlus, ZipPlusHint, ZipTypes},
            test_utils::*,
        },
        pcs_transcript::{PcsProverTranscript, PcsVerifierTranscript},
    };
    use crypto_bigint::U64;
    use crypto_primitives::{
        Field, FromWithConfig, IntoWithConfig, PrimeField,
        crypto_bigint_boxed_monty::BoxedMontyField, crypto_bigint_int::Int,
    };
    use itertools::Itertools;
    use num_traits::{ConstOne, ConstZero, Zero};
    use rand::{Rng, prelude::ThreadRng};
    use std::mem::size_of;
    use zinc_poly::{
        mle::{DenseMultilinearExtension, MultilinearExtensionRand},
        univariate::binary::BinaryPoly,
    };
    use zinc_transcript::traits::{Transcribable, Transcript};
    use zinc_utils::CHECKED;

    const INT_LIMBS: usize = U64::LIMBS;

    const N: usize = INT_LIMBS;
    const K: usize = INT_LIMBS * 4;
    const M: usize = INT_LIMBS * 8;
    const DEGREE_PLUS_ONE: usize = 3;

    type F = BoxedMontyField;

    type Zt = TestZipTypes<N, K, M>;
    type C = RaaSignFlippingCode<Zt, TestRaaConfig, 4>;

    type PolyZt = TestBinPolyZipTypes<K, M, DEGREE_PLUS_ONE>;
    type PolyC = RaaCode<PolyZt, TestRaaConfig, 4>;

    type TestZip = ZipPlus<Zt, C>;
    type TestPolyZip = ZipPlus<PolyZt, PolyC>;

    #[test]
    fn successful_verification_of_valid_proof() {
        let num_vars = 4;
        {
            let (pp, comm, point_f, eval_f, mut transcript) =
                setup_full_protocol::<F, N, K, M>(num_vars);
            let (field_cfg, projecting_element) =
                get_field_and_projecting_element::<Zt, F>(&mut transcript.fs_transcript);

            let result = TestZip::verify::<_, CHECKED>(
                &mut transcript,
                &pp,
                &comm,
                &field_cfg,
                &projecting_element,
                &point_f,
                &eval_f,
            );
            assert!(result.is_ok(), "Verification failed: {result:?}")
        };
        {
            let (pp, comm, point_f, eval_f, mut transcript) =
                setup_full_protocol_poly::<F, N, K, M, DEGREE_PLUS_ONE>(num_vars);
            let (field_cfg, projecting_element) =
                get_field_and_projecting_element::<PolyZt, F>(&mut transcript.fs_transcript);

            let result = TestPolyZip::verify::<_, CHECKED>(
                &mut transcript,
                &pp,
                &comm,
                &field_cfg,
                &projecting_element,
                &point_f,
                &eval_f,
            );

            assert!(result.is_ok(), "Verification failed: {result:?}");
        }
    }

    #[test]
    fn verification_fails_with_incorrect_evaluation() {
        let num_vars = 4;

        {
            let (pp, comm, point_f, eval_f, mut transcript) =
                setup_full_protocol::<F, N, K, M>(num_vars);
            let (field_cfg, projecting_element) =
                get_field_and_projecting_element::<Zt, F>(&mut transcript.fs_transcript);

            let result = TestZip::verify::<_, CHECKED>(
                &mut transcript,
                &pp,
                &comm,
                &field_cfg,
                &projecting_element,
                &point_f,
                &(eval_f + F::one_with_cfg(&field_cfg)),
            );

            assert!(result.is_err());
        }

        {
            let (pp, comm, point_f, eval_f, mut transcript) =
                setup_full_protocol_poly::<F, N, K, M, DEGREE_PLUS_ONE>(num_vars);
            let (field_cfg, projecting_element) =
                get_field_and_projecting_element::<PolyZt, F>(&mut transcript.fs_transcript);

            let result = TestPolyZip::verify::<_, CHECKED>(
                &mut transcript,
                &pp,
                &comm,
                &field_cfg,
                &projecting_element,
                &point_f,
                &(eval_f + F::one_with_cfg(&field_cfg)),
            );

            assert!(result.is_err());
        }
    }

    #[test]
    fn verification_fails_with_tampered_proof() {
        fn tamper(mut proof: PcsVerifierTranscript) -> PcsVerifierTranscript {
            proof.stream.get_mut()[0] ^= 0x01;
            proof
        }
        let num_vars = 4;

        {
            let (pp, comm, point_f, eval_f, proof) = setup_full_protocol::<F, N, K, M>(num_vars);
            let mut tampered = tamper(proof);
            let (field_cfg, projecting_element) =
                get_field_and_projecting_element::<Zt, F>(&mut tampered.fs_transcript);
            let result = TestZip::verify::<_, CHECKED>(
                &mut tampered,
                &pp,
                &comm,
                &field_cfg,
                &projecting_element,
                &point_f,
                &eval_f,
            );
            assert!(result.is_err());
        }

        {
            let (pp, comm, point_f, eval_f, proof) =
                setup_full_protocol_poly::<F, N, K, M, DEGREE_PLUS_ONE>(num_vars);
            let mut tampered = tamper(proof);
            let (field_cfg, projecting_element) =
                get_field_and_projecting_element::<PolyZt, F>(&mut tampered.fs_transcript);
            let result = TestPolyZip::verify::<_, CHECKED>(
                &mut tampered,
                &pp,
                &comm,
                &field_cfg,
                &projecting_element,
                &point_f,
                &eval_f,
            );
            assert!(result.is_err());
        }
    }

    #[test]
    fn verification_fails_with_wrong_commitment() {
        let num_vars = 4;
        {
            let (pp, _comm_poly1, point_f, eval_f, mut transcript) =
                setup_full_protocol::<F, N, K, M>(num_vars);
            let (field_cfg, projecting_element) =
                get_field_and_projecting_element::<Zt, F>(&mut transcript.fs_transcript);

            let poly2: DenseMultilinearExtension<_> =
                (20..(20 + (1 << num_vars))).map(Int::from).collect();

            let (_, comm_poly2) = TestZip::commit(&pp, &poly2).unwrap();

            let result = TestZip::verify::<_, CHECKED>(
                &mut transcript,
                &pp,
                &comm_poly2,
                &field_cfg,
                &projecting_element,
                &point_f,
                &eval_f,
            );

            assert!(result.is_err());
        }

        {
            let (pp, _comm_poly1, point_f, eval_f, mut transcript) =
                setup_full_protocol_poly::<F, N, K, M, DEGREE_PLUS_ONE>(num_vars);
            let (field_cfg, projecting_element) =
                get_field_and_projecting_element::<PolyZt, F>(&mut transcript.fs_transcript);

            let different_evals = {
                let different_eval_coeffs: Vec<_> = (1..=((1 << num_vars)
                    * (DEGREE_PLUS_ONE - 1) as i8))
                    .map(|x| (x % 3 == 0).into())
                    .collect_vec();
                different_eval_coeffs
                    .chunks_exact(DEGREE_PLUS_ONE - 1)
                    .map(BinaryPoly::new)
                    .collect_vec()
            };

            let poly2 = DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                different_evals,
                Zero::zero(),
            );
            let (_, comm_poly2) = TestPolyZip::commit(&pp, &poly2).unwrap();

            let result = TestPolyZip::verify::<_, CHECKED>(
                &mut transcript,
                &pp,
                &comm_poly2,
                &field_cfg,
                &projecting_element,
                &point_f,
                &eval_f,
            );

            assert!(result.is_err());
        }
    }

    #[test]
    fn verification_fails_with_invalid_point_size() {
        let num_vars = 4;

        let make_invalid_point = |cfg: &<F as PrimeField>::Config| {
            let mut invalid_point = vec![];
            for i in 0..=num_vars {
                invalid_point.push(F::from_with_cfg(100 + i as i32, cfg));
            }
            invalid_point
        };

        {
            let (pp, comm, _point_f, eval_f, mut transcript) =
                setup_full_protocol_poly::<F, N, K, M, DEGREE_PLUS_ONE>(num_vars);
            let (field_cfg, projecting_element) =
                get_field_and_projecting_element::<PolyZt, F>(&mut transcript.fs_transcript);
            let invalid_point = make_invalid_point(eval_f.cfg());

            let result = TestPolyZip::verify::<_, CHECKED>(
                &mut transcript,
                &pp,
                &comm,
                &field_cfg,
                &projecting_element,
                &invalid_point,
                &eval_f,
            );

            assert!(matches!(result, Err(..)));
        }

        {
            let (pp, comm, _point_f, eval_f, mut transcript) =
                setup_full_protocol::<F, N, K, M>(num_vars);
            let (field_cfg, projecting_element) =
                get_field_and_projecting_element::<Zt, F>(&mut transcript.fs_transcript);
            let invalid_point = make_invalid_point(eval_f.cfg());

            let result = TestZip::verify::<_, CHECKED>(
                &mut transcript,
                &pp,
                &comm,
                &field_cfg,
                &projecting_element,
                &invalid_point,
                &eval_f,
            );

            assert!(matches!(result, Err(..)));
        }
    }

    #[test]
    fn verification_fails_due_to_incorrect_polynomial() {
        let num_vars = 4;
        let (pp, mle1) = setup_test_params(num_vars);

        let (data, comm) = TestZip::commit(&pp, &mle1).unwrap();

        let mle2: DenseMultilinearExtension<_> = (20..=35).map(Int::from).collect();

        let point: Vec<<Zt as ZipTypes>::Pt> =
            (0..num_vars).map(|i| Int::from(i as i32 + 2)).collect();

        let mut prover_transcript = PcsProverTranscript::new_from_commitment(&comm).unwrap();
        let (field_cfg, projecting_element) =
            get_field_and_projecting_element::<Zt, F>(&mut prover_transcript.fs_transcript);

        TestZip::test::<CHECKED>(&mut prover_transcript, &pp, &mle2, &data)
            .expect("test phase should succeed");
        let _eval_f = TestZip::evaluate::<F, CHECKED>(
            &mut prover_transcript,
            &pp,
            &mle2,
            &point,
            &field_cfg,
            &projecting_element,
        )
        .expect("evaluation phase should succeed");

        let eval_mle1 = mle1
            .evaluate(&point, Zero::zero())
            .expect("Failed to evaluate polynomial");

        let point_f: Vec<F> = point.iter().map(|v| v.into_with_cfg(&field_cfg)).collect();
        let eval_mle1_f = eval_mle1.into_with_cfg(&field_cfg);

        let mut verifier_transcript = prover_transcript.into_verification_transcript();
        let (field_cfg, projecting_element) =
            get_field_and_projecting_element::<Zt, F>(&mut verifier_transcript.fs_transcript);

        let verification_result = TestZip::verify::<_, CHECKED>(
            &mut verifier_transcript,
            &pp,
            &comm,
            &field_cfg,
            &projecting_element,
            &point_f,
            &eval_mle1_f,
        );

        assert!(verification_result.is_err());
    }

    #[test]
    fn verification_fails_due_to_a_hint_that_is_not_close() {
        let num_vars = 4;
        let (pp, mle) = setup_test_params(num_vars);

        let (original_data, comm) = TestZip::commit(&pp, &mle).unwrap();

        let mut corrupted_data = original_data.cw_matrix.clone();
        {
            let mut corrupted_rows = corrupted_data.to_rows_slices_mut();
            let codeword_len = pp.linear_code.codeword_len();
            // Proximity distance is half the codeword length for the default spec.
            // We corrupt more than half of the first row to ensure it's not close.
            let corruption_count = codeword_len / 2 + 1;
            for i in corrupted_rows[0].iter_mut().take(corruption_count) {
                *i += Int::ONE;
            }
        }

        let corrupted_merkle_tree = MerkleTree::new(&corrupted_data.to_rows_slices());
        let corrupted_data = ZipPlusHint::new(corrupted_data, corrupted_merkle_tree);

        let point: Vec<<Zt as ZipTypes>::Pt> =
            (0..num_vars).map(|i| Int::from(i as i32 + 2)).collect();

        let mut prover_transcript = PcsProverTranscript::new_from_commitment(&comm).unwrap();
        let (field_cfg, projecting_element) =
            get_field_and_projecting_element::<Zt, F>(&mut prover_transcript.fs_transcript);

        TestZip::test::<CHECKED>(&mut prover_transcript, &pp, &mle, &corrupted_data)
            .expect("test phase should succeed");
        let eval_f = TestZip::evaluate::<F, CHECKED>(
            &mut prover_transcript,
            &pp,
            &mle,
            &point,
            &field_cfg,
            &projecting_element,
        )
        .expect("evaluation phase should succeed");

        let point_f: Vec<F> = point.iter().map(|v| v.into_with_cfg(&field_cfg)).collect();

        let mut verifier_transcript = prover_transcript.into_verification_transcript();
        verifier_transcript.fs_transcript.absorb_slice(&comm.root);
        let (field_cfg, projecting_element) =
            get_field_and_projecting_element::<Zt, F>(&mut verifier_transcript.fs_transcript);

        let verification_result = TestZip::verify::<_, CHECKED>(
            &mut verifier_transcript,
            &pp,
            &comm,
            &field_cfg,
            &projecting_element,
            &point_f,
            &eval_f,
        );

        assert!(verification_result.is_err());
    }

    #[test]
    fn verification_fails_due_to_incorrect_evaluation() {
        let num_vars = 4;
        let (pp, mle) = setup_test_params(num_vars);

        let (data, comm) = TestZip::commit(&pp, &mle).unwrap();

        let point: Vec<<Zt as ZipTypes>::Pt> =
            (0..num_vars).map(|i| Int::from(i as i32 + 2)).collect();

        let mut prover_transcript = PcsProverTranscript::new_from_commitment(&comm).unwrap();
        let (field_cfg, projecting_element) =
            get_field_and_projecting_element::<Zt, F>(&mut prover_transcript.fs_transcript);

        TestZip::test::<CHECKED>(&mut prover_transcript, &pp, &mle, &data)
            .expect("test phase should succeed");
        let correct_eval_f = TestZip::evaluate::<F, CHECKED>(
            &mut prover_transcript,
            &pp,
            &mle,
            &point,
            &field_cfg,
            &projecting_element,
        )
        .expect("evaluation phase should succeed");

        let incorrect_eval_f = correct_eval_f + F::one_with_cfg(&field_cfg);
        let point_f: Vec<F> = point.iter().map(|v| v.into_with_cfg(&field_cfg)).collect();

        let mut verifier_transcript = prover_transcript.into_verification_transcript();
        verifier_transcript.fs_transcript.absorb_slice(&comm.root);
        let (field_cfg, projecting_element) =
            get_field_and_projecting_element::<Zt, F>(&mut verifier_transcript.fs_transcript);

        let verification_result = TestZip::verify::<_, CHECKED>(
            &mut verifier_transcript,
            &pp,
            &comm,
            &field_cfg,
            &projecting_element,
            &point_f,
            &incorrect_eval_f, // Use the wrong evaluation here
        );

        assert!(verification_result.is_err());
    }

    #[test]
    fn verification_fails_if_proximity_check_is_invalid() {
        let poly_size = 8; // row_len=4, num_rows=2 -> proximity checks are active

        let linear_code = C::new(poly_size);
        let pp = TestZip::setup(poly_size, linear_code);

        let mle: DenseMultilinearExtension<_> = (0..poly_size as i32)
            .map(<Zt as ZipTypes>::Eval::from)
            .collect();

        let (data, comm) = TestZip::commit(&pp, &mle).expect("commit should succeed");

        let point = [0i64, 0i64, 0i64]
            .into_iter()
            .map(Int::<1>::from)
            .collect::<Vec<_>>();
        let eval = mle.evaluate(&point, Zero::zero()).unwrap();

        let mut prover_transcript = PcsProverTranscript::new_from_commitment(&comm).unwrap();
        let (field_cfg, projecting_element) =
            get_field_and_projecting_element::<Zt, F>(&mut prover_transcript.fs_transcript);

        TestZip::test::<CHECKED>(&mut prover_transcript, &pp, &mle, &data)
            .expect("test phase should succeed");
        let eval_f = TestZip::evaluate::<F, CHECKED>(
            &mut prover_transcript,
            &pp,
            &mle,
            &point,
            &field_cfg,
            &projecting_element,
        )
        .expect("evaluation phase should succeed");

        assert_eq!(
            eval_f,
            eval.into_with_cfg(&field_cfg),
            "Evaluation mismatch after opening"
        );

        let point_f: Vec<F> = point.iter().map(|v| v.into_with_cfg(&field_cfg)).collect();

        let row_len = pp.linear_code.row_len();
        let bytes_per_int = M * size_of::<crypto_bigint::Word>();
        let first_combined_row_bytes = row_len * bytes_per_int;

        let mut verifier_transcript = prover_transcript.into_verification_transcript();
        assert!(
            first_combined_row_bytes <= verifier_transcript.stream.get_ref().len(),
            "proof too small to tamper"
        );

        let flip_at = bytes_per_int * (row_len / 2);
        verifier_transcript.stream.get_mut()[flip_at] ^= 0x01;

        verifier_transcript.fs_transcript.absorb_slice(&comm.root);
        let (field_cfg, projecting_element) =
            get_field_and_projecting_element::<Zt, F>(&mut verifier_transcript.fs_transcript);

        let res = TestZip::verify::<_, CHECKED>(
            &mut verifier_transcript,
            &pp,
            &comm,
            &field_cfg,
            &projecting_element,
            &point_f,
            &eval_f,
        );

        match res {
            Err(ZipError::InvalidPcsOpen(msg)) => {
                assert_eq!(msg, "Proximity failure");
            }
            Ok(()) => panic!("verification unexpectedly succeeded"),
            Err(e) => panic!("unexpected error: {e:?}"),
        }
    }

    #[test]
    fn verification_fails_if_proximity_check_is_invalid_2() {
        fn evaluate_in_field<R>(
            evaluations: &[R],
            point: &[F],
            cfg: &<F as PrimeField>::Config,
        ) -> F
        where
            F: for<'a> FromWithConfig<&'a R>,
        {
            let num_vars = point.len();
            assert_eq!(evaluations.len(), 1 << num_vars);
            let mut current_evals: Vec<F> = evaluations
                .iter()
                .map(|v| v.into_with_cfg(cfg))
                .collect_vec();
            for p in point.iter().take(num_vars) {
                let one_minus_p_i = F::one_with_cfg(cfg) - p;
                let mut next_evals = Vec::with_capacity(current_evals.len() / 2);
                for j in (0..current_evals.len()).step_by(2) {
                    let val = current_evals[j].clone() * &one_minus_p_i + &current_evals[j + 1] * p;
                    next_evals.push(val);
                }
                current_evals = next_evals;
            }
            current_evals[0].clone()
        }

        let mut rng = ThreadRng::default();

        let n = 3;
        let poly_size = 1 << n;
        let linear_code: C = C::new(poly_size);
        let pp = TestZip::setup(poly_size, linear_code);
        let mle: DenseMultilinearExtension<_> = (0..poly_size)
            .map(|_| <Zt as ZipTypes>::Eval::from(rng.random::<i8>()))
            .collect();
        let point: Vec<_> = (0..n)
            .map(|_| rng.random::<<Zt as ZipTypes>::Pt>())
            .collect();

        let (mut data, comm) = TestZip::commit(&pp, &mle).unwrap();
        data.cw_matrix.to_rows_slices_mut()[0][0] += Int::ONE;

        let mut prover_transcript = PcsProverTranscript::new_from_commitment(&comm).unwrap();
        let (field_cfg, projecting_element) =
            get_field_and_projecting_element::<Zt, F>(&mut prover_transcript.fs_transcript);

        TestZip::test::<CHECKED>(&mut prover_transcript, &pp, &mle, &data).unwrap();
        let _eval_f = TestZip::evaluate::<F, CHECKED>(
            &mut prover_transcript,
            &pp,
            &mle,
            &point,
            &field_cfg,
            &projecting_element,
        )
        .unwrap();

        let point_f = point
            .iter()
            .map(|v| v.into_with_cfg(&field_cfg))
            .collect_vec();
        let eval_f = evaluate_in_field(&mle, &point_f, &field_cfg);

        let mut verifier_transcript = prover_transcript.into_verification_transcript();
        verifier_transcript.fs_transcript.absorb_slice(&comm.root);
        let (field_cfg, projecting_element) =
            get_field_and_projecting_element::<Zt, F>(&mut verifier_transcript.fs_transcript);

        let verification_result = TestZip::verify::<_, CHECKED>(
            &mut verifier_transcript,
            &pp,
            &comm,
            &field_cfg,
            &projecting_element,
            &point_f,
            &eval_f,
        );

        assert!(verification_result.is_err());
    }

    #[test]
    fn verification_fails_if_evaluation_consistency_check_is_invalid() {
        let poly_size = 8;
        let linear_code = C::new(poly_size);
        let pp = TestZip::setup(poly_size, linear_code);

        let mle: DenseMultilinearExtension<_> =
            (0..poly_size as i32).map(Int::<INT_LIMBS>::from).collect();

        let (data, comm) = TestZip::commit(&pp, &mle).expect("commit should succeed");

        let point: Vec<<Zt as ZipTypes>::Pt> =
            [0i64, 0i64, 0i64].into_iter().map(Int::from).collect_vec();

        let mut prover_transcript = PcsProverTranscript::new_from_commitment(&comm).unwrap();
        let (field_cfg, projecting_element) =
            get_field_and_projecting_element::<Zt, F>(&mut prover_transcript.fs_transcript);

        TestZip::test::<CHECKED>(&mut prover_transcript, &pp, &mle, &data).unwrap();
        let eval_f = TestZip::evaluate::<F, CHECKED>(
            &mut prover_transcript,
            &pp,
            &mle,
            &point,
            &field_cfg,
            &projecting_element,
        )
        .unwrap();

        let point_f: Vec<F> = point.iter().map(|v| v.into_with_cfg(&field_cfg)).collect();

        let row_len = pp.linear_code.row_len();
        // Two elements: one for value and one for module
        let bytes_per_field = eval_f.inner().get_num_bytes() * 2;
        let q0_bytes = row_len * bytes_per_field;

        let mut verifier_transcript = prover_transcript.into_verification_transcript();
        verifier_transcript.fs_transcript.absorb_slice(&comm.root);
        let _ = get_field_and_projecting_element::<Zt, F>(&mut verifier_transcript.fs_transcript);

        assert!(
            verifier_transcript.stream.get_ref().len() >= q0_bytes,
            "proof too small to contain q_0_combined_row"
        );

        let tail_start = verifier_transcript.stream.get_ref().len() - q0_bytes;
        // Field element is serialized as: modulus (get_num_bytes) + value
        // (get_num_bytes) We want to tamper with the value portion, which
        // starts at bytes_per_field / 2
        let value_offset = bytes_per_field / 2;
        let flip_at = tail_start + value_offset;
        verifier_transcript.stream.get_mut()[flip_at] ^= 0x01;

        let res = TestZip::verify::<_, CHECKED>(
            &mut verifier_transcript,
            &pp,
            &comm,
            &field_cfg,
            &projecting_element,
            &point_f,
            &eval_f,
        );

        match res {
            Err(ZipError::InvalidPcsOpen(msg)) => {
                assert_eq!(msg, "Evaluation consistency failure");
            }
            Ok(()) => panic!("verification unexpectedly succeeded"),
            Err(e) => panic!("unexpected error: {e:?}"),
        }
    }

    #[test]
    fn verification_succeeds_for_zero_polynomial() {
        let poly_size = 8;
        let linear_code = C::new(poly_size);
        let pp = TestZip::setup(poly_size, linear_code);

        let mle: DenseMultilinearExtension<_> = (0..poly_size).map(|_| Int::ZERO).collect();

        let (data, comm) = TestZip::commit(&pp, &mle).expect("commit should succeed");

        let point: Vec<<Zt as ZipTypes>::Pt> =
            [0i64, 0i64, 0i64].into_iter().map(Int::from).collect_vec();

        let mut prover_transcript = PcsProverTranscript::new_from_commitment(&comm).unwrap();
        let (field_cfg, projecting_element) =
            get_field_and_projecting_element::<Zt, F>(&mut prover_transcript.fs_transcript);

        TestZip::test::<CHECKED>(&mut prover_transcript, &pp, &mle, &data).unwrap();
        let eval_f = TestZip::evaluate::<F, CHECKED>(
            &mut prover_transcript,
            &pp,
            &mle,
            &point,
            &field_cfg,
            &projecting_element,
        )
        .unwrap();

        let point_f: Vec<F> = point.iter().map(|v| v.into_with_cfg(&field_cfg)).collect();

        let mut verifier_transcript = prover_transcript.into_verification_transcript();
        verifier_transcript.fs_transcript.absorb_slice(&comm.root);
        let (field_cfg, projecting_element) =
            get_field_and_projecting_element::<Zt, F>(&mut verifier_transcript.fs_transcript);

        let res = TestZip::verify::<_, CHECKED>(
            &mut verifier_transcript,
            &pp,
            &comm,
            &field_cfg,
            &projecting_element,
            &point_f,
            &eval_f,
        );
        assert!(res.is_ok());
    }

    #[test]
    fn verification_succeeds_at_zero_point() {
        let num_vars = 3;
        let poly_size = 1 << num_vars;
        let linear_code = C::new(poly_size);
        let pp = TestZip::setup(poly_size, linear_code);

        let mle: DenseMultilinearExtension<_> =
            (1..=poly_size as i32).map(Int::<INT_LIMBS>::from).collect();

        let (data, comm) = TestZip::commit(&pp, &mle).expect("commit should succeed");

        let point: Vec<<Zt as ZipTypes>::Pt> = vec![Int::ZERO; num_vars];

        let mut prover_transcript = PcsProverTranscript::new_from_commitment(&comm).unwrap();
        let (field_cfg, projecting_element) =
            get_field_and_projecting_element::<Zt, F>(&mut prover_transcript.fs_transcript);

        TestZip::test::<CHECKED>(&mut prover_transcript, &pp, &mle, &data).unwrap();
        let eval_f = TestZip::evaluate::<F, CHECKED>(
            &mut prover_transcript,
            &pp,
            &mle,
            &point,
            &field_cfg,
            &projecting_element,
        )
        .unwrap();

        let point_f: Vec<F> = point.iter().map(|v| v.into_with_cfg(&field_cfg)).collect();

        let mut verifier_transcript = prover_transcript.into_verification_transcript();
        verifier_transcript.fs_transcript.absorb_slice(&comm.root);
        let (field_cfg, projecting_element) =
            get_field_and_projecting_element::<Zt, F>(&mut verifier_transcript.fs_transcript);

        let res = TestZip::verify::<_, CHECKED>(
            &mut verifier_transcript,
            &pp,
            &comm,
            &field_cfg,
            &projecting_element,
            &point_f,
            &eval_f,
        );
        assert!(res.is_ok());
    }

    #[test]
    fn verification_succeeds_when_polynomial_coefficients_are_max_bit_size() {
        let num_vars = 4;
        let (pp, _) = setup_test_params(num_vars);

        let mut evals: Vec<<Zt as ZipTypes>::Eval> =
            (0..1 << num_vars as i32).map(Int::from).collect();
        evals[1] = Int::from(i64::MAX);
        let poly = DenseMultilinearExtension::from_evaluations_vec(num_vars, evals, Zero::zero());

        let (data, comm) = TestZip::commit(&pp, &poly).unwrap();

        // A point of [1, 0, 0, 0] will evaluate to poly.evaluations[1].
        let mut point = vec![<Zt as ZipTypes>::Pt::ZERO; num_vars];
        point[0] = <Zt as ZipTypes>::Pt::ONE;

        let mut prover_transcript = PcsProverTranscript::new_from_commitment(&comm).unwrap();
        let (field_cfg, projecting_element) =
            get_field_and_projecting_element::<Zt, F>(&mut prover_transcript.fs_transcript);

        TestZip::test::<CHECKED>(&mut prover_transcript, &pp, &poly, &data).unwrap();
        let eval_f = TestZip::evaluate::<F, CHECKED>(
            &mut prover_transcript,
            &pp,
            &poly,
            &point,
            &field_cfg,
            &projecting_element,
        )
        .unwrap();

        let point_f: Vec<F> = point.iter().map(|v| v.into_with_cfg(&field_cfg)).collect();

        let mut verifier_transcript = prover_transcript.into_verification_transcript();
        verifier_transcript.fs_transcript.absorb_slice(&comm.root);
        let (field_cfg, projecting_element) =
            get_field_and_projecting_element::<Zt, F>(&mut verifier_transcript.fs_transcript);

        let verification_result = TestZip::verify::<_, CHECKED>(
            &mut verifier_transcript,
            &pp,
            &comm,
            &field_cfg,
            &projecting_element,
            &point_f,
            &eval_f,
        );

        assert!(
            verification_result.is_ok(),
            "Verification failed: {verification_result:?}",
        );
    }

    #[test]
    fn verification_succeeds_with_minimal_polynomial_size_mu_is_2() {
        let num_vars = 2;
        let (pp, poly) = setup_test_params(num_vars);

        let (hint, comm) = TestZip::commit(&pp, &poly).unwrap();

        let point: Vec<<Zt as ZipTypes>::Pt> = vec![Int::from(1), Int::from(2)];

        let mut prover_transcript = PcsProverTranscript::new_from_commitment(&comm).unwrap();
        let (field_cfg, projecting_element) =
            get_field_and_projecting_element::<Zt, F>(&mut prover_transcript.fs_transcript);

        TestZip::test::<CHECKED>(&mut prover_transcript, &pp, &poly, &hint).unwrap();
        let eval_f = TestZip::evaluate::<F, CHECKED>(
            &mut prover_transcript,
            &pp,
            &poly,
            &point,
            &field_cfg,
            &projecting_element,
        )
        .unwrap();

        let point_f: Vec<F> = point.iter().map(|v| v.into_with_cfg(&field_cfg)).collect();

        let mut verifier_transcript = prover_transcript.into_verification_transcript();
        verifier_transcript.fs_transcript.absorb_slice(&comm.root);
        let (field_cfg, projecting_element) =
            get_field_and_projecting_element::<Zt, F>(&mut verifier_transcript.fs_transcript);

        let verification_result = TestZip::verify::<_, CHECKED>(
            &mut verifier_transcript,
            &pp,
            &comm,
            &field_cfg,
            &projecting_element,
            &point_f,
            &eval_f,
        );

        assert!(verification_result.is_ok());
    }

    #[test]
    fn verification_fails_if_proximity_values_are_too_large() {
        let poly_size = 8;
        let linear_code = C::new(poly_size);
        let pp = TestZip::setup(poly_size, linear_code);

        let mle: DenseMultilinearExtension<_> =
            (1..=poly_size as i32).map(Int::<INT_LIMBS>::from).collect();

        let (data, comm) = TestZip::commit(&pp, &mle).expect("commit should succeed");

        let point: Vec<<Zt as ZipTypes>::Pt> =
            [0i64, 0i64, 0i64].into_iter().map(Int::from).collect_vec();

        let mut prover_transcript = PcsProverTranscript::new_from_commitment(&comm).unwrap();
        let (field_cfg, projecting_element) =
            get_field_and_projecting_element::<Zt, F>(&mut prover_transcript.fs_transcript);

        TestZip::test::<CHECKED>(&mut prover_transcript, &pp, &mle, &data).unwrap();
        let eval_f = TestZip::evaluate::<F, CHECKED>(
            &mut prover_transcript,
            &pp,
            &mle,
            &point,
            &field_cfg,
            &projecting_element,
        )
        .unwrap();

        let point_f: Vec<F> = point.iter().map(|v| v.into_with_cfg(&field_cfg)).collect();

        let row_len = pp.linear_code.row_len();
        let bytes_per_int = M * 8;
        let first_section_bytes = row_len * bytes_per_int;

        let mut verifier_transcript = prover_transcript.into_verification_transcript();
        assert!(
            first_section_bytes <= verifier_transcript.stream.get_ref().len(),
            "proof too small to tamper u'"
        );

        for b in &mut verifier_transcript.stream.get_mut()[0..bytes_per_int] {
            *b = 0xFF;
        }

        verifier_transcript.fs_transcript.absorb_slice(&comm.root);
        let (field_cfg, projecting_element) =
            get_field_and_projecting_element::<Zt, F>(&mut verifier_transcript.fs_transcript);

        let res = TestZip::verify::<_, CHECKED>(
            &mut verifier_transcript,
            &pp,
            &comm,
            &field_cfg,
            &projecting_element,
            &point_f,
            &eval_f,
        );
        assert!(res.is_err());
    }

    /// Mirrors: `Zip/Verify: RandomField<4>, poly_size = 2^12 (Int limbs = 1)`
    #[test]
    fn bench_p12_verify() {
        fn inner<const P: usize>() {
            let mut rng = ThreadRng::default();
            // Match the benchmark's transcript usage for linear code construction
            let poly_size = 1 << P;
            let linear_code = C::new(poly_size);
            let pp = TestZip::setup(poly_size, linear_code);

            let mle = DenseMultilinearExtension::rand(P, &mut rng);
            let (data, commitment) = TestZip::commit(&pp, &mle).expect("commit");

            // Same point choice as the bench
            let point = vec![1i64; P].iter().map(|v| v.into()).collect_vec();

            let mut prover_transcript =
                PcsProverTranscript::new_from_commitment(&commitment).unwrap();
            let (field_cfg, projecting_element) =
                get_field_and_projecting_element::<Zt, F>(&mut prover_transcript.fs_transcript);

            // Prover produces a proof once (exactly as in the bench)
            TestZip::test::<CHECKED>(&mut prover_transcript, &pp, &mle, &data).unwrap();
            let eval_f = TestZip::evaluate::<F, CHECKED>(
                &mut prover_transcript,
                &pp,
                &mle,
                &point,
                &field_cfg,
                &projecting_element,
            )
            .unwrap();

            let point_f: Vec<F> = point.iter().map(|v| v.into_with_cfg(&field_cfg)).collect();

            let mut verifier_transcript = prover_transcript.into_verification_transcript();
            verifier_transcript
                .fs_transcript
                .absorb_slice(&commitment.root);
            let (field_cfg, projecting_element) =
                get_field_and_projecting_element::<Zt, F>(&mut verifier_transcript.fs_transcript);

            // Verifier replays verification from the same proof (also like the bench)
            TestZip::verify::<_, CHECKED>(
                &mut verifier_transcript,
                &pp,
                &commitment,
                &field_cfg,
                &projecting_element,
                &point_f,
                &eval_f,
            )
            .expect("verify");
        }

        inner::<12>();
    }

    /// Mirrors: `Zip+/Verify` for `poly_size=2^12`
    #[test]
    fn bench_p12_verify_poly() {
        fn inner<const P: usize>() {
            let mut rng = ThreadRng::default();
            // Match the benchmark's transcript usage for linear code construction
            let poly_size = 1 << P;
            let linear_code = PolyC::new(poly_size);
            let pp = TestPolyZip::setup(poly_size, linear_code);

            let mle = DenseMultilinearExtension::rand(P, &mut rng);
            let (data, commitment) = TestPolyZip::commit(&pp, &mle).expect("commit");

            // Same point choice as the bench
            let point = vec![1i64; P].iter().map(|v| (*v).into()).collect_vec();

            let mut prover_transcript =
                PcsProverTranscript::new_from_commitment(&commitment).unwrap();
            let (field_cfg, projecting_element) =
                get_field_and_projecting_element::<PolyZt, F>(&mut prover_transcript.fs_transcript);

            // Prover produces a proof once (exactly as in the bench)
            TestPolyZip::test::<CHECKED>(&mut prover_transcript, &pp, &mle, &data).unwrap();
            let eval_f = TestPolyZip::evaluate::<F, CHECKED>(
                &mut prover_transcript,
                &pp,
                &mle,
                &point,
                &field_cfg,
                &projecting_element,
            )
            .unwrap();

            let point_f: Vec<F> = point.iter().map(|v| v.into_with_cfg(&field_cfg)).collect();

            let mut verifier_transcript = prover_transcript.into_verification_transcript();
            verifier_transcript
                .fs_transcript
                .absorb_slice(&commitment.root);
            let (field_cfg, projecting_element) = get_field_and_projecting_element::<PolyZt, F>(
                &mut verifier_transcript.fs_transcript,
            );

            // Verifier replays verification from the same proof (also like the bench)
            TestPolyZip::verify::<_, CHECKED>(
                &mut verifier_transcript,
                &pp,
                &commitment,
                &field_cfg,
                &projecting_element,
                &point_f,
                &eval_f,
            )
            .expect("verify");
        }

        inner::<19>();
    }
}
