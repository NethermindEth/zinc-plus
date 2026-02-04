use crate::{
    ZipError,
    code::LinearCode,
    merkle::MtHash,
    pcs::{
        ZipPlusProof, ZipPlusTestTranscript,
        structs::{ZipPlus, ZipPlusCommitment, ZipPlusParams, ZipTypes},
        utils::{ColumnOpening, point_to_tensor, validate_input},
    },
    pcs_transcript::PcsTranscript,
};
use crypto_primitives::{FromPrimitiveWithConfig, FromWithConfig, IntoWithConfig, PrimeField};
use itertools::Itertools;
use num_traits::{ConstOne, ConstZero, Zero};
use zinc_poly::Polynomial;
use zinc_transcript::KeccakTranscript;
use zinc_transcript::traits::{Transcribable, Transcript};
use zinc_utils::{
    UNCHECKED,
    from_ref::FromRef,
    inner_product::{InnerProduct, MBSInnerProduct},
    mul_by_scalar::MulByScalar,
    projectable_to_field::ProjectableToField,
};

impl<Zt: ZipTypes, Lc: LinearCode<Zt>> ZipPlus<Zt, Lc> {
    pub fn verify_test_phase<const CHECK_FOR_OVERFLOW: bool>(
        vp: &ZipPlusParams<Zt, Lc>,
        comm: &ZipPlusCommitment,
        test_transcript: ZipPlusTestTranscript,
    ) -> Result<(), ZipError> {
        let mut transcript: PcsTranscript = test_transcript.into();
        transcript.fs_transcript = KeccakTranscript::default();
        transcript.stream.set_position(0);

        Self::verify_testing::<CHECK_FOR_OVERFLOW>(vp, &comm.root, &mut transcript)?;

        Ok(())
    }

    pub fn verify<F, const CHECK_FOR_OVERFLOW: bool>(
        vp: &ZipPlusParams<Zt, Lc>,
        comm: &ZipPlusCommitment,
        point_f: &[F],
        eval_f: &F,
        proof: &ZipPlusProof,
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

        let mut transcript: PcsTranscript = proof.clone().into();

        let columns_opened =
            Self::verify_testing::<CHECK_FOR_OVERFLOW>(vp, &comm.root, &mut transcript)?;

        let field_cfg = transcript
            .fs_transcript
            .get_random_field_cfg::<F, Zt::Fmod, Zt::PrimeTest>();
        let projecting_element: Zt::Chal = transcript.fs_transcript.get_challenge();
        let projecting_element: F = (&projecting_element).into_with_cfg(&field_cfg);

        Self::verify_evaluation(
            vp,
            point_f,
            eval_f,
            &columns_opened,
            &mut transcript,
            projecting_element,
            &field_cfg,
        )?;

        Ok(())
    }

    #[allow(clippy::arithmetic_side_effects, clippy::type_complexity)]
    pub(super) fn verify_testing<const CHECK_FOR_OVERFLOW: bool>(
        vp: &ZipPlusParams<Zt, Lc>,
        root: &MtHash,
        transcript: &mut PcsTranscript,
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

        let mut columns_opened: Vec<(usize, Vec<Zt::Cw>)> =
            Vec::with_capacity(Zt::NUM_COLUMN_OPENINGS);

        for _ in 0..Zt::NUM_COLUMN_OPENINGS {
            let column_idx = transcript.squeeze_challenge_idx(vp.linear_code.codeword_len());
            let column_values = transcript.read_const_many(vp.num_rows)?;

            if let Some((ref alphas, ref coeffs, ref encoded_combined_row)) = encoded_combined_rows
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

            ColumnOpening::verify_column(root, &column_values, column_idx, transcript).map_err(
                |e| ZipError::InvalidPcsOpen(format!("Column opening verification failed: {e}")),
            )?;
            // TODO: Verify column opening is taking a long time.
            columns_opened.push((column_idx, column_values));
        }

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
        transcript: &mut PcsTranscript,
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
        for (column_idx, column_values) in columns_opened.iter() {
            Self::verify_proximity_q_0(
                &q_0,
                &encoded_combined_row,
                column_values,
                *column_idx,
                vp.num_rows,
                &project,
                field_cfg,
            )?;
        }

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
            // TODO: this inner product is taking a long time.
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
            ZipPlusProof,
            structs::{ZipPlus, ZipPlusHint, ZipTypes},
            test_utils::*,
        },
    };
    use crypto_bigint::{Random, U64};
    use crypto_primitives::{
        Field, FromWithConfig, IntoWithConfig, PrimeField,
        crypto_bigint_boxed_monty::BoxedMontyField, crypto_bigint_int::Int,
    };
    use itertools::Itertools;
    use num_traits::{ConstOne, ConstZero, Zero};
    use rand::prelude::*;
    use zinc_poly::{
        mle::{DenseMultilinearExtension, MultilinearExtensionRand},
        univariate::binary::BinaryPoly,
    };
    use zinc_transcript::traits::Transcribable;
    use zinc_utils::CHECKED;

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

    #[test]
    fn successful_verification_of_valid_proof() {
        let num_vars = 4;
        {
            let (pp, comm, point_f, eval_f, proof) = setup_full_protocol::<F, N, K, M>(num_vars);

            let result = TestZip::verify::<_, CHECKED>(&pp, &comm, &point_f, &eval_f, &proof);
            assert!(result.is_ok(), "Verification failed: {result:?}")
        };
        {
            let (pp, comm, point_f, eval_f, proof) =
                setup_full_protocol_poly::<F, N, K, M, DEGREE_PLUS_ONE>(num_vars);

            let result = TestPolyZip::verify::<_, CHECKED>(&pp, &comm, &point_f, &eval_f, &proof);

            assert!(result.is_ok(), "Verification failed: {result:?}");
        }
    }

    #[test]
    fn verification_fails_with_incorrect_evaluation() {
        let num_vars = 4;

        {
            let (pp, comm, point_f, eval_f, proof) = setup_full_protocol::<F, N, K, M>(num_vars);
            let cfg = eval_f.cfg().clone();

            let result = TestZip::verify::<_, CHECKED>(
                &pp,
                &comm,
                &point_f,
                &(eval_f + F::one_with_cfg(&cfg)),
                &proof,
            );

            assert!(result.is_err());
        }

        {
            let (pp, comm, point_f, eval_f, proof) =
                setup_full_protocol_poly::<F, N, K, M, DEGREE_PLUS_ONE>(num_vars);
            let cfg = eval_f.cfg().clone();

            let result = TestPolyZip::verify::<_, CHECKED>(
                &pp,
                &comm,
                &point_f,
                &(eval_f + F::one_with_cfg(&cfg)),
                &proof,
            );

            assert!(result.is_err());
        }
    }

    #[test]
    fn verification_fails_with_tampered_proof() {
        fn tamper(proof: ZipPlusProof) -> ZipPlusProof {
            let mut tampered = proof.0.clone();
            tampered[0] ^= 0x01;
            ZipPlusProof(tampered)
        }
        let num_vars = 4;

        {
            let (pp, comm, point_f, eval, proof) = setup_full_protocol::<F, N, K, M>(num_vars);
            let tampered = tamper(proof);
            let result = TestZip::verify::<_, CHECKED>(&pp, &comm, &point_f, &eval, &tampered);
            assert!(result.is_err());
        }

        {
            let (pp, comm, point_f, eval_f, proof) =
                setup_full_protocol_poly::<F, N, K, M, DEGREE_PLUS_ONE>(num_vars);
            let tampered = tamper(proof);
            let result =
                TestPolyZip::verify::<_, CHECKED>(&pp, &comm, &point_f, &eval_f, &tampered);
            assert!(result.is_err());
        }
    }

    #[test]
    fn verification_fails_with_wrong_commitment() {
        let num_vars = 4;
        {
            let (pp, _comm_poly1, point_f, eval_f, proof_poly1) =
                setup_full_protocol::<F, N, K, M>(num_vars);

            let poly2: DenseMultilinearExtension<_> =
                (20..(20 + (1 << num_vars))).map(Int::from).collect();

            let (_, comm_poly2) = TestZip::commit(&pp, &poly2).unwrap();

            let result =
                TestZip::verify::<_, CHECKED>(&pp, &comm_poly2, &point_f, &eval_f, &proof_poly1);

            assert!(result.is_err());
        }

        {
            let (pp, _comm_poly1, point_f, eval_f, proof_poly1) =
                setup_full_protocol_poly::<F, N, K, M, DEGREE_PLUS_ONE>(num_vars);

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
                &pp,
                &comm_poly2,
                &point_f,
                &eval_f,
                &proof_poly1,
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
            let (pp, comm, _point_f, eval_f, proof) =
                setup_full_protocol_poly::<F, N, K, M, DEGREE_PLUS_ONE>(num_vars);
            let invalid_point = make_invalid_point(eval_f.cfg());

            let result =
                TestPolyZip::verify::<_, CHECKED>(&pp, &comm, &invalid_point, &eval_f, &proof);

            assert!(matches!(result, Err(..)));
        }

        {
            let (pp, comm, _point_f, eval_f, proof) = setup_full_protocol::<F, N, K, M>(num_vars);
            let invalid_point = make_invalid_point(eval_f.cfg());

            let result = TestZip::verify::<_, CHECKED>(&pp, &comm, &invalid_point, &eval_f, &proof);

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

        let test_mle2_proof =
            TestZip::test::<CHECKED>(&pp, &mle2, &data).expect("test phase should succeed");
        let (eval_f, eval_mle2_proof) =
            TestZip::evaluate::<F, CHECKED>(&pp, &mle2, &point, test_mle2_proof)
                .expect("evaluation phase should succeed");

        let eval_mle1 = mle1
            .evaluate(&point, Zero::zero())
            .expect("Failed to evaluate polynomial");
        let field_cfg = eval_f.cfg().clone();

        let point_f: Vec<F> = point.iter().map(|v| v.into_with_cfg(&field_cfg)).collect();
        let eval_mle1_f = eval_mle1.into_with_cfg(&field_cfg);

        let verification_result =
            TestZip::verify::<_, CHECKED>(&pp, &comm, &point_f, &eval_mle1_f, &eval_mle2_proof);

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

        let test_transcript = TestZip::test::<CHECKED>(&pp, &mle, &corrupted_data)
            .expect("test phase should succeed");
        let (eval_f, proof) = TestZip::evaluate::<F, CHECKED>(&pp, &mle, &point, test_transcript)
            .expect("evaluation phase should succeed");
        let field_cfg = eval_f.cfg().clone();

        let expected_eval = mle
            .evaluate(&point, Zero::zero())
            .expect("Failed to evaluate polynomial");
        assert_eq!(eval_f, expected_eval.into_with_cfg(&field_cfg));

        let point_f: Vec<F> = point.iter().map(|v| v.into_with_cfg(&field_cfg)).collect();

        let verification_result =
            TestZip::verify::<_, CHECKED>(&pp, &comm, &point_f, &eval_f, &proof);

        assert!(verification_result.is_err());
    }

    #[test]
    fn verification_fails_due_to_incorrect_evaluation() {
        let num_vars = 4;
        let (pp, mle) = setup_test_params(num_vars);

        let (data, comm) = TestZip::commit(&pp, &mle).unwrap();

        let point: Vec<<Zt as ZipTypes>::Pt> =
            (0..num_vars).map(|i| Int::from(i as i32 + 2)).collect();

        let test_transcript =
            TestZip::test::<CHECKED>(&pp, &mle, &data).expect("test phase should succeed");
        let (correct_eval_f, proof) =
            TestZip::evaluate::<F, CHECKED>(&pp, &mle, &point, test_transcript)
                .expect("evaluation phase should succeed");
        let field_cfg = correct_eval_f.cfg().clone();

        let incorrect_eval_f = correct_eval_f + F::one_with_cfg(&field_cfg);
        let point_f: Vec<F> = point.iter().map(|v| v.into_with_cfg(&field_cfg)).collect();

        let verification_result = TestZip::verify::<_, CHECKED>(
            &pp,
            &comm,
            &point_f,
            &incorrect_eval_f, // Use the wrong evaluation here
            &proof,
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

        let test_transcript =
            TestZip::test::<CHECKED>(&pp, &mle, &data).expect("test phase should succeed");
        let (eval_f, mut proof) =
            TestZip::evaluate::<F, CHECKED>(&pp, &mle, &point, test_transcript)
                .expect("evaluation phase should succeed");
        let field_cfg = eval_f.cfg().clone();

        assert_eq!(
            eval_f,
            eval.into_with_cfg(&field_cfg),
            "Evaluation mismatch after opening"
        );

        let point_f: Vec<F> = point.iter().map(|v| v.into_with_cfg(&field_cfg)).collect();

        let row_len = pp.linear_code.row_len();
        let bytes_per_int = M * size_of::<crypto_bigint::Word>();
        let first_combined_row_bytes = row_len * bytes_per_int;
        assert!(
            first_combined_row_bytes <= proof.0.len(),
            "proof too small to tamper"
        );

        let flip_at = bytes_per_int * (row_len / 2);
        proof.0[flip_at] ^= 0x01;

        let res = TestZip::verify::<_, CHECKED>(&pp, &comm, &point_f, &eval_f, &proof);

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
            .map(|_| <Zt as ZipTypes>::Pt::random(&mut rng))
            .collect();

        let (mut data, comm) = TestZip::commit(&pp, &mle).unwrap();
        data.cw_matrix.to_rows_slices_mut()[0][0] += Int::ONE;

        let test_transcript = TestZip::test::<CHECKED>(&pp, &mle, &data).unwrap();
        let (eval_f, proof) =
            TestZip::evaluate::<F, CHECKED>(&pp, &mle, &point, test_transcript).unwrap();
        let field_cfg = eval_f.cfg().clone();

        let point_f = point
            .iter()
            .map(|v| v.into_with_cfg(&field_cfg))
            .collect_vec();
        let eval_f = evaluate_in_field(&mle, &point_f, &field_cfg);
        let verification_result =
            TestZip::verify::<_, CHECKED>(&pp, &comm, &point_f, &eval_f, &proof);

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

        let test_transcript = TestZip::test::<CHECKED>(&pp, &mle, &data).unwrap();
        let (eval_f, mut proof) =
            TestZip::evaluate::<F, CHECKED>(&pp, &mle, &point, test_transcript).unwrap();
        let field_cfg = eval_f.cfg().clone();

        let point_f: Vec<F> = point.iter().map(|v| v.into_with_cfg(&field_cfg)).collect();

        let row_len = pp.linear_code.row_len();
        // Two elements: one for value and one for module
        let bytes_per_field = eval_f.inner().get_num_bytes() * 2;
        let q0_bytes = row_len * bytes_per_field;
        assert!(
            proof.0.len() >= q0_bytes,
            "proof too small to contain q_0_combined_row"
        );

        let tail_start = proof.0.len() - q0_bytes;
        let flip_at = tail_start + (bytes_per_field / 4);
        proof.0[flip_at] ^= 0x01;

        let res = TestZip::verify::<_, CHECKED>(&pp, &comm, &point_f, &eval_f, &proof);

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

        let test_transcript = TestZip::test::<CHECKED>(&pp, &mle, &data).unwrap();
        let (real_eval_f, proof) =
            TestZip::evaluate::<F, CHECKED>(&pp, &mle, &point, test_transcript).unwrap();
        let field_cfg = real_eval_f.cfg().clone();

        let eval_f = mle
            .evaluate(&point, Zero::zero())
            .unwrap()
            .into_with_cfg(&field_cfg);
        let point_f: Vec<F> = point.iter().map(|v| v.into_with_cfg(&field_cfg)).collect();

        let res = TestZip::verify::<_, CHECKED>(&pp, &comm, &point_f, &eval_f, &proof);
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

        let test_transcript = TestZip::test::<CHECKED>(&pp, &mle, &data).unwrap();
        let (real_eval_f, proof) =
            TestZip::evaluate::<F, CHECKED>(&pp, &mle, &point, test_transcript).unwrap();
        let field_cfg = real_eval_f.cfg().clone();

        let eval_f = mle
            .evaluate(&point, Zero::zero())
            .unwrap()
            .into_with_cfg(&field_cfg);
        let point_f: Vec<F> = point.iter().map(|v| v.into_with_cfg(&field_cfg)).collect();

        let res = TestZip::verify::<_, CHECKED>(&pp, &comm, &point_f, &eval_f, &proof);
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

        let test_transcript = TestZip::test::<CHECKED>(&pp, &poly, &data).unwrap();
        let (eval_f, proof) =
            TestZip::evaluate::<F, CHECKED>(&pp, &poly, &point, test_transcript).unwrap();
        let field_cfg = eval_f.cfg().clone();

        let expected_eval = poly
            .evaluate(&point, Zero::zero())
            .expect("failed to evaluate polynomial");
        assert_eq!(eval_f, expected_eval.into_with_cfg(&field_cfg));

        let point_f: Vec<F> = point.iter().map(|v| v.into_with_cfg(&field_cfg)).collect();

        let verification_result =
            TestZip::verify::<_, CHECKED>(&pp, &comm, &point_f, &eval_f, &proof);

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

        let test_transcript = TestZip::test::<CHECKED>(&pp, &poly, &hint).unwrap();
        let (eval_f, proof) =
            TestZip::evaluate::<F, CHECKED>(&pp, &poly, &point, test_transcript).unwrap();
        let field_cfg = eval_f.cfg().clone();

        let expected_eval = poly
            .evaluate(&point, Zero::zero())
            .expect("failed to evaluate polynomial");
        assert_eq!(eval_f, expected_eval.into_with_cfg(&field_cfg));

        let point_f: Vec<F> = point.iter().map(|v| v.into_with_cfg(&field_cfg)).collect();

        let verification_result =
            TestZip::verify::<_, CHECKED>(&pp, &comm, &point_f, &eval_f, &proof);

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

        let test_transcript = TestZip::test::<CHECKED>(&pp, &mle, &data).unwrap();
        let (real_eval_f, mut proof) =
            TestZip::evaluate::<F, CHECKED>(&pp, &mle, &point, test_transcript).unwrap();
        let field_cfg = real_eval_f.cfg().clone();

        let eval_f = mle
            .evaluate(&point, Zero::zero())
            .unwrap()
            .into_with_cfg(&field_cfg);
        let point_f: Vec<F> = point.iter().map(|v| v.into_with_cfg(&field_cfg)).collect();

        let row_len = pp.linear_code.row_len();
        let bytes_per_int = M * 8;
        let first_section_bytes = row_len * bytes_per_int;
        assert!(
            first_section_bytes <= proof.0.len(),
            "proof too small to tamper u'"
        );

        for b in &mut proof.0[0..bytes_per_int] {
            *b = 0xFF;
        }

        let res = TestZip::verify::<_, CHECKED>(&pp, &comm, &point_f, &eval_f, &proof);
        assert!(res.is_err());
    }

    /// Mirrors: `Zip/Verify: RandomField<4>, poly_size = 2^12 (Int limbs = 1)`
    #[test]
    fn bench_p12_verify() {
        fn inner<const P: usize>() {
            let mut rng = ThreadRng::default();
            // Match the benchmark’s transcript usage for linear code construction
            let poly_size = 1 << P;
            let linear_code = C::new(poly_size);
            let pp = TestZip::setup(poly_size, linear_code);

            let mle = DenseMultilinearExtension::rand(P, &mut rng);
            let (data, commitment) = TestZip::commit(&pp, &mle).expect("commit");

            // Same point choice as the bench
            let point = vec![1i64; P].iter().map(|v| v.into()).collect_vec();
            let eval = *mle.last().expect("nonempty evals");

            // Prover produces a proof once (exactly as in the bench)
            let test_transcript = TestZip::test::<CHECKED>(&pp, &mle, &data).unwrap();
            let (eval_f, proof) =
                TestZip::evaluate::<F, CHECKED>(&pp, &mle, &point, test_transcript).unwrap();
            let field_cfg = eval_f.cfg().clone();

            assert_eq!(
                eval_f,
                eval.into_with_cfg(&field_cfg),
                "Evaluation mismatch after opening"
            );

            let point_f: Vec<F> = point.iter().map(|v| v.into_with_cfg(&field_cfg)).collect();

            // Verifier replays verification from the same proof (also like the bench)
            TestZip::verify::<_, CHECKED>(&pp, &commitment, &point_f, &eval_f, &proof)
                .expect("verify");
        }

        inner::<12>();
    }

    /// Mirrors: `Zip+/Verify` for `poly_size=2^12`
    #[test]
    fn bench_p12_verify_poly() {
        fn inner<const P: usize>() {
            let mut rng = ThreadRng::default();
            // Match the benchmark’s transcript usage for linear code construction
            let poly_size = 1 << P;
            let linear_code = PolyC::new(poly_size);
            let pp = TestPolyZip::setup(poly_size, linear_code);

            let mle = DenseMultilinearExtension::rand(P, &mut rng);
            let (data, commitment) = TestPolyZip::commit(&pp, &mle).expect("commit");

            // Same point choice as the bench
            let point = vec![1i64; P].iter().map(|v| (*v).into()).collect_vec();

            // Prover produces a proof once (exactly as in the bench)
            let test_proof = TestPolyZip::test::<CHECKED>(&pp, &mle, &data).unwrap();
            let (eval_f, eval_proof) =
                TestPolyZip::evaluate::<F, CHECKED>(&pp, &mle, &point, test_proof).unwrap();
            let field_cfg = eval_f.cfg().clone();

            let point_f: Vec<F> = point.iter().map(|v| v.into_with_cfg(&field_cfg)).collect();

            // Verifier replays verification from the same proof (also like the bench)
            TestPolyZip::verify::<_, CHECKED>(&pp, &commitment, &point_f, &eval_f, &eval_proof)
                .expect("verify");
        }

        inner::<19>();
    }
}
