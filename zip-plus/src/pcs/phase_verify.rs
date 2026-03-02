use crate::{
    ZipError,
    code::LinearCode,
    pcs::{
        ZipPlusProof,
        structs::{ZipPlus, ZipPlusCommitment, ZipPlusParams, ZipTypes},
        utils::{point_to_tensor, validate_input},
    },
    pcs_transcript::PcsTranscript,
};
use crypto_primitives::{FromPrimitiveWithConfig, FromWithConfig, IntoWithConfig, PrimeField};
use itertools::Itertools;
use num_traits::{ConstOne, ConstZero, Zero};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use zinc_poly::Polynomial;
use zinc_transcript::traits::{Transcribable, Transcript};
use zinc_utils::{
    UNCHECKED, cfg_into_iter,
    from_ref::FromRef,
    inner_product::{InnerProduct, MBSInnerProduct},
    mul_by_scalar::MulByScalar,
};

// References main.pdf for the new Zip+ protocol
impl<Zt: ZipTypes, Lc: LinearCode<Zt>> ZipPlus<Zt, Lc> {
    #[allow(clippy::arithmetic_side_effects, clippy::type_complexity)]
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
            + for<'a> FromWithConfig<&'a Zt::CombR>
            + for<'a> FromWithConfig<&'a Zt::Chal>
            + for<'a> MulByScalar<&'a F>,
        F::Inner: FromRef<Zt::Fmod> + Transcribable,
    {
        let mut transcript: PcsTranscript = proof.clone().into();
        let field_cfg = transcript
            .fs_transcript
            .get_random_field_cfg::<F, Zt::Fmod, Zt::PrimeTest>();
        Self::verify_with_field_cfg::<F, CHECK_FOR_OVERFLOW>(
            vp, comm, point_f, eval_f, transcript, &field_cfg,
        )
    }

    /// Like [`Self::verify`], but accepts a pre-computed `field_cfg` and a
    /// [`PcsTranscript`] whose `fs_transcript` has already been advanced
    /// past `get_random_field_cfg`.
    ///
    /// This avoids the redundant MillerRabin primality search when the
    /// caller already holds the PCS field configuration (e.g. because it
    /// derived `eval_f` / `point_f` from the same deterministic transcript).
    #[allow(clippy::arithmetic_side_effects, clippy::type_complexity)]
    pub fn verify_with_field_cfg<F, const CHECK_FOR_OVERFLOW: bool>(
        vp: &ZipPlusParams<Zt, Lc>,
        comm: &ZipPlusCommitment,
        point_f: &[F],
        eval_f: &F,
        mut transcript: PcsTranscript,
        field_cfg: &F::Config,
    ) -> Result<(), ZipError>
    where
        F: FromPrimitiveWithConfig
            + FromRef<F>
            + for<'a> FromWithConfig<&'a Zt::CombR>
            + for<'a> FromWithConfig<&'a Zt::Chal>
            + for<'a> MulByScalar<&'a F>,
        F::Inner: FromRef<Zt::Fmod> + Transcribable,
    {
        let batch_size = comm.batch_size;
        validate_input::<Zt, Lc, _>("verify", vp.num_vars, batch_size, &[], &[point_f])?;

        let num_rows = vp.num_rows;
        let row_len = vp.linear_code.row_len();

        // TODO Lift q0, q1 back to int and take following dot products on ints instead
        // of MBSInnerProduct in field (see comboned row)
        let (q_0, q_1) = point_to_tensor(vp.num_rows, point_f, field_cfg)?;
        let zero_f = F::zero_with_cfg(field_cfg);

        let degree_bound = Zt::Comb::DEGREE_BOUND;
        let mut per_poly_alphas: Vec<Vec<Zt::Chal>> = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            let alphas: Vec<Zt::Chal> = if degree_bound.is_zero() {
                vec![Zt::Chal::ONE]
            } else {
                transcript.fs_transcript.get_challenges(degree_bound + 1)
            };

            per_poly_alphas.push(alphas);
        }

        let b: Vec<F> = transcript.read_field_elements(num_rows, field_cfg)?;

        // Check eval claim: <q_0, b> == evals_f
        if MBSInnerProduct::inner_product::<UNCHECKED>(&q_0, &b, zero_f.clone())? != *eval_f {
            return Err(ZipError::InvalidPcsOpen(
                "Evaluation consistency failure".into(),
            ));
        }

        let coeffs: Vec<Zt::Chal> = if num_rows == 1 {
            vec![Zt::Chal::ONE]
        } else {
            transcript.fs_transcript.get_challenges(num_rows)
        };

        let combined_row: Vec<Zt::CombR> = transcript.read_const_many(row_len)?;
        let encoded_combined_row: Vec<Zt::CombR> = vp.linear_code.encode_wide(&combined_row);

        // Check: <w, q_1> = <s, b>
        // It is safe to use inner_product_unchecked because we're in a field.
        // NOTE: CombR entries (Int<M>) can exceed the field's bit-width, so the
        // CombR→F lift must reduce mod p before truncating limbs.
        // MontyField's FromWithConfig does this; BoxedMontyField's does not and will
        // panic.
        let lhs = MBSInnerProduct::mapped_inner_product::<_, _, _, _, UNCHECKED>(
            &combined_row,
            &q_1,
            zero_f.clone(),
            |cr| cr.into_with_cfg(field_cfg),
        )?;

        let rhs = MBSInnerProduct::mapped_inner_product::<_, _, _, _, UNCHECKED>(
            &coeffs,
            &b,
            zero_f.clone(),
            |cr| cr.into_with_cfg(field_cfg),
        )?;

        if lhs != rhs {
            return Err(ZipError::InvalidPcsOpen(
                "Eval-proximity link failure".into(),
            ));
        }

        let columns_and_proofs: Vec<_> = (0..Zt::NUM_COLUMN_OPENINGS)
            .map(|_| -> Result<_, ZipError> {
                let column_idx = transcript.squeeze_challenge_idx(vp.linear_code.codeword_len());
                let column_values = transcript.read_const_many(batch_size * vp.num_rows)?;
                let proof = transcript.read_merkle_proof().map_err(|e| {
                    ZipError::InvalidPcsOpen(format!("Failed to read Merkle a proof: {e}"))
                })?;

                Ok((column_idx, column_values, proof))
            })
            .try_collect()?;

        cfg_into_iter!(columns_and_proofs).try_for_each(
            |(column_idx, column_values, proof)| -> Result<(), ZipError> {
                Self::verify_column_testing_batched::<CHECK_FOR_OVERFLOW>(
                    &per_poly_alphas,
                    &coeffs,
                    &encoded_combined_row,
                    &column_values,
                    column_idx,
                    vp.num_rows,
                    batch_size,
                )?;

                proof
                    .verify(&comm.root, &column_values, column_idx)
                    .map_err(|e| {
                        ZipError::InvalidPcsOpen(format!("Column opening verification failed: {e}"))
                    })?;

                Ok(())
            },
        )?;

        Ok(())
    }

    // Proximity check: M * w[column] = sum_m(sum_j(s_j * sum_i(r_i * v_i)[column]))
    // v_i are encoded rows, r_i are alphas and where j = (0..k2) i.e. over rows
    // and m is number of polys batched
    pub(super) fn verify_column_testing_batched<const CHECK_FOR_OVERFLOW: bool>(
        per_poly_alphas: &[Vec<Zt::Chal>],
        coeffs: &[Zt::Chal],
        encoded_combined_row: &[Zt::CombR],
        all_column_entries: &[Zt::Cw],
        column: usize,
        num_rows: usize,
        batch_size: usize,
    ) -> Result<(), ZipError> {
        #[allow(clippy::arithmetic_side_effects)]
        let all_column_entries_comb =
            (0..batch_size).try_fold(Zt::CombR::ZERO, |acc, i| -> Result<_, ZipError> {
                let column_entries: Vec<_> = all_column_entries[i * num_rows..(i + 1) * num_rows]
                    .iter()
                    .map(Zt::Comb::from_ref)
                    .map(|p| {
                        Zt::CombDotChal::inner_product::<CHECK_FOR_OVERFLOW>(
                            &p,
                            &per_poly_alphas[i],
                            Zt::CombR::ZERO,
                        )
                    })
                    .try_collect()?;

                Ok(acc
                    + Zt::ArrCombRDotChal::inner_product::<CHECK_FOR_OVERFLOW>(
                        &column_entries,
                        coeffs,
                        Zt::CombR::ZERO,
                    )?)
            })?;

        if all_column_entries_comb != encoded_combined_row[column] {
            return Err(ZipError::InvalidPcsOpen("Proximity failure".into()));
        }

        Ok(())
    }

    pub(crate) fn verify_proximity_q_0<F>(
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
            ZipPlusProof,
            structs::{ZipPlus, ZipPlusHint, ZipTypes},
            test_utils::*,
        },
        pcs_transcript::PcsTranscript,
    };
    use crypto_bigint::U64;
    use crypto_primitives::{
        FromWithConfig, IntoWithConfig, PrimeField, crypto_bigint_int::Int,
        crypto_bigint_monty::MontyField,
    };
    use itertools::Itertools;
    use num_traits::{ConstOne, ConstZero, Zero};
    use rand::prelude::*;
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

    type F = MontyField<K>;

    type Zt = TestZipTypes<N, K, M>;
    type C = RaaSignFlippingCode<Zt, TestRaaConfig, 4>;

    type PolyZt = TestPolyZipTypes<K, M, DEGREE_PLUS_ONE>;
    type PolyC = RaaCode<PolyZt, TestRaaConfig, 4>;

    type TestZip = ZipPlus<Zt, C>;
    type TestPolyZip = ZipPlus<PolyZt, PolyC>;

    fn field_cfg() -> <F as PrimeField>::Config {
        let mut t = PcsTranscript::new();
        t.fs_transcript
            .get_random_field_cfg::<F, <Zt as ZipTypes>::Fmod, <Zt as ZipTypes>::PrimeTest>()
    }

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
            let cfg = *eval_f.cfg();
            let tampered = eval_f + F::one_with_cfg(&cfg);

            let result = TestZip::verify::<_, CHECKED>(&pp, &comm, &point_f, &tampered, &proof);

            assert!(result.is_err());
        }

        {
            let (pp, comm, point_f, eval_f, proof) =
                setup_full_protocol_poly::<F, N, K, M, DEGREE_PLUS_ONE>(num_vars);
            let cfg = *eval_f.cfg();
            let tampered = eval_f + F::one_with_cfg(&cfg);

            let result = TestPolyZip::verify::<_, CHECKED>(&pp, &comm, &point_f, &tampered, &proof);

            assert!(result.is_err());
        }
    }

    #[test]
    fn verification_fails_with_tampered_proof() {
        fn tamper(proof: ZipPlusProof) -> ZipPlusProof {
            let mut tampered = proof.0.clone();
            // Byte 0 is the 1-byte LENGTH_NUM_BYTES prefix for b field elements.
            // Flip byte 1 (first byte of the first b element's VALUE) instead.
            tampered[1] ^= 0x01;
            ZipPlusProof(tampered)
        }
        let num_vars = 4;

        {
            let (pp, comm, point_f, eval_f, proof) = setup_full_protocol::<F, N, K, M>(num_vars);
            let tampered = tamper(proof);
            let result = TestZip::verify::<_, CHECKED>(&pp, &comm, &point_f, &eval_f, &tampered);
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

            let (_, comm_poly2) = TestZip::commit_single(&pp, &poly2).unwrap();

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
            let (_, comm_poly2) = TestPolyZip::commit_single(&pp, &poly2).unwrap();

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

        let (hint, comm) = TestZip::commit_single(&pp, &mle1).unwrap();

        let mle2: DenseMultilinearExtension<_> = (20..=35).map(Int::from).collect();

        let int_point: Vec<<Zt as ZipTypes>::Pt> =
            (0..num_vars).map(|i| Int::from(i as i32 + 2)).collect();
        let cfg = field_cfg();
        let point_f: Vec<F> = int_point.iter().map(|v| v.into_with_cfg(&cfg)).collect();

        let (eval, proof) =
            TestZip::prove::<F, CHECKED>(&pp, std::slice::from_ref(&mle2), &point_f, &hint)
                .expect("prove should succeed");

        let verification_result =
            TestZip::verify::<_, CHECKED>(&pp, &comm, &point_f, &eval, &proof);

        assert!(verification_result.is_err());
    }

    #[test]
    fn verification_fails_due_to_a_hint_that_is_not_close() {
        let num_vars = 4;
        let (pp, mle) = setup_test_params(num_vars);

        let (original_hint, comm) = TestZip::commit_single(&pp, &mle).unwrap();

        let mut corrupted_data = original_hint.cw_matrices[0].clone();
        {
            let mut corrupted_rows = corrupted_data.to_rows_slices_mut();
            let codeword_len = pp.linear_code.codeword_len();
            let corruption_count = codeword_len / 2 + 1;
            for i in corrupted_rows[0].iter_mut().take(corruption_count) {
                *i += Int::ONE;
            }
        }

        let corrupted_merkle_tree = MerkleTree::new(&corrupted_data.to_rows_slices());
        let corrupted_hint = ZipPlusHint::new(vec![corrupted_data], corrupted_merkle_tree);

        let int_point: Vec<<Zt as ZipTypes>::Pt> =
            (0..num_vars).map(|i| Int::from(i as i32 + 2)).collect();
        let cfg = field_cfg();
        let point_f: Vec<F> = int_point.iter().map(|v| v.into_with_cfg(&cfg)).collect();

        let (eval, proof) =
            TestZip::prove::<F, CHECKED>(&pp, std::slice::from_ref(&mle), &point_f, &corrupted_hint)
                .expect("prove should succeed");

        let verification_result =
            TestZip::verify::<_, CHECKED>(&pp, &comm, &point_f, &eval, &proof);

        assert!(verification_result.is_err());
    }

    #[test]
    fn verification_fails_due_to_incorrect_evaluation() {
        let num_vars = 4;
        let (pp, mle) = setup_test_params(num_vars);

        let (hint, comm) = TestZip::commit_single(&pp, &mle).unwrap();

        let int_point: Vec<<Zt as ZipTypes>::Pt> =
            (0..num_vars).map(|i| Int::from(i as i32 + 2)).collect();
        let cfg = field_cfg();
        let point_f: Vec<F> = int_point.iter().map(|v| v.into_with_cfg(&cfg)).collect();

        let (eval, proof) =
            TestZip::prove::<F, CHECKED>(&pp, std::slice::from_ref(&mle), &point_f, &hint)
                .expect("prove should succeed");

        let incorrect_eval_f = eval + F::one_with_cfg(&cfg);

        let verification_result =
            TestZip::verify::<_, CHECKED>(&pp, &comm, &point_f, &incorrect_eval_f, &proof);

        assert!(verification_result.is_err());
    }

    #[test]
    fn verification_fails_if_proximity_check_is_invalid() {
        let poly_size: usize = 8;
        let num_vars = poly_size.ilog2() as usize;
        let row_len = 1 << (num_vars / 2);
        let linear_code = C::new(row_len);
        let pp = TestZip::setup(poly_size, linear_code);

        let mle: DenseMultilinearExtension<_> = (0..poly_size as i32)
            .map(<Zt as ZipTypes>::Eval::from)
            .collect();

        let (hint, comm) = TestZip::commit_single(&pp, &mle).expect("commit should succeed");

        let int_point = [0i64, 0i64, 0i64]
            .into_iter()
            .map(Int::<1>::from)
            .collect::<Vec<_>>();
        let cfg = field_cfg();
        let point_f: Vec<F> = int_point.iter().map(|v| v.into_with_cfg(&cfg)).collect();

        let (eval, mut proof) =
            TestZip::prove::<F, CHECKED>(&pp, std::slice::from_ref(&mle), &point_f, &hint)
                .expect("prove should succeed");

        // New transcript layout: [b field elems] [combined_row] [column openings...]
        // To trigger "Proximity failure", corrupt a column value (past b +
        // combined_row).
        let num_bytes_f = eval.inner().get_num_bytes();
        // MontyField uses ConstTranscribable: no length prefix, value only (no modulus)
        let b_section_size = pp.num_rows * num_bytes_f;
        let bytes_per_comb_r = M * size_of::<crypto_bigint::Word>();
        let combined_row_size = pp.linear_code.row_len() * bytes_per_comb_r;
        let column_values_start = b_section_size + combined_row_size;
        let bytes_per_cw = K * size_of::<crypto_bigint::Word>();
        assert!(
            column_values_start + bytes_per_cw <= proof.0.len(),
            "proof too small to tamper column values"
        );

        let flip_at = column_values_start + bytes_per_cw / 2;
        proof.0[flip_at] ^= 0x01;

        let res = TestZip::verify::<_, CHECKED>(&pp, &comm, &point_f, &eval, &proof);

        match res {
            Err(ZipError::InvalidPcsOpen(msg)) => {
                assert_eq!(msg, "Proximity failure");
            }
            Ok(()) => panic!("verification unexpectedly succeeded"),
            Err(e) => panic!("unexpected error: {e:?}"),
        }
    }

    #[test]
    fn verification_fails_if_evaluation_consistency_check_is_invalid() {
        let poly_size: usize = 8;
        let num_vars = poly_size.ilog2() as usize;
        let row_len = 1 << (num_vars / 2);
        let linear_code = C::new(row_len);
        let pp = TestZip::setup(poly_size, linear_code);

        let mle: DenseMultilinearExtension<_> =
            (0..poly_size as i32).map(Int::<INT_LIMBS>::from).collect();

        let (hint, comm) = TestZip::commit_single(&pp, &mle).expect("commit should succeed");

        let int_point: Vec<<Zt as ZipTypes>::Pt> =
            [0i64, 0i64, 0i64].into_iter().map(Int::from).collect_vec();
        let cfg = field_cfg();
        let point_f: Vec<F> = int_point.iter().map(|v| v.into_with_cfg(&cfg)).collect();

        let (eval, mut proof) =
            TestZip::prove::<F, CHECKED>(&pp, std::slice::from_ref(&mle), &point_f, &hint).unwrap();

        // Transcript starts with b field elements (no length prefix for
        // MontyField's ConstTranscribable, value-only serialization).
        // Flip a byte inside the first b element to corrupt eval consistency.
        let num_bytes_f = eval.inner().get_num_bytes();
        let flip_at = num_bytes_f / 4;
        assert!(
            flip_at < proof.0.len(),
            "proof too small to tamper b section"
        );
        proof.0[flip_at] ^= 0x01;

        let res = TestZip::verify::<_, CHECKED>(&pp, &comm, &point_f, &eval, &proof);

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

        let (hint, comm) = TestZip::commit_single(&pp, &mle).expect("commit should succeed");

        let int_point: Vec<<Zt as ZipTypes>::Pt> =
            [0i64, 0i64, 0i64].into_iter().map(Int::from).collect_vec();
        let cfg = field_cfg();
        let point_f: Vec<F> = int_point.iter().map(|v| v.into_with_cfg(&cfg)).collect();

        let (eval, proof) =
            TestZip::prove::<F, CHECKED>(&pp, std::slice::from_ref(&mle), &point_f, &hint).unwrap();

        let res = TestZip::verify::<_, CHECKED>(&pp, &comm, &point_f, &eval, &proof);
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

        let (hint, comm) = TestZip::commit_single(&pp, &mle).expect("commit should succeed");

        let int_point: Vec<<Zt as ZipTypes>::Pt> = vec![Int::ZERO; num_vars];
        let cfg = field_cfg();
        let point_f: Vec<F> = int_point.iter().map(|v| v.into_with_cfg(&cfg)).collect();

        let (eval, proof) =
            TestZip::prove::<F, CHECKED>(&pp, std::slice::from_ref(&mle), &point_f, &hint).unwrap();

        let res = TestZip::verify::<_, CHECKED>(&pp, &comm, &point_f, &eval, &proof);
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

        let (hint, comm) = TestZip::commit_single(&pp, &poly).unwrap();

        let mut int_point = vec![<Zt as ZipTypes>::Pt::ZERO; num_vars];
        int_point[0] = <Zt as ZipTypes>::Pt::ONE;
        let cfg = field_cfg();
        let point_f: Vec<F> = int_point.iter().map(|v| v.into_with_cfg(&cfg)).collect();

        let (eval_f, proof) =
            TestZip::prove::<F, CHECKED>(&pp, std::slice::from_ref(&poly), &point_f, &hint).unwrap();

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

        let (hint, comm) = TestZip::commit_single(&pp, &poly).unwrap();

        let int_point: Vec<<Zt as ZipTypes>::Pt> = vec![Int::from(1), Int::from(2)];
        let cfg = field_cfg();
        let point_f: Vec<F> = int_point.iter().map(|v| v.into_with_cfg(&cfg)).collect();

        let (eval, proof) =
            TestZip::prove::<F, CHECKED>(&pp, std::slice::from_ref(&poly), &point_f, &hint).unwrap();

        let verification_result =
            TestZip::verify::<_, CHECKED>(&pp, &comm, &point_f, &eval, &proof);

        assert!(verification_result.is_ok());
    }

    #[test]
    fn verification_fails_at_proximity_link_check_if_combined_row_is_corrupted() {
        let poly_size = 8;
        let linear_code = C::new(poly_size);
        let pp = TestZip::setup(poly_size, linear_code);

        let mle: DenseMultilinearExtension<_> =
            (1..=poly_size as i32).map(Int::<INT_LIMBS>::from).collect();

        let (hint, comm) = TestZip::commit_single(&pp, &mle).expect("commit should succeed");

        let int_point: Vec<<Zt as ZipTypes>::Pt> =
            [0i64, 0i64, 0i64].into_iter().map(Int::from).collect_vec();
        let cfg = field_cfg();
        let point_f: Vec<F> = int_point.iter().map(|v| v.into_with_cfg(&cfg)).collect();

        let (eval, mut proof) =
            TestZip::prove::<F, CHECKED>(&pp, std::slice::from_ref(&mle), &point_f, &hint).unwrap();

        // Offset past b section to reach combined_row (CombR = Int<M>).
        let num_bytes_f = eval.inner().get_num_bytes();
        let b_section_size = 1 + pp.num_rows * 2 * num_bytes_f;
        let bytes_to_corrupt = M * size_of::<crypto_bigint::Word>();
        assert!(
            b_section_size + bytes_to_corrupt <= proof.0.len(),
            "proof too small to tamper combined_row"
        );

        for b in &mut proof.0[b_section_size..b_section_size + bytes_to_corrupt] {
            *b = 0xFF;
        }

        let res = TestZip::verify::<_, CHECKED>(&pp, &comm, &point_f, &eval, &proof);
        assert!(res.is_err());
    }

    /// Mirrors: `Zip/Verify: RandomField<4>, poly_size = 2^12 (Int limbs = 1)`
    #[test]
    fn bench_p12_verify() {
        fn inner<const P: usize>() {
            let mut rng = ThreadRng::default();
            // Match the benchmark’s transcript usage for linear code construction
            let poly_size = 1 << P;
            let row_len = 1 << (P / 2);
            let linear_code = C::new(row_len);
            let pp = TestZip::setup(poly_size, linear_code);

            let mle = DenseMultilinearExtension::rand(P, &mut rng);
            let (hint, commitment) = TestZip::commit_single(&pp, &mle).expect("commit");

            let field_cfg = {
                let mut t = PcsTranscript::new();
                t.fs_transcript
                    .get_random_field_cfg::<F, <Zt as ZipTypes>::Fmod, <Zt as ZipTypes>::PrimeTest>()
            };
            let point_f: Vec<F> = (0..P)
                .map(|_| (&Int::<INT_LIMBS>::from(1i64)).into_with_cfg(&field_cfg))
                .collect();

            let (eval, proof) =
                TestZip::prove::<F, CHECKED>(&pp, std::slice::from_ref(&mle), &point_f, &hint)
                    .unwrap();

            let zero_f = F::zero_with_cfg(&field_cfg);
            let mle_f = DenseMultilinearExtension::from_evaluations_vec(
                P,
                mle.iter().map(|c| c.into_with_cfg(&field_cfg)).collect(),
                zero_f.clone(),
            );
            let expected_eval = mle_f.evaluate(&point_f, zero_f).unwrap();
            assert_eq!(eval, expected_eval, "prover returned wrong eval");

            TestZip::verify::<_, CHECKED>(&pp, &commitment, &point_f, &eval, &proof)
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
            let row_len = 1 << (P / 2);
            let linear_code = PolyC::new(row_len);
            let pp = TestPolyZip::setup(poly_size, linear_code);

            let mle = DenseMultilinearExtension::rand(P, &mut rng);
            let (hint, commitment) = TestPolyZip::commit_single(&pp, &mle).expect("commit");

            let field_cfg = {
                let mut t = PcsTranscript::new();
                t.fs_transcript
                    .get_random_field_cfg::<F, <Zt as ZipTypes>::Fmod, <Zt as ZipTypes>::PrimeTest>()
            };
            let point_f: Vec<F> = (0..P)
                .map(|_| (&(1i128)).into_with_cfg(&field_cfg))
                .collect();

            let (eval, proof) =
                TestPolyZip::prove::<F, CHECKED>(&pp, std::slice::from_ref(&mle), &point_f, &hint)
                    .unwrap();

            TestPolyZip::verify::<_, CHECKED>(&pp, &commitment, &point_f, &eval, &proof)
                .expect("verify");
        }

        inner::<12>();
    }

    fn batched_prove_verify_inner<const BATCH: usize>(num_vars: usize) {
        let poly_size = 1 << num_vars;
        let linear_code = C::new(poly_size);
        let pp = TestZip::setup(poly_size, linear_code);

        let polys: Vec<DenseMultilinearExtension<_>> = (0..BATCH)
            .map(|b| {
                let base = (b * poly_size) as i32;
                (base + 1..=base + poly_size as i32)
                    .map(Int::<INT_LIMBS>::from)
                    .collect()
            })
            .collect();

        let (hint, comm) = TestZip::commit(&pp, &polys).unwrap();
        let int_point: Vec<<Zt as ZipTypes>::Pt> =
            (0..num_vars).map(|i| Int::from(i as i32 + 2)).collect();
        let cfg = field_cfg();
        let point_f: Vec<F> = int_point.iter().map(|v| v.into_with_cfg(&cfg)).collect();

        let (eval, proof) = TestZip::prove::<F, CHECKED>(&pp, &polys, &point_f, &hint).unwrap();

        let res = TestZip::verify::<_, CHECKED>(&pp, &comm, &point_f, &eval, &proof);
        assert!(
            res.is_ok(),
            "Batched verify (batch={BATCH}) failed: {res:?}"
        );
    }

    #[test]
    fn batched_prove_verify_batch_2() {
        batched_prove_verify_inner::<2>(4);
    }

    #[test]
    fn batched_prove_verify_batch_5() {
        batched_prove_verify_inner::<5>(4);
    }

    #[test]
    fn batched_prove_verify_batch_1_roundtrip() {
        batched_prove_verify_inner::<1>(4);
    }

    #[test]
    fn batched_verify_fails_with_tampered_eval() {
        let num_vars = 4;
        let poly_size = 1 << num_vars;
        let linear_code = C::new(poly_size);
        let pp = TestZip::setup(poly_size, linear_code);

        let polys: Vec<DenseMultilinearExtension<_>> = vec![
            (1..=poly_size as i32).map(Int::from).collect(),
            (17..=16 + poly_size as i32).map(Int::from).collect(),
        ];

        let (hint, comm) = TestZip::commit(&pp, &polys).unwrap();
        let int_point: Vec<<Zt as ZipTypes>::Pt> = (0..num_vars).map(|i| Int::from(i + 2)).collect();
        let cfg = field_cfg();
        let point_f: Vec<F> = int_point.iter().map(|v| v.into_with_cfg(&cfg)).collect();

        let (eval, proof) = TestZip::prove::<F, CHECKED>(&pp, &polys, &point_f, &hint).unwrap();
        let tampered_eval = eval + F::one_with_cfg(&cfg);

        let res = TestZip::verify::<_, CHECKED>(&pp, &comm, &point_f, &tampered_eval, &proof);
        assert!(res.is_err(), "Should fail when eval is tampered");
    }
}
