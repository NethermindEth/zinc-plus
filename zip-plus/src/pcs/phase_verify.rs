use crate::{
    ZipError,
    code::LinearCode,
    pcs::{
        structs::{ZipPlus, ZipPlusCommitment, ZipPlusParams, ZipTypes},
        utils::{point_to_tensor, validate_input},
    },
    pcs_transcript::PcsVerifierTranscript,
};
use crypto_primitives::{FromPrimitiveWithConfig, FromWithConfig, IntoWithConfig};
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

impl<Zt: ZipTypes, Lc: LinearCode<Zt>> ZipPlus<Zt, Lc> {
    /// Verifies an opening proof for one or more committed multilinear
    /// polynomials at an evaluation point, using the Zip+ protocol.
    ///
    /// This replaces the old two-phase (verify_testing + verify_evaluation)
    /// approach. The old protocol performed two proximity checks (one in CombR,
    /// one in F via a separate `projecting_element` γ) and one eval consistency
    /// check. The merged protocol eliminates the F-domain proximity check
    /// entirely, replacing it with a coherence check between `b` and `w`
    /// that ties the single CombR proximity check to the evaluation claim.
    ///
    /// # Verification checks (4 total)
    ///
    /// 1. **Eval consistency**: `<q_0, b> == eval_f`. Ensures the claimed
    ///    evaluation matches the `b` vector written by the prover, where `b_j =
    ///    sum_i(<w'_ij, q_1>)` and `w'_ij` is the j-th decoded row of poly i
    ///    after taking the random linear combination `<entry, alphas_i>` of
    ///    every entry.
    ///
    /// 2. **Coherence** (b-w): `<w, q_1> == <s, b>`. Ensures `b` and `w` are
    ///    derived from the same underlying rows `w'_j`, tying the
    ///    proximity-tested `w` to the eval-tested `b`.
    ///
    /// 3. **Proximity** (per opened column, batched across polys): `Enc(w)[col]
    ///    == sum_i(sum_j(s_j * <v_ij[col], alphas_i>))`. For each poly i and
    ///    row j, takes the random linear combination `<v_ij[col], alphas_i>` of
    ///    the Cw column entry to get a CombR value, combines rows with
    ///    coefficients `s`, sums across polys. Compares against the encoded
    ///    combined row.
    ///
    /// 4. **Merkle proof** (per opened column): verifies the column values
    ///    against `comm.root`, ensuring that the data matches what was
    ///    committed.
    ///
    /// Chain of trust: Merkle (check 4) → column data authentic → proximity
    /// (check 3) → `w` is a valid codeword consistent with columns →
    /// coherence (check 2) → `b` is consistent with `w` →
    /// eval consistency (check 1) → `eval_f` is correct.
    ///
    /// # Algorithm
    /// 1. Computes `(q_0, q_1) = point_to_tensor(point_f)`.
    /// 2. Per polynomial, re-derives `alphas` from the transcript.
    /// 3. Reads `b` (length `num_rows`) from the transcript.
    /// 4. **Check 1**: asserts `<q_0, b> == eval_f`.
    /// 5. Re-derives combination coefficients `s` (or `[1]` when `num_rows ==
    ///    1`).
    /// 6. Reads combined row `w` (CombR, length `row_len`) and encodes it.
    /// 7. **Check 2**: asserts `<w, q_1> == <s, b>`.
    /// 8. For each of `NUM_COLUMN_OPENINGS`: a. Squeezes column index, reads
    ///    per-poly column values + Merkle proof. b. **Check 3**:
    ///    `verify_column_testing_batched`. c. **Check 4**:
    ///    `proof.verify(comm.root, column_values, col)`.
    ///
    /// # Parameters
    /// - `vp`: Public parameters (same as prover's `pp`).
    /// - `comm`: The `ZipPlusCommitment` (Merkle root + batch size) from the
    ///   commit phase.
    /// - `point_f`: The evaluation point in field `F` (length `num_vars`).
    /// - `eval_f`: The claimed combined evaluation `<q_0, b>`.
    /// - `proof`: The `ZipPlusProof` produced by `prove`.
    ///
    /// # Returns
    /// `Ok(())` if all four checks pass.
    ///
    /// # Errors
    /// - `ZipError::InvalidPcsParam` if inputs are malformed.
    /// - `ZipError::InvalidPcsOpen("Evaluation consistency failure")` if check
    ///   1 fails.
    /// - `ZipError::InvalidPcsOpen("Coherence failure")` if check 2 fails.
    /// - `ZipError::InvalidPcsOpen("Proximity failure")` if check 3 fails.
    /// - `ZipError::InvalidPcsOpen("Column opening verification failed: ...")`
    ///   if check 4 (Merkle) fails.
    #[allow(clippy::arithmetic_side_effects, clippy::type_complexity)]
    pub fn verify<F, const CHECK_FOR_OVERFLOW: bool>(
        transcript: &mut PcsVerifierTranscript,
        vp: &ZipPlusParams<Zt, Lc>,
        comm: &ZipPlusCommitment,
        field_cfg: &F::Config,
        point_f: &[F],
        eval_f: &F,
    ) -> Result<(), ZipError>
    where
        F: FromPrimitiveWithConfig
            + FromRef<F>
            + for<'a> FromWithConfig<&'a Zt::CombR>
            + for<'a> FromWithConfig<&'a Zt::Chal>
            + for<'a> MulByScalar<&'a F>,
        F::Inner: Transcribable,
        F::Modulus: FromRef<Zt::Fmod> + Transcribable,
    {
        let batch_size = comm.batch_size;
        validate_input::<Zt, Lc, _>("verify", vp.num_vars, batch_size, &[], &[point_f])?;

        let num_rows = vp.num_rows;
        let row_len = vp.linear_code.row_len();

        // TODO Lift q0, q1 back to int and take following dot products on ints instead
        // of MBSInnerProduct in field (see combined_row)
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

        let b: Vec<F> = transcript.read_field_elements(num_rows)?;

        // Check 1: <q_0, b> == eval_f
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

        // Check 2: <w, q_1> == <s, b>
        // Ensures b and w are derived from the same underlying rows w'_j.
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
            return Err(ZipError::InvalidPcsOpen("Coherence failure".into()));
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

    // Check 3: Enc(w)[col] == sum_i( sum_j( s_j * <v_ij[col], alphas_i> ) )
    // For each poly i and row j, takes the random linear combination
    // <v_ij[col], alphas_i> of the Cw column entry to CombR,
    // combines rows with coefficients s, sums across polys.
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
        FromWithConfig, IntoWithConfig, PrimeField, crypto_bigint_int::Int,
        crypto_bigint_monty::MontyField,
    };
    use itertools::Itertools;
    use num_traits::{ConstOne, ConstZero, Zero};
    use rand::prelude::*;
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

    type F = MontyField<K>;

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
            let (pp, comm, point_f, eval_f, mut transcript) =
                setup_full_protocol::<F, N, K, M>(num_vars);
            let (field_cfg, projecting_element) =
                get_field_and_projecting_element::<Zt, F>(&mut transcript.fs_transcript);

            let result = TestZip::verify::<_, CHECKED>(
                &mut transcript,
                &pp,
                &comm,
                &field_cfg,
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
            let tampered = eval_f + F::one_with_cfg(&field_cfg);

            let result = TestZip::verify::<_, CHECKED>(
                &mut transcript,
                &pp,
                &comm,
                &field_cfg,
                &point_f,
                &tampered,
            );

            assert!(result.is_err());
        }

        {
            let (pp, comm, point_f, eval_f, mut transcript) =
                setup_full_protocol_poly::<F, N, K, M, DEGREE_PLUS_ONE>(num_vars);
            let (field_cfg, projecting_element) =
                get_field_and_projecting_element::<PolyZt, F>(&mut transcript.fs_transcript);
            let tampered = eval_f + F::one_with_cfg(&field_cfg);

            let result = TestPolyZip::verify::<_, CHECKED>(
                &mut transcript,
                &pp,
                &comm,
                &field_cfg,
                &point_f,
                &tampered,
            );

            assert!(result.is_err());
        }
    }

    #[test]
    fn verification_fails_with_tampered_proof() {
        fn tamper(mut proof: PcsVerifierTranscript) -> PcsVerifierTranscript {
            // Byte 0 is the 1-byte LENGTH_NUM_BYTES prefix for b field elements.
            // Flip byte 1 (first byte of the first b element's VALUE) instead.
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

            let (_, comm_poly2) = TestZip::commit_single(&pp, &poly2).unwrap();

            let result = TestZip::verify::<_, CHECKED>(
                &mut transcript,
                &pp,
                &comm_poly2,
                &field_cfg,
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
            let (_, comm_poly2) = TestPolyZip::commit_single(&pp, &poly2).unwrap();

            let result = TestPolyZip::verify::<_, CHECKED>(
                &mut transcript,
                &pp,
                &comm_poly2,
                &field_cfg,
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

        let (hint, comm) = TestZip::commit_single(&pp, &mle1).unwrap();

        let mle2: DenseMultilinearExtension<_> = (20..=35).map(Int::from).collect();

        let point: Vec<<Zt as ZipTypes>::Pt> =
            (0..num_vars).map(|i| Int::from(i as i32 + 2)).collect();

        let mut prover_transcript = PcsProverTranscript::new_from_commitment(&comm).unwrap();
        let (field_cfg, projecting_element) =
            get_field_and_projecting_element::<Zt, F>(&mut prover_transcript.fs_transcript);

        let _eval_f = TestZip::prove_single::<F, CHECKED>(
            &mut prover_transcript,
            &pp,
            &mle2,
            &point,
            &hint,
            &field_cfg,
        )
        .unwrap();

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
            &point_f,
            &eval_mle1_f,
        );
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

        let point: Vec<<Zt as ZipTypes>::Pt> =
            (0..num_vars).map(|i| Int::from(i as i32 + 2)).collect();

        let mut prover_transcript = PcsProverTranscript::new_from_commitment(&comm).unwrap();
        let (field_cfg, projecting_element) =
            get_field_and_projecting_element::<Zt, F>(&mut prover_transcript.fs_transcript);

        let eval_f = TestZip::prove_single::<F, CHECKED>(
            &mut prover_transcript,
            &pp,
            &mle,
            &point,
            &corrupted_hint,
            &field_cfg,
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
            &point_f,
            &eval_f,
        );

        assert!(verification_result.is_err());
    }

    #[test]
    fn verification_fails_due_to_incorrect_evaluation() {
        let num_vars = 4;
        let (pp, mle) = setup_test_params(num_vars);

        let (hint, comm) = TestZip::commit_single(&pp, &mle).unwrap();

        let point: Vec<<Zt as ZipTypes>::Pt> =
            (0..num_vars).map(|i| Int::from(i as i32 + 2)).collect();

        let mut prover_transcript = PcsProverTranscript::new_from_commitment(&comm).unwrap();
        let (field_cfg, projecting_element) =
            get_field_and_projecting_element::<Zt, F>(&mut prover_transcript.fs_transcript);

        let eval_f = TestZip::prove_single::<F, CHECKED>(
            &mut prover_transcript,
            &pp,
            &mle,
            &point,
            &hint,
            &field_cfg,
        )
        .unwrap();

        let incorrect_eval_f = eval_f + F::one_with_cfg(&field_cfg);
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
            &point_f,
            &incorrect_eval_f, // Use the wrong evaluation here
        );

        assert!(verification_result.is_err());
    }

    #[test]
    fn verification_fails_if_proximity_check_is_invalid() {
        let poly_size = 8;

        let linear_code = C::new(poly_size);
        let pp = TestZip::setup(poly_size, linear_code);

        let mle: DenseMultilinearExtension<_> = (0..poly_size as i32)
            .map(<Zt as ZipTypes>::Eval::from)
            .collect();

        let (hint, comm) = TestZip::commit_single(&pp, &mle).expect("commit should succeed");

        let point = [0, 0, 0]
            .into_iter()
            .map(Int::<1>::from)
            .collect::<Vec<_>>();

        let mut prover_transcript = PcsProverTranscript::new_from_commitment(&comm).unwrap();
        let (field_cfg, projecting_element) =
            get_field_and_projecting_element::<Zt, F>(&mut prover_transcript.fs_transcript);

        let eval_f = TestZip::prove_single::<F, CHECKED>(
            &mut prover_transcript,
            &pp,
            &mle,
            &point,
            &hint,
            &field_cfg,
        )
        .unwrap();

        let point_f: Vec<F> = point.iter().map(|v| v.into_with_cfg(&field_cfg)).collect();

        // New transcript layout: [b field elems] [combined_row] [column openings...]
        // To trigger "Proximity failure", corrupt a column value (past b +
        // combined_row).
        let row_len = pp.linear_code.row_len();
        let num_bytes_f = eval_f.inner().get_num_bytes();
        let b_section_size = 1 + pp.num_rows * 2 * num_bytes_f;
        let bytes_per_comb_r = M * size_of::<crypto_bigint::Word>();
        let combined_row_size = row_len * bytes_per_comb_r;
        let column_values_start = b_section_size + combined_row_size;
        let bytes_per_cw = K * size_of::<crypto_bigint::Word>();

        let mut verifier_transcript = prover_transcript.into_verification_transcript();
        assert!(
            column_values_start + bytes_per_cw <= verifier_transcript.stream.get_ref().len(),
            "proof too small to tamper column values"
        );

        let flip_at = column_values_start + bytes_per_cw / 2;
        verifier_transcript.stream.get_mut()[flip_at] ^= 0x01;

        verifier_transcript.fs_transcript.absorb_slice(&comm.root);
        let (field_cfg, projecting_element) =
            get_field_and_projecting_element::<Zt, F>(&mut verifier_transcript.fs_transcript);

        let res = TestZip::verify::<_, CHECKED>(
            &mut verifier_transcript,
            &pp,
            &comm,
            &field_cfg,
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
    fn verification_fails_if_evaluation_consistency_check_is_invalid() {
        let poly_size = 8;
        let linear_code = C::new(poly_size);
        let pp = TestZip::setup(poly_size, linear_code);

        let mle: DenseMultilinearExtension<_> =
            (0..poly_size as i32).map(Int::<INT_LIMBS>::from).collect();

        let (hint, comm) = TestZip::commit_single(&pp, &mle).expect("commit should succeed");

        let point: Vec<<Zt as ZipTypes>::Pt> = [0, 0, 0].into_iter().map(Int::from).collect_vec();

        let mut prover_transcript = PcsProverTranscript::new_from_commitment(&comm).unwrap();
        let (field_cfg, projecting_element) =
            get_field_and_projecting_element::<Zt, F>(&mut prover_transcript.fs_transcript);

        let eval_f = TestZip::prove_single::<F, CHECKED>(
            &mut prover_transcript,
            &pp,
            &mle,
            &point,
            &hint,
            &field_cfg,
        )
        .unwrap();

        let point_f: Vec<F> = point.iter().map(|v| v.into_with_cfg(&field_cfg)).collect();

        // New transcript starts with b field elements: [1-byte prefix][modulus|value
        // per elem]. Flip a byte inside the first b element's VALUE to corrupt
        // eval consistency.
        let num_bytes_f_mod = eval_f.modulus().get_num_bytes();
        let num_bytes_f_val = eval_f.inner().get_num_bytes();
        let flip_at = 1 + num_bytes_f_mod + num_bytes_f_val / 4;

        let mut verifier_transcript = prover_transcript.into_verification_transcript();
        verifier_transcript.fs_transcript.absorb_slice(&comm.root);
        let _ = get_field_and_projecting_element::<Zt, F>(&mut verifier_transcript.fs_transcript);
        assert!(
            flip_at < verifier_transcript.stream.get_ref().len(),
            "proof too small to tamper b section"
        );
        verifier_transcript.stream.get_mut()[flip_at] ^= 0x01;

        let res = TestZip::verify::<_, CHECKED>(
            &mut verifier_transcript,
            &pp,
            &comm,
            &field_cfg,
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

        let (hint, comm) = TestZip::commit_single(&pp, &mle).expect("commit should succeed");

        let point: Vec<<Zt as ZipTypes>::Pt> = [0, 0, 0].into_iter().map(Int::from).collect_vec();

        let mut prover_transcript = PcsProverTranscript::new_from_commitment(&comm).unwrap();
        let (field_cfg, projecting_element) =
            get_field_and_projecting_element::<Zt, F>(&mut prover_transcript.fs_transcript);

        let eval_f = TestZip::prove_single::<F, CHECKED>(
            &mut prover_transcript,
            &pp,
            &mle,
            &point,
            &hint,
            &field_cfg,
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

        let (hint, comm) = TestZip::commit_single(&pp, &mle).expect("commit should succeed");

        let point: Vec<<Zt as ZipTypes>::Pt> = vec![Int::ZERO; num_vars];

        let mut prover_transcript = PcsProverTranscript::new_from_commitment(&comm).unwrap();
        let (field_cfg, projecting_element) =
            get_field_and_projecting_element::<Zt, F>(&mut prover_transcript.fs_transcript);

        let eval_f = TestZip::prove_single::<F, CHECKED>(
            &mut prover_transcript,
            &pp,
            &mle,
            &point,
            &hint,
            &field_cfg,
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

        let (hint, comm) = TestZip::commit_single(&pp, &poly).unwrap();

        let mut point = vec![<Zt as ZipTypes>::Pt::ZERO; num_vars];
        point[0] = <Zt as ZipTypes>::Pt::ONE;

        let mut prover_transcript = PcsProverTranscript::new_from_commitment(&comm).unwrap();
        let (field_cfg, projecting_element) =
            get_field_and_projecting_element::<Zt, F>(&mut prover_transcript.fs_transcript);

        let eval_f = TestZip::prove_single::<F, CHECKED>(
            &mut prover_transcript,
            &pp,
            &poly,
            &point,
            &hint,
            &field_cfg,
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

        let (hint, comm) = TestZip::commit_single(&pp, &poly).unwrap();

        let point: Vec<<Zt as ZipTypes>::Pt> = vec![Int::from(1), Int::from(2)];

        let mut prover_transcript = PcsProverTranscript::new_from_commitment(&comm).unwrap();
        let (field_cfg, projecting_element) =
            get_field_and_projecting_element::<Zt, F>(&mut prover_transcript.fs_transcript);

        let eval_f = TestZip::prove_single::<F, CHECKED>(
            &mut prover_transcript,
            &pp,
            &poly,
            &point,
            &hint,
            &field_cfg,
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
            &point_f,
            &eval_f,
        );

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

        let point: Vec<<Zt as ZipTypes>::Pt> = [0, 0, 0].into_iter().map(Int::from).collect_vec();

        let mut prover_transcript = PcsProverTranscript::new_from_commitment(&comm).unwrap();
        let (field_cfg, projecting_element) =
            get_field_and_projecting_element::<Zt, F>(&mut prover_transcript.fs_transcript);

        let eval_f = TestZip::prove_single::<F, CHECKED>(
            &mut prover_transcript,
            &pp,
            &mle,
            &point,
            &hint,
            &field_cfg,
        )
        .unwrap();

        let point_f: Vec<F> = point.iter().map(|v| v.into_with_cfg(&field_cfg)).collect();

        // Offset past b section to reach combined_row (CombR = Int<M>).
        let num_bytes_f = eval_f.inner().get_num_bytes();
        let b_section_size = 1 + pp.num_rows * 2 * num_bytes_f;
        let bytes_to_corrupt = M * size_of::<crypto_bigint::Word>();

        let mut verifier_transcript = prover_transcript.into_verification_transcript();
        assert!(
            b_section_size + bytes_to_corrupt <= verifier_transcript.stream.get_ref().len(),
            "proof too small to tamper combined_row"
        );

        for b in &mut verifier_transcript.stream.get_mut()
            [b_section_size..b_section_size + bytes_to_corrupt]
        {
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
            let (hint, commitment) = TestZip::commit_single(&pp, &mle).expect("commit");

            let point = vec![1i64; P].iter().map(|v| v.into()).collect_vec();

            let mut prover_transcript =
                PcsProverTranscript::new_from_commitment(&commitment).unwrap();
            let (field_cfg, projecting_element) =
                get_field_and_projecting_element::<Zt, F>(&mut prover_transcript.fs_transcript);

            let eval_f = TestZip::prove_single::<F, CHECKED>(
                &mut prover_transcript,
                &pp,
                &mle,
                &point,
                &hint,
                &field_cfg,
            )
            .unwrap();

            let point_f: Vec<F> = point.iter().map(|v| v.into_with_cfg(&field_cfg)).collect();

            let mut verifier_transcript = prover_transcript.into_verification_transcript();
            verifier_transcript
                .fs_transcript
                .absorb_slice(&commitment.root);
            let (field_cfg, projecting_element) =
                get_field_and_projecting_element::<Zt, F>(&mut verifier_transcript.fs_transcript);

            let zero_f = F::zero_with_cfg(&field_cfg);
            let mle_f = DenseMultilinearExtension::from_evaluations_vec(
                P,
                mle.iter().map(|c| c.into_with_cfg(&field_cfg)).collect(),
                zero_f.clone(),
            );
            let expected_eval_f = mle_f.evaluate(&point_f, zero_f).unwrap();
            assert_eq!(eval_f, expected_eval_f, "prover returned wrong eval");

            TestZip::verify::<_, CHECKED>(
                &mut verifier_transcript,
                &pp,
                &commitment,
                &field_cfg,
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
            let (hint, comm) = TestPolyZip::commit_single(&pp, &mle).expect("commit");

            let point = vec![1i64; P].iter().map(|v| (*v).into()).collect_vec();

            let mut prover_transcript = PcsProverTranscript::new_from_commitment(&comm).unwrap();
            let (field_cfg, projecting_element) =
                get_field_and_projecting_element::<PolyZt, F>(&mut prover_transcript.fs_transcript);

            let eval_f = TestPolyZip::prove_single::<F, CHECKED>(
                &mut prover_transcript,
                &pp,
                &mle,
                &point,
                &hint,
                &field_cfg,
            )
            .unwrap();

            let point_f: Vec<F> = point.iter().map(|v| v.into_with_cfg(&field_cfg)).collect();

            let mut verifier_transcript = prover_transcript.into_verification_transcript();
            verifier_transcript.fs_transcript.absorb_slice(&comm.root);
            let (field_cfg, projecting_element) = get_field_and_projecting_element::<PolyZt, F>(
                &mut verifier_transcript.fs_transcript,
            );

            // Verifier replays verification from the same proof (also like the bench)
            TestPolyZip::verify::<_, CHECKED>(
                &mut verifier_transcript,
                &pp,
                &comm,
                &field_cfg,
                &point_f,
                &eval_f,
            )
            .expect("verify");
        }

        inner::<19>();
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
        let point: Vec<<Zt as ZipTypes>::Pt> =
            (0..num_vars).map(|i| Int::from(i as i32 + 2)).collect();

        let mut prover_transcript = PcsProverTranscript::new_from_commitment(&comm).unwrap();
        let (field_cfg, projecting_element) =
            get_field_and_projecting_element::<PolyZt, F>(&mut prover_transcript.fs_transcript);

        let eval_f = TestZip::prove::<F, CHECKED>(
            &mut prover_transcript,
            &pp,
            &polys,
            &point,
            &hint,
            &field_cfg,
        )
        .unwrap();

        let point_f: Vec<F> = point.iter().map(|v| v.into_with_cfg(&field_cfg)).collect();

        let mut verifier_transcript = prover_transcript.into_verification_transcript();
        verifier_transcript.fs_transcript.absorb_slice(&comm.root);
        let (field_cfg, projecting_element) =
            get_field_and_projecting_element::<PolyZt, F>(&mut verifier_transcript.fs_transcript);

        let res = TestZip::verify::<_, CHECKED>(
            &mut verifier_transcript,
            &pp,
            &comm,
            &field_cfg,
            &point_f,
            &eval_f,
        );
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
        let point: Vec<<Zt as ZipTypes>::Pt> = (0..num_vars).map(|i| Int::from(i + 2)).collect();

        let mut prover_transcript = PcsProverTranscript::new_from_commitment(&comm).unwrap();
        let (field_cfg, projecting_element) =
            get_field_and_projecting_element::<PolyZt, F>(&mut prover_transcript.fs_transcript);

        let eval_f = TestZip::prove::<F, CHECKED>(
            &mut prover_transcript,
            &pp,
            &polys,
            &point,
            &hint,
            &field_cfg,
        )
        .unwrap();
        let tampered_eval = eval_f + F::one_with_cfg(&field_cfg);

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
            &point_f,
            &tampered_eval,
        );
        assert!(res.is_err(), "Should fail when eval is tampered");
    }
}
