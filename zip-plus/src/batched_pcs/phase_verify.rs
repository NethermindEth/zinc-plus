use crate::{
    ZipError,
    code::LinearCode,
    merkle::MtHash,
    pcs::{
        structs::ZipTypes,
        utils::point_to_tensor,
    },
    pcs_transcript::PcsTranscript,
};
use crypto_primitives::{FromPrimitiveWithConfig, FromWithConfig, IntoWithConfig};
use itertools::Itertools;
use num_traits::{CheckedAdd, ConstOne, ConstZero, Zero};
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

use super::structs::{
    BatchedZipPlus, BatchedZipPlusCommitment, BatchedZipPlusParams, BatchedZipPlusProof,
};

impl<Zt: ZipTypes, Lc: LinearCode<Zt>> BatchedZipPlus<Zt, Lc> {
    /// Verifies a batched ZipPlus proof.
    ///
    /// All polynomials are evaluated at the same point. Verification checks
    /// that the committed batch is consistent with the proof-internal
    /// evaluation scalars and batched row.
    ///
    /// # Parameters
    /// - `vp`: Public (verifier) parameters.
    /// - `comm`: The batched commitment (single Merkle root + batch size).
    /// - `point`: The shared evaluation point (raw, not yet in field).
    /// - `proof`: The batched proof.
    pub fn verify<F, const CHECK_FOR_OVERFLOW: bool>(
        vp: &BatchedZipPlusParams<Zt, Lc>,
        comm: &BatchedZipPlusCommitment,
        point: &[Zt::Pt],
        proof: &BatchedZipPlusProof,
    ) -> Result<(), ZipError>
    where
        F: FromPrimitiveWithConfig
            + FromRef<F>
            + for<'a> FromWithConfig<&'a Zt::Chal>
            + for<'a> FromWithConfig<&'a Zt::Pt>
            + for<'a> MulByScalar<&'a F>,
        F::Inner: FromRef<Zt::Fmod> + Transcribable,
        Zt::Cw: ProjectableToField<F>,
    {
        let batch_size = comm.batch_size;

        let mut transcript: PcsTranscript = proof.clone().into();

        // columns_opened: Vec<(column_idx, Vec<Vec<Zt::Cw>>)>
        // For each opened column, we have a Vec of column values per polynomial.
        let columns_opened = Self::verify_testing::<CHECK_FOR_OVERFLOW>(
            vp,
            &comm.root,
            batch_size,
            &mut transcript,
        )?;

        let field_cfg = transcript
            .fs_transcript
            .get_random_field_cfg::<F, Zt::Fmod, Zt::PrimeTest>();
        let projecting_element: Zt::Chal = transcript.fs_transcript.get_challenge();
        let projecting_element: F = (&projecting_element).into_with_cfg(&field_cfg);

        // Sample batching challenge β (must match prover's β)
        let beta: Zt::Chal = transcript.fs_transcript.get_challenge();
        let beta: F = (&beta).into_with_cfg(&field_cfg);

        // Convert raw point to field elements using the PCS's own field_cfg
        let point_f: Vec<F> = point
            .iter()
            .map(|v| v.into_with_cfg(&field_cfg))
            .collect();

        Self::verify_evaluation(
            vp,
            &point_f,
            batch_size,
            &columns_opened,
            &mut transcript,
            projecting_element,
            beta,
            &field_cfg,
        )?;

        Ok(())
    }

    /// Verifies the testing phase of a batched proof.
    ///
    /// Batching: per-polynomial alpha challenges project each polynomial's
    /// BPoly entries into integers. A single set of row-combination
    /// coefficients is shared. The prover sends a single `combined_row`
    /// that is the sum of all per-polynomial combined rows. The verifier
    /// encodes that single row and checks proximity across all polynomials
    /// simultaneously.
    ///
    /// Returns a vector of opened columns, where each entry is
    /// `(column_idx, per_poly_column_values)` with `per_poly_column_values`
    /// being a `Vec<Vec<Zt::Cw>>` indexed by polynomial index.
    #[allow(clippy::arithmetic_side_effects, clippy::type_complexity)]
    fn verify_testing<const CHECK_FOR_OVERFLOW: bool>(
        vp: &BatchedZipPlusParams<Zt, Lc>,
        root: &MtHash,
        batch_size: usize,
        transcript: &mut PcsTranscript,
    ) -> Result<Vec<(usize, Vec<Vec<Zt::Cw>>)>, ZipError> {
        // Phase 1: Sample per-polynomial alpha challenges
        let all_alphas: Vec<Vec<Zt::Chal>> = (0..batch_size)
            .map(|_| {
                if Zt::Comb::DEGREE_BOUND.is_zero() {
                    vec![Zt::Chal::ONE]
                } else {
                    transcript
                        .fs_transcript
                        .get_challenges::<Zt::Chal>(Zt::Comb::DEGREE_BOUND + 1)
                }
            })
            .collect();

        // Phase 2: Sample a single set of row-combination coefficients (shared)
        let coeffs = if vp.num_rows > 1 {
            transcript.fs_transcript.get_challenges(vp.num_rows)
        } else {
            vec![Zt::Chal::ONE]
        };

        // Phase 3: Read the single combined row (sum over all polynomials)
        let combined_row: Vec<Zt::CombR> =
            transcript.read_const_many(vp.linear_code.row_len())?;

        // Phase 4: Encode the single combined row
        let encoded_combined_row: Vec<Zt::CombR> =
            vp.linear_code.encode_wide(&combined_row);

        // Read column openings from transcript sequentially
        let columns_and_proofs: Vec<_> = (0..Zt::NUM_COLUMN_OPENINGS)
            .map(|_| -> Result<_, ZipError> {
                let column_idx =
                    transcript.squeeze_challenge_idx(vp.linear_code.codeword_len());

                // Read column values for each polynomial in order
                let per_poly_column_values: Vec<Vec<Zt::Cw>> = (0..batch_size)
                    .map(|_| transcript.read_const_many(vp.num_rows))
                    .try_collect()?;

                let proof = transcript.read_merkle_proof().map_err(|e| {
                    ZipError::InvalidPcsOpen(format!(
                        "Failed to read batched Merkle proof: {e}"
                    ))
                })?;

                Ok((column_idx, per_poly_column_values, proof))
            })
            .try_collect()?;

        let columns_opened: Vec<(usize, Vec<Vec<Zt::Cw>>)> =
            cfg_into_iter!(columns_and_proofs)
                .map(
                    |(column_idx, per_poly_column_values, proof)| -> Result<_, ZipError> {
                        // Verify proximity across ALL polynomials.
                        // Sum each polynomial's alpha-projected, row-combined
                        // column contribution and compare to the encoded
                        // combined row.
                        Self::verify_batched_column_testing::<CHECK_FOR_OVERFLOW>(
                            &all_alphas,
                            &coeffs,
                            &encoded_combined_row,
                            &per_poly_column_values,
                            column_idx,
                            vp.num_rows,
                        )?;

                        // Verify Merkle proof against the shared root.
                        // The leaf was hashed from the concatenation of all
                        // polynomials' column values.
                        let all_column_values: Vec<Zt::Cw> = per_poly_column_values
                            .iter()
                            .flat_map(|v| v.iter().cloned())
                            .collect();

                        proof
                            .verify(root, &all_column_values, column_idx)
                            .map_err(|e| {
                                ZipError::InvalidPcsOpen(format!(
                                    "Batched column opening verification failed: {e}"
                                ))
                            })?;

                        Ok((column_idx, per_poly_column_values))
                    },
                )
                .collect::<Result<_, _>>()?;

        Ok(columns_opened)
    }

    /// Verifies a single column's proximity for a batch of polynomials.
    ///
    /// For each polynomial, alpha-projects its column entries and combines
    /// rows with the shared `coeffs`. The per-polynomial results are summed
    /// and compared to `encoded_combined_row[column]`.
    #[allow(clippy::arithmetic_side_effects)]
    fn verify_batched_column_testing<const CHECK_FOR_OVERFLOW: bool>(
        all_alphas: &[Vec<Zt::Chal>],
        coeffs: &[Zt::Chal],
        encoded_combined_row: &[Zt::CombR],
        per_poly_column_values: &[Vec<Zt::Cw>],
        column: usize,
        num_rows: usize,
    ) -> Result<(), ZipError> {
        let mut total: Zt::CombR = Zt::CombR::ZERO;

        for (poly_column, alphas) in per_poly_column_values.iter().zip(all_alphas.iter()) {
            let column_entries_comb: Zt::CombR = if num_rows > 1 {
                let column_entries: Vec<_> = poly_column
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
                    &Zt::Comb::from_ref(&poly_column[0]),
                    alphas,
                    Zt::CombR::ZERO,
                )?
            };
            total = total.checked_add(&column_entries_comb)
                .expect("Overflow when summing per-polynomial column contributions: CombR is too narrow for batch size");
        }

        if total != encoded_combined_row[column] {
            return Err(ZipError::InvalidPcsOpen("Proximity failure".into()));
        }
        Ok(())
    }

    /// Verifies the evaluation phase for a batched proof.
    ///
    /// The prover writes a single batched row `r̂ = Σ β^i · r_i` and
    /// per-polynomial evaluation scalars to the proof transcript. The verifier
    /// reads both from the proof (in the correct field context) and checks:
    ///
    /// 1. `⟨r̂, q_1⟩ == Σ β^i · v_i` (batched evaluation consistency).
    /// 2. For each opened column `j`: the β-weighted sum of per-polynomial
    ///    field-projected, q_0-combined column values equals `encode(r̂)[j]`.
    #[allow(clippy::too_many_arguments)]
    fn verify_evaluation<F>(
        vp: &BatchedZipPlusParams<Zt, Lc>,
        point_f: &[F],
        batch_size: usize,
        columns_opened: &[(usize, Vec<Vec<Zt::Cw>>)],
        transcript: &mut PcsTranscript,
        projecting_element: F,
        beta: F,
        field_cfg: &F::Config,
    ) -> Result<(), ZipError>
    where
        F: FromPrimitiveWithConfig + FromRef<F> + for<'a> MulByScalar<&'a F>,
        F::Inner: FromRef<Zt::Fmod> + Transcribable,
        Zt::Cw: ProjectableToField<F>,
    {
        let (q_0, q_1) = point_to_tensor(vp.num_rows, point_f, field_cfg)?;
        let project = Zt::Cw::prepare_projection(&projecting_element);

        // Read the single batched row from the transcript
        let batched_row =
            transcript.read_field_elements(vp.linear_code.row_len(), field_cfg)?;

        // Read per-polynomial evaluation scalars from the proof transcript.
        // These are in the PCS's field context (same field_cfg as batched_row).
        let proof_evals: Vec<F> =
            transcript.read_field_elements(batch_size, field_cfg)?;

        // Verify batched evaluation consistency: ⟨batched_row, q_1⟩ == Σ β^i · v_i
        // Uses the proof-internal eval scalars (correct field context).
        let batched_eval = MBSInnerProduct::inner_product::<UNCHECKED>(
            &batched_row,
            &q_1,
            F::zero_with_cfg(field_cfg),
        )?;

        let mut expected_batched_eval = F::zero_with_cfg(field_cfg);
        let mut beta_power = F::one_with_cfg(field_cfg);
        for eval in &proof_evals {
            let term = eval
                .mul_by_scalar::<UNCHECKED>(&beta_power)
                .expect("Field multiplication cannot overflow");
            expected_batched_eval += term;
            beta_power = beta_power
                .mul_by_scalar::<UNCHECKED>(&beta)
                .expect("Field multiplication cannot overflow");
        }

        if batched_eval != expected_batched_eval {
            return Err(ZipError::InvalidPcsOpen(
                "Batched evaluation consistency failure".into(),
            ));
        }

        // Encode the batched row under the linear code
        let encoded_batched_row: Vec<F> =
            vp.linear_code.encode_f(&batched_row);

        // Pre-compute β powers for proximity checks
        let mut beta_powers: Vec<F> = Vec::with_capacity(batch_size);
        let mut bp = F::one_with_cfg(field_cfg);
        for _ in 0..batch_size {
            beta_powers.push(bp.clone());
            bp = bp
                .mul_by_scalar::<UNCHECKED>(&beta)
                .expect("Field multiplication cannot overflow");
        }

        // Verify proximity for each opened column across all polynomials
        cfg_iter!(columns_opened).try_for_each(
            |(column_idx, per_poly_column_values)| -> Result<(), ZipError> {
                let mut batched_column_value = F::zero_with_cfg(field_cfg);

                for (poly_idx, column_values) in
                    per_poly_column_values.iter().enumerate()
                {
                    let per_poly_val = if vp.num_rows > 1 {
                        let projected: Vec<F> =
                            column_values.iter().map(&project).collect();
                        MBSInnerProduct::inner_product::<UNCHECKED>(
                            &q_0,
                            &projected,
                            F::zero_with_cfg(field_cfg),
                        )?
                    } else {
                        project(&column_values[0])
                    };
                    let term = per_poly_val
                        .mul_by_scalar::<UNCHECKED>(&beta_powers[poly_idx])
                        .expect("Field multiplication cannot overflow");
                    batched_column_value += term;
                }

                if batched_column_value != encoded_batched_row[*column_idx] {
                    return Err(ZipError::InvalidPcsOpen(
                        "Batched proximity failure in eval phase".into(),
                    ));
                }
                Ok(())
            },
        )?;

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
    use super::*;
    use crate::{
        batched_pcs::structs::BatchedZipPlus,
        pcs::test_utils::*,
    };
    use crypto_bigint::U64;
    use crypto_primitives::IntoWithConfig;
    use crypto_primitives::{
        PrimeField,
        crypto_bigint_boxed_monty::BoxedMontyField, crypto_bigint_int::Int,
    };
    use zinc_utils::CHECKED;

    const INT_LIMBS: usize = U64::LIMBS;
    const N: usize = INT_LIMBS;
    const K: usize = INT_LIMBS * 4;
    const M: usize = INT_LIMBS * 8;

    type F = BoxedMontyField;
    type Zt = TestZipTypes<N, K, M>;
    type C = crate::code::raa_sign_flip::RaaSignFlippingCode<Zt, TestRaaConfig, 4>;
    type TestBatchedZip = BatchedZipPlus<Zt, C>;

    /// Helper: run full batched protocol and return all outputs for verification
    fn setup_batched_protocol(
        num_vars: usize,
        polys: Vec<zinc_poly::mle::DenseMultilinearExtension<<Zt as ZipTypes>::Eval>>,
    ) -> (
        BatchedZipPlusParams<Zt, C>,
        BatchedZipPlusCommitment,
        Vec<<Zt as ZipTypes>::Pt>,
        BatchedZipPlusProof,
    ) {
        let (pp, _) = setup_test_params::<N, K, M>(num_vars);

        let (hint, comm) = TestBatchedZip::commit(&pp, &polys).unwrap();

        let test_transcript =
            TestBatchedZip::test::<CHECKED>(&pp, &polys, &hint).unwrap();

        // Evaluation point shared across all polynomials
        let point: Vec<<Zt as ZipTypes>::Pt> =
            (0..num_vars).map(|i| Int::from(i as i32 + 2)).collect();

        let (_evals_f, proof) =
            TestBatchedZip::evaluate::<F, CHECKED>(&pp, &polys, &point, test_transcript)
                .unwrap();

        (pp, comm, point, proof)
    }

    #[test]
    fn successful_batched_verification_single_poly() {
        let num_vars = 4;
        let poly: zinc_poly::mle::DenseMultilinearExtension<_> =
            (1..=16).map(Int::from).collect();

        let (pp, comm, point, proof) =
            setup_batched_protocol(num_vars, vec![poly]);

        let result =
            TestBatchedZip::verify::<F, CHECKED>(&pp, &comm, &point, &proof);
        assert!(result.is_ok(), "Single-poly batched verification failed: {result:?}");
    }

    #[test]
    fn successful_batched_verification_multiple_polys() {
        let num_vars = 4;
        let poly1: zinc_poly::mle::DenseMultilinearExtension<_> =
            (1..=16).map(Int::from).collect();
        let poly2: zinc_poly::mle::DenseMultilinearExtension<_> =
            (17..=32).map(Int::from).collect();

        let (pp, comm, point, proof) =
            setup_batched_protocol(num_vars, vec![poly1, poly2]);

        let result =
            TestBatchedZip::verify::<F, CHECKED>(&pp, &comm, &point, &proof);
        assert!(result.is_ok(), "Multi-poly batched verification failed: {result:?}");
    }

    #[test]
    fn batched_verification_fails_with_tampered_proof() {
        let num_vars = 4;
        let poly1: zinc_poly::mle::DenseMultilinearExtension<_> =
            (1..=16).map(Int::from).collect();
        let poly2: zinc_poly::mle::DenseMultilinearExtension<_> =
            (17..=32).map(Int::from).collect();

        let (pp, comm, point, proof) =
            setup_batched_protocol(num_vars, vec![poly1, poly2]);

        // Tamper with the proof bytes (flip a byte in the eval phase data).
        let mut proof_bytes: Vec<u8> = {
            let tx: PcsTranscript = proof.into();
            tx.stream.into_inner()
        };
        // Flip a byte near the end (in the eval phase region)
        let idx = proof_bytes.len() - 10;
        proof_bytes[idx] ^= 0xFF;
        let tampered_proof: super::BatchedZipPlusProof = {
            let tx = PcsTranscript {
                fs_transcript: zinc_transcript::KeccakTranscript::default(),
                stream: std::io::Cursor::new(proof_bytes),
            };
            tx.into()
        };

        let result =
            TestBatchedZip::verify::<F, CHECKED>(&pp, &comm, &point, &tampered_proof);
        assert!(result.is_err(), "Verification should fail with tampered proof");
    }

    #[test]
    fn batched_verification_five_polynomials() {
        let num_vars = 4;
        let polys: Vec<zinc_poly::mle::DenseMultilinearExtension<_>> = (0..5)
            .map(|offset| {
                let start = offset * 16 + 1;
                (start..start + 16).map(Int::from).collect()
            })
            .collect();

        let (pp, comm, point, proof) =
            setup_batched_protocol(num_vars, polys);

        let result =
            TestBatchedZip::verify::<F, CHECKED>(&pp, &comm, &point, &proof);
        assert!(result.is_ok(), "5-poly batched verification failed: {result:?}");
    }
}
