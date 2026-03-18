use crate::{
    ZipError,
    code::LinearCode,
    combine_rows,
    pcs::{
        structs::{ZipPlus, ZipPlusHint, ZipPlusParams, ZipTypes},
        utils::{point_to_tensor, validate_input},
    },
    pcs_transcript::PcsProverTranscript,
};
use crypto_primitives::{FromWithConfig, IntoWithConfig, PrimeField};
use itertools::Itertools;
use num_traits::{ConstOne, ConstZero, Zero};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use zinc_poly::{Polynomial, mle::DenseMultilinearExtension};
use zinc_transcript::traits::{Transcribable, Transcript};
use zinc_utils::{
    UNCHECKED, cfg_chunks, cfg_iter, cfg_iter_mut,
    from_ref::FromRef,
    inner_product::{InnerProduct, MBSInnerProduct},
    mul_by_scalar::MulByScalar,
};

impl<Zt: ZipTypes, Lc: LinearCode<Zt>> ZipPlus<Zt, Lc> {
    /// Generates an opening proof for one or more committed multilinear
    /// polynomials at an evaluation point, using the Zip+ protocol.
    ///
    /// This replaces the old two-phase (test + evaluate) approach with a single
    /// merged phase. The key idea: alpha-projection (Eval → CombR) is used for
    /// *both* the proximity argument and the evaluation claim, eliminating the
    /// separate field-domain projection via `projecting_element` γ.
    ///
    /// # Algorithm
    /// 1. Computes points: `(q_0, q_1) = point_to_tensor(point)` where `q_0`
    ///    (length `num_rows`) combines rows and `q_1` (length `row_len`)
    ///    combines columns.
    /// 2. Per polynomial, samples random challenges `alphas` (`[α_0, …, α_d]`).
    ///    For each decoded row `w_j` takes the inner product `<entry, alphas>`
    ///    of every entry in the row, producing `w'_j` — a row of `CombR`
    ///    integers.
    /// 3. Computes `b` (length `num_rows`), accumulated across all polys: `b_j
    ///    += <w'_j, q_1>` for each row `j`.
    /// 4. Writes `b` to the transcript and computes `eval = <q_0, b>`.
    /// 5. Samples combination coefficients `betas` (or hardcodes `[1]` when
    ///    `num_rows == 1`) and computes `combined_row` (CombR, length
    ///    `row_len`) = `sum_i(sum_j(s_j * w'_ij))`, accumulated across all
    ///    polynomials
    /// 6. Writes `combined_row` to the transcript.
    /// 7. Opens `NUM_COLUMN_OPENINGS` Merkle columns: for each, squeezes a
    ///    column index, writes per-polynomial column values (Cw entries), and
    ///    appends the Merkle proof.
    ///
    /// # Transcript layout
    /// ```text
    /// [field_cfg sampled]
    /// [per-poly alphas sampled]
    /// [b written as F elements]
    /// [coeffs s sampled (or hardcoded [1])]
    /// [combined_row written as CombR]
    /// [column openings: idx, per-poly column values, merkle proof] × NUM_COLUMN_OPENINGS
    /// ```
    ///
    /// # Parameters
    /// - `pp`: Public parameters containing `num_vars`, `num_rows`, and the
    ///   linear code configuration.
    /// - `polys`: Slice of multilinear polynomials (batch). All must have
    ///   `num_vars` variables matching `pp`.
    /// - `point`: The evaluation point (in `Zt::Pt` coordinates, length
    ///   `num_vars`).
    /// - `commit_hint`: The `ZipPlusHint` returned by `commit`, containing
    ///   per-polynomial codeword matrices and the shared Merkle tree.
    ///
    /// # Returns
    /// A `Result` containing:
    /// - `F`: The combined evaluation `<q_0, b>`, which equals
    ///   `sum_i(alpha_projected_eval_i(point))` across all batched polys.
    /// - `ZipPlusProof`: The serialized transcript (b, combined_row, column
    ///   openings + Merkle proofs) for the verifier.
    ///
    /// # Errors
    /// - Returns `ZipError::InvalidPcsParam` if any polynomial has more
    ///   variables than `pp` supports.
    /// - Returns `ZipError::OverflowError` (when `CHECK_FOR_OVERFLOW` is true)
    ///   if intermediate CombR sums exceed the integer precision.
    pub fn prove<F, const CHECK_FOR_OVERFLOW: bool>(
        transcript: &mut PcsProverTranscript,
        pp: &ZipPlusParams<Zt, Lc>,
        polys: &[DenseMultilinearExtension<Zt::Eval>],
        point: &[Zt::Pt],
        commit_hint: &ZipPlusHint<Zt::Cw>,
        field_cfg: &F::Config,
    ) -> Result<F, ZipError>
    where
        F: PrimeField
            + for<'a> FromWithConfig<&'a Zt::CombR>
            + for<'a> FromWithConfig<&'a Zt::Pt>
            + for<'a> MulByScalar<&'a F>
            + FromRef<F>,
        F::Inner: Transcribable,
        F::Modulus: Transcribable,
    {
        let point = point
            .iter()
            .map(|v| v.into_with_cfg(field_cfg))
            .collect::<Vec<F>>();
        Self::prove_f::<F, CHECK_FOR_OVERFLOW>(
            transcript,
            pp,
            polys,
            &point,
            commit_hint,
            field_cfg,
        )
    }

    /// See [`Self::prove`] for details.
    /// This version takes the evaluation point already mapped to the field
    #[allow(clippy::arithmetic_side_effects)]
    pub fn prove_f<F, const CHECK_FOR_OVERFLOW: bool>(
        transcript: &mut PcsProverTranscript,
        pp: &ZipPlusParams<Zt, Lc>,
        polys: &[DenseMultilinearExtension<Zt::Eval>],
        point: &[F],
        commit_hint: &ZipPlusHint<Zt::Cw>,
        field_cfg: &F::Config,
    ) -> Result<F, ZipError>
    where
        F: PrimeField
            + for<'a> FromWithConfig<&'a Zt::CombR>
            + for<'a> MulByScalar<&'a F>
            + FromRef<F>,
        F::Inner: Transcribable,
        F::Modulus: Transcribable,
    {
        let batch_size = polys.len();
        validate_input::<Zt, Lc, _>("prove", pp.num_vars, batch_size, polys, &[point])?;

        let num_rows = pp.num_rows;
        let row_len = pp.linear_code.row_len();

        // TODO Lift q0, q1 back to int and take following dot products on ints instead
        // of MBSInnerProduct in field (see comboned row) We prove evaluations
        // over the field, so integers need to be mapped to field elements first
        let (q_0, q_1) = point_to_tensor(num_rows, point, field_cfg)?;

        let degree_bound = Zt::Comb::DEGREE_BOUND;
        let polys_as_comb_r: Vec<Vec<Zt::CombR>> = polys
            .iter()
            .map(|poly| {
                let alphas = if degree_bound.is_zero() {
                    vec![Zt::Chal::ONE]
                } else {
                    transcript.fs_transcript.get_challenges(degree_bound + 1)
                };

                cfg_iter!(poly.evaluations)
                    .map(|eval| {
                        Zt::EvalDotChal::inner_product::<CHECK_FOR_OVERFLOW>(
                            eval,
                            &alphas,
                            Zt::CombR::ZERO,
                        )
                        .map_err(ZipError::from)
                    })
                    .collect()
            })
            .try_collect()?;

        let zero_f = F::zero_with_cfg(field_cfg);

        // Compute per-polynomial row dot products, then sum across polynomials.
        #[cfg(feature = "parallel")]
        let b = {
            let per_poly_b: Vec<Vec<F>> = polys_as_comb_r
                .par_iter()
                .map(|poly_comb_r| {
                    poly_comb_r
                        .par_chunks(row_len)
                        .map(|row| {
                            let row_f: Vec<F> =
                                row.iter().map(|int| int.into_with_cfg(field_cfg)).collect();
                            MBSInnerProduct::inner_product::<UNCHECKED>(
                                &row_f,
                                &q_1,
                                zero_f.clone(),
                            )
                        })
                        .collect::<Result<Vec<F>, _>>()
                })
                .collect::<Result<_, _>>()?;

            let mut b = vec![zero_f.clone(); num_rows];
            for poly_b in &per_poly_b {
                b.iter_mut().zip(poly_b).for_each(|(a, d)| *a += d);
            }
            b
        };

        #[cfg(not(feature = "parallel"))]
        let b = polys_as_comb_r.iter().try_fold(
            vec![zero_f.clone(); num_rows],
            |mut acc, poly_comb_r| -> Result<_, ZipError> {
                let row_dots: Vec<F> = cfg_chunks!(poly_comb_r, row_len)
                    .map(|row| {
                        let row_f: Vec<F> =
                            row.iter().map(|int| int.into_with_cfg(field_cfg)).collect();
                        MBSInnerProduct::inner_product::<UNCHECKED>(&row_f, &q_1, zero_f.clone())
                    })
                    .collect::<Result<_, _>>()?;

                acc.iter_mut().zip(row_dots).for_each(|(a, d)| *a += d);

                Ok(acc)
            },
        )?;

        transcript.write_field_elements(&b)?;
        // Compute eval = <q_0, b> (inner product in field), <q_2, b> in paper
        // It is safe to use inner_product_unchecked because we're in a field.
        let eval = MBSInnerProduct::inner_product::<UNCHECKED>(&q_0, &b, zero_f.clone())?;

        // combined_row[col] = sum_i( sum_j( s_j * w'_ij[col] ) ) over all polys i, rows
        // j Inner: combine_rows! computes sum_j(s_j * w'_ij) per poly
        // Outer: try_fold accumulates across polys
        let coeffs = if pp.num_rows == 1 {
            vec![Zt::Chal::ONE]
        } else {
            transcript
                .fs_transcript
                .get_challenges::<Zt::Chal>(num_rows)
        };

        #[cfg(feature = "parallel")]
        let combined_row: Vec<Zt::CombR> = {
            let per_poly_rows: Vec<Vec<Zt::CombR>> = polys_as_comb_r
                .par_iter()
                .map(|poly| {
                    Ok::<_, ZipError>(combine_rows!(
                        CHECK_FOR_OVERFLOW,
                        &coeffs,
                        poly.iter(),
                        |eval: &Zt::CombR| Ok::<_, ZipError>(eval.clone()),
                        row_len,
                        Zt::CombR::ZERO
                    ))
                })
                .collect::<Result<_, _>>()?;

            let mut combined = vec![Zt::CombR::ZERO; row_len];
            for row in &per_poly_rows {
                combined.iter_mut().zip(row.iter()).for_each(|(a, r)| {
                    if CHECK_FOR_OVERFLOW {
                        *a = zinc_utils::add!(
                            *a,
                            r,
                            "Addition overflow while summing combined rows across polys"
                        );
                    } else {
                        *a += r;
                    }
                });
            }
            combined
        };

        #[cfg(not(feature = "parallel"))]
        let combined_row: Vec<Zt::CombR> = polys_as_comb_r.iter().try_fold(
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
                        *a = zinc_utils::add!(
                            *a,
                            r,
                            "Addition overflow while summing combined rows across polys"
                        );
                    } else {
                        *a += r;
                    }
                });

                Ok(acc)
            },
        )?;

        transcript.write_const_many(&combined_row)?;
        for _ in 0..Zt::NUM_COLUMN_OPENINGS {
            let column_idx = transcript.squeeze_challenge_idx(pp.linear_code.codeword_len());
            Self::open_merkle_trees_for_column(transcript, commit_hint, column_idx)?;
        }

        Ok(eval)
    }

    /// See [`Self::prove`] for details.
    #[inline(always)]
    pub fn prove_single<F, const CHECK_FOR_OVERFLOW: bool>(
        transcript: &mut PcsProverTranscript,
        pp: &ZipPlusParams<Zt, Lc>,
        poly: &DenseMultilinearExtension<Zt::Eval>,
        point: &[Zt::Pt],
        commit_hint: &ZipPlusHint<Zt::Cw>,
        field_cfg: &F::Config,
    ) -> Result<F, ZipError>
    where
        F: PrimeField
            + for<'a> FromWithConfig<&'a Zt::CombR>
            + for<'a> FromWithConfig<&'a Zt::Chal>
            + for<'a> FromWithConfig<&'a Zt::Pt>
            + for<'a> MulByScalar<&'a F>
            + FromRef<F>,
        F::Inner: Transcribable,
        F::Modulus: FromRef<Zt::Fmod> + Transcribable,
    {
        Self::prove::<F, CHECK_FOR_OVERFLOW>(
            transcript,
            pp,
            std::slice::from_ref(poly),
            point,
            commit_hint,
            field_cfg,
        )
    }

    pub(super) fn open_merkle_trees_for_column(
        transcript: &mut PcsProverTranscript,
        commit_hint: &ZipPlusHint<Zt::Cw>,
        column_idx: usize,
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
            structs::{ZipPlus, ZipPlusHint},
            test_utils::*,
        },
        pcs_transcript::PcsProverTranscript,
    };
    use crypto_bigint::U64;
    use crypto_primitives::{
        IntoWithConfig, crypto_bigint_boxed_monty::BoxedMontyField, crypto_bigint_int::Int,
    };
    use num_traits::{ConstOne, Zero};
    use zinc_poly::mle::DenseMultilinearExtension;
    use zinc_utils::{CHECKED, from_ref::FromRef};

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

    fn test_point(num_vars: usize) -> Vec<Int<INT_LIMBS>> {
        (0..num_vars).map(|i| Int::from(i as i32 + 2)).collect()
    }

    #[test]
    fn prove_succeeds_for_single_poly() {
        let num_vars = 4;
        let (pp, poly) = setup_test_params(num_vars);
        let (hint, comm) = TestZip::commit_single(&pp, &poly).unwrap();
        let point = test_point(num_vars);

        let mut transcript = PcsProverTranscript::new_from_commitment(&comm);
        let field_cfg = get_field_cfg::<Zt, F>(&mut transcript.fs_transcript);

        let result = TestZip::prove_single::<F, CHECKED>(
            &mut transcript,
            &pp,
            &poly,
            &point,
            &hint,
            &field_cfg,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn prove_succeeds_for_poly_type() {
        let num_vars = 4;
        let (pp, poly) = setup_poly_test_params(num_vars);
        let (hint, comm) = TestPolyZip::commit_single(&pp, &poly).unwrap();
        let point: Vec<i128> = (0..num_vars).map(|i| i as i128 + 2).collect();

        let mut transcript = PcsProverTranscript::new_from_commitment(&comm);
        let field_cfg = get_field_cfg::<Zt, F>(&mut transcript.fs_transcript);

        let result = TestPolyZip::prove_single::<F, CHECKED>(
            &mut transcript,
            &pp,
            &poly,
            &point,
            &hint,
            &field_cfg,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn prove_succeeds_with_corrupted_codeword() {
        let num_vars = 4;
        let (pp, poly) = setup_test_params(num_vars);
        let (mut hint, comm) = TestZip::commit_single(&pp, &poly).unwrap();

        {
            let mut rows = hint.cw_matrices[0].to_rows_slices_mut();
            assert!(!rows.is_empty());
            rows[0][0] += Int::ONE;
        }

        let corrupted_tree = {
            let all_rows: Vec<&[_]> = hint.cw_matrices.iter().flat_map(|m| m.as_rows()).collect();
            MerkleTree::new(&all_rows)
        };
        let corrupted_hint = ZipPlusHint::new(hint.cw_matrices, corrupted_tree);

        let point = test_point(num_vars);

        let mut transcript = PcsProverTranscript::new_from_commitment(&comm);
        let field_cfg = get_field_cfg::<Zt, F>(&mut transcript.fs_transcript);

        let result = TestZip::prove_single::<F, CHECKED>(
            &mut transcript,
            &pp,
            &poly,
            &point,
            &corrupted_hint,
            &field_cfg,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn prove_rejects_oversized_polynomial() {
        let num_vars = 4;
        let (pp, _) = setup_test_params(num_vars);
        let oversized_poly: DenseMultilinearExtension<_> = (0..1 << 5).map(Int::from).collect();

        let (hint, comm) =
            TestZip::commit_single(&pp, &setup_test_params::<N, K, M>(num_vars).1).unwrap();

        let point = test_point(num_vars);

        let mut transcript = PcsProverTranscript::new_from_commitment(&comm);
        let field_cfg = get_field_cfg::<Zt, F>(&mut transcript.fs_transcript);

        let result = TestZip::prove_single::<F, CHECKED>(
            &mut transcript,
            &pp,
            &oversized_poly,
            &point,
            &hint,
            &field_cfg,
        );
        assert!(result.is_err());
    }

    /// For TestZipTypes (degree_bound = 0), alphas = [1] so prove eval
    /// equals poly(point) lifted to F.
    #[test]
    fn prove_returns_correct_evaluation() {
        let num_vars = 4;
        let (pp, poly) = setup_test_params(num_vars);
        let (hint, comm) = TestZip::commit_single(&pp, &poly).unwrap();
        let point = test_point(num_vars);

        let mut transcript = PcsProverTranscript::new_from_commitment(&comm);
        let field_cfg = get_field_cfg::<Zt, F>(&mut transcript.fs_transcript);

        let eval_f = TestZip::prove_single::<F, CHECKED>(
            &mut transcript,
            &pp,
            &poly,
            &point,
            &hint,
            &field_cfg,
        )
        .unwrap();

        let poly_wide: DenseMultilinearExtension<Int<M>> =
            poly.evaluations.iter().map(Int::from_ref).collect();
        let expected_int = poly_wide.evaluate(&point, Zero::zero()).unwrap();
        let expected_f: F = (&expected_int).into_with_cfg(&field_cfg);

        assert_eq!(eval_f, expected_f);
    }

    fn make_batch_polys(
        num_vars: usize,
        batch_size: usize,
    ) -> Vec<DenseMultilinearExtension<Int<INT_LIMBS>>> {
        let poly_size = 1 << num_vars;
        (0..batch_size)
            .map(|b| {
                let base = (b * poly_size) as i32;
                (base + 1..=base + poly_size as i32)
                    .map(Int::from)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn prove_succeeds_for_batch() {
        let num_vars = 4;
        let (pp, _) = setup_test_params(num_vars);
        let polys = make_batch_polys(num_vars, 2);

        let (hint, comm) = TestZip::commit(&pp, &polys).unwrap();
        let point = test_point(num_vars);

        let mut transcript = PcsProverTranscript::new_from_commitment(&comm);
        let field_cfg = get_field_cfg::<Zt, F>(&mut transcript.fs_transcript);

        let result =
            TestZip::prove::<F, CHECKED>(&mut transcript, &pp, &polys, &point, &hint, &field_cfg);
        assert!(result.is_ok())
    }

    #[test]
    fn prove_succeeds_for_batch_5() {
        let num_vars = 4;
        let (pp, _) = setup_test_params(num_vars);
        let polys = make_batch_polys(num_vars, 5);

        let (hint, comm) = TestZip::commit(&pp, &polys).unwrap();
        let point = test_point(num_vars);

        let mut transcript = PcsProverTranscript::new_from_commitment(&comm);
        let field_cfg = get_field_cfg::<Zt, F>(&mut transcript.fs_transcript);

        let result =
            TestZip::prove::<F, CHECKED>(&mut transcript, &pp, &polys, &point, &hint, &field_cfg);
        assert!(result.is_ok())
    }

    #[test]
    fn prove_with_corrupted_codeword_for_batch() {
        let num_vars = 4;
        let (pp, _) = setup_test_params(num_vars);
        let polys = make_batch_polys(num_vars, 2);

        let (mut hint, comm) = TestZip::commit(&pp, &polys).unwrap();

        hint.cw_matrices[0].to_rows_slices_mut()[0][0] += Int::ONE;

        let corrupted_tree = {
            let all_rows: Vec<&[_]> = hint.cw_matrices.iter().flat_map(|m| m.as_rows()).collect();
            MerkleTree::new(&all_rows)
        };
        let corrupted_hint = ZipPlusHint::new(hint.cw_matrices, corrupted_tree);

        let point = test_point(num_vars);

        let mut transcript = PcsProverTranscript::new_from_commitment(&comm);
        let field_cfg = get_field_cfg::<Zt, F>(&mut transcript.fs_transcript);

        let result = TestZip::prove::<F, CHECKED>(
            &mut transcript,
            &pp,
            &polys,
            &point,
            &corrupted_hint,
            &field_cfg,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn prove_rejects_oversized_polynomial_in_batch() {
        let num_vars = 4;
        let (pp, _) = setup_test_params(num_vars);
        let oversized: DenseMultilinearExtension<_> = (0..1 << 5).map(Int::from).collect();
        let normal: DenseMultilinearExtension<_> = (1..=16).map(Int::from).collect();
        let polys = vec![normal, oversized];

        let (hint, comm) = TestZip::commit(&pp, &make_batch_polys(num_vars, 2)).unwrap();

        let point = test_point(num_vars);

        let mut transcript = PcsProverTranscript::new_from_commitment(&comm);
        let field_cfg = get_field_cfg::<Zt, F>(&mut transcript.fs_transcript);

        let result =
            TestZip::prove::<F, CHECKED>(&mut transcript, &pp, &polys, &point, &hint, &field_cfg);
        assert!(result.is_err());
    }
}
