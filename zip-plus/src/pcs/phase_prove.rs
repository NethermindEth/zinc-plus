use crate::{
    ZipError,
    code::LinearCode,
    combine_rows,
    pcs::{
        ZipPlusProof,
        structs::{ZipPlus, ZipPlusHint, ZipPlusParams, ZipTypes},
        utils::{point_to_tensor, validate_input},
    },
    pcs_transcript::PcsTranscript,
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

// References main.pdf for the new Zip+ protocol
impl<Zt: ZipTypes, Lc: LinearCode<Zt>> ZipPlus<Zt, Lc> {
    #[allow(clippy::arithmetic_side_effects)]
    pub fn prove<F, const CHECK_FOR_OVERFLOW: bool>(
        pp: &ZipPlusParams<Zt, Lc>,
        polys: &[DenseMultilinearExtension<Zt::Eval>],
        point: &[F],
        commit_hint: &ZipPlusHint<Zt::Cw>,
    ) -> Result<(F, ZipPlusProof), ZipError>
    where
        F: PrimeField
            + for<'a> FromWithConfig<&'a Zt::CombR>
            + for<'a> FromWithConfig<&'a Zt::Chal>
            + for<'a> MulByScalar<&'a F>
            + FromRef<F>,
        F::Inner: FromRef<Zt::Fmod> + Transcribable,
        F::Modulus: FromRef<Zt::Fmod> + Transcribable,
    {
        Self::prove_with_seed::<F, CHECK_FOR_OVERFLOW>(pp, polys, point, commit_hint, &[])
    }

    /// Like [`Self::prove`], but absorbs `seed` bytes into the PCS
    /// Fiat-Shamir transcript **before** deriving the field configuration.
    ///
    /// Pipeline callers pass the Merkle commitment root so the PCS prime
    /// matches the PIOP prime (both absorb the same commitment).
    #[allow(clippy::arithmetic_side_effects)]
    pub fn prove_with_seed<F, const CHECK_FOR_OVERFLOW: bool>(
        pp: &ZipPlusParams<Zt, Lc>,
        polys: &[DenseMultilinearExtension<Zt::Eval>],
        point: &[F],
        commit_hint: &ZipPlusHint<Zt::Cw>,
        seed: &[u8],
    ) -> Result<(F, ZipPlusProof), ZipError>
    where
        F: PrimeField
            + for<'a> FromWithConfig<&'a Zt::CombR>
            + for<'a> FromWithConfig<&'a Zt::Chal>
            + for<'a> MulByScalar<&'a F>
            + FromRef<F>,
        F::Inner: FromRef<Zt::Fmod> + Transcribable,
        F::Modulus: FromRef<Zt::Fmod> + Transcribable,
    {
        let batch_size = polys.len();
        validate_input::<Zt, Lc, _>(
            "prove",
            pp.num_vars,
            batch_size,
            &polys.iter().collect_vec(),
            &[point],
        )?;

        let num_rows = pp.num_rows;
        let row_len = pp.linear_code.row_len();
        let mut transcript = PcsTranscript::new();
        if !seed.is_empty() {
            transcript.fs_transcript.absorb(seed);
        }
        let field_cfg = transcript
            .fs_transcript
            .get_random_field_cfg::<F, Zt::Fmod, Zt::PrimeTest>();

        let (q_0, q_1) = point_to_tensor(num_rows, point, &field_cfg)?;

        // ── 1. Combine each polynomial's evaluations with FS challenges ──
        // Pre-squeeze all per-polynomial alpha vectors up-front so the FS
        // transcript advances identically to the original sequential code,
        // then evaluate the inner products across *all* polynomials in one
        // parallel section (avoids sequential gaps between rayon batches).
        let degree_bound = Zt::Comb::DEGREE_BOUND;
        let all_alphas: Vec<Vec<Zt::Chal>> = polys
            .iter()
            .map(|_| {
                if degree_bound.is_zero() {
                    vec![Zt::Chal::ONE]
                } else {
                    transcript.fs_transcript.get_challenges(degree_bound + 1)
                }
            })
            .collect();

        let polys_as_comb_r: Vec<Vec<Zt::CombR>> = cfg_iter!(polys)
            .zip(cfg_iter!(all_alphas))
            .map(|(poly, alphas)| {
                poly.evaluations
                    .iter()
                    .map(|eval| {
                        Zt::EvalDotChal::inner_product::<CHECK_FOR_OVERFLOW>(
                            eval,
                            alphas,
                            Zt::CombR::ZERO,
                        )
                        .map_err(ZipError::from)
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;

        // ── 2. Compute b = Σ_poly <row, q_1> per row ──────────────────
        // Parallel map across polynomials, then sequential reduce.
        let zero_f = F::zero_with_cfg(&field_cfg);
        let per_poly_dots: Vec<Vec<F>> = cfg_iter!(polys_as_comb_r)
            .map(|poly_comb_r| {
                cfg_chunks!(poly_comb_r, row_len)
                    .map(|row| {
                        let row_f: Vec<F> = row
                            .iter()
                            .map(|int| int.into_with_cfg(&field_cfg))
                            .collect();
                        MBSInnerProduct::inner_product::<UNCHECKED>(&row_f, &q_1, zero_f.clone())
                    })
                    .collect::<Result<_, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;

        let b = per_poly_dots
            .into_iter()
            .fold(vec![zero_f.clone(); num_rows], |mut acc, dots| {
                acc.iter_mut().zip(dots).for_each(|(a, d)| *a += d);
                acc
            });

        transcript.write_field_elements(&b)?;
        // Compute eval = <q_0, b> (inner product in field), <q_2, b> in paper
        // It is safe to use inner_product_unchecked because we're in a field.
        let eval = MBSInnerProduct::inner_product::<UNCHECKED>(&q_0, &b, zero_f.clone())?;

        // ── 3. Compute combined_row ────────────────────────────────────
        // combined_row_i[col] = sum_j(s_j * int_rows_i[j][col]) for each column
        // combined_row: Vec<CombR> (length row_len), s_j in paper
        let coeffs = if pp.num_rows == 1 {
            vec![Zt::Chal::ONE]
        } else {
            transcript
                .fs_transcript
                .get_challenges::<Zt::Chal>(num_rows)
        };

        // Parallel map across polys, sequential reduce.
        let per_poly_rows: Vec<Vec<Zt::CombR>> = cfg_iter!(polys_as_comb_r)
            .map(|poly| -> Result<_, ZipError> {
                Ok(combine_rows!(
                    CHECK_FOR_OVERFLOW,
                    &coeffs,
                    poly.iter(),
                    |eval: &Zt::CombR| Ok::<_, ZipError>(eval.clone()),
                    row_len,
                    Zt::CombR::ZERO
                ))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let combined_row: Vec<Zt::CombR> = per_poly_rows.into_iter().fold(
            vec![Zt::CombR::ZERO; row_len],
            |mut acc, row| {
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
                acc
            },
        );

        // ── 4. Column openings ─────────────────────────────────────────
        // Pre-squeeze all column indices (FS-only, no stream dependency),
        // then write openings sequentially to preserve byte-stream order.
        transcript.write_const_many(&combined_row)?;

        // ── 4a. Grinding (proof-of-work) ──────────────────────────────
        // If GRINDING_BITS > 0, the prover searches for a nonce that
        // commits extra bits of security, allowing fewer column openings.
        transcript.grind(Zt::GRINDING_BITS)?;

        let column_indices: Vec<usize> = (0..Zt::NUM_COLUMN_OPENINGS)
            .map(|_| transcript.squeeze_challenge_idx(pp.linear_code.codeword_len()))
            .collect();
        for column_idx in column_indices {
            Self::open_merkle_trees_for_column(commit_hint, column_idx, &mut transcript)?;
        }

        Ok((eval, transcript.into()))
    }

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
        IntoWithConfig, PrimeField, crypto_bigint_boxed_monty::BoxedMontyField,
        crypto_bigint_int::Int,
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

    fn field_cfg() -> <F as PrimeField>::Config {
        let mut t = PcsTranscript::new();
        t.fs_transcript
            .get_random_field_cfg::<F, <Zt as ZipTypes>::Fmod, <Zt as ZipTypes>::PrimeTest>()
    }

    fn test_point_f(num_vars: usize) -> Vec<F> {
        let cfg = field_cfg();
        test_point(num_vars)
            .iter()
            .map(|v| v.into_with_cfg(&cfg))
            .collect()
    }

    #[test]
    fn prove_succeeds_for_single_poly() {
        let num_vars = 4;
        let (pp, poly) = setup_test_params(num_vars);
        let (hint, _) = TestZip::commit_single(&pp, &poly).unwrap();
        let point = test_point_f(num_vars);

        let result = TestZip::prove::<F, CHECKED>(&pp, &[poly], &point, &hint);
        assert!(result.is_ok());
    }

    #[test]
    fn prove_succeeds_for_poly_type() {
        let num_vars = 4;
        let (pp, poly) = setup_poly_test_params(num_vars);
        let (hint, _) = TestPolyZip::commit_single(&pp, &poly).unwrap();
        let cfg = field_cfg();
        let point: Vec<F> = (0..num_vars)
            .map(|i| (&(i as i128 + 2)).into_with_cfg(&cfg))
            .collect();

        let result = TestPolyZip::prove::<F, CHECKED>(&pp, &[poly], &point, &hint);
        assert!(result.is_ok());
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
            let all_rows: Vec<&[_]> = hint.cw_matrices.iter().flat_map(|m| m.as_rows()).collect();
            MerkleTree::new(&all_rows)
        };
        let corrupted_hint = ZipPlusHint::new(hint.cw_matrices, corrupted_tree);

        let point = test_point_f(num_vars);
        let result = TestZip::prove::<F, CHECKED>(&pp, &[poly], &point, &corrupted_hint);
        assert!(result.is_ok());
    }

    #[test]
    fn prove_rejects_oversized_polynomial() {
        let num_vars = 4;
        let (pp, _) = setup_test_params(num_vars);
        let oversized_poly: DenseMultilinearExtension<_> = (0..1 << 5).map(Int::from).collect();

        let (hint, _) =
            TestZip::commit_single(&pp, &setup_test_params::<N, K, M>(num_vars).1).unwrap();

        let point = test_point_f(num_vars);
        let result = TestZip::prove::<F, CHECKED>(&pp, &[oversized_poly], &point, &hint);
        assert!(result.is_err());
    }

    /// For TestZipTypes (degree_bound = 0), alphas = [1] so prove eval
    /// equals poly(point) lifted to F.
    #[test]
    fn prove_returns_correct_evaluation() {
        let num_vars = 4;
        let (pp, poly) = setup_test_params(num_vars);
        let (hint, _) = TestZip::commit_single(&pp, &poly).unwrap();
        let int_point = test_point(num_vars);
        let cfg = field_cfg();
        let point_f: Vec<F> = int_point.iter().map(|v| v.into_with_cfg(&cfg)).collect();

        let (eval, _) =
            TestZip::prove::<F, CHECKED>(&pp, std::slice::from_ref(&poly), &point_f, &hint)
                .unwrap();

        let poly_wide: DenseMultilinearExtension<Int<M>> =
            poly.evaluations.iter().map(Int::from_ref).collect();
        let expected_int = poly_wide.evaluate(&int_point, Zero::zero()).unwrap();
        let expected_f: F = (&expected_int).into_with_cfg(&cfg);

        assert_eq!(eval, expected_f);
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

        let (hint, _) = TestZip::commit(&pp, &polys).unwrap();
        let point = test_point_f(num_vars);

        let result = TestZip::prove::<F, CHECKED>(&pp, &polys, &point, &hint);
        assert!(result.is_ok())
    }

    #[test]
    fn prove_succeeds_for_batch_5() {
        let num_vars = 4;
        let (pp, _) = setup_test_params(num_vars);
        let polys = make_batch_polys(num_vars, 5);

        let (hint, _) = TestZip::commit(&pp, &polys).unwrap();
        let point = test_point_f(num_vars);

        let result = TestZip::prove::<F, CHECKED>(&pp, &polys, &point, &hint);
        assert!(result.is_ok())
    }

    #[test]
    fn prove_with_corrupted_codeword_for_batch() {
        let num_vars = 4;
        let (pp, _) = setup_test_params(num_vars);
        let polys = make_batch_polys(num_vars, 2);

        let (mut hint, _) = TestZip::commit(&pp, &polys).unwrap();

        hint.cw_matrices[0].to_rows_slices_mut()[0][0] += Int::ONE;

        let corrupted_tree = {
            let all_rows: Vec<&[_]> = hint.cw_matrices.iter().flat_map(|m| m.as_rows()).collect();
            MerkleTree::new(&all_rows)
        };
        let corrupted_hint = ZipPlusHint::new(hint.cw_matrices, corrupted_tree);

        let point = test_point_f(num_vars);
        let result = TestZip::prove::<F, CHECKED>(&pp, &polys, &point, &corrupted_hint);
        assert!(result.is_ok());
    }

    #[test]
    fn prove_rejects_oversized_polynomial_in_batch() {
        let num_vars = 4;
        let (pp, _) = setup_test_params(num_vars);
        let oversized: DenseMultilinearExtension<_> = (0..1 << 5).map(Int::from).collect();
        let normal: DenseMultilinearExtension<_> = (1..=16).map(Int::from).collect();
        let polys = vec![normal, oversized];

        let (hint, _) = TestZip::commit(&pp, &make_batch_polys(num_vars, 2)).unwrap();
        let point = test_point_f(num_vars);
        let result = TestZip::prove::<F, CHECKED>(&pp, &polys, &point, &hint);
        assert!(result.is_err());
    }
}
