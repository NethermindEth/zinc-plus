use crate::{
    ZipError,
    code::LinearCode,
    merkle::MtHash,
    pcs::{
        structs::{
            MulByScalar, ProjectableToField, ZipPlus, ZipPlusCommitment, ZipPlusParams, ZipTypes,
        },
        utils::{ColumnOpening, point_to_tensor, validate_input},
    },
    pcs_transcript::PcsTranscript,
    poly::Polynomial,
    traits::{FromRef, Transcribable, Transcript},
    utils::inner_product,
};
use ark_std::iterable::Iterable;
use crypto_primitives::PrimeField;
use itertools::Itertools;

impl<Zt: ZipTypes> ZipPlus<Zt> {
    pub fn verify<F>(
        vp: &ZipPlusParams<Zt>,
        comm: &ZipPlusCommitment,
        point_f: &[F],
        eval_f: &F,
        transcript: &mut PcsTranscript,
    ) -> Result<(), ZipError>
    where
        F: PrimeField
            + FromRef<F>
            + FromRef<Zt::Chal>
            + FromRef<<Zt::Code as LinearCode<Zt::Eval, Zt::Cw, Zt::CombR>>::Inner>
            + for<'a> MulByScalar<&'a F>,
        F::Inner: Transcribable,
        Zt::Cw: ProjectableToField<F>,
    {
        validate_input::<Zt, _>("verify", vp.num_vars, &[], [point_f])?;

        let columns_opened = Self::verify_testing(vp, &comm.root, transcript)?;

        let projecting_element: Zt::Chal = transcript.fs_transcript.get_challenge();
        let projecting_element: F = F::from_ref(&projecting_element);

        Self::verify_evaluation(
            vp,
            point_f,
            eval_f,
            &columns_opened,
            transcript,
            projecting_element,
        )?;

        Ok(())
    }

    pub fn batch_verify<'a, F>(
        vp: &ZipPlusParams<Zt>,
        comms: impl Iterable<Item = &'a ZipPlusCommitment>,
        points: &[Vec<F>],
        evals: &[F],
        transcript: &mut PcsTranscript,
    ) -> Result<(), ZipError>
    where
        F: PrimeField
            + FromRef<F>
            + FromRef<Zt::Chal>
            + FromRef<<Zt::Code as LinearCode<Zt::Eval, Zt::Cw, Zt::CombR>>::Inner>
            + for<'b> MulByScalar<&'b F>,
        F::Inner: Transcribable,
        Zt::Cw: ProjectableToField<F>,
    {
        for (i, (eval, comm)) in evals.iter().zip(comms.iter()).enumerate() {
            Self::verify(vp, comm, &points[i], eval, transcript)?;
        }
        Ok(())
    }

    #[allow(clippy::type_complexity)]
    pub(super) fn verify_testing(
        vp: &ZipPlusParams<Zt>,
        root: &MtHash,
        transcript: &mut PcsTranscript,
    ) -> Result<Vec<(usize, Vec<Zt::Cw>)>, ZipError> {
        // Gather the coeffs and encoded combined rows per proximity test
        let encoded_combined_rows: Option<(Vec<Zt::Chal>, Vec<Zt::Chal>, Vec<Zt::CombR>)> = if vp
            .num_rows
            > 1
        {
            // Values to evaluate the coefficients at
            let alphas = transcript
                .fs_transcript
                .get_challenges(Zt::Comb::DEGREE_BOUND);

            // Coefficients for the linear combination of polynomial with evaluated
            // coefficients
            let coeffs = transcript.fs_transcript.get_challenges(vp.num_rows);

            let combined_row: Vec<Zt::CombR> = transcript.read_many(vp.linear_code.row_len())?;

            let encoded_combined_row: Vec<Zt::CombR> = vp.linear_code.encode_wide(&combined_row);
            Some((alphas, coeffs, encoded_combined_row))
        } else {
            None
        };

        let mut columns_opened: Vec<(usize, Vec<Zt::Cw>)> =
            Vec::with_capacity(Zt::NUM_COLUMN_OPENINGS);

        for _ in 0..Zt::NUM_COLUMN_OPENINGS {
            let column_idx = transcript.squeeze_challenge_idx(vp.linear_code.codeword_len());
            let column_values = transcript.read_many(vp.num_rows)?;

            if let Some((ref alphas, ref coeffs, ref encoded_combined_row)) = encoded_combined_rows
            {
                Self::verify_column_testing(
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

    pub(super) fn verify_column_testing(
        alphas: &[Zt::Chal],
        coeffs: &[Zt::Chal],
        encoded_combined_row: &[Zt::CombR],
        column_entries: &[Zt::Cw],
        column: usize,
        num_rows: usize,
    ) -> Result<(), ZipError> {
        let column_entries_comb: Zt::CombR = if num_rows > 1 {
            let column_entries: Result<Vec<Zt::CombR>, _> = column_entries
                .iter()
                .map(Zt::Comb::from_ref)
                .map(|p| p.evaluate_at_point(alphas))
                .collect();
            inner_product(coeffs.iter(), column_entries?.iter())
        } else {
            Zt::Comb::from_ref(&column_entries[0]).evaluate_at_point(alphas)?
        };

        if column_entries_comb != encoded_combined_row[column] {
            return Err(ZipError::InvalidPcsOpen("Proximity failure".into()));
        }
        Ok(())
    }

    fn verify_evaluation<F>(
        vp: &ZipPlusParams<Zt>,
        point_f: &[F],
        eval_f: &F,
        columns_opened: &[(usize, Vec<Zt::Cw>)],
        transcript: &mut PcsTranscript,
        projecting_element: F,
    ) -> Result<(), ZipError>
    where
        F: PrimeField
            + FromRef<F>
            + FromRef<<Zt::Code as LinearCode<Zt::Eval, Zt::Cw, Zt::CombR>>::Inner>
            + for<'a> MulByScalar<&'a F>,
        F::Inner: Transcribable,
        Zt::Cw: ProjectableToField<F>,
    {
        let q_0_combined_row = transcript.read_field_elements(vp.linear_code.row_len())?;
        let encoded_combined_row = vp.linear_code.encode_f(&q_0_combined_row);

        let (q_0, q_1) = point_to_tensor(vp.num_rows, point_f)?;

        if inner_product(&q_0_combined_row, &q_1) != *eval_f {
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
            )?;
        }

        Ok(())
    }

    fn verify_proximity_q_0<F>(
        q_0: &Vec<F>,
        encoded_q_0_combined_row: &[F],
        column_entries: &[Zt::Cw],
        column: usize,
        num_rows: usize,
        project: &impl Fn(&<Zt as ZipTypes>::Cw) -> F,
    ) -> Result<(), ZipError>
    where
        F: PrimeField + for<'a> MulByScalar<&'a F>,
        Zt::Cw: ProjectableToField<F>,
    {
        let column_entries_comb = if num_rows > 1 {
            let column_entries = column_entries.iter().map(project).collect_vec();
            inner_product(q_0, &column_entries)
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
        code::LinearCode,
        field::F256,
        pcs::{
            structs::{ZipPlus, ZipTypes},
            test_utils::*,
        },
        pcs_transcript::PcsTranscript,
        poly::{
            dense::DensePolynomial,
            mle::{DenseMultilinearExtension, MultilinearExtensionRand},
        },
        traits::FromRef,
        transcript::KeccakTranscript,
        utils::WORD_FACTOR,
    };
    use crypto_bigint::{Random, U256, const_monty_params};
    use crypto_primitives::{Ring, crypto_bigint_int::Int};
    use itertools::Itertools;
    use num_traits::{ConstOne, ConstZero};
    use rand::prelude::*;

    const INT_LIMBS: usize = WORD_FACTOR;
    const FIELD_LIMBS: usize = 4 * WORD_FACTOR;

    const N: usize = INT_LIMBS;
    const K: usize = INT_LIMBS * 4;
    const M: usize = INT_LIMBS * 8;
    const DEGREE: usize = 2;

    const_monty_params!(
        ModP,
        U256,
        "0000000000000000000000000000000000000000B933426489189CB5B47D567F"
    );

    type F = F256<ModP>;

    type Zt = TestZipTypes<N, K, M>;
    type C = <Zt as ZipTypes>::Code;

    type PolyZt = TestPolyZipTypes<N, K, M, DEGREE>;

    type TestZip = ZipPlus<Zt>;
    type TestPolyZip = ZipPlus<PolyZt>;

    #[test]
    fn successful_verification_of_valid_proof() {
        let num_vars = 4;
        {
            let (pp, comm, point_f, eval_f, proof) = setup_full_protocol::<F, N, K, M>(num_vars);

            let mut verifier_transcript = PcsTranscript::from_proof(&proof);
            let result = TestZip::verify(&pp, &comm, &point_f, &eval_f, &mut verifier_transcript);
            assert!(result.is_ok(), "Verification failed: {result:?}")
        };
        {
            let (pp, comm, point_f, eval_f, proof) =
                setup_full_protocol_poly::<F, N, K, M, DEGREE>(num_vars);

            let mut verifier_transcript = PcsTranscript::from_proof(&proof);
            let result =
                TestPolyZip::verify(&pp, &comm, &point_f, &eval_f, &mut verifier_transcript);

            assert!(result.is_ok(), "Verification failed: {result:?}");
        }
    }

    #[test]
    fn verification_fails_with_incorrect_evaluation() {
        let num_vars = 4;

        {
            let (pp, comm, point_f, eval_f, proof) = setup_full_protocol::<F, N, K, M>(num_vars);

            let mut verifier_transcript = PcsTranscript::from_proof(&proof);
            let result = TestZip::verify(
                &pp,
                &comm,
                &point_f,
                &(eval_f + F::ONE),
                &mut verifier_transcript,
            );

            assert!(result.is_err());
        }

        {
            let (pp, comm, point_f, eval_f, proof) =
                setup_full_protocol_poly::<F, N, K, M, DEGREE>(num_vars);

            let mut verifier_transcript = PcsTranscript::from_proof(&proof);
            let result = TestPolyZip::verify(
                &pp,
                &comm,
                &point_f,
                &(eval_f + F::ONE),
                &mut verifier_transcript,
            );

            assert!(result.is_err());
        }
    }

    #[test]
    fn verification_fails_with_tampered_proof() {
        fn tamper(proof: Vec<u8>) -> Vec<u8> {
            let mut tampered = proof.clone();
            tampered[0] ^= 0x01;
            tampered
        }
        let num_vars = 4;

        {
            let (pp, comm, point_f, eval, proof) = setup_full_protocol::<F, N, K, M>(num_vars);
            let tampered = tamper(proof);
            let mut verifier_transcript = PcsTranscript::from_proof(&tampered);
            let result = TestZip::verify(&pp, &comm, &point_f, &eval, &mut verifier_transcript);
            assert!(result.is_err());
        }

        {
            let (pp, comm, point_f, eval_f, proof) =
                setup_full_protocol_poly::<F, N, K, M, DEGREE>(num_vars);
            let tampered = tamper(proof);
            let mut verifier_transcript = PcsTranscript::from_proof(&tampered);
            let result =
                TestPolyZip::verify(&pp, &comm, &point_f, &eval_f, &mut verifier_transcript);
            assert!(result.is_err());
        }
    }

    #[test]
    fn verification_fails_with_wrong_commitment() {
        let num_vars = 4;
        {
            let (pp, _comm_poly1, point_f, eval_f, proof_poly1) =
                setup_full_protocol::<F, N, K, M>(num_vars);

            let different_evals: Vec<_> = (20..(20 + (1 << num_vars))).map(Int::from).collect();
            let poly2 = DenseMultilinearExtension::from_evaluations_vec(num_vars, different_evals);
            let (_, comm_poly2) = TestZip::commit(&pp, &poly2).unwrap();

            let mut transcript = PcsTranscript::from_proof(&proof_poly1);
            let result = TestZip::verify(&pp, &comm_poly2, &point_f, &eval_f, &mut transcript);

            assert!(result.is_err());
        }

        {
            let (pp, _comm_poly1, point_f, eval_f, proof_poly1) =
                setup_full_protocol_poly::<F, N, K, M, DEGREE>(num_vars);

            let different_evals = {
                let different_eval_coeffs: Vec<_> = (1..=((1 << num_vars) * DEGREE as i32))
                    .map(|x| Int::from_i32(x + 20))
                    .collect_vec();
                different_eval_coeffs
                    .chunks_exact(DEGREE)
                    .map(DensePolynomial::new)
                    .collect_vec()
            };

            let poly2 = DenseMultilinearExtension::from_evaluations_vec(num_vars, different_evals);
            let (_, comm_poly2) = TestPolyZip::commit(&pp, &poly2).unwrap();

            let mut transcript = PcsTranscript::from_proof(&proof_poly1);
            let result = TestPolyZip::verify(&pp, &comm_poly2, &point_f, &eval_f, &mut transcript);

            assert!(result.is_err());
        }
    }

    #[test]
    fn verification_fails_with_invalid_point_size() {
        let num_vars = 4;
        let mut invalid_point = vec![];
        for i in 0..=num_vars {
            invalid_point.push(F::from(100 + i as i32));
        }

        {
            let (pp, comm, _point_f, eval_f, proof) =
                setup_full_protocol_poly::<F, N, K, M, DEGREE>(num_vars);

            let mut transcript = PcsTranscript::from_proof(&proof);
            let result = TestPolyZip::verify(&pp, &comm, &invalid_point, &eval_f, &mut transcript);

            assert!(matches!(result, Err(..)));
        }

        {
            let (pp, comm, _point_f, eval, proof) = setup_full_protocol::<F, N, K, M>(num_vars);

            let mut transcript = PcsTranscript::from_proof(&proof);
            let result = TestZip::verify(&pp, &comm, &invalid_point, &eval, &mut transcript);

            assert!(matches!(result, Err(..)));
        }
    }

    #[test]
    fn verification_fails_if_proximity_check_is_invalid() {
        let mut keccak = MockTranscript::default();
        let poly_size = 8; // row_len=4, num_rows=2 -> proximity checks are active
        let n = 3;

        let linear_code = C::new(poly_size, true, &mut keccak);
        let pp = TestZip::setup(poly_size, linear_code);

        let evaluations: Vec<_> = (0..poly_size as i32)
            .map(<Zt as ZipTypes>::Eval::from)
            .collect();
        let mle = DenseMultilinearExtension::from_evaluations_slice(n, &evaluations);

        let (data, comm) = TestZip::commit(&pp, &mle).expect("commit should succeed");

        let point = [0i64, 0i64, 0i64]
            .into_iter()
            .map(Int::<1>::from)
            .collect::<Vec<_>>();
        let point_f: Vec<F> = point.iter().map(F::from).collect_vec();
        let eval = mle.evaluate(&point).unwrap();

        let mut prover_tr = PcsTranscript::new();
        let eval_f =
            TestZip::open(&pp, &mle, &data, &point, &mut prover_tr).expect("open should succeed");
        assert_eq!(
            eval_f,
            F::from_ref(&eval),
            "Evaluation mismatch after opening"
        );
        let mut proof = prover_tr.into_proof();

        let row_len = pp.linear_code.row_len();
        let bytes_per_int = M * size_of::<crypto_bigint::Word>();
        let first_combined_row_bytes = row_len * bytes_per_int;
        assert!(
            first_combined_row_bytes <= proof.len(),
            "proof too small to tamper"
        );

        let flip_at = bytes_per_int * (row_len / 2);
        proof[flip_at] ^= 0x01;

        let mut ver_tr = PcsTranscript::from_proof(&proof);
        let res = TestZip::verify(&pp, &comm, &point_f, &eval_f, &mut ver_tr);

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
        fn evaluate_in_field<R>(evaluations: &[R], point: &[F]) -> F
        where
            R: Ring,
            F: FromRef<R>,
        {
            let num_vars = point.len();
            assert_eq!(evaluations.len(), 1 << num_vars);
            let mut current_evals: Vec<F> = evaluations.iter().map(F::from_ref).collect_vec();
            for p in point.iter().take(num_vars) {
                let one_minus_p_i = F::ONE - p;
                let mut next_evals = Vec::with_capacity(current_evals.len() / 2);
                for j in (0..current_evals.len()).step_by(2) {
                    let val = current_evals[j] * one_minus_p_i + current_evals[j + 1] * p;
                    next_evals.push(val);
                }
                current_evals = next_evals;
            }
            current_evals[0]
        }

        let mut rng = ThreadRng::default();

        let n = 3;
        let poly_size = 1 << n;
        let mut keccak_transcript = KeccakTranscript::new();
        let linear_code: C = C::new(poly_size, true, &mut keccak_transcript);
        let param = TestZip::setup(poly_size, linear_code);
        let evaluations: Vec<_> = (0..poly_size)
            .map(|_| <Zt as ZipTypes>::Eval::from(rng.random::<i8>()))
            .collect();
        let mle = DenseMultilinearExtension::from_evaluations_slice(n, &evaluations);
        let point: Vec<_> = (0..n)
            .map(|_| <Zt as ZipTypes>::Pt::random(&mut rng))
            .collect();
        let point_f = point.iter().map(F::from_ref).collect_vec();

        let (mut data, comm) = TestZip::commit(&param, &mle).unwrap();
        if !data.rows.is_empty() {
            data.rows[0] += Int::ONE;
        }

        let mut prover_transcript = PcsTranscript::new();
        let _: F = TestZip::open(&param, &mle, &data, &point, &mut prover_transcript).unwrap();
        let proof = prover_transcript.into_proof();

        let mut verifier_transcript = PcsTranscript::from_proof(&proof);
        let eval_f = evaluate_in_field(&mle.evaluations, &point_f);
        let verification_result =
            TestZip::verify(&param, &comm, &point_f, &eval_f, &mut verifier_transcript);

        assert!(verification_result.is_err());
    }

    #[test]
    fn verification_fails_if_evaluation_consistency_check_is_invalid() {
        let mut keccak = MockTranscript::default();
        let poly_size = 8;
        let linear_code = C::new(poly_size, true, &mut keccak);
        let pp = TestZip::setup(poly_size, linear_code);

        let evaluations: Vec<_> = (0..poly_size as i32).map(Int::<INT_LIMBS>::from).collect();
        let n = 3;
        let mle = DenseMultilinearExtension::from_evaluations_slice(n, &evaluations);

        let (data, comm) = TestZip::commit(&pp, &mle).expect("commit should succeed");

        let point: Vec<<Zt as ZipTypes>::Pt> =
            [0i64, 0i64, 0i64].into_iter().map(Int::from).collect_vec();
        let point_f: Vec<F> = point.iter().map(F::from).collect_vec();
        let eval_f = mle.evaluate(&point).unwrap().into();

        let mut prover_tr = PcsTranscript::new();
        let _: F =
            TestZip::open(&pp, &mle, &data, &point, &mut prover_tr).expect("open should succeed");
        let mut proof = prover_tr.into_proof();

        let row_len = pp.linear_code.row_len();
        let bytes_per_field = FIELD_LIMBS * size_of::<crypto_bigint::Word>();
        let q0_bytes = row_len * bytes_per_field;
        assert!(
            proof.len() >= q0_bytes,
            "proof too small to contain q_0_combined_row"
        );

        let tail_start = proof.len() - q0_bytes;
        let flip_at = tail_start + (bytes_per_field / 2);
        proof[flip_at] ^= 0x01;

        let mut ver_tr = PcsTranscript::from_proof(&proof);
        let res = TestZip::verify(&pp, &comm, &point_f, &eval_f, &mut ver_tr);

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
        let mut keccak = MockTranscript::default();
        let poly_size = 8;
        let linear_code = C::new(poly_size, true, &mut keccak);
        let pp = TestZip::setup(poly_size, linear_code);

        let evaluations: Vec<_> = vec![Int::<INT_LIMBS>::from(0); poly_size];
        let n = 3;
        let mle = DenseMultilinearExtension::from_evaluations_slice(n, &evaluations);

        let (data, comm) = TestZip::commit(&pp, &mle).expect("commit should succeed");

        let point: Vec<<Zt as ZipTypes>::Pt> =
            [0i64, 0i64, 0i64].into_iter().map(Int::from).collect_vec();
        let point_f: Vec<F> = point.iter().map(F::from).collect_vec();
        let eval_f = mle.evaluate(&point).unwrap().into();
        let mut prover_tr = PcsTranscript::new();
        let _: F =
            TestZip::open(&pp, &mle, &data, &point, &mut prover_tr).expect("open should succeed");
        let proof = prover_tr.into_proof();

        let mut ver_tr = PcsTranscript::from_proof(&proof);
        let res = TestZip::verify(&pp, &comm, &point_f, &eval_f, &mut ver_tr);
        assert!(res.is_ok());
    }

    #[test]
    fn verification_succeeds_at_zero_point() {
        let mut keccak = MockTranscript::default();
        let poly_size = 8;
        let linear_code = C::new(poly_size, true, &mut keccak);
        let pp = TestZip::setup(poly_size, linear_code);

        let evaluations: Vec<_> = (1..=poly_size as i32).map(Int::<INT_LIMBS>::from).collect();
        let n = 3;
        let mle = DenseMultilinearExtension::from_evaluations_slice(n, &evaluations);

        let (data, comm) = TestZip::commit(&pp, &mle).expect("commit should succeed");

        let point: Vec<<Zt as ZipTypes>::Pt> = vec![Int::ZERO; n];
        let point_f: Vec<F> = point.iter().map(F::from).collect_vec();

        let eval_f = mle.evaluate(&point).unwrap().into();

        let mut prover_tr = PcsTranscript::new();
        let _: F =
            TestZip::open(&pp, &mle, &data, &point, &mut prover_tr).expect("open should succeed");
        let proof = prover_tr.into_proof();

        let mut ver_tr = PcsTranscript::from_proof(&proof);
        let res = TestZip::verify(&pp, &comm, &point_f, &eval_f, &mut ver_tr);
        assert!(res.is_ok());
    }

    #[test]
    fn verification_fails_if_proximity_values_are_too_large() {
        let mut keccak = MockTranscript::default();
        let poly_size = 8;
        let linear_code = C::new(poly_size, true, &mut keccak);
        let pp = TestZip::setup(poly_size, linear_code);

        let evaluations: Vec<_> = (1..=poly_size as i32).map(Int::<INT_LIMBS>::from).collect();
        let n = 3;
        let mle = DenseMultilinearExtension::from_evaluations_slice(n, &evaluations);

        let (data, comm) = TestZip::commit(&pp, &mle).expect("commit should succeed");

        let point: Vec<<Zt as ZipTypes>::Pt> =
            [0i64, 0i64, 0i64].into_iter().map(Int::from).collect_vec();
        let point_f: Vec<F> = point.iter().map(F::from).collect_vec();
        let eval_f = mle.evaluate(&point).unwrap().into();

        let mut prover_tr = PcsTranscript::new();
        let _: F =
            TestZip::open(&pp, &mle, &data, &point, &mut prover_tr).expect("open should succeed");
        let mut proof = prover_tr.into_proof();

        let row_len = pp.linear_code.row_len();
        let bytes_per_int = M * 8;
        let first_section_bytes = row_len * bytes_per_int;
        assert!(
            first_section_bytes <= proof.len(),
            "proof too small to tamper u'"
        );

        for b in &mut proof[0..bytes_per_int] {
            *b = 0xFF;
        }

        let mut ver_tr = PcsTranscript::from_proof(&proof);
        let res = TestZip::verify(&pp, &comm, &point_f, &eval_f, &mut ver_tr);
        assert!(res.is_err());
    }

    /// Mirrors: `Zip/Verify: RandomField<4>, poly_size = 2^12 (Int limbs = 1)`
    #[test]
    fn bench_p12_verify() {
        fn inner<const P: usize>() {
            let mut rng = ThreadRng::default();
            // Match the benchmark’s transcript usage for linear code construction
            let mut keccak_transcript = KeccakTranscript::new();
            let poly_size = 1 << P;
            let linear_code = C::new(poly_size, true, &mut keccak_transcript);
            let params = TestZip::setup(poly_size, linear_code);

            let poly = DenseMultilinearExtension::rand(P, &mut rng);
            let (data, commitment) = TestZip::commit(&params, &poly).expect("commit");

            // Same point choice as the bench
            let point = vec![1i64; P].iter().map(|v| v.into()).collect_vec();
            let eval = *poly.evaluations.last().expect("nonempty evals");

            // Prover produces a proof once (exactly as in the bench)
            let mut prover_tx = PcsTranscript::new();
            let eval_f: F =
                TestZip::open(&params, &poly, &data, &point, &mut prover_tx).expect("open");
            let proof = prover_tx.into_proof();
            assert_eq!(eval_f, eval.into(), "Evaluation mismatch after opening");

            // Verifier replays verification from the same proof (also like the bench)
            let mut verifier_tx = PcsTranscript::from_proof(&proof);
            TestZip::verify(
                &params,
                &commitment,
                &point.iter().map(F::from).collect::<Vec<_>>(),
                &eval_f,
                &mut verifier_tx,
            )
            .expect("verify");
        }

        inner::<12>();
    }
}
