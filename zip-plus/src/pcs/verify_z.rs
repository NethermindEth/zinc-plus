use crate::{
    ZipError,
    code::LinearCode,
    pcs::{
        structs::{MultilinearZip, MultilinearZipCommitment, MultilinearZipParams},
        utils::{ColumnOpening, MtHash, point_to_tensor, validate_input},
    },
    pcs_transcript::PcsTranscript,
    poly::dense::DenseMultilinearExtension,
    traits::Transcribable,
    utils::{expand, inner_product},
};
use ark_std::iterable::Iterable;
use crypto_primitives::{PrimeField, crypto_bigint_int::Int};
use itertools::Itertools;

impl<const N: usize, const L: usize, const K: usize, const M: usize, LC: LinearCode<N, L, K, M>>
    MultilinearZip<N, L, K, M, LC>
{
    pub fn verify<F>(
        vp: &MultilinearZipParams<N, L, K, M, LC>,
        comm: &MultilinearZipCommitment,
        point: &[F],
        eval: &F,
        transcript: &mut PcsTranscript,
    ) -> Result<(), ZipError>
    where
        F: PrimeField + for<'a> From<&'a Int<L>> + for<'a> From<&'a Int<K>>,
        F::Inner: Transcribable,
    {
        let no_polys = Vec::<DenseMultilinearExtension<bool>>::new();
        validate_input("verify", vp.num_vars, &no_polys, [point])?;

        let columns_opened = Self::verify_testing(vp, &comm.root, transcript)?;

        Self::verify_evaluation_z(vp, point, eval, &columns_opened, transcript)?;

        Ok(())
    }

    pub fn batch_verify_z<'a, F>(
        vp: &MultilinearZipParams<N, L, K, M, LC>,
        comms: impl Iterable<Item = &'a MultilinearZipCommitment>,
        points: &[Vec<F>],
        evals: &[F],
        transcript: &mut PcsTranscript,
    ) -> Result<(), ZipError>
    where
        F: PrimeField + for<'b> From<&'b Int<L>> + for<'b> From<&'b Int<K>>,
        F::Inner: Transcribable,
    {
        for (i, (eval, comm)) in evals.iter().zip(comms.iter()).enumerate() {
            Self::verify(vp, comm, &points[i], eval, transcript)?;
        }
        Ok(())
    }

    #[allow(clippy::type_complexity)]
    pub(super) fn verify_testing(
        vp: &MultilinearZipParams<N, L, K, M, LC>,
        root: &MtHash,
        transcript: &mut PcsTranscript,
    ) -> Result<Vec<(usize, Vec<Int<K>>)>, ZipError> {
        // Gather the coeffs and encoded combined rows per proximity test
        let mut encoded_combined_rows: Vec<(Vec<Int<N>>, Vec<Int<M>>)> =
            Vec::with_capacity(vp.linear_code.num_proximity_testing());

        if vp.num_rows > 1 {
            for _ in 0..vp.linear_code.num_proximity_testing() {
                let coeffs = transcript.fs_transcript.get_integer_challenges(vp.num_rows);

                let combined_row: Vec<Int<M>> =
                    transcript.read_integers(vp.linear_code.row_len())?;

                let encoded_combined_row: Vec<Int<M>> = vp.linear_code.encode_wide(&combined_row);
                encoded_combined_rows.push((coeffs, encoded_combined_row));
            }
        }

        let mut columns_opened: Vec<(usize, Vec<Int<K>>)> =
            Vec::with_capacity(vp.linear_code.num_column_opening());

        for _ in 0..vp.linear_code.num_column_opening() {
            let column_idx = transcript.squeeze_challenge_idx(vp.linear_code.codeword_len());
            let column_values = transcript.read_integers(vp.num_rows)?;

            for (coeffs, encoded_combined_row) in encoded_combined_rows.iter() {
                Self::verify_column_testing(
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
        coeffs: &[Int<N>],
        encoded_combined_row: &[Int<M>],
        column_entries: &[Int<K>],
        column: usize,
        num_rows: usize,
    ) -> Result<(), ZipError> {
        let column_entries_comb: Int<M> = if num_rows > 1 {
            let coeffs: Vec<Int<M>> = coeffs.iter().map(expand::<N, M>).collect();
            let column_entries: Vec<Int<M>> = column_entries.iter().map(expand::<K, M>).collect();
            inner_product(coeffs.iter(), column_entries.iter())
        } else {
            expand(&column_entries[0])
        };

        if column_entries_comb != encoded_combined_row[column] {
            return Err(ZipError::InvalidPcsOpen("Proximity failure".into()));
        }
        Ok(())
    }

    fn verify_evaluation_z<F>(
        vp: &MultilinearZipParams<N, L, K, M, LC>,
        point: &[F],
        eval: &F,
        columns_opened: &[(usize, Vec<Int<K>>)],
        transcript: &mut PcsTranscript,
    ) -> Result<(), ZipError>
    where
        F: PrimeField + for<'a> From<&'a Int<L>> + for<'a> From<&'a Int<K>>,
        F::Inner: Transcribable,
    {
        let q_0_combined_row = transcript.read_field_elements(vp.linear_code.row_len())?;
        let encoded_combined_row = vp.linear_code.encode_f(&q_0_combined_row);

        let (q_0, q_1) = point_to_tensor(vp.num_rows, point)?;

        if inner_product(&q_0_combined_row, &q_1) != *eval {
            return Err(ZipError::InvalidPcsOpen(
                "Evaluation consistency failure".into(),
            ));
        }
        for (column_idx, column_values) in columns_opened.iter() {
            Self::verify_proximity_q_0(
                &q_0,
                &encoded_combined_row,
                column_values,
                *column_idx,
                vp.num_rows,
            )?;
        }

        Ok(())
    }

    fn verify_proximity_q_0<F>(
        q_0: &Vec<F>,
        encoded_q_0_combined_row: &[F],
        column_entries: &[Int<K>],
        column: usize,
        num_rows: usize,
    ) -> Result<(), ZipError>
    where
        F: PrimeField + for<'b> From<&'b Int<K>>,
    {
        let column_entries_comb = if num_rows > 1 {
            let column_entries = column_entries.iter().map(F::from).collect_vec();
            inner_product(q_0, &column_entries)
            // TODO: this inner product is taking a long time.
        } else {
            F::from(&column_entries.first().expect("No column entries").resize())
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
    use crypto_bigint::{Random, U256, const_monty_params};
    use crypto_primitives::crypto_bigint_int::Int;
    use itertools::Itertools;
    use num_traits::{ConstOne, One};
    use rand::Rng;

    use crate::{
        ZipError,
        code::{DefaultLinearCodeSpec, LinearCode, raa::RaaCode},
        field::F256,
        pcs::{
            structs::{MultilinearZip, MultilinearZipCommitment, MultilinearZipParams},
            tests::MockTranscript,
        },
        pcs_transcript::PcsTranscript,
        poly::{dense::DenseMultilinearExtension, mle::MultilinearExtensionRand},
        transcript::KeccakTranscript,
        utils::WORD_FACTOR,
    };

    const INT_LIMBS: usize = WORD_FACTOR;
    const FIELD_LIMBS: usize = 4 * WORD_FACTOR;

    const N: usize = INT_LIMBS;
    const L: usize = INT_LIMBS * 2;
    const K: usize = INT_LIMBS * 4;
    const M: usize = INT_LIMBS * 8;

    const_monty_params!(
        ModP,
        U256,
        "0000000000000000000000000000000000000000B933426489189CB5B47D567F"
    );

    type F = F256<ModP>;
    type LC = RaaCode<N, L, K, M>;
    type TestZip = MultilinearZip<N, L, K, M, LC>;

    #[allow(clippy::type_complexity)]
    fn setup_full_protocol(
        num_vars: usize,
    ) -> (
        MultilinearZipParams<N, L, K, M, LC>,
        MultilinearZipCommitment,
        Vec<F>,
        F,
        Vec<u8>,
    ) {
        let poly_size = 1 << num_vars;
        let evaluations: Vec<_> = (0..poly_size as i32).map(Int::<INT_LIMBS>::from).collect();
        let poly = DenseMultilinearExtension::from_evaluations_vec(num_vars, evaluations);

        let mut keccak = MockTranscript::default();
        let linear_code = LC::new(&DefaultLinearCodeSpec, poly_size, &mut keccak);
        let pp = TestZip::setup(poly_size, linear_code);

        let (data, comm) = TestZip::commit(&pp, &poly).unwrap();

        let point_int: Vec<Int<INT_LIMBS>> =
            (0..num_vars).map(|i| Int::from(i as i32 + 2)).collect();
        let point_f: Vec<F> = point_int.iter().map(F::from).collect_vec();

        let mut prover_transcript = PcsTranscript::new();
        TestZip::open(&pp, &poly, &data, &point_f, &mut prover_transcript).unwrap();
        let proof = prover_transcript.into_proof();

        let eval: F = match poly.evaluate(&point_int) {
            None => panic!("failed to evaluate polynomial"),
            Some(p) => p,
        }
        .into();

        (pp, comm, point_f, eval, proof)
    }

    #[test]
    fn successful_verification_of_valid_proof() {
        let num_vars = 4;
        let (pp, comm, point_f, eval, proof) = setup_full_protocol(num_vars);

        let mut verifier_transcript = PcsTranscript::from_proof(&proof);
        let result = TestZip::verify(&pp, &comm, &point_f, &eval, &mut verifier_transcript);

        assert!(result.is_ok());
    }

    #[test]
    fn verification_fails_with_incorrect_evaluation() {
        let num_vars = 4;
        let (pp, comm, point_f, eval, proof) = setup_full_protocol(num_vars);

        let one = F::one();

        let incorrect_eval = eval + one;
        let mut verifier_transcript = PcsTranscript::from_proof(&proof);
        let result = TestZip::verify(
            &pp,
            &comm,
            &point_f,
            &incorrect_eval,
            &mut verifier_transcript,
        );

        assert!(result.is_err());
    }

    #[test]
    fn verification_fails_with_tampered_proof() {
        let num_vars = 4;
        let (pp, comm, point_f, eval, proof) = setup_full_protocol(num_vars);

        let mut tampered = proof.clone();
        tampered[0] ^= 0x01;

        let mut verifier_transcript = PcsTranscript::from_proof(&tampered);
        let result = TestZip::verify(&pp, &comm, &point_f, &eval, &mut verifier_transcript);

        assert!(result.is_err());
    }

    #[test]
    fn verification_fails_with_wrong_commitment() {
        let num_vars = 4;
        let (pp, _comm_poly1, point_f, eval, proof_poly1) = setup_full_protocol(num_vars);

        let different_evals: Vec<_> = (20..(20 + (1 << num_vars))).map(Int::from).collect();
        let poly2 = DenseMultilinearExtension::from_evaluations_vec(num_vars, different_evals);
        let (_, comm_poly2) = TestZip::commit(&pp, &poly2).unwrap();

        let mut transcript = PcsTranscript::from_proof(&proof_poly1);
        let result = TestZip::verify(&pp, &comm_poly2, &point_f, &eval, &mut transcript);

        assert!(result.is_err());
    }

    #[test]
    fn verification_fails_with_invalid_point_size() {
        let num_vars = 4;
        let (pp, comm, _point_f, eval, proof) = setup_full_protocol(num_vars);
        let mut invalid_point = vec![];
        for i in 0..=num_vars {
            invalid_point.push(F::from(100 + i as i32));
        }

        let mut transcript = PcsTranscript::from_proof(&proof);
        let result = TestZip::verify(&pp, &comm, &invalid_point, &eval, &mut transcript);

        assert!(matches!(result, Err(..)));
    }

    #[test]
    fn verification_fails_if_proximity_check_is_invalid() {
        let mut keccak = MockTranscript::default();
        let poly_size = 8; // row_len=4, num_rows=2 -> proximity checks are active
        let linear_code = LC::new(&DefaultLinearCodeSpec, poly_size, &mut keccak);
        let pp = TestZip::setup(poly_size, linear_code);

        let evaluations: Vec<_> = (0..poly_size as i32).map(Int::<INT_LIMBS>::from).collect();
        let n = 3;
        let mle = DenseMultilinearExtension::from_evaluations_slice(n, &evaluations);

        let (data, comm) = TestZip::commit(&pp, &mle).expect("commit should succeed");

        let point_int = [0i64, 0i64, 0i64]
            .into_iter()
            .map(Int::from)
            .collect::<Vec<_>>();
        let point: Vec<F> = point_int.iter().map(F::from).collect_vec();
        let eval = mle.evaluate(&point_int).unwrap().into();

        let mut prover_tr = PcsTranscript::new();
        TestZip::open(&pp, &mle, &data, &point, &mut prover_tr).expect("open should succeed");
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
        let res = TestZip::verify(&pp, &comm, &point, &eval, &mut ver_tr);

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
        fn evaluate_in_field(evaluations: &[Int<INT_LIMBS>], point: &[F]) -> F {
            let num_vars = point.len();
            assert_eq!(evaluations.len(), 1 << num_vars);
            let mut current_evals: Vec<F> = evaluations.iter().map(F::from).collect_vec();
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

        let mut rng = rand::rng();

        let n = 3;
        let poly_size = 1 << n;
        let mut keccak_transcript = KeccakTranscript::new();
        let linear_code: LC = LC::new(&DefaultLinearCodeSpec, poly_size, &mut keccak_transcript);
        let param = TestZip::setup(poly_size, linear_code);
        let evaluations: Vec<_> = (0..poly_size)
            .map(|_| Int::<INT_LIMBS>::from(rng.random::<i8>()))
            .collect();
        let mle = DenseMultilinearExtension::from_evaluations_slice(n, &evaluations);
        let point_int: Vec<_> = (0..n).map(|_| Int::<INT_LIMBS>::random(&mut rng)).collect();
        let point_f = point_int.into_iter().map(F::from).collect_vec();

        let (mut data, comm) = TestZip::commit(&param, &mle).unwrap();
        if !data.rows.is_empty() {
            data.rows[0] += Int::<{ 4 * INT_LIMBS }>::from(1);
        }

        let mut prover_transcript = PcsTranscript::new();
        TestZip::open(&param, &mle, &data, &point_f, &mut prover_transcript).unwrap();
        let proof = prover_transcript.into_proof();

        let mut verifier_transcript = PcsTranscript::from_proof(&proof);
        let eval = evaluate_in_field(&mle.evaluations, &point_f);
        let verification_result =
            TestZip::verify(&param, &comm, &point_f, &eval, &mut verifier_transcript);

        assert!(verification_result.is_err());
    }

    #[test]
    fn verification_fails_if_evaluation_consistency_check_is_invalid() {
        let mut keccak = MockTranscript::default();
        let poly_size = 8;
        let linear_code = LC::new(&DefaultLinearCodeSpec, poly_size, &mut keccak);
        let pp = TestZip::setup(poly_size, linear_code);

        let evaluations: Vec<_> = (0..poly_size as i32).map(Int::<INT_LIMBS>::from).collect();
        let n = 3;
        let mle = DenseMultilinearExtension::from_evaluations_slice(n, &evaluations);

        let (data, comm) = TestZip::commit(&pp, &mle).expect("commit should succeed");

        let point_int = [0i64, 0i64, 0i64]
            .into_iter()
            .map(Int::from)
            .collect::<Vec<_>>();
        let point: Vec<F> = point_int.iter().map(F::from).collect_vec();
        let eval = mle.evaluate(&point_int).unwrap().into();

        let mut prover_tr = PcsTranscript::new();
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
        let res = TestZip::verify(&pp, &comm, &point, &eval, &mut ver_tr);

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
        let linear_code = LC::new(&DefaultLinearCodeSpec, poly_size, &mut keccak);
        let pp = TestZip::setup(poly_size, linear_code);

        let evaluations: Vec<_> = vec![Int::<INT_LIMBS>::from(0); poly_size];
        let n = 3;
        let mle = DenseMultilinearExtension::from_evaluations_slice(n, &evaluations);

        let (data, comm) = TestZip::commit(&pp, &mle).expect("commit should succeed");

        let point_int = [0i64, 0i64, 0i64]
            .into_iter()
            .map(Int::from)
            .collect::<Vec<_>>();
        let point: Vec<F> = point_int.iter().map(F::from).collect_vec();
        let eval = mle.evaluate(&point_int).unwrap().into();
        let mut prover_tr = PcsTranscript::new();
        TestZip::open(&pp, &mle, &data, &point, &mut prover_tr).expect("open should succeed");
        let proof = prover_tr.into_proof();

        let mut ver_tr = PcsTranscript::from_proof(&proof);
        let res = TestZip::verify(&pp, &comm, &point, &eval, &mut ver_tr);
        assert!(res.is_ok());
    }

    #[test]
    fn verification_succeeds_at_zero_point() {
        let mut keccak = MockTranscript::default();
        let poly_size = 8;
        let linear_code = LC::new(&DefaultLinearCodeSpec, poly_size, &mut keccak);
        let pp = TestZip::setup(poly_size, linear_code);

        let evaluations: Vec<_> = (1..=poly_size as i32).map(Int::<INT_LIMBS>::from).collect();
        let n = 3;
        let mle = DenseMultilinearExtension::from_evaluations_slice(n, &evaluations);

        let (data, comm) = TestZip::commit(&pp, &mle).expect("commit should succeed");

        let point_int = vec![Int::from(0i64); n];
        let point: Vec<F> = point_int.iter().map(F::from).collect_vec();

        let eval = mle.evaluate(&point_int).unwrap().into();

        let mut prover_tr = PcsTranscript::new();
        TestZip::open(&pp, &mle, &data, &point, &mut prover_tr).expect("open should succeed");
        let proof = prover_tr.into_proof();

        let mut ver_tr = PcsTranscript::from_proof(&proof);
        let res = TestZip::verify(&pp, &comm, &point, &eval, &mut ver_tr);
        assert!(res.is_ok());
    }

    #[test]
    fn verification_fails_if_proximity_values_are_too_large() {
        let mut keccak = MockTranscript::default();
        let poly_size = 8;
        let linear_code = LC::new(&DefaultLinearCodeSpec, poly_size, &mut keccak);
        let pp = TestZip::setup(poly_size, linear_code);

        let evaluations: Vec<_> = (1..=poly_size as i32).map(Int::<INT_LIMBS>::from).collect();
        let n = 3;
        let mle = DenseMultilinearExtension::from_evaluations_slice(n, &evaluations);

        let (data, comm) = TestZip::commit(&pp, &mle).expect("commit should succeed");

        let point_int = [0i64, 0i64, 0i64]
            .into_iter()
            .map(Int::from)
            .collect::<Vec<_>>();
        let point: Vec<F> = point_int.iter().map(F::from).collect_vec();
        let eval = mle.evaluate(&point_int).unwrap().into();

        let mut prover_tr = PcsTranscript::new();
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
        let res = TestZip::verify(&pp, &comm, &point, &eval, &mut ver_tr);
        assert!(res.is_err());
    }

    /// Mirrors: `Zip/Verify: RandomField<4>, poly_size = 2^12 (Int limbs = 1)`
    #[test]
    fn bench_p12_verify() {
        fn inner<const P: usize>() {
            let mut rng = rand::rng();
            // Match the benchmark’s transcript usage for linear code construction
            let mut keccak_transcript = KeccakTranscript::new();
            let poly_size = 1 << P;
            let linear_code = LC::new(&DefaultLinearCodeSpec, poly_size, &mut keccak_transcript);
            let params = TestZip::setup(poly_size, linear_code);

            let poly = DenseMultilinearExtension::rand(P, &mut rng);
            let (data, commitment) = TestZip::commit(&params, &poly).expect("commit");

            // Same point choice as the bench
            let point = vec![1i64; P];
            let eval = *poly.evaluations.last().expect("nonempty evals");

            // Prover produces a proof once (exactly as in the bench)
            let mut prover_tx = PcsTranscript::new();
            TestZip::open(
                &params,
                &poly,
                &data,
                &point.iter().map(F::from).collect::<Vec<_>>(),
                &mut prover_tx,
            )
            .expect("open");
            let proof = prover_tx.into_proof();

            // Verifier replays verification from the same proof (also like the bench)
            let mut verifier_tx = PcsTranscript::from_proof(&proof);
            TestZip::verify(
                &params,
                &commitment,
                &point.iter().map(F::from).collect::<Vec<_>>(),
                &eval.into(),
                &mut verifier_tx,
            )
            .expect("verify");
        }

        inner::<12>();
    }
}
