use crate::{
    ZipError,
    code::LinearCode,
    merkle::MtHash,
    pcs::{
        structs::{
            ChallengeRing, CodewordRing, EvaluationRing, LinearCombinationRing, MulByScalar,
            MultilinearZip, MultilinearZipCommitment, MultilinearZipParams,
        },
        utils::{ColumnOpening, point_to_tensor, validate_input},
    },
    pcs_transcript::PcsTranscript,
    poly::mle::DenseMultilinearExtension,
    traits::{FromRef, Transcribable, Transcript},
    utils::inner_product,
};
use ark_std::iterable::Iterable;
use crypto_primitives::PrimeField;
use itertools::Itertools;

impl<
    Eval: EvaluationRing,
    Cw: CodewordRing,
    Chal: ChallengeRing,
    Comb: LinearCombinationRing<Eval, Cw, Chal>,
    C: LinearCode<Eval, Cw, Comb>,
> MultilinearZip<Eval, Cw, Chal, Comb, C>
{
    pub fn verify<F>(
        vp: &MultilinearZipParams<Eval, Cw, Comb, C>,
        comm: &MultilinearZipCommitment,
        point: &[F],
        eval: &F,
        transcript: &mut PcsTranscript,
    ) -> Result<(), ZipError>
    where
        F: PrimeField + FromRef<F> + FromRef<C::Inner> + FromRef<Cw> + for<'a> MulByScalar<&'a F>,
        F::Inner: Transcribable,
    {
        let no_polys = Vec::<DenseMultilinearExtension<bool>>::new();
        validate_input("verify", vp.num_vars, &no_polys, [point])?;

        let columns_opened = Self::verify_testing(vp, &comm.root, transcript)?;

        Self::verify_evaluation_z(vp, point, eval, &columns_opened, transcript)?;

        Ok(())
    }

    pub fn batch_verify_z<'a, F>(
        vp: &MultilinearZipParams<Eval, Cw, Comb, C>,
        comms: impl Iterable<Item = &'a MultilinearZipCommitment>,
        points: &[Vec<F>],
        evals: &[F],
        transcript: &mut PcsTranscript,
    ) -> Result<(), ZipError>
    where
        F: PrimeField + FromRef<F> + FromRef<C::Inner> + FromRef<Cw> + for<'b> MulByScalar<&'b F>,
        F::Inner: Transcribable,
    {
        for (i, (eval, comm)) in evals.iter().zip(comms.iter()).enumerate() {
            Self::verify(vp, comm, &points[i], eval, transcript)?;
        }
        Ok(())
    }

    #[allow(clippy::type_complexity)]
    pub(super) fn verify_testing(
        vp: &MultilinearZipParams<Eval, Cw, Comb, C>,
        root: &MtHash,
        transcript: &mut PcsTranscript,
    ) -> Result<Vec<(usize, Vec<Cw>)>, ZipError> {
        // Gather the coeffs and encoded combined rows per proximity test
        let mut encoded_combined_rows: Vec<(Vec<Chal>, Vec<Comb>)> =
            Vec::with_capacity(vp.linear_code.num_proximity_testing());

        if vp.num_rows > 1 {
            for _ in 0..vp.linear_code.num_proximity_testing() {
                let coeffs = transcript.fs_transcript.get_challenges(vp.num_rows);

                let combined_row: Vec<Comb> = transcript.read_many(vp.linear_code.row_len())?;

                let encoded_combined_row: Vec<Comb> = vp.linear_code.encode_wide(&combined_row);
                encoded_combined_rows.push((coeffs, encoded_combined_row));
            }
        }

        let mut columns_opened: Vec<(usize, Vec<Cw>)> =
            Vec::with_capacity(vp.linear_code.num_column_opening());

        for _ in 0..vp.linear_code.num_column_opening() {
            let column_idx = transcript.squeeze_challenge_idx(vp.linear_code.codeword_len());
            let column_values = transcript.read_many(vp.num_rows)?;

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
        coeffs: &[Chal],
        encoded_combined_row: &[Comb],
        column_entries: &[Cw],
        column: usize,
        num_rows: usize,
    ) -> Result<(), ZipError> {
        let column_entries_comb: Comb = if num_rows > 1 {
            let column_entries: Vec<Comb> = column_entries.iter().map(Comb::from_ref).collect();
            inner_product(coeffs.iter(), column_entries.iter())
        } else {
            Comb::from_ref(&column_entries[0])
        };

        if column_entries_comb != encoded_combined_row[column] {
            return Err(ZipError::InvalidPcsOpen("Proximity failure".into()));
        }
        Ok(())
    }

    fn verify_evaluation_z<F>(
        vp: &MultilinearZipParams<Eval, Cw, Comb, C>,
        point: &[F],
        eval: &F,
        columns_opened: &[(usize, Vec<Cw>)],
        transcript: &mut PcsTranscript,
    ) -> Result<(), ZipError>
    where
        F: PrimeField + FromRef<F> + FromRef<C::Inner> + FromRef<Cw> + for<'a> MulByScalar<&'a F>,
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
        column_entries: &[Cw],
        column: usize,
        num_rows: usize,
    ) -> Result<(), ZipError>
    where
        F: PrimeField + FromRef<Cw> + for<'a> MulByScalar<&'a F>,
    {
        let column_entries_comb = if num_rows > 1 {
            let column_entries = column_entries.iter().map(F::from_ref).collect_vec();
            inner_product(q_0, &column_entries)
            // TODO: this inner product is taking a long time.
        } else {
            F::from_ref(column_entries.first().expect("No column entries"))
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
    use rand::prelude::*;

    use crate::{
        ZipError,
        code::{DefaultLinearCodeSpec, LinearCode, raa::RaaCode},
        field::F256,
        pcs::{
            structs::{MultilinearZip, MultilinearZipCommitment, MultilinearZipParams},
            tests::MockTranscript,
        },
        pcs_transcript::PcsTranscript,
        poly::{
            dense::DensePolynomial,
            mle::{DenseMultilinearExtension, MultilinearExtensionRand},
        },
        transcript::KeccakTranscript,
        utils::WORD_FACTOR,
    };

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
    type C = RaaCode<Int<N>, Int<K>, Int<M>>;
    type PolyC = RaaCode<
        DensePolynomial<Int<N>, DEGREE>,
        DensePolynomial<Int<K>, DEGREE>,
        DensePolynomial<Int<M>, DEGREE>,
    >;
    type TestZip = MultilinearZip<Int<N>, Int<K>, Int<N>, Int<M>, C>;
    type TestPolyZip = MultilinearZip<
        DensePolynomial<Int<N>, DEGREE>,
        DensePolynomial<Int<K>, DEGREE>,
        Int<N>,
        DensePolynomial<Int<M>, DEGREE>,
        PolyC,
    >;

    #[allow(clippy::type_complexity)]
    fn setup_full_protocol(
        num_vars: usize,
    ) -> (
        MultilinearZipParams<Int<N>, Int<K>, Int<M>, C>,
        MultilinearZipCommitment,
        Vec<F>,
        F,
        Vec<u8>,
    ) {
        let poly_size = 1 << num_vars;
        let evaluations: Vec<_> = (0..poly_size as i32).map(Int::<N>::from).collect();
        let poly = DenseMultilinearExtension::from_evaluations_vec(num_vars, evaluations);

        let mut keccak = MockTranscript::default();
        let linear_code = C::new(&DefaultLinearCodeSpec, poly_size, true, &mut keccak);
        let pp = TestZip::setup(poly_size, linear_code);

        let (data, comm) = TestZip::commit(&pp, &poly).unwrap();

        let point_int: Vec<Int<N>> = (0..num_vars).map(|i| Int::from(i as i32 + 2)).collect();
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

    #[allow(clippy::type_complexity)]
    fn setup_full_protocol_poly(
        num_vars: usize,
    ) -> (
        MultilinearZipParams<
            DensePolynomial<Int<N>, DEGREE>,
            DensePolynomial<Int<K>, DEGREE>,
            DensePolynomial<Int<M>, DEGREE>,
            PolyC,
        >,
        MultilinearZipCommitment,
        Vec<F>,
        F,
        Vec<u8>,
    ) {
        let poly_size = 1 << num_vars;
        let eval_coeffs: Vec<_> = (0..poly_size as i32).map(Int::<N>::from).collect_vec();
        let evaluations = eval_coeffs
            .windows(DEGREE + 1)
            .map(DensePolynomial::new)
            .collect_vec();
        let poly = DenseMultilinearExtension::from_evaluations_vec(num_vars, evaluations);

        let mut keccak = MockTranscript::default();
        let linear_code = PolyC::new(&DefaultLinearCodeSpec, poly_size, true, &mut keccak);
        let pp = TestPolyZip::setup(poly_size, linear_code);

        let (data, comm) = TestPolyZip::commit(&pp, &poly).unwrap();

        let point_poly: Vec<DensePolynomial<Int<N>, DEGREE>> = (0..num_vars)
            .map(|i| {
                let start = i as i32 + 2;
                let vec = (start..=(start + DEGREE as i32))
                    .map(Int::<N>::from)
                    .collect_vec();
                DensePolynomial::new(vec)
            })
            .collect();
        let point_f: Vec<F> = point_poly.iter().map(F::from).collect_vec();

        let mut prover_transcript = PcsTranscript::new();
        TestPolyZip::open(&pp, &poly, &data, &point_f, &mut prover_transcript).unwrap();
        let proof = prover_transcript.into_proof();

        let eval: F = match poly.evaluate(&point_poly) {
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
    fn successful_verification_of_valid_proof_poly() {
        let num_vars = 4;
        let (pp, comm, point_f, eval, proof) = setup_full_protocol_poly(num_vars);

        let mut verifier_transcript = PcsTranscript::from_proof(&proof);
        let result = TestPolyZip::verify(&pp, &comm, &point_f, &eval, &mut verifier_transcript);

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
        let linear_code = C::new(&DefaultLinearCodeSpec, poly_size, true, &mut keccak);
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

        let mut rng = ThreadRng::default();

        let n = 3;
        let poly_size = 1 << n;
        let mut keccak_transcript = KeccakTranscript::new();
        let linear_code: C = C::new(
            &DefaultLinearCodeSpec,
            poly_size,
            true,
            &mut keccak_transcript,
        );
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
        let linear_code = C::new(&DefaultLinearCodeSpec, poly_size, true, &mut keccak);
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
        let linear_code = C::new(&DefaultLinearCodeSpec, poly_size, true, &mut keccak);
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
        let linear_code = C::new(&DefaultLinearCodeSpec, poly_size, true, &mut keccak);
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
        let linear_code = C::new(&DefaultLinearCodeSpec, poly_size, true, &mut keccak);
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
            let mut rng = ThreadRng::default();
            // Match the benchmark’s transcript usage for linear code construction
            let mut keccak_transcript = KeccakTranscript::new();
            let poly_size = 1 << P;
            let linear_code = C::new(
                &DefaultLinearCodeSpec,
                poly_size,
                true,
                &mut keccak_transcript,
            );
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
