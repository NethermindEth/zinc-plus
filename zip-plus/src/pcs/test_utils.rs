#![allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::type_complexity
)]

use crate::{
    code::{LinearCode, raa::RaaCode},
    pcs::structs::{
        MulByScalar, ProjectableToField, ZipPlus, ZipPlusCommitment, ZipPlusParams, ZipTypes,
    },
    pcs_transcript::{PcsTranscript, ZipPlusEvaluationProof},
    poly::{dense::DensePolynomial, mle::DenseMultilinearExtension},
    traits::{FromRef, Transcribable, Transcript},
};
use crypto_primitives::{PrimeField, crypto_bigint_int::Int};
use itertools::Itertools;

const REPETITION_FACTOR: usize = 4;

pub struct TestZipTypes<const N: usize, const K: usize, const M: usize> {}
impl<const N: usize, const K: usize, const M: usize> ZipTypes for TestZipTypes<N, K, M> {
    const NUM_COLUMN_OPENINGS: usize = 650;
    type EvalR = Int<N>;
    type Eval = Int<N>;
    type CwR = Int<K>;
    type Cw = Int<K>;
    type Chal = Int<N>;
    type Pt = Int<N>;
    type CombR = Int<M>;
    type Comb = Int<M>;
}

pub struct TestPolyZipTypes<const N: usize, const K: usize, const M: usize, const DEGREE: usize> {
}
impl<const N: usize, const K: usize, const M: usize, const DEGREE: usize> ZipTypes
    for TestPolyZipTypes<N, K, M, DEGREE>
{
    const NUM_COLUMN_OPENINGS: usize = 650;
    type EvalR = Int<N>;
    type Eval = DensePolynomial<Int<N>, DEGREE>;
    type CwR = Int<K>;
    type Cw = DensePolynomial<Int<K>, DEGREE>;
    type Chal = Int<N>;
    type Pt = Int<N>;
    type CombR = Int<M>;
    type Comb = DensePolynomial<Int<M>, DEGREE>;
}

/// Helper function to set up common parameters for tests.
pub fn setup_test_params<const N: usize, const K: usize, const M: usize>(
    num_vars: usize,
) -> (
    ZipPlusParams<TestZipTypes<N, K, M>, RaaCode<TestZipTypes<N, K, M>, REPETITION_FACTOR>>,
    DenseMultilinearExtension<Int<N>>,
) {
    setup_test_params_inner(num_vars, |poly_size| {
        (1..=poly_size as i32).map(Int::from).collect()
    })
}

/// Helper function to set up common parameters for tests.
pub fn setup_poly_test_params<
    const N: usize,
    const K: usize,
    const M: usize,
    const DEGREE: usize,
>(
    num_vars: usize,
) -> (
    ZipPlusParams<
        TestPolyZipTypes<N, K, M, DEGREE>,
        RaaCode<TestPolyZipTypes<N, K, M, DEGREE>, REPETITION_FACTOR>,
    >,
    DenseMultilinearExtension<DensePolynomial<Int<N>, DEGREE>>,
) {
    setup_test_params_inner(num_vars, |poly_size| {
        let eval_coeffs: Vec<_> = (1..=(poly_size * DEGREE) as i32)
            .map(Int::from)
            .collect_vec();
        eval_coeffs
            .chunks_exact(DEGREE)
            .map(DensePolynomial::new)
            .collect_vec()
    })
}

fn setup_test_params_inner<Zt: ZipTypes, Lc: LinearCode<Zt>>(
    num_vars: usize,
    prepare_evaluations: impl FnOnce(usize) -> Vec<Zt::Eval>,
) -> (ZipPlusParams<Zt, Lc>, DenseMultilinearExtension<Zt::Eval>) {
    let poly_size = 1 << num_vars;
    let num_rows = 1 << num_vars.div_ceil(2);

    let mut transcript = MockTranscript::default();
    let code = Lc::new(poly_size, true, &mut transcript);
    let pp = ZipPlusParams::new(num_vars, num_rows, code);

    let evaluations = prepare_evaluations(poly_size);
    let poly = DenseMultilinearExtension::from_evaluations_vec(num_vars, evaluations);

    (pp, poly)
}

pub fn setup_full_protocol<F, const N: usize, const K: usize, const M: usize>(
    num_vars: usize,
) -> (
    ZipPlusParams<TestZipTypes<N, K, M>, RaaCode<TestZipTypes<N, K, M>, REPETITION_FACTOR>>,
    ZipPlusCommitment,
    Vec<F>,
    F,
    ZipPlusEvaluationProof,
)
where
    F: PrimeField + FromRef<Int<N>> + for<'a> MulByScalar<&'a F>,
    F::Inner: Transcribable,
    Int<N>: ProjectableToField<F>,
{
    setup_full_protocol_inner::<_, _, _, N>(num_vars, setup_test_params, || {
        (0..num_vars).map(|i| Int::from(i as i32 + 2)).collect()
    })
}

pub fn setup_full_protocol_poly<
    F,
    const N: usize,
    const K: usize,
    const M: usize,
    const DEGREE: usize,
>(
    num_vars: usize,
) -> (
    ZipPlusParams<
        TestPolyZipTypes<N, K, M, DEGREE>,
        RaaCode<TestPolyZipTypes<N, K, M, DEGREE>, REPETITION_FACTOR>,
    >,
    ZipPlusCommitment,
    Vec<F>,
    F,
    ZipPlusEvaluationProof,
)
where
    F: PrimeField + FromRef<Int<N>> + for<'a> MulByScalar<&'a F>,
    F::Inner: Transcribable,
    DensePolynomial<Int<N>, DEGREE>: ProjectableToField<F>,
{
    setup_full_protocol_inner::<_, _, _, N>(num_vars, setup_poly_test_params, || {
        (0..num_vars).map(|i| Int::from(i as i32 + 2)).collect()
    })
}

fn setup_full_protocol_inner<Zt, Lc, F, const N: usize>(
    num_vars: usize,
    setup: impl FnOnce(usize) -> (ZipPlusParams<Zt, Lc>, DenseMultilinearExtension<Zt::Eval>),
    prepare_evaluation_point: impl FnOnce() -> Vec<Zt::Pt>,
) -> (
    ZipPlusParams<Zt, Lc>,
    ZipPlusCommitment,
    Vec<F>,
    F,
    ZipPlusEvaluationProof,
)
where
    Zt: ZipTypes,
    Zt::Eval: for<'a> MulByScalar<&'a Zt::Pt>,
    Lc: LinearCode<Zt>,
    F: PrimeField + FromRef<Zt::Chal> + FromRef<Zt::Pt> + for<'a> MulByScalar<&'a F>,
    F::Inner: Transcribable,
    Zt::Eval: ProjectableToField<F>,
{
    let (pp, poly) = setup(num_vars);

    let (data, comm) = ZipPlus::commit(&pp, &poly).unwrap();

    let point: Vec<Zt::Pt> = prepare_evaluation_point();

    let test_proof = ZipPlus::test(&pp, &poly, &data).unwrap();

    let projecting_element: F = {
        let mut transcript: PcsTranscript = test_proof.clone().into();
        let projecting_element: Zt::Chal = transcript.fs_transcript.get_challenge();
        F::from_ref(&projecting_element)
    };

    let (eval_f, eval_proof) = ZipPlus::evaluate(&pp, &poly, &point, test_proof).unwrap();

    // Verify the evaluation is done correctly
    {
        let expected_eval = poly
            .evaluate(&point)
            .expect("failed to evaluate polynomial");
        let project = Zt::Eval::prepare_projection(&projecting_element);
        assert_eq!(eval_f, project(&expected_eval));
    }

    let point_f = point.iter().map(F::from_ref).collect_vec();

    (pp, comm, point_f, eval_f, eval_proof)
}

#[derive(Default)]
pub struct MockTranscript {
    pub counter: i64,
}

impl Transcript for MockTranscript {
    fn get_challenge<T: Transcribable>(&mut self) -> T {
        self.counter += 1;
        let mut bytes = vec![0u8; T::NUM_BYTES];
        let counter_bytes = self.counter.to_le_bytes();
        bytes[..counter_bytes.len()].copy_from_slice(&counter_bytes);
        T::read_transcription_bytes(&bytes)
    }
}
