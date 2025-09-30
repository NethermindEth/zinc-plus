#![allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::type_complexity
)]

use crate::{
    code::{DefaultLinearCodeSpec, LinearCode, raa::RaaCode},
    pcs::structs::{
        AsPackable, MulByScalar, ProjectableToField, ZipPlus, ZipPlusCommitment, ZipPlusParams,
        ZipTypes,
    },
    pcs_transcript::PcsTranscript,
    poly::{Polynomial, dense::DensePolynomial, mle::DenseMultilinearExtension},
    traits::{FromRef, Transcribable, Transcript},
};
use crypto_primitives::{PrimeField, Ring, crypto_bigint_int::Int};
use itertools::Itertools;

pub struct TestZipTypes<const N: usize, const K: usize, const M: usize> {}
impl<const N: usize, const K: usize, const M: usize> ZipTypes for TestZipTypes<N, K, M> {
    type EvalR = Int<N>;
    type Eval = Int<N>;
    type CwR = Int<K>;
    type Cw = Int<K>;
    type Chal = Int<N>;
    type Pt = Int<N>;
    type CombR = Int<M>;
    type Comb = Int<M>;
    type Code = RaaCode<Int<N>, Int<K>, Int<M>>;
}

pub struct TestPolyZipTypes<const N: usize, const K: usize, const M: usize, const DEGREE: usize> {
}
impl<const N: usize, const K: usize, const M: usize, const DEGREE: usize> ZipTypes
    for TestPolyZipTypes<N, K, M, DEGREE>
{
    type EvalR = Int<N>;
    type Eval = DensePolynomial<Int<N>, DEGREE>;
    type CwR = Int<K>;
    type Cw = DensePolynomial<Int<K>, DEGREE>;
    type Chal = Int<N>;
    type Pt = Int<N>;
    type CombR = Int<M>;
    type Comb = DensePolynomial<Int<M>, DEGREE>;
    type Code = RaaCode<DensePolynomial<Int<N>, DEGREE>, DensePolynomial<Int<K>, DEGREE>, Int<M>>;
}

/// Helper function to set up common parameters for tests.
pub fn setup_test_params<const N: usize, const K: usize, const M: usize>(
    num_vars: usize,
) -> (
    ZipPlusParams<TestZipTypes<N, K, M>>,
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
    ZipPlusParams<TestPolyZipTypes<N, K, M, DEGREE>>,
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

fn setup_test_params_inner<Zt: ZipTypes>(
    num_vars: usize,
    prepare_evaluations: impl FnOnce(usize) -> Vec<Zt::Eval>,
) -> (ZipPlusParams<Zt>, DenseMultilinearExtension<Zt::Eval>) {
    let poly_size = 1 << num_vars;
    let num_rows = 1 << num_vars.div_ceil(2);

    let mut transcript = MockTranscript::default();
    let code =
        <Zt as ZipTypes>::Code::new(&DefaultLinearCodeSpec, poly_size, true, &mut transcript);
    let pp = ZipPlusParams::new(num_vars, num_rows, code);

    let evaluations = prepare_evaluations(poly_size);
    let poly = DenseMultilinearExtension::from_evaluations_vec(num_vars, evaluations);

    (pp, poly)
}

pub fn setup_full_protocol<F, const N: usize, const K: usize, const M: usize>(
    num_vars: usize,
) -> (
    ZipPlusParams<TestZipTypes<N, K, M>>,
    ZipPlusCommitment,
    Vec<F>,
    F,
    Vec<u8>,
)
where
    F: PrimeField + FromRef<Int<N>> + for<'a> MulByScalar<&'a F>,
    F::Inner: Transcribable,
    Int<N>: ProjectableToField<F>,
{
    setup_full_protocol_inner::<_, _, _, _, _, _, N>(num_vars, setup_test_params, || {
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
    ZipPlusParams<TestPolyZipTypes<N, K, M, DEGREE>>,
    ZipPlusCommitment,
    Vec<F>,
    F,
    Vec<u8>,
)
where
    F: PrimeField + FromRef<Int<N>> + for<'a> MulByScalar<&'a F>,
    F::Inner: Transcribable,
    DensePolynomial<Int<N>, DEGREE>: ProjectableToField<F>,
{
    setup_full_protocol_inner::<_, _, _, _, _, _, N>(num_vars, setup_poly_test_params, || {
        (0..num_vars).map(|i| Int::from(i as i32 + 2)).collect()
    })
}

fn setup_full_protocol_inner<Eval, Cw, Pt, CombR, Zt, F, const N: usize>(
    num_vars: usize,
    setup: impl FnOnce(usize) -> (ZipPlusParams<Zt>, DenseMultilinearExtension<Eval>),
    prepare_evaluation_point: impl FnOnce() -> Vec<Pt>,
) -> (ZipPlusParams<Zt>, ZipPlusCommitment, Vec<F>, F, Vec<u8>)
where
    Eval: Ring + for<'a> MulByScalar<&'a Pt>,
    Cw: Ring + FromRef<Eval> + AsPackable + Transcribable,
    CombR: Ring + FromRef<CombR> + Transcribable,
    Zt: ZipTypes<Eval = Eval, Cw = Cw, Pt = Pt, CombR = CombR, Code = RaaCode<Eval, Cw, CombR>>,
    F: PrimeField + FromRef<Zt::Chal> + FromRef<Zt::Pt> + for<'a> MulByScalar<&'a F>,
    F::Inner: Transcribable,
    Eval: ProjectableToField<F>,
{
    let (pp, poly) = setup(num_vars);

    let (data, comm) = ZipPlus::commit(&pp, &poly).unwrap();

    let point: Vec<Pt> = prepare_evaluation_point();

    let mut prover_transcript = PcsTranscript::new();
    let eval_f = ZipPlus::open(&pp, &poly, &data, &point, &mut prover_transcript).unwrap();

    let proof = prover_transcript.into_proof();

    // Verify the evaluation is done correctly
    {
        let expected_eval = poly
            .evaluate(&point)
            .expect("failed to evaluate polynomial");
        let project = Eval::prepare_projection(&read_field_projecting_element(&pp, &proof));
        assert_eq!(eval_f, project(&expected_eval));
    }

    let point_f = point.iter().map(F::from_ref).collect_vec();

    (pp, comm, point_f, eval_f, proof)
}

pub fn read_field_projecting_element<Zt, F>(pp: &ZipPlusParams<Zt>, proof: &[u8]) -> F
where
    Zt: ZipTypes,
    Zt::Eval: ProjectableToField<F>,
    F: PrimeField + FromRef<Zt::Chal>,
{
    let mut transcript = PcsTranscript::from_proof(proof);
    // Advance the transcript to the point where we can get the projecting element
    // TODO: This shouldn't be necessary after we split testing and evaluation
    // phases
    if pp.num_rows > 1 {
        for _ in 0..pp.linear_code.num_proximity_testing() {
            let _ = transcript
                .read_many::<Zt::CombR>(pp.linear_code.row_len())
                .unwrap();
            let _ = transcript
                .fs_transcript
                .get_challenges::<Zt::Chal>(Zt::Comb::DEGREE_BOUND);
            let _ = transcript
                .fs_transcript
                .get_challenges::<Zt::Chal>(pp.num_rows);
        }
    }
    for _ in 0..pp.linear_code.num_column_opening() {
        let _ = transcript.squeeze_challenge_idx(pp.linear_code.codeword_len());
        let _ = transcript.read_many::<Zt::Cw>(pp.num_rows).unwrap();
        let _ = transcript.read_merkle_proof().unwrap();
    }
    let projecting_element: Zt::Chal = transcript.fs_transcript.get_challenge();
    F::from_ref(&projecting_element)
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
