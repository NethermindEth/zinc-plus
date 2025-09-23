#![allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::type_complexity
)]

use crate::{
    code::{DefaultLinearCodeSpec, raa::RaaCode},
    pcs::structs::{
        AsPackable, MulByScalar, MultilinearZipCommitment, MultilinearZipParams, ZipPlus, ZipTypes,
    },
    pcs_transcript::PcsTranscript,
    poly::{dense::DensePolynomial, mle::DenseMultilinearExtension},
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
    type CombR = Int<M>;
    type Comb = DensePolynomial<Int<M>, DEGREE>;
    type Code = RaaCode<
        DensePolynomial<Int<N>, DEGREE>,
        DensePolynomial<Int<K>, DEGREE>,
        DensePolynomial<Int<M>, DEGREE>,
    >;
}

/// Helper function to set up common parameters for tests.
pub fn setup_test_params<const N: usize, const K: usize, const M: usize>(
    num_vars: usize,
) -> (
    MultilinearZipParams<TestZipTypes<N, K, M>>,
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
    MultilinearZipParams<TestPolyZipTypes<N, K, M, DEGREE>>,
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

fn setup_test_params_inner<Eval, Cw, Comb, Zt>(
    num_vars: usize,
    prepare_evaluations: impl FnOnce(usize) -> Vec<Zt::Eval>,
) -> (
    MultilinearZipParams<Zt>,
    DenseMultilinearExtension<Zt::Eval>,
)
where
    Eval: Ring,
    Cw: Ring + FromRef<Eval>,
    Comb: Ring + FromRef<Comb>,
    Zt: ZipTypes<Eval = Eval, Cw = Cw, Comb = Comb, Code = RaaCode<Eval, Cw, Comb>>,
{
    let poly_size = 1 << num_vars;
    let num_rows = 1 << num_vars.div_ceil(2);

    let mut transcript = MockTranscript::default();
    let code = RaaCode::new(&DefaultLinearCodeSpec, poly_size, true, &mut transcript);
    let pp = MultilinearZipParams::new(num_vars, num_rows, code);

    let evaluations = prepare_evaluations(poly_size);
    let poly = DenseMultilinearExtension::from_evaluations_vec(num_vars, evaluations);

    (pp, poly)
}

pub fn setup_full_protocol<F, const N: usize, const K: usize, const M: usize>(
    num_vars: usize,
) -> (
    MultilinearZipParams<TestZipTypes<N, K, M>>,
    MultilinearZipCommitment,
    Vec<F>,
    F,
    Vec<u8>,
)
where
    F: PrimeField + FromRef<Int<N>> + for<'a> MulByScalar<&'a F>,
    F::Inner: Transcribable,
{
    setup_full_protocol_inner::<_, _, _, _, _, N>(num_vars, setup_test_params, || {
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
    MultilinearZipParams<TestPolyZipTypes<N, K, M, DEGREE>>,
    MultilinearZipCommitment,
    Vec<F>,
    F,
    Vec<u8>,
)
where
    F: PrimeField + FromRef<DensePolynomial<Int<N>, DEGREE>> + for<'a> MulByScalar<&'a F>,
    F::Inner: Transcribable,
{
    setup_full_protocol_inner::<_, _, _, _, _, N>(num_vars, setup_poly_test_params, || {
        (0..num_vars)
            .map(|i| {
                let start = i as i32 + 2;
                let vec = (start..=(start + DEGREE as i32))
                    .map(Int::<N>::from)
                    .collect_vec();
                DensePolynomial::new(vec)
            })
            .collect()
    })
}

fn setup_full_protocol_inner<Eval, Cw, Comb, Zt, F, const N: usize>(
    num_vars: usize,
    setup: impl FnOnce(usize) -> (MultilinearZipParams<Zt>, DenseMultilinearExtension<Eval>),
    prepare_evaluation_point: impl FnOnce() -> Vec<Eval>,
) -> (
    MultilinearZipParams<Zt>,
    MultilinearZipCommitment,
    Vec<F>,
    F,
    Vec<u8>,
)
where
    Eval: Ring,
    Cw: Ring + FromRef<Eval> + AsPackable,
    Comb: Ring + FromRef<Comb>,
    Zt: ZipTypes<Eval = Eval, Cw = Cw, Comb = Comb, Code = RaaCode<Eval, Cw, Comb>>,
    F: PrimeField + FromRef<Eval> + for<'a> MulByScalar<&'a F>,
    F::Inner: Transcribable,
{
    let (pp, poly) = setup(num_vars);

    let (data, comm) = ZipPlus::commit(&pp, &poly).unwrap();

    let point: Vec<Eval> = prepare_evaluation_point();
    let point_f: Vec<F> = point.iter().map(F::from_ref).collect_vec();

    let mut prover_transcript = PcsTranscript::new();
    ZipPlus::open(&pp, &poly, &data, &point_f, &mut prover_transcript).unwrap();
    let proof = prover_transcript.into_proof();

    let eval = F::from_ref(
        &poly
            .evaluate(&point)
            .expect("failed to evaluate polynomial"),
    );

    (pp, comm, point_f, eval, proof)
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
