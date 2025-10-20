#![allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::type_complexity
)]

use crate::{
    code::{LinearCode, raa::RaaCode},
    pcs::{
        ZipPlusProof,
        structs::{
            MulByScalar, ProjectableToField, ZipPlus, ZipPlusCommitment, ZipPlusParams, ZipTypes,
        },
    },
    pcs_transcript::PcsTranscript,
    poly::{dense::DensePolynomial, mle::DenseMultilinearExtension},
    traits::{FromRef, Transcribable, Transcript},
    utils::UintSemiring,
};
use crypto_bigint::{BoxedUint, Uint};
use crypto_primes::hazmat::MillerRabin;
use crypto_primitives::{
    FromWithConfig, IntoWithConfig, PrimeField, crypto_bigint_boxed_monty::BoxedMontyField,
    crypto_bigint_int::Int,
};
use itertools::Itertools;
use num_traits::Zero;

const REPETITION_FACTOR: usize = 4;

pub struct TestZipTypes<const N: usize, const K: usize, const M: usize> {}
impl<const N: usize, const K: usize, const M: usize> ZipTypes for TestZipTypes<N, K, M> {
    const NUM_COLUMN_OPENINGS: usize = 650;
    type EvalR = Int<N>;
    type Eval = Int<N>;
    type CwR = Int<K>;
    type Cw = Int<K>;
    type Fmod = UintSemiring<K>;
    type PrimeTest = MillerRabin<Uint<K>>;
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
    type Fmod = UintSemiring<K>;
    type PrimeTest = MillerRabin<Uint<K>>;
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

    let code = Lc::new(poly_size, true);
    let pp = ZipPlusParams::new(num_vars, num_rows, code);

    let evaluations = prepare_evaluations(poly_size);
    let poly = DenseMultilinearExtension::from_evaluations_vec(num_vars, evaluations, Zero::zero());

    (pp, poly)
}

pub fn setup_full_protocol<F, const N: usize, const K: usize, const M: usize>(
    num_vars: usize,
) -> (
    ZipPlusParams<TestZipTypes<N, K, M>, RaaCode<TestZipTypes<N, K, M>, REPETITION_FACTOR>>,
    ZipPlusCommitment,
    Vec<F>,
    F,
    ZipPlusProof,
)
where
    F: PrimeField + for<'a> FromWithConfig<&'a Int<N>> + for<'a> MulByScalar<&'a F>,
    F::Inner: FromRef<UintSemiring<K>> + Transcribable,
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
    ZipPlusProof,
)
where
    F: PrimeField + for<'a> FromWithConfig<&'a Int<N>> + for<'a> MulByScalar<&'a F>,
    F::Inner: FromRef<UintSemiring<K>> + Transcribable,
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
    ZipPlusProof,
)
where
    Zt: ZipTypes,
    Zt::Eval: for<'a> MulByScalar<&'a Zt::Pt>,
    Lc: LinearCode<Zt>,
    F: PrimeField
        + for<'a> FromWithConfig<&'a Zt::Chal>
        + for<'a> FromWithConfig<&'a Zt::Pt>
        + for<'a> MulByScalar<&'a F>,
    F::Inner: FromRef<Zt::Fmod> + Transcribable,
    Zt::Eval: ProjectableToField<F>,
{
    let (pp, poly) = setup(num_vars);

    let (data, comm) = ZipPlus::commit(&pp, &poly).unwrap();

    let point: Vec<Zt::Pt> = prepare_evaluation_point();

    let transcript = ZipPlus::test(&pp, &poly, &data).unwrap();

    let (field_cfg, projecting_element) = {
        let mut transcript: PcsTranscript = transcript.clone().into();
        let field_modulus = F::Inner::from_ref(
            &transcript
                .fs_transcript
                .get_prime::<Zt::Fmod, Zt::PrimeTest>(),
        );
        let field_cfg = F::make_cfg(&field_modulus).unwrap();
        let projecting_element: Zt::Chal = transcript.fs_transcript.get_challenge();
        let projecting_element: F = (&projecting_element).into_with_cfg(&field_cfg);
        (field_cfg, projecting_element)
    };

    let (eval_f, proof) = ZipPlus::evaluate(&pp, &poly, &point, transcript).unwrap();

    // Verify the evaluation is done correctly
    {
        let expected_eval = poly
            .evaluate(&point, Zero::zero())
            .expect("failed to evaluate polynomial");
        let project = Zt::Eval::prepare_projection(&projecting_element);
        assert_eq!(eval_f, project(&expected_eval));
    }

    let point_f = point
        .iter()
        .map(|v| v.into_with_cfg(&field_cfg))
        .collect_vec();

    (pp, comm, point_f, eval_f, proof)
}

pub fn get_dyn_config(hex_modulus: &str) -> <BoxedMontyField as PrimeField>::Config {
    let modulus =
        BoxedUint::from_str_radix_vartime(hex_modulus, 16).expect("Invalid modulus hex string");
    BoxedMontyField::make_cfg(&modulus).expect("Failed to create field config")
}
