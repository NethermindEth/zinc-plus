#![allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::type_complexity
)]

use crate::{
    code::{
        LinearCode,
        raa::{RaaCode, RaaConfig},
        raa_sign_flip::RaaSignFlippingCode,
    },
    pcs::{
        ZipPlusProof,
        structs::{ZipPlus, ZipPlusCommitment, ZipPlusParams, ZipTypes},
    },
    pcs_transcript::PcsTranscript,
};
use crypto_primitives::{
    FromWithConfig, IntSemiring, IntoWithConfig, PrimeField, boolean::Boolean,
    crypto_bigint_int::Int, crypto_bigint_uint::Uint,
};
use itertools::Itertools;
use num_traits::Zero;
use zinc_poly::{mle::DenseMultilinearExtension, univariate::dense::DensePolynomial};
use zinc_primality::MillerRabin;
use zinc_transcript::traits::{Transcribable, Transcript};
use zinc_utils::{
    from_ref::FromRef, mul_by_scalar::MulByScalar, projectable_to_field::ProjectableToField,
};

const REPETITION_FACTOR: usize = 4;

pub const RAA_CFG: RaaConfig = RaaConfig {
    check_for_overflows: true,
    permute_in_place: false,
};

pub struct TestZipTypes<const N: usize, const K: usize, const M: usize> {}
impl<const N: usize, const K: usize, const M: usize> ZipTypes for TestZipTypes<N, K, M> {
    const NUM_COLUMN_OPENINGS: usize = 650;
    type Eval = Int<N>;
    type Cw = Int<K>;
    type Fmod = Uint<K>;
    type PrimeTest = MillerRabin;
    type Chal = Int<N>;
    type Pt = Int<N>;
    type CombR = Int<M>;
    type Comb = Self::CombR;
}

pub struct TestPolyZipTypes<const K: usize, const M: usize, const DEGREE_PLUS_ONE: usize> {}
impl<const K: usize, const M: usize, const DEGREE_PLUS_ONE: usize> ZipTypes
    for TestPolyZipTypes<K, M, DEGREE_PLUS_ONE>
{
    const NUM_COLUMN_OPENINGS: usize = 650;
    type Eval = DensePolynomial<Boolean, DEGREE_PLUS_ONE>;
    type Cw = DensePolynomial<i32, DEGREE_PLUS_ONE>;
    type Fmod = Uint<K>;
    type PrimeTest = MillerRabin;
    type Chal = i128;
    type Pt = i128;
    type CombR = Int<M>;
    type Comb = DensePolynomial<Self::CombR, DEGREE_PLUS_ONE>;
}

/// Helper function to set up common parameters for tests.
pub fn setup_test_params<const N: usize, const K: usize, const M: usize>(
    num_vars: usize,
) -> (
    ZipPlusParams<
        TestZipTypes<N, K, M>,
        RaaSignFlippingCode<TestZipTypes<N, K, M>, REPETITION_FACTOR>,
    >,
    DenseMultilinearExtension<<TestZipTypes<N, K, M> as ZipTypes>::Eval>,
) {
    setup_test_params_inner(num_vars, |poly_size| {
        (1..=poly_size as i32).map(Int::from).collect()
    })
}

/// Helper function to set up common parameters for tests.
pub fn setup_poly_test_params<const K: usize, const M: usize, const DEGREE_PLUS_ONE: usize>(
    num_vars: usize,
) -> (
    ZipPlusParams<
        TestPolyZipTypes<K, M, DEGREE_PLUS_ONE>,
        RaaCode<TestPolyZipTypes<K, M, DEGREE_PLUS_ONE>, REPETITION_FACTOR>,
    >,
    DenseMultilinearExtension<<TestPolyZipTypes<K, M, DEGREE_PLUS_ONE> as ZipTypes>::Eval>,
) {
    setup_test_params_inner(num_vars, |poly_size| {
        let eval_coeffs: Vec<_> = (1..=(poly_size * (DEGREE_PLUS_ONE - 1)) as i8)
            .map(|v| v.is_odd().into())
            .collect_vec();
        eval_coeffs
            .chunks_exact(DEGREE_PLUS_ONE - 1)
            .map(DensePolynomial::new)
            .collect_vec()
    })
}

fn setup_test_params_inner<Zt: ZipTypes, Lc: LinearCode<Zt, Config = RaaConfig>>(
    num_vars: usize,
    prepare_evaluations: impl FnOnce(usize) -> Vec<Zt::Eval>,
) -> (ZipPlusParams<Zt, Lc>, DenseMultilinearExtension<Zt::Eval>) {
    let poly_size = 1 << num_vars;
    let num_rows = 1 << num_vars.div_ceil(2);

    let code = Lc::new(poly_size, RAA_CFG);
    let pp = ZipPlusParams::new(num_vars, num_rows, code);

    let evaluations = prepare_evaluations(poly_size);
    let poly = DenseMultilinearExtension::from_evaluations_vec(num_vars, evaluations, Zero::zero());

    (pp, poly)
}

pub fn setup_full_protocol<F, const N: usize, const K: usize, const M: usize>(
    num_vars: usize,
) -> (
    ZipPlusParams<
        TestZipTypes<N, K, M>,
        RaaSignFlippingCode<TestZipTypes<N, K, M>, REPETITION_FACTOR>,
    >,
    ZipPlusCommitment,
    Vec<F>,
    F,
    ZipPlusProof,
)
where
    F: PrimeField
        + for<'a> FromWithConfig<&'a <TestZipTypes<N, K, M> as ZipTypes>::Chal>
        + for<'a> MulByScalar<&'a F>,
    F::Inner: FromRef<<TestZipTypes<N, K, M> as ZipTypes>::Fmod> + Transcribable,
    <TestZipTypes<N, K, M> as ZipTypes>::Eval: ProjectableToField<F>,
    <TestZipTypes<N, K, M> as ZipTypes>::Comb: ProjectableToField<F>,
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
    const DEGREE_PLUS_ONE: usize,
>(
    num_vars: usize,
) -> (
    ZipPlusParams<
        TestPolyZipTypes<K, M, DEGREE_PLUS_ONE>,
        RaaCode<TestPolyZipTypes<K, M, DEGREE_PLUS_ONE>, REPETITION_FACTOR>,
    >,
    ZipPlusCommitment,
    Vec<F>,
    F,
    ZipPlusProof,
)
where
    F: PrimeField
        + for<'a> FromWithConfig<&'a <TestPolyZipTypes<K, M, DEGREE_PLUS_ONE> as ZipTypes>::Chal>
        + for<'a> MulByScalar<&'a F>,
    F::Inner: FromRef<<TestPolyZipTypes<K, M, DEGREE_PLUS_ONE> as ZipTypes>::Fmod> + Transcribable,
    <TestPolyZipTypes<K, M, DEGREE_PLUS_ONE> as ZipTypes>::Eval: ProjectableToField<F>,
    <TestPolyZipTypes<K, M, DEGREE_PLUS_ONE> as ZipTypes>::Comb: ProjectableToField<F>,
{
    setup_full_protocol_inner::<_, _, _, N>(num_vars, setup_poly_test_params, || {
        (0..num_vars).map(|i| i as i128 + 2).collect()
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
    Lc: LinearCode<Zt>,
    F: PrimeField
        + for<'a> FromWithConfig<&'a Zt::Chal>
        + for<'a> FromWithConfig<&'a Zt::Pt>
        + for<'a> MulByScalar<&'a F>,
    F::Inner: FromRef<Zt::Fmod> + Transcribable,
    Zt::Eval: ProjectableToField<F>,
    Zt::Comb: ProjectableToField<F> + for<'a> MulByScalar<&'a Zt::Pt>,
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
        // Widen up polynomial for evaluation
        let poly = DenseMultilinearExtension {
            evaluations: poly
                .evaluations
                .iter()
                .map(Zt::Comb::from_ref)
                .collect_vec(),
            num_vars,
        };
        let expected_eval = poly
            .evaluate(&point, Zero::zero())
            .expect("failed to evaluate polynomial");
        let project = Zt::Comb::prepare_projection(&projecting_element);
        assert_eq!(eval_f, project(&expected_eval));
    }

    let point_f = point
        .iter()
        .map(|v| v.into_with_cfg(&field_cfg))
        .collect_vec();

    (pp, comm, point_f, eval_f, proof)
}
