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
    pcs::structs::{ZipPlus, ZipPlusCommitment, ZipPlusParams, ZipTypes},
    pcs_transcript::{PcsProverTranscript, PcsVerifierTranscript},
};
use crypto_primitives::{
    FromWithConfig, IntSemiring, IntoWithConfig, PrimeField, crypto_bigint_int::Int,
    crypto_bigint_uint::Uint,
};
use itertools::Itertools;
use num_traits::Zero;
use zinc_poly::{
    mle::DenseMultilinearExtension,
    univariate::{
        binary::{BinaryPoly, BinaryPolyInnerProduct},
        dense::{DensePolyInnerProduct, DensePolynomial},
    },
};
use zinc_primality::MillerRabin;
use zinc_transcript::traits::{Transcribable, Transcript};
use zinc_utils::{
    CHECKED,
    from_ref::FromRef,
    inner_product::{MBSInnerProduct, ScalarProduct},
    mul_by_scalar::MulByScalar,
    projectable_to_field::ProjectableToField,
};

pub const REPETITION_FACTOR: usize = 4;

#[derive(Clone, Copy)]
pub struct TestRaaConfig;

impl RaaConfig for TestRaaConfig {
    const PERMUTE_IN_PLACE: bool = false;
    const CHECK_FOR_OVERFLOWS: bool = true;
}

pub struct TestZipTypes<const N: usize, const K: usize, const M: usize> {}
impl<const N: usize, const K: usize, const M: usize> ZipTypes for TestZipTypes<N, K, M> {
    const NUM_COLUMN_OPENINGS: usize = 200;
    type Eval = Int<N>;
    type Cw = Int<K>;
    type Fmod = Uint<K>;
    type PrimeTest = MillerRabin;
    type Chal = Int<N>;
    type Pt = Int<N>;
    type CombR = Int<M>;
    type Comb = Self::CombR;
    type EvalDotChal = ScalarProduct;
    type CombDotChal = ScalarProduct;
    type ArrCombRDotChal = MBSInnerProduct;
}

pub struct TestBinPolyZipTypes<const K: usize, const M: usize, const DEGREE_PLUS_ONE: usize> {}
impl<const K: usize, const M: usize, const DEGREE_PLUS_ONE: usize> ZipTypes
    for TestBinPolyZipTypes<K, M, DEGREE_PLUS_ONE>
{
    const NUM_COLUMN_OPENINGS: usize = 200;
    type Eval = BinaryPoly<DEGREE_PLUS_ONE>;
    type Cw = DensePolynomial<i64, DEGREE_PLUS_ONE>;
    type Fmod = Uint<K>;
    type PrimeTest = MillerRabin;
    type Chal = i128;
    type Pt = i128;
    type CombR = Int<M>;
    type Comb = DensePolynomial<Self::CombR, DEGREE_PLUS_ONE>;
    type EvalDotChal = BinaryPolyInnerProduct<Self::Chal, DEGREE_PLUS_ONE>;
    type CombDotChal = DensePolyInnerProduct<
        Self::CombR,
        Self::Chal,
        Self::CombR,
        MBSInnerProduct,
        DEGREE_PLUS_ONE,
    >;
    type ArrCombRDotChal = MBSInnerProduct;
}

pub struct TestArbPolyZipTypes<
    const N: usize,
    const K: usize,
    const M: usize,
    const DEGREE_PLUS_ONE: usize,
> {}
impl<const N: usize, const K: usize, const M: usize, const DEGREE_PLUS_ONE: usize> ZipTypes
    for TestArbPolyZipTypes<N, K, M, DEGREE_PLUS_ONE>
{
    const NUM_COLUMN_OPENINGS: usize = 200;
    type Eval = DensePolynomial<Int<N>, DEGREE_PLUS_ONE>;
    type Cw = DensePolynomial<Int<K>, DEGREE_PLUS_ONE>;
    type Fmod = Uint<K>;
    type PrimeTest = MillerRabin;
    type Chal = i128;
    type Pt = i128;
    type CombR = Int<M>;
    type Comb = DensePolynomial<Self::CombR, DEGREE_PLUS_ONE>;
    type EvalDotChal =
        DensePolyInnerProduct<Int<N>, Self::Chal, Self::CombR, MBSInnerProduct, DEGREE_PLUS_ONE>;
    type CombDotChal = DensePolyInnerProduct<
        Self::CombR,
        Self::Chal,
        Self::CombR,
        MBSInnerProduct,
        DEGREE_PLUS_ONE,
    >;
    type ArrCombRDotChal = MBSInnerProduct;
}

/// Helper function to set up common parameters for tests.
pub fn setup_test_params<const N: usize, const K: usize, const M: usize>(
    num_vars: usize,
) -> (
    ZipPlusParams<
        TestZipTypes<N, K, M>,
        RaaSignFlippingCode<TestZipTypes<N, K, M>, TestRaaConfig, REPETITION_FACTOR>,
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
        TestBinPolyZipTypes<K, M, DEGREE_PLUS_ONE>,
        RaaCode<TestBinPolyZipTypes<K, M, DEGREE_PLUS_ONE>, TestRaaConfig, REPETITION_FACTOR>,
    >,
    DenseMultilinearExtension<<TestBinPolyZipTypes<K, M, DEGREE_PLUS_ONE> as ZipTypes>::Eval>,
) {
    setup_test_params_inner(num_vars, |poly_size| {
        let eval_coeffs: Vec<_> = (1..=(poly_size * (DEGREE_PLUS_ONE - 1)) as i8)
            .map(|v| v.is_odd().into())
            .collect_vec();
        eval_coeffs
            .chunks_exact(DEGREE_PLUS_ONE - 1)
            .map(BinaryPoly::new)
            .collect_vec()
    })
}

fn setup_test_params_inner<Zt: ZipTypes, Lc: LinearCode<Zt>>(
    num_vars: usize,
    prepare_evaluations: impl FnOnce(usize) -> Vec<Zt::Eval>,
) -> (ZipPlusParams<Zt, Lc>, DenseMultilinearExtension<Zt::Eval>) {
    let poly_size = 1 << num_vars;
    let num_rows = 1 << num_vars.div_ceil(2);

    let code = Lc::new(poly_size);
    let pp = ZipPlusParams::new(num_vars, num_rows, code);

    let evaluations = prepare_evaluations(poly_size);
    // We know the length of the evaluations is a power of two.
    let poly = DenseMultilinearExtension {
        num_vars,
        evaluations,
    };

    (pp, poly)
}

pub fn setup_full_protocol<F, const N: usize, const K: usize, const M: usize>(
    num_vars: usize,
) -> (
    ZipPlusParams<
        TestZipTypes<N, K, M>,
        RaaSignFlippingCode<TestZipTypes<N, K, M>, TestRaaConfig, REPETITION_FACTOR>,
    >,
    ZipPlusCommitment,
    Vec<F>,
    F,
    PcsVerifierTranscript,
)
where
    F: PrimeField
        + for<'a> FromWithConfig<&'a <TestZipTypes<N, K, M> as ZipTypes>::Chal>
        + for<'a> FromWithConfig<&'a <TestZipTypes<N, K, M> as ZipTypes>::CombR>
        + for<'a> MulByScalar<&'a F>
        + FromRef<F>,
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
        TestBinPolyZipTypes<K, M, DEGREE_PLUS_ONE>,
        RaaCode<TestBinPolyZipTypes<K, M, DEGREE_PLUS_ONE>, TestRaaConfig, REPETITION_FACTOR>,
    >,
    ZipPlusCommitment,
    Vec<F>,
    F,
    PcsVerifierTranscript,
)
where
    F: PrimeField
        + for<'a> FromWithConfig<&'a <TestBinPolyZipTypes<K, M, DEGREE_PLUS_ONE> as ZipTypes>::Chal>
        + for<'a> FromWithConfig<&'a <TestBinPolyZipTypes<K, M, DEGREE_PLUS_ONE> as ZipTypes>::CombR>
        + for<'a> MulByScalar<&'a F>
        + FromRef<F>
        + 'static,
    F::Inner:
        FromRef<<TestBinPolyZipTypes<K, M, DEGREE_PLUS_ONE> as ZipTypes>::Fmod> + Transcribable,
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
    PcsVerifierTranscript,
)
where
    Zt: ZipTypes,
    Lc: LinearCode<Zt>,
    F: PrimeField
        + for<'a> FromWithConfig<&'a Zt::Chal>
        + for<'a> FromWithConfig<&'a Zt::Pt>
        + for<'a> MulByScalar<&'a F>
        + FromRef<F>,
    F::Inner: FromRef<Zt::Fmod> + Transcribable,
    Zt::Comb: for<'a> MulByScalar<&'a Zt::Pt>,
    Zt::Eval: ProjectableToField<F>,
    Zt::Comb: ProjectableToField<F>,
{
    let (pp, poly) = setup(num_vars);

    let (data, comm) = ZipPlus::commit(&pp, &poly).unwrap();

    let mut transcript = PcsProverTranscript::new_from_commitment(&comm).unwrap();
    let (field_cfg, projecting_element) =
        get_field_and_projecting_element::<Zt, F>(&mut transcript.fs_transcript);

    let point: Vec<Zt::Pt> = prepare_evaluation_point();

    ZipPlus::test::<CHECKED>(&mut transcript, &pp, &poly, &data).unwrap();
    let eval_f = ZipPlus::evaluate::<_, CHECKED>(
        &mut transcript,
        &pp,
        &poly,
        &point,
        &field_cfg,
        &projecting_element,
    )
    .unwrap();

    // Verify the evaluation is done correctly
    {
        // Widen up polynomial for evaluation
        let poly: DenseMultilinearExtension<_> = poly.iter().map(Zt::Comb::from_ref).collect();

        let expected_eval = poly
            .evaluate(&point, Zero::zero())
            .expect("failed to evaluate polynomial");
        let projecting_element_f: F = (&projecting_element).into_with_cfg(&field_cfg);
        let project = Zt::Comb::prepare_projection(&projecting_element_f);
        assert_eq!(eval_f, project(&expected_eval));
    }

    let point_f = point
        .iter()
        .map(|v| v.into_with_cfg(&field_cfg))
        .collect_vec();

    let mut transcript = transcript.into_verification_transcript();

    transcript.fs_transcript.absorb_slice(&comm.root.0);

    (pp, comm, point_f, eval_f, transcript)
}

pub fn get_field_and_projecting_element<Zt, F>(
    transcript: &mut impl Transcript,
) -> (F::Config, Zt::Chal)
where
    Zt: ZipTypes,
    F: PrimeField,
    F::Inner: FromRef<Zt::Fmod>,
{
    let field_cfg = transcript.get_random_field_cfg::<F, Zt::Fmod, Zt::PrimeTest>();
    let projecting_element: Zt::Chal = transcript.get_challenge();
    (field_cfg, projecting_element)
}
