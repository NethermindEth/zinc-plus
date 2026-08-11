#![allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::type_complexity,
    clippy::unwrap_used
)]

use crate::{
    code::{
        LinearCode,
        iprs::{IprsCode, PnttConfigF65537},
    },
    pcs::structs::{ZipPlus, ZipPlusCommitment, ZipPlusParams, ZipTypes},
    pcs_transcript::{PcsProverTranscript, PcsVerifierTranscript},
};
use crypto_primitives::{
    BaseFieldConfig, IntSemiring, ProjectElementWithConfig, crypto_bigint_int::Int,
    crypto_bigint_uint::Uint,
};
use itertools::Itertools;
use zinc_poly::{
    mle::DenseMultilinearExtension,
    univariate::{
        binary::{BinaryPoly, BinaryPolyInnerProduct},
        dense::{DensePolyInnerProduct, DensePolynomial},
    },
};
use zinc_primality::MillerRabin;
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_utils::{
    CHECKED,
    from_ref::FromRef,
    inner_product::{MBSInnerProduct, PreparedFieldInnerProduct, ScalarProduct},
    mul_by_scalar::MulByScalar,
};

pub const IPRS_DEPTH: usize = 1;
pub const IPRS_ROW_LEN: usize = 32 * (1 << (3 * IPRS_DEPTH)); // I.e. BASE_DIM = 32

/// Linear code repetition factor (inverse rate).
pub const REP_FACTOR: usize = 8;

pub type TestIprsConfig = PnttConfigF65537;

#[derive(Debug, Clone)]
pub struct TestZipTypes<const N: usize, const K: usize, const M: usize> {}
impl<const N: usize, const K: usize, const M: usize> ZipTypes for TestZipTypes<N, K, M> {
    const NUM_COLUMN_OPENINGS: usize = 100;
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

#[derive(Debug, Clone)]
pub struct TestBinPolyZipTypes<const K: usize, const M: usize, const DEGREE_PLUS_ONE: usize> {}
impl<const K: usize, const M: usize, const DEGREE_PLUS_ONE: usize> ZipTypes
    for TestBinPolyZipTypes<K, M, DEGREE_PLUS_ONE>
{
    const NUM_COLUMN_OPENINGS: usize = 100;
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
        (),
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
        IprsCode<TestZipTypes<N, K, M>, TestIprsConfig, REP_FACTOR, CHECKED>,
    >,
    DenseMultilinearExtension<<TestZipTypes<N, K, M> as ZipTypes>::Eval>,
) {
    setup_test_params_inner(
        num_vars,
        IprsCode::new(IPRS_ROW_LEN, IPRS_DEPTH).unwrap(),
        |poly_size| (1..=poly_size as i32).map(Int::from).collect(),
    )
}

/// Helper function to set up common parameters for tests.
pub fn setup_poly_test_params<const K: usize, const M: usize, const DEGREE_PLUS_ONE: usize>(
    num_vars: usize,
) -> (
    ZipPlusParams<
        TestBinPolyZipTypes<K, M, DEGREE_PLUS_ONE>,
        IprsCode<TestBinPolyZipTypes<K, M, DEGREE_PLUS_ONE>, TestIprsConfig, REP_FACTOR, CHECKED>,
    >,
    DenseMultilinearExtension<<TestBinPolyZipTypes<K, M, DEGREE_PLUS_ONE> as ZipTypes>::Eval>,
) {
    setup_test_params_inner(
        num_vars,
        IprsCode::new(IPRS_ROW_LEN, IPRS_DEPTH).unwrap(),
        |poly_size| {
            let degree = DEGREE_PLUS_ONE - 1;
            let eval_coeffs: Vec<_> = (1..=(poly_size * degree) as i64)
                .map(|v| v.is_odd().into())
                .collect_vec();
            eval_coeffs
                .chunks_exact(degree)
                .map(BinaryPoly::new_padded)
                .collect_vec()
        },
    )
}

pub fn setup_test_params_inner<Zt: ZipTypes, Lc: LinearCode<Zt>>(
    num_vars: usize,
    code: Lc,
    prepare_evaluations: impl FnOnce(usize) -> Vec<Zt::Eval>,
) -> (ZipPlusParams<Zt, Lc>, DenseMultilinearExtension<Zt::Eval>) {
    let poly_size: usize = 1 << num_vars;
    let row_len = code.row_len();
    let num_rows = poly_size.div_ceil(row_len);

    let pp = ZipPlusParams::new(num_vars, num_rows, code);

    let evaluations = prepare_evaluations(poly_size);
    // We know the length of the evaluations is a power of two.
    let poly = DenseMultilinearExtension {
        num_vars,
        evaluations,
    };

    (pp, poly)
}

pub fn setup_full_protocol<C, const N: usize, const K: usize, const M: usize>(
    num_vars: usize,
) -> (
    ZipPlusParams<
        TestZipTypes<N, K, M>,
        IprsCode<TestZipTypes<N, K, M>, TestIprsConfig, REP_FACTOR, CHECKED>,
    >,
    ZipPlusCommitment,
    Vec<C::Element>,
    C::Element,
    PcsVerifierTranscript,
)
where
    C: BaseFieldConfig
        + ProjectElementWithConfig<<TestZipTypes<N, K, M> as ZipTypes>::Pt>
        + ProjectElementWithConfig<<TestZipTypes<N, K, M> as ZipTypes>::CombR>
        + PreparedFieldInnerProduct<<TestZipTypes<N, K, M> as ZipTypes>::CombR>
        + Sync,
    C::Element: ConstTranscribable,
    C::Integer: ConstTranscribable,
    C::Integer: FromRef<<TestZipTypes<N, K, M> as ZipTypes>::Fmod>,
{
    setup_full_protocol_inner::<_, _, C, N>(num_vars, setup_test_params, || {
        (0..num_vars).map(|i| Int::from(i as i32 + 2)).collect()
    })
}

pub fn setup_full_protocol_poly<
    C,
    const N: usize,
    const K: usize,
    const M: usize,
    const DEGREE_PLUS_ONE: usize,
>(
    num_vars: usize,
) -> (
    ZipPlusParams<
        TestBinPolyZipTypes<K, M, DEGREE_PLUS_ONE>,
        IprsCode<TestBinPolyZipTypes<K, M, DEGREE_PLUS_ONE>, TestIprsConfig, REP_FACTOR, CHECKED>,
    >,
    ZipPlusCommitment,
    Vec<C::Element>,
    C::Element,
    PcsVerifierTranscript,
)
where
    C: BaseFieldConfig
        + ProjectElementWithConfig<<TestBinPolyZipTypes<K, M, DEGREE_PLUS_ONE> as ZipTypes>::Pt>
        + ProjectElementWithConfig<<TestBinPolyZipTypes<K, M, DEGREE_PLUS_ONE> as ZipTypes>::CombR>
        + PreparedFieldInnerProduct<<TestBinPolyZipTypes<K, M, DEGREE_PLUS_ONE> as ZipTypes>::CombR>
        + Sync,
    C::Element: ConstTranscribable,
    C::Integer: ConstTranscribable,
    C::Integer: FromRef<<TestBinPolyZipTypes<K, M, DEGREE_PLUS_ONE> as ZipTypes>::Fmod>,
{
    setup_full_protocol_inner::<_, _, C, N>(num_vars, setup_poly_test_params, || {
        (0..num_vars).map(|i| i as i128 + 2).collect()
    })
}

pub fn setup_full_protocol_inner<Zt, Lc, C, const N: usize>(
    num_vars: usize,
    setup: impl FnOnce(usize) -> (ZipPlusParams<Zt, Lc>, DenseMultilinearExtension<Zt::Eval>),
    prepare_evaluation_point: impl FnOnce() -> Vec<Zt::Pt>,
) -> (
    ZipPlusParams<Zt, Lc>,
    ZipPlusCommitment,
    Vec<C::Element>,
    C::Element,
    PcsVerifierTranscript,
)
where
    Zt: ZipTypes,
    Zt::CombR: MulByScalar<Zt::Chal>,
    Lc: LinearCode<Zt>,
    C: BaseFieldConfig
        + ProjectElementWithConfig<Zt::Pt>
        + ProjectElementWithConfig<Zt::CombR>
        + PreparedFieldInnerProduct<Zt::CombR>
        + Sync,
    C::Element: ConstTranscribable,
    C::Integer: ConstTranscribable,
    C::Integer: FromRef<Zt::Fmod>,
{
    let (pp, poly) = setup(num_vars);
    let (hint, comm) = ZipPlus::commit_single(&pp, &poly).unwrap();

    let mut transcript = PcsProverTranscript::new_from_commitment(&comm);
    let field_cfg = get_field_cfg::<Zt, C>(&mut transcript.fs_transcript);
    let point: Vec<Zt::Pt> = prepare_evaluation_point();

    let eval_f =
        ZipPlus::prove_single::<C, CHECKED>(&mut transcript, &pp, &poly, &point, &hint, &field_cfg)
            .unwrap();

    let point_f = point.iter().map(|v| field_cfg.project(v)).collect_vec();

    let mut transcript = transcript.into_verification_transcript();

    transcript.fs_transcript.absorb_slice(&comm.root.0);

    (pp, comm, point_f, eval_f, transcript)
}

pub fn get_field_cfg<Zt, C>(transcript: &mut impl Transcript) -> C
where
    Zt: ZipTypes,
    C: BaseFieldConfig,
    C::Integer: FromRef<Zt::Fmod>,
{
    transcript.get_random_field_cfg::<C, Zt::Fmod, Zt::PrimeTest>()
}
