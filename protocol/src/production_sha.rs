//! Production SHA ProjectionFold protocol helpers.
//!
//! This module is intentionally separate from the existing single-instance
//! `Proof`: production ProjectionFold has a different transcript order and
//! derives folded commitments only after SumFold fixes the instance-axis point.

use std::{borrow::Cow, io::Cursor, marker::PhantomData};

use crate::{
    ZincTypes,
    multipoint_reduction::{prove_multipoint_reduction, verify_multipoint_reduction},
    pcs::{
        AllHyraxPCSTypes, PCSCommitments, PCSOpeningProof, PCSParams, PCSProverData,
        PCSVerifierParams, ZincPCSTypes,
    },
};
use ark_ec::AffineRepr;
use crypto_primitives::{FromPrimitiveWithConfig, PrimeField};
use num_traits::{ConstZero, Zero};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use thiserror::Error;
#[cfg(debug_assertions)]
use zinc_piop::neutron_nova::validate_projected_trace;
use zinc_piop::{
    combined_poly_resolver::Proof as CombinedPolyResolverProof,
    ideal_check::Proof as IdealCheckProof,
    multipoint_eval::{
        MultipointEval, MultipointEvalError, Proof as MultipointEvalProof,
        Subclaim as MultipointSubclaim,
    },
    neutron_nova::SumFoldError,
    neutron_nova::{
        InstanceFoldClaim, LinearResidualCoeffTable, MleTable, NUM_NONZERO_SHA_FAMILIES,
        NUM_SHA_RESIDUAL_FAMILIES, ProjectedPublic, ProjectedTrace, ProjectionFoldWitness,
        SHA_ROW_COUNT, SHA_ROW_VARS, SHA_WORD_BITS, ShaBinaryFoldField, ShaBooleanitySource,
        ShaIntCol, ShaProjectionError, ShaPublicCol, ShaPublicWordCol, ShaResidualFamily,
        ShaWordCol, beta_aggregate_nonzero_ideal_polys_with_weights, bit_slice_index,
        build_booleanity_weights, build_dense_sha_sumfold_group, build_folded_row_sumcheck_group,
        build_linear_residual_coeff_tables_with_row_weights,
        build_production_sha_sumfold_group_from_prefix_accumulators, build_sha_lambda_powers,
        build_sha_residual_eval_powers, build_sha_sumfold_linear_accumulator,
        build_sha_sumfold_quadratic_prefix_accumulator, derive_instance_fold_claim,
        expression_folded_row_sum_with_row_weights, expression_folded_row_sum_with_vectors,
        fold_projected_traces, folded_row_integrand_sum, production_sha_booleanity_sources,
        production_sha_nonzero_families, sha_int_at_point_with_weights_unchecked,
        sha_public_at_point, sha_public_at_point_with_weights,
        sha_word_bits_at_point_with_weights_unchecked, verify_folded_row_sumcheck_claim,
        verify_fresh_sha_ideal_polys,
    },
    sumcheck::{
        SumCheckError,
        multi_degree::{MultiDegreeSumcheck, MultiDegreeSumcheckGroup, MultiDegreeSumcheckProof},
    },
};
use zinc_poly::{
    EvaluatablePolynomial, EvaluationError,
    mle::DenseMultilinearExtension,
    univariate::{
        binary::BinaryPoly,
        dense::DensePolynomial,
        dynamic::over_field::{DynamicPolyFInnerProduct, DynamicPolynomialF},
        nat_evaluation::NatEvaluatedPoly,
    },
    utils::{ArithErrors, build_eq_x_r_vec, eq_eval},
};
use zinc_transcript::Blake3Transcript;
use zinc_transcript::traits::{GenTranscribable, Transcribable, Transcript};
use zinc_uair::{ShiftSpec, Uair, UairSignature, UairTrace, UairWitness};
use zinc_utils::{
    UNCHECKED, cfg_into_iter, cfg_iter, delayed_reduction::DelayedFieldProductSum,
    inner_product::FieldFieldInnerProduct, inner_product::InnerProduct,
    inner_transparent_field::InnerTransparentField,
};
use zip_plus::{
    ZipError,
    pcs::{
        generic::{FoldablePCS, PCS},
        hyrax::{BinaryLanes, DensePolyScalarLanes, HyraxFieldBridge, HyraxPCS, IntScalarLane},
    },
    pcs_transcript::{PcsProverTranscript, PcsVerifierTranscript},
};

/// Serialized production ProjectionFold proof object.
///
/// This object carries verifier messages and claimed evaluations only. Folding
/// weights, batching powers, folded accumulator values, and prover working
/// caches are derived from the transcript/setup or kept as prover-local state.
#[derive(Clone, Debug)]
pub struct ProductionLinearIdealFoldProof<P, Zt, F, const D: usize>
where
    Zt: ZincTypes<D>,
    F: PrimeField,
    P: ZincPCSTypes<Zt, F, D>,
{
    pub instance_commitments: Vec<PCSCommitments<P, Zt, F, D>>,
    pub ideal_check: IdealCheckProof<F>,
    pub sumfold_proof: MultiDegreeSumcheckProof<F>,
    pub resolver: CombinedPolyResolverProof<F>,
    pub combined_sumcheck: MultiDegreeSumcheckProof<F>,
    pub multipoint_eval: MultipointEvalProof<F>,
    pub witness_lifted_evals: Vec<DynamicPolynomialF<F>>,
    pub opening_proof: PCSOpeningProof<P, Zt, F, D>,
}

#[derive(Clone, Debug)]
pub struct ProductionShaWitnessPolys<Zt, const D: usize>
where
    Zt: ZincTypes<D>,
{
    pub binary: MleTable<BinaryPoly<D>>,
    pub arbitrary: MleTable<DensePolynomial<Zt::Int, D>>,
    pub int: MleTable<Zt::Int>,
}

#[derive(Clone, Debug)]
pub struct ProductionShaProverInstance<Zt, F, const D: usize>
where
    Zt: ZincTypes<D>,
    F: PrimeField,
{
    pub trace: ProjectedTrace<F>,
    pub public: ProjectedPublic<F>,
    pub witness_polys: ProductionShaWitnessPolys<Zt, D>,
}

pub trait ProductionShaProjectionAdapter<Zt, F, const D: usize>: Uair
where
    Zt: ZincTypes<D>,
    F: PrimeField,
{
    fn production_sha_pcs_batch_sizes() -> (usize, usize, usize) {
        (ShaWordCol::COUNT, 0, ShaIntCol::COUNT)
    }

    fn project_production_sha_public(
        shape: &UairShape<Self>,
        public_trace: &UairTrace<'_, Zt::Int, Zt::Int, D>,
        field_cfg: &F::Config,
    ) -> Result<ProjectedPublic<F>, ProductionShaError<F>>
    where
        Self: Sized;

    fn project_production_sha_witness(
        shape: &UairShape<Self>,
        public_trace: &UairTrace<'_, Zt::Int, Zt::Int, D>,
        witness_trace: &UairTrace<'_, Zt::Int, Zt::Int, D>,
        field_cfg: &F::Config,
    ) -> Result<
        (
            ProjectedTrace<F>,
            ProjectedPublic<F>,
            ProductionShaWitnessPolys<Zt, D>,
        ),
        ProductionShaError<F>,
    >
    where
        Self: Sized;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShaEndpointEvals<F> {
    pub sources: Vec<ShaSourceEndpointEval<F>>,
    pub int_sources: Vec<ShaIntEndpointEval<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShaSourceEndpointEval<F> {
    pub col: ShaWordCol,
    pub shift: usize,
    pub scalarized: F,
    pub bits: [F; 32],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShaIntEndpointEval<F> {
    pub col: ShaIntCol,
    pub scalar: F,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShaMpSource {
    WordBit { col: ShaWordCol, bit: usize },
    Int { col: ShaIntCol },
    Public { col: ShaPublicCol },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShaMpShiftSource {
    WordBit {
        col: ShaWordCol,
        bit: usize,
        shift: usize,
    },
    Public {
        col: ShaPublicCol,
        shift: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShaMultipointLayout {
    pub sources: Vec<ShaMpSource>,
    pub shifts: Vec<ShaMpShiftSource>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VirtualChMajEndpoint<F> {
    pub ch1: [F; 32],
    pub ch2: [F; 32],
    pub maj: [F; 32],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProductionShaChallenges<F> {
    pub r_ic: [F; SHA_ROW_VARS],
    pub a: F,
    pub lambda: F,
    pub rho: F,
    pub xi: F,
    pub beta: Vec<F>,
}

const SHA256_ROUND_CONSTANTS: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];
const SHA_IDEAL_EVAL_POWER_COUNT: usize = 62;

const PRODUCTION_SHA_FRESH_BATCH_DOMAIN: &[u8] = b"PF_CONCISE_SHA256_FRESH_BATCH_V1";

#[derive(Clone, Debug)]
pub struct UairShape<U: Uair> {
    pub num_vars: usize,
    pub signature: UairSignature,
    _marker: PhantomData<U>,
}

impl<U: Uair> UairShape<U> {
    pub fn new(num_vars: usize) -> Self {
        Self {
            num_vars,
            signature: U::signature(),
            _marker: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct UairInstance<'a, PolyCoeff: Clone, Int: Clone, Commitments, const D: usize> {
    pub public_trace: UairTrace<'a, PolyCoeff, Int, D>,
    pub commitments: Commitments,
}

#[derive(Clone, Debug)]
pub struct LinearIdealFoldProverParams<P, U, Zt, F, const D: usize>
where
    U: Uair,
    Zt: ZincTypes<D>,
    F: PrimeField,
    P: ZincPCSTypes<Zt, F, D>,
{
    pub pcs_params: PCSParams<P, Zt, F, D>,
    pub field_cfg: F::Config,
    pub prefix_vars: usize,
    _marker: PhantomData<U>,
}

impl<P, U, Zt, F, const D: usize> LinearIdealFoldProverParams<P, U, Zt, F, D>
where
    U: Uair,
    Zt: ZincTypes<D>,
    F: PrimeField,
    P: ZincPCSTypes<Zt, F, D>,
{
    pub fn new(
        pcs_params: PCSParams<P, Zt, F, D>,
        field_cfg: F::Config,
        prefix_vars: usize,
    ) -> Self {
        Self {
            pcs_params,
            field_cfg,
            prefix_vars,
            _marker: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LinearIdealFoldVerifierParams<P, U, Zt, F, const D: usize>
where
    U: Uair,
    Zt: ZincTypes<D>,
    F: PrimeField,
    P: ZincPCSTypes<Zt, F, D>,
{
    pub pcs_params: PCSVerifierParams<P, Zt, F, D>,
    pub field_cfg: F::Config,
    _marker: PhantomData<U>,
}

impl<P, U, Zt, F, const D: usize> LinearIdealFoldVerifierParams<P, U, Zt, F, D>
where
    U: Uair,
    Zt: ZincTypes<D>,
    F: PrimeField,
    P: ZincPCSTypes<Zt, F, D>,
{
    pub fn new(pcs_params: PCSVerifierParams<P, Zt, F, D>, field_cfg: F::Config) -> Self {
        Self {
            pcs_params,
            field_cfg,
            _marker: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct VerifiedLinearIdealFoldSetup<P, U, Zt, F, const D: usize>
where
    U: Uair,
    Zt: ZincTypes<D>,
    F: PrimeField,
    P: ZincPCSTypes<Zt, F, D>,
{
    pub pcs_params: PCSVerifierParams<P, Zt, F, D>,
    pub shape: UairShape<U>,
    pub field_cfg: F::Config,
}

#[derive(Clone, Debug)]
pub struct LinearIdealFoldProveOutput<Instance, FoldedInstance, FoldedWitness, Proof> {
    pub fresh_instances: Vec<Instance>,
    pub folded_instance: FoldedInstance,
    pub folded_witness: FoldedWitness,
    pub proof: Proof,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldedLinearIdealInstance<F: PrimeField, Commitments, Public> {
    pub target: F,
    pub commitments: Commitments,
    pub public: Public,
}

#[derive(Clone, Debug)]
pub struct FoldedLinearIdealWitness<Witness> {
    pub witness: Witness,
}

type ProductionShaFreshArtifacts<P, Zt, F, const D: usize> = (
    Vec<
        UairInstance<
            'static,
            <Zt as ZincTypes<D>>::Int,
            <Zt as ZincTypes<D>>::Int,
            PCSCommitments<P, Zt, F, D>,
            D,
        >,
    >,
    Vec<PCSCommitments<P, Zt, F, D>>,
    Vec<PCSProverData<P, Zt, F, D>>,
    Vec<ProjectedTrace<F>>,
    Vec<ProjectedPublic<F>>,
);

type ProductionShaSumfoldAccumulators<F> = (Vec<F>, Vec<F>, MultiDegreeSumcheckGroup<F>);

type ProductionShaFoldAfterSumfold<P, Zt, F, const D: usize> = (
    ProjectionFoldWitness<F>,
    ProjectedPublic<F>,
    F,
    InstanceFoldClaim<F>,
    PCSCommitments<P, Zt, F, D>,
    PCSProverData<P, Zt, F, D>,
);

type ProductionShaEndpointMultipoint<F> = (
    CombinedPolyResolverProof<F>,
    ShaEndpointEvals<F>,
    MultipointEvalProof<F>,
    Vec<F>,
);

type ProductionShaPcsOpening<P, Zt, F, const D: usize> =
    (Vec<DynamicPolynomialF<F>>, PCSOpeningProof<P, Zt, F, D>);

type ProductionShaVerifierAggregate<F> = (
    [DynamicPolynomialF<F>; NUM_NONZERO_SHA_FAMILIES],
    F,
    F,
    F,
    F,
    F,
);

#[derive(Clone, Debug)]
pub struct ProductionShaFoldedWitness<P, Zt, F, const D: usize>
where
    Zt: ZincTypes<D>,
    F: PrimeField,
    P: ZincPCSTypes<Zt, F, D>,
{
    pub trace: ProjectedTrace<F>,
    pub opening_witness: PCSProverData<P, Zt, F, D>,
}

pub trait ProductionShaFoldedPcsOpen<Zt, F, const D: usize>: ZincPCSTypes<Zt, F, D>
where
    Zt: ZincTypes<D>,
    F: PrimeField,
{
    #[allow(clippy::too_many_arguments)]
    fn prove_folded_pcs_opening(
        pcs_params: &PCSParams<Self, Zt, F, D>,
        folded_commitments: &PCSCommitments<Self, Zt, F, D>,
        folded_trace: &ProjectedTrace<F>,
        folded_prover_data: &PCSProverData<Self, Zt, F, D>,
        r_0: &[F],
        folded_lifted_evals: &[DynamicPolynomialF<F>],
        field_cfg: &F::Config,
    ) -> Result<PCSOpeningProof<Self, Zt, F, D>, ProductionShaError<F>>
    where
        Self: Sized;
}

impl<Zt, F, C, const D: usize> ProductionShaFoldedPcsOpen<Zt, F, D> for AllHyraxPCSTypes<C>
where
    Zt: ZincTypes<D>,
    F: HyraxFieldBridge<C> + DelayedFieldProductSum,
    F::Inner: Transcribable,
    F::Modulus: Transcribable,
    C: AffineRepr,
    HyraxPCS<C, BinaryLanes>: PCS<
            F,
            BinaryPoly<D>,
            D,
            CommitmentKey = zip_plus::pcs::hyrax::HyraxCommitmentKey<C>,
            ProverData = zip_plus::pcs::hyrax::HyraxProverData<C>,
            OpeningProof = Vec<u8>,
        >,
    HyraxPCS<C, DensePolyScalarLanes>: PCS<
            F,
            DensePolynomial<Zt::Int, D>,
            D,
            CommitmentKey = zip_plus::pcs::hyrax::HyraxCommitmentKey<C>,
            ProverData = zip_plus::pcs::hyrax::HyraxProverData<C>,
            OpeningProof = Vec<u8>,
        >,
    HyraxPCS<C, IntScalarLane>: PCS<
            F,
            Zt::Int,
            D,
            CommitmentKey = zip_plus::pcs::hyrax::HyraxCommitmentKey<C>,
            ProverData = zip_plus::pcs::hyrax::HyraxProverData<C>,
            OpeningProof = Vec<u8>,
        >,
{
    fn prove_folded_pcs_opening(
        pcs_params: &PCSParams<Self, Zt, F, D>,
        folded_commitments: &PCSCommitments<Self, Zt, F, D>,
        folded_trace: &ProjectedTrace<F>,
        folded_prover_data: &PCSProverData<Self, Zt, F, D>,
        r_0: &[F],
        folded_lifted_evals: &[DynamicPolynomialF<F>],
        field_cfg: &F::Config,
    ) -> Result<PCSOpeningProof<Self, Zt, F, D>, ProductionShaError<F>> {
        prove_production_sha_hyrax_pcs_opening::<C, Zt, F, D>(
            pcs_params,
            folded_commitments,
            folded_trace,
            folded_prover_data,
            r_0,
            folded_lifted_evals,
            field_cfg,
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedShaSumFold<F: PrimeField> {
    pub r_b: Vec<F>,
    pub c_sf: F,
}

pub type LinearIdealFoldError<F> = ProductionShaError<F>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldedRowSumcheckOutput<F> {
    pub r_star: Vec<F>,
    pub r_star_eq_weights: Vec<F>,
    pub terminal_value: F,
    pub endpoint_evals: Option<ShaEndpointEvals<F>>,
}

#[derive(Debug, Error)]
pub enum ProductionShaError<F: PrimeField> {
    #[error("production SHA requires at least two fresh instances, got {0}")]
    InstanceCountTooSmall(usize),
    #[error("instance count must be a power of two, got {0}")]
    InstanceCountNotPowerOfTwo(usize),
    #[error("length mismatch for {label}: got {got}, expected {expected}")]
    LengthMismatch {
        label: &'static str,
        got: usize,
        expected: usize,
    },
    #[error("non-canonical proof object: {0}")]
    NonCanonicalProofObject(&'static str),
    #[error("production SHA public selector column {col:?} is not boolean at row {row}")]
    NonBooleanPublicSelector { col: ShaPublicCol, row: usize },
    #[error("production SHA public selector column {col:?} is all zero")]
    EmptyPublicSelector { col: ShaPublicCol },
    #[error(
        "production SHA public selector column {col:?} does not match the fixed row layout at row {row}"
    )]
    InvalidPublicSelector { col: ShaPublicCol, row: usize },
    #[error("production SHA public K column does not match SHA-256 constants at row {row}")]
    InvalidRoundConstant { row: usize },
    #[error("production SHA requires {expected}-bit word polynomials, got D={got}")]
    UnsupportedProductionShaWordDegree { got: usize, expected: usize },
    #[error("unsupported production SHA PCS shape: {0}")]
    UnsupportedProductionShaPcsShape(&'static str),
    #[error("production SHA prover not implemented: {0}")]
    ProverNotImplemented(&'static str),
    #[error("PCS opening transcript has trailing bytes")]
    TrailingPcsOpeningBytes,
    #[error("{label} expected exactly one sumcheck group, got {got}")]
    UnexpectedSumcheckGroupCount { label: &'static str, got: usize },
    #[error("SumFold proof has degree {degree}, expected at most 3")]
    SumFoldDegreeTooHigh { degree: usize },
    #[error("SumFold terminal evaluation mismatch")]
    SumFoldTerminalMismatch,
    #[error("row sumcheck proof has degree {degree}, expected at most 3")]
    RowSumcheckDegreeTooHigh { degree: usize },
    #[error("row sumcheck terminal evaluation mismatch")]
    RowSumcheckTerminalMismatch,
    #[error("endpoint scalarization mismatch for {col:?} shift {shift}")]
    EndpointScalarizationMismatch { col: ShaWordCol, shift: usize },
    #[error("missing endpoint eval for {col:?} shift {shift}")]
    MissingEndpointEval { col: ShaWordCol, shift: usize },
    #[error("ideal membership failed")]
    IdealMembership,
    #[error("PCS error: {0}")]
    Pcs(#[from] ZipError),
    #[error("sumcheck error: {0}")]
    Sumcheck(#[from] SumCheckError<F>),
    #[error("multipoint evaluation error: {0}")]
    Multipoint(#[from] MultipointEvalError<F>),
    #[error("SumFold error: {0}")]
    SumFold(#[from] SumFoldError),
    #[error("SHA projection error: {0}")]
    ShaProjection(#[from] ShaProjectionError),
    #[error("equality polynomial error: {0}")]
    Eq(#[from] ArithErrors),
    #[error("polynomial evaluation error: {0}")]
    PolyEval(#[from] EvaluationError),
}

pub fn absorb_projected_sha_publics<F>(
    transcript: &mut impl Transcript,
    publics: &[zinc_piop::neutron_nova::ProjectedPublic<F>],
    field_cfg: &F::Config,
) where
    F: PrimeField,
    F::Inner: Transcribable,
    F::Modulus: Transcribable,
{
    let mut field_buf = runtime_field_transcript_buf::<F>(field_cfg);
    let mut encoded = Vec::with_capacity(
        publics.len()
            * ShaPublicWordCol::COUNT
            * SHA_ROW_COUNT
            * F::Inner::get_num_bytes(F::zero_with_cfg(field_cfg).inner()),
    );
    let zero = F::zero_with_cfg(field_cfg);

    fn push_u64(buf: &mut Vec<u8>, value: usize) {
        buf.extend_from_slice(&(value as u64).to_le_bytes());
    }

    fn push_field_inners<F>(buf: &mut Vec<u8>, values: &[F], scratch: &mut [u8])
    where
        F: PrimeField,
        F::Inner: Transcribable,
    {
        for value in values {
            value.inner().write_transcription_bytes_exact(scratch);
            buf.extend_from_slice(scratch);
        }
    }

    transcript.absorb_slice(b"production_sha_publics_begin");
    encoded.extend_from_slice(b"compact_v1");
    zero.modulus()
        .write_transcription_bytes_exact(&mut field_buf);
    encoded.extend_from_slice(&field_buf);
    push_u64(&mut encoded, publics.len());
    push_u64(&mut encoded, ShaPublicCol::COUNT);
    push_u64(&mut encoded, ShaPublicWordCol::COUNT);
    push_u64(&mut encoded, SHA_ROW_COUNT);
    for (instance_idx, public) in publics.iter().enumerate() {
        push_u64(&mut encoded, instance_idx);
        push_u64(&mut encoded, public.columns.len());
        match &public.bit_slices {
            Some(bit_slices) => {
                encoded.push(1);
                push_u64(&mut encoded, bit_slices.len());
            }
            None => encoded.push(0),
        }
        for public_col in production_sha_public_word_column_map() {
            let col_idx = public_col.index();
            let col = &public.columns[col_idx];
            push_u64(&mut encoded, col_idx);
            push_u64(&mut encoded, col.evaluations.len());
            push_field_inners::<F>(&mut encoded, &col.evaluations, &mut field_buf);
        }
    }
    transcript.absorb_slice(&encoded);
    transcript.absorb_slice(b"production_sha_publics_end");
}

fn absorb_uair_shape_metadata<U: Uair>(transcript: &mut impl Transcript, shape: &UairShape<U>) {
    let sig = &shape.signature;

    transcript.absorb_slice(b"uair_shape_metadata_begin");
    transcript.absorb_slice(&(shape.num_vars as u64).to_le_bytes());
    absorb_uair_column_counts(
        transcript,
        sig.total_cols().num_binary_poly_cols(),
        sig.total_cols().num_arbitrary_poly_cols(),
        sig.total_cols().num_int_cols(),
    );
    absorb_uair_column_counts(
        transcript,
        sig.public_cols().num_binary_poly_cols(),
        sig.public_cols().num_arbitrary_poly_cols(),
        sig.public_cols().num_int_cols(),
    );
    absorb_uair_column_counts(
        transcript,
        sig.witness_cols().num_binary_poly_cols(),
        sig.witness_cols().num_arbitrary_poly_cols(),
        sig.witness_cols().num_int_cols(),
    );
    transcript.absorb_slice(&(sig.shifts().len() as u64).to_le_bytes());
    for shift in sig.shifts() {
        transcript.absorb_slice(&(shift.source_col() as u64).to_le_bytes());
        transcript.absorb_slice(&(shift.shift_amount() as u64).to_le_bytes());
    }
    transcript.absorb_slice(b"uair_shape_metadata_end");
}

fn absorb_uair_column_counts(
    transcript: &mut impl Transcript,
    binary: usize,
    arbitrary: usize,
    int: usize,
) {
    transcript.absorb_slice(&(binary as u64).to_le_bytes());
    transcript.absorb_slice(&(arbitrary as u64).to_le_bytes());
    transcript.absorb_slice(&(int as u64).to_le_bytes());
}

fn runtime_field_transcript_buf<F>(field_cfg: &F::Config) -> Vec<u8>
where
    F: PrimeField,
    F::Inner: Transcribable,
{
    vec![0u8; F::zero_with_cfg(field_cfg).inner().get_num_bytes()]
}

fn absorb_sha_resolver_proof<F>(
    transcript: &mut impl Transcript,
    proof: &CombinedPolyResolverProof<F>,
    field_cfg: &F::Config,
) where
    F: PrimeField,
    F::Inner: Transcribable,
    F::Modulus: Transcribable,
{
    let mut field_buf = runtime_field_transcript_buf::<F>(field_cfg);
    fn absorb_vec<F>(
        transcript: &mut impl Transcript,
        label: &'static [u8],
        values: &[F],
        field_buf: &mut [u8],
    ) where
        F: PrimeField,
        F::Inner: Transcribable,
        F::Modulus: Transcribable,
    {
        transcript.absorb_slice(label);
        transcript.absorb_slice(&(values.len() as u64).to_le_bytes());
        transcript.absorb_random_field_slice(values, field_buf);
    }

    transcript.absorb_slice(b"production_sha_resolver_begin");
    absorb_vec(transcript, b"up", &proof.up_evals, &mut field_buf);
    absorb_vec(transcript, b"down", &proof.down_evals, &mut field_buf);
    absorb_vec(
        transcript,
        b"bit_slice",
        &proof.bit_slice_evals,
        &mut field_buf,
    );
    absorb_vec(
        transcript,
        b"bit_op_down",
        &proof.bit_op_down_evals,
        &mut field_buf,
    );
    absorb_vec(
        transcript,
        b"shifted_bit_slice",
        &proof.shifted_bit_slice_evals,
        &mut field_buf,
    );
    transcript.absorb_slice(b"production_sha_resolver_end");
}

fn absorb_public_uair_trace<Zt, const D: usize>(
    transcript: &mut impl Transcript,
    instance_idx: usize,
    trace: &UairTrace<'_, Zt::Int, Zt::Int, D>,
) where
    Zt: ZincTypes<D>,
{
    fn push_u64(buf: &mut Vec<u8>, value: usize) {
        buf.extend_from_slice(&(value as u64).to_le_bytes());
    }

    fn push_transcribable<T: Transcribable>(buf: &mut Vec<u8>, value: &T, scratch: &mut Vec<u8>) {
        let len = value.get_num_bytes() + T::LENGTH_NUM_BYTES;
        scratch.resize(len, 0);
        value.write_transcription_bytes_subset(scratch);
        buf.extend_from_slice(scratch);
    }

    let mut encoded = Vec::new();
    let mut scratch = Vec::new();
    encoded.extend_from_slice(b"compact_v1");
    push_u64(&mut encoded, instance_idx);
    push_u64(&mut encoded, trace.binary_poly.len());
    for (col_idx, col) in trace.binary_poly.iter().enumerate() {
        push_u64(&mut encoded, col_idx);
        push_u64(&mut encoded, col.num_vars);
        push_u64(&mut encoded, col.evaluations.len());
        for value in &col.evaluations {
            push_transcribable(&mut encoded, value, &mut scratch);
        }
    }
    push_u64(&mut encoded, trace.arbitrary_poly.len());
    for (col_idx, col) in trace.arbitrary_poly.iter().enumerate() {
        push_u64(&mut encoded, col_idx);
        push_u64(&mut encoded, col.num_vars);
        push_u64(&mut encoded, col.evaluations.len());
        for poly in &col.evaluations {
            for coeff in poly.iter() {
                push_transcribable(&mut encoded, coeff, &mut scratch);
            }
        }
    }
    push_u64(&mut encoded, trace.int.len());
    for (col_idx, col) in trace.int.iter().enumerate() {
        push_u64(&mut encoded, col_idx);
        push_u64(&mut encoded, col.num_vars);
        push_u64(&mut encoded, col.evaluations.len());
        for value in &col.evaluations {
            push_transcribable(&mut encoded, value, &mut scratch);
        }
    }

    transcript.absorb_slice(b"uair_public_trace_begin");
    transcript.absorb_slice(&encoded);
    transcript.absorb_slice(b"uair_public_trace_end");
}

fn absorb_production_sha_statement_metadata(transcript: &mut impl Transcript) {
    transcript.absorb_slice(PRODUCTION_SHA_FRESH_BATCH_DOMAIN);
    transcript.absorb_slice(b"production_sha_statement_metadata_begin");

    transcript.absorb_slice(b"row_layout");
    transcript.absorb_slice(&(SHA_ROW_VARS as u64).to_le_bytes());
    transcript.absorb_slice(&(SHA_ROW_COUNT as u64).to_le_bytes());
    for (start, end) in [(0u64, 3u64), (0, 15), (0, 47), (0, 63), (64, 67), (68, 71)] {
        transcript.absorb_slice(&start.to_le_bytes());
        transcript.absorb_slice(&end.to_le_bytes());
    }

    transcript.absorb_slice(b"sha_word_column_order");
    for col in ShaWordCol::ALL {
        transcript.absorb_slice(&(col.index() as u64).to_le_bytes());
    }
    transcript.absorb_slice(b"sha_int_column_order");
    for col in ShaIntCol::ALL {
        transcript.absorb_slice(&(col.index() as u64).to_le_bytes());
    }
    transcript.absorb_slice(b"sha_public_column_order");
    for col in ShaPublicCol::ALL {
        transcript.absorb_slice(&(col.index() as u64).to_le_bytes());
    }

    transcript.absorb_slice(b"sha_residual_family_order");
    for family in ShaResidualFamily::ALL {
        transcript.absorb_slice(&(family.index() as u64).to_le_bytes());
    }
    transcript.absorb_slice(b"sha_nonzero_ideal_ids");
    for family in production_sha_nonzero_families() {
        transcript.absorb_slice(&(family.index() as u64).to_le_bytes());
        let ideal_id: &[u8] = match family {
            ShaResidualFamily::R0BigSigmaA | ShaResidualFamily::R1BigSigmaE => b"X32_MINUS_1",
            ShaResidualFamily::R4Schedule
            | ShaResidualFamily::R5UpdateA
            | ShaResidualFamily::R6UpdateE
            | ShaResidualFamily::R9FeedForwardA
            | ShaResidualFamily::R10FeedForwardE => b"X_MINUS_2",
            _ => b"UNEXPECTED_NONZERO_IDEAL",
        };
        transcript.absorb_slice(ideal_id);
    }

    transcript.absorb_slice(b"sha256_k_constants");
    for constant in SHA256_ROUND_CONSTANTS {
        transcript.absorb_slice(&(constant as u64).to_le_bytes());
    }

    transcript.absorb_slice(b"production_sha_statement_metadata_end");
}

pub fn absorb_production_sha_commitments<P, Zt, F, const D: usize>(
    transcript: &mut impl Transcript,
    label: &'static [u8],
    commitments: &[PCSCommitments<P, Zt, F, D>],
) where
    Zt: ZincTypes<D>,
    F: PrimeField,
    P: ZincPCSTypes<Zt, F, D>,
{
    transcript.absorb_slice(label);
    transcript.absorb_slice(&(commitments.len() as u64).to_le_bytes());
    for (instance_idx, commitment) in commitments.iter().enumerate() {
        transcript.absorb_slice(&(instance_idx as u64).to_le_bytes());
        P::BinaryPCS::absorb_commitment(transcript, &commitment.binary);
        P::ArbitraryPCS::absorb_commitment(transcript, &commitment.arbitrary);
        P::IntPCS::absorb_commitment(transcript, &commitment.int);
    }
}

pub fn absorb_fresh_sha_ideal_polys<F>(
    transcript: &mut impl Transcript,
    ideal_polys: &[[DynamicPolynomialF<F>; NUM_NONZERO_SHA_FAMILIES]],
    field_cfg: &F::Config,
) where
    F: PrimeField,
    F::Inner: Transcribable,
    F::Modulus: Transcribable,
{
    let mut field_buf = runtime_field_transcript_buf::<F>(field_cfg);
    transcript.absorb_slice(b"production_sha_fresh_ideals_begin");
    transcript.absorb_slice(&(ideal_polys.len() as u64).to_le_bytes());
    for (instance_idx, instance) in ideal_polys.iter().enumerate() {
        transcript.absorb_slice(&(instance_idx as u64).to_le_bytes());
        transcript.absorb_slice(&(instance.len() as u64).to_le_bytes());
        for (family_idx, poly) in instance.iter().enumerate() {
            transcript.absorb_slice(&(family_idx as u64).to_le_bytes());
            transcript.absorb_slice(&(poly.coeffs.len() as u64).to_le_bytes());
            transcript.absorb_random_field_slice(&poly.coeffs, &mut field_buf);
        }
    }
    transcript.absorb_slice(b"production_sha_fresh_ideals_end");
}

pub fn absorb_aggregate_sha_ideal_polys<F>(
    transcript: &mut impl Transcript,
    ideal_polys: &[DynamicPolynomialF<F>; NUM_NONZERO_SHA_FAMILIES],
    field_cfg: &F::Config,
) where
    F: PrimeField,
    F::Inner: Transcribable,
    F::Modulus: Transcribable,
{
    let mut field_buf = runtime_field_transcript_buf::<F>(field_cfg);
    transcript.absorb_slice(b"production_sha_aggregate_ideals_begin");
    transcript.absorb_slice(&(ideal_polys.len() as u64).to_le_bytes());
    for (family_idx, poly) in ideal_polys.iter().enumerate() {
        transcript.absorb_slice(&(family_idx as u64).to_le_bytes());
        transcript.absorb_slice(&(poly.coeffs.len() as u64).to_le_bytes());
        transcript.absorb_random_field_slice(&poly.coeffs, &mut field_buf);
    }
    transcript.absorb_slice(b"production_sha_aggregate_ideals_end");
}

pub fn absorb_sha_endpoint_evals<F>(
    transcript: &mut impl Transcript,
    endpoint_evals: &ShaEndpointEvals<F>,
    field_cfg: &F::Config,
) where
    F: PrimeField,
    F::Inner: Transcribable,
    F::Modulus: Transcribable,
{
    let mut field_buf = runtime_field_transcript_buf::<F>(field_cfg);
    transcript.absorb_slice(b"production_sha_endpoint_evals_begin");
    transcript.absorb_slice(&(endpoint_evals.sources.len() as u64).to_le_bytes());
    for source in &endpoint_evals.sources {
        transcript.absorb_slice(&(source.col.index() as u64).to_le_bytes());
        transcript.absorb_slice(&(source.shift as u64).to_le_bytes());
        transcript.absorb_random_field(&source.scalarized, &mut field_buf);
        transcript.absorb_random_field_slice(&source.bits, &mut field_buf);
    }
    transcript.absorb_slice(&(endpoint_evals.int_sources.len() as u64).to_le_bytes());
    for source in &endpoint_evals.int_sources {
        transcript.absorb_slice(&(source.col.index() as u64).to_le_bytes());
        transcript.absorb_random_field(&source.scalar, &mut field_buf);
    }
    transcript.absorb_slice(b"production_sha_endpoint_evals_end");
}

pub fn absorb_folded_lifted_evals<F>(
    transcript: &mut impl Transcript,
    lifted_evals: &[DynamicPolynomialF<F>],
    field_cfg: &F::Config,
) where
    F: PrimeField,
    F::Inner: Transcribable,
    F::Modulus: Transcribable,
{
    let mut field_buf = runtime_field_transcript_buf::<F>(field_cfg);
    transcript.absorb_slice(b"production_sha_folded_lifted_evals_begin");
    transcript.absorb_slice(&(lifted_evals.len() as u64).to_le_bytes());
    for (idx, lifted_eval) in lifted_evals.iter().enumerate() {
        transcript.absorb_slice(&(idx as u64).to_le_bytes());
        transcript.absorb_slice(&(lifted_eval.coeffs.len() as u64).to_le_bytes());
        transcript.absorb_random_field_slice(&lifted_eval.coeffs, &mut field_buf);
    }
    transcript.absorb_slice(b"production_sha_folded_lifted_evals_end");
}

pub fn sample_pre_ideal_challenge<F>(
    transcript: &mut impl Transcript,
    field_cfg: &F::Config,
) -> [F; SHA_ROW_VARS]
where
    F: DelayedFieldProductSum,
    F::Inner: Transcribable,
{
    std::array::from_fn(|_| transcript.get_transcribable_field_challenge(field_cfg))
}

pub fn sample_instance_batch_challenge<F>(
    transcript: &mut impl Transcript,
    instance_count: usize,
    field_cfg: &F::Config,
) -> Result<Vec<F>, ProductionShaError<F>>
where
    F: PrimeField,
    F::Inner: Transcribable,
{
    if !instance_count.is_power_of_two() {
        return Err(ProductionShaError::InstanceCountNotPowerOfTwo(
            instance_count,
        ));
    }
    let ell = usize::try_from(instance_count.trailing_zeros()).expect("ell fits usize");
    Ok(transcript.get_transcribable_field_challenges(ell, field_cfg))
}

pub fn sample_post_aggregate_ideal_challenges<F>(
    transcript: &mut impl Transcript,
    field_cfg: &F::Config,
) -> (F, F, F, F)
where
    F: PrimeField,
    F::Inner: Transcribable,
{
    (
        transcript.get_transcribable_field_challenge(field_cfg),
        transcript.get_transcribable_field_challenge(field_cfg),
        transcript.get_transcribable_field_challenge(field_cfg),
        transcript.get_transcribable_field_challenge(field_cfg),
    )
}

pub fn sample_post_ideal_challenges<F>(
    transcript: &mut impl Transcript,
    instance_count: usize,
    field_cfg: &F::Config,
) -> Result<(F, F, F, F, Vec<F>), ProductionShaError<F>>
where
    F: PrimeField,
    F::Inner: Transcribable,
{
    let (a, lambda, rho, xi) = sample_post_aggregate_ideal_challenges(transcript, field_cfg);
    Ok((
        a,
        lambda,
        rho,
        xi,
        sample_instance_batch_challenge(transcript, instance_count, field_cfg)?,
    ))
}

pub fn check_fresh_sha_ideal_membership<F>(
    ideal_polys: &[[DynamicPolynomialF<F>; NUM_NONZERO_SHA_FAMILIES]],
    field_cfg: &F::Config,
) -> Result<(), ProductionShaError<F>>
where
    F: DelayedFieldProductSum,
{
    verify_fresh_sha_ideal_polys(ideal_polys, field_cfg)?;
    Ok(())
}

pub fn check_aggregate_sha_ideal_membership<F>(
    ideal_polys: &[DynamicPolynomialF<F>; NUM_NONZERO_SHA_FAMILIES],
    field_cfg: &F::Config,
) -> Result<(), ProductionShaError<F>>
where
    F: PrimeField,
    F::Inner: Transcribable,
    F::Modulus: Transcribable,
{
    verify_fresh_sha_ideal_polys(std::slice::from_ref(ideal_polys), field_cfg)?;
    Ok(())
}

fn ensure_production_sha_word_degree<F, const D: usize>() -> Result<(), ProductionShaError<F>>
where
    F: PrimeField,
{
    if D != SHA_WORD_BITS {
        return Err(ProductionShaError::UnsupportedProductionShaWordDegree {
            got: D,
            expected: SHA_WORD_BITS,
        });
    }
    Ok(())
}

fn validate_production_sha_batch_sizes<F>(
    binary: usize,
    arbitrary: usize,
    int: usize,
) -> Result<(), ProductionShaError<F>>
where
    F: PrimeField,
{
    if binary != ShaWordCol::COUNT {
        return Err(ProductionShaError::UnsupportedProductionShaPcsShape(
            "production SHA expects one binary commitment batch per SHA word column",
        ));
    }
    if arbitrary != 0 {
        return Err(ProductionShaError::UnsupportedProductionShaPcsShape(
            "production SHA expects no arbitrary witness columns",
        ));
    }
    if int != ShaIntCol::COUNT {
        return Err(ProductionShaError::UnsupportedProductionShaPcsShape(
            "production SHA expects one int commitment batch per SHA int column",
        ));
    }
    Ok(())
}

pub fn commit_production_sha_instance<P, Zt, F, const D: usize>(
    pcs_params: &PCSParams<P, Zt, F, D>,
    witness_polys: &ProductionShaWitnessPolys<Zt, D>,
) -> Result<(PCSProverData<P, Zt, F, D>, PCSCommitments<P, Zt, F, D>), ProductionShaError<F>>
where
    Zt: ZincTypes<D>,
    F: PrimeField,
    P: ZincPCSTypes<Zt, F, D>,
    P::BinaryPCS: FoldablePCS<F, BinaryPoly<D>, D>,
    P::ArbitraryPCS: FoldablePCS<F, DensePolynomial<Zt::Int, D>, D>,
    P::IntPCS: FoldablePCS<F, Zt::Int, D>,
{
    ensure_production_sha_word_degree::<F, D>()?;
    validate_production_sha_batch_sizes::<F>(
        witness_polys.binary.len(),
        witness_polys.arbitrary.len(),
        witness_polys.int.len(),
    )?;
    let (binary_data, binary_commitment) =
        P::BinaryPCS::commit(&pcs_params.binary, &witness_polys.binary)?;
    let (arbitrary_data, arbitrary_commitment) =
        P::ArbitraryPCS::commit(&pcs_params.arbitrary, &witness_polys.arbitrary)?;
    let (int_data, int_commitment) = P::IntPCS::commit(&pcs_params.int, &witness_polys.int)?;
    Ok((
        PCSProverData {
            binary: binary_data,
            arbitrary: arbitrary_data,
            int: int_data,
        },
        PCSCommitments {
            binary: binary_commitment,
            arbitrary: arbitrary_commitment,
            int: int_commitment,
        },
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn prove_linear_ideal_fold<P, U, Zt, F, const D: usize>(
    pp: &LinearIdealFoldProverParams<P, U, Zt, F, D>,
    shape: &UairShape<U>,
    witnesses: &[UairWitness<'_, Zt::Int, Zt::Int, D>],
    transcript: &mut impl Transcript,
) -> Result<
    LinearIdealFoldProveOutput<
        UairInstance<'static, Zt::Int, Zt::Int, PCSCommitments<P, Zt, F, D>, D>,
        FoldedLinearIdealInstance<F, PCSCommitments<P, Zt, F, D>, ProjectedPublic<F>>,
        FoldedLinearIdealWitness<ProductionShaFoldedWitness<P, Zt, F, D>>,
        ProductionLinearIdealFoldProof<P, Zt, F, D>,
    >,
    LinearIdealFoldError<F>,
>
where
    U: Uair + ProductionShaProjectionAdapter<Zt, F, D> + Sync,
    Zt: ZincTypes<D>,
    F: InnerTransparentField
        + DelayedFieldProductSum
        + ShaBinaryFoldField
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: Transcribable + Zero + Default + Send + Sync,
    F::Modulus: Transcribable,
    P: ZincPCSTypes<Zt, F, D>,
    P: ProductionShaFoldedPcsOpen<Zt, F, D>,
    P::BinaryPCS: FoldablePCS<F, BinaryPoly<D>, D>,
    P::ArbitraryPCS: FoldablePCS<F, DensePolynomial<Zt::Int, D>, D>,
    P::IntPCS: FoldablePCS<F, Zt::Int, D>,
{
    let field_cfg = &pp.field_cfg;
    ensure_production_sha_word_degree::<F, D>()?;
    if witnesses.len() < 2 {
        return Err(ProductionShaError::InstanceCountTooSmall(witnesses.len()));
    }
    if !witnesses.len().is_power_of_two() {
        return Err(ProductionShaError::InstanceCountNotPowerOfTwo(
            witnesses.len(),
        ));
    }

    let booleanity_sources = production_sha_booleanity_sources();
    absorb_production_sha_statement_metadata(transcript);
    absorb_uair_shape_metadata(transcript, shape);

    let (fresh_instances, instance_commitments, instance_prover_data, traces, publics) =
        prove_fresh_instances_phase::<P, U, Zt, F, D>(
            &pp.pcs_params,
            shape,
            witnesses,
            transcript,
            field_cfg,
        )?;
    validate_production_sha_publics(&publics, field_cfg)?;

    tracing::info_span!(
        target: "zinc_protocol::production_sha",
        "absorb_fresh_commitments",
        side = "prove",
        phase = "absorb_fresh_commitments",
    )
    .in_scope(|| {
        absorb_production_sha_commitments::<P, Zt, F, D>(
            transcript,
            b"production_sha_fresh_commitments",
            &instance_commitments,
        )
    });
    tracing::info_span!(
        target: "zinc_protocol::production_sha",
        "absorb_projected_publics",
        side = "prove",
        phase = "absorb_projected_publics",
    )
    .in_scope(|| absorb_projected_sha_publics(transcript, &publics, field_cfg));

    let r_ic = sample_pre_ideal_challenge(transcript, field_cfg);
    let r_ic_eq_weights = build_eq_x_r_vec(&r_ic, field_cfg)?;
    let coeff_tables =
        build_residual_coeff_tables_phase(&traces, &publics, &r_ic_eq_weights, field_cfg)?;

    let beta = sample_instance_batch_challenge(transcript, witnesses.len(), field_cfg)?;
    let beta_eq_weights = build_eq_x_r_vec(&beta, field_cfg)?;
    let (ideal_check, aggregate_ideal_polys) =
        prove_aggregate_ideal_phase(&coeff_tables, &beta_eq_weights, transcript, field_cfg)?;

    let (a, lambda, rho, xi) = sample_post_aggregate_ideal_challenges(transcript, field_cfg);
    let a_powers = build_sha_residual_eval_powers(&a, field_cfg);
    let lambda_powers = build_sha_lambda_powers(&lambda, field_cfg);
    let booleanity_weights =
        build_booleanity_weights(&rho, &xi, booleanity_sources.len(), field_cfg);
    let initial_claim = evaluate_aggregate_sha_ideal_claim_with_powers(
        &aggregate_ideal_polys,
        &a_powers,
        &lambda_powers,
        field_cfg,
    )?;

    let (_linear_accumulator, _quadratic_prefix_accumulator, sumfold_group) =
        build_sumfold_accumulators_phase(
            &traces,
            &beta,
            &beta_eq_weights,
            &r_ic_eq_weights,
            &coeff_tables,
            &a_powers,
            &lambda_powers,
            &booleanity_weights,
            &booleanity_sources,
            pp.prefix_vars,
            field_cfg,
        )?;

    let (sumfold_proof, sumfold_r_b) = prove_sumfold_phase(
        transcript,
        sumfold_group,
        &initial_claim,
        beta.len(),
        field_cfg,
    )?;

    let provisional_sumfold_output = derive_instance_fold_claim(
        &beta,
        sumfold_r_b.clone(),
        F::one_with_cfg(field_cfg),
        witnesses.len(),
        field_cfg,
    )?;

    let (folded, folded_public, row_claim, sumfold_output, folded_commitments, folded_prover_data) =
        prove_fold_after_sumfold_phase::<P, Zt, F, D>(
            &traces,
            &publics,
            &provisional_sumfold_output,
            &beta,
            sumfold_r_b,
            &r_ic_eq_weights,
            &a_powers,
            &lambda_powers,
            &booleanity_weights,
            &booleanity_sources,
            &instance_commitments,
            &instance_prover_data,
            field_cfg,
        )?;
    absorb_production_sha_commitments::<P, Zt, F, D>(
        transcript,
        b"production_sha_derived_folded_commitments",
        std::slice::from_ref(&folded_commitments),
    );

    verify_folded_row_sumcheck_claim(&row_claim, sumfold_output.final_round_sumcheck_claim())?;
    let (combined_sumcheck, row_output) = prove_row_sumcheck_phase(
        transcript,
        &folded.trace,
        &folded_public,
        &r_ic,
        &r_ic_eq_weights,
        &a_powers,
        &lambda_powers,
        &booleanity_weights,
        &booleanity_sources,
        &row_claim,
        field_cfg,
    )?;

    let (resolver, _resolver_endpoint_evals, multipoint_eval, r_0) =
        prove_endpoint_multipoint_phase(
            transcript,
            &folded.trace,
            &folded_public,
            &row_output,
            &r_ic,
            &a,
            &a_powers,
            &lambda_powers,
            &booleanity_weights,
            &booleanity_sources,
            field_cfg,
        )?;

    let r_0_eq_weights = build_eq_x_r_vec(&r_0, field_cfg)?;
    let (witness_lifted_evals, opening_proof) = prove_pcs_opening_phase::<P, Zt, F, D>(
        transcript,
        &folded.trace,
        &folded_commitments,
        &folded_prover_data,
        &r_0,
        &r_0_eq_weights,
        &pp.pcs_params,
        field_cfg,
    )?;

    Ok(LinearIdealFoldProveOutput {
        fresh_instances,
        folded_instance: FoldedLinearIdealInstance {
            target: sumfold_output.final_round_sumcheck_claim().clone(),
            commitments: folded_commitments,
            public: folded_public,
        },
        folded_witness: FoldedLinearIdealWitness {
            witness: ProductionShaFoldedWitness {
                trace: folded.trace,
                opening_witness: folded_prover_data,
            },
        },
        proof: ProductionLinearIdealFoldProof {
            instance_commitments,
            ideal_check,
            sumfold_proof,
            resolver,
            combined_sumcheck,
            multipoint_eval,
            witness_lifted_evals,
            opening_proof,
        },
    })
}

#[tracing::instrument(
    target = "zinc_protocol::production_sha",
    level = "info",
    skip_all,
    fields(side = "prove", phase = "fresh_instances", instances = witnesses.len())
)]
#[allow(clippy::too_many_arguments)]
fn prove_fresh_instances_phase<P, U, Zt, F, const D: usize>(
    pcs_params: &PCSParams<P, Zt, F, D>,
    shape: &UairShape<U>,
    witnesses: &[UairWitness<'_, Zt::Int, Zt::Int, D>],
    transcript: &mut impl Transcript,
    field_cfg: &F::Config,
) -> Result<ProductionShaFreshArtifacts<P, Zt, F, D>, ProductionShaError<F>>
where
    U: Uair + ProductionShaProjectionAdapter<Zt, F, D> + Sync,
    Zt: ZincTypes<D>,
    F: PrimeField,
    P: ZincPCSTypes<Zt, F, D>,
    P::BinaryPCS: FoldablePCS<F, BinaryPoly<D>, D>,
    P::ArbitraryPCS: FoldablePCS<F, DensePolynomial<Zt::Int, D>, D>,
    P::IntPCS: FoldablePCS<F, Zt::Int, D>,
{
    for (instance_idx, witness) in witnesses.iter().enumerate() {
        let public_trace = public_uair_trace_view(&witness.trace, &shape.signature)?;
        absorb_public_uair_trace::<Zt, D>(transcript, instance_idx, &public_trace);
    }

    let artifacts = cfg_iter!(witnesses)
        .map(|witness| {
            let public_trace = public_uair_trace_view(&witness.trace, &shape.signature)?;
            let witness_trace = witness_uair_trace_view(&witness.trace, &shape.signature)?;
            let (trace, public, witness_polys) = tracing::info_span!(
                target: "zinc_protocol::production_sha",
                "fresh_project_instance",
                side = "prove",
                phase = "fresh_project_instance",
            )
            .in_scope(|| {
                U::project_production_sha_witness(shape, &public_trace, &witness_trace, field_cfg)
            })?;
            let (data, commitment) = tracing::info_span!(
                target: "zinc_protocol::production_sha",
                "fresh_commit_instance",
                side = "prove",
                phase = "fresh_commit_instance",
            )
            .in_scope(|| {
                commit_production_sha_instance::<P, Zt, F, D>(pcs_params, &witness_polys)
            })?;

            Ok((
                UairInstance {
                    public_trace: own_uair_trace(&public_trace),
                    commitments: commitment.clone(),
                },
                commitment,
                data,
                trace,
                public,
            ))
        })
        .collect::<Result<Vec<_>, ProductionShaError<F>>>()?;

    let mut fresh_instances = Vec::with_capacity(artifacts.len());
    let mut instance_commitments = Vec::with_capacity(artifacts.len());
    let mut instance_prover_data = Vec::with_capacity(artifacts.len());
    let mut traces = Vec::with_capacity(artifacts.len());
    let mut publics = Vec::with_capacity(artifacts.len());

    for (fresh_instance, commitment, data, trace, public) in artifacts {
        fresh_instances.push(fresh_instance);
        instance_commitments.push(commitment);
        instance_prover_data.push(data);
        traces.push(trace);
        publics.push(public);
    }

    Ok((
        fresh_instances,
        instance_commitments,
        instance_prover_data,
        traces,
        publics,
    ))
}

#[tracing::instrument(
    target = "zinc_protocol::production_sha",
    level = "info",
    skip_all,
    fields(side = "prove", phase = "residual_coeff_tables", instances = traces.len())
)]
fn build_residual_coeff_tables_phase<F>(
    traces: &[ProjectedTrace<F>],
    publics: &[ProjectedPublic<F>],
    r_ic_eq_weights: &[F],
    field_cfg: &F::Config,
) -> Result<Vec<LinearResidualCoeffTable<F>>, ProductionShaError<F>>
where
    F: PrimeField,
    F::Inner: Transcribable,
    F::Modulus: Transcribable,
{
    build_linear_residual_coeff_tables_with_row_weights(traces, publics, r_ic_eq_weights, field_cfg)
        .map_err(ProductionShaError::from)
}

#[tracing::instrument(
    target = "zinc_protocol::production_sha",
    level = "info",
    skip_all,
    fields(side = "prove", phase = "aggregate_ideal", instances = coeff_tables.len())
)]
fn prove_aggregate_ideal_phase<F>(
    coeff_tables: &[LinearResidualCoeffTable<F>],
    beta_eq_weights: &[F],
    transcript: &mut impl Transcript,
    field_cfg: &F::Config,
) -> Result<
    (
        IdealCheckProof<F>,
        [DynamicPolynomialF<F>; NUM_NONZERO_SHA_FAMILIES],
    ),
    ProductionShaError<F>,
>
where
    F: PrimeField,
    F::Inner: Transcribable,
    F::Modulus: Transcribable,
{
    let aggregate_ideal_polys =
        beta_aggregate_nonzero_ideal_polys_with_weights(coeff_tables, beta_eq_weights)?;
    let ideal_check = IdealCheckProof {
        combined_mle_values: aggregate_ideal_polys.iter().cloned().collect(),
    };
    #[cfg(debug_assertions)]
    check_aggregate_sha_ideal_membership(&aggregate_ideal_polys, field_cfg)?;
    absorb_aggregate_sha_ideal_polys(transcript, &aggregate_ideal_polys, field_cfg);
    Ok((ideal_check, aggregate_ideal_polys))
}

#[tracing::instrument(
    target = "zinc_protocol::production_sha",
    level = "info",
    skip_all,
    fields(
        side = "prove",
        phase = "sumfold_accumulators",
        instances = traces.len(),
        prefix_vars,
    )
)]
#[allow(clippy::too_many_arguments)]
fn build_sumfold_accumulators_phase<F>(
    traces: &[ProjectedTrace<F>],
    beta: &[F],
    beta_eq_weights: &[F],
    r_ic_eq_weights: &[F],
    coeff_tables: &[LinearResidualCoeffTable<F>],
    a_powers: &[F],
    lambda_powers: &[F],
    booleanity_weights: &[F],
    booleanity_sources: &[ShaBooleanitySource],
    prefix_vars: usize,
    field_cfg: &F::Config,
) -> Result<ProductionShaSumfoldAccumulators<F>, ProductionShaError<F>>
where
    F: InnerTransparentField + DelayedFieldProductSum + Send + Sync + 'static,
    F::Inner: Zero,
{
    let linear_accumulator = tracing::info_span!(
        target: "zinc_protocol::production_sha",
        "sumfold_linear_accumulator",
        side = "prove",
        phase = "sumfold_linear_accumulator",
    )
    .in_scope(|| {
        build_sha_sumfold_linear_accumulator(coeff_tables, a_powers, lambda_powers, field_cfg)
    })?;
    let quadratic_prefix_accumulator = tracing::info_span!(
        target: "zinc_protocol::production_sha",
        "sumfold_quadratic_prefix_accumulator",
        side = "prove",
        phase = "sumfold_quadratic_prefix_accumulator",
    )
    .in_scope(|| {
        build_sha_sumfold_quadratic_prefix_accumulator(
            traces,
            booleanity_sources,
            prefix_vars,
            r_ic_eq_weights,
            booleanity_weights,
            field_cfg,
        )
    })?;
    let sumfold_group = tracing::info_span!(
        target: "zinc_protocol::production_sha",
        "sumfold_group",
        side = "prove",
        phase = "sumfold_group",
    )
    .in_scope(|| {
        build_production_sha_sumfold_group_from_prefix_accumulators(
            traces,
            beta,
            beta_eq_weights,
            r_ic_eq_weights,
            &linear_accumulator,
            &quadratic_prefix_accumulator,
            booleanity_weights,
            booleanity_sources,
            prefix_vars,
            field_cfg,
        )
    })?;
    Ok((
        linear_accumulator,
        quadratic_prefix_accumulator,
        sumfold_group,
    ))
}

#[tracing::instrument(
    target = "zinc_protocol::production_sha",
    level = "info",
    skip_all,
    fields(side = "prove", phase = "sumfold_prove", instance_vars)
)]
fn prove_sumfold_phase<F>(
    transcript: &mut impl Transcript,
    sumfold_group: MultiDegreeSumcheckGroup<F>,
    initial_claim: &F,
    instance_vars: usize,
    field_cfg: &F::Config,
) -> Result<(MultiDegreeSumcheckProof<F>, Vec<F>), ProductionShaError<F>>
where
    F: InnerTransparentField
        + DelayedFieldProductSum
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: Transcribable + Zero + Send + Sync,
    F::Modulus: Transcribable,
{
    prove_optimized_sha_sumfold_with_weights(
        transcript,
        sumfold_group,
        initial_claim,
        instance_vars,
        field_cfg,
    )
}

#[tracing::instrument(
    target = "zinc_protocol::production_sha",
    level = "info",
    skip_all,
    fields(side = "prove", phase = "fold_after_sumfold", instances = traces.len())
)]
#[allow(clippy::too_many_arguments)]
fn prove_fold_after_sumfold_phase<P, Zt, F, const D: usize>(
    traces: &[ProjectedTrace<F>],
    publics: &[ProjectedPublic<F>],
    provisional_sumfold_output: &InstanceFoldClaim<F>,
    beta: &[F],
    sumfold_r_b: Vec<F>,
    r_ic_eq_weights: &[F],
    a_powers: &[F],
    lambda_powers: &[F],
    booleanity_weights: &[F],
    booleanity_sources: &[ShaBooleanitySource],
    instance_commitments: &[PCSCommitments<P, Zt, F, D>],
    instance_prover_data: &[PCSProverData<P, Zt, F, D>],
    field_cfg: &F::Config,
) -> Result<ProductionShaFoldAfterSumfold<P, Zt, F, D>, ProductionShaError<F>>
where
    Zt: ZincTypes<D>,
    F: InnerTransparentField + DelayedFieldProductSum + ShaBinaryFoldField + Send + Sync + 'static,
    F::Inner: Zero,
    P: ZincPCSTypes<Zt, F, D>,
    P::BinaryPCS: FoldablePCS<F, BinaryPoly<D>, D>,
    P::ArbitraryPCS: FoldablePCS<F, DensePolynomial<Zt::Int, D>, D>,
    P::IntPCS: FoldablePCS<F, Zt::Int, D>,
{
    let (folded, folded_public) = tracing::info_span!(
        target: "zinc_protocol::production_sha",
        "fold_projected_traces",
        side = "prove",
        phase = "fold_projected_traces",
    )
    .in_scope(|| fold_projected_traces(traces, publics, provisional_sumfold_output, field_cfg))?;
    let row_claim = tracing::info_span!(
        target: "zinc_protocol::production_sha",
        "fold_row_claim",
        side = "prove",
        phase = "fold_row_claim",
    )
    .in_scope(|| {
        production_sha_folded_row_sum_fast(
            &folded.trace,
            &folded_public,
            r_ic_eq_weights,
            a_powers,
            lambda_powers,
            booleanity_weights,
            booleanity_sources,
            field_cfg,
        )
    })?;
    let sumfold_output = tracing::info_span!(
        target: "zinc_protocol::production_sha",
        "fold_claim",
        side = "prove",
        phase = "fold_claim",
    )
    .in_scope(|| {
        derive_instance_fold_claim_from_row_claim(
            beta,
            sumfold_r_b,
            &row_claim,
            traces.len(),
            field_cfg,
        )
    })?;
    let folded_commitments = tracing::info_span!(
        target: "zinc_protocol::production_sha",
        "fold_commitments",
        side = "prove",
        phase = "fold_commitments",
    )
    .in_scope(|| {
        fold_pcs_commitments::<P, Zt, F, D>(
            instance_commitments,
            sumfold_output.eq_instance_weights(),
            field_cfg,
        )
    })?;
    let folded_prover_data = tracing::info_span!(
        target: "zinc_protocol::production_sha",
        "fold_prover_data",
        side = "prove",
        phase = "fold_prover_data",
    )
    .in_scope(|| {
        fold_pcs_prover_data::<P, Zt, F, D>(
            instance_prover_data,
            sumfold_output.eq_instance_weights(),
            field_cfg,
        )
    })?;

    Ok((
        folded,
        folded_public,
        row_claim,
        sumfold_output,
        folded_commitments,
        folded_prover_data,
    ))
}

#[allow(clippy::too_many_arguments)]
fn production_sha_folded_row_sum_fast<F>(
    trace: &ProjectedTrace<F>,
    public: &ProjectedPublic<F>,
    row_weights: &[F],
    a_powers: &[F],
    lambda_powers: &[F],
    booleanity_weights: &[F],
    booleanity_sources: &[ShaBooleanitySource],
    field_cfg: &F::Config,
) -> Result<F, ProductionShaError<F>>
where
    F: InnerTransparentField + DelayedFieldProductSum + Send + Sync + 'static,
    F::Inner: Zero,
{
    #[cfg(debug_assertions)]
    validate_projected_trace(trace)?;
    if row_weights.len() != SHA_ROW_COUNT {
        return Err(ProductionShaError::LengthMismatch {
            label: "row weights",
            got: row_weights.len(),
            expected: SHA_ROW_COUNT,
        });
    }
    if a_powers.len() < SHA_IDEAL_EVAL_POWER_COUNT {
        return Err(ProductionShaError::LengthMismatch {
            label: "a powers",
            got: a_powers.len(),
            expected: SHA_IDEAL_EVAL_POWER_COUNT,
        });
    }
    if lambda_powers.len() != NUM_SHA_RESIDUAL_FAMILIES {
        return Err(ProductionShaError::LengthMismatch {
            label: "lambda powers",
            got: lambda_powers.len(),
            expected: NUM_SHA_RESIDUAL_FAMILIES,
        });
    }
    if booleanity_weights.len() != booleanity_sources.len() {
        return Err(ProductionShaError::LengthMismatch {
            label: "booleanity weights",
            got: booleanity_weights.len(),
            expected: booleanity_sources.len(),
        });
    }

    let weight_vec = |shift: usize| {
        (0..SHA_WORD_BITS)
            .map(|bit| {
                if bit >= shift {
                    a_powers[bit - shift].clone()
                } else {
                    F::zero_with_cfg(field_cfg)
                }
            })
            .collect::<Vec<_>>()
    };
    let rot_vec = |shift: usize| {
        (0..SHA_WORD_BITS)
            .map(|bit| a_powers[(bit + shift) % SHA_WORD_BITS].clone())
            .collect::<Vec<_>>()
    };

    let word_weights = a_powers[..SHA_WORD_BITS].to_vec();
    let rot25_weights = rot_vec(25);
    let rot14_weights = rot_vec(14);
    let rot15_weights = rot_vec(15);
    let rot13_weights = rot_vec(13);
    let shift0_weights = weight_vec(0);
    let shift2_weights = weight_vec(2);
    let shift3_weights = weight_vec(3);
    let shift5_weights = weight_vec(5);
    let shift8_weights = weight_vec(8);
    let shift9_weights = weight_vec(9);
    let shift10_weights = weight_vec(10);
    let rho_sig0 = a_powers[10].clone() + &a_powers[19] + &a_powers[30];
    let rho_sig1 = a_powers[7].clone() + &a_powers[21] + &a_powers[26];
    let low_mu_coeff = production_sha_pow_two(32, field_cfg);
    let high_mu_w_coeff = production_sha_pow_two(34, field_cfg);
    let high_mu_3_bit_coeff = production_sha_pow_two(35, field_cfg);
    let high_mu_1_bit_coeff = production_sha_pow_two(33, field_cfg);
    let one = F::one_with_cfg(field_cfg);
    let two = one.clone() + &one;

    let values = cfg_iter!(row_weights)
        .enumerate()
        .map(|(row, row_weight)| {
            let word_eval_with = |col: ShaWordCol, shift: usize, weights: &[F]| {
                trace_word_eval_at_row_with_weights(trace, col, row, shift, weights, field_cfg)
            };
            let word_eval =
                |col: ShaWordCol, shift: usize| word_eval_with(col, shift, &word_weights);
            let public_word_eval = |col: ShaPublicCol| {
                public_word_or_const_eval_at_row(public, col, row, &word_weights, field_cfg)
            };

            let a_word = word_eval(ShaWordCol::A, 0)?;
            let e_word = word_eval(ShaWordCol::E, 0)?;
            let sigma0 = word_eval(ShaWordCol::Sigma0, 0)?;
            let sigma1 = word_eval(ShaWordCol::Sigma1, 0)?;
            let w = word_eval(ShaWordCol::W, 0)?;
            let small_sigma0 = word_eval(ShaWordCol::SmallSigma0, 0)?;
            let small_sigma1 = word_eval(ShaWordCol::SmallSigma1, 0)?;
            let ov_sigma0 = word_eval(ShaWordCol::OvSigma0, 0)?;
            let ov_sigma1 = word_eval(ShaWordCol::OvSigma1, 0)?;
            let ov_small_sigma0 = word_eval(ShaWordCol::OvSmallSigma0, 0)?;
            let ov_small_sigma1 = word_eval(ShaWordCol::OvSmallSigma1, 0)?;

            let mu = |low_weights: &[F], high_weights: &[F], high_coeff: &F| {
                Ok::<F, ProductionShaError<F>>(
                    word_eval_with(ShaWordCol::MuPacked, 0, low_weights)? * &low_mu_coeff
                        - word_eval_with(ShaWordCol::MuPacked, 0, high_weights)? * high_coeff,
                )
            };
            let mu_w = mu(&shift0_weights, &shift2_weights, &high_mu_w_coeff)?;
            let mu_a = mu(&shift2_weights, &shift5_weights, &high_mu_3_bit_coeff)?;
            let mu_e = mu(&shift5_weights, &shift8_weights, &high_mu_3_bit_coeff)?;
            let mu_ff_a = mu(&shift8_weights, &shift9_weights, &high_mu_1_bit_coeff)?;
            let mu_ff_e = mu(&shift9_weights, &shift10_weights, &high_mu_1_bit_coeff)?;

            let r0 = a_word.clone() * &rho_sig0 - &sigma0 - two.clone() * &ov_sigma0;
            let r1 = e_word.clone() * &rho_sig1 - &sigma1 - two.clone() * &ov_sigma1;
            let r2 = word_eval_with(ShaWordCol::W, 0, &rot25_weights)?
                + word_eval_with(ShaWordCol::W, 0, &rot14_weights)?
                + word_eval_with(ShaWordCol::W, 0, &shift3_weights)?
                - &small_sigma0
                - two.clone() * &ov_small_sigma0;
            let r3 = word_eval_with(ShaWordCol::W, 0, &rot15_weights)?
                + word_eval_with(ShaWordCol::W, 0, &rot13_weights)?
                + word_eval_with(ShaWordCol::W, 0, &shift10_weights)?
                - &small_sigma1
                - two.clone() * &ov_small_sigma1;
            let r4 = word_eval(ShaWordCol::W, 16)?
                - &w
                - word_eval(ShaWordCol::SmallSigma0, 1)?
                - word_eval(ShaWordCol::W, 9)?
                - word_eval(ShaWordCol::SmallSigma1, 14)?
                + &mu_w
                + trace_int_at_row(trace, ShaIntCol::CompSchedule, row, field_cfg)?;
            let r5 = word_eval(ShaWordCol::A, 4)?
                - &e_word
                - word_eval(ShaWordCol::Sigma1, 3)?
                - word_eval(ShaWordCol::Uef, 3)?
                - word_eval(ShaWordCol::UNegEg, 3)?
                - public_scalar_at_row(public, ShaPublicCol::K, row, 3, field_cfg)?
                - &w
                - word_eval(ShaWordCol::Sigma0, 3)?
                - word_eval(ShaWordCol::Maj, 3)?
                + &mu_a
                + trace_int_at_row(trace, ShaIntCol::CompUpdateA, row, field_cfg)?;
            let r6 = word_eval(ShaWordCol::E, 4)?
                - &a_word
                - &e_word
                - word_eval(ShaWordCol::Sigma1, 3)?
                - word_eval(ShaWordCol::Uef, 3)?
                - word_eval(ShaWordCol::UNegEg, 3)?
                - public_scalar_at_row(public, ShaPublicCol::K, row, 3, field_cfg)?
                - &w
                + &mu_e
                + trace_int_at_row(trace, ShaIntCol::CompUpdateE, row, field_cfg)?;

            let s_init = public_scalar_at_row(public, ShaPublicCol::SInit, row, 0, field_cfg)?;
            let s_msg = public_scalar_at_row(public, ShaPublicCol::SMsg, row, 0, field_cfg)?;
            let s_sched = public_scalar_at_row(public, ShaPublicCol::SSched, row, 0, field_cfg)?;
            let s_upd = public_scalar_at_row(public, ShaPublicCol::SUpd, row, 0, field_cfg)?;
            let s_ff = public_scalar_at_row(public, ShaPublicCol::SFf, row, 0, field_cfg)?;
            let s_out = public_scalar_at_row(public, ShaPublicCol::SOut, row, 0, field_cfg)?;

            let r7 = (a_word.clone() - public_word_eval(ShaPublicCol::PAIn)?) * &s_init
                + (a_word.clone() - public_word_eval(ShaPublicCol::PAOut)?) * &s_out;
            let r8 = (e_word.clone() - public_word_eval(ShaPublicCol::PEIn)?) * &s_init
                + (e_word.clone() - public_word_eval(ShaPublicCol::PEOut)?) * &s_out;
            let r9 = word_eval(ShaWordCol::A, 4)?
                - &a_word
                - public_scalar_at_row(public, ShaPublicCol::PAIn, row, 0, field_cfg)?
                + &mu_ff_a
                + trace_int_at_row(trace, ShaIntCol::CompFeedForwardA, row, field_cfg)?;
            let r10 = word_eval(ShaWordCol::E, 4)?
                - &e_word
                - public_scalar_at_row(public, ShaPublicCol::PEIn, row, 0, field_cfg)?
                + &mu_ff_e
                + trace_int_at_row(trace, ShaIntCol::CompFeedForwardE, row, field_cfg)?;
            let r11 = (w - public_word_eval(ShaPublicCol::Message)?) * &s_msg;
            let r12 = trace_int_at_row(trace, ShaIntCol::CompSchedule, row, field_cfg)? * &s_sched;
            let r13 = trace_int_at_row(trace, ShaIntCol::CompUpdateA, row, field_cfg)? * &s_upd;
            let r14 = trace_int_at_row(trace, ShaIntCol::CompUpdateE, row, field_cfg)? * &s_upd;
            let r15 = trace_int_at_row(trace, ShaIntCol::CompFeedForwardA, row, field_cfg)? * &s_ff;
            let r16 = trace_int_at_row(trace, ShaIntCol::CompFeedForwardE, row, field_cfg)? * &s_ff;
            let r17 = word_eval_with(ShaWordCol::MuPacked, 0, &shift10_weights)?;

            let residuals = [
                r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17,
            ];
            let linear = FieldFieldInnerProduct::inner_product::<UNCHECKED>(
                &residuals,
                lambda_powers,
                F::zero_with_cfg(field_cfg),
            )
            .map_err(|_| {
                ProductionShaError::NonCanonicalProofObject(
                    "production SHA row residual dot product failed",
                )
            })?;

            let mut bool_sum = F::zero_with_cfg(field_cfg);
            for (source, weight) in booleanity_sources.iter().zip(booleanity_weights.iter()) {
                let d = booleanity_source_value_at_fast(trace, row, source, field_cfg)?;
                bool_sum += weight.clone() * (d.clone() * (d - one.clone()));
            }

            Ok(row_weight.clone() * (linear + bool_sum))
        })
        .collect::<Result<Vec<_>, ProductionShaError<F>>>()?;

    folded_row_integrand_sum(&values, field_cfg).map_err(ProductionShaError::from)
}

fn trace_word_eval_at_row_with_weights<F>(
    trace: &ProjectedTrace<F>,
    col: ShaWordCol,
    row: usize,
    shift: usize,
    weights: &[F],
    field_cfg: &F::Config,
) -> Result<F, ProductionShaError<F>>
where
    F: PrimeField,
{
    let mut acc = F::zero_with_cfg(field_cfg);
    for (bit, weight) in weights.iter().enumerate().take(SHA_WORD_BITS) {
        acc += trace_word_bit_at_row(trace, col, row, shift, bit, field_cfg)? * weight;
    }
    Ok(acc)
}

fn public_word_or_const_eval_at_row<F>(
    public: &ProjectedPublic<F>,
    col: ShaPublicCol,
    row: usize,
    weights: &[F],
    field_cfg: &F::Config,
) -> Result<F, ProductionShaError<F>>
where
    F: PrimeField,
{
    let Some(_col_idx) = public_word_col_index(col) else {
        return public_scalar_at_row(public, col, row, 0, field_cfg);
    };
    if public.bit_slices.is_none() {
        return public_scalar_at_row(public, col, row, 0, field_cfg);
    }
    let mut acc = F::zero_with_cfg(field_cfg);
    for (bit, weight) in weights.iter().enumerate().take(SHA_WORD_BITS) {
        acc += public_word_bit_at_row(public, col, row, bit, field_cfg)? * weight;
    }
    Ok(acc)
}

fn booleanity_source_value_at_fast<F>(
    trace: &ProjectedTrace<F>,
    row: usize,
    source: &ShaBooleanitySource,
    field_cfg: &F::Config,
) -> Result<F, ProductionShaError<F>>
where
    F: PrimeField,
{
    let one = F::one_with_cfg(field_cfg);
    let two = one.clone() + &one;
    match *source {
        ShaBooleanitySource::WordBit { col, bit } => {
            trace_word_bit_at_row(trace, col, row, 0, bit, field_cfg)
        }
        ShaBooleanitySource::VirtualCh1 { bit } => {
            Ok(
                trace_word_bit_at_row(trace, ShaWordCol::E, row, 2, bit, field_cfg)?
                    + &trace_word_bit_at_row(trace, ShaWordCol::E, row, 1, bit, field_cfg)?
                    - two.clone()
                        * trace_word_bit_at_row(trace, ShaWordCol::Uef, row, 2, bit, field_cfg)?,
            )
        }
        ShaBooleanitySource::VirtualCh2 { bit } => {
            Ok(
                trace_word_bit_at_row(trace, ShaWordCol::E, row, 2, bit, field_cfg)?
                    - &trace_word_bit_at_row(trace, ShaWordCol::E, row, 0, bit, field_cfg)?
                    + two.clone()
                        * trace_word_bit_at_row(trace, ShaWordCol::UNegEg, row, 2, bit, field_cfg)?
                    + two.clone()
                        * trace_word_bit_at_row(
                            trace,
                            ShaWordCol::Ch2Comp,
                            row,
                            0,
                            bit,
                            field_cfg,
                        )?,
            )
        }
        ShaBooleanitySource::VirtualMaj { bit } => {
            Ok(
                trace_word_bit_at_row(trace, ShaWordCol::A, row, 0, bit, field_cfg)?
                    + &trace_word_bit_at_row(trace, ShaWordCol::A, row, 1, bit, field_cfg)?
                    + &trace_word_bit_at_row(trace, ShaWordCol::A, row, 2, bit, field_cfg)?
                    - two.clone()
                        * trace_word_bit_at_row(trace, ShaWordCol::Maj, row, 2, bit, field_cfg)?
                    - two.clone()
                        * trace_word_bit_at_row(
                            trace,
                            ShaWordCol::MajComp,
                            row,
                            0,
                            bit,
                            field_cfg,
                        )?,
            )
        }
    }
}

#[tracing::instrument(
    target = "zinc_protocol::production_sha",
    level = "info",
    skip_all,
    fields(side = "prove", phase = "row_sumcheck")
)]
#[allow(clippy::too_many_arguments)]
fn prove_row_sumcheck_phase<F>(
    transcript: &mut impl Transcript,
    trace: &ProjectedTrace<F>,
    public: &ProjectedPublic<F>,
    r_ic: &[F; SHA_ROW_VARS],
    r_ic_eq_weights: &[F],
    a_powers: &[F],
    lambda_powers: &[F],
    booleanity_weights: &[F],
    booleanity_sources: &[ShaBooleanitySource],
    row_claim: &F,
    field_cfg: &F::Config,
) -> Result<(MultiDegreeSumcheckProof<F>, FoldedRowSumcheckOutput<F>), ProductionShaError<F>>
where
    F: InnerTransparentField
        + DelayedFieldProductSum
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: Transcribable + Zero,
    F::Modulus: Transcribable,
{
    let (combined_sumcheck, row_output) =
        prove_expression_folded_row_sumcheck_with_output_and_vectors(
            transcript,
            trace,
            public,
            r_ic,
            r_ic_eq_weights,
            a_powers,
            lambda_powers,
            booleanity_weights,
            booleanity_sources,
            field_cfg,
        )?;
    verify_folded_row_sumcheck_claim(&combined_sumcheck.claimed_sums()[0], row_claim)?;
    Ok((combined_sumcheck, row_output))
}

#[tracing::instrument(
    target = "zinc_protocol::production_sha",
    level = "info",
    skip_all,
    fields(side = "prove", phase = "endpoint_multipoint")
)]
#[allow(clippy::too_many_arguments)]
fn prove_endpoint_multipoint_phase<F>(
    transcript: &mut impl Transcript,
    trace: &ProjectedTrace<F>,
    folded_public: &ProjectedPublic<F>,
    row_output: &FoldedRowSumcheckOutput<F>,
    r_ic: &[F; SHA_ROW_VARS],
    a: &F,
    a_powers: &[F],
    lambda_powers: &[F],
    booleanity_weights: &[F],
    booleanity_sources: &[ShaBooleanitySource],
    field_cfg: &F::Config,
) -> Result<ProductionShaEndpointMultipoint<F>, ProductionShaError<F>>
where
    F: InnerTransparentField
        + DelayedFieldProductSum
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: Transcribable + Zero + Default + Send + Sync,
    F::Modulus: Transcribable,
{
    #[cfg(not(debug_assertions))]
    let _ = (
        r_ic,
        a,
        a_powers,
        lambda_powers,
        booleanity_weights,
        booleanity_sources,
    );

    let endpoint_evals = tracing::info_span!(
        target: "zinc_protocol::production_sha",
        "endpoint_build_evals",
        side = "prove",
        phase = "endpoint_build_evals",
    )
    .in_scope(|| {
        row_output.endpoint_evals.clone().map_or_else(
            || {
                build_sha_endpoint_evals_from_trace_with_row_weights(
                    trace,
                    &row_output.r_star_eq_weights,
                    a,
                    field_cfg,
                )
            },
            Ok,
        )
    })?;
    let resolver = tracing::info_span!(
        target: "zinc_protocol::production_sha",
        "endpoint_resolver",
        side = "prove",
        phase = "endpoint_resolver",
    )
    .in_scope(|| {
        let resolver = sha_resolver_from_endpoint_evals(&endpoint_evals)?;
        absorb_sha_resolver_proof(transcript, &resolver, field_cfg);
        Ok::<_, ProductionShaError<F>>(resolver)
    })?;
    #[cfg(debug_assertions)]
    {
        let resolver_endpoint_evals = sha_endpoint_evals_from_resolver(&resolver, a, field_cfg)?;
        let terminal = tracing::info_span!(
            target: "zinc_protocol::production_sha",
            "endpoint_terminal",
            side = "prove",
            phase = "endpoint_terminal",
        )
        .in_scope(|| {
            reconstruct_folded_row_terminal_from_endpoints_with_vectors(
                &resolver_endpoint_evals,
                folded_public,
                r_ic,
                &row_output.r_star,
                &row_output.r_star_eq_weights,
                a_powers,
                lambda_powers,
                booleanity_weights,
                booleanity_sources,
                field_cfg,
            )
        })?;
        verify_folded_row_terminal_value(row_output, &terminal)?;
    }

    let (multipoint_eval, r_0) = tracing::info_span!(
        target: "zinc_protocol::production_sha",
        "endpoint_reduce",
        side = "prove",
        phase = "endpoint_reduce",
    )
    .in_scope(|| {
        prove_sha_endpoint_multipoint_with_row_weights(
            transcript,
            trace,
            folded_public,
            &endpoint_evals,
            &row_output.r_star,
            &row_output.r_star_eq_weights,
            field_cfg,
        )
    })?;
    Ok((resolver, endpoint_evals, multipoint_eval, r_0))
}

#[tracing::instrument(
    target = "zinc_protocol::production_sha",
    level = "info",
    skip_all,
    fields(side = "prove", phase = "pcs_opening")
)]
#[allow(clippy::too_many_arguments)]
fn prove_pcs_opening_phase<P, Zt, F, const D: usize>(
    transcript: &mut impl Transcript,
    folded_trace: &ProjectedTrace<F>,
    folded_commitments: &PCSCommitments<P, Zt, F, D>,
    folded_prover_data: &PCSProverData<P, Zt, F, D>,
    r_0: &[F],
    r_0_eq_weights: &[F],
    pcs_params: &PCSParams<P, Zt, F, D>,
    field_cfg: &F::Config,
) -> Result<ProductionShaPcsOpening<P, Zt, F, D>, ProductionShaError<F>>
where
    Zt: ZincTypes<D>,
    F: PrimeField + DelayedFieldProductSum,
    F::Inner: Transcribable,
    F::Modulus: Transcribable,
    P: ZincPCSTypes<Zt, F, D>,
    P: ProductionShaFoldedPcsOpen<Zt, F, D>,
    P::BinaryPCS: FoldablePCS<F, BinaryPoly<D>, D>,
    P::ArbitraryPCS: FoldablePCS<F, DensePolynomial<Zt::Int, D>, D>,
    P::IntPCS: FoldablePCS<F, Zt::Int, D>,
{
    let witness_lifted_evals = tracing::info_span!(
        target: "zinc_protocol::production_sha",
        "pcs_lifted_evals",
        side = "prove",
        phase = "pcs_lifted_evals",
    )
    .in_scope(|| {
        build_folded_sha_pcs_lifted_evals_with_row_weights(folded_trace, r_0_eq_weights, field_cfg)
    })?;
    tracing::info_span!(
        target: "zinc_protocol::production_sha",
        "pcs_absorb_lifted_evals",
        side = "prove",
        phase = "pcs_absorb_lifted_evals",
    )
    .in_scope(|| absorb_folded_lifted_evals(transcript, &witness_lifted_evals, field_cfg));
    let opening_proof = tracing::info_span!(
        target: "zinc_protocol::production_sha",
        "pcs_open_core",
        side = "prove",
        phase = "pcs_open_core",
    )
    .in_scope(|| {
        P::prove_folded_pcs_opening(
            pcs_params,
            folded_commitments,
            folded_trace,
            folded_prover_data,
            r_0,
            &witness_lifted_evals,
            field_cfg,
        )
    })?;
    Ok((witness_lifted_evals, opening_proof))
}

fn public_uair_trace_view<'a, PolyCoeff, Int, F, const D: usize>(
    trace: &'a UairTrace<'_, PolyCoeff, Int, D>,
    sig: &UairSignature,
) -> Result<UairTrace<'a, PolyCoeff, Int, D>, ProductionShaError<F>>
where
    PolyCoeff: Clone,
    Int: Clone,
    F: PrimeField,
{
    let public = sig.public_cols();
    validate_uair_trace_shape(trace, sig)?;
    Ok(UairTrace {
        binary_poly: Cow::Borrowed(
            trace
                .binary_poly
                .get(..public.num_binary_poly_cols())
                .ok_or(ProductionShaError::LengthMismatch {
                    label: "UAIR public binary columns",
                    got: trace.binary_poly.len(),
                    expected: public.num_binary_poly_cols(),
                })?,
        ),
        arbitrary_poly: Cow::Borrowed(
            trace
                .arbitrary_poly
                .get(..public.num_arbitrary_poly_cols())
                .ok_or(ProductionShaError::LengthMismatch {
                    label: "UAIR public arbitrary columns",
                    got: trace.arbitrary_poly.len(),
                    expected: public.num_arbitrary_poly_cols(),
                })?,
        ),
        int: Cow::Borrowed(trace.int.get(..public.num_int_cols()).ok_or(
            ProductionShaError::LengthMismatch {
                label: "UAIR public int columns",
                got: trace.int.len(),
                expected: public.num_int_cols(),
            },
        )?),
    })
}

fn witness_uair_trace_view<'a, PolyCoeff, Int, F, const D: usize>(
    trace: &'a UairTrace<'_, PolyCoeff, Int, D>,
    sig: &UairSignature,
) -> Result<UairTrace<'a, PolyCoeff, Int, D>, ProductionShaError<F>>
where
    PolyCoeff: Clone,
    Int: Clone,
    F: PrimeField,
{
    let public = sig.public_cols();
    let total = sig.total_cols();
    validate_uair_trace_shape(trace, sig)?;
    Ok(UairTrace {
        binary_poly: Cow::Borrowed(
            trace
                .binary_poly
                .get(public.num_binary_poly_cols()..total.num_binary_poly_cols())
                .ok_or(ProductionShaError::LengthMismatch {
                    label: "UAIR witness binary columns",
                    got: trace.binary_poly.len(),
                    expected: total.num_binary_poly_cols(),
                })?,
        ),
        arbitrary_poly: Cow::Borrowed(
            trace
                .arbitrary_poly
                .get(public.num_arbitrary_poly_cols()..total.num_arbitrary_poly_cols())
                .ok_or(ProductionShaError::LengthMismatch {
                    label: "UAIR witness arbitrary columns",
                    got: trace.arbitrary_poly.len(),
                    expected: total.num_arbitrary_poly_cols(),
                })?,
        ),
        int: Cow::Borrowed(
            trace
                .int
                .get(public.num_int_cols()..total.num_int_cols())
                .ok_or(ProductionShaError::LengthMismatch {
                    label: "UAIR witness int columns",
                    got: trace.int.len(),
                    expected: total.num_int_cols(),
                })?,
        ),
    })
}

fn validate_uair_trace_shape<PolyCoeff, Int, F, const D: usize>(
    trace: &UairTrace<'_, PolyCoeff, Int, D>,
    sig: &UairSignature,
) -> Result<(), ProductionShaError<F>>
where
    PolyCoeff: Clone,
    Int: Clone,
    F: PrimeField,
{
    let total = sig.total_cols();
    if trace.binary_poly.len() != total.num_binary_poly_cols() {
        return Err(ProductionShaError::LengthMismatch {
            label: "UAIR binary columns",
            got: trace.binary_poly.len(),
            expected: total.num_binary_poly_cols(),
        });
    }
    if trace.arbitrary_poly.len() != total.num_arbitrary_poly_cols() {
        return Err(ProductionShaError::LengthMismatch {
            label: "UAIR arbitrary columns",
            got: trace.arbitrary_poly.len(),
            expected: total.num_arbitrary_poly_cols(),
        });
    }
    if trace.int.len() != total.num_int_cols() {
        return Err(ProductionShaError::LengthMismatch {
            label: "UAIR int columns",
            got: trace.int.len(),
            expected: total.num_int_cols(),
        });
    }
    Ok(())
}

fn own_uair_trace<PolyCoeff, Int, const D: usize>(
    trace: &UairTrace<'_, PolyCoeff, Int, D>,
) -> UairTrace<'static, PolyCoeff, Int, D>
where
    PolyCoeff: Clone,
    Int: Clone,
{
    UairTrace {
        binary_poly: Cow::Owned(trace.binary_poly.iter().cloned().collect()),
        arbitrary_poly: Cow::Owned(trace.arbitrary_poly.iter().cloned().collect()),
        int: Cow::Owned(trace.int.iter().cloned().collect()),
    }
}

pub fn setup_verify_linear_ideal_fold<P, U, Zt, F, const D: usize>(
    params: LinearIdealFoldVerifierParams<P, U, Zt, F, D>,
    shape: UairShape<U>,
) -> Result<VerifiedLinearIdealFoldSetup<P, U, Zt, F, D>, LinearIdealFoldError<F>>
where
    U: Uair + ProductionShaProjectionAdapter<Zt, F, D>,
    Zt: ZincTypes<D>,
    F: PrimeField + FromPrimitiveWithConfig,
    F::Inner: Transcribable,
    F::Modulus: Transcribable,
    P: ZincPCSTypes<Zt, F, D>,
    P::BinaryPCS: FoldablePCS<F, BinaryPoly<D>, D>,
    P::ArbitraryPCS: FoldablePCS<F, DensePolynomial<Zt::Int, D>, D>,
    P::IntPCS: FoldablePCS<F, Zt::Int, D>,
{
    ensure_production_sha_word_degree::<F, D>()?;
    if shape.num_vars != SHA_ROW_VARS {
        return Err(ProductionShaError::LengthMismatch {
            label: "production SHA row variables",
            got: shape.num_vars,
            expected: SHA_ROW_VARS,
        });
    }

    let (binary, arbitrary, int) = U::production_sha_pcs_batch_sizes();
    validate_production_sha_batch_sizes::<F>(binary, arbitrary, int)?;

    Ok(VerifiedLinearIdealFoldSetup {
        pcs_params: params.pcs_params,
        shape,
        field_cfg: params.field_cfg,
    })
}

pub fn verify_linear_ideal_fold<P, U, Zt, F, const D: usize>(
    vs: &VerifiedLinearIdealFoldSetup<P, U, Zt, F, D>,
    instances: &[UairInstance<'_, Zt::Int, Zt::Int, PCSCommitments<P, Zt, F, D>, D>],
    proof: &ProductionLinearIdealFoldProof<P, Zt, F, D>,
    transcript: &mut impl Transcript,
) -> Result<
    FoldedLinearIdealInstance<F, PCSCommitments<P, Zt, F, D>, ProjectedPublic<F>>,
    LinearIdealFoldError<F>,
>
where
    U: Uair + ProductionShaProjectionAdapter<Zt, F, D>,
    Zt: ZincTypes<D>,
    F: InnerTransparentField
        + DelayedFieldProductSum
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: Transcribable + Zero + Default + Send + Sync,
    F::Modulus: Transcribable,
    P: ZincPCSTypes<Zt, F, D>,
    P::BinaryPCS: FoldablePCS<F, BinaryPoly<D>, D>,
    P::ArbitraryPCS: FoldablePCS<F, DensePolynomial<Zt::Int, D>, D>,
    P::IntPCS: FoldablePCS<F, Zt::Int, D>,
{
    let field_cfg = &vs.field_cfg;
    ensure_production_sha_word_degree::<F, D>()?;
    if instances.len() < 2 {
        return Err(ProductionShaError::InstanceCountTooSmall(instances.len()));
    }
    if !instances.len().is_power_of_two() {
        return Err(ProductionShaError::InstanceCountNotPowerOfTwo(
            instances.len(),
        ));
    }
    if proof.instance_commitments.len() != instances.len() {
        return Err(ProductionShaError::LengthMismatch {
            label: "proof commitments/instances",
            got: proof.instance_commitments.len(),
            expected: instances.len(),
        });
    }

    absorb_production_sha_statement_metadata(transcript);
    absorb_uair_shape_metadata(transcript, &vs.shape);

    let publics =
        verify_public_projection_phase::<P, U, Zt, F, D>(vs, instances, proof, transcript)?;

    let booleanity_sources = production_sha_booleanity_sources();

    let r_ic = sample_pre_ideal_challenge(transcript, field_cfg);
    let beta = sample_instance_batch_challenge(transcript, instances.len(), field_cfg)?;

    let (_aggregate_ideal_polys, a, lambda, rho, xi, initial_claim) =
        verify_aggregate_ideal_phase(&proof.ideal_check, transcript, field_cfg)?;

    let sumfold_output = verify_sumfold_phase(
        transcript,
        &proof.sumfold_proof,
        &initial_claim,
        &beta,
        beta.len(),
        instances.len(),
        field_cfg,
    )?;

    let folded_commitments = verify_fold_commitments_phase::<P, Zt, F, D>(
        &proof.instance_commitments,
        sumfold_output.eq_instance_weights(),
        field_cfg,
    )?;
    absorb_production_sha_commitments::<P, Zt, F, D>(
        transcript,
        b"production_sha_derived_folded_commitments",
        std::slice::from_ref(&folded_commitments),
    );

    let row_output = verify_row_sumcheck_phase(
        transcript,
        &proof.combined_sumcheck,
        sumfold_output.final_round_sumcheck_claim(),
        field_cfg,
    )?;

    let folded_public =
        verify_fold_publics_phase(&publics, sumfold_output.eq_instance_weights(), field_cfg)?;

    let subclaim = verify_endpoint_multipoint_phase(
        transcript,
        proof,
        &folded_public,
        &row_output,
        &r_ic,
        &a,
        &lambda,
        &rho,
        &xi,
        &booleanity_sources,
        field_cfg,
    )?;

    verify_pcs_phase::<P, Zt, F, D>(
        transcript,
        &vs.pcs_params,
        &folded_commitments,
        &subclaim.sumcheck_subclaim.point,
        &proof.witness_lifted_evals,
        &proof.opening_proof,
        field_cfg,
    )?;

    Ok(FoldedLinearIdealInstance {
        target: sumfold_output.final_round_sumcheck_claim().clone(),
        commitments: folded_commitments,
        public: folded_public,
    })
}

#[tracing::instrument(
    target = "zinc_protocol::production_sha",
    level = "info",
    skip_all,
    fields(side = "verify", phase = "public_projection", instances = instances.len())
)]
fn verify_public_projection_phase<P, U, Zt, F, const D: usize>(
    vs: &VerifiedLinearIdealFoldSetup<P, U, Zt, F, D>,
    instances: &[UairInstance<'_, Zt::Int, Zt::Int, PCSCommitments<P, Zt, F, D>, D>],
    proof: &ProductionLinearIdealFoldProof<P, Zt, F, D>,
    transcript: &mut impl Transcript,
) -> Result<Vec<ProjectedPublic<F>>, ProductionShaError<F>>
where
    U: Uair + ProductionShaProjectionAdapter<Zt, F, D>,
    Zt: ZincTypes<D>,
    F: PrimeField + FromPrimitiveWithConfig,
    F::Inner: Transcribable,
    F::Modulus: Transcribable,
    P: ZincPCSTypes<Zt, F, D>,
{
    let field_cfg = &vs.field_cfg;
    let mut publics = Vec::with_capacity(instances.len());
    for (instance_idx, instance) in instances.iter().enumerate() {
        validate_public_uair_trace_shape::<Zt::Int, Zt::Int, F, D>(
            &instance.public_trace,
            &vs.shape.signature,
        )?;
        if !pcs_commitments_match::<P, Zt, F, D>(
            &instance.commitments,
            &proof.instance_commitments[instance_idx],
        ) {
            return Err(ProductionShaError::NonCanonicalProofObject(
                "instance commitments do not match proof commitments",
            ));
        }
        absorb_public_uair_trace::<Zt, D>(transcript, instance_idx, &instance.public_trace);
        publics.push(U::project_production_sha_public(
            &vs.shape,
            &instance.public_trace,
            field_cfg,
        )?);
    }

    validate_production_sha_publics(&publics, field_cfg)?;
    absorb_production_sha_commitments::<P, Zt, F, D>(
        transcript,
        b"production_sha_fresh_commitments",
        &proof.instance_commitments,
    );
    absorb_projected_sha_publics(transcript, &publics, field_cfg);
    Ok(publics)
}

#[tracing::instrument(
    target = "zinc_protocol::production_sha",
    level = "info",
    skip_all,
    fields(side = "verify", phase = "aggregate_ideal_verify")
)]
fn verify_aggregate_ideal_phase<F>(
    proof: &IdealCheckProof<F>,
    transcript: &mut impl Transcript,
    field_cfg: &F::Config,
) -> Result<ProductionShaVerifierAggregate<F>, ProductionShaError<F>>
where
    F: PrimeField + DelayedFieldProductSum,
    F::Inner: Transcribable,
    F::Modulus: Transcribable,
{
    let aggregate_ideal_polys = aggregate_sha_ideal_polys_from_proof(proof)?;
    check_aggregate_sha_ideal_membership(&aggregate_ideal_polys, field_cfg)?;
    absorb_aggregate_sha_ideal_polys(transcript, &aggregate_ideal_polys, field_cfg);

    let (a, lambda, rho, xi) = sample_post_aggregate_ideal_challenges(transcript, field_cfg);
    let initial_claim =
        evaluate_aggregate_sha_ideal_claim(&aggregate_ideal_polys, &a, &lambda, field_cfg)?;
    Ok((aggregate_ideal_polys, a, lambda, rho, xi, initial_claim))
}

#[tracing::instrument(
    target = "zinc_protocol::production_sha",
    level = "info",
    skip_all,
    fields(side = "verify", phase = "sumfold_verify", instance_vars, instances)
)]
fn verify_sumfold_phase<F>(
    transcript: &mut impl Transcript,
    proof: &MultiDegreeSumcheckProof<F>,
    initial_claim: &F,
    beta: &[F],
    instance_vars: usize,
    instances: usize,
    field_cfg: &F::Config,
) -> Result<InstanceFoldClaim<F>, ProductionShaError<F>>
where
    F: InnerTransparentField
        + DelayedFieldProductSum
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: Transcribable + Zero + Default + Send + Sync,
    F::Modulus: Transcribable,
{
    let verified_sumfold =
        verify_full_sha_sumfold(transcript, proof, initial_claim, instance_vars, field_cfg)?;
    Ok(derive_instance_fold_claim(
        beta,
        verified_sumfold.r_b,
        verified_sumfold.c_sf,
        instances,
        field_cfg,
    )?)
}

#[tracing::instrument(
    target = "zinc_protocol::production_sha",
    level = "info",
    skip_all,
    fields(side = "verify", phase = "fold_after_sumfold", instances = commitments.len())
)]
fn verify_fold_commitments_phase<P, Zt, F, const D: usize>(
    commitments: &[PCSCommitments<P, Zt, F, D>],
    weights: &[F],
    field_cfg: &F::Config,
) -> Result<PCSCommitments<P, Zt, F, D>, ProductionShaError<F>>
where
    Zt: ZincTypes<D>,
    F: PrimeField,
    F::Inner: Transcribable,
    F::Modulus: Transcribable,
    P: ZincPCSTypes<Zt, F, D>,
    P::BinaryPCS: FoldablePCS<F, BinaryPoly<D>, D>,
    P::ArbitraryPCS: FoldablePCS<F, DensePolynomial<Zt::Int, D>, D>,
    P::IntPCS: FoldablePCS<F, Zt::Int, D>,
{
    fold_pcs_commitments::<P, Zt, F, D>(commitments, weights, field_cfg)
}

#[tracing::instrument(
    target = "zinc_protocol::production_sha",
    level = "info",
    skip_all,
    fields(side = "verify", phase = "row_sumcheck_verify")
)]
fn verify_row_sumcheck_phase<F>(
    transcript: &mut impl Transcript,
    proof: &MultiDegreeSumcheckProof<F>,
    final_round_sumcheck_claim: &F,
    field_cfg: &F::Config,
) -> Result<FoldedRowSumcheckOutput<F>, ProductionShaError<F>>
where
    F: InnerTransparentField
        + DelayedFieldProductSum
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: Transcribable + Zero + Default + Send + Sync,
    F::Modulus: Transcribable,
{
    verify_folded_row_sumcheck(transcript, proof, final_round_sumcheck_claim, field_cfg)
}

#[tracing::instrument(
    target = "zinc_protocol::production_sha",
    level = "info",
    skip_all,
    fields(side = "verify", phase = "fold_after_sumfold", instances = publics.len())
)]
fn verify_fold_publics_phase<F>(
    publics: &[ProjectedPublic<F>],
    weights: &[F],
    field_cfg: &F::Config,
) -> Result<ProjectedPublic<F>, ProductionShaError<F>>
where
    F: PrimeField,
{
    fold_projected_publics(publics, weights, field_cfg)
}

#[tracing::instrument(
    target = "zinc_protocol::production_sha",
    level = "info",
    skip_all,
    fields(side = "verify", phase = "endpoint_multipoint_verify")
)]
#[allow(clippy::too_many_arguments)]
fn verify_endpoint_multipoint_phase<P, Zt, F, const D: usize>(
    transcript: &mut impl Transcript,
    proof: &ProductionLinearIdealFoldProof<P, Zt, F, D>,
    folded_public: &ProjectedPublic<F>,
    row_output: &FoldedRowSumcheckOutput<F>,
    r_ic: &[F; SHA_ROW_VARS],
    a: &F,
    lambda: &F,
    rho: &F,
    xi: &F,
    booleanity_sources: &[ShaBooleanitySource],
    field_cfg: &F::Config,
) -> Result<MultipointSubclaim<F>, ProductionShaError<F>>
where
    Zt: ZincTypes<D>,
    F: InnerTransparentField
        + DelayedFieldProductSum
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: Transcribable + Zero + Default + Send + Sync,
    F::Modulus: Transcribable,
    P: ZincPCSTypes<Zt, F, D>,
{
    absorb_sha_resolver_proof(transcript, &proof.resolver, field_cfg);
    let endpoint_evals = sha_endpoint_evals_from_resolver(&proof.resolver, a, field_cfg)?;
    let terminal = reconstruct_folded_row_terminal_from_endpoints(
        &endpoint_evals,
        folded_public,
        r_ic,
        &row_output.r_star,
        a,
        lambda,
        rho,
        xi,
        booleanity_sources,
        field_cfg,
    )?;
    verify_folded_row_terminal_value(row_output, &terminal)?;

    let (subclaim, shift_specs) = verify_sha_endpoint_multipoint(
        transcript,
        &proof.multipoint_eval,
        &endpoint_evals,
        folded_public,
        &row_output.r_star,
        field_cfg,
    )?;
    let open_evals = multipoint_open_evals_from_pcs_lifted(
        &proof.witness_lifted_evals,
        &production_sha_multipoint_layout(),
        folded_public,
        &subclaim.sumcheck_subclaim.point,
        field_cfg,
    )?;
    verify_sha_endpoint_multipoint_open_evals(&subclaim, &open_evals, &shift_specs, field_cfg)?;
    Ok(subclaim)
}

#[tracing::instrument(
    target = "zinc_protocol::production_sha",
    level = "info",
    skip_all,
    fields(side = "verify", phase = "pcs_verify")
)]
fn verify_pcs_phase<P, Zt, F, const D: usize>(
    transcript: &mut impl Transcript,
    pcs_params: &PCSVerifierParams<P, Zt, F, D>,
    folded_commitments: &PCSCommitments<P, Zt, F, D>,
    point: &[F],
    witness_lifted_evals: &[DynamicPolynomialF<F>],
    opening_proof: &PCSOpeningProof<P, Zt, F, D>,
    field_cfg: &F::Config,
) -> Result<(), ProductionShaError<F>>
where
    Zt: ZincTypes<D>,
    F: PrimeField,
    F::Inner: Transcribable,
    F::Modulus: Transcribable,
    P: ZincPCSTypes<Zt, F, D>,
    P::BinaryPCS: FoldablePCS<F, BinaryPoly<D>, D>,
    P::ArbitraryPCS: FoldablePCS<F, DensePolynomial<Zt::Int, D>, D>,
    P::IntPCS: FoldablePCS<F, Zt::Int, D>,
{
    absorb_folded_lifted_evals(transcript, witness_lifted_evals, field_cfg);
    verify_production_sha_pcs_opening::<P, Zt, F, D>(
        pcs_params,
        folded_commitments,
        point,
        witness_lifted_evals,
        opening_proof,
        field_cfg,
    )
}

fn validate_public_uair_trace_shape<PolyCoeff, Int, F, const D: usize>(
    trace: &UairTrace<'_, PolyCoeff, Int, D>,
    sig: &UairSignature,
) -> Result<(), ProductionShaError<F>>
where
    PolyCoeff: Clone,
    Int: Clone,
    F: PrimeField,
{
    let public = sig.public_cols();
    if trace.binary_poly.len() != public.num_binary_poly_cols() {
        return Err(ProductionShaError::LengthMismatch {
            label: "UAIR public binary columns",
            got: trace.binary_poly.len(),
            expected: public.num_binary_poly_cols(),
        });
    }
    if trace.arbitrary_poly.len() != public.num_arbitrary_poly_cols() {
        return Err(ProductionShaError::LengthMismatch {
            label: "UAIR public arbitrary columns",
            got: trace.arbitrary_poly.len(),
            expected: public.num_arbitrary_poly_cols(),
        });
    }
    if trace.int.len() != public.num_int_cols() {
        return Err(ProductionShaError::LengthMismatch {
            label: "UAIR public int columns",
            got: trace.int.len(),
            expected: public.num_int_cols(),
        });
    }
    Ok(())
}

fn pcs_commitments_match<P, Zt, F, const D: usize>(
    lhs: &PCSCommitments<P, Zt, F, D>,
    rhs: &PCSCommitments<P, Zt, F, D>,
) -> bool
where
    Zt: ZincTypes<D>,
    F: PrimeField,
    P: ZincPCSTypes<Zt, F, D>,
{
    fn commitment_bytes<C, W>(commitment: &C, write: W) -> Vec<u8>
    where
        W: FnOnce(&C, &mut Vec<u8>),
    {
        let mut bytes = Vec::new();
        write(commitment, &mut bytes);
        bytes
    }

    commitment_bytes(&lhs.binary, P::BinaryPCS::write_commitment_bytes)
        == commitment_bytes(&rhs.binary, P::BinaryPCS::write_commitment_bytes)
        && commitment_bytes(&lhs.arbitrary, P::ArbitraryPCS::write_commitment_bytes)
            == commitment_bytes(&rhs.arbitrary, P::ArbitraryPCS::write_commitment_bytes)
        && commitment_bytes(&lhs.int, P::IntPCS::write_commitment_bytes)
            == commitment_bytes(&rhs.int, P::IntPCS::write_commitment_bytes)
}

fn aggregate_sha_ideal_polys_from_proof<F>(
    proof: &IdealCheckProof<F>,
) -> Result<[DynamicPolynomialF<F>; NUM_NONZERO_SHA_FAMILIES], ProductionShaError<F>>
where
    F: PrimeField,
{
    let got = proof.combined_mle_values.len();
    proof
        .combined_mle_values
        .clone()
        .try_into()
        .map_err(|_| ProductionShaError::LengthMismatch {
            label: "aggregate SHA ideal polynomial count",
            got,
            expected: NUM_NONZERO_SHA_FAMILIES,
        })
}

fn scalarize_sha_endpoint_bits<F>(bits: &[F; SHA_WORD_BITS], a: &F, field_cfg: &F::Config) -> F
where
    F: PrimeField,
{
    let powers = zinc_utils::powers(a.clone(), F::one_with_cfg(field_cfg), SHA_WORD_BITS);
    bits.iter()
        .zip(powers.iter())
        .fold(F::zero_with_cfg(field_cfg), |acc, (bit, power)| {
            acc + bit.clone() * power
        })
}

fn sha_endpoint_evals_from_resolver<F>(
    resolver: &CombinedPolyResolverProof<F>,
    a: &F,
    field_cfg: &F::Config,
) -> Result<ShaEndpointEvals<F>, ProductionShaError<F>>
where
    F: DelayedFieldProductSum,
{
    if !resolver.down_evals.is_empty() {
        return Err(ProductionShaError::LengthMismatch {
            label: "production SHA resolver down evals",
            got: resolver.down_evals.len(),
            expected: 0,
        });
    }
    if !resolver.bit_op_down_evals.is_empty() {
        return Err(ProductionShaError::LengthMismatch {
            label: "production SHA resolver bit-op down evals",
            got: resolver.bit_op_down_evals.len(),
            expected: 0,
        });
    }

    let word_sources = production_sha_endpoint_word_sources();
    let unshifted_words = word_sources.iter().filter(|(_, shift)| *shift == 0).count();
    let shifted_words = word_sources.len() - unshifted_words;
    let expected_unshifted_bits = unshifted_words * SHA_WORD_BITS;
    let expected_shifted_bits = shifted_words * SHA_WORD_BITS;
    if resolver.bit_slice_evals.len() != expected_unshifted_bits {
        return Err(ProductionShaError::LengthMismatch {
            label: "production SHA resolver unshifted bit slices",
            got: resolver.bit_slice_evals.len(),
            expected: expected_unshifted_bits,
        });
    }
    if resolver.shifted_bit_slice_evals.len() != expected_shifted_bits {
        return Err(ProductionShaError::LengthMismatch {
            label: "production SHA resolver shifted bit slices",
            got: resolver.shifted_bit_slice_evals.len(),
            expected: expected_shifted_bits,
        });
    }

    let mut unshifted_idx = 0usize;
    let mut shifted_idx = 0usize;
    let mut sources = Vec::with_capacity(word_sources.len());
    for (col, shift) in word_sources {
        let bit_slice = if shift == 0 {
            let start = unshifted_idx * SHA_WORD_BITS;
            unshifted_idx += 1;
            &resolver.bit_slice_evals[start..start + SHA_WORD_BITS]
        } else {
            let start = shifted_idx * SHA_WORD_BITS;
            shifted_idx += 1;
            &resolver.shifted_bit_slice_evals[start..start + SHA_WORD_BITS]
        };
        let bits: [F; SHA_WORD_BITS] = std::array::from_fn(|idx| bit_slice[idx].clone());
        let scalarized = scalarize_sha_endpoint_bits(&bits, a, field_cfg);
        sources.push(ShaSourceEndpointEval {
            col,
            shift,
            scalarized,
            bits,
        });
    }

    let int_sources = production_sha_endpoint_int_sources();
    if resolver.up_evals.len() != int_sources.len() {
        return Err(ProductionShaError::LengthMismatch {
            label: "production SHA resolver int evals",
            got: resolver.up_evals.len(),
            expected: int_sources.len(),
        });
    }
    let int_sources = int_sources
        .into_iter()
        .zip(resolver.up_evals.iter())
        .map(|(col, scalar)| ShaIntEndpointEval {
            col,
            scalar: scalar.clone(),
        })
        .collect();

    Ok(ShaEndpointEvals {
        sources,
        int_sources,
    })
}

fn sha_resolver_from_endpoint_evals<F>(
    endpoint_evals: &ShaEndpointEvals<F>,
) -> Result<CombinedPolyResolverProof<F>, ProductionShaError<F>>
where
    F: PrimeField,
{
    validate_sha_endpoint_layout(endpoint_evals)?;

    let mut bit_slice_evals = Vec::new();
    let mut shifted_bit_slice_evals = Vec::new();
    for source in &endpoint_evals.sources {
        if source.shift == 0 {
            bit_slice_evals.extend(source.bits.iter().cloned());
        } else {
            shifted_bit_slice_evals.extend(source.bits.iter().cloned());
        }
    }

    Ok(CombinedPolyResolverProof {
        up_evals: endpoint_evals
            .int_sources
            .iter()
            .map(|source| source.scalar.clone())
            .collect(),
        down_evals: Vec::new(),
        bit_slice_evals,
        bit_op_down_evals: Vec::new(),
        shifted_bit_slice_evals,
    })
}

fn verify_production_sha_pcs_opening<P, Zt, F, const D: usize>(
    pcs_params: &PCSVerifierParams<P, Zt, F, D>,
    folded_commitments: &PCSCommitments<P, Zt, F, D>,
    r_0: &[F],
    folded_lifted_evals: &[DynamicPolynomialF<F>],
    opening_proof: &PCSOpeningProof<P, Zt, F, D>,
    field_cfg: &F::Config,
) -> Result<(), ProductionShaError<F>>
where
    Zt: ZincTypes<D>,
    F: PrimeField,
    F::Inner: Transcribable,
    F::Modulus: Transcribable,
    P: ZincPCSTypes<Zt, F, D>,
{
    ensure_production_sha_word_degree::<F, D>()?;
    validate_production_sha_batch_sizes::<F>(
        P::BinaryPCS::batch_size(&folded_commitments.binary),
        P::ArbitraryPCS::batch_size(&folded_commitments.arbitrary),
        P::IntPCS::batch_size(&folded_commitments.int),
    )?;
    let (binary_lifted, int_lifted) = split_folded_sha_pcs_lifted_evals(folded_lifted_evals)?;
    let arbitrary_lifted: &[DynamicPolynomialF<F>] = &[];

    let mut transcript = PcsVerifierTranscript {
        fs_transcript: Blake3Transcript::default(),
        stream: Cursor::default(),
    };
    let mut transcription_buf = vec![0u8; F::zero_with_cfg(field_cfg).inner().get_num_bytes()];

    P::BinaryPCS::absorb_commitment(&mut transcript.fs_transcript, &folded_commitments.binary);
    absorb_pcs_lifted_evals(
        &mut transcript.fs_transcript,
        binary_lifted,
        &mut transcription_buf,
    );
    P::BinaryPCS::verify_open::<true>(
        &mut transcript,
        &pcs_params.binary,
        &folded_commitments.binary,
        r_0,
        binary_lifted,
        &opening_proof.binary,
        field_cfg,
    )?;

    P::ArbitraryPCS::absorb_commitment(
        &mut transcript.fs_transcript,
        &folded_commitments.arbitrary,
    );
    absorb_pcs_lifted_evals(
        &mut transcript.fs_transcript,
        arbitrary_lifted,
        &mut transcription_buf,
    );
    P::ArbitraryPCS::verify_open::<true>(
        &mut transcript,
        &pcs_params.arbitrary,
        &folded_commitments.arbitrary,
        r_0,
        arbitrary_lifted,
        &opening_proof.arbitrary,
        field_cfg,
    )?;

    P::IntPCS::absorb_commitment(&mut transcript.fs_transcript, &folded_commitments.int);
    absorb_pcs_lifted_evals(
        &mut transcript.fs_transcript,
        int_lifted,
        &mut transcription_buf,
    );
    P::IntPCS::verify_open::<true>(
        &mut transcript,
        &pcs_params.int,
        &folded_commitments.int,
        r_0,
        int_lifted,
        &opening_proof.int,
        field_cfg,
    )?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn prove_production_sha_hyrax_pcs_opening<C, Zt, F, const D: usize>(
    pcs_params: &PCSParams<AllHyraxPCSTypes<C>, Zt, F, D>,
    folded_commitments: &PCSCommitments<AllHyraxPCSTypes<C>, Zt, F, D>,
    folded_trace: &ProjectedTrace<F>,
    folded_prover_data: &PCSProverData<AllHyraxPCSTypes<C>, Zt, F, D>,
    r_0: &[F],
    folded_lifted_evals: &[DynamicPolynomialF<F>],
    field_cfg: &F::Config,
) -> Result<PCSOpeningProof<AllHyraxPCSTypes<C>, Zt, F, D>, ProductionShaError<F>>
where
    Zt: ZincTypes<D>,
    F: HyraxFieldBridge<C> + DelayedFieldProductSum,
    F::Inner: Transcribable,
    F::Modulus: Transcribable,
    C: AffineRepr,
    HyraxPCS<C, BinaryLanes>: PCS<
            F,
            BinaryPoly<D>,
            D,
            CommitmentKey = zip_plus::pcs::hyrax::HyraxCommitmentKey<C>,
            ProverData = zip_plus::pcs::hyrax::HyraxProverData<C>,
            OpeningProof = Vec<u8>,
        >,
    HyraxPCS<C, DensePolyScalarLanes>: PCS<
            F,
            DensePolynomial<Zt::Int, D>,
            D,
            CommitmentKey = zip_plus::pcs::hyrax::HyraxCommitmentKey<C>,
            ProverData = zip_plus::pcs::hyrax::HyraxProverData<C>,
            OpeningProof = Vec<u8>,
        >,
    HyraxPCS<C, IntScalarLane>: PCS<
            F,
            Zt::Int,
            D,
            CommitmentKey = zip_plus::pcs::hyrax::HyraxCommitmentKey<C>,
            ProverData = zip_plus::pcs::hyrax::HyraxProverData<C>,
            OpeningProof = Vec<u8>,
        >,
{
    ensure_production_sha_word_degree::<F, D>()?;
    validate_production_sha_batch_sizes::<F>(
        HyraxPCS::<C, BinaryLanes>::batch_size(&folded_commitments.binary),
        HyraxPCS::<C, DensePolyScalarLanes>::batch_size(&folded_commitments.arbitrary),
        HyraxPCS::<C, IntScalarLane>::batch_size(&folded_commitments.int),
    )?;
    let (binary_lifted, int_lifted) = split_folded_sha_pcs_lifted_evals(folded_lifted_evals)?;
    let arbitrary_lifted: &[DynamicPolynomialF<F>] = &[];

    let arbitrary_scalar_lanes: Vec<Vec<Vec<C::ScalarField>>> = Vec::new();
    let binary_field_lanes = folded_sha_binary_field_lanes(folded_trace);
    let int_field_lanes = folded_sha_int_field_lanes(folded_trace);

    let mut transcript = PcsProverTranscript {
        fs_transcript: Blake3Transcript::default(),
        stream: Cursor::default(),
    };
    let mut transcription_buf = vec![0u8; F::zero_with_cfg(field_cfg).inner().get_num_bytes()];

    HyraxPCS::<C, BinaryLanes>::absorb_commitment(
        &mut transcript.fs_transcript,
        &folded_commitments.binary,
    );
    absorb_pcs_lifted_evals(
        &mut transcript.fs_transcript,
        binary_lifted,
        &mut transcription_buf,
    );
    let binary_start = transcript.stream.position() as usize;
    HyraxPCS::<C, BinaryLanes>::prove_open_field_lanes_single_row::<F, true>(
        &mut transcript,
        &pcs_params.binary,
        &binary_field_lanes,
        r_0,
        &folded_prover_data.binary,
        field_cfg,
    )?;
    let binary_end = transcript.stream.position() as usize;
    let binary = transcript.stream.get_ref()[binary_start..binary_end].to_vec();

    HyraxPCS::<C, DensePolyScalarLanes>::absorb_commitment(
        &mut transcript.fs_transcript,
        &folded_commitments.arbitrary,
    );
    absorb_pcs_lifted_evals(
        &mut transcript.fs_transcript,
        arbitrary_lifted,
        &mut transcription_buf,
    );
    let arbitrary_start = transcript.stream.position() as usize;
    HyraxPCS::<C, DensePolyScalarLanes>::prove_open_scalar_lanes::<F, true>(
        &mut transcript,
        &pcs_params.arbitrary,
        &arbitrary_scalar_lanes,
        r_0,
        &folded_prover_data.arbitrary,
        field_cfg,
    )?;
    let arbitrary_end = transcript.stream.position() as usize;
    let arbitrary = transcript.stream.get_ref()[arbitrary_start..arbitrary_end].to_vec();

    HyraxPCS::<C, IntScalarLane>::absorb_commitment(
        &mut transcript.fs_transcript,
        &folded_commitments.int,
    );
    absorb_pcs_lifted_evals(
        &mut transcript.fs_transcript,
        int_lifted,
        &mut transcription_buf,
    );
    let int_start = transcript.stream.position() as usize;
    HyraxPCS::<C, IntScalarLane>::prove_open_field_lanes_single_row::<F, true>(
        &mut transcript,
        &pcs_params.int,
        &int_field_lanes,
        r_0,
        &folded_prover_data.int,
        field_cfg,
    )?;
    let int_end = transcript.stream.position() as usize;
    let int = transcript.stream.get_ref()[int_start..int_end].to_vec();

    Ok(PCSOpeningProof {
        binary,
        arbitrary,
        int,
    })
}

#[cfg(test)]
fn evaluate_fresh_targets_from_ideal_polys<F>(
    ideal_polys: &[[DynamicPolynomialF<F>; NUM_NONZERO_SHA_FAMILIES]],
    a: &F,
    lambda: &F,
    field_cfg: &F::Config,
) -> Result<Vec<F>, ProductionShaError<F>>
where
    F: DelayedFieldProductSum,
{
    let lambda_powers = zinc_utils::powers(
        lambda.clone(),
        F::one_with_cfg(field_cfg),
        NUM_SHA_RESIDUAL_FAMILIES,
    );
    let a_powers = zinc_utils::powers(
        a.clone(),
        F::one_with_cfg(field_cfg),
        SHA_IDEAL_EVAL_POWER_COUNT,
    );
    let nonzero_lambda_powers = selected_nonzero_sha_lambda_powers(&lambda_powers)?;
    ideal_polys
        .iter()
        .map(|instance| {
            let mut values: [F; NUM_NONZERO_SHA_FAMILIES] =
                std::array::from_fn(|_| F::zero_with_cfg(field_cfg));
            for (slot, poly) in instance.iter().enumerate() {
                values[slot] = evaluate_production_sha_poly_at_powers(poly, &a_powers, field_cfg)?;
            }
            FieldFieldInnerProduct::inner_product::<UNCHECKED>(
                &values,
                &nonzero_lambda_powers,
                F::zero_with_cfg(field_cfg),
            )
            .map_err(|_| {
                ProductionShaError::NonCanonicalProofObject(
                    "production SHA nonzero-family dot product failed",
                )
            })
        })
        .collect()
}

#[allow(dead_code)]
fn beta_aggregate_sha_ideal_polys<F>(
    ideal_polys: &[[DynamicPolynomialF<F>; NUM_NONZERO_SHA_FAMILIES]],
    beta: &[F],
    field_cfg: &F::Config,
) -> Result<[DynamicPolynomialF<F>; NUM_NONZERO_SHA_FAMILIES], ProductionShaError<F>>
where
    F: PrimeField,
{
    let weights = build_eq_x_r_vec(beta, field_cfg)?;
    if weights.len() != ideal_polys.len() {
        return Err(ProductionShaError::LengthMismatch {
            label: "beta weights/fresh ideal polys",
            got: weights.len(),
            expected: ideal_polys.len(),
        });
    }

    let mut aggregate: [DynamicPolynomialF<F>; NUM_NONZERO_SHA_FAMILIES] =
        std::array::from_fn(|_| DynamicPolynomialF::ZERO);
    for (weight, instance) in weights.iter().zip(ideal_polys) {
        for (slot, poly) in instance.iter().enumerate() {
            let weighted = scale_production_sha_poly(poly, weight);
            aggregate[slot] += &weighted;
        }
    }
    aggregate.iter_mut().for_each(DynamicPolynomialF::trim);
    Ok(aggregate)
}

#[allow(dead_code)]
fn scale_production_sha_poly<F>(poly: &DynamicPolynomialF<F>, scalar: &F) -> DynamicPolynomialF<F>
where
    F: PrimeField,
{
    DynamicPolynomialF::new_trimmed(
        poly.coeffs
            .iter()
            .map(|coeff| coeff.clone() * scalar)
            .collect::<Vec<_>>(),
    )
}

fn evaluate_aggregate_sha_ideal_claim<F>(
    ideal_polys: &[DynamicPolynomialF<F>; NUM_NONZERO_SHA_FAMILIES],
    a: &F,
    lambda: &F,
    field_cfg: &F::Config,
) -> Result<F, ProductionShaError<F>>
where
    F: DelayedFieldProductSum,
{
    let lambda_powers = zinc_utils::powers(
        lambda.clone(),
        F::one_with_cfg(field_cfg),
        NUM_SHA_RESIDUAL_FAMILIES,
    );
    let a_powers = zinc_utils::powers(
        a.clone(),
        F::one_with_cfg(field_cfg),
        SHA_IDEAL_EVAL_POWER_COUNT,
    );
    evaluate_aggregate_sha_ideal_claim_with_powers(
        ideal_polys,
        &a_powers,
        &lambda_powers,
        field_cfg,
    )
}

fn evaluate_aggregate_sha_ideal_claim_with_powers<F>(
    ideal_polys: &[DynamicPolynomialF<F>; NUM_NONZERO_SHA_FAMILIES],
    a_powers: &[F],
    lambda_powers: &[F],
    field_cfg: &F::Config,
) -> Result<F, ProductionShaError<F>>
where
    F: DelayedFieldProductSum,
{
    if a_powers.len() < SHA_IDEAL_EVAL_POWER_COUNT {
        return Err(ProductionShaError::LengthMismatch {
            label: "aggregate SHA ideal a powers",
            got: a_powers.len(),
            expected: SHA_IDEAL_EVAL_POWER_COUNT,
        });
    }
    let mut values: [F; NUM_NONZERO_SHA_FAMILIES] =
        std::array::from_fn(|_| F::zero_with_cfg(field_cfg));
    for (slot, poly) in ideal_polys.iter().enumerate() {
        values[slot] = evaluate_production_sha_poly_at_powers(poly, a_powers, field_cfg)?;
    }
    lambda_weighted_nonzero_sha_values(&values, lambda_powers, field_cfg)
}

fn selected_nonzero_sha_lambda_powers<F>(
    lambda_powers: &[F],
) -> Result<[F; NUM_NONZERO_SHA_FAMILIES], ProductionShaError<F>>
where
    F: PrimeField,
{
    if lambda_powers.len() != NUM_SHA_RESIDUAL_FAMILIES {
        return Err(ProductionShaError::LengthMismatch {
            label: "lambda powers",
            got: lambda_powers.len(),
            expected: NUM_SHA_RESIDUAL_FAMILIES,
        });
    }
    Ok(std::array::from_fn(|slot| {
        lambda_powers[production_sha_nonzero_families()[slot].index()].clone()
    }))
}

fn lambda_weighted_nonzero_sha_values<F>(
    values: &[F; NUM_NONZERO_SHA_FAMILIES],
    lambda_powers: &[F],
    field_cfg: &F::Config,
) -> Result<F, ProductionShaError<F>>
where
    F: DelayedFieldProductSum,
{
    let weights = selected_nonzero_sha_lambda_powers(lambda_powers)?;
    FieldFieldInnerProduct::inner_product::<UNCHECKED>(
        values,
        &weights,
        F::zero_with_cfg(field_cfg),
    )
    .map_err(|_| {
        ProductionShaError::NonCanonicalProofObject(
            "production SHA nonzero-family dot product failed",
        )
    })
}

fn lambda_weighted_sha_residual_polys_at_powers<F>(
    residuals: &[DynamicPolynomialF<F>],
    a_powers: &[F],
    lambda_powers: &[F],
    field_cfg: &F::Config,
) -> Result<F, ProductionShaError<F>>
where
    F: DelayedFieldProductSum,
{
    if residuals.len() != NUM_SHA_RESIDUAL_FAMILIES {
        return Err(ProductionShaError::LengthMismatch {
            label: "SHA residual families",
            got: residuals.len(),
            expected: NUM_SHA_RESIDUAL_FAMILIES,
        });
    }
    if lambda_powers.len() != NUM_SHA_RESIDUAL_FAMILIES {
        return Err(ProductionShaError::LengthMismatch {
            label: "lambda powers",
            got: lambda_powers.len(),
            expected: NUM_SHA_RESIDUAL_FAMILIES,
        });
    }
    let mut values: [F; NUM_SHA_RESIDUAL_FAMILIES] =
        std::array::from_fn(|_| F::zero_with_cfg(field_cfg));
    for (idx, residual) in residuals.iter().enumerate() {
        values[idx] = evaluate_production_sha_poly_at_powers(residual, a_powers, field_cfg)?;
    }
    FieldFieldInnerProduct::inner_product::<UNCHECKED>(
        &values,
        lambda_powers,
        F::zero_with_cfg(field_cfg),
    )
    .map_err(|_| {
        ProductionShaError::NonCanonicalProofObject(
            "production SHA residual-family dot product failed",
        )
    })
}

fn evaluate_production_sha_poly_at_powers<F>(
    poly: &DynamicPolynomialF<F>,
    powers: &[F],
    field_cfg: &F::Config,
) -> Result<F, ProductionShaError<F>>
where
    F: DelayedFieldProductSum,
{
    if poly.coeffs.is_empty() {
        return Ok(F::zero_with_cfg(field_cfg));
    }
    if poly.coeffs.len() > powers.len() {
        return Err(ProductionShaError::NonCanonicalProofObject(
            "production SHA polynomial exceeds scalarization power bound",
        ));
    }
    DynamicPolyFInnerProduct::inner_product::<UNCHECKED>(
        &poly.coeffs,
        &powers[..poly.coeffs.len()],
        F::zero_with_cfg(field_cfg),
    )
    .map_err(|_| {
        ProductionShaError::NonCanonicalProofObject("production SHA polynomial dot product failed")
    })
}

#[allow(dead_code)]
fn eq_weighted_sum<F>(
    point: &[F],
    values: &[F],
    field_cfg: &F::Config,
) -> Result<F, ProductionShaError<F>>
where
    F: DelayedFieldProductSum,
{
    let expected = 1usize
        .checked_shl(u32::try_from(point.len()).map_err(|_| {
            ProductionShaError::LengthMismatch {
                label: "eq point",
                got: point.len(),
                expected: usize::BITS as usize,
            }
        })?)
        .ok_or(ProductionShaError::LengthMismatch {
            label: "eq point",
            got: point.len(),
            expected: usize::BITS as usize,
        })?;
    if values.len() != expected {
        return Err(ProductionShaError::LengthMismatch {
            label: "eq-weighted values",
            got: values.len(),
            expected,
        });
    }
    let weights = build_eq_x_r_vec(point, field_cfg)?;
    FieldFieldInnerProduct::inner_product::<UNCHECKED>(
        &weights,
        values,
        F::zero_with_cfg(field_cfg),
    )
    .map_err(|_| {
        ProductionShaError::NonCanonicalProofObject("eq-weighted value dot product failed")
    })
}

fn fold_mle_tables<F>(
    kind: &'static str,
    tables: &[&MleTable<F>],
    theta: &[F],
    field_cfg: &F::Config,
) -> Result<MleTable<F>, ProductionShaError<F>>
where
    F: PrimeField,
{
    if tables.len() != theta.len() {
        return Err(ProductionShaError::LengthMismatch {
            label: kind,
            got: tables.len(),
            expected: theta.len(),
        });
    }
    let first = tables.first().ok_or(ProductionShaError::LengthMismatch {
        label: kind,
        got: 0,
        expected: 1,
    })?;
    let col_count = first.len();
    let first_col = first.first().ok_or(ProductionShaError::LengthMismatch {
        label: kind,
        got: 0,
        expected: 1,
    })?;
    let row_count = first_col.evaluations.len();
    let num_vars = first_col.num_vars;
    let mut folded = vec![vec![F::zero_with_cfg(field_cfg); row_count]; col_count];
    for (table, weight) in tables.iter().zip(theta.iter()) {
        if table.len() != col_count {
            return Err(ProductionShaError::LengthMismatch {
                label: kind,
                got: table.len(),
                expected: col_count,
            });
        }
        for (col_idx, column) in table.iter().enumerate() {
            if column.num_vars != num_vars {
                return Err(ProductionShaError::LengthMismatch {
                    label: kind,
                    got: column.num_vars,
                    expected: num_vars,
                });
            }
            if column.evaluations.len() != row_count {
                return Err(ProductionShaError::LengthMismatch {
                    label: kind,
                    got: column.evaluations.len(),
                    expected: row_count,
                });
            }
            for (out, value) in folded[col_idx].iter_mut().zip(column.evaluations.iter()) {
                *out += weight.clone() * value;
            }
        }
    }
    Ok(folded
        .into_iter()
        .map(|evaluations| DenseMultilinearExtension {
            evaluations,
            num_vars,
        })
        .collect())
}

fn fold_optional_mle_tables<F>(
    kind: &'static str,
    tables: &[Option<&MleTable<F>>],
    theta: &[F],
    field_cfg: &F::Config,
) -> Result<Option<MleTable<F>>, ProductionShaError<F>>
where
    F: PrimeField,
{
    let has_table = tables
        .first()
        .ok_or(ProductionShaError::LengthMismatch {
            label: kind,
            got: 0,
            expected: 1,
        })?
        .is_some();
    if !has_table {
        if tables.iter().any(Option::is_some) {
            return Err(ProductionShaError::LengthMismatch {
                label: kind,
                got: 1,
                expected: 0,
            });
        }
        return Ok(None);
    }
    let present = tables
        .iter()
        .map(|table| {
            table.ok_or(ProductionShaError::LengthMismatch {
                label: kind,
                got: 0,
                expected: 1,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    fold_mle_tables(kind, &present, theta, field_cfg).map(Some)
}

fn fold_projected_publics<F>(
    publics: &[ProjectedPublic<F>],
    theta: &[F],
    field_cfg: &F::Config,
) -> Result<ProjectedPublic<F>, ProductionShaError<F>>
where
    F: PrimeField,
{
    if publics.len() != theta.len() {
        return Err(ProductionShaError::LengthMismatch {
            label: "publics/theta",
            got: publics.len(),
            expected: theta.len(),
        });
    }
    let first = publics.first().ok_or(ProductionShaError::LengthMismatch {
        label: "publics",
        got: 0,
        expected: 1,
    })?;
    let columns = fold_mle_tables(
        "public columns",
        &publics
            .iter()
            .map(|public| &public.columns)
            .collect::<Vec<_>>(),
        theta,
        field_cfg,
    )?;
    let bit_slices = fold_optional_mle_tables(
        "public bit slices",
        &publics
            .iter()
            .map(|public| public.bit_slices.as_ref())
            .collect::<Vec<_>>(),
        theta,
        field_cfg,
    )?;
    if first.bit_slices.is_none() != bit_slices.is_none() {
        return Err(ProductionShaError::LengthMismatch {
            label: "public bit slice presence",
            got: usize::from(bit_slices.is_some()),
            expected: usize::from(first.bit_slices.is_some()),
        });
    }
    Ok(ProjectedPublic {
        columns,
        bit_slices,
    })
}

fn validate_production_sha_publics<F>(
    publics: &[ProjectedPublic<F>],
    field_cfg: &F::Config,
) -> Result<(), ProductionShaError<F>>
where
    F: PrimeField + FromPrimitiveWithConfig,
{
    for public in publics {
        if public.columns.len() != ShaPublicCol::COUNT {
            return Err(ProductionShaError::LengthMismatch {
                label: "SHA public column count",
                got: public.columns.len(),
                expected: ShaPublicCol::COUNT,
            });
        }
        for col in &public.columns {
            if col.num_vars != SHA_ROW_VARS || col.evaluations.len() != SHA_ROW_COUNT {
                return Err(ProductionShaError::LengthMismatch {
                    label: "SHA public row count",
                    got: col.evaluations.len(),
                    expected: SHA_ROW_COUNT,
                });
            }
        }
        for selector in [
            ShaPublicCol::SInit,
            ShaPublicCol::SMsg,
            ShaPublicCol::SSched,
            ShaPublicCol::SUpd,
            ShaPublicCol::SFf,
            ShaPublicCol::SOut,
        ] {
            let col = &public.columns[selector.index()];
            for (row, value) in col.evaluations.iter().enumerate() {
                let expected = production_sha_selector_expected(selector, row, field_cfg);
                if value != &expected {
                    if *value != F::zero_with_cfg(field_cfg) && *value != F::one_with_cfg(field_cfg)
                    {
                        return Err(ProductionShaError::NonBooleanPublicSelector {
                            col: selector,
                            row,
                        });
                    }
                    return Err(ProductionShaError::InvalidPublicSelector { col: selector, row });
                }
            }
        }

        let k_col = &public.columns[ShaPublicCol::K.index()];
        for (row, value) in k_col.evaluations.iter().enumerate() {
            let expected = production_sha_k_expected(row, field_cfg);
            if value != &expected {
                return Err(ProductionShaError::InvalidRoundConstant { row });
            }
        }

        validate_production_sha_public_word_columns(public, field_cfg)?;
    }
    Ok(())
}

fn validate_production_sha_public_word_columns<F>(
    public: &ProjectedPublic<F>,
    field_cfg: &F::Config,
) -> Result<(), ProductionShaError<F>>
where
    F: PrimeField,
{
    let bit_slices =
        public
            .bit_slices
            .as_ref()
            .ok_or(ProductionShaError::NonCanonicalProofObject(
                "production SHA public word columns are required",
            ))?;
    if bit_slices.len() != ShaPublicWordCol::COUNT * SHA_WORD_BITS {
        return Err(ProductionShaError::LengthMismatch {
            label: "SHA public word column count",
            got: bit_slices.len(),
            expected: ShaPublicWordCol::COUNT * SHA_WORD_BITS,
        });
    }

    for (word_idx, public_col) in production_sha_public_word_column_map().iter().enumerate() {
        let scalar_col = &public.columns[public_col.index()];
        for row in 0..SHA_ROW_COUNT {
            let mut bits = Vec::with_capacity(SHA_WORD_BITS);
            for bit in 0..SHA_WORD_BITS {
                let table_idx = bit_slice_index(word_idx, bit, SHA_WORD_BITS);
                let bit_col =
                    bit_slices
                        .get(table_idx)
                        .ok_or(ProductionShaError::LengthMismatch {
                            label: "SHA public word bit column",
                            got: table_idx,
                            expected: bit_slices.len(),
                        })?;
                if bit_col.num_vars != SHA_ROW_VARS || bit_col.evaluations.len() != SHA_ROW_COUNT {
                    return Err(ProductionShaError::LengthMismatch {
                        label: "SHA public word row count",
                        got: bit_col.evaluations.len(),
                        expected: SHA_ROW_COUNT,
                    });
                }
                let bit = bit_col.evaluations[row].clone();
                if bit != F::zero_with_cfg(field_cfg) && bit != F::one_with_cfg(field_cfg) {
                    return Err(ProductionShaError::NonCanonicalProofObject(
                        "production SHA public word bit is not boolean",
                    ));
                }
                bits.push(bit);
            }
            if bits.len() != SHA_WORD_BITS {
                return Err(ProductionShaError::LengthMismatch {
                    label: "SHA public word bit count",
                    got: bits.len(),
                    expected: SHA_WORD_BITS,
                });
            }
            let scalarized = scalarize_sha_public_word_bits_at_two(&bits, field_cfg);
            if scalarized != scalar_col.evaluations[row] {
                return Err(ProductionShaError::NonCanonicalProofObject(
                    "production SHA public word bits do not match scalar public column",
                ));
            }
        }
        debug_assert_eq!(Some(word_idx), public_word_col_index(*public_col));
    }
    Ok(())
}

fn production_sha_public_word_column_map() -> [ShaPublicCol; ShaPublicWordCol::COUNT] {
    [
        ShaPublicCol::PAIn,
        ShaPublicCol::PEIn,
        ShaPublicCol::PAOut,
        ShaPublicCol::PEOut,
        ShaPublicCol::Message,
    ]
}

fn scalarize_sha_public_word_bits_at_two<F>(bits: &[F], field_cfg: &F::Config) -> F
where
    F: PrimeField,
{
    let two = F::one_with_cfg(field_cfg) + F::one_with_cfg(field_cfg);
    let mut power = F::one_with_cfg(field_cfg);
    let mut out = F::zero_with_cfg(field_cfg);
    for bit in bits {
        out += bit.clone() * &power;
        power *= &two;
    }
    out
}

fn production_sha_selector_expected<F>(
    selector: ShaPublicCol,
    row: usize,
    field_cfg: &F::Config,
) -> F
where
    F: PrimeField,
{
    let active = match selector {
        ShaPublicCol::SInit => row < 4,
        ShaPublicCol::SMsg => row < 16,
        ShaPublicCol::SSched => row < 48,
        ShaPublicCol::SUpd => row < 64,
        ShaPublicCol::SFf => (64..68).contains(&row),
        ShaPublicCol::SOut => (68..72).contains(&row),
        _ => false,
    };
    if active {
        F::one_with_cfg(field_cfg)
    } else {
        F::zero_with_cfg(field_cfg)
    }
}

fn production_sha_k_expected<F>(row: usize, field_cfg: &F::Config) -> F
where
    F: PrimeField + FromPrimitiveWithConfig,
{
    if (3..67).contains(&row) {
        F::from_with_cfg(SHA256_ROUND_CONSTANTS[row - 3] as u64, field_cfg)
    } else {
        F::zero_with_cfg(field_cfg)
    }
}

#[allow(dead_code)]
fn build_folded_sha_pcs_lifted_evals<F>(
    folded_trace: &ProjectedTrace<F>,
    r_0: &[F],
    field_cfg: &F::Config,
) -> Result<Vec<DynamicPolynomialF<F>>, ProductionShaError<F>>
where
    F: PrimeField + DelayedFieldProductSum,
{
    let row_weights = build_eq_x_r_vec(r_0, field_cfg)?;
    build_folded_sha_pcs_lifted_evals_with_row_weights(folded_trace, &row_weights, field_cfg)
}

fn build_folded_sha_pcs_lifted_evals_with_row_weights<F>(
    folded_trace: &ProjectedTrace<F>,
    row_weights: &[F],
    field_cfg: &F::Config,
) -> Result<Vec<DynamicPolynomialF<F>>, ProductionShaError<F>>
where
    F: PrimeField + DelayedFieldProductSum,
{
    if row_weights.len() != SHA_ROW_COUNT {
        return Err(ProductionShaError::LengthMismatch {
            label: "row weights",
            got: row_weights.len(),
            expected: SHA_ROW_COUNT,
        });
    }
    #[cfg(debug_assertions)]
    validate_projected_trace(folded_trace)?;
    let word_lifted = cfg_iter!(&ShaWordCol::ALL)
        .map(|&col| {
            let coeffs = sha_word_bits_at_point_with_weights_unchecked(
                folded_trace,
                col,
                0,
                row_weights,
                field_cfg,
            )?
            .to_vec();
            Ok(DynamicPolynomialF::new_trimmed(coeffs))
        })
        .collect::<Result<Vec<_>, ProductionShaError<F>>>()?;
    let int_lifted = cfg_iter!(&ShaIntCol::ALL)
        .map(|&col| {
            Ok(DynamicPolynomialF::new_trimmed([
                sha_int_at_point_with_weights_unchecked(folded_trace, col, row_weights, field_cfg)?,
            ]))
        })
        .collect::<Result<Vec<_>, ProductionShaError<F>>>()?;
    Ok(word_lifted.into_iter().chain(int_lifted).collect())
}

fn split_folded_sha_pcs_lifted_evals<F>(
    lifted_evals: &[DynamicPolynomialF<F>],
) -> Result<(&[DynamicPolynomialF<F>], &[DynamicPolynomialF<F>]), ProductionShaError<F>>
where
    F: PrimeField,
{
    let expected = ShaWordCol::COUNT + ShaIntCol::COUNT;
    if lifted_evals.len() != expected {
        return Err(ProductionShaError::LengthMismatch {
            label: "folded SHA PCS lifted evals",
            got: lifted_evals.len(),
            expected,
        });
    }
    validate_folded_sha_pcs_lifted_evals_canonical(lifted_evals)?;
    Ok(lifted_evals.split_at(ShaWordCol::COUNT))
}

fn validate_folded_sha_pcs_lifted_evals_canonical<F>(
    lifted_evals: &[DynamicPolynomialF<F>],
) -> Result<(), ProductionShaError<F>>
where
    F: PrimeField,
{
    for (idx, lifted_eval) in lifted_evals.iter().enumerate() {
        let max_len = if idx < ShaWordCol::COUNT {
            SHA_WORD_BITS
        } else {
            1
        };
        if lifted_eval.coeffs.len() > max_len {
            return Err(ProductionShaError::NonCanonicalProofObject(
                "folded SHA lifted eval has too many coefficients",
            ));
        }
        if lifted_eval.coeffs.last().is_some_and(F::is_zero) {
            return Err(ProductionShaError::NonCanonicalProofObject(
                "folded SHA lifted eval has trailing zero coefficients",
            ));
        }
    }
    Ok(())
}

#[allow(dead_code)]
fn folded_sha_binary_scalar_lanes<C, F>(
    folded_trace: &ProjectedTrace<F>,
) -> Result<Vec<Vec<Vec<C::ScalarField>>>, ZipError>
where
    C: AffineRepr,
    F: HyraxFieldBridge<C>,
{
    let lanes = cfg_into_iter!(0..ShaWordCol::COUNT * SHA_WORD_BITS)
        .map(|flat_idx| {
            let col_idx = flat_idx / SHA_WORD_BITS;
            let bit = flat_idx % SHA_WORD_BITS;
            let column = &folded_trace.bit_slices[bit_slice_index(col_idx, bit, SHA_WORD_BITS)];
            column
                .evaluations
                .iter()
                .map(F::field_to_scalar)
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut out = Vec::with_capacity(ShaWordCol::COUNT);
    let mut lanes = lanes.into_iter();
    for _ in 0..ShaWordCol::COUNT {
        let mut col_lanes = Vec::with_capacity(SHA_WORD_BITS);
        for _ in 0..SHA_WORD_BITS {
            col_lanes.push(
                lanes
                    .next()
                    .expect("flat binary scalar lane count is exact"),
            );
        }
        out.push(col_lanes);
    }
    Ok(out)
}

#[allow(dead_code)]
fn folded_sha_int_scalar_lanes<C, F>(
    folded_trace: &ProjectedTrace<F>,
) -> Result<Vec<Vec<Vec<C::ScalarField>>>, ZipError>
where
    C: AffineRepr,
    F: HyraxFieldBridge<C>,
{
    cfg_iter!(&ShaIntCol::ALL)
        .map(|col| {
            let column = &folded_trace.int_columns[col.index()];
            let lane = column
                .evaluations
                .iter()
                .map(F::field_to_scalar)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(vec![lane])
        })
        .collect()
}

fn folded_sha_binary_field_lanes<F>(folded_trace: &ProjectedTrace<F>) -> Vec<Vec<&[F]>>
where
    F: PrimeField,
{
    ShaWordCol::ALL
        .iter()
        .map(|col| {
            (0..SHA_WORD_BITS)
                .map(|bit| {
                    folded_trace.bit_slices[bit_slice_index(col.index(), bit, SHA_WORD_BITS)]
                        .evaluations
                        .as_slice()
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn folded_sha_int_field_lanes<F>(folded_trace: &ProjectedTrace<F>) -> Vec<Vec<&[F]>>
where
    F: PrimeField,
{
    ShaIntCol::ALL
        .iter()
        .map(|col| vec![folded_trace.int_columns[col.index()].evaluations.as_slice()])
        .collect()
}

fn absorb_pcs_lifted_evals<F>(
    transcript: &mut impl Transcript,
    lifted_evals: &[DynamicPolynomialF<F>],
    transcription_buf: &mut Vec<u8>,
) where
    F: PrimeField,
    F::Inner: Transcribable,
    F::Modulus: Transcribable,
{
    for lifted_eval in lifted_evals {
        transcript.absorb_random_field_slice(&lifted_eval.coeffs, transcription_buf);
    }
}

fn multipoint_open_evals_from_pcs_lifted<F>(
    lifted_evals: &[DynamicPolynomialF<F>],
    layout: &ShaMultipointLayout,
    folded_public: &ProjectedPublic<F>,
    r_0: &[F],
    field_cfg: &F::Config,
) -> Result<Vec<F>, ProductionShaError<F>>
where
    F: PrimeField,
{
    split_folded_sha_pcs_lifted_evals(lifted_evals)?;
    layout
        .sources
        .iter()
        .map(|source| match *source {
            ShaMpSource::Public { col } => {
                sha_public_at_point(folded_public, col, 0, r_0, field_cfg)
                    .map_err(ProductionShaError::from)
            }
            ShaMpSource::WordBit { col, bit } => Ok(lifted_evals[col.index()]
                .coeffs
                .get(bit)
                .cloned()
                .unwrap_or_else(|| F::zero_with_cfg(field_cfg))),
            ShaMpSource::Int { col } => Ok(lifted_evals[ShaWordCol::COUNT + col.index()]
                .coeffs
                .first()
                .cloned()
                .unwrap_or_else(|| F::zero_with_cfg(field_cfg))),
        })
        .collect()
}

pub fn prove_sha_sumfold_targets<F>(
    transcript: &mut impl Transcript,
    fresh_targets: &[F],
    beta: &[F],
    prefix_vars: usize,
    field_cfg: &F::Config,
) -> Result<(MultiDegreeSumcheckProof<F>, InstanceFoldClaim<F>), ProductionShaError<F>>
where
    F: InnerTransparentField
        + DelayedFieldProductSum
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: Transcribable + Zero,
    F::Modulus: Transcribable,
{
    let claims = zinc_piop::neutron_nova::LinearInstanceClaims::new(fresh_targets.to_vec())?;
    let group = claims.build_hybrid_sumcheck_group(beta, prefix_vars, field_cfg)?;
    let (proof, states) =
        MultiDegreeSumcheck::prove_as_subprotocol(transcript, vec![group], claims.ell(), field_cfg);
    let r_b = states
        .first()
        .ok_or(ProductionShaError::LengthMismatch {
            label: "sumfold states",
            got: 0,
            expected: 1,
        })?
        .randomness
        .clone();
    let c_sf = sumfold_expected_eval(beta, fresh_targets, &r_b, field_cfg)?;
    let output = derive_instance_fold_claim(beta, r_b, c_sf, fresh_targets.len(), field_cfg)?;
    Ok((proof, output))
}

pub fn verify_sha_sumfold_targets<F>(
    transcript: &mut impl Transcript,
    proof: &MultiDegreeSumcheckProof<F>,
    fresh_targets: &[F],
    beta: &[F],
    field_cfg: &F::Config,
) -> Result<InstanceFoldClaim<F>, ProductionShaError<F>>
where
    F: InnerTransparentField
        + DelayedFieldProductSum
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: Transcribable + Zero,
    F::Modulus: Transcribable,
{
    require_single_sumcheck_group(proof, "SHA SumFold")?;
    for &degree in proof.degrees() {
        if degree > 3 {
            return Err(ProductionShaError::SumFoldDegreeTooHigh { degree });
        }
    }
    let claims = zinc_piop::neutron_nova::LinearInstanceClaims::new(fresh_targets.to_vec())?;
    let subclaims =
        MultiDegreeSumcheck::verify_as_subprotocol(transcript, claims.ell(), proof, field_cfg)?;
    let r_b = subclaims.point().to_vec();
    let c_sf = subclaims.expected_evaluations()[0].clone();
    if c_sf != sumfold_expected_eval(beta, fresh_targets, &r_b, field_cfg)? {
        return Err(ProductionShaError::SumFoldTerminalMismatch);
    }
    Ok(derive_instance_fold_claim(
        beta,
        r_b,
        c_sf,
        fresh_targets.len(),
        field_cfg,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn prove_full_sha_sumfold<F>(
    transcript: &mut impl Transcript,
    traces: &[ProjectedTrace<F>],
    publics: &[ProjectedPublic<F>],
    initial_claim: &F,
    beta: &[F],
    r_ic: &[F; SHA_ROW_VARS],
    a: &F,
    lambda: &F,
    rho: &F,
    xi: &F,
    booleanity_sources: &[ShaBooleanitySource],
    field_cfg: &F::Config,
) -> Result<(MultiDegreeSumcheckProof<F>, InstanceFoldClaim<F>), ProductionShaError<F>>
where
    F: InnerTransparentField
        + DelayedFieldProductSum
        + ShaBinaryFoldField
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: Transcribable + Zero,
    F::Modulus: Transcribable,
{
    let group = build_dense_sha_sumfold_group(
        traces,
        publics,
        beta,
        r_ic,
        a,
        lambda,
        rho,
        xi,
        booleanity_sources,
        field_cfg,
    )?;
    let ell = beta.len();
    let (proof, states) =
        MultiDegreeSumcheck::prove_as_subprotocol(transcript, vec![group], ell, field_cfg);
    require_single_sumcheck_group(&proof, "SHA SumFold")?;
    if proof.claimed_sums()[0] != *initial_claim {
        return Err(ProductionShaError::SumFoldTerminalMismatch);
    }

    let r_b = states
        .first()
        .ok_or(ProductionShaError::LengthMismatch {
            label: "sumfold states",
            got: 0,
            expected: 1,
        })?
        .randomness
        .clone();
    let provisional = derive_instance_fold_claim(
        beta,
        r_b.clone(),
        F::one_with_cfg(field_cfg),
        traces.len(),
        field_cfg,
    )?;
    let (folded, folded_public) =
        zinc_piop::neutron_nova::fold_projected_traces(traces, publics, &provisional, field_cfg)?;
    let post_sumfold_claim = zinc_piop::neutron_nova::expression_folded_row_sum(
        &folded.trace,
        &folded_public,
        r_ic,
        a,
        lambda,
        rho,
        xi,
        booleanity_sources,
        field_cfg,
    )?;
    let d = eq_eval(beta, &r_b, F::one_with_cfg(field_cfg))?;
    let c_sf = d * post_sumfold_claim;
    Ok((
        proof,
        derive_instance_fold_claim(beta, r_b, c_sf, traces.len(), field_cfg)?,
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn prove_optimized_sha_sumfold<F>(
    transcript: &mut impl Transcript,
    traces: &[ProjectedTrace<F>],
    publics: &[ProjectedPublic<F>],
    initial_claim: &F,
    beta: &[F],
    r_ic: &[F; SHA_ROW_VARS],
    a: &F,
    lambda: &F,
    rho: &F,
    xi: &F,
    booleanity_sources: &[ShaBooleanitySource],
    coeff_tables: &[LinearResidualCoeffTable<F>],
    prefix_vars: usize,
    field_cfg: &F::Config,
) -> Result<(MultiDegreeSumcheckProof<F>, InstanceFoldClaim<F>), ProductionShaError<F>>
where
    F: InnerTransparentField
        + DelayedFieldProductSum
        + ShaBinaryFoldField
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: Transcribable + Zero,
    F::Modulus: Transcribable,
{
    let beta_eq_weights = build_eq_x_r_vec(beta, field_cfg)?;
    let r_ic_eq_weights = build_eq_x_r_vec(r_ic, field_cfg)?;
    let a_powers = build_sha_residual_eval_powers(a, field_cfg);
    let lambda_powers = build_sha_lambda_powers(lambda, field_cfg);
    let booleanity_weights = build_booleanity_weights(rho, xi, booleanity_sources.len(), field_cfg);
    let linear_accumulator =
        build_sha_sumfold_linear_accumulator(coeff_tables, &a_powers, &lambda_powers, field_cfg)?;
    let quadratic_prefix_accumulator = build_sha_sumfold_quadratic_prefix_accumulator(
        traces,
        booleanity_sources,
        prefix_vars,
        &r_ic_eq_weights,
        &booleanity_weights,
        field_cfg,
    )?;
    let group = build_production_sha_sumfold_group_from_prefix_accumulators(
        traces,
        beta,
        &beta_eq_weights,
        &r_ic_eq_weights,
        &linear_accumulator,
        &quadratic_prefix_accumulator,
        &booleanity_weights,
        booleanity_sources,
        prefix_vars,
        field_cfg,
    )?;
    let (proof, r_b) = prove_optimized_sha_sumfold_with_weights(
        transcript,
        group,
        initial_claim,
        beta.len(),
        field_cfg,
    )?;
    let provisional = derive_instance_fold_claim(
        beta,
        r_b.clone(),
        F::one_with_cfg(field_cfg),
        traces.len(),
        field_cfg,
    )?;
    let (folded, folded_public) =
        zinc_piop::neutron_nova::fold_projected_traces(traces, publics, &provisional, field_cfg)?;
    let row_claim = expression_folded_row_sum_with_vectors(
        &folded.trace,
        &folded_public,
        &r_ic_eq_weights,
        &a_powers,
        &lambda_powers,
        &booleanity_weights,
        booleanity_sources,
        field_cfg,
    )?;
    Ok((
        proof,
        derive_instance_fold_claim_from_row_claim(beta, r_b, &row_claim, traces.len(), field_cfg)?,
    ))
}

pub fn prove_optimized_sha_sumfold_with_weights<F>(
    transcript: &mut impl Transcript,
    group: MultiDegreeSumcheckGroup<F>,
    initial_claim: &F,
    instance_vars: usize,
    field_cfg: &F::Config,
) -> Result<(MultiDegreeSumcheckProof<F>, Vec<F>), ProductionShaError<F>>
where
    F: InnerTransparentField
        + DelayedFieldProductSum
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: Transcribable + Zero,
    F::Modulus: Transcribable,
{
    let (proof, states) = MultiDegreeSumcheck::prove_as_subprotocol(
        transcript,
        vec![group],
        instance_vars,
        field_cfg,
    );
    require_single_sumcheck_group(&proof, "SHA SumFold")?;
    if proof.claimed_sums()[0] != *initial_claim {
        return Err(ProductionShaError::SumFoldTerminalMismatch);
    }

    Ok((
        proof,
        states
            .first()
            .ok_or(ProductionShaError::LengthMismatch {
                label: "sumfold states",
                got: 0,
                expected: 1,
            })?
            .randomness
            .clone(),
    ))
}

pub fn derive_instance_fold_claim_from_row_claim<F>(
    beta: &[F],
    r_b: Vec<F>,
    row_claim: &F,
    instance_count: usize,
    field_cfg: &F::Config,
) -> Result<InstanceFoldClaim<F>, ProductionShaError<F>>
where
    F: PrimeField,
{
    let d = eq_eval(beta, &r_b, F::one_with_cfg(field_cfg))?;
    let c_sf = d * row_claim;
    derive_instance_fold_claim(beta, r_b, c_sf, instance_count, field_cfg)
        .map_err(ProductionShaError::from)
}

pub fn verify_full_sha_sumfold<F>(
    transcript: &mut impl Transcript,
    proof: &MultiDegreeSumcheckProof<F>,
    initial_claim: &F,
    instance_vars: usize,
    field_cfg: &F::Config,
) -> Result<VerifiedShaSumFold<F>, ProductionShaError<F>>
where
    F: InnerTransparentField
        + DelayedFieldProductSum
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: Transcribable + Zero,
    F::Modulus: Transcribable,
{
    require_single_sumcheck_group(proof, "SHA SumFold")?;
    for &degree in proof.degrees() {
        if degree > 3 {
            return Err(ProductionShaError::SumFoldDegreeTooHigh { degree });
        }
    }
    let Some(claimed_sum) = proof.claimed_sums().first() else {
        return Err(ProductionShaError::LengthMismatch {
            label: "SHA SumFold claimed sums",
            got: 0,
            expected: 1,
        });
    };
    if claimed_sum != initial_claim {
        return Err(ProductionShaError::SumFoldTerminalMismatch);
    }

    let subclaims =
        MultiDegreeSumcheck::verify_as_subprotocol(transcript, instance_vars, proof, field_cfg)?;
    let r_b = subclaims.point().to_vec();
    let c_sf = subclaims.expected_evaluations()[0].clone();
    Ok(VerifiedShaSumFold { r_b, c_sf })
}

pub fn fold_pcs_commitments<P, Zt, F, const D: usize>(
    commitments: &[PCSCommitments<P, Zt, F, D>],
    theta: &[F],
    field_cfg: &F::Config,
) -> Result<PCSCommitments<P, Zt, F, D>, ProductionShaError<F>>
where
    Zt: ZincTypes<D>,
    F: PrimeField,
    P: ZincPCSTypes<Zt, F, D>,
    P::BinaryPCS: FoldablePCS<F, BinaryPoly<D>, D>,
    P::ArbitraryPCS: FoldablePCS<F, DensePolynomial<Zt::Int, D>, D>,
    P::IntPCS: FoldablePCS<F, Zt::Int, D>,
{
    if commitments.len() != theta.len() {
        return Err(ProductionShaError::LengthMismatch {
            label: "commitments/theta",
            got: commitments.len(),
            expected: theta.len(),
        });
    }
    let binary = commitments
        .iter()
        .map(|commitment| &commitment.binary)
        .collect::<Vec<_>>();
    let arbitrary = commitments
        .iter()
        .map(|commitment| &commitment.arbitrary)
        .collect::<Vec<_>>();
    let int = commitments
        .iter()
        .map(|commitment| &commitment.int)
        .collect::<Vec<_>>();
    Ok(PCSCommitments {
        binary: P::BinaryPCS::fold_commitment_refs(&binary, theta, field_cfg)?,
        arbitrary: P::ArbitraryPCS::fold_commitment_refs(&arbitrary, theta, field_cfg)?,
        int: P::IntPCS::fold_commitment_refs(&int, theta, field_cfg)?,
    })
}

pub fn fold_pcs_prover_data<P, Zt, F, const D: usize>(
    prover_data: &[PCSProverData<P, Zt, F, D>],
    theta: &[F],
    field_cfg: &F::Config,
) -> Result<PCSProverData<P, Zt, F, D>, ProductionShaError<F>>
where
    Zt: ZincTypes<D>,
    F: PrimeField,
    P: ZincPCSTypes<Zt, F, D>,
    P::BinaryPCS: FoldablePCS<F, BinaryPoly<D>, D>,
    P::ArbitraryPCS: FoldablePCS<F, DensePolynomial<Zt::Int, D>, D>,
    P::IntPCS: FoldablePCS<F, Zt::Int, D>,
{
    if prover_data.len() != theta.len() {
        return Err(ProductionShaError::LengthMismatch {
            label: "prover_data/theta",
            got: prover_data.len(),
            expected: theta.len(),
        });
    }
    let binary = prover_data
        .iter()
        .map(|data| data.binary.clone())
        .collect::<Vec<_>>();
    let arbitrary = prover_data
        .iter()
        .map(|data| data.arbitrary.clone())
        .collect::<Vec<_>>();
    let int = prover_data
        .iter()
        .map(|data| data.int.clone())
        .collect::<Vec<_>>();
    Ok(PCSProverData {
        binary: P::BinaryPCS::fold_prover_data(&binary, theta, field_cfg)?,
        arbitrary: P::ArbitraryPCS::fold_prover_data(&arbitrary, theta, field_cfg)?,
        int: P::IntPCS::fold_prover_data(&int, theta, field_cfg)?,
    })
}

pub fn prove_folded_row_sumcheck<F>(
    transcript: &mut impl Transcript,
    row_integrand_values: &[F],
    post_sumfold_claim: &F,
    field_cfg: &F::Config,
) -> Result<MultiDegreeSumcheckProof<F>, ProductionShaError<F>>
where
    F: InnerTransparentField
        + DelayedFieldProductSum
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: Transcribable + Zero,
    F::Modulus: Transcribable,
{
    let claimed = folded_row_integrand_sum(row_integrand_values, field_cfg)?;
    verify_folded_row_sumcheck_claim(&claimed, post_sumfold_claim)?;
    let group = build_folded_row_sumcheck_group(row_integrand_values, field_cfg)?;
    let (proof, _) =
        MultiDegreeSumcheck::prove_as_subprotocol(transcript, vec![group], SHA_ROW_VARS, field_cfg);
    Ok(proof)
}

#[derive(Clone)]
struct RowExpressionOffsets {
    word: [[usize; ROW_EXPR_WORD_SHIFT_SLOTS]; ShaWordCol::COUNT],
    int: [usize; ShaIntCol::COUNT],
    public_scalar: [[usize; ROW_EXPR_PUBLIC_SHIFT_SLOTS]; ShaPublicCol::COUNT],
    public_word: [usize; ShaPublicCol::COUNT],
}

const ROW_EXPR_MISSING_SOURCE: usize = usize::MAX;
const ROW_EXPR_WORD_SHIFT_SLOTS: usize = 17;
const ROW_EXPR_PUBLIC_SHIFT_SLOTS: usize = 4;

#[derive(Clone)]
struct RowExpressionLayout {
    word_sources: Vec<(ShaWordCol, usize)>,
    int_sources: Vec<ShaIntCol>,
    public_scalar_sources: Vec<(ShaPublicCol, usize)>,
    public_word_sources: Vec<ShaPublicCol>,
    word_offset: usize,
    int_offset: usize,
    public_scalar_offset: usize,
    public_word_offset: usize,
}

impl RowExpressionLayout {
    fn new() -> Self {
        let word_sources = production_sha_endpoint_word_sources();
        let int_sources = production_sha_endpoint_int_sources();
        let mut public_scalar_sources = ShaPublicCol::ALL
            .iter()
            .copied()
            .map(|col| (col, 0))
            .collect::<Vec<_>>();
        public_scalar_sources.push((ShaPublicCol::K, 3));
        let public_word_sources = vec![
            ShaPublicCol::PAIn,
            ShaPublicCol::PEIn,
            ShaPublicCol::PAOut,
            ShaPublicCol::PEOut,
            ShaPublicCol::Message,
        ];
        let word_offset = 1;
        let int_offset = word_offset + word_sources.len() * SHA_WORD_BITS;
        let public_scalar_offset = int_offset + int_sources.len();
        let public_word_offset = public_scalar_offset + public_scalar_sources.len();
        Self {
            word_sources,
            int_sources,
            public_scalar_sources,
            public_word_sources,
            word_offset,
            int_offset,
            public_scalar_offset,
            public_word_offset,
        }
    }

    fn offsets(&self) -> RowExpressionOffsets {
        let mut word = [[ROW_EXPR_MISSING_SOURCE; ROW_EXPR_WORD_SHIFT_SLOTS]; ShaWordCol::COUNT];
        for (idx, &(col, shift)) in self.word_sources.iter().enumerate() {
            word[col.index()][shift] = idx;
        }

        let mut int = [ROW_EXPR_MISSING_SOURCE; ShaIntCol::COUNT];
        for (idx, &col) in self.int_sources.iter().enumerate() {
            int[col.index()] = idx;
        }

        let mut public_scalar =
            [[ROW_EXPR_MISSING_SOURCE; ROW_EXPR_PUBLIC_SHIFT_SLOTS]; ShaPublicCol::COUNT];
        for (idx, &(col, shift)) in self.public_scalar_sources.iter().enumerate() {
            public_scalar[col.index()][shift] = idx;
        }

        let mut public_word = [ROW_EXPR_MISSING_SOURCE; ShaPublicCol::COUNT];
        for (idx, &col) in self.public_word_sources.iter().enumerate() {
            public_word[col.index()] = idx;
        }

        RowExpressionOffsets {
            word,
            int,
            public_scalar,
            public_word,
        }
    }
}

fn trace_word_bit_at_row<F>(
    trace: &ProjectedTrace<F>,
    col: ShaWordCol,
    row: usize,
    shift: usize,
    bit: usize,
    field_cfg: &F::Config,
) -> Result<F, ProductionShaError<F>>
where
    F: PrimeField,
{
    if bit >= SHA_WORD_BITS {
        return Err(ProductionShaError::LengthMismatch {
            label: "SHA word bit index",
            got: bit,
            expected: SHA_WORD_BITS,
        });
    }
    let Some(shifted) = row.checked_add(shift) else {
        return Ok(F::zero_with_cfg(field_cfg));
    };
    if shifted >= SHA_ROW_COUNT {
        return Ok(F::zero_with_cfg(field_cfg));
    }
    trace
        .bit_slices
        .get(bit_slice_index(col.index(), bit, SHA_WORD_BITS))
        .and_then(|column| column.evaluations.get(shifted))
        .cloned()
        .ok_or(ProductionShaError::LengthMismatch {
            label: "SHA trace word bit",
            got: shifted,
            expected: SHA_ROW_COUNT,
        })
}

fn trace_int_at_row<F>(
    trace: &ProjectedTrace<F>,
    col: ShaIntCol,
    row: usize,
    field_cfg: &F::Config,
) -> Result<F, ProductionShaError<F>>
where
    F: PrimeField,
{
    if row >= SHA_ROW_COUNT {
        return Ok(F::zero_with_cfg(field_cfg));
    }
    trace
        .int_columns
        .get(col.index())
        .and_then(|column| column.evaluations.get(row))
        .cloned()
        .ok_or(ProductionShaError::LengthMismatch {
            label: "SHA trace int row",
            got: row,
            expected: SHA_ROW_COUNT,
        })
}

fn public_scalar_at_row<F>(
    public: &ProjectedPublic<F>,
    col: ShaPublicCol,
    row: usize,
    shift: usize,
    field_cfg: &F::Config,
) -> Result<F, ProductionShaError<F>>
where
    F: PrimeField,
{
    let Some(shifted) = row.checked_add(shift) else {
        return Ok(F::zero_with_cfg(field_cfg));
    };
    if shifted >= SHA_ROW_COUNT {
        return Ok(F::zero_with_cfg(field_cfg));
    }
    public
        .columns
        .get(col.index())
        .and_then(|column| column.evaluations.get(shifted))
        .cloned()
        .ok_or(ProductionShaError::LengthMismatch {
            label: "SHA public scalar row",
            got: shifted,
            expected: SHA_ROW_COUNT,
        })
}

fn public_word_col_index(col: ShaPublicCol) -> Option<usize> {
    match col {
        ShaPublicCol::PAIn => Some(0),
        ShaPublicCol::PEIn => Some(1),
        ShaPublicCol::PAOut => Some(2),
        ShaPublicCol::PEOut => Some(3),
        ShaPublicCol::Message => Some(4),
        _ => None,
    }
}

fn public_word_bit_at_row<F>(
    public: &ProjectedPublic<F>,
    col: ShaPublicCol,
    row: usize,
    bit: usize,
    field_cfg: &F::Config,
) -> Result<F, ProductionShaError<F>>
where
    F: PrimeField,
{
    if bit >= SHA_WORD_BITS {
        return Err(ProductionShaError::LengthMismatch {
            label: "SHA public word bit index",
            got: bit,
            expected: SHA_WORD_BITS,
        });
    }
    if row >= SHA_ROW_COUNT {
        return Ok(F::zero_with_cfg(field_cfg));
    }
    let col_idx = public_word_col_index(col).ok_or(ProductionShaError::NonCanonicalProofObject(
        "SHA public column is not a public word",
    ))?;
    let bit_slices =
        public
            .bit_slices
            .as_ref()
            .ok_or(ProductionShaError::NonCanonicalProofObject(
                "production SHA public word columns are required",
            ))?;
    let table_idx = bit_slice_index(col_idx, bit, SHA_WORD_BITS);
    bit_slices
        .get(table_idx)
        .and_then(|column| column.evaluations.get(row))
        .cloned()
        .ok_or(ProductionShaError::LengthMismatch {
            label: "SHA public word bit row",
            got: row,
            expected: SHA_ROW_COUNT,
        })
}

fn production_sha_pow_two<F>(exp: usize, field_cfg: &F::Config) -> F
where
    F: PrimeField,
{
    let two = F::one_with_cfg(field_cfg) + F::one_with_cfg(field_cfg);
    let mut out = F::one_with_cfg(field_cfg);
    for _ in 0..exp {
        out *= &two;
    }
    out
}

fn row_expr_mle_from_table_shift<F>(
    label: &'static str,
    table: &MleTable<F>,
    col_idx: usize,
    shift: usize,
    zero: &F,
    zero_inner: &F::Inner,
) -> Result<DenseMultilinearExtension<F::Inner>, ProductionShaError<F>>
where
    F: InnerTransparentField,
{
    let column = table
        .get(col_idx)
        .ok_or(ProductionShaError::LengthMismatch {
            label,
            got: col_idx,
            expected: table.len(),
        })?;
    if column.num_vars != SHA_ROW_VARS || column.evaluations.len() != SHA_ROW_COUNT {
        return Err(ProductionShaError::LengthMismatch {
            label,
            got: column.evaluations.len(),
            expected: SHA_ROW_COUNT,
        });
    }

    let mut evaluations = Vec::with_capacity(SHA_ROW_COUNT);
    if shift < SHA_ROW_COUNT {
        evaluations.extend(
            column.evaluations[shift..]
                .iter()
                .map(|value| value.inner().clone()),
        );
    }
    evaluations.resize(SHA_ROW_COUNT, zero_inner.clone());
    debug_assert_eq!(evaluations.len(), SHA_ROW_COUNT);
    let _ = zero;
    Ok(DenseMultilinearExtension::from_evaluations_vec(
        SHA_ROW_VARS,
        evaluations,
        zero_inner.clone(),
    ))
}

#[allow(clippy::too_many_arguments)]
fn build_production_sha_row_expression_sumcheck_group<F>(
    trace: &ProjectedTrace<F>,
    public: &ProjectedPublic<F>,
    r_ic: &[F; SHA_ROW_VARS],
    a: &F,
    lambda: &F,
    rho: &F,
    xi: &F,
    booleanity_sources: &[ShaBooleanitySource],
    field_cfg: &F::Config,
) -> Result<MultiDegreeSumcheckGroup<F>, ProductionShaError<F>>
where
    F: InnerTransparentField + DelayedFieldProductSum + Send + Sync + 'static,
    F::Inner: Zero,
{
    let row_weights = build_eq_x_r_vec(r_ic, field_cfg)?;
    build_production_sha_row_expression_sumcheck_group_with_row_weights(
        trace,
        public,
        &row_weights,
        a,
        lambda,
        rho,
        xi,
        booleanity_sources,
        field_cfg,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_production_sha_row_expression_sumcheck_group_with_row_weights<F>(
    trace: &ProjectedTrace<F>,
    public: &ProjectedPublic<F>,
    row_weights: &[F],
    a: &F,
    lambda: &F,
    rho: &F,
    xi: &F,
    booleanity_sources: &[ShaBooleanitySource],
    field_cfg: &F::Config,
) -> Result<MultiDegreeSumcheckGroup<F>, ProductionShaError<F>>
where
    F: InnerTransparentField + DelayedFieldProductSum + Send + Sync + 'static,
    F::Inner: Zero,
{
    let a_powers = build_sha_residual_eval_powers(a, field_cfg);
    let lambda_powers = build_sha_lambda_powers(lambda, field_cfg);
    let booleanity_weights = build_booleanity_weights(rho, xi, booleanity_sources.len(), field_cfg);
    build_production_sha_row_expression_sumcheck_group_with_vectors(
        trace,
        public,
        row_weights,
        &a_powers,
        &lambda_powers,
        &booleanity_weights,
        booleanity_sources,
        field_cfg,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_production_sha_row_expression_sumcheck_group_with_vectors<F>(
    trace: &ProjectedTrace<F>,
    public: &ProjectedPublic<F>,
    row_weights: &[F],
    a_powers: &[F],
    lambda_powers: &[F],
    booleanity_weights: &[F],
    booleanity_sources: &[ShaBooleanitySource],
    field_cfg: &F::Config,
) -> Result<MultiDegreeSumcheckGroup<F>, ProductionShaError<F>>
where
    F: InnerTransparentField + DelayedFieldProductSum + Send + Sync + 'static,
    F::Inner: Zero,
{
    if row_weights.len() != SHA_ROW_COUNT {
        return Err(ProductionShaError::LengthMismatch {
            label: "row weights",
            got: row_weights.len(),
            expected: SHA_ROW_COUNT,
        });
    }
    if a_powers.len() < SHA_IDEAL_EVAL_POWER_COUNT {
        return Err(ProductionShaError::LengthMismatch {
            label: "a powers",
            got: a_powers.len(),
            expected: SHA_IDEAL_EVAL_POWER_COUNT,
        });
    }
    if lambda_powers.len() != NUM_SHA_RESIDUAL_FAMILIES {
        return Err(ProductionShaError::LengthMismatch {
            label: "lambda powers",
            got: lambda_powers.len(),
            expected: NUM_SHA_RESIDUAL_FAMILIES,
        });
    }
    if booleanity_weights.len() != booleanity_sources.len() {
        return Err(ProductionShaError::LengthMismatch {
            label: "booleanity weights",
            got: booleanity_weights.len(),
            expected: booleanity_sources.len(),
        });
    }
    let zero = F::zero_with_cfg(field_cfg);
    let zero_inner = zero.inner().clone();
    let layout = RowExpressionLayout::new();
    let mut mles = Vec::with_capacity(
        1 + layout.word_sources.len() * SHA_WORD_BITS
            + layout.int_sources.len()
            + layout.public_scalar_sources.len()
            + layout.public_word_sources.len() * SHA_WORD_BITS,
    );

    mles.push(DenseMultilinearExtension::from_evaluations_vec(
        SHA_ROW_VARS,
        row_weights
            .iter()
            .map(|value| value.inner().clone())
            .collect(),
        zero_inner.clone(),
    ));

    for &(col, shift) in &layout.word_sources {
        for bit in 0..SHA_WORD_BITS {
            mles.push(row_expr_mle_from_table_shift(
                "SHA trace word bit",
                &trace.bit_slices,
                bit_slice_index(col.index(), bit, SHA_WORD_BITS),
                shift,
                &zero,
                &zero_inner,
            )?);
        }
    }

    for &col in &layout.int_sources {
        mles.push(row_expr_mle_from_table_shift(
            "SHA trace int",
            &trace.int_columns,
            col.index(),
            0,
            &zero,
            &zero_inner,
        )?);
    }

    for &(col, shift) in &layout.public_scalar_sources {
        mles.push(row_expr_mle_from_table_shift(
            "SHA public scalar",
            &public.columns,
            col.index(),
            shift,
            &zero,
            &zero_inner,
        )?);
    }

    for &col in &layout.public_word_sources {
        let bit_slices =
            public
                .bit_slices
                .as_ref()
                .ok_or(ProductionShaError::NonCanonicalProofObject(
                    "production SHA public word columns are required",
                ))?;
        let col_idx = public_word_col_index(col).ok_or(
            ProductionShaError::NonCanonicalProofObject("SHA public column is not a public word"),
        )?;
        for bit in 0..SHA_WORD_BITS {
            mles.push(row_expr_mle_from_table_shift(
                "SHA public word bit",
                bit_slices,
                bit_slice_index(col_idx, bit, SHA_WORD_BITS),
                0,
                &zero,
                &zero_inner,
            )?);
        }
    }

    let offsets = layout.offsets();
    let word_weights = a_powers[..SHA_WORD_BITS].to_vec();
    let rot_weights = |shift: usize| {
        (0..SHA_WORD_BITS)
            .map(|bit| a_powers[(bit + shift) % SHA_WORD_BITS].clone())
            .collect::<Vec<_>>()
    };
    let shift_weights = |shift: usize| {
        (0..SHA_WORD_BITS)
            .map(|bit| {
                if bit >= shift {
                    a_powers[bit - shift].clone()
                } else {
                    F::zero_with_cfg(field_cfg)
                }
            })
            .collect::<Vec<_>>()
    };
    let rot25_weights = rot_weights(25);
    let rot14_weights = rot_weights(14);
    let rot15_weights = rot_weights(15);
    let rot13_weights = rot_weights(13);
    let shift3_weights = shift_weights(3);
    let shift10_weights = shift_weights(10);
    let shift0_weights = shift_weights(0);
    let shift2_weights = shift_weights(2);
    let shift5_weights = shift_weights(5);
    let shift8_weights = shift_weights(8);
    let shift9_weights = shift_weights(9);
    let lambda_powers = lambda_powers.to_vec();
    let booleanity_weights = booleanity_weights.to_vec();
    let booleanity_sources = booleanity_sources.to_vec();
    let one = F::one_with_cfg(field_cfg);
    let two = one.clone() + &one;
    let rho_sig0 = a_powers[10].clone() + &a_powers[19] + &a_powers[30];
    let rho_sig1 = a_powers[7].clone() + &a_powers[21] + &a_powers[26];
    let low_mu_coeff = production_sha_pow_two(32, field_cfg);
    let high_mu_w_coeff = production_sha_pow_two(34, field_cfg);
    let high_mu_3_bit_coeff = production_sha_pow_two(35, field_cfg);
    let high_mu_1_bit_coeff = production_sha_pow_two(33, field_cfg);

    Ok(MultiDegreeSumcheckGroup::new(
        3,
        mles,
        Box::new(move |values: &[F]| {
            let zero = zero.clone();
            let dot = |lhs: &[F], rhs: &[F]| {
                FieldFieldInnerProduct::inner_product::<UNCHECKED>(lhs, rhs, zero.clone())
                    .expect("row expression dot product lengths match")
            };
            let word_source_idx = |col: ShaWordCol, shift: usize| {
                let idx = offsets.word[col.index()][shift];
                debug_assert_ne!(idx, ROW_EXPR_MISSING_SOURCE);
                idx
            };
            let word_bits = |col: ShaWordCol, shift: usize| {
                let source_idx = word_source_idx(col, shift);
                let base = layout.word_offset + source_idx * SHA_WORD_BITS;
                &values[base..base + SHA_WORD_BITS]
            };
            let word_eval =
                |col: ShaWordCol, shift: usize| dot(word_bits(col, shift), &word_weights);
            let word_eval_with =
                |col: ShaWordCol, shift: usize, weights: &[F]| dot(word_bits(col, shift), weights);
            let word_bit =
                |col: ShaWordCol, shift: usize, bit: usize| word_bits(col, shift)[bit].clone();
            let int_value = |col: ShaIntCol| {
                let idx = offsets.int[col.index()];
                debug_assert_ne!(idx, ROW_EXPR_MISSING_SOURCE);
                values[layout.int_offset + idx].clone()
            };
            let public_scalar = |col: ShaPublicCol, shift: usize| {
                let idx = offsets.public_scalar[col.index()][shift];
                debug_assert_ne!(idx, ROW_EXPR_MISSING_SOURCE);
                values[layout.public_scalar_offset + idx].clone()
            };
            let public_word_or_const_eval = |col: ShaPublicCol| {
                let idx = offsets.public_word[col.index()];
                if idx == ROW_EXPR_MISSING_SOURCE {
                    public_scalar(col, 0)
                } else {
                    let base = layout.public_word_offset + idx * SHA_WORD_BITS;
                    dot(&values[base..base + SHA_WORD_BITS], &word_weights)
                }
            };

            let a_word = word_eval(ShaWordCol::A, 0);
            let e_word = word_eval(ShaWordCol::E, 0);
            let sigma0 = word_eval(ShaWordCol::Sigma0, 0);
            let sigma1 = word_eval(ShaWordCol::Sigma1, 0);
            let w = word_eval(ShaWordCol::W, 0);
            let small_sigma0 = word_eval(ShaWordCol::SmallSigma0, 0);
            let small_sigma1 = word_eval(ShaWordCol::SmallSigma1, 0);
            let ov_sigma0 = word_eval(ShaWordCol::OvSigma0, 0);
            let ov_sigma1 = word_eval(ShaWordCol::OvSigma1, 0);
            let ov_small_sigma0 = word_eval(ShaWordCol::OvSmallSigma0, 0);
            let ov_small_sigma1 = word_eval(ShaWordCol::OvSmallSigma1, 0);

            let mu = |low_weights: &[F], high_weights: &[F], high_coeff: &F| {
                word_eval_with(ShaWordCol::MuPacked, 0, low_weights) * &low_mu_coeff
                    - word_eval_with(ShaWordCol::MuPacked, 0, high_weights) * high_coeff
            };
            let mu_w = mu(&shift0_weights, &shift2_weights, &high_mu_w_coeff);
            let mu_a = mu(&shift2_weights, &shift5_weights, &high_mu_3_bit_coeff);
            let mu_e = mu(&shift5_weights, &shift8_weights, &high_mu_3_bit_coeff);
            let mu_ff_a = mu(&shift8_weights, &shift9_weights, &high_mu_1_bit_coeff);
            let mu_ff_e = mu(&shift9_weights, &shift10_weights, &high_mu_1_bit_coeff);

            let r0 = a_word.clone() * &rho_sig0 - &sigma0 - two.clone() * &ov_sigma0;
            let r1 = e_word.clone() * &rho_sig1 - &sigma1 - two.clone() * &ov_sigma1;
            let r2 = word_eval_with(ShaWordCol::W, 0, &rot25_weights)
                + word_eval_with(ShaWordCol::W, 0, &rot14_weights)
                + word_eval_with(ShaWordCol::W, 0, &shift3_weights)
                - &small_sigma0
                - two.clone() * &ov_small_sigma0;
            let r3 = word_eval_with(ShaWordCol::W, 0, &rot15_weights)
                + word_eval_with(ShaWordCol::W, 0, &rot13_weights)
                + word_eval_with(ShaWordCol::W, 0, &shift10_weights)
                - &small_sigma1
                - two.clone() * &ov_small_sigma1;
            let r4 = word_eval(ShaWordCol::W, 16)
                - &w
                - word_eval(ShaWordCol::SmallSigma0, 1)
                - word_eval(ShaWordCol::W, 9)
                - word_eval(ShaWordCol::SmallSigma1, 14)
                + &mu_w
                + int_value(ShaIntCol::CompSchedule);
            let r5 = word_eval(ShaWordCol::A, 4)
                - &e_word
                - word_eval(ShaWordCol::Sigma1, 3)
                - word_eval(ShaWordCol::Uef, 3)
                - word_eval(ShaWordCol::UNegEg, 3)
                - public_scalar(ShaPublicCol::K, 3)
                - &w
                - word_eval(ShaWordCol::Sigma0, 3)
                - word_eval(ShaWordCol::Maj, 3)
                + &mu_a
                + int_value(ShaIntCol::CompUpdateA);
            let r6 = word_eval(ShaWordCol::E, 4)
                - &a_word
                - &e_word
                - word_eval(ShaWordCol::Sigma1, 3)
                - word_eval(ShaWordCol::Uef, 3)
                - word_eval(ShaWordCol::UNegEg, 3)
                - public_scalar(ShaPublicCol::K, 3)
                - &w
                + &mu_e
                + int_value(ShaIntCol::CompUpdateE);

            let s_init = public_scalar(ShaPublicCol::SInit, 0);
            let s_msg = public_scalar(ShaPublicCol::SMsg, 0);
            let s_sched = public_scalar(ShaPublicCol::SSched, 0);
            let s_upd = public_scalar(ShaPublicCol::SUpd, 0);
            let s_ff = public_scalar(ShaPublicCol::SFf, 0);
            let s_out = public_scalar(ShaPublicCol::SOut, 0);

            let r7 = (a_word.clone() - public_word_or_const_eval(ShaPublicCol::PAIn)) * &s_init
                + (a_word.clone() - public_word_or_const_eval(ShaPublicCol::PAOut)) * &s_out;
            let r8 = (e_word.clone() - public_word_or_const_eval(ShaPublicCol::PEIn)) * &s_init
                + (e_word.clone() - public_word_or_const_eval(ShaPublicCol::PEOut)) * &s_out;
            let r9 = word_eval(ShaWordCol::A, 4) - &a_word - public_scalar(ShaPublicCol::PAIn, 0)
                + &mu_ff_a
                + int_value(ShaIntCol::CompFeedForwardA);
            let r10 = word_eval(ShaWordCol::E, 4) - &e_word - public_scalar(ShaPublicCol::PEIn, 0)
                + &mu_ff_e
                + int_value(ShaIntCol::CompFeedForwardE);
            let r11 = (w - public_word_or_const_eval(ShaPublicCol::Message)) * &s_msg;
            let r12 = int_value(ShaIntCol::CompSchedule) * &s_sched;
            let r13 = int_value(ShaIntCol::CompUpdateA) * &s_upd;
            let r14 = int_value(ShaIntCol::CompUpdateE) * &s_upd;
            let r15 = int_value(ShaIntCol::CompFeedForwardA) * &s_ff;
            let r16 = int_value(ShaIntCol::CompFeedForwardE) * &s_ff;
            let r17 = word_eval_with(ShaWordCol::MuPacked, 0, &shift10_weights);
            let residuals = [
                r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17,
            ];
            let linear = FieldFieldInnerProduct::inner_product::<UNCHECKED>(
                &residuals,
                &lambda_powers,
                zero.clone(),
            )
            .expect("row expression residual dot product lengths match");

            let mut bool_sum = zero.clone();
            for (source, weight) in booleanity_sources.iter().zip(booleanity_weights.iter()) {
                let d = match *source {
                    ShaBooleanitySource::WordBit { col, bit } => word_bit(col, 0, bit),
                    ShaBooleanitySource::VirtualCh1 { bit: bit_idx } => {
                        word_bit(ShaWordCol::E, 2, bit_idx) + &word_bit(ShaWordCol::E, 1, bit_idx)
                            - two.clone() * word_bit(ShaWordCol::Uef, 2, bit_idx)
                    }
                    ShaBooleanitySource::VirtualCh2 { bit: bit_idx } => {
                        word_bit(ShaWordCol::E, 2, bit_idx) - &word_bit(ShaWordCol::E, 0, bit_idx)
                            + two.clone() * word_bit(ShaWordCol::UNegEg, 2, bit_idx)
                            + two.clone() * word_bit(ShaWordCol::Ch2Comp, 0, bit_idx)
                    }
                    ShaBooleanitySource::VirtualMaj { bit: bit_idx } => {
                        word_bit(ShaWordCol::A, 0, bit_idx)
                            + &word_bit(ShaWordCol::A, 1, bit_idx)
                            + &word_bit(ShaWordCol::A, 2, bit_idx)
                            - two.clone() * word_bit(ShaWordCol::Maj, 2, bit_idx)
                            - two.clone() * word_bit(ShaWordCol::MajComp, 0, bit_idx)
                    }
                };
                bool_sum += weight.clone() * (d.clone() * (d - one.clone()));
            }

            values[0].clone() * (linear + bool_sum)
        }),
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn prove_expression_folded_row_sumcheck<F>(
    transcript: &mut impl Transcript,
    trace: &ProjectedTrace<F>,
    public: &ProjectedPublic<F>,
    r_ic: &[F; SHA_ROW_VARS],
    a: &F,
    lambda: &F,
    rho: &F,
    xi: &F,
    booleanity_sources: &[ShaBooleanitySource],
    post_sumfold_claim: &F,
    field_cfg: &F::Config,
) -> Result<MultiDegreeSumcheckProof<F>, ProductionShaError<F>>
where
    F: InnerTransparentField
        + DelayedFieldProductSum
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: Transcribable + Zero,
    F::Modulus: Transcribable,
{
    let claimed = zinc_piop::neutron_nova::expression_folded_row_sum(
        trace,
        public,
        r_ic,
        a,
        lambda,
        rho,
        xi,
        booleanity_sources,
        field_cfg,
    )?;
    verify_folded_row_sumcheck_claim(&claimed, post_sumfold_claim)?;
    let group = build_production_sha_row_expression_sumcheck_group(
        trace,
        public,
        r_ic,
        a,
        lambda,
        rho,
        xi,
        booleanity_sources,
        field_cfg,
    )?;
    let (proof, _) =
        MultiDegreeSumcheck::prove_as_subprotocol(transcript, vec![group], SHA_ROW_VARS, field_cfg);
    Ok(proof)
}

#[allow(clippy::too_many_arguments)]
pub fn prove_expression_folded_row_sumcheck_with_output<F>(
    transcript: &mut impl Transcript,
    trace: &ProjectedTrace<F>,
    public: &ProjectedPublic<F>,
    r_ic: &[F; SHA_ROW_VARS],
    a: &F,
    lambda: &F,
    rho: &F,
    xi: &F,
    booleanity_sources: &[ShaBooleanitySource],
    post_sumfold_claim: &F,
    field_cfg: &F::Config,
) -> Result<(MultiDegreeSumcheckProof<F>, FoldedRowSumcheckOutput<F>), ProductionShaError<F>>
where
    F: InnerTransparentField
        + DelayedFieldProductSum
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: Transcribable + Zero,
    F::Modulus: Transcribable,
{
    let r_ic_eq_weights = build_eq_x_r_vec(r_ic, field_cfg)?;
    prove_expression_folded_row_sumcheck_with_output_and_weights(
        transcript,
        trace,
        public,
        r_ic,
        &r_ic_eq_weights,
        a,
        lambda,
        rho,
        xi,
        booleanity_sources,
        post_sumfold_claim,
        field_cfg,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn prove_expression_folded_row_sumcheck_with_output_and_weights<F>(
    transcript: &mut impl Transcript,
    trace: &ProjectedTrace<F>,
    public: &ProjectedPublic<F>,
    r_ic: &[F; SHA_ROW_VARS],
    r_ic_eq_weights: &[F],
    a: &F,
    lambda: &F,
    rho: &F,
    xi: &F,
    booleanity_sources: &[ShaBooleanitySource],
    post_sumfold_claim: &F,
    field_cfg: &F::Config,
) -> Result<(MultiDegreeSumcheckProof<F>, FoldedRowSumcheckOutput<F>), ProductionShaError<F>>
where
    F: InnerTransparentField
        + DelayedFieldProductSum
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: Transcribable + Zero,
    F::Modulus: Transcribable,
{
    let a_powers = build_sha_residual_eval_powers(a, field_cfg);
    let lambda_powers = build_sha_lambda_powers(lambda, field_cfg);
    let booleanity_weights = build_booleanity_weights(rho, xi, booleanity_sources.len(), field_cfg);
    let claimed = expression_folded_row_sum_with_row_weights(
        trace,
        public,
        r_ic_eq_weights,
        a,
        lambda,
        rho,
        xi,
        booleanity_sources,
        field_cfg,
    )?;
    verify_folded_row_sumcheck_claim(&claimed, post_sumfold_claim)?;
    prove_expression_folded_row_sumcheck_with_output_and_vectors(
        transcript,
        trace,
        public,
        r_ic,
        r_ic_eq_weights,
        &a_powers,
        &lambda_powers,
        &booleanity_weights,
        booleanity_sources,
        field_cfg,
    )
}

fn row_sumcheck_terminal_from_proof<F>(
    proof: &MultiDegreeSumcheckProof<F>,
    challenges: &[F],
) -> Result<F, ProductionShaError<F>>
where
    F: FromPrimitiveWithConfig,
{
    if proof.group_messages().len() != 1 {
        return Err(ProductionShaError::UnexpectedSumcheckGroupCount {
            label: "row sumcheck terminal",
            got: proof.group_messages().len(),
        });
    }
    if proof.claimed_sums().len() != 1 {
        return Err(ProductionShaError::UnexpectedSumcheckGroupCount {
            label: "row sumcheck terminal claimed sums",
            got: proof.claimed_sums().len(),
        });
    }
    let degree = proof
        .degrees()
        .first()
        .ok_or(ProductionShaError::LengthMismatch {
            label: "row sumcheck terminal degrees",
            got: 0,
            expected: 1,
        })?;
    let messages = proof
        .group_messages()
        .first()
        .expect("checked row sumcheck group count");
    if messages.len() != challenges.len() {
        return Err(ProductionShaError::LengthMismatch {
            label: "row sumcheck terminal rounds",
            got: messages.len(),
            expected: challenges.len(),
        });
    }

    let mut expected = proof.claimed_sums()[0].clone();
    for (message, challenge) in messages.iter().zip(challenges) {
        let tail = &message.0.tail_evaluations;
        if tail.len() != *degree {
            return Err(ProductionShaError::LengthMismatch {
                label: "row sumcheck terminal degree",
                got: tail.len(),
                expected: *degree,
            });
        }
        let constant = match tail.first() {
            Some(p1) => expected.clone() - p1,
            None => expected.clone(),
        };
        let mut evaluations = Vec::with_capacity(tail.len() + 1);
        evaluations.push(constant);
        evaluations.extend_from_slice(tail);
        expected = NatEvaluatedPoly::new(evaluations).evaluate_at_point(challenge)?;
    }

    Ok(expected)
}

#[allow(clippy::too_many_arguments)]
pub fn prove_expression_folded_row_sumcheck_with_output_and_vectors<F>(
    transcript: &mut impl Transcript,
    trace: &ProjectedTrace<F>,
    public: &ProjectedPublic<F>,
    r_ic: &[F; SHA_ROW_VARS],
    r_ic_eq_weights: &[F],
    a_powers: &[F],
    lambda_powers: &[F],
    booleanity_weights: &[F],
    booleanity_sources: &[ShaBooleanitySource],
    field_cfg: &F::Config,
) -> Result<(MultiDegreeSumcheckProof<F>, FoldedRowSumcheckOutput<F>), ProductionShaError<F>>
where
    F: InnerTransparentField
        + DelayedFieldProductSum
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: Transcribable + Zero,
    F::Modulus: Transcribable,
{
    #[cfg(not(debug_assertions))]
    let _ = r_ic;

    let group = tracing::info_span!(
        target: "zinc_protocol::production_sha",
        "row_sumcheck_build_group",
        side = "prove",
        phase = "row_sumcheck_build_group",
    )
    .in_scope(|| {
        build_production_sha_row_expression_sumcheck_group_with_vectors(
            trace,
            public,
            r_ic_eq_weights,
            a_powers,
            lambda_powers,
            booleanity_weights,
            booleanity_sources,
            field_cfg,
        )
    })?;
    let (proof, states) = tracing::info_span!(
        target: "zinc_protocol::production_sha",
        "row_sumcheck_prove_core",
        side = "prove",
        phase = "row_sumcheck_prove_core",
    )
    .in_scope(|| {
        MultiDegreeSumcheck::prove_as_subprotocol(transcript, vec![group], SHA_ROW_VARS, field_cfg)
    });
    let r_star = states
        .first()
        .ok_or(ProductionShaError::LengthMismatch {
            label: "folded row states",
            got: 0,
            expected: 1,
        })?
        .randomness
        .clone();
    let r_star_eq_weights = build_eq_x_r_vec(&r_star, field_cfg)?;
    let a = a_powers.get(1).ok_or(ProductionShaError::LengthMismatch {
        label: "a powers",
        got: a_powers.len(),
        expected: 2,
    })?;
    let endpoint_evals = tracing::info_span!(
        target: "zinc_protocol::production_sha",
        "row_sumcheck_endpoint_evals",
        side = "prove",
        phase = "row_sumcheck_endpoint_evals",
    )
    .in_scope(|| {
        build_sha_endpoint_evals_from_trace_with_row_weights(
            trace,
            &r_star_eq_weights,
            a,
            field_cfg,
        )
    })?;
    let terminal_value = tracing::info_span!(
        target: "zinc_protocol::production_sha",
        "row_sumcheck_terminal",
        side = "prove",
        phase = "row_sumcheck_terminal",
    )
    .in_scope(|| row_sumcheck_terminal_from_proof(&proof, &r_star))?;
    #[cfg(debug_assertions)]
    {
        let reconstructed_terminal = tracing::info_span!(
            target: "zinc_protocol::production_sha",
            "row_sumcheck_terminal_debug",
            side = "prove",
            phase = "row_sumcheck_terminal_debug",
        )
        .in_scope(|| {
            reconstruct_folded_row_terminal_from_endpoints_with_vectors(
                &endpoint_evals,
                public,
                r_ic,
                &r_star,
                &r_star_eq_weights,
                a_powers,
                lambda_powers,
                booleanity_weights,
                booleanity_sources,
                field_cfg,
            )
        })?;
        if terminal_value != reconstructed_terminal {
            return Err(ProductionShaError::RowSumcheckTerminalMismatch);
        }
    }
    Ok((
        proof,
        FoldedRowSumcheckOutput {
            r_star,
            r_star_eq_weights,
            terminal_value,
            endpoint_evals: Some(endpoint_evals),
        },
    ))
}

pub fn verify_folded_row_sumcheck<F>(
    transcript: &mut impl Transcript,
    proof: &MultiDegreeSumcheckProof<F>,
    post_sumfold_claim: &F,
    field_cfg: &F::Config,
) -> Result<FoldedRowSumcheckOutput<F>, ProductionShaError<F>>
where
    F: InnerTransparentField
        + DelayedFieldProductSum
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: Transcribable + Zero,
    F::Modulus: Transcribable,
{
    require_single_sumcheck_group(proof, "folded row sumcheck")?;
    for &degree in proof.degrees() {
        if degree > 3 {
            return Err(ProductionShaError::RowSumcheckDegreeTooHigh { degree });
        }
    }
    let Some(claimed_sum) = proof.claimed_sums().first() else {
        return Err(ProductionShaError::LengthMismatch {
            label: "folded row claimed sums",
            got: 0,
            expected: 1,
        });
    };
    verify_folded_row_sumcheck_claim(claimed_sum, post_sumfold_claim)?;
    let subclaims =
        MultiDegreeSumcheck::verify_as_subprotocol(transcript, SHA_ROW_VARS, proof, field_cfg)?;
    let r_star = subclaims.point().to_vec();
    let r_star_eq_weights = build_eq_x_r_vec(&r_star, field_cfg)?;
    Ok(FoldedRowSumcheckOutput {
        r_star,
        r_star_eq_weights,
        terminal_value: subclaims.expected_evaluations()[0].clone(),
        endpoint_evals: None,
    })
}

pub fn verify_folded_row_terminal_value<F>(
    output: &FoldedRowSumcheckOutput<F>,
    reconstructed_terminal: &F,
) -> Result<(), ProductionShaError<F>>
where
    F: PrimeField,
{
    if &output.terminal_value != reconstructed_terminal {
        return Err(ProductionShaError::RowSumcheckTerminalMismatch);
    }
    Ok(())
}

pub fn production_sha_endpoint_word_sources() -> Vec<(ShaWordCol, usize)> {
    let mut sources = Vec::new();
    let mut push = |col, shift| {
        if !sources.contains(&(col, shift)) {
            sources.push((col, shift));
        }
    };

    for col in ShaWordCol::ALL {
        push(col, 0);
    }
    for (col, shifts) in [
        (ShaWordCol::A, &[1usize, 2, 4][..]),
        (ShaWordCol::E, &[1usize, 2, 4][..]),
        (ShaWordCol::Sigma0, &[3usize][..]),
        (ShaWordCol::Sigma1, &[3usize][..]),
        (ShaWordCol::W, &[3usize, 9, 16][..]),
        (ShaWordCol::SmallSigma0, &[1usize][..]),
        (ShaWordCol::SmallSigma1, &[14usize][..]),
        (ShaWordCol::Uef, &[2usize, 3][..]),
        (ShaWordCol::UNegEg, &[2usize, 3][..]),
        (ShaWordCol::Maj, &[2usize, 3][..]),
    ] {
        for &shift in shifts {
            push(col, shift);
        }
    }
    sources
}

pub fn production_sha_endpoint_int_sources() -> Vec<ShaIntCol> {
    ShaIntCol::ALL.to_vec()
}

pub fn production_sha_multipoint_layout() -> ShaMultipointLayout {
    let mut sources = Vec::new();
    let mut push_source = |source| {
        if !sources.contains(&source) {
            sources.push(source);
        }
    };

    for (col, _) in production_sha_endpoint_word_sources() {
        for bit in 0..32 {
            push_source(ShaMpSource::WordBit { col, bit });
        }
    }
    for col in production_sha_endpoint_int_sources() {
        push_source(ShaMpSource::Int { col });
    }
    for col in ShaPublicCol::ALL {
        push_source(ShaMpSource::Public { col });
    }

    let mut shifts = Vec::new();
    let mut push_shift = |shift| {
        if !shifts.contains(&shift) {
            shifts.push(shift);
        }
    };
    for (col, shift) in production_sha_endpoint_word_sources() {
        if shift == 0 {
            continue;
        }
        for bit in 0..32 {
            push_shift(ShaMpShiftSource::WordBit { col, bit, shift });
        }
    }
    push_shift(ShaMpShiftSource::Public {
        col: ShaPublicCol::K,
        shift: 3,
    });

    ShaMultipointLayout { sources, shifts }
}

pub fn build_sha_endpoint_evals_from_trace<F>(
    trace: &ProjectedTrace<F>,
    r_star: &[F],
    a: &F,
    field_cfg: &F::Config,
) -> Result<ShaEndpointEvals<F>, ProductionShaError<F>>
where
    F: DelayedFieldProductSum,
{
    let row_weights = build_eq_x_r_vec(r_star, field_cfg)?;
    build_sha_endpoint_evals_from_trace_with_row_weights(trace, &row_weights, a, field_cfg)
}

pub fn build_sha_endpoint_evals_from_trace_with_row_weights<F>(
    trace: &ProjectedTrace<F>,
    row_weights: &[F],
    a: &F,
    field_cfg: &F::Config,
) -> Result<ShaEndpointEvals<F>, ProductionShaError<F>>
where
    F: DelayedFieldProductSum,
{
    if row_weights.len() != SHA_ROW_COUNT {
        return Err(ProductionShaError::LengthMismatch {
            label: "row weights",
            got: row_weights.len(),
            expected: SHA_ROW_COUNT,
        });
    }
    #[cfg(debug_assertions)]
    validate_projected_trace(trace)?;
    let mut sources = Vec::new();
    for (col, shift) in production_sha_endpoint_word_sources() {
        let bits =
            sha_endpoint_word_bits_with_row_weights(trace, col, shift, row_weights, field_cfg)?;
        sources.push(ShaSourceEndpointEval {
            col,
            shift,
            scalarized: scalarize_sha_endpoint_bits(&bits, a, field_cfg),
            bits,
        });
    }
    let mut int_sources = Vec::new();
    for col in production_sha_endpoint_int_sources() {
        int_sources.push(ShaIntEndpointEval {
            col,
            scalar: sha_endpoint_int_with_row_weights(trace, col, row_weights, field_cfg)?,
        });
    }
    Ok(ShaEndpointEvals {
        sources,
        int_sources,
    })
}

fn sha_endpoint_word_bits_with_row_weights<F>(
    trace: &ProjectedTrace<F>,
    col: ShaWordCol,
    shift: usize,
    row_weights: &[F],
    field_cfg: &F::Config,
) -> Result<[F; SHA_WORD_BITS], ProductionShaError<F>>
where
    F: DelayedFieldProductSum,
{
    let active_len = SHA_ROW_COUNT.saturating_sub(shift);
    if active_len == 0 {
        return Ok(std::array::from_fn(|_| F::zero_with_cfg(field_cfg)));
    }
    let weights = &row_weights[..active_len];
    let values_start = shift;
    let values_end = shift + active_len;
    let bits = cfg_into_iter!(0..SHA_WORD_BITS)
        .map(|bit| {
            let table_idx = bit_slice_index(col.index(), bit, SHA_WORD_BITS);
            let column =
                trace
                    .bit_slices
                    .get(table_idx)
                    .ok_or(ProductionShaError::LengthMismatch {
                        label: "SHA endpoint bit column",
                        got: table_idx,
                        expected: trace.bit_slices.len(),
                    })?;
            FieldFieldInnerProduct::inner_product::<UNCHECKED>(
                weights,
                &column.evaluations[values_start..values_end],
                F::zero_with_cfg(field_cfg),
            )
            .map_err(|_| {
                ProductionShaError::NonCanonicalProofObject(
                    "SHA endpoint bit row-weight dot product failed",
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    bits.try_into()
        .map_err(|bits: Vec<F>| ProductionShaError::LengthMismatch {
            label: "SHA endpoint word bits",
            got: bits.len(),
            expected: SHA_WORD_BITS,
        })
}

fn sha_endpoint_int_with_row_weights<F>(
    trace: &ProjectedTrace<F>,
    col: ShaIntCol,
    row_weights: &[F],
    field_cfg: &F::Config,
) -> Result<F, ProductionShaError<F>>
where
    F: DelayedFieldProductSum,
{
    let column = trace
        .int_columns
        .get(col.index())
        .ok_or(ProductionShaError::LengthMismatch {
            label: "SHA endpoint int column",
            got: col.index(),
            expected: trace.int_columns.len(),
        })?;
    FieldFieldInnerProduct::inner_product::<UNCHECKED>(
        row_weights,
        &column.evaluations,
        F::zero_with_cfg(field_cfg),
    )
    .map_err(|_| {
        ProductionShaError::NonCanonicalProofObject(
            "SHA endpoint int row-weight dot product failed",
        )
    })
}

pub fn validate_sha_endpoint_layout<F>(
    endpoint_evals: &ShaEndpointEvals<F>,
) -> Result<(), ProductionShaError<F>>
where
    F: PrimeField,
{
    let word_sources = production_sha_endpoint_word_sources();
    if endpoint_evals.sources.len() != word_sources.len() {
        return Err(ProductionShaError::LengthMismatch {
            label: "SHA endpoint word-source count",
            got: endpoint_evals.sources.len(),
            expected: word_sources.len(),
        });
    }
    for (got, expected) in endpoint_evals.sources.iter().zip(word_sources.iter()) {
        if (got.col, got.shift) != *expected {
            return Err(ProductionShaError::NonCanonicalProofObject(
                "SHA endpoint word sources are not in canonical order",
            ));
        }
    }

    let int_sources = production_sha_endpoint_int_sources();
    if endpoint_evals.int_sources.len() != int_sources.len() {
        return Err(ProductionShaError::LengthMismatch {
            label: "SHA endpoint int-source count",
            got: endpoint_evals.int_sources.len(),
            expected: int_sources.len(),
        });
    }
    for (got, expected) in endpoint_evals.int_sources.iter().zip(int_sources.iter()) {
        if got.col != *expected {
            return Err(ProductionShaError::NonCanonicalProofObject(
                "SHA endpoint int sources are not in canonical order",
            ));
        }
    }
    Ok(())
}

pub fn prove_sha_endpoint_multipoint<F>(
    transcript: &mut impl Transcript,
    folded_trace: &ProjectedTrace<F>,
    folded_public: &ProjectedPublic<F>,
    endpoint_evals: &ShaEndpointEvals<F>,
    r_star: &[F],
    field_cfg: &F::Config,
) -> Result<(MultipointEvalProof<F>, Vec<F>), ProductionShaError<F>>
where
    F: InnerTransparentField
        + DelayedFieldProductSum
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: Transcribable + Zero + Default + Send + Sync,
    F::Modulus: Transcribable,
{
    let row_weights = build_eq_x_r_vec(r_star, field_cfg)?;
    prove_sha_endpoint_multipoint_with_row_weights(
        transcript,
        folded_trace,
        folded_public,
        endpoint_evals,
        r_star,
        &row_weights,
        field_cfg,
    )
}

pub fn prove_sha_endpoint_multipoint_with_row_weights<F>(
    transcript: &mut impl Transcript,
    folded_trace: &ProjectedTrace<F>,
    folded_public: &ProjectedPublic<F>,
    endpoint_evals: &ShaEndpointEvals<F>,
    r_star: &[F],
    row_weights: &[F],
    field_cfg: &F::Config,
) -> Result<(MultipointEvalProof<F>, Vec<F>), ProductionShaError<F>>
where
    F: InnerTransparentField
        + DelayedFieldProductSum
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: Transcribable + Zero + Default + Send + Sync,
    F::Modulus: Transcribable,
{
    if row_weights.len() != SHA_ROW_COUNT {
        return Err(ProductionShaError::LengthMismatch {
            label: "row weights",
            got: row_weights.len(),
            expected: SHA_ROW_COUNT,
        });
    }
    validate_sha_endpoint_layout(endpoint_evals)?;
    let layout = tracing::info_span!(
        target: "zinc_protocol::production_sha",
        "multipoint_layout",
        side = "prove",
        phase = "multipoint_layout",
    )
    .in_scope(production_sha_multipoint_layout);
    let trace_mles = tracing::info_span!(
        target: "zinc_protocol::production_sha",
        "multipoint_trace_mles",
        side = "prove",
        phase = "multipoint_trace_mles",
        sources = layout.sources.len(),
    )
    .in_scope(|| sha_multipoint_trace_mles(folded_trace, folded_public, &layout, field_cfg))?;
    let up_evals = tracing::info_span!(
        target: "zinc_protocol::production_sha",
        "multipoint_up_evals",
        side = "prove",
        phase = "multipoint_up_evals",
        sources = layout.sources.len(),
    )
    .in_scope(|| {
        sha_multipoint_up_evals_with_row_weights(
            endpoint_evals,
            folded_public,
            row_weights,
            &layout,
            field_cfg,
        )
    })?;
    let (shift_specs, down_evals) = tracing::info_span!(
        target: "zinc_protocol::production_sha",
        "multipoint_down_evals",
        side = "prove",
        phase = "multipoint_down_evals",
        shifts = layout.shifts.len(),
    )
    .in_scope(|| {
        sha_multipoint_shift_specs_and_down_evals_with_row_weights(
            endpoint_evals,
            folded_public,
            row_weights,
            &layout,
            field_cfg,
        )
    })?;
    tracing::info_span!(
        target: "zinc_protocol::production_sha",
        "multipoint_sumcheck",
        side = "prove",
        phase = "multipoint_sumcheck",
        sources = trace_mles.len(),
        shifts = shift_specs.len(),
    )
    .in_scope(|| {
        prove_multipoint_reduction(
            transcript,
            &trace_mles,
            r_star,
            &up_evals,
            &down_evals,
            &shift_specs,
            field_cfg,
        )
        .map_err(ProductionShaError::from)
    })
}

pub fn verify_sha_endpoint_multipoint<F>(
    transcript: &mut impl Transcript,
    proof: &MultipointEvalProof<F>,
    endpoint_evals: &ShaEndpointEvals<F>,
    folded_public: &ProjectedPublic<F>,
    r_star: &[F],
    field_cfg: &F::Config,
) -> Result<(MultipointSubclaim<F>, Vec<ShiftSpec>), ProductionShaError<F>>
where
    F: InnerTransparentField
        + DelayedFieldProductSum
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: Transcribable + Zero + Default + Send + Sync,
    F::Modulus: Transcribable,
{
    validate_sha_endpoint_layout(endpoint_evals)?;
    let layout = production_sha_multipoint_layout();
    let up_evals =
        sha_multipoint_up_evals(endpoint_evals, folded_public, r_star, &layout, field_cfg)?;
    let (shift_specs, down_evals) = sha_multipoint_shift_specs_and_down_evals(
        endpoint_evals,
        folded_public,
        r_star,
        &layout,
        field_cfg,
    )?;
    let subclaim = verify_multipoint_reduction(
        transcript,
        proof.clone(),
        r_star,
        &up_evals,
        &down_evals,
        &shift_specs,
        SHA_ROW_VARS,
        field_cfg,
    )?;
    Ok((subclaim, shift_specs))
}

pub fn verify_sha_endpoint_multipoint_open_evals<F>(
    subclaim: &MultipointSubclaim<F>,
    open_evals: &[F],
    shift_specs: &[ShiftSpec],
    field_cfg: &F::Config,
) -> Result<(), ProductionShaError<F>>
where
    F: InnerTransparentField
        + DelayedFieldProductSum
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: Transcribable + Zero + Default + Send + Sync,
    F::Modulus: Transcribable,
{
    Ok(MultipointEval::verify_subclaim(
        subclaim,
        open_evals,
        shift_specs,
        field_cfg,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn reconstruct_folded_row_terminal_from_endpoints<F>(
    endpoint_evals: &ShaEndpointEvals<F>,
    folded_public: &ProjectedPublic<F>,
    r_ic: &[F; SHA_ROW_VARS],
    r_star: &[F],
    a: &F,
    lambda: &F,
    rho: &F,
    xi: &F,
    booleanity_sources: &[ShaBooleanitySource],
    field_cfg: &F::Config,
) -> Result<F, ProductionShaError<F>>
where
    F: DelayedFieldProductSum,
{
    if r_star.len() != SHA_ROW_VARS {
        return Err(ProductionShaError::LengthMismatch {
            label: "r_star",
            got: r_star.len(),
            expected: SHA_ROW_VARS,
        });
    }
    let row_weights = build_eq_x_r_vec(r_star, field_cfg)?;
    reconstruct_folded_row_terminal_from_endpoints_with_row_weights(
        endpoint_evals,
        folded_public,
        r_ic,
        r_star,
        &row_weights,
        a,
        lambda,
        rho,
        xi,
        booleanity_sources,
        field_cfg,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn reconstruct_folded_row_terminal_from_endpoints_with_row_weights<F>(
    endpoint_evals: &ShaEndpointEvals<F>,
    folded_public: &ProjectedPublic<F>,
    r_ic: &[F; SHA_ROW_VARS],
    r_star: &[F],
    row_weights: &[F],
    a: &F,
    lambda: &F,
    rho: &F,
    xi: &F,
    booleanity_sources: &[ShaBooleanitySource],
    field_cfg: &F::Config,
) -> Result<F, ProductionShaError<F>>
where
    F: DelayedFieldProductSum,
{
    if r_star.len() != SHA_ROW_VARS {
        return Err(ProductionShaError::LengthMismatch {
            label: "r_star",
            got: r_star.len(),
            expected: SHA_ROW_VARS,
        });
    }
    if row_weights.len() != SHA_ROW_COUNT {
        return Err(ProductionShaError::LengthMismatch {
            label: "row weights",
            got: row_weights.len(),
            expected: SHA_ROW_COUNT,
        });
    }
    let a_powers = build_sha_residual_eval_powers(a, field_cfg);
    let lambda_powers = build_sha_lambda_powers(lambda, field_cfg);
    let booleanity_weights = build_booleanity_weights(rho, xi, booleanity_sources.len(), field_cfg);
    reconstruct_folded_row_terminal_from_endpoints_with_vectors(
        endpoint_evals,
        folded_public,
        r_ic,
        r_star,
        row_weights,
        &a_powers,
        &lambda_powers,
        &booleanity_weights,
        booleanity_sources,
        field_cfg,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn reconstruct_folded_row_terminal_from_endpoints_with_vectors<F>(
    endpoint_evals: &ShaEndpointEvals<F>,
    folded_public: &ProjectedPublic<F>,
    r_ic: &[F; SHA_ROW_VARS],
    r_star: &[F],
    row_weights: &[F],
    a_powers: &[F],
    lambda_powers: &[F],
    booleanity_weights: &[F],
    booleanity_sources: &[ShaBooleanitySource],
    field_cfg: &F::Config,
) -> Result<F, ProductionShaError<F>>
where
    F: DelayedFieldProductSum,
{
    if r_star.len() != SHA_ROW_VARS {
        return Err(ProductionShaError::LengthMismatch {
            label: "r_star",
            got: r_star.len(),
            expected: SHA_ROW_VARS,
        });
    }
    if row_weights.len() != SHA_ROW_COUNT {
        return Err(ProductionShaError::LengthMismatch {
            label: "row weights",
            got: row_weights.len(),
            expected: SHA_ROW_COUNT,
        });
    }
    if a_powers.len() < SHA_IDEAL_EVAL_POWER_COUNT {
        return Err(ProductionShaError::LengthMismatch {
            label: "a powers",
            got: a_powers.len(),
            expected: SHA_IDEAL_EVAL_POWER_COUNT,
        });
    }
    if lambda_powers.len() != NUM_SHA_RESIDUAL_FAMILIES {
        return Err(ProductionShaError::LengthMismatch {
            label: "lambda powers",
            got: lambda_powers.len(),
            expected: NUM_SHA_RESIDUAL_FAMILIES,
        });
    }
    if booleanity_weights.len() != booleanity_sources.len() {
        return Err(ProductionShaError::LengthMismatch {
            label: "booleanity weights",
            got: booleanity_weights.len(),
            expected: booleanity_sources.len(),
        });
    }
    validate_sha_endpoint_layout(endpoint_evals)?;
    verify_endpoint_scalarization_with_powers(endpoint_evals, a_powers, field_cfg)?;

    let residuals = residual_polys_from_endpoints_with_row_weights(
        endpoint_evals,
        folded_public,
        row_weights,
        field_cfg,
    )?;
    let linear = lambda_weighted_sha_residual_polys_at_powers(
        &residuals,
        a_powers,
        lambda_powers,
        field_cfg,
    )?;

    let mut bool_terms = Vec::with_capacity(booleanity_sources.len());
    for source in booleanity_sources {
        let d = booleanity_endpoint_value(endpoint_evals, source, field_cfg)?;
        bool_terms.push(d.clone() * (d - F::one_with_cfg(field_cfg)));
    }
    let bool_sum = FieldFieldInnerProduct::inner_product::<UNCHECKED>(
        booleanity_weights,
        &bool_terms,
        F::zero_with_cfg(field_cfg),
    )
    .map_err(|_| {
        ProductionShaError::NonCanonicalProofObject("endpoint booleanity dot product failed")
    })?;

    let row_weight = eq_eval(r_ic, r_star, F::one_with_cfg(field_cfg))?;
    Ok(row_weight * (linear + bool_sum))
}

pub fn verify_endpoint_scalarization<F>(
    endpoint_evals: &ShaEndpointEvals<F>,
    a: &F,
    field_cfg: &F::Config,
) -> Result<(), ProductionShaError<F>>
where
    F: DelayedFieldProductSum,
{
    let powers = zinc_utils::powers(a.clone(), F::one_with_cfg(field_cfg), 32);
    verify_endpoint_scalarization_with_powers(endpoint_evals, &powers, field_cfg)
}

pub fn verify_endpoint_scalarization_with_powers<F>(
    endpoint_evals: &ShaEndpointEvals<F>,
    a_powers: &[F],
    field_cfg: &F::Config,
) -> Result<(), ProductionShaError<F>>
where
    F: DelayedFieldProductSum,
{
    if a_powers.len() < SHA_WORD_BITS {
        return Err(ProductionShaError::LengthMismatch {
            label: "endpoint scalarization powers",
            got: a_powers.len(),
            expected: SHA_WORD_BITS,
        });
    }
    for source in &endpoint_evals.sources {
        let recombined = zinc_utils::inner_product::FieldFieldInnerProduct::inner_product::<
            UNCHECKED,
        >(
            &source.bits,
            &a_powers[..SHA_WORD_BITS],
            F::zero_with_cfg(field_cfg),
        )
        .map_err(|_| {
            ProductionShaError::NonCanonicalProofObject("endpoint scalarization dot product failed")
        })?;
        if recombined != source.scalarized {
            return Err(ProductionShaError::EndpointScalarizationMismatch {
                col: source.col,
                shift: source.shift,
            });
        }
    }
    Ok(())
}

pub fn reconstruct_virtual_ch_maj_endpoint<F>(
    endpoint_evals: &ShaEndpointEvals<F>,
    field_cfg: &F::Config,
) -> Result<VirtualChMajEndpoint<F>, ProductionShaError<F>>
where
    F: PrimeField,
{
    let two = F::one_with_cfg(field_cfg) + F::one_with_cfg(field_cfg);
    let bits = |col, shift| source_bits(endpoint_evals, col, shift);
    let e0 = bits(ShaWordCol::E, 0)?;
    let e1 = bits(ShaWordCol::E, 1)?;
    let e2 = bits(ShaWordCol::E, 2)?;
    let a0 = bits(ShaWordCol::A, 0)?;
    let a1 = bits(ShaWordCol::A, 1)?;
    let a2 = bits(ShaWordCol::A, 2)?;
    let uef2 = bits(ShaWordCol::Uef, 2)?;
    let uneg_eg2 = bits(ShaWordCol::UNegEg, 2)?;
    let ch2_comp0 = bits(ShaWordCol::Ch2Comp, 0)?;
    let maj2 = bits(ShaWordCol::Maj, 2)?;
    let maj_comp0 = bits(ShaWordCol::MajComp, 0)?;

    Ok(VirtualChMajEndpoint {
        ch1: std::array::from_fn(|idx| e2[idx].clone() + &e1[idx] - two.clone() * &uef2[idx]),
        ch2: std::array::from_fn(|idx| {
            e2[idx].clone() - &e0[idx]
                + two.clone() * &uneg_eg2[idx]
                + two.clone() * &ch2_comp0[idx]
        }),
        maj: std::array::from_fn(|idx| {
            a0[idx].clone() + &a1[idx] + &a2[idx]
                - two.clone() * &maj2[idx]
                - two.clone() * &maj_comp0[idx]
        }),
    })
}

pub fn booleanity_endpoint_value<F>(
    endpoint_evals: &ShaEndpointEvals<F>,
    source: &ShaBooleanitySource,
    field_cfg: &F::Config,
) -> Result<F, ProductionShaError<F>>
where
    F: PrimeField,
{
    match source {
        ShaBooleanitySource::WordBit { col, bit } => Ok(source_bits(endpoint_evals, *col, 0)?
            .get(*bit)
            .cloned()
            .ok_or(ProductionShaError::LengthMismatch {
                label: "endpoint bit",
                got: *bit,
                expected: 32,
            })?),
        ShaBooleanitySource::VirtualCh1 { bit } => Ok(reconstruct_virtual_ch_maj_endpoint(
            endpoint_evals,
            field_cfg,
        )?
        .ch1
        .get(*bit)
        .cloned()
        .ok_or(ProductionShaError::LengthMismatch {
            label: "virtual Ch1 bit",
            got: *bit,
            expected: 32,
        })?),
        ShaBooleanitySource::VirtualCh2 { bit } => Ok(reconstruct_virtual_ch_maj_endpoint(
            endpoint_evals,
            field_cfg,
        )?
        .ch2
        .get(*bit)
        .cloned()
        .ok_or(ProductionShaError::LengthMismatch {
            label: "virtual Ch2 bit",
            got: *bit,
            expected: 32,
        })?),
        ShaBooleanitySource::VirtualMaj { bit } => Ok(reconstruct_virtual_ch_maj_endpoint(
            endpoint_evals,
            field_cfg,
        )?
        .maj
        .get(*bit)
        .cloned()
        .ok_or(ProductionShaError::LengthMismatch {
            label: "virtual Maj bit",
            got: *bit,
            expected: 32,
        })?),
    }
}

fn sha_multipoint_trace_mles<F>(
    folded_trace: &ProjectedTrace<F>,
    folded_public: &ProjectedPublic<F>,
    layout: &ShaMultipointLayout,
    field_cfg: &F::Config,
) -> Result<Vec<DenseMultilinearExtension<F::Inner>>, ProductionShaError<F>>
where
    F: InnerTransparentField,
{
    let zero_inner = F::zero_with_cfg(field_cfg).inner().clone();
    layout
        .sources
        .iter()
        .map(|source| {
            let values = (0..SHA_ROW_COUNT)
                .map(|row| {
                    sha_mp_source_row_value(folded_trace, folded_public, *source, row, field_cfg)
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(DenseMultilinearExtension::from_evaluations_vec(
                SHA_ROW_VARS,
                values.iter().map(|value| value.inner().clone()).collect(),
                zero_inner.clone(),
            ))
        })
        .collect()
}

fn sha_multipoint_up_evals<F>(
    endpoint_evals: &ShaEndpointEvals<F>,
    folded_public: &ProjectedPublic<F>,
    r_star: &[F],
    layout: &ShaMultipointLayout,
    field_cfg: &F::Config,
) -> Result<Vec<F>, ProductionShaError<F>>
where
    F: PrimeField,
{
    let row_weights = build_eq_x_r_vec(r_star, field_cfg)?;
    sha_multipoint_up_evals_with_row_weights(
        endpoint_evals,
        folded_public,
        &row_weights,
        layout,
        field_cfg,
    )
}

fn sha_multipoint_up_evals_with_row_weights<F>(
    endpoint_evals: &ShaEndpointEvals<F>,
    folded_public: &ProjectedPublic<F>,
    row_weights: &[F],
    layout: &ShaMultipointLayout,
    field_cfg: &F::Config,
) -> Result<Vec<F>, ProductionShaError<F>>
where
    F: PrimeField,
{
    layout
        .sources
        .iter()
        .map(|source| {
            sha_mp_source_endpoint_value_with_row_weights(
                endpoint_evals,
                folded_public,
                row_weights,
                *source,
                field_cfg,
            )
        })
        .collect()
}

fn sha_multipoint_shift_specs_and_down_evals<F>(
    endpoint_evals: &ShaEndpointEvals<F>,
    folded_public: &ProjectedPublic<F>,
    r_star: &[F],
    layout: &ShaMultipointLayout,
    field_cfg: &F::Config,
) -> Result<(Vec<ShiftSpec>, Vec<F>), ProductionShaError<F>>
where
    F: PrimeField,
{
    let row_weights = build_eq_x_r_vec(r_star, field_cfg)?;
    sha_multipoint_shift_specs_and_down_evals_with_row_weights(
        endpoint_evals,
        folded_public,
        &row_weights,
        layout,
        field_cfg,
    )
}

fn sha_multipoint_shift_specs_and_down_evals_with_row_weights<F>(
    endpoint_evals: &ShaEndpointEvals<F>,
    folded_public: &ProjectedPublic<F>,
    row_weights: &[F],
    layout: &ShaMultipointLayout,
    field_cfg: &F::Config,
) -> Result<(Vec<ShiftSpec>, Vec<F>), ProductionShaError<F>>
where
    F: PrimeField,
{
    let mut specs = Vec::with_capacity(layout.shifts.len());
    let mut evals = Vec::with_capacity(layout.shifts.len());
    for shift in &layout.shifts {
        let (source, amount, value) = match *shift {
            ShaMpShiftSource::WordBit { col, bit, shift } => (
                ShaMpSource::WordBit { col, bit },
                shift,
                source_bits(endpoint_evals, col, shift)?
                    .get(bit)
                    .cloned()
                    .ok_or(ProductionShaError::LengthMismatch {
                        label: "shifted word bit",
                        got: bit,
                        expected: 32,
                    })?,
            ),
            ShaMpShiftSource::Public { col, shift } => (
                ShaMpSource::Public { col },
                shift,
                sha_public_at_point_with_weights(
                    folded_public,
                    col,
                    shift,
                    row_weights,
                    field_cfg,
                )?,
            ),
        };
        let source_idx = layout
            .sources
            .iter()
            .position(|candidate| *candidate == source)
            .ok_or(ProductionShaError::LengthMismatch {
                label: "multipoint shift source",
                got: 0,
                expected: 1,
            })?;
        specs.push(ShiftSpec::new(source_idx, amount));
        evals.push(value);
    }
    Ok((specs, evals))
}

fn sha_mp_source_row_value<F>(
    folded_trace: &ProjectedTrace<F>,
    folded_public: &ProjectedPublic<F>,
    source: ShaMpSource,
    row: usize,
    field_cfg: &F::Config,
) -> Result<F, ProductionShaError<F>>
where
    F: PrimeField,
{
    match source {
        ShaMpSource::WordBit { col, bit } => folded_trace
            .bit_slices
            .get(bit_slice_index(col.index(), bit, SHA_WORD_BITS))
            .and_then(|column| column.evaluations.get(row))
            .cloned()
            .ok_or(ProductionShaError::LengthMismatch {
                label: "multipoint word bit source",
                got: row,
                expected: SHA_ROW_COUNT,
            }),
        ShaMpSource::Int { col } => folded_trace
            .int_columns
            .get(col.index())
            .and_then(|column| column.evaluations.get(row))
            .cloned()
            .ok_or(ProductionShaError::LengthMismatch {
                label: "multipoint int source",
                got: row,
                expected: SHA_ROW_COUNT,
            }),
        ShaMpSource::Public { col } => folded_public
            .columns
            .get(col.index())
            .and_then(|column| column.evaluations.get(row))
            .cloned()
            .ok_or(ProductionShaError::LengthMismatch {
                label: "multipoint public source",
                got: row,
                expected: SHA_ROW_COUNT,
            }),
    }
    .map(|value| {
        let _ = field_cfg;
        value
    })
}

fn sha_mp_source_endpoint_value_with_row_weights<F>(
    endpoint_evals: &ShaEndpointEvals<F>,
    folded_public: &ProjectedPublic<F>,
    row_weights: &[F],
    source: ShaMpSource,
    field_cfg: &F::Config,
) -> Result<F, ProductionShaError<F>>
where
    F: PrimeField,
{
    match source {
        ShaMpSource::WordBit { col, bit } => source_bits(endpoint_evals, col, 0)?
            .get(bit)
            .cloned()
            .ok_or(ProductionShaError::LengthMismatch {
                label: "endpoint word bit",
                got: bit,
                expected: 32,
            }),
        ShaMpSource::Int { col } => endpoint_evals
            .int_sources
            .iter()
            .find(|source| source.col == col)
            .map(|source| source.scalar.clone())
            .ok_or(ProductionShaError::LengthMismatch {
                label: "endpoint int source",
                got: endpoint_evals.int_sources.len(),
                expected: ShaIntCol::COUNT,
            }),
        ShaMpSource::Public { col } => Ok(sha_public_at_point_with_weights(
            folded_public,
            col,
            0,
            row_weights,
            field_cfg,
        )?),
    }
}

fn residual_polys_from_endpoints_with_row_weights<F>(
    endpoint_evals: &ShaEndpointEvals<F>,
    folded_public: &ProjectedPublic<F>,
    row_weights: &[F],
    field_cfg: &F::Config,
) -> Result<Vec<DynamicPolynomialF<F>>, ProductionShaError<F>>
where
    F: DelayedFieldProductSum,
{
    if row_weights.len() != SHA_ROW_COUNT {
        return Err(ProductionShaError::LengthMismatch {
            label: "row weights",
            got: row_weights.len(),
            expected: SHA_ROW_COUNT,
        });
    }
    let one = F::one_with_cfg(field_cfg);
    let two = one.clone() + &one;
    let rho_sig0 = sparse_endpoint_poly::<F>(&[10, 19, 30], field_cfg);
    let rho_sig1 = sparse_endpoint_poly::<F>(&[7, 21, 26], field_cfg);

    let a = endpoint_word_poly(endpoint_evals, ShaWordCol::A, 0, field_cfg)?;
    let e = endpoint_word_poly(endpoint_evals, ShaWordCol::E, 0, field_cfg)?;
    let sigma0 = endpoint_word_poly(endpoint_evals, ShaWordCol::Sigma0, 0, field_cfg)?;
    let sigma1 = endpoint_word_poly(endpoint_evals, ShaWordCol::Sigma1, 0, field_cfg)?;
    let w = endpoint_word_poly(endpoint_evals, ShaWordCol::W, 0, field_cfg)?;
    let small_sigma0 = endpoint_word_poly(endpoint_evals, ShaWordCol::SmallSigma0, 0, field_cfg)?;
    let small_sigma1 = endpoint_word_poly(endpoint_evals, ShaWordCol::SmallSigma1, 0, field_cfg)?;
    let ov_sigma0 = endpoint_word_poly(endpoint_evals, ShaWordCol::OvSigma0, 0, field_cfg)?;
    let ov_sigma1 = endpoint_word_poly(endpoint_evals, ShaWordCol::OvSigma1, 0, field_cfg)?;
    let ov_small_sigma0 =
        endpoint_word_poly(endpoint_evals, ShaWordCol::OvSmallSigma0, 0, field_cfg)?;
    let ov_small_sigma1 =
        endpoint_word_poly(endpoint_evals, ShaWordCol::OvSmallSigma1, 0, field_cfg)?;

    let r0 = (&a * &rho_sig0) - &sigma0 - &scale_endpoint_poly(&ov_sigma0, &two);
    let r1 = (&e * &rho_sig1) - &sigma1 - &scale_endpoint_poly(&ov_sigma1, &two);
    let r2 = endpoint_word_poly(endpoint_evals, ShaWordCol::W, 0, field_cfg)?.rot_c(25)
        + &endpoint_word_poly(endpoint_evals, ShaWordCol::W, 0, field_cfg)?.rot_c(14)
        + &endpoint_word_poly(endpoint_evals, ShaWordCol::W, 0, field_cfg)?.shift_r_c(3)
        - &small_sigma0
        - &scale_endpoint_poly(&ov_small_sigma0, &two);
    let r3 = endpoint_word_poly(endpoint_evals, ShaWordCol::W, 0, field_cfg)?.rot_c(15)
        + &endpoint_word_poly(endpoint_evals, ShaWordCol::W, 0, field_cfg)?.rot_c(13)
        + &endpoint_word_poly(endpoint_evals, ShaWordCol::W, 0, field_cfg)?.shift_r_c(10)
        - &small_sigma1
        - &scale_endpoint_poly(&ov_small_sigma1, &two);

    let mu_w = endpoint_mu_contribution(endpoint_evals, 0, 2, field_cfg)?;
    let mu_a = endpoint_mu_contribution(endpoint_evals, 2, 5, field_cfg)?;
    let mu_e = endpoint_mu_contribution(endpoint_evals, 5, 8, field_cfg)?;
    let mu_ff_a = endpoint_mu_contribution(endpoint_evals, 8, 9, field_cfg)?;
    let mu_ff_e = endpoint_mu_contribution(endpoint_evals, 9, 10, field_cfg)?;

    let r4 = endpoint_word_poly(endpoint_evals, ShaWordCol::W, 16, field_cfg)?
        - &w
        - &endpoint_word_poly(endpoint_evals, ShaWordCol::SmallSigma0, 1, field_cfg)?
        - &endpoint_word_poly(endpoint_evals, ShaWordCol::W, 9, field_cfg)?
        - &endpoint_word_poly(endpoint_evals, ShaWordCol::SmallSigma1, 14, field_cfg)?
        + &mu_w
        + &endpoint_int_const_poly(endpoint_evals, ShaIntCol::CompSchedule, field_cfg)?;

    let r5 = endpoint_word_poly(endpoint_evals, ShaWordCol::A, 4, field_cfg)?
        - &e
        - &endpoint_word_poly(endpoint_evals, ShaWordCol::Sigma1, 3, field_cfg)?
        - &endpoint_word_poly(endpoint_evals, ShaWordCol::Uef, 3, field_cfg)?
        - &endpoint_word_poly(endpoint_evals, ShaWordCol::UNegEg, 3, field_cfg)?
        - &endpoint_public_const_poly_with_row_weights(
            folded_public,
            ShaPublicCol::K,
            3,
            row_weights,
            field_cfg,
        )?
        - &w
        - &endpoint_word_poly(endpoint_evals, ShaWordCol::Sigma0, 3, field_cfg)?
        - &endpoint_word_poly(endpoint_evals, ShaWordCol::Maj, 3, field_cfg)?
        + &mu_a
        + &endpoint_int_const_poly(endpoint_evals, ShaIntCol::CompUpdateA, field_cfg)?;

    let r6 = endpoint_word_poly(endpoint_evals, ShaWordCol::E, 4, field_cfg)?
        - &a
        - &e
        - &endpoint_word_poly(endpoint_evals, ShaWordCol::Sigma1, 3, field_cfg)?
        - &endpoint_word_poly(endpoint_evals, ShaWordCol::Uef, 3, field_cfg)?
        - &endpoint_word_poly(endpoint_evals, ShaWordCol::UNegEg, 3, field_cfg)?
        - &endpoint_public_const_poly_with_row_weights(
            folded_public,
            ShaPublicCol::K,
            3,
            row_weights,
            field_cfg,
        )?
        - &w
        + &mu_e
        + &endpoint_int_const_poly(endpoint_evals, ShaIntCol::CompUpdateE, field_cfg)?;

    let s_init = sha_public_at_point_with_weights(
        folded_public,
        ShaPublicCol::SInit,
        0,
        row_weights,
        field_cfg,
    )?;
    let s_msg = sha_public_at_point_with_weights(
        folded_public,
        ShaPublicCol::SMsg,
        0,
        row_weights,
        field_cfg,
    )?;
    let s_sched = sha_public_at_point_with_weights(
        folded_public,
        ShaPublicCol::SSched,
        0,
        row_weights,
        field_cfg,
    )?;
    let s_upd = sha_public_at_point_with_weights(
        folded_public,
        ShaPublicCol::SUpd,
        0,
        row_weights,
        field_cfg,
    )?;
    let s_ff = sha_public_at_point_with_weights(
        folded_public,
        ShaPublicCol::SFf,
        0,
        row_weights,
        field_cfg,
    )?;
    let s_out = sha_public_at_point_with_weights(
        folded_public,
        ShaPublicCol::SOut,
        0,
        row_weights,
        field_cfg,
    )?;

    let r7 = scale_endpoint_poly(
        &(a.clone()
            - &endpoint_public_word_or_const_poly_with_row_weights(
                folded_public,
                ShaPublicCol::PAIn,
                row_weights,
                field_cfg,
            )?),
        &s_init,
    ) + &scale_endpoint_poly(
        &(a.clone()
            - &endpoint_public_word_or_const_poly_with_row_weights(
                folded_public,
                ShaPublicCol::PAOut,
                row_weights,
                field_cfg,
            )?),
        &s_out,
    );
    let r8 = scale_endpoint_poly(
        &(e.clone()
            - &endpoint_public_word_or_const_poly_with_row_weights(
                folded_public,
                ShaPublicCol::PEIn,
                row_weights,
                field_cfg,
            )?),
        &s_init,
    ) + &scale_endpoint_poly(
        &(e.clone()
            - &endpoint_public_word_or_const_poly_with_row_weights(
                folded_public,
                ShaPublicCol::PEOut,
                row_weights,
                field_cfg,
            )?),
        &s_out,
    );

    let r9 = endpoint_word_poly(endpoint_evals, ShaWordCol::A, 4, field_cfg)?
        - &a
        - &endpoint_public_const_poly_with_row_weights(
            folded_public,
            ShaPublicCol::PAIn,
            0,
            row_weights,
            field_cfg,
        )?
        + &mu_ff_a
        + &endpoint_int_const_poly(endpoint_evals, ShaIntCol::CompFeedForwardA, field_cfg)?;
    let r10 = endpoint_word_poly(endpoint_evals, ShaWordCol::E, 4, field_cfg)?
        - &e
        - &endpoint_public_const_poly_with_row_weights(
            folded_public,
            ShaPublicCol::PEIn,
            0,
            row_weights,
            field_cfg,
        )?
        + &mu_ff_e
        + &endpoint_int_const_poly(endpoint_evals, ShaIntCol::CompFeedForwardE, field_cfg)?;
    let r11 = scale_endpoint_poly(
        &(w - &endpoint_public_word_or_const_poly_with_row_weights(
            folded_public,
            ShaPublicCol::Message,
            row_weights,
            field_cfg,
        )?),
        &s_msg,
    );

    let comp_schedule =
        endpoint_int_const_poly(endpoint_evals, ShaIntCol::CompSchedule, field_cfg)?;
    let comp_update_a = endpoint_int_const_poly(endpoint_evals, ShaIntCol::CompUpdateA, field_cfg)?;
    let comp_update_e = endpoint_int_const_poly(endpoint_evals, ShaIntCol::CompUpdateE, field_cfg)?;
    let comp_ff_a =
        endpoint_int_const_poly(endpoint_evals, ShaIntCol::CompFeedForwardA, field_cfg)?;
    let comp_ff_e =
        endpoint_int_const_poly(endpoint_evals, ShaIntCol::CompFeedForwardE, field_cfg)?;

    let r12 = scale_endpoint_poly(&comp_schedule, &s_sched);
    let r13 = scale_endpoint_poly(&comp_update_a, &s_upd);
    let r14 = scale_endpoint_poly(&comp_update_e, &s_upd);
    let r15 = scale_endpoint_poly(&comp_ff_a, &s_ff);
    let r16 = scale_endpoint_poly(&comp_ff_e, &s_ff);
    let r17 = endpoint_word_poly(endpoint_evals, ShaWordCol::MuPacked, 0, field_cfg)?.shift_r_c(10);

    let mut residuals = vec![
        r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17,
    ];
    residuals.iter_mut().for_each(DynamicPolynomialF::trim);
    Ok(residuals)
}

fn endpoint_word_poly<F>(
    endpoint_evals: &ShaEndpointEvals<F>,
    col: ShaWordCol,
    shift: usize,
    field_cfg: &F::Config,
) -> Result<DynamicPolynomialF<F>, ProductionShaError<F>>
where
    F: PrimeField,
{
    let mut coeffs = source_bits(endpoint_evals, col, shift)?.to_vec();
    coeffs.resize(32, F::zero_with_cfg(field_cfg));
    Ok(DynamicPolynomialF::new_trimmed(coeffs))
}

fn endpoint_int_const_poly<F>(
    endpoint_evals: &ShaEndpointEvals<F>,
    col: ShaIntCol,
    field_cfg: &F::Config,
) -> Result<DynamicPolynomialF<F>, ProductionShaError<F>>
where
    F: PrimeField,
{
    let value = endpoint_evals
        .int_sources
        .iter()
        .find(|source| source.col == col)
        .map(|source| source.scalar.clone())
        .ok_or(ProductionShaError::LengthMismatch {
            label: "endpoint int source",
            got: endpoint_evals.int_sources.len(),
            expected: ShaIntCol::COUNT,
        })?;
    Ok(endpoint_const_poly(value, field_cfg))
}

fn endpoint_public_const_poly_with_row_weights<F>(
    folded_public: &ProjectedPublic<F>,
    col: ShaPublicCol,
    shift: usize,
    row_weights: &[F],
    field_cfg: &F::Config,
) -> Result<DynamicPolynomialF<F>, ProductionShaError<F>>
where
    F: PrimeField,
{
    Ok(endpoint_const_poly(
        sha_public_at_point_with_weights(folded_public, col, shift, row_weights, field_cfg)?,
        field_cfg,
    ))
}

fn endpoint_public_word_or_const_poly_with_row_weights<F>(
    folded_public: &ProjectedPublic<F>,
    col: ShaPublicCol,
    row_weights: &[F],
    field_cfg: &F::Config,
) -> Result<DynamicPolynomialF<F>, ProductionShaError<F>>
where
    F: DelayedFieldProductSum,
{
    let Some(col_idx) = public_word_col_index(col) else {
        return endpoint_public_const_poly_with_row_weights(
            folded_public,
            col,
            0,
            row_weights,
            field_cfg,
        );
    };
    let bit_slices =
        folded_public
            .bit_slices
            .as_ref()
            .ok_or(ProductionShaError::NonCanonicalProofObject(
                "production SHA public word columns are required",
            ))?;
    let mut coeffs = Vec::with_capacity(SHA_WORD_BITS);
    for bit in 0..SHA_WORD_BITS {
        let table_idx = bit_slice_index(col_idx, bit, SHA_WORD_BITS);
        let bit_column = bit_slices
            .get(table_idx)
            .ok_or(ProductionShaError::LengthMismatch {
                label: "SHA public word bit column",
                got: table_idx,
                expected: bit_slices.len(),
            })?;
        if bit_column.num_vars != SHA_ROW_VARS || bit_column.evaluations.len() != SHA_ROW_COUNT {
            return Err(ProductionShaError::LengthMismatch {
                label: "SHA public word row count",
                got: bit_column.evaluations.len(),
                expected: SHA_ROW_COUNT,
            });
        }
        coeffs.push(
            FieldFieldInnerProduct::inner_product::<UNCHECKED>(
                row_weights,
                &bit_column.evaluations,
                F::zero_with_cfg(field_cfg),
            )
            .map_err(|_| {
                ProductionShaError::NonCanonicalProofObject(
                    "SHA public word row-weight dot product failed",
                )
            })?,
        );
    }
    Ok(DynamicPolynomialF::new_trimmed(coeffs))
}

fn endpoint_mu_contribution<F>(
    endpoint_evals: &ShaEndpointEvals<F>,
    low: usize,
    high: usize,
    field_cfg: &F::Config,
) -> Result<DynamicPolynomialF<F>, ProductionShaError<F>>
where
    F: PrimeField,
{
    let packed = endpoint_word_poly(endpoint_evals, ShaWordCol::MuPacked, 0, field_cfg)?
        .shift_r_c(low as u32);
    let tail = endpoint_word_poly(endpoint_evals, ShaWordCol::MuPacked, 0, field_cfg)?
        .shift_r_c(high as u32);
    let low_coeff = endpoint_pow_two(32, field_cfg);
    let high_coeff = endpoint_pow_two(32 + high - low, field_cfg);
    Ok(scale_endpoint_poly(&packed, &low_coeff) - &scale_endpoint_poly(&tail, &high_coeff))
}

fn sparse_endpoint_poly<F>(positions: &[usize], field_cfg: &F::Config) -> DynamicPolynomialF<F>
where
    F: PrimeField,
{
    let mut coeffs = vec![F::zero_with_cfg(field_cfg); 32];
    for &pos in positions {
        coeffs[pos] = F::one_with_cfg(field_cfg);
    }
    DynamicPolynomialF::new_trimmed(coeffs)
}

fn scale_endpoint_poly<F>(poly: &DynamicPolynomialF<F>, scalar: &F) -> DynamicPolynomialF<F>
where
    F: PrimeField,
{
    if poly.is_zero() || F::is_zero(scalar) {
        return DynamicPolynomialF::ZERO;
    }
    DynamicPolynomialF::new_trimmed(
        poly.coeffs
            .iter()
            .map(|coeff| coeff.clone() * scalar)
            .collect::<Vec<_>>(),
    )
}

fn endpoint_const_poly<F>(value: F, field_cfg: &F::Config) -> DynamicPolynomialF<F>
where
    F: PrimeField,
{
    if F::is_zero(&value) {
        DynamicPolynomialF::ZERO
    } else {
        let _ = field_cfg;
        DynamicPolynomialF::constant_poly(value)
    }
}

fn endpoint_pow_two<F>(exp: usize, field_cfg: &F::Config) -> F
where
    F: PrimeField,
{
    let two = F::one_with_cfg(field_cfg) + F::one_with_cfg(field_cfg);
    let mut out = F::one_with_cfg(field_cfg);
    for _ in 0..exp {
        out *= &two;
    }
    out
}

fn source_bits<F>(
    endpoint_evals: &ShaEndpointEvals<F>,
    col: ShaWordCol,
    shift: usize,
) -> Result<&[F; 32], ProductionShaError<F>>
where
    F: PrimeField,
{
    endpoint_evals
        .sources
        .iter()
        .find(|source| source.col == col && source.shift == shift)
        .map(|source| &source.bits)
        .ok_or(ProductionShaError::MissingEndpointEval { col, shift })
}

fn sumfold_expected_eval<F>(
    beta: &[F],
    fresh_targets: &[F],
    r_b: &[F],
    field_cfg: &F::Config,
) -> Result<F, ProductionShaError<F>>
where
    F: DelayedFieldProductSum,
{
    let d = eq_eval(beta, r_b, F::one_with_cfg(field_cfg))?;
    let weights = build_eq_x_r_vec(r_b, field_cfg)?;
    if weights.len() != fresh_targets.len() {
        return Err(ProductionShaError::LengthMismatch {
            label: "sumfold target weights",
            got: weights.len(),
            expected: fresh_targets.len(),
        });
    }
    let claim_at_r = FieldFieldInnerProduct::inner_product::<UNCHECKED>(
        &weights,
        fresh_targets,
        F::zero_with_cfg(field_cfg),
    )
    .map_err(|_| {
        ProductionShaError::NonCanonicalProofObject("SumFold expected-value dot product failed")
    })?;
    Ok(d * claim_at_r)
}

fn require_single_sumcheck_group<F>(
    proof: &MultiDegreeSumcheckProof<F>,
    label: &'static str,
) -> Result<(), ProductionShaError<F>>
where
    F: PrimeField,
{
    let group_count = proof.degrees().len();
    if group_count != 1 {
        return Err(ProductionShaError::UnexpectedSumcheckGroupCount {
            label,
            got: group_count,
        });
    }
    let claimed_count = proof.claimed_sums().len();
    if claimed_count != 1 {
        return Err(ProductionShaError::LengthMismatch {
            label: "sumcheck claimed sums",
            got: claimed_count,
            expected: 1,
        });
    }
    Ok(())
}

#[allow(dead_code)]
fn family_weight_index(family: ShaResidualFamily) -> usize {
    family.index()
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{fixed_prime, pcs::AllHyraxPCSTypes};
    use ark_ec::{CurveGroup, PrimeGroup};
    use core::fmt::Debug;
    use crypto_primitives::{
        FromWithConfig, crypto_bigint_boxed_monty::BoxedMontyField, crypto_bigint_int::Int,
        crypto_bigint_uint::Uint,
    };
    use zinc_piop::neutron_nova::{
        SHA_ROW_COUNT, SHA_WORD_BITS, expression_folded_row_sum, fold_projected_traces,
    };
    use zinc_poly::mle::MultilinearExtensionWithConfig;
    use zinc_poly::univariate::{binary::BinaryPolyInnerProduct, dense::DensePolyInnerProduct};
    use zinc_primality::MillerRabin;
    use zinc_test_uair::{
        EC_FP_INT_LIMBS, SHA256_INITIAL_STATE, Sha256CompressionSliceUair,
        sha256::cols as sha256_cols, sha256_padded_message_blocks,
        synthesize_sha256_chain_witnesses,
    };
    use zinc_transcript::{Blake3Transcript, traits::Transcript};
    use zinc_utils::inner_product::{MBSInnerProduct, ScalarProduct};
    use zip_plus::{
        code::iprs::{IprsCode, PnttConfigF65537},
        pcs::{
            hyrax::{HyraxBlindingMode, HyraxPCS},
            structs::ZipTypes,
        },
    };

    type F = BoxedMontyField;
    type ShaInt = Int<EC_FP_INT_LIMBS>;
    const TEST_DEGREE_PLUS_ONE: usize = 32;
    const TEST_REP: usize = 4;
    const TEST_CHECKED: bool = false;
    const TEST_FIELD_LIMBS: usize = 4;
    const TEST_NUM_COLUMN_OPENINGS: usize = 150;

    fn cfg() -> <F as PrimeField>::Config {
        fixed_prime::secp256k1_field_cfg::<F, Uint<TEST_FIELD_LIMBS>>()
    }

    fn f(value: u64) -> F {
        F::from_with_cfg(value, &cfg())
    }

    #[derive(Debug, Clone)]
    struct TestShaBinaryZipTypes;

    impl ZipTypes for TestShaBinaryZipTypes {
        const NUM_COLUMN_OPENINGS: usize = TEST_NUM_COLUMN_OPENINGS;
        type Eval = BinaryPoly<TEST_DEGREE_PLUS_ONE>;
        type Cw = DensePolynomial<i64, TEST_DEGREE_PLUS_ONE>;
        type Fmod = Uint<TEST_FIELD_LIMBS>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<{ EC_FP_INT_LIMBS * 4 }>;
        type Comb = DensePolynomial<Self::CombR, TEST_DEGREE_PLUS_ONE>;
        type EvalDotChal = BinaryPolyInnerProduct<Self::Chal, TEST_DEGREE_PLUS_ONE>;
        type CombDotChal = DensePolyInnerProduct<
            Self::CombR,
            Self::Chal,
            Self::CombR,
            MBSInnerProduct,
            TEST_DEGREE_PLUS_ONE,
        >;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    #[derive(Debug, Clone)]
    struct TestShaArbitraryZipTypes;

    impl ZipTypes for TestShaArbitraryZipTypes {
        const NUM_COLUMN_OPENINGS: usize = TEST_NUM_COLUMN_OPENINGS;
        type Eval = DensePolynomial<ShaInt, TEST_DEGREE_PLUS_ONE>;
        type Cw = DensePolynomial<Int<6>, TEST_DEGREE_PLUS_ONE>;
        type Fmod = Uint<TEST_FIELD_LIMBS>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<{ EC_FP_INT_LIMBS * 4 }>;
        type Comb = DensePolynomial<Self::CombR, TEST_DEGREE_PLUS_ONE>;
        type EvalDotChal = DensePolyInnerProduct<
            ShaInt,
            Self::Chal,
            Self::CombR,
            MBSInnerProduct,
            TEST_DEGREE_PLUS_ONE,
        >;
        type CombDotChal = DensePolyInnerProduct<
            Self::CombR,
            Self::Chal,
            Self::CombR,
            MBSInnerProduct,
            TEST_DEGREE_PLUS_ONE,
        >;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    #[derive(Debug, Clone)]
    struct TestShaIntZipTypes;

    impl ZipTypes for TestShaIntZipTypes {
        const NUM_COLUMN_OPENINGS: usize = TEST_NUM_COLUMN_OPENINGS;
        type Eval = ShaInt;
        type Cw = Int<6>;
        type Fmod = Uint<TEST_FIELD_LIMBS>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<{ EC_FP_INT_LIMBS * 4 }>;
        type Comb = Self::CombR;
        type EvalDotChal = ScalarProduct;
        type CombDotChal = ScalarProduct;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    #[derive(Clone, Debug)]
    struct TestShaZincTypes;

    impl ZincTypes<TEST_DEGREE_PLUS_ONE> for TestShaZincTypes {
        type Int = ShaInt;
        type Chal = i128;
        type Pt = i128;
        type Fmod = Uint<TEST_FIELD_LIMBS>;
        type PrimeTest = MillerRabin;

        type BinaryZt = TestShaBinaryZipTypes;
        type ArbitraryZt = TestShaArbitraryZipTypes;
        type IntZt = TestShaIntZipTypes;

        type BinaryLc = IprsCode<Self::BinaryZt, PnttConfigF65537, TEST_REP, TEST_CHECKED>;
        type ArbitraryLc = IprsCode<Self::ArbitraryZt, PnttConfigF65537, TEST_REP, TEST_CHECKED>;
        type IntLc = IprsCode<Self::IntZt, PnttConfigF65537, TEST_REP, TEST_CHECKED>;
    }

    fn sha_binary_col<'a>(
        public_trace: &'a UairTrace<'_, ShaInt, ShaInt, TEST_DEGREE_PLUS_ONE>,
        witness_trace: &'a UairTrace<'_, ShaInt, ShaInt, TEST_DEGREE_PLUS_ONE>,
        flat_col: usize,
    ) -> Result<
        &'a DenseMultilinearExtension<BinaryPoly<TEST_DEGREE_PLUS_ONE>>,
        ProductionShaError<F>,
    > {
        if flat_col < sha256_cols::NUM_BIN_PUB {
            public_trace
                .binary_poly
                .get(flat_col)
                .ok_or(ProductionShaError::LengthMismatch {
                    label: "SHA binary public source columns",
                    got: public_trace.binary_poly.len(),
                    expected: flat_col + 1,
                })
        } else {
            let witness_col = flat_col - sha256_cols::NUM_BIN_PUB;
            witness_trace
                .binary_poly
                .get(witness_col)
                .ok_or(ProductionShaError::LengthMismatch {
                    label: "SHA binary witness source columns",
                    got: witness_trace.binary_poly.len(),
                    expected: witness_col + 1,
                })
        }
    }

    fn sha_int_col<'a>(
        public_trace: &'a UairTrace<'_, ShaInt, ShaInt, TEST_DEGREE_PLUS_ONE>,
        witness_trace: &'a UairTrace<'_, ShaInt, ShaInt, TEST_DEGREE_PLUS_ONE>,
        flat_col: usize,
    ) -> Result<&'a DenseMultilinearExtension<ShaInt>, ProductionShaError<F>> {
        if flat_col < sha256_cols::NUM_INT_PUB {
            public_trace
                .int
                .get(flat_col)
                .ok_or(ProductionShaError::LengthMismatch {
                    label: "SHA int public source columns",
                    got: public_trace.int.len(),
                    expected: flat_col + 1,
                })
        } else {
            let witness_col = flat_col - sha256_cols::NUM_INT_PUB;
            witness_trace
                .int
                .get(witness_col)
                .ok_or(ProductionShaError::LengthMismatch {
                    label: "SHA int witness source columns",
                    got: witness_trace.int.len(),
                    expected: witness_col + 1,
                })
        }
    }

    fn project_binary_source(
        col: &DenseMultilinearExtension<BinaryPoly<TEST_DEGREE_PLUS_ONE>>,
        field_cfg: &<F as PrimeField>::Config,
    ) -> Result<Vec<Vec<F>>, ProductionShaError<F>> {
        if col.evaluations.len() < SHA_ROW_COUNT {
            return Err(ProductionShaError::LengthMismatch {
                label: "SHA binary source rows",
                got: col.evaluations.len(),
                expected: SHA_ROW_COUNT,
            });
        }
        Ok(col
            .evaluations
            .iter()
            .take(SHA_ROW_COUNT)
            .map(|poly| {
                poly.iter()
                    .take(SHA_WORD_BITS)
                    .map(|bit| {
                        if bit.into_inner() {
                            F::one_with_cfg(field_cfg)
                        } else {
                            F::zero_with_cfg(field_cfg)
                        }
                    })
                    .collect()
            })
            .collect())
    }

    fn project_int_source(
        col: &DenseMultilinearExtension<ShaInt>,
        field_cfg: &<F as PrimeField>::Config,
    ) -> Result<Vec<F>, ProductionShaError<F>> {
        if col.evaluations.len() < SHA_ROW_COUNT {
            return Err(ProductionShaError::LengthMismatch {
                label: "SHA int source rows",
                got: col.evaluations.len(),
                expected: SHA_ROW_COUNT,
            });
        }
        Ok(col
            .evaluations
            .iter()
            .take(SHA_ROW_COUNT)
            .map(|value| F::from_with_cfg(value, field_cfg))
            .collect())
    }

    fn truncate_sha_row_domain<Eval: Clone>(
        col: &DenseMultilinearExtension<Eval>,
        label: &'static str,
    ) -> Result<DenseMultilinearExtension<Eval>, ProductionShaError<F>> {
        if col.evaluations.len() < SHA_ROW_COUNT {
            return Err(ProductionShaError::LengthMismatch {
                label,
                got: col.evaluations.len(),
                expected: SHA_ROW_COUNT,
            });
        }
        Ok(DenseMultilinearExtension {
            evaluations: col.evaluations[..SHA_ROW_COUNT].to_vec(),
            num_vars: SHA_ROW_VARS,
        })
    }

    fn word_scalar_at_two(bits: &[F], field_cfg: &<F as PrimeField>::Config) -> F {
        let two = F::one_with_cfg(field_cfg) + F::one_with_cfg(field_cfg);
        let mut power = F::one_with_cfg(field_cfg);
        let mut value = F::zero_with_cfg(field_cfg);
        for bit in bits {
            value += bit.clone() * &power;
            power *= &two;
        }
        value
    }

    fn mle_table_from_columns<T>(columns: Vec<Vec<T>>) -> MleTable<T> {
        columns
            .into_iter()
            .map(|evaluations| DenseMultilinearExtension {
                evaluations,
                num_vars: SHA_ROW_VARS,
            })
            .collect()
    }

    fn flatten_bit_columns<T>(columns: Vec<Vec<Vec<T>>>) -> MleTable<T> {
        let mut flattened = (0..columns.len() * SHA_WORD_BITS)
            .map(|_| Vec::with_capacity(SHA_ROW_COUNT))
            .collect::<Vec<_>>();
        for (col_idx, rows) in columns.into_iter().enumerate() {
            for bits in rows {
                for (bit, value) in bits.into_iter().enumerate() {
                    flattened[bit_slice_index(col_idx, bit, SHA_WORD_BITS)].push(value);
                }
            }
        }
        mle_table_from_columns(flattened)
    }

    fn scalarize_bit_slices_plain(
        bit_slices: &MleTable<F>,
        a: &F,
        field_cfg: &<F as PrimeField>::Config,
    ) -> Result<MleTable<F>, ProductionShaError<F>> {
        let powers = zinc_utils::powers(a.clone(), F::one_with_cfg(field_cfg), SHA_WORD_BITS);
        let word_count = bit_slices.len() / SHA_WORD_BITS;
        let mut words = Vec::with_capacity(word_count);
        for col_idx in 0..word_count {
            let mut out_col = Vec::with_capacity(SHA_ROW_COUNT);
            for row in 0..SHA_ROW_COUNT {
                let mut value = F::zero_with_cfg(field_cfg);
                for (bit, power) in powers.iter().enumerate() {
                    let bit_col = &bit_slices[bit_slice_index(col_idx, bit, SHA_WORD_BITS)];
                    if bit_col.num_vars != SHA_ROW_VARS
                        || bit_col.evaluations.len() != SHA_ROW_COUNT
                    {
                        return Err(ProductionShaError::LengthMismatch {
                            label: "SHA scalarized bit-slice rows",
                            got: bit_col.evaluations.len(),
                            expected: SHA_ROW_COUNT,
                        });
                    }
                    value += bit_col.evaluations[row].clone() * power;
                }
                out_col.push(value);
            }
            words.push(out_col);
        }
        Ok(mle_table_from_columns(words))
    }

    fn projected_public_from_sources(
        pa_a: &[Vec<F>],
        pa_e: &[Vec<F>],
        message: &[Vec<F>],
        field_cfg: &<F as PrimeField>::Config,
    ) -> MleTable<F> {
        let mut columns =
            vec![vec![F::zero_with_cfg(field_cfg); SHA_ROW_COUNT]; ShaPublicCol::COUNT];
        for row in 0..SHA_ROW_COUNT {
            columns[ShaPublicCol::K.index()][row] = production_sha_k_expected(row, field_cfg);
            columns[ShaPublicCol::PAIn.index()][row] = word_scalar_at_two(&pa_a[row], field_cfg);
            columns[ShaPublicCol::PEIn.index()][row] = word_scalar_at_two(&pa_e[row], field_cfg);
            columns[ShaPublicCol::PAOut.index()][row] = word_scalar_at_two(&pa_a[row], field_cfg);
            columns[ShaPublicCol::PEOut.index()][row] = word_scalar_at_two(&pa_e[row], field_cfg);
            columns[ShaPublicCol::Message.index()][row] =
                word_scalar_at_two(&message[row], field_cfg);
        }
        for selector in [
            ShaPublicCol::SInit,
            ShaPublicCol::SMsg,
            ShaPublicCol::SSched,
            ShaPublicCol::SUpd,
            ShaPublicCol::SFf,
            ShaPublicCol::SOut,
        ] {
            for row in 0..SHA_ROW_COUNT {
                columns[selector.index()][row] =
                    production_sha_selector_expected(selector, row, field_cfg);
            }
        }
        mle_table_from_columns(columns)
    }

    impl ProductionShaProjectionAdapter<TestShaZincTypes, F, TEST_DEGREE_PLUS_ONE>
        for Sha256CompressionSliceUair<ShaInt>
    {
        fn project_production_sha_public(
            _shape: &UairShape<Self>,
            public_trace: &UairTrace<'_, ShaInt, ShaInt, TEST_DEGREE_PLUS_ONE>,
            field_cfg: &<F as PrimeField>::Config,
        ) -> Result<ProjectedPublic<F>, ProductionShaError<F>> {
            let empty_witness = UairTrace {
                binary_poly: Cow::Borrowed(&[]),
                arbitrary_poly: Cow::Borrowed(&[]),
                int: Cow::Borrowed(&[]),
            };
            let pa_a = project_binary_source(
                sha_binary_col(public_trace, &empty_witness, sha256_cols::PA_A)?,
                field_cfg,
            )?;
            let pa_e = project_binary_source(
                sha_binary_col(public_trace, &empty_witness, sha256_cols::PA_E)?,
                field_cfg,
            )?;
            let message = project_binary_source(
                sha_binary_col(public_trace, &empty_witness, sha256_cols::PA_M)?,
                field_cfg,
            )?;
            let public_columns = projected_public_from_sources(&pa_a, &pa_e, &message, field_cfg);
            Ok(ProjectedPublic {
                columns: public_columns,
                bit_slices: Some(flatten_bit_columns(vec![
                    pa_a.clone(),
                    pa_e.clone(),
                    pa_a,
                    pa_e,
                    message,
                ])),
            })
        }

        fn project_production_sha_witness(
            _shape: &UairShape<Self>,
            public_trace: &UairTrace<'_, ShaInt, ShaInt, TEST_DEGREE_PLUS_ONE>,
            witness_trace: &UairTrace<'_, ShaInt, ShaInt, TEST_DEGREE_PLUS_ONE>,
            field_cfg: &<F as PrimeField>::Config,
        ) -> Result<
            (
                ProjectedTrace<F>,
                ProjectedPublic<F>,
                ProductionShaWitnessPolys<TestShaZincTypes, TEST_DEGREE_PLUS_ONE>,
            ),
            ProductionShaError<F>,
        > {
            let word_sources = [
                sha256_cols::W_A,
                sha256_cols::W_E,
                sha256_cols::W_SIG0,
                sha256_cols::W_SIG1,
                sha256_cols::W_W,
                sha256_cols::W_LSIG0,
                sha256_cols::W_LSIG1,
                sha256_cols::W_U_EF,
                sha256_cols::W_U_NEG_E_G,
                sha256_cols::W_MAJ,
                sha256_cols::W_MU_PACKED,
                sha256_cols::PA_OV_SIG0,
                sha256_cols::PA_OV_SIG1,
                sha256_cols::PA_OV_LSIG0,
                sha256_cols::PA_OV_LSIG1,
                sha256_cols::PA_R_CH2_COMP,
                sha256_cols::PA_R_MAJ_COMP,
            ];
            let int_sources = [
                sha256_cols::PA_C_C7,
                sha256_cols::PA_C_C8,
                sha256_cols::PA_C_C9,
                sha256_cols::PA_C_FF_A,
                sha256_cols::PA_C_FF_E,
            ];

            let bit_columns = word_sources
                .iter()
                .map(|&col| {
                    project_binary_source(
                        sha_binary_col(public_trace, witness_trace, col)?,
                        field_cfg,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let bit_slices = flatten_bit_columns(bit_columns.clone());
            let scalarized = scalarize_bit_slices_plain(
                &bit_slices,
                &F::from_with_cfg(2u64, field_cfg),
                field_cfg,
            )?;
            let pa_a = project_binary_source(
                sha_binary_col(public_trace, witness_trace, sha256_cols::PA_A)?,
                field_cfg,
            )?;
            let pa_e = project_binary_source(
                sha_binary_col(public_trace, witness_trace, sha256_cols::PA_E)?,
                field_cfg,
            )?;
            let message = project_binary_source(
                sha_binary_col(public_trace, witness_trace, sha256_cols::PA_M)?,
                field_cfg,
            )?;
            let public_columns = projected_public_from_sources(&pa_a, &pa_e, &message, field_cfg);
            let int_columns = int_sources
                .iter()
                .map(|&col| {
                    project_int_source(sha_int_col(public_trace, witness_trace, col)?, field_cfg)
                })
                .collect::<Result<Vec<_>, _>>()?;

            let trace = ProjectedTrace {
                bit_slices,
                scalarized,
                int_columns: mle_table_from_columns(int_columns.clone()),
                public_columns: public_columns.clone(),
            };
            let public = ProjectedPublic {
                columns: public_columns,
                bit_slices: Some(flatten_bit_columns(vec![
                    pa_a.clone(),
                    pa_e.clone(),
                    pa_a,
                    pa_e,
                    message,
                ])),
            };
            Ok((
                trace,
                public,
                ProductionShaWitnessPolys {
                    binary: word_sources
                        .iter()
                        .map(|&col| {
                            truncate_sha_row_domain(
                                sha_binary_col(public_trace, witness_trace, col)?,
                                "SHA binary witness row-domain projection",
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                    arbitrary: Vec::new(),
                    int: int_sources
                        .iter()
                        .map(|&col| {
                            truncate_sha_row_domain(
                                sha_int_col(public_trace, witness_trace, col)?,
                                "SHA int witness row-domain projection",
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                },
            ))
        }
    }

    fn zero_trace_with_scalar_challenge(a: &F) -> ProjectedTrace<F> {
        let field_cfg = cfg();
        let zero = F::zero_with_cfg(&field_cfg);
        let bit_slices =
            flatten_bit_columns(vec![
                vec![vec![zero.clone(); SHA_WORD_BITS]; SHA_ROW_COUNT];
                ShaWordCol::COUNT
            ]);
        let scalarized = scalarize_bit_slices_plain(&bit_slices, a, &field_cfg).unwrap();
        ProjectedTrace {
            bit_slices,
            scalarized,
            int_columns: mle_table_from_columns(vec![
                vec![zero.clone(); SHA_ROW_COUNT];
                ShaIntCol::COUNT
            ]),
            public_columns: mle_table_from_columns(vec![
                vec![zero; SHA_ROW_COUNT];
                ShaPublicCol::COUNT
            ]),
        }
    }

    fn zero_public() -> ProjectedPublic<F> {
        let field_cfg = cfg();
        ProjectedPublic {
            columns: mle_table_from_columns(vec![
                vec![F::zero_with_cfg(&field_cfg); SHA_ROW_COUNT];
                ShaPublicCol::COUNT
            ]),
            bit_slices: Some(flatten_bit_columns(vec![
                vec![
                    vec![
                        F::zero_with_cfg(
                            &field_cfg
                        );
                        SHA_WORD_BITS
                    ];
                    SHA_ROW_COUNT
                ];
                ShaPublicWordCol::COUNT
            ])),
        }
    }

    fn fixed_layout_public() -> ProjectedPublic<F> {
        let field_cfg = cfg();
        let mut public = zero_public();
        for selector in [
            ShaPublicCol::SInit,
            ShaPublicCol::SMsg,
            ShaPublicCol::SSched,
            ShaPublicCol::SUpd,
            ShaPublicCol::SFf,
            ShaPublicCol::SOut,
        ] {
            for row in 0..SHA_ROW_COUNT {
                public.columns[selector.index()].evaluations[row] =
                    production_sha_selector_expected(selector, row, &field_cfg);
            }
        }
        for row in 0..SHA_ROW_COUNT {
            public.columns[ShaPublicCol::K.index()].evaluations[row] =
                production_sha_k_expected(row, &field_cfg);
        }
        public
    }

    fn sparse_r_ic() -> [F; SHA_ROW_VARS] {
        std::array::from_fn(|idx| f(idx as u64 + 2))
    }

    fn rescalarize_endpoint_source(source: &mut ShaSourceEndpointEval<F>, a: &F) {
        let field_cfg = cfg();
        let powers = zinc_utils::powers(a.clone(), F::one_with_cfg(&field_cfg), 32);
        source.scalarized = source
            .bits
            .iter()
            .zip(powers.iter())
            .fold(F::zero_with_cfg(&field_cfg), |acc, (bit, power)| {
                acc + bit.clone() * power
            });
    }

    fn endpoint_source(col: ShaWordCol, shift: usize, seed: u64) -> ShaSourceEndpointEval<F> {
        let field_cfg = cfg();
        let bits = std::array::from_fn(|idx| f(seed + idx as u64 + 1));
        let powers = zinc_utils::powers(f(7), F::one_with_cfg(&field_cfg), 32);
        let scalarized = bits
            .iter()
            .zip(powers.iter())
            .fold(F::zero_with_cfg(&field_cfg), |acc, (bit, power)| {
                acc + bit.clone() * power
            });
        ShaSourceEndpointEval {
            col,
            shift,
            scalarized,
            bits,
        }
    }

    fn endpoint_evals_for_virtuals() -> ShaEndpointEvals<F> {
        ShaEndpointEvals {
            sources: vec![
                endpoint_source(ShaWordCol::E, 0, 10),
                endpoint_source(ShaWordCol::E, 1, 20),
                endpoint_source(ShaWordCol::E, 2, 30),
                endpoint_source(ShaWordCol::A, 0, 40),
                endpoint_source(ShaWordCol::A, 1, 50),
                endpoint_source(ShaWordCol::A, 2, 60),
                endpoint_source(ShaWordCol::Uef, 2, 70),
                endpoint_source(ShaWordCol::UNegEg, 2, 80),
                endpoint_source(ShaWordCol::Ch2Comp, 0, 90),
                endpoint_source(ShaWordCol::Maj, 2, 100),
                endpoint_source(ShaWordCol::MajComp, 0, 110),
            ],
            int_sources: Vec::new(),
        }
    }

    fn hyrax_key_pair<C, Lanes>(
        width: usize,
        offset: u64,
    ) -> (
        zip_plus::pcs::hyrax::HyraxCommitmentKey<C>,
        zip_plus::pcs::hyrax::HyraxVerifierKey<C>,
    )
    where
        C: AffineRepr,
        Lanes: Clone + Debug + Send + Sync,
    {
        let generator = C::Group::generator();
        let bases = (0..width)
            .map(|idx| {
                let scalar = C::ScalarField::from(
                    offset + u64::try_from(idx).expect("Hyrax basis index fits u64") + 1,
                );
                (generator * scalar).into_affine()
            })
            .collect::<Vec<_>>();
        let h = generator
            * C::ScalarField::from(
                offset + u64::try_from(width).expect("Hyrax width fits u64") + 1,
            );
        HyraxPCS::<C, Lanes>::setup_from_bases_with_blinding(
            width,
            bases,
            h,
            HyraxBlindingMode::Unblinded,
        )
        .expect("Hyrax test setup must be valid")
    }

    fn all_hyrax_test_pcs_params<C>() -> (
        PCSParams<AllHyraxPCSTypes<C>, TestShaZincTypes, F, TEST_DEGREE_PLUS_ONE>,
        PCSVerifierParams<AllHyraxPCSTypes<C>, TestShaZincTypes, F, TEST_DEGREE_PLUS_ONE>,
    )
    where
        C: AffineRepr,
        AllHyraxPCSTypes<C>: ZincPCSTypes<
                TestShaZincTypes,
                F,
                TEST_DEGREE_PLUS_ONE,
                BinaryPCS = HyraxPCS<C, BinaryLanes>,
                ArbitraryPCS = HyraxPCS<C, DensePolyScalarLanes>,
                IntPCS = HyraxPCS<C, IntScalarLane>,
            >,
    {
        let width = SHA_ROW_COUNT;
        let (binary_ck, binary_vk) = hyrax_key_pair::<C, BinaryLanes>(width, 0);
        let (arbitrary_ck, arbitrary_vk) = hyrax_key_pair::<C, DensePolyScalarLanes>(width, 1_000);
        let (int_ck, int_vk) = hyrax_key_pair::<C, IntScalarLane>(width, 2_000);

        (
            PCSParams::<AllHyraxPCSTypes<C>, TestShaZincTypes, F, TEST_DEGREE_PLUS_ONE> {
                binary: binary_ck,
                arbitrary: arbitrary_ck,
                int: int_ck,
            },
            PCSVerifierParams::<AllHyraxPCSTypes<C>, TestShaZincTypes, F, TEST_DEGREE_PLUS_ONE> {
                binary: binary_vk,
                arbitrary: arbitrary_vk,
                int: int_vk,
            },
        )
    }

    #[test]
    fn linear_ideal_fold_proves_and_verifies_eight_sha_instances_with_hyrax() {
        type C = ark_bn254::G1Affine;
        type P = AllHyraxPCSTypes<C>;
        type U = Sha256CompressionSliceUair<ShaInt>;

        let field_cfg = fixed_prime::field_cfg_from_curve_scalar::<F, Uint<TEST_FIELD_LIMBS>, C>();
        let initial_state = SHA256_INITIAL_STATE;
        let message = vec!["hello world"; 40].join(" ");
        let message_blocks = sha256_padded_message_blocks::<8>(message.as_bytes())
            .expect("test message should canonically pad to 8 SHA-256 blocks");
        let (witnesses, _final_state) =
            synthesize_sha256_chain_witnesses::<ShaInt, 8>(initial_state, message_blocks)
                .expect("SHA-256 UAIR witnesses synthesize");
        let shape = UairShape::<U>::new(SHA_ROW_VARS);
        let (pcs_params, pcs_verifier_params) = all_hyrax_test_pcs_params::<C>();
        let pp =
            LinearIdealFoldProverParams::<P, U, TestShaZincTypes, F, TEST_DEGREE_PLUS_ONE>::new(
                pcs_params,
                field_cfg.clone(),
                3,
            );
        let vs = setup_verify_linear_ideal_fold::<P, U, TestShaZincTypes, F, TEST_DEGREE_PLUS_ONE>(
            LinearIdealFoldVerifierParams::new(pcs_verifier_params, field_cfg),
            shape.clone(),
        )
        .expect("production SHA verifier setup succeeds");

        let mut prover_transcript = Blake3Transcript::new();
        let output = prove_linear_ideal_fold::<P, U, TestShaZincTypes, F, TEST_DEGREE_PLUS_ONE>(
            &pp,
            &shape,
            &witnesses,
            &mut prover_transcript,
        )
        .expect("production SHA ProjectionFold proof succeeds");

        let mut verifier_transcript = Blake3Transcript::new();
        let verified = verify_linear_ideal_fold::<P, U, TestShaZincTypes, F, TEST_DEGREE_PLUS_ONE>(
            &vs,
            &output.fresh_instances,
            &output.proof,
            &mut verifier_transcript,
        )
        .expect("production SHA ProjectionFold proof verifies");

        assert_eq!(verified.target, output.folded_instance.target);
        assert_eq!(verified.public, output.folded_instance.public);
        assert!(pcs_commitments_match::<
            P,
            TestShaZincTypes,
            F,
            TEST_DEGREE_PLUS_ONE,
        >(
            &verified.commitments, &output.folded_instance.commitments
        ));
    }

    #[test]
    fn optimized_sumfold_claim_feeds_folded_row_sumcheck_with_tail_for_eight_sha_instances() {
        type U = Sha256CompressionSliceUair<ShaInt>;

        let field_cfg = cfg();
        let initial_state = SHA256_INITIAL_STATE;
        let message = vec!["hello world"; 40].join(" ");
        let message_blocks = sha256_padded_message_blocks::<8>(message.as_bytes())
            .expect("test message should canonically pad to eight SHA-256 blocks");
        let (witnesses, _final_state) =
            synthesize_sha256_chain_witnesses::<ShaInt, 8>(initial_state, message_blocks)
                .expect("SHA-256 UAIR witnesses synthesize");
        let shape = UairShape::<U>::new(SHA_ROW_VARS);

        let (traces, publics): (Vec<_>, Vec<_>) = witnesses
            .iter()
            .map(|witness| {
                let public_trace =
                    public_uair_trace_view::<ShaInt, ShaInt, F, TEST_DEGREE_PLUS_ONE>(
                        &witness.trace,
                        &shape.signature,
                    )
                    .unwrap();
                let witness_trace =
                    witness_uair_trace_view::<ShaInt, ShaInt, F, TEST_DEGREE_PLUS_ONE>(
                        &witness.trace,
                        &shape.signature,
                    )
                    .unwrap();
                let (trace, public, _witness_polys) = U::project_production_sha_witness(
                    &shape,
                    &public_trace,
                    &witness_trace,
                    &field_cfg,
                )
                .unwrap();
                (trace, public)
            })
            .unzip();
        validate_production_sha_publics(&publics, &field_cfg).unwrap();

        let r_ic = sparse_r_ic();
        let r_ic_eq_weights = build_eq_x_r_vec(&r_ic, &field_cfg).unwrap();
        let coeff_tables = build_linear_residual_coeff_tables_with_row_weights(
            &traces,
            &publics,
            &r_ic_eq_weights,
            &field_cfg,
        )
        .unwrap();
        let beta = vec![f(13), f(17), f(19)];
        let beta_eq_weights = build_eq_x_r_vec(&beta, &field_cfg).unwrap();
        let aggregate_ideal_polys =
            beta_aggregate_nonzero_ideal_polys_with_weights(&coeff_tables, &beta_eq_weights)
                .unwrap();
        let ideal_check = IdealCheckProof {
            combined_mle_values: aggregate_ideal_polys.iter().cloned().collect(),
        };
        let aggregate_ideal_polys = aggregate_sha_ideal_polys_from_proof(&ideal_check).unwrap();
        check_aggregate_sha_ideal_membership(&aggregate_ideal_polys, &field_cfg).unwrap();

        let a = f(5);
        let lambda = f(7);
        let rho = f(11);
        let xi = f(13);
        let booleanity_sources = production_sha_booleanity_sources();
        let a_powers = build_sha_residual_eval_powers(&a, &field_cfg);
        let lambda_powers = build_sha_lambda_powers(&lambda, &field_cfg);
        let booleanity_weights =
            build_booleanity_weights(&rho, &xi, booleanity_sources.len(), &field_cfg);
        let initial_claim =
            evaluate_aggregate_sha_ideal_claim(&aggregate_ideal_polys, &a, &lambda, &field_cfg)
                .unwrap();
        let linear_accumulator = build_sha_sumfold_linear_accumulator(
            &coeff_tables,
            &a_powers,
            &lambda_powers,
            &field_cfg,
        )
        .unwrap();
        let prefix_vars = 2;
        let quadratic_prefix_accumulator = build_sha_sumfold_quadratic_prefix_accumulator(
            &traces,
            &booleanity_sources,
            prefix_vars,
            &r_ic_eq_weights,
            &booleanity_weights,
            &field_cfg,
        )
        .unwrap();
        assert_eq!(linear_accumulator.len(), traces.len());
        assert_eq!(quadratic_prefix_accumulator.len(), 18);

        let group = build_production_sha_sumfold_group_from_prefix_accumulators(
            &traces,
            &beta,
            &beta_eq_weights,
            &r_ic_eq_weights,
            &linear_accumulator,
            &quadratic_prefix_accumulator,
            &booleanity_weights,
            &booleanity_sources,
            prefix_vars,
            &field_cfg,
        )
        .unwrap();
        let mut sumfold_prover_transcript = Blake3Transcript::new();
        sumfold_prover_transcript.absorb_slice(b"sha-sumfold-row-bridge");
        let (sumfold_proof, r_b) = prove_optimized_sha_sumfold_with_weights(
            &mut sumfold_prover_transcript,
            group,
            &initial_claim,
            beta.len(),
            &field_cfg,
        )
        .unwrap();

        let mut sumfold_verifier_transcript = Blake3Transcript::new();
        sumfold_verifier_transcript.absorb_slice(b"sha-sumfold-row-bridge");
        let verified_sumfold = verify_full_sha_sumfold(
            &mut sumfold_verifier_transcript,
            &sumfold_proof,
            &initial_claim,
            beta.len(),
            &field_cfg,
        )
        .unwrap();
        assert_eq!(verified_sumfold.r_b, r_b);

        let provisional = derive_instance_fold_claim(
            &beta,
            r_b.clone(),
            F::one_with_cfg(&field_cfg),
            traces.len(),
            &field_cfg,
        )
        .unwrap();
        let (folded, folded_public) =
            fold_projected_traces(&traces, &publics, &provisional, &field_cfg).unwrap();
        let row_claim = expression_folded_row_sum_with_vectors(
            &folded.trace,
            &folded_public,
            &r_ic_eq_weights,
            &a_powers,
            &lambda_powers,
            &booleanity_weights,
            &booleanity_sources,
            &field_cfg,
        )
        .unwrap();
        let sumfold_output = derive_instance_fold_claim_from_row_claim(
            &beta,
            r_b,
            &row_claim,
            traces.len(),
            &field_cfg,
        )
        .unwrap();
        assert_eq!(verified_sumfold.c_sf, *sumfold_output.c_sf());
        assert_eq!(sumfold_output.final_round_sumcheck_claim(), &row_claim);

        let mut row_prover_transcript = Blake3Transcript::new();
        row_prover_transcript.absorb_slice(b"sha-folded-row-bridge");
        let (row_proof, row_output) = prove_expression_folded_row_sumcheck_with_output_and_vectors(
            &mut row_prover_transcript,
            &folded.trace,
            &folded_public,
            &r_ic,
            &r_ic_eq_weights,
            &a_powers,
            &lambda_powers,
            &booleanity_weights,
            &booleanity_sources,
            &field_cfg,
        )
        .unwrap();
        assert_eq!(row_proof.claimed_sums(), &[row_claim.clone()]);

        let mut row_verifier_transcript = Blake3Transcript::new();
        row_verifier_transcript.absorb_slice(b"sha-folded-row-bridge");
        let verified_row = verify_folded_row_sumcheck(
            &mut row_verifier_transcript,
            &row_proof,
            &row_claim,
            &field_cfg,
        )
        .unwrap();
        verify_folded_row_terminal_value(&verified_row, &row_output.terminal_value).unwrap();
    }

    #[test]
    fn fresh_ideal_coefficients_are_bound_before_a() {
        let field_cfg = cfg();
        let ideals = vec![std::array::from_fn(|idx| {
            DynamicPolynomialF::new_trimmed(vec![f(idx as u64 + 1), f(99)])
        })];
        let mut tampered = ideals.clone();
        tampered[0][0].coeffs[0] += f(1);

        let sample_a = |values: &[[DynamicPolynomialF<F>; NUM_NONZERO_SHA_FAMILIES]]| {
            let mut transcript = Blake3Transcript::new();
            transcript.absorb_slice(b"fresh-commitments-and-public-inputs");
            let _r_ic = sample_pre_ideal_challenge::<F>(&mut transcript, &field_cfg);
            absorb_fresh_sha_ideal_polys(&mut transcript, values, &field_cfg);
            let (a, _, _, _, _) =
                sample_post_ideal_challenges::<F>(&mut transcript, 1, &field_cfg).unwrap();
            a
        };

        assert_ne!(sample_a(&ideals), sample_a(&tampered));
    }

    #[test]
    fn fresh_ideal_absorption_binds_polynomial_slot_structure() {
        let field_cfg = cfg();
        let mut packed = vec![std::array::from_fn(|_| {
            DynamicPolynomialF::new(Vec::<F>::new())
        })];
        packed[0][0] = DynamicPolynomialF::new(vec![f(1), f(2)]);

        let mut split = vec![std::array::from_fn(|_| {
            DynamicPolynomialF::new(Vec::<F>::new())
        })];
        split[0][0] = DynamicPolynomialF::new(vec![f(1)]);
        split[0][1] = DynamicPolynomialF::new(vec![f(2)]);

        let sample_a = |values: &[[DynamicPolynomialF<F>; NUM_NONZERO_SHA_FAMILIES]]| {
            let mut transcript = Blake3Transcript::new();
            transcript.absorb_slice(b"fresh-commitments-and-public-inputs");
            let _r_ic = sample_pre_ideal_challenge::<F>(&mut transcript, &field_cfg);
            absorb_fresh_sha_ideal_polys(&mut transcript, values, &field_cfg);
            let (a, _, _, _, _) =
                sample_post_ideal_challenges::<F>(&mut transcript, 1, &field_cfg).unwrap();
            a
        };

        assert_ne!(sample_a(&packed), sample_a(&split));
    }

    #[test]
    fn aggregate_ideal_claim_matches_old_per_instance_targets() {
        let field_cfg = cfg();
        let mut ideal_polys = Vec::new();
        for instance_idx in 0..4 {
            ideal_polys.push(std::array::from_fn(|slot| {
                let family = production_sha_nonzero_families()[slot];
                match family {
                    ShaResidualFamily::R0BigSigmaA | ShaResidualFamily::R1BigSigmaE => {
                        let c = f((instance_idx * 10 + slot + 1) as u64);
                        let mut coeffs = vec![F::zero_with_cfg(&field_cfg); 33];
                        coeffs[0] = -c.clone();
                        coeffs[32] = c;
                        DynamicPolynomialF::new_trimmed(coeffs)
                    }
                    _ => {
                        let c = f((instance_idx * 10 + slot + 1) as u64);
                        DynamicPolynomialF::new_trimmed(vec![-f(2) * &c, c])
                    }
                }
            }));
        }
        let beta = vec![f(3), f(5)];
        let a = f(7);
        let lambda = f(11);

        let aggregate = beta_aggregate_sha_ideal_polys(&ideal_polys, &beta, &field_cfg).unwrap();
        let aggregate_claim =
            evaluate_aggregate_sha_ideal_claim(&aggregate, &a, &lambda, &field_cfg).unwrap();
        let fresh_targets =
            evaluate_fresh_targets_from_ideal_polys(&ideal_polys, &a, &lambda, &field_cfg).unwrap();
        let old_claim = eq_weighted_sum(&beta, &fresh_targets, &field_cfg).unwrap();

        assert_eq!(aggregate_claim, old_claim);
    }

    #[test]
    fn aggregate_ideal_membership_rejects_wrong_family_polynomial() {
        let field_cfg = cfg();
        let mut aggregate = std::array::from_fn(|slot| {
            let family = production_sha_nonzero_families()[slot];
            match family {
                ShaResidualFamily::R0BigSigmaA | ShaResidualFamily::R1BigSigmaE => {
                    let mut coeffs = vec![F::zero_with_cfg(&field_cfg); 33];
                    coeffs[0] = -f(3);
                    coeffs[32] = f(3);
                    DynamicPolynomialF::new_trimmed(coeffs)
                }
                _ => DynamicPolynomialF::new_trimmed(vec![-f(10), f(5)]),
            }
        });
        check_aggregate_sha_ideal_membership(&aggregate, &field_cfg).unwrap();

        aggregate[2] = DynamicPolynomialF::new_trimmed(vec![f(1)]);
        assert!(matches!(
            check_aggregate_sha_ideal_membership(&aggregate, &field_cfg),
            Err(ProductionShaError::ShaProjection(
                ShaProjectionError::IdealMembership
            ))
        ));
    }

    #[test]
    fn aggregate_ideal_absorption_precedes_scalarization_challenges() {
        let field_cfg = cfg();
        let aggregate = std::array::from_fn(|slot| {
            DynamicPolynomialF::new_trimmed(vec![f(slot as u64 + 1), f(slot as u64 + 2)])
        });
        let mut tampered = aggregate.clone();
        tampered[0].coeffs[0] += f(1);

        let sample_a = |values: &[DynamicPolynomialF<F>; NUM_NONZERO_SHA_FAMILIES]| {
            let mut transcript = Blake3Transcript::new();
            transcript.absorb_slice(b"fresh-commitments-and-public-inputs");
            let _r_ic = sample_pre_ideal_challenge::<F>(&mut transcript, &field_cfg);
            let _beta =
                sample_instance_batch_challenge::<F>(&mut transcript, 4, &field_cfg).unwrap();
            absorb_aggregate_sha_ideal_polys(&mut transcript, values, &field_cfg);
            let (a, _, _, _) =
                sample_post_aggregate_ideal_challenges::<F>(&mut transcript, &field_cfg);
            a
        };

        assert_ne!(sample_a(&aggregate), sample_a(&tampered));
    }

    #[test]
    fn production_public_validation_requires_fixed_selectors_and_k() {
        let field_cfg = cfg();
        let valid = fixed_layout_public();
        validate_production_sha_publics(std::slice::from_ref(&valid), &field_cfg).unwrap();

        let mut bad_selector = valid.clone();
        bad_selector.columns[ShaPublicCol::SOut.index()].evaluations[0] = f(1);
        assert!(matches!(
            validate_production_sha_publics(&[bad_selector], &field_cfg),
            Err(ProductionShaError::InvalidPublicSelector {
                col: ShaPublicCol::SOut,
                row: 0
            })
        ));

        let mut non_boolean_selector = valid.clone();
        non_boolean_selector.columns[ShaPublicCol::SInit.index()].evaluations[0] = f(2);
        assert!(matches!(
            validate_production_sha_publics(&[non_boolean_selector], &field_cfg),
            Err(ProductionShaError::NonBooleanPublicSelector {
                col: ShaPublicCol::SInit,
                row: 0
            })
        ));

        let mut bad_k = valid;
        bad_k.columns[ShaPublicCol::K.index()].evaluations[3] += f(1);
        assert!(matches!(
            validate_production_sha_publics(&[bad_k], &field_cfg),
            Err(ProductionShaError::InvalidRoundConstant { row: 3 })
        ));
    }

    #[test]
    fn sumfold_outputs_instance_fold_point_before_weights() {
        let field_cfg = cfg();
        let fresh_targets = vec![f(2), f(5), f(7), f(11)];
        let beta = vec![f(13), f(17)];

        let mut prover_transcript = Blake3Transcript::new();
        prover_transcript.absorb_slice(b"bound-before-sumfold");
        let (proof, prover_output) =
            prove_sha_sumfold_targets(&mut prover_transcript, &fresh_targets, &beta, 1, &field_cfg)
                .unwrap();

        let mut verifier_transcript = Blake3Transcript::new();
        verifier_transcript.absorb_slice(b"bound-before-sumfold");
        let verifier_output = verify_sha_sumfold_targets(
            &mut verifier_transcript,
            &proof,
            &fresh_targets,
            &beta,
            &field_cfg,
        )
        .unwrap();

        assert_eq!(verifier_output, prover_output);
        assert_eq!(
            prover_output.eq_instance_weights(),
            build_eq_x_r_vec(prover_output.r_b(), &field_cfg).unwrap()
        );
        assert_eq!(
            prover_output.eq_instance_weights().len(),
            fresh_targets.len()
        );

        let d = eq_eval(&beta, prover_output.r_b(), F::one_with_cfg(&field_cfg)).unwrap();
        assert_eq!(
            prover_output.c_sf(),
            &(d * prover_output.final_round_sumcheck_claim())
        );

        let mut bad_targets = fresh_targets;
        bad_targets[0] += f(1);
        let mut bad_transcript = Blake3Transcript::new();
        bad_transcript.absorb_slice(b"bound-before-sumfold");
        assert!(
            verify_sha_sumfold_targets(
                &mut bad_transcript,
                &proof,
                &bad_targets,
                &beta,
                &field_cfg
            )
            .is_err()
        );
    }

    #[test]
    fn sumfold_verifier_rejects_extra_groups() {
        let field_cfg = cfg();
        let fresh_targets = vec![f(2), f(5), f(7), f(11)];
        let beta = vec![f(13), f(17)];
        let claims =
            zinc_piop::neutron_nova::LinearInstanceClaims::new(fresh_targets.clone()).unwrap();
        let group_0 = claims
            .build_hybrid_sumcheck_group(&beta, 1, &field_cfg)
            .unwrap();
        let group_1 = claims
            .build_hybrid_sumcheck_group(&beta, 1, &field_cfg)
            .unwrap();

        let mut prover_transcript = Blake3Transcript::new();
        prover_transcript.absorb_slice(b"bound-before-sumfold");
        let (proof, _) = MultiDegreeSumcheck::prove_as_subprotocol(
            &mut prover_transcript,
            vec![group_0, group_1],
            claims.ell(),
            &field_cfg,
        );

        let mut verifier_transcript = Blake3Transcript::new();
        verifier_transcript.absorb_slice(b"bound-before-sumfold");
        assert!(matches!(
            verify_sha_sumfold_targets(
                &mut verifier_transcript,
                &proof,
                &fresh_targets,
                &beta,
                &field_cfg,
            ),
            Err(ProductionShaError::UnexpectedSumcheckGroupCount {
                label: "SHA SumFold",
                got: 2
            })
        ));
    }

    #[test]
    fn full_sha_sumfold_derives_fold_weights_after_instance_sumcheck() {
        let field_cfg = cfg();
        let a = f(5);
        let traces = vec![
            zero_trace_with_scalar_challenge(&a),
            zero_trace_with_scalar_challenge(&a),
        ];
        let publics = vec![zero_public(), zero_public()];
        let beta = vec![f(13)];
        let r_ic = sparse_r_ic();
        let lambda = f(17);
        let rho = f(19);
        let xi = f(23);
        let booleanity_sources = vec![
            ShaBooleanitySource::WordBit {
                col: ShaWordCol::A,
                bit: 0,
            },
            ShaBooleanitySource::VirtualMaj { bit: 0 },
        ];
        let initial_claim = F::zero_with_cfg(&field_cfg);

        let mut prover_transcript = Blake3Transcript::new();
        prover_transcript.absorb_slice(b"full-sha-sumfold-context");
        let (proof, prover_output) = prove_full_sha_sumfold(
            &mut prover_transcript,
            &traces,
            &publics,
            &initial_claim,
            &beta,
            &r_ic,
            &a,
            &lambda,
            &rho,
            &xi,
            &booleanity_sources,
            &field_cfg,
        )
        .unwrap();

        let mut verifier_transcript = Blake3Transcript::new();
        verifier_transcript.absorb_slice(b"full-sha-sumfold-context");
        let verified_sumfold = verify_full_sha_sumfold(
            &mut verifier_transcript,
            &proof,
            &initial_claim,
            beta.len(),
            &field_cfg,
        )
        .unwrap();
        let verifier_output = derive_instance_fold_claim(
            &beta,
            verified_sumfold.r_b,
            verified_sumfold.c_sf,
            traces.len(),
            &field_cfg,
        )
        .unwrap();

        assert_eq!(verifier_output, prover_output);
        assert_eq!(
            prover_output.eq_instance_weights(),
            build_eq_x_r_vec(prover_output.r_b(), &field_cfg).unwrap()
        );
        assert_eq!(prover_output.eq_instance_weights().len(), traces.len());

        let (folded, folded_public) =
            fold_projected_traces(&traces, &publics, &prover_output, &field_cfg).unwrap();
        let folded_sum = expression_folded_row_sum(
            &folded.trace,
            &folded_public,
            &r_ic,
            &a,
            &lambda,
            &rho,
            &xi,
            &booleanity_sources,
            &field_cfg,
        )
        .unwrap();
        assert_eq!(prover_output.final_round_sumcheck_claim(), &folded_sum);

        let mut bad_transcript = Blake3Transcript::new();
        bad_transcript.absorb_slice(b"full-sha-sumfold-context");
        assert!(
            verify_full_sha_sumfold(&mut bad_transcript, &proof, &f(1), beta.len(), &field_cfg)
                .is_err()
        );
    }

    #[test]
    fn folded_row_sumcheck_claim_matches_folded_integrand_sum() {
        let field_cfg = cfg();
        let row_integrand_values = (0..(1usize << SHA_ROW_VARS))
            .map(|idx| f((idx as u64).wrapping_mul(3) + 1))
            .collect::<Vec<_>>();
        let final_round_sumcheck_claim =
            folded_row_integrand_sum(&row_integrand_values, &field_cfg).unwrap();

        let mut prover_transcript = Blake3Transcript::new();
        prover_transcript.absorb_slice(b"folded-row-context");
        let proof = prove_folded_row_sumcheck(
            &mut prover_transcript,
            &row_integrand_values,
            &final_round_sumcheck_claim,
            &field_cfg,
        )
        .unwrap();
        assert_eq!(proof.claimed_sums(), &[final_round_sumcheck_claim.clone()]);

        let mut verifier_transcript = Blake3Transcript::new();
        verifier_transcript.absorb_slice(b"folded-row-context");
        let output = verify_folded_row_sumcheck(
            &mut verifier_transcript,
            &proof,
            &final_round_sumcheck_claim,
            &field_cfg,
        )
        .unwrap();
        assert_eq!(output.r_star.len(), SHA_ROW_VARS);

        let row_weights = build_eq_x_r_vec(&output.r_star, &field_cfg).unwrap();
        let terminal = row_weights
            .iter()
            .zip(row_integrand_values.iter())
            .fold(F::zero_with_cfg(&field_cfg), |acc, (weight, value)| {
                acc + weight.clone() * value
            });
        verify_folded_row_terminal_value(&output, &terminal).unwrap();

        let mut bad_terminal = terminal;
        bad_terminal += f(1);
        assert!(verify_folded_row_terminal_value(&output, &bad_terminal).is_err());

        let mut bad_claim = final_round_sumcheck_claim;
        bad_claim += f(1);
        let mut bad_transcript = Blake3Transcript::new();
        bad_transcript.absorb_slice(b"folded-row-context");
        assert!(
            verify_folded_row_sumcheck(&mut bad_transcript, &proof, &bad_claim, &field_cfg)
                .is_err()
        );
    }

    #[test]
    fn folded_row_verifier_rejects_extra_groups() {
        let field_cfg = cfg();
        let row_integrand_values = (0..(1usize << SHA_ROW_VARS))
            .map(|idx| f((idx as u64).wrapping_mul(5) + 9))
            .collect::<Vec<_>>();
        let post_sumfold_claim =
            folded_row_integrand_sum(&row_integrand_values, &field_cfg).unwrap();
        let group_0 = build_folded_row_sumcheck_group(&row_integrand_values, &field_cfg).unwrap();
        let group_1 = build_folded_row_sumcheck_group(&row_integrand_values, &field_cfg).unwrap();

        let mut prover_transcript = Blake3Transcript::new();
        prover_transcript.absorb_slice(b"folded-row-context");
        let (proof, _) = MultiDegreeSumcheck::prove_as_subprotocol(
            &mut prover_transcript,
            vec![group_0, group_1],
            SHA_ROW_VARS,
            &field_cfg,
        );

        let mut verifier_transcript = Blake3Transcript::new();
        verifier_transcript.absorb_slice(b"folded-row-context");
        assert!(matches!(
            verify_folded_row_sumcheck(
                &mut verifier_transcript,
                &proof,
                &post_sumfold_claim,
                &field_cfg
            ),
            Err(ProductionShaError::UnexpectedSumcheckGroupCount {
                label: "folded row sumcheck",
                got: 2
            })
        ));
    }

    #[test]
    fn expression_folded_row_terminal_is_reconstructed_from_endpoints() {
        let field_cfg = cfg();
        let a = f(5);
        let trace = zero_trace_with_scalar_challenge(&a);
        let public = zero_public();
        let r_ic = sparse_r_ic();
        let lambda = f(7);
        let rho = f(11);
        let xi = f(13);
        let booleanity_sources = vec![
            ShaBooleanitySource::WordBit {
                col: ShaWordCol::A,
                bit: 0,
            },
            ShaBooleanitySource::VirtualCh1 { bit: 0 },
        ];
        let post_sumfold_claim = expression_folded_row_sum(
            &trace,
            &public,
            &r_ic,
            &a,
            &lambda,
            &rho,
            &xi,
            &booleanity_sources,
            &field_cfg,
        )
        .unwrap();

        let mut prover_transcript = Blake3Transcript::new();
        prover_transcript.absorb_slice(b"expression-row-context");
        let proof = prove_expression_folded_row_sumcheck(
            &mut prover_transcript,
            &trace,
            &public,
            &r_ic,
            &a,
            &lambda,
            &rho,
            &xi,
            &booleanity_sources,
            &post_sumfold_claim,
            &field_cfg,
        )
        .unwrap();

        let mut verifier_transcript = Blake3Transcript::new();
        verifier_transcript.absorb_slice(b"expression-row-context");
        let output = verify_folded_row_sumcheck(
            &mut verifier_transcript,
            &proof,
            &post_sumfold_claim,
            &field_cfg,
        )
        .unwrap();
        let endpoint_evals =
            build_sha_endpoint_evals_from_trace(&trace, &output.r_star, &a, &field_cfg).unwrap();
        let terminal = reconstruct_folded_row_terminal_from_endpoints(
            &endpoint_evals,
            &public,
            &r_ic,
            &output.r_star,
            &a,
            &lambda,
            &rho,
            &xi,
            &booleanity_sources,
            &field_cfg,
        )
        .unwrap();
        verify_folded_row_terminal_value(&output, &terminal).unwrap();

        let mut bad_terminal = terminal;
        bad_terminal += f(1);
        assert!(verify_folded_row_terminal_value(&output, &bad_terminal).is_err());

        let mut bad_endpoints = endpoint_evals;
        bad_endpoints.sources[0].bits[0] += f(1);
        assert!(
            reconstruct_folded_row_terminal_from_endpoints(
                &bad_endpoints,
                &public,
                &r_ic,
                &output.r_star,
                &a,
                &lambda,
                &rho,
                &xi,
                &booleanity_sources,
                &field_cfg
            )
            .is_err()
        );
    }

    #[test]
    fn endpoint_multipoint_reduces_all_sources_and_rejects_bad_openings() {
        let field_cfg = cfg();
        let a = f(5);
        let trace = zero_trace_with_scalar_challenge(&a);
        let public = zero_public();
        let r_star = vec![f(2), f(3), f(5), f(7), f(11), f(13), f(17)];
        let endpoint_evals =
            build_sha_endpoint_evals_from_trace(&trace, &r_star, &a, &field_cfg).unwrap();

        let mut prover_transcript = Blake3Transcript::new();
        prover_transcript.absorb_slice(b"endpoint-multipoint-context");
        let (proof, r_0) = prove_sha_endpoint_multipoint(
            &mut prover_transcript,
            &trace,
            &public,
            &endpoint_evals,
            &r_star,
            &field_cfg,
        )
        .unwrap();

        let mut verifier_transcript = Blake3Transcript::new();
        verifier_transcript.absorb_slice(b"endpoint-multipoint-context");
        let (subclaim, shift_specs) = verify_sha_endpoint_multipoint(
            &mut verifier_transcript,
            &proof,
            &endpoint_evals,
            &public,
            &r_star,
            &field_cfg,
        )
        .unwrap();
        assert_eq!(subclaim.sumcheck_subclaim.point, r_0);

        let layout = production_sha_multipoint_layout();
        let trace_mles = sha_multipoint_trace_mles(&trace, &public, &layout, &field_cfg).unwrap();
        let open_evals = trace_mles
            .iter()
            .map(|mle| mle.clone().evaluate_with_config(&r_0, &field_cfg).unwrap())
            .collect::<Vec<_>>();
        verify_sha_endpoint_multipoint_open_evals(&subclaim, &open_evals, &shift_specs, &field_cfg)
            .unwrap();

        let mut bad_open_evals = open_evals.clone();
        bad_open_evals[0] += f(1);
        assert!(
            verify_sha_endpoint_multipoint_open_evals(
                &subclaim,
                &bad_open_evals,
                &shift_specs,
                &field_cfg
            )
            .is_err()
        );

        let mut bad_endpoint_evals = endpoint_evals;
        bad_endpoint_evals.sources[0].bits[0] += f(1);
        rescalarize_endpoint_source(&mut bad_endpoint_evals.sources[0], &a);
        let mut bad_verifier_transcript = Blake3Transcript::new();
        bad_verifier_transcript.absorb_slice(b"endpoint-multipoint-context");
        assert!(
            verify_sha_endpoint_multipoint(
                &mut bad_verifier_transcript,
                &proof,
                &bad_endpoint_evals,
                &public,
                &r_star,
                &field_cfg
            )
            .is_err()
        );
    }

    #[test]
    fn fresh_ideal_objects_must_be_trimmed_and_degree_capped() {
        let field_cfg = cfg();
        let mut ideals = vec![std::array::from_fn(|_| {
            DynamicPolynomialF::new(Vec::<F>::new())
        })];
        ideals[0][0] = DynamicPolynomialF::new(vec![f(1), F::zero_with_cfg(&field_cfg)]);
        assert!(matches!(
            check_fresh_sha_ideal_membership(&ideals, &field_cfg),
            Err(ProductionShaError::ShaProjection(
                ShaProjectionError::NonCanonicalProofObject(_)
            ))
        ));

        let mut high_degree = vec![std::array::from_fn(|_| {
            DynamicPolynomialF::new(Vec::<F>::new())
        })];
        high_degree[0][2] = DynamicPolynomialF::new(vec![f(1); 33]);
        assert!(matches!(
            check_fresh_sha_ideal_membership(&high_degree, &field_cfg),
            Err(ProductionShaError::ShaProjection(
                ShaProjectionError::NonCanonicalProofObject(_)
            ))
        ));
    }

    #[test]
    fn endpoint_layout_must_be_exact_and_canonical() {
        let field_cfg = cfg();
        let a = f(5);
        let trace = zero_trace_with_scalar_challenge(&a);
        let r_star = vec![f(2), f(3), f(5), f(7), f(11), f(13), f(17)];
        let endpoint_evals =
            build_sha_endpoint_evals_from_trace(&trace, &r_star, &a, &field_cfg).unwrap();
        validate_sha_endpoint_layout(&endpoint_evals).unwrap();

        let mut missing = endpoint_evals.clone();
        missing.sources.pop();
        assert!(validate_sha_endpoint_layout(&missing).is_err());

        let mut reordered = endpoint_evals;
        reordered.sources.swap(0, 1);
        assert!(matches!(
            validate_sha_endpoint_layout(&reordered),
            Err(ProductionShaError::NonCanonicalProofObject(_))
        ));
    }

    #[test]
    fn pcs_lifted_evals_drive_multipoint_sources_and_recompute_publics() {
        let field_cfg = cfg();
        let mut public = zero_public();
        public.columns[ShaPublicCol::K.index()].evaluations[0] = f(99);
        let r_0 = vec![F::zero_with_cfg(&field_cfg); SHA_ROW_VARS];
        let layout = production_sha_multipoint_layout();

        let mut lifted = vec![
            DynamicPolynomialF::new_trimmed(vec![F::zero_with_cfg(&field_cfg)]);
            ShaWordCol::COUNT + ShaIntCol::COUNT
        ];
        lifted[ShaWordCol::A.index()] = DynamicPolynomialF::new_trimmed(vec![f(3)]);
        lifted[ShaWordCol::COUNT + ShaIntCol::CompSchedule.index()] =
            DynamicPolynomialF::new_trimmed(vec![f(7)]);

        let open_evals =
            multipoint_open_evals_from_pcs_lifted(&lifted, &layout, &public, &r_0, &field_cfg)
                .unwrap();
        let a0_idx = layout
            .sources
            .iter()
            .position(|source| {
                *source
                    == ShaMpSource::WordBit {
                        col: ShaWordCol::A,
                        bit: 0,
                    }
            })
            .unwrap();
        let int_idx = layout
            .sources
            .iter()
            .position(|source| {
                *source
                    == ShaMpSource::Int {
                        col: ShaIntCol::CompSchedule,
                    }
            })
            .unwrap();
        let public_idx = layout
            .sources
            .iter()
            .position(|source| {
                *source
                    == ShaMpSource::Public {
                        col: ShaPublicCol::K,
                    }
            })
            .unwrap();

        assert_eq!(open_evals[a0_idx], f(3));
        assert_eq!(open_evals[int_idx], f(7));
        assert_eq!(open_evals[public_idx], f(99));
    }

    #[test]
    fn folded_lifted_evals_must_be_canonical_and_32_bit() {
        let field_cfg = cfg();
        let mut lifted = vec![DynamicPolynomialF::ZERO; ShaWordCol::COUNT + ShaIntCol::COUNT];
        split_folded_sha_pcs_lifted_evals(&lifted).unwrap();
        ensure_production_sha_word_degree::<F, 32>().unwrap();
        assert!(matches!(
            ensure_production_sha_word_degree::<F, 8>(),
            Err(ProductionShaError::UnsupportedProductionShaWordDegree {
                got: 8,
                expected: 32
            })
        ));

        lifted[ShaWordCol::A.index()] =
            DynamicPolynomialF::new(vec![F::zero_with_cfg(&field_cfg); SHA_WORD_BITS + 1]);
        assert!(matches!(
            split_folded_sha_pcs_lifted_evals(&lifted),
            Err(ProductionShaError::NonCanonicalProofObject(_))
        ));

        lifted[ShaWordCol::A.index()] =
            DynamicPolynomialF::new(vec![f(1), F::zero_with_cfg(&field_cfg)]);
        assert!(matches!(
            split_folded_sha_pcs_lifted_evals(&lifted),
            Err(ProductionShaError::NonCanonicalProofObject(_))
        ));

        lifted[ShaWordCol::A.index()] = DynamicPolynomialF::ZERO;
        lifted[ShaWordCol::COUNT + ShaIntCol::CompSchedule.index()] =
            DynamicPolynomialF::new(vec![f(1), f(2)]);
        assert!(matches!(
            split_folded_sha_pcs_lifted_evals(&lifted),
            Err(ProductionShaError::NonCanonicalProofObject(_))
        ));
    }

    #[test]
    fn production_sha_requires_exact_commitment_batch_sizes() {
        validate_production_sha_batch_sizes::<F>(ShaWordCol::COUNT, 0, ShaIntCol::COUNT).unwrap();

        assert!(matches!(
            validate_production_sha_batch_sizes::<F>(0, 0, ShaIntCol::COUNT),
            Err(ProductionShaError::UnsupportedProductionShaPcsShape(_))
        ));
        assert!(matches!(
            validate_production_sha_batch_sizes::<F>(ShaWordCol::COUNT, 1, ShaIntCol::COUNT),
            Err(ProductionShaError::UnsupportedProductionShaPcsShape(_))
        ));
        assert!(matches!(
            validate_production_sha_batch_sizes::<F>(ShaWordCol::COUNT, 0, 0),
            Err(ProductionShaError::UnsupportedProductionShaPcsShape(_))
        ));
    }

    #[test]
    fn scalarization_and_virtual_endpoints_use_source_bits_only() {
        let field_cfg = cfg();
        let mut endpoint_evals = endpoint_evals_for_virtuals();
        verify_endpoint_scalarization(&endpoint_evals, &f(7), &field_cfg).unwrap();

        let virtuals = reconstruct_virtual_ch_maj_endpoint(&endpoint_evals, &field_cfg).unwrap();
        let two = f(2);
        for bit in 0..SHA_WORD_BITS {
            let e0 = source_bits(&endpoint_evals, ShaWordCol::E, 0).unwrap()[bit].clone();
            let e1 = source_bits(&endpoint_evals, ShaWordCol::E, 1).unwrap()[bit].clone();
            let e2 = source_bits(&endpoint_evals, ShaWordCol::E, 2).unwrap()[bit].clone();
            let a0 = source_bits(&endpoint_evals, ShaWordCol::A, 0).unwrap()[bit].clone();
            let a1 = source_bits(&endpoint_evals, ShaWordCol::A, 1).unwrap()[bit].clone();
            let a2 = source_bits(&endpoint_evals, ShaWordCol::A, 2).unwrap()[bit].clone();
            let uef2 = source_bits(&endpoint_evals, ShaWordCol::Uef, 2).unwrap()[bit].clone();
            let uneg_eg2 =
                source_bits(&endpoint_evals, ShaWordCol::UNegEg, 2).unwrap()[bit].clone();
            let ch2_comp0 =
                source_bits(&endpoint_evals, ShaWordCol::Ch2Comp, 0).unwrap()[bit].clone();
            let maj2 = source_bits(&endpoint_evals, ShaWordCol::Maj, 2).unwrap()[bit].clone();
            let maj_comp0 =
                source_bits(&endpoint_evals, ShaWordCol::MajComp, 0).unwrap()[bit].clone();

            assert_eq!(virtuals.ch1[bit], e2.clone() + e1 - two.clone() * uef2);
            assert_eq!(
                virtuals.ch2[bit],
                e2 - e0 + two.clone() * uneg_eg2 + two.clone() * ch2_comp0
            );
            assert_eq!(
                virtuals.maj[bit],
                a0 + a1 + a2 - two.clone() * maj2 - two.clone() * maj_comp0
            );
        }

        endpoint_evals.sources[0].scalarized += f(1);
        assert!(verify_endpoint_scalarization(&endpoint_evals, &f(7), &field_cfg).is_err());
    }
}
