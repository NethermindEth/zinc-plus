//! Production SHA ProjectionFold protocol helpers.
//!
//! This module is intentionally separate from the existing single-instance
//! `Proof`: production ProjectionFold has a different transcript order and
//! derives folded commitments only after SumFold fixes the instance-axis point.

use std::io::Cursor;

use crate::{
    ZincTypes,
    pcs::{
        AllHyraxPCSTypes, PCSCommitments, PCSParams, PCSProverData, PCSVerifierParams,
        ProductionShaPCS, ZincPCSTypes,
    },
};
use ark_ec::AffineRepr;
use crypto_primitives::{FromPrimitiveWithConfig, PrimeField};
use num_traits::{ConstZero, Zero};
use thiserror::Error;
use zinc_piop::{
    multipoint_eval::{
        MultipointEval, MultipointEvalError, Proof as MultipointEvalProof,
        Subclaim as MultipointSubclaim,
    },
    neutron_nova::SumFoldError,
    neutron_nova::{
        NUM_NONZERO_SHA_FAMILIES, NUM_SHA_RESIDUAL_FAMILIES, ProjectedShaPublic, ProjectedShaTrace,
        SHA_ROW_COUNT, SHA_ROW_VARS, SHA_WORD_BITS, ShaBooleanitySource, ShaIntCol,
        ShaProjectionError, ShaPublicCol, ShaResidualFamily, ShaSumFoldOutput, ShaWordCol,
        build_dense_sha_sumfold_group, build_expression_folded_row_sumcheck_group,
        build_folded_row_sumcheck_group, build_fresh_sha_ideal_cache, evaluate_fresh_sha_targets,
        finalize_sha_sumfold, fold_projected_sha_traces, folded_row_integrand_sum,
        production_sha_booleanity_sources, production_sha_nonzero_families,
        production_sha_nonzero_ideals, sha_int_at_point, sha_public_at_point,
        sha_scalarized_word_at_point, sha_word_bits_at_point, verify_folded_row_sumcheck_claim,
    },
    sumcheck::{
        SumCheckError,
        multi_degree::{MultiDegreeSumcheck, MultiDegreeSumcheckProof},
    },
};
use zinc_poly::{
    EvaluatablePolynomial, EvaluationError,
    mle::DenseMultilinearExtension,
    univariate::{
        binary::BinaryPoly, dense::DensePolynomial, dynamic::over_field::DynamicPolynomialF,
    },
    utils::{ArithErrors, build_eq_x_r_vec, eq_eval},
};
use zinc_transcript::Blake3Transcript;
use zinc_transcript::traits::{ConstTranscribable, Transcribable, Transcript};
use zinc_uair::ShiftSpec;
use zinc_uair::ideal::IdealCheck;
use zinc_utils::{
    delayed_reduction::DelayedFieldProductSum, inner_transparent_field::InnerTransparentField,
};
use zip_plus::{
    ZipError,
    pcs::{
        generic::{FoldablePCS, PCS},
        hyrax::{
            BinaryLanes, DensePolyScalarLanes, HyraxCommitment, HyraxCommitmentKey,
            HyraxFieldBridge, HyraxPCS, HyraxProverData, HyraxVerifierKey, IntScalarLane,
        },
    },
    pcs_transcript::{PcsProverTranscript, PcsVerifierTranscript},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProductionShaProof<F: PrimeField, Commitments> {
    pub instance_commitments: Vec<Commitments>,
    pub fresh_ideal_polys: Vec<[DynamicPolynomialF<F>; NUM_NONZERO_SHA_FAMILIES]>,
    pub sumfold_proof: MultiDegreeSumcheckProof<F>,
    pub folded_row_sumcheck: MultiDegreeSumcheckProof<F>,
    pub endpoint_evals: ShaEndpointEvals<F>,
    pub multipoint_eval: MultipointEvalProof<F>,
    pub folded_lifted_evals: Vec<DynamicPolynomialF<F>>,
    pub pcs_opening_bytes: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct ProductionShaWitnessPolys<Zt, const D: usize>
where
    Zt: ZincTypes<D>,
{
    pub binary: Vec<DenseMultilinearExtension<BinaryPoly<D>>>,
    pub arbitrary: Vec<DenseMultilinearExtension<DensePolynomial<Zt::Int, D>>>,
    pub int: Vec<DenseMultilinearExtension<Zt::Int>>,
}

#[derive(Clone, Debug)]
pub struct ProductionShaProverInstance<Zt, F, const D: usize>
where
    Zt: ZincTypes<D>,
    F: PrimeField,
{
    pub trace: ProjectedShaTrace<F>,
    pub public: ProjectedShaPublic<F>,
    pub witness_polys: ProductionShaWitnessPolys<Zt, D>,
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

const PRODUCTION_SHA_FRESH_BATCH_DOMAIN: &[u8] = b"PF_CONCISE_SHA256_FRESH_BATCH_V1";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldedRowSumcheckOutput<F> {
    pub r_star: Vec<F>,
    pub terminal_value: F,
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

pub trait ProductionShaOpeningPCS<Zt, F, const D: usize>:
    ProductionShaPCS<Zt, F, D> + Sized
where
    Zt: ZincTypes<D>,
    F: PrimeField,
    Self::BinaryPCS: FoldablePCS<F, BinaryPoly<D>, D>,
    Self::ArbitraryPCS: FoldablePCS<F, DensePolynomial<Zt::Int, D>, D>,
    Self::IntPCS: FoldablePCS<F, Zt::Int, D>,
{
    fn prove_folded_sha_opening(
        pcs_params: &PCSParams<Self, Zt, F, D>,
        folded_prover_data: &PCSProverData<Self, Zt, F, D>,
        folded_commitments: &PCSCommitments<Self, Zt, F, D>,
        folded_trace: &ProjectedShaTrace<F>,
        r_0: &[F],
        folded_lifted_evals: &[DynamicPolynomialF<F>],
        field_cfg: &F::Config,
    ) -> Result<Vec<u8>, ProductionShaError<F>>
    where
        F::Inner: ConstTranscribable + Transcribable,
        F::Modulus: ConstTranscribable + Transcribable;

    fn verify_folded_sha_opening(
        pcs_params: &PCSVerifierParams<Self, Zt, F, D>,
        folded_commitments: &PCSCommitments<Self, Zt, F, D>,
        r_0: &[F],
        folded_lifted_evals: &[DynamicPolynomialF<F>],
        pcs_opening_bytes: &[u8],
        field_cfg: &F::Config,
    ) -> Result<(), ProductionShaError<F>>
    where
        F::Inner: ConstTranscribable + Transcribable,
        F::Modulus: ConstTranscribable + Transcribable;
}

pub fn absorb_projected_sha_publics<F>(
    transcript: &mut impl Transcript,
    publics: &[zinc_piop::neutron_nova::ProjectedShaPublic<F>],
) where
    F: PrimeField,
    F::Inner: ConstTranscribable + Transcribable,
    F::Modulus: Transcribable,
{
    let mut buf = vec![0; F::Inner::NUM_BYTES];
    transcript.absorb_slice(b"production_sha_publics_begin");
    transcript.absorb_slice(&(publics.len() as u64).to_le_bytes());
    for (instance_idx, public) in publics.iter().enumerate() {
        transcript.absorb_slice(&(instance_idx as u64).to_le_bytes());
        transcript.absorb_slice(&(public.columns.columns.len() as u64).to_le_bytes());
        for (col_idx, col) in public.columns.columns.iter().enumerate() {
            transcript.absorb_slice(&(col_idx as u64).to_le_bytes());
            transcript.absorb_slice(&(col.len() as u64).to_le_bytes());
            transcript.absorb_random_field_slice(col, &mut buf);
        }
    }
    transcript.absorb_slice(b"production_sha_publics_end");
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
) where
    F: PrimeField,
    F::Inner: ConstTranscribable + Transcribable,
    F::Modulus: Transcribable,
{
    let mut buf = vec![0; F::Inner::NUM_BYTES];
    transcript.absorb_slice(b"production_sha_fresh_ideals_begin");
    transcript.absorb_slice(&(ideal_polys.len() as u64).to_le_bytes());
    for (instance_idx, instance) in ideal_polys.iter().enumerate() {
        transcript.absorb_slice(&(instance_idx as u64).to_le_bytes());
        transcript.absorb_slice(&(instance.len() as u64).to_le_bytes());
        for (family_idx, poly) in instance.iter().enumerate() {
            transcript.absorb_slice(&(family_idx as u64).to_le_bytes());
            transcript.absorb_slice(&(poly.coeffs.len() as u64).to_le_bytes());
            transcript.absorb_random_field_slice(&poly.coeffs, &mut buf);
        }
    }
    transcript.absorb_slice(b"production_sha_fresh_ideals_end");
}

pub fn absorb_sha_endpoint_evals<F>(
    transcript: &mut impl Transcript,
    endpoint_evals: &ShaEndpointEvals<F>,
) where
    F: PrimeField,
    F::Inner: ConstTranscribable + Transcribable,
    F::Modulus: Transcribable,
{
    let mut buf = vec![0; F::Inner::NUM_BYTES];
    transcript.absorb_slice(b"production_sha_endpoint_evals_begin");
    transcript.absorb_slice(&(endpoint_evals.sources.len() as u64).to_le_bytes());
    for source in &endpoint_evals.sources {
        transcript.absorb_slice(&(source.col.index() as u64).to_le_bytes());
        transcript.absorb_slice(&(source.shift as u64).to_le_bytes());
        transcript.absorb_random_field(&source.scalarized, &mut buf);
        transcript.absorb_random_field_slice(&source.bits, &mut buf);
    }
    transcript.absorb_slice(&(endpoint_evals.int_sources.len() as u64).to_le_bytes());
    for source in &endpoint_evals.int_sources {
        transcript.absorb_slice(&(source.col.index() as u64).to_le_bytes());
        transcript.absorb_random_field(&source.scalar, &mut buf);
    }
    transcript.absorb_slice(b"production_sha_endpoint_evals_end");
}

pub fn absorb_folded_lifted_evals<F>(
    transcript: &mut impl Transcript,
    lifted_evals: &[DynamicPolynomialF<F>],
) where
    F: PrimeField,
    F::Inner: ConstTranscribable + Transcribable,
    F::Modulus: Transcribable,
{
    let mut buf = vec![0; F::Inner::NUM_BYTES];
    transcript.absorb_slice(b"production_sha_folded_lifted_evals_begin");
    transcript.absorb_slice(&(lifted_evals.len() as u64).to_le_bytes());
    for (idx, lifted_eval) in lifted_evals.iter().enumerate() {
        transcript.absorb_slice(&(idx as u64).to_le_bytes());
        transcript.absorb_slice(&(lifted_eval.coeffs.len() as u64).to_le_bytes());
        transcript.absorb_random_field_slice(&lifted_eval.coeffs, &mut buf);
    }
    transcript.absorb_slice(b"production_sha_folded_lifted_evals_end");
}

pub fn sample_pre_ideal_challenge<F>(
    transcript: &mut impl Transcript,
    field_cfg: &F::Config,
) -> [F; SHA_ROW_VARS]
where
    F: PrimeField,
    F::Inner: ConstTranscribable,
{
    std::array::from_fn(|_| transcript.get_field_challenge(field_cfg))
}

pub fn sample_post_ideal_challenges<F>(
    transcript: &mut impl Transcript,
    instance_count: usize,
    field_cfg: &F::Config,
) -> Result<(F, F, F, F, Vec<F>), ProductionShaError<F>>
where
    F: PrimeField,
    F::Inner: ConstTranscribable,
{
    if !instance_count.is_power_of_two() {
        return Err(ProductionShaError::InstanceCountNotPowerOfTwo(
            instance_count,
        ));
    }
    let ell = usize::try_from(instance_count.trailing_zeros()).expect("ell fits usize");
    Ok((
        transcript.get_field_challenge(field_cfg),
        transcript.get_field_challenge(field_cfg),
        transcript.get_field_challenge(field_cfg),
        transcript.get_field_challenge(field_cfg),
        transcript.get_field_challenges(ell, field_cfg),
    ))
}

pub fn check_fresh_sha_ideal_membership<F>(
    ideal_polys: &[[DynamicPolynomialF<F>; NUM_NONZERO_SHA_FAMILIES]],
    field_cfg: &F::Config,
) -> Result<(), ProductionShaError<F>>
where
    F: PrimeField,
{
    validate_fresh_sha_ideal_polys_canonical(ideal_polys)?;
    let ideals = production_sha_nonzero_ideals(field_cfg);
    for values in ideal_polys {
        for (ideal, value) in ideals.iter().zip(values.iter()) {
            if !ideal
                .contains(value)
                .map_err(|_| ProductionShaError::IdealMembership)?
            {
                return Err(ProductionShaError::IdealMembership);
            }
        }
    }
    Ok(())
}

impl<Zt, F, C, const D: usize> ProductionShaOpeningPCS<Zt, F, D> for AllHyraxPCSTypes<C>
where
    Zt: ZincTypes<D>,
    F: HyraxFieldBridge<C>,
    C: AffineRepr,
    HyraxPCS<C, BinaryLanes>: PCS<
            F,
            BinaryPoly<D>,
            D,
            CommitmentKey = HyraxCommitmentKey<C>,
            VerifierKey = HyraxVerifierKey<C>,
            Commitment = HyraxCommitment<C>,
            ProverData = HyraxProverData<C>,
        > + FoldablePCS<F, BinaryPoly<D>, D>,
    HyraxPCS<C, DensePolyScalarLanes>: PCS<
            F,
            DensePolynomial<Zt::Int, D>,
            D,
            CommitmentKey = HyraxCommitmentKey<C>,
            VerifierKey = HyraxVerifierKey<C>,
            Commitment = HyraxCommitment<C>,
            ProverData = HyraxProverData<C>,
        > + FoldablePCS<F, DensePolynomial<Zt::Int, D>, D>,
    HyraxPCS<C, IntScalarLane>: PCS<
            F,
            Zt::Int,
            D,
            CommitmentKey = HyraxCommitmentKey<C>,
            VerifierKey = HyraxVerifierKey<C>,
            Commitment = HyraxCommitment<C>,
            ProverData = HyraxProverData<C>,
        > + FoldablePCS<F, Zt::Int, D>,
{
    fn prove_folded_sha_opening(
        pcs_params: &PCSParams<Self, Zt, F, D>,
        folded_prover_data: &PCSProverData<Self, Zt, F, D>,
        folded_commitments: &PCSCommitments<Self, Zt, F, D>,
        folded_trace: &ProjectedShaTrace<F>,
        r_0: &[F],
        folded_lifted_evals: &[DynamicPolynomialF<F>],
        field_cfg: &F::Config,
    ) -> Result<Vec<u8>, ProductionShaError<F>>
    where
        F::Inner: ConstTranscribable + Transcribable,
        F::Modulus: ConstTranscribable + Transcribable,
    {
        ensure_production_sha_word_degree::<F, D>()?;
        validate_production_sha_batch_sizes::<F>(
            <Self::BinaryPCS as PCS<F, BinaryPoly<D>, D>>::batch_size(&folded_commitments.binary),
            <Self::ArbitraryPCS as PCS<F, DensePolynomial<Zt::Int, D>, D>>::batch_size(
                &folded_commitments.arbitrary,
            ),
            <Self::IntPCS as PCS<F, Zt::Int, D>>::batch_size(&folded_commitments.int),
        )?;
        let (binary_lifted, int_lifted) = split_folded_sha_pcs_lifted_evals(folded_lifted_evals)?;

        let mut transcript = PcsProverTranscript {
            fs_transcript: Blake3Transcript::default(),
            stream: Cursor::default(),
        };
        let mut transcription_buf = vec![0u8; F::Inner::NUM_BYTES];

        <HyraxPCS<C, BinaryLanes> as PCS<F, BinaryPoly<D>, D>>::absorb_commitment(
            &mut transcript.fs_transcript,
            &folded_commitments.binary,
        );
        absorb_pcs_lifted_evals(
            &mut transcript.fs_transcript,
            binary_lifted,
            &mut transcription_buf,
        );
        let binary_scalar_lanes = folded_sha_binary_scalar_lanes::<C, F>(folded_trace);
        HyraxPCS::<C, BinaryLanes>::prove_open_scalar_lanes::<F, true>(
            &mut transcript,
            &pcs_params.binary,
            &binary_scalar_lanes,
            r_0,
            &folded_prover_data.binary,
            field_cfg,
        )?;

        <HyraxPCS<C, IntScalarLane> as PCS<F, Zt::Int, D>>::absorb_commitment(
            &mut transcript.fs_transcript,
            &folded_commitments.int,
        );
        absorb_pcs_lifted_evals(
            &mut transcript.fs_transcript,
            int_lifted,
            &mut transcription_buf,
        );
        let int_scalar_lanes = folded_sha_int_scalar_lanes::<C, F>(folded_trace);
        HyraxPCS::<C, IntScalarLane>::prove_open_scalar_lanes::<F, true>(
            &mut transcript,
            &pcs_params.int,
            &int_scalar_lanes,
            r_0,
            &folded_prover_data.int,
            field_cfg,
        )?;

        Ok(transcript.stream.into_inner())
    }

    fn verify_folded_sha_opening(
        pcs_params: &PCSVerifierParams<Self, Zt, F, D>,
        folded_commitments: &PCSCommitments<Self, Zt, F, D>,
        r_0: &[F],
        folded_lifted_evals: &[DynamicPolynomialF<F>],
        pcs_opening_bytes: &[u8],
        field_cfg: &F::Config,
    ) -> Result<(), ProductionShaError<F>>
    where
        F::Inner: ConstTranscribable + Transcribable,
        F::Modulus: ConstTranscribable + Transcribable,
    {
        ensure_production_sha_word_degree::<F, D>()?;
        validate_production_sha_batch_sizes::<F>(
            <Self::BinaryPCS as PCS<F, BinaryPoly<D>, D>>::batch_size(&folded_commitments.binary),
            <Self::ArbitraryPCS as PCS<F, DensePolynomial<Zt::Int, D>, D>>::batch_size(
                &folded_commitments.arbitrary,
            ),
            <Self::IntPCS as PCS<F, Zt::Int, D>>::batch_size(&folded_commitments.int),
        )?;
        let (binary_lifted, int_lifted) = split_folded_sha_pcs_lifted_evals(folded_lifted_evals)?;

        let mut transcript = PcsVerifierTranscript {
            fs_transcript: Blake3Transcript::default(),
            stream: Cursor::new(pcs_opening_bytes.to_vec()),
        };
        let mut transcription_buf = vec![0u8; F::Inner::NUM_BYTES];

        <HyraxPCS<C, BinaryLanes> as PCS<F, BinaryPoly<D>, D>>::absorb_commitment(
            &mut transcript.fs_transcript,
            &folded_commitments.binary,
        );
        absorb_pcs_lifted_evals(
            &mut transcript.fs_transcript,
            binary_lifted,
            &mut transcription_buf,
        );
        <HyraxPCS<C, BinaryLanes> as PCS<F, BinaryPoly<D>, D>>::verify_open::<true>(
            &mut transcript,
            &pcs_params.binary,
            &folded_commitments.binary,
            r_0,
            binary_lifted,
            field_cfg,
        )?;

        <HyraxPCS<C, IntScalarLane> as PCS<F, Zt::Int, D>>::absorb_commitment(
            &mut transcript.fs_transcript,
            &folded_commitments.int,
        );
        absorb_pcs_lifted_evals(
            &mut transcript.fs_transcript,
            int_lifted,
            &mut transcription_buf,
        );
        <HyraxPCS<C, IntScalarLane> as PCS<F, Zt::Int, D>>::verify_open::<true>(
            &mut transcript,
            &pcs_params.int,
            &folded_commitments.int,
            r_0,
            int_lifted,
            field_cfg,
        )?;

        if transcript.stream.position() != pcs_opening_bytes.len() as u64 {
            return Err(ProductionShaError::TrailingPcsOpeningBytes);
        }
        Ok(())
    }
}

fn validate_fresh_sha_ideal_polys_canonical<F>(
    ideal_polys: &[[DynamicPolynomialF<F>; NUM_NONZERO_SHA_FAMILIES]],
) -> Result<(), ProductionShaError<F>>
where
    F: PrimeField,
{
    for instance in ideal_polys {
        for (slot, poly) in instance.iter().enumerate() {
            if poly.coeffs.last().is_some_and(F::is_zero) {
                return Err(ProductionShaError::NonCanonicalProofObject(
                    "fresh ideal polynomial has trailing zero coefficients",
                ));
            }
            let family = production_sha_nonzero_families()[slot];
            let max_degree = match family {
                ShaResidualFamily::R0BigSigmaA | ShaResidualFamily::R1BigSigmaE => 61,
                ShaResidualFamily::R4Schedule
                | ShaResidualFamily::R5UpdateA
                | ShaResidualFamily::R6UpdateE
                | ShaResidualFamily::R9FeedForwardA
                | ShaResidualFamily::R10FeedForwardE => 31,
                _ => {
                    return Err(ProductionShaError::NonCanonicalProofObject(
                        "unexpected nonzero SHA ideal family",
                    ));
                }
            };
            if poly.coeffs.len() > max_degree + 1 {
                return Err(ProductionShaError::NonCanonicalProofObject(
                    "fresh ideal polynomial exceeds production degree cap",
                ));
            }
        }
    }
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
    P: ProductionShaPCS<Zt, F, D>,
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
pub fn prove_production_sha_core<P, Zt, F, const D: usize>(
    transcript: &mut impl Transcript,
    pcs_params: &PCSParams<P, Zt, F, D>,
    instances: &[ProductionShaProverInstance<Zt, F, D>],
    field_cfg: &F::Config,
) -> Result<ProductionShaProof<F, PCSCommitments<P, Zt, F, D>>, ProductionShaError<F>>
where
    Zt: ZincTypes<D>,
    F: InnerTransparentField
        + DelayedFieldProductSum
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: ConstTranscribable + Transcribable + num_traits::Zero + Default + Send + Sync,
    F::Modulus: ConstTranscribable + Transcribable,
    P: ProductionShaOpeningPCS<Zt, F, D>,
    P::BinaryPCS: FoldablePCS<F, BinaryPoly<D>, D>,
    P::ArbitraryPCS: FoldablePCS<F, DensePolynomial<Zt::Int, D>, D>,
    P::IntPCS: FoldablePCS<F, Zt::Int, D>,
{
    ensure_production_sha_word_degree::<F, D>()?;
    if instances.len() < 2 {
        return Err(ProductionShaError::InstanceCountTooSmall(instances.len()));
    }
    if !instances.len().is_power_of_two() {
        return Err(ProductionShaError::InstanceCountNotPowerOfTwo(
            instances.len(),
        ));
    }
    let booleanity_sources = production_sha_booleanity_sources();
    absorb_production_sha_statement_metadata(transcript);

    let mut prover_data = Vec::with_capacity(instances.len());
    let mut instance_commitments = Vec::with_capacity(instances.len());
    for instance in instances {
        let (data, commitment) =
            commit_production_sha_instance::<P, Zt, F, D>(pcs_params, &instance.witness_polys)?;
        prover_data.push(data);
        instance_commitments.push(commitment);
    }

    let traces = instances
        .iter()
        .map(|instance| instance.trace.clone())
        .collect::<Vec<_>>();
    let publics = instances
        .iter()
        .map(|instance| instance.public.clone())
        .collect::<Vec<_>>();
    validate_production_sha_publics(&publics, field_cfg)?;

    absorb_production_sha_commitments::<P, Zt, F, D>(
        transcript,
        b"production_sha_fresh_commitments",
        &instance_commitments,
    );
    absorb_projected_sha_publics(transcript, &publics);

    let r_ic = sample_pre_ideal_challenge(transcript, field_cfg);
    let mut ideal_cache = build_fresh_sha_ideal_cache(&traces, &publics, r_ic.clone(), field_cfg)?;
    absorb_fresh_sha_ideal_polys(transcript, &ideal_cache.ideal_polys);
    check_fresh_sha_ideal_membership(&ideal_cache.ideal_polys, field_cfg)?;

    let (a, lambda, rho, xi, beta) =
        sample_post_ideal_challenges(transcript, instances.len(), field_cfg)?;
    evaluate_fresh_sha_targets(&mut ideal_cache, &a, &lambda, field_cfg)?;
    let initial_claim = eq_weighted_sum(&beta, &ideal_cache.fresh_targets, field_cfg)?;

    let (sumfold_proof, sumfold_output) = prove_full_sha_sumfold(
        transcript,
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
        field_cfg,
    )?;

    let (folded, folded_public) =
        fold_projected_sha_traces(&traces, &publics, &sumfold_output, field_cfg)?;
    let folded_commitments = fold_pcs_commitments::<P, Zt, F, D>(
        &instance_commitments,
        sumfold_output.theta(),
        field_cfg,
    )?;
    let folded_prover_data =
        fold_pcs_prover_data::<P, Zt, F, D>(&prover_data, sumfold_output.theta(), field_cfg)?;
    absorb_production_sha_commitments::<P, Zt, F, D>(
        transcript,
        b"production_sha_derived_folded_commitments",
        std::slice::from_ref(&folded_commitments),
    );

    let (folded_row_sumcheck, row_output) = prove_expression_folded_row_sumcheck_with_output(
        transcript,
        &folded.trace,
        &folded_public,
        &r_ic,
        &a,
        &lambda,
        &rho,
        &xi,
        &booleanity_sources,
        sumfold_output.t_prime(),
        field_cfg,
    )?;
    let endpoint_evals =
        build_sha_endpoint_evals_from_trace(&folded.trace, &row_output.r_star, field_cfg)?;
    absorb_sha_endpoint_evals(transcript, &endpoint_evals);
    let terminal = reconstruct_folded_row_terminal_from_endpoints(
        &endpoint_evals,
        &folded_public,
        &r_ic,
        &row_output.r_star,
        &a,
        &lambda,
        &rho,
        &xi,
        &booleanity_sources,
        field_cfg,
    )?;
    verify_folded_row_terminal_value(&row_output, &terminal)?;

    let (multipoint_eval, r_0) = prove_sha_endpoint_multipoint(
        transcript,
        &folded.trace,
        &folded_public,
        &endpoint_evals,
        &row_output.r_star,
        field_cfg,
    )?;
    let folded_lifted_evals = build_folded_sha_pcs_lifted_evals(&folded.trace, &r_0, field_cfg)?;
    absorb_folded_lifted_evals(transcript, &folded_lifted_evals);
    let pcs_opening_bytes = P::prove_folded_sha_opening(
        pcs_params,
        &folded_prover_data,
        &folded_commitments,
        &folded.trace,
        &r_0,
        &folded_lifted_evals,
        field_cfg,
    )?;
    transcript.absorb_slice(b"production_sha_pcs_opening_bytes");
    transcript.absorb_slice(&(pcs_opening_bytes.len() as u64).to_le_bytes());
    transcript.absorb_slice(&pcs_opening_bytes);

    Ok(ProductionShaProof {
        instance_commitments,
        fresh_ideal_polys: ideal_cache.ideal_polys,
        sumfold_proof,
        folded_row_sumcheck,
        endpoint_evals,
        multipoint_eval,
        folded_lifted_evals,
        pcs_opening_bytes,
    })
}

pub fn verify_production_sha_core<P, Zt, F, const D: usize>(
    transcript: &mut impl Transcript,
    pcs_params: &PCSVerifierParams<P, Zt, F, D>,
    proof: &ProductionShaProof<F, PCSCommitments<P, Zt, F, D>>,
    publics: &[ProjectedShaPublic<F>],
    field_cfg: &F::Config,
) -> Result<(), ProductionShaError<F>>
where
    Zt: ZincTypes<D>,
    F: InnerTransparentField
        + DelayedFieldProductSum
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: ConstTranscribable + Transcribable + num_traits::Zero + Default + Send + Sync,
    F::Modulus: ConstTranscribable + Transcribable,
    P: ProductionShaOpeningPCS<Zt, F, D>,
    P::BinaryPCS: FoldablePCS<F, BinaryPoly<D>, D>,
    P::ArbitraryPCS: FoldablePCS<F, DensePolynomial<Zt::Int, D>, D>,
    P::IntPCS: FoldablePCS<F, Zt::Int, D>,
{
    ensure_production_sha_word_degree::<F, D>()?;
    if publics.len() < 2 {
        return Err(ProductionShaError::InstanceCountTooSmall(publics.len()));
    }
    if !publics.len().is_power_of_two() {
        return Err(ProductionShaError::InstanceCountNotPowerOfTwo(
            publics.len(),
        ));
    }
    if proof.instance_commitments.len() != publics.len() {
        return Err(ProductionShaError::LengthMismatch {
            label: "commitments/publics",
            got: proof.instance_commitments.len(),
            expected: publics.len(),
        });
    }
    if proof.fresh_ideal_polys.len() != publics.len() {
        return Err(ProductionShaError::LengthMismatch {
            label: "fresh ideals/publics",
            got: proof.fresh_ideal_polys.len(),
            expected: publics.len(),
        });
    }
    validate_production_sha_publics(publics, field_cfg)?;
    let booleanity_sources = production_sha_booleanity_sources();
    absorb_production_sha_statement_metadata(transcript);

    absorb_production_sha_commitments::<P, Zt, F, D>(
        transcript,
        b"production_sha_fresh_commitments",
        &proof.instance_commitments,
    );
    absorb_projected_sha_publics(transcript, publics);

    let r_ic = sample_pre_ideal_challenge(transcript, field_cfg);
    absorb_fresh_sha_ideal_polys(transcript, &proof.fresh_ideal_polys);
    check_fresh_sha_ideal_membership(&proof.fresh_ideal_polys, field_cfg)?;

    let (a, lambda, rho, xi, beta) =
        sample_post_ideal_challenges(transcript, publics.len(), field_cfg)?;
    let fresh_targets =
        evaluate_fresh_targets_from_ideal_polys(&proof.fresh_ideal_polys, &a, &lambda, field_cfg)?;
    let initial_claim = eq_weighted_sum(&beta, &fresh_targets, field_cfg)?;

    let sumfold_output = verify_full_sha_sumfold(
        transcript,
        &proof.sumfold_proof,
        &initial_claim,
        &beta,
        publics.len(),
        field_cfg,
    )?;
    let folded_commitments = fold_pcs_commitments::<P, Zt, F, D>(
        &proof.instance_commitments,
        sumfold_output.theta(),
        field_cfg,
    )?;
    absorb_production_sha_commitments::<P, Zt, F, D>(
        transcript,
        b"production_sha_derived_folded_commitments",
        std::slice::from_ref(&folded_commitments),
    );

    let row_output = verify_folded_row_sumcheck(
        transcript,
        &proof.folded_row_sumcheck,
        sumfold_output.t_prime(),
        field_cfg,
    )?;
    absorb_sha_endpoint_evals(transcript, &proof.endpoint_evals);
    let folded_public = fold_projected_sha_publics(publics, sumfold_output.theta(), field_cfg)?;
    let terminal = reconstruct_folded_row_terminal_from_endpoints(
        &proof.endpoint_evals,
        &folded_public,
        &r_ic,
        &row_output.r_star,
        &a,
        &lambda,
        &rho,
        &xi,
        &booleanity_sources,
        field_cfg,
    )?;
    verify_folded_row_terminal_value(&row_output, &terminal)?;

    let (subclaim, shift_specs) = verify_sha_endpoint_multipoint(
        transcript,
        &proof.multipoint_eval,
        &proof.endpoint_evals,
        &folded_public,
        &row_output.r_star,
        field_cfg,
    )?;
    let open_evals = multipoint_open_evals_from_pcs_lifted(
        &proof.folded_lifted_evals,
        &production_sha_multipoint_layout(),
        &folded_public,
        &subclaim.sumcheck_subclaim.point,
        field_cfg,
    )?;
    verify_sha_endpoint_multipoint_open_evals(&subclaim, &open_evals, &shift_specs, field_cfg)?;
    absorb_folded_lifted_evals(transcript, &proof.folded_lifted_evals);
    P::verify_folded_sha_opening(
        pcs_params,
        &folded_commitments,
        &subclaim.sumcheck_subclaim.point,
        &proof.folded_lifted_evals,
        &proof.pcs_opening_bytes,
        field_cfg,
    )?;
    transcript.absorb_slice(b"production_sha_pcs_opening_bytes");
    transcript.absorb_slice(&(proof.pcs_opening_bytes.len() as u64).to_le_bytes());
    transcript.absorb_slice(&proof.pcs_opening_bytes);

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn prove_production_sha<P, Zt, F, const D: usize>(
    transcript: &mut impl Transcript,
    pcs_params: &PCSParams<P, Zt, F, D>,
    instances: &[ProductionShaProverInstance<Zt, F, D>],
    field_cfg: &F::Config,
) -> Result<ProductionShaProof<F, PCSCommitments<P, Zt, F, D>>, ProductionShaError<F>>
where
    Zt: ZincTypes<D>,
    F: InnerTransparentField
        + DelayedFieldProductSum
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: ConstTranscribable + Transcribable + num_traits::Zero + Default + Send + Sync,
    F::Modulus: ConstTranscribable + Transcribable,
    P: ProductionShaOpeningPCS<Zt, F, D>,
    P::BinaryPCS: FoldablePCS<F, BinaryPoly<D>, D>,
    P::ArbitraryPCS: FoldablePCS<F, DensePolynomial<Zt::Int, D>, D>,
    P::IntPCS: FoldablePCS<F, Zt::Int, D>,
{
    prove_production_sha_core::<P, Zt, F, D>(transcript, pcs_params, instances, field_cfg)
}

pub fn verify_production_sha<P, Zt, F, const D: usize>(
    transcript: &mut impl Transcript,
    pcs_params: &PCSVerifierParams<P, Zt, F, D>,
    proof: &ProductionShaProof<F, PCSCommitments<P, Zt, F, D>>,
    publics: &[ProjectedShaPublic<F>],
    field_cfg: &F::Config,
) -> Result<(), ProductionShaError<F>>
where
    Zt: ZincTypes<D>,
    F: InnerTransparentField
        + DelayedFieldProductSum
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: ConstTranscribable + Transcribable + num_traits::Zero + Default + Send + Sync,
    F::Modulus: ConstTranscribable + Transcribable,
    P: ProductionShaOpeningPCS<Zt, F, D>,
    P::BinaryPCS: FoldablePCS<F, BinaryPoly<D>, D>,
    P::ArbitraryPCS: FoldablePCS<F, DensePolynomial<Zt::Int, D>, D>,
    P::IntPCS: FoldablePCS<F, Zt::Int, D>,
{
    verify_production_sha_core::<P, Zt, F, D>(transcript, pcs_params, proof, publics, field_cfg)
}

fn evaluate_fresh_targets_from_ideal_polys<F>(
    ideal_polys: &[[DynamicPolynomialF<F>; NUM_NONZERO_SHA_FAMILIES]],
    a: &F,
    lambda: &F,
    field_cfg: &F::Config,
) -> Result<Vec<F>, ProductionShaError<F>>
where
    F: PrimeField,
{
    let lambda_powers = zinc_utils::powers(
        lambda.clone(),
        F::one_with_cfg(field_cfg),
        NUM_SHA_RESIDUAL_FAMILIES,
    );
    ideal_polys
        .iter()
        .map(|instance| {
            let mut target = F::zero_with_cfg(field_cfg);
            for (slot, family) in production_sha_nonzero_families().iter().enumerate() {
                target +=
                    lambda_powers[family.index()].clone() * instance[slot].evaluate_at_point(a)?;
            }
            Ok(target)
        })
        .collect()
}

fn eq_weighted_sum<F>(
    point: &[F],
    values: &[F],
    field_cfg: &F::Config,
) -> Result<F, ProductionShaError<F>>
where
    F: PrimeField,
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
    Ok(weights
        .iter()
        .zip(values.iter())
        .fold(F::zero_with_cfg(field_cfg), |acc, (weight, value)| {
            acc + weight.clone() * value
        }))
}

fn fold_projected_sha_publics<F>(
    publics: &[ProjectedShaPublic<F>],
    theta: &[F],
    field_cfg: &F::Config,
) -> Result<ProjectedShaPublic<F>, ProductionShaError<F>>
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
    let col_count = first.columns.columns.len();
    let row_count = first
        .columns
        .columns
        .first()
        .map(|col| col.len())
        .unwrap_or(0);
    let mut columns = vec![vec![F::zero_with_cfg(field_cfg); row_count]; col_count];
    for (public, weight) in publics.iter().zip(theta.iter()) {
        if public.columns.columns.len() != col_count {
            return Err(ProductionShaError::LengthMismatch {
                label: "public column count",
                got: public.columns.columns.len(),
                expected: col_count,
            });
        }
        for (col_idx, col) in public.columns.columns.iter().enumerate() {
            if col.len() != row_count {
                return Err(ProductionShaError::LengthMismatch {
                    label: "public row count",
                    got: col.len(),
                    expected: row_count,
                });
            }
            for (out, value) in columns[col_idx].iter_mut().zip(col.iter()) {
                *out += weight.clone() * value;
            }
        }
    }
    Ok(ProjectedShaPublic {
        columns: zinc_piop::neutron_nova::ShaPublicColumns { columns },
    })
}

fn validate_production_sha_publics<F>(
    publics: &[ProjectedShaPublic<F>],
    field_cfg: &F::Config,
) -> Result<(), ProductionShaError<F>>
where
    F: PrimeField + FromPrimitiveWithConfig,
{
    for public in publics {
        if public.columns.columns.len() != ShaPublicCol::COUNT {
            return Err(ProductionShaError::LengthMismatch {
                label: "SHA public column count",
                got: public.columns.columns.len(),
                expected: ShaPublicCol::COUNT,
            });
        }
        for col in &public.columns.columns {
            if col.len() != SHA_ROW_COUNT {
                return Err(ProductionShaError::LengthMismatch {
                    label: "SHA public row count",
                    got: col.len(),
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
            let col = &public.columns.columns[selector.index()];
            for (row, value) in col.iter().enumerate() {
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

        let k_col = &public.columns.columns[ShaPublicCol::K.index()];
        for (row, value) in k_col.iter().enumerate() {
            let expected = production_sha_k_expected(row, field_cfg);
            if value != &expected {
                return Err(ProductionShaError::InvalidRoundConstant { row });
            }
        }
    }
    Ok(())
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

fn build_folded_sha_pcs_lifted_evals<F>(
    folded_trace: &ProjectedShaTrace<F>,
    r_0: &[F],
    field_cfg: &F::Config,
) -> Result<Vec<DynamicPolynomialF<F>>, ProductionShaError<F>>
where
    F: PrimeField + DelayedFieldProductSum,
{
    let mut lifted = Vec::with_capacity(ShaWordCol::COUNT + ShaIntCol::COUNT);
    for col in ShaWordCol::ALL {
        let coeffs = sha_word_bits_at_point(folded_trace, col, 0, r_0, field_cfg)?.to_vec();
        lifted.push(DynamicPolynomialF::new_trimmed(coeffs));
    }
    for col in ShaIntCol::ALL {
        lifted.push(DynamicPolynomialF::new_trimmed([sha_int_at_point(
            folded_trace,
            col,
            r_0,
            field_cfg,
        )?]));
    }
    Ok(lifted)
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

fn folded_sha_binary_scalar_lanes<C, F>(
    folded_trace: &ProjectedShaTrace<F>,
) -> Vec<Vec<Vec<C::ScalarField>>>
where
    C: AffineRepr,
    F: HyraxFieldBridge<C>,
{
    ShaWordCol::ALL
        .iter()
        .map(|col| {
            (0..32)
                .map(|bit| {
                    (0..SHA_ROW_COUNT)
                        .map(|row| {
                            F::field_to_scalar(
                                &folded_trace.bit_slices.columns[col.index()][row][bit],
                            )
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn folded_sha_int_scalar_lanes<C, F>(
    folded_trace: &ProjectedShaTrace<F>,
) -> Vec<Vec<Vec<C::ScalarField>>>
where
    C: AffineRepr,
    F: HyraxFieldBridge<C>,
{
    ShaIntCol::ALL
        .iter()
        .map(|col| {
            vec![
                (0..SHA_ROW_COUNT)
                    .map(|row| {
                        F::field_to_scalar(&folded_trace.int_columns.columns[col.index()][row])
                    })
                    .collect::<Vec<_>>(),
            ]
        })
        .collect()
}

fn absorb_pcs_lifted_evals<F>(
    transcript: &mut impl Transcript,
    lifted_evals: &[DynamicPolynomialF<F>],
    transcription_buf: &mut Vec<u8>,
) where
    F: PrimeField,
    F::Inner: ConstTranscribable + Transcribable,
    F::Modulus: Transcribable,
{
    for lifted_eval in lifted_evals {
        transcript.absorb_random_field_slice(&lifted_eval.coeffs, transcription_buf);
    }
}

fn multipoint_open_evals_from_pcs_lifted<F>(
    lifted_evals: &[DynamicPolynomialF<F>],
    layout: &ShaMultipointLayout,
    folded_public: &ProjectedShaPublic<F>,
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
) -> Result<(MultiDegreeSumcheckProof<F>, ShaSumFoldOutput<F>), ProductionShaError<F>>
where
    F: InnerTransparentField
        + DelayedFieldProductSum
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: ConstTranscribable + num_traits::Zero,
    F::Modulus: ConstTranscribable,
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
    let output = finalize_sha_sumfold(beta, r_b, c_sf, fresh_targets.len(), field_cfg)?;
    Ok((proof, output))
}

pub fn verify_sha_sumfold_targets<F>(
    transcript: &mut impl Transcript,
    proof: &MultiDegreeSumcheckProof<F>,
    fresh_targets: &[F],
    beta: &[F],
    field_cfg: &F::Config,
) -> Result<ShaSumFoldOutput<F>, ProductionShaError<F>>
where
    F: InnerTransparentField
        + DelayedFieldProductSum
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: ConstTranscribable + num_traits::Zero,
    F::Modulus: ConstTranscribable,
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
    Ok(finalize_sha_sumfold(
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
    traces: &[ProjectedShaTrace<F>],
    publics: &[ProjectedShaPublic<F>],
    initial_claim: &F,
    beta: &[F],
    r_ic: &[F; SHA_ROW_VARS],
    a: &F,
    lambda: &F,
    rho: &F,
    xi: &F,
    booleanity_sources: &[ShaBooleanitySource],
    field_cfg: &F::Config,
) -> Result<(MultiDegreeSumcheckProof<F>, ShaSumFoldOutput<F>), ProductionShaError<F>>
where
    F: InnerTransparentField
        + DelayedFieldProductSum
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: ConstTranscribable + num_traits::Zero,
    F::Modulus: ConstTranscribable,
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
    let provisional = finalize_sha_sumfold(
        beta,
        r_b.clone(),
        F::one_with_cfg(field_cfg),
        traces.len(),
        field_cfg,
    )?;
    let (folded, folded_public) = zinc_piop::neutron_nova::fold_projected_sha_traces(
        traces,
        publics,
        &provisional,
        field_cfg,
    )?;
    let t_prime = zinc_piop::neutron_nova::expression_folded_row_sum(
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
    let c_sf = d * t_prime;
    Ok((
        proof,
        finalize_sha_sumfold(beta, r_b, c_sf, traces.len(), field_cfg)?,
    ))
}

pub fn verify_full_sha_sumfold<F>(
    transcript: &mut impl Transcript,
    proof: &MultiDegreeSumcheckProof<F>,
    initial_claim: &F,
    beta: &[F],
    instance_count: usize,
    field_cfg: &F::Config,
) -> Result<ShaSumFoldOutput<F>, ProductionShaError<F>>
where
    F: InnerTransparentField
        + DelayedFieldProductSum
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: ConstTranscribable + num_traits::Zero,
    F::Modulus: ConstTranscribable,
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
        MultiDegreeSumcheck::verify_as_subprotocol(transcript, beta.len(), proof, field_cfg)?;
    let r_b = subclaims.point().to_vec();
    let c_sf = subclaims.expected_evaluations()[0].clone();
    Ok(finalize_sha_sumfold(
        beta,
        r_b,
        c_sf,
        instance_count,
        field_cfg,
    )?)
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
        .map(|commitment| commitment.binary.clone())
        .collect::<Vec<_>>();
    let arbitrary = commitments
        .iter()
        .map(|commitment| commitment.arbitrary.clone())
        .collect::<Vec<_>>();
    let int = commitments
        .iter()
        .map(|commitment| commitment.int.clone())
        .collect::<Vec<_>>();
    Ok(PCSCommitments {
        binary: P::BinaryPCS::fold_commitments(&binary, theta, field_cfg)?,
        arbitrary: P::ArbitraryPCS::fold_commitments(&arbitrary, theta, field_cfg)?,
        int: P::IntPCS::fold_commitments(&int, theta, field_cfg)?,
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
    t_prime: &F,
    field_cfg: &F::Config,
) -> Result<MultiDegreeSumcheckProof<F>, ProductionShaError<F>>
where
    F: InnerTransparentField
        + DelayedFieldProductSum
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: ConstTranscribable + num_traits::Zero,
    F::Modulus: ConstTranscribable,
{
    let claimed = folded_row_integrand_sum(row_integrand_values, field_cfg)?;
    verify_folded_row_sumcheck_claim(&claimed, t_prime)?;
    let group = build_folded_row_sumcheck_group(row_integrand_values, field_cfg)?;
    let (proof, _) =
        MultiDegreeSumcheck::prove_as_subprotocol(transcript, vec![group], SHA_ROW_VARS, field_cfg);
    Ok(proof)
}

#[allow(clippy::too_many_arguments)]
pub fn prove_expression_folded_row_sumcheck<F>(
    transcript: &mut impl Transcript,
    trace: &ProjectedShaTrace<F>,
    public: &ProjectedShaPublic<F>,
    r_ic: &[F; SHA_ROW_VARS],
    a: &F,
    lambda: &F,
    rho: &F,
    xi: &F,
    booleanity_sources: &[ShaBooleanitySource],
    t_prime: &F,
    field_cfg: &F::Config,
) -> Result<MultiDegreeSumcheckProof<F>, ProductionShaError<F>>
where
    F: InnerTransparentField
        + DelayedFieldProductSum
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: ConstTranscribable + num_traits::Zero,
    F::Modulus: ConstTranscribable,
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
    verify_folded_row_sumcheck_claim(&claimed, t_prime)?;
    let group = build_expression_folded_row_sumcheck_group(
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
    trace: &ProjectedShaTrace<F>,
    public: &ProjectedShaPublic<F>,
    r_ic: &[F; SHA_ROW_VARS],
    a: &F,
    lambda: &F,
    rho: &F,
    xi: &F,
    booleanity_sources: &[ShaBooleanitySource],
    t_prime: &F,
    field_cfg: &F::Config,
) -> Result<(MultiDegreeSumcheckProof<F>, FoldedRowSumcheckOutput<F>), ProductionShaError<F>>
where
    F: InnerTransparentField
        + DelayedFieldProductSum
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: ConstTranscribable + num_traits::Zero,
    F::Modulus: ConstTranscribable,
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
    verify_folded_row_sumcheck_claim(&claimed, t_prime)?;
    let group = build_expression_folded_row_sumcheck_group(
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
    let (proof, states) =
        MultiDegreeSumcheck::prove_as_subprotocol(transcript, vec![group], SHA_ROW_VARS, field_cfg);
    let r_star = states
        .first()
        .ok_or(ProductionShaError::LengthMismatch {
            label: "folded row states",
            got: 0,
            expected: 1,
        })?
        .randomness
        .clone();
    let endpoint_evals = build_sha_endpoint_evals_from_trace(trace, &r_star, field_cfg)?;
    let terminal_value = reconstruct_folded_row_terminal_from_endpoints(
        &endpoint_evals,
        public,
        r_ic,
        &r_star,
        a,
        lambda,
        rho,
        xi,
        booleanity_sources,
        field_cfg,
    )?;
    Ok((
        proof,
        FoldedRowSumcheckOutput {
            r_star,
            terminal_value,
        },
    ))
}

pub fn verify_folded_row_sumcheck<F>(
    transcript: &mut impl Transcript,
    proof: &MultiDegreeSumcheckProof<F>,
    t_prime: &F,
    field_cfg: &F::Config,
) -> Result<FoldedRowSumcheckOutput<F>, ProductionShaError<F>>
where
    F: InnerTransparentField
        + DelayedFieldProductSum
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: ConstTranscribable + num_traits::Zero,
    F::Modulus: ConstTranscribable,
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
    verify_folded_row_sumcheck_claim(claimed_sum, t_prime)?;
    let subclaims =
        MultiDegreeSumcheck::verify_as_subprotocol(transcript, SHA_ROW_VARS, proof, field_cfg)?;
    Ok(FoldedRowSumcheckOutput {
        r_star: subclaims.point().to_vec(),
        terminal_value: subclaims.expected_evaluations()[0].clone(),
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
    trace: &ProjectedShaTrace<F>,
    r_star: &[F],
    field_cfg: &F::Config,
) -> Result<ShaEndpointEvals<F>, ProductionShaError<F>>
where
    F: PrimeField + DelayedFieldProductSum,
{
    let mut sources = Vec::new();
    for (col, shift) in production_sha_endpoint_word_sources() {
        sources.push(ShaSourceEndpointEval {
            col,
            shift,
            scalarized: sha_scalarized_word_at_point(trace, col, shift, r_star, field_cfg)?,
            bits: sha_word_bits_at_point(trace, col, shift, r_star, field_cfg)?,
        });
    }
    let mut int_sources = Vec::new();
    for col in production_sha_endpoint_int_sources() {
        int_sources.push(ShaIntEndpointEval {
            col,
            scalar: sha_int_at_point(trace, col, r_star, field_cfg)?,
        });
    }
    Ok(ShaEndpointEvals {
        sources,
        int_sources,
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
    folded_trace: &ProjectedShaTrace<F>,
    folded_public: &ProjectedShaPublic<F>,
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
    F::Inner: ConstTranscribable + num_traits::Zero + Default + Send + Sync,
    F::Modulus: ConstTranscribable,
{
    validate_sha_endpoint_layout(endpoint_evals)?;
    let layout = production_sha_multipoint_layout();
    let trace_mles = sha_multipoint_trace_mles(folded_trace, folded_public, &layout, field_cfg)?;
    let up_evals =
        sha_multipoint_up_evals(endpoint_evals, folded_public, r_star, &layout, field_cfg)?;
    let (shift_specs, down_evals) = sha_multipoint_shift_specs_and_down_evals(
        endpoint_evals,
        folded_public,
        r_star,
        &layout,
        field_cfg,
    )?;
    let (proof, state) = MultipointEval::prove_as_subprotocol(
        transcript,
        &trace_mles,
        r_star,
        &up_evals,
        &down_evals,
        &shift_specs,
        field_cfg,
    )?;
    Ok((proof, state.eval_point))
}

pub fn verify_sha_endpoint_multipoint<F>(
    transcript: &mut impl Transcript,
    proof: &MultipointEvalProof<F>,
    endpoint_evals: &ShaEndpointEvals<F>,
    folded_public: &ProjectedShaPublic<F>,
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
    F::Inner: ConstTranscribable + num_traits::Zero + Default + Send + Sync,
    F::Modulus: ConstTranscribable,
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
    let subclaim = MultipointEval::verify_as_subprotocol(
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
    F::Inner: ConstTranscribable + num_traits::Zero + Default + Send + Sync,
    F::Modulus: ConstTranscribable,
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
    folded_public: &ProjectedShaPublic<F>,
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
    F: PrimeField,
{
    if r_star.len() != SHA_ROW_VARS {
        return Err(ProductionShaError::LengthMismatch {
            label: "r_star",
            got: r_star.len(),
            expected: SHA_ROW_VARS,
        });
    }

    validate_sha_endpoint_layout(endpoint_evals)?;
    verify_endpoint_scalarization(endpoint_evals, a, field_cfg)?;

    let residuals =
        residual_polys_from_endpoints(endpoint_evals, folded_public, r_star, field_cfg)?;
    let lambda_powers =
        zinc_utils::powers(lambda.clone(), F::one_with_cfg(field_cfg), residuals.len());
    let mut linear = F::zero_with_cfg(field_cfg);
    for (residual, weight) in residuals.iter().zip(lambda_powers.iter()) {
        linear += weight.clone() * residual.evaluate_at_point(a)?;
    }

    let rho_powers = zinc_utils::powers(
        rho.clone(),
        F::one_with_cfg(field_cfg),
        booleanity_sources.len(),
    );
    let mut bool_sum = F::zero_with_cfg(field_cfg);
    for (source, rho_power) in booleanity_sources.iter().zip(rho_powers.iter()) {
        let d = booleanity_endpoint_value(endpoint_evals, source, field_cfg)?;
        bool_sum += rho_power.clone() * d.clone() * (d - F::one_with_cfg(field_cfg));
    }

    let row_weight = eq_eval(r_ic, r_star, F::one_with_cfg(field_cfg))?;
    Ok(row_weight * (linear + xi.clone() * bool_sum))
}

pub fn verify_endpoint_scalarization<F>(
    endpoint_evals: &ShaEndpointEvals<F>,
    a: &F,
    field_cfg: &F::Config,
) -> Result<(), ProductionShaError<F>>
where
    F: PrimeField,
{
    let powers = zinc_utils::powers(a.clone(), F::one_with_cfg(field_cfg), 32);
    for source in &endpoint_evals.sources {
        let recombined = source
            .bits
            .iter()
            .zip(powers.iter())
            .fold(F::zero_with_cfg(field_cfg), |acc, (bit, power)| {
                acc + bit.clone() * power
            });
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
    folded_trace: &ProjectedShaTrace<F>,
    folded_public: &ProjectedShaPublic<F>,
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
    folded_public: &ProjectedShaPublic<F>,
    r_star: &[F],
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
            sha_mp_source_endpoint_value(endpoint_evals, folded_public, r_star, *source, field_cfg)
        })
        .collect()
}

fn sha_multipoint_shift_specs_and_down_evals<F>(
    endpoint_evals: &ShaEndpointEvals<F>,
    folded_public: &ProjectedShaPublic<F>,
    r_star: &[F],
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
                sha_public_at_point(folded_public, col, shift, r_star, field_cfg)?,
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
    folded_trace: &ProjectedShaTrace<F>,
    folded_public: &ProjectedShaPublic<F>,
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
            .columns
            .get(col.index())
            .and_then(|rows| rows.get(row))
            .and_then(|bits| bits.get(bit))
            .cloned()
            .ok_or(ProductionShaError::LengthMismatch {
                label: "multipoint word bit source",
                got: row,
                expected: SHA_ROW_COUNT,
            }),
        ShaMpSource::Int { col } => folded_trace
            .int_columns
            .columns
            .get(col.index())
            .and_then(|rows| rows.get(row))
            .cloned()
            .ok_or(ProductionShaError::LengthMismatch {
                label: "multipoint int source",
                got: row,
                expected: SHA_ROW_COUNT,
            }),
        ShaMpSource::Public { col } => folded_public
            .columns
            .columns
            .get(col.index())
            .and_then(|rows| rows.get(row))
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

fn sha_mp_source_endpoint_value<F>(
    endpoint_evals: &ShaEndpointEvals<F>,
    folded_public: &ProjectedShaPublic<F>,
    r_star: &[F],
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
        ShaMpSource::Public { col } => Ok(sha_public_at_point(
            folded_public,
            col,
            0,
            r_star,
            field_cfg,
        )?),
    }
}

fn residual_polys_from_endpoints<F>(
    endpoint_evals: &ShaEndpointEvals<F>,
    folded_public: &ProjectedShaPublic<F>,
    r_star: &[F],
    field_cfg: &F::Config,
) -> Result<Vec<DynamicPolynomialF<F>>, ProductionShaError<F>>
where
    F: PrimeField,
{
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
        - &endpoint_public_const_poly(folded_public, ShaPublicCol::K, 3, r_star, field_cfg)?
        - &endpoint_word_poly(endpoint_evals, ShaWordCol::W, 3, field_cfg)?
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
        - &endpoint_public_const_poly(folded_public, ShaPublicCol::K, 3, r_star, field_cfg)?
        - &endpoint_word_poly(endpoint_evals, ShaWordCol::W, 3, field_cfg)?
        + &mu_e
        + &endpoint_int_const_poly(endpoint_evals, ShaIntCol::CompUpdateE, field_cfg)?;

    let s_init = sha_public_at_point(folded_public, ShaPublicCol::SInit, 0, r_star, field_cfg)?;
    let s_msg = sha_public_at_point(folded_public, ShaPublicCol::SMsg, 0, r_star, field_cfg)?;
    let s_sched = sha_public_at_point(folded_public, ShaPublicCol::SSched, 0, r_star, field_cfg)?;
    let s_upd = sha_public_at_point(folded_public, ShaPublicCol::SUpd, 0, r_star, field_cfg)?;
    let s_ff = sha_public_at_point(folded_public, ShaPublicCol::SFf, 0, r_star, field_cfg)?;
    let s_out = sha_public_at_point(folded_public, ShaPublicCol::SOut, 0, r_star, field_cfg)?;

    let r7 = scale_endpoint_poly(
        &(a.clone()
            - &endpoint_public_const_poly(
                folded_public,
                ShaPublicCol::PAIn,
                0,
                r_star,
                field_cfg,
            )?),
        &s_init,
    ) + &scale_endpoint_poly(
        &(a.clone()
            - &endpoint_public_const_poly(
                folded_public,
                ShaPublicCol::PAOut,
                0,
                r_star,
                field_cfg,
            )?),
        &s_out,
    );
    let r8 = scale_endpoint_poly(
        &(e.clone()
            - &endpoint_public_const_poly(
                folded_public,
                ShaPublicCol::PEIn,
                0,
                r_star,
                field_cfg,
            )?),
        &s_init,
    ) + &scale_endpoint_poly(
        &(e.clone()
            - &endpoint_public_const_poly(
                folded_public,
                ShaPublicCol::PEOut,
                0,
                r_star,
                field_cfg,
            )?),
        &s_out,
    );

    let r9 = endpoint_word_poly(endpoint_evals, ShaWordCol::A, 4, field_cfg)?
        - &a
        - &endpoint_public_const_poly(folded_public, ShaPublicCol::PAIn, 0, r_star, field_cfg)?
        + &mu_ff_a
        + &endpoint_int_const_poly(endpoint_evals, ShaIntCol::CompFeedForwardA, field_cfg)?;
    let r10 = endpoint_word_poly(endpoint_evals, ShaWordCol::E, 4, field_cfg)?
        - &e
        - &endpoint_public_const_poly(folded_public, ShaPublicCol::PEIn, 0, r_star, field_cfg)?
        + &mu_ff_e
        + &endpoint_int_const_poly(endpoint_evals, ShaIntCol::CompFeedForwardE, field_cfg)?;
    let r11 = scale_endpoint_poly(
        &(w - &endpoint_public_const_poly(
            folded_public,
            ShaPublicCol::Message,
            0,
            r_star,
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

fn endpoint_public_const_poly<F>(
    folded_public: &ProjectedShaPublic<F>,
    col: ShaPublicCol,
    shift: usize,
    r_star: &[F],
    field_cfg: &F::Config,
) -> Result<DynamicPolynomialF<F>, ProductionShaError<F>>
where
    F: PrimeField,
{
    Ok(endpoint_const_poly(
        sha_public_at_point(folded_public, col, shift, r_star, field_cfg)?,
        field_cfg,
    ))
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
    F: PrimeField,
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
    let claim_at_r = weights
        .iter()
        .zip(fresh_targets)
        .fold(F::zero_with_cfg(field_cfg), |acc, (weight, target)| {
            acc + weight.clone() * target
        });
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

    use crate::fixed_prime;
    use crypto_primitives::{
        FromWithConfig, crypto_bigint_monty::MontyField, crypto_bigint_uint::Uint,
    };
    use zinc_piop::neutron_nova::{
        SHA_ROW_COUNT, SHA_WORD_BITS, ShaBitSliceColumns, ShaIntColumns, ShaPublicColumns,
        expression_folded_row_sum, fold_projected_sha_traces, scalarize_trace_words,
    };
    use zinc_poly::mle::MultilinearExtensionWithConfig;
    use zinc_transcript::{Blake3Transcript, traits::Transcript};

    type F = MontyField<4>;

    fn cfg() -> <F as PrimeField>::Config {
        fixed_prime::secp256k1_field_cfg::<F, Uint<4>>()
    }

    fn f(value: u64) -> F {
        F::from_with_cfg(value, &cfg())
    }

    fn zero_trace_with_scalar_challenge(a: &F) -> ProjectedShaTrace<F> {
        let field_cfg = cfg();
        let zero = F::zero_with_cfg(&field_cfg);
        let bit_slices = ShaBitSliceColumns {
            columns: vec![
                vec![vec![zero.clone(); SHA_WORD_BITS]; SHA_ROW_COUNT];
                ShaWordCol::COUNT
            ],
        };
        let scalarized_words = scalarize_trace_words(&bit_slices, a, &field_cfg).unwrap();
        ProjectedShaTrace {
            rows: SHA_ROW_COUNT,
            bit_slices,
            scalarized_words,
            int_columns: ShaIntColumns {
                columns: vec![vec![zero.clone(); SHA_ROW_COUNT]; ShaIntCol::COUNT],
            },
            public_columns: ShaPublicColumns {
                columns: vec![vec![zero; SHA_ROW_COUNT]; ShaPublicCol::COUNT],
            },
        }
    }

    fn zero_public() -> ProjectedShaPublic<F> {
        let field_cfg = cfg();
        ProjectedShaPublic {
            columns: ShaPublicColumns {
                columns: vec![
                    vec![F::zero_with_cfg(&field_cfg); SHA_ROW_COUNT];
                    ShaPublicCol::COUNT
                ],
            },
        }
    }

    fn fixed_layout_public() -> ProjectedShaPublic<F> {
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
                public.columns.columns[selector.index()][row] =
                    production_sha_selector_expected(selector, row, &field_cfg);
            }
        }
        for row in 0..SHA_ROW_COUNT {
            public.columns.columns[ShaPublicCol::K.index()][row] =
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
            absorb_fresh_sha_ideal_polys(&mut transcript, values);
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
            absorb_fresh_sha_ideal_polys(&mut transcript, values);
            let (a, _, _, _, _) =
                sample_post_ideal_challenges::<F>(&mut transcript, 1, &field_cfg).unwrap();
            a
        };

        assert_ne!(sample_a(&packed), sample_a(&split));
    }

    #[test]
    fn production_public_validation_requires_fixed_selectors_and_k() {
        let field_cfg = cfg();
        let valid = fixed_layout_public();
        validate_production_sha_publics(std::slice::from_ref(&valid), &field_cfg).unwrap();

        let mut bad_selector = valid.clone();
        bad_selector.columns.columns[ShaPublicCol::SOut.index()][0] = f(1);
        assert!(matches!(
            validate_production_sha_publics(&[bad_selector], &field_cfg),
            Err(ProductionShaError::InvalidPublicSelector {
                col: ShaPublicCol::SOut,
                row: 0
            })
        ));

        let mut non_boolean_selector = valid.clone();
        non_boolean_selector.columns.columns[ShaPublicCol::SInit.index()][0] = f(2);
        assert!(matches!(
            validate_production_sha_publics(&[non_boolean_selector], &field_cfg),
            Err(ProductionShaError::NonBooleanPublicSelector {
                col: ShaPublicCol::SInit,
                row: 0
            })
        ));

        let mut bad_k = valid;
        bad_k.columns.columns[ShaPublicCol::K.index()][3] += f(1);
        assert!(matches!(
            validate_production_sha_publics(&[bad_k], &field_cfg),
            Err(ProductionShaError::InvalidRoundConstant { row: 3 })
        ));
    }

    #[test]
    fn sumfold_outputs_instance_fold_point_before_theta() {
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
            prover_output.theta(),
            build_eq_x_r_vec(prover_output.r_b(), &field_cfg).unwrap()
        );
        assert_eq!(prover_output.theta().len(), fresh_targets.len());

        let d = eq_eval(&beta, prover_output.r_b(), F::one_with_cfg(&field_cfg)).unwrap();
        assert_eq!(prover_output.c_sf(), &(d * prover_output.t_prime()));

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
        let verifier_output = verify_full_sha_sumfold(
            &mut verifier_transcript,
            &proof,
            &initial_claim,
            &beta,
            traces.len(),
            &field_cfg,
        )
        .unwrap();

        assert_eq!(verifier_output, prover_output);
        assert_eq!(
            prover_output.theta(),
            build_eq_x_r_vec(prover_output.r_b(), &field_cfg).unwrap()
        );
        assert_eq!(prover_output.theta().len(), traces.len());

        let (folded, folded_public) =
            fold_projected_sha_traces(&traces, &publics, &prover_output, &field_cfg).unwrap();
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
        assert_eq!(prover_output.t_prime(), &folded_sum);

        let mut bad_transcript = Blake3Transcript::new();
        bad_transcript.absorb_slice(b"full-sha-sumfold-context");
        assert!(
            verify_full_sha_sumfold(
                &mut bad_transcript,
                &proof,
                &f(1),
                &beta,
                traces.len(),
                &field_cfg
            )
            .is_err()
        );
    }

    #[test]
    fn folded_row_sumcheck_claim_is_t_prime() {
        let field_cfg = cfg();
        let row_integrand_values = (0..(1usize << SHA_ROW_VARS))
            .map(|idx| f((idx as u64).wrapping_mul(3) + 1))
            .collect::<Vec<_>>();
        let t_prime = folded_row_integrand_sum(&row_integrand_values, &field_cfg).unwrap();

        let mut prover_transcript = Blake3Transcript::new();
        prover_transcript.absorb_slice(b"folded-row-context");
        let proof = prove_folded_row_sumcheck(
            &mut prover_transcript,
            &row_integrand_values,
            &t_prime,
            &field_cfg,
        )
        .unwrap();
        assert_eq!(proof.claimed_sums(), &[t_prime.clone()]);

        let mut verifier_transcript = Blake3Transcript::new();
        verifier_transcript.absorb_slice(b"folded-row-context");
        let output =
            verify_folded_row_sumcheck(&mut verifier_transcript, &proof, &t_prime, &field_cfg)
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

        let mut bad_t = t_prime;
        bad_t += f(1);
        let mut bad_transcript = Blake3Transcript::new();
        bad_transcript.absorb_slice(b"folded-row-context");
        assert!(
            verify_folded_row_sumcheck(&mut bad_transcript, &proof, &bad_t, &field_cfg).is_err()
        );
    }

    #[test]
    fn folded_row_verifier_rejects_extra_groups() {
        let field_cfg = cfg();
        let row_integrand_values = (0..(1usize << SHA_ROW_VARS))
            .map(|idx| f((idx as u64).wrapping_mul(5) + 9))
            .collect::<Vec<_>>();
        let t_prime = folded_row_integrand_sum(&row_integrand_values, &field_cfg).unwrap();
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
            verify_folded_row_sumcheck(&mut verifier_transcript, &proof, &t_prime, &field_cfg),
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
        let t_prime = expression_folded_row_sum(
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
            &t_prime,
            &field_cfg,
        )
        .unwrap();

        let mut verifier_transcript = Blake3Transcript::new();
        verifier_transcript.absorb_slice(b"expression-row-context");
        let output =
            verify_folded_row_sumcheck(&mut verifier_transcript, &proof, &t_prime, &field_cfg)
                .unwrap();
        let endpoint_evals =
            build_sha_endpoint_evals_from_trace(&trace, &output.r_star, &field_cfg).unwrap();
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
            build_sha_endpoint_evals_from_trace(&trace, &r_star, &field_cfg).unwrap();

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
            Err(ProductionShaError::NonCanonicalProofObject(_))
        ));

        let mut high_degree = vec![std::array::from_fn(|_| {
            DynamicPolynomialF::new(Vec::<F>::new())
        })];
        high_degree[0][2] = DynamicPolynomialF::new(vec![f(1); 33]);
        assert!(matches!(
            check_fresh_sha_ideal_membership(&high_degree, &field_cfg),
            Err(ProductionShaError::NonCanonicalProofObject(_))
        ));
    }

    #[test]
    fn endpoint_layout_must_be_exact_and_canonical() {
        let field_cfg = cfg();
        let trace = zero_trace_with_scalar_challenge(&f(5));
        let r_star = vec![f(2), f(3), f(5), f(7), f(11), f(13), f(17)];
        let endpoint_evals =
            build_sha_endpoint_evals_from_trace(&trace, &r_star, &field_cfg).unwrap();
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
        public.columns.columns[ShaPublicCol::K.index()][0] = f(99);
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
