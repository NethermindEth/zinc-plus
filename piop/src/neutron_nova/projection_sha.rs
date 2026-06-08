//! Production SHA-256 ProjectionFold helpers.
//!
//! This module implements the SHA-specific data model and reference
//! computations used by the production ProjectionFold flow:
//!
//! fresh ideal checks -> SumFold over instances -> post-SumFold folding ->
//! folded row check over the 128-row SHA domain.

use crate::ideal_check::batched_ideal_check;
use crate::neutron_nova::{SumFoldError, accumulator::dmr_flush_adds};
use crate::{
    CombFn,
    sumcheck::multi_degree::{MultiDegreeSumcheckGroup, PrefixFastPath, PrefixRoundOutput},
};
use crypto_primitives::{PrimeField, crypto_bigint_uint::Uint};
use num_traits::{ConstZero, Zero};
use thiserror::Error;
use zinc_poly::{
    mle::DenseMultilinearExtension,
    univariate::dynamic::over_field::{DynamicPolyFInnerProduct, DynamicPolynomialF},
    utils::{ArithErrors, build_eq_x_r_vec, eq_eval},
};
use zinc_uair::{
    ideal::{Ideal, IdealCheck, IdealCheckError, rotation::RotationIdeal},
    ideal_collector::IdealOrZero,
};
use zinc_utils::{
    UNCHECKED,
    delayed_reduction::{DelayedFieldProductSum, DelayedModularReduction, MontgomeryLimbs},
    from_ref::FromRef,
    inner_product::{FieldFieldInnerProduct, InnerProduct},
    inner_transparent_field::InnerTransparentField,
    powers,
};

pub const SHA_ROW_VARS: usize = 7;
pub const SHA_ROW_COUNT: usize = 128;
pub const SHA_WORD_BITS: usize = 32;
pub const NUM_SHA_RESIDUAL_FAMILIES: usize = 18;
pub const NUM_NONZERO_SHA_FAMILIES: usize = 7;
const SHA_RESIDUAL_EVAL_POWER_COUNT: usize = 62;

pub type MleColumn<T> = DenseMultilinearExtension<T>;
pub type MleTable<T> = Vec<MleColumn<T>>;

const NONZERO_SHA_FAMILIES: [ShaResidualFamily; NUM_NONZERO_SHA_FAMILIES] = [
    ShaResidualFamily::R0BigSigmaA,
    ShaResidualFamily::R1BigSigmaE,
    ShaResidualFamily::R4Schedule,
    ShaResidualFamily::R5UpdateA,
    ShaResidualFamily::R6UpdateE,
    ShaResidualFamily::R9FeedForwardA,
    ShaResidualFamily::R10FeedForwardE,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShaResidualFamily {
    R0BigSigmaA,
    R1BigSigmaE,
    R2SmallSigma0,
    R3SmallSigma1,
    R4Schedule,
    R5UpdateA,
    R6UpdateE,
    R7PinA,
    R8PinE,
    R9FeedForwardA,
    R10FeedForwardE,
    R11MessagePin,
    R12CompSchedule,
    R13CompUpdateA,
    R14CompUpdateE,
    R15CompFeedForwardA,
    R16CompFeedForwardE,
    R17CarryHighBits,
}

impl ShaResidualFamily {
    pub const ALL: [Self; NUM_SHA_RESIDUAL_FAMILIES] = [
        Self::R0BigSigmaA,
        Self::R1BigSigmaE,
        Self::R2SmallSigma0,
        Self::R3SmallSigma1,
        Self::R4Schedule,
        Self::R5UpdateA,
        Self::R6UpdateE,
        Self::R7PinA,
        Self::R8PinE,
        Self::R9FeedForwardA,
        Self::R10FeedForwardE,
        Self::R11MessagePin,
        Self::R12CompSchedule,
        Self::R13CompUpdateA,
        Self::R14CompUpdateE,
        Self::R15CompFeedForwardA,
        Self::R16CompFeedForwardE,
        Self::R17CarryHighBits,
    ];

    pub fn index(self) -> usize {
        match self {
            Self::R0BigSigmaA => 0,
            Self::R1BigSigmaE => 1,
            Self::R2SmallSigma0 => 2,
            Self::R3SmallSigma1 => 3,
            Self::R4Schedule => 4,
            Self::R5UpdateA => 5,
            Self::R6UpdateE => 6,
            Self::R7PinA => 7,
            Self::R8PinE => 8,
            Self::R9FeedForwardA => 9,
            Self::R10FeedForwardE => 10,
            Self::R11MessagePin => 11,
            Self::R12CompSchedule => 12,
            Self::R13CompUpdateA => 13,
            Self::R14CompUpdateE => 14,
            Self::R15CompFeedForwardA => 15,
            Self::R16CompFeedForwardE => 16,
            Self::R17CarryHighBits => 17,
        }
    }

    pub fn is_nonzero_ideal(self) -> bool {
        matches!(
            self,
            Self::R0BigSigmaA
                | Self::R1BigSigmaE
                | Self::R4Schedule
                | Self::R5UpdateA
                | Self::R6UpdateE
                | Self::R9FeedForwardA
                | Self::R10FeedForwardE
        )
    }
}

#[repr(usize)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShaWordCol {
    A = 0,
    E = 1,
    Sigma0 = 2,
    Sigma1 = 3,
    W = 4,
    SmallSigma0 = 5,
    SmallSigma1 = 6,
    Uef = 7,
    UNegEg = 8,
    Maj = 9,
    MuPacked = 10,
    OvSigma0 = 11,
    OvSigma1 = 12,
    OvSmallSigma0 = 13,
    OvSmallSigma1 = 14,
    Ch2Comp = 15,
    MajComp = 16,
}

impl ShaWordCol {
    pub const ALL: [Self; 17] = [
        Self::A,
        Self::E,
        Self::Sigma0,
        Self::Sigma1,
        Self::W,
        Self::SmallSigma0,
        Self::SmallSigma1,
        Self::Uef,
        Self::UNegEg,
        Self::Maj,
        Self::MuPacked,
        Self::OvSigma0,
        Self::OvSigma1,
        Self::OvSmallSigma0,
        Self::OvSmallSigma1,
        Self::Ch2Comp,
        Self::MajComp,
    ];

    pub const COUNT: usize = 17;

    pub fn index(self) -> usize {
        self as usize
    }
}

#[repr(usize)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShaIntCol {
    CompSchedule = 0,
    CompUpdateA = 1,
    CompUpdateE = 2,
    CompFeedForwardA = 3,
    CompFeedForwardE = 4,
}

impl ShaIntCol {
    pub const ALL: [Self; 5] = [
        Self::CompSchedule,
        Self::CompUpdateA,
        Self::CompUpdateE,
        Self::CompFeedForwardA,
        Self::CompFeedForwardE,
    ];

    pub const COUNT: usize = 5;

    pub fn index(self) -> usize {
        self as usize
    }
}

#[repr(usize)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShaPublicCol {
    K = 0,
    PAIn = 1,
    PEIn = 2,
    PAOut = 3,
    PEOut = 4,
    Message = 5,
    SInit = 6,
    SMsg = 7,
    SSched = 8,
    SUpd = 9,
    SFf = 10,
    SOut = 11,
}

impl ShaPublicCol {
    pub const ALL: [Self; 12] = [
        Self::K,
        Self::PAIn,
        Self::PEIn,
        Self::PAOut,
        Self::PEOut,
        Self::Message,
        Self::SInit,
        Self::SMsg,
        Self::SSched,
        Self::SUpd,
        Self::SFf,
        Self::SOut,
    ];

    pub const COUNT: usize = 12;

    pub fn index(self) -> usize {
        self as usize
    }
}

#[repr(usize)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShaPublicWordCol {
    PAIn = 0,
    PEIn = 1,
    PAOut = 2,
    PEOut = 3,
    Message = 4,
}

impl ShaPublicWordCol {
    pub const ALL: [Self; 5] = [
        Self::PAIn,
        Self::PEIn,
        Self::PAOut,
        Self::PEOut,
        Self::Message,
    ];

    pub const COUNT: usize = 5;

    pub fn index(self) -> usize {
        self as usize
    }
}

impl ShaPublicCol {
    fn public_word_col(self) -> Option<ShaPublicWordCol> {
        match self {
            Self::PAIn => Some(ShaPublicWordCol::PAIn),
            Self::PEIn => Some(ShaPublicWordCol::PEIn),
            Self::PAOut => Some(ShaPublicWordCol::PAOut),
            Self::PEOut => Some(ShaPublicWordCol::PEOut),
            Self::Message => Some(ShaPublicWordCol::Message),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectedTrace<F> {
    /// Flattened as `[word_col * SHA_WORD_BITS + bit][row]`.
    pub bit_slices: MleTable<F>,
    /// Indexed as `[word_col][row]`.
    pub scalarized: MleTable<F>,
    /// Indexed as `[int_col][row]`.
    pub int_columns: MleTable<F>,
    /// Indexed as `[public_col][row]`.
    pub public_columns: MleTable<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectedPublic<F> {
    /// Indexed as `[public_col][row]`.
    pub columns: MleTable<F>,
    /// Flattened as `[public_word_col * SHA_WORD_BITS + bit][row]`.
    pub bit_slices: Option<MleTable<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FreshIdealEvaluationCache<F: PrimeField> {
    pub r_ic: [F; SHA_ROW_VARS],
    pub ideal_polys: Vec<[DynamicPolynomialF<F>; NUM_NONZERO_SHA_FAMILIES]>,
    pub taus_at_a: Vec<[F; NUM_NONZERO_SHA_FAMILIES]>,
    pub fresh_targets: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearResidualCoeffTable<F: PrimeField> {
    /// Indexed by residual family.
    pub coeffs: Vec<DynamicPolynomialF<F>>,
}

impl<F> LinearResidualCoeffTable<F>
where
    F: PrimeField,
{
    pub fn coeffs_for_family(&self, family: ShaResidualFamily) -> Option<&DynamicPolynomialF<F>> {
        self.coeffs.get(family.index())
    }
}

pub fn beta_aggregate_nonzero_ideal_polys<F>(
    tables: &[LinearResidualCoeffTable<F>],
    beta: &[F],
    field_cfg: &F::Config,
) -> Result<[DynamicPolynomialF<F>; NUM_NONZERO_SHA_FAMILIES], ShaProjectionError>
where
    F: PrimeField,
{
    let weights = build_eq_x_r_vec(beta, field_cfg)?;
    beta_aggregate_nonzero_ideal_polys_with_weights(tables, &weights)
}

pub fn beta_aggregate_nonzero_ideal_polys_with_weights<F>(
    tables: &[LinearResidualCoeffTable<F>],
    beta_eq_weights: &[F],
) -> Result<[DynamicPolynomialF<F>; NUM_NONZERO_SHA_FAMILIES], ShaProjectionError>
where
    F: PrimeField,
{
    if beta_eq_weights.len() != tables.len() {
        return Err(ShaProjectionError::InstanceCountMismatch {
            got: beta_eq_weights.len(),
            expected: tables.len(),
        });
    }

    let mut aggregate: [DynamicPolynomialF<F>; NUM_NONZERO_SHA_FAMILIES] =
        std::array::from_fn(|_| DynamicPolynomialF::ZERO);
    for (weight, table) in beta_eq_weights.iter().zip(tables) {
        for (slot, family) in NONZERO_SHA_FAMILIES.iter().enumerate() {
            let residual =
                table
                    .coeffs
                    .get(family.index())
                    .ok_or(ShaProjectionError::MissingColumn {
                        kind: "linear_residual_coeffs",
                        col: family.index(),
                    })?;
            let weighted = scale_poly(residual, weight);
            aggregate[slot] += &weighted;
        }
    }
    aggregate.iter_mut().for_each(DynamicPolynomialF::trim);
    Ok(aggregate)
}

pub fn build_sha_residual_eval_powers<F>(a: &F, field_cfg: &F::Config) -> Vec<F>
where
    F: PrimeField,
{
    powers(
        a.clone(),
        F::one_with_cfg(field_cfg),
        SHA_RESIDUAL_EVAL_POWER_COUNT,
    )
}

pub fn build_sha_lambda_powers<F>(lambda: &F, field_cfg: &F::Config) -> Vec<F>
where
    F: PrimeField,
{
    powers(
        lambda.clone(),
        F::one_with_cfg(field_cfg),
        NUM_SHA_RESIDUAL_FAMILIES,
    )
}

pub fn build_booleanity_weights<F>(
    rho: &F,
    xi: &F,
    source_count: usize,
    field_cfg: &F::Config,
) -> Vec<F>
where
    F: PrimeField,
{
    powers(rho.clone(), F::one_with_cfg(field_cfg), source_count)
        .into_iter()
        .map(|rho_power| xi.clone() * rho_power)
        .collect()
}

pub fn build_sha_sumfold_linear_accumulator<F>(
    tables: &[LinearResidualCoeffTable<F>],
    a_powers: &[F],
    lambda_powers: &[F],
    field_cfg: &F::Config,
) -> Result<Vec<F>, ShaProjectionError>
where
    F: DelayedFieldProductSum,
{
    if lambda_powers.len() != NUM_SHA_RESIDUAL_FAMILIES {
        return Err(ShaProjectionError::MissingColumn {
            kind: "lambda_powers",
            col: lambda_powers.len(),
        });
    }
    if a_powers.len() < SHA_RESIDUAL_EVAL_POWER_COUNT {
        return Err(ShaProjectionError::MissingColumn {
            kind: "a_powers",
            col: a_powers.len(),
        });
    }
    tables
        .iter()
        .map(|table| {
            if table.coeffs.len() != NUM_SHA_RESIDUAL_FAMILIES {
                return Err(ShaProjectionError::MissingColumn {
                    kind: "linear_residual_coeffs",
                    col: table.coeffs.len(),
                });
            }
            let mut target = F::zero_with_cfg(field_cfg);
            for (family_idx, residual) in table.coeffs.iter().enumerate() {
                target += lambda_powers[family_idx].clone()
                    * evaluate_poly_at_powers_dmr(residual, &a_powers, field_cfg)?;
            }
            Ok(target)
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InstanceFoldClaim<F> {
    pub r_b: Vec<F>,
    pub c_sf: F,
    pub final_round_sumcheck_claim: F,
    pub eq_instance_weights: Vec<F>,
}

impl<F> InstanceFoldClaim<F> {
    pub fn r_b(&self) -> &[F] {
        &self.r_b
    }

    pub fn c_sf(&self) -> &F {
        &self.c_sf
    }

    pub fn final_round_sumcheck_claim(&self) -> &F {
        &self.final_round_sumcheck_claim
    }

    pub fn eq_instance_weights(&self) -> &[F] {
        &self.eq_instance_weights
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldedCommitments<C> {
    pub commitments: Vec<C>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectionFoldAccumulator<F, C = F> {
    pub instance_fold_claim: InstanceFoldClaim<F>,
    pub commitments: FoldedCommitments<C>,
    pub public: ProjectedPublic<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectionFoldWitness<F> {
    pub trace: ProjectedTrace<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShaBooleanitySource {
    WordBit { col: ShaWordCol, bit: usize },
    VirtualCh1 { bit: usize },
    VirtualCh2 { bit: usize },
    VirtualMaj { bit: usize },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VirtualChMajValues<F> {
    pub ch1: [F; SHA_WORD_BITS],
    pub ch2: [F; SHA_WORD_BITS],
    pub maj: [F; SHA_WORD_BITS],
}

#[derive(Clone, Debug)]
pub enum ShaProductionIdeal<F: PrimeField> {
    RotX2(RotationIdeal<F, 1>),
    RotXw1,
}

impl<F: PrimeField> FromRef<ShaProductionIdeal<F>> for ShaProductionIdeal<F> {
    fn from_ref(value: &ShaProductionIdeal<F>) -> Self {
        value.clone()
    }
}

impl<F: PrimeField> Ideal for ShaProductionIdeal<F> {}

impl<F: PrimeField> IdealCheck<DynamicPolynomialF<F>> for ShaProductionIdeal<F> {
    fn contains(&self, value: &DynamicPolynomialF<F>) -> Result<bool, IdealCheckError> {
        match self {
            ShaProductionIdeal::RotX2(ideal) => IdealOrZero::NonZero(ideal.clone()).contains(value),
            ShaProductionIdeal::RotXw1 => {
                if value.coeffs.is_empty() {
                    return Ok(true);
                }
                let one = F::one_with_cfg(value.coeffs[0].cfg());
                IdealOrZero::NonZero(RotationIdeal::<F, 32>::new(one)).contains(value)
            }
        }
    }
}

#[derive(Clone, Debug, Error)]
pub enum ShaProjectionError {
    #[error("expected {expected} rows, got {got}")]
    RowCount { expected: usize, got: usize },
    #[error("row index out of range: {row}")]
    RowIndexOutOfRange { row: usize },
    #[error("{kind} column {col} is missing")]
    MissingColumn { kind: &'static str, col: usize },
    #[error("{kind} column {col} row length mismatch: got {got}, expected {expected}")]
    ColumnRowCount {
        kind: &'static str,
        col: usize,
        got: usize,
        expected: usize,
    },
    #[error("word column {col} row {row} bit length mismatch: got {got}, expected {expected}")]
    BitCount {
        col: usize,
        row: usize,
        got: usize,
        expected: usize,
    },
    #[error("row-batching point length mismatch: got {got}, expected 7")]
    RowPointLength { got: usize },
    #[error("instance count must be a power of two, got {got}")]
    InstanceCountNotPowerOfTwo { got: usize },
    #[error("instance count mismatch: got {got}, expected {expected}")]
    InstanceCountMismatch { got: usize, expected: usize },
    #[error("public word column presence mismatch across folded instances")]
    PublicWordColumnPresenceMismatch,
    #[error("folding weight count mismatch: got {got}, expected {expected}")]
    FoldingWeightCount { got: usize, expected: usize },
    #[error("SumFold denominator eq(beta, r_b) is zero")]
    ZeroSumFoldDenominator,
    #[error("scalarization mismatch for word column {col}")]
    ScalarizationMismatch { col: usize },
    #[error("folded row sumcheck claim does not match SumFold target")]
    FoldedRowClaimMismatch,
    #[error("booleanity bit index out of range: {bit}")]
    BitIndexOutOfRange { bit: usize },
    #[error("non-canonical proof object: {0}")]
    NonCanonicalProofObject(&'static str),
    #[error("ideal membership check failed")]
    IdealMembership,
    #[error("polynomial evaluation failed: {0}")]
    PolynomialEvaluation(#[from] zinc_poly::EvaluationError),
    #[error("inner product failed: {0}")]
    InnerProduct(#[from] zinc_utils::inner_product::InnerProductError),
    #[error("equality table construction failed: {0}")]
    EqTable(#[from] ArithErrors),
    #[error("sumfold helper failed: {0}")]
    SumFold(#[from] SumFoldError),
}

pub fn production_sha_nonzero_families() -> &'static [ShaResidualFamily] {
    &NONZERO_SHA_FAMILIES
}

pub fn production_sha_nonzero_ideals<F: PrimeField>(
    field_cfg: &F::Config,
) -> [ShaProductionIdeal<F>; NUM_NONZERO_SHA_FAMILIES] {
    let two = F::one_with_cfg(field_cfg) + F::one_with_cfg(field_cfg);
    [
        ShaProductionIdeal::RotXw1,
        ShaProductionIdeal::RotXw1,
        ShaProductionIdeal::RotX2(RotationIdeal::new(two.clone())),
        ShaProductionIdeal::RotX2(RotationIdeal::new(two.clone())),
        ShaProductionIdeal::RotX2(RotationIdeal::new(two.clone())),
        ShaProductionIdeal::RotX2(RotationIdeal::new(two.clone())),
        ShaProductionIdeal::RotX2(RotationIdeal::new(two)),
    ]
}

fn production_sha_ideal_max_degree(family: ShaResidualFamily) -> Result<usize, ShaProjectionError> {
    match family {
        ShaResidualFamily::R0BigSigmaA | ShaResidualFamily::R1BigSigmaE => Ok(61),
        ShaResidualFamily::R4Schedule
        | ShaResidualFamily::R5UpdateA
        | ShaResidualFamily::R6UpdateE
        | ShaResidualFamily::R9FeedForwardA
        | ShaResidualFamily::R10FeedForwardE => Ok(31),
        _ => Err(ShaProjectionError::NonCanonicalProofObject(
            "unexpected nonzero SHA ideal family",
        )),
    }
}

pub fn validate_fresh_sha_ideal_polys_canonical<F>(
    ideal_polys: &[[DynamicPolynomialF<F>; NUM_NONZERO_SHA_FAMILIES]],
) -> Result<(), ShaProjectionError>
where
    F: PrimeField,
{
    for instance in ideal_polys {
        for (slot, poly) in instance.iter().enumerate() {
            if poly.coeffs.last().is_some_and(F::is_zero) {
                return Err(ShaProjectionError::NonCanonicalProofObject(
                    "fresh ideal polynomial has trailing zero coefficients",
                ));
            }

            let family = production_sha_nonzero_families()[slot];
            let max_degree = production_sha_ideal_max_degree(family)?;
            if poly.coeffs.len() > max_degree + 1 {
                return Err(ShaProjectionError::NonCanonicalProofObject(
                    "fresh ideal polynomial exceeds production degree cap",
                ));
            }
        }
    }
    Ok(())
}

pub fn verify_fresh_sha_ideal_polys<F>(
    ideal_polys: &[[DynamicPolynomialF<F>; NUM_NONZERO_SHA_FAMILIES]],
    field_cfg: &F::Config,
) -> Result<(), ShaProjectionError>
where
    F: PrimeField,
{
    validate_fresh_sha_ideal_polys_canonical(ideal_polys)?;

    let ideals = production_sha_nonzero_ideals(field_cfg);
    for values in ideal_polys {
        batched_ideal_check(&ideals, values).map_err(|_err| ShaProjectionError::IdealMembership)?;
    }
    Ok(())
}

pub fn build_sha_ideal_values_at_point<F>(
    trace: &ProjectedTrace<F>,
    public: &ProjectedPublic<F>,
    r_ic: &[F; SHA_ROW_VARS],
    field_cfg: &F::Config,
) -> Result<[DynamicPolynomialF<F>; NUM_NONZERO_SHA_FAMILIES], ShaProjectionError>
where
    F: PrimeField,
{
    validate_trace(trace)?;
    validate_public(public)?;

    let row_weights = build_eq_x_r_vec(r_ic, field_cfg)?;
    let mut out: [DynamicPolynomialF<F>; NUM_NONZERO_SHA_FAMILIES] =
        std::array::from_fn(|_| DynamicPolynomialF::ZERO);

    for (row, row_weight) in row_weights.iter().enumerate().take(SHA_ROW_COUNT) {
        let residuals = residual_polys_at_row(trace, public, row, field_cfg)?;
        for (slot, family) in NONZERO_SHA_FAMILIES.iter().enumerate() {
            let weighted = scale_poly(&residuals[family.index()], row_weight);
            out[slot] += &weighted;
        }
    }
    out.iter_mut().for_each(DynamicPolynomialF::trim);
    Ok(out)
}

pub fn check_sha_ideal_values<F>(
    values: &[DynamicPolynomialF<F>; NUM_NONZERO_SHA_FAMILIES],
    field_cfg: &F::Config,
) -> Result<(), ShaProjectionError>
where
    F: PrimeField,
{
    let ideals = production_sha_nonzero_ideals(field_cfg);
    batched_ideal_check(&ideals, values).map_err(|_err| ShaProjectionError::IdealMembership)
}

#[allow(clippy::arithmetic_side_effects)]
pub fn bit_slice_index(col: usize, bit: usize, bits_per_col: usize) -> usize {
    col * bits_per_col + bit
}

fn mle_table_from_columns<T>(columns: Vec<Vec<T>>, num_vars: usize) -> MleTable<T> {
    columns
        .into_iter()
        .map(|evaluations| DenseMultilinearExtension {
            evaluations,
            num_vars,
        })
        .collect()
}

#[cfg(test)]
fn flatten_bit_columns<T>(
    columns: Vec<Vec<Vec<T>>>,
    bits_per_col: usize,
    num_vars: usize,
    kind: &'static str,
) -> Result<MleTable<T>, ShaProjectionError> {
    let mut flattened = (0..columns.len() * bits_per_col)
        .map(|_| Vec::new())
        .collect::<Vec<Vec<T>>>();
    for (col_idx, rows) in columns.into_iter().enumerate() {
        if rows.len() != SHA_ROW_COUNT {
            return Err(ShaProjectionError::ColumnRowCount {
                kind,
                col: col_idx,
                got: rows.len(),
                expected: SHA_ROW_COUNT,
            });
        }
        for (row, bits) in rows.into_iter().enumerate() {
            if bits.len() != bits_per_col {
                return Err(ShaProjectionError::BitCount {
                    col: col_idx,
                    row,
                    got: bits.len(),
                    expected: bits_per_col,
                });
            }
            for (bit, value) in bits.into_iter().enumerate() {
                flattened[bit_slice_index(col_idx, bit, bits_per_col)].push(value);
            }
        }
    }
    Ok(mle_table_from_columns(flattened, num_vars))
}

pub fn build_fresh_sha_ideal_cache<F>(
    traces: &[ProjectedTrace<F>],
    publics: &[ProjectedPublic<F>],
    r_ic: [F; SHA_ROW_VARS],
    field_cfg: &F::Config,
) -> Result<FreshIdealEvaluationCache<F>, ShaProjectionError>
where
    F: PrimeField,
{
    if traces.len() != publics.len() {
        return Err(ShaProjectionError::InstanceCountMismatch {
            got: publics.len(),
            expected: traces.len(),
        });
    }
    let ideal_polys = traces
        .iter()
        .zip(publics)
        .map(|(trace, public)| build_sha_ideal_values_at_point(trace, public, &r_ic, field_cfg))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(FreshIdealEvaluationCache {
        r_ic,
        ideal_polys,
        taus_at_a: Vec::new(),
        fresh_targets: Vec::new(),
    })
}

pub fn build_linear_residual_coeff_tables<F>(
    traces: &[ProjectedTrace<F>],
    publics: &[ProjectedPublic<F>],
    r_ic: &[F; SHA_ROW_VARS],
    field_cfg: &F::Config,
) -> Result<Vec<LinearResidualCoeffTable<F>>, ShaProjectionError>
where
    F: PrimeField,
{
    let row_weights = build_eq_x_r_vec(r_ic, field_cfg)?;
    build_linear_residual_coeff_tables_with_row_weights(traces, publics, &row_weights, field_cfg)
}

pub fn build_linear_residual_coeff_tables_with_row_weights<F>(
    traces: &[ProjectedTrace<F>],
    publics: &[ProjectedPublic<F>],
    row_weights: &[F],
    field_cfg: &F::Config,
) -> Result<Vec<LinearResidualCoeffTable<F>>, ShaProjectionError>
where
    F: PrimeField,
{
    if traces.len() != publics.len() {
        return Err(ShaProjectionError::InstanceCountMismatch {
            got: publics.len(),
            expected: traces.len(),
        });
    }
    if row_weights.len() != SHA_ROW_COUNT {
        return Err(ShaProjectionError::ColumnRowCount {
            kind: "row_weights",
            col: 0,
            got: row_weights.len(),
            expected: SHA_ROW_COUNT,
        });
    }
    traces
        .iter()
        .zip(publics)
        .map(|(trace, public)| {
            validate_trace(trace)?;
            validate_public(public)?;
            let mut coeffs = vec![DynamicPolynomialF::ZERO; NUM_SHA_RESIDUAL_FAMILIES];
            for (row, row_weight) in row_weights.iter().enumerate().take(SHA_ROW_COUNT) {
                let residuals = residual_polys_at_row(trace, public, row, field_cfg)?;
                for (family_idx, residual) in residuals.iter().enumerate() {
                    let weighted = scale_poly(residual, row_weight);
                    coeffs[family_idx] += &weighted;
                }
            }
            coeffs.iter_mut().for_each(DynamicPolynomialF::trim);
            Ok(LinearResidualCoeffTable { coeffs })
        })
        .collect::<Result<Vec<_>, _>>()
}

pub fn check_fresh_sha_ideal_cache<F>(
    cache: &FreshIdealEvaluationCache<F>,
    field_cfg: &F::Config,
) -> Result<(), ShaProjectionError>
where
    F: PrimeField,
{
    verify_fresh_sha_ideal_polys(&cache.ideal_polys, field_cfg)
}

pub fn evaluate_fresh_sha_targets<F>(
    cache: &mut FreshIdealEvaluationCache<F>,
    a: &F,
    lambda: &F,
    field_cfg: &F::Config,
) -> Result<(), ShaProjectionError>
where
    F: DelayedFieldProductSum,
{
    let one = F::one_with_cfg(field_cfg);
    let zero = F::zero_with_cfg(field_cfg);
    let lambda_powers = powers(lambda.clone(), one, NUM_SHA_RESIDUAL_FAMILIES);
    let a_powers = powers(
        a.clone(),
        F::one_with_cfg(field_cfg),
        SHA_RESIDUAL_EVAL_POWER_COUNT,
    );

    cache.taus_at_a.clear();
    cache.fresh_targets.clear();

    for ideal_polys in &cache.ideal_polys {
        let mut tau_values = Vec::with_capacity(NUM_NONZERO_SHA_FAMILIES);
        for poly in ideal_polys {
            tau_values.push(evaluate_poly_at_powers_dmr(poly, &a_powers, field_cfg)?);
        }
        let taus: [F; NUM_NONZERO_SHA_FAMILIES] = tau_values
            .try_into()
            .unwrap_or_else(|_| unreachable!("exactly seven SHA ideal values were evaluated"));
        let mut target = zero.clone();
        for (slot, family) in NONZERO_SHA_FAMILIES.iter().enumerate() {
            target += lambda_powers[family.index()].clone() * &taus[slot];
        }
        cache.taus_at_a.push(taus);
        cache.fresh_targets.push(target);
    }
    Ok(())
}

pub fn derive_instance_fold_claim<F>(
    beta: &[F],
    r_b: Vec<F>,
    c_sf: F,
    instance_count: usize,
    field_cfg: &F::Config,
) -> Result<InstanceFoldClaim<F>, ShaProjectionError>
where
    F: PrimeField,
{
    if !instance_count.is_power_of_two() {
        return Err(ShaProjectionError::InstanceCountNotPowerOfTwo {
            got: instance_count,
        });
    }
    let ell = usize::try_from(instance_count.trailing_zeros()).expect("ell fits usize");
    if beta.len() != ell {
        return Err(ShaProjectionError::InstanceCountMismatch {
            got: beta.len(),
            expected: ell,
        });
    }
    if r_b.len() != ell {
        return Err(ShaProjectionError::InstanceCountMismatch {
            got: r_b.len(),
            expected: ell,
        });
    }

    let one = F::one_with_cfg(field_cfg);
    let d = eq_eval(beta, &r_b, one)?;
    if F::is_zero(&d) {
        return Err(ShaProjectionError::ZeroSumFoldDenominator);
    }

    let eq_instance_weights = build_eq_x_r_vec(&r_b, field_cfg)?;
    debug_assert_eq!(eq_instance_weights.len(), instance_count);
    let final_round_sumcheck_claim = c_sf.clone() / d;
    Ok(InstanceFoldClaim {
        r_b,
        c_sf,
        final_round_sumcheck_claim,
        eq_instance_weights,
    })
}

pub fn fold_projected_traces<F>(
    traces: &[ProjectedTrace<F>],
    publics: &[ProjectedPublic<F>],
    sumfold: &InstanceFoldClaim<F>,
    field_cfg: &F::Config,
) -> Result<(ProjectionFoldWitness<F>, ProjectedPublic<F>), ShaProjectionError>
where
    F: PrimeField,
{
    if traces.len() != publics.len() {
        return Err(ShaProjectionError::InstanceCountMismatch {
            got: publics.len(),
            expected: traces.len(),
        });
    }
    if sumfold.eq_instance_weights.len() != traces.len() {
        return Err(ShaProjectionError::FoldingWeightCount {
            got: sumfold.eq_instance_weights.len(),
            expected: traces.len(),
        });
    }
    for trace in traces {
        validate_trace(trace)?;
    }
    for public in publics {
        validate_public(public)?;
    }

    let folded_trace = ProjectedTrace {
        bit_slices: fold_mle_tables(
            "bit_slices",
            traces.iter().map(|trace| &trace.bit_slices),
            &sumfold.eq_instance_weights,
            field_cfg,
        )?,
        scalarized: fold_mle_tables(
            "scalarized",
            traces.iter().map(|trace| &trace.scalarized),
            &sumfold.eq_instance_weights,
            field_cfg,
        )?,
        int_columns: fold_mle_tables(
            "int_columns",
            traces.iter().map(|trace| &trace.int_columns),
            &sumfold.eq_instance_weights,
            field_cfg,
        )?,
        public_columns: fold_mle_tables(
            "public_columns",
            traces.iter().map(|trace| &trace.public_columns),
            &sumfold.eq_instance_weights,
            field_cfg,
        )?,
    };
    let folded_public = ProjectedPublic {
        columns: fold_mle_tables(
            "public.columns",
            publics.iter().map(|public| &public.columns),
            &sumfold.eq_instance_weights,
            field_cfg,
        )?,
        bit_slices: fold_optional_mle_tables(
            "public.bit_slices",
            publics.iter().map(|public| public.bit_slices.as_ref()),
            &sumfold.eq_instance_weights,
            field_cfg,
        )?,
    };

    Ok((
        ProjectionFoldWitness {
            trace: folded_trace,
        },
        folded_public,
    ))
}

pub fn scalarize_bit_slices<F>(
    bit_slices: &MleTable<F>,
    a: &F,
    field_cfg: &F::Config,
) -> Result<MleTable<F>, ShaProjectionError>
where
    F: PrimeField + MontgomeryLimbs + DelayedFieldProductSum + Send + Sync,
{
    let powers = powers(a.clone(), F::one_with_cfg(field_cfg), SHA_WORD_BITS);
    if bit_slices.len() % SHA_WORD_BITS != 0 {
        return Err(ShaProjectionError::MissingColumn {
            kind: "bit_slices",
            col: bit_slices.len(),
        });
    }
    let word_col_count = bit_slices.len() / SHA_WORD_BITS;
    let mut words = Vec::with_capacity(word_col_count);
    for col_idx in 0..word_col_count {
        let mut out_col = Vec::with_capacity(SHA_ROW_COUNT);
        for row in 0..SHA_ROW_COUNT {
            let mut bits = Vec::with_capacity(SHA_WORD_BITS);
            for bit in 0..SHA_WORD_BITS {
                bits.push(scalar_from_table(
                    "bit_slices",
                    bit_slices,
                    bit_slice_index(col_idx, bit, SHA_WORD_BITS),
                    row,
                    field_cfg,
                )?);
            }
            out_col.push(project_binary_bits_conditional_add_dmr(
                &bits, &powers, field_cfg,
            )?);
        }
        words.push(out_col);
    }
    Ok(mle_table_from_columns(words, SHA_ROW_VARS))
}

pub fn verify_folded_scalarization_links<F>(
    trace: &ProjectedTrace<F>,
    a: &F,
    word_cols: &[ShaWordCol],
    field_cfg: &F::Config,
) -> Result<(), ShaProjectionError>
where
    F: PrimeField + DelayedFieldProductSum,
{
    validate_trace(trace)?;
    let powers = powers(a.clone(), F::one_with_cfg(field_cfg), SHA_WORD_BITS);
    for col in word_cols {
        let col_idx = col.index();
        for row in 0..SHA_ROW_COUNT {
            let mut bits = Vec::with_capacity(SHA_WORD_BITS);
            for bit in 0..SHA_WORD_BITS {
                bits.push(scalar_from_table(
                    "bit_slices",
                    &trace.bit_slices,
                    bit_slice_index(col_idx, bit, SHA_WORD_BITS),
                    row,
                    field_cfg,
                )?);
            }
            let recombined = project_bits_dmr(&bits, &powers, field_cfg)?;
            let scalar =
                scalar_from_table("scalarized", &trace.scalarized, col_idx, row, field_cfg)?;
            if recombined != scalar {
                return Err(ShaProjectionError::ScalarizationMismatch { col: col_idx });
            }
        }
    }
    Ok(())
}

pub fn verify_folded_scalarization_links_at_point<F>(
    trace: &ProjectedTrace<F>,
    a: &F,
    r_star: &[F; SHA_ROW_VARS],
    word_cols: &[ShaWordCol],
    field_cfg: &F::Config,
) -> Result<(), ShaProjectionError>
where
    F: PrimeField + DelayedFieldProductSum,
{
    for col in word_cols {
        verify_folded_shifted_scalarization_link_at_point(trace, a, r_star, *col, 0, field_cfg)?;
    }
    Ok(())
}

pub fn verify_folded_shifted_scalarization_link_at_point<F>(
    trace: &ProjectedTrace<F>,
    a: &F,
    r_star: &[F; SHA_ROW_VARS],
    col: ShaWordCol,
    shift: usize,
    field_cfg: &F::Config,
) -> Result<(), ShaProjectionError>
where
    F: PrimeField + DelayedFieldProductSum,
{
    validate_trace(trace)?;
    let row_weights = build_eq_x_r_vec(r_star, field_cfg)?;
    let powers = powers(a.clone(), F::one_with_cfg(field_cfg), SHA_WORD_BITS);
    let mut word_eval = F::zero_with_cfg(field_cfg);
    let mut bit_rows = Vec::with_capacity(SHA_ROW_COUNT);

    for (row, row_weight) in row_weights.iter().enumerate() {
        word_eval += row_weight.clone()
            * scalarized_word_at_shifted_or_zero(trace, col, row, shift, field_cfg)?;
        let mut bits = Vec::with_capacity(SHA_WORD_BITS);
        for bit in 0..SHA_WORD_BITS {
            bits.push(bit_at_shifted_or_zero(
                trace, col, row, shift, bit, field_cfg,
            )?);
        }
        bit_rows.push(project_bits_dmr(&bits, &powers, field_cfg)?);
    }
    let bit_eval = FieldFieldInnerProduct::inner_product::<UNCHECKED>(
        &row_weights,
        &bit_rows,
        F::zero_with_cfg(field_cfg),
    )?;

    if word_eval != bit_eval {
        return Err(ShaProjectionError::ScalarizationMismatch { col: col.index() });
    }
    Ok(())
}

pub fn reconstruct_virtual_ch_maj_at_row<F>(
    trace: &ProjectedTrace<F>,
    row: usize,
    field_cfg: &F::Config,
) -> Result<VirtualChMajValues<F>, ShaProjectionError>
where
    F: PrimeField,
{
    validate_trace(trace)?;
    reconstruct_virtual_ch_maj_at_row_unchecked(trace, row, field_cfg)
}

fn reconstruct_virtual_ch_maj_at_row_unchecked<F>(
    trace: &ProjectedTrace<F>,
    row: usize,
    field_cfg: &F::Config,
) -> Result<VirtualChMajValues<F>, ShaProjectionError>
where
    F: PrimeField,
{
    if row >= SHA_ROW_COUNT {
        return Err(ShaProjectionError::RowIndexOutOfRange { row });
    }
    let two = F::one_with_cfg(field_cfg) + F::one_with_cfg(field_cfg);
    let ch1 = build_virtual_bit_array(|bit| {
        Ok(
            bit_at_shifted_or_zero(trace, ShaWordCol::E, row, 2, bit, field_cfg)?
                + bit_at_shifted_or_zero(trace, ShaWordCol::E, row, 1, bit, field_cfg)?
                - two.clone()
                    * bit_at_shifted_or_zero(trace, ShaWordCol::Uef, row, 2, bit, field_cfg)?,
        )
    });
    let ch2 = build_virtual_bit_array(|bit| {
        Ok(
            bit_at_shifted_or_zero(trace, ShaWordCol::E, row, 2, bit, field_cfg)?
                - bit_at_shifted_or_zero(trace, ShaWordCol::E, row, 0, bit, field_cfg)?
                + two.clone()
                    * bit_at_shifted_or_zero(trace, ShaWordCol::UNegEg, row, 2, bit, field_cfg)?
                + two.clone()
                    * bit_at_shifted_or_zero(trace, ShaWordCol::Ch2Comp, row, 0, bit, field_cfg)?,
        )
    });
    let maj = build_virtual_bit_array(|bit| {
        Ok(
            bit_at_shifted_or_zero(trace, ShaWordCol::A, row, 0, bit, field_cfg)?
                + bit_at_shifted_or_zero(trace, ShaWordCol::A, row, 1, bit, field_cfg)?
                + bit_at_shifted_or_zero(trace, ShaWordCol::A, row, 2, bit, field_cfg)?
                - two.clone()
                    * bit_at_shifted_or_zero(trace, ShaWordCol::Maj, row, 2, bit, field_cfg)?
                - two.clone()
                    * bit_at_shifted_or_zero(trace, ShaWordCol::MajComp, row, 0, bit, field_cfg)?,
        )
    });

    Ok(VirtualChMajValues {
        ch1: ch1?,
        ch2: ch2?,
        maj: maj?,
    })
}

pub fn folded_row_integrand_values<F>(
    trace: &ProjectedTrace<F>,
    public: &ProjectedPublic<F>,
    r_ic: &[F; SHA_ROW_VARS],
    a: &F,
    lambda: &F,
    rho: &F,
    xi: &F,
    booleanity_sources: &[ShaBooleanitySource],
    field_cfg: &F::Config,
) -> Result<Vec<F>, ShaProjectionError>
where
    F: InnerTransparentField + DelayedFieldProductSum,
    F::Inner: Zero,
{
    validate_trace(trace)?;
    validate_public(public)?;
    let row_weights = build_eq_x_r_vec(r_ic, field_cfg)?;
    folded_row_integrand_values_with_row_weights(
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
pub fn folded_row_integrand_values_with_row_weights<F>(
    trace: &ProjectedTrace<F>,
    public: &ProjectedPublic<F>,
    row_weights: &[F],
    a: &F,
    lambda: &F,
    rho: &F,
    xi: &F,
    booleanity_sources: &[ShaBooleanitySource],
    field_cfg: &F::Config,
) -> Result<Vec<F>, ShaProjectionError>
where
    F: InnerTransparentField + DelayedFieldProductSum,
    F::Inner: Zero,
{
    let lambda_powers = build_sha_lambda_powers(lambda, field_cfg);
    let a_powers = build_sha_residual_eval_powers(a, field_cfg);
    let booleanity_weights = build_booleanity_weights(rho, xi, booleanity_sources.len(), field_cfg);
    folded_row_integrand_values_with_vectors(
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
pub fn folded_row_integrand_values_with_vectors<F>(
    trace: &ProjectedTrace<F>,
    public: &ProjectedPublic<F>,
    row_weights: &[F],
    a_powers: &[F],
    lambda_powers: &[F],
    booleanity_weights: &[F],
    booleanity_sources: &[ShaBooleanitySource],
    field_cfg: &F::Config,
) -> Result<Vec<F>, ShaProjectionError>
where
    F: InnerTransparentField + DelayedFieldProductSum,
    F::Inner: Zero,
{
    validate_trace(trace)?;
    validate_public(public)?;
    if row_weights.len() != SHA_ROW_COUNT {
        return Err(ShaProjectionError::ColumnRowCount {
            kind: "row_weights",
            col: 0,
            got: row_weights.len(),
            expected: SHA_ROW_COUNT,
        });
    }
    if lambda_powers.len() != NUM_SHA_RESIDUAL_FAMILIES {
        return Err(ShaProjectionError::MissingColumn {
            kind: "lambda_powers",
            col: lambda_powers.len(),
        });
    }
    if a_powers.len() < SHA_RESIDUAL_EVAL_POWER_COUNT {
        return Err(ShaProjectionError::MissingColumn {
            kind: "a_powers",
            col: a_powers.len(),
        });
    }
    if booleanity_weights.len() != booleanity_sources.len() {
        return Err(ShaProjectionError::ColumnRowCount {
            kind: "booleanity_weights",
            col: 0,
            got: booleanity_weights.len(),
            expected: booleanity_sources.len(),
        });
    }
    let needs_virtuals = sources_need_virtuals(booleanity_sources);

    let mut out = Vec::with_capacity(SHA_ROW_COUNT);
    for row in 0..SHA_ROW_COUNT {
        let linear = sha_linear_residual_row_value_with_powers(
            trace,
            public,
            row,
            &a_powers,
            &lambda_powers,
            field_cfg,
        )?;

        let mut bool_sum = F::zero_with_cfg(field_cfg);
        let virtuals = if needs_virtuals {
            Some(reconstruct_virtual_ch_maj_at_row_unchecked(
                trace, row, field_cfg,
            )?)
        } else {
            None
        };
        for (idx, source) in booleanity_sources.iter().enumerate() {
            let d = booleanity_source_value_at_row_with_virtuals(
                trace,
                row,
                source,
                virtuals.as_ref(),
                field_cfg,
            )?;
            let term = d.clone() * (d - F::one_with_cfg(field_cfg));
            bool_sum += booleanity_weights[idx].clone() * term;
        }
        out.push(row_weights[row].clone() * (linear + bool_sum));
    }
    Ok(out)
}

pub fn build_folded_row_sumcheck_group<F>(
    row_integrand_values: &[F],
    field_cfg: &F::Config,
) -> Result<MultiDegreeSumcheckGroup<F>, ShaProjectionError>
where
    F: InnerTransparentField + DelayedFieldProductSum + Send + Sync + 'static,
    F::Inner: Zero,
{
    if row_integrand_values.len() != SHA_ROW_COUNT {
        return Err(ShaProjectionError::ColumnRowCount {
            kind: "row_integrand",
            col: 0,
            got: row_integrand_values.len(),
            expected: SHA_ROW_COUNT,
        });
    }
    let zero_inner = F::zero_with_cfg(field_cfg).inner().clone();
    let integrand = DenseMultilinearExtension::from_evaluations_vec(
        SHA_ROW_VARS,
        row_integrand_values
            .iter()
            .map(|value| value.inner().clone())
            .collect(),
        zero_inner,
    );
    Ok(MultiDegreeSumcheckGroup::new(
        1,
        vec![integrand],
        Box::new(|values: &[F]| values[0].clone()),
    ))
}

pub fn folded_row_integrand_sum<F>(
    row_integrand_values: &[F],
    field_cfg: &F::Config,
) -> Result<F, ShaProjectionError>
where
    F: PrimeField,
{
    if row_integrand_values.len() != SHA_ROW_COUNT {
        return Err(ShaProjectionError::ColumnRowCount {
            kind: "row_integrand",
            col: 0,
            got: row_integrand_values.len(),
            expected: SHA_ROW_COUNT,
        });
    }
    Ok(row_integrand_values
        .iter()
        .fold(F::zero_with_cfg(field_cfg), |acc, value| acc + value))
}

/// Canonical booleanity sources for the production SHA ProjectionFold flow.
///
/// This includes every committed binary-polynomial SHA source bit and the
/// three virtual Ch/Maj residual families. The virtual values are reconstructed
/// from source bit slices; they are never independent witness columns.
pub fn production_sha_booleanity_sources() -> Vec<ShaBooleanitySource> {
    let mut sources = Vec::with_capacity(ShaWordCol::COUNT * SHA_WORD_BITS + 3 * SHA_WORD_BITS);
    for col_idx in 0..ShaWordCol::COUNT {
        let col = ShaWordCol::ALL[col_idx];
        for bit in 0..SHA_WORD_BITS {
            sources.push(ShaBooleanitySource::WordBit { col, bit });
        }
    }
    for bit in 0..SHA_WORD_BITS {
        sources.push(ShaBooleanitySource::VirtualCh1 { bit });
        sources.push(ShaBooleanitySource::VirtualCh2 { bit });
        sources.push(ShaBooleanitySource::VirtualMaj { bit });
    }
    sources
}

/// Evaluate the linear SHA residual scalarization at one row.
pub fn sha_linear_residual_row_value<F>(
    trace: &ProjectedTrace<F>,
    public: &ProjectedPublic<F>,
    row: usize,
    a: &F,
    lambda: &F,
    field_cfg: &F::Config,
) -> Result<F, ShaProjectionError>
where
    F: DelayedFieldProductSum,
{
    let lambda_powers = powers(
        lambda.clone(),
        F::one_with_cfg(field_cfg),
        NUM_SHA_RESIDUAL_FAMILIES,
    );
    let a_powers = powers(
        a.clone(),
        F::one_with_cfg(field_cfg),
        SHA_RESIDUAL_EVAL_POWER_COUNT,
    );
    sha_linear_residual_row_value_with_powers(
        trace,
        public,
        row,
        &a_powers,
        &lambda_powers,
        field_cfg,
    )
}

/// Evaluate the row-weighted linear SHA residual scalarization for one
/// instance.
pub fn sha_linear_residual_sum<F>(
    trace: &ProjectedTrace<F>,
    public: &ProjectedPublic<F>,
    r_ic: &[F; SHA_ROW_VARS],
    a: &F,
    lambda: &F,
    field_cfg: &F::Config,
) -> Result<F, ShaProjectionError>
where
    F: DelayedFieldProductSum,
{
    validate_trace(trace)?;
    validate_public(public)?;
    let row_weights = build_eq_x_r_vec(r_ic, field_cfg)?;
    let lambda_powers = powers(
        lambda.clone(),
        F::one_with_cfg(field_cfg),
        NUM_SHA_RESIDUAL_FAMILIES,
    );
    let a_powers = powers(
        a.clone(),
        F::one_with_cfg(field_cfg),
        SHA_RESIDUAL_EVAL_POWER_COUNT,
    );
    sha_linear_residual_sum_with_weights(
        trace,
        public,
        &row_weights,
        &a_powers,
        &lambda_powers,
        field_cfg,
    )
}

fn sha_linear_residual_sum_with_weights<F>(
    trace: &ProjectedTrace<F>,
    public: &ProjectedPublic<F>,
    row_weights: &[F],
    a_powers: &[F],
    lambda_powers: &[F],
    field_cfg: &F::Config,
) -> Result<F, ShaProjectionError>
where
    F: DelayedFieldProductSum,
{
    let mut sum = F::zero_with_cfg(field_cfg);
    for (row, row_weight) in row_weights.iter().enumerate() {
        sum += row_weight.clone()
            * sha_linear_residual_row_value_with_powers(
                trace,
                public,
                row,
                a_powers,
                lambda_powers,
                field_cfg,
            )?;
    }
    Ok(sum)
}

fn sha_linear_residual_row_value_with_powers<F>(
    trace: &ProjectedTrace<F>,
    public: &ProjectedPublic<F>,
    row: usize,
    a_powers: &[F],
    lambda_powers: &[F],
    field_cfg: &F::Config,
) -> Result<F, ShaProjectionError>
where
    F: DelayedFieldProductSum,
{
    let residuals = residual_values_at_row_with_powers(trace, public, row, a_powers, field_cfg)?;
    FieldFieldInnerProduct::inner_product::<UNCHECKED>(
        &residuals,
        lambda_powers,
        F::zero_with_cfg(field_cfg),
    )
    .map_err(ShaProjectionError::from)
}

#[derive(Clone, Debug)]
struct BinaryPrefixTailTable<F> {
    values: Vec<F>,
    prefix_vars: usize,
    tail_len: usize,
}

impl<F> BinaryPrefixTailTable<F>
where
    F: PrimeField,
{
    fn new(values: Vec<F>, prefix_vars: usize, tail_len: usize) -> Self {
        debug_assert_eq!(values.len(), binary_len(prefix_vars) * tail_len);
        Self {
            values,
            prefix_vars,
            tail_len,
        }
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn bind_first_axis(&mut self, r: &F, field_cfg: &F::Config) {
        debug_assert!(self.prefix_vars > 0);
        let rest_len = binary_len(self.prefix_vars - 1);
        let old_prefix_len = binary_len(self.prefix_vars);
        let one = F::one_with_cfg(field_cfg);
        let one_minus_r = one - r;
        let mut next = vec![F::zero_with_cfg(field_cfg); rest_len * self.tail_len];

        for tail in 0..self.tail_len {
            let old_tail_offset = tail * old_prefix_len;
            let new_tail_offset = tail * rest_len;
            for rest in 0..rest_len {
                let base = old_tail_offset + (rest << 1);
                next[new_tail_offset + rest] =
                    self.values[base].clone() * &one_minus_r + self.values[base + 1].clone() * r;
            }
        }

        self.values = next;
        self.prefix_vars -= 1;
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn value_with_first_axis(&self, rest: usize, tail: usize, x: &F, field_cfg: &F::Config) -> F {
        debug_assert!(self.prefix_vars > 0);
        let prefix_len = binary_len(self.prefix_vars);
        let base = tail * prefix_len + (rest << 1);
        let one = F::one_with_cfg(field_cfg);
        self.values[base].clone() * (one - x) + self.values[base + 1].clone() * x
    }
}

#[derive(Clone, Debug)]
struct TernaryPrefixTailTable<F> {
    values: Vec<F>,
    prefix_vars: usize,
    tail_len: usize,
}

impl<F> TernaryPrefixTailTable<F>
where
    F: PrimeField,
{
    fn new(
        values: Vec<F>,
        prefix_vars: usize,
        tail_len: usize,
    ) -> Result<Self, ShaProjectionError> {
        debug_assert_eq!(values.len(), checked_ternary_len(prefix_vars)? * tail_len);
        Ok(Self {
            values,
            prefix_vars,
            tail_len,
        })
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn bind_first_axis(&mut self, r: &F, field_cfg: &F::Config) -> Result<(), ShaProjectionError> {
        debug_assert!(self.prefix_vars > 0);
        let rest_len = checked_ternary_len(self.prefix_vars - 1)?;
        let old_prefix_len = checked_ternary_len(self.prefix_vars)?;
        let one = F::one_with_cfg(field_cfg);
        let one_minus_r = one - r;
        let quadratic = r.clone() * (r.clone() - F::one_with_cfg(field_cfg));
        let mut next = vec![F::zero_with_cfg(field_cfg); rest_len * self.tail_len];

        for tail in 0..self.tail_len {
            let old_tail_offset = tail * old_prefix_len;
            let new_tail_offset = tail * rest_len;
            for rest in 0..rest_len {
                let base = old_tail_offset + rest * 3;
                next[new_tail_offset + rest] = self.values[base].clone() * &one_minus_r
                    + self.values[base + 1].clone() * r
                    + self.values[base + 2].clone() * &quadratic;
            }
        }

        self.values = next;
        self.prefix_vars -= 1;
        Ok(())
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn value_with_first_axis(
        &self,
        rest: usize,
        tail: usize,
        x: &F,
        field_cfg: &F::Config,
    ) -> Result<F, ShaProjectionError> {
        debug_assert!(self.prefix_vars > 0);
        let prefix_len = checked_ternary_len(self.prefix_vars)?;
        let base = tail * prefix_len + rest * 3;
        let one = F::one_with_cfg(field_cfg);
        Ok(self.values[base].clone() * (one - x)
            + self.values[base + 1].clone() * x
            + self.values[base + 2].clone()
                * (x.clone() * (x.clone() - F::one_with_cfg(field_cfg))))
    }
}

struct RelationSumFoldPrefixFastPath<F: PrimeField> {
    traces: Box<[ProjectedTrace<F>]>,
    beta: Vec<F>,
    booleanity_sources: Vec<ShaBooleanitySource>,
    linear: BinaryPrefixTailTable<F>,
    booleanity: TernaryPrefixTailTable<F>,
    tail_eq_weights: Vec<F>,
    prefix_suffix_eq_weights: Vec<Vec<F>>,
    total_prefix_vars: usize,
    round: usize,
    eq_bound: F,
}

#[derive(Clone, Debug)]
struct TernaryCoeffPlan {
    support_mask: usize,
    finite_bits: usize,
    vertices: Vec<(usize, bool)>,
}

impl<F> RelationSumFoldPrefixFastPath<F>
where
    F: InnerTransparentField + DelayedFieldProductSum + Send + Sync + 'static,
    F::Inner: Zero,
{
    #[allow(clippy::too_many_arguments)]
    fn new(
        traces: &[ProjectedTrace<F>],
        publics: &[ProjectedPublic<F>],
        beta: &[F],
        r_ic: &[F; SHA_ROW_VARS],
        a: &F,
        lambda: &F,
        rho: &F,
        xi: &F,
        booleanity_sources: &[ShaBooleanitySource],
        prefix_vars: usize,
        field_cfg: &F::Config,
    ) -> Result<Self, ShaProjectionError> {
        Self::new_owned(
            traces.to_vec().into_boxed_slice(),
            publics,
            beta,
            r_ic,
            a,
            lambda,
            rho,
            xi,
            booleanity_sources,
            prefix_vars,
            field_cfg,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_owned(
        traces: Box<[ProjectedTrace<F>]>,
        publics: &[ProjectedPublic<F>],
        beta: &[F],
        r_ic: &[F; SHA_ROW_VARS],
        a: &F,
        lambda: &F,
        rho: &F,
        xi: &F,
        booleanity_sources: &[ShaBooleanitySource],
        prefix_vars: usize,
        field_cfg: &F::Config,
    ) -> Result<Self, ShaProjectionError> {
        let coeff_tables = build_linear_residual_coeff_tables(&traces, publics, r_ic, field_cfg)?;
        let row_weights = build_eq_x_r_vec(r_ic, field_cfg)?;
        Self::new_owned_with_linear_cache(
            traces,
            publics,
            beta,
            r_ic,
            &row_weights,
            a,
            lambda,
            rho,
            xi,
            booleanity_sources,
            prefix_vars,
            &coeff_tables,
            field_cfg,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_owned_with_linear_cache(
        traces: Box<[ProjectedTrace<F>]>,
        publics: &[ProjectedPublic<F>],
        beta: &[F],
        _r_ic: &[F; SHA_ROW_VARS],
        row_weights: &[F],
        a: &F,
        lambda: &F,
        rho: &F,
        xi: &F,
        booleanity_sources: &[ShaBooleanitySource],
        prefix_vars: usize,
        coeff_tables: &[LinearResidualCoeffTable<F>],
        field_cfg: &F::Config,
    ) -> Result<Self, ShaProjectionError> {
        let ell = validate_sha_sumfold_inputs(&traces, publics, beta)?;
        if prefix_vars == 0 || prefix_vars > ell {
            return Err(SumFoldError::Ell0TooLarge {
                ell0: prefix_vars,
                ell,
            }
            .into());
        }
        if coeff_tables.len() != traces.len() {
            return Err(ShaProjectionError::InstanceCountMismatch {
                got: coeff_tables.len(),
                expected: traces.len(),
            });
        }
        if row_weights.len() != SHA_ROW_COUNT {
            return Err(ShaProjectionError::ColumnRowCount {
                kind: "row_weights",
                col: 0,
                got: row_weights.len(),
                expected: SHA_ROW_COUNT,
            });
        }

        let tail_vars = ell - prefix_vars;
        let tail_len = binary_len(tail_vars);
        let a_powers = build_sha_residual_eval_powers(a, field_cfg);
        let lambda_powers = build_sha_lambda_powers(lambda, field_cfg);
        let booleanity_weights =
            build_booleanity_weights(rho, xi, booleanity_sources.len(), field_cfg);
        let linear_values = build_sha_sumfold_linear_accumulator(
            coeff_tables,
            &a_powers,
            &lambda_powers,
            field_cfg,
        )?;
        let quadratic_values = build_sha_booleanity_prefix_tail_table(
            &traces,
            booleanity_sources,
            prefix_vars,
            tail_len,
            row_weights,
            &booleanity_weights,
            field_cfg,
        );
        Self::new_owned_with_accumulators(
            traces,
            beta,
            &linear_values,
            &quadratic_values?,
            booleanity_sources,
            prefix_vars,
            field_cfg,
        )
    }

    fn new_owned_with_accumulators(
        traces: Box<[ProjectedTrace<F>]>,
        beta: &[F],
        linear_values: &[F],
        quadratic_values: &[F],
        booleanity_sources: &[ShaBooleanitySource],
        prefix_vars: usize,
        field_cfg: &F::Config,
    ) -> Result<Self, ShaProjectionError> {
        let ell = validate_sha_sumfold_traces(&traces, beta)?;
        if prefix_vars == 0 || prefix_vars > ell {
            return Err(SumFoldError::Ell0TooLarge {
                ell0: prefix_vars,
                ell,
            }
            .into());
        }

        let tail_vars = ell - prefix_vars;
        let tail_len = binary_len(tail_vars);
        let linear_len = binary_len(prefix_vars) * tail_len;
        if linear_values.len() != linear_len {
            return Err(ShaProjectionError::InstanceCountMismatch {
                got: linear_values.len(),
                expected: linear_len,
            });
        }
        let quadratic_len = checked_ternary_len(prefix_vars)? * tail_len;
        if quadratic_values.len() != quadratic_len {
            return Err(ShaProjectionError::InstanceCountMismatch {
                got: quadratic_values.len(),
                expected: quadratic_len,
            });
        }

        let linear = BinaryPrefixTailTable::new(linear_values.to_vec(), prefix_vars, tail_len);
        let booleanity =
            TernaryPrefixTailTable::new(quadratic_values.to_vec(), prefix_vars, tail_len)?;

        let tail_eq_weights = eq_weights_or_one(&beta[prefix_vars..], field_cfg)?;
        let mut prefix_suffix_eq_weights = Vec::with_capacity(prefix_vars);
        for round in 0..prefix_vars {
            prefix_suffix_eq_weights
                .push(eq_weights_or_one(&beta[round + 1..prefix_vars], field_cfg)?);
        }

        Ok(Self {
            traces,
            beta: beta.to_vec(),
            booleanity_sources: booleanity_sources.to_vec(),
            linear,
            booleanity,
            tail_eq_weights,
            prefix_suffix_eq_weights,
            total_prefix_vars: prefix_vars,
            round: 0,
            eq_bound: F::one_with_cfg(field_cfg),
        })
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn bind_previous_round(
        &mut self,
        r: &F,
        field_cfg: &F::Config,
    ) -> Result<(), ShaProjectionError> {
        let beta_idx = self.round - 1;
        self.eq_bound *= eq_one_var(&self.beta[beta_idx], r, field_cfg);
        self.linear.bind_first_axis(r, field_cfg);
        self.booleanity.bind_first_axis(r, field_cfg)
    }

    fn round_value_at(&self, x: &F, field_cfg: &F::Config) -> Result<F, ShaProjectionError> {
        debug_assert!(self.round < self.total_prefix_vars);
        let suffix_weights = &self.prefix_suffix_eq_weights[self.round];
        let rest_len = suffix_weights.len();
        let mut acc = F::zero_with_cfg(field_cfg);

        for tail in 0..self.tail_eq_weights.len() {
            for (rest, suffix_weight) in suffix_weights.iter().enumerate().take(rest_len) {
                let linear = self.linear.value_with_first_axis(rest, tail, x, field_cfg);
                let ternary_rest = binary_bits_to_ternary_index(rest, self.linear.prefix_vars - 1);
                let booleanity =
                    self.booleanity
                        .value_with_first_axis(ternary_rest, tail, x, field_cfg)?;
                acc += self.tail_eq_weights[tail].clone() * suffix_weight * (linear + booleanity);
            }
        }

        Ok(self.eq_bound.clone() * eq_one_var(&self.beta[self.round], x, field_cfg) * acc)
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn finish_tail_mles(
        mut self,
        prefix_challenges: &[F],
        field_cfg: &F::Config,
    ) -> Result<Vec<DenseMultilinearExtension<F::Inner>>, ShaProjectionError> {
        debug_assert_eq!(prefix_challenges.len(), self.total_prefix_vars);
        let tail_vars = self.beta.len() - self.total_prefix_vars;
        if tail_vars == 0 {
            return Ok(Vec::new());
        }

        while self.linear.prefix_vars > 0 {
            let next_axis = self.total_prefix_vars - self.linear.prefix_vars;
            let r = &prefix_challenges[next_axis];
            self.linear.bind_first_axis(r, field_cfg);
        }

        let tail_len = binary_len(tail_vars);
        debug_assert_eq!(self.linear.values.len(), tail_len);

        let prefix_weights = eq_weights_or_one(prefix_challenges, field_cfg)?;
        let eq_prefix_at_r = eq_eval(
            prefix_challenges,
            &self.beta[..self.total_prefix_vars],
            F::one_with_cfg(field_cfg),
        )?;
        let zero_inner = F::zero_with_cfg(field_cfg).inner().clone();

        let mut mles = Vec::with_capacity(2 + self.booleanity_sources.len() * SHA_ROW_COUNT);
        mles.push(DenseMultilinearExtension::from_evaluations_vec(
            tail_vars,
            self.tail_eq_weights
                .iter()
                .map(|tail_weight| (eq_prefix_at_r.clone() * tail_weight).inner().clone())
                .collect(),
            zero_inner.clone(),
        ));
        mles.push(DenseMultilinearExtension::from_evaluations_vec(
            tail_vars,
            self.linear
                .values
                .iter()
                .map(|value| value.inner().clone())
                .collect(),
            zero_inner.clone(),
        ));

        let source_tail_values = bind_sha_booleanity_sources_to_prefix(
            &self.traces,
            &self.booleanity_sources,
            self.total_prefix_vars,
            tail_len,
            &prefix_weights,
            field_cfg,
        )?;

        for values in source_tail_values {
            mles.push(DenseMultilinearExtension::from_evaluations_vec(
                tail_vars,
                values.iter().map(|value| value.inner().clone()).collect(),
                zero_inner.clone(),
            ));
        }

        Ok(mles)
    }
}

impl<F> PrefixFastPath<F> for RelationSumFoldPrefixFastPath<F>
where
    F: InnerTransparentField + DelayedFieldProductSum + Send + Sync + 'static,
    F::Inner: Zero,
{
    fn prefix_len(&self) -> usize {
        self.total_prefix_vars
    }

    fn prove_prefix_round(
        &mut self,
        verifier_msg: &Option<F>,
        config: &F::Config,
    ) -> PrefixRoundOutput<F> {
        if let Some(r) = verifier_msg {
            self.bind_previous_round(r, config)
                .expect("validated SHA prefix table should bind");
        }

        let zero = F::zero_with_cfg(config);
        let one = F::one_with_cfg(config);
        let two = one.clone() + &one;
        let three = two.clone() + &one;

        let p0 = self
            .round_value_at(&zero, config)
            .expect("validated SHA prefix table should evaluate at 0");
        let p1 = self
            .round_value_at(&one, config)
            .expect("validated SHA prefix table should evaluate at 1");
        let p2 = self
            .round_value_at(&two, config)
            .expect("validated SHA prefix table should evaluate at 2");
        let p3 = self
            .round_value_at(&three, config)
            .expect("validated SHA prefix table should evaluate at 3");

        let asserted_sum = if self.round == 0 {
            Some(p0 + &p1)
        } else {
            None
        };
        self.round += 1;

        PrefixRoundOutput {
            asserted_sum,
            tail_evaluations: vec![p1, p2, p3],
        }
    }

    fn finish_prefix(
        self: Box<Self>,
        prefix_challenges: &[F],
        config: &F::Config,
    ) -> Vec<DenseMultilinearExtension<F::Inner>> {
        self.finish_tail_mles(prefix_challenges, config)
            .expect("validated SHA prefix fast path should finish")
    }
}

/// Build the production SHA SumFold group over the instance axis.
///
/// The group proves the expression
///
/// `eq(beta, b) * (L(b) + xi * B(b))`
///
/// where `L` is the row-weighted linear SHA residual scalarization and `B`
/// is built from source booleanity MLEs. Unlike a table of fresh targets, the
/// booleanity part is evaluated from source MLEs, so the terminal at `r_b`
/// is the folded booleanity expression.
#[allow(clippy::too_many_arguments)]
pub fn build_dense_sha_sumfold_group<F>(
    traces: &[ProjectedTrace<F>],
    publics: &[ProjectedPublic<F>],
    beta: &[F],
    r_ic: &[F; SHA_ROW_VARS],
    a: &F,
    lambda: &F,
    rho: &F,
    xi: &F,
    booleanity_sources: &[ShaBooleanitySource],
    field_cfg: &F::Config,
) -> Result<MultiDegreeSumcheckGroup<F>, ShaProjectionError>
where
    F: InnerTransparentField + DelayedFieldProductSum + Send + Sync + 'static,
    F::Inner: Zero,
{
    let row_weights = build_eq_x_r_vec(r_ic, field_cfg)?;
    let beta_eq_weights = build_eq_x_r_vec(beta, field_cfg)?;
    build_dense_sha_sumfold_group_with_weights(
        traces,
        publics,
        beta,
        &beta_eq_weights,
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
pub fn build_dense_sha_sumfold_group_with_weights<F>(
    traces: &[ProjectedTrace<F>],
    publics: &[ProjectedPublic<F>],
    beta: &[F],
    beta_eq_weights: &[F],
    row_weights: &[F],
    a: &F,
    lambda: &F,
    rho: &F,
    xi: &F,
    booleanity_sources: &[ShaBooleanitySource],
    field_cfg: &F::Config,
) -> Result<MultiDegreeSumcheckGroup<F>, ShaProjectionError>
where
    F: InnerTransparentField + DelayedFieldProductSum + Send + Sync + 'static,
    F::Inner: Zero,
{
    let _ell = validate_sha_sumfold_inputs(traces, publics, beta)?;
    if beta_eq_weights.len() != traces.len() {
        return Err(ShaProjectionError::InstanceCountMismatch {
            got: beta_eq_weights.len(),
            expected: traces.len(),
        });
    }
    if row_weights.len() != SHA_ROW_COUNT {
        return Err(ShaProjectionError::ColumnRowCount {
            kind: "row_weights",
            col: 0,
            got: row_weights.len(),
            expected: SHA_ROW_COUNT,
        });
    }
    let lambda_powers = build_sha_lambda_powers(lambda, field_cfg);
    let a_powers = build_sha_residual_eval_powers(a, field_cfg);
    let booleanity_weights = build_booleanity_weights(rho, xi, booleanity_sources.len(), field_cfg);
    let linear_values = traces
        .iter()
        .zip(publics.iter())
        .map(|(trace, public)| {
            sha_linear_residual_sum_with_weights(
                trace,
                public,
                &row_weights,
                &a_powers,
                &lambda_powers,
                field_cfg,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    build_dense_sha_sumfold_group_from_accumulators(
        traces,
        beta,
        beta_eq_weights,
        row_weights,
        &linear_values,
        &booleanity_weights,
        booleanity_sources,
        field_cfg,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_dense_sha_sumfold_group_from_accumulators<F>(
    traces: &[ProjectedTrace<F>],
    beta: &[F],
    beta_eq_weights: &[F],
    row_weights: &[F],
    linear_accumulator: &[F],
    booleanity_weights: &[F],
    booleanity_sources: &[ShaBooleanitySource],
    field_cfg: &F::Config,
) -> Result<MultiDegreeSumcheckGroup<F>, ShaProjectionError>
where
    F: InnerTransparentField + DelayedFieldProductSum + Send + Sync + 'static,
    F::Inner: Zero,
{
    let ell = validate_sha_sumfold_traces(traces, beta)?;
    if beta_eq_weights.len() != traces.len() {
        return Err(ShaProjectionError::InstanceCountMismatch {
            got: beta_eq_weights.len(),
            expected: traces.len(),
        });
    }
    if row_weights.len() != SHA_ROW_COUNT {
        return Err(ShaProjectionError::ColumnRowCount {
            kind: "row_weights",
            col: 0,
            got: row_weights.len(),
            expected: SHA_ROW_COUNT,
        });
    }
    if linear_accumulator.len() != traces.len() {
        return Err(ShaProjectionError::InstanceCountMismatch {
            got: linear_accumulator.len(),
            expected: traces.len(),
        });
    }
    if booleanity_weights.len() != booleanity_sources.len() {
        return Err(ShaProjectionError::ColumnRowCount {
            kind: "booleanity_weights",
            col: 0,
            got: booleanity_weights.len(),
            expected: booleanity_sources.len(),
        });
    }

    let zero_inner = F::zero_with_cfg(field_cfg).inner().clone();
    let mut mles = Vec::with_capacity(2 + booleanity_sources.len() * SHA_ROW_COUNT);

    mles.push(DenseMultilinearExtension::from_evaluations_vec(
        ell,
        beta_eq_weights
            .iter()
            .map(|value| value.inner().clone())
            .collect(),
        zero_inner.clone(),
    ));
    mles.push(DenseMultilinearExtension::from_evaluations_vec(
        ell,
        linear_accumulator
            .iter()
            .map(|value| value.inner().clone())
            .collect(),
        zero_inner.clone(),
    ));

    for source in booleanity_sources {
        for row in 0..SHA_ROW_COUNT {
            let values = traces
                .iter()
                .map(|trace| booleanity_source_value_at_row(trace, row, source, field_cfg))
                .collect::<Result<Vec<_>, _>>()?;
            mles.push(DenseMultilinearExtension::from_evaluations_vec(
                ell,
                values.iter().map(|value| value.inner().clone()).collect(),
                zero_inner.clone(),
            ));
        }
    }

    Ok(MultiDegreeSumcheckGroup::new(
        3,
        mles,
        sha_weighted_sumfold_comb_fn(row_weights.to_vec(), booleanity_weights.to_vec(), field_cfg),
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn build_production_sha_sumfold_group_from_prefix_accumulators<F>(
    traces: &[ProjectedTrace<F>],
    beta: &[F],
    beta_eq_weights: &[F],
    row_weights: &[F],
    linear_accumulator: &[F],
    quadratic_prefix_accumulator: &[F],
    booleanity_weights: &[F],
    booleanity_sources: &[ShaBooleanitySource],
    prefix_vars: usize,
    field_cfg: &F::Config,
) -> Result<MultiDegreeSumcheckGroup<F>, ShaProjectionError>
where
    F: InnerTransparentField + DelayedFieldProductSum + Send + Sync + 'static,
    F::Inner: Zero,
{
    let ell = validate_sha_sumfold_traces(traces, beta)?;
    if beta_eq_weights.len() != traces.len() {
        return Err(ShaProjectionError::InstanceCountMismatch {
            got: beta_eq_weights.len(),
            expected: traces.len(),
        });
    }
    if row_weights.len() != SHA_ROW_COUNT {
        return Err(ShaProjectionError::ColumnRowCount {
            kind: "row_weights",
            col: 0,
            got: row_weights.len(),
            expected: SHA_ROW_COUNT,
        });
    }
    if linear_accumulator.len() != traces.len() {
        return Err(ShaProjectionError::InstanceCountMismatch {
            got: linear_accumulator.len(),
            expected: traces.len(),
        });
    }
    if booleanity_weights.len() != booleanity_sources.len() {
        return Err(ShaProjectionError::ColumnRowCount {
            kind: "booleanity_weights",
            col: 0,
            got: booleanity_weights.len(),
            expected: booleanity_sources.len(),
        });
    }
    if prefix_vars > ell {
        return Err(SumFoldError::Ell0TooLarge {
            ell0: prefix_vars,
            ell,
        }
        .into());
    }
    if prefix_vars == 0 {
        if !quadratic_prefix_accumulator.is_empty() {
            return Err(ShaProjectionError::InstanceCountMismatch {
                got: quadratic_prefix_accumulator.len(),
                expected: 0,
            });
        }
        return build_dense_sha_sumfold_group_from_accumulators(
            traces,
            beta,
            beta_eq_weights,
            row_weights,
            linear_accumulator,
            booleanity_weights,
            booleanity_sources,
            field_cfg,
        );
    }

    let fast_path = RelationSumFoldPrefixFastPath::new_owned_with_accumulators(
        traces.to_vec().into_boxed_slice(),
        beta,
        linear_accumulator,
        quadratic_prefix_accumulator,
        booleanity_sources,
        prefix_vars,
        field_cfg,
    );

    Ok(MultiDegreeSumcheckGroup::with_prefix_fast(
        3,
        Vec::new(),
        sha_weighted_sumfold_comb_fn(row_weights.to_vec(), booleanity_weights.to_vec(), field_cfg),
        Box::new(fast_path?),
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn build_production_sha_sumfold_group<F>(
    traces: &[ProjectedTrace<F>],
    publics: &[ProjectedPublic<F>],
    beta: &[F],
    r_ic: &[F; SHA_ROW_VARS],
    a: &F,
    lambda: &F,
    rho: &F,
    xi: &F,
    booleanity_sources: &[ShaBooleanitySource],
    prefix_vars: usize,
    field_cfg: &F::Config,
) -> Result<MultiDegreeSumcheckGroup<F>, ShaProjectionError>
where
    F: InnerTransparentField + DelayedFieldProductSum + Send + Sync + 'static,
    F::Inner: Zero,
{
    let ell = validate_sha_sumfold_inputs(traces, publics, beta)?;
    if prefix_vars > ell {
        return Err(SumFoldError::Ell0TooLarge {
            ell0: prefix_vars,
            ell,
        }
        .into());
    }
    if prefix_vars == 0 {
        return build_dense_sha_sumfold_group(
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
        );
    }

    let fast_path = RelationSumFoldPrefixFastPath::new(
        traces,
        publics,
        beta,
        r_ic,
        a,
        lambda,
        rho,
        xi,
        booleanity_sources,
        prefix_vars,
        field_cfg,
    );

    Ok(MultiDegreeSumcheckGroup::with_prefix_fast(
        3,
        Vec::new(),
        sha_sumfold_comb_fn(
            build_eq_x_r_vec(r_ic, field_cfg)?,
            powers(
                rho.clone(),
                F::one_with_cfg(field_cfg),
                booleanity_sources.len(),
            ),
            xi.clone(),
            field_cfg,
        ),
        Box::new(fast_path?),
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn build_production_sha_sumfold_group_with_linear_cache<F>(
    traces: &[ProjectedTrace<F>],
    publics: &[ProjectedPublic<F>],
    linear_cache: &[LinearResidualCoeffTable<F>],
    beta: &[F],
    r_ic: &[F; SHA_ROW_VARS],
    a: &F,
    lambda: &F,
    rho: &F,
    xi: &F,
    booleanity_sources: &[ShaBooleanitySource],
    prefix_vars: usize,
    field_cfg: &F::Config,
) -> Result<MultiDegreeSumcheckGroup<F>, ShaProjectionError>
where
    F: InnerTransparentField + DelayedFieldProductSum + Send + Sync + 'static,
    F::Inner: Zero,
{
    let row_weights = build_eq_x_r_vec(r_ic, field_cfg)?;
    let beta_eq_weights = build_eq_x_r_vec(beta, field_cfg)?;
    build_production_sha_sumfold_group_with_linear_cache_and_weights(
        traces,
        publics,
        linear_cache,
        beta,
        &beta_eq_weights,
        r_ic,
        &row_weights,
        a,
        lambda,
        rho,
        xi,
        booleanity_sources,
        prefix_vars,
        field_cfg,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn build_production_sha_sumfold_group_with_linear_cache_and_weights<F>(
    traces: &[ProjectedTrace<F>],
    publics: &[ProjectedPublic<F>],
    linear_cache: &[LinearResidualCoeffTable<F>],
    beta: &[F],
    beta_eq_weights: &[F],
    r_ic: &[F; SHA_ROW_VARS],
    row_weights: &[F],
    a: &F,
    lambda: &F,
    rho: &F,
    xi: &F,
    booleanity_sources: &[ShaBooleanitySource],
    prefix_vars: usize,
    field_cfg: &F::Config,
) -> Result<MultiDegreeSumcheckGroup<F>, ShaProjectionError>
where
    F: InnerTransparentField + DelayedFieldProductSum + Send + Sync + 'static,
    F::Inner: Zero,
{
    let ell = validate_sha_sumfold_inputs(traces, publics, beta)?;
    if linear_cache.len() != traces.len() {
        return Err(ShaProjectionError::InstanceCountMismatch {
            got: linear_cache.len(),
            expected: traces.len(),
        });
    }
    if beta_eq_weights.len() != traces.len() {
        return Err(ShaProjectionError::InstanceCountMismatch {
            got: beta_eq_weights.len(),
            expected: traces.len(),
        });
    }
    if row_weights.len() != SHA_ROW_COUNT {
        return Err(ShaProjectionError::ColumnRowCount {
            kind: "row_weights",
            col: 0,
            got: row_weights.len(),
            expected: SHA_ROW_COUNT,
        });
    }
    if prefix_vars > ell {
        return Err(SumFoldError::Ell0TooLarge {
            ell0: prefix_vars,
            ell,
        }
        .into());
    }
    if prefix_vars == 0 {
        return build_dense_sha_sumfold_group_with_linear_cache_and_weights(
            traces,
            linear_cache,
            beta,
            beta_eq_weights,
            row_weights,
            a,
            lambda,
            rho,
            xi,
            booleanity_sources,
            field_cfg,
        );
    }

    let fast_path = RelationSumFoldPrefixFastPath::new_owned_with_linear_cache(
        traces.to_vec().into_boxed_slice(),
        publics,
        beta,
        r_ic,
        row_weights,
        a,
        lambda,
        rho,
        xi,
        booleanity_sources,
        prefix_vars,
        linear_cache,
        field_cfg,
    );

    Ok(MultiDegreeSumcheckGroup::with_prefix_fast(
        3,
        Vec::new(),
        sha_sumfold_comb_fn(
            row_weights.to_vec(),
            powers(
                rho.clone(),
                F::one_with_cfg(field_cfg),
                booleanity_sources.len(),
            ),
            xi.clone(),
            field_cfg,
        ),
        Box::new(fast_path?),
    ))
}

#[allow(clippy::too_many_arguments)]
fn build_dense_sha_sumfold_group_with_linear_cache_and_weights<F>(
    traces: &[ProjectedTrace<F>],
    linear_cache: &[LinearResidualCoeffTable<F>],
    beta: &[F],
    beta_eq_weights: &[F],
    row_weights: &[F],
    a: &F,
    lambda: &F,
    rho: &F,
    xi: &F,
    booleanity_sources: &[ShaBooleanitySource],
    field_cfg: &F::Config,
) -> Result<MultiDegreeSumcheckGroup<F>, ShaProjectionError>
where
    F: InnerTransparentField + DelayedFieldProductSum + Send + Sync + 'static,
    F::Inner: Zero,
{
    if beta_eq_weights.len() != traces.len() {
        return Err(ShaProjectionError::InstanceCountMismatch {
            got: beta_eq_weights.len(),
            expected: traces.len(),
        });
    }
    if row_weights.len() != SHA_ROW_COUNT {
        return Err(ShaProjectionError::ColumnRowCount {
            kind: "row_weights",
            col: 0,
            got: row_weights.len(),
            expected: SHA_ROW_COUNT,
        });
    }
    let a_powers = build_sha_residual_eval_powers(a, field_cfg);
    let lambda_powers = build_sha_lambda_powers(lambda, field_cfg);
    let booleanity_weights = build_booleanity_weights(rho, xi, booleanity_sources.len(), field_cfg);
    let linear_values =
        build_sha_sumfold_linear_accumulator(linear_cache, &a_powers, &lambda_powers, field_cfg)?;
    build_dense_sha_sumfold_group_from_accumulators(
        traces,
        beta,
        beta_eq_weights,
        row_weights,
        &linear_values,
        &booleanity_weights,
        booleanity_sources,
        field_cfg,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn build_production_sha_sumfold_group_owned<F>(
    traces: Box<[ProjectedTrace<F>]>,
    publics: &[ProjectedPublic<F>],
    beta: &[F],
    r_ic: &[F; SHA_ROW_VARS],
    a: &F,
    lambda: &F,
    rho: &F,
    xi: &F,
    booleanity_sources: &[ShaBooleanitySource],
    prefix_vars: usize,
    field_cfg: &F::Config,
) -> Result<MultiDegreeSumcheckGroup<F>, ShaProjectionError>
where
    F: InnerTransparentField + DelayedFieldProductSum + Send + Sync + 'static,
    F::Inner: Zero,
{
    let ell = validate_sha_sumfold_inputs(&traces, publics, beta)?;
    if prefix_vars > ell {
        return Err(SumFoldError::Ell0TooLarge {
            ell0: prefix_vars,
            ell,
        }
        .into());
    }
    if prefix_vars == 0 {
        return build_dense_sha_sumfold_group(
            &traces,
            publics,
            beta,
            r_ic,
            a,
            lambda,
            rho,
            xi,
            booleanity_sources,
            field_cfg,
        );
    }

    let fast_path = RelationSumFoldPrefixFastPath::new_owned(
        traces,
        publics,
        beta,
        r_ic,
        a,
        lambda,
        rho,
        xi,
        booleanity_sources,
        prefix_vars,
        field_cfg,
    );

    Ok(MultiDegreeSumcheckGroup::with_prefix_fast(
        3,
        Vec::new(),
        sha_sumfold_comb_fn(
            build_eq_x_r_vec(r_ic, field_cfg)?,
            powers(
                rho.clone(),
                F::one_with_cfg(field_cfg),
                booleanity_sources.len(),
            ),
            xi.clone(),
            field_cfg,
        ),
        Box::new(fast_path?),
    ))
}

fn sha_sumfold_comb_fn<F>(
    row_weights: Vec<F>,
    rho_powers: Vec<F>,
    xi: F,
    field_cfg: &F::Config,
) -> CombFn<F>
where
    F: PrimeField + Send + Sync + 'static,
{
    let booleanity_weights = rho_powers
        .into_iter()
        .map(|rho_power| xi.clone() * rho_power)
        .collect();
    sha_weighted_sumfold_comb_fn(row_weights, booleanity_weights, field_cfg)
}

fn sha_weighted_sumfold_comb_fn<F>(
    row_weights: Vec<F>,
    booleanity_weights: Vec<F>,
    field_cfg: &F::Config,
) -> CombFn<F>
where
    F: PrimeField + Send + Sync + 'static,
{
    let zero_for_comb = F::zero_with_cfg(field_cfg);
    let one_for_comb = F::one_with_cfg(field_cfg);
    Box::new(move |values: &[F]| {
        let eq_beta = values[0].clone();
        let linear = values[1].clone();
        let mut bool_sum = zero_for_comb.clone();
        let mut cursor = 2usize;
        for booleanity_weight in &booleanity_weights {
            for row_weight in &row_weights {
                let d = values[cursor].clone();
                cursor += 1;
                let term = d.clone() * (d - one_for_comb.clone());
                bool_sum += row_weight.clone() * booleanity_weight * term;
            }
        }
        eq_beta * (linear + bool_sum)
    })
}

fn validate_sha_sumfold_inputs<F>(
    traces: &[ProjectedTrace<F>],
    publics: &[ProjectedPublic<F>],
    beta: &[F],
) -> Result<usize, ShaProjectionError> {
    if traces.is_empty() {
        return Err(ShaProjectionError::InstanceCountNotPowerOfTwo { got: 0 });
    }
    if traces.len() != publics.len() {
        return Err(ShaProjectionError::InstanceCountMismatch {
            got: publics.len(),
            expected: traces.len(),
        });
    }
    if !traces.len().is_power_of_two() {
        return Err(ShaProjectionError::InstanceCountNotPowerOfTwo { got: traces.len() });
    }
    let ell = usize::try_from(traces.len().trailing_zeros()).expect("trailing_zeros fits usize");
    if beta.len() != ell {
        return Err(ShaProjectionError::InstanceCountMismatch {
            got: beta.len(),
            expected: ell,
        });
    }
    for trace in traces {
        validate_trace(trace)?;
    }
    for public in publics {
        validate_public(public)?;
    }
    Ok(ell)
}

fn binary_len(vars: usize) -> usize {
    1usize
        .checked_shl(u32::try_from(vars).expect("vars fits u32"))
        .expect("binary domain size fits usize")
}

fn checked_ternary_len(vars: usize) -> Result<usize, ShaProjectionError> {
    let mut size = 1usize;
    for _ in 0..vars {
        size = size
            .checked_mul(3)
            .ok_or(SumFoldError::DomainTooLarge { ell: vars })?;
    }
    Ok(size)
}

fn eq_weights_or_one<F>(point: &[F], field_cfg: &F::Config) -> Result<Vec<F>, ShaProjectionError>
where
    F: PrimeField,
{
    if point.is_empty() {
        Ok(vec![F::one_with_cfg(field_cfg)])
    } else {
        Ok(build_eq_x_r_vec(point, field_cfg)?)
    }
}

fn eq_one_var<F>(beta: &F, x: &F, field_cfg: &F::Config) -> F
where
    F: PrimeField,
{
    let one = F::one_with_cfg(field_cfg);
    x.clone() * beta + (one.clone() - x) * (one - beta)
}

fn validate_sha_sumfold_traces<F>(
    traces: &[ProjectedTrace<F>],
    beta: &[F],
) -> Result<usize, ShaProjectionError> {
    if traces.is_empty() {
        return Err(ShaProjectionError::InstanceCountNotPowerOfTwo { got: 0 });
    }
    if !traces.len().is_power_of_two() {
        return Err(ShaProjectionError::InstanceCountNotPowerOfTwo { got: traces.len() });
    }
    let ell = usize::try_from(traces.len().trailing_zeros()).expect("trailing_zeros fits usize");
    if beta.len() != ell {
        return Err(ShaProjectionError::InstanceCountMismatch {
            got: beta.len(),
            expected: ell,
        });
    }
    for trace in traces {
        validate_trace(trace)?;
    }
    Ok(ell)
}

#[allow(clippy::too_many_arguments)]
pub fn build_sha_sumfold_quadratic_prefix_accumulator<F>(
    traces: &[ProjectedTrace<F>],
    booleanity_sources: &[ShaBooleanitySource],
    prefix_vars: usize,
    row_weights: &[F],
    booleanity_weights: &[F],
    field_cfg: &F::Config,
) -> Result<Vec<F>, ShaProjectionError>
where
    F: PrimeField,
{
    if traces.is_empty() {
        return Err(ShaProjectionError::InstanceCountNotPowerOfTwo { got: 0 });
    }
    if !traces.len().is_power_of_two() {
        return Err(ShaProjectionError::InstanceCountNotPowerOfTwo { got: traces.len() });
    }
    let ell = usize::try_from(traces.len().trailing_zeros()).expect("trailing_zeros fits usize");
    if prefix_vars > ell {
        return Err(SumFoldError::Ell0TooLarge {
            ell0: prefix_vars,
            ell,
        }
        .into());
    }
    if row_weights.len() != SHA_ROW_COUNT {
        return Err(ShaProjectionError::ColumnRowCount {
            kind: "row_weights",
            col: 0,
            got: row_weights.len(),
            expected: SHA_ROW_COUNT,
        });
    }
    if booleanity_weights.len() != booleanity_sources.len() {
        return Err(ShaProjectionError::ColumnRowCount {
            kind: "booleanity_weights",
            col: 0,
            got: booleanity_weights.len(),
            expected: booleanity_sources.len(),
        });
    }
    for trace in traces {
        validate_trace(trace)?;
    }
    if prefix_vars == 0 {
        return Ok(Vec::new());
    }

    let tail_len = binary_len(ell - prefix_vars);
    build_sha_booleanity_prefix_tail_table(
        traces,
        booleanity_sources,
        prefix_vars,
        tail_len,
        row_weights,
        booleanity_weights,
        field_cfg,
    )
}

#[allow(clippy::arithmetic_side_effects)]
fn build_sha_booleanity_prefix_tail_table<F>(
    traces: &[ProjectedTrace<F>],
    booleanity_sources: &[ShaBooleanitySource],
    prefix_vars: usize,
    tail_len: usize,
    row_weights: &[F],
    booleanity_weights: &[F],
    field_cfg: &F::Config,
) -> Result<Vec<F>, ShaProjectionError>
where
    F: PrimeField,
{
    let prefix_len = binary_len(prefix_vars);
    let ternary_len = checked_ternary_len(prefix_vars)?;
    let mut table = vec![F::zero_with_cfg(field_cfg); ternary_len * tail_len];
    if booleanity_sources.is_empty() {
        return Ok(table);
    }

    let needs_virtuals = sources_need_virtuals(booleanity_sources);
    let coeff_plans = ternary_coeff_plans(prefix_vars)?;
    let mut source_values =
        vec![F::zero_with_cfg(field_cfg); prefix_len * booleanity_sources.len()];
    for tail in 0..tail_len {
        for (row, row_weight) in row_weights.iter().enumerate().take(SHA_ROW_COUNT) {
            fill_booleanity_source_prefix_values(
                traces,
                booleanity_sources,
                prefix_vars,
                tail,
                row,
                needs_virtuals,
                &mut source_values,
                field_cfg,
            )?;

            for (source_idx, booleanity_weight) in booleanity_weights.iter().enumerate() {
                let source_weight = row_weight.clone() * booleanity_weight;
                for (ternary_idx, plan) in coeff_plans.iter().enumerate() {
                    let coeff = booleanity_degree_two_coeff(
                        &source_values,
                        booleanity_sources.len(),
                        source_idx,
                        plan,
                        field_cfg,
                    );
                    table[tail * ternary_len + ternary_idx] += source_weight.clone() * coeff;
                }
            }
        }
    }
    Ok(table)
}

#[allow(clippy::arithmetic_side_effects)]
fn bind_sha_booleanity_sources_to_prefix<F>(
    traces: &[ProjectedTrace<F>],
    booleanity_sources: &[ShaBooleanitySource],
    prefix_vars: usize,
    tail_len: usize,
    prefix_weights: &[F],
    field_cfg: &F::Config,
) -> Result<Vec<Vec<F>>, ShaProjectionError>
where
    F: PrimeField,
{
    let prefix_len = binary_len(prefix_vars);
    let source_count = booleanity_sources.len();
    let needs_virtuals = sources_need_virtuals(booleanity_sources);
    let mut source_values = vec![F::zero_with_cfg(field_cfg); prefix_len * source_count];
    let mut out = vec![vec![F::zero_with_cfg(field_cfg); tail_len]; source_count * SHA_ROW_COUNT];

    for tail in 0..tail_len {
        for row in 0..SHA_ROW_COUNT {
            fill_booleanity_source_prefix_values(
                traces,
                booleanity_sources,
                prefix_vars,
                tail,
                row,
                needs_virtuals,
                &mut source_values,
                field_cfg,
            )?;
            for source_idx in 0..source_count {
                let mut acc = F::zero_with_cfg(field_cfg);
                for (prefix, weight) in prefix_weights.iter().enumerate().take(prefix_len) {
                    acc += weight.clone() * &source_values[prefix * source_count + source_idx];
                }
                out[source_idx * SHA_ROW_COUNT + row][tail] = acc;
            }
        }
    }

    Ok(out)
}

#[allow(clippy::arithmetic_side_effects)]
fn fill_booleanity_source_prefix_values<F>(
    traces: &[ProjectedTrace<F>],
    booleanity_sources: &[ShaBooleanitySource],
    prefix_vars: usize,
    tail: usize,
    row: usize,
    needs_virtuals: bool,
    out: &mut [F],
    field_cfg: &F::Config,
) -> Result<(), ShaProjectionError>
where
    F: PrimeField,
{
    let prefix_len = binary_len(prefix_vars);
    let source_count = booleanity_sources.len();
    debug_assert_eq!(out.len(), prefix_len * source_count);

    for prefix in 0..prefix_len {
        let instance_idx = prefix + (tail << prefix_vars);
        let trace = &traces[instance_idx];
        let virtuals = if needs_virtuals {
            Some(reconstruct_virtual_ch_maj_at_row_unchecked(
                trace, row, field_cfg,
            )?)
        } else {
            None
        };

        for (source_idx, source) in booleanity_sources.iter().enumerate() {
            let value = match source {
                ShaBooleanitySource::WordBit { col, bit } => {
                    bit_at_shifted_or_zero(trace, *col, row, 0, *bit, field_cfg)?
                }
                ShaBooleanitySource::VirtualCh1 { bit } => {
                    virtual_bit_at(&virtuals.as_ref().expect("virtuals computed").ch1, *bit)?
                }
                ShaBooleanitySource::VirtualCh2 { bit } => {
                    virtual_bit_at(&virtuals.as_ref().expect("virtuals computed").ch2, *bit)?
                }
                ShaBooleanitySource::VirtualMaj { bit } => {
                    virtual_bit_at(&virtuals.as_ref().expect("virtuals computed").maj, *bit)?
                }
            };
            out[prefix * source_count + source_idx] = value;
        }
    }

    Ok(())
}

#[allow(clippy::arithmetic_side_effects)]
fn booleanity_degree_two_coeff<F>(
    source_values: &[F],
    source_count: usize,
    source_idx: usize,
    plan: &TernaryCoeffPlan,
    field_cfg: &F::Config,
) -> F
where
    F: PrimeField,
{
    let value_at =
        |prefix: usize| -> F { source_values[prefix * source_count + source_idx].clone() };
    if plan.support_mask == 0 {
        let d = value_at(plan.finite_bits);
        return d.clone() * (d - F::one_with_cfg(field_cfg));
    }

    let mut delta = F::zero_with_cfg(field_cfg);
    for (prefix, positive) in &plan.vertices {
        if *positive {
            delta += value_at(*prefix);
        } else {
            delta -= value_at(*prefix);
        }
    }

    delta.clone() * delta
}

#[allow(clippy::arithmetic_side_effects)]
fn ternary_point_parts(mut index: usize, prefix_vars: usize) -> (usize, usize) {
    let mut support_mask = 0usize;
    let mut finite_bits = 0usize;
    for var in 0..prefix_vars {
        let digit = index % 3;
        index /= 3;
        match digit {
            0 => {}
            1 => finite_bits |= 1usize << var,
            2 => support_mask |= 1usize << var,
            _ => unreachable!("ternary digit is always 0, 1, or 2"),
        }
    }
    (support_mask, finite_bits)
}

fn sources_need_virtuals(booleanity_sources: &[ShaBooleanitySource]) -> bool {
    booleanity_sources.iter().any(|source| {
        matches!(
            source,
            ShaBooleanitySource::VirtualCh1 { .. }
                | ShaBooleanitySource::VirtualCh2 { .. }
                | ShaBooleanitySource::VirtualMaj { .. }
        )
    })
}

#[allow(clippy::arithmetic_side_effects)]
fn ternary_coeff_plans(prefix_vars: usize) -> Result<Vec<TernaryCoeffPlan>, ShaProjectionError> {
    let ternary_len = checked_ternary_len(prefix_vars)?;
    let mut plans = Vec::with_capacity(ternary_len);
    for ternary_idx in 0..ternary_len {
        let (support_mask, finite_bits) = ternary_point_parts(ternary_idx, prefix_vars);
        let mut vertices = Vec::new();
        if support_mask != 0 {
            let mut support_bits = [0usize; usize::BITS as usize];
            let mut mask = support_mask;
            let mut support_size = 0usize;
            while mask != 0 {
                let bit = mask & mask.wrapping_neg();
                support_bits[support_size] = bit;
                support_size += 1;
                mask ^= bit;
            }
            vertices.reserve(1usize << support_size);
            for vertex in 0..(1usize << support_size) {
                let mut prefix = finite_bits;
                for (pos, bit) in support_bits[..support_size].iter().enumerate() {
                    if ((vertex >> pos) & 1) == 1 {
                        prefix |= *bit;
                    }
                }
                let positive = (support_size
                    - usize::try_from(vertex.count_ones()).expect("count_ones fits usize"))
                    % 2
                    == 0;
                vertices.push((prefix, positive));
            }
        }
        plans.push(TernaryCoeffPlan {
            support_mask,
            finite_bits,
            vertices,
        });
    }
    Ok(plans)
}

#[allow(clippy::arithmetic_side_effects)]
fn binary_bits_to_ternary_index(mut bits: usize, vars: usize) -> usize {
    let mut index = 0usize;
    let mut scale = 1usize;
    for _ in 0..vars {
        if bits & 1 == 1 {
            index += scale;
        }
        bits >>= 1;
        scale *= 3;
    }
    index
}

/// Build the expression-backed folded row sumcheck group.
///
/// The terminal at the verifier challenge is tied to source MLE endpoint
/// values, including booleanity sources, rather than to an opaque MLE of
/// precomputed row-integrand values.
#[allow(clippy::too_many_arguments)]
pub fn build_expression_folded_row_sumcheck_group<F>(
    trace: &ProjectedTrace<F>,
    public: &ProjectedPublic<F>,
    r_ic: &[F; SHA_ROW_VARS],
    a: &F,
    lambda: &F,
    rho: &F,
    xi: &F,
    booleanity_sources: &[ShaBooleanitySource],
    field_cfg: &F::Config,
) -> Result<MultiDegreeSumcheckGroup<F>, ShaProjectionError>
where
    F: InnerTransparentField + DelayedFieldProductSum + Send + Sync + 'static,
    F::Inner: Zero,
{
    let row_weights = build_eq_x_r_vec(r_ic, field_cfg)?;
    build_expression_folded_row_sumcheck_group_with_row_weights(
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
pub fn build_expression_folded_row_sumcheck_group_with_row_weights<F>(
    trace: &ProjectedTrace<F>,
    public: &ProjectedPublic<F>,
    row_weights: &[F],
    a: &F,
    lambda: &F,
    rho: &F,
    xi: &F,
    booleanity_sources: &[ShaBooleanitySource],
    field_cfg: &F::Config,
) -> Result<MultiDegreeSumcheckGroup<F>, ShaProjectionError>
where
    F: InnerTransparentField + DelayedFieldProductSum + Send + Sync + 'static,
    F::Inner: Zero,
{
    validate_trace(trace)?;
    validate_public(public)?;
    if row_weights.len() != SHA_ROW_COUNT {
        return Err(ShaProjectionError::ColumnRowCount {
            kind: "row_weights",
            col: 0,
            got: row_weights.len(),
            expected: SHA_ROW_COUNT,
        });
    }

    let zero = F::zero_with_cfg(field_cfg);
    let zero_inner = zero.inner().clone();
    let mut mles = Vec::with_capacity(2 + booleanity_sources.len());

    mles.push(DenseMultilinearExtension::from_evaluations_vec(
        SHA_ROW_VARS,
        row_weights
            .iter()
            .map(|value| value.inner().clone())
            .collect(),
        zero_inner.clone(),
    ));

    let linear_values = (0..SHA_ROW_COUNT)
        .map(|row| sha_linear_residual_row_value(trace, public, row, a, lambda, field_cfg))
        .collect::<Result<Vec<_>, _>>()?;
    mles.push(DenseMultilinearExtension::from_evaluations_vec(
        SHA_ROW_VARS,
        linear_values
            .iter()
            .map(|value| value.inner().clone())
            .collect(),
        zero_inner.clone(),
    ));

    let needs_virtuals = sources_need_virtuals(booleanity_sources);
    let mut source_values = (0..booleanity_sources.len())
        .map(|_| Vec::with_capacity(SHA_ROW_COUNT))
        .collect::<Vec<_>>();
    for row in 0..SHA_ROW_COUNT {
        let virtuals = if needs_virtuals {
            Some(reconstruct_virtual_ch_maj_at_row_unchecked(
                trace, row, field_cfg,
            )?)
        } else {
            None
        };
        for (source_idx, source) in booleanity_sources.iter().enumerate() {
            source_values[source_idx].push(booleanity_source_value_at_row_with_virtuals(
                trace,
                row,
                source,
                virtuals.as_ref(),
                field_cfg,
            )?);
        }
    }
    for values in source_values {
        mles.push(DenseMultilinearExtension::from_evaluations_vec(
            SHA_ROW_VARS,
            values.iter().map(|value| value.inner().clone()).collect(),
            zero_inner.clone(),
        ));
    }

    let rho_powers = powers(
        rho.clone(),
        F::one_with_cfg(field_cfg),
        booleanity_sources.len(),
    );
    let xi = xi.clone();
    let zero_for_comb = F::zero_with_cfg(field_cfg);
    let one_for_comb = F::one_with_cfg(field_cfg);
    Ok(MultiDegreeSumcheckGroup::new(
        3,
        mles,
        Box::new(move |values: &[F]| {
            let row_weight = values[0].clone();
            let linear = values[1].clone();
            let mut bool_sum = zero_for_comb.clone();
            for (d, rho_power) in values[2..].iter().zip(rho_powers.iter()) {
                let term = d.clone() * (d.clone() - one_for_comb.clone());
                bool_sum += rho_power.clone() * term;
            }
            row_weight * (linear + xi.clone() * bool_sum)
        }),
    ))
}

/// Claimed sum for the expression-backed folded row check.
#[allow(clippy::too_many_arguments)]
pub fn expression_folded_row_sum<F>(
    trace: &ProjectedTrace<F>,
    public: &ProjectedPublic<F>,
    r_ic: &[F; SHA_ROW_VARS],
    a: &F,
    lambda: &F,
    rho: &F,
    xi: &F,
    booleanity_sources: &[ShaBooleanitySource],
    field_cfg: &F::Config,
) -> Result<F, ShaProjectionError>
where
    F: InnerTransparentField + DelayedFieldProductSum,
    F::Inner: Zero,
{
    let values = folded_row_integrand_values(
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
    folded_row_integrand_sum(&values, field_cfg)
}

/// Claimed sum for the expression-backed folded row check with precomputed
/// `eq(r_ic, row)` weights.
#[allow(clippy::too_many_arguments)]
pub fn expression_folded_row_sum_with_row_weights<F>(
    trace: &ProjectedTrace<F>,
    public: &ProjectedPublic<F>,
    row_weights: &[F],
    a: &F,
    lambda: &F,
    rho: &F,
    xi: &F,
    booleanity_sources: &[ShaBooleanitySource],
    field_cfg: &F::Config,
) -> Result<F, ShaProjectionError>
where
    F: InnerTransparentField + DelayedFieldProductSum,
    F::Inner: Zero,
{
    let values = folded_row_integrand_values_with_row_weights(
        trace,
        public,
        row_weights,
        a,
        lambda,
        rho,
        xi,
        booleanity_sources,
        field_cfg,
    )?;
    folded_row_integrand_sum(&values, field_cfg)
}

#[allow(clippy::too_many_arguments)]
pub fn expression_folded_row_sum_with_vectors<F>(
    trace: &ProjectedTrace<F>,
    public: &ProjectedPublic<F>,
    row_weights: &[F],
    a_powers: &[F],
    lambda_powers: &[F],
    booleanity_weights: &[F],
    booleanity_sources: &[ShaBooleanitySource],
    field_cfg: &F::Config,
) -> Result<F, ShaProjectionError>
where
    F: InnerTransparentField + DelayedFieldProductSum,
    F::Inner: Zero,
{
    let values = folded_row_integrand_values_with_vectors(
        trace,
        public,
        row_weights,
        a_powers,
        lambda_powers,
        booleanity_weights,
        booleanity_sources,
        field_cfg,
    )?;
    folded_row_integrand_sum(&values, field_cfg)
}

pub fn sha_word_bits_at_point<F>(
    trace: &ProjectedTrace<F>,
    col: ShaWordCol,
    shift: usize,
    point: &[F],
    field_cfg: &F::Config,
) -> Result<[F; SHA_WORD_BITS], ShaProjectionError>
where
    F: PrimeField,
{
    if point.len() != SHA_ROW_VARS {
        return Err(ShaProjectionError::RowPointLength { got: point.len() });
    }
    validate_trace(trace)?;
    let row_weights = build_eq_x_r_vec(point, field_cfg)?;
    sha_word_bits_at_point_with_weights(trace, col, shift, &row_weights, field_cfg)
}

pub fn sha_word_bits_at_point_with_weights<F>(
    trace: &ProjectedTrace<F>,
    col: ShaWordCol,
    shift: usize,
    row_weights: &[F],
    field_cfg: &F::Config,
) -> Result<[F; SHA_WORD_BITS], ShaProjectionError>
where
    F: PrimeField,
{
    validate_trace(trace)?;
    if row_weights.len() != SHA_ROW_COUNT {
        return Err(ShaProjectionError::ColumnRowCount {
            kind: "row_weights",
            col: 0,
            got: row_weights.len(),
            expected: SHA_ROW_COUNT,
        });
    }
    let mut bits: [F; SHA_WORD_BITS] = std::array::from_fn(|_| F::zero_with_cfg(field_cfg));
    for (row, row_weight) in row_weights.iter().enumerate() {
        for (bit, out) in bits.iter_mut().enumerate() {
            *out += row_weight.clone()
                * bit_at_shifted_or_zero(trace, col, row, shift, bit, field_cfg)?;
        }
    }
    Ok(bits)
}

pub fn sha_scalarized_word_at_point<F>(
    trace: &ProjectedTrace<F>,
    col: ShaWordCol,
    shift: usize,
    point: &[F],
    field_cfg: &F::Config,
) -> Result<F, ShaProjectionError>
where
    F: PrimeField,
{
    if point.len() != SHA_ROW_VARS {
        return Err(ShaProjectionError::RowPointLength { got: point.len() });
    }
    validate_trace(trace)?;
    let row_weights = build_eq_x_r_vec(point, field_cfg)?;
    sha_scalarized_word_at_point_with_weights(trace, col, shift, &row_weights, field_cfg)
}

pub fn sha_scalarized_word_at_point_with_weights<F>(
    trace: &ProjectedTrace<F>,
    col: ShaWordCol,
    shift: usize,
    row_weights: &[F],
    field_cfg: &F::Config,
) -> Result<F, ShaProjectionError>
where
    F: PrimeField,
{
    validate_trace(trace)?;
    if row_weights.len() != SHA_ROW_COUNT {
        return Err(ShaProjectionError::ColumnRowCount {
            kind: "row_weights",
            col: 0,
            got: row_weights.len(),
            expected: SHA_ROW_COUNT,
        });
    }
    let mut value = F::zero_with_cfg(field_cfg);
    for (row, row_weight) in row_weights.iter().enumerate() {
        value += row_weight.clone()
            * scalarized_word_at_shifted_or_zero(trace, col, row, shift, field_cfg)?;
    }
    Ok(value)
}

pub fn sha_int_at_point<F>(
    trace: &ProjectedTrace<F>,
    col: ShaIntCol,
    point: &[F],
    field_cfg: &F::Config,
) -> Result<F, ShaProjectionError>
where
    F: PrimeField,
{
    if point.len() != SHA_ROW_VARS {
        return Err(ShaProjectionError::RowPointLength { got: point.len() });
    }
    validate_trace(trace)?;
    let row_weights = build_eq_x_r_vec(point, field_cfg)?;
    sha_int_at_point_with_weights(trace, col, &row_weights, field_cfg)
}

pub fn sha_int_at_point_with_weights<F>(
    trace: &ProjectedTrace<F>,
    col: ShaIntCol,
    row_weights: &[F],
    field_cfg: &F::Config,
) -> Result<F, ShaProjectionError>
where
    F: PrimeField,
{
    validate_trace(trace)?;
    if row_weights.len() != SHA_ROW_COUNT {
        return Err(ShaProjectionError::ColumnRowCount {
            kind: "row_weights",
            col: 0,
            got: row_weights.len(),
            expected: SHA_ROW_COUNT,
        });
    }
    let mut value = F::zero_with_cfg(field_cfg);
    for (row, row_weight) in row_weights.iter().enumerate() {
        value += row_weight.clone() * int_scalar(trace, col, row, field_cfg)?;
    }
    Ok(value)
}

pub fn sha_public_at_point<F>(
    public: &ProjectedPublic<F>,
    col: ShaPublicCol,
    shift: usize,
    point: &[F],
    field_cfg: &F::Config,
) -> Result<F, ShaProjectionError>
where
    F: PrimeField,
{
    if point.len() != SHA_ROW_VARS {
        return Err(ShaProjectionError::RowPointLength { got: point.len() });
    }
    validate_public(public)?;
    let row_weights = build_eq_x_r_vec(point, field_cfg)?;
    sha_public_at_point_with_weights(public, col, shift, &row_weights, field_cfg)
}

pub fn sha_public_at_point_with_weights<F>(
    public: &ProjectedPublic<F>,
    col: ShaPublicCol,
    shift: usize,
    row_weights: &[F],
    field_cfg: &F::Config,
) -> Result<F, ShaProjectionError>
where
    F: PrimeField,
{
    validate_public(public)?;
    if row_weights.len() != SHA_ROW_COUNT {
        return Err(ShaProjectionError::ColumnRowCount {
            kind: "row_weights",
            col: 0,
            got: row_weights.len(),
            expected: SHA_ROW_COUNT,
        });
    }
    let mut value = F::zero_with_cfg(field_cfg);
    for (row, row_weight) in row_weights.iter().enumerate() {
        let shifted = row.checked_add(shift).unwrap_or(SHA_ROW_COUNT);
        value += row_weight.clone() * public_scalar(public, col, shifted, field_cfg)?;
    }
    Ok(value)
}

pub fn verify_folded_row_sumcheck_claim<F>(
    claimed_sum: &F,
    final_round_sumcheck_claim: &F,
) -> Result<(), ShaProjectionError>
where
    F: PrimeField,
{
    if claimed_sum != final_round_sumcheck_claim {
        return Err(ShaProjectionError::FoldedRowClaimMismatch);
    }
    Ok(())
}

pub fn residual_polys_at_row<F>(
    trace: &ProjectedTrace<F>,
    public: &ProjectedPublic<F>,
    row: usize,
    field_cfg: &F::Config,
) -> Result<[DynamicPolynomialF<F>; NUM_SHA_RESIDUAL_FAMILIES], ShaProjectionError>
where
    F: PrimeField,
{
    let zero = F::zero_with_cfg(field_cfg);
    let one = F::one_with_cfg(field_cfg);
    let two = one.clone() + &one;
    let rho_sig0 = sparse_poly::<F>(&[10, 19, 30], field_cfg);
    let rho_sig1 = sparse_poly::<F>(&[7, 21, 26], field_cfg);

    let a = word_poly(trace, ShaWordCol::A, row, field_cfg)?;
    let e = word_poly(trace, ShaWordCol::E, row, field_cfg)?;
    let sigma0 = word_poly(trace, ShaWordCol::Sigma0, row, field_cfg)?;
    let sigma1 = word_poly(trace, ShaWordCol::Sigma1, row, field_cfg)?;
    let w = word_poly(trace, ShaWordCol::W, row, field_cfg)?;
    let small_sigma0 = word_poly(trace, ShaWordCol::SmallSigma0, row, field_cfg)?;
    let small_sigma1 = word_poly(trace, ShaWordCol::SmallSigma1, row, field_cfg)?;
    let ov_sigma0 = word_poly(trace, ShaWordCol::OvSigma0, row, field_cfg)?;
    let ov_sigma1 = word_poly(trace, ShaWordCol::OvSigma1, row, field_cfg)?;
    let ov_small_sigma0 = word_poly(trace, ShaWordCol::OvSmallSigma0, row, field_cfg)?;
    let ov_small_sigma1 = word_poly(trace, ShaWordCol::OvSmallSigma1, row, field_cfg)?;

    let r0 = (&a * &rho_sig0) - &sigma0 - &scale_poly(&ov_sigma0, &two);
    let r1 = (&e * &rho_sig1) - &sigma1 - &scale_poly(&ov_sigma1, &two);
    let r2 = word_poly(trace, ShaWordCol::W, row, field_cfg)?.rot_c(25)
        + &word_poly(trace, ShaWordCol::W, row, field_cfg)?.rot_c(14)
        + &word_poly(trace, ShaWordCol::W, row, field_cfg)?.shift_r_c(3)
        - &small_sigma0
        - &scale_poly(&ov_small_sigma0, &two);
    let r3 = word_poly(trace, ShaWordCol::W, row, field_cfg)?.rot_c(15)
        + &word_poly(trace, ShaWordCol::W, row, field_cfg)?.rot_c(13)
        + &word_poly(trace, ShaWordCol::W, row, field_cfg)?.shift_r_c(10)
        - &small_sigma1
        - &scale_poly(&ov_small_sigma1, &two);

    let mu_w = mu_contribution(trace, row, 0, 2, field_cfg)?;
    let mu_a = mu_contribution(trace, row, 2, 5, field_cfg)?;
    let mu_e = mu_contribution(trace, row, 5, 8, field_cfg)?;
    let mu_ff_a = mu_contribution(trace, row, 8, 9, field_cfg)?;
    let mu_ff_e = mu_contribution(trace, row, 9, 10, field_cfg)?;

    let r4 = word_poly_shifted(trace, ShaWordCol::W, row, 16, field_cfg)?
        - &w
        - &word_poly_shifted(trace, ShaWordCol::SmallSigma0, row, 1, field_cfg)?
        - &word_poly_shifted(trace, ShaWordCol::W, row, 9, field_cfg)?
        - &word_poly_shifted(trace, ShaWordCol::SmallSigma1, row, 14, field_cfg)?
        + &mu_w
        + &int_const_poly(trace, ShaIntCol::CompSchedule, row, field_cfg)?;

    let r5 = word_poly_shifted(trace, ShaWordCol::A, row, 4, field_cfg)?
        - &e
        - &word_poly_shifted(trace, ShaWordCol::Sigma1, row, 3, field_cfg)?
        - &word_poly_shifted(trace, ShaWordCol::Uef, row, 3, field_cfg)?
        - &word_poly_shifted(trace, ShaWordCol::UNegEg, row, 3, field_cfg)?
        - &public_const_poly(public, ShaPublicCol::K, row + 3, field_cfg)?
        - &w
        - &word_poly_shifted(trace, ShaWordCol::Sigma0, row, 3, field_cfg)?
        - &word_poly_shifted(trace, ShaWordCol::Maj, row, 3, field_cfg)?
        + &mu_a
        + &int_const_poly(trace, ShaIntCol::CompUpdateA, row, field_cfg)?;

    let r6 = word_poly_shifted(trace, ShaWordCol::E, row, 4, field_cfg)?
        - &a
        - &e
        - &word_poly_shifted(trace, ShaWordCol::Sigma1, row, 3, field_cfg)?
        - &word_poly_shifted(trace, ShaWordCol::Uef, row, 3, field_cfg)?
        - &word_poly_shifted(trace, ShaWordCol::UNegEg, row, 3, field_cfg)?
        - &public_const_poly(public, ShaPublicCol::K, row + 3, field_cfg)?
        - &w
        + &mu_e
        + &int_const_poly(trace, ShaIntCol::CompUpdateE, row, field_cfg)?;

    let s_init = public_scalar(public, ShaPublicCol::SInit, row, field_cfg)?;
    let s_msg = public_scalar(public, ShaPublicCol::SMsg, row, field_cfg)?;
    let s_sched = public_scalar(public, ShaPublicCol::SSched, row, field_cfg)?;
    let s_upd = public_scalar(public, ShaPublicCol::SUpd, row, field_cfg)?;
    let s_ff = public_scalar(public, ShaPublicCol::SFf, row, field_cfg)?;
    let s_out = public_scalar(public, ShaPublicCol::SOut, row, field_cfg)?;

    let r7 = scale_poly(
        &(a.clone() - &public_word_or_const_poly(public, ShaPublicCol::PAIn, row, field_cfg)?),
        &s_init,
    ) + &scale_poly(
        &(a.clone() - &public_word_or_const_poly(public, ShaPublicCol::PAOut, row, field_cfg)?),
        &s_out,
    );
    let r8 = scale_poly(
        &(e.clone() - &public_word_or_const_poly(public, ShaPublicCol::PEIn, row, field_cfg)?),
        &s_init,
    ) + &scale_poly(
        &(e.clone() - &public_word_or_const_poly(public, ShaPublicCol::PEOut, row, field_cfg)?),
        &s_out,
    );

    let r9 = word_poly_shifted(trace, ShaWordCol::A, row, 4, field_cfg)?
        - &a
        - &public_const_poly(public, ShaPublicCol::PAIn, row, field_cfg)?
        + &mu_ff_a
        + &int_const_poly(trace, ShaIntCol::CompFeedForwardA, row, field_cfg)?;
    let r10 = word_poly_shifted(trace, ShaWordCol::E, row, 4, field_cfg)?
        - &e
        - &public_const_poly(public, ShaPublicCol::PEIn, row, field_cfg)?
        + &mu_ff_e
        + &int_const_poly(trace, ShaIntCol::CompFeedForwardE, row, field_cfg)?;
    let r11 = scale_poly(
        &(w - &public_word_or_const_poly(public, ShaPublicCol::Message, row, field_cfg)?),
        &s_msg,
    );

    let comp_schedule = int_const_poly(trace, ShaIntCol::CompSchedule, row, field_cfg)?;
    let comp_update_a = int_const_poly(trace, ShaIntCol::CompUpdateA, row, field_cfg)?;
    let comp_update_e = int_const_poly(trace, ShaIntCol::CompUpdateE, row, field_cfg)?;
    let comp_ff_a = int_const_poly(trace, ShaIntCol::CompFeedForwardA, row, field_cfg)?;
    let comp_ff_e = int_const_poly(trace, ShaIntCol::CompFeedForwardE, row, field_cfg)?;

    let r12 = scale_poly(&comp_schedule, &s_sched);
    let r13 = scale_poly(&comp_update_a, &s_upd);
    let r14 = scale_poly(&comp_update_e, &s_upd);
    let r15 = scale_poly(&comp_ff_a, &s_ff);
    let r16 = scale_poly(&comp_ff_e, &s_ff);
    let r17 = word_poly(trace, ShaWordCol::MuPacked, row, field_cfg)?.shift_r_c(10);

    let mut residuals = [
        r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17,
    ];
    residuals.iter_mut().for_each(DynamicPolynomialF::trim);
    debug_assert_eq!(residuals.len(), NUM_SHA_RESIDUAL_FAMILIES);
    let _ = zero;
    Ok(residuals)
}

fn residual_values_at_row_with_powers<F>(
    trace: &ProjectedTrace<F>,
    public: &ProjectedPublic<F>,
    row: usize,
    a_powers: &[F],
    field_cfg: &F::Config,
) -> Result<[F; NUM_SHA_RESIDUAL_FAMILIES], ShaProjectionError>
where
    F: DelayedFieldProductSum,
{
    let polies = residual_polys_at_row(trace, public, row, field_cfg)?;
    let mut out: [F; NUM_SHA_RESIDUAL_FAMILIES] =
        std::array::from_fn(|_| F::zero_with_cfg(field_cfg));
    for (idx, poly) in polies.iter().enumerate() {
        out[idx] = evaluate_poly_at_powers_dmr(poly, a_powers, field_cfg)?;
    }
    Ok(out)
}

fn validate_trace<F>(trace: &ProjectedTrace<F>) -> Result<(), ShaProjectionError> {
    validate_table(
        "bit_slices",
        &trace.bit_slices,
        ShaWordCol::COUNT * SHA_WORD_BITS,
    )?;
    validate_table("scalarized", &trace.scalarized, ShaWordCol::COUNT)?;
    validate_table("int_columns", &trace.int_columns, ShaIntCol::COUNT)?;
    validate_table("public_columns", &trace.public_columns, ShaPublicCol::COUNT)
}

fn validate_public<F>(public: &ProjectedPublic<F>) -> Result<(), ShaProjectionError> {
    validate_table("public.columns", &public.columns, ShaPublicCol::COUNT)?;
    if let Some(bit_slices) = &public.bit_slices {
        validate_table(
            "public.bit_slices",
            bit_slices,
            ShaPublicWordCol::COUNT * SHA_WORD_BITS,
        )?;
    }
    Ok(())
}

fn validate_table<F>(
    kind: &'static str,
    columns: &MleTable<F>,
    expected_cols: usize,
) -> Result<(), ShaProjectionError> {
    if columns.len() != expected_cols {
        return Err(ShaProjectionError::MissingColumn {
            kind,
            col: columns.len(),
        });
    }
    for (col, values) in columns.iter().enumerate() {
        if values.num_vars != SHA_ROW_VARS || values.evaluations.len() != SHA_ROW_COUNT {
            return Err(ShaProjectionError::ColumnRowCount {
                kind,
                col,
                got: values.evaluations.len(),
                expected: SHA_ROW_COUNT,
            });
        }
    }
    Ok(())
}

fn word_poly<F>(
    trace: &ProjectedTrace<F>,
    col: ShaWordCol,
    row: usize,
    field_cfg: &F::Config,
) -> Result<DynamicPolynomialF<F>, ShaProjectionError>
where
    F: PrimeField,
{
    if row >= SHA_ROW_COUNT {
        return Ok(DynamicPolynomialF::ZERO);
    }
    let col_idx = col.index();
    let mut coeffs = Vec::with_capacity(SHA_WORD_BITS);
    for bit in 0..SHA_WORD_BITS {
        coeffs.push(scalar_from_table(
            "bit_slices",
            &trace.bit_slices,
            bit_slice_index(col_idx, bit, SHA_WORD_BITS),
            row,
            field_cfg,
        )?);
    }
    coeffs.resize(SHA_WORD_BITS, F::zero_with_cfg(field_cfg));
    Ok(DynamicPolynomialF::new_trimmed(coeffs))
}

fn word_poly_shifted<F>(
    trace: &ProjectedTrace<F>,
    col: ShaWordCol,
    row: usize,
    shift: usize,
    field_cfg: &F::Config,
) -> Result<DynamicPolynomialF<F>, ShaProjectionError>
where
    F: PrimeField,
{
    match row.checked_add(shift) {
        Some(shifted) if shifted < SHA_ROW_COUNT => word_poly(trace, col, shifted, field_cfg),
        _ => Ok(DynamicPolynomialF::ZERO),
    }
}

fn bit_at_shifted_or_zero<F>(
    trace: &ProjectedTrace<F>,
    col: ShaWordCol,
    row: usize,
    shift: usize,
    bit: usize,
    field_cfg: &F::Config,
) -> Result<F, ShaProjectionError>
where
    F: PrimeField,
{
    if bit >= SHA_WORD_BITS {
        return Err(ShaProjectionError::BitIndexOutOfRange { bit });
    }
    let Some(shifted) = row.checked_add(shift) else {
        return Ok(F::zero_with_cfg(field_cfg));
    };
    if shifted >= SHA_ROW_COUNT {
        return Ok(F::zero_with_cfg(field_cfg));
    }
    let col_idx = col.index();
    scalar_from_table(
        "bit_slices",
        &trace.bit_slices,
        bit_slice_index(col_idx, bit, SHA_WORD_BITS),
        shifted,
        field_cfg,
    )
}

fn int_const_poly<F>(
    trace: &ProjectedTrace<F>,
    col: ShaIntCol,
    row: usize,
    field_cfg: &F::Config,
) -> Result<DynamicPolynomialF<F>, ShaProjectionError>
where
    F: PrimeField,
{
    Ok(const_poly(
        int_scalar(trace, col, row, field_cfg)?,
        field_cfg,
    ))
}

fn public_const_poly<F>(
    public: &ProjectedPublic<F>,
    col: ShaPublicCol,
    row: usize,
    field_cfg: &F::Config,
) -> Result<DynamicPolynomialF<F>, ShaProjectionError>
where
    F: PrimeField,
{
    Ok(const_poly(
        public_scalar(public, col, row, field_cfg)?,
        field_cfg,
    ))
}

fn public_word_or_const_poly<F>(
    public: &ProjectedPublic<F>,
    col: ShaPublicCol,
    row: usize,
    field_cfg: &F::Config,
) -> Result<DynamicPolynomialF<F>, ShaProjectionError>
where
    F: PrimeField,
{
    let Some(word_col) = col.public_word_col() else {
        return public_const_poly(public, col, row, field_cfg);
    };
    let Some(bit_slices) = &public.bit_slices else {
        return public_const_poly(public, col, row, field_cfg);
    };
    if row >= SHA_ROW_COUNT {
        return Ok(DynamicPolynomialF::ZERO);
    }
    let col_idx = word_col.index();
    let mut bits = Vec::with_capacity(SHA_WORD_BITS);
    for bit in 0..SHA_WORD_BITS {
        bits.push(scalar_from_table(
            "public.bit_slices",
            bit_slices,
            bit_slice_index(col_idx, bit, SHA_WORD_BITS),
            row,
            field_cfg,
        )?);
    }
    Ok(DynamicPolynomialF::new_trimmed(bits))
}

fn int_scalar<F>(
    trace: &ProjectedTrace<F>,
    col: ShaIntCol,
    row: usize,
    field_cfg: &F::Config,
) -> Result<F, ShaProjectionError>
where
    F: PrimeField,
{
    if row >= SHA_ROW_COUNT {
        return Ok(F::zero_with_cfg(field_cfg));
    }
    scalar_from_table(
        "int_columns",
        &trace.int_columns,
        col.index(),
        row,
        field_cfg,
    )
}

fn public_scalar<F>(
    public: &ProjectedPublic<F>,
    col: ShaPublicCol,
    row: usize,
    field_cfg: &F::Config,
) -> Result<F, ShaProjectionError>
where
    F: PrimeField,
{
    if row >= SHA_ROW_COUNT {
        return Ok(F::zero_with_cfg(field_cfg));
    }
    scalar_from_table(
        "public.columns",
        &public.columns,
        col.index(),
        row,
        field_cfg,
    )
}

fn scalar_from_table<F>(
    kind: &'static str,
    columns: &MleTable<F>,
    col: usize,
    row: usize,
    field_cfg: &F::Config,
) -> Result<F, ShaProjectionError>
where
    F: PrimeField,
{
    let values = columns
        .get(col)
        .ok_or(ShaProjectionError::MissingColumn { kind, col })?;
    if values.num_vars != SHA_ROW_VARS || values.evaluations.len() != SHA_ROW_COUNT {
        return Err(ShaProjectionError::ColumnRowCount {
            kind,
            col,
            got: values.evaluations.len(),
            expected: SHA_ROW_COUNT,
        });
    }
    Ok(values
        .evaluations
        .get(row)
        .cloned()
        .unwrap_or_else(|| F::zero_with_cfg(field_cfg)))
}

fn const_poly<F: PrimeField>(value: F, _field_cfg: &F::Config) -> DynamicPolynomialF<F> {
    DynamicPolynomialF::new_trimmed([value])
}

fn sparse_poly<F: PrimeField>(indices: &[usize], field_cfg: &F::Config) -> DynamicPolynomialF<F> {
    let mut coeffs = vec![F::zero_with_cfg(field_cfg); SHA_WORD_BITS];
    for &idx in indices {
        coeffs[idx] = F::one_with_cfg(field_cfg);
    }
    DynamicPolynomialF::new_trimmed(coeffs)
}

fn scale_poly<F: PrimeField>(poly: &DynamicPolynomialF<F>, scalar: &F) -> DynamicPolynomialF<F> {
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

fn pow_two<F: PrimeField>(exp: usize, field_cfg: &F::Config) -> F {
    let two = F::one_with_cfg(field_cfg) + F::one_with_cfg(field_cfg);
    let mut out = F::one_with_cfg(field_cfg);
    for _ in 0..exp {
        out *= &two;
    }
    out
}

fn mu_contribution<F>(
    trace: &ProjectedTrace<F>,
    row: usize,
    low: usize,
    high: usize,
    field_cfg: &F::Config,
) -> Result<DynamicPolynomialF<F>, ShaProjectionError>
where
    F: PrimeField,
{
    let packed = word_poly(trace, ShaWordCol::MuPacked, row, field_cfg)?.shift_r_c(low as u32);
    let tail = word_poly(trace, ShaWordCol::MuPacked, row, field_cfg)?.shift_r_c(high as u32);
    let low_coeff = pow_two(32, field_cfg);
    let high_coeff = pow_two(32 + high - low, field_cfg);
    Ok(scale_poly(&packed, &low_coeff) - &scale_poly(&tail, &high_coeff))
}

fn evaluate_poly_at_powers_dmr<F>(
    poly: &DynamicPolynomialF<F>,
    powers: &[F],
    field_cfg: &F::Config,
) -> Result<F, ShaProjectionError>
where
    F: DelayedFieldProductSum,
{
    if poly.coeffs.is_empty() {
        return Ok(F::zero_with_cfg(field_cfg));
    }
    if poly.coeffs.len() > powers.len() {
        return Err(ShaProjectionError::NonCanonicalProofObject(
            "SHA polynomial exceeds precomputed scalarization power bound",
        ));
    }
    DynamicPolyFInnerProduct::inner_product::<UNCHECKED>(
        &poly.coeffs,
        &powers[..poly.coeffs.len()],
        F::zero_with_cfg(field_cfg),
    )
    .map_err(ShaProjectionError::from)
}

fn project_bits_dmr<F>(
    bits: &[F],
    powers: &[F],
    field_cfg: &F::Config,
) -> Result<F, ShaProjectionError>
where
    F: DelayedFieldProductSum,
{
    if bits.len() > powers.len() {
        return Err(ShaProjectionError::NonCanonicalProofObject(
            "SHA bit projection exceeds precomputed scalarization power bound",
        ));
    }
    FieldFieldInnerProduct::inner_product::<UNCHECKED>(
        bits,
        &powers[..bits.len()],
        F::zero_with_cfg(field_cfg),
    )
    .map_err(ShaProjectionError::from)
}

fn project_binary_bits_conditional_add_dmr<F>(
    bits: &[F],
    powers: &[F],
    field_cfg: &F::Config,
) -> Result<F, ShaProjectionError>
where
    F: MontgomeryLimbs + DelayedFieldProductSum + Send + Sync,
{
    if bits.len() > powers.len() {
        return Err(ShaProjectionError::NonCanonicalProofObject(
            "SHA binary bit projection exceeds precomputed scalarization power bound",
        ));
    }
    let one = F::one_with_cfg(field_cfg);
    let reduction_params = F::barrett_reduction_params(field_cfg);
    let flush_adds = dmr_flush_adds(&reduction_params);
    let mut bucket = Uint::<5>::zero();
    let mut pending_adds = 0usize;
    let mut acc = F::zero_with_cfg(field_cfg);

    for (bit, power) in bits.iter().zip(powers.iter()) {
        if F::is_zero(bit) {
            continue;
        }
        if bit != &one {
            return project_bits_dmr(bits, powers, field_cfg);
        }

        <Uint<5> as DelayedModularReduction<F>>::add(&mut bucket, power);
        pending_adds = pending_adds.saturating_add(1);
        if pending_adds >= flush_adds {
            let pending = std::mem::replace(&mut bucket, Uint::zero());
            acc += <Uint<5> as DelayedModularReduction<F>>::reduce(
                pending,
                field_cfg,
                &reduction_params,
            );
            pending_adds = 0;
        }
    }

    if !bucket.is_zero() {
        acc +=
            <Uint<5> as DelayedModularReduction<F>>::reduce(bucket, field_cfg, &reduction_params);
    }
    Ok(acc)
}

fn build_virtual_bit_array<F, G>(mut f: G) -> Result<[F; SHA_WORD_BITS], ShaProjectionError>
where
    G: FnMut(usize) -> Result<F, ShaProjectionError>,
{
    let mut values = Vec::with_capacity(SHA_WORD_BITS);
    for bit in 0..SHA_WORD_BITS {
        values.push(f(bit)?);
    }
    Ok(values
        .try_into()
        .unwrap_or_else(|_| unreachable!("exactly 32 virtual bits were built")))
}

fn scalarized_word_at_shifted_or_zero<F>(
    trace: &ProjectedTrace<F>,
    col: ShaWordCol,
    row: usize,
    shift: usize,
    field_cfg: &F::Config,
) -> Result<F, ShaProjectionError>
where
    F: PrimeField,
{
    let Some(shifted) = row.checked_add(shift) else {
        return Ok(F::zero_with_cfg(field_cfg));
    };
    if shifted >= SHA_ROW_COUNT {
        return Ok(F::zero_with_cfg(field_cfg));
    }
    let col_idx = col.index();
    scalar_from_table("scalarized", &trace.scalarized, col_idx, shifted, field_cfg)
}

fn booleanity_source_value_at_row<F>(
    trace: &ProjectedTrace<F>,
    row: usize,
    source: &ShaBooleanitySource,
    field_cfg: &F::Config,
) -> Result<F, ShaProjectionError>
where
    F: PrimeField,
{
    let virtuals = if matches!(
        source,
        ShaBooleanitySource::VirtualCh1 { .. }
            | ShaBooleanitySource::VirtualCh2 { .. }
            | ShaBooleanitySource::VirtualMaj { .. }
    ) {
        Some(reconstruct_virtual_ch_maj_at_row(trace, row, field_cfg)?)
    } else {
        None
    };
    booleanity_source_value_at_row_with_virtuals(trace, row, source, virtuals.as_ref(), field_cfg)
}

fn booleanity_source_value_at_row_with_virtuals<F>(
    trace: &ProjectedTrace<F>,
    row: usize,
    source: &ShaBooleanitySource,
    virtuals: Option<&VirtualChMajValues<F>>,
    field_cfg: &F::Config,
) -> Result<F, ShaProjectionError>
where
    F: PrimeField,
{
    match source {
        ShaBooleanitySource::WordBit { col, bit } => {
            bit_at_shifted_or_zero(trace, *col, row, 0, *bit, field_cfg)
        }
        ShaBooleanitySource::VirtualCh1 { bit } => {
            virtual_bit_at(&virtuals.expect("virtual source needs row cache").ch1, *bit)
        }
        ShaBooleanitySource::VirtualCh2 { bit } => {
            virtual_bit_at(&virtuals.expect("virtual source needs row cache").ch2, *bit)
        }
        ShaBooleanitySource::VirtualMaj { bit } => {
            virtual_bit_at(&virtuals.expect("virtual source needs row cache").maj, *bit)
        }
    }
}

fn virtual_bit_at<F: Clone>(
    bits: &[F; SHA_WORD_BITS],
    bit: usize,
) -> Result<F, ShaProjectionError> {
    bits.get(bit)
        .cloned()
        .ok_or(ShaProjectionError::BitIndexOutOfRange { bit })
}

fn fold_mle_tables<'a, F, I>(
    kind: &'static str,
    tables: I,
    theta: &[F],
    field_cfg: &F::Config,
) -> Result<MleTable<F>, ShaProjectionError>
where
    F: PrimeField + 'a,
    I: IntoIterator<Item = &'a MleTable<F>>,
{
    let tables = tables.into_iter().collect::<Vec<_>>();
    if tables.len() != theta.len() {
        return Err(ShaProjectionError::FoldingWeightCount {
            got: theta.len(),
            expected: tables.len(),
        });
    }
    let Some(first) = tables.first() else {
        return Ok(Vec::new());
    };
    let mut out = first
        .iter()
        .map(|column| DenseMultilinearExtension {
            evaluations: vec![F::zero_with_cfg(field_cfg); column.evaluations.len()],
            num_vars: column.num_vars,
        })
        .collect::<MleTable<F>>();
    for (table, weight) in tables.iter().zip(theta) {
        if table.len() != first.len() {
            return Err(ShaProjectionError::InstanceCountMismatch {
                got: table.len(),
                expected: first.len(),
            });
        }
        for (col_idx, col) in table.iter().enumerate() {
            if col.num_vars != out[col_idx].num_vars
                || col.evaluations.len() != out[col_idx].evaluations.len()
            {
                return Err(ShaProjectionError::ColumnRowCount {
                    kind,
                    col: col_idx,
                    got: col.evaluations.len(),
                    expected: out[col_idx].evaluations.len(),
                });
            }
            for (out, value) in out[col_idx].evaluations.iter_mut().zip(&col.evaluations) {
                *out += weight.clone() * value;
            }
        }
    }
    Ok(out)
}

fn fold_optional_mle_tables<'a, F, I>(
    kind: &'static str,
    tables: I,
    theta: &[F],
    field_cfg: &F::Config,
) -> Result<Option<MleTable<F>>, ShaProjectionError>
where
    F: PrimeField + 'a,
    I: IntoIterator<Item = Option<&'a MleTable<F>>>,
{
    let tables = tables.into_iter().collect::<Vec<_>>();
    if tables.iter().all(Option::is_none) {
        return Ok(None);
    }
    let mut present = Vec::with_capacity(tables.len());
    for table in tables {
        let Some(table) = table else {
            return Err(ShaProjectionError::PublicWordColumnPresenceMismatch);
        };
        present.push(table);
    }
    fold_mle_tables(kind, present, theta, field_cfg).map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sumcheck::multi_degree::MultiDegreeSumcheck;
    use crate::test_utils::test_config;
    use crypto_primitives::{FromWithConfig, crypto_bigint_monty::MontyField};
    use zinc_poly::EvaluatablePolynomial;
    use zinc_transcript::Blake3Transcript;

    type F = MontyField<4>;

    fn f(value: u64) -> F {
        F::from_with_cfg(value, &test_config())
    }

    fn zero_table(cols: usize) -> MleTable<F> {
        let cfg = test_config();
        mle_table_from_columns(
            vec![vec![F::zero_with_cfg(&cfg); SHA_ROW_COUNT]; cols],
            SHA_ROW_VARS,
        )
    }

    fn set_word_bit(
        trace: &mut ProjectedTrace<F>,
        col: ShaWordCol,
        row: usize,
        bit: usize,
        value: F,
    ) {
        let idx = bit_slice_index(col.index(), bit, SHA_WORD_BITS);
        trace.bit_slices[idx].evaluations[row] = value;
    }

    fn word_bit(trace: &ProjectedTrace<F>, col: ShaWordCol, row: usize, bit: usize) -> &F {
        &trace.bit_slices[bit_slice_index(col.index(), bit, SHA_WORD_BITS)].evaluations[row]
    }

    fn zero_trace() -> ProjectedTrace<F> {
        let cfg = test_config();
        let zero = F::zero_with_cfg(&cfg);
        let bits = vec![vec![vec![zero.clone(); SHA_WORD_BITS]; SHA_ROW_COUNT]; ShaWordCol::COUNT];
        let bit_slices =
            flatten_bit_columns(bits, SHA_WORD_BITS, SHA_ROW_VARS, "bit_slices").unwrap();
        let scalarized = scalarize_bit_slices(&bit_slices, &f(5), &cfg).unwrap();
        ProjectedTrace {
            bit_slices,
            scalarized,
            int_columns: zero_table(ShaIntCol::COUNT),
            public_columns: zero_table(ShaPublicCol::COUNT),
        }
    }

    fn zero_public() -> ProjectedPublic<F> {
        ProjectedPublic {
            columns: zero_table(ShaPublicCol::COUNT),
            bit_slices: None,
        }
    }

    fn synthetic_boolean_trace(instance_idx: u64, a: &F) -> ProjectedTrace<F> {
        let cfg = test_config();
        let zero = F::zero_with_cfg(&cfg);
        let mut bits =
            vec![vec![vec![zero.clone(); SHA_WORD_BITS]; SHA_ROW_COUNT]; ShaWordCol::COUNT];
        for (col_idx, col) in bits.iter_mut().enumerate() {
            for (row_idx, row) in col.iter_mut().enumerate() {
                for (bit_idx, bit) in row.iter_mut().enumerate() {
                    let selector = instance_idx
                        + u64::try_from(col_idx * 17 + row_idx * 3 + bit_idx)
                            .expect("test selector fits u64");
                    if selector % 2 == 1 {
                        *bit = f(1);
                    }
                }
            }
        }
        let bit_slices =
            flatten_bit_columns(bits, SHA_WORD_BITS, SHA_ROW_VARS, "bit_slices").unwrap();
        let scalarized = scalarize_bit_slices(&bit_slices, a, &cfg).unwrap();
        ProjectedTrace {
            bit_slices,
            scalarized,
            int_columns: zero_table(ShaIntCol::COUNT),
            public_columns: zero_table(ShaPublicCol::COUNT),
        }
    }

    fn prove_and_verify_sumfold(
        group: MultiDegreeSumcheckGroup<F>,
        ell: usize,
    ) -> (
        crate::sumcheck::multi_degree::MultiDegreeSumcheckProof<F>,
        Vec<F>,
        Vec<F>,
    ) {
        let cfg = test_config();
        let mut prover_transcript = Blake3Transcript::new();
        let (proof, _) = MultiDegreeSumcheck::prove_as_subprotocol(
            &mut prover_transcript,
            vec![group],
            ell,
            &cfg,
        );

        let mut verifier_transcript = Blake3Transcript::new();
        let subclaims =
            MultiDegreeSumcheck::verify_as_subprotocol(&mut verifier_transcript, ell, &proof, &cfg)
                .expect("sumcheck proof should verify");

        (
            proof,
            subclaims.point().to_vec(),
            subclaims.expected_evaluations().to_vec(),
        )
    }

    fn naive_project_bits(bits: &[F], powers: &[F]) -> F {
        bits.iter()
            .zip(powers.iter())
            .fold(F::zero_with_cfg(&test_config()), |acc, (bit, power)| {
                acc + bit.clone() * power
            })
    }

    #[test]
    fn dmr_bit_projection_matches_naive_for_binary_and_field_bits() {
        let cfg = test_config();
        let a = f(7);
        let powers = powers(a, F::one_with_cfg(&cfg), SHA_WORD_BITS);
        let zero = F::zero_with_cfg(&cfg);
        let mut binary_bits = vec![zero.clone(); SHA_WORD_BITS];
        binary_bits[0] = f(1);
        binary_bits[5] = f(1);
        binary_bits[31] = f(1);

        let binary_expected = naive_project_bits(&binary_bits, &powers);
        assert_eq!(
            project_binary_bits_conditional_add_dmr(&binary_bits, &powers, &cfg).unwrap(),
            binary_expected
        );
        assert_eq!(
            project_bits_dmr(&binary_bits, &powers, &cfg).unwrap(),
            binary_expected
        );

        let mut field_bits = vec![zero; SHA_WORD_BITS];
        field_bits[3] = f(2);
        field_bits[9] = f(11);
        let field_expected = naive_project_bits(&field_bits, &powers);
        assert_eq!(
            project_binary_bits_conditional_add_dmr(&field_bits, &powers, &cfg).unwrap(),
            field_expected
        );
        assert_eq!(
            project_bits_dmr(&field_bits, &powers, &cfg).unwrap(),
            field_expected
        );
    }

    #[test]
    fn dmr_residual_evaluation_matches_polynomial_evaluation() {
        let cfg = test_config();
        let a = f(5);
        let trace = synthetic_boolean_trace(3, &a);
        let public = zero_public();
        let row = 17usize;
        let a_powers = powers(
            a.clone(),
            F::one_with_cfg(&cfg),
            SHA_RESIDUAL_EVAL_POWER_COUNT,
        );
        let residuals =
            residual_values_at_row_with_powers(&trace, &public, row, &a_powers, &cfg).unwrap();
        let polies = residual_polys_at_row(&trace, &public, row, &cfg).unwrap();

        for (value, poly) in residuals.iter().zip(polies.iter()) {
            assert_eq!(value, &poly.evaluate_at_point(&a).unwrap());
        }
    }

    #[test]
    fn dmr_fresh_sha_targets_match_reference_evaluation() {
        let cfg = test_config();
        let a = f(13);
        let lambda = f(17);
        let mut cache = FreshIdealEvaluationCache {
            r_ic: std::array::from_fn(|_| F::zero_with_cfg(&cfg)),
            ideal_polys: vec![std::array::from_fn(|slot| {
                DynamicPolynomialF::new_trimmed([
                    f(u64::try_from(slot + 1).unwrap()),
                    f(u64::try_from(slot + 2).unwrap()),
                    f(u64::try_from(slot + 3).unwrap()),
                ])
            })],
            taus_at_a: Vec::new(),
            fresh_targets: Vec::new(),
        };

        evaluate_fresh_sha_targets(&mut cache, &a, &lambda, &cfg).unwrap();

        let lambda_powers = powers(lambda, F::one_with_cfg(&cfg), NUM_SHA_RESIDUAL_FAMILIES);
        let mut expected_target = F::zero_with_cfg(&cfg);
        for (slot, family) in NONZERO_SHA_FAMILIES.iter().enumerate() {
            let expected_tau = cache.ideal_polys[0][slot].evaluate_at_point(&a).unwrap();
            assert_eq!(cache.taus_at_a[0][slot], expected_tau);
            expected_target += lambda_powers[family.index()].clone() * expected_tau;
        }
        assert_eq!(cache.fresh_targets[0], expected_target);
    }

    #[test]
    fn zero_trace_ideal_cache_checks_and_targets_are_zero() {
        let cfg = test_config();
        let trace = zero_trace();
        let public = zero_public();
        let mut r_ic = std::array::from_fn(|_| F::zero_with_cfg(&cfg));
        r_ic[0] = f(3);
        r_ic[1] = f(7);

        let mut cache =
            build_fresh_sha_ideal_cache(&[trace], &[public], r_ic, &cfg).expect("cache builds");
        check_fresh_sha_ideal_cache(&cache, &cfg).expect("zero ideals pass");
        evaluate_fresh_sha_targets(&mut cache, &f(11), &f(13), &cfg).unwrap();

        assert_eq!(cache.ideal_polys.len(), 1);
        for poly in &cache.ideal_polys[0] {
            assert!(poly.is_zero());
        }
        for tau in &cache.taus_at_a[0] {
            assert_eq!(tau, &F::zero_with_cfg(&cfg));
        }
        assert_eq!(cache.fresh_targets[0], F::zero_with_cfg(&cfg));
    }

    #[test]
    fn beta_aggregate_with_weights_matches_wrapper() {
        let cfg = test_config();
        let table = |offset: u64| LinearResidualCoeffTable {
            coeffs: (0..NUM_SHA_RESIDUAL_FAMILIES)
                .map(|idx| {
                    DynamicPolynomialF::new_trimmed([
                        f(offset + idx as u64 + 1),
                        f(offset + idx as u64 + 101),
                    ])
                })
                .collect(),
        };
        let tables = vec![table(0), table(1_000)];
        let beta = [f(17)];
        let beta_eq_weights = zinc_poly::utils::build_eq_x_r_vec(&beta, &cfg).unwrap();

        let wrapped = beta_aggregate_nonzero_ideal_polys(&tables, &beta, &cfg).unwrap();
        let cached =
            beta_aggregate_nonzero_ideal_polys_with_weights(&tables, &beta_eq_weights).unwrap();

        assert_eq!(cached, wrapped);
    }

    #[test]
    fn tampered_ideal_cache_fails_membership() {
        let cfg = test_config();
        let trace = zero_trace();
        let public = zero_public();
        let r_ic = std::array::from_fn(|_| F::zero_with_cfg(&cfg));
        let mut values = build_sha_ideal_values_at_point(&trace, &public, &r_ic, &cfg).unwrap();
        values[0] = DynamicPolynomialF::new_trimmed([f(1)]);

        assert!(matches!(
            check_sha_ideal_values(&values, &cfg),
            Err(ShaProjectionError::IdealMembership)
        ));
    }

    #[test]
    fn fresh_sha_ideal_polys_are_verified_by_reusable_helper() {
        let cfg = test_config();
        let valid_zero = vec![std::array::from_fn(|_| {
            DynamicPolynomialF::new(Vec::<F>::new())
        })];
        verify_fresh_sha_ideal_polys(&valid_zero, &cfg).expect("zero ideal set passes");

        let mut tampered_x_minus_two = valid_zero.clone();
        tampered_x_minus_two[0][2] = DynamicPolynomialF::new_trimmed([f(1)]);
        assert!(matches!(
            verify_fresh_sha_ideal_polys(&tampered_x_minus_two, &cfg),
            Err(ShaProjectionError::IdealMembership)
        ));

        let mut trailing_zero = valid_zero.clone();
        trailing_zero[0][0] = DynamicPolynomialF::new(vec![f(1), F::zero_with_cfg(&cfg)]);
        assert!(matches!(
            verify_fresh_sha_ideal_polys(&trailing_zero, &cfg),
            Err(ShaProjectionError::NonCanonicalProofObject(_))
        ));

        let mut high_degree = valid_zero;
        high_degree[0][2] = DynamicPolynomialF::new(vec![f(1); 33]);
        assert!(matches!(
            verify_fresh_sha_ideal_polys(&high_degree, &cfg),
            Err(ShaProjectionError::NonCanonicalProofObject(_))
        ));
    }

    #[test]
    fn scalarization_links_check_folded_words() {
        let cfg = test_config();
        let mut trace = zero_trace();
        set_word_bit(&mut trace, ShaWordCol::A, 0, 0, f(1));
        set_word_bit(&mut trace, ShaWordCol::A, 0, 3, f(1));
        trace.scalarized = scalarize_bit_slices(&trace.bit_slices, &f(5), &cfg).unwrap();

        verify_folded_scalarization_links(&trace, &f(5), &[ShaWordCol::A], &cfg)
            .expect("scalarization should pass");

        trace.scalarized[ShaWordCol::A.index()].evaluations[0] += f(1);
        assert!(matches!(
            verify_folded_scalarization_links(&trace, &f(5), &[ShaWordCol::A], &cfg),
            Err(ShaProjectionError::ScalarizationMismatch { .. })
        ));
    }

    #[test]
    fn scalarization_links_check_endpoint_and_shifted_sources() {
        let cfg = test_config();
        let mut trace = zero_trace();
        set_word_bit(&mut trace, ShaWordCol::A, 0, 1, f(1));
        set_word_bit(&mut trace, ShaWordCol::A, 1, 0, f(1));
        set_word_bit(&mut trace, ShaWordCol::A, 1, 2, f(1));
        trace.scalarized = scalarize_bit_slices(&trace.bit_slices, &f(3), &cfg).unwrap();
        let r_star = std::array::from_fn(|_| F::zero_with_cfg(&cfg));

        verify_folded_scalarization_links_at_point(&trace, &f(3), &r_star, &[ShaWordCol::A], &cfg)
            .expect("endpoint scalarization should pass");
        verify_folded_shifted_scalarization_link_at_point(
            &trace,
            &f(3),
            &r_star,
            ShaWordCol::A,
            1,
            &cfg,
        )
        .expect("shifted endpoint scalarization should pass");

        trace.scalarized[ShaWordCol::A.index()].evaluations[0] += f(1);
        assert!(matches!(
            verify_folded_scalarization_links_at_point(
                &trace,
                &f(3),
                &r_star,
                &[ShaWordCol::A],
                &cfg
            ),
            Err(ShaProjectionError::ScalarizationMismatch { .. })
        ));
    }

    #[test]
    fn instance_fold_claim_derives_weights_after_endpoint() {
        let cfg = test_config();
        let beta = vec![f(2), f(3)];
        let r_b = vec![f(5), f(7)];
        let c_sf = f(11);
        let out = derive_instance_fold_claim(&beta, r_b.clone(), c_sf.clone(), 4, &cfg).unwrap();
        let d = eq_eval(&beta, &r_b, F::one_with_cfg(&cfg)).unwrap();

        assert_eq!(out.final_round_sumcheck_claim(), &(c_sf / d));
        assert_eq!(
            out.eq_instance_weights(),
            build_eq_x_r_vec(&r_b, &cfg).unwrap()
        );
    }

    #[test]
    fn folding_uses_eq_instance_weights() {
        let cfg = test_config();
        let beta = vec![f(2)];
        let r_b = vec![f(3)];
        let out = derive_instance_fold_claim(&beta, r_b, f(9), 2, &cfg).unwrap();

        let mut left = zero_trace();
        let mut right = zero_trace();
        set_word_bit(&mut left, ShaWordCol::A, 0, 0, f(1));
        set_word_bit(&mut right, ShaWordCol::A, 0, 0, f(2));
        left.scalarized = scalarize_bit_slices(&left.bit_slices, &f(5), &cfg).unwrap();
        right.scalarized = scalarize_bit_slices(&right.bit_slices, &f(5), &cfg).unwrap();

        let (folded, _public) = fold_projected_traces(
            &[left.clone(), right.clone()],
            &[zero_public(), zero_public()],
            &out,
            &cfg,
        )
        .unwrap();
        let expected = out.eq_instance_weights()[0].clone() * word_bit(&left, ShaWordCol::A, 0, 0)
            + out.eq_instance_weights()[1].clone() * word_bit(&right, ShaWordCol::A, 0, 0);
        assert_eq!(*word_bit(&folded.trace, ShaWordCol::A, 0, 0), expected);
    }

    #[test]
    fn virtual_ch_maj_reconstructs_from_source_bits() {
        let cfg = test_config();
        let mut trace = zero_trace();
        set_word_bit(&mut trace, ShaWordCol::E, 2, 0, f(1));
        set_word_bit(&mut trace, ShaWordCol::E, 1, 0, f(1));
        set_word_bit(&mut trace, ShaWordCol::Uef, 2, 0, f(1));
        set_word_bit(&mut trace, ShaWordCol::A, 0, 1, f(1));
        set_word_bit(&mut trace, ShaWordCol::A, 1, 1, f(1));
        set_word_bit(&mut trace, ShaWordCol::A, 2, 1, f(1));
        set_word_bit(&mut trace, ShaWordCol::Maj, 2, 1, f(1));

        let virtuals = reconstruct_virtual_ch_maj_at_row(&trace, 0, &cfg).unwrap();

        assert_eq!(virtuals.ch1[0], F::zero_with_cfg(&cfg));
        assert_eq!(virtuals.maj[1], f(1));
    }

    #[test]
    fn malformed_virtual_sources_return_errors() {
        let cfg = test_config();
        let trace = zero_trace();
        assert!(matches!(
            reconstruct_virtual_ch_maj_at_row(&trace, SHA_ROW_COUNT, &cfg),
            Err(ShaProjectionError::RowIndexOutOfRange { .. })
        ));

        let public = zero_public();
        let r_ic = std::array::from_fn(|_| F::zero_with_cfg(&cfg));
        assert!(matches!(
            folded_row_integrand_values(
                &trace,
                &public,
                &r_ic,
                &f(3),
                &f(5),
                &f(7),
                &f(11),
                &[ShaBooleanitySource::VirtualMaj { bit: SHA_WORD_BITS }],
                &cfg,
            ),
            Err(ShaProjectionError::BitIndexOutOfRange { .. })
        ));
    }

    #[test]
    fn production_sha_sumfold_prefix_tail_matches_dense_sumcheck() {
        let cfg = test_config();
        let ell = 3usize;
        let a = f(3);
        let traces = (0..(1usize << ell))
            .map(|idx| synthetic_boolean_trace(u64::try_from(idx).unwrap(), &a))
            .collect::<Vec<_>>();
        let publics = vec![zero_public(); traces.len()];
        let beta = vec![f(5), f(7), f(11)];
        let r_ic = [f(2), f(3), f(5), f(7), f(11), f(13), f(17)];
        let lambda = f(19);
        let rho = f(23);
        let xi = f(29);
        let sources = vec![
            ShaBooleanitySource::WordBit {
                col: ShaWordCol::A,
                bit: 0,
            },
            ShaBooleanitySource::WordBit {
                col: ShaWordCol::E,
                bit: 1,
            },
            ShaBooleanitySource::WordBit {
                col: ShaWordCol::W,
                bit: 2,
            },
        ];

        for prefix_vars in [1usize, 2, 3] {
            let dense = build_dense_sha_sumfold_group(
                &traces, &publics, &beta, &r_ic, &a, &lambda, &rho, &xi, &sources, &cfg,
            )
            .unwrap();
            let optimized = build_production_sha_sumfold_group(
                &traces,
                &publics,
                &beta,
                &r_ic,
                &a,
                &lambda,
                &rho,
                &xi,
                &sources,
                prefix_vars,
                &cfg,
            )
            .unwrap();

            let (dense_proof, dense_point, dense_expected) = prove_and_verify_sumfold(dense, ell);
            let (optimized_proof, optimized_point, optimized_expected) =
                prove_and_verify_sumfold(optimized, ell);

            assert_eq!(optimized_proof, dense_proof);
            assert_eq!(optimized_point, dense_point);
            assert_eq!(optimized_expected, dense_expected);
        }
    }

    #[test]
    fn production_sha_sumfold_feeds_folded_row_sumcheck() {
        let cfg = test_config();
        let ell = 2usize;
        let a = f(3);
        let traces = (0..(1usize << ell))
            .map(|idx| synthetic_boolean_trace(u64::try_from(idx).unwrap(), &a))
            .collect::<Vec<_>>();
        let publics = vec![zero_public(); traces.len()];
        let beta = vec![f(5), f(7)];
        let r_ic = [f(2), f(3), f(5), f(7), f(11), f(13), f(17)];
        let lambda = f(19);
        let rho = f(23);
        let xi = f(29);
        let sources = vec![
            ShaBooleanitySource::WordBit {
                col: ShaWordCol::A,
                bit: 0,
            },
            ShaBooleanitySource::WordBit {
                col: ShaWordCol::E,
                bit: 1,
            },
        ];

        let sumfold_group = build_production_sha_sumfold_group(
            &traces, &publics, &beta, &r_ic, &a, &lambda, &rho, &xi, &sources, 1, &cfg,
        )
        .unwrap();
        let (_proof, r_b, expected) = prove_and_verify_sumfold(sumfold_group, ell);
        let sumfold =
            derive_instance_fold_claim(&beta, r_b, expected[0].clone(), traces.len(), &cfg)
                .unwrap();
        let (folded_witness, folded_public) =
            fold_projected_traces(&traces, &publics, &sumfold, &cfg).unwrap();

        let folded_claim = expression_folded_row_sum(
            &folded_witness.trace,
            &folded_public,
            &r_ic,
            &a,
            &lambda,
            &rho,
            &xi,
            &sources,
            &cfg,
        )
        .unwrap();
        assert_eq!(&folded_claim, sumfold.final_round_sumcheck_claim());

        let row_group = build_expression_folded_row_sumcheck_group(
            &folded_witness.trace,
            &folded_public,
            &r_ic,
            &a,
            &lambda,
            &rho,
            &xi,
            &sources,
            &cfg,
        )
        .unwrap();
        let mut row_prover_transcript = Blake3Transcript::new();
        let (row_proof, _) = MultiDegreeSumcheck::prove_as_subprotocol(
            &mut row_prover_transcript,
            vec![row_group],
            SHA_ROW_VARS,
            &cfg,
        );
        let mut row_verifier_transcript = Blake3Transcript::new();
        MultiDegreeSumcheck::verify_as_subprotocol(
            &mut row_verifier_transcript,
            SHA_ROW_VARS,
            &row_proof,
            &cfg,
        )
        .expect("folded row sumcheck proof should verify");
        verify_folded_row_sumcheck_claim(
            &row_proof.claimed_sums()[0],
            sumfold.final_round_sumcheck_claim(),
        )
        .expect("folded row claim matches T'");
    }

    #[test]
    fn folded_row_group_claims_row_integrand_sum() {
        let cfg = test_config();
        let trace = zero_trace();
        let public = zero_public();
        let r_ic = std::array::from_fn(|_| F::zero_with_cfg(&cfg));
        let values = folded_row_integrand_values(
            &trace,
            &public,
            &r_ic,
            &f(3),
            &f(5),
            &f(7),
            &f(11),
            &[],
            &cfg,
        )
        .unwrap();
        let group = build_folded_row_sumcheck_group(&values, &cfg).unwrap();
        let mut transcript = Blake3Transcript::new();
        let (proof, _) =
            MultiDegreeSumcheck::prove_as_subprotocol(&mut transcript, vec![group], 7, &cfg);

        assert_eq!(proof.claimed_sums()[0], F::zero_with_cfg(&cfg));
        verify_folded_row_sumcheck_claim(&proof.claimed_sums()[0], &F::zero_with_cfg(&cfg))
            .expect("row claim matches T'");
        assert!(matches!(
            verify_folded_row_sumcheck_claim(&proof.claimed_sums()[0], &f(1)),
            Err(ShaProjectionError::FoldedRowClaimMismatch)
        ));
        assert_eq!(
            folded_row_integrand_sum(&values, &cfg).unwrap(),
            F::zero_with_cfg(&cfg)
        );
    }
}
