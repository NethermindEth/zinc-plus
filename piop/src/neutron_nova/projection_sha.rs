//! Production SHA-256 ProjectionFold helpers.
//!
//! This module implements the SHA-specific data model and reference
//! computations used by the production ProjectionFold flow:
//!
//! fresh ideal checks -> SumFold over instances -> post-SumFold folding ->
//! folded row check over the 128-row SHA domain.

use std::{borrow::Borrow, collections::HashMap};

use crate::ideal_check::batched_ideal_check;
use crate::neutron_nova::SumFoldError;
use crate::{
    CombFn,
    sumcheck::multi_degree::{MultiDegreeSumcheckGroup, PrefixFastPath, PrefixRoundOutput},
};
use ark_ff::{MontBackend, MontConfig};
use crypto_primitives::{
    PrimeField, ark_ff_fp::Fp as ArkFp, crypto_bigint_boxed_monty::BoxedMontyField,
    crypto_bigint_monty::MontyField, crypto_bigint_uint::Uint,
};
use num_traits::{ConstZero, Zero};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
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
    UNCHECKED, cfg_chunks, cfg_into_iter, cfg_iter,
    delayed_reduction::{
        BarrettDelayedReduction, DelayedFieldProductSum, DelayedFieldProductSumAlgorithm,
        DelayedModularReductionAlgorithm, MontgomeryLimbs, MontgomeryProductSum4,
        ProductAccumulator4,
    },
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
const SHA_DIRECT_ROW_CHUNK: usize = 8;
const SHA_SUFFIX_DMR_PAIR_CHUNK: usize = 256;
const SHA_SUFFIX_DMR_WEIGHT_CHUNK: usize = 512;

pub type MleColumn<T> = DenseMultilinearExtension<T>;
pub type MleTable<T> = Vec<MleColumn<T>>;

pub trait ShaBinaryFoldField: PrimeField + Send + Sync + Sized {
    fn fold_binary_mle_tables(
        kind: &'static str,
        tables: &[&MleTable<Self>],
        theta: &[Self],
        field_cfg: &Self::Config,
    ) -> Result<MleTable<Self>, ShaProjectionError>;
}

impl ShaBinaryFoldField for MontyField<4> {
    fn fold_binary_mle_tables(
        kind: &'static str,
        tables: &[&MleTable<Self>],
        theta: &[Self],
        field_cfg: &Self::Config,
    ) -> Result<MleTable<Self>, ShaProjectionError> {
        fold_binary_mle_tables_montgomery(kind, tables, theta, field_cfg)
    }
}

impl ShaBinaryFoldField for BoxedMontyField {
    fn fold_binary_mle_tables(
        kind: &'static str,
        tables: &[&MleTable<Self>],
        theta: &[Self],
        field_cfg: &Self::Config,
    ) -> Result<MleTable<Self>, ShaProjectionError> {
        fold_binary_mle_tables_generic(kind, tables, theta, field_cfg)
    }
}

impl<M, const N: usize> ShaBinaryFoldField for ArkFp<MontBackend<M, N>, N>
where
    M: MontConfig<N>,
{
    fn fold_binary_mle_tables(
        kind: &'static str,
        tables: &[&MleTable<Self>],
        theta: &[Self],
        field_cfg: &Self::Config,
    ) -> Result<MleTable<Self>, ShaProjectionError> {
        fold_binary_mle_tables_generic(kind, tables, theta, field_cfg)
    }
}

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
            add_scaled_poly_assign(&mut aggregate[slot], residual, weight);
        }
    }
    aggregate.iter_mut().for_each(DynamicPolynomialF::trim);
    Ok(aggregate)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShaAggregateIdealWeightPlan<F: PrimeField> {
    /// Indexed as `[instance][row]`.
    pub beta_row_weights: Vec<Vec<F>>,
}

impl<F> ShaAggregateIdealWeightPlan<F>
where
    F: PrimeField,
{
    pub fn new(beta_eq_weights: &[F], row_weights: &[F]) -> Result<Self, ShaProjectionError> {
        if row_weights.len() != SHA_ROW_COUNT {
            return Err(ShaProjectionError::ColumnRowCount {
                kind: "row_weights",
                col: 0,
                got: row_weights.len(),
                expected: SHA_ROW_COUNT,
            });
        }
        Ok(Self {
            beta_row_weights: beta_eq_weights
                .iter()
                .map(|beta_weight| {
                    row_weights
                        .iter()
                        .map(|row_weight| beta_weight.clone() * row_weight)
                        .collect()
                })
                .collect(),
        })
    }

    pub fn instance_count(&self) -> usize {
        self.beta_row_weights.len()
    }
}

pub fn beta_aggregate_nonzero_ideal_polys_direct_with_weights<F, Trace, Public>(
    traces: &[Trace],
    publics: &[Public],
    plan: &ShaAggregateIdealWeightPlan<F>,
    field_cfg: &F::Config,
) -> Result<[DynamicPolynomialF<F>; NUM_NONZERO_SHA_FAMILIES], ShaProjectionError>
where
    F: PrimeField + Send + Sync,
    Trace: Borrow<ProjectedTrace<F>> + Sync,
    Public: Borrow<ProjectedPublic<F>> + Sync,
{
    if traces.len() != publics.len() {
        return Err(ShaProjectionError::InstanceCountMismatch {
            got: publics.len(),
            expected: traces.len(),
        });
    }
    if plan.instance_count() != traces.len() {
        return Err(ShaProjectionError::InstanceCountMismatch {
            got: plan.instance_count(),
            expected: traces.len(),
        });
    }

    #[cfg(debug_assertions)]
    {
        for ((trace, public), beta_row_weights) in traces
            .iter()
            .zip(publics.iter())
            .zip(plan.beta_row_weights.iter())
        {
            validate_trace(trace.borrow())?;
            validate_public(public.borrow())?;
            if beta_row_weights.len() != SHA_ROW_COUNT {
                return Err(ShaProjectionError::ColumnRowCount {
                    kind: "beta_row_weights",
                    col: 0,
                    got: beta_row_weights.len(),
                    expected: SHA_ROW_COUNT,
                });
            }
        }
    }
    #[cfg(not(debug_assertions))]
    {
        for beta_row_weights in plan.beta_row_weights.iter() {
            if beta_row_weights.len() != SHA_ROW_COUNT {
                return Err(ShaProjectionError::ColumnRowCount {
                    kind: "beta_row_weights",
                    col: 0,
                    got: beta_row_weights.len(),
                    expected: SHA_ROW_COUNT,
                });
            }
        }
    }

    let constants = ShaResidualPolyConstants::new(field_cfg);
    let tasks = sha_direct_row_tasks(traces.len(), SHA_ROW_COUNT);
    let partials = cfg_iter!(&tasks)
        .map(|&(instance_idx, row_start, row_end)| {
            let trace = traces[instance_idx].borrow();
            let public = publics[instance_idx].borrow();
            let beta_row_weights = &plan.beta_row_weights[instance_idx];
            let mut acc = NonzeroResidualCoeffAccumulator::new(field_cfg);
            for row in row_start..row_end {
                accumulate_nonzero_ideal_row_fixed(
                    &mut acc,
                    trace,
                    public,
                    row,
                    &beta_row_weights[row],
                    &constants,
                    field_cfg,
                )?;
            }
            Ok(acc)
        })
        .collect::<Vec<Result<_, ShaProjectionError>>>();

    let mut aggregate = NonzeroResidualCoeffAccumulator::new(field_cfg);
    for partial in partials {
        aggregate.add_assign(partial?);
    }
    Ok(aggregate.into_polys())
}

fn sha_direct_row_tasks(instance_count: usize, row_count: usize) -> Vec<(usize, usize, usize)> {
    let chunk = SHA_DIRECT_ROW_CHUNK.min(row_count).max(1);
    let row_chunks = (row_count + chunk - 1) / chunk;
    let mut tasks = Vec::with_capacity(instance_count * row_chunks);
    for instance_idx in 0..instance_count {
        for row_start in (0..row_count).step_by(chunk) {
            tasks.push((instance_idx, row_start, (row_start + chunk).min(row_count)));
        }
    }
    tasks
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

fn selected_nonzero_sha_lambda_powers<F>(
    lambda_powers: &[F],
) -> Result<[F; NUM_NONZERO_SHA_FAMILIES], ShaProjectionError>
where
    F: PrimeField,
{
    if lambda_powers.len() != NUM_SHA_RESIDUAL_FAMILIES {
        return Err(ShaProjectionError::MissingColumn {
            kind: "lambda_powers",
            col: lambda_powers.len(),
        });
    }
    Ok(std::array::from_fn(|slot| {
        lambda_powers[NONZERO_SHA_FAMILIES[slot].index()].clone()
    }))
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
    cfg_iter!(tables)
        .map(|table| {
            if table.coeffs.len() != NUM_SHA_RESIDUAL_FAMILIES {
                return Err(ShaProjectionError::MissingColumn {
                    kind: "linear_residual_coeffs",
                    col: table.coeffs.len(),
                });
            }
            let mut values: [F; NUM_SHA_RESIDUAL_FAMILIES] =
                std::array::from_fn(|_| F::zero_with_cfg(field_cfg));
            for (family_idx, residual) in table.coeffs.iter().enumerate() {
                values[family_idx] = evaluate_poly_at_powers_dmr(residual, &a_powers, field_cfg)?;
            }
            FieldFieldInnerProduct::inner_product::<UNCHECKED>(
                &values,
                lambda_powers,
                F::zero_with_cfg(field_cfg),
            )
            .map_err(ShaProjectionError::from)
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShaLinearResidualWeightPlan<F: PrimeField> {
    pub row_weights: Vec<F>,
    pub a_powers: Vec<F>,
    pub lambda_powers: Vec<F>,
}

impl<F> ShaLinearResidualWeightPlan<F>
where
    F: PrimeField,
{
    pub fn new(
        row_weights: &[F],
        a_powers: &[F],
        lambda_powers: &[F],
    ) -> Result<Self, ShaProjectionError> {
        if row_weights.len() != SHA_ROW_COUNT {
            return Err(ShaProjectionError::ColumnRowCount {
                kind: "row_weights",
                col: 0,
                got: row_weights.len(),
                expected: SHA_ROW_COUNT,
            });
        }
        if a_powers.len() < SHA_RESIDUAL_EVAL_POWER_COUNT {
            return Err(ShaProjectionError::MissingColumn {
                kind: "a_powers",
                col: a_powers.len(),
            });
        }
        if lambda_powers.len() != NUM_SHA_RESIDUAL_FAMILIES {
            return Err(ShaProjectionError::MissingColumn {
                kind: "lambda_powers",
                col: lambda_powers.len(),
            });
        }
        Ok(Self {
            row_weights: row_weights.to_vec(),
            a_powers: a_powers.to_vec(),
            lambda_powers: lambda_powers.to_vec(),
        })
    }
}

pub fn build_sha_sumfold_linear_accumulator_direct_with_weights<F, Trace, Public>(
    traces: &[Trace],
    publics: &[Public],
    plan: &ShaLinearResidualWeightPlan<F>,
    field_cfg: &F::Config,
) -> Result<Vec<F>, ShaProjectionError>
where
    F: ShaLinearAccumulatorField,
    Trace: Borrow<ProjectedTrace<F>> + Sync,
    Public: Borrow<ProjectedPublic<F>> + Sync,
{
    F::build_sha_sumfold_linear_accumulator_direct_with_weights(traces, publics, plan, field_cfg)
}

pub trait ShaLinearAccumulatorField: DelayedFieldProductSum + Send + Sync + Sized {
    fn build_sha_sumfold_linear_accumulator_direct_with_weights<Trace, Public>(
        traces: &[Trace],
        publics: &[Public],
        plan: &ShaLinearResidualWeightPlan<Self>,
        field_cfg: &Self::Config,
    ) -> Result<Vec<Self>, ShaProjectionError>
    where
        Trace: Borrow<ProjectedTrace<Self>> + Sync,
        Public: Borrow<ProjectedPublic<Self>> + Sync;
}

impl ShaLinearAccumulatorField for MontyField<4> {
    fn build_sha_sumfold_linear_accumulator_direct_with_weights<Trace, Public>(
        traces: &[Trace],
        publics: &[Public],
        plan: &ShaLinearResidualWeightPlan<Self>,
        field_cfg: &Self::Config,
    ) -> Result<Vec<Self>, ShaProjectionError>
    where
        Trace: Borrow<ProjectedTrace<Self>> + Sync,
        Public: Borrow<ProjectedPublic<Self>> + Sync,
    {
        build_sha_sumfold_linear_accumulator_direct_with_weights_dmr(
            traces, publics, plan, field_cfg,
        )
    }
}

impl ShaLinearAccumulatorField for BoxedMontyField {
    fn build_sha_sumfold_linear_accumulator_direct_with_weights<Trace, Public>(
        traces: &[Trace],
        publics: &[Public],
        plan: &ShaLinearResidualWeightPlan<Self>,
        field_cfg: &Self::Config,
    ) -> Result<Vec<Self>, ShaProjectionError>
    where
        Trace: Borrow<ProjectedTrace<Self>> + Sync,
        Public: Borrow<ProjectedPublic<Self>> + Sync,
    {
        build_sha_sumfold_linear_accumulator_direct_with_weights_generic(
            traces, publics, plan, field_cfg,
        )
    }
}

impl<M, const N: usize> ShaLinearAccumulatorField for ArkFp<MontBackend<M, N>, N>
where
    M: MontConfig<N>,
{
    fn build_sha_sumfold_linear_accumulator_direct_with_weights<Trace, Public>(
        traces: &[Trace],
        publics: &[Public],
        plan: &ShaLinearResidualWeightPlan<Self>,
        field_cfg: &Self::Config,
    ) -> Result<Vec<Self>, ShaProjectionError>
    where
        Trace: Borrow<ProjectedTrace<Self>> + Sync,
        Public: Borrow<ProjectedPublic<Self>> + Sync,
    {
        build_sha_sumfold_linear_accumulator_direct_with_weights_generic(
            traces, publics, plan, field_cfg,
        )
    }
}

pub trait ShaSuffixScannerField: PrimeField + Send + Sync + Sized {
    fn suffix_reduced_body_buckets(
        linear_claims: &[Self],
        booleanity_claims: &[Vec<Self>],
        source_row_weights: &[Self],
        suffix_eq_weights: &[Self],
        field_cfg: &Self::Config,
    ) -> (Self, Self) {
        suffix_reduced_body_buckets_generic(
            linear_claims,
            booleanity_claims,
            source_row_weights,
            suffix_eq_weights,
            field_cfg,
        )
    }

    fn suffix_direct_one_body_bucket(
        linear_claims: &[Self],
        booleanity_claims: &[Vec<Self>],
        source_row_weights: &[Self],
        suffix_eq_weights: &[Self],
        field_cfg: &Self::Config,
    ) -> Self {
        suffix_direct_one_body_bucket_generic(
            linear_claims,
            booleanity_claims,
            source_row_weights,
            suffix_eq_weights,
            field_cfg,
        )
    }

    fn suffix_reduced_body_buckets_flat(
        linear_claims: &[Self],
        booleanity_values: &[Self],
        source_row_count: usize,
        source_row_weights: &[Self],
        suffix_eq_weights: &[Self],
        field_cfg: &Self::Config,
    ) -> (Self, Self) {
        suffix_reduced_body_buckets_flat_generic(
            linear_claims,
            booleanity_values,
            source_row_count,
            source_row_weights,
            suffix_eq_weights,
            field_cfg,
        )
    }

    fn suffix_direct_one_body_bucket_flat(
        linear_claims: &[Self],
        booleanity_values: &[Self],
        source_row_count: usize,
        source_row_weights: &[Self],
        suffix_eq_weights: &[Self],
        field_cfg: &Self::Config,
    ) -> Self {
        suffix_direct_one_body_bucket_flat_generic(
            linear_claims,
            booleanity_values,
            source_row_count,
            source_row_weights,
            suffix_eq_weights,
            field_cfg,
        )
    }

    fn suffix_fold_prepare_next_round_flat(
        linear_claims: &mut [Self],
        booleanity_values: &mut [Self],
        source_row_count: usize,
        source_row_weights: &[Self],
        suffix_eq_weights: &[Self],
        r: &Self,
        need_t_one: bool,
        field_cfg: &Self::Config,
    ) -> (Self, Self, Option<Self>) {
        suffix_fold_prepare_next_round_flat_generic(
            linear_claims,
            booleanity_values,
            source_row_count,
            source_row_weights,
            suffix_eq_weights,
            r,
            need_t_one,
            field_cfg,
        )
    }
}

impl ShaSuffixScannerField for MontyField<4> {
    fn suffix_reduced_body_buckets(
        linear_claims: &[Self],
        booleanity_claims: &[Vec<Self>],
        source_row_weights: &[Self],
        suffix_eq_weights: &[Self],
        field_cfg: &Self::Config,
    ) -> (Self, Self) {
        let product_sum = MontgomeryProductSum4::<Self>::new(field_cfg);
        suffix_reduced_body_buckets_dmr_with_algorithm(
            &product_sum,
            linear_claims,
            booleanity_claims,
            source_row_weights,
            suffix_eq_weights,
            field_cfg,
        )
    }

    fn suffix_direct_one_body_bucket(
        linear_claims: &[Self],
        booleanity_claims: &[Vec<Self>],
        source_row_weights: &[Self],
        suffix_eq_weights: &[Self],
        field_cfg: &Self::Config,
    ) -> Self {
        let product_sum = MontgomeryProductSum4::<Self>::new(field_cfg);
        suffix_direct_one_body_bucket_dmr_with_algorithm(
            &product_sum,
            linear_claims,
            booleanity_claims,
            source_row_weights,
            suffix_eq_weights,
            field_cfg,
        )
    }

    fn suffix_reduced_body_buckets_flat(
        linear_claims: &[Self],
        booleanity_values: &[Self],
        source_row_count: usize,
        source_row_weights: &[Self],
        suffix_eq_weights: &[Self],
        field_cfg: &Self::Config,
    ) -> (Self, Self) {
        let product_sum = MontgomeryProductSum4::<Self>::new(field_cfg);
        suffix_reduced_body_buckets_flat_dmr_with_algorithm(
            &product_sum,
            linear_claims,
            booleanity_values,
            source_row_count,
            source_row_weights,
            suffix_eq_weights,
            field_cfg,
        )
    }

    fn suffix_direct_one_body_bucket_flat(
        linear_claims: &[Self],
        booleanity_values: &[Self],
        source_row_count: usize,
        source_row_weights: &[Self],
        suffix_eq_weights: &[Self],
        field_cfg: &Self::Config,
    ) -> Self {
        let product_sum = MontgomeryProductSum4::<Self>::new(field_cfg);
        suffix_direct_one_body_bucket_flat_dmr_with_algorithm(
            &product_sum,
            linear_claims,
            booleanity_values,
            source_row_count,
            source_row_weights,
            suffix_eq_weights,
            field_cfg,
        )
    }

    fn suffix_fold_prepare_next_round_flat(
        linear_claims: &mut [Self],
        booleanity_values: &mut [Self],
        source_row_count: usize,
        source_row_weights: &[Self],
        suffix_eq_weights: &[Self],
        r: &Self,
        need_t_one: bool,
        field_cfg: &Self::Config,
    ) -> (Self, Self, Option<Self>) {
        let product_sum = MontgomeryProductSum4::<Self>::new(field_cfg);
        suffix_fold_prepare_next_round_flat_dmr_with_algorithm(
            &product_sum,
            linear_claims,
            booleanity_values,
            source_row_count,
            source_row_weights,
            suffix_eq_weights,
            r,
            need_t_one,
            field_cfg,
        )
    }
}

impl ShaSuffixScannerField for BoxedMontyField {}

impl<M, const N: usize> ShaSuffixScannerField for ArkFp<MontBackend<M, N>, N> where M: MontConfig<N> {}

#[allow(clippy::arithmetic_side_effects)]
fn suffix_pair_weight<F>(suffix_eq_weights: &[F], rest: usize) -> F
where
    F: PrimeField,
{
    debug_assert!(suffix_eq_weights.len() >= 2);
    debug_assert!(rest < (suffix_eq_weights.len() >> 1));
    suffix_eq_weights[rest << 1].clone() + suffix_eq_weights[(rest << 1) + 1].clone()
}

#[allow(clippy::arithmetic_side_effects)]
fn suffix_reduced_body_buckets_generic<F>(
    linear_claims: &[F],
    booleanity_claims: &[Vec<F>],
    source_row_weights: &[F],
    suffix_eq_weights: &[F],
    field_cfg: &F::Config,
) -> (F, F)
where
    F: PrimeField,
{
    debug_assert_eq!(linear_claims.len(), suffix_eq_weights.len());
    debug_assert_eq!(booleanity_claims.len(), source_row_weights.len());
    for values in booleanity_claims {
        debug_assert_eq!(values.len(), suffix_eq_weights.len());
    }

    let rest_len = suffix_eq_weights.len() >> 1;
    let one = F::one_with_cfg(field_cfg);
    let mut linear_zero = F::zero_with_cfg(field_cfg);
    let mut quadratic_zero = F::zero_with_cfg(field_cfg);
    let mut quadratic_infinity = F::zero_with_cfg(field_cfg);

    for rest in 0..rest_len {
        let weight = suffix_pair_weight(suffix_eq_weights, rest);
        linear_zero += weight.clone() * linear_claims[rest << 1].clone();

        let mut source_zero = F::zero_with_cfg(field_cfg);
        let mut source_infinity = F::zero_with_cfg(field_cfg);
        for (values, scale) in booleanity_claims.iter().zip(source_row_weights) {
            if F::is_zero(scale) {
                continue;
            }
            let even = values[rest << 1].clone();
            let odd = values[(rest << 1) + 1].clone();
            source_zero += scale.clone() * even.clone() * (even.clone() - one.clone());
            let delta = odd - even;
            source_infinity += scale.clone() * delta.clone() * delta;
        }

        quadratic_zero += weight.clone() * source_zero;
        quadratic_infinity += weight * source_infinity;
    }

    (linear_zero + quadratic_zero, quadratic_infinity)
}

#[allow(clippy::arithmetic_side_effects)]
fn suffix_direct_one_body_bucket_generic<F>(
    linear_claims: &[F],
    booleanity_claims: &[Vec<F>],
    source_row_weights: &[F],
    suffix_eq_weights: &[F],
    field_cfg: &F::Config,
) -> F
where
    F: PrimeField,
{
    debug_assert_eq!(linear_claims.len(), suffix_eq_weights.len());
    debug_assert_eq!(booleanity_claims.len(), source_row_weights.len());
    for values in booleanity_claims {
        debug_assert_eq!(values.len(), suffix_eq_weights.len());
    }

    let rest_len = suffix_eq_weights.len() >> 1;
    let one = F::one_with_cfg(field_cfg);
    let mut acc = F::zero_with_cfg(field_cfg);

    for rest in 0..rest_len {
        let weight = suffix_pair_weight(suffix_eq_weights, rest);
        acc += weight.clone() * linear_claims[(rest << 1) + 1].clone();

        let mut source_one = F::zero_with_cfg(field_cfg);
        for (values, scale) in booleanity_claims.iter().zip(source_row_weights) {
            if F::is_zero(scale) {
                continue;
            }
            let odd = values[(rest << 1) + 1].clone();
            source_one += scale.clone() * odd.clone() * (odd - one.clone());
        }
        acc += weight * source_one;
    }

    acc
}

#[inline(always)]
#[allow(clippy::arithmetic_side_effects)]
fn suffix_flat_index(tail: usize, source_row: usize, source_row_count: usize) -> usize {
    tail * source_row_count + source_row
}

#[allow(clippy::arithmetic_side_effects)]
fn suffix_reduced_body_buckets_flat_generic<F>(
    linear_claims: &[F],
    booleanity_values: &[F],
    source_row_count: usize,
    source_row_weights: &[F],
    suffix_eq_weights: &[F],
    field_cfg: &F::Config,
) -> (F, F)
where
    F: PrimeField,
{
    debug_assert_eq!(linear_claims.len(), suffix_eq_weights.len());
    debug_assert_eq!(source_row_weights.len(), source_row_count);
    debug_assert_eq!(
        booleanity_values.len(),
        suffix_eq_weights.len() * source_row_count
    );

    let rest_len = suffix_eq_weights.len() >> 1;
    let one = F::one_with_cfg(field_cfg);
    let mut linear_zero = F::zero_with_cfg(field_cfg);
    let mut quadratic_zero = F::zero_with_cfg(field_cfg);
    let mut quadratic_infinity = F::zero_with_cfg(field_cfg);

    for rest in 0..rest_len {
        let even_tail = rest << 1;
        let odd_tail = even_tail + 1;
        let weight = suffix_pair_weight(suffix_eq_weights, rest);
        linear_zero += weight.clone() * linear_claims[even_tail].clone();

        let mut source_zero = F::zero_with_cfg(field_cfg);
        let mut source_infinity = F::zero_with_cfg(field_cfg);
        for (source_row, scale) in source_row_weights.iter().enumerate() {
            if F::is_zero(scale) {
                continue;
            }
            let even = booleanity_values
                [suffix_flat_index(even_tail, source_row, source_row_count)]
            .clone();
            let odd = booleanity_values[suffix_flat_index(odd_tail, source_row, source_row_count)]
                .clone();
            source_zero += scale.clone() * even.clone() * (even.clone() - one.clone());
            let delta = odd - even;
            source_infinity += scale.clone() * delta.clone() * delta;
        }

        quadratic_zero += weight.clone() * source_zero;
        quadratic_infinity += weight * source_infinity;
    }

    (linear_zero + quadratic_zero, quadratic_infinity)
}

#[allow(clippy::arithmetic_side_effects)]
fn suffix_direct_one_body_bucket_flat_generic<F>(
    linear_claims: &[F],
    booleanity_values: &[F],
    source_row_count: usize,
    source_row_weights: &[F],
    suffix_eq_weights: &[F],
    field_cfg: &F::Config,
) -> F
where
    F: PrimeField,
{
    debug_assert_eq!(linear_claims.len(), suffix_eq_weights.len());
    debug_assert_eq!(source_row_weights.len(), source_row_count);
    debug_assert_eq!(
        booleanity_values.len(),
        suffix_eq_weights.len() * source_row_count
    );

    let rest_len = suffix_eq_weights.len() >> 1;
    let one = F::one_with_cfg(field_cfg);
    let mut acc = F::zero_with_cfg(field_cfg);

    for rest in 0..rest_len {
        let odd_tail = (rest << 1) + 1;
        let weight = suffix_pair_weight(suffix_eq_weights, rest);
        acc += weight.clone() * linear_claims[odd_tail].clone();

        let mut source_one = F::zero_with_cfg(field_cfg);
        for (source_row, scale) in source_row_weights.iter().enumerate() {
            if F::is_zero(scale) {
                continue;
            }
            let odd = booleanity_values[suffix_flat_index(odd_tail, source_row, source_row_count)]
                .clone();
            source_one += scale.clone() * odd.clone() * (odd - one.clone());
        }
        acc += weight * source_one;
    }

    acc
}

#[allow(clippy::arithmetic_side_effects)]
fn suffix_fold_prepare_next_round_flat_generic<F>(
    linear_claims: &mut [F],
    booleanity_values: &mut [F],
    source_row_count: usize,
    source_row_weights: &[F],
    suffix_eq_weights: &[F],
    r: &F,
    need_t_one: bool,
    field_cfg: &F::Config,
) -> (F, F, Option<F>)
where
    F: PrimeField,
{
    debug_assert_eq!(linear_claims.len(), suffix_eq_weights.len());
    debug_assert_eq!(source_row_weights.len(), source_row_count);
    debug_assert_eq!(
        booleanity_values.len(),
        suffix_eq_weights.len() * source_row_count
    );
    debug_assert_eq!(suffix_eq_weights.len() & 3, 0);

    let next_pair_count = suffix_eq_weights.len() >> 2;
    let one = F::one_with_cfg(field_cfg);
    let mut linear_zero = F::zero_with_cfg(field_cfg);
    let mut linear_one = F::zero_with_cfg(field_cfg);
    let mut quadratic_zero = F::zero_with_cfg(field_cfg);
    let mut quadratic_infinity = F::zero_with_cfg(field_cfg);
    let mut quadratic_one = F::zero_with_cfg(field_cfg);

    for next_pair in 0..next_pair_count {
        let old_base = next_pair << 2;
        let folded_even = next_pair << 1;
        let folded_odd = folded_even + 1;
        let pair_weight = suffix_eq_weights[old_base].clone()
            + &suffix_eq_weights[old_base + 1]
            + &suffix_eq_weights[old_base + 2]
            + &suffix_eq_weights[old_base + 3];

        let l00 = linear_claims[old_base].clone();
        let l01 = linear_claims[old_base + 1].clone();
        let l10 = linear_claims[old_base + 2].clone();
        let l11 = linear_claims[old_base + 3].clone();
        let l0 = l00.clone() + r.clone() * (l01 - l00);
        let l1 = l10.clone() + r.clone() * (l11 - l10);
        linear_claims[folded_even] = l0.clone();
        linear_claims[folded_odd] = l1.clone();
        linear_zero += pair_weight.clone() * l0;
        if need_t_one {
            linear_one += pair_weight.clone() * l1;
        }

        let mut source_zero = F::zero_with_cfg(field_cfg);
        let mut source_infinity = F::zero_with_cfg(field_cfg);
        let mut source_one = F::zero_with_cfg(field_cfg);
        for (source_row, scale) in source_row_weights.iter().enumerate() {
            let old_00 = suffix_flat_index(old_base, source_row, source_row_count);
            let old_01 = suffix_flat_index(old_base + 1, source_row, source_row_count);
            let old_10 = suffix_flat_index(old_base + 2, source_row, source_row_count);
            let old_11 = suffix_flat_index(old_base + 3, source_row, source_row_count);
            let d00 = booleanity_values[old_00].clone();
            let d01 = booleanity_values[old_01].clone();
            let d10 = booleanity_values[old_10].clone();
            let d11 = booleanity_values[old_11].clone();
            let d0 = d00.clone() + r.clone() * (d01 - d00);
            let d1 = d10.clone() + r.clone() * (d11 - d10);
            let new_0 = suffix_flat_index(folded_even, source_row, source_row_count);
            let new_1 = suffix_flat_index(folded_odd, source_row, source_row_count);
            booleanity_values[new_0] = d0.clone();
            booleanity_values[new_1] = d1.clone();

            if F::is_zero(scale) {
                continue;
            }
            source_zero += scale.clone() * d0.clone() * (d0.clone() - one.clone());
            let delta = d1.clone() - d0;
            source_infinity += scale.clone() * delta.clone() * delta;
            if need_t_one {
                source_one += scale.clone() * d1.clone() * (d1 - one.clone());
            }
        }

        quadratic_zero += pair_weight.clone() * source_zero;
        quadratic_infinity += pair_weight.clone() * source_infinity;
        if need_t_one {
            quadratic_one += pair_weight * source_one;
        }
    }

    let t_one = need_t_one.then(|| linear_one + quadratic_one);
    (linear_zero + quadratic_zero, quadratic_infinity, t_one)
}

#[inline(always)]
fn add_product_sum4<F>(
    product_sum: &MontgomeryProductSum4<'_, F>,
    total: &mut F,
    acc: &mut ProductAccumulator4,
    lhs: &F,
    rhs: &F,
) where
    F: MontgomeryLimbs + Send + Sync,
{
    if F::is_zero(lhs) || F::is_zero(rhs) {
        return;
    }
    product_sum.add_product(acc, lhs, rhs);
    if acc.pending_products() >= product_sum.flush_products() {
        *total += product_sum.reduce_products(*acc);
        *acc = product_sum.zero_accumulator();
    }
}

#[inline(always)]
fn finish_product_sum4<F>(
    product_sum: &MontgomeryProductSum4<'_, F>,
    mut total: F,
    acc: ProductAccumulator4,
) -> F
where
    F: MontgomeryLimbs + Send + Sync,
{
    if acc.pending_products() != 0 {
        total += product_sum.reduce_products(acc);
    }
    total
}

#[allow(clippy::arithmetic_side_effects)]
fn suffix_reduced_body_buckets_dmr_with_algorithm<F>(
    product_sum: &MontgomeryProductSum4<'_, F>,
    linear_claims: &[F],
    booleanity_claims: &[Vec<F>],
    source_row_weights: &[F],
    suffix_eq_weights: &[F],
    field_cfg: &F::Config,
) -> (F, F)
where
    F: MontgomeryLimbs + Send + Sync,
{
    debug_assert_eq!(linear_claims.len(), suffix_eq_weights.len());
    debug_assert_eq!(suffix_eq_weights.len() & 1, 0);
    debug_assert_eq!(booleanity_claims.len(), source_row_weights.len());
    for values in booleanity_claims {
        debug_assert_eq!(values.len(), suffix_eq_weights.len());
    }

    let zero = F::zero_with_cfg(field_cfg);
    let flush_products = product_sum.flush_products();
    let chunk_buckets: Vec<_> = cfg_chunks!(suffix_eq_weights, SHA_SUFFIX_DMR_WEIGHT_CHUNK)
        .enumerate()
        .map(|(chunk_idx, suffix_eq_chunk)| {
            let chunk_product_sum =
                MontgomeryProductSum4::<F>::new_with_flush_products(field_cfg, flush_products);
            suffix_reduced_body_buckets_dmr_chunk(
                &chunk_product_sum,
                linear_claims,
                booleanity_claims,
                source_row_weights,
                suffix_eq_chunk,
                chunk_idx * SHA_SUFFIX_DMR_PAIR_CHUNK,
                field_cfg,
            )
        })
        .collect();

    let mut linear_zero = zero.clone();
    let mut quadratic_zero = zero.clone();
    let mut quadratic_infinity = zero;
    for (linear_chunk, quadratic_zero_chunk, quadratic_infinity_chunk) in chunk_buckets {
        linear_zero += linear_chunk;
        quadratic_zero += quadratic_zero_chunk;
        quadratic_infinity += quadratic_infinity_chunk;
    }

    (linear_zero + quadratic_zero, quadratic_infinity)
}

#[allow(clippy::arithmetic_side_effects)]
fn suffix_reduced_body_buckets_dmr_chunk<F>(
    product_sum: &MontgomeryProductSum4<'_, F>,
    linear_claims: &[F],
    booleanity_claims: &[Vec<F>],
    source_row_weights: &[F],
    suffix_eq_chunk: &[F],
    pair_offset: usize,
    field_cfg: &F::Config,
) -> (F, F, F)
where
    F: MontgomeryLimbs + Send + Sync,
{
    debug_assert_eq!(suffix_eq_chunk.len() & 1, 0);

    let rest_len = suffix_eq_chunk.len() >> 1;
    let one = F::one_with_cfg(field_cfg);
    let zero = F::zero_with_cfg(field_cfg);

    let mut linear_total = zero.clone();
    let mut linear_acc = product_sum.zero_accumulator();
    let mut quadratic_zero_total = zero.clone();
    let mut quadratic_zero_acc = product_sum.zero_accumulator();
    let mut quadratic_infinity_total = zero.clone();
    let mut quadratic_infinity_acc = product_sum.zero_accumulator();

    for local_rest in 0..rest_len {
        let rest = pair_offset + local_rest;
        let weight = suffix_pair_weight(suffix_eq_chunk, local_rest);
        add_product_sum4(
            product_sum,
            &mut linear_total,
            &mut linear_acc,
            &weight,
            &linear_claims[rest << 1],
        );

        let mut source_zero_total = zero.clone();
        let mut source_zero_acc = product_sum.zero_accumulator();
        let mut source_infinity_total = zero.clone();
        let mut source_infinity_acc = product_sum.zero_accumulator();

        for (values, scale) in booleanity_claims.iter().zip(source_row_weights) {
            if F::is_zero(scale) {
                continue;
            }
            let even = values[rest << 1].clone();
            let odd = values[(rest << 1) + 1].clone();
            let zero_term = even.clone() * (even.clone() - one.clone());
            add_product_sum4(
                product_sum,
                &mut source_zero_total,
                &mut source_zero_acc,
                scale,
                &zero_term,
            );

            let delta = odd - even;
            let infinity_term = delta.clone() * delta;
            add_product_sum4(
                product_sum,
                &mut source_infinity_total,
                &mut source_infinity_acc,
                scale,
                &infinity_term,
            );
        }

        let source_zero = finish_product_sum4(product_sum, source_zero_total, source_zero_acc);
        add_product_sum4(
            product_sum,
            &mut quadratic_zero_total,
            &mut quadratic_zero_acc,
            &weight,
            &source_zero,
        );

        let source_infinity =
            finish_product_sum4(product_sum, source_infinity_total, source_infinity_acc);
        add_product_sum4(
            product_sum,
            &mut quadratic_infinity_total,
            &mut quadratic_infinity_acc,
            &weight,
            &source_infinity,
        );
    }

    let linear_zero = finish_product_sum4(product_sum, linear_total, linear_acc);
    let quadratic_zero = finish_product_sum4(product_sum, quadratic_zero_total, quadratic_zero_acc);
    let quadratic_infinity = finish_product_sum4(
        product_sum,
        quadratic_infinity_total,
        quadratic_infinity_acc,
    );

    (linear_zero, quadratic_zero, quadratic_infinity)
}

#[allow(clippy::arithmetic_side_effects)]
fn suffix_direct_one_body_bucket_dmr_with_algorithm<F>(
    product_sum: &MontgomeryProductSum4<'_, F>,
    linear_claims: &[F],
    booleanity_claims: &[Vec<F>],
    source_row_weights: &[F],
    suffix_eq_weights: &[F],
    field_cfg: &F::Config,
) -> F
where
    F: MontgomeryLimbs + Send + Sync,
{
    debug_assert_eq!(linear_claims.len(), suffix_eq_weights.len());
    debug_assert_eq!(suffix_eq_weights.len() & 1, 0);
    debug_assert_eq!(booleanity_claims.len(), source_row_weights.len());
    for values in booleanity_claims {
        debug_assert_eq!(values.len(), suffix_eq_weights.len());
    }

    let zero = F::zero_with_cfg(field_cfg);
    let flush_products = product_sum.flush_products();
    let chunk_buckets: Vec<_> = cfg_chunks!(suffix_eq_weights, SHA_SUFFIX_DMR_WEIGHT_CHUNK)
        .enumerate()
        .map(|(chunk_idx, suffix_eq_chunk)| {
            let chunk_product_sum =
                MontgomeryProductSum4::<F>::new_with_flush_products(field_cfg, flush_products);
            suffix_direct_one_body_bucket_dmr_chunk(
                &chunk_product_sum,
                linear_claims,
                booleanity_claims,
                source_row_weights,
                suffix_eq_chunk,
                chunk_idx * SHA_SUFFIX_DMR_PAIR_CHUNK,
                field_cfg,
            )
        })
        .collect();

    let mut total = zero;
    for chunk_bucket in chunk_buckets {
        total += chunk_bucket;
    }

    total
}

#[allow(clippy::arithmetic_side_effects)]
fn suffix_direct_one_body_bucket_dmr_chunk<F>(
    product_sum: &MontgomeryProductSum4<'_, F>,
    linear_claims: &[F],
    booleanity_claims: &[Vec<F>],
    source_row_weights: &[F],
    suffix_eq_chunk: &[F],
    pair_offset: usize,
    field_cfg: &F::Config,
) -> F
where
    F: MontgomeryLimbs + Send + Sync,
{
    debug_assert_eq!(suffix_eq_chunk.len() & 1, 0);

    let rest_len = suffix_eq_chunk.len() >> 1;
    let one = F::one_with_cfg(field_cfg);
    let zero = F::zero_with_cfg(field_cfg);
    let mut total = zero.clone();
    let mut acc = product_sum.zero_accumulator();

    for local_rest in 0..rest_len {
        let rest = pair_offset + local_rest;
        let weight = suffix_pair_weight(suffix_eq_chunk, local_rest);
        add_product_sum4(
            product_sum,
            &mut total,
            &mut acc,
            &weight,
            &linear_claims[(rest << 1) + 1],
        );

        let mut source_one_total = zero.clone();
        let mut source_one_acc = product_sum.zero_accumulator();
        for (values, scale) in booleanity_claims.iter().zip(source_row_weights) {
            if F::is_zero(scale) {
                continue;
            }
            let odd = values[(rest << 1) + 1].clone();
            let one_term = odd.clone() * (odd - one.clone());
            add_product_sum4(
                product_sum,
                &mut source_one_total,
                &mut source_one_acc,
                scale,
                &one_term,
            );
        }
        let source_one = finish_product_sum4(product_sum, source_one_total, source_one_acc);
        add_product_sum4(product_sum, &mut total, &mut acc, &weight, &source_one);
    }

    finish_product_sum4(product_sum, total, acc)
}

#[allow(clippy::arithmetic_side_effects)]
fn suffix_reduced_body_buckets_flat_dmr_with_algorithm<F>(
    product_sum: &MontgomeryProductSum4<'_, F>,
    linear_claims: &[F],
    booleanity_values: &[F],
    source_row_count: usize,
    source_row_weights: &[F],
    suffix_eq_weights: &[F],
    field_cfg: &F::Config,
) -> (F, F)
where
    F: MontgomeryLimbs + Send + Sync,
{
    debug_assert_eq!(linear_claims.len(), suffix_eq_weights.len());
    debug_assert_eq!(suffix_eq_weights.len() & 1, 0);
    debug_assert_eq!(source_row_weights.len(), source_row_count);
    debug_assert_eq!(
        booleanity_values.len(),
        suffix_eq_weights.len() * source_row_count
    );

    let rest_len = suffix_eq_weights.len() >> 1;
    let one = F::one_with_cfg(field_cfg);
    let zero = F::zero_with_cfg(field_cfg);
    let mut linear_total = zero.clone();
    let mut linear_acc = product_sum.zero_accumulator();
    let mut quadratic_zero_total = zero.clone();
    let mut quadratic_zero_acc = product_sum.zero_accumulator();
    let mut quadratic_infinity_total = zero.clone();
    let mut quadratic_infinity_acc = product_sum.zero_accumulator();

    for rest in 0..rest_len {
        let even_tail = rest << 1;
        let odd_tail = even_tail + 1;
        let weight = suffix_pair_weight(suffix_eq_weights, rest);
        add_product_sum4(
            product_sum,
            &mut linear_total,
            &mut linear_acc,
            &weight,
            &linear_claims[even_tail],
        );

        let mut source_zero_total = zero.clone();
        let mut source_zero_acc = product_sum.zero_accumulator();
        let mut source_infinity_total = zero.clone();
        let mut source_infinity_acc = product_sum.zero_accumulator();
        for (source_row, scale) in source_row_weights.iter().enumerate() {
            if F::is_zero(scale) {
                continue;
            }
            let even = booleanity_values
                [suffix_flat_index(even_tail, source_row, source_row_count)]
            .clone();
            let odd = booleanity_values[suffix_flat_index(odd_tail, source_row, source_row_count)]
                .clone();
            let zero_term = even.clone() * (even.clone() - one.clone());
            add_product_sum4(
                product_sum,
                &mut source_zero_total,
                &mut source_zero_acc,
                scale,
                &zero_term,
            );

            let delta = odd - even;
            let infinity_term = delta.clone() * delta;
            add_product_sum4(
                product_sum,
                &mut source_infinity_total,
                &mut source_infinity_acc,
                scale,
                &infinity_term,
            );
        }

        let source_zero = finish_product_sum4(product_sum, source_zero_total, source_zero_acc);
        add_product_sum4(
            product_sum,
            &mut quadratic_zero_total,
            &mut quadratic_zero_acc,
            &weight,
            &source_zero,
        );

        let source_infinity =
            finish_product_sum4(product_sum, source_infinity_total, source_infinity_acc);
        add_product_sum4(
            product_sum,
            &mut quadratic_infinity_total,
            &mut quadratic_infinity_acc,
            &weight,
            &source_infinity,
        );
    }

    let linear_zero = finish_product_sum4(product_sum, linear_total, linear_acc);
    let quadratic_zero = finish_product_sum4(product_sum, quadratic_zero_total, quadratic_zero_acc);
    let quadratic_infinity = finish_product_sum4(
        product_sum,
        quadratic_infinity_total,
        quadratic_infinity_acc,
    );
    (linear_zero + quadratic_zero, quadratic_infinity)
}

#[allow(clippy::arithmetic_side_effects)]
fn suffix_direct_one_body_bucket_flat_dmr_with_algorithm<F>(
    product_sum: &MontgomeryProductSum4<'_, F>,
    linear_claims: &[F],
    booleanity_values: &[F],
    source_row_count: usize,
    source_row_weights: &[F],
    suffix_eq_weights: &[F],
    field_cfg: &F::Config,
) -> F
where
    F: MontgomeryLimbs + Send + Sync,
{
    debug_assert_eq!(linear_claims.len(), suffix_eq_weights.len());
    debug_assert_eq!(suffix_eq_weights.len() & 1, 0);
    debug_assert_eq!(source_row_weights.len(), source_row_count);
    debug_assert_eq!(
        booleanity_values.len(),
        suffix_eq_weights.len() * source_row_count
    );

    let rest_len = suffix_eq_weights.len() >> 1;
    let one = F::one_with_cfg(field_cfg);
    let zero = F::zero_with_cfg(field_cfg);
    let mut total = zero.clone();
    let mut acc = product_sum.zero_accumulator();

    for rest in 0..rest_len {
        let odd_tail = (rest << 1) + 1;
        let weight = suffix_pair_weight(suffix_eq_weights, rest);
        add_product_sum4(
            product_sum,
            &mut total,
            &mut acc,
            &weight,
            &linear_claims[odd_tail],
        );

        let mut source_one_total = zero.clone();
        let mut source_one_acc = product_sum.zero_accumulator();
        for (source_row, scale) in source_row_weights.iter().enumerate() {
            if F::is_zero(scale) {
                continue;
            }
            let odd = booleanity_values[suffix_flat_index(odd_tail, source_row, source_row_count)]
                .clone();
            let one_term = odd.clone() * (odd - one.clone());
            add_product_sum4(
                product_sum,
                &mut source_one_total,
                &mut source_one_acc,
                scale,
                &one_term,
            );
        }
        let source_one = finish_product_sum4(product_sum, source_one_total, source_one_acc);
        add_product_sum4(product_sum, &mut total, &mut acc, &weight, &source_one);
    }

    finish_product_sum4(product_sum, total, acc)
}

#[allow(clippy::arithmetic_side_effects)]
fn suffix_fold_prepare_next_round_flat_dmr_with_algorithm<F>(
    product_sum: &MontgomeryProductSum4<'_, F>,
    linear_claims: &mut [F],
    booleanity_values: &mut [F],
    source_row_count: usize,
    source_row_weights: &[F],
    suffix_eq_weights: &[F],
    r: &F,
    need_t_one: bool,
    field_cfg: &F::Config,
) -> (F, F, Option<F>)
where
    F: MontgomeryLimbs + Send + Sync,
{
    debug_assert_eq!(linear_claims.len(), suffix_eq_weights.len());
    debug_assert_eq!(source_row_weights.len(), source_row_count);
    debug_assert_eq!(
        booleanity_values.len(),
        suffix_eq_weights.len() * source_row_count
    );
    debug_assert_eq!(suffix_eq_weights.len() & 3, 0);

    let next_pair_count = suffix_eq_weights.len() >> 2;
    let one = F::one_with_cfg(field_cfg);
    let zero = F::zero_with_cfg(field_cfg);
    let mut linear_zero_total = zero.clone();
    let mut linear_zero_acc = product_sum.zero_accumulator();
    let mut linear_one_total = zero.clone();
    let mut linear_one_acc = product_sum.zero_accumulator();
    let mut quadratic_zero_total = zero.clone();
    let mut quadratic_zero_acc = product_sum.zero_accumulator();
    let mut quadratic_infinity_total = zero.clone();
    let mut quadratic_infinity_acc = product_sum.zero_accumulator();
    let mut quadratic_one_total = zero.clone();
    let mut quadratic_one_acc = product_sum.zero_accumulator();

    for next_pair in 0..next_pair_count {
        let old_base = next_pair << 2;
        let folded_even = next_pair << 1;
        let folded_odd = folded_even + 1;
        let pair_weight = suffix_eq_weights[old_base].clone()
            + &suffix_eq_weights[old_base + 1]
            + &suffix_eq_weights[old_base + 2]
            + &suffix_eq_weights[old_base + 3];

        let l00 = linear_claims[old_base].clone();
        let l01 = linear_claims[old_base + 1].clone();
        let l10 = linear_claims[old_base + 2].clone();
        let l11 = linear_claims[old_base + 3].clone();
        let l0 = l00.clone() + r.clone() * (l01 - l00);
        let l1 = l10.clone() + r.clone() * (l11 - l10);
        linear_claims[folded_even] = l0.clone();
        linear_claims[folded_odd] = l1.clone();
        add_product_sum4(
            product_sum,
            &mut linear_zero_total,
            &mut linear_zero_acc,
            &pair_weight,
            &l0,
        );
        if need_t_one {
            add_product_sum4(
                product_sum,
                &mut linear_one_total,
                &mut linear_one_acc,
                &pair_weight,
                &l1,
            );
        }

        let mut source_zero_total = zero.clone();
        let mut source_zero_acc = product_sum.zero_accumulator();
        let mut source_infinity_total = zero.clone();
        let mut source_infinity_acc = product_sum.zero_accumulator();
        let mut source_one_total = zero.clone();
        let mut source_one_acc = product_sum.zero_accumulator();
        for (source_row, scale) in source_row_weights.iter().enumerate() {
            let old_00 = suffix_flat_index(old_base, source_row, source_row_count);
            let old_01 = suffix_flat_index(old_base + 1, source_row, source_row_count);
            let old_10 = suffix_flat_index(old_base + 2, source_row, source_row_count);
            let old_11 = suffix_flat_index(old_base + 3, source_row, source_row_count);
            let d00 = booleanity_values[old_00].clone();
            let d01 = booleanity_values[old_01].clone();
            let d10 = booleanity_values[old_10].clone();
            let d11 = booleanity_values[old_11].clone();
            let d0 = d00.clone() + r.clone() * (d01 - d00);
            let d1 = d10.clone() + r.clone() * (d11 - d10);
            let new_0 = suffix_flat_index(folded_even, source_row, source_row_count);
            let new_1 = suffix_flat_index(folded_odd, source_row, source_row_count);
            booleanity_values[new_0] = d0.clone();
            booleanity_values[new_1] = d1.clone();

            if F::is_zero(scale) {
                continue;
            }
            let zero_term = d0.clone() * (d0.clone() - one.clone());
            add_product_sum4(
                product_sum,
                &mut source_zero_total,
                &mut source_zero_acc,
                scale,
                &zero_term,
            );

            let delta = d1.clone() - d0;
            let infinity_term = delta.clone() * delta;
            add_product_sum4(
                product_sum,
                &mut source_infinity_total,
                &mut source_infinity_acc,
                scale,
                &infinity_term,
            );

            if need_t_one {
                let one_term = d1.clone() * (d1 - one.clone());
                add_product_sum4(
                    product_sum,
                    &mut source_one_total,
                    &mut source_one_acc,
                    scale,
                    &one_term,
                );
            }
        }

        let source_zero = finish_product_sum4(product_sum, source_zero_total, source_zero_acc);
        add_product_sum4(
            product_sum,
            &mut quadratic_zero_total,
            &mut quadratic_zero_acc,
            &pair_weight,
            &source_zero,
        );

        let source_infinity =
            finish_product_sum4(product_sum, source_infinity_total, source_infinity_acc);
        add_product_sum4(
            product_sum,
            &mut quadratic_infinity_total,
            &mut quadratic_infinity_acc,
            &pair_weight,
            &source_infinity,
        );

        if need_t_one {
            let source_one = finish_product_sum4(product_sum, source_one_total, source_one_acc);
            add_product_sum4(
                product_sum,
                &mut quadratic_one_total,
                &mut quadratic_one_acc,
                &pair_weight,
                &source_one,
            );
        }
    }

    let linear_zero = finish_product_sum4(product_sum, linear_zero_total, linear_zero_acc);
    let quadratic_zero = finish_product_sum4(product_sum, quadratic_zero_total, quadratic_zero_acc);
    let quadratic_infinity = finish_product_sum4(
        product_sum,
        quadratic_infinity_total,
        quadratic_infinity_acc,
    );
    let t_one = if need_t_one {
        let linear_one = finish_product_sum4(product_sum, linear_one_total, linear_one_acc);
        let quadratic_one =
            finish_product_sum4(product_sum, quadratic_one_total, quadratic_one_acc);
        Some(linear_one + quadratic_one)
    } else {
        None
    };

    (linear_zero + quadratic_zero, quadratic_infinity, t_one)
}

fn build_sha_sumfold_linear_accumulator_direct_with_weights_generic<F, Trace, Public>(
    traces: &[Trace],
    publics: &[Public],
    plan: &ShaLinearResidualWeightPlan<F>,
    field_cfg: &F::Config,
) -> Result<Vec<F>, ShaProjectionError>
where
    F: DelayedFieldProductSum + Send + Sync,
    Trace: Borrow<ProjectedTrace<F>> + Sync,
    Public: Borrow<ProjectedPublic<F>> + Sync,
{
    if traces.len() != publics.len() {
        return Err(ShaProjectionError::InstanceCountMismatch {
            got: publics.len(),
            expected: traces.len(),
        });
    }
    #[cfg(debug_assertions)]
    {
        for (trace, public) in traces.iter().zip(publics.iter()) {
            validate_trace(trace.borrow())?;
            validate_public(public.borrow())?;
        }
    }
    let tasks = sha_direct_row_tasks(traces.len(), plan.row_weights.len());
    let partials = cfg_iter!(&tasks)
        .map(|&(instance_idx, row_start, row_end)| {
            sha_linear_residual_partial_sum_with_weights(
                traces[instance_idx].borrow(),
                publics[instance_idx].borrow(),
                &plan.row_weights,
                row_start,
                row_end,
                &plan.a_powers,
                &plan.lambda_powers,
                field_cfg,
            )
            .map(|partial| (instance_idx, partial))
        })
        .collect::<Vec<Result<_, ShaProjectionError>>>();
    let mut out = vec![F::zero_with_cfg(field_cfg); traces.len()];
    for partial in partials {
        let (instance_idx, value) = partial?;
        out[instance_idx] += value;
    }
    Ok(out)
}

fn build_sha_sumfold_linear_accumulator_direct_with_weights_dmr<F, Trace, Public>(
    traces: &[Trace],
    publics: &[Public],
    plan: &ShaLinearResidualWeightPlan<F>,
    field_cfg: &F::Config,
) -> Result<Vec<F>, ShaProjectionError>
where
    F: MontgomeryLimbs + DelayedFieldProductSum + Send + Sync,
    Trace: Borrow<ProjectedTrace<F>> + Sync,
    Public: Borrow<ProjectedPublic<F>> + Sync,
{
    if traces.len() != publics.len() {
        return Err(ShaProjectionError::InstanceCountMismatch {
            got: publics.len(),
            expected: traces.len(),
        });
    }
    let eval_plan = ShaDirectResidualEvalPlan::new(&plan.a_powers, &plan.lambda_powers, field_cfg)?;
    let reducer = BarrettDelayedReduction::<F>::new(field_cfg);
    #[cfg(debug_assertions)]
    {
        for (trace, public) in traces.iter().zip(publics.iter()) {
            validate_trace(trace.borrow())?;
            validate_public(public.borrow())?;
        }
    }
    let tasks = sha_direct_row_tasks(traces.len(), plan.row_weights.len());
    let partials = cfg_iter!(&tasks)
        .map(|&(instance_idx, row_start, row_end)| {
            sha_linear_residual_partial_sum_with_plan_dmr(
                traces[instance_idx].borrow(),
                publics[instance_idx].borrow(),
                &plan.row_weights,
                row_start,
                row_end,
                &eval_plan,
                &reducer,
                field_cfg,
            )
            .map(|partial| (instance_idx, partial))
        })
        .collect::<Vec<Result<_, ShaProjectionError>>>();
    let mut out = vec![F::zero_with_cfg(field_cfg); traces.len()];
    for partial in partials {
        let (instance_idx, value) = partial?;
        out[instance_idx] += value;
    }
    Ok(out)
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
            add_scaled_poly_assign(&mut out[slot], &residuals[family.index()], row_weight);
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

pub fn build_linear_residual_coeff_tables<F, Trace, Public>(
    traces: &[Trace],
    publics: &[Public],
    r_ic: &[F; SHA_ROW_VARS],
    field_cfg: &F::Config,
) -> Result<Vec<LinearResidualCoeffTable<F>>, ShaProjectionError>
where
    F: PrimeField,
    Trace: Borrow<ProjectedTrace<F>> + Sync,
    Public: Borrow<ProjectedPublic<F>> + Sync,
{
    let row_weights = build_eq_x_r_vec(r_ic, field_cfg)?;
    build_linear_residual_coeff_tables_with_row_weights(traces, publics, &row_weights, field_cfg)
}

pub fn build_linear_residual_coeff_tables_with_row_weights<F, Trace, Public>(
    traces: &[Trace],
    publics: &[Public],
    row_weights: &[F],
    field_cfg: &F::Config,
) -> Result<Vec<LinearResidualCoeffTable<F>>, ShaProjectionError>
where
    F: PrimeField,
    Trace: Borrow<ProjectedTrace<F>> + Sync,
    Public: Borrow<ProjectedPublic<F>> + Sync,
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
    let constants: ShaResidualPolyConstants<F> = ShaResidualPolyConstants::new(field_cfg);
    cfg_iter!(traces)
        .zip(cfg_iter!(publics))
        .map(|(trace, public)| {
            let trace = trace.borrow();
            let public = public.borrow();
            #[cfg(debug_assertions)]
            {
                validate_trace(trace)?;
                validate_public(public)?;
            }
            let partials = cfg_chunks!(row_weights, 64)
                .enumerate()
                .map(|(chunk_idx, row_weight_chunk)| {
                    let mut partial = FixedResidualCoeffAccumulator::new(
                        NUM_SHA_RESIDUAL_FAMILIES,
                        SHA_RESIDUAL_EVAL_POWER_COUNT,
                        field_cfg,
                    );
                    let row_offset = chunk_idx * 64;
                    for (row_in_chunk, row_weight) in row_weight_chunk.iter().enumerate() {
                        let row = row_offset + row_in_chunk;
                        accumulate_residual_row_fixed(
                            &mut partial,
                            trace,
                            public,
                            row,
                            row_weight,
                            &constants,
                            field_cfg,
                        )?;
                    }
                    Ok(partial)
                })
                .collect::<Result<Vec<_>, ShaProjectionError>>()?;
            let mut coeffs = FixedResidualCoeffAccumulator::new(
                NUM_SHA_RESIDUAL_FAMILIES,
                SHA_RESIDUAL_EVAL_POWER_COUNT,
                field_cfg,
            );
            for partial in partials {
                coeffs.add_assign(partial);
            }
            Ok(coeffs.into_table())
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
    let nonzero_lambda_powers = selected_nonzero_sha_lambda_powers(&lambda_powers)?;

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
        let target = FieldFieldInnerProduct::inner_product::<UNCHECKED>(
            &nonzero_lambda_powers,
            &taus,
            zero.clone(),
        )
        .map_err(ShaProjectionError::from)?;
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

pub fn fold_projected_traces<F, Trace, Public>(
    traces: &[Trace],
    publics: &[Public],
    sumfold: &InstanceFoldClaim<F>,
    field_cfg: &F::Config,
) -> Result<(ProjectionFoldWitness<F>, ProjectedPublic<F>), ShaProjectionError>
where
    F: ShaBinaryFoldField,
    Trace: Borrow<ProjectedTrace<F>>,
    Public: Borrow<ProjectedPublic<F>>,
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
    #[cfg(debug_assertions)]
    {
        for trace in traces {
            validate_trace(trace.borrow())?;
        }
        for public in publics {
            validate_public(public.borrow())?;
        }
    }

    let folded_public_columns = fold_mle_tables(
        "public.columns",
        publics.iter().map(|public| &public.borrow().columns),
        &sumfold.eq_instance_weights,
        field_cfg,
    )?;
    let folded_trace = ProjectedTrace {
        bit_slices: fold_binary_mle_tables(
            "bit_slices",
            traces.iter().map(|trace| &trace.borrow().bit_slices),
            &sumfold.eq_instance_weights,
            field_cfg,
        )?,
        scalarized: fold_mle_tables(
            "scalarized",
            traces.iter().map(|trace| &trace.borrow().scalarized),
            &sumfold.eq_instance_weights,
            field_cfg,
        )?,
        int_columns: fold_mle_tables(
            "int_columns",
            traces.iter().map(|trace| &trace.borrow().int_columns),
            &sumfold.eq_instance_weights,
            field_cfg,
        )?,
        public_columns: folded_public_columns.clone(),
    };
    let folded_public = ProjectedPublic {
        columns: folded_public_columns,
        bit_slices: fold_optional_binary_mle_tables(
            "public.bit_slices",
            publics
                .iter()
                .map(|public| public.borrow().bit_slices.as_ref()),
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
    let reducer = BarrettDelayedReduction::<F>::new(field_cfg);
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
                &bits, &powers, field_cfg, &reducer,
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

#[inline(always)]
fn double_bool_or_mul<F>(bit: F, zero: &F, one: &F, two: &F) -> F
where
    F: PrimeField,
{
    if &bit == zero {
        zero.clone()
    } else if &bit == one {
        two.clone()
    } else {
        two.clone() * bit
    }
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
    let zero = F::zero_with_cfg(field_cfg);
    let one = F::one_with_cfg(field_cfg);
    let two = one.clone() + &one;
    let ch1 = build_virtual_bit_array(|bit| {
        Ok(
            bit_at_shifted_or_zero_fast(trace, ShaWordCol::E, row, 2, bit, field_cfg)
                + bit_at_shifted_or_zero_fast(trace, ShaWordCol::E, row, 1, bit, field_cfg)
                - double_bool_or_mul(
                    bit_at_shifted_or_zero_fast(trace, ShaWordCol::Uef, row, 2, bit, field_cfg),
                    &zero,
                    &one,
                    &two,
                ),
        )
    });
    let ch2 = build_virtual_bit_array(|bit| {
        Ok(
            bit_at_shifted_or_zero_fast(trace, ShaWordCol::E, row, 2, bit, field_cfg)
                - bit_at_shifted_or_zero_fast(trace, ShaWordCol::E, row, 0, bit, field_cfg)
                + double_bool_or_mul(
                    bit_at_shifted_or_zero_fast(trace, ShaWordCol::UNegEg, row, 2, bit, field_cfg),
                    &zero,
                    &one,
                    &two,
                )
                + double_bool_or_mul(
                    bit_at_shifted_or_zero_fast(trace, ShaWordCol::Ch2Comp, row, 0, bit, field_cfg),
                    &zero,
                    &one,
                    &two,
                ),
        )
    });
    let maj = build_virtual_bit_array(|bit| {
        Ok(
            bit_at_shifted_or_zero_fast(trace, ShaWordCol::A, row, 0, bit, field_cfg)
                + bit_at_shifted_or_zero_fast(trace, ShaWordCol::A, row, 1, bit, field_cfg)
                + bit_at_shifted_or_zero_fast(trace, ShaWordCol::A, row, 2, bit, field_cfg)
                - double_bool_or_mul(
                    bit_at_shifted_or_zero_fast(trace, ShaWordCol::Maj, row, 2, bit, field_cfg),
                    &zero,
                    &one,
                    &two,
                )
                - double_bool_or_mul(
                    bit_at_shifted_or_zero_fast(trace, ShaWordCol::MajComp, row, 0, bit, field_cfg),
                    &zero,
                    &one,
                    &two,
                ),
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

        let mut bool_terms = Vec::with_capacity(booleanity_sources.len());
        let virtuals = if needs_virtuals {
            Some(reconstruct_virtual_ch_maj_at_row_unchecked(
                trace, row, field_cfg,
            )?)
        } else {
            None
        };
        for source in booleanity_sources {
            let d = booleanity_source_value_at_row_with_virtuals(
                trace,
                row,
                source,
                virtuals.as_ref(),
                field_cfg,
            )?;
            let term = d.clone() * (d - F::one_with_cfg(field_cfg));
            bool_terms.push(term);
        }
        let bool_sum = FieldFieldInnerProduct::inner_product::<UNCHECKED>(
            booleanity_weights,
            &bool_terms,
            F::zero_with_cfg(field_cfg),
        )
        .map_err(ShaProjectionError::from)?;
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
    sha_linear_residual_partial_sum_with_weights(
        trace,
        public,
        row_weights,
        0,
        row_weights.len(),
        a_powers,
        lambda_powers,
        field_cfg,
    )
}

#[allow(clippy::too_many_arguments)]
fn sha_linear_residual_partial_sum_with_weights<F>(
    trace: &ProjectedTrace<F>,
    public: &ProjectedPublic<F>,
    row_weights: &[F],
    row_start: usize,
    row_end: usize,
    a_powers: &[F],
    lambda_powers: &[F],
    field_cfg: &F::Config,
) -> Result<F, ShaProjectionError>
where
    F: DelayedFieldProductSum,
{
    let mut values = Vec::with_capacity(row_end - row_start);
    for row in row_start..row_end {
        values.push(sha_linear_residual_row_value_with_powers(
            trace,
            public,
            row,
            a_powers,
            lambda_powers,
            field_cfg,
        )?);
    }
    FieldFieldInnerProduct::inner_product::<UNCHECKED>(
        &row_weights[row_start..row_end],
        &values,
        F::zero_with_cfg(field_cfg),
    )
    .map_err(ShaProjectionError::from)
}

#[allow(clippy::too_many_arguments)]
fn sha_linear_residual_partial_sum_with_plan_dmr<F>(
    trace: &ProjectedTrace<F>,
    public: &ProjectedPublic<F>,
    row_weights: &[F],
    row_start: usize,
    row_end: usize,
    eval_plan: &ShaDirectResidualEvalPlan<F>,
    reducer: &BarrettDelayedReduction<'_, F>,
    field_cfg: &F::Config,
) -> Result<F, ShaProjectionError>
where
    F: MontgomeryLimbs + DelayedFieldProductSum + Send + Sync,
{
    let mut values = Vec::with_capacity(row_end - row_start);
    for row in row_start..row_end {
        values.push(sha_linear_residual_row_value_direct(
            trace, public, row, eval_plan, reducer, field_cfg,
        )?);
    }
    FieldFieldInnerProduct::inner_product::<UNCHECKED>(
        &row_weights[row_start..row_end],
        &values,
        F::zero_with_cfg(field_cfg),
    )
    .map_err(ShaProjectionError::from)
}

#[derive(Clone, Debug)]
struct ShaDirectResidualEvalPlan<F: PrimeField> {
    word_weights: Vec<F>,
    w0_base_weights: Vec<F>,
    mu_packed_weights: Vec<F>,
    lambda_powers: Vec<F>,
    rho_sig0: F,
    rho_sig1: F,
    two: F,
}

impl<F> ShaDirectResidualEvalPlan<F>
where
    F: PrimeField,
{
    fn new(
        a_powers: &[F],
        lambda_powers: &[F],
        field_cfg: &F::Config,
    ) -> Result<Self, ShaProjectionError> {
        if a_powers.len() < SHA_RESIDUAL_EVAL_POWER_COUNT {
            return Err(ShaProjectionError::MissingColumn {
                kind: "a_powers",
                col: a_powers.len(),
            });
        }
        if lambda_powers.len() != NUM_SHA_RESIDUAL_FAMILIES {
            return Err(ShaProjectionError::MissingColumn {
                kind: "lambda_powers",
                col: lambda_powers.len(),
            });
        }

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
        let rot_weights = |shift: usize| {
            (0..SHA_WORD_BITS)
                .map(|bit| a_powers[(bit + shift) % SHA_WORD_BITS].clone())
                .collect::<Vec<_>>()
        };
        let word_weights = a_powers[..SHA_WORD_BITS].to_vec();
        let rot25_weights = rot_weights(25);
        let rot14_weights = rot_weights(14);
        let rot15_weights = rot_weights(15);
        let rot13_weights = rot_weights(13);
        let shift0_weights = shift_weights(0);
        let shift2_weights = shift_weights(2);
        let shift3_weights = shift_weights(3);
        let shift5_weights = shift_weights(5);
        let shift8_weights = shift_weights(8);
        let shift9_weights = shift_weights(9);
        let shift10_weights = shift_weights(10);

        let zero = F::zero_with_cfg(field_cfg);
        let mut w0_base_weights = vec![zero.clone(); SHA_WORD_BITS];
        add_scaled_weights(&mut w0_base_weights, &rot25_weights, &lambda_powers[2]);
        add_scaled_weights(&mut w0_base_weights, &rot14_weights, &lambda_powers[2]);
        add_scaled_weights(&mut w0_base_weights, &shift3_weights, &lambda_powers[2]);
        add_scaled_weights(&mut w0_base_weights, &rot15_weights, &lambda_powers[3]);
        add_scaled_weights(&mut w0_base_weights, &rot13_weights, &lambda_powers[3]);
        add_scaled_weights(&mut w0_base_weights, &shift10_weights, &lambda_powers[3]);
        let w0_word_coeff =
            zero.clone() - &(lambda_powers[4].clone() + &lambda_powers[5] + &lambda_powers[6]);
        add_scaled_weights(&mut w0_base_weights, &word_weights, &w0_word_coeff);

        let low_mu_coeff = pow_two(32, field_cfg);
        let high_mu_w_coeff = pow_two(34, field_cfg);
        let high_mu_3_bit_coeff = pow_two(35, field_cfg);
        let high_mu_1_bit_coeff = pow_two(33, field_cfg);
        let mut mu_packed_weights = vec![zero.clone(); SHA_WORD_BITS];
        add_scaled_weights(
            &mut mu_packed_weights,
            &shift0_weights,
            &(lambda_powers[4].clone() * &low_mu_coeff),
        );
        add_scaled_weights(
            &mut mu_packed_weights,
            &shift2_weights,
            &(lambda_powers[5].clone() * &low_mu_coeff
                - &(lambda_powers[4].clone() * &high_mu_w_coeff)),
        );
        add_scaled_weights(
            &mut mu_packed_weights,
            &shift5_weights,
            &(lambda_powers[6].clone() * &low_mu_coeff
                - &(lambda_powers[5].clone() * &high_mu_3_bit_coeff)),
        );
        add_scaled_weights(
            &mut mu_packed_weights,
            &shift8_weights,
            &(lambda_powers[9].clone() * &low_mu_coeff
                - &(lambda_powers[6].clone() * &high_mu_3_bit_coeff)),
        );
        add_scaled_weights(
            &mut mu_packed_weights,
            &shift9_weights,
            &(lambda_powers[10].clone() * &low_mu_coeff
                - &(lambda_powers[9].clone() * &high_mu_1_bit_coeff)),
        );
        add_scaled_weights(
            &mut mu_packed_weights,
            &shift10_weights,
            &(lambda_powers[17].clone() - &(lambda_powers[10].clone() * &high_mu_1_bit_coeff)),
        );

        let one = F::one_with_cfg(field_cfg);
        let two = one.clone() + &one;
        Ok(Self {
            word_weights,
            w0_base_weights,
            mu_packed_weights,
            lambda_powers: lambda_powers.to_vec(),
            rho_sig0: a_powers[10].clone() + &a_powers[19] + &a_powers[30],
            rho_sig1: a_powers[7].clone() + &a_powers[21] + &a_powers[26],
            two,
        })
    }
}

fn add_scaled_weights<F>(dst: &mut [F], weights: &[F], scalar: &F)
where
    F: PrimeField,
{
    if F::is_zero(scalar) {
        return;
    }
    debug_assert_eq!(dst.len(), weights.len());
    for (dst, weight) in dst.iter_mut().zip(weights.iter()) {
        *dst += weight.clone() * scalar;
    }
}

fn sha_linear_residual_row_value_direct<F>(
    trace: &ProjectedTrace<F>,
    public: &ProjectedPublic<F>,
    row: usize,
    plan: &ShaDirectResidualEvalPlan<F>,
    reducer: &BarrettDelayedReduction<'_, F>,
    field_cfg: &F::Config,
) -> Result<F, ShaProjectionError>
where
    F: MontgomeryLimbs + DelayedFieldProductSum + Send + Sync,
{
    let lambda = &plan.lambda_powers;
    let s_init = public_scalar_shifted_fast(public, ShaPublicCol::SInit, row, 0, field_cfg);
    let s_msg = public_scalar_shifted_fast(public, ShaPublicCol::SMsg, row, 0, field_cfg);
    let s_sched = public_scalar_shifted_fast(public, ShaPublicCol::SSched, row, 0, field_cfg);
    let s_upd = public_scalar_shifted_fast(public, ShaPublicCol::SUpd, row, 0, field_cfg);
    let s_ff = public_scalar_shifted_fast(public, ShaPublicCol::SFf, row, 0, field_cfg);
    let s_out = public_scalar_shifted_fast(public, ShaPublicCol::SOut, row, 0, field_cfg);
    let s_init_out = s_init.clone() + &s_out;
    let zero = F::zero_with_cfg(field_cfg);
    let mut linear = zero.clone();

    let mut add_trace_word = |col: ShaWordCol,
                              shift: usize,
                              weights: &[F],
                              coeff: &F|
     -> Result<(), ShaProjectionError> {
        if F::is_zero(coeff) {
            return Ok(());
        }
        linear += trace_word_eval_at_row_with_weights(
            trace, col, row, shift, weights, field_cfg, reducer,
        )? * coeff;
        Ok(())
    };

    add_trace_word(
        ShaWordCol::A,
        0,
        &plan.word_weights,
        &(lambda[0].clone() * &plan.rho_sig0 - &lambda[6] + lambda[7].clone() * &s_init_out
            - &lambda[9]),
    )?;
    add_trace_word(
        ShaWordCol::E,
        0,
        &plan.word_weights,
        &(lambda[1].clone() * &plan.rho_sig1 - &lambda[5] - &lambda[6]
            + lambda[8].clone() * &s_init_out
            - &lambda[10]),
    )?;
    add_trace_word(
        ShaWordCol::Sigma0,
        0,
        &plan.word_weights,
        &(zero.clone() - &lambda[0]),
    )?;
    add_trace_word(
        ShaWordCol::Sigma1,
        0,
        &plan.word_weights,
        &(zero.clone() - &lambda[1]),
    )?;
    add_trace_word(
        ShaWordCol::SmallSigma0,
        0,
        &plan.word_weights,
        &(zero.clone() - &lambda[2]),
    )?;
    add_trace_word(
        ShaWordCol::SmallSigma1,
        0,
        &plan.word_weights,
        &(zero.clone() - &lambda[3]),
    )?;
    add_trace_word(
        ShaWordCol::OvSigma0,
        0,
        &plan.word_weights,
        &(zero.clone() - &(lambda[0].clone() * &plan.two)),
    )?;
    add_trace_word(
        ShaWordCol::OvSigma1,
        0,
        &plan.word_weights,
        &(zero.clone() - &(lambda[1].clone() * &plan.two)),
    )?;
    add_trace_word(
        ShaWordCol::OvSmallSigma0,
        0,
        &plan.word_weights,
        &(zero.clone() - &(lambda[2].clone() * &plan.two)),
    )?;
    add_trace_word(
        ShaWordCol::OvSmallSigma1,
        0,
        &plan.word_weights,
        &(zero.clone() - &(lambda[3].clone() * &plan.two)),
    )?;

    let w0_selector_coeff = lambda[11].clone() * &s_msg;
    if F::is_zero(&w0_selector_coeff) {
        add_trace_word(
            ShaWordCol::W,
            0,
            &plan.w0_base_weights,
            &F::one_with_cfg(field_cfg),
        )?;
    } else {
        let w0_weights: [F; SHA_WORD_BITS] = std::array::from_fn(|bit| {
            plan.w0_base_weights[bit].clone() + w0_selector_coeff.clone() * &plan.word_weights[bit]
        });
        add_trace_word(ShaWordCol::W, 0, &w0_weights, &F::one_with_cfg(field_cfg))?;
    }

    add_trace_word(ShaWordCol::W, 16, &plan.word_weights, &lambda[4])?;
    add_trace_word(
        ShaWordCol::SmallSigma0,
        1,
        &plan.word_weights,
        &(zero.clone() - &lambda[4]),
    )?;
    add_trace_word(
        ShaWordCol::W,
        9,
        &plan.word_weights,
        &(zero.clone() - &lambda[4]),
    )?;
    add_trace_word(
        ShaWordCol::SmallSigma1,
        14,
        &plan.word_weights,
        &(zero.clone() - &lambda[4]),
    )?;
    add_trace_word(
        ShaWordCol::A,
        4,
        &plan.word_weights,
        &(lambda[5].clone() + &lambda[9]),
    )?;
    add_trace_word(
        ShaWordCol::E,
        4,
        &plan.word_weights,
        &(lambda[6].clone() + &lambda[10]),
    )?;
    let neg_l5_l6 = zero.clone() - &(lambda[5].clone() + &lambda[6]);
    add_trace_word(ShaWordCol::Sigma1, 3, &plan.word_weights, &neg_l5_l6)?;
    add_trace_word(ShaWordCol::Uef, 3, &plan.word_weights, &neg_l5_l6)?;
    add_trace_word(ShaWordCol::UNegEg, 3, &plan.word_weights, &neg_l5_l6)?;
    add_trace_word(
        ShaWordCol::Sigma0,
        3,
        &plan.word_weights,
        &(zero.clone() - &lambda[5]),
    )?;
    add_trace_word(
        ShaWordCol::Maj,
        3,
        &plan.word_weights,
        &(zero.clone() - &lambda[5]),
    )?;
    add_trace_word(
        ShaWordCol::MuPacked,
        0,
        &plan.mu_packed_weights,
        &F::one_with_cfg(field_cfg),
    )?;

    let mut add_public_word = |col: ShaPublicCol, coeff: &F| -> Result<(), ShaProjectionError> {
        if F::is_zero(coeff) {
            return Ok(());
        }
        linear += public_word_or_const_eval_at_row_with_weights(
            public,
            col,
            row,
            &plan.word_weights,
            field_cfg,
            reducer,
        )? * coeff;
        Ok(())
    };
    add_public_word(
        ShaPublicCol::PAIn,
        &(zero.clone() - &(lambda[7].clone() * &s_init)),
    )?;
    add_public_word(
        ShaPublicCol::PAOut,
        &(zero.clone() - &(lambda[7].clone() * &s_out)),
    )?;
    add_public_word(
        ShaPublicCol::PEIn,
        &(zero.clone() - &(lambda[8].clone() * &s_init)),
    )?;
    add_public_word(
        ShaPublicCol::PEOut,
        &(zero.clone() - &(lambda[8].clone() * &s_out)),
    )?;
    add_public_word(
        ShaPublicCol::Message,
        &(zero.clone() - &(lambda[11].clone() * &s_msg)),
    )?;

    linear += public_scalar_shifted_fast(public, ShaPublicCol::K, row, 3, field_cfg) * &neg_l5_l6;
    linear += public_scalar_shifted_fast(public, ShaPublicCol::PAIn, row, 0, field_cfg)
        * &(zero.clone() - &lambda[9]);
    linear += public_scalar_shifted_fast(public, ShaPublicCol::PEIn, row, 0, field_cfg)
        * &(zero.clone() - &lambda[10]);

    linear += int_scalar_fast(trace, ShaIntCol::CompSchedule, row, field_cfg)
        * &(lambda[4].clone() + lambda[12].clone() * &s_sched);
    linear += int_scalar_fast(trace, ShaIntCol::CompUpdateA, row, field_cfg)
        * &(lambda[5].clone() + lambda[13].clone() * &s_upd);
    linear += int_scalar_fast(trace, ShaIntCol::CompUpdateE, row, field_cfg)
        * &(lambda[6].clone() + lambda[14].clone() * &s_upd);
    linear += int_scalar_fast(trace, ShaIntCol::CompFeedForwardA, row, field_cfg)
        * &(lambda[9].clone() + lambda[15].clone() * &s_ff);
    linear += int_scalar_fast(trace, ShaIntCol::CompFeedForwardE, row, field_cfg)
        * &(lambda[10].clone() + lambda[16].clone() * &s_ff);

    Ok(linear)
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
struct LinearClaimTable<F> {
    values: Vec<F>,
}

impl<F> LinearClaimTable<F>
where
    F: PrimeField,
{
    fn new(values: Vec<F>) -> Result<Self, ShaProjectionError> {
        if values.is_empty() || !values.len().is_power_of_two() {
            return Err(ShaProjectionError::InstanceCountNotPowerOfTwo { got: values.len() });
        }
        Ok(Self { values })
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn finite_bucket(
        &self,
        weights: &CollapsedSuffixEqWeights<F>,
        branch: usize,
        field_cfg: &F::Config,
    ) -> F {
        debug_assert!(branch < 2);
        debug_assert_eq!(weights.len(), self.values.len());
        debug_assert!(self.values.len() >= 2);
        let mut acc = F::zero_with_cfg(field_cfg);
        for rest in 0..(self.values.len() >> 1) {
            acc += weights.pair_weight(rest) * self.values[(rest << 1) + branch].clone();
        }
        acc
    }

    fn zero_bucket(&self, weights: &CollapsedSuffixEqWeights<F>, field_cfg: &F::Config) -> F {
        self.finite_bucket(weights, 0, field_cfg)
    }

    fn one_bucket(&self, weights: &CollapsedSuffixEqWeights<F>, field_cfg: &F::Config) -> F {
        self.finite_bucket(weights, 1, field_cfg)
    }

    fn fold_in_place(&mut self, r: &F) {
        fold_binary_claim_vector(&mut self.values, r);
    }
}

#[derive(Clone, Debug)]
struct BooleanityClaimTable<F> {
    values: Vec<F>,
    tail_len: usize,
    source_row_count: usize,
}

impl<F> BooleanityClaimTable<F>
where
    F: PrimeField,
{
    fn new(
        values: Vec<F>,
        tail_len: usize,
        source_row_count: usize,
    ) -> Result<Self, ShaProjectionError> {
        if tail_len == 0 || !tail_len.is_power_of_two() {
            return Err(ShaProjectionError::InstanceCountNotPowerOfTwo { got: tail_len });
        }
        let expected_len = tail_len.checked_mul(source_row_count).ok_or(
            ShaProjectionError::InstanceCountMismatch {
                got: values.len(),
                expected: usize::MAX,
            },
        )?;
        if values.len() != expected_len {
            return Err(ShaProjectionError::InstanceCountMismatch {
                got: values.len(),
                expected: expected_len,
            });
        }
        Ok(Self {
            values,
            tail_len,
            source_row_count,
        })
    }

    #[cfg(test)]
    fn from_source_major(
        source_values: &[Vec<F>],
        tail_len: usize,
    ) -> Result<Self, ShaProjectionError> {
        for values in source_values {
            if values.len() != tail_len {
                return Err(ShaProjectionError::InstanceCountMismatch {
                    got: values.len(),
                    expected: tail_len,
                });
            }
        }
        let source_row_count = source_values.len();
        let mut values = Vec::with_capacity(tail_len * source_row_count);
        for tail in 0..tail_len {
            for source_row_values in source_values {
                values.push(source_row_values[tail].clone());
            }
        }
        Self::new(values, tail_len, source_row_count)
    }

    fn tail_len(&self) -> usize {
        self.tail_len
    }

    fn source_row_count(&self) -> usize {
        self.source_row_count
    }

    fn values(&self) -> &[F] {
        &self.values
    }

    fn values_mut(&mut self) -> &mut [F] {
        &mut self.values
    }

    fn value(&self, tail: usize, source_row: usize) -> &F {
        &self.values[suffix_flat_index(tail, source_row, self.source_row_count)]
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn fold_in_place(&mut self, r: &F) {
        debug_assert!(self.tail_len.is_power_of_two());
        debug_assert!(self.tail_len >= 2);
        let half = self.tail_len >> 1;
        for tail in 0..half {
            let even_tail = tail << 1;
            let odd_tail = even_tail + 1;
            for source_row in 0..self.source_row_count {
                let even = self.values
                    [suffix_flat_index(even_tail, source_row, self.source_row_count)]
                .clone();
                let odd = self.values
                    [suffix_flat_index(odd_tail, source_row, self.source_row_count)]
                .clone();
                let folded = even.clone() + r.clone() * (odd - even);
                let out_idx = suffix_flat_index(tail, source_row, self.source_row_count);
                self.values[out_idx] = folded;
            }
        }
        self.truncate_tail_len(half);
    }

    fn truncate_tail_len(&mut self, tail_len: usize) {
        debug_assert!(tail_len <= self.tail_len);
        self.tail_len = tail_len;
        self.values.truncate(tail_len * self.source_row_count);
    }

    fn source_row_values(&self, source_row: usize) -> Vec<F> {
        (0..self.tail_len)
            .map(|tail| self.value(tail, source_row).clone())
            .collect()
    }
}

#[derive(Clone, Debug)]
struct CollapsedSuffixEqWeights<F> {
    values: Vec<F>,
}

impl<F> CollapsedSuffixEqWeights<F>
where
    F: PrimeField,
{
    fn new(beta_suffix: &[F], field_cfg: &F::Config) -> Result<Self, ShaProjectionError> {
        Self::from_values(eq_weights_or_one(beta_suffix, field_cfg)?)
    }

    fn from_values(values: Vec<F>) -> Result<Self, ShaProjectionError> {
        if values.is_empty() || !values.len().is_power_of_two() {
            return Err(ShaProjectionError::InstanceCountNotPowerOfTwo { got: values.len() });
        }
        Ok(Self { values })
    }

    fn len(&self) -> usize {
        self.values.len()
    }

    fn pair_count(&self) -> usize {
        debug_assert!(self.values.len() >= 2);
        self.values.len() >> 1
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn pair_weight(&self, rest: usize) -> F {
        debug_assert!(rest < self.pair_count());
        self.values[rest << 1].clone() + self.values[(rest << 1) + 1].clone()
    }

    fn collapse_current_axis(&mut self) {
        collapse_binary_weight_vector(&mut self.values);
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
    tail_traces: Option<Box<[ProjectedTrace<F>]>>,
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
{
    #[allow(clippy::too_many_arguments)]
    fn new<Trace, Public>(
        traces: &[Trace],
        publics: &[Public],
        beta: &[F],
        r_ic: &[F; SHA_ROW_VARS],
        a: &F,
        lambda: &F,
        rho: &F,
        xi: &F,
        booleanity_sources: &[ShaBooleanitySource],
        prefix_vars: usize,
        field_cfg: &F::Config,
    ) -> Result<Self, ShaProjectionError>
    where
        Trace: Borrow<ProjectedTrace<F>> + Sync,
        Public: Borrow<ProjectedPublic<F>> + Sync,
    {
        let coeff_tables = build_linear_residual_coeff_tables(traces, publics, r_ic, field_cfg)?;
        let row_weights = build_eq_x_r_vec(r_ic, field_cfg)?;
        Self::new_with_linear_cache(
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
    fn new_with_linear_cache<Trace, Public>(
        traces: &[Trace],
        publics: &[Public],
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
    ) -> Result<Self, ShaProjectionError>
    where
        Trace: Borrow<ProjectedTrace<F>> + Sync,
        Public: Borrow<ProjectedPublic<F>> + Sync,
    {
        let ell = validate_sha_sumfold_inputs(traces, publics, beta)?;
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
            traces,
            booleanity_sources,
            prefix_vars,
            tail_len,
            row_weights,
            &booleanity_weights,
            field_cfg,
        );
        Self::new_with_accumulators(
            traces,
            beta,
            &linear_values,
            &quadratic_values?,
            booleanity_sources,
            prefix_vars,
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

    fn new_with_accumulators<Trace>(
        traces: &[Trace],
        beta: &[F],
        linear_values: &[F],
        quadratic_values: &[F],
        booleanity_sources: &[ShaBooleanitySource],
        prefix_vars: usize,
        field_cfg: &F::Config,
    ) -> Result<Self, ShaProjectionError>
    where
        Trace: Borrow<ProjectedTrace<F>> + Sync,
    {
        let ell = validate_sha_sumfold_traces(traces, beta)?;
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

        let tail_traces = if tail_vars == 0 {
            None
        } else {
            Some(
                traces
                    .iter()
                    .map(|trace| trace.borrow().clone())
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            )
        };

        Ok(Self {
            tail_traces,
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

    fn new_owned_with_accumulators(
        traces: Box<[ProjectedTrace<F>]>,
        beta: &[F],
        linear_values: &[F],
        quadratic_values: &[F],
        booleanity_sources: &[ShaBooleanitySource],
        prefix_vars: usize,
        field_cfg: &F::Config,
    ) -> Result<Self, ShaProjectionError> {
        let mut fast_path = Self::new_with_accumulators(
            &traces,
            beta,
            linear_values,
            quadratic_values,
            booleanity_sources,
            prefix_vars,
            field_cfg,
        )?;
        if fast_path.beta.len() > fast_path.total_prefix_vars {
            fast_path.tail_traces = Some(traces);
        }
        Ok(fast_path)
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
    fn quadratic_reduced_buckets(
        &self,
        field_cfg: &F::Config,
    ) -> Result<(F, F), ShaProjectionError> {
        debug_assert!(self.round < self.total_prefix_vars);
        debug_assert!(self.booleanity.prefix_vars > 0);
        let suffix_weights = &self.prefix_suffix_eq_weights[self.round];
        let rest_len = suffix_weights.len();
        let prefix_len = checked_ternary_len(self.booleanity.prefix_vars)?;
        let mut at_zero = F::zero_with_cfg(field_cfg);
        let mut at_infinity = F::zero_with_cfg(field_cfg);

        for tail in 0..self.tail_eq_weights.len() {
            for (rest, suffix_weight) in suffix_weights.iter().enumerate().take(rest_len) {
                let ternary_rest =
                    binary_bits_to_ternary_index(rest, self.booleanity.prefix_vars - 1);
                let base = tail * prefix_len + ternary_rest * 3;
                let weight = self.tail_eq_weights[tail].clone() * suffix_weight;
                at_zero += weight.clone() * self.booleanity.values[base].clone();
                at_infinity += weight * self.booleanity.values[base + 2].clone();
            }
        }

        Ok((at_zero, at_infinity))
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn quadratic_one_bucket(&self, field_cfg: &F::Config) -> Result<F, ShaProjectionError> {
        debug_assert!(self.round < self.total_prefix_vars);
        debug_assert!(self.booleanity.prefix_vars > 0);
        let suffix_weights = &self.prefix_suffix_eq_weights[self.round];
        let rest_len = suffix_weights.len();
        let prefix_len = checked_ternary_len(self.booleanity.prefix_vars)?;
        let mut at_one = F::zero_with_cfg(field_cfg);

        for tail in 0..self.tail_eq_weights.len() {
            for (rest, suffix_weight) in suffix_weights.iter().enumerate().take(rest_len) {
                let ternary_rest =
                    binary_bits_to_ternary_index(rest, self.booleanity.prefix_vars - 1);
                let base = tail * prefix_len + ternary_rest * 3;
                let weight = self.tail_eq_weights[tail].clone() * suffix_weight;
                at_one += weight * self.booleanity.values[base + 1].clone();
            }
        }

        Ok(at_one)
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

        let traces = self
            .tail_traces
            .as_deref()
            .expect("tail traces must be present when tail variables remain");
        let source_tail_values = bind_sha_booleanity_sources_to_prefix(
            traces,
            &self.booleanity_sources,
            self.total_prefix_vars,
            tail_len,
            &prefix_weights,
            field_cfg,
        )?;

        for source_row in 0..source_tail_values.source_row_count() {
            let values = source_tail_values.source_row_values(source_row);
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

struct RelationSumFoldAllRoundsFastPath<F: PrimeField> {
    prefix: RelationSumFoldPrefixFastPath<F>,
    total_vars: usize,
    prefix_vars: usize,
    live_linear: Option<LinearClaimTable<F>>,
    live_eq_weights: Option<CollapsedSuffixEqWeights<F>>,
    row_weights: Vec<F>,
    booleanity_weights: Vec<F>,
    prefix_challenges: Vec<F>,
    current_claim: Option<F>,
    last_round_evaluations: Option<[F; 4]>,
    suffix: Option<RelationSumFoldSuffixState<F>>,
}

impl<F> RelationSumFoldAllRoundsFastPath<F>
where
    F: InnerTransparentField
        + DelayedFieldProductSum
        + ShaSuffixScannerField
        + Send
        + Sync
        + 'static,
{
    fn new(
        prefix: RelationSumFoldPrefixFastPath<F>,
        row_weights: Vec<F>,
        booleanity_weights: Vec<F>,
    ) -> Result<Self, ShaProjectionError> {
        if row_weights.len() != SHA_ROW_COUNT {
            return Err(ShaProjectionError::ColumnRowCount {
                kind: "row_weights",
                col: 0,
                got: row_weights.len(),
                expected: SHA_ROW_COUNT,
            });
        }
        if booleanity_weights.len() != prefix.booleanity_sources.len() {
            return Err(ShaProjectionError::ColumnRowCount {
                kind: "booleanity_weights",
                col: 0,
                got: booleanity_weights.len(),
                expected: prefix.booleanity_sources.len(),
            });
        }

        let total_vars = prefix.beta.len();
        let prefix_vars = prefix.total_prefix_vars;
        Ok(Self {
            prefix,
            total_vars,
            prefix_vars,
            live_linear: None,
            live_eq_weights: None,
            row_weights,
            booleanity_weights,
            prefix_challenges: Vec::with_capacity(prefix_vars),
            current_claim: None,
            last_round_evaluations: None,
            suffix: None,
        })
    }

    fn new_with_initial_claim(
        prefix: RelationSumFoldPrefixFastPath<F>,
        row_weights: Vec<F>,
        booleanity_weights: Vec<F>,
        initial_claim: F,
        field_cfg: &F::Config,
    ) -> Result<Self, ShaProjectionError> {
        let live_eq_weights = CollapsedSuffixEqWeights::new(&prefix.beta, field_cfg)?;
        let live_linear = LinearClaimTable::new(prefix.linear.values.clone())?;
        let mut out = Self::new(prefix, row_weights, booleanity_weights)?;
        out.current_claim = Some(initial_claim);
        out.live_linear = Some(live_linear);
        out.live_eq_weights = Some(live_eq_weights);
        Ok(out)
    }

    fn absorb_previous_challenge(
        &mut self,
        r: &F,
        field_cfg: &F::Config,
    ) -> Result<(), ShaProjectionError> {
        let current_claim = self.evaluate_last_round_at(r, field_cfg);
        self.current_claim = Some(current_claim.clone());

        if let Some(linear) = self.live_linear.as_mut() {
            linear.fold_in_place(r);
            self.live_eq_weights
                .as_mut()
                .expect("live linear mode should carry active equality weights")
                .collapse_current_axis();
        }

        if self.prefix.round <= self.prefix_vars && self.prefix.booleanity.prefix_vars > 0 {
            let beta_idx = self.prefix.round - 1;
            self.prefix.eq_bound *= eq_one_var(&self.prefix.beta[beta_idx], r, field_cfg);
            if self.live_linear.is_none() {
                self.prefix.linear.bind_first_axis(r, field_cfg);
            }
            self.prefix.booleanity.bind_first_axis(r, field_cfg)?;
            self.prefix_challenges.push(r.clone());
            if self.prefix.booleanity.prefix_vars == 0 && self.prefix_vars < self.total_vars {
                self.initialize_suffix(field_cfg)?;
            }
        } else if let Some(suffix) = self.suffix.as_mut() {
            suffix.bind_previous_challenge(r, &current_claim, field_cfg);
        }

        Ok(())
    }

    fn evaluate_last_round_at(&self, r: &F, field_cfg: &F::Config) -> F {
        let evaluations = self
            .last_round_evaluations
            .as_ref()
            .expect("previous SHA SumFold round must be available before verifier challenge");
        eval_cubic_from_zero_one_two_three(evaluations, r, field_cfg)
    }

    fn initialize_suffix(&mut self, field_cfg: &F::Config) -> Result<(), ShaProjectionError> {
        debug_assert_eq!(self.prefix_challenges.len(), self.prefix_vars);
        let tail_vars = self.total_vars - self.prefix_vars;
        if tail_vars == 0 {
            return Ok(());
        }

        let tail_len = binary_len(tail_vars);
        let linear_claims = self
            .live_linear
            .as_ref()
            .map(|linear| linear.values.clone())
            .unwrap_or_else(|| self.prefix.linear.values.clone());
        debug_assert_eq!(linear_claims.len(), tail_len);
        let prefix_weights = eq_weights_or_one(&self.prefix_challenges, field_cfg)?;
        let traces = self
            .prefix
            .tail_traces
            .as_deref()
            .expect("tail traces must be present for all-round SHA suffix fast path");
        let booleanity_claims = bind_sha_booleanity_sources_to_prefix(
            traces,
            &self.prefix.booleanity_sources,
            self.prefix_vars,
            tail_len,
            &prefix_weights,
            field_cfg,
        )?;

        self.suffix = Some(RelationSumFoldSuffixState::new(
            self.prefix.beta.clone(),
            self.prefix_vars,
            self.prefix.eq_bound.clone(),
            LinearClaimTable::new(linear_claims)?,
            booleanity_claims,
            self.row_weights.clone(),
            self.booleanity_weights.clone(),
            field_cfg,
        )?);
        self.live_linear = None;
        self.live_eq_weights = None;
        Ok(())
    }

    fn prove_current_prefix_round(&mut self, field_cfg: &F::Config) -> PrefixRoundOutput<F> {
        if self.live_linear.is_some() {
            return self.prove_current_prefix_round_with_live_linear(field_cfg);
        }

        let zero = F::zero_with_cfg(field_cfg);
        let one = F::one_with_cfg(field_cfg);
        let two = one.clone() + &one;
        let three = two.clone() + &one;

        let p0 = self
            .prefix
            .round_value_at(&zero, field_cfg)
            .expect("validated SHA prefix table should evaluate at 0");
        let p1 = self
            .prefix
            .round_value_at(&one, field_cfg)
            .expect("validated SHA prefix table should evaluate at 1");
        let p2 = self
            .prefix
            .round_value_at(&two, field_cfg)
            .expect("validated SHA prefix table should evaluate at 2");
        let p3 = self
            .prefix
            .round_value_at(&three, field_cfg)
            .expect("validated SHA prefix table should evaluate at 3");

        let asserted_sum = if self.prefix.round == 0 {
            Some(p0.clone() + &p1)
        } else {
            None
        };
        self.last_round_evaluations = Some([p0, p1.clone(), p2.clone(), p3.clone()]);
        self.prefix.round += 1;

        PrefixRoundOutput {
            asserted_sum,
            tail_evaluations: vec![p1, p2, p3],
        }
    }

    fn prove_current_prefix_round_with_live_linear(
        &mut self,
        field_cfg: &F::Config,
    ) -> PrefixRoundOutput<F> {
        let linear = self
            .live_linear
            .as_ref()
            .expect("live linear mode should own a linear claim table");
        let live_eq_weights = self
            .live_eq_weights
            .as_ref()
            .expect("live linear mode should carry active equality weights");
        let linear_zero = linear.zero_bucket(live_eq_weights, field_cfg);
        let (quadratic_zero, quadratic_infinity) = self
            .prefix
            .quadratic_reduced_buckets(field_cfg)
            .expect("validated SHA prefix table should produce reduced buckets");
        let t_zero = linear_zero + quadratic_zero;
        let beta_idx = self.prefix.round;
        let zero = F::zero_with_cfg(field_cfg);
        let one = F::one_with_cfg(field_cfg);
        let two = one.clone() + &one;
        let three = two.clone() + &one;
        let e_zero = self.prefix.eq_bound.clone()
            * eq_one_var(&self.prefix.beta[beta_idx], &zero, field_cfg);
        let e_one =
            self.prefix.eq_bound.clone() * eq_one_var(&self.prefix.beta[beta_idx], &one, field_cfg);
        let current_claim = self
            .current_claim
            .as_ref()
            .expect("claim-aware SHA fast path should carry a current claim");
        let t_one = if F::is_zero(&e_one) {
            linear.one_bucket(live_eq_weights, field_cfg)
                + self
                    .prefix
                    .quadratic_one_bucket(field_cfg)
                    .expect("validated SHA prefix table should produce a one bucket")
        } else {
            (current_claim.clone() - e_zero.clone() * t_zero.clone()) / e_one
        };

        let eval_at = |x: &F| {
            let eq = self.prefix.eq_bound.clone()
                * eq_one_var(&self.prefix.beta[beta_idx], x, field_cfg);
            eq * eval_quadratic_from_zero_one_infinity(
                &t_zero,
                &t_one,
                &quadratic_infinity,
                x,
                field_cfg,
            )
        };

        let p0 = e_zero * t_zero.clone();
        let p1 = eval_at(&one);
        let p2 = eval_at(&two);
        let p3 = eval_at(&three);
        let asserted_sum = if self.prefix.round == 0 {
            Some(current_claim.clone())
        } else {
            None
        };
        self.last_round_evaluations = Some([p0, p1.clone(), p2.clone(), p3.clone()]);
        self.prefix.round += 1;

        PrefixRoundOutput {
            asserted_sum,
            tail_evaluations: vec![p1, p2, p3],
        }
    }

    fn prove_current_suffix_round(&mut self, field_cfg: &F::Config) -> PrefixRoundOutput<F> {
        let current_claim = self
            .current_claim
            .as_ref()
            .expect("SHA suffix SumFold claim should be fixed by previous challenge")
            .clone();
        let suffix = self
            .suffix
            .as_mut()
            .expect("SHA suffix fast path should be initialized after L0");
        let evaluations = suffix.prove_round(&current_claim, field_cfg);
        let [_p0, p1, p2, p3] = evaluations.clone();
        self.last_round_evaluations = Some(evaluations);
        PrefixRoundOutput {
            asserted_sum: None,
            tail_evaluations: vec![p1, p2, p3],
        }
    }
}

impl<F> PrefixFastPath<F> for RelationSumFoldAllRoundsFastPath<F>
where
    F: InnerTransparentField
        + DelayedFieldProductSum
        + ShaSuffixScannerField
        + Send
        + Sync
        + 'static,
{
    fn prefix_len(&self) -> usize {
        self.total_vars
    }

    fn requires_full_prefix(&self) -> bool {
        true
    }

    fn prove_prefix_round(
        &mut self,
        verifier_msg: &Option<F>,
        config: &F::Config,
    ) -> PrefixRoundOutput<F> {
        if let Some(r) = verifier_msg {
            self.absorb_previous_challenge(r, config)
                .expect("validated SHA all-round fast path should bind previous challenge");
        }

        if self.prefix.round < self.prefix_vars {
            self.prove_current_prefix_round(config)
        } else {
            self.prove_current_suffix_round(config)
        }
    }

    fn finish_prefix(
        self: Box<Self>,
        _prefix_challenges: &[F],
        _config: &F::Config,
    ) -> Vec<DenseMultilinearExtension<F::Inner>> {
        Vec::new()
    }
}

#[derive(Clone, Debug)]
struct SuffixRoundBuckets<F> {
    t_zero: F,
    t_infinity: F,
    t_one_direct: Option<F>,
}

struct RelationSumFoldSuffixState<F: PrimeField> {
    beta: Vec<F>,
    suffix_start: usize,
    round: usize,
    alpha: F,
    linear_claims: LinearClaimTable<F>,
    booleanity_claims: BooleanityClaimTable<F>,
    source_row_weights: Vec<F>,
    suffix_eq_weights: CollapsedSuffixEqWeights<F>,
    prepared_round: Option<SuffixRoundBuckets<F>>,
}

impl<F> RelationSumFoldSuffixState<F>
where
    F: ShaSuffixScannerField,
{
    #[allow(clippy::too_many_arguments)]
    fn new(
        beta: Vec<F>,
        suffix_start: usize,
        alpha: F,
        linear_claims: LinearClaimTable<F>,
        booleanity_claims: BooleanityClaimTable<F>,
        row_weights: Vec<F>,
        booleanity_weights: Vec<F>,
        field_cfg: &F::Config,
    ) -> Result<Self, ShaProjectionError> {
        let suffix_vars = beta.len() - suffix_start;
        let suffix_len = binary_len(suffix_vars);
        if linear_claims.values.len() != suffix_len {
            return Err(ShaProjectionError::InstanceCountMismatch {
                got: linear_claims.values.len(),
                expected: suffix_len,
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
        let expected_booleanity_claims = booleanity_weights.len() * SHA_ROW_COUNT;
        if booleanity_claims.source_row_count() != expected_booleanity_claims {
            return Err(ShaProjectionError::ColumnRowCount {
                kind: "booleanity_claims",
                col: 0,
                got: booleanity_claims.source_row_count(),
                expected: expected_booleanity_claims,
            });
        }
        if booleanity_claims.tail_len() != suffix_len {
            return Err(ShaProjectionError::InstanceCountMismatch {
                got: booleanity_claims.tail_len(),
                expected: suffix_len,
            });
        }

        let suffix_eq_weights = CollapsedSuffixEqWeights::new(&beta[suffix_start..], field_cfg)?;
        let mut source_row_weights = Vec::with_capacity(expected_booleanity_claims);
        for booleanity_weight in &booleanity_weights {
            for row_weight in &row_weights {
                source_row_weights.push(row_weight.clone() * booleanity_weight);
            }
        }

        Ok(Self {
            beta,
            suffix_start,
            round: 0,
            alpha,
            linear_claims,
            booleanity_claims,
            source_row_weights,
            suffix_eq_weights,
            prepared_round: None,
        })
    }

    fn prove_round(&mut self, current_claim: &F, field_cfg: &F::Config) -> [F; 4] {
        debug_assert!(self.round < self.suffix_vars());
        self.debug_assert_live_lengths();
        let need_t_one = self.current_round_needs_direct_one(field_cfg);
        let buckets = self
            .prepared_round
            .take()
            .unwrap_or_else(|| self.scan_round_buckets(need_t_one, field_cfg));
        let evaluations = self.round_evaluations_from_buckets(current_claim, &buckets, field_cfg);
        self.round += 1;
        evaluations
    }

    fn round_evaluations_from_buckets(
        &self,
        current_claim: &F,
        buckets: &SuffixRoundBuckets<F>,
        field_cfg: &F::Config,
    ) -> [F; 4] {
        let beta_idx = self.suffix_start + self.round;
        let zero = F::zero_with_cfg(field_cfg);
        let one = F::one_with_cfg(field_cfg);
        let two = one.clone() + &one;
        let three = two.clone() + &one;
        let e_zero = self.alpha.clone() * eq_one_var(&self.beta[beta_idx], &zero, field_cfg);
        let e_one = self.alpha.clone() * eq_one_var(&self.beta[beta_idx], &one, field_cfg);
        let t_zero = &buckets.t_zero;
        let t_infinity = &buckets.t_infinity;
        let t_one = if F::is_zero(&e_one) {
            buckets
                .t_one_direct
                .clone()
                .expect("zero e1 suffix round should carry a direct T1 bucket")
        } else {
            (current_claim.clone() - e_zero.clone() * buckets.t_zero.clone()) / e_one
        };

        let eval_at = |x: &F| {
            let eq = self.alpha.clone() * eq_one_var(&self.beta[beta_idx], x, field_cfg);
            eq * eval_quadratic_from_zero_one_infinity(t_zero, &t_one, t_infinity, x, field_cfg)
        };

        let p0 = e_zero * t_zero.clone();
        let p1 = eval_at(&one);
        let p2 = eval_at(&two);
        let p3 = eval_at(&three);
        [p0, p1, p2, p3]
    }

    fn bind_previous_challenge(&mut self, r: &F, _current_claim: &F, field_cfg: &F::Config) {
        debug_assert!(self.round > 0);
        debug_assert!(self.prepared_round.is_none());
        let beta_idx = self.suffix_start + self.round - 1;
        self.alpha *= eq_one_var(&self.beta[beta_idx], r, field_cfg);

        if self.round < self.suffix_vars() {
            let need_t_one = self.current_round_needs_direct_one(field_cfg);
            let source_row_count = self.booleanity_claims.source_row_count();
            let (t_zero, t_infinity, t_one_direct) = F::suffix_fold_prepare_next_round_flat(
                &mut self.linear_claims.values,
                self.booleanity_claims.values_mut(),
                source_row_count,
                &self.source_row_weights,
                &self.suffix_eq_weights.values,
                r,
                need_t_one,
                field_cfg,
            );
            let new_len = self.linear_claims.values.len() >> 1;
            self.linear_claims.values.truncate(new_len);
            self.booleanity_claims.truncate_tail_len(new_len);
            self.suffix_eq_weights.collapse_current_axis();
            self.prepared_round = Some(SuffixRoundBuckets {
                t_zero,
                t_infinity,
                t_one_direct,
            });
        } else {
            self.linear_claims.fold_in_place(r);
            self.booleanity_claims.fold_in_place(r);
            self.suffix_eq_weights.collapse_current_axis();
            self.prepared_round = None;
        }
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn reduced_body_buckets(&self, field_cfg: &F::Config) -> (F, F) {
        self.debug_assert_live_lengths();
        F::suffix_reduced_body_buckets_flat(
            &self.linear_claims.values,
            self.booleanity_claims.values(),
            self.booleanity_claims.source_row_count(),
            &self.source_row_weights,
            &self.suffix_eq_weights.values,
            field_cfg,
        )
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn direct_one_body_bucket(&self, field_cfg: &F::Config) -> F {
        self.debug_assert_live_lengths();
        F::suffix_direct_one_body_bucket_flat(
            &self.linear_claims.values,
            self.booleanity_claims.values(),
            self.booleanity_claims.source_row_count(),
            &self.source_row_weights,
            &self.suffix_eq_weights.values,
            field_cfg,
        )
    }

    fn scan_round_buckets(&self, need_t_one: bool, field_cfg: &F::Config) -> SuffixRoundBuckets<F> {
        let (t_zero, t_infinity) = self.reduced_body_buckets(field_cfg);
        let t_one_direct = need_t_one.then(|| self.direct_one_body_bucket(field_cfg));
        SuffixRoundBuckets {
            t_zero,
            t_infinity,
            t_one_direct,
        }
    }

    fn current_round_needs_direct_one(&self, field_cfg: &F::Config) -> bool {
        let beta_idx = self.suffix_start + self.round;
        let one = F::one_with_cfg(field_cfg);
        let e_one = self.alpha.clone() * eq_one_var(&self.beta[beta_idx], &one, field_cfg);
        F::is_zero(&e_one)
    }

    fn suffix_vars(&self) -> usize {
        self.beta.len() - self.suffix_start
    }

    fn debug_assert_live_lengths(&self) {
        debug_assert_eq!(
            self.linear_claims.values.len(),
            self.suffix_eq_weights.len()
        );
        debug_assert_eq!(
            self.booleanity_claims.tail_len(),
            self.suffix_eq_weights.len()
        );
        debug_assert_eq!(
            self.booleanity_claims.values().len(),
            self.suffix_eq_weights.len() * self.booleanity_claims.source_row_count()
        );
    }
}

fn eval_quadratic_from_zero_one_infinity<F>(
    at_zero: &F,
    at_one: &F,
    at_infinity: &F,
    x: &F,
    _field_cfg: &F::Config,
) -> F
where
    F: PrimeField,
{
    let linear_coeff = at_one.clone() - at_zero - at_infinity;
    at_zero.clone() + linear_coeff * x + at_infinity.clone() * x.clone() * x
}

#[allow(clippy::arithmetic_side_effects)]
fn eval_cubic_from_zero_one_two_three<F>(evaluations: &[F; 4], x: &F, field_cfg: &F::Config) -> F
where
    F: PrimeField,
{
    let one = F::one_with_cfg(field_cfg);
    let two = one.clone() + &one;
    let three = two.clone() + &one;
    let six = three.clone() * &two;
    let first = evaluations[1].clone() - &evaluations[0];
    let second = evaluations[2].clone() - evaluations[1].clone() * &two + &evaluations[0];
    let third = evaluations[3].clone() - evaluations[2].clone() * &three
        + evaluations[1].clone() * &three
        - &evaluations[0];
    evaluations[0].clone()
        + x.clone() * first
        + x.clone() * (x.clone() - one.clone()) * second / two
        + x.clone() * (x.clone() - one) * (x.clone() - three + F::one_with_cfg(field_cfg)) * third
            / six
}

#[allow(clippy::arithmetic_side_effects)]
fn fold_binary_claim_vector<F>(values: &mut Vec<F>, r: &F)
where
    F: PrimeField,
{
    debug_assert!(values.len().is_power_of_two());
    debug_assert!(values.len() >= 2);
    let half = values.len() >> 1;
    for idx in 0..half {
        let even = values[idx << 1].clone();
        let odd = values[(idx << 1) + 1].clone();
        values[idx] = even.clone() + r.clone() * (odd - even);
    }
    values.truncate(half);
}

#[allow(clippy::arithmetic_side_effects)]
fn collapse_binary_weight_vector<F>(values: &mut Vec<F>)
where
    F: PrimeField,
{
    debug_assert!(values.len().is_power_of_two());
    debug_assert!(values.len() >= 2);
    let half = values.len() >> 1;
    for idx in 0..half {
        values[idx] = values[idx << 1].clone() + values[(idx << 1) + 1].clone();
    }
    values.truncate(half);
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
pub fn build_dense_sha_sumfold_group<F, Trace, Public>(
    traces: &[Trace],
    publics: &[Public],
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
    Trace: Borrow<ProjectedTrace<F>> + Sync,
    Public: Borrow<ProjectedPublic<F>> + Sync,
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
pub fn build_dense_sha_sumfold_group_with_weights<F, Trace, Public>(
    traces: &[Trace],
    publics: &[Public],
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
    Trace: Borrow<ProjectedTrace<F>> + Sync,
    Public: Borrow<ProjectedPublic<F>> + Sync,
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
                trace.borrow(),
                public.borrow(),
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
fn build_dense_sha_sumfold_group_from_accumulators<F, Trace>(
    traces: &[Trace],
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
    Trace: Borrow<ProjectedTrace<F>> + Sync,
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

    let need_virtuals = sources_need_virtuals(booleanity_sources);
    let mut source_values =
        vec![Vec::with_capacity(traces.len()); booleanity_sources.len() * SHA_ROW_COUNT];
    for row in 0..SHA_ROW_COUNT {
        let virtuals_by_trace = if need_virtuals {
            traces
                .iter()
                .map(|trace| reconstruct_virtual_ch_maj_at_row(trace.borrow(), row, field_cfg))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            Vec::new()
        };
        for (trace_idx, trace) in traces.iter().enumerate() {
            let virtuals = need_virtuals.then(|| &virtuals_by_trace[trace_idx]);
            for (source_idx, source) in booleanity_sources.iter().enumerate() {
                source_values[source_idx * SHA_ROW_COUNT + row].push(
                    booleanity_source_value_at_row_with_virtuals(
                        trace.borrow(),
                        row,
                        source,
                        virtuals,
                        field_cfg,
                    )?,
                );
            }
        }
    }
    for values in source_values {
        mles.push(DenseMultilinearExtension::from_evaluations_vec(
            ell,
            values.iter().map(|value| value.inner().clone()).collect(),
            zero_inner.clone(),
        ));
    }

    Ok(MultiDegreeSumcheckGroup::new(
        3,
        mles,
        sha_weighted_sumfold_comb_fn(row_weights.to_vec(), booleanity_weights.to_vec(), field_cfg),
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn build_production_sha_sumfold_group_from_prefix_accumulators<F, Trace>(
    traces: &[Trace],
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
    F: InnerTransparentField
        + DelayedFieldProductSum
        + ShaSuffixScannerField
        + Send
        + Sync
        + 'static,
    Trace: Borrow<ProjectedTrace<F>> + Sync,
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

    let prefix_fast_path = RelationSumFoldPrefixFastPath::new_with_accumulators(
        traces,
        beta,
        linear_accumulator,
        quadratic_prefix_accumulator,
        booleanity_sources,
        prefix_vars,
        field_cfg,
    )?;
    let fast_path = RelationSumFoldAllRoundsFastPath::new(
        prefix_fast_path,
        row_weights.to_vec(),
        booleanity_weights.to_vec(),
    )?;

    Ok(MultiDegreeSumcheckGroup::with_prefix_fast(
        3,
        Vec::new(),
        sha_weighted_sumfold_comb_fn(row_weights.to_vec(), booleanity_weights.to_vec(), field_cfg),
        Box::new(fast_path),
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn build_production_sha_sumfold_group_from_prefix_accumulators_with_initial_claim<F, Trace>(
    traces: &[Trace],
    beta: &[F],
    beta_eq_weights: &[F],
    row_weights: &[F],
    linear_accumulator: &[F],
    quadratic_prefix_accumulator: &[F],
    booleanity_weights: &[F],
    booleanity_sources: &[ShaBooleanitySource],
    prefix_vars: usize,
    initial_claim: &F,
    field_cfg: &F::Config,
) -> Result<MultiDegreeSumcheckGroup<F>, ShaProjectionError>
where
    F: InnerTransparentField
        + DelayedFieldProductSum
        + ShaSuffixScannerField
        + Send
        + Sync
        + 'static,
    Trace: Borrow<ProjectedTrace<F>> + Sync,
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

    let prefix_fast_path = RelationSumFoldPrefixFastPath::new_with_accumulators(
        traces,
        beta,
        linear_accumulator,
        quadratic_prefix_accumulator,
        booleanity_sources,
        prefix_vars,
        field_cfg,
    )?;
    let fast_path = RelationSumFoldAllRoundsFastPath::new_with_initial_claim(
        prefix_fast_path,
        row_weights.to_vec(),
        booleanity_weights.to_vec(),
        initial_claim.clone(),
        field_cfg,
    )?;

    Ok(MultiDegreeSumcheckGroup::with_prefix_fast(
        3,
        Vec::new(),
        sha_weighted_sumfold_comb_fn(row_weights.to_vec(), booleanity_weights.to_vec(), field_cfg),
        Box::new(fast_path),
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn build_production_sha_sumfold_group<F, Trace, Public>(
    traces: &[Trace],
    publics: &[Public],
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
    F: InnerTransparentField
        + DelayedFieldProductSum
        + ShaSuffixScannerField
        + Send
        + Sync
        + 'static,
    Trace: Borrow<ProjectedTrace<F>> + Sync,
    Public: Borrow<ProjectedPublic<F>> + Sync,
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

    let row_weights = build_eq_x_r_vec(r_ic, field_cfg)?;
    let booleanity_weights = build_booleanity_weights(rho, xi, booleanity_sources.len(), field_cfg);
    let prefix_fast_path = RelationSumFoldPrefixFastPath::new(
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
    )?;
    let fast_path = RelationSumFoldAllRoundsFastPath::new(
        prefix_fast_path,
        row_weights.clone(),
        booleanity_weights.clone(),
    )?;

    Ok(MultiDegreeSumcheckGroup::with_prefix_fast(
        3,
        Vec::new(),
        sha_weighted_sumfold_comb_fn(row_weights, booleanity_weights, field_cfg),
        Box::new(fast_path),
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn build_production_sha_sumfold_group_with_linear_cache<F, Trace, Public>(
    traces: &[Trace],
    publics: &[Public],
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
    F: InnerTransparentField
        + DelayedFieldProductSum
        + ShaSuffixScannerField
        + Send
        + Sync
        + 'static,
    Trace: Borrow<ProjectedTrace<F>> + Sync,
    Public: Borrow<ProjectedPublic<F>> + Sync,
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
pub fn build_production_sha_sumfold_group_with_linear_cache_and_weights<F, Trace, Public>(
    traces: &[Trace],
    publics: &[Public],
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
    F: InnerTransparentField
        + DelayedFieldProductSum
        + ShaSuffixScannerField
        + Send
        + Sync
        + 'static,
    Trace: Borrow<ProjectedTrace<F>> + Sync,
    Public: Borrow<ProjectedPublic<F>> + Sync,
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

    let booleanity_weights = build_booleanity_weights(rho, xi, booleanity_sources.len(), field_cfg);
    let prefix_fast_path = RelationSumFoldPrefixFastPath::new_with_linear_cache(
        traces,
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
    )?;
    let fast_path = RelationSumFoldAllRoundsFastPath::new(
        prefix_fast_path,
        row_weights.to_vec(),
        booleanity_weights.clone(),
    )?;

    Ok(MultiDegreeSumcheckGroup::with_prefix_fast(
        3,
        Vec::new(),
        sha_weighted_sumfold_comb_fn(row_weights.to_vec(), booleanity_weights, field_cfg),
        Box::new(fast_path),
    ))
}

#[allow(clippy::too_many_arguments)]
fn build_dense_sha_sumfold_group_with_linear_cache_and_weights<F, Trace>(
    traces: &[Trace],
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
    Trace: Borrow<ProjectedTrace<F>> + Sync,
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
    F: InnerTransparentField
        + DelayedFieldProductSum
        + ShaSuffixScannerField
        + Send
        + Sync
        + 'static,
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

    let row_weights = build_eq_x_r_vec(r_ic, field_cfg)?;
    let booleanity_weights = build_booleanity_weights(rho, xi, booleanity_sources.len(), field_cfg);
    let prefix_fast_path = RelationSumFoldPrefixFastPath::new_owned(
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
    )?;
    let fast_path = RelationSumFoldAllRoundsFastPath::new(
        prefix_fast_path,
        row_weights.clone(),
        booleanity_weights.clone(),
    )?;

    Ok(MultiDegreeSumcheckGroup::with_prefix_fast(
        3,
        Vec::new(),
        sha_weighted_sumfold_comb_fn(row_weights, booleanity_weights, field_cfg),
        Box::new(fast_path),
    ))
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

fn validate_sha_sumfold_inputs<F, Trace, Public>(
    traces: &[Trace],
    publics: &[Public],
    beta: &[F],
) -> Result<usize, ShaProjectionError>
where
    F: PrimeField,
    Trace: Borrow<ProjectedTrace<F>>,
    Public: Borrow<ProjectedPublic<F>>,
{
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
        validate_trace(trace.borrow())?;
    }
    for public in publics {
        validate_public(public.borrow())?;
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

fn validate_sha_sumfold_traces<F, Trace>(
    traces: &[Trace],
    beta: &[F],
) -> Result<usize, ShaProjectionError>
where
    F: PrimeField,
    Trace: Borrow<ProjectedTrace<F>>,
{
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
    #[cfg(debug_assertions)]
    {
        for trace in traces {
            validate_trace(trace.borrow())?;
        }
    }
    Ok(ell)
}

#[allow(clippy::too_many_arguments)]
pub fn build_sha_sumfold_quadratic_prefix_accumulator<F, Trace>(
    traces: &[Trace],
    booleanity_sources: &[ShaBooleanitySource],
    prefix_vars: usize,
    row_weights: &[F],
    booleanity_weights: &[F],
    field_cfg: &F::Config,
) -> Result<Vec<F>, ShaProjectionError>
where
    F: PrimeField,
    Trace: Borrow<ProjectedTrace<F>> + Sync,
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
    #[cfg(debug_assertions)]
    {
        for trace in traces {
            validate_trace(trace.borrow())?;
        }
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
fn build_sha_booleanity_prefix_tail_table<F, Trace>(
    traces: &[Trace],
    booleanity_sources: &[ShaBooleanitySource],
    prefix_vars: usize,
    tail_len: usize,
    row_weights: &[F],
    booleanity_weights: &[F],
    field_cfg: &F::Config,
) -> Result<Vec<F>, ShaProjectionError>
where
    F: PrimeField,
    Trace: Borrow<ProjectedTrace<F>> + Sync,
{
    let prefix_len = binary_len(prefix_vars);
    let ternary_len = checked_ternary_len(prefix_vars)?;
    let mut table = vec![F::zero_with_cfg(field_cfg); ternary_len * tail_len];
    if booleanity_sources.is_empty() {
        return Ok(table);
    }
    if is_canonical_production_booleanity_sources(booleanity_sources) {
        return build_sha_booleanity_prefix_tail_table_production_fast(
            traces,
            prefix_vars,
            tail_len,
            row_weights,
            booleanity_weights,
            field_cfg,
        );
    }

    let coeff_plans = ternary_coeff_plans(prefix_vars)?;
    let word_bit_source_count = booleanity_sources
        .iter()
        .take_while(|source| matches!(source, ShaBooleanitySource::WordBit { .. }))
        .count();
    let suffix_sources = &booleanity_sources[word_bit_source_count..];
    let suffix_count = suffix_sources.len();
    let suffix_needs_virtuals = sources_need_virtuals(suffix_sources);
    let small_square_fields: Vec<F> = small_square_field_table(field_cfg);
    let mask_count = 1usize << prefix_len;
    let mut mask_coeff_table = Vec::with_capacity(mask_count * ternary_len);
    for mask in 0..mask_count {
        let source_mask = u8::try_from(mask).map_err(|_| {
            ShaProjectionError::NonCanonicalProofObject(
                "booleanity prefix masks require at most eight prefix entries",
            )
        })?;
        for plan in &coeff_plans {
            mask_coeff_table.push(booleanity_word_bit_mask_degree_two_coeff(
                source_mask,
                plan,
                &small_square_fields,
                field_cfg,
            ));
        }
    }
    let one = F::one_with_cfg(field_cfg);
    let partials = cfg_chunks!(row_weights, 8)
        .enumerate()
        .map(|(chunk_idx, row_weight_chunk)| {
            let row_offset = chunk_idx * 8;
            let mut partial = vec![F::zero_with_cfg(field_cfg); ternary_len * tail_len];
            let mut suffix_values = vec![F::zero_with_cfg(field_cfg); prefix_len * suffix_count];
            let mut mask_weights = vec![F::zero_with_cfg(field_cfg); mask_count];
            let mut touched_masks = Vec::new();
            for tail in 0..tail_len {
                for (row_in_chunk, row_weight) in row_weight_chunk.iter().enumerate() {
                    let row = row_offset + row_in_chunk;
                    for (source_idx, source) in booleanity_sources[..word_bit_source_count]
                        .iter()
                        .enumerate()
                    {
                        let ShaBooleanitySource::WordBit { col, bit } = source else {
                            unreachable!("word-bit prefix only contains word-bit sources");
                        };
                        let mask = booleanity_word_bit_prefix_mask(
                            traces,
                            *col,
                            *bit,
                            prefix_vars,
                            tail,
                            row,
                            field_cfg,
                        );
                        let mask_idx = usize::from(mask);
                        if F::is_zero(&mask_weights[mask_idx]) {
                            touched_masks.push(mask_idx);
                        }
                        mask_weights[mask_idx] += booleanity_weights[source_idx].clone();
                    }

                    for &mask_idx in &touched_masks {
                        let source_weight = row_weight.clone() * &mask_weights[mask_idx];
                        let coeff_offset = mask_idx * ternary_len;
                        for ternary_idx in 0..ternary_len {
                            let coeff = &mask_coeff_table[coeff_offset + ternary_idx];
                            if F::is_zero(&coeff) {
                                continue;
                            }
                            partial[tail * ternary_len + ternary_idx] +=
                                source_weight.clone() * coeff;
                        }
                        mask_weights[mask_idx] = F::zero_with_cfg(field_cfg);
                    }
                    touched_masks.clear();

                    if suffix_count != 0 {
                        fill_booleanity_source_prefix_values(
                            traces,
                            suffix_sources,
                            prefix_vars,
                            tail,
                            row,
                            suffix_needs_virtuals,
                            &mut suffix_values,
                            field_cfg,
                        )?;
                    }

                    let mut generic_suffixes = Vec::new();
                    for suffix_idx in 0..suffix_count {
                        let source_idx = word_bit_source_count + suffix_idx;
                        let booleanity_weight = &booleanity_weights[source_idx];
                        if let Some(mask_idx) = booleanity_prefix_values_binary_mask(
                            &suffix_values,
                            suffix_count,
                            suffix_idx,
                            &one,
                        ) {
                            if F::is_zero(&mask_weights[mask_idx]) {
                                touched_masks.push(mask_idx);
                            }
                            mask_weights[mask_idx] += booleanity_weight.clone();
                        } else {
                            generic_suffixes.push(suffix_idx);
                        }
                    }

                    for &mask_idx in &touched_masks {
                        let source_weight = row_weight.clone() * &mask_weights[mask_idx];
                        let coeff_offset = mask_idx * ternary_len;
                        for ternary_idx in 0..ternary_len {
                            let coeff = &mask_coeff_table[coeff_offset + ternary_idx];
                            if F::is_zero(&coeff) {
                                continue;
                            }
                            partial[tail * ternary_len + ternary_idx] +=
                                source_weight.clone() * coeff;
                        }
                        mask_weights[mask_idx] = F::zero_with_cfg(field_cfg);
                    }
                    touched_masks.clear();

                    for suffix_idx in generic_suffixes {
                        let source_idx = word_bit_source_count + suffix_idx;
                        let booleanity_weight = &booleanity_weights[source_idx];
                        let source_weight = row_weight.clone() * booleanity_weight;
                        for (ternary_idx, plan) in coeff_plans.iter().enumerate() {
                            let coeff = booleanity_degree_two_coeff(
                                &suffix_values,
                                suffix_count,
                                suffix_idx,
                                plan,
                                field_cfg,
                            );
                            if F::is_zero(&coeff) {
                                continue;
                            }
                            partial[tail * ternary_len + ternary_idx] +=
                                source_weight.clone() * coeff;
                        }
                    }
                }
            }
            Ok(partial)
        })
        .collect::<Result<Vec<_>, ShaProjectionError>>()?;
    for partial in partials {
        for (acc, value) in table.iter_mut().zip(partial) {
            *acc += value;
        }
    }
    Ok(table)
}

fn is_canonical_production_booleanity_sources(booleanity_sources: &[ShaBooleanitySource]) -> bool {
    let word_bit_count = ShaWordCol::COUNT * SHA_WORD_BITS;
    if booleanity_sources.len() != word_bit_count + 3 * SHA_WORD_BITS {
        return false;
    }
    for (col_idx, col) in ShaWordCol::ALL.iter().enumerate() {
        for bit in 0..SHA_WORD_BITS {
            let source_idx = col_idx * SHA_WORD_BITS + bit;
            if booleanity_sources[source_idx] != (ShaBooleanitySource::WordBit { col: *col, bit }) {
                return false;
            }
        }
    }
    for bit in 0..SHA_WORD_BITS {
        let base = word_bit_count + bit * 3;
        if booleanity_sources[base] != (ShaBooleanitySource::VirtualCh1 { bit })
            || booleanity_sources[base + 1] != (ShaBooleanitySource::VirtualCh2 { bit })
            || booleanity_sources[base + 2] != (ShaBooleanitySource::VirtualMaj { bit })
        {
            return false;
        }
    }
    true
}

#[allow(clippy::arithmetic_side_effects)]
fn build_sha_booleanity_prefix_tail_table_production_fast<F, Trace>(
    traces: &[Trace],
    prefix_vars: usize,
    tail_len: usize,
    row_weights: &[F],
    booleanity_weights: &[F],
    field_cfg: &F::Config,
) -> Result<Vec<F>, ShaProjectionError>
where
    F: PrimeField,
    Trace: Borrow<ProjectedTrace<F>> + Sync,
{
    let prefix_len = binary_len(prefix_vars);
    let ternary_len = checked_ternary_len(prefix_vars)?;
    let coeff_plans = ternary_coeff_plans(prefix_vars)?;

    let word_bit_count = ShaWordCol::COUNT * SHA_WORD_BITS;
    let one = F::one_with_cfg(field_cfg);
    let partials = cfg_chunks!(row_weights, 8)
        .enumerate()
        .map(|(chunk_idx, row_weight_chunk)| {
            let row_offset = chunk_idx * 8;
            let mut partial = vec![F::zero_with_cfg(field_cfg); ternary_len * tail_len];
            let mut mask_weights = HashMap::new();
            let mut touched_masks = Vec::new();
            let mut virtuals_by_prefix = Vec::with_capacity(prefix_len);
            let mut generic_values = vec![F::zero_with_cfg(field_cfg); prefix_len];

            for tail in 0..tail_len {
                for (row_in_chunk, row_weight) in row_weight_chunk.iter().enumerate() {
                    let row = row_offset + row_in_chunk;
                    for col_idx in 0..ShaWordCol::COUNT {
                        for bit in 0..SHA_WORD_BITS {
                            let source_idx = col_idx * SHA_WORD_BITS + bit;
                            add_booleanity_mask_weight(
                                &mut mask_weights,
                                &mut touched_masks,
                                booleanity_table_prefix_mask(
                                    traces,
                                    bit_slice_index(col_idx, bit, SHA_WORD_BITS),
                                    prefix_vars,
                                    prefix_len,
                                    tail,
                                    row,
                                ),
                                &booleanity_weights[source_idx],
                            );
                        }
                    }

                    virtuals_by_prefix.clear();
                    for prefix in 0..prefix_len {
                        let instance_idx = prefix + (tail << prefix_vars);
                        virtuals_by_prefix.push(reconstruct_virtual_ch_maj_at_row_unchecked(
                            traces[instance_idx].borrow(),
                            row,
                            field_cfg,
                        )?);
                    }

                    for bit in 0..SHA_WORD_BITS {
                        for family_idx in 0..3 {
                            let source_idx = word_bit_count + bit * 3 + family_idx;
                            let mut mask_idx = 0usize;
                            let mut is_binary = true;
                            for (prefix, virtuals) in virtuals_by_prefix.iter().enumerate() {
                                let value = virtual_family_bit(virtuals, family_idx, bit);
                                generic_values[prefix] = value.clone();
                                if F::is_zero(value) {
                                    continue;
                                }
                                if value == &one {
                                    mask_idx |= 1usize << prefix;
                                } else {
                                    is_binary = false;
                                }
                            }
                            if is_binary {
                                add_booleanity_mask_weight(
                                    &mut mask_weights,
                                    &mut touched_masks,
                                    mask_idx,
                                    &booleanity_weights[source_idx],
                                );
                            } else {
                                let source_weight =
                                    row_weight.clone() * &booleanity_weights[source_idx];
                                for (ternary_idx, plan) in coeff_plans.iter().enumerate() {
                                    let coeff = booleanity_degree_two_coeff_from_prefix_values(
                                        &generic_values,
                                        plan,
                                        field_cfg,
                                    );
                                    if F::is_zero(&coeff) {
                                        continue;
                                    }
                                    partial[tail * ternary_len + ternary_idx] +=
                                        source_weight.clone() * coeff;
                                }
                            }
                        }
                    }

                    flush_booleanity_mask_weights(
                        &mut partial,
                        tail,
                        ternary_len,
                        &coeff_plans,
                        &mut mask_weights,
                        &mut touched_masks,
                        row_weight,
                    );
                }
            }
            Ok(partial)
        })
        .collect::<Result<Vec<_>, ShaProjectionError>>()?;

    let mut table = vec![F::zero_with_cfg(field_cfg); ternary_len * tail_len];
    for partial in partials {
        for (acc, value) in table.iter_mut().zip(partial) {
            *acc += value;
        }
    }
    Ok(table)
}

fn add_booleanity_mask_weight<F>(
    mask_weights: &mut HashMap<usize, F>,
    touched_masks: &mut Vec<usize>,
    mask_idx: usize,
    weight: &F,
) where
    F: PrimeField,
{
    if F::is_zero(weight) {
        return;
    }
    use std::collections::hash_map::Entry;
    match mask_weights.entry(mask_idx) {
        Entry::Occupied(mut entry) => {
            *entry.get_mut() += weight;
        }
        Entry::Vacant(entry) => {
            touched_masks.push(mask_idx);
            entry.insert(weight.clone());
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn flush_booleanity_mask_weights<F>(
    partial: &mut [F],
    tail: usize,
    ternary_len: usize,
    coeff_plans: &[TernaryCoeffPlan],
    mask_weights: &mut HashMap<usize, F>,
    touched_masks: &mut Vec<usize>,
    row_weight: &F,
) where
    F: PrimeField,
{
    for &mask_idx in touched_masks.iter() {
        let Some(mask_weight) = mask_weights.remove(&mask_idx) else {
            continue;
        };
        if F::is_zero(&mask_weight) {
            continue;
        }
        let source_weight = row_weight.clone() * mask_weight;
        for (ternary_idx, plan) in coeff_plans.iter().enumerate() {
            let coeff = booleanity_word_bit_mask_degree_two_coeff_small(mask_idx, plan);
            if coeff == 0 {
                continue;
            }
            add_small_coeff_product(
                &mut partial[tail * ternary_len + ternary_idx],
                &source_weight,
                coeff,
            );
        }
    }
    touched_masks.clear();
}

fn add_small_coeff_product<F>(acc: &mut F, value: &F, coeff: usize)
where
    F: PrimeField,
{
    if coeff == 0 {
        return;
    }
    if coeff == 1 {
        *acc += value;
        return;
    }

    let mut remaining = coeff;
    let mut addend = value.clone();
    let mut term = None;
    while remaining != 0 {
        if remaining & 1 == 1 {
            match &mut term {
                Some(term) => *term += &addend,
                None => term = Some(addend.clone()),
            }
        }
        remaining >>= 1;
        if remaining != 0 {
            let doubled = addend.clone();
            addend += &doubled;
        }
    }
    if let Some(term) = term {
        *acc += term;
    }
}

#[allow(clippy::arithmetic_side_effects)]
fn booleanity_table_prefix_mask<F, Trace>(
    traces: &[Trace],
    table_idx: usize,
    prefix_vars: usize,
    prefix_len: usize,
    tail: usize,
    row: usize,
) -> usize
where
    F: PrimeField,
    Trace: Borrow<ProjectedTrace<F>>,
{
    let mut value_mask = 0usize;
    for prefix in 0..prefix_len {
        let instance_idx = prefix + (tail << prefix_vars);
        let trace = traces[instance_idx].borrow();
        debug_assert!(instance_idx < traces.len());
        debug_assert!(table_idx < trace.bit_slices.len());
        debug_assert!(row < trace.bit_slices[table_idx].evaluations.len());
        if !F::is_zero(&trace.bit_slices[table_idx].evaluations[row]) {
            value_mask |= 1usize << prefix;
        }
    }
    value_mask
}

fn virtual_family_bit<F>(virtuals: &VirtualChMajValues<F>, family_idx: usize, bit: usize) -> &F {
    match family_idx {
        0 => &virtuals.ch1[bit],
        1 => &virtuals.ch2[bit],
        2 => &virtuals.maj[bit],
        _ => unreachable!("production virtual family index is in 0..3"),
    }
}

#[allow(clippy::arithmetic_side_effects)]
fn booleanity_degree_two_coeff_from_prefix_values<F>(
    values: &[F],
    plan: &TernaryCoeffPlan,
    field_cfg: &F::Config,
) -> F
where
    F: PrimeField,
{
    if plan.support_mask == 0 {
        let d = values[plan.finite_bits].clone();
        return d.clone() * (d - F::one_with_cfg(field_cfg));
    }

    let mut delta = F::zero_with_cfg(field_cfg);
    for (prefix, positive) in &plan.vertices {
        if *positive {
            delta += &values[*prefix];
        } else {
            delta -= &values[*prefix];
        }
    }
    delta.clone() * delta
}

#[allow(clippy::arithmetic_side_effects)]
fn booleanity_word_bit_prefix_mask<F, Trace>(
    traces: &[Trace],
    col: ShaWordCol,
    bit: usize,
    prefix_vars: usize,
    tail: usize,
    row: usize,
    field_cfg: &F::Config,
) -> u8
where
    F: PrimeField,
    Trace: Borrow<ProjectedTrace<F>>,
{
    let prefix_len = binary_len(prefix_vars);
    let mut value_mask = 0u8;
    for prefix in 0..prefix_len {
        let instance_idx = prefix + (tail << prefix_vars);
        let trace = traces[instance_idx].borrow();
        if !F::is_zero(&bit_at_shifted_or_zero_fast(
            trace, col, row, 0, bit, field_cfg,
        )) {
            value_mask |= 1u8 << prefix;
        }
    }
    value_mask
}

#[allow(clippy::arithmetic_side_effects)]
fn bind_sha_booleanity_sources_to_prefix<F>(
    traces: &[ProjectedTrace<F>],
    booleanity_sources: &[ShaBooleanitySource],
    prefix_vars: usize,
    tail_len: usize,
    prefix_weights: &[F],
    field_cfg: &F::Config,
) -> Result<BooleanityClaimTable<F>, ShaProjectionError>
where
    F: DelayedFieldProductSum,
{
    let prefix_len = binary_len(prefix_vars);
    let source_count = booleanity_sources.len();
    let source_row_count = source_count * SHA_ROW_COUNT;
    let needs_virtuals = sources_need_virtuals(booleanity_sources);
    let mut source_values = vec![F::zero_with_cfg(field_cfg); prefix_len * source_count];
    let mut out = vec![F::zero_with_cfg(field_cfg); tail_len * source_row_count];
    let mut source_column_values = Vec::with_capacity(prefix_len);

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
                source_column_values.clear();
                for prefix in 0..prefix_len {
                    source_column_values
                        .push(source_values[prefix * source_count + source_idx].clone());
                }
                let acc = FieldFieldInnerProduct::inner_product::<UNCHECKED>(
                    prefix_weights,
                    &source_column_values,
                    F::zero_with_cfg(field_cfg),
                )
                .map_err(ShaProjectionError::from)?;
                let source_row = source_idx * SHA_ROW_COUNT + row;
                out[suffix_flat_index(tail, source_row, source_row_count)] = acc;
            }
        }
    }

    BooleanityClaimTable::new(out, tail_len, source_row_count)
}

#[allow(clippy::arithmetic_side_effects)]
fn fill_booleanity_source_prefix_values<F, Trace>(
    traces: &[Trace],
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
    Trace: Borrow<ProjectedTrace<F>>,
{
    let prefix_len = binary_len(prefix_vars);
    let source_count = booleanity_sources.len();
    debug_assert_eq!(out.len(), prefix_len * source_count);

    for prefix in 0..prefix_len {
        let instance_idx = prefix + (tail << prefix_vars);
        let trace = traces[instance_idx].borrow();
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

fn booleanity_prefix_values_binary_mask<F>(
    source_values: &[F],
    source_count: usize,
    source_idx: usize,
    one: &F,
) -> Option<usize>
where
    F: PrimeField,
{
    if source_count == 0 {
        return Some(0);
    }
    let prefix_len = source_values.len() / source_count;
    if prefix_len > usize::BITS as usize {
        return None;
    }
    let mut mask = 0usize;
    for prefix in 0..prefix_len {
        let value = &source_values[prefix * source_count + source_idx];
        if F::is_zero(value) {
            continue;
        }
        if value == one {
            mask |= 1usize << prefix;
        } else {
            return None;
        }
    }
    Some(mask)
}

fn booleanity_word_bit_mask_degree_two_coeff<F>(
    source_mask: u8,
    plan: &TernaryCoeffPlan,
    small_square_fields: &[F],
    field_cfg: &F::Config,
) -> F
where
    F: PrimeField,
{
    if plan.support_mask == 0 {
        return F::zero_with_cfg(field_cfg);
    }
    let mut delta = 0i32;
    for (prefix, positive) in &plan.vertices {
        if ((source_mask >> prefix) & 1) == 0 {
            continue;
        }
        if *positive {
            delta += 1;
        } else {
            delta -= 1;
        }
    }
    let square = usize::try_from(delta * delta).expect("square is non-negative");
    small_square_fields
        .get(square)
        .cloned()
        .unwrap_or_else(|| small_usize_to_field(square, field_cfg))
}

fn booleanity_word_bit_mask_degree_two_coeff_small(
    source_mask: usize,
    plan: &TernaryCoeffPlan,
) -> usize {
    if plan.support_mask == 0 {
        return 0;
    }
    let mut delta = 0i32;
    for (prefix, positive) in &plan.vertices {
        if ((source_mask >> prefix) & 1) == 0 {
            continue;
        }
        if *positive {
            delta += 1;
        } else {
            delta -= 1;
        }
    }
    usize::try_from(delta * delta).expect("booleanity coefficient square is non-negative")
}

fn small_square_field_table<F>(field_cfg: &F::Config) -> Vec<F>
where
    F: PrimeField,
{
    (0..=64)
        .map(|value| small_usize_to_field(value, field_cfg))
        .collect()
}

fn small_usize_to_field<F>(value: usize, field_cfg: &F::Config) -> F
where
    F: PrimeField,
{
    let one = F::one_with_cfg(field_cfg);
    let mut out = F::zero_with_cfg(field_cfg);
    for _ in 0..value {
        out += &one;
    }
    out
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
    sha_word_bits_at_point_with_weights_unchecked(trace, col, shift, row_weights, field_cfg)
}

pub fn sha_word_bits_at_point_with_weights_unchecked<F>(
    trace: &ProjectedTrace<F>,
    col: ShaWordCol,
    shift: usize,
    row_weights: &[F],
    field_cfg: &F::Config,
) -> Result<[F; SHA_WORD_BITS], ShaProjectionError>
where
    F: PrimeField,
{
    let mut bits: [F; SHA_WORD_BITS] = std::array::from_fn(|_| F::zero_with_cfg(field_cfg));
    for (row, row_weight) in row_weights.iter().enumerate() {
        for (bit, out) in bits.iter_mut().enumerate() {
            *out += row_weight.clone()
                * bit_at_shifted_or_zero(trace, col, row, shift, bit, field_cfg)?;
        }
    }
    Ok(bits)
}

pub fn sha_word_bits_at_point_with_weights_inner_product_unchecked<F>(
    trace: &ProjectedTrace<F>,
    col: ShaWordCol,
    shift: usize,
    row_weights: &[F],
    field_cfg: &F::Config,
) -> Result<[F; SHA_WORD_BITS], ShaProjectionError>
where
    F: DelayedFieldProductSum,
{
    let mut values_by_bit: [Vec<F>; SHA_WORD_BITS] =
        std::array::from_fn(|_| Vec::with_capacity(row_weights.len()));
    for row in 0..row_weights.len() {
        for (bit, values) in values_by_bit.iter_mut().enumerate() {
            values.push(bit_at_shifted_or_zero(
                trace, col, row, shift, bit, field_cfg,
            )?);
        }
    }

    let mut bits: [F; SHA_WORD_BITS] = std::array::from_fn(|_| F::zero_with_cfg(field_cfg));
    for (out, values) in bits.iter_mut().zip(values_by_bit.iter()) {
        *out = FieldFieldInnerProduct::inner_product::<UNCHECKED>(
            row_weights,
            values,
            F::zero_with_cfg(field_cfg),
        )
        .map_err(ShaProjectionError::from)?;
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
    sha_int_at_point_with_weights_unchecked(trace, col, row_weights, field_cfg)
}

pub fn sha_int_at_point_with_weights_unchecked<F>(
    trace: &ProjectedTrace<F>,
    col: ShaIntCol,
    row_weights: &[F],
    field_cfg: &F::Config,
) -> Result<F, ShaProjectionError>
where
    F: PrimeField,
{
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
    let rho_sig0 = sparse_poly::<F>(&[10, 19, 30], field_cfg);
    let rho_sig1 = sparse_poly::<F>(&[7, 21, 26], field_cfg);
    residual_polys_at_row_with_rotation_polys(trace, public, row, &rho_sig0, &rho_sig1, field_cfg)
}

fn residual_polys_at_row_with_rotation_polys<F>(
    trace: &ProjectedTrace<F>,
    public: &ProjectedPublic<F>,
    row: usize,
    rho_sig0: &DynamicPolynomialF<F>,
    rho_sig1: &DynamicPolynomialF<F>,
    field_cfg: &F::Config,
) -> Result<[DynamicPolynomialF<F>; NUM_SHA_RESIDUAL_FAMILIES], ShaProjectionError>
where
    F: PrimeField,
{
    let constants = ShaResidualPolyConstants {
        rho_sig0: rho_sig0.clone(),
        rho_sig1: rho_sig1.clone(),
        two: F::one_with_cfg(field_cfg) + F::one_with_cfg(field_cfg),
        low_mu_coeff: pow_two(32, field_cfg),
        high_mu_w_coeff: pow_two(34, field_cfg),
        high_mu_3_bit_coeff: pow_two(35, field_cfg),
        high_mu_1_bit_coeff: pow_two(33, field_cfg),
    };
    residual_polys_at_row_with_constants(trace, public, row, &constants, field_cfg)
}

struct ShaResidualPolyConstants<F: PrimeField> {
    rho_sig0: DynamicPolynomialF<F>,
    rho_sig1: DynamicPolynomialF<F>,
    two: F,
    low_mu_coeff: F,
    high_mu_w_coeff: F,
    high_mu_3_bit_coeff: F,
    high_mu_1_bit_coeff: F,
}

impl<F: PrimeField> ShaResidualPolyConstants<F> {
    fn new(field_cfg: &F::Config) -> Self {
        Self {
            rho_sig0: sparse_poly::<F>(&[10, 19, 30], field_cfg),
            rho_sig1: sparse_poly::<F>(&[7, 21, 26], field_cfg),
            two: F::one_with_cfg(field_cfg) + F::one_with_cfg(field_cfg),
            low_mu_coeff: pow_two(32, field_cfg),
            high_mu_w_coeff: pow_two(34, field_cfg),
            high_mu_3_bit_coeff: pow_two(35, field_cfg),
            high_mu_1_bit_coeff: pow_two(33, field_cfg),
        }
    }
}

#[derive(Clone, Debug)]
struct NonzeroResidualCoeffAccumulator<F: PrimeField> {
    coeffs: [Vec<F>; NUM_NONZERO_SHA_FAMILIES],
    one: F,
}

impl<F> NonzeroResidualCoeffAccumulator<F>
where
    F: PrimeField,
{
    fn new(field_cfg: &F::Config) -> Self {
        Self {
            coeffs: std::array::from_fn(|_| {
                vec![F::zero_with_cfg(field_cfg); SHA_RESIDUAL_EVAL_POWER_COUNT]
            }),
            one: F::one_with_cfg(field_cfg),
        }
    }

    fn add_assign(&mut self, rhs: Self) {
        for (dst_family, rhs_family) in self.coeffs.iter_mut().zip(rhs.coeffs) {
            for (dst, rhs) in dst_family.iter_mut().zip(rhs_family) {
                *dst += rhs;
            }
        }
    }

    fn into_polys(mut self) -> [DynamicPolynomialF<F>; NUM_NONZERO_SHA_FAMILIES] {
        std::array::from_fn(|slot| {
            let mut coeffs = std::mem::take(&mut self.coeffs[slot]);
            while coeffs.last().is_some_and(F::is_zero) {
                coeffs.pop();
            }
            DynamicPolynomialF { coeffs }
        })
    }

    #[inline(always)]
    fn add_scaled_to_slot(&mut self, slot: usize, coeff_idx: usize, value: &F, scale: &F) {
        if F::is_zero(scale) || F::is_zero(value) {
            return;
        }
        debug_assert!(slot < NUM_NONZERO_SHA_FAMILIES);
        debug_assert!(coeff_idx < self.coeffs[slot].len());
        if value == &self.one {
            self.coeffs[slot][coeff_idx] += scale;
        } else {
            self.coeffs[slot][coeff_idx] += value.clone() * scale;
        }
    }

    fn add_trace_word_scaled(
        &mut self,
        slot: usize,
        trace: &ProjectedTrace<F>,
        col: ShaWordCol,
        row: usize,
        row_shift: usize,
        scale: &F,
        field_cfg: &F::Config,
    ) -> Result<(), ShaProjectionError> {
        if F::is_zero(scale) {
            return Ok(());
        }
        for bit in 0..SHA_WORD_BITS {
            let value = bit_at_shifted_or_zero(trace, col, row, row_shift, bit, field_cfg)?;
            self.add_scaled_to_slot(slot, bit, &value, scale);
        }
        Ok(())
    }

    fn add_trace_word_shift_r_scaled(
        &mut self,
        slot: usize,
        trace: &ProjectedTrace<F>,
        col: ShaWordCol,
        row: usize,
        shift: usize,
        scale: &F,
        field_cfg: &F::Config,
    ) -> Result<(), ShaProjectionError> {
        debug_assert!(shift < SHA_WORD_BITS);
        if F::is_zero(scale) {
            return Ok(());
        }
        for out_bit in 0..(SHA_WORD_BITS - shift) {
            let value = bit_at_shifted_or_zero(trace, col, row, 0, out_bit + shift, field_cfg)?;
            self.add_scaled_to_slot(slot, out_bit, &value, scale);
        }
        Ok(())
    }

    fn add_trace_word_sparse_product_scaled(
        &mut self,
        slot: usize,
        trace: &ProjectedTrace<F>,
        col: ShaWordCol,
        row: usize,
        shifts: &[usize],
        scale: &F,
        field_cfg: &F::Config,
    ) -> Result<(), ShaProjectionError> {
        if F::is_zero(scale) {
            return Ok(());
        }
        for bit in 0..SHA_WORD_BITS {
            let value = bit_at_shifted_or_zero(trace, col, row, 0, bit, field_cfg)?;
            for &shift in shifts {
                self.add_scaled_to_slot(slot, bit + shift, &value, scale);
            }
        }
        Ok(())
    }

    fn add_trace_int_const_scaled(
        &mut self,
        slot: usize,
        trace: &ProjectedTrace<F>,
        col: ShaIntCol,
        row: usize,
        scale: &F,
        field_cfg: &F::Config,
    ) -> Result<(), ShaProjectionError> {
        let value = int_scalar(trace, col, row, field_cfg)?;
        self.add_scaled_to_slot(slot, 0, &value, scale);
        Ok(())
    }

    fn add_public_scalar_const_scaled(
        &mut self,
        slot: usize,
        public: &ProjectedPublic<F>,
        col: ShaPublicCol,
        row: usize,
        scale: &F,
        field_cfg: &F::Config,
    ) -> Result<(), ShaProjectionError> {
        let value = public_scalar(public, col, row, field_cfg)?;
        self.add_scaled_to_slot(slot, 0, &value, scale);
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct FixedResidualCoeffAccumulator<F: PrimeField> {
    coeffs: Vec<Vec<F>>,
    one: F,
}

impl<F> FixedResidualCoeffAccumulator<F>
where
    F: PrimeField,
{
    fn new(family_count: usize, coeff_count: usize, field_cfg: &F::Config) -> Self {
        Self {
            coeffs: (0..family_count)
                .map(|_| vec![F::zero_with_cfg(field_cfg); coeff_count])
                .collect(),
            one: F::one_with_cfg(field_cfg),
        }
    }

    fn add_assign(&mut self, rhs: Self) {
        for (dst_family, rhs_family) in self.coeffs.iter_mut().zip(rhs.coeffs) {
            for (dst, rhs) in dst_family.iter_mut().zip(rhs_family) {
                *dst += rhs;
            }
        }
    }

    fn into_table(mut self) -> LinearResidualCoeffTable<F> {
        let coeffs = self
            .coeffs
            .drain(..)
            .map(|mut coeffs| {
                while coeffs.last().is_some_and(F::is_zero) {
                    coeffs.pop();
                }
                DynamicPolynomialF { coeffs }
            })
            .collect();
        LinearResidualCoeffTable { coeffs }
    }

    #[inline(always)]
    fn add_scaled_to_family_idx(
        &mut self,
        family_idx: usize,
        coeff_idx: usize,
        value: &F,
        scale: &F,
    ) {
        if F::is_zero(scale) || F::is_zero(value) {
            return;
        }
        debug_assert!(family_idx < self.coeffs.len());
        debug_assert!(coeff_idx < self.coeffs[family_idx].len());
        if value == &self.one {
            self.coeffs[family_idx][coeff_idx] += scale;
        } else {
            self.coeffs[family_idx][coeff_idx] += value.clone() * scale;
        }
    }

    #[inline(always)]
    fn add_scaled(&mut self, family: ShaResidualFamily, coeff_idx: usize, value: &F, scale: &F) {
        self.add_scaled_to_family_idx(family.index(), coeff_idx, value, scale);
    }

    #[inline(always)]
    fn add_const_scaled(&mut self, family: ShaResidualFamily, value: &F, scale: &F) {
        self.add_scaled(family, 0, value, scale);
    }

    fn add_trace_word_scaled(
        &mut self,
        family: ShaResidualFamily,
        trace: &ProjectedTrace<F>,
        col: ShaWordCol,
        row: usize,
        row_shift: usize,
        scale: &F,
        field_cfg: &F::Config,
    ) -> Result<(), ShaProjectionError> {
        if F::is_zero(scale) {
            return Ok(());
        }
        for bit in 0..SHA_WORD_BITS {
            let value = bit_at_shifted_or_zero(trace, col, row, row_shift, bit, field_cfg)?;
            self.add_scaled(family, bit, &value, scale);
        }
        Ok(())
    }

    fn add_trace_word_rot_scaled(
        &mut self,
        family: ShaResidualFamily,
        trace: &ProjectedTrace<F>,
        col: ShaWordCol,
        row: usize,
        rot: usize,
        scale: &F,
        field_cfg: &F::Config,
    ) -> Result<(), ShaProjectionError> {
        debug_assert!(rot < SHA_WORD_BITS);
        if F::is_zero(scale) {
            return Ok(());
        }
        for bit in 0..SHA_WORD_BITS {
            let value = bit_at_shifted_or_zero(trace, col, row, 0, bit, field_cfg)?;
            self.add_scaled(family, (bit + rot) % SHA_WORD_BITS, &value, scale);
        }
        Ok(())
    }

    fn add_trace_word_shift_r_scaled(
        &mut self,
        family: ShaResidualFamily,
        trace: &ProjectedTrace<F>,
        col: ShaWordCol,
        row: usize,
        shift: usize,
        scale: &F,
        field_cfg: &F::Config,
    ) -> Result<(), ShaProjectionError> {
        debug_assert!(shift < SHA_WORD_BITS);
        if F::is_zero(scale) {
            return Ok(());
        }
        for out_bit in 0..(SHA_WORD_BITS - shift) {
            let value = bit_at_shifted_or_zero(trace, col, row, 0, out_bit + shift, field_cfg)?;
            self.add_scaled(family, out_bit, &value, scale);
        }
        Ok(())
    }

    fn add_trace_word_sparse_product_scaled(
        &mut self,
        family: ShaResidualFamily,
        trace: &ProjectedTrace<F>,
        col: ShaWordCol,
        row: usize,
        shifts: &[usize],
        scale: &F,
        field_cfg: &F::Config,
    ) -> Result<(), ShaProjectionError> {
        if F::is_zero(scale) {
            return Ok(());
        }
        for bit in 0..SHA_WORD_BITS {
            let value = bit_at_shifted_or_zero(trace, col, row, 0, bit, field_cfg)?;
            for &shift in shifts {
                self.add_scaled(family, bit + shift, &value, scale);
            }
        }
        Ok(())
    }

    fn add_trace_int_const_scaled(
        &mut self,
        family: ShaResidualFamily,
        trace: &ProjectedTrace<F>,
        col: ShaIntCol,
        row: usize,
        scale: &F,
        field_cfg: &F::Config,
    ) -> Result<(), ShaProjectionError> {
        let value = int_scalar(trace, col, row, field_cfg)?;
        self.add_const_scaled(family, &value, scale);
        Ok(())
    }

    fn add_public_scalar_const_scaled(
        &mut self,
        family: ShaResidualFamily,
        public: &ProjectedPublic<F>,
        col: ShaPublicCol,
        row: usize,
        scale: &F,
        field_cfg: &F::Config,
    ) -> Result<(), ShaProjectionError> {
        let value = public_scalar(public, col, row, field_cfg)?;
        self.add_const_scaled(family, &value, scale);
        Ok(())
    }

    fn add_public_word_or_const_scaled(
        &mut self,
        family: ShaResidualFamily,
        public: &ProjectedPublic<F>,
        col: ShaPublicCol,
        row: usize,
        scale: &F,
        field_cfg: &F::Config,
    ) -> Result<(), ShaProjectionError> {
        if F::is_zero(scale) {
            return Ok(());
        }
        let Some(word_col) = col.public_word_col() else {
            return self.add_public_scalar_const_scaled(family, public, col, row, scale, field_cfg);
        };
        let Some(bit_slices) = &public.bit_slices else {
            return self.add_public_scalar_const_scaled(family, public, col, row, scale, field_cfg);
        };
        if row >= SHA_ROW_COUNT {
            return Ok(());
        }
        let col_idx = word_col.index();
        for bit in 0..SHA_WORD_BITS {
            let value = scalar_from_table(
                "public.bit_slices",
                bit_slices,
                bit_slice_index(col_idx, bit, SHA_WORD_BITS),
                row,
                field_cfg,
            )?;
            self.add_scaled(family, bit, &value, scale);
        }
        Ok(())
    }
}

#[inline(always)]
fn neg<F: PrimeField>(value: &F) -> F {
    -value.clone()
}

#[inline(always)]
fn scaled<F: PrimeField>(lhs: &F, rhs: &F) -> F {
    lhs.clone() * rhs
}

fn add_mu_contribution<F>(
    acc: &mut FixedResidualCoeffAccumulator<F>,
    family: ShaResidualFamily,
    trace: &ProjectedTrace<F>,
    row: usize,
    low_shift: usize,
    high_shift: usize,
    high_coeff: &F,
    row_weight: &F,
    constants: &ShaResidualPolyConstants<F>,
    field_cfg: &F::Config,
) -> Result<(), ShaProjectionError>
where
    F: PrimeField,
{
    let low_scale = scaled(row_weight, &constants.low_mu_coeff);
    let high_scale = neg(&scaled(row_weight, high_coeff));
    acc.add_trace_word_shift_r_scaled(
        family,
        trace,
        ShaWordCol::MuPacked,
        row,
        low_shift,
        &low_scale,
        field_cfg,
    )?;
    acc.add_trace_word_shift_r_scaled(
        family,
        trace,
        ShaWordCol::MuPacked,
        row,
        high_shift,
        &high_scale,
        field_cfg,
    )
}

fn add_nonzero_mu_contribution<F>(
    acc: &mut NonzeroResidualCoeffAccumulator<F>,
    slot: usize,
    trace: &ProjectedTrace<F>,
    row: usize,
    low_shift: usize,
    high_shift: usize,
    high_coeff: &F,
    row_weight: &F,
    constants: &ShaResidualPolyConstants<F>,
    field_cfg: &F::Config,
) -> Result<(), ShaProjectionError>
where
    F: PrimeField,
{
    let low_scale = scaled(row_weight, &constants.low_mu_coeff);
    let high_scale = neg(&scaled(row_weight, high_coeff));
    acc.add_trace_word_shift_r_scaled(
        slot,
        trace,
        ShaWordCol::MuPacked,
        row,
        low_shift,
        &low_scale,
        field_cfg,
    )?;
    acc.add_trace_word_shift_r_scaled(
        slot,
        trace,
        ShaWordCol::MuPacked,
        row,
        high_shift,
        &high_scale,
        field_cfg,
    )
}

#[allow(clippy::arithmetic_side_effects)]
fn accumulate_nonzero_ideal_row_fixed<F>(
    acc: &mut NonzeroResidualCoeffAccumulator<F>,
    trace: &ProjectedTrace<F>,
    public: &ProjectedPublic<F>,
    row: usize,
    row_weight: &F,
    constants: &ShaResidualPolyConstants<F>,
    field_cfg: &F::Config,
) -> Result<(), ShaProjectionError>
where
    F: PrimeField,
{
    if F::is_zero(row_weight) {
        return Ok(());
    }

    let minus_row = neg(row_weight);
    let minus_two_row = neg(&scaled(row_weight, &constants.two));

    // R0/R1: big-sigma residuals.
    acc.add_trace_word_sparse_product_scaled(
        0,
        trace,
        ShaWordCol::A,
        row,
        &[10, 19, 30],
        row_weight,
        field_cfg,
    )?;
    acc.add_trace_word_scaled(0, trace, ShaWordCol::Sigma0, row, 0, &minus_row, field_cfg)?;
    acc.add_trace_word_scaled(
        0,
        trace,
        ShaWordCol::OvSigma0,
        row,
        0,
        &minus_two_row,
        field_cfg,
    )?;

    acc.add_trace_word_sparse_product_scaled(
        1,
        trace,
        ShaWordCol::E,
        row,
        &[7, 21, 26],
        row_weight,
        field_cfg,
    )?;
    acc.add_trace_word_scaled(1, trace, ShaWordCol::Sigma1, row, 0, &minus_row, field_cfg)?;
    acc.add_trace_word_scaled(
        1,
        trace,
        ShaWordCol::OvSigma1,
        row,
        0,
        &minus_two_row,
        field_cfg,
    )?;

    // R4: schedule transition.
    acc.add_trace_word_scaled(2, trace, ShaWordCol::W, row, 16, row_weight, field_cfg)?;
    for (col, shift) in [
        (ShaWordCol::W, 0usize),
        (ShaWordCol::SmallSigma0, 1),
        (ShaWordCol::W, 9),
        (ShaWordCol::SmallSigma1, 14),
    ] {
        acc.add_trace_word_scaled(2, trace, col, row, shift, &minus_row, field_cfg)?;
    }
    add_nonzero_mu_contribution(
        acc,
        2,
        trace,
        row,
        0,
        2,
        &constants.high_mu_w_coeff,
        row_weight,
        constants,
        field_cfg,
    )?;
    acc.add_trace_int_const_scaled(
        2,
        trace,
        ShaIntCol::CompSchedule,
        row,
        row_weight,
        field_cfg,
    )?;

    // R5/R6: compression round updates.
    acc.add_trace_word_scaled(3, trace, ShaWordCol::A, row, 4, row_weight, field_cfg)?;
    for (col, shift) in [
        (ShaWordCol::E, 0usize),
        (ShaWordCol::Sigma1, 3),
        (ShaWordCol::Uef, 3),
        (ShaWordCol::UNegEg, 3),
        (ShaWordCol::W, 0),
        (ShaWordCol::Sigma0, 3),
        (ShaWordCol::Maj, 3),
    ] {
        acc.add_trace_word_scaled(3, trace, col, row, shift, &minus_row, field_cfg)?;
    }
    acc.add_public_scalar_const_scaled(3, public, ShaPublicCol::K, row + 3, &minus_row, field_cfg)?;
    add_nonzero_mu_contribution(
        acc,
        3,
        trace,
        row,
        2,
        5,
        &constants.high_mu_3_bit_coeff,
        row_weight,
        constants,
        field_cfg,
    )?;
    acc.add_trace_int_const_scaled(3, trace, ShaIntCol::CompUpdateA, row, row_weight, field_cfg)?;

    acc.add_trace_word_scaled(4, trace, ShaWordCol::E, row, 4, row_weight, field_cfg)?;
    for (col, shift) in [
        (ShaWordCol::A, 0usize),
        (ShaWordCol::E, 0),
        (ShaWordCol::Sigma1, 3),
        (ShaWordCol::Uef, 3),
        (ShaWordCol::UNegEg, 3),
        (ShaWordCol::W, 0),
    ] {
        acc.add_trace_word_scaled(4, trace, col, row, shift, &minus_row, field_cfg)?;
    }
    acc.add_public_scalar_const_scaled(4, public, ShaPublicCol::K, row + 3, &minus_row, field_cfg)?;
    add_nonzero_mu_contribution(
        acc,
        4,
        trace,
        row,
        5,
        8,
        &constants.high_mu_3_bit_coeff,
        row_weight,
        constants,
        field_cfg,
    )?;
    acc.add_trace_int_const_scaled(4, trace, ShaIntCol::CompUpdateE, row, row_weight, field_cfg)?;

    // R9/R10: feed-forward rows.
    acc.add_trace_word_scaled(5, trace, ShaWordCol::A, row, 4, row_weight, field_cfg)?;
    acc.add_trace_word_scaled(5, trace, ShaWordCol::A, row, 0, &minus_row, field_cfg)?;
    acc.add_public_scalar_const_scaled(5, public, ShaPublicCol::PAIn, row, &minus_row, field_cfg)?;
    add_nonzero_mu_contribution(
        acc,
        5,
        trace,
        row,
        8,
        9,
        &constants.high_mu_1_bit_coeff,
        row_weight,
        constants,
        field_cfg,
    )?;
    acc.add_trace_int_const_scaled(
        5,
        trace,
        ShaIntCol::CompFeedForwardA,
        row,
        row_weight,
        field_cfg,
    )?;

    acc.add_trace_word_scaled(6, trace, ShaWordCol::E, row, 4, row_weight, field_cfg)?;
    acc.add_trace_word_scaled(6, trace, ShaWordCol::E, row, 0, &minus_row, field_cfg)?;
    acc.add_public_scalar_const_scaled(6, public, ShaPublicCol::PEIn, row, &minus_row, field_cfg)?;
    add_nonzero_mu_contribution(
        acc,
        6,
        trace,
        row,
        9,
        10,
        &constants.high_mu_1_bit_coeff,
        row_weight,
        constants,
        field_cfg,
    )?;
    acc.add_trace_int_const_scaled(
        6,
        trace,
        ShaIntCol::CompFeedForwardE,
        row,
        row_weight,
        field_cfg,
    )?;

    Ok(())
}

#[allow(clippy::arithmetic_side_effects)]
fn accumulate_residual_row_fixed<F>(
    acc: &mut FixedResidualCoeffAccumulator<F>,
    trace: &ProjectedTrace<F>,
    public: &ProjectedPublic<F>,
    row: usize,
    row_weight: &F,
    constants: &ShaResidualPolyConstants<F>,
    field_cfg: &F::Config,
) -> Result<(), ShaProjectionError>
where
    F: PrimeField,
{
    if F::is_zero(row_weight) {
        return Ok(());
    }

    let minus_row = neg(row_weight);
    let minus_two_row = neg(&scaled(row_weight, &constants.two));

    // R0/R1: big-sigma residuals. Multiplication by the sparse rotation
    // polynomial is just three coefficient shifts.
    acc.add_trace_word_sparse_product_scaled(
        ShaResidualFamily::R0BigSigmaA,
        trace,
        ShaWordCol::A,
        row,
        &[10, 19, 30],
        row_weight,
        field_cfg,
    )?;
    acc.add_trace_word_scaled(
        ShaResidualFamily::R0BigSigmaA,
        trace,
        ShaWordCol::Sigma0,
        row,
        0,
        &minus_row,
        field_cfg,
    )?;
    acc.add_trace_word_scaled(
        ShaResidualFamily::R0BigSigmaA,
        trace,
        ShaWordCol::OvSigma0,
        row,
        0,
        &minus_two_row,
        field_cfg,
    )?;

    acc.add_trace_word_sparse_product_scaled(
        ShaResidualFamily::R1BigSigmaE,
        trace,
        ShaWordCol::E,
        row,
        &[7, 21, 26],
        row_weight,
        field_cfg,
    )?;
    acc.add_trace_word_scaled(
        ShaResidualFamily::R1BigSigmaE,
        trace,
        ShaWordCol::Sigma1,
        row,
        0,
        &minus_row,
        field_cfg,
    )?;
    acc.add_trace_word_scaled(
        ShaResidualFamily::R1BigSigmaE,
        trace,
        ShaWordCol::OvSigma1,
        row,
        0,
        &minus_two_row,
        field_cfg,
    )?;

    // R2/R3: small-sigma residuals over the message schedule word.
    for rot in [25usize, 14] {
        acc.add_trace_word_rot_scaled(
            ShaResidualFamily::R2SmallSigma0,
            trace,
            ShaWordCol::W,
            row,
            rot,
            row_weight,
            field_cfg,
        )?;
    }
    acc.add_trace_word_shift_r_scaled(
        ShaResidualFamily::R2SmallSigma0,
        trace,
        ShaWordCol::W,
        row,
        3,
        row_weight,
        field_cfg,
    )?;
    acc.add_trace_word_scaled(
        ShaResidualFamily::R2SmallSigma0,
        trace,
        ShaWordCol::SmallSigma0,
        row,
        0,
        &minus_row,
        field_cfg,
    )?;
    acc.add_trace_word_scaled(
        ShaResidualFamily::R2SmallSigma0,
        trace,
        ShaWordCol::OvSmallSigma0,
        row,
        0,
        &minus_two_row,
        field_cfg,
    )?;

    for rot in [15usize, 13] {
        acc.add_trace_word_rot_scaled(
            ShaResidualFamily::R3SmallSigma1,
            trace,
            ShaWordCol::W,
            row,
            rot,
            row_weight,
            field_cfg,
        )?;
    }
    acc.add_trace_word_shift_r_scaled(
        ShaResidualFamily::R3SmallSigma1,
        trace,
        ShaWordCol::W,
        row,
        10,
        row_weight,
        field_cfg,
    )?;
    acc.add_trace_word_scaled(
        ShaResidualFamily::R3SmallSigma1,
        trace,
        ShaWordCol::SmallSigma1,
        row,
        0,
        &minus_row,
        field_cfg,
    )?;
    acc.add_trace_word_scaled(
        ShaResidualFamily::R3SmallSigma1,
        trace,
        ShaWordCol::OvSmallSigma1,
        row,
        0,
        &minus_two_row,
        field_cfg,
    )?;

    // R4: schedule transition.
    acc.add_trace_word_scaled(
        ShaResidualFamily::R4Schedule,
        trace,
        ShaWordCol::W,
        row,
        16,
        row_weight,
        field_cfg,
    )?;
    for (col, shift) in [
        (ShaWordCol::W, 0usize),
        (ShaWordCol::SmallSigma0, 1),
        (ShaWordCol::W, 9),
        (ShaWordCol::SmallSigma1, 14),
    ] {
        acc.add_trace_word_scaled(
            ShaResidualFamily::R4Schedule,
            trace,
            col,
            row,
            shift,
            &minus_row,
            field_cfg,
        )?;
    }
    add_mu_contribution(
        acc,
        ShaResidualFamily::R4Schedule,
        trace,
        row,
        0,
        2,
        &constants.high_mu_w_coeff,
        row_weight,
        constants,
        field_cfg,
    )?;
    acc.add_trace_int_const_scaled(
        ShaResidualFamily::R4Schedule,
        trace,
        ShaIntCol::CompSchedule,
        row,
        row_weight,
        field_cfg,
    )?;

    // R5/R6: compression round updates.
    acc.add_trace_word_scaled(
        ShaResidualFamily::R5UpdateA,
        trace,
        ShaWordCol::A,
        row,
        4,
        row_weight,
        field_cfg,
    )?;
    for (col, shift) in [
        (ShaWordCol::E, 0usize),
        (ShaWordCol::Sigma1, 3),
        (ShaWordCol::Uef, 3),
        (ShaWordCol::UNegEg, 3),
        (ShaWordCol::W, 0),
        (ShaWordCol::Sigma0, 3),
        (ShaWordCol::Maj, 3),
    ] {
        acc.add_trace_word_scaled(
            ShaResidualFamily::R5UpdateA,
            trace,
            col,
            row,
            shift,
            &minus_row,
            field_cfg,
        )?;
    }
    acc.add_public_scalar_const_scaled(
        ShaResidualFamily::R5UpdateA,
        public,
        ShaPublicCol::K,
        row + 3,
        &minus_row,
        field_cfg,
    )?;
    add_mu_contribution(
        acc,
        ShaResidualFamily::R5UpdateA,
        trace,
        row,
        2,
        5,
        &constants.high_mu_3_bit_coeff,
        row_weight,
        constants,
        field_cfg,
    )?;
    acc.add_trace_int_const_scaled(
        ShaResidualFamily::R5UpdateA,
        trace,
        ShaIntCol::CompUpdateA,
        row,
        row_weight,
        field_cfg,
    )?;

    acc.add_trace_word_scaled(
        ShaResidualFamily::R6UpdateE,
        trace,
        ShaWordCol::E,
        row,
        4,
        row_weight,
        field_cfg,
    )?;
    for (col, shift) in [
        (ShaWordCol::A, 0usize),
        (ShaWordCol::E, 0),
        (ShaWordCol::Sigma1, 3),
        (ShaWordCol::Uef, 3),
        (ShaWordCol::UNegEg, 3),
        (ShaWordCol::W, 0),
    ] {
        acc.add_trace_word_scaled(
            ShaResidualFamily::R6UpdateE,
            trace,
            col,
            row,
            shift,
            &minus_row,
            field_cfg,
        )?;
    }
    acc.add_public_scalar_const_scaled(
        ShaResidualFamily::R6UpdateE,
        public,
        ShaPublicCol::K,
        row + 3,
        &minus_row,
        field_cfg,
    )?;
    add_mu_contribution(
        acc,
        ShaResidualFamily::R6UpdateE,
        trace,
        row,
        5,
        8,
        &constants.high_mu_3_bit_coeff,
        row_weight,
        constants,
        field_cfg,
    )?;
    acc.add_trace_int_const_scaled(
        ShaResidualFamily::R6UpdateE,
        trace,
        ShaIntCol::CompUpdateE,
        row,
        row_weight,
        field_cfg,
    )?;

    // R7/R8: pin input/output public words at active selector rows.
    let s_init = public_scalar(public, ShaPublicCol::SInit, row, field_cfg)?;
    let s_out = public_scalar(public, ShaPublicCol::SOut, row, field_cfg)?;
    let init_scale = scaled(row_weight, &s_init);
    let out_scale = scaled(row_weight, &s_out);
    let neg_init_scale = neg(&init_scale);
    let neg_out_scale = neg(&out_scale);
    acc.add_trace_word_scaled(
        ShaResidualFamily::R7PinA,
        trace,
        ShaWordCol::A,
        row,
        0,
        &init_scale,
        field_cfg,
    )?;
    acc.add_public_word_or_const_scaled(
        ShaResidualFamily::R7PinA,
        public,
        ShaPublicCol::PAIn,
        row,
        &neg_init_scale,
        field_cfg,
    )?;
    acc.add_trace_word_scaled(
        ShaResidualFamily::R7PinA,
        trace,
        ShaWordCol::A,
        row,
        0,
        &out_scale,
        field_cfg,
    )?;
    acc.add_public_word_or_const_scaled(
        ShaResidualFamily::R7PinA,
        public,
        ShaPublicCol::PAOut,
        row,
        &neg_out_scale,
        field_cfg,
    )?;

    acc.add_trace_word_scaled(
        ShaResidualFamily::R8PinE,
        trace,
        ShaWordCol::E,
        row,
        0,
        &init_scale,
        field_cfg,
    )?;
    acc.add_public_word_or_const_scaled(
        ShaResidualFamily::R8PinE,
        public,
        ShaPublicCol::PEIn,
        row,
        &neg_init_scale,
        field_cfg,
    )?;
    acc.add_trace_word_scaled(
        ShaResidualFamily::R8PinE,
        trace,
        ShaWordCol::E,
        row,
        0,
        &out_scale,
        field_cfg,
    )?;
    acc.add_public_word_or_const_scaled(
        ShaResidualFamily::R8PinE,
        public,
        ShaPublicCol::PEOut,
        row,
        &neg_out_scale,
        field_cfg,
    )?;

    // R9/R10: feed-forward rows.
    acc.add_trace_word_scaled(
        ShaResidualFamily::R9FeedForwardA,
        trace,
        ShaWordCol::A,
        row,
        4,
        row_weight,
        field_cfg,
    )?;
    acc.add_trace_word_scaled(
        ShaResidualFamily::R9FeedForwardA,
        trace,
        ShaWordCol::A,
        row,
        0,
        &minus_row,
        field_cfg,
    )?;
    acc.add_public_scalar_const_scaled(
        ShaResidualFamily::R9FeedForwardA,
        public,
        ShaPublicCol::PAIn,
        row,
        &minus_row,
        field_cfg,
    )?;
    add_mu_contribution(
        acc,
        ShaResidualFamily::R9FeedForwardA,
        trace,
        row,
        8,
        9,
        &constants.high_mu_1_bit_coeff,
        row_weight,
        constants,
        field_cfg,
    )?;
    acc.add_trace_int_const_scaled(
        ShaResidualFamily::R9FeedForwardA,
        trace,
        ShaIntCol::CompFeedForwardA,
        row,
        row_weight,
        field_cfg,
    )?;

    acc.add_trace_word_scaled(
        ShaResidualFamily::R10FeedForwardE,
        trace,
        ShaWordCol::E,
        row,
        4,
        row_weight,
        field_cfg,
    )?;
    acc.add_trace_word_scaled(
        ShaResidualFamily::R10FeedForwardE,
        trace,
        ShaWordCol::E,
        row,
        0,
        &minus_row,
        field_cfg,
    )?;
    acc.add_public_scalar_const_scaled(
        ShaResidualFamily::R10FeedForwardE,
        public,
        ShaPublicCol::PEIn,
        row,
        &minus_row,
        field_cfg,
    )?;
    add_mu_contribution(
        acc,
        ShaResidualFamily::R10FeedForwardE,
        trace,
        row,
        9,
        10,
        &constants.high_mu_1_bit_coeff,
        row_weight,
        constants,
        field_cfg,
    )?;
    acc.add_trace_int_const_scaled(
        ShaResidualFamily::R10FeedForwardE,
        trace,
        ShaIntCol::CompFeedForwardE,
        row,
        row_weight,
        field_cfg,
    )?;

    // R11-R17: selector and high-bit/carry residuals.
    let s_msg = public_scalar(public, ShaPublicCol::SMsg, row, field_cfg)?;
    let msg_scale = scaled(row_weight, &s_msg);
    let neg_msg_scale = neg(&msg_scale);
    acc.add_trace_word_scaled(
        ShaResidualFamily::R11MessagePin,
        trace,
        ShaWordCol::W,
        row,
        0,
        &msg_scale,
        field_cfg,
    )?;
    acc.add_public_word_or_const_scaled(
        ShaResidualFamily::R11MessagePin,
        public,
        ShaPublicCol::Message,
        row,
        &neg_msg_scale,
        field_cfg,
    )?;

    let s_sched = public_scalar(public, ShaPublicCol::SSched, row, field_cfg)?;
    let s_upd = public_scalar(public, ShaPublicCol::SUpd, row, field_cfg)?;
    let s_ff = public_scalar(public, ShaPublicCol::SFf, row, field_cfg)?;
    acc.add_trace_int_const_scaled(
        ShaResidualFamily::R12CompSchedule,
        trace,
        ShaIntCol::CompSchedule,
        row,
        &scaled(row_weight, &s_sched),
        field_cfg,
    )?;
    acc.add_trace_int_const_scaled(
        ShaResidualFamily::R13CompUpdateA,
        trace,
        ShaIntCol::CompUpdateA,
        row,
        &scaled(row_weight, &s_upd),
        field_cfg,
    )?;
    acc.add_trace_int_const_scaled(
        ShaResidualFamily::R14CompUpdateE,
        trace,
        ShaIntCol::CompUpdateE,
        row,
        &scaled(row_weight, &s_upd),
        field_cfg,
    )?;
    acc.add_trace_int_const_scaled(
        ShaResidualFamily::R15CompFeedForwardA,
        trace,
        ShaIntCol::CompFeedForwardA,
        row,
        &scaled(row_weight, &s_ff),
        field_cfg,
    )?;
    acc.add_trace_int_const_scaled(
        ShaResidualFamily::R16CompFeedForwardE,
        trace,
        ShaIntCol::CompFeedForwardE,
        row,
        &scaled(row_weight, &s_ff),
        field_cfg,
    )?;
    acc.add_trace_word_shift_r_scaled(
        ShaResidualFamily::R17CarryHighBits,
        trace,
        ShaWordCol::MuPacked,
        row,
        10,
        row_weight,
        field_cfg,
    )?;

    Ok(())
}

fn residual_polys_at_row_with_constants<F>(
    trace: &ProjectedTrace<F>,
    public: &ProjectedPublic<F>,
    row: usize,
    constants: &ShaResidualPolyConstants<F>,
    field_cfg: &F::Config,
) -> Result<[DynamicPolynomialF<F>; NUM_SHA_RESIDUAL_FAMILIES], ShaProjectionError>
where
    F: PrimeField,
{
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
    let w_rot25 = w.rot_c(25);
    let w_rot14 = w.rot_c(14);
    let w_rot15 = w.rot_c(15);
    let w_rot13 = w.rot_c(13);
    let w_shift3 = w.shift_r_c(3);
    let w_shift9 = word_poly_shifted(trace, ShaWordCol::W, row, 9, field_cfg)?;
    let w_shift16 = word_poly_shifted(trace, ShaWordCol::W, row, 16, field_cfg)?;
    let small_sigma0_shift1 = word_poly_shifted(trace, ShaWordCol::SmallSigma0, row, 1, field_cfg)?;
    let small_sigma1_shift14 =
        word_poly_shifted(trace, ShaWordCol::SmallSigma1, row, 14, field_cfg)?;
    let a_shift4 = word_poly_shifted(trace, ShaWordCol::A, row, 4, field_cfg)?;
    let e_shift4 = word_poly_shifted(trace, ShaWordCol::E, row, 4, field_cfg)?;
    let sigma0_shift3 = word_poly_shifted(trace, ShaWordCol::Sigma0, row, 3, field_cfg)?;
    let sigma1_shift3 = word_poly_shifted(trace, ShaWordCol::Sigma1, row, 3, field_cfg)?;
    let uef_shift3 = word_poly_shifted(trace, ShaWordCol::Uef, row, 3, field_cfg)?;
    let uneg_eg_shift3 = word_poly_shifted(trace, ShaWordCol::UNegEg, row, 3, field_cfg)?;
    let maj_shift3 = word_poly_shifted(trace, ShaWordCol::Maj, row, 3, field_cfg)?;
    let public_k_shift3 = public_const_poly(public, ShaPublicCol::K, row + 3, field_cfg)?;
    let comp_schedule = int_const_poly(trace, ShaIntCol::CompSchedule, row, field_cfg)?;
    let comp_update_a = int_const_poly(trace, ShaIntCol::CompUpdateA, row, field_cfg)?;
    let comp_update_e = int_const_poly(trace, ShaIntCol::CompUpdateE, row, field_cfg)?;
    let comp_ff_a = int_const_poly(trace, ShaIntCol::CompFeedForwardA, row, field_cfg)?;
    let comp_ff_e = int_const_poly(trace, ShaIntCol::CompFeedForwardE, row, field_cfg)?;

    let r0 = (&a * &constants.rho_sig0) - &sigma0 - &scale_poly(&ov_sigma0, &constants.two);
    let r1 = (&e * &constants.rho_sig1) - &sigma1 - &scale_poly(&ov_sigma1, &constants.two);
    let r2 = w_rot25 + &w_rot14 + &w_shift3
        - &small_sigma0
        - &scale_poly(&ov_small_sigma0, &constants.two);
    let r3 = w_rot15 + &w_rot13 + &w.shift_r_c(10)
        - &small_sigma1
        - &scale_poly(&ov_small_sigma1, &constants.two);

    let mu_packed = word_poly(trace, ShaWordCol::MuPacked, row, field_cfg)?;
    let mu_shift2 = mu_packed.shift_r_c(2);
    let mu_shift5 = mu_packed.shift_r_c(5);
    let mu_shift8 = mu_packed.shift_r_c(8);
    let mu_shift9 = mu_packed.shift_r_c(9);
    let mu_shift10 = mu_packed.shift_r_c(10);
    let mu = |low: &DynamicPolynomialF<F>, high: &DynamicPolynomialF<F>, high_coeff: &F| {
        scale_poly(low, &constants.low_mu_coeff) - &scale_poly(high, high_coeff)
    };
    let mu_w = mu(&mu_packed, &mu_shift2, &constants.high_mu_w_coeff);
    let mu_a = mu(&mu_shift2, &mu_shift5, &constants.high_mu_3_bit_coeff);
    let mu_e = mu(&mu_shift5, &mu_shift8, &constants.high_mu_3_bit_coeff);
    let mu_ff_a = mu(&mu_shift8, &mu_shift9, &constants.high_mu_1_bit_coeff);
    let mu_ff_e = mu(&mu_shift9, &mu_shift10, &constants.high_mu_1_bit_coeff);

    let r4 = w_shift16 - &w - &small_sigma0_shift1 - &w_shift9 - &small_sigma1_shift14
        + &mu_w
        + &comp_schedule;

    let r5 = a_shift4.clone()
        - &e
        - &sigma1_shift3
        - &uef_shift3
        - &uneg_eg_shift3
        - &public_k_shift3
        - &w
        - &sigma0_shift3
        - &maj_shift3
        + &mu_a
        + &comp_update_a;

    let r6 = e_shift4.clone()
        - &a
        - &e
        - &sigma1_shift3
        - &uef_shift3
        - &uneg_eg_shift3
        - &public_k_shift3
        - &w
        + &mu_e
        + &comp_update_e;

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

    let r9 = a_shift4 - &a - &public_const_poly(public, ShaPublicCol::PAIn, row, field_cfg)?
        + &mu_ff_a
        + &comp_ff_a;
    let r10 = e_shift4 - &e - &public_const_poly(public, ShaPublicCol::PEIn, row, field_cfg)?
        + &mu_ff_e
        + &comp_ff_e;
    let r11 = scale_poly(
        &(w - &public_word_or_const_poly(public, ShaPublicCol::Message, row, field_cfg)?),
        &s_msg,
    );

    let r12 = scale_poly(&comp_schedule, &s_sched);
    let r13 = scale_poly(&comp_update_a, &s_upd);
    let r14 = scale_poly(&comp_update_e, &s_upd);
    let r15 = scale_poly(&comp_ff_a, &s_ff);
    let r16 = scale_poly(&comp_ff_e, &s_ff);
    let r17 = mu_shift10;

    let mut residuals = [
        r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17,
    ];
    residuals.iter_mut().for_each(DynamicPolynomialF::trim);
    debug_assert_eq!(residuals.len(), NUM_SHA_RESIDUAL_FAMILIES);
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

pub fn validate_projected_trace<F>(trace: &ProjectedTrace<F>) -> Result<(), ShaProjectionError> {
    validate_trace(trace)
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

fn bit_at_shifted_or_zero_fast<F>(
    trace: &ProjectedTrace<F>,
    col: ShaWordCol,
    row: usize,
    shift: usize,
    bit: usize,
    field_cfg: &F::Config,
) -> F
where
    F: PrimeField,
{
    debug_assert!(bit < SHA_WORD_BITS);
    let Some(shifted) = row.checked_add(shift) else {
        return F::zero_with_cfg(field_cfg);
    };
    if shifted >= SHA_ROW_COUNT {
        return F::zero_with_cfg(field_cfg);
    }
    let table_idx = bit_slice_index(col.index(), bit, SHA_WORD_BITS);
    debug_assert!(table_idx < trace.bit_slices.len());
    debug_assert!(shifted < trace.bit_slices[table_idx].evaluations.len());
    trace.bit_slices[table_idx].evaluations[shifted].clone()
}

fn trace_word_eval_at_row_with_weights<F>(
    trace: &ProjectedTrace<F>,
    col: ShaWordCol,
    row: usize,
    shift: usize,
    weights: &[F],
    field_cfg: &F::Config,
    reducer: &BarrettDelayedReduction<'_, F>,
) -> Result<F, ShaProjectionError>
where
    F: MontgomeryLimbs + DelayedFieldProductSum + Send + Sync,
{
    let Some(shifted) = row.checked_add(shift) else {
        return Ok(F::zero_with_cfg(field_cfg));
    };
    if shifted >= SHA_ROW_COUNT {
        return Ok(F::zero_with_cfg(field_cfg));
    }
    project_bit_slice_word_row_conditional_add_dmr(
        &trace.bit_slices,
        col.index(),
        shifted,
        weights,
        field_cfg,
        reducer,
    )
}

fn int_scalar_fast<F>(
    trace: &ProjectedTrace<F>,
    col: ShaIntCol,
    row: usize,
    field_cfg: &F::Config,
) -> F
where
    F: PrimeField,
{
    if row >= SHA_ROW_COUNT {
        return F::zero_with_cfg(field_cfg);
    }
    debug_assert!(col.index() < trace.int_columns.len());
    debug_assert!(row < trace.int_columns[col.index()].evaluations.len());
    trace.int_columns[col.index()].evaluations[row].clone()
}

fn public_scalar_shifted_fast<F>(
    public: &ProjectedPublic<F>,
    col: ShaPublicCol,
    row: usize,
    shift: usize,
    field_cfg: &F::Config,
) -> F
where
    F: PrimeField,
{
    let Some(shifted) = row.checked_add(shift) else {
        return F::zero_with_cfg(field_cfg);
    };
    if shifted >= SHA_ROW_COUNT {
        return F::zero_with_cfg(field_cfg);
    }
    debug_assert!(col.index() < public.columns.len());
    debug_assert!(shifted < public.columns[col.index()].evaluations.len());
    public.columns[col.index()].evaluations[shifted].clone()
}

fn public_word_or_const_eval_at_row_with_weights<F>(
    public: &ProjectedPublic<F>,
    col: ShaPublicCol,
    row: usize,
    weights: &[F],
    field_cfg: &F::Config,
    reducer: &BarrettDelayedReduction<'_, F>,
) -> Result<F, ShaProjectionError>
where
    F: MontgomeryLimbs + DelayedFieldProductSum + Send + Sync,
{
    let Some(word_col) = col.public_word_col() else {
        return Ok(public_scalar_shifted_fast(public, col, row, 0, field_cfg));
    };
    let Some(bit_slices) = &public.bit_slices else {
        return Ok(public_scalar_shifted_fast(public, col, row, 0, field_cfg));
    };
    if row >= SHA_ROW_COUNT {
        return Ok(F::zero_with_cfg(field_cfg));
    }
    project_bit_slice_word_row_conditional_add_dmr(
        bit_slices,
        word_col.index(),
        row,
        weights,
        field_cfg,
        reducer,
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

fn add_scaled_poly_assign<F: PrimeField>(
    acc: &mut DynamicPolynomialF<F>,
    poly: &DynamicPolynomialF<F>,
    scalar: &F,
) {
    if poly.is_zero() || F::is_zero(scalar) {
        return;
    }
    if acc.coeffs.len() < poly.coeffs.len() {
        acc.coeffs
            .resize_with(poly.coeffs.len(), || F::zero_with_cfg(scalar.cfg()));
    }
    for (dst, coeff) in acc.coeffs.iter_mut().zip(&poly.coeffs) {
        *dst += coeff.clone() * scalar;
    }
}

fn pow_two<F: PrimeField>(exp: usize, field_cfg: &F::Config) -> F {
    let two = F::one_with_cfg(field_cfg) + F::one_with_cfg(field_cfg);
    let mut out = F::one_with_cfg(field_cfg);
    for _ in 0..exp {
        out *= &two;
    }
    out
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

fn project_bit_slice_word_row_conditional_add_dmr<F>(
    bit_slices: &MleTable<F>,
    col_idx: usize,
    row: usize,
    powers: &[F],
    field_cfg: &F::Config,
    reducer: &BarrettDelayedReduction<'_, F>,
) -> Result<F, ShaProjectionError>
where
    F: MontgomeryLimbs + DelayedFieldProductSum + Send + Sync,
{
    if SHA_WORD_BITS > powers.len() {
        return Err(ShaProjectionError::NonCanonicalProofObject(
            "SHA binary bit projection exceeds precomputed scalarization power bound",
        ));
    }
    let one = F::one_with_cfg(field_cfg);
    let mut bucket = Uint::<5>::zero();
    let mut pending_adds = 0usize;
    let mut acc = F::zero_with_cfg(field_cfg);

    for (bit, power) in powers.iter().enumerate().take(SHA_WORD_BITS) {
        let table_idx = bit_slice_index(col_idx, bit, SHA_WORD_BITS);
        let bit_col = bit_slices
            .get(table_idx)
            .ok_or(ShaProjectionError::MissingColumn {
                kind: "bit_slices",
                col: table_idx,
            })?;
        let bit_value = bit_col
            .evaluations
            .get(row)
            .ok_or(ShaProjectionError::ColumnRowCount {
                kind: "bit_slices",
                col: table_idx,
                got: bit_col.evaluations.len(),
                expected: row + 1,
            })?;
        if F::is_zero(bit_value) {
            continue;
        }
        if bit_value != &one {
            let bits = (0..SHA_WORD_BITS)
                .map(|fallback_bit| {
                    let fallback_idx = bit_slice_index(col_idx, fallback_bit, SHA_WORD_BITS);
                    bit_slices
                        .get(fallback_idx)
                        .and_then(|col| col.evaluations.get(row))
                        .cloned()
                        .ok_or(ShaProjectionError::MissingColumn {
                            kind: "bit_slices",
                            col: fallback_idx,
                        })
                })
                .collect::<Result<Vec<_>, ShaProjectionError>>()?;
            return project_bits_dmr(&bits, powers, field_cfg);
        }

        reducer.add(&mut bucket, power);
        pending_adds = pending_adds.saturating_add(1);
        if pending_adds >= reducer.flush_adds() {
            let pending = std::mem::replace(&mut bucket, Uint::zero());
            acc += reducer.reduce(pending);
            pending_adds = 0;
        }
    }

    if !bucket.is_zero() {
        acc += reducer.reduce(bucket);
    }
    Ok(acc)
}

fn project_binary_bits_conditional_add_dmr<F>(
    bits: &[F],
    powers: &[F],
    field_cfg: &F::Config,
    reducer: &BarrettDelayedReduction<'_, F>,
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

        reducer.add(&mut bucket, power);
        pending_adds = pending_adds.saturating_add(1);
        if pending_adds >= reducer.flush_adds() {
            let pending = std::mem::replace(&mut bucket, Uint::zero());
            acc += reducer.reduce(pending);
            pending_adds = 0;
        }
    }

    if !bucket.is_zero() {
        acc += reducer.reduce(bucket);
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

fn fold_binary_mle_tables<'a, F, I>(
    kind: &'static str,
    tables: I,
    theta: &[F],
    field_cfg: &F::Config,
) -> Result<MleTable<F>, ShaProjectionError>
where
    F: ShaBinaryFoldField + 'a,
    I: IntoIterator<Item = &'a MleTable<F>>,
{
    let tables = tables.into_iter().collect::<Vec<_>>();
    F::fold_binary_mle_tables(kind, &tables, theta, field_cfg)
}

fn fold_binary_mle_tables_generic<F>(
    kind: &'static str,
    tables: &[&MleTable<F>],
    theta: &[F],
    field_cfg: &F::Config,
) -> Result<MleTable<F>, ShaProjectionError>
where
    F: PrimeField,
{
    fold_mle_tables(kind, tables.iter().copied(), theta, field_cfg)
}

fn fold_binary_mle_tables_montgomery<F>(
    kind: &'static str,
    tables: &[&MleTable<F>],
    theta: &[F],
    field_cfg: &F::Config,
) -> Result<MleTable<F>, ShaProjectionError>
where
    F: PrimeField + MontgomeryLimbs + Send + Sync,
{
    if tables.len() != theta.len() {
        return Err(ShaProjectionError::FoldingWeightCount {
            got: theta.len(),
            expected: tables.len(),
        });
    }
    let Some(&first) = tables.first() else {
        return Ok(Vec::new());
    };
    for table in tables {
        if table.len() != first.len() {
            return Err(ShaProjectionError::InstanceCountMismatch {
                got: table.len(),
                expected: first.len(),
            });
        }
    }

    let one = F::one_with_cfg(field_cfg);
    let reducer = BarrettDelayedReduction::<F>::new(field_cfg);
    cfg_into_iter!(0..first.len())
        .map(|col_idx| {
            let template = &first[col_idx];
            let mut evaluations = vec![F::zero_with_cfg(field_cfg); template.evaluations.len()];
            for table in tables {
                let col = &table[col_idx];
                if col.num_vars != template.num_vars
                    || col.evaluations.len() != template.evaluations.len()
                {
                    return Err(ShaProjectionError::ColumnRowCount {
                        kind,
                        col: col_idx,
                        got: col.evaluations.len(),
                        expected: template.evaluations.len(),
                    });
                }
            }
            for (row, out) in evaluations.iter_mut().enumerate() {
                *out = fold_binary_row_values_montgomery_dmr(
                    tables, theta, col_idx, row, &one, field_cfg, &reducer,
                );
            }
            Ok(DenseMultilinearExtension {
                evaluations,
                num_vars: template.num_vars,
            })
        })
        .collect::<Result<MleTable<F>, ShaProjectionError>>()
}

fn fold_binary_row_values_montgomery_dmr<F>(
    tables: &[&MleTable<F>],
    theta: &[F],
    col_idx: usize,
    row: usize,
    one: &F,
    field_cfg: &F::Config,
    reducer: &BarrettDelayedReduction<'_, F>,
) -> F
where
    F: PrimeField + MontgomeryLimbs + Send + Sync,
{
    let mut bucket = Uint::<5>::zero();
    let mut pending_adds = 0usize;
    let mut acc = F::zero_with_cfg(field_cfg);

    for (table, weight) in tables.iter().zip(theta) {
        let value = &table[col_idx].evaluations[row];
        if F::is_zero(value) {
            continue;
        }
        if value != one {
            return fold_row_values_naive(tables, theta, col_idx, row, field_cfg);
        }
        reducer.add(&mut bucket, weight);
        pending_adds = pending_adds.saturating_add(1);
        if pending_adds >= reducer.flush_adds() {
            let pending = std::mem::replace(&mut bucket, Uint::zero());
            acc += reducer.reduce(pending);
            pending_adds = 0;
        }
    }

    if !bucket.is_zero() {
        acc += reducer.reduce(bucket);
    }
    acc
}

fn fold_row_values_naive<F>(
    tables: &[&MleTable<F>],
    theta: &[F],
    col_idx: usize,
    row: usize,
    field_cfg: &F::Config,
) -> F
where
    F: PrimeField,
{
    let mut acc = F::zero_with_cfg(field_cfg);
    for (table, weight) in tables.iter().zip(theta) {
        acc += weight.clone() * &table[col_idx].evaluations[row];
    }
    acc
}

fn fold_optional_binary_mle_tables<'a, F, I>(
    kind: &'static str,
    tables: I,
    theta: &[F],
    field_cfg: &F::Config,
) -> Result<Option<MleTable<F>>, ShaProjectionError>
where
    F: ShaBinaryFoldField + 'a,
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
    fold_binary_mle_tables(kind, present, theta, field_cfg).map(Some)
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
    for table in &tables {
        if table.len() != first.len() {
            return Err(ShaProjectionError::InstanceCountMismatch {
                got: table.len(),
                expected: first.len(),
            });
        }
    }
    cfg_into_iter!(0..first.len())
        .map(|col_idx| {
            let template = &first[col_idx];
            let mut evaluations = vec![F::zero_with_cfg(field_cfg); template.evaluations.len()];
            for (table, weight) in tables.iter().zip(theta) {
                let col = &table[col_idx];
                if col.num_vars != template.num_vars
                    || col.evaluations.len() != template.evaluations.len()
                {
                    return Err(ShaProjectionError::ColumnRowCount {
                        kind,
                        col: col_idx,
                        got: col.evaluations.len(),
                        expected: template.evaluations.len(),
                    });
                }
                for (out, value) in evaluations.iter_mut().zip(&col.evaluations) {
                    *out += weight.clone() * value;
                }
            }
            Ok(DenseMultilinearExtension {
                evaluations,
                num_vars: template.num_vars,
            })
        })
        .collect::<Result<MleTable<F>, ShaProjectionError>>()
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
        let reducer = BarrettDelayedReduction::<F>::new(&cfg);
        assert_eq!(
            project_binary_bits_conditional_add_dmr(&binary_bits, &powers, &cfg, &reducer).unwrap(),
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
            project_binary_bits_conditional_add_dmr(&field_bits, &powers, &cfg, &reducer).unwrap(),
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
    fn fixed_residual_coeff_table_matches_dynamic_reference() {
        let cfg = test_config();
        let a = f(5);
        let mut trace = synthetic_boolean_trace(11, &a);
        for (col_idx, column) in trace.int_columns.iter_mut().enumerate() {
            for (row_idx, value) in column.evaluations.iter_mut().enumerate() {
                *value = f(u64::try_from((col_idx + 3) * (row_idx % 17 + 1)).unwrap());
            }
        }

        let mut public = zero_public();
        for (col_idx, column) in public.columns.iter_mut().enumerate() {
            for (row_idx, value) in column.evaluations.iter_mut().enumerate() {
                *value = f(u64::try_from((col_idx + 5) * (row_idx % 19 + 1)).unwrap());
            }
        }

        let zero = F::zero_with_cfg(&cfg);
        let mut public_bits =
            vec![vec![vec![zero; SHA_WORD_BITS]; SHA_ROW_COUNT]; ShaPublicWordCol::COUNT];
        for (col_idx, col) in public_bits.iter_mut().enumerate() {
            for (row_idx, row) in col.iter_mut().enumerate() {
                for (bit_idx, bit) in row.iter_mut().enumerate() {
                    if (col_idx + row_idx + bit_idx) % 3 == 0 {
                        *bit = f(1);
                    }
                }
            }
        }
        public.bit_slices =
            Some(flatten_bit_columns(public_bits, SHA_WORD_BITS, SHA_ROW_VARS, "public").unwrap());

        let row_weights = (0..SHA_ROW_COUNT)
            .map(|row| f(u64::try_from(row % 23 + 1).unwrap()))
            .collect::<Vec<_>>();
        let fixed = build_linear_residual_coeff_tables_with_row_weights(
            &[trace.clone()],
            &[public.clone()],
            &row_weights,
            &cfg,
        )
        .unwrap();

        let constants = ShaResidualPolyConstants::new(&cfg);
        let mut expected = vec![DynamicPolynomialF::ZERO; NUM_SHA_RESIDUAL_FAMILIES];
        for (row, row_weight) in row_weights.iter().enumerate() {
            let residuals =
                residual_polys_at_row_with_constants(&trace, &public, row, &constants, &cfg)
                    .unwrap();
            for (family_idx, residual) in residuals.iter().enumerate() {
                add_scaled_poly_assign(&mut expected[family_idx], residual, row_weight);
            }
        }
        expected.iter_mut().for_each(DynamicPolynomialF::trim);

        assert_eq!(fixed[0].coeffs, expected);
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
    fn direct_aggregate_and_linear_accumulators_match_residual_table_path() {
        let cfg = test_config();
        let a = f(5);
        let mut traces = vec![
            synthetic_boolean_trace(11, &a),
            synthetic_boolean_trace(29, &a),
        ];
        for (trace_idx, trace) in traces.iter_mut().enumerate() {
            for (col_idx, column) in trace.int_columns.iter_mut().enumerate() {
                for (row_idx, value) in column.evaluations.iter_mut().enumerate() {
                    *value = f(
                        u64::try_from((trace_idx + 2) * (col_idx + 3) * (row_idx % 17 + 1))
                            .unwrap(),
                    );
                }
            }
        }

        let mut publics = vec![zero_public(), zero_public()];
        for (public_idx, public) in publics.iter_mut().enumerate() {
            for (col_idx, column) in public.columns.iter_mut().enumerate() {
                for (row_idx, value) in column.evaluations.iter_mut().enumerate() {
                    *value = f(u64::try_from(
                        (public_idx + 5) * (col_idx + 7) * (row_idx % 19 + 1),
                    )
                    .unwrap());
                }
            }
        }

        let row_weights = (0..SHA_ROW_COUNT)
            .map(|row| f(u64::try_from(row % 23 + 1).unwrap()))
            .collect::<Vec<_>>();
        let coeff_tables = build_linear_residual_coeff_tables_with_row_weights(
            &traces,
            &publics,
            &row_weights,
            &cfg,
        )
        .unwrap();

        let beta = [f(17)];
        let beta_eq_weights = zinc_poly::utils::build_eq_x_r_vec(&beta, &cfg).unwrap();
        let aggregate_plan =
            ShaAggregateIdealWeightPlan::new(&beta_eq_weights, &row_weights).unwrap();
        let direct_aggregate = beta_aggregate_nonzero_ideal_polys_direct_with_weights(
            &traces,
            &publics,
            &aggregate_plan,
            &cfg,
        )
        .unwrap();
        let table_aggregate =
            beta_aggregate_nonzero_ideal_polys_with_weights(&coeff_tables, &beta_eq_weights)
                .unwrap();
        assert_eq!(direct_aggregate, table_aggregate);

        let a_powers = build_sha_residual_eval_powers(&f(31), &cfg);
        let lambda_powers = build_sha_lambda_powers(&f(37), &cfg);
        let linear_plan =
            ShaLinearResidualWeightPlan::new(&row_weights, &a_powers, &lambda_powers).unwrap();
        let direct_linear = build_sha_sumfold_linear_accumulator_direct_with_weights(
            &traces,
            &publics,
            &linear_plan,
            &cfg,
        )
        .unwrap();
        let table_linear =
            build_sha_sumfold_linear_accumulator(&coeff_tables, &a_powers, &lambda_powers, &cfg)
                .unwrap();
        assert_eq!(direct_linear, table_linear);
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
    fn suffix_linear_claim_vector_folds_adjacent_pairs() {
        let mut table = LinearClaimTable::new(vec![f(2), f(7), f(11), f(19)]).unwrap();
        let r0 = f(5);
        table.fold_in_place(&r0);
        assert_eq!(
            table.values,
            vec![
                f(2) + r0.clone() * (f(7) - f(2)),
                f(11) + r0.clone() * (f(19) - f(11)),
            ]
        );

        let r1 = f(13);
        let expected = table.values[0].clone()
            + r1.clone() * (table.values[1].clone() - table.values[0].clone());
        table.fold_in_place(&r1);
        assert_eq!(table.values, vec![expected]);
    }

    #[test]
    fn collapsed_suffix_eq_weights_match_rebuilt_tail_weights() {
        let cfg = test_config();
        let beta = vec![f(5), f(7), f(11)];
        let mut weights = CollapsedSuffixEqWeights::new(&beta, &cfg).unwrap();

        for round in 0..beta.len() {
            let expected_pair_weights = eq_weights_or_one(&beta[round + 1..], &cfg).unwrap();
            assert_eq!(weights.pair_count(), expected_pair_weights.len());
            for (rest, expected) in expected_pair_weights.iter().enumerate() {
                assert_eq!(weights.pair_weight(rest), *expected);
            }

            weights.collapse_current_axis();
            assert_eq!(weights.values, expected_pair_weights);
        }
    }

    #[test]
    fn collapsed_suffix_eq_weights_handle_single_round_suffix() {
        let cfg = test_config();
        let beta = vec![f(5)];
        let mut weights = CollapsedSuffixEqWeights::new(&beta, &cfg).unwrap();

        assert_eq!(weights.pair_count(), 1);
        assert_eq!(weights.pair_weight(0), F::one_with_cfg(&cfg));
        weights.collapse_current_axis();
        assert_eq!(weights.values, vec![F::one_with_cfg(&cfg)]);
    }

    #[test]
    fn booleanity_claim_table_preserves_tail_major_layout() {
        let tail_len = 4;
        let source_count = 2;
        let source_row_count = source_count * SHA_ROW_COUNT;
        let source_major = (0..source_row_count)
            .map(|source_row| {
                (0..tail_len)
                    .map(|tail| f(u64::try_from(source_row * 100 + tail + 1).unwrap()))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let table = BooleanityClaimTable::from_source_major(&source_major, tail_len).unwrap();

        assert_eq!(table.tail_len(), tail_len);
        assert_eq!(table.source_row_count(), source_row_count);
        for source_idx in 0..source_count {
            for row in 0..SHA_ROW_COUNT {
                let source_row = source_idx * SHA_ROW_COUNT + row;
                for tail in 0..tail_len {
                    assert_eq!(
                        table.values()[suffix_flat_index(tail, source_row, source_row_count)],
                        source_major[source_row][tail]
                    );
                    assert_eq!(
                        *table.value(tail, source_row),
                        source_major[source_row][tail]
                    );
                }
            }
        }
    }

    #[test]
    fn suffix_fused_fold_prepare_matches_separate_fold_then_scan() {
        let cfg = test_config();
        let suffix_eq_weights = eq_weights_or_one(&[f(23), f(29), f(31)], &cfg).unwrap();
        let linear_claims = vec![f(2), f(3), f(5), f(7), f(11), f(13), f(17), f(19)];
        let source_row_weights = vec![f(37), F::zero_with_cfg(&cfg), f(41), f(43)];
        let source_major = vec![
            vec![f(47), f(53), f(59), f(61), f(67), f(71), f(73), f(79)],
            vec![f(83), f(89), f(97), f(101), f(103), f(107), f(109), f(113)],
            vec![
                f(127),
                f(131),
                f(137),
                f(139),
                f(149),
                f(151),
                f(157),
                f(163),
            ],
            vec![
                f(167),
                f(173),
                f(179),
                f(181),
                f(191),
                f(193),
                f(197),
                f(199),
            ],
        ];
        let table = BooleanityClaimTable::from_source_major(&source_major, 8).unwrap();
        let r = f(211);

        for need_t_one in [false, true] {
            let mut expected_linear = LinearClaimTable::new(linear_claims.clone()).unwrap();
            let mut expected_table = table.clone();
            let mut expected_weights =
                CollapsedSuffixEqWeights::from_values(suffix_eq_weights.clone()).unwrap();
            expected_linear.fold_in_place(&r);
            expected_table.fold_in_place(&r);
            expected_weights.collapse_current_axis();

            let expected_reduced = <F as ShaSuffixScannerField>::suffix_reduced_body_buckets_flat(
                &expected_linear.values,
                expected_table.values(),
                expected_table.source_row_count(),
                &source_row_weights,
                &expected_weights.values,
                &cfg,
            );
            let expected_one = <F as ShaSuffixScannerField>::suffix_direct_one_body_bucket_flat(
                &expected_linear.values,
                expected_table.values(),
                expected_table.source_row_count(),
                &source_row_weights,
                &expected_weights.values,
                &cfg,
            );

            let mut actual_linear = linear_claims.clone();
            let mut actual_table = table.clone();
            let actual_source_row_count = actual_table.source_row_count();
            let actual = <F as ShaSuffixScannerField>::suffix_fold_prepare_next_round_flat(
                &mut actual_linear,
                actual_table.values_mut(),
                actual_source_row_count,
                &source_row_weights,
                &suffix_eq_weights,
                &r,
                need_t_one,
                &cfg,
            );
            actual_linear.truncate(expected_linear.values.len());
            actual_table.truncate_tail_len(expected_table.tail_len());

            assert_eq!(actual_linear, expected_linear.values);
            assert_eq!(actual_table.values(), expected_table.values());
            assert_eq!((actual.0, actual.1), expected_reduced);
            if need_t_one {
                assert_eq!(actual.2, Some(expected_one));
            } else {
                assert!(actual.2.is_none());
            }
        }
    }

    #[test]
    fn suffix_fused_state_prepares_t_one_only_when_needed() {
        let cfg = test_config();
        let source_major = vec![vec![f(11), f(13), f(17), f(19)]; SHA_ROW_COUNT];
        let row_weights = vec![f(2); SHA_ROW_COUNT];
        let booleanity_weights = vec![f(3)];

        for (next_beta, should_prepare_t_one) in [(f(7), false), (F::zero_with_cfg(&cfg), true)] {
            let mut suffix = RelationSumFoldSuffixState::new(
                vec![f(5), next_beta],
                0,
                F::one_with_cfg(&cfg),
                LinearClaimTable::new(vec![f(23), f(29), f(31), f(37)]).unwrap(),
                BooleanityClaimTable::from_source_major(&source_major, 4).unwrap(),
                row_weights.clone(),
                booleanity_weights.clone(),
                &cfg,
            )
            .unwrap();
            let current_claim = f(41);
            let r = f(43);
            let evaluations = suffix.prove_round(&current_claim, &cfg);
            let next_claim = eval_cubic_from_zero_one_two_three(&evaluations, &r, &cfg);
            suffix.bind_previous_challenge(&r, &next_claim, &cfg);

            let prepared = suffix
                .prepared_round
                .clone()
                .expect("one suffix round should remain after the first bind");
            assert_eq!(prepared.t_one_direct.is_some(), should_prepare_t_one);
            let expected = suffix.round_evaluations_from_buckets(&next_claim, &prepared, &cfg);
            assert_eq!(suffix.prove_round(&next_claim, &cfg), expected);
            assert!(suffix.prepared_round.is_none());
        }
    }

    #[test]
    fn suffix_final_fold_does_not_prepare_next_round() {
        let cfg = test_config();
        let mut suffix = RelationSumFoldSuffixState::new(
            vec![f(5)],
            0,
            F::one_with_cfg(&cfg),
            LinearClaimTable::new(vec![f(7), f(11)]).unwrap(),
            BooleanityClaimTable::from_source_major(&vec![vec![f(13), f(17)]; SHA_ROW_COUNT], 2)
                .unwrap(),
            vec![f(19); SHA_ROW_COUNT],
            vec![f(23)],
            &cfg,
        )
        .unwrap();
        let current_claim = f(29);
        let r = f(31);
        let evaluations = suffix.prove_round(&current_claim, &cfg);
        let next_claim = eval_cubic_from_zero_one_two_three(&evaluations, &r, &cfg);

        suffix.bind_previous_challenge(&r, &next_claim, &cfg);

        assert!(suffix.prepared_round.is_none());
        assert_eq!(suffix.linear_claims.values.len(), 1);
        assert_eq!(suffix.booleanity_claims.tail_len(), 1);
        assert_eq!(suffix.suffix_eq_weights.len(), 1);
    }

    #[test]
    fn suffix_buckets_use_collapsed_pair_weights() {
        let cfg = test_config();
        let beta = vec![f(5), f(7)];
        let linear_claims = LinearClaimTable::new(vec![f(2), f(3), f(5), f(7)]).unwrap();
        let mut booleanity_claims = vec![vec![F::zero_with_cfg(&cfg); 4]; SHA_ROW_COUNT];
        booleanity_claims[0] = vec![f(11), f(13), f(17), f(19)];
        let booleanity_claim_table =
            BooleanityClaimTable::from_source_major(&booleanity_claims, 4).unwrap();
        let mut row_weights = vec![F::zero_with_cfg(&cfg); SHA_ROW_COUNT];
        row_weights[0] = f(23);
        let booleanity_weights = vec![f(29)];
        let suffix = RelationSumFoldSuffixState::new(
            beta.clone(),
            0,
            F::one_with_cfg(&cfg),
            linear_claims,
            booleanity_claim_table,
            row_weights,
            booleanity_weights.clone(),
            &cfg,
        )
        .unwrap();

        let tail_weights = eq_weights_or_one(&beta[1..], &cfg).unwrap();
        let mut expected_linear_zero = F::zero_with_cfg(&cfg);
        let mut expected_quadratic_zero = F::zero_with_cfg(&cfg);
        let mut expected_quadratic_infinity = F::zero_with_cfg(&cfg);
        let mut expected_one = F::zero_with_cfg(&cfg);
        let one = F::one_with_cfg(&cfg);

        for (rest, weight) in tail_weights.iter().enumerate() {
            let linear_even = suffix.linear_claims.values[rest << 1].clone();
            let linear_odd = suffix.linear_claims.values[(rest << 1) + 1].clone();
            expected_linear_zero += weight.clone() * linear_even;
            expected_one += weight.clone() * linear_odd;

            let even = suffix.booleanity_claims.value(rest << 1, 0).clone();
            let odd = suffix.booleanity_claims.value((rest << 1) + 1, 0).clone();
            let scale = f(23) * &booleanity_weights[0] * weight;
            expected_quadratic_zero += scale.clone() * even.clone() * (even.clone() - one.clone());
            let delta = odd.clone() - even;
            expected_quadratic_infinity += scale.clone() * delta.clone() * delta;
            expected_one += scale * odd.clone() * (odd - one.clone());
        }

        let (actual_zero, actual_infinity) = suffix.reduced_body_buckets(&cfg);
        assert_eq!(actual_zero, expected_linear_zero + expected_quadratic_zero);
        assert_eq!(actual_infinity, expected_quadratic_infinity);
        assert_eq!(suffix.direct_one_body_bucket(&cfg), expected_one);
    }

    #[test]
    fn suffix_dmr_scanner_matches_generic_with_forced_flush() {
        let cfg = test_config();
        let beta = vec![f(23), f(29), f(31)];
        let suffix_eq_weights = eq_weights_or_one(&beta, &cfg).unwrap();
        let linear_claims = vec![f(2), f(3), f(5), f(7), f(11), f(13), f(17), f(19)];
        let source_row_weights = vec![f(37), F::zero_with_cfg(&cfg), f(41), f(43)];
        let booleanity_claims = vec![
            vec![f(47), f(53), f(59), f(61), f(67), f(71), f(73), f(79)],
            vec![f(83), f(89), f(97), f(101), f(103), f(107), f(109), f(113)],
            vec![
                f(127),
                f(131),
                f(137),
                f(139),
                f(149),
                f(151),
                f(157),
                f(163),
            ],
            vec![
                f(167),
                f(173),
                f(179),
                f(181),
                f(191),
                f(193),
                f(197),
                f(199),
            ],
        ];

        let expected_reduced = suffix_reduced_body_buckets_generic(
            &linear_claims,
            &booleanity_claims,
            &source_row_weights,
            &suffix_eq_weights,
            &cfg,
        );
        let expected_one = suffix_direct_one_body_bucket_generic(
            &linear_claims,
            &booleanity_claims,
            &source_row_weights,
            &suffix_eq_weights,
            &cfg,
        );

        assert_eq!(
            <F as ShaSuffixScannerField>::suffix_reduced_body_buckets(
                &linear_claims,
                &booleanity_claims,
                &source_row_weights,
                &suffix_eq_weights,
                &cfg,
            ),
            expected_reduced
        );
        assert_eq!(
            <F as ShaSuffixScannerField>::suffix_direct_one_body_bucket(
                &linear_claims,
                &booleanity_claims,
                &source_row_weights,
                &suffix_eq_weights,
                &cfg,
            ),
            expected_one
        );

        let forced_flush = MontgomeryProductSum4::<F>::new_with_flush_products(&cfg, 1);
        assert_eq!(
            suffix_reduced_body_buckets_dmr_with_algorithm(
                &forced_flush,
                &linear_claims,
                &booleanity_claims,
                &source_row_weights,
                &suffix_eq_weights,
                &cfg,
            ),
            expected_reduced
        );
        assert_eq!(
            suffix_direct_one_body_bucket_dmr_with_algorithm(
                &forced_flush,
                &linear_claims,
                &booleanity_claims,
                &source_row_weights,
                &suffix_eq_weights,
                &cfg,
            ),
            expected_one
        );
    }

    #[test]
    fn all_round_suffix_linear_claims_match_bound_prefix_tail() {
        let cfg = test_config();
        let ell = 3usize;
        let prefix_vars = 1usize;
        let a = f(3);
        let traces = (0..(1usize << ell))
            .map(|idx| synthetic_boolean_trace(u64::try_from(idx).unwrap(), &a))
            .collect::<Vec<_>>();
        let beta = vec![f(5), f(7), f(11)];
        let r_ic = [f(2), f(3), f(5), f(7), f(11), f(13), f(17)];
        let row_weights = build_eq_x_r_vec(&r_ic, &cfg).unwrap();
        let a_powers = build_sha_residual_eval_powers(&a, &cfg);
        let lambda_powers = build_sha_lambda_powers(&f(19), &cfg);
        let booleanity_weights = build_booleanity_weights(&f(23), &f(29), 2, &cfg);
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
        let publics = vec![zero_public(); traces.len()];
        let coeff_tables =
            build_linear_residual_coeff_tables(&traces, &publics, &r_ic, &cfg).unwrap();
        let linear_accumulator =
            build_sha_sumfold_linear_accumulator(&coeff_tables, &a_powers, &lambda_powers, &cfg)
                .unwrap();
        let quadratic_prefix_accumulator = build_sha_sumfold_quadratic_prefix_accumulator(
            &traces,
            &sources,
            prefix_vars,
            &row_weights,
            &booleanity_weights,
            &cfg,
        )
        .unwrap();

        let r0 = f(31);
        let mut reference_prefix = RelationSumFoldPrefixFastPath::new_with_accumulators(
            &traces,
            &beta,
            &linear_accumulator,
            &quadratic_prefix_accumulator,
            &sources,
            prefix_vars,
            &cfg,
        )
        .unwrap();
        reference_prefix.round = 1;
        reference_prefix.bind_previous_round(&r0, &cfg).unwrap();
        let expected_linear_claims = reference_prefix.linear.values.clone();

        let prefix_fast = RelationSumFoldPrefixFastPath::new_with_accumulators(
            &traces,
            &beta,
            &linear_accumulator,
            &quadratic_prefix_accumulator,
            &sources,
            prefix_vars,
            &cfg,
        )
        .unwrap();
        let mut all_round =
            RelationSumFoldAllRoundsFastPath::new(prefix_fast, row_weights, booleanity_weights)
                .unwrap();
        let _ = all_round.prove_prefix_round(&None, &cfg);
        all_round.absorb_previous_challenge(&r0, &cfg).unwrap();

        assert_eq!(
            &all_round
                .suffix
                .as_ref()
                .expect("suffix state initialized")
                .linear_claims
                .values,
            &expected_linear_claims
        );
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
    fn production_sha_sumfold_suffix_zero_beta_matches_dense_sumcheck() {
        let cfg = test_config();
        let ell = 2usize;
        let a = f(3);
        let traces = (0..(1usize << ell))
            .map(|idx| synthetic_boolean_trace(u64::try_from(idx).unwrap(), &a))
            .collect::<Vec<_>>();
        let publics = vec![zero_public(); traces.len()];
        let beta = vec![f(5), F::zero_with_cfg(&cfg)];
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

        let dense = build_dense_sha_sumfold_group(
            &traces, &publics, &beta, &r_ic, &a, &lambda, &rho, &xi, &sources, &cfg,
        )
        .unwrap();
        let optimized = build_production_sha_sumfold_group(
            &traces, &publics, &beta, &r_ic, &a, &lambda, &rho, &xi, &sources, 1, &cfg,
        )
        .unwrap();

        let (dense_proof, dense_point, dense_expected) = prove_and_verify_sumfold(dense, ell);
        let (optimized_proof, optimized_point, optimized_expected) =
            prove_and_verify_sumfold(optimized, ell);

        assert_eq!(optimized_proof, dense_proof);
        assert_eq!(optimized_point, dense_point);
        assert_eq!(optimized_expected, dense_expected);
    }

    #[test]
    fn production_sha_sumfold_live_linear_claims_match_dense_sumcheck() {
        let cfg = test_config();
        let ell = 3usize;
        let a = f(3);
        let traces = (0..(1usize << ell))
            .map(|idx| synthetic_boolean_trace(u64::try_from(idx).unwrap(), &a))
            .collect::<Vec<_>>();
        let publics = vec![zero_public(); traces.len()];
        let beta = vec![f(5), F::zero_with_cfg(&cfg), f(11)];
        let beta_eq_weights = build_eq_x_r_vec(&beta, &cfg).unwrap();
        let r_ic = [f(2), f(3), f(5), f(7), f(11), f(13), f(17)];
        let row_weights = build_eq_x_r_vec(&r_ic, &cfg).unwrap();
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
        let a_powers = build_sha_residual_eval_powers(&a, &cfg);
        let lambda_powers = build_sha_lambda_powers(&lambda, &cfg);
        let booleanity_weights = build_booleanity_weights(&rho, &xi, sources.len(), &cfg);
        let coeff_tables =
            build_linear_residual_coeff_tables(&traces, &publics, &r_ic, &cfg).unwrap();
        let linear_accumulator =
            build_sha_sumfold_linear_accumulator(&coeff_tables, &a_powers, &lambda_powers, &cfg)
                .unwrap();

        for prefix_vars in [1usize, 2, 3] {
            let dense = build_dense_sha_sumfold_group(
                &traces, &publics, &beta, &r_ic, &a, &lambda, &rho, &xi, &sources, &cfg,
            )
            .unwrap();
            let (dense_proof, dense_point, dense_expected) = prove_and_verify_sumfold(dense, ell);
            let initial_claim = dense_proof.claimed_sums()[0].clone();
            let quadratic_prefix_accumulator = build_sha_sumfold_quadratic_prefix_accumulator(
                &traces,
                &sources,
                prefix_vars,
                &row_weights,
                &booleanity_weights,
                &cfg,
            )
            .unwrap();
            let optimized =
                build_production_sha_sumfold_group_from_prefix_accumulators_with_initial_claim(
                    &traces,
                    &beta,
                    &beta_eq_weights,
                    &row_weights,
                    &linear_accumulator,
                    &quadratic_prefix_accumulator,
                    &booleanity_weights,
                    &sources,
                    prefix_vars,
                    &initial_claim,
                    &cfg,
                )
                .unwrap();
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
