//! Production SHA-256 ProjectionFold helpers.
//!
//! This module implements the SHA-specific data model and reference
//! computations used by the production ProjectionFold flow:
//!
//! fresh ideal checks -> SumFold over instances -> post-SumFold folding ->
//! folded row check over the 128-row SHA domain.

use crate::ideal_check::batched_ideal_check;
use crate::neutron_nova::SumFoldError;
use crate::sumcheck::multi_degree::MultiDegreeSumcheckGroup;
use crypto_primitives::PrimeField;
use num_traits::{ConstZero, Zero};
use thiserror::Error;
use zinc_poly::{
    EvaluatablePolynomial,
    mle::DenseMultilinearExtension,
    univariate::dynamic::over_field::DynamicPolynomialF,
    utils::{ArithErrors, build_eq_x_r_vec, eq_eval},
};
use zinc_uair::{
    ideal::{Ideal, IdealCheck, IdealCheckError, rotation::RotationIdeal},
    ideal_collector::IdealOrZero,
};
use zinc_utils::{
    delayed_reduction::DelayedFieldProductSum, from_ref::FromRef,
    inner_transparent_field::InnerTransparentField, powers,
};

pub const SHA_ROW_VARS: usize = 7;
pub const SHA_ROW_COUNT: usize = 128;
pub const SHA_WORD_BITS: usize = 32;
pub const NUM_SHA_RESIDUAL_FAMILIES: usize = 18;
pub const NUM_NONZERO_SHA_FAMILIES: usize = 7;

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
    pub const COUNT: usize = 12;

    pub fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShaBitSliceColumns<F> {
    /// Indexed as `[word_col][row][bit]`.
    pub columns: Vec<Vec<Vec<F>>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShaScalarizedRows<F> {
    /// Indexed as `[word_col][row]`.
    pub words: Vec<Vec<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShaIntColumns<F> {
    /// Indexed as `[int_col][row]`.
    pub columns: Vec<Vec<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShaPublicColumns<F> {
    /// Indexed as `[public_col][row]`.
    pub columns: Vec<Vec<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectedShaTrace<F> {
    pub rows: usize,
    pub bit_slices: ShaBitSliceColumns<F>,
    pub scalarized_words: ShaScalarizedRows<F>,
    pub int_columns: ShaIntColumns<F>,
    pub public_columns: ShaPublicColumns<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectedShaPublic<F> {
    pub columns: ShaPublicColumns<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FreshShaIdealCache<F: PrimeField> {
    pub r_ic: [F; SHA_ROW_VARS],
    pub ideal_polys: Vec<[DynamicPolynomialF<F>; NUM_NONZERO_SHA_FAMILIES]>,
    pub taus_at_a: Vec<[F; NUM_NONZERO_SHA_FAMILIES]>,
    pub fresh_targets: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShaSumFoldOutput<F> {
    r_b: Vec<F>,
    c_sf: F,
    t_prime: F,
    theta: Vec<F>,
}

impl<F> ShaSumFoldOutput<F> {
    pub fn r_b(&self) -> &[F] {
        &self.r_b
    }

    pub fn c_sf(&self) -> &F {
        &self.c_sf
    }

    pub fn t_prime(&self) -> &F {
        &self.t_prime
    }

    pub fn theta(&self) -> &[F] {
        &self.theta
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldedCommitments<C> {
    pub commitments: Vec<C>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldedShaAccumulator<F, C = F> {
    pub t_prime: F,
    pub folded_commitments: FoldedCommitments<C>,
    pub folded_public: ProjectedShaPublic<F>,
    pub r_b: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldedShaWitness<F> {
    pub trace: ProjectedShaTrace<F>,
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
    #[error("ideal membership check failed")]
    IdealMembership,
    #[error("polynomial evaluation failed: {0}")]
    PolynomialEvaluation(#[from] zinc_poly::EvaluationError),
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

pub fn build_sha_ideal_values_at_point<F>(
    trace: &ProjectedShaTrace<F>,
    public: &ProjectedShaPublic<F>,
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

pub fn build_fresh_sha_ideal_cache<F>(
    traces: &[ProjectedShaTrace<F>],
    publics: &[ProjectedShaPublic<F>],
    r_ic: [F; SHA_ROW_VARS],
    field_cfg: &F::Config,
) -> Result<FreshShaIdealCache<F>, ShaProjectionError>
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

    Ok(FreshShaIdealCache {
        r_ic,
        ideal_polys,
        taus_at_a: Vec::new(),
        fresh_targets: Vec::new(),
    })
}

pub fn check_fresh_sha_ideal_cache<F>(
    cache: &FreshShaIdealCache<F>,
    field_cfg: &F::Config,
) -> Result<(), ShaProjectionError>
where
    F: PrimeField,
{
    for values in &cache.ideal_polys {
        check_sha_ideal_values(values, field_cfg)?;
    }
    Ok(())
}

pub fn evaluate_fresh_sha_targets<F>(
    cache: &mut FreshShaIdealCache<F>,
    a: &F,
    lambda: &F,
    field_cfg: &F::Config,
) -> Result<(), ShaProjectionError>
where
    F: PrimeField,
{
    let one = F::one_with_cfg(field_cfg);
    let zero = F::zero_with_cfg(field_cfg);
    let lambda_powers = powers(lambda.clone(), one, NUM_SHA_RESIDUAL_FAMILIES);

    cache.taus_at_a.clear();
    cache.fresh_targets.clear();

    for ideal_polys in &cache.ideal_polys {
        let taus: [F; NUM_NONZERO_SHA_FAMILIES] = std::array::from_fn(|idx| {
            ideal_polys[idx]
                .evaluate_at_point(a)
                .expect("field polynomial evaluation cannot fail")
        });
        let mut target = zero.clone();
        for (slot, family) in NONZERO_SHA_FAMILIES.iter().enumerate() {
            target += lambda_powers[family.index()].clone() * &taus[slot];
        }
        cache.taus_at_a.push(taus);
        cache.fresh_targets.push(target);
    }
    Ok(())
}

pub fn finalize_sha_sumfold<F>(
    beta: &[F],
    r_b: Vec<F>,
    c_sf: F,
    instance_count: usize,
    field_cfg: &F::Config,
) -> Result<ShaSumFoldOutput<F>, ShaProjectionError>
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

    let theta = build_eq_x_r_vec(&r_b, field_cfg)?;
    debug_assert_eq!(theta.len(), instance_count);
    let t_prime = c_sf.clone() / d;
    Ok(ShaSumFoldOutput {
        r_b,
        c_sf,
        t_prime,
        theta,
    })
}

pub fn fold_projected_sha_traces<F>(
    traces: &[ProjectedShaTrace<F>],
    publics: &[ProjectedShaPublic<F>],
    sumfold: &ShaSumFoldOutput<F>,
    field_cfg: &F::Config,
) -> Result<(FoldedShaWitness<F>, ProjectedShaPublic<F>), ShaProjectionError>
where
    F: PrimeField,
{
    if traces.len() != publics.len() {
        return Err(ShaProjectionError::InstanceCountMismatch {
            got: publics.len(),
            expected: traces.len(),
        });
    }
    if sumfold.theta.len() != traces.len() {
        return Err(ShaProjectionError::FoldingWeightCount {
            got: sumfold.theta.len(),
            expected: traces.len(),
        });
    }
    for trace in traces {
        validate_trace(trace)?;
    }
    for public in publics {
        validate_public(public)?;
    }

    let folded_trace = ProjectedShaTrace {
        rows: SHA_ROW_COUNT,
        bit_slices: ShaBitSliceColumns {
            columns: fold_3d(
                traces.iter().map(|trace| &trace.bit_slices.columns),
                &sumfold.theta,
                field_cfg,
            )?,
        },
        scalarized_words: ShaScalarizedRows {
            words: fold_2d(
                traces.iter().map(|trace| &trace.scalarized_words.words),
                &sumfold.theta,
                field_cfg,
            )?,
        },
        int_columns: ShaIntColumns {
            columns: fold_2d(
                traces.iter().map(|trace| &trace.int_columns.columns),
                &sumfold.theta,
                field_cfg,
            )?,
        },
        public_columns: ShaPublicColumns {
            columns: fold_2d(
                traces.iter().map(|trace| &trace.public_columns.columns),
                &sumfold.theta,
                field_cfg,
            )?,
        },
    };
    let folded_public = ProjectedShaPublic {
        columns: ShaPublicColumns {
            columns: fold_2d(
                publics.iter().map(|public| &public.columns.columns),
                &sumfold.theta,
                field_cfg,
            )?,
        },
    };

    Ok((
        FoldedShaWitness {
            trace: folded_trace,
        },
        folded_public,
    ))
}

pub fn scalarize_trace_words<F>(
    bit_slices: &ShaBitSliceColumns<F>,
    a: &F,
    field_cfg: &F::Config,
) -> Result<ShaScalarizedRows<F>, ShaProjectionError>
where
    F: PrimeField,
{
    let powers = powers(a.clone(), F::one_with_cfg(field_cfg), SHA_WORD_BITS);
    let mut words = Vec::with_capacity(bit_slices.columns.len());
    for (col_idx, col) in bit_slices.columns.iter().enumerate() {
        if col.len() != SHA_ROW_COUNT {
            return Err(ShaProjectionError::ColumnRowCount {
                kind: "bit_slices",
                col: col_idx,
                got: col.len(),
                expected: SHA_ROW_COUNT,
            });
        }
        let mut out_col = Vec::with_capacity(SHA_ROW_COUNT);
        for (row, bits) in col.iter().enumerate() {
            if bits.len() != SHA_WORD_BITS {
                return Err(ShaProjectionError::BitCount {
                    col: col_idx,
                    row,
                    got: bits.len(),
                    expected: SHA_WORD_BITS,
                });
            }
            out_col.push(project_bits(bits, &powers, field_cfg));
        }
        words.push(out_col);
    }
    Ok(ShaScalarizedRows { words })
}

pub fn verify_folded_scalarization_links<F>(
    trace: &ProjectedShaTrace<F>,
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
        let bit_col =
            trace
                .bit_slices
                .columns
                .get(col_idx)
                .ok_or(ShaProjectionError::MissingColumn {
                    kind: "bit_slices",
                    col: col_idx,
                })?;
        let scalar_col =
            trace
                .scalarized_words
                .words
                .get(col_idx)
                .ok_or(ShaProjectionError::MissingColumn {
                    kind: "scalarized_words",
                    col: col_idx,
                })?;
        for row in 0..SHA_ROW_COUNT {
            let recombined = project_bits(&bit_col[row], &powers, field_cfg);
            if recombined != scalar_col[row] {
                return Err(ShaProjectionError::ScalarizationMismatch { col: col_idx });
            }
        }
    }
    Ok(())
}

pub fn verify_folded_scalarization_links_at_point<F>(
    trace: &ProjectedShaTrace<F>,
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
    trace: &ProjectedShaTrace<F>,
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
    let mut bit_eval = F::zero_with_cfg(field_cfg);

    for (row, row_weight) in row_weights.iter().enumerate() {
        word_eval += row_weight.clone()
            * scalarized_word_at_shifted_or_zero(trace, col, row, shift, field_cfg)?;
        for (bit, power) in powers.iter().enumerate() {
            let source_bit = bit_at_shifted_or_zero(trace, col, row, shift, bit, field_cfg)?;
            bit_eval += row_weight.clone() * power.clone() * source_bit;
        }
    }

    if word_eval != bit_eval {
        return Err(ShaProjectionError::ScalarizationMismatch { col: col.index() });
    }
    Ok(())
}

pub fn reconstruct_virtual_ch_maj_at_row<F>(
    trace: &ProjectedShaTrace<F>,
    row: usize,
    field_cfg: &F::Config,
) -> Result<VirtualChMajValues<F>, ShaProjectionError>
where
    F: PrimeField,
{
    validate_trace(trace)?;
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
    trace: &ProjectedShaTrace<F>,
    public: &ProjectedShaPublic<F>,
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
    let lambda_powers = powers(
        lambda.clone(),
        F::one_with_cfg(field_cfg),
        NUM_SHA_RESIDUAL_FAMILIES,
    );
    let rho_powers = powers(
        rho.clone(),
        F::one_with_cfg(field_cfg),
        booleanity_sources.len(),
    );

    let mut out = Vec::with_capacity(SHA_ROW_COUNT);
    for row in 0..SHA_ROW_COUNT {
        let residuals = residual_values_at_row(trace, public, row, a, field_cfg)?;
        let mut linear = F::zero_with_cfg(field_cfg);
        for family in ShaResidualFamily::ALL {
            linear += lambda_powers[family.index()].clone() * &residuals[family.index()];
        }

        let mut bool_sum = F::zero_with_cfg(field_cfg);
        for (idx, source) in booleanity_sources.iter().enumerate() {
            let d = booleanity_source_value_at_row(trace, row, source, field_cfg)?;
            let term = d.clone() * (d - F::one_with_cfg(field_cfg));
            bool_sum += rho_powers[idx].clone() * term;
        }
        out.push(row_weights[row].clone() * (linear + xi.clone() * bool_sum));
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

pub fn verify_folded_row_sumcheck_claim<F>(
    claimed_sum: &F,
    t_prime: &F,
) -> Result<(), ShaProjectionError>
where
    F: PrimeField,
{
    if claimed_sum != t_prime {
        return Err(ShaProjectionError::FoldedRowClaimMismatch);
    }
    Ok(())
}

pub fn residual_polys_at_row<F>(
    trace: &ProjectedShaTrace<F>,
    public: &ProjectedShaPublic<F>,
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
        - &word_poly_shifted(trace, ShaWordCol::W, row, 3, field_cfg)?
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
        - &word_poly_shifted(trace, ShaWordCol::W, row, 3, field_cfg)?
        + &mu_e
        + &int_const_poly(trace, ShaIntCol::CompUpdateE, row, field_cfg)?;

    let s_init = public_scalar(public, ShaPublicCol::SInit, row, field_cfg)?;
    let s_msg = public_scalar(public, ShaPublicCol::SMsg, row, field_cfg)?;
    let s_sched = public_scalar(public, ShaPublicCol::SSched, row, field_cfg)?;
    let s_upd = public_scalar(public, ShaPublicCol::SUpd, row, field_cfg)?;
    let s_ff = public_scalar(public, ShaPublicCol::SFf, row, field_cfg)?;
    let s_out = public_scalar(public, ShaPublicCol::SOut, row, field_cfg)?;

    let r7 = scale_poly(
        &(a.clone() - &public_const_poly(public, ShaPublicCol::PAIn, row, field_cfg)?),
        &s_init,
    ) + &scale_poly(
        &(a.clone() - &public_const_poly(public, ShaPublicCol::PAOut, row, field_cfg)?),
        &s_out,
    );
    let r8 = scale_poly(
        &(e.clone() - &public_const_poly(public, ShaPublicCol::PEIn, row, field_cfg)?),
        &s_init,
    ) + &scale_poly(
        &(e.clone() - &public_const_poly(public, ShaPublicCol::PEOut, row, field_cfg)?),
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
        &(w - &public_const_poly(public, ShaPublicCol::Message, row, field_cfg)?),
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

fn residual_values_at_row<F>(
    trace: &ProjectedShaTrace<F>,
    public: &ProjectedShaPublic<F>,
    row: usize,
    a: &F,
    field_cfg: &F::Config,
) -> Result<[F; NUM_SHA_RESIDUAL_FAMILIES], ShaProjectionError>
where
    F: PrimeField,
{
    let polies = residual_polys_at_row(trace, public, row, field_cfg)?;
    let mut out: [F; NUM_SHA_RESIDUAL_FAMILIES] =
        std::array::from_fn(|_| F::zero_with_cfg(field_cfg));
    for (idx, poly) in polies.iter().enumerate() {
        out[idx] = poly.evaluate_at_point(a)?;
    }
    Ok(out)
}

fn validate_trace<F>(trace: &ProjectedShaTrace<F>) -> Result<(), ShaProjectionError> {
    if trace.rows != SHA_ROW_COUNT {
        return Err(ShaProjectionError::RowCount {
            expected: SHA_ROW_COUNT,
            got: trace.rows,
        });
    }
    validate_bit_columns(&trace.bit_slices)?;
    validate_matrix(
        "scalarized_words",
        &trace.scalarized_words.words,
        SHA_ROW_COUNT,
    )?;
    validate_matrix("int_columns", &trace.int_columns.columns, SHA_ROW_COUNT)?;
    validate_matrix(
        "public_columns",
        &trace.public_columns.columns,
        SHA_ROW_COUNT,
    )
}

fn validate_public<F>(public: &ProjectedShaPublic<F>) -> Result<(), ShaProjectionError> {
    validate_matrix("public.columns", &public.columns.columns, SHA_ROW_COUNT)
}

fn validate_matrix<F>(
    kind: &'static str,
    columns: &[Vec<F>],
    rows: usize,
) -> Result<(), ShaProjectionError> {
    for (col, values) in columns.iter().enumerate() {
        if values.len() != rows {
            return Err(ShaProjectionError::ColumnRowCount {
                kind,
                col,
                got: values.len(),
                expected: rows,
            });
        }
    }
    Ok(())
}

fn validate_bit_columns<F>(bit_slices: &ShaBitSliceColumns<F>) -> Result<(), ShaProjectionError> {
    for (col, rows) in bit_slices.columns.iter().enumerate() {
        if rows.len() != SHA_ROW_COUNT {
            return Err(ShaProjectionError::ColumnRowCount {
                kind: "bit_slices",
                col,
                got: rows.len(),
                expected: SHA_ROW_COUNT,
            });
        }
        for (row, bits) in rows.iter().enumerate() {
            if bits.len() != SHA_WORD_BITS {
                return Err(ShaProjectionError::BitCount {
                    col,
                    row,
                    got: bits.len(),
                    expected: SHA_WORD_BITS,
                });
            }
        }
    }
    Ok(())
}

fn word_poly<F>(
    trace: &ProjectedShaTrace<F>,
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
    let rows = trace
        .bit_slices
        .columns
        .get(col_idx)
        .ok_or(ShaProjectionError::MissingColumn {
            kind: "bit_slices",
            col: col_idx,
        })?;
    let bits = rows.get(row).ok_or(ShaProjectionError::ColumnRowCount {
        kind: "bit_slices",
        col: col_idx,
        got: rows.len(),
        expected: SHA_ROW_COUNT,
    })?;
    if bits.len() != SHA_WORD_BITS {
        return Err(ShaProjectionError::BitCount {
            col: col_idx,
            row,
            got: bits.len(),
            expected: SHA_WORD_BITS,
        });
    }
    let mut coeffs = bits.clone();
    coeffs.resize(SHA_WORD_BITS, F::zero_with_cfg(field_cfg));
    Ok(DynamicPolynomialF::new_trimmed(coeffs))
}

fn word_poly_shifted<F>(
    trace: &ProjectedShaTrace<F>,
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
    trace: &ProjectedShaTrace<F>,
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
    let rows = trace
        .bit_slices
        .columns
        .get(col_idx)
        .ok_or(ShaProjectionError::MissingColumn {
            kind: "bit_slices",
            col: col_idx,
        })?;
    if rows.len() != SHA_ROW_COUNT {
        return Err(ShaProjectionError::ColumnRowCount {
            kind: "bit_slices",
            col: col_idx,
            got: rows.len(),
            expected: SHA_ROW_COUNT,
        });
    }
    let bits = &rows[shifted];
    if bits.len() != SHA_WORD_BITS {
        return Err(ShaProjectionError::BitCount {
            col: col_idx,
            row: shifted,
            got: bits.len(),
            expected: SHA_WORD_BITS,
        });
    }
    Ok(bits[bit].clone())
}

fn int_const_poly<F>(
    trace: &ProjectedShaTrace<F>,
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
    public: &ProjectedShaPublic<F>,
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

fn int_scalar<F>(
    trace: &ProjectedShaTrace<F>,
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
    scalar_from_matrix(
        "int_columns",
        &trace.int_columns.columns,
        col.index(),
        row,
        field_cfg,
    )
}

fn public_scalar<F>(
    public: &ProjectedShaPublic<F>,
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
    scalar_from_matrix(
        "public.columns",
        &public.columns.columns,
        col.index(),
        row,
        field_cfg,
    )
}

fn scalar_from_matrix<F>(
    kind: &'static str,
    columns: &[Vec<F>],
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
    Ok(values
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
    trace: &ProjectedShaTrace<F>,
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

fn project_bits<F: PrimeField>(bits: &[F], powers: &[F], field_cfg: &F::Config) -> F {
    let mut acc = F::zero_with_cfg(field_cfg);
    for (bit, power) in bits.iter().zip(powers.iter()) {
        acc += bit.clone() * power;
    }
    acc
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
    trace: &ProjectedShaTrace<F>,
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
    let rows =
        trace
            .scalarized_words
            .words
            .get(col_idx)
            .ok_or(ShaProjectionError::MissingColumn {
                kind: "scalarized_words",
                col: col_idx,
            })?;
    if rows.len() != SHA_ROW_COUNT {
        return Err(ShaProjectionError::ColumnRowCount {
            kind: "scalarized_words",
            col: col_idx,
            got: rows.len(),
            expected: SHA_ROW_COUNT,
        });
    }
    Ok(rows[shifted].clone())
}

fn booleanity_source_value_at_row<F>(
    trace: &ProjectedShaTrace<F>,
    row: usize,
    source: &ShaBooleanitySource,
    field_cfg: &F::Config,
) -> Result<F, ShaProjectionError>
where
    F: PrimeField,
{
    match source {
        ShaBooleanitySource::WordBit { col, bit } => {
            bit_at_shifted_or_zero(trace, *col, row, 0, *bit, field_cfg)
        }
        ShaBooleanitySource::VirtualCh1 { bit } => virtual_bit_at(
            &reconstruct_virtual_ch_maj_at_row(trace, row, field_cfg)?.ch1,
            *bit,
        ),
        ShaBooleanitySource::VirtualCh2 { bit } => virtual_bit_at(
            &reconstruct_virtual_ch_maj_at_row(trace, row, field_cfg)?.ch2,
            *bit,
        ),
        ShaBooleanitySource::VirtualMaj { bit } => virtual_bit_at(
            &reconstruct_virtual_ch_maj_at_row(trace, row, field_cfg)?.maj,
            *bit,
        ),
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

fn fold_2d<'a, F, I>(
    matrices: I,
    theta: &[F],
    field_cfg: &F::Config,
) -> Result<Vec<Vec<F>>, ShaProjectionError>
where
    F: PrimeField + 'a,
    I: IntoIterator<Item = &'a Vec<Vec<F>>>,
{
    let matrices = matrices.into_iter().collect::<Vec<_>>();
    if matrices.len() != theta.len() {
        return Err(ShaProjectionError::FoldingWeightCount {
            got: theta.len(),
            expected: matrices.len(),
        });
    }
    let Some(first) = matrices.first() else {
        return Ok(Vec::new());
    };
    let mut out = vec![vec![F::zero_with_cfg(field_cfg); SHA_ROW_COUNT]; first.len()];
    for (matrix, weight) in matrices.iter().zip(theta) {
        if matrix.len() != first.len() {
            return Err(ShaProjectionError::InstanceCountMismatch {
                got: matrix.len(),
                expected: first.len(),
            });
        }
        for (col_idx, col) in matrix.iter().enumerate() {
            if col.len() != SHA_ROW_COUNT {
                return Err(ShaProjectionError::ColumnRowCount {
                    kind: "fold_2d",
                    col: col_idx,
                    got: col.len(),
                    expected: SHA_ROW_COUNT,
                });
            }
            for row in 0..SHA_ROW_COUNT {
                out[col_idx][row] += weight.clone() * &col[row];
            }
        }
    }
    Ok(out)
}

fn fold_3d<'a, F, I>(
    tensors: I,
    theta: &[F],
    field_cfg: &F::Config,
) -> Result<Vec<Vec<Vec<F>>>, ShaProjectionError>
where
    F: PrimeField + 'a,
    I: IntoIterator<Item = &'a Vec<Vec<Vec<F>>>>,
{
    let tensors = tensors.into_iter().collect::<Vec<_>>();
    if tensors.len() != theta.len() {
        return Err(ShaProjectionError::FoldingWeightCount {
            got: theta.len(),
            expected: tensors.len(),
        });
    }
    let Some(first) = tensors.first() else {
        return Ok(Vec::new());
    };
    let mut out =
        vec![vec![vec![F::zero_with_cfg(field_cfg); SHA_WORD_BITS]; SHA_ROW_COUNT]; first.len()];
    for (tensor, weight) in tensors.iter().zip(theta) {
        if tensor.len() != first.len() {
            return Err(ShaProjectionError::InstanceCountMismatch {
                got: tensor.len(),
                expected: first.len(),
            });
        }
        for (col_idx, col) in tensor.iter().enumerate() {
            if col.len() != SHA_ROW_COUNT {
                return Err(ShaProjectionError::ColumnRowCount {
                    kind: "fold_3d",
                    col: col_idx,
                    got: col.len(),
                    expected: SHA_ROW_COUNT,
                });
            }
            for row in 0..SHA_ROW_COUNT {
                if col[row].len() != SHA_WORD_BITS {
                    return Err(ShaProjectionError::BitCount {
                        col: col_idx,
                        row,
                        got: col[row].len(),
                        expected: SHA_WORD_BITS,
                    });
                }
                for bit in 0..SHA_WORD_BITS {
                    out[col_idx][row][bit] += weight.clone() * &col[row][bit];
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sumcheck::multi_degree::MultiDegreeSumcheck;
    use crate::test_utils::test_config;
    use crypto_primitives::{FromWithConfig, crypto_bigint_monty::MontyField};
    use zinc_transcript::Blake3Transcript;

    type F = MontyField<4>;

    fn f(value: u64) -> F {
        F::from_with_cfg(value, &test_config())
    }

    fn zero_trace() -> ProjectedShaTrace<F> {
        let cfg = test_config();
        let zero = F::zero_with_cfg(&cfg);
        let bits = vec![vec![vec![zero.clone(); SHA_WORD_BITS]; SHA_ROW_COUNT]; ShaWordCol::COUNT];
        let bit_slices = ShaBitSliceColumns { columns: bits };
        let scalarized_words = scalarize_trace_words(&bit_slices, &f(5), &cfg).unwrap();
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
        let cfg = test_config();
        ProjectedShaPublic {
            columns: ShaPublicColumns {
                columns: vec![vec![F::zero_with_cfg(&cfg); SHA_ROW_COUNT]; ShaPublicCol::COUNT],
            },
        }
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
    fn scalarization_links_check_folded_words() {
        let cfg = test_config();
        let mut trace = zero_trace();
        trace.bit_slices.columns[ShaWordCol::A.index()][0][0] = f(1);
        trace.bit_slices.columns[ShaWordCol::A.index()][0][3] = f(1);
        trace.scalarized_words = scalarize_trace_words(&trace.bit_slices, &f(5), &cfg).unwrap();

        verify_folded_scalarization_links(&trace, &f(5), &[ShaWordCol::A], &cfg)
            .expect("scalarization should pass");

        trace.scalarized_words.words[ShaWordCol::A.index()][0] += f(1);
        assert!(matches!(
            verify_folded_scalarization_links(&trace, &f(5), &[ShaWordCol::A], &cfg),
            Err(ShaProjectionError::ScalarizationMismatch { .. })
        ));
    }

    #[test]
    fn scalarization_links_check_endpoint_and_shifted_sources() {
        let cfg = test_config();
        let mut trace = zero_trace();
        trace.bit_slices.columns[ShaWordCol::A.index()][0][1] = f(1);
        trace.bit_slices.columns[ShaWordCol::A.index()][1][0] = f(1);
        trace.bit_slices.columns[ShaWordCol::A.index()][1][2] = f(1);
        trace.scalarized_words = scalarize_trace_words(&trace.bit_slices, &f(3), &cfg).unwrap();
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

        trace.scalarized_words.words[ShaWordCol::A.index()][0] += f(1);
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
    fn sumfold_output_derives_theta_after_endpoint() {
        let cfg = test_config();
        let beta = vec![f(2), f(3)];
        let r_b = vec![f(5), f(7)];
        let c_sf = f(11);
        let out = finalize_sha_sumfold(&beta, r_b.clone(), c_sf.clone(), 4, &cfg).unwrap();
        let d = eq_eval(&beta, &r_b, F::one_with_cfg(&cfg)).unwrap();

        assert_eq!(out.t_prime(), &(c_sf / d));
        assert_eq!(out.theta(), build_eq_x_r_vec(&r_b, &cfg).unwrap());
    }

    #[test]
    fn folding_uses_sumfold_theta() {
        let cfg = test_config();
        let beta = vec![f(2)];
        let r_b = vec![f(3)];
        let out = finalize_sha_sumfold(&beta, r_b, f(9), 2, &cfg).unwrap();

        let mut left = zero_trace();
        let mut right = zero_trace();
        left.bit_slices.columns[ShaWordCol::A.index()][0][0] = f(1);
        right.bit_slices.columns[ShaWordCol::A.index()][0][0] = f(2);
        left.scalarized_words = scalarize_trace_words(&left.bit_slices, &f(5), &cfg).unwrap();
        right.scalarized_words = scalarize_trace_words(&right.bit_slices, &f(5), &cfg).unwrap();

        let (folded, _public) = fold_projected_sha_traces(
            &[left.clone(), right.clone()],
            &[zero_public(), zero_public()],
            &out,
            &cfg,
        )
        .unwrap();
        let expected = out.theta()[0].clone() * &left.bit_slices.columns[0][0][0]
            + out.theta()[1].clone() * &right.bit_slices.columns[0][0][0];
        assert_eq!(folded.trace.bit_slices.columns[0][0][0], expected);
    }

    #[test]
    fn virtual_ch_maj_reconstructs_from_source_bits() {
        let cfg = test_config();
        let mut trace = zero_trace();
        trace.bit_slices.columns[ShaWordCol::E.index()][2][0] = f(1);
        trace.bit_slices.columns[ShaWordCol::E.index()][1][0] = f(1);
        trace.bit_slices.columns[ShaWordCol::Uef.index()][2][0] = f(1);
        trace.bit_slices.columns[ShaWordCol::A.index()][0][1] = f(1);
        trace.bit_slices.columns[ShaWordCol::A.index()][1][1] = f(1);
        trace.bit_slices.columns[ShaWordCol::A.index()][2][1] = f(1);
        trace.bit_slices.columns[ShaWordCol::Maj.index()][2][1] = f(1);

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
