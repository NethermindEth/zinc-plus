use crate::neutron_nova::{RowWeights, sumfold::checked_domain_size};
use crate::sumcheck::{
    multi_degree::{MultiDegreeSumcheckGroup, PrefixFastPath, PrefixRoundOutput},
    prover::ProverState as SumcheckProverState,
};
use crypto_primitives::{FromPrimitiveWithConfig, PrimeField, crypto_bigint_uint::Uint};
use num_traits::Zero;
use std::array;
use thiserror::Error;
use zinc_poly::{
    mle::DenseMultilinearExtension,
    univariate::binary::BinaryPoly,
    utils::{build_eq_x_r_inner, build_eq_x_r_vec, eq_eval},
};
use zinc_uair::UairTrace;
use zinc_utils::{
    UNCHECKED,
    delayed_reduction::{
        BarrettDelayedReduction, DelayedFieldProductSum, DelayedModularReductionAlgorithm,
        MontgomeryLimbs, MontgomeryProductSum4,
    },
    inner_product::FieldFieldInnerProduct,
    inner_transparent_field::InnerTransparentField,
};

use super::{LinearPrefixTable, SumFoldError};

const PREFIX_TILE_SIZE: usize = 8;
/// Precomputed equality weights for the instance-axis SumFold split.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumFoldEqWeights<F: PrimeField> {
    pub prefix_eq_weights: Vec<F>,
    pub tail_eq_weights: Vec<F>,
}

/// Precomputed multiplication weights used by the linear CPR accumulator.
#[derive(Debug)]
pub struct LinearCprWeights<'a, F: PrimeField> {
    pub row_weights: &'a RowWeights<F>,
    pub tail_eq_weights: &'a [F],
}

impl<F: PrimeField> Clone for LinearCprWeights<'_, F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: PrimeField> Copy for LinearCprWeights<'_, F> {}

/// Precomputed scalar weights applied after row/tail DMR reduction.
#[derive(Debug)]
pub struct LinearCprScalarWeights<'a, F: PrimeField> {
    pub family_weights: &'a [F],
    pub scalarization_powers: &'a [F],
}

impl<F: PrimeField> Clone for LinearCprScalarWeights<'_, F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: PrimeField> Copy for LinearCprScalarWeights<'_, F> {}

/// One linear CPR family described as small-coefficient binary source terms.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearFamilySpec<F: PrimeField> {
    pub family_idx: usize,
    pub active_rows: Vec<usize>,
    pub terms: Vec<LinearTermSpec<F>>,
}

/// One binary source term in a linear CPR family.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearTermSpec<F: PrimeField> {
    pub source: LinearBinarySource,
    pub coeffs_by_active_row: Vec<CoeffClass<F>>,
}

/// Binary source read by a linear term.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LinearBinarySource {
    Column { col_idx: usize },
    ShiftedColumn { col_idx: usize, shift: usize },
}

impl LinearBinarySource {
    fn col_idx(&self) -> usize {
        match self {
            Self::Column { col_idx } | Self::ShiftedColumn { col_idx, .. } => *col_idx,
        }
    }
}

/// Coefficient class for post-DMR bucket weighting.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CoeffClass<F: PrimeField> {
    Zero,
    Small(i64),
    Large(F),
}

impl<F> CoeffClass<F>
where
    F: FromPrimitiveWithConfig,
{
    fn is_zero(&self) -> bool {
        matches!(self, Self::Zero | Self::Small(0))
    }

    fn to_field(&self, field_cfg: &F::Config) -> F {
        match self {
            Self::Zero => F::zero_with_cfg(field_cfg),
            Self::Small(value) => F::from_with_cfg(*value, field_cfg),
            Self::Large(value) => value.clone(),
        }
    }
}

#[derive(Clone, Debug, Error)]
pub enum LinearCprAccumulatorError {
    #[error("linear CPR accumulator needs at least one trace")]
    EmptyTraces,
    #[error("trace count must be a power of two, got {len}")]
    TraceCountNotPowerOfTwo { len: usize },
    #[error("prefix_vars={prefix_vars} must be at most ell={ell}")]
    PrefixVarsTooLarge { prefix_vars: usize, ell: usize },
    #[error("domain size is too large for {vars} variables")]
    DomainTooLarge { vars: usize },
    #[error("{label} length mismatch: got {got}, expected {expected}")]
    LengthMismatch {
        label: &'static str,
        got: usize,
        expected: usize,
    },
    #[error("trace {trace_idx} has {got} binary columns, expected {expected}")]
    BinaryColumnCountMismatch {
        trace_idx: usize,
        got: usize,
        expected: usize,
    },
    #[error("trace {trace_idx} binary column {col_idx} has {got} rows, expected {expected}")]
    BinaryColumnRowMismatch {
        trace_idx: usize,
        col_idx: usize,
        got: usize,
        expected: usize,
    },
    #[error(
        "family {family_idx} references family weight {weight_idx}, but only {len} weights exist"
    )]
    FamilyWeightOutOfRange {
        family_idx: usize,
        weight_idx: usize,
        len: usize,
    },
    #[error("family {family_idx} active row {row} is out of range for {rows} rows")]
    ActiveRowOutOfRange {
        family_idx: usize,
        row: usize,
        rows: usize,
    },
    #[error("family {family_idx} term {term_idx} has {got} coefficients, expected {expected}")]
    TermCoeffLengthMismatch {
        family_idx: usize,
        term_idx: usize,
        got: usize,
        expected: usize,
    },
    #[error(
        "family {family_idx} term {term_idx} references binary column {col_idx}, but only {cols} exist"
    )]
    SourceColumnOutOfRange {
        family_idx: usize,
        term_idx: usize,
        col_idx: usize,
        cols: usize,
    },
    #[error("sumfold helper failed: {0}")]
    SumFold(#[from] SumFoldError),
    #[error("equality table construction failed: {0}")]
    EqTable(#[from] zinc_poly::utils::ArithErrors),
}

#[derive(Clone, Debug)]
struct PreparedFamily<F: PrimeField> {
    family_idx: usize,
    active_rows: Vec<usize>,
    coeff_values: Vec<F>,
    terms: Vec<PreparedTerm>,
}

#[derive(Clone, Debug)]
struct PreparedTerm {
    source: LinearBinarySource,
    coeff_indices_by_active_row: Vec<Option<usize>>,
}

/// Build prefix/tail equality weights for a SumFold instance-axis split.
pub fn build_sumfold_eq_weights<F>(
    beta: &[F],
    prefix_vars: usize,
    field_cfg: &F::Config,
) -> Result<SumFoldEqWeights<F>, LinearCprAccumulatorError>
where
    F: PrimeField,
{
    let ell = beta.len();
    if prefix_vars > ell {
        return Err(LinearCprAccumulatorError::PrefixVarsTooLarge { prefix_vars, ell });
    }

    let prefix_eq_weights = if prefix_vars == 0 {
        vec![F::one_with_cfg(field_cfg)]
    } else {
        zinc_poly::utils::build_eq_x_r_vec(&beta[..prefix_vars], field_cfg)?
    };
    let tail_vars = ell - prefix_vars;
    let tail_eq_weights = if tail_vars == 0 {
        vec![F::one_with_cfg(field_cfg)]
    } else {
        zinc_poly::utils::build_eq_x_r_vec(&beta[prefix_vars..], field_cfg)?
    };

    Ok(SumFoldEqWeights {
        prefix_eq_weights,
        tail_eq_weights,
    })
}

/// Build the optimized linear CPR prefix table from small-value binary traces.
#[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
pub fn build_linear_cpr_prefix_table<F, PolyCoeff, Int, const D: usize>(
    traces: &[UairTrace<'_, PolyCoeff, Int, D>],
    prefix_vars: usize,
    families: &[LinearFamilySpec<F>],
    weights: LinearCprWeights<'_, F>,
    scalar_weights: LinearCprScalarWeights<'_, F>,
    field_cfg: &F::Config,
) -> Result<LinearPrefixTable<F>, LinearCprAccumulatorError>
where
    F: MontgomeryLimbs + DelayedFieldProductSum + FromPrimitiveWithConfig + Send + Sync + 'static,
    PolyCoeff: Clone,
    Int: Clone,
{
    let ell = validate_trace_count(traces.len())?;
    if prefix_vars > ell {
        return Err(LinearCprAccumulatorError::PrefixVarsTooLarge { prefix_vars, ell });
    }

    let prefix_len = checked_domain_size(prefix_vars)?;
    let tail_len = checked_domain_size(ell - prefix_vars)?;
    if weights.tail_eq_weights.len() != tail_len {
        return Err(LinearCprAccumulatorError::LengthMismatch {
            label: "tail_eq_weights",
            got: weights.tail_eq_weights.len(),
            expected: tail_len,
        });
    }
    if scalar_weights.scalarization_powers.len() < D {
        return Err(LinearCprAccumulatorError::LengthMismatch {
            label: "scalarization_powers",
            got: scalar_weights.scalarization_powers.len(),
            expected: D,
        });
    }

    let num_binary_cols = validate_traces(traces, weights.row_weights.len())?;
    let prepared = prepare_families(
        families,
        scalar_weights.family_weights.len(),
        num_binary_cols,
        weights.row_weights.len(),
        field_cfg,
    )?;

    let mut table_values = vec![F::zero_with_cfg(field_cfg); prefix_len];
    let reducer = BarrettDelayedReduction::<F>::new(field_cfg);

    for family in &prepared {
        let family_weight = &scalar_weights.family_weights[family.family_idx];
        if family.coeff_values.is_empty() {
            continue;
        }

        let mut tile_start = 0usize;
        while tile_start < prefix_len {
            let tile_len = PREFIX_TILE_SIZE.min(prefix_len - tile_start);
            accumulate_family_tile::<F, PolyCoeff, Int, D>(
                traces,
                family,
                tile_start,
                tile_len,
                prefix_vars,
                tail_len,
                &weights,
                scalar_weights.scalarization_powers,
                family_weight,
                field_cfg,
                &reducer,
                &mut table_values,
            )?;
            tile_start += tile_len;
        }
    }

    LinearPrefixTable::from_values_for_prefix_vars(table_values, ell, prefix_vars)
        .map_err(LinearCprAccumulatorError::from)
}

/// Build the post-prefix CPR claim table `claim(r_prefix, tail)`.
///
/// Unlike [`build_linear_cpr_prefix_table`], this does not apply tail equality
/// weights. The resumed ordinary sumcheck keeps
/// `eq(beta_prefix, r_prefix) * eq(beta_tail, tail)` as its separate equality
/// MLE, so this table only binds the CPR claim MLE along the prefix variables.
#[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
pub fn build_linear_cpr_prefix_bound_tail_table<F, PolyCoeff, Int, const D: usize>(
    traces: &[UairTrace<'_, PolyCoeff, Int, D>],
    prefix_vars: usize,
    prefix_point: &[F],
    families: &[LinearFamilySpec<F>],
    row_weights: &RowWeights<F>,
    scalar_weights: LinearCprScalarWeights<'_, F>,
    field_cfg: &F::Config,
) -> Result<DenseMultilinearExtension<F::Inner>, LinearCprAccumulatorError>
where
    F: InnerTransparentField + FromPrimitiveWithConfig,
    F::Inner: Zero,
    PolyCoeff: Clone,
    Int: Clone,
{
    let ell = validate_trace_count(traces.len())?;
    if prefix_vars > ell {
        return Err(LinearCprAccumulatorError::PrefixVarsTooLarge { prefix_vars, ell });
    }
    if prefix_point.len() != prefix_vars {
        return Err(LinearCprAccumulatorError::LengthMismatch {
            label: "prefix_point",
            got: prefix_point.len(),
            expected: prefix_vars,
        });
    }
    if scalar_weights.scalarization_powers.len() < D {
        return Err(LinearCprAccumulatorError::LengthMismatch {
            label: "scalarization_powers",
            got: scalar_weights.scalarization_powers.len(),
            expected: D,
        });
    }

    let prefix_len = checked_domain_size(prefix_vars)?;
    let tail_vars = ell - prefix_vars;
    let tail_len = checked_domain_size(tail_vars)?;
    let prefix_weights = if prefix_vars == 0 {
        vec![F::one_with_cfg(field_cfg)]
    } else {
        zinc_poly::utils::build_eq_x_r_vec(prefix_point, field_cfg)?
    };

    let num_binary_cols = validate_traces(traces, row_weights.len())?;
    let prepared = prepare_families(
        families,
        scalar_weights.family_weights.len(),
        num_binary_cols,
        row_weights.len(),
        field_cfg,
    )?;

    let zero = F::zero_with_cfg(field_cfg);
    let mut tail_values = vec![zero.clone(); tail_len];
    for (tail, tail_value) in tail_values.iter_mut().enumerate() {
        for (prefix, prefix_weight) in prefix_weights.iter().enumerate().take(prefix_len) {
            let instance_idx = prefix + (tail << prefix_vars);
            let trace = &traces[instance_idx];

            for family in &prepared {
                let family_weight = &scalar_weights.family_weights[family.family_idx];
                let mut family_value = zero.clone();

                for (active_pos, &row) in family.active_rows.iter().enumerate() {
                    let row_weight = &row_weights.as_slice()[row];
                    for term in &family.terms {
                        let Some(coeff_idx) = term.coeff_indices_by_active_row[active_pos] else {
                            continue;
                        };
                        let Some(poly) = source_poly::<PolyCoeff, Int, D>(trace, &term.source, row)
                        else {
                            continue;
                        };
                        let coeff_value = &family.coeff_values[coeff_idx];

                        for (bit_idx, bit) in poly.iter().enumerate().take(D) {
                            if bit.into_inner() {
                                let mut contribution = row_weight.clone();
                                contribution *= coeff_value;
                                contribution *= &scalar_weights.scalarization_powers[bit_idx];
                                family_value += contribution;
                            }
                        }
                    }
                }

                *tail_value += prefix_weight.clone() * family_weight * family_value;
            }
        }
    }

    let zero_inner = zero.inner().clone();
    Ok(DenseMultilinearExtension::from_evaluations_vec(
        tail_vars,
        tail_values
            .iter()
            .map(|value| value.inner().clone())
            .collect(),
        zero_inner,
    ))
}

struct LinearCprPrefixFastPath<
    F: PrimeField,
    PolyCoeff: Clone + 'static,
    Int: Clone + 'static,
    const D: usize,
> {
    traces: Vec<UairTrace<'static, PolyCoeff, Int, D>>,
    families: Vec<LinearFamilySpec<F>>,
    row_weights: RowWeights<F>,
    family_weights: Vec<F>,
    scalarization_powers: Vec<F>,
    beta: Vec<F>,
    prefix_vars: usize,
    prefix_state: SumcheckProverState<F>,
}

impl<F, PolyCoeff, Int, const D: usize> LinearCprPrefixFastPath<F, PolyCoeff, Int, D>
where
    F: MontgomeryLimbs
        + InnerTransparentField
        + DelayedFieldProductSum
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: Zero,
    PolyCoeff: Clone + Send + Sync + 'static,
    Int: Clone + Send + Sync + 'static,
{
    #[allow(clippy::too_many_arguments)]
    fn new(
        traces: Vec<UairTrace<'static, PolyCoeff, Int, D>>,
        prefix_vars: usize,
        beta: Vec<F>,
        families: Vec<LinearFamilySpec<F>>,
        row_weights: RowWeights<F>,
        family_weights: Vec<F>,
        scalarization_powers: Vec<F>,
        field_cfg: &F::Config,
    ) -> Result<Self, LinearCprAccumulatorError> {
        let eq_weights = build_sumfold_eq_weights(&beta, prefix_vars, field_cfg)?;
        let table = build_linear_cpr_prefix_table(
            &traces,
            prefix_vars,
            &families,
            LinearCprWeights {
                row_weights: &row_weights,
                tail_eq_weights: &eq_weights.tail_eq_weights,
            },
            LinearCprScalarWeights {
                family_weights: &family_weights,
                scalarization_powers: &scalarization_powers,
            },
            field_cfg,
        )?;
        let eq_prefix = build_eq_x_r_inner(&beta[..prefix_vars], field_cfg)?;
        let prefix_state =
            SumcheckProverState::new(vec![eq_prefix, table.to_mle(field_cfg)], prefix_vars, 2);

        Ok(Self {
            traces,
            families,
            row_weights,
            family_weights,
            scalarization_powers,
            beta,
            prefix_vars,
            prefix_state,
        })
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn finish_tail_mles(
        self,
        prefix_challenges: &[F],
        field_cfg: &F::Config,
    ) -> Vec<DenseMultilinearExtension<F::Inner>> {
        debug_assert_eq!(prefix_challenges.len(), self.prefix_vars);
        let tail_vars = self.beta.len() - self.prefix_vars;
        let beta_tail_weights = build_eq_x_r_vec(&self.beta[self.prefix_vars..], field_cfg)
            .expect("tail beta equality table should build");
        let eq_prefix_at_r = eq_eval(
            prefix_challenges,
            &self.beta[..self.prefix_vars],
            F::one_with_cfg(field_cfg),
        )
        .expect("prefix challenge and beta prefix lengths match");

        let tail_claims = build_linear_cpr_prefix_bound_tail_table(
            &self.traces,
            self.prefix_vars,
            prefix_challenges,
            &self.families,
            &self.row_weights,
            LinearCprScalarWeights {
                family_weights: &self.family_weights,
                scalarization_powers: &self.scalarization_powers,
            },
            field_cfg,
        )
        .expect("validated CPR tail table should build");

        let zero_inner = F::zero_with_cfg(field_cfg).inner().clone();
        let scaled_eq_tail = DenseMultilinearExtension::from_evaluations_vec(
            tail_vars,
            beta_tail_weights
                .iter()
                .map(|weight| (eq_prefix_at_r.clone() * weight).inner().clone())
                .collect(),
            zero_inner,
        );

        vec![scaled_eq_tail, tail_claims]
    }
}

impl<F, PolyCoeff, Int, const D: usize> PrefixFastPath<F>
    for LinearCprPrefixFastPath<F, PolyCoeff, Int, D>
where
    F: MontgomeryLimbs
        + InnerTransparentField
        + DelayedFieldProductSum
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: Zero,
    PolyCoeff: Clone + Send + Sync + 'static,
    Int: Clone + Send + Sync + 'static,
{
    fn prefix_len(&self) -> usize {
        self.prefix_vars
    }

    fn prove_prefix_round(
        &mut self,
        verifier_msg: &Option<F>,
        config: &F::Config,
    ) -> PrefixRoundOutput<F> {
        let msg = self.prefix_state.prove_round(
            verifier_msg,
            |values: &[F]| values[0].clone() * &values[1],
            config,
        );
        let asserted_sum = if self.prefix_state.round == 1 {
            self.prefix_state.asserted_sum.clone()
        } else {
            None
        };

        PrefixRoundOutput {
            asserted_sum,
            tail_evaluations: msg.0.tail_evaluations,
        }
    }

    fn finish_prefix(
        self: Box<Self>,
        prefix_challenges: &[F],
        config: &F::Config,
    ) -> Vec<DenseMultilinearExtension<F::Inner>> {
        self.finish_tail_mles(prefix_challenges, config)
    }
}

/// Build a hybrid SumFold group for linear CPR over owned traces.
///
/// `prefix_vars = 0` falls back to an ordinary full sumcheck group over dense
/// CPR claims. Positive prefixes emit CPR prefix-table sumcheck messages, bind
/// the prefix at the sampled challenges, and resume the ordinary degree-2
/// sumcheck over the tail variables.
#[allow(clippy::too_many_arguments)]
pub fn build_linear_cpr_hybrid_sumcheck_group<F, PolyCoeff, Int, const D: usize>(
    traces: Vec<UairTrace<'static, PolyCoeff, Int, D>>,
    prefix_vars: usize,
    beta: &[F],
    families: Vec<LinearFamilySpec<F>>,
    row_weights: RowWeights<F>,
    family_weights: Vec<F>,
    scalarization_powers: Vec<F>,
    field_cfg: &F::Config,
) -> Result<MultiDegreeSumcheckGroup<F>, LinearCprAccumulatorError>
where
    F: MontgomeryLimbs
        + InnerTransparentField
        + DelayedFieldProductSum
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: Zero,
    PolyCoeff: Clone + Send + Sync + 'static,
    Int: Clone + Send + Sync + 'static,
{
    let ell = validate_trace_count(traces.len())?;
    if beta.len() != ell {
        return Err(LinearCprAccumulatorError::LengthMismatch {
            label: "beta",
            got: beta.len(),
            expected: ell,
        });
    }
    if prefix_vars > ell {
        return Err(LinearCprAccumulatorError::PrefixVarsTooLarge { prefix_vars, ell });
    }
    if prefix_vars == 0 {
        let claims = build_linear_cpr_prefix_bound_tail_table(
            &traces,
            0,
            &[],
            &families,
            &row_weights,
            LinearCprScalarWeights {
                family_weights: &family_weights,
                scalarization_powers: &scalarization_powers,
            },
            field_cfg,
        )?;
        let eq_beta = build_eq_x_r_inner(beta, field_cfg)?;
        return Ok(MultiDegreeSumcheckGroup::new(
            2,
            vec![eq_beta, claims],
            Box::new(|values: &[F]| values[0].clone() * &values[1]),
        ));
    }
    if prefix_vars >= ell {
        return Err(LinearCprAccumulatorError::SumFold(
            SumFoldError::HybridPrefixNeedsTail {
                ell0: prefix_vars,
                ell,
            },
        ));
    }

    let fast_path = LinearCprPrefixFastPath::new(
        traces,
        prefix_vars,
        beta.to_vec(),
        families,
        row_weights,
        family_weights,
        scalarization_powers,
        field_cfg,
    )?;
    Ok(MultiDegreeSumcheckGroup::with_prefix_fast(
        2,
        Vec::new(),
        Box::new(|values: &[F]| values[0].clone() * &values[1]),
        Box::new(fast_path),
    ))
}

#[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
fn accumulate_family_tile<F, PolyCoeff, Int, const D: usize>(
    traces: &[UairTrace<'_, PolyCoeff, Int, D>],
    family: &PreparedFamily<F>,
    tile_start: usize,
    tile_len: usize,
    prefix_vars: usize,
    tail_len: usize,
    weights: &LinearCprWeights<'_, F>,
    scalarization_powers: &[F],
    family_weight: &F,
    field_cfg: &F::Config,
    reducer: &BarrettDelayedReduction<'_, F>,
    table_values: &mut [F],
) -> Result<(), LinearCprAccumulatorError>
where
    F: MontgomeryLimbs + DelayedFieldProductSum + FromPrimitiveWithConfig + Send + Sync + 'static,
    PolyCoeff: Clone,
    Int: Clone,
{
    let bucket_count = tile_len
        .checked_mul(family.coeff_values.len())
        .ok_or(LinearCprAccumulatorError::DomainTooLarge { vars: prefix_vars })?;
    let mut buckets: Vec<[Uint<5>; D]> = vec![[Uint::zero(); D]; bucket_count];
    let zero = F::zero_with_cfg(field_cfg);
    let mut lane_accs: Vec<[F; D]> = (0..bucket_count)
        .map(|_| array::from_fn(|_| zero.clone()))
        .collect();
    let product_sum = MontgomeryProductSum4::<F>::new(field_cfg);
    let mut pending_adds = 0usize;
    let flush_adds = reducer.flush_adds();

    for tail in 0..tail_len {
        let tail_weight = &weights.tail_eq_weights[tail];
        for (active_pos, &row) in family.active_rows.iter().enumerate() {
            let omega = tail_weight.clone() * &weights.row_weights.as_slice()[row];

            for term in &family.terms {
                let Some(coeff_idx) = term.coeff_indices_by_active_row[active_pos] else {
                    continue;
                };

                for prefix_offset in 0..tile_len {
                    let prefix = tile_start + prefix_offset;
                    let instance_idx = prefix + (tail << prefix_vars);
                    let Some(poly) =
                        source_poly::<PolyCoeff, Int, D>(&traces[instance_idx], &term.source, row)
                    else {
                        continue;
                    };
                    let bucket_idx = prefix_offset * family.coeff_values.len() + coeff_idx;
                    pending_adds = pending_adds.saturating_add(add_poly_bits_to_bucket(
                        poly,
                        &omega,
                        reducer,
                        &mut buckets[bucket_idx],
                    ));

                    if pending_adds >= flush_adds {
                        flush_buckets_into_lanes(&mut buckets, &mut lane_accs, reducer);
                        pending_adds = 0;
                    }
                }
            }
        }
    }

    flush_buckets_into_lanes(&mut buckets, &mut lane_accs, reducer);

    for prefix_offset in 0..tile_len {
        let prefix = tile_start + prefix_offset;
        let mut family_value = zero.clone();
        for (coeff_idx, coeff_value) in family.coeff_values.iter().enumerate() {
            let bucket_idx = prefix_offset * family.coeff_values.len() + coeff_idx;
            let projected = FieldFieldInnerProduct::inner_product_with_algorithm::<UNCHECKED, _>(
                &product_sum,
                &lane_accs[bucket_idx],
                &scalarization_powers[..D],
                zero.clone(),
            )
            .expect("lane accumulator and scalarization powers have matching lengths");
            family_value += coeff_value.clone() * projected;
        }
        table_values[prefix] += family_weight.clone() * family_value;
    }

    Ok(())
}

fn validate_trace_count(len: usize) -> Result<usize, LinearCprAccumulatorError> {
    if len == 0 {
        return Err(LinearCprAccumulatorError::EmptyTraces);
    }
    if !len.is_power_of_two() {
        return Err(LinearCprAccumulatorError::TraceCountNotPowerOfTwo { len });
    }
    Ok(usize::try_from(len.trailing_zeros()).expect("trailing_zeros fits usize"))
}

fn validate_traces<PolyCoeff, Int, const D: usize>(
    traces: &[UairTrace<'_, PolyCoeff, Int, D>],
    expected_rows: usize,
) -> Result<usize, LinearCprAccumulatorError>
where
    PolyCoeff: Clone,
    Int: Clone,
{
    let num_binary_cols = traces
        .first()
        .expect("trace count was already validated as non-empty")
        .binary_poly
        .len();
    for (trace_idx, trace) in traces.iter().enumerate() {
        if trace.binary_poly.len() != num_binary_cols {
            return Err(LinearCprAccumulatorError::BinaryColumnCountMismatch {
                trace_idx,
                got: trace.binary_poly.len(),
                expected: num_binary_cols,
            });
        }
        for (col_idx, column) in trace.binary_poly.iter().enumerate() {
            if column.evaluations.len() != expected_rows {
                return Err(LinearCprAccumulatorError::BinaryColumnRowMismatch {
                    trace_idx,
                    col_idx,
                    got: column.evaluations.len(),
                    expected: expected_rows,
                });
            }
        }
    }
    Ok(num_binary_cols)
}

fn prepare_families<F>(
    families: &[LinearFamilySpec<F>],
    family_weight_len: usize,
    num_binary_cols: usize,
    row_count: usize,
    field_cfg: &F::Config,
) -> Result<Vec<PreparedFamily<F>>, LinearCprAccumulatorError>
where
    F: FromPrimitiveWithConfig,
{
    let mut prepared = Vec::with_capacity(families.len());
    for family in families {
        if family.family_idx >= family_weight_len {
            return Err(LinearCprAccumulatorError::FamilyWeightOutOfRange {
                family_idx: family.family_idx,
                weight_idx: family.family_idx,
                len: family_weight_len,
            });
        }
        for &row in &family.active_rows {
            if row >= row_count {
                return Err(LinearCprAccumulatorError::ActiveRowOutOfRange {
                    family_idx: family.family_idx,
                    row,
                    rows: row_count,
                });
            }
        }

        let mut coeff_classes = Vec::<CoeffClass<F>>::new();
        let mut terms = Vec::with_capacity(family.terms.len());
        for (term_idx, term) in family.terms.iter().enumerate() {
            let col_idx = term.source.col_idx();
            if col_idx >= num_binary_cols {
                return Err(LinearCprAccumulatorError::SourceColumnOutOfRange {
                    family_idx: family.family_idx,
                    term_idx,
                    col_idx,
                    cols: num_binary_cols,
                });
            }
            if term.coeffs_by_active_row.len() != family.active_rows.len() {
                return Err(LinearCprAccumulatorError::TermCoeffLengthMismatch {
                    family_idx: family.family_idx,
                    term_idx,
                    got: term.coeffs_by_active_row.len(),
                    expected: family.active_rows.len(),
                });
            }

            let mut coeff_indices = Vec::with_capacity(term.coeffs_by_active_row.len());
            for coeff in &term.coeffs_by_active_row {
                if coeff.is_zero() {
                    coeff_indices.push(None);
                    continue;
                }
                let idx = match coeff_classes.iter().position(|existing| existing == coeff) {
                    Some(idx) => idx,
                    None => {
                        coeff_classes.push(coeff.clone());
                        coeff_classes.len() - 1
                    }
                };
                coeff_indices.push(Some(idx));
            }
            terms.push(PreparedTerm {
                source: term.source.clone(),
                coeff_indices_by_active_row: coeff_indices,
            });
        }

        let coeff_values = coeff_classes
            .iter()
            .map(|coeff| coeff.to_field(field_cfg))
            .collect();

        prepared.push(PreparedFamily {
            family_idx: family.family_idx,
            active_rows: family.active_rows.clone(),
            coeff_values,
            terms,
        });
    }
    Ok(prepared)
}

fn source_poly<'a, PolyCoeff, Int, const D: usize>(
    trace: &'a UairTrace<'_, PolyCoeff, Int, D>,
    source: &LinearBinarySource,
    row: usize,
) -> Option<&'a BinaryPoly<D>>
where
    PolyCoeff: Clone,
    Int: Clone,
{
    match source {
        LinearBinarySource::Column { col_idx } => trace.binary_poly[*col_idx].evaluations.get(row),
        LinearBinarySource::ShiftedColumn { col_idx, shift } => row
            .checked_add(*shift)
            .and_then(|shifted_row| trace.binary_poly[*col_idx].evaluations.get(shifted_row)),
    }
}

#[allow(clippy::arithmetic_side_effects)]
fn add_poly_bits_to_bucket<F, const D: usize>(
    poly: &BinaryPoly<D>,
    weight: &F,
    reducer: &BarrettDelayedReduction<'_, F>,
    bucket: &mut [Uint<5>; D],
) -> usize
where
    F: MontgomeryLimbs + Send + Sync,
{
    if D <= 64 {
        let mut bits = 0u64;
        for (bit_idx, coeff) in poly.iter().enumerate().take(D) {
            if coeff.into_inner() {
                bits |= 1u64 << bit_idx;
            }
        }

        let mut adds = 0usize;
        while bits != 0 {
            let bit_idx =
                usize::try_from(bits.trailing_zeros()).expect("trailing_zeros fits usize");
            reducer.add(&mut bucket[bit_idx], weight);
            bits &= bits - 1;
            adds += 1;
        }
        adds
    } else {
        let mut adds = 0usize;
        for (bit_idx, coeff) in poly.iter().enumerate().take(D) {
            if coeff.into_inner() {
                reducer.add(&mut bucket[bit_idx], weight);
                adds += 1;
            }
        }
        adds
    }
}

fn flush_buckets_into_lanes<F, const D: usize>(
    buckets: &mut [[Uint<5>; D]],
    lane_accs: &mut [[F; D]],
    reducer: &BarrettDelayedReduction<'_, F>,
) where
    F: MontgomeryLimbs + Send + Sync,
{
    for (bucket_lanes, acc_lanes) in buckets.iter_mut().zip(lane_accs.iter_mut()) {
        for (bucket, acc) in bucket_lanes.iter_mut().zip(acc_lanes.iter_mut()) {
            if bucket.is_zero() {
                continue;
            }
            let pending = std::mem::replace(bucket, Uint::zero());
            *acc += reducer.reduce(pending);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        neutron_nova::LinearInstanceClaims, sumcheck::multi_degree::MultiDegreeSumcheck,
        test_utils::test_config,
    };
    use crypto_primitives::{FromWithConfig, crypto_bigint_monty::MontyField};
    use std::borrow::Cow;
    use zinc_poly::{
        mle::{DenseMultilinearExtension, MultilinearExtensionWithConfig},
        utils::{build_eq_x_r_vec, eq_eval},
    };
    use zinc_transcript::Blake3Transcript;
    use zinc_utils::powers;

    type F = MontyField<4>;
    type Trace = UairTrace<'static, F, F, 32>;

    fn f(value: u64) -> F {
        F::from_with_cfg(value, &test_config())
    }

    fn binary_column(patterns: &[u32]) -> DenseMultilinearExtension<BinaryPoly<32>> {
        assert!(patterns.len().is_power_of_two());
        DenseMultilinearExtension::from_evaluations_vec(
            usize::try_from(patterns.len().trailing_zeros()).expect("trailing_zeros fits usize"),
            patterns
                .iter()
                .copied()
                .map(BinaryPoly::<32>::from)
                .collect(),
            BinaryPoly::<32>::zero(),
        )
    }

    fn trace_from_columns(col0: &[u32], col1: &[u32]) -> Trace {
        UairTrace {
            binary_poly: Cow::Owned(vec![binary_column(col0), binary_column(col1)]),
            arbitrary_poly: Cow::Owned(Vec::new()),
            int: Cow::Owned(Vec::new()),
        }
    }

    fn sample_traces() -> Vec<Trace> {
        vec![
            trace_from_columns(
                &[0x0000_0001, 0x0000_0002, 0x8000_0001, 0x0001_0010],
                &[0x0000_0005, 0x0000_000a, 0x0000_0101, 0x0000_1000],
            ),
            trace_from_columns(
                &[0x0000_0003, 0x0000_0010, 0x0000_00f0, 0x0000_f000],
                &[0x0000_0006, 0x0000_0009, 0x0000_0f00, 0x0000_00ff],
            ),
            trace_from_columns(
                &[0x0000_0100, 0x0000_0201, 0x0000_0402, 0x0000_0804],
                &[0x0000_0011, 0x0000_0022, 0x0000_0044, 0x0000_0088],
            ),
            trace_from_columns(
                &[0x0000_aaaa, 0x0000_5555, 0x0000_3333, 0x0000_cccc],
                &[0x0000_1234, 0x0000_4321, 0x0000_00f1, 0x0000_0f10],
            ),
        ]
    }

    fn sample_families() -> Vec<LinearFamilySpec<F>> {
        vec![
            LinearFamilySpec {
                family_idx: 0,
                active_rows: vec![0, 1, 2],
                terms: vec![
                    LinearTermSpec {
                        source: LinearBinarySource::Column { col_idx: 0 },
                        coeffs_by_active_row: vec![
                            CoeffClass::Small(1),
                            CoeffClass::Small(-1),
                            CoeffClass::Zero,
                        ],
                    },
                    LinearTermSpec {
                        source: LinearBinarySource::Column { col_idx: 1 },
                        coeffs_by_active_row: vec![
                            CoeffClass::Small(2),
                            CoeffClass::Small(1),
                            CoeffClass::Small(-2),
                        ],
                    },
                ],
            },
            LinearFamilySpec {
                family_idx: 1,
                active_rows: vec![1, 3],
                terms: vec![
                    LinearTermSpec {
                        source: LinearBinarySource::Column { col_idx: 0 },
                        coeffs_by_active_row: vec![CoeffClass::Large(f(7)), CoeffClass::Small(-1)],
                    },
                    LinearTermSpec {
                        source: LinearBinarySource::ShiftedColumn {
                            col_idx: 1,
                            shift: 1,
                        },
                        coeffs_by_active_row: vec![CoeffClass::Small(1), CoeffClass::Small(3)],
                    },
                ],
            },
        ]
    }

    fn scalar_weights() -> Vec<F> {
        powers(f(3), F::one_with_cfg(&test_config()), 32)
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn naive_linear_cpr_claims(
        traces: &[Trace],
        families: &[LinearFamilySpec<F>],
        row_weights: &RowWeights<F>,
        scalar_weights: LinearCprScalarWeights<'_, F>,
    ) -> Vec<F> {
        let cfg = test_config();
        let mut claims = vec![F::zero_with_cfg(&cfg); traces.len()];

        for (trace, claim) in traces.iter().zip(claims.iter_mut()) {
            for family in families {
                let mut family_value = F::zero_with_cfg(&cfg);
                for (active_pos, &row) in family.active_rows.iter().enumerate() {
                    for term in &family.terms {
                        let coeff = &term.coeffs_by_active_row[active_pos];
                        if coeff.is_zero() {
                            continue;
                        }
                        let coeff_value = coeff.to_field(&cfg);
                        let Some(poly) = source_poly::<F, F, 32>(trace, &term.source, row) else {
                            continue;
                        };

                        for (bit_idx, bit) in poly.iter().enumerate().take(32) {
                            if bit.into_inner() {
                                let mut contribution = row_weights.as_slice()[row].clone();
                                contribution *= &coeff_value;
                                contribution *= &scalar_weights.scalarization_powers[bit_idx];
                                family_value += contribution;
                            }
                        }
                    }
                }
                *claim += scalar_weights.family_weights[family.family_idx].clone() * family_value;
            }
        }

        claims
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn naive_linear_cpr_table(
        traces: &[Trace],
        prefix_vars: usize,
        families: &[LinearFamilySpec<F>],
        weights: LinearCprWeights<'_, F>,
        scalar_weights: LinearCprScalarWeights<'_, F>,
    ) -> Vec<F> {
        let cfg = test_config();
        let ell =
            usize::try_from(traces.len().trailing_zeros()).expect("trailing_zeros fits usize");
        let prefix_len = 1usize << prefix_vars;
        let tail_len = 1usize << (ell - prefix_vars);
        let claims = naive_linear_cpr_claims(traces, families, weights.row_weights, scalar_weights);
        let mut table = vec![F::zero_with_cfg(&cfg); prefix_len];

        for (prefix, value) in table.iter_mut().enumerate() {
            for tail in 0..tail_len {
                let instance_idx = prefix + (tail << prefix_vars);
                *value += weights.tail_eq_weights[tail].clone() * &claims[instance_idx];
            }
        }

        table
    }

    fn build_table_for_prefix_vars(
        prefix_vars: usize,
    ) -> (LinearPrefixTable<F>, SumFoldEqWeights<F>) {
        let cfg = test_config();
        let traces = sample_traces();
        let families = sample_families();
        let beta = vec![f(19), f(23)];
        let eq_weights = build_sumfold_eq_weights(&beta, prefix_vars, &cfg).unwrap();
        let row_weights = RowWeights::new(&[f(11), f(13)], &cfg).unwrap();
        let family_weights = vec![f(5), f(17)];
        let scalarization_powers = scalar_weights();

        let table = build_linear_cpr_prefix_table(
            &traces,
            prefix_vars,
            &families,
            LinearCprWeights {
                row_weights: &row_weights,
                tail_eq_weights: &eq_weights.tail_eq_weights,
            },
            LinearCprScalarWeights {
                family_weights: &family_weights,
                scalarization_powers: &scalarization_powers,
            },
            &cfg,
        )
        .unwrap();

        let expected = naive_linear_cpr_table(
            &traces,
            prefix_vars,
            &families,
            LinearCprWeights {
                row_weights: &row_weights,
                tail_eq_weights: &eq_weights.tail_eq_weights,
            },
            LinearCprScalarWeights {
                family_weights: &family_weights,
                scalarization_powers: &scalarization_powers,
            },
        );
        assert_eq!(table.values(), expected.as_slice());

        (table, eq_weights)
    }

    #[test]
    fn optimized_linear_cpr_table_matches_expanded_formula_for_live_prefix() {
        let (table, eq_weights) = build_table_for_prefix_vars(2);

        assert_eq!(table.ell(), 2);
        assert_eq!(table.ell0(), 2);
        assert_eq!(
            eq_weights.tail_eq_weights,
            vec![F::one_with_cfg(&test_config())]
        );
    }

    #[test]
    fn optimized_linear_cpr_table_matches_expanded_formula_for_windowed_prefix() {
        let (table, eq_weights) = build_table_for_prefix_vars(1);

        assert_eq!(table.ell(), 2);
        assert_eq!(table.ell0(), 1);
        assert_eq!(table.len(), 2);
        assert_eq!(eq_weights.tail_eq_weights.len(), 2);
    }

    #[test]
    fn optimized_linear_cpr_prefix_bound_tail_table_matches_naive_claim_binding() {
        let cfg = test_config();
        let traces = sample_traces();
        let families = sample_families();
        let prefix_vars = 1;
        let prefix_point = vec![f(29)];
        let row_weights = RowWeights::new(&[f(11), f(13)], &cfg).unwrap();
        let family_weights = vec![f(5), f(17)];
        let scalarization_powers = scalar_weights();
        let scalar_weights = LinearCprScalarWeights {
            family_weights: &family_weights,
            scalarization_powers: &scalarization_powers,
        };

        let tail_mle = build_linear_cpr_prefix_bound_tail_table(
            &traces,
            prefix_vars,
            &prefix_point,
            &families,
            &row_weights,
            scalar_weights,
            &cfg,
        )
        .unwrap();
        let dense_claims =
            naive_linear_cpr_claims(&traces, &families, &row_weights, scalar_weights);
        let prefix_weights = build_eq_x_r_vec(&prefix_point, &cfg).unwrap();

        let tail_len = 1usize << (2 - prefix_vars);
        let mut expected = vec![F::zero_with_cfg(&cfg); tail_len];
        for (tail, value) in expected.iter_mut().enumerate() {
            for (prefix, weight) in prefix_weights.iter().enumerate() {
                let instance_idx = prefix + (tail << prefix_vars);
                *value += weight.clone() * &dense_claims[instance_idx];
            }
        }

        assert_eq!(tail_mle.num_vars, 1);
        assert_eq!(
            tail_mle.evaluations,
            expected
                .iter()
                .map(|value| value.inner().clone())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn optimized_linear_cpr_hybrid_sumfold_matches_full_ordinary_sumcheck() {
        let cfg = test_config();
        let traces = sample_traces();
        let families = sample_families();
        let beta = vec![f(19), f(23)];
        let row_weights = RowWeights::new(&[f(11), f(13)], &cfg).unwrap();
        let family_weights = vec![f(5), f(17)];
        let scalarization_powers = scalar_weights();
        let scalar_weights = LinearCprScalarWeights {
            family_weights: &family_weights,
            scalarization_powers: &scalarization_powers,
        };
        let dense_claims =
            naive_linear_cpr_claims(&traces, &families, &row_weights, scalar_weights);
        let claims = LinearInstanceClaims::from_claims_for_ell(dense_claims, 2).unwrap();

        let full_group = claims.build_full_sumcheck_group(&beta, &cfg).unwrap();
        let optimized_group = build_linear_cpr_hybrid_sumcheck_group(
            traces.clone(),
            1,
            &beta,
            families.clone(),
            row_weights.clone(),
            family_weights.clone(),
            scalarization_powers.clone(),
            &cfg,
        )
        .unwrap();
        let fallback_group = build_linear_cpr_hybrid_sumcheck_group(
            traces,
            0,
            &beta,
            families,
            row_weights,
            family_weights,
            scalarization_powers,
            &cfg,
        )
        .unwrap();

        let mut prover_transcript = Blake3Transcript::new();
        let (full_proof, _states) = MultiDegreeSumcheck::prove_as_subprotocol(
            &mut prover_transcript,
            vec![full_group],
            2,
            &cfg,
        );
        let mut prover_transcript = Blake3Transcript::new();
        let (optimized_proof, _states) = MultiDegreeSumcheck::prove_as_subprotocol(
            &mut prover_transcript,
            vec![optimized_group],
            2,
            &cfg,
        );
        assert_eq!(optimized_proof, full_proof);
        let mut prover_transcript = Blake3Transcript::new();
        let (fallback_proof, _states) = MultiDegreeSumcheck::prove_as_subprotocol(
            &mut prover_transcript,
            vec![fallback_group],
            2,
            &cfg,
        );
        assert_eq!(fallback_proof, full_proof);

        let mut verifier_transcript = Blake3Transcript::new();
        let full_subclaims = MultiDegreeSumcheck::verify_as_subprotocol(
            &mut verifier_transcript,
            2,
            &full_proof,
            &cfg,
        )
        .expect("full linear CPR sumcheck should verify");
        let mut verifier_transcript = Blake3Transcript::new();
        let optimized_subclaims = MultiDegreeSumcheck::verify_as_subprotocol(
            &mut verifier_transcript,
            2,
            &optimized_proof,
            &cfg,
        )
        .expect("optimized linear CPR sumcheck should verify");
        assert_eq!(optimized_subclaims.point(), full_subclaims.point());
        assert_eq!(
            optimized_subclaims.expected_evaluations(),
            full_subclaims.expected_evaluations()
        );

        let point = full_subclaims.point();
        let eq_at_point =
            eq_eval(point, &beta, F::one_with_cfg(&cfg)).expect("same number of variables");
        let zero_inner = F::zero_with_cfg(&cfg).inner().clone();
        let claims_mle = DenseMultilinearExtension::from_evaluations_vec(
            2,
            claims
                .claims()
                .iter()
                .map(|value| value.inner().clone())
                .collect(),
            zero_inner,
        );
        let claim_eval = claims_mle.evaluate_with_config(point, &cfg).unwrap();
        assert_eq!(
            full_subclaims.expected_evaluations()[0],
            eq_at_point * claim_eval
        );
    }

    #[test]
    fn optimized_linear_cpr_flushes_dmr_for_small_modulus() {
        let cfg = test_config();
        let trace_count = 2048usize;
        let traces: Vec<_> = (0..trace_count)
            .map(|_| trace_from_columns(&[u32::MAX], &[0]))
            .collect();
        let families = vec![LinearFamilySpec {
            family_idx: 0,
            active_rows: vec![0],
            terms: vec![LinearTermSpec {
                source: LinearBinarySource::Column { col_idx: 0 },
                coeffs_by_active_row: vec![CoeffClass::Small(1)],
            }],
        }];
        let row_weights = RowWeights::new(&[], &cfg).unwrap();
        let max = -F::from_with_cfg(1u64, &cfg);
        let tail_eq_weights = vec![max; trace_count];
        let family_weights = vec![F::one_with_cfg(&cfg)];
        let scalarization_powers = scalar_weights();
        let weights = LinearCprWeights {
            row_weights: &row_weights,
            tail_eq_weights: &tail_eq_weights,
        };
        let scalar_weights = LinearCprScalarWeights {
            family_weights: &family_weights,
            scalarization_powers: &scalarization_powers,
        };

        let table =
            build_linear_cpr_prefix_table(&traces, 0, &families, weights, scalar_weights, &cfg)
                .unwrap();
        let expected = naive_linear_cpr_table(&traces, 0, &families, weights, scalar_weights);

        assert_eq!(table.values(), expected.as_slice());
    }

    #[test]
    fn optimized_linear_cpr_validation_errors_are_reported() {
        let cfg = test_config();
        let traces = sample_traces();
        let families = sample_families();
        let row_weights = RowWeights::new(&[f(11), f(13)], &cfg).unwrap();
        let family_weights = vec![f(5), f(17)];
        let scalarization_powers = scalar_weights();

        let err = build_linear_cpr_prefix_table(
            &traces,
            3,
            &families,
            LinearCprWeights {
                row_weights: &row_weights,
                tail_eq_weights: &[F::one_with_cfg(&cfg)],
            },
            LinearCprScalarWeights {
                family_weights: &family_weights,
                scalarization_powers: &scalarization_powers,
            },
            &cfg,
        )
        .expect_err("too many prefix variables should be rejected");
        assert!(matches!(
            err,
            LinearCprAccumulatorError::PrefixVarsTooLarge {
                prefix_vars: 3,
                ell: 2
            }
        ));

        let err = build_linear_cpr_prefix_table(
            &traces,
            2,
            &families,
            LinearCprWeights {
                row_weights: &row_weights,
                tail_eq_weights: &[],
            },
            LinearCprScalarWeights {
                family_weights: &family_weights,
                scalarization_powers: &scalarization_powers,
            },
            &cfg,
        )
        .expect_err("wrong tail weight length should be rejected");
        assert!(matches!(
            err,
            LinearCprAccumulatorError::LengthMismatch {
                label: "tail_eq_weights",
                got: 0,
                expected: 1
            }
        ));

        let mut bad_families = sample_families();
        bad_families[0].terms[0].coeffs_by_active_row.pop();
        let eq_weights = build_sumfold_eq_weights(&[f(19), f(23)], 2, &cfg).unwrap();
        let err = build_linear_cpr_prefix_table(
            &traces,
            2,
            &bad_families,
            LinearCprWeights {
                row_weights: &row_weights,
                tail_eq_weights: &eq_weights.tail_eq_weights,
            },
            LinearCprScalarWeights {
                family_weights: &family_weights,
                scalarization_powers: &scalarization_powers,
            },
            &cfg,
        )
        .expect_err("wrong coefficient vector length should be rejected");
        assert!(matches!(
            err,
            LinearCprAccumulatorError::TermCoeffLengthMismatch {
                family_idx: 0,
                term_idx: 0,
                got: 2,
                expected: 3
            }
        ));
    }
}
