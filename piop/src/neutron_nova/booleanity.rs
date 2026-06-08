use crate::neutron_nova::RowWeights;
use crypto_primitives::{FromPrimitiveWithConfig, PrimeField, crypto_bigint_uint::Uint};
use num_traits::Zero;
use std::array;
use thiserror::Error;
use zinc_poly::univariate::binary::BinaryPoly;
use zinc_uair::UairTrace;
use zinc_utils::{
    UNCHECKED,
    delayed_reduction::{
        BarrettDelayedReduction, DelayedFieldProductSum, DelayedModularReductionAlgorithm,
        MontgomeryLimbs, MontgomeryProductSum4,
    },
    inner_product::FieldFieldInnerProduct,
};

const MAX_DMR_BUCKET_ARRAYS: usize = 256;
const MAX_BOOLEANITY_PREFIX_VARS: usize = 10;

/// Precomputed equality weights used by the booleanity accumulator.
#[derive(Debug)]
pub struct BooleanityWeights<'a, F: PrimeField> {
    pub row_weights: &'a RowWeights<F>,
    pub tail_eq_weights: &'a [F],
}

impl<F: PrimeField> Clone for BooleanityWeights<'_, F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: PrimeField> Copy for BooleanityWeights<'_, F> {}

/// Precomputed scalarization weights for booleanity lanes.
#[derive(Debug)]
pub struct BooleanityScalarWeights<'a, F: PrimeField> {
    /// Indexed as `col_idx * D + bit_idx`.
    pub rho_powers: &'a [F],
}

impl<F: PrimeField> Clone for BooleanityScalarWeights<'_, F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: PrimeField> Copy for BooleanityScalarWeights<'_, F> {}

/// One point in `{0, 1, infinity}^prefix_vars`, represented as `(S, a)`.
///
/// `support_mask` marks coordinates set to infinity. `finite_bits` stores
/// Boolean assignments in original coordinate positions. The two masks are
/// canonical and must not overlap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExtendedPrefixPoint {
    support_mask: usize,
    finite_bits: usize,
}

impl ExtendedPrefixPoint {
    pub fn new(
        support_mask: usize,
        finite_bits: usize,
    ) -> Result<Self, BooleanityAccumulatorError> {
        if support_mask & finite_bits != 0 {
            return Err(BooleanityAccumulatorError::ExtendedPointNotCanonical);
        }
        Ok(Self {
            support_mask,
            finite_bits,
        })
    }

    pub fn support_mask(self) -> usize {
        self.support_mask
    }

    pub fn finite_bits(self) -> usize {
        self.finite_bits
    }

    pub fn support_size(self) -> usize {
        usize::try_from(self.support_mask.count_ones()).expect("count_ones fits usize")
    }

    pub fn is_finite_only(self) -> bool {
        self.support_mask == 0
    }
}

/// Dense table over `{0, 1, infinity}^prefix_vars`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BooleanityPrefixTable<F: PrimeField> {
    values: Vec<F>,
    ell: usize,
    prefix_vars: usize,
    num_binary_cols: usize,
}

impl<F: PrimeField> BooleanityPrefixTable<F> {
    pub fn values(&self) -> &[F] {
        &self.values
    }

    pub fn ell(&self) -> usize {
        self.ell
    }

    pub fn prefix_vars(&self) -> usize {
        self.prefix_vars
    }

    pub fn num_binary_cols(&self) -> usize {
        self.num_binary_cols
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn value_at_point(
        &self,
        point: ExtendedPrefixPoint,
    ) -> Result<&F, BooleanityAccumulatorError> {
        let index = extended_point_index(point, self.prefix_vars)?;
        self.values
            .get(index)
            .ok_or(BooleanityAccumulatorError::ExtendedPointIndexOutOfRange {
                index,
                domain_size: self.values.len(),
            })
    }
}

#[derive(Clone, Debug, Error)]
pub enum BooleanityAccumulatorError {
    #[error("booleanity accumulator needs at least one trace")]
    EmptyTraces,
    #[error("trace count must be a power of two, got {len}")]
    TraceCountNotPowerOfTwo { len: usize },
    #[error("prefix_vars={prefix_vars} must be at most ell={ell}")]
    PrefixVarsTooLarge { prefix_vars: usize, ell: usize },
    #[error("booleanity prefix_vars={prefix_vars} exceeds supported maximum {max}")]
    PrefixVarsExceedsSupported { prefix_vars: usize, max: usize },
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
    #[error("extended point index {index} is out of range for domain size {domain_size}")]
    ExtendedPointIndexOutOfRange { index: usize, domain_size: usize },
    #[error("extended point uses bits outside prefix_vars={prefix_vars}")]
    ExtendedPointOutOfRange { prefix_vars: usize },
    #[error("extended point has finite bits set inside the infinity support")]
    ExtendedPointNotCanonical,
    #[error("accumulator bucket count overflow for {entries} entries and stride {stride}")]
    BucketCountOverflow { entries: usize, stride: usize },
}

#[derive(Clone, Copy, Debug)]
struct ExtendedTableEntry {
    table_index: usize,
    point: ExtendedPrefixPoint,
}

/// Build the optimized booleanity prefix table from small-value binary traces.
#[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
pub fn build_booleanity_prefix_table<F, PolyCoeff, Int, const D: usize>(
    traces: &[UairTrace<'_, PolyCoeff, Int, D>],
    prefix_vars: usize,
    weights: BooleanityWeights<'_, F>,
    scalar_weights: BooleanityScalarWeights<'_, F>,
    field_cfg: &F::Config,
) -> Result<BooleanityPrefixTable<F>, BooleanityAccumulatorError>
where
    F: MontgomeryLimbs + DelayedFieldProductSum + FromPrimitiveWithConfig + Send + Sync + 'static,
    PolyCoeff: Clone,
    Int: Clone,
{
    let ell = validate_trace_count(traces.len())?;
    if prefix_vars > ell {
        return Err(BooleanityAccumulatorError::PrefixVarsTooLarge { prefix_vars, ell });
    }
    if prefix_vars > MAX_BOOLEANITY_PREFIX_VARS {
        return Err(BooleanityAccumulatorError::PrefixVarsExceedsSupported {
            prefix_vars,
            max: MAX_BOOLEANITY_PREFIX_VARS,
        });
    }

    let prefix_len = binary_domain_size(prefix_vars)?;
    let tail_len = binary_domain_size(ell - prefix_vars)?;
    if weights.tail_eq_weights.len() != tail_len {
        return Err(BooleanityAccumulatorError::LengthMismatch {
            label: "tail_eq_weights",
            got: weights.tail_eq_weights.len(),
            expected: tail_len,
        });
    }

    let num_binary_cols = validate_traces(traces, weights.row_weights.len())?;
    let expected_rho_powers = num_binary_cols
        .checked_mul(D)
        .ok_or(BooleanityAccumulatorError::DomainTooLarge { vars: prefix_vars })?;
    if scalar_weights.rho_powers.len() < expected_rho_powers {
        return Err(BooleanityAccumulatorError::LengthMismatch {
            label: "rho_powers",
            got: scalar_weights.rho_powers.len(),
            expected: expected_rho_powers,
        });
    }

    let domain_len = ternary_domain_size(prefix_vars)?;
    let mut table_values = vec![F::zero_with_cfg(field_cfg); domain_len];
    let entries_by_support_size = extended_entries_by_support_size(prefix_vars)?;
    let row_count = weights.row_weights.len();
    let omega = precompute_row_tail_weights(weights, field_cfg)?;
    let reducer = BarrettDelayedReduction::<F>::new(field_cfg);
    let word_count = bit_word_count(D);
    let prefix_word_len = prefix_len
        .checked_mul(word_count)
        .ok_or(BooleanityAccumulatorError::DomainTooLarge { vars: prefix_vars })?;
    let mut prefix_words = vec![0u64; prefix_word_len];

    for support_size in 1..=prefix_vars {
        let entries = &entries_by_support_size[support_size];
        if entries.is_empty() {
            continue;
        }

        let max_magnitude = max_delta_magnitude(prefix_vars, support_size)?;
        let _ = max_magnitude
            .checked_mul(max_magnitude)
            .ok_or(BooleanityAccumulatorError::DomainTooLarge { vars: prefix_vars })?;
        let tile_len = adaptive_tile_len(max_magnitude).min(entries.len());
        for tile in entries.chunks(tile_len) {
            for col_idx in 0..num_binary_cols {
                accumulate_column_tile::<F, PolyCoeff, Int, D>(
                    traces,
                    col_idx,
                    tile,
                    prefix_vars,
                    prefix_len,
                    tail_len,
                    row_count,
                    word_count,
                    max_magnitude,
                    &omega,
                    &mut prefix_words,
                    &scalar_weights.rho_powers[col_idx * D..col_idx * D + D],
                    field_cfg,
                    &reducer,
                    &mut table_values,
                )?;
            }
        }
    }

    Ok(BooleanityPrefixTable {
        values: table_values,
        ell,
        prefix_vars,
        num_binary_cols,
    })
}

#[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
fn accumulate_column_tile<F, PolyCoeff, Int, const D: usize>(
    traces: &[UairTrace<'_, PolyCoeff, Int, D>],
    col_idx: usize,
    tile: &[ExtendedTableEntry],
    prefix_vars: usize,
    prefix_len: usize,
    tail_len: usize,
    row_count: usize,
    word_count: usize,
    max_magnitude: usize,
    omega: &[F],
    prefix_words: &mut [u64],
    rho_powers: &[F],
    field_cfg: &F::Config,
    reducer: &BarrettDelayedReduction<'_, F>,
    table_values: &mut [F],
) -> Result<(), BooleanityAccumulatorError>
where
    F: MontgomeryLimbs + DelayedFieldProductSum + FromPrimitiveWithConfig + Send + Sync + 'static,
    PolyCoeff: Clone,
    Int: Clone,
{
    let bucket_stride = max_magnitude + 1;
    let bucket_count = tile.len().checked_mul(bucket_stride).ok_or(
        BooleanityAccumulatorError::BucketCountOverflow {
            entries: tile.len(),
            stride: bucket_stride,
        },
    )?;
    let mut buckets: Vec<[Uint<5>; D]> = vec![[Uint::zero(); D]; bucket_count];
    let zero = F::zero_with_cfg(field_cfg);
    let mut lane_accs: Vec<[F; D]> = (0..bucket_count)
        .map(|_| array::from_fn(|_| zero.clone()))
        .collect();
    let product_sum = MontgomeryProductSum4::<F>::new(field_cfg);
    let mut pending_adds = 0usize;
    let flush_adds = reducer.flush_adds();

    for tail in 0..tail_len {
        let omega_offset = tail * row_count;
        for row in 0..row_count {
            gather_prefix_words::<PolyCoeff, Int, D>(
                traces,
                col_idx,
                tail,
                row,
                prefix_vars,
                prefix_len,
                word_count,
                prefix_words,
            );
            let weight = &omega[omega_offset + row];

            for (entry_offset, entry) in tile.iter().enumerate() {
                pending_adds = pending_adds.saturating_add(accumulate_entry_deltas::<F, D>(
                    entry.point,
                    entry_offset,
                    bucket_stride,
                    word_count,
                    prefix_words,
                    weight,
                    reducer,
                    &mut buckets,
                ));

                if pending_adds >= flush_adds {
                    flush_buckets_into_lanes(&mut buckets, &mut lane_accs, reducer);
                    pending_adds = 0;
                }
            }
        }
    }

    flush_buckets_into_lanes(&mut buckets, &mut lane_accs, reducer);

    for (entry_offset, entry) in tile.iter().enumerate() {
        for magnitude in 1..=max_magnitude {
            let bucket_idx = entry_offset * bucket_stride + magnitude;
            let projected = FieldFieldInnerProduct::inner_product_with_algorithm::<UNCHECKED, _>(
                &product_sum,
                &lane_accs[bucket_idx],
                rho_powers,
                zero.clone(),
            )
            .expect("lane accumulator and rho powers have matching lengths");
            let magnitude_square = magnitude
                .checked_mul(magnitude)
                .ok_or(BooleanityAccumulatorError::DomainTooLarge { vars: prefix_vars })?;
            let scale = F::from_with_cfg(
                u64::try_from(magnitude_square).map_err(|_| {
                    BooleanityAccumulatorError::DomainTooLarge { vars: prefix_vars }
                })?,
                field_cfg,
            );
            table_values[entry.table_index] += scale * projected;
        }
    }
    Ok(())
}

#[allow(clippy::arithmetic_side_effects)]
fn gather_prefix_words<PolyCoeff, Int, const D: usize>(
    traces: &[UairTrace<'_, PolyCoeff, Int, D>],
    col_idx: usize,
    tail: usize,
    row: usize,
    prefix_vars: usize,
    prefix_len: usize,
    word_count: usize,
    out: &mut [u64],
) where
    PolyCoeff: Clone,
    Int: Clone,
{
    for prefix in 0..prefix_len {
        let instance_idx = prefix + (tail << prefix_vars);
        let poly = &traces[instance_idx].binary_poly[col_idx].evaluations[row];
        write_poly_words(
            poly,
            &mut out[prefix * word_count..(prefix + 1) * word_count],
        );
    }
}

#[allow(clippy::arithmetic_side_effects)]
fn accumulate_entry_deltas<F, const D: usize>(
    point: ExtendedPrefixPoint,
    entry_offset: usize,
    bucket_stride: usize,
    word_count: usize,
    prefix_words: &[u64],
    weight: &F,
    reducer: &BarrettDelayedReduction<'_, F>,
    buckets: &mut [[Uint<5>; D]],
) -> usize
where
    F: MontgomeryLimbs + Send + Sync,
{
    match point.support_size() {
        1 => accumulate_support_one::<F, D>(
            point,
            entry_offset,
            bucket_stride,
            word_count,
            prefix_words,
            weight,
            reducer,
            buckets,
        ),
        2 => accumulate_support_two::<F, D>(
            point,
            entry_offset,
            bucket_stride,
            word_count,
            prefix_words,
            weight,
            reducer,
            buckets,
        ),
        _ => accumulate_support_general::<F, D>(
            point,
            entry_offset,
            bucket_stride,
            word_count,
            prefix_words,
            weight,
            reducer,
            buckets,
        ),
    }
}

#[allow(clippy::arithmetic_side_effects)]
fn accumulate_support_one<F, const D: usize>(
    point: ExtendedPrefixPoint,
    entry_offset: usize,
    bucket_stride: usize,
    word_count: usize,
    prefix_words: &[u64],
    weight: &F,
    reducer: &BarrettDelayedReduction<'_, F>,
    buckets: &mut [[Uint<5>; D]],
) -> usize
where
    F: MontgomeryLimbs + Send + Sync,
{
    let support_bit = point.support_mask;
    let base = point.finite_bits & !support_bit;
    let idx0 = base;
    let idx1 = base | support_bit;
    let bucket_idx = entry_offset * bucket_stride + 1;
    let mut adds = 0usize;
    for word_idx in 0..word_count {
        let mask = word_at(prefix_words, idx0, word_count, word_idx)
            ^ word_at(prefix_words, idx1, word_count, word_idx);
        adds += add_mask_word_to_bucket(mask, word_idx, weight, reducer, &mut buckets[bucket_idx]);
    }
    adds
}

#[allow(clippy::arithmetic_side_effects)]
fn accumulate_support_two<F, const D: usize>(
    point: ExtendedPrefixPoint,
    entry_offset: usize,
    bucket_stride: usize,
    word_count: usize,
    prefix_words: &[u64],
    weight: &F,
    reducer: &BarrettDelayedReduction<'_, F>,
    buckets: &mut [[Uint<5>; D]],
) -> usize
where
    F: MontgomeryLimbs + Send + Sync,
{
    let first = point.support_mask & point.support_mask.wrapping_neg();
    let second = point.support_mask ^ first;
    let base = point.finite_bits & !point.support_mask;
    let idx00 = base;
    let idx10 = base | first;
    let idx01 = base | second;
    let idx11 = base | first | second;
    let bucket_1 = entry_offset * bucket_stride + 1;
    let bucket_2 = entry_offset * bucket_stride + 2;
    let mut adds = 0usize;

    for word_idx in 0..word_count {
        let valid_mask = valid_word_mask::<D>(word_idx);
        let d00 = word_at(prefix_words, idx00, word_count, word_idx) & valid_mask;
        let d10 = word_at(prefix_words, idx10, word_count, word_idx) & valid_mask;
        let d01 = word_at(prefix_words, idx01, word_count, word_idx) & valid_mask;
        let d11 = word_at(prefix_words, idx11, word_count, word_idx) & valid_mask;
        let mask_1 = ((d11 ^ d00) ^ (d10 ^ d01)) & valid_mask;
        let mask_2 = ((d11 & d00 & !d10 & !d01) | (!d11 & !d00 & d10 & d01)) & valid_mask;
        adds += add_mask_word_to_bucket(mask_1, word_idx, weight, reducer, &mut buckets[bucket_1]);
        adds += add_mask_word_to_bucket(mask_2, word_idx, weight, reducer, &mut buckets[bucket_2]);
    }

    adds
}

#[allow(clippy::arithmetic_side_effects)]
fn accumulate_support_general<F, const D: usize>(
    point: ExtendedPrefixPoint,
    entry_offset: usize,
    bucket_stride: usize,
    word_count: usize,
    prefix_words: &[u64],
    weight: &F,
    reducer: &BarrettDelayedReduction<'_, F>,
    buckets: &mut [[Uint<5>; D]],
) -> usize
where
    F: MontgomeryLimbs + Send + Sync,
{
    let mut support_bits = [0usize; usize::BITS as usize];
    let support_size = support_bit_masks_into(point.support_mask, &mut support_bits);
    let base = point.finite_bits & !point.support_mask;
    let mut deltas = [0i64; D];
    for vertex in 0..(1usize << support_size) {
        let mut prefix = base;
        for (pos, bit) in support_bits[..support_size].iter().enumerate() {
            if ((vertex >> pos) & 1) == 1 {
                prefix |= *bit;
            }
        }
        let sign = if (support_size - vertex.count_ones() as usize) % 2 == 0 {
            1i64
        } else {
            -1i64
        };
        for word_idx in 0..word_count {
            let mut word = word_at(prefix_words, prefix, word_count, word_idx);
            while word != 0 {
                let bit =
                    usize::try_from(word.trailing_zeros()).expect("trailing_zeros fits usize");
                let lane = word_idx * 64 + bit;
                if lane < D {
                    deltas[lane] += sign;
                }
                word &= word - 1;
            }
        }
    }

    let mut adds = 0usize;
    for (lane, delta) in deltas.iter().enumerate() {
        let magnitude = usize::try_from(delta.unsigned_abs()).expect("delta magnitude fits usize");
        if magnitude == 0 {
            continue;
        }
        let bucket_idx = entry_offset * bucket_stride + magnitude;
        reducer.add(&mut buckets[bucket_idx][lane], weight);
        adds += 1;
    }
    adds
}

#[allow(clippy::arithmetic_side_effects)]
fn add_mask_word_to_bucket<F, const D: usize>(
    mut mask: u64,
    word_idx: usize,
    weight: &F,
    reducer: &BarrettDelayedReduction<'_, F>,
    bucket: &mut [Uint<5>; D],
) -> usize
where
    F: MontgomeryLimbs + Send + Sync,
{
    let mut adds = 0usize;
    while mask != 0 {
        let bit = usize::try_from(mask.trailing_zeros()).expect("trailing_zeros fits usize");
        let lane = word_idx * 64 + bit;
        if lane < D {
            reducer.add(&mut bucket[lane], weight);
            adds += 1;
        }
        mask &= mask - 1;
    }
    adds
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

fn validate_trace_count(len: usize) -> Result<usize, BooleanityAccumulatorError> {
    if len == 0 {
        return Err(BooleanityAccumulatorError::EmptyTraces);
    }
    if !len.is_power_of_two() {
        return Err(BooleanityAccumulatorError::TraceCountNotPowerOfTwo { len });
    }
    Ok(usize::try_from(len.trailing_zeros()).expect("trailing_zeros fits usize"))
}

fn validate_traces<PolyCoeff, Int, const D: usize>(
    traces: &[UairTrace<'_, PolyCoeff, Int, D>],
    expected_rows: usize,
) -> Result<usize, BooleanityAccumulatorError>
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
            return Err(BooleanityAccumulatorError::BinaryColumnCountMismatch {
                trace_idx,
                got: trace.binary_poly.len(),
                expected: num_binary_cols,
            });
        }
        for (col_idx, column) in trace.binary_poly.iter().enumerate() {
            if column.evaluations.len() != expected_rows {
                return Err(BooleanityAccumulatorError::BinaryColumnRowMismatch {
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

fn precompute_row_tail_weights<F>(
    weights: BooleanityWeights<'_, F>,
    field_cfg: &F::Config,
) -> Result<Vec<F>, BooleanityAccumulatorError>
where
    F: PrimeField,
{
    let row_count = weights.row_weights.len();
    let total = weights.tail_eq_weights.len().checked_mul(row_count).ok_or(
        BooleanityAccumulatorError::DomainTooLarge {
            vars: weights.tail_eq_weights.len(),
        },
    )?;
    let mut omega = Vec::with_capacity(total);
    for tail_weight in weights.tail_eq_weights {
        for row_weight in weights.row_weights.as_slice() {
            omega.push(tail_weight.clone() * row_weight);
        }
    }
    if omega.is_empty() {
        omega.push(F::zero_with_cfg(field_cfg));
    }
    Ok(omega)
}

fn adaptive_tile_len(max_magnitude: usize) -> usize {
    (MAX_DMR_BUCKET_ARRAYS / (max_magnitude + 1)).max(1)
}

fn max_delta_magnitude(
    prefix_vars: usize,
    support_size: usize,
) -> Result<usize, BooleanityAccumulatorError> {
    let shift = support_size
        .checked_sub(1)
        .ok_or(BooleanityAccumulatorError::DomainTooLarge { vars: prefix_vars })?;
    let shift = u32::try_from(shift)
        .map_err(|_| BooleanityAccumulatorError::DomainTooLarge { vars: prefix_vars })?;
    1usize
        .checked_shl(shift)
        .ok_or(BooleanityAccumulatorError::DomainTooLarge { vars: prefix_vars })
}

fn binary_domain_size(vars: usize) -> Result<usize, BooleanityAccumulatorError> {
    let shift =
        u32::try_from(vars).map_err(|_| BooleanityAccumulatorError::DomainTooLarge { vars })?;
    1usize
        .checked_shl(shift)
        .ok_or(BooleanityAccumulatorError::DomainTooLarge { vars })
}

pub fn ternary_domain_size(vars: usize) -> Result<usize, BooleanityAccumulatorError> {
    let mut size = 1usize;
    for _ in 0..vars {
        size = size
            .checked_mul(3)
            .ok_or(BooleanityAccumulatorError::DomainTooLarge { vars })?;
    }
    Ok(size)
}

#[allow(clippy::arithmetic_side_effects)]
pub fn extended_point_from_index(
    mut index: usize,
    prefix_vars: usize,
) -> Result<ExtendedPrefixPoint, BooleanityAccumulatorError> {
    let domain_size = ternary_domain_size(prefix_vars)?;
    if index >= domain_size {
        return Err(BooleanityAccumulatorError::ExtendedPointIndexOutOfRange {
            index,
            domain_size,
        });
    }

    let mut support_mask = 0usize;
    let mut finite_bits = 0usize;
    for var in 0..prefix_vars {
        let digit = index % 3;
        index /= 3;
        match digit {
            0 => {}
            1 => finite_bits |= 1usize << var,
            2 => support_mask |= 1usize << var,
            _ => unreachable!("ternary digit must be 0, 1, or 2"),
        }
    }
    ExtendedPrefixPoint::new(support_mask, finite_bits)
}

#[allow(clippy::arithmetic_side_effects)]
pub fn extended_point_index(
    point: ExtendedPrefixPoint,
    prefix_vars: usize,
) -> Result<usize, BooleanityAccumulatorError> {
    let _ = ternary_domain_size(prefix_vars)?;
    let allowed_bits = binary_domain_size(prefix_vars)?.saturating_sub(1);
    if point.support_mask & !allowed_bits != 0 || point.finite_bits & !allowed_bits != 0 {
        return Err(BooleanityAccumulatorError::ExtendedPointOutOfRange { prefix_vars });
    }
    if point.support_mask & point.finite_bits != 0 {
        return Err(BooleanityAccumulatorError::ExtendedPointNotCanonical);
    }

    let mut index = 0usize;
    let mut scale = 1usize;
    for var in 0..prefix_vars {
        let bit = 1usize << var;
        let digit = if point.support_mask & bit != 0 {
            2
        } else if point.finite_bits & bit != 0 {
            1
        } else {
            0
        };
        index = index
            .checked_add(digit * scale)
            .ok_or(BooleanityAccumulatorError::DomainTooLarge { vars: prefix_vars })?;
        scale = scale
            .checked_mul(3)
            .ok_or(BooleanityAccumulatorError::DomainTooLarge { vars: prefix_vars })?;
    }
    Ok(index)
}

fn extended_entries_by_support_size(
    prefix_vars: usize,
) -> Result<Vec<Vec<ExtendedTableEntry>>, BooleanityAccumulatorError> {
    let domain_len = ternary_domain_size(prefix_vars)?;
    let mut entries = vec![Vec::new(); prefix_vars + 1];
    for table_index in 0..domain_len {
        let point = extended_point_from_index(table_index, prefix_vars)?;
        let support_size = point.support_size();
        if support_size == 0 {
            continue;
        }
        entries[support_size].push(ExtendedTableEntry { table_index, point });
    }
    Ok(entries)
}

fn support_bit_masks_into(mut support_mask: usize, out: &mut [usize]) -> usize {
    let mut len = 0usize;
    while support_mask != 0 {
        let bit = support_mask & support_mask.wrapping_neg();
        out[len] = bit;
        len += 1;
        support_mask ^= bit;
    }
    len
}

#[cfg(test)]
fn support_bit_masks(support_mask: usize) -> Vec<usize> {
    let mut bits =
        vec![0usize; usize::try_from(support_mask.count_ones()).expect("count_ones fits usize")];
    let len = support_bit_masks_into(support_mask, &mut bits);
    debug_assert_eq!(len, bits.len());
    bits
}

fn bit_word_count(degree: usize) -> usize {
    degree.div_ceil(64)
}

#[allow(clippy::arithmetic_side_effects)]
fn write_poly_words<const D: usize>(poly: &BinaryPoly<D>, out: &mut [u64]) {
    out.fill(0);
    for (bit_idx, coeff) in poly.iter().enumerate().take(D) {
        if coeff.into_inner() {
            out[bit_idx / 64] |= 1u64 << (bit_idx % 64);
        }
    }
}

fn word_at(prefix_words: &[u64], prefix: usize, word_count: usize, word_idx: usize) -> u64 {
    prefix_words[prefix * word_count + word_idx]
}

#[allow(clippy::arithmetic_side_effects)]
fn valid_word_mask<const D: usize>(word_idx: usize) -> u64 {
    let remaining = D.saturating_sub(word_idx * 64);
    match remaining {
        0 => 0,
        1..=63 => (1u64 << remaining) - 1,
        _ => u64::MAX,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{neutron_nova::build_sumfold_eq_weights, test_utils::test_config};
    use crypto_primitives::{FromWithConfig, crypto_bigint_monty::MontyField};
    use std::borrow::Cow;
    use zinc_poly::mle::DenseMultilinearExtension;
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

    fn sample_traces_ell3() -> Vec<Trace> {
        (0..8u32)
            .map(|i| {
                trace_from_columns(
                    &[
                        0x0000_0001 ^ i,
                        0x0000_00f0 ^ (i << 4),
                        0x0000_3333 ^ (i * 0x1111),
                        0x8000_0001 ^ (i << 8),
                    ],
                    &[
                        0x0000_0005 ^ (i << 1),
                        0x0000_0a0a ^ (i << 5),
                        0x0000_f00f ^ (i * 3),
                        0x0001_0010 ^ (i << 9),
                    ],
                )
            })
            .collect()
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn naive_table(
        traces: &[Trace],
        prefix_vars: usize,
        weights: BooleanityWeights<'_, F>,
        scalar_weights: BooleanityScalarWeights<'_, F>,
    ) -> Vec<F> {
        let cfg = test_config();
        let ell =
            usize::try_from(traces.len().trailing_zeros()).expect("trailing_zeros fits usize");
        let tail_len = 1usize << (ell - prefix_vars);
        let domain_len = ternary_domain_size(prefix_vars).unwrap();
        let mut out = vec![F::zero_with_cfg(&cfg); domain_len];

        for (index, value) in out.iter_mut().enumerate() {
            let point = extended_point_from_index(index, prefix_vars).unwrap();
            if point.is_finite_only() {
                continue;
            }
            for tail in 0..tail_len {
                for row in 0..weights.row_weights.len() {
                    for col_idx in 0..traces[0].binary_poly.len() {
                        for bit_idx in 0..32 {
                            let delta = naive_delta_bit(
                                traces,
                                col_idx,
                                bit_idx,
                                tail,
                                row,
                                prefix_vars,
                                point,
                            );
                            if delta == 0 {
                                continue;
                            }
                            let mut contribution = weights.tail_eq_weights[tail].clone();
                            contribution *= &weights.row_weights.as_slice()[row];
                            contribution *= &scalar_weights.rho_powers[col_idx * 32 + bit_idx];
                            let delta_square =
                                u64::try_from(delta * delta).expect("delta square fits u64");
                            contribution *= F::from_with_cfg(delta_square, &cfg);
                            *value += contribution;
                        }
                    }
                }
            }
        }
        out
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn naive_delta_bit(
        traces: &[Trace],
        col_idx: usize,
        bit_idx: usize,
        tail: usize,
        row: usize,
        prefix_vars: usize,
        point: ExtendedPrefixPoint,
    ) -> i64 {
        let support_bits = support_bit_masks(point.support_mask);
        let support_size = support_bits.len();
        let base = point.finite_bits & !point.support_mask;
        let mut delta = 0i64;
        for vertex in 0..(1usize << support_size) {
            let mut prefix = base;
            for (pos, bit) in support_bits.iter().enumerate() {
                if ((vertex >> pos) & 1) == 1 {
                    prefix |= *bit;
                }
            }
            let sign = if (support_size - vertex.count_ones() as usize) % 2 == 0 {
                1i64
            } else {
                -1i64
            };
            let instance_idx = prefix + (tail << prefix_vars);
            let bit = traces[instance_idx].binary_poly[col_idx].evaluations[row]
                .iter()
                .nth(bit_idx)
                .expect("bit index in range")
                .into_inner();
            if bit {
                delta += sign;
            }
        }
        delta
    }

    fn rho_powers() -> Vec<F> {
        powers(f(7), F::one_with_cfg(&test_config()), 64)
    }

    #[test]
    fn extended_point_index_round_trips() {
        for prefix_vars in 0..=4 {
            let domain_len = ternary_domain_size(prefix_vars).unwrap();
            for index in 0..domain_len {
                let point = extended_point_from_index(index, prefix_vars).unwrap();
                assert_eq!(extended_point_index(point, prefix_vars).unwrap(), index);
            }
        }
    }

    #[test]
    fn extended_point_rejects_noncanonical_overlap() {
        assert!(matches!(
            ExtendedPrefixPoint::new(0b01, 0b01),
            Err(BooleanityAccumulatorError::ExtendedPointNotCanonical)
        ));

        let point = ExtendedPrefixPoint {
            support_mask: 0b01,
            finite_bits: 0b01,
        };
        assert!(matches!(
            extended_point_index(point, 1),
            Err(BooleanityAccumulatorError::ExtendedPointNotCanonical)
        ));
    }

    #[test]
    fn optimized_booleanity_table_matches_naive_for_support_one_and_two() {
        let cfg = test_config();
        let traces = sample_traces_ell3();
        let beta = vec![f(3), f(5), f(11)];
        let eq_weights = build_sumfold_eq_weights(&beta, 2, &cfg).unwrap();
        let row_weights = RowWeights::new(&[f(13), f(17)], &cfg).unwrap();
        let rho = rho_powers();
        let weights = BooleanityWeights {
            row_weights: &row_weights,
            tail_eq_weights: &eq_weights.tail_eq_weights,
        };
        let scalar_weights = BooleanityScalarWeights { rho_powers: &rho };

        let table = build_booleanity_prefix_table(&traces, 2, weights, scalar_weights, &cfg)
            .expect("booleanity table should build");
        let expected = naive_table(&traces, 2, weights, scalar_weights);
        assert_eq!(table.values(), expected.as_slice());
    }

    #[test]
    fn optimized_booleanity_table_matches_naive_for_general_support() {
        let cfg = test_config();
        let traces = sample_traces_ell3();
        let eq_weights = build_sumfold_eq_weights(&[f(3), f(5), f(11)], 3, &cfg).unwrap();
        let row_weights = RowWeights::new(&[f(13), f(17)], &cfg).unwrap();
        let rho = rho_powers();
        let weights = BooleanityWeights {
            row_weights: &row_weights,
            tail_eq_weights: &eq_weights.tail_eq_weights,
        };
        let scalar_weights = BooleanityScalarWeights { rho_powers: &rho };

        let table = build_booleanity_prefix_table(&traces, 3, weights, scalar_weights, &cfg)
            .expect("booleanity table should build");
        let expected = naive_table(&traces, 3, weights, scalar_weights);
        assert_eq!(table.values(), expected.as_slice());
    }

    #[test]
    fn finite_only_entries_are_zero_initially() {
        let cfg = test_config();
        let traces = sample_traces_ell3();
        let eq_weights = build_sumfold_eq_weights(&[f(3), f(5), f(11)], 2, &cfg).unwrap();
        let row_weights = RowWeights::new(&[f(13), f(17)], &cfg).unwrap();
        let rho = rho_powers();
        let table = build_booleanity_prefix_table(
            &traces,
            2,
            BooleanityWeights {
                row_weights: &row_weights,
                tail_eq_weights: &eq_weights.tail_eq_weights,
            },
            BooleanityScalarWeights { rho_powers: &rho },
            &cfg,
        )
        .unwrap();

        for index in 0..table.len() {
            let point = extended_point_from_index(index, table.prefix_vars()).unwrap();
            if point.is_finite_only() {
                assert_eq!(table.values()[index], F::zero_with_cfg(&cfg));
            }
        }
    }

    #[test]
    fn booleanity_validation_errors_are_reported() {
        let cfg = test_config();
        let traces = sample_traces_ell3();
        let row_weights = RowWeights::new(&[f(13), f(17)], &cfg).unwrap();
        let rho = rho_powers();
        let empty_traces: &[Trace] = &[];

        let err = build_booleanity_prefix_table(
            empty_traces,
            0,
            BooleanityWeights {
                row_weights: &row_weights,
                tail_eq_weights: &[F::one_with_cfg(&cfg)],
            },
            BooleanityScalarWeights { rho_powers: &rho },
            &cfg,
        )
        .expect_err("empty traces should be rejected");
        assert!(matches!(err, BooleanityAccumulatorError::EmptyTraces));

        let err = build_booleanity_prefix_table(
            &traces[..3],
            1,
            BooleanityWeights {
                row_weights: &row_weights,
                tail_eq_weights: &[F::one_with_cfg(&cfg), F::one_with_cfg(&cfg)],
            },
            BooleanityScalarWeights { rho_powers: &rho },
            &cfg,
        )
        .expect_err("non-power-of-two trace count should be rejected");
        assert!(matches!(
            err,
            BooleanityAccumulatorError::TraceCountNotPowerOfTwo { len: 3 }
        ));

        let err = build_booleanity_prefix_table(
            &traces,
            4,
            BooleanityWeights {
                row_weights: &row_weights,
                tail_eq_weights: &[F::one_with_cfg(&cfg)],
            },
            BooleanityScalarWeights { rho_powers: &rho },
            &cfg,
        )
        .expect_err("too many prefix vars should be rejected");
        assert!(matches!(
            err,
            BooleanityAccumulatorError::PrefixVarsTooLarge {
                prefix_vars: 4,
                ell: 3
            }
        ));

        let oversized_prefix_traces =
            vec![traces[0].clone(); 1usize << (MAX_BOOLEANITY_PREFIX_VARS + 1)];
        let err = build_booleanity_prefix_table(
            &oversized_prefix_traces,
            MAX_BOOLEANITY_PREFIX_VARS + 1,
            BooleanityWeights {
                row_weights: &row_weights,
                tail_eq_weights: &[F::one_with_cfg(&cfg)],
            },
            BooleanityScalarWeights { rho_powers: &rho },
            &cfg,
        )
        .expect_err("unsupported prefix var count should be rejected before allocation");
        assert!(matches!(
            err,
            BooleanityAccumulatorError::PrefixVarsExceedsSupported {
                prefix_vars,
                max
            } if prefix_vars == MAX_BOOLEANITY_PREFIX_VARS + 1
                && max == MAX_BOOLEANITY_PREFIX_VARS
        ));

        let err = build_booleanity_prefix_table(
            &traces,
            2,
            BooleanityWeights {
                row_weights: &row_weights,
                tail_eq_weights: &[],
            },
            BooleanityScalarWeights { rho_powers: &rho },
            &cfg,
        )
        .expect_err("wrong tail weight length should be rejected");
        assert!(matches!(
            err,
            BooleanityAccumulatorError::LengthMismatch {
                label: "tail_eq_weights",
                got: 0,
                expected: 2
            }
        ));

        let short_rho = vec![F::one_with_cfg(&cfg); 3];
        let eq_weights = build_sumfold_eq_weights(&[f(3), f(5), f(11)], 2, &cfg).unwrap();
        let err = build_booleanity_prefix_table(
            &traces,
            2,
            BooleanityWeights {
                row_weights: &row_weights,
                tail_eq_weights: &eq_weights.tail_eq_weights,
            },
            BooleanityScalarWeights {
                rho_powers: &short_rho,
            },
            &cfg,
        )
        .expect_err("short rho powers should be rejected");
        assert!(matches!(
            err,
            BooleanityAccumulatorError::LengthMismatch {
                label: "rho_powers",
                got: 3,
                expected: 64
            }
        ));

        let mut mismatched_cols = traces.clone();
        mismatched_cols[1].binary_poly.to_mut().pop();
        let err = build_booleanity_prefix_table(
            &mismatched_cols,
            2,
            BooleanityWeights {
                row_weights: &row_weights,
                tail_eq_weights: &eq_weights.tail_eq_weights,
            },
            BooleanityScalarWeights { rho_powers: &rho },
            &cfg,
        )
        .expect_err("wrong binary column count should be rejected");
        assert!(matches!(
            err,
            BooleanityAccumulatorError::BinaryColumnCountMismatch {
                trace_idx: 1,
                got: 1,
                expected: 2
            }
        ));

        let bad_row_weights = RowWeights::new(&[f(13)], &cfg).unwrap();
        let err = build_booleanity_prefix_table(
            &traces,
            2,
            BooleanityWeights {
                row_weights: &bad_row_weights,
                tail_eq_weights: &eq_weights.tail_eq_weights,
            },
            BooleanityScalarWeights { rho_powers: &rho },
            &cfg,
        )
        .expect_err("wrong row count should be rejected");
        assert!(matches!(
            err,
            BooleanityAccumulatorError::BinaryColumnRowMismatch { .. }
        ));
    }
}
