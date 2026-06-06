use crypto_primitives::{PrimeField, crypto_bigint_uint::Uint};
use num_traits::Zero;
use std::array;
use thiserror::Error;
use zinc_poly::{
    mle::DenseMultilinearExtension,
    univariate::binary::BinaryPoly,
    utils::{ArithErrors, build_eq_x_r_vec},
};
use zinc_utils::{
    UNCHECKED,
    delayed_reduction::{
        BarrettReductionParams, DelayedFieldProductSum, DelayedModularReduction, MontgomeryLimbs,
    },
    inner_product::{FieldFieldInnerProduct, InnerProduct},
    powers,
};

const DMR_FLUSH_ADDS: usize = 1 << 20;

pub(crate) fn dmr_flush_adds(reduction_params: &BarrettReductionParams) -> usize {
    if reduction_params.modulus[3] == 0 {
        1
    } else {
        DMR_FLUSH_ADDS
    }
}

/// Errors produced by NeutronNova row-space accumulation helpers.
#[derive(Clone, Debug, Error)]
pub enum AccumulatorError {
    #[error("row weight length mismatch: weights={weights}, rows={rows}")]
    RowWeightLengthMismatch { weights: usize, rows: usize },
    #[error("bit index {bit_idx} is out of range for degree bound {degree}")]
    BitIndexOutOfRange { bit_idx: usize, degree: usize },
    #[error("projection powers length mismatch: got {got}, expected at least {expected}")]
    ProjectionPowersLengthMismatch { got: usize, expected: usize },
    #[error("row-weight construction failed: {0}")]
    RowWeights(#[from] ArithErrors),
}

/// Equality weights over the Boolean row space.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowWeights<F: PrimeField> {
    weights: Vec<F>,
}

impl<F: PrimeField> RowWeights<F> {
    /// Build weights `eq(r, z)` for all Boolean rows `z`.
    ///
    /// The zero-variable row space is supported as a single row of weight 1.
    pub fn new(row_point: &[F], field_cfg: &F::Config) -> Result<Self, AccumulatorError> {
        let weights = if row_point.is_empty() {
            vec![F::one_with_cfg(field_cfg)]
        } else {
            build_eq_x_r_vec(row_point, field_cfg)?
        };
        Ok(Self { weights })
    }

    /// Build row weights and zero the final row to match current CPR parity.
    pub fn new_with_last_row_zero(
        row_point: &[F],
        field_cfg: &F::Config,
    ) -> Result<Self, AccumulatorError> {
        let mut weights = Self::new(row_point, field_cfg)?;
        weights.zero_last_row(field_cfg);
        Ok(weights)
    }

    /// Set the final row weight to zero in-place.
    pub fn zero_last_row(&mut self, field_cfg: &F::Config) {
        if let Some(last) = self.weights.last_mut() {
            *last = F::zero_with_cfg(field_cfg);
        }
    }

    pub fn as_slice(&self) -> &[F] {
        &self.weights
    }

    pub fn len(&self) -> usize {
        self.weights.len()
    }

    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }
}

/// DMR-backed bit buckets for one small-value binary-polynomial column.
#[derive(Clone, Debug)]
pub struct SmallValueBitAccumulator<'a, F: PrimeField, const D: usize> {
    buckets: [Uint<5>; D],
    lane_accs: [F; D],
    pending_adds: usize,
    flush_adds: usize,
    field_cfg: &'a F::Config,
    reduction_params: BarrettReductionParams,
}

impl<'a, F, const D: usize> SmallValueBitAccumulator<'a, F, D>
where
    F: MontgomeryLimbs + Send + Sync,
{
    pub fn new(field_cfg: &'a F::Config) -> Self {
        let reduction_params = F::barrett_reduction_params(field_cfg);
        let flush_adds = dmr_flush_adds(&reduction_params);
        let zero = F::zero_with_cfg(field_cfg);
        Self {
            buckets: [Uint::zero(); D],
            lane_accs: array::from_fn(|_| zero.clone()),
            pending_adds: 0,
            flush_adds,
            field_cfg,
            reduction_params,
        }
    }

    /// Pending unreduced DMR buckets.
    ///
    /// Flushed contributions live in the reduced lane accumulators, so this is
    /// only useful for low-level tests and diagnostics.
    pub fn pending_buckets(&self) -> &[Uint<5>] {
        &self.buckets
    }

    pub fn add_bit_weight(&mut self, bit_idx: usize, weight: &F) -> Result<(), AccumulatorError> {
        let Some(bucket) = self.buckets.get_mut(bit_idx) else {
            return Err(AccumulatorError::BitIndexOutOfRange { bit_idx, degree: D });
        };
        <Uint<5> as DelayedModularReduction<F>>::add(bucket, weight);
        self.pending_adds = self.pending_adds.saturating_add(1);
        if self.pending_adds >= self.flush_adds {
            self.flush_buckets();
        }
        Ok(())
    }

    #[allow(clippy::arithmetic_side_effects)]
    pub fn add_binary_poly(
        &mut self,
        poly: &BinaryPoly<D>,
        weight: &F,
    ) -> Result<(), AccumulatorError> {
        if D <= 64 {
            let mut bits = 0u64;
            for (bit_idx, coeff) in poly.iter().enumerate().take(D) {
                if coeff.into_inner() {
                    bits |= 1u64 << bit_idx;
                }
            }

            while bits != 0 {
                let bit_idx =
                    usize::try_from(bits.trailing_zeros()).expect("trailing_zeros fits usize");
                self.add_bit_weight(bit_idx, weight)?;
                bits &= bits - 1;
            }
        } else {
            for (bit_idx, coeff) in poly.iter().enumerate().take(D) {
                if coeff.into_inner() {
                    self.add_bit_weight(bit_idx, weight)?;
                }
            }
        }
        Ok(())
    }

    pub fn reduce_buckets(mut self) -> Vec<F> {
        self.flush_buckets();
        self.lane_accs.into_iter().collect()
    }

    fn flush_buckets(&mut self) {
        for (bucket, acc) in self.buckets.iter_mut().zip(self.lane_accs.iter_mut()) {
            if bucket.is_zero() {
                continue;
            }
            let pending = std::mem::replace(bucket, Uint::zero());
            *acc += <Uint<5> as DelayedModularReduction<F>>::reduce(
                pending,
                self.field_cfg,
                &self.reduction_params,
            );
        }
        self.pending_adds = 0;
    }
}

impl<F, const D: usize> SmallValueBitAccumulator<'_, F, D>
where
    F: MontgomeryLimbs + DelayedFieldProductSum + Send + Sync,
{
    pub fn project(mut self, projection_powers: &[F]) -> Result<F, AccumulatorError> {
        if projection_powers.len() < D {
            return Err(AccumulatorError::ProjectionPowersLengthMismatch {
                got: projection_powers.len(),
                expected: D,
            });
        }

        self.flush_buckets();
        let zero = F::zero_with_cfg(self.field_cfg);
        Ok(FieldFieldInnerProduct::inner_product::<UNCHECKED>(
            &self.lane_accs,
            &projection_powers[..D],
            zero,
        )
        .expect("bucket and projection-power lengths match"))
    }
}

/// Accumulate one binary-polynomial column projected by `projecting_element`.
///
/// This computes the bucket-first form:
/// first `S_j = sum_z bit_j(col[z]) * row_weight[z]`, then
/// `sum_j S_j * projecting_element^j`.
pub fn accumulate_binary_column_projected<F, const D: usize>(
    column: &DenseMultilinearExtension<BinaryPoly<D>>,
    row_weights: &RowWeights<F>,
    projecting_element: &F,
    field_cfg: &F::Config,
) -> Result<F, AccumulatorError>
where
    F: MontgomeryLimbs + DelayedFieldProductSum + Send + Sync,
{
    if column.evaluations.len() != row_weights.len() {
        return Err(AccumulatorError::RowWeightLengthMismatch {
            weights: row_weights.len(),
            rows: column.evaluations.len(),
        });
    }

    let one = F::one_with_cfg(field_cfg);
    let projection_powers: Vec<F> = powers(projecting_element.clone(), one, D);
    let mut accumulator = SmallValueBitAccumulator::<F, D>::new(field_cfg);

    for (poly, weight) in column.iter().zip(row_weights.as_slice()) {
        accumulator.add_binary_poly(poly, weight)?;
    }

    accumulator.project(&projection_powers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::test_config;
    use crypto_bigint::{Odd, modular::MontyParams};
    use crypto_primitives::{FromWithConfig, crypto_bigint_monty::MontyField};
    use zinc_utils::powers;

    type F = MontyField<4>;

    fn field_cfg() -> MontyParams<4> {
        let modulus = crypto_bigint::Uint::<4>::from_words([
            0xFFFF_FFFE_FFFF_FC2F,
            0xFFFF_FFFF_FFFF_FFFF,
            0xFFFF_FFFF_FFFF_FFFF,
            0xFFFF_FFFF_FFFF_FFFF,
        ]);
        MontyParams::new(Odd::new(modulus).expect("secp256k1 modulus is odd"))
    }

    fn f(value: u64, cfg: &MontyParams<4>) -> F {
        F::from_with_cfg(value, cfg)
    }

    fn binary_col(patterns: &[u32]) -> DenseMultilinearExtension<BinaryPoly<32>> {
        DenseMultilinearExtension::from_evaluations_vec(
            usize::try_from(patterns.len().next_power_of_two().trailing_zeros())
                .expect("trailing_zeros fits usize"),
            patterns
                .iter()
                .copied()
                .map(BinaryPoly::<32>::from)
                .collect(),
            BinaryPoly::<32>::zero(),
        )
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn naive_projected_sum(
        column: &DenseMultilinearExtension<BinaryPoly<32>>,
        row_weights: &RowWeights<F>,
        projecting_element: &F,
        cfg: &MontyParams<4>,
    ) -> F {
        let zero = F::zero_with_cfg(&cfg);
        let one = F::one_with_cfg(&cfg);
        let powers = powers(projecting_element.clone(), one, 32);

        column
            .iter()
            .zip(row_weights.as_slice())
            .fold(zero, |mut acc, (poly, row_weight)| {
                for (bit_idx, coeff) in poly.iter().enumerate().take(32) {
                    if coeff.into_inner() {
                        acc += row_weight.clone() * &powers[bit_idx];
                    }
                }
                acc
            })
    }

    #[test]
    fn projected_binary_column_matches_naive_row_space_sum() {
        let cfg = field_cfg();
        let point = vec![f(3, &cfg), f(5, &cfg), f(7, &cfg)];
        let row_weights = RowWeights::new(&point, &cfg).unwrap();
        let column = binary_col(&[
            0x0000_0001,
            0x8000_0001,
            0x0f0f_00f0,
            0xf000_00ff,
            0x0101_0101,
            0x1111_2222,
            0xdead_beef,
            0xffff_0000,
        ]);
        let projecting_element = f(11, &cfg);

        let got =
            accumulate_binary_column_projected(&column, &row_weights, &projecting_element, &cfg)
                .unwrap();
        let expected = naive_projected_sum(&column, &row_weights, &projecting_element, &cfg);

        assert_eq!(got, expected);
    }

    #[test]
    fn small_value_bit_accumulator_flushes_for_small_modulus() {
        let cfg = test_config();
        let max = -F::from_with_cfg(1u64, &cfg);
        let mut accumulator = SmallValueBitAccumulator::<F, 32>::new(&cfg);
        let mut expected = F::zero_with_cfg(&cfg);

        for _ in 0..2048 {
            accumulator.add_bit_weight(7, &max).unwrap();
            expected += &max;
        }

        let lanes = accumulator.reduce_buckets();
        assert_eq!(lanes[7], expected);
        assert!(
            lanes
                .iter()
                .enumerate()
                .all(|(lane, value)| lane == 7 || F::is_zero(value))
        );
    }

    #[test]
    fn last_row_zero_helper_matches_manual_zeroing() {
        let cfg = field_cfg();
        let point = vec![f(3, &cfg), f(5, &cfg)];

        let mut manual = RowWeights::new(&point, &cfg).unwrap();
        manual.zero_last_row(&cfg);
        let helper = RowWeights::new_with_last_row_zero(&point, &cfg).unwrap();

        assert_eq!(helper, manual);
        assert_eq!(helper.as_slice().last().unwrap(), &F::zero_with_cfg(&cfg));
    }

    #[test]
    fn projected_binary_column_rejects_row_weight_mismatch() {
        let cfg = field_cfg();
        let row_weights = RowWeights::new(&[f(3, &cfg), f(5, &cfg)], &cfg).unwrap();
        let column = binary_col(&[1, 2, 3, 4, 5, 6, 7, 8]);

        let err = accumulate_binary_column_projected(&column, &row_weights, &f(11, &cfg), &cfg)
            .expect_err("mismatched row weights should be rejected");

        assert!(matches!(
            err,
            AccumulatorError::RowWeightLengthMismatch {
                weights: 4,
                rows: 8
            }
        ));
    }
}
