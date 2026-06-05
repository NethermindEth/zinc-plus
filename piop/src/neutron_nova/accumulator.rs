use crypto_primitives::{PrimeField, crypto_bigint_uint::Uint};
use num_traits::Zero;
use std::marker::PhantomData;
use thiserror::Error;
use zinc_poly::{
    mle::DenseMultilinearExtension,
    univariate::binary::BinaryPoly,
    utils::{ArithErrors, build_eq_x_r_vec},
};
use zinc_utils::{
    UNCHECKED,
    delayed_reduction::{DelayedFieldProductSum, DelayedModularReduction, MontgomeryLimbs},
    inner_product::{FieldFieldInnerProduct, InnerProduct},
    powers,
};

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
pub struct SmallValueBitAccumulator<F: PrimeField, const D: usize> {
    buckets: [Uint<5>; D],
    _field: PhantomData<F>,
}

impl<F: PrimeField, const D: usize> Default for SmallValueBitAccumulator<F, D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: PrimeField, const D: usize> SmallValueBitAccumulator<F, D> {
    pub fn new() -> Self {
        Self {
            buckets: [Uint::zero(); D],
            _field: PhantomData,
        }
    }

    pub fn buckets(&self) -> &[Uint<5>] {
        &self.buckets
    }
}

impl<F, const D: usize> SmallValueBitAccumulator<F, D>
where
    F: MontgomeryLimbs + Send + Sync,
{
    pub fn add_bit_weight(&mut self, bit_idx: usize, weight: &F) -> Result<(), AccumulatorError> {
        let Some(bucket) = self.buckets.get_mut(bit_idx) else {
            return Err(AccumulatorError::BitIndexOutOfRange { bit_idx, degree: D });
        };
        <Uint<5> as DelayedModularReduction<F>>::add(bucket, weight);
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

    pub fn reduce_buckets(
        self,
        field_cfg: &F::Config,
        reduction_params: &zinc_utils::delayed_reduction::BarrettReductionParams,
    ) -> Vec<F> {
        self.buckets
            .into_iter()
            .map(|bucket| {
                <Uint<5> as DelayedModularReduction<F>>::reduce(bucket, field_cfg, reduction_params)
            })
            .collect()
    }
}

impl<F, const D: usize> SmallValueBitAccumulator<F, D>
where
    F: MontgomeryLimbs + DelayedFieldProductSum + Send + Sync,
{
    pub fn project(
        self,
        projection_powers: &[F],
        field_cfg: &F::Config,
        reduction_params: &zinc_utils::delayed_reduction::BarrettReductionParams,
    ) -> Result<F, AccumulatorError> {
        if projection_powers.len() < D {
            return Err(AccumulatorError::ProjectionPowersLengthMismatch {
                got: projection_powers.len(),
                expected: D,
            });
        }

        let zero = F::zero_with_cfg(field_cfg);
        let bucket_evals = self.reduce_buckets(field_cfg, reduction_params);
        Ok(FieldFieldInnerProduct::inner_product::<UNCHECKED>(
            &bucket_evals,
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
    let reduction_params = F::barrett_reduction_params(field_cfg);
    let mut accumulator = SmallValueBitAccumulator::<F, D>::new();

    for (poly, weight) in column.iter().zip(row_weights.as_slice()) {
        accumulator.add_binary_poly(poly, weight)?;
    }

    accumulator.project(&projection_powers, field_cfg, &reduction_params)
}

#[cfg(test)]
mod tests {
    use super::*;
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
    ) -> F {
        let cfg = field_cfg();
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
        let expected = naive_projected_sum(&column, &row_weights, &projecting_element);

        assert_eq!(got, expected);
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
