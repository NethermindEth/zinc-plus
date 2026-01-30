#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::mem::MaybeUninit;

use crypto_primitives::{Field, FromWithConfig, PrimeField, Semiring};
use itertools::{Itertools, max};
use zinc_poly::{
    CoefficientProjectable,
    mle::{DenseMultilinearExtension, dense::CollectDenseMleWithZero},
    univariate::dynamic::DynamicPolynomial,
};
use zinc_uair::{ConstraintBuilder, Uair, ideal::DummyIdeal};
use zinc_utils::{cfg_iter, cfg_iter_mut, from_ref::FromRef};

use crate::ideal_check::structs::IdealCheckTypes;

pub struct CombinedPolyRowBuilder<F: PrimeField> {
    combined_evaluations: Vec<DynamicPolynomial<F>>,
}

impl<F: PrimeField> ConstraintBuilder for CombinedPolyRowBuilder<F> {
    type Expr = DynamicPolynomial<F>;
    type Ideal = DummyIdeal;

    #[allow(clippy::arithmetic_side_effects)]
    fn assert_in_ideal(&mut self, expr: Self::Expr, _ideal: &Self::Ideal) {
        self.combined_evaluations.push(expr);
    }
}

impl<F: PrimeField> CombinedPolyRowBuilder<F> {
    pub fn new(num_constraints: usize) -> Self {
        Self {
            combined_evaluations: Vec::with_capacity(num_constraints),
        }
    }
}

#[allow(clippy::arithmetic_side_effects)]
pub fn compute_combined_polynomials<
    IcTypes: IdealCheckTypes<DEGREE_PLUS_ONE>,
    U,
    const DEGREE_PLUS_ONE: usize,
>(
    trace: &[DenseMultilinearExtension<IcTypes::Witness>],
    projecting_element: &IcTypes::F,
    num_constraints: usize,
) -> Vec<Vec<DenseMultilinearExtension<<IcTypes::F as Field>::Inner>>>
where
    U: Uair<IcTypes::Witness>,
{
    let num_rows = trace[0].len();
    let num_cols = trace.len();

    let mut trace_matrix: Vec<Vec<DynamicPolynomial<IcTypes::F>>> = (0..num_rows)
        .map(|_| Vec::with_capacity(num_cols))
        .collect();

    cfg_iter_mut!(trace_matrix)
        .enumerate()
        .for_each(|(row_num, row)| {
            row.spare_capacity_mut()
                .iter_mut()
                .enumerate()
                .for_each(|(col_num, cell)| {
                    *cell = MaybeUninit::new(
                        trace[col_num][row_num]
                            .project_coefficients(projecting_element)
                            .into(),
                    );
                });

            unsafe { row.set_len(num_cols) };
        });

    let field_zero = IcTypes::F::zero_with_cfg(projecting_element.cfg());

    let mut combined_poly_rows: Vec<(usize, Vec<DynamicPolynomial<IcTypes::F>>)> =
        cfg_iter!(trace_matrix)
            .zip(cfg_iter!(trace_matrix).skip(1))
            .map(|(up, down)| {
                let mut constraint_builder = CombinedPolyRowBuilder::new(num_constraints);
                U::constrain_general(
                    &mut constraint_builder,
                    up,
                    down,
                    |x| x.project_coefficients(projecting_element).into(),
                    |x, y| {
                        Some(
                            DynamicPolynomial::from(y.project_coefficients(projecting_element)) * x,
                        )
                    },
                    DummyIdeal::from_ref,
                );

                let mut combined_evaluations = constraint_builder.combined_evaluations;

                combined_evaluations
                    .iter_mut()
                    .for_each(|eval| eval.trim_with_zero(&field_zero));

                let max_degree = max(combined_evaluations
                    .iter()
                    .map(|eval| eval.degree_with_zero(&field_zero).unwrap_or(0)))
                .expect("the iter can't be empty");

                (max_degree, combined_evaluations)
            })
            .collect();

    combined_poly_rows.push(
        combined_poly_rows
            .last()
            .expect("shouldn't be empty")
            .clone(),
    );

    let max_degree = *max(combined_poly_rows.iter().map(|(max_degree, _)| max_degree))
        .expect("the iter can't be empty");

    let result: Vec<Vec<DenseMultilinearExtension<<IcTypes::F as Field>::Inner>>> = (0
        ..num_constraints)
        .map(|constraint| {
            (0..=max_degree)
                .map(|coeff| {
                    combined_poly_rows
                        .iter()
                        .map(|(_, row)| {
                            if coeff >= row[constraint].coeffs.len() {
                                field_zero.inner().clone()
                            } else {
                                row[constraint].coeffs[coeff].inner().clone()
                            }
                        })
                        .collect_dense_mle_with_zero(field_zero.inner())
                })
                .collect_vec()
        })
        .collect_vec();

    result
}
