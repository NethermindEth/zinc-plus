use crypto_primitives::{FromWithConfig, PrimeField, Semiring};
use itertools::Itertools;
use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::collections::HashMap;
use zinc_poly::{
    EvaluatablePolynomial, EvaluationError,
    mle::DenseMultilinearExtension,
    univariate::{
        binary::BinaryPoly, dense::DensePolynomial, dynamic::over_field::DynamicPolynomialF,
    },
};
use zinc_uair::{Uair, collect_scalars::collect_scalars};
use zinc_utils::{cfg_extend, cfg_into_iter, cfg_iter, cfg_iter_mut};

/// Row-indexed trace matrix: `trace[row][col]`.
/// Each row contains all column values for that row.
/// Used by `compute_combined_polynomials` for non-linear constraints.
pub type RowMajorTrace<F> = Vec<Vec<DynamicPolynomialF<F>>>;

/// Column-indexed trace matrix: `trace[col][row]`.
/// Each column is a `DenseMultilinearExtension` over the hypercube.
/// Used by `evaluate_combined_polynomials` for linear constraints (MLE-first).
pub type ColumnMajorTrace<F> = Vec<DenseMultilinearExtension<DynamicPolynomialF<F>>>;

/// Project a multi-typed trace onto F[X], returning a row-indexed (transposed)
/// matrix. Result: `trace[row][col]` where columns are ordered as binary_poly,
/// arbitrary_poly, int.
///
/// Use this for the combined polynomial approach (non-linear constraints).
#[allow(clippy::arithmetic_side_effects)]
pub fn project_trace_coeffs_row_major<F, PolyCoeff, Int, const DEGREE_PLUS_ONE: usize>(
    binary_poly_trace: &[DenseMultilinearExtension<BinaryPoly<DEGREE_PLUS_ONE>>],
    arbitrary_poly_trace: &[DenseMultilinearExtension<
        DensePolynomial<PolyCoeff, DEGREE_PLUS_ONE>,
    >],
    int_trace: &[DenseMultilinearExtension<Int>],
    field_cfg: &F::Config,
) -> RowMajorTrace<F>
where
    F: FromWithConfig<PolyCoeff> + FromWithConfig<Int> + Send + Sync,
    PolyCoeff: Clone + Send + Sync,
    Int: Clone + Send + Sync,
{
    let zero = F::zero_with_cfg(field_cfg);
    let one = F::one_with_cfg(field_cfg);

    let binary_len = binary_poly_trace.len();
    let arbitrary_len = arbitrary_poly_trace.len();
    let num_cols = binary_poly_trace.len() + arbitrary_poly_trace.len() + int_trace.len();

    // Determine number of rows from the first non-empty column
    let num_rows = binary_poly_trace
        .first()
        .map(|c| c.len())
        .or_else(|| arbitrary_poly_trace.first().map(|c| c.len()))
        .or_else(|| int_trace.first().map(|c| c.len()))
        .unwrap_or(0);

    // Build row-by-row
    cfg_into_iter!(0..num_rows)
        .map(|row_idx| {
            let mut row: Vec<DynamicPolynomialF<F>> = Vec::with_capacity(num_cols);
            let spare = row.spare_capacity_mut();

            // Binary poly columns
            cfg_iter_mut!(spare[..binary_len])
                .zip(cfg_iter!(binary_poly_trace))
                .for_each(|(slot, col)| {
                    let binary_poly = &col.evaluations[row_idx];
                    slot.write(
                        binary_poly
                            .iter()
                            .map(|coeff| {
                                if coeff.into_inner() {
                                    one.clone()
                                } else {
                                    zero.clone()
                                }
                            })
                            .collect(),
                    );
                });

            // Arbitrary poly columns
            cfg_iter_mut!(spare[binary_len..binary_len + arbitrary_len])
                .zip(cfg_iter!(arbitrary_poly_trace))
                .for_each(|(slot, col)| {
                    let arbitrary_poly = &col.evaluations[row_idx];
                    slot.write(
                        arbitrary_poly
                            .iter()
                            .map(|coeff| F::from_with_cfg(coeff.clone(), field_cfg))
                            .collect(),
                    );
                });

            // Int columns
            cfg_iter_mut!(spare[binary_len + arbitrary_len..])
                .zip(cfg_iter!(int_trace))
                .for_each(|(slot, col)| {
                    let int_val = &col.evaluations[row_idx];
                    slot.write(DynamicPolynomialF {
                        coeffs: vec![F::from_with_cfg(int_val.clone(), field_cfg)],
                    });
                });

            // SAFETY: All slots have been initialized above.
            unsafe { row.set_len(num_cols) };
            row
        })
        .collect()
}

/// Project a multi-typed trace onto F[X], returning a column-indexed matrix.
/// Result: `trace[col]` is a
/// `DenseMultilinearExtension<DynamicPolynomialF<F>>`.
///
/// Use this for the MLE-first approach (linear constraints only).
#[allow(clippy::arithmetic_side_effects)]
pub fn project_trace_coeffs_column_major<F, PolyCoeff, Int, const DEGREE_PLUS_ONE: usize>(
    binary_poly_trace: &[DenseMultilinearExtension<BinaryPoly<DEGREE_PLUS_ONE>>],
    arbitrary_poly_trace: &[DenseMultilinearExtension<
        DensePolynomial<PolyCoeff, DEGREE_PLUS_ONE>,
    >],
    int_trace: &[DenseMultilinearExtension<Int>],
    field_cfg: &F::Config,
) -> ColumnMajorTrace<F>
where
    F: FromWithConfig<PolyCoeff> + FromWithConfig<Int> + Send + Sync,
    PolyCoeff: Clone + Send + Sync,
    Int: Clone + Send + Sync,
{
    let zero = F::zero_with_cfg(field_cfg);
    let one = F::one_with_cfg(field_cfg);

    let num_vars = [
        binary_poly_trace.first().map(|c| c.num_vars),
        arbitrary_poly_trace.first().map(|c| c.num_vars),
        int_trace.first().map(|c| c.num_vars),
    ]
    .into_iter()
    .flatten()
    .max()
    .unwrap_or(0);

    let mut result =
        Vec::with_capacity(binary_poly_trace.len() + arbitrary_poly_trace.len() + int_trace.len());

    // Binary poly columns
    cfg_extend!(
        result,
        cfg_iter!(binary_poly_trace).map(|column| {
            let evaluations: Vec<DynamicPolynomialF<F>> = column
                .iter()
                .map(|binary_poly| {
                    binary_poly
                        .iter()
                        .map(|coeff| {
                            if coeff.into_inner() {
                                one.clone()
                            } else {
                                zero.clone()
                            }
                        })
                        .collect()
                })
                .collect();
            DenseMultilinearExtension {
                evaluations,
                num_vars,
            }
        })
    );

    // Arbitrary poly columns
    cfg_extend!(
        result,
        cfg_iter!(arbitrary_poly_trace).map(|column| {
            let evaluations: Vec<DynamicPolynomialF<F>> = column
                .iter()
                .map(|arbitrary_poly| {
                    arbitrary_poly
                        .iter()
                        .map(|coeff| F::from_with_cfg(coeff.clone(), field_cfg))
                        .collect()
                })
                .collect();
            DenseMultilinearExtension {
                evaluations,
                num_vars,
            }
        })
    );

    // Int columns
    cfg_extend!(
        result,
        cfg_iter!(int_trace).map(|column| {
            let evaluations: Vec<DynamicPolynomialF<F>> = column
                .iter()
                .map(|int| DynamicPolynomialF {
                    coeffs: vec![F::from_with_cfg(int.clone(), field_cfg)],
                })
                .collect();
            DenseMultilinearExtension {
                evaluations,
                num_vars,
            }
        })
    );

    result
}

/// Evaluate a row-indexed trace along F[X]->F and return column-indexed MLEs.
/// Takes a `TransposedTrace` (row-indexed) and returns column-indexed
/// `Vec<DenseMultilinearExtension<F::Inner>>` for sumcheck compatibility.
/// Each polynomial is evaluated at the projecting element.
#[allow(clippy::arithmetic_side_effects)]
pub fn evaluate_trace_to_column_mles<F: PrimeField + 'static>(
    trace: &RowMajorTrace<F>,
    projecting_element: &F,
) -> Vec<DenseMultilinearExtension<F::Inner>> {
    let num_rows = trace.len();
    let num_cols = trace.first().map(|r| r.len()).unwrap_or(0);
    let num_vars = num_rows.next_power_of_two().trailing_zeros() as usize;
    let zero = F::zero_with_cfg(projecting_element.cfg()).inner().clone();

    // Build column-indexed MLEs from row-indexed trace
    cfg_into_iter!(0..num_cols)
        .map(|col_idx| {
            let evaluations: Vec<F::Inner> = (0..num_rows)
                .map(|row_idx| {
                    let poly = &trace[row_idx][col_idx];
                    if poly.is_zero() {
                        zero.clone()
                    } else {
                        poly.evaluate_at_point(projecting_element)
                            .expect("dynamic poly evaluation does not fail")
                            .inner()
                            .clone()
                    }
                })
                .collect();
            DenseMultilinearExtension::from_evaluations_vec(num_vars, evaluations, zero.clone())
        })
        .collect()
}

/// Project scalars of a UAIR onto F[X].
pub fn project_scalars<F: PrimeField, U: Uair>(
    project: impl Fn(&U::Scalar) -> DynamicPolynomialF<F>,
) -> HashMap<U::Scalar, DynamicPolynomialF<F>> {
    let uair_scalars = collect_scalars::<U>();

    // TODO(Ilia): if there's a lot of scalars
    //             we should do this in parallel probably.
    uair_scalars
        .into_iter()
        .map(|scalar| {
            (scalar.clone(), {
                let mut dynamic_poly = project(&scalar);

                dynamic_poly.trim();

                dynamic_poly
            })
        })
        .collect()
}

/// Project scalars of a UAIR along F[X] -> F.
pub fn project_scalars_to_field<R: Semiring + 'static, F: PrimeField>(
    scalars: HashMap<R, DynamicPolynomialF<F>>,
    projecting_element: &F,
) -> Result<HashMap<R, F>, (R, F, EvaluationError)> {
    // TODO(Ilia): Parallelising this might be good for big UAIRs.
    //             We'd conditionally route between sequential and parallel
    //             projection depending on how many scalars the UAIR has.
    scalars
        .into_iter()
        .map(
            |(scalar, value)| -> Result<(R, F), (R, F, EvaluationError)> {
                Ok((
                    scalar.clone(),
                    value
                        .evaluate_at_point(projecting_element)
                        .map_err(|err| (scalar.clone(), projecting_element.clone(), err))?,
                ))
            },
        )
        .try_collect()
}
