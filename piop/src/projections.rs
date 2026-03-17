#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crypto_primitives::{FromWithConfig, PrimeField, Semiring};
use std::{collections::HashMap, iter};
use zinc_poly::{
    EvaluationError,
    mle::DenseMultilinearExtension,
    univariate::dynamic::over_field::{DynamicPolyFInnerProduct, DynamicPolynomialF},
};
use zinc_uair::{Uair, UairTrace, collect_scalars::collect_scalars};
use zinc_utils::{
    UNCHECKED, cfg_extend, cfg_into_iter, cfg_iter, cfg_iter_mut, inner_product::InnerProduct,
    powers,
};

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
    trace: &UairTrace<PolyCoeff, Int, DEGREE_PLUS_ONE>,
    field_cfg: &F::Config,
) -> RowMajorTrace<F>
where
    F: for<'a> FromWithConfig<&'a PolyCoeff> + for<'a> FromWithConfig<&'a Int> + Send + Sync,
    PolyCoeff: Clone + Send + Sync,
    Int: Clone + Send + Sync,
{
    let zero = F::zero_with_cfg(field_cfg);
    let one = F::one_with_cfg(field_cfg);

    let binary_len = trace.binary_poly.len();
    let arbitrary_len = trace.arbitrary_poly.len();
    let num_cols = trace.binary_poly.len() + trace.arbitrary_poly.len() + trace.int.len();

    // Determine number of rows from the first non-empty column
    let num_rows = trace
        .binary_poly
        .first()
        .map(|c| c.len())
        .or_else(|| trace.arbitrary_poly.first().map(|c| c.len()))
        .or_else(|| trace.int.first().map(|c| c.len()))
        .unwrap_or(0);

    // Preallocate the result matrix with the correct number of rows and columns.
    // (We have to work around the fact that cloned Vec doesn't keep its capacity)
    let mut result: RowMajorTrace<F> = iter::repeat_with(|| Vec::with_capacity(num_cols))
        .take(num_rows)
        .collect();

    // Build row-by-row
    cfg_iter_mut!(result)
        .enumerate()
        .for_each(|(row_idx, row)| {
            let spare = row.spare_capacity_mut();

            // Binary poly columns
            cfg_iter_mut!(spare[..binary_len])
                .zip(cfg_iter!(trace.binary_poly))
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
                .zip(cfg_iter!(trace.arbitrary_poly))
                .for_each(|(slot, col)| {
                    let arbitrary_poly = &col.evaluations[row_idx];
                    slot.write(
                        arbitrary_poly
                            .iter()
                            .map(|coeff| F::from_with_cfg(coeff, field_cfg))
                            .collect(),
                    );
                });

            // Int columns
            cfg_iter_mut!(spare[binary_len + arbitrary_len..])
                .zip(cfg_iter!(trace.int))
                .for_each(|(slot, col)| {
                    let int_val = &col.evaluations[row_idx];
                    slot.write(DynamicPolynomialF {
                        coeffs: vec![F::from_with_cfg(int_val, field_cfg)],
                    });
                });

            // SAFETY: All slots have been initialized above.
            unsafe { row.set_len(num_cols) };
        });
    result
}

/// Project a multi-typed trace onto `F[X]`, returning a column-indexed matrix.
/// Result: `trace[col]` is a
/// `DenseMultilinearExtension<DynamicPolynomialF<F>>`.
///
/// Use this for the MLE-first approach (linear constraints only).
#[allow(clippy::arithmetic_side_effects)]
pub fn project_trace_coeffs_column_major<F, PolyCoeff, Int, const DEGREE_PLUS_ONE: usize>(
    trace: &UairTrace<PolyCoeff, Int, DEGREE_PLUS_ONE>,
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
        trace.binary_poly.first().map(|c| c.num_vars),
        trace.arbitrary_poly.first().map(|c| c.num_vars),
        trace.int.first().map(|c| c.num_vars),
    ]
    .into_iter()
    .flatten()
    .max()
    .unwrap_or(0);

    let mut result =
        Vec::with_capacity(trace.binary_poly.len() + trace.arbitrary_poly.len() + trace.int.len());

    // Binary poly columns
    cfg_extend!(
        result,
        cfg_iter!(trace.binary_poly).map(|column| {
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
        cfg_iter!(trace.arbitrary_poly).map(|column| {
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
        cfg_iter!(trace.int).map(|column| {
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
    let zero = F::zero_with_cfg(projecting_element.cfg());
    let one = F::one_with_cfg(projecting_element.cfg());

    let max_coeffs_len = trace
        .iter()
        .flat_map(|row| row.iter())
        .map(|poly| poly.degree().map_or(0, |d| d + 1))
        .max()
        .unwrap_or(0)
        .max(1);
    let projection_powers: Vec<F> = powers(projecting_element.clone(), one, max_coeffs_len);

    // Build column-indexed MLEs from row-indexed trace
    cfg_into_iter!(0..num_cols)
        .map(|col_idx| {
            let evaluations: Vec<F::Inner> = (0..num_rows)
                .map(|row_idx| {
                    let poly = &trace[row_idx][col_idx];
                    let deg = poly.degree().map_or(0, |d| d + 1);
                    DynamicPolyFInnerProduct::inner_product::<UNCHECKED>(
                        &poly.coeffs[..deg],
                        &projection_powers[..deg],
                        zero.clone(),
                    )
                    .expect("inner product cannot fail here")
                    .inner()
                    .clone()
                })
                .collect();
            DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                evaluations,
                zero.inner().clone(),
            )
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
#[allow(clippy::arithmetic_side_effects)]
pub fn project_scalars_to_field<R: Semiring + 'static, F: PrimeField>(
    scalars: HashMap<R, DynamicPolynomialF<F>>,
    projecting_element: &F,
) -> Result<HashMap<R, F>, (R, F, EvaluationError)> {
    // TODO(Ilia): Parallelising this might be good for big UAIRs.
    //             We'd conditionally route between sequential and parallel
    //             projection depending on how many scalars the UAIR has.
    let one = F::one_with_cfg(projecting_element.cfg());
    let zero = F::zero_with_cfg(projecting_element.cfg());

    let max_coeffs_len = scalars
        .values()
        .map(|poly| poly.degree().map_or(0, |d| d + 1))
        .max()
        .unwrap_or(0)
        .max(1);

    let projection_powers: Vec<F> = powers(projecting_element.clone(), one, max_coeffs_len);

    Ok(scalars
        .into_iter()
        .map(|(scalar, value)| {
            let deg = value.degree().map_or(0, |d| d + 1);
            (
                scalar,
                DynamicPolyFInnerProduct::inner_product::<UNCHECKED>(
                    &value.coeffs[..deg],
                    &projection_powers[..deg],
                    zero.clone(),
                )
                .expect("inner product cannot fail here"),
            )
        })
        .collect())
}
