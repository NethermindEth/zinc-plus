#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crypto_primitives::{FieldConfig, ProjectElementWithConfig, Semiring, SemiringConfig, Wrapper};
use std::{collections::HashMap, iter};
use zinc_poly::{
    EvaluationError,
    mle::DenseMultilinearExtension,
    univariate::dynamic::{DynamicPolynomial, HasDynamicPolynomialConfig},
};
use zinc_uair::{BitOpSpec, Uair, UairTrace, collect_scalars::collect_scalars};
use zinc_utils::{
    UNCHECKED, cfg_extend, cfg_into_iter, cfg_iter, cfg_iter_mut,
    inner_product::{InnerProduct, NativeInnerProduct},
    powers,
};

/// Row-indexed trace matrix: `trace[row][col]`.
/// Each row contains all column values for that row.
/// Used by `evaluate_for_constraints`.
pub type RowMajorTrace<F> = Vec<Vec<DynamicPolynomial<F>>>;

/// Column-indexed trace matrix: `trace[col][row]`.
/// Each column is a `DenseMultilinearExtension` over the hypercube.
/// Used by `evaluate_combined_polynomials` (MLE-first approach).
pub type ColumnMajorTrace<F> = Vec<DenseMultilinearExtension<DynamicPolynomial<F>>>;

/// Holds the projected trace in either row-major or column-major layout,
/// depending on which ideal check approach (MLE-first or combined) is used.
#[derive(Clone, Debug)]
pub enum ProjectedTrace<F> {
    RowMajor(RowMajorTrace<F>),
    ColumnMajor(ColumnMajorTrace<F>),
}

#[derive(Clone, Debug)]
pub struct ProjectedScalars<From: Semiring, To: Clone> {
    inner: HashMap<From, To>,
}

impl<From: Semiring, To: Clone> ProjectedScalars<From, To> {
    // TODO(alex): Maybe return results?
    #[inline]
    pub fn get(&self, scalar: &From) -> Option<To> {
        // TODO(alex): Lookup key often is DensePolynomial<R, 32> which is relatively
        //             expensive to hash. If this becomes a bottleneck we can consider
        //             using e.g. a raw point cache or something.
        self.inner.get(scalar).cloned()
    }
}

/// Project a multi-typed trace onto F[X], returning a row-indexed (transposed)
/// matrix. Result: `trace[row][col]` where columns are ordered as binary_poly,
/// arbitrary_poly, int.
///
/// Use this for the combined polynomial approach.
#[allow(clippy::arithmetic_side_effects)]
pub fn project_trace_coeffs_row_major<C, PolyCoeff, Int, const DB: usize, const DA: usize>(
    trace: &UairTrace<PolyCoeff, Int, DB, DA>,
    field_cfg: &C,
) -> RowMajorTrace<C::Element>
where
    C: FieldConfig + ProjectElementWithConfig<PolyCoeff> + ProjectElementWithConfig<Int>,
    PolyCoeff: Clone + Send + Sync,
    Int: Clone + Send + Sync,
{
    let zero = field_cfg.zero();
    let one = field_cfg.one();

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
    let mut result: RowMajorTrace<C::Element> = iter::repeat_with(|| Vec::with_capacity(num_cols))
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
                            .map(|coeff| field_cfg.project(coeff))
                            .collect(),
                    );
                });

            // Int columns
            cfg_iter_mut!(spare[binary_len + arbitrary_len..])
                .zip(cfg_iter!(trace.int))
                .for_each(|(slot, col)| {
                    let int_val = &col.evaluations[row_idx];
                    slot.write(DynamicPolynomial {
                        coeffs: vec![field_cfg.project(int_val)],
                    });
                });

            // SAFETY: All slots have been initialized above.
            unsafe { row.set_len(num_cols) };
        });
    result
}

/// Project a multi-typed trace onto `F[X]`, returning a column-indexed matrix.
/// Result: `trace[col]` is a
/// `DenseMultilinearExtension<DynamicPolynomial<F>>`.
///
/// Use this for the MLE-first approach.
#[allow(clippy::arithmetic_side_effects)]
pub fn project_trace_coeffs_column_major<C, PolyCoeff, Int, const DB: usize, const DA: usize>(
    trace: &UairTrace<PolyCoeff, Int, DB, DA>,
    field_cfg: &C,
) -> ColumnMajorTrace<C::Element>
where
    C: FieldConfig + ProjectElementWithConfig<PolyCoeff> + ProjectElementWithConfig<Int>,
    PolyCoeff: Clone + Send + Sync,
    Int: Clone + Send + Sync,
{
    let zero = field_cfg.zero();
    let one = field_cfg.one();

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
            let evaluations: Vec<DynamicPolynomial<C::Element>> = column
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
            let evaluations: Vec<DynamicPolynomial<C::Element>> = column
                .iter()
                .map(|arbitrary_poly| {
                    arbitrary_poly
                        .iter()
                        .map(|coeff| field_cfg.project(coeff))
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
            let evaluations: Vec<DynamicPolynomial<C::Element>> = column
                .iter()
                .map(|int| DynamicPolynomial {
                    coeffs: vec![field_cfg.project(int)],
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

/// Transpose a column-indexed trace into a row-indexed trace.
///
/// `result[row][col] = trace[col].evaluations[row].clone()`. Used by the
/// MLE-first prover when it needs to fall back to the row-major
/// `evaluate_for_constraints` path for non-linear constraints.
pub fn column_major_to_row_major<F: Clone + Send + Sync>(
    trace: &ColumnMajorTrace<F>,
) -> RowMajorTrace<F> {
    let num_rows = trace.first().map(|c| c.evaluations.len()).unwrap_or(0);

    cfg_into_iter!(0..num_rows)
        .map(|row_idx| {
            trace
                .iter()
                .map(|col| col.evaluations[row_idx].clone())
                .collect()
        })
        .collect()
}

/// Evaluate a projected trace along `F[X] -> F` and return column-indexed
/// MLEs (`Vec<DenseMultilinearExtension<F>>`) for sumcheck
/// compatibility. Dispatches on the trace layout internally.
#[allow(clippy::arithmetic_side_effects)]
pub fn evaluate_trace_to_column_mles<C>(
    field_cfg: &C,
    trace: &ProjectedTrace<C::Element>,
    projecting_element: &C::Element,
) -> Vec<DenseMultilinearExtension<C::Element>>
where
    C: FieldConfig,
{
    let zero = field_cfg.zero();
    let poly_cfg = field_cfg.dyn_poly_cfg();

    let max_coeffs_len = {
        // Iterators have different types, so this is easier
        macro_rules! common_code {
            ($v:expr) => {
                $v.iter()
                    .flat_map(|row| row.iter())
                    .map(|poly| poly_cfg.degree(poly).map_or(0, |d| d + 1))
                    .max()
                    .unwrap_or(0)
                    .max(1)
            };
        }
        match trace {
            ProjectedTrace::RowMajor(t) => common_code!(t),
            ProjectedTrace::ColumnMajor(t) => common_code!(t),
        }
    };

    let projection_powers: Vec<C::Element> = powers(field_cfg, projecting_element, max_coeffs_len);

    let evaluate_poly = |poly: &DynamicPolynomial<C::Element>| -> C::Element {
        let deg = field_cfg.dyn_poly_cfg().degree(poly).map_or(0, |d| d + 1);
        NativeInnerProduct::inner_product::<UNCHECKED>(
            field_cfg,
            &poly.coeffs[..deg],
            &projection_powers[..deg],
            zero.clone(),
        )
        .expect("inner product cannot fail here")
    };

    match trace {
        ProjectedTrace::RowMajor(t) => {
            let num_rows = t.len();
            let num_cols = t.first().map(|r| r.len()).unwrap_or(0);
            let num_vars = num_rows.next_power_of_two().trailing_zeros() as usize;

            cfg_into_iter!(0..num_cols)
                .map(|col_idx| {
                    let evaluations: Vec<C::Element> = (0..num_rows)
                        .map(|row_idx| evaluate_poly(&t[row_idx][col_idx]))
                        .collect();
                    DenseMultilinearExtension::from_evaluations_vec(
                        num_vars,
                        evaluations,
                        zero.clone(),
                    )
                })
                .collect()
        }
        ProjectedTrace::ColumnMajor(t) => cfg_iter!(t)
            .map(|col_mle| {
                let evaluations: Vec<C::Element> = cfg_iter!(col_mle).map(evaluate_poly).collect();
                DenseMultilinearExtension::from_evaluations_vec(
                    col_mle.num_vars,
                    evaluations,
                    zero.clone(),
                )
            })
            .collect(),
    }
}

/// Build the projected MLE of a bit-op virtual column.
///
/// The bit operation is applied to each source cell's coefficients before
/// evaluating the cell at `projecting_element`.
pub fn build_bit_op_virtual_mle<C, const D: usize>(
    trace: &ProjectedTrace<C::Element>,
    spec: &BitOpSpec,
    projecting_element: &C::Element,
    field_cfg: &C,
) -> DenseMultilinearExtension<C::Element>
where
    C: FieldConfig,
{
    let zero = field_cfg.zero();
    let projection_powers: Vec<C::Element> = powers(field_cfg, projecting_element, D);

    let c = spec.op().count();
    assert!(
        c > 0 && c < D,
        "BitOp count {c} out of range for cell width D = {D}",
    );

    let evaluate_with_bit_op = |cell: &DynamicPolynomial<C::Element>| -> C::Element {
        let transformed = spec.op().transform::<C, D>(cell, field_cfg);
        NativeInnerProduct::inner_product::<UNCHECKED>(
            field_cfg,
            &transformed.coeffs,
            &projection_powers,
            zero.clone(),
        )
        .expect("inner product cannot fail here")
    };

    match trace {
        ProjectedTrace::RowMajor(t) => {
            let num_rows = t.len();
            let num_vars = num_rows.next_power_of_two().trailing_zeros() as usize;
            let evaluations: Vec<C::Element> = (0..num_rows)
                .map(|row_idx| evaluate_with_bit_op(&t[row_idx][spec.source_col()]))
                .collect();
            DenseMultilinearExtension::from_evaluations_vec(num_vars, evaluations, zero.clone())
        }
        ProjectedTrace::ColumnMajor(t) => {
            let col_mle = &t[spec.source_col()];
            let evaluations: Vec<C::Element> = col_mle.iter().map(evaluate_with_bit_op).collect();
            DenseMultilinearExtension::from_evaluations_vec(
                col_mle.num_vars,
                evaluations,
                zero.clone(),
            )
        }
    }
}

/// Project scalars of a UAIR onto $F[X]$.
pub fn project_scalars<C: SemiringConfig, U: Uair>(
    field_cfg: &C,
    project: impl Fn(&U::Scalar) -> DynamicPolynomial<C::Element>,
) -> ProjectedScalars<U::Scalar, DynamicPolynomial<C::Element>> {
    let uair_scalars = collect_scalars::<U>();
    let poly_cfg = field_cfg.dyn_poly_cfg();

    // TODO(Ilia): if there's a lot of scalars
    //             we should do this in parallel probably.
    let inner = uair_scalars
        .into_iter()
        .map(|scalar| {
            let mut dynamic_poly = project(&scalar);
            poly_cfg.trim(&mut dynamic_poly);
            (scalar, dynamic_poly)
        })
        .collect();

    ProjectedScalars { inner }
}

/// Project scalars of a UAIR along F[X] -> F.
#[allow(clippy::arithmetic_side_effects, clippy::type_complexity)]
pub fn project_scalars_to_field<R: Semiring, C: FieldConfig>(
    field_cfg: &C,
    scalars: ProjectedScalars<R, DynamicPolynomial<C::Element>>,
    projecting_element: &C::Element,
) -> Result<ProjectedScalars<R, C::Element>, (R, C::Element, EvaluationError)> {
    // TODO(Ilia): Parallelising this might be good for big UAIRs.
    //             We'd conditionally route between sequential and parallel
    //             projection depending on how many scalars the UAIR has.
    let zero = field_cfg.zero();
    let poly_cfg = field_cfg.dyn_poly_cfg();

    let max_coeffs_len = scalars
        .inner
        .values()
        .map(|poly| poly_cfg.degree(poly).map_or(0, |d| d + 1))
        .max()
        .unwrap_or(0)
        .max(1);

    let projection_powers: Vec<C::Element> = powers(field_cfg, projecting_element, max_coeffs_len);

    let inner = scalars
        .inner
        .into_iter()
        .map(|(scalar, value)| {
            let deg = poly_cfg.degree(&value).map_or(0, |d| d + 1);
            (
                scalar,
                NativeInnerProduct::inner_product::<UNCHECKED>(
                    field_cfg,
                    &value.coeffs[..deg],
                    &projection_powers[..deg],
                    zero.clone(),
                )
                .expect("inner product cannot fail here"),
            )
        })
        .collect();

    Ok(ProjectedScalars { inner })
}

#[cfg(test)]
#[allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::clone_on_copy,
    clippy::redundant_clone
)]
mod tests {
    use super::*;
    use crate::test_utils::{LIMBS, test_config};
    use crypto_primitives::crypto_bigint_monty::{MontyField, MontyFieldElement};
    use zinc_uair::BitOp;
    use zinc_utils::inner_product::NativeInnerProduct;

    type F = MontyField<LIMBS>;
    type E = MontyFieldElement<LIMBS>;

    fn f(value: u32, cfg: &F) -> E {
        cfg.project(&value)
    }

    fn poly(coeffs: Vec<E>) -> DynamicPolynomial<E> {
        DynamicPolynomial { coeffs }
    }

    fn expected_eval(coeffs: &[E], projecting_element: &E, cfg: &F) -> E {
        let powers = powers(cfg, projecting_element, coeffs.len());
        NativeInnerProduct::inner_product::<UNCHECKED>(cfg, coeffs, &powers, cfg.zero())
            .expect("inner product cannot fail here")
    }

    #[test]
    fn builds_bit_op_virtual_mle_from_row_major_trace() {
        let cfg = test_config();
        let alpha = f(2, &cfg);
        let zero = cfg.zero();
        let row0 = [f(1, &cfg), f(2, &cfg), f(3, &cfg), f(4, &cfg)];
        let row1 = [f(5, &cfg), f(6, &cfg), f(7, &cfg), f(8, &cfg)];
        let trace =
            ProjectedTrace::RowMajor(vec![vec![poly(row0.to_vec())], vec![poly(row1.to_vec())]]);

        let mle = build_bit_op_virtual_mle::<F, 4>(
            &trace,
            &BitOpSpec::new(0, BitOp::ShR(2)),
            &alpha,
            &cfg,
        );

        assert_eq!(mle.num_vars, 1);
        assert_eq!(
            mle.evaluations,
            vec![
                expected_eval(
                    &[row0[2].clone(), row0[3].clone(), zero.clone(), zero.clone()],
                    &alpha,
                    &cfg
                ),
                expected_eval(
                    &[row1[2].clone(), row1[3].clone(), zero.clone(), zero],
                    &alpha,
                    &cfg
                ),
            ],
        );
    }

    #[test]
    fn builds_bit_op_virtual_mle_from_column_major_trace() {
        let cfg = test_config();
        let alpha = f(3, &cfg);
        let row0 = [f(1, &cfg), f(2, &cfg), f(3, &cfg), f(4, &cfg)];
        let row1 = [f(5, &cfg), f(6, &cfg), f(7, &cfg), f(8, &cfg)];
        let trace = ProjectedTrace::ColumnMajor(vec![DenseMultilinearExtension {
            evaluations: vec![poly(row0.to_vec()), poly(row1.to_vec())],
            num_vars: 1,
        }]);

        let mle = build_bit_op_virtual_mle::<F, 4>(
            &trace,
            &BitOpSpec::new(0, BitOp::Rot(1)),
            &alpha,
            &cfg,
        );

        assert_eq!(mle.num_vars, 1);
        assert_eq!(
            mle.evaluations,
            vec![
                expected_eval(
                    &[
                        row0[1].clone(),
                        row0[2].clone(),
                        row0[3].clone(),
                        row0[0].clone()
                    ],
                    &alpha,
                    &cfg,
                ),
                expected_eval(
                    &[
                        row1[1].clone(),
                        row1[2].clone(),
                        row1[3].clone(),
                        row1[0].clone()
                    ],
                    &alpha,
                    &cfg,
                ),
            ],
        );
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn rejects_bit_op_count_at_cell_width() {
        let cfg = test_config();
        let alpha = f(2, &cfg);
        let trace = ProjectedTrace::RowMajor(vec![vec![poly(vec![f(1, &cfg), f(2, &cfg)])]]);

        let _ = build_bit_op_virtual_mle::<F, 2>(
            &trace,
            &BitOpSpec::new(0, BitOp::Rot(2)),
            &alpha,
            &cfg,
        );
    }
}
