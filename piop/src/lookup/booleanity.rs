//! Algebraic booleanity check for binary_poly columns.
//!
//! Replaces an external lookup argument: we prove that each entry of a
//! `binary_poly` witness column `v` really is a binary polynomial of
//! degree `< D` by writing `v = Σ_{i=0}^{D-1} X^i · v_i` with bit-slice
//! MLEs `v_i` over `F`, and folding `v_i · (v_i - 1) · eq(r, b) = 0` into
//! the existing CPR multi-degree sumcheck.
//!
//! See [`compute_bit_slices_flat`] (prover) and
//! [`verify_bit_decomposition_consistency`] (verifier).

use crypto_primitives::PrimeField;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use thiserror::Error;
use zinc_poly::{mle::DenseMultilinearExtension, univariate::binary::BinaryPoly};
use zinc_utils::{cfg_iter, inner_transparent_field::InnerTransparentField};

/// Build bit-slice MLEs over `F::Inner` for every binary_poly column.
///
/// Output ordering is **flat, column-major-then-bit-major**: index
/// `col_idx * D + bit_idx` is `MLE<F::Inner>` whose evaluations are the
/// `bit_idx`-th bit of each row's `BinaryPoly<D>` cast to 0/1 in `F::Inner`.
///
/// Length: `trace_binary_poly.len() * D`.
#[allow(clippy::arithmetic_side_effects)]
pub fn compute_bit_slices_flat<F, const D: usize>(
    trace_binary_poly: &[DenseMultilinearExtension<BinaryPoly<D>>],
    field_cfg: &F::Config,
) -> Vec<DenseMultilinearExtension<F::Inner>>
where
    F: InnerTransparentField,
    F::Inner: Clone + Send + Sync,
{
    let zero_inner = F::zero_with_cfg(field_cfg).into_inner();
    let one_inner = F::one_with_cfg(field_cfg).into_inner();

    cfg_iter!(trace_binary_poly)
        .flat_map(|col| {
            let num_vars = col.num_vars;
            // Per-column transpose: bit_evals[bit_idx][row_idx] = bit
            let mut bit_evals: Vec<Vec<F::Inner>> = (0..D)
                .map(|_| Vec::with_capacity(col.evaluations.len()))
                .collect();
            for bp in &col.evaluations {
                for (bit_idx, coeff) in bp.iter().enumerate() {
                    bit_evals[bit_idx].push(if coeff.into_inner() {
                        one_inner.clone()
                    } else {
                        zero_inner.clone()
                    });
                }
            }
            bit_evals
                .into_iter()
                .map(move |evaluations| DenseMultilinearExtension {
                    evaluations,
                    num_vars,
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Verifier check: each binary_poly column's projected MLE evaluation at
/// `r*` (`up_evals[col_idx]`) must equal `Σ_i a^i · bit_slice_evals[i]`,
/// where `a` is the field-projection element used to send `F[X] → F`.
///
/// This is the algebraic analog of the user-described lifting check
/// (`MLE[v](r') = Σ_i X^i · MLE[v_i](r')` followed by projection): in
/// projected `F`-land, `ψ_a(MLE[v](r*)) = Σ_i a^i · MLE[v_i](r*)`. With
/// overwhelming probability over the random `a`, the equation pins down
/// each bit-slice eval against the true bit-decomposition of the
/// committed parent column.
///
/// `parent_evals_per_col` and `bit_slice_evals` are slices into the
/// CPR `up_evals` and `bit_slice_evals` proof fields; binary_poly columns
/// are assumed to occupy the prefix `[0, num_binary_poly_cols)` of the
/// trace column layout (matching `project_trace_coeffs_*` ordering).
#[allow(clippy::arithmetic_side_effects)]
pub fn verify_bit_decomposition_consistency<F: PrimeField>(
    parent_evals_per_col: &[F],
    bit_slice_evals: &[F],
    projecting_element: &F,
    bits_per_col: usize,
) -> Result<(), BooleanityError<F>> {
    if bit_slice_evals.len() != parent_evals_per_col.len() * bits_per_col {
        return Err(BooleanityError::WrongBitSliceEvalCount {
            got: bit_slice_evals.len(),
            expected: parent_evals_per_col.len() * bits_per_col,
        });
    }

    if bits_per_col == 0 {
        return Ok(());
    }

    let zero = F::zero_with_cfg(projecting_element.cfg());
    let one = F::one_with_cfg(projecting_element.cfg());

    // Powers [1, a, a^2, ..., a^{bits_per_col - 1}].
    let mut a_powers: Vec<F> = Vec::with_capacity(bits_per_col);
    let mut acc = one;
    for _ in 0..bits_per_col {
        a_powers.push(acc.clone());
        acc *= projecting_element;
    }

    for (col_idx, parent_eval) in parent_evals_per_col.iter().enumerate() {
        let base = col_idx * bits_per_col;
        let recombined =
            bit_slice_evals[base..base + bits_per_col]
                .iter()
                .zip(&a_powers)
                .fold(zero.clone(), |acc, (bit_eval, a_pow)| {
                    acc + bit_eval.clone() * a_pow
                });

        if &recombined != parent_eval {
            return Err(BooleanityError::ConsistencyMismatch {
                col_idx,
                got: recombined,
                expected: parent_eval.clone(),
            });
        }
    }

    Ok(())
}

#[derive(Debug, Error)]
pub enum BooleanityError<F: PrimeField> {
    #[error(
        "wrong bit-slice evaluation count: got {got}, expected {expected} (= num_binary_poly_cols * D)"
    )]
    WrongBitSliceEvalCount { got: usize, expected: usize },
    #[error(
        "bit-decomposition consistency mismatch on binary_poly column {col_idx}: got Σ a^i·bᵢ = {got:?}, expected parent eval {expected:?}"
    )]
    ConsistencyMismatch { col_idx: usize, got: F, expected: F },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_primitives::{boolean::Boolean, crypto_bigint_monty::MontyField};

    type F = MontyField<4>;

    fn test_cfg() -> <F as crypto_primitives::PrimeField>::Config {
        crate::test_utils::test_config()
    }

    fn col_from_u8s(patterns: &[u8]) -> DenseMultilinearExtension<BinaryPoly<8>> {
        use std::array;
        let evaluations: Vec<BinaryPoly<8>> = patterns
            .iter()
            .map(|&p| {
                let coeffs: [Boolean; 8] =
                    array::from_fn(|i| Boolean::new((p >> i) & 1 != 0));
                BinaryPoly::<8>::new(coeffs)
            })
            .collect();
        let num_vars = evaluations.len().next_power_of_two().trailing_zeros() as usize;
        DenseMultilinearExtension {
            num_vars,
            evaluations,
        }
    }

    #[test]
    fn bit_slices_round_trip_recovers_original_bits() {
        let cfg = test_cfg();
        let col = col_from_u8s(&[0b00000000, 0b11111111, 0b10101010, 0b01010101]);
        let bit_slices = compute_bit_slices_flat::<F, 8>(std::slice::from_ref(&col), &cfg);

        assert_eq!(bit_slices.len(), 8);
        let one = F::one_with_cfg(&cfg).into_inner();
        let zero = F::zero_with_cfg(&cfg).into_inner();
        for (row, p) in [0b00000000u8, 0b11111111, 0b10101010, 0b01010101]
            .iter()
            .enumerate()
        {
            for bit in 0..8 {
                let want = if (p >> bit) & 1 != 0 {
                    one.clone()
                } else {
                    zero.clone()
                };
                assert_eq!(bit_slices[bit].evaluations[row], want, "row {row} bit {bit}");
            }
        }
    }

    #[test]
    fn consistency_check_accepts_honest_decomposition() {
        let cfg = test_cfg();
        let one = F::one_with_cfg(&cfg);
        let zero = F::zero_with_cfg(&cfg);

        // Synthesize bit-slice evals as field elements directly (some 0/1
        // pattern) and compute the matching parent eval = Σ_i b_i · a^i.
        let bits: [u32; 8] = [1, 0, 1, 1, 0, 0, 0, 1]; // 0b10001101 = 0x8d
        let bit_evals: Vec<F> = bits
            .iter()
            .map(|&b| if b == 1 { one.clone() } else { zero.clone() })
            .collect();
        let a = one.clone() + one.clone() + one.clone(); // a = 3

        let mut parent_eval = zero.clone();
        let mut a_pow = one.clone();
        for be in &bit_evals {
            parent_eval = parent_eval + be.clone() * a_pow.clone();
            a_pow = a_pow * a.clone();
        }

        verify_bit_decomposition_consistency(
            std::slice::from_ref(&parent_eval),
            &bit_evals,
            &a,
            8,
        )
        .expect("honest decomposition should satisfy consistency check");
    }

    #[test]
    fn consistency_check_rejects_tampered_bit() {
        let cfg = test_cfg();
        let one = F::one_with_cfg(&cfg);
        let zero = F::zero_with_cfg(&cfg);

        let bits: [u32; 4] = [1, 1, 1, 1];
        let bit_evals: Vec<F> = bits
            .iter()
            .map(|&b| if b == 1 { one.clone() } else { zero.clone() })
            .collect();
        let a = one.clone() + one.clone(); // a = 2

        let mut parent_eval = zero.clone();
        let mut a_pow = one.clone();
        for be in &bit_evals {
            parent_eval = parent_eval + be.clone() * a_pow.clone();
            a_pow = a_pow * a.clone();
        }

        let mut tampered = bit_evals.clone();
        tampered[0] = tampered[0].clone() + one;

        let res = verify_bit_decomposition_consistency(
            std::slice::from_ref(&parent_eval),
            &tampered,
            &a,
            4,
        );
        assert!(matches!(res, Err(BooleanityError::ConsistencyMismatch { .. })));
    }

    #[test]
    fn consistency_check_no_op_when_no_binary_poly_columns() {
        let cfg = test_cfg();
        let one = F::one_with_cfg(&cfg);
        let parent_evals: Vec<F> = vec![];
        let bit_evals: Vec<F> = vec![];
        verify_bit_decomposition_consistency(&parent_evals, &bit_evals, &one, 8).unwrap();
    }

    #[test]
    fn consistency_check_rejects_wrong_eval_count() {
        let cfg = test_cfg();
        let one = F::one_with_cfg(&cfg);
        let parent_evals = vec![one.clone()]; // 1 binary_poly column
        let bit_evals: Vec<F> = vec![one.clone(), one.clone()]; // only 2 bits, expected 8
        let res = verify_bit_decomposition_consistency(&parent_evals, &bit_evals, &one, 8);
        assert!(matches!(res, Err(BooleanityError::WrongBitSliceEvalCount { .. })));
    }
}
