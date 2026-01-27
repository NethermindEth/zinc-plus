use rand::{rngs::StdRng, seq::SliceRandom};
use rand_core::SeedableRng;

/// Computes a linear combination of multiple evaluation rows into a single
/// combined row.
///
/// Given a flat `evaluations` iterator, interpreted as a matrix with `row_len`
/// columns, this function treats each consecutive `row_len` values as one row.
/// The output is a single row, computed by multiplying each input row by the
/// corresponding coefficient from `coeffs`, and summing these scaled rows
/// column-wise.
///
/// This is equivalent to performing a matrix-vector multiplication where the
/// matrix is formed by the evaluations and the vector is formed by the
/// coefficients.
///
/// # Arguments
///
/// - `coeffs`: Coefficients applied to each row.
/// - `evals_iter`: Iterator over flattened evaluations arranged row-wise.
/// - `convert_eval`: Function to convert each evaluation to the desired type.
///   Should be just `Ok::<_, ZipError>` if no conversion is
///   needed.
/// - `add_scaled`: Function to add a scaled evaluation to an accumulator.
/// - `row_len`: Number of columns per evaluation row.
/// - `zero`: Additive neutral element of `El`.
///
/// # Returns
///
/// A vector of length `row_len` representing the combined row.
#[macro_export]
#[allow(unused_imports)]
macro_rules! combine_rows {
    ($coeffs:expr, $evals_iter:expr, $convert_eval:expr, $add_scaled:expr, $row_len:expr, $zero:expr) => {{
        let row_len = $row_len;
        let mut combined_row = Vec::with_capacity(row_len);

        cfg_iter_mut!(combined_row.spare_capacity_mut())
            .enumerate()
            .try_for_each(|(column, combined)| -> Result<(), ZipError> {
                let mut acc = $zero;

                for (eval, coeff) in $evals_iter
                    .skip(column)
                    .step_by(row_len)
                    .zip($coeffs.iter())
                {
                    let eval = $convert_eval(eval)?;
                    let scaled = eval
                        .mul_by_scalar(coeff)
                        .expect("Cannot multiply evaluation by coefficient");
                    acc = $add_scaled(acc, scaled);
                }

                *combined = std::mem::MaybeUninit::new(acc);
                Ok(())
            })?;

        // Safety: We initialized all elements in the combined_row.
        unsafe {
            combined_row.set_len(row_len);
        }

        combined_row
    }};
}

/// Reorder the elements in slice using the given randomness seed
pub(super) fn shuffle_seeded<T>(slice: &mut [T], seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    slice.shuffle(&mut rng);
}

#[cfg(test)]
mod test {
    use crate::ZipError;
    use zinc_utils::{cfg_iter_mut, mul_by_scalar::MulByScalar};

    #[cfg(feature = "parallel")]
    use rayon::prelude::*;

    #[test]
    fn test_basic_combination() -> Result<(), ZipError> {
        let coeffs = [1, 2];
        let evaluations = [3, 4, 5, 6];
        let row_len = 2;

        let result = combine_rows!(
            &coeffs,
            evaluations.iter(),
            Ok::<_, ZipError>,
            |acc, scaled| acc + scaled,
            row_len,
            0_i32
        );

        assert_eq!(result, vec![(3 + 2 * 5), (4 + 2 * 6)]);
        Ok(())
    }

    #[test]
    fn test_second_combination() -> Result<(), ZipError> {
        let coeffs = [3, 4];
        let evaluations = [2, 4, 6, 8];
        let row_len = 2;

        let result = combine_rows!(
            &coeffs,
            evaluations.iter(),
            Ok::<_, ZipError>,
            |acc, scaled| acc + scaled,
            row_len,
            0_i32
        );

        assert_eq!(result, vec![(3 * 2 + 4 * 6), (3 * 4 + 4 * 8)]);
        Ok(())
    }

    #[test]
    fn test_large_values() -> Result<(), ZipError> {
        let coeffs = [1000, -500];
        let evaluations = [2000, -3000, 4000, -5000];
        let row_len = 2;

        let result = combine_rows!(
            &coeffs,
            evaluations.iter(),
            Ok::<_, ZipError>,
            |acc, scaled| acc + scaled,
            row_len,
            0_i32
        );

        assert_eq!(
            result,
            vec![
                (1000 * 2000 + (-500) * 4000),
                (1000 * -3000 + (-500) * -5000)
            ]
        );
        Ok(())
    }
}
