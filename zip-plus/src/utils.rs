use ark_std::cfg_iter_mut;
use num_traits::CheckedAdd;
use rand::{rngs::StdRng, seq::SliceRandom};
use rand_core::SeedableRng;
use std::{
    iter::{Iterator, Sum},
    mem::MaybeUninit,
};
use zinc_utils::{add, mul_by_scalar::MulByScalar};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub(crate) fn inner_product<'a, 'b, Coeff, El, L, R>(lhs: L, rhs: R, zero: El) -> El
where
    Coeff: Clone + 'a + 'b,
    El: Clone + CheckedAdd + for<'z> MulByScalar<&'z Coeff> + 'a + 'b,
    L: IntoIterator<Item = &'a Coeff>,
    R: IntoIterator<Item = &'b El>,
{
    lhs.into_iter()
        .zip(rhs)
        .map(|(lhs, rhs)| {
            rhs.mul_by_scalar(lhs)
                .expect("Cannot multiply a codeword element by a coefficient")
        })
        .reduce(|acc, product| add!(acc, &product))
        .unwrap_or(zero)
}

/// Computes a linear combination of multiple evaluation rows into a single
/// combined row.
///
/// Given a flat `evaluations` vector, interpreted as a matrix with `row_len`
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
/// - `evaluations`: Flattened evaluations arranged row-wise.
/// - `row_len`: Number of columns per evaluation row.
///
/// # Returns
///
/// A vector of length `row_len` representing the combined row.
pub(super) fn combine_rows<Coeff, El>(
    coeffs: &[Coeff],
    evaluations: &[El],
    row_len: usize,
) -> Vec<El>
where
    Coeff: Send + Sync,
    El: Clone + Sum + for<'z> MulByScalar<&'z Coeff> + Send + Sync,
{
    let mut combined_row = Vec::with_capacity(row_len);

    cfg_iter_mut!(combined_row.spare_capacity_mut())
        .enumerate()
        .for_each(|(column, combined)| {
            *combined = MaybeUninit::new(
                coeffs
                    .iter()
                    .zip(evaluations.iter().skip(column).step_by(row_len))
                    .map(|(coeff, eval)| {
                        eval.mul_by_scalar(coeff)
                            .expect("Cannot multiply evaluation by coefficient")
                    })
                    .sum(),
            );
        });

    // Safety: We initialized all elements in the combined_row.
    unsafe {
        combined_row.set_len(row_len);
    }

    combined_row
}

/// Reorder the elements in slice using the given randomness seed
pub(super) fn shuffle_seeded<T>(slice: &mut [T], seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    slice.shuffle(&mut rng);
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_inner_product_basic() {
        let lhs = [1, 2, 3];
        let rhs = [4, 5, 6];
        assert_eq!(inner_product(lhs.iter(), rhs.iter(), 0), 4 + 2 * 5 + 3 * 6);
    }

    #[test]
    fn test_basic_combination() {
        let coeffs = vec![1, 2];
        let evaluations = vec![3, 4, 5, 6];
        let row_len = 2;

        let result = combine_rows(&coeffs, &evaluations, row_len);

        assert_eq!(result, vec![(3 + 2 * 5), (4 + 2 * 6)]);
    }

    #[test]
    fn test_second_combination() {
        let coeffs = vec![3, 4];
        let evaluations = vec![2, 4, 6, 8];
        let row_len = 2;

        let result = combine_rows(&coeffs, &evaluations, row_len);

        assert_eq!(result, vec![(3 * 2 + 4 * 6), (3 * 4 + 4 * 8)]);
    }
    #[test]
    fn test_large_values() {
        let coeffs = vec![1000, -500];
        let evaluations = vec![2000, -3000, 4000, -5000];
        let row_len = 2;

        let result = combine_rows(&coeffs, &evaluations, row_len);

        assert_eq!(
            result,
            vec![
                (1000 * 2000 + (-500) * 4000),
                (1000 * -3000 + (-500) * -5000)
            ]
        );
    }
}
