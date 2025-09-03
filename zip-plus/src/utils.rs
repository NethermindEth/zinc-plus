use ark_std::{
    ops::{Add, Mul},
    vec,
    vec::Vec,
};
use num_integer::Integer as NumInteger;
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};

use crate::traits::{Integer, Words};

pub(crate) fn inner_product<'a, 'b, T, L, R>(lhs: L, rhs: R) -> T
where
    T: Clone + Mul<Output = T> + Add<Output = T> + Default + 'a + 'b,
    L: IntoIterator<Item = &'a T>,
    R: IntoIterator<Item = &'b T>,
{
    lhs.into_iter()
        .zip(rhs)
        .map(|(lhs, rhs)| lhs.clone() * rhs.clone())
        .reduce(|acc, product| acc + product)
        .unwrap_or_default()
}

pub(crate) fn div_ceil(dividend: usize, divisor: usize) -> usize {
    NumInteger::div_ceil(&dividend, &divisor)
}

pub(crate) fn num_threads() -> usize {
    #[cfg(feature = "parallel")]
    return rayon::current_num_threads();

    #[cfg(not(feature = "parallel"))]
    return 1;
}

pub(crate) fn parallelize_iter<I, T, F>(iter: I, f: F)
where
    I: Send + Iterator<Item = T>,
    T: Send,
    F: Fn(T) + Send + Sync + Clone,
{
    #[cfg(feature = "parallel")]
    rayon::scope(|scope| {
        iter.for_each(|item| {
            let f = &f;
            scope.spawn(move |_| f(item))
        })
    });

    #[cfg(not(feature = "parallel"))]
    iter.for_each(f);
}

pub(crate) fn parallelize<T, F>(v: &mut [T], f: F)
where
    T: Send,
    F: Fn((&mut [T], usize)) + Send + Sync + Clone,
{
    #[cfg(feature = "parallel")]
    {
        use crate::utils::div_ceil;
        let num_threads = num_threads();
        let chunk_size = div_ceil(v.len(), num_threads);
        if chunk_size < num_threads {
            f((v, 0));
        } else {
            parallelize_iter(v.chunks_mut(chunk_size).zip((0..).step_by(chunk_size)), f);
        }
    }

    #[cfg(not(feature = "parallel"))]
    f((v, 0));
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
pub(super) fn combine_rows<'a, F, C, E>(coeffs: C, evaluations: E, row_len: usize) -> Vec<F>
where
    F: Clone
        + Default
        + Send
        + Sync
        + for<'b> ark_std::ops::AddAssign<&'b F>
        + for<'b> ark_std::ops::Mul<&'b F, Output = F>,
    C: IntoIterator<Item = F> + Sync,
    E: IntoIterator<Item = F> + Sync,
    C::IntoIter: Clone + Send + Sync,
    E::IntoIter: Clone + Send + Sync,
{
    let coeffs_iter = coeffs.into_iter();
    let evaluations_iter = evaluations.into_iter();

    let mut combined_row = vec![F::default(); row_len];
    parallelize(&mut combined_row, |(combined_row, offset)| {
        combined_row
            .iter_mut()
            .zip(offset..)
            .for_each(|(combined, column)| {
                *combined = F::default();
                coeffs_iter
                    .clone()
                    .zip(evaluations_iter.clone().skip(column).step_by(row_len))
                    .for_each(|(coeff, eval)| {
                        *combined += &(coeff * &eval);
                    });
            })
    });

    combined_row
}

pub(super) fn expand<N: Integer, M: Integer + for<'a> From<&'a N>>(narrow_int: &N) -> M {
    assert!(
        N::W::num_words() <= M::W::num_words(),
        "Cannot squeeze a wide integer into a narrow integer."
    );

    M::from(narrow_int)
}

/// Reorder the elements in slice using the given randomness seed
pub(super) fn shuffle_seeded<T>(slice: &mut [T], seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    slice.shuffle(&mut rng);
}

#[cfg(test)]
mod test {

    use crypto_bigint::Random;
    use num_traits::{ConstOne, ConstZero};

    use crate::{
        field::Int,
        traits::Integer,
        utils::{expand, inner_product},
    };
    use crate::pcs::utils::AsWords;

    #[test]
    fn test_inner_product_basic() {
        let lhs = [1, 2, 3];
        let rhs = [4, 5, 6];
        assert_eq!(inner_product(lhs.iter(), rhs.iter()), 4 + 2 * 5 + 3 * 6);
    }

    #[test]
    fn test_expand_normal() {
        let input_words = [1u64, 2u64];
        let input = Int::<2>::from(input_words);
        let expanded = expand::<Int<2>, Int<4>>(&input);

        let expected_words = [1u64, 2u64, 0u64, 0u64];
        assert_eq!(expanded.as_words(), expected_words);
    }

    #[test]
    fn test_expand_identity() {
        let input_words = [42u64, 99u64];
        let input = Int::<2>::from(input_words);
        let expanded = expand::<Int<2>, Int<2>>(&input);

        let expected_words = [42u64, 99u64];
        assert_eq!(expanded.as_words(), expected_words);
    }

    #[test]
    #[should_panic(expected = "Cannot squeeze a wide integer into a narrow integer.")]
    fn test_expand_invalid() {
        let input = Int::<4>::from([1, 2, 3, 4]);
        // N = 4, M = 2 → should panic
        let _ = expand::<Int<4>, Int<2>>(&input);
    }

    #[test]
    fn test_expand_zero_padding() {
        let input = Int::<1>::from([123]);
        let expanded = expand::<Int<1>, Int<3>>(&input);

        let expected_words = [123u64, 0u64, 0u64];
        assert_eq!(expanded.as_words(), expected_words);
    }

    #[test]
    fn test_expand_all_zeros() {
        let input = Int::<2>::from([0u64, 0u64]);
        let expanded = expand::<Int<2>, Int<4>>(&input);

        let expected_words = [0u64, 0u64, 0u64, 0u64];
        assert_eq!(expanded.as_words(), expected_words);
    }
    #[test]
    fn test_expand_negative_number_identity() {
        // Example negative number in two's complement for 2 words
        let negative_val = Int::<2>::from([!0u64, !0u64]); // -1
        let expanded = expand::<Int<2>, Int<2>>(&negative_val);

        assert_eq!(expanded, Int::<2>::ZERO - &Int::<2>::ONE);
    }

    #[test]
    fn test_expand_negative_number_wider() {
        let mut rg = ark_std::test_rng();

        let mut positive_val = Int::<2>::random(&mut rg);
        if positive_val < Int::ZERO {
            positive_val = Int::<2>::ZERO - &positive_val;
        }

        let expanded_positive = expand::<Int<2>, Int<4>>(&positive_val);

        let negative_val = Int::<2>::ZERO - &positive_val;
        let expanded_negative = expand::<Int<2>, Int<4>>(&negative_val);

        let expected_negative = Int::<4>::ZERO - &expanded_positive;

        assert_eq!(expanded_negative, expected_negative);
    }
}
