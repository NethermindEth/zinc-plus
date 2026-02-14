use crate::{code::LinearCode, pcs::structs::ZipTypes, utils::shuffle_seeded};
use crypto_primitives::PrimeField;
use num_traits::CheckedAdd;
use std::{marker::PhantomData, ops::AddAssign};
use zinc_poly::ConstCoeffBitWidth;
use zinc_utils::{add, from_ref::FromRef, mul};

pub trait RaaConfig: Copy + Send + Sync {
    /// Whether to permute the codeword in place, instead of copying it using a
    /// precomputed permutation.
    const PERMUTE_IN_PLACE: bool;
    /// Whether to check for overflows during encoding
    // TODO: Unify with `CHECK_FOR_OVERFLOW` in `zinc_poly`
    const CHECK_FOR_OVERFLOWS: bool;
}

/// Implementation of a repeat-accumulate-accumulate (RAA) codes over the binary
/// field, as defined by the Blaze paper (https://eprint.iacr.org/2024/1609)
#[derive(Debug, Clone)]
pub struct RaaCode<Zt: ZipTypes, Config: RaaConfig, const REP: usize> {
    pub(crate) row_len: usize,
    /// Randomness seed for the first permutation
    pub(crate) perm_1_seed: u64,

    /// Randomness seed for the second permutation
    pub(crate) perm_2_seed: u64,

    /// First permutation
    pub(crate) perm_1: Vec<usize>,

    /// Second permutation
    pub(crate) perm_2: Vec<usize>,

    phantom: PhantomData<(Zt, Config)>,
}

impl<Zt: ZipTypes, Config: RaaConfig, const REP: usize> RaaCode<Zt, Config, REP> {
    /// Do the actual encoding, as per RAA spec
    fn encode_inner<In, Out>(&self, row: &[In]) -> Vec<Out>
    where
        Out: CheckedAdd + for<'a> AddAssign<&'a Out> + FromRef<In> + Clone,
    {
        debug_assert_eq!(
            row.len(),
            self.row_len,
            "Row length must match the code's row length"
        );

        let mut result: Vec<Out> = repeat(row, REP);
        if Config::PERMUTE_IN_PLACE {
            shuffle_seeded(&mut result, self.perm_1_seed);
        } else {
            result = clone_shuffled(&result, &self.perm_1);
        }
        if Config::CHECK_FOR_OVERFLOWS {
            accumulate(&mut result);
        } else {
            accumulate_unchecked(&mut result);
        }
        if Config::PERMUTE_IN_PLACE {
            shuffle_seeded(&mut result, self.perm_2_seed);
        } else {
            result = clone_shuffled(&result, &self.perm_2);
        }
        if Config::CHECK_FOR_OVERFLOWS {
            accumulate(&mut result);
        } else {
            accumulate_unchecked(&mut result);
        }
        debug_assert_eq!(result.len(), self.codeword_len());
        result
    }
}

impl<Zt: ZipTypes, Config: RaaConfig, const REP: usize> LinearCode<Zt>
    for RaaCode<Zt, Config, REP>
{
    const REPETITION_FACTOR: usize = REP;

    fn new(row_len: usize) -> Self {
        assert!(
            REP.is_power_of_two(),
            "Repetition factor must be a power of two"
        );
        assert!(
            row_len.is_power_of_two(),
            "Row length must be a power of two"
        );

        // Width of each entry in codeword vector, in bits.
        // For RAA it's initial_bits + 2*log2(codeword_len),
        // where codeword_len = row_len * REP and the factor of 2
        // comes from the two accumulation steps.
        let codeword_width_bits = {
            let initial_bits =
                u32::try_from(Zt::Eval::COEFF_BIT_WIDTH).expect("Size of EvalR type is too large");

            let row_len_log = row_len.ilog2();
            let rep_factor_log = REP.ilog2();
            add!(initial_bits, add!(mul!(row_len_log, 2), mul!(rep_factor_log, 2)))
        };
        let codeword_type_bits =
            u32::try_from(Zt::Cw::COEFF_BIT_WIDTH).expect("Size of CwR type is too large");
        assert!(
            codeword_type_bits >= codeword_width_bits,
            "Cannot fit {codeword_width_bits}-bit wide codeword entries in {} bits entries",
            codeword_type_bits
        );

        // We don't need a secure/unpredictable randomness here, so use fixed seeds
        const PERM_1_SEED: u64 = 1;
        const PERM_2_SEED: u64 = 2;

        let codeword_len = mul!(row_len, REP);

        let mut perm_1: Vec<usize> = (0..codeword_len).collect();
        shuffle_seeded(&mut perm_1, PERM_1_SEED);
        let mut perm_2: Vec<usize> = (0..codeword_len).collect();
        shuffle_seeded(&mut perm_2, PERM_2_SEED);

        Self {
            row_len,
            perm_1_seed: PERM_1_SEED,
            perm_2_seed: PERM_2_SEED,
            perm_1,
            perm_2,
            phantom: PhantomData,
        }
    }

    fn row_len(&self) -> usize {
        self.row_len
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn codeword_len(&self) -> usize {
        self.row_len * REP
    }

    fn encode(&self, row: &[Zt::Eval]) -> Vec<Zt::Cw> {
        self.encode_inner(row)
    }

    fn encode_wide(&self, row: &[Zt::CombR]) -> Vec<Zt::CombR> {
        self.encode_inner(row)
    }

    fn encode_f<F>(&self, row: &[F]) -> Vec<F>
    where
        F: PrimeField + FromRef<F>,
    {
        self.encode_inner(row)
    }
}

/// Repeat the given slice N times, e.g `[1,2,3] => [1,2,3,1,2,3]`
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn repeat<In, Out: FromRef<In> + Clone>(
    input: &[In],
    repetition_factor: usize,
) -> Vec<Out> {
    input
        .iter()
        .map(Out::from_ref)
        .cycle()
        .take(input.len() * repetition_factor)
        .collect()
}

/// Perform an operation equivalent to multiplying the slice in-place by the
/// accumulation matrix from the RAA code - a lower triangular matrix of the
/// appropriate size, i.e. a matrix looking like this:
///
/// ```text
/// 1 0 0 0
/// 1 1 0 0
/// 1 1 1 0
/// 1 1 1 1
/// ```
#[allow(clippy::arithmetic_side_effects)] // Clippy is too dumb to realize `i - 1` is safe here
pub(crate) fn accumulate<I>(input: &mut [I])
where
    I: CheckedAdd + Clone,
{
    if let Some(first) = input.first().cloned() {
        let mut acc = first;
        for curr in input.iter_mut().skip(1) {
            acc = add!(*curr, acc, "Accumulation overflow");
            *curr = acc.clone();
        }
    }
}

#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn accumulate_unchecked<I>(input: &mut [I])
where
    I: for<'a> AddAssign<&'a I> + Clone,
{
    if let Some(first) = input.first().cloned() {
        let mut acc = first;
        for i in 1..input.len() {
            // Avoid bound checking
            unsafe {
                acc += input.get_unchecked(i);
                *input.get_unchecked_mut(i) = acc.clone();
            };
        }
    }
}

/// Clone the data using a precomputed permutation.
pub(crate) fn clone_shuffled<T>(data: &[T], perm: &[usize]) -> Vec<T>
where
    T: Clone,
{
    perm.iter().map(|&i| data[i].clone()).collect()
}

#[cfg(test)]
mod tests {
    use crypto_bigint::U64;
    use crypto_primitives::crypto_bigint_int::Int;
    use num_traits::Zero;

    use super::*;
    use crate::{code::LinearCode, pcs::test_utils::TestZipTypes, utils::shuffle_seeded};

    const REPETITION_FACTOR: usize = 4;

    // Define common types for testing
    const INT_LIMBS: usize = U64::LIMBS;

    const N: usize = INT_LIMBS;
    const K: usize = INT_LIMBS * 4;
    const M: usize = INT_LIMBS * 8;

    #[derive(Clone, Copy)]
    struct RaaConfigGeneric<const PERMUTE_IN_PLACE: bool, const CHECK_FOR_OVERFLOWS: bool>;

    impl<const PERMUTE_IN_PLACE: bool, const CHECK_FOR_OVERFLOWS: bool> RaaConfig
        for RaaConfigGeneric<PERMUTE_IN_PLACE, CHECK_FOR_OVERFLOWS>
    {
        const PERMUTE_IN_PLACE: bool = PERMUTE_IN_PLACE;
        const CHECK_FOR_OVERFLOWS: bool = CHECK_FOR_OVERFLOWS;
    }

    macro_rules! test_raa {
        ($zt:ty, $row_len:expr, $f:expr) => {
            test_raa!($zt, $row_len, $f, RaaConfigGeneric<false, false>);
            test_raa!($zt, $row_len, $f, RaaConfigGeneric<false, true>);
            test_raa!($zt, $row_len, $f, RaaConfigGeneric<true, false>);
            test_raa!($zt, $row_len, $f, RaaConfigGeneric<true, true>);
        };

        ($zt:ty, $row_len:expr, $f:expr, $config:ty) => {
            {
                let code = RaaCode::<$zt, $config, REPETITION_FACTOR>::new($row_len);
                $f(&code)
            }
        };
    }

    #[test]
    fn repeat_function_duplicates_row_correctly() {
        type I = Int<N>;
        let input = [10, 20].map(I::from);

        let repetition_factor = 3;

        let repeated_output = repeat::<_, I>(&input, repetition_factor);

        let expected_output: Vec<_> = [10, 20, 10, 20, 10, 20].into_iter().map(I::from).collect();
        assert_eq!(
            repeated_output, expected_output,
            "Failed on repetition factor > 1"
        );

        let empty_input: Vec<I> = vec![];
        let repeated_empty = repeat::<_, I>(&empty_input, 5);
        assert!(repeated_empty.is_empty(), "Failed on empty input vector");

        let repeated_once = repeat::<_, I>(&input, 1);
        assert_eq!(repeated_once, input, "Failed on repetition factor of 1");
    }

    #[test]
    fn accumulate_function_computes_cumulative_sum() {
        type I = Int<N>;
        let mut input1: Vec<I> = [1, 2, 3, 4].into_iter().map(I::from).collect();
        let expected1: Vec<I> = [1, 3, 6, 10].into_iter().map(I::from).collect();
        accumulate(&mut input1);
        assert_eq!(input1, expected1, "Failed on positive integers");

        let mut input1: Vec<I> = [1, 2, 3, 4].into_iter().map(I::from).collect();
        accumulate_unchecked(&mut input1);
        assert_eq!(input1, expected1, "Failed on positive integers");

        let mut input2: Vec<I> = [5, 0, 2, 0].into_iter().map(I::from).collect();
        let expected2: Vec<I> = [5, 5, 7, 7].into_iter().map(I::from).collect();
        accumulate(&mut input2);
        assert_eq!(input2, expected2, "Failed on vector with zeros");

        let mut input3: Vec<I> = [-1, 5, -10, 2].into_iter().map(I::from).collect();
        let expected3: Vec<I> = [-1, 4, -6, -4].into_iter().map(I::from).collect();
        accumulate(&mut input3);
        assert_eq!(input3, expected3, "Failed on vector with negative numbers");

        let mut empty_input: Vec<I> = vec![];
        let expected_empty: Vec<I> = vec![];
        accumulate(&mut empty_input);
        assert_eq!(empty_input, expected_empty, "Failed on empty vector");
    }

    #[test]
    fn shuffle_is_deterministic_for_a_given_seed() {
        type I = Int<N>;
        let original: Vec<I> = (1..=10).map(I::from).collect();
        let mut vec1 = original.clone();
        let mut vec2 = original.clone();
        let mut vec3 = original.clone();

        let seed1 = 12345;
        let seed2 = 54321;

        shuffle_seeded(&mut vec1, seed1);
        shuffle_seeded(&mut vec2, seed1);
        shuffle_seeded(&mut vec3, seed2);

        assert_eq!(
            vec1, vec2,
            "Shuffling with the same seed should produce the same result"
        );
        assert_ne!(
            vec1, vec3,
            "Shuffling with different seeds should produce different results"
        );
        assert_ne!(
            vec1, original,
            "Shuffled vector should not be the same as the original"
        );
        assert_ne!(
            vec3, original,
            "Shuffled vector should not be the same as the original"
        );
    }

    #[test]
    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn encoding_preserves_linearity() {
        test_raa!(TestZipTypes<N, K, M>, 4, |code: &RaaCode<_, _, _>| {
            let a: Vec<Int<N>> = (1..=4).map(Int::<N>::from).collect();
            let b: Vec<Int<N>> = (5..=8).map(Int::<N>::from).collect();
            let sum_ab: Vec<Int<N>> = a.iter().zip(b.iter()).map(|(x, y)| *x + y).collect();

            let encode_a: Vec<Int<K>> = code.encode(&a);
            let encode_b: Vec<Int<K>> = code.encode(&b);
            let encode_sum_ab: Vec<Int<K>> = code.encode(&sum_ab);

            let sum_encode_ab: Vec<Int<K>> = encode_a
                .iter()
                .zip(encode_b.iter())
                .map(|(x, y)| *x + y)
                .collect();

            assert_eq!(encode_sum_ab, sum_encode_ab);
        });
    }

    /// Since our shuffle seeds are fixed, we can test the encoding
    /// against a known output.
    #[test]
    fn encoding_produces_predictable_results() {
        let a: Vec<Int<N>> = (1..=4).map(Int::<N>::from).collect();

        test_raa!(TestZipTypes<N, K, M>, 4, |code: &RaaCode<_, _, _>| {
            let encode_a: Vec<Int<K>> = code.encode(&a);
            assert_eq!(
                encode_a,
                [
                    0x1E, 0x36, 0x39, 0x5A, 0x70, 0x7E, 0xA5, 0xC1, 0xCB, 0xDC, 0xF9, 0x11E, 0x124,
                    0x14C, 0x14D, 0x160
                ]
                .map(Int::<K>::from)
            );
        });
    }

    #[test]
    fn encoding_zero_vector_results_in_zero_codeword() {
        test_raa!(TestZipTypes<N, K, M>, 4, |code: &RaaCode<_, _, _>| {
            let zero_vector: Vec<_> = vec![Int::<N>::zero(); code.row_len()];
            let encoded_vector: Vec<Int<K>> = code.encode(&zero_vector);

            let expected_codeword: Vec<Int<K>> = vec![Int::zero(); code.codeword_len()];

            assert_eq!(
                encoded_vector, expected_codeword,
                "Encoding a zero vector should result in a zero codeword"
            );
        });
    }

    #[test]
    fn in_place_permutation_should_not_affect_order() {
        let data: Vec<Int<N>> = (1..=1024).map(Int::<N>::from).collect();
        let row_len = data.len();
        let codeword_1: Vec<Int<K>> = {
            let code_in_place = RaaCode::<
                TestZipTypes<N, K, M>,
                RaaConfigGeneric<true, true>,
                REPETITION_FACTOR,
            >::new(row_len);
            code_in_place.encode(&data)
        };

        let codeword_2: Vec<Int<K>> = {
            let code_cloning = RaaCode::<
                TestZipTypes<N, K, M>,
                RaaConfigGeneric<false, true>,
                REPETITION_FACTOR,
            >::new(row_len);
            code_cloning.encode(&data)
        };
        assert_eq!(
            codeword_1, codeword_2,
            "In-place permutation should not affect the final codeword"
        );
    }

    #[test]
    #[should_panic]
    fn constructor_panics_on_insufficient_codeword_width() {
        const N: usize = 1;
        const K: usize = 1;

        let _code = RaaCode::<
            TestZipTypes<N, K, N>,
            RaaConfigGeneric<false, true>,
            REPETITION_FACTOR,
        >::new(1 << 15);
    }

    #[test]
    #[should_panic(expected = "Row length must match the code's row length")]
    #[cfg(debug_assertions)]
    fn encode_panics_on_mismatched_row_length() {
        test_raa!(TestZipTypes<N, K, M>, 4, |code: &RaaCode<_, _, _>| {
            let incorrect_row = vec![Int::<N>::from(1), Int::<N>::from(2), Int::<N>::from(3)];
            let _: Vec<Int<K>> = code.encode(&incorrect_row);
        });
    }
}
