use crypto_primitives::{PrimeField, Ring};
use num_traits::{CheckedAdd, Zero};
use std::{marker::PhantomData, ops::AddAssign};

use crate::{
    add,
    code::{LinearCode, LinearCodeSpec},
    mul, sub,
    traits::{FromRef, Transcript},
    utils::shuffle_seeded,
};

/// Implementation of a repeat-accumulate-accumulate (RAA) codes over the binary
/// field, as defined by the Blaze paper (https://eprint.iacr.org/2024/1609)
#[derive(Debug, Clone)]
pub struct RaaCode<Eval: Ring, Cw: Ring, Comb: Ring> {
    /// Whether to check for overflows during encoding
    check_for_overflows: bool,

    row_len: usize,

    repetition_factor: usize,

    num_column_opening: usize,

    num_proximity_testing: usize,

    /// Randomness seed for the first permutation
    perm_1_seed: u64,

    /// Randomness seed for the second permutation
    perm_2_seed: u64,

    phantom: PhantomData<(Eval, Cw, Comb)>,
}

impl<Eval, Cw, Comb> RaaCode<Eval, Cw, Comb>
where
    Eval: Ring,
    Cw: Ring + FromRef<Eval>,
    Comb: Ring + FromRef<Comb>,
{
    pub fn new<S: LinearCodeSpec, T: Transcript>(
        spec: &S,
        poly_size: usize,
        check_for_overflows: bool,
        transcript: &mut T,
    ) -> Self {
        // Taken from original Zip codes

        let num_vars = poly_size.ilog2();
        let two_pow_num_vars = 1_usize
            .checked_shl(num_vars)
            .expect("2 ** num_vars overflows");
        let row_len: usize = two_pow_num_vars
            .isqrt()
            .checked_next_power_of_two()
            .expect("row_len overflow");
        let repetition_factor = spec.repetition_factor();

        let num_column_opening = spec.num_column_opening();
        let n_0 = sub!(two_pow_num_vars, 1).min(20);
        let num_proximity_testing = spec.num_proximity_testing(row_len, n_0);

        // Width of each entry in codeword vector, in bits.
        // For RAA it's initial_bits + 2*log(repetition_factor) + num_variables
        let codeword_width_bits = {
            // let initial_bits = crypto_bigint::Int::<N>::BITS;

            let initial_bits = u32::try_from(size_of::<Eval>())
                .ok()
                .and_then(|b| b.checked_mul(8))
                .expect("Size of Eval type is too large");

            let rep_factor_log = repetition_factor
                .checked_next_power_of_two()
                .expect("Repetition factor is too large")
                .ilog2();
            let num_vars_even = if num_vars.is_multiple_of(2) {
                num_vars
            } else {
                add!(num_vars, 1)
            };
            add!(initial_bits, add!(num_vars_even, mul!(rep_factor_log, 2)))
        };
        let codeword_type_bits = u32::try_from(size_of::<Cw>())
            .ok()
            .and_then(|b| b.checked_mul(8))
            .expect("Size of Cw type is too large");
        assert!(
            codeword_type_bits >= codeword_width_bits,
            "Cannot fit {codeword_width_bits}-bit wide codeword entries in {} bits entries",
            codeword_type_bits
        );

        let perm_1_seed = transcript.get_challenge();
        let perm_2_seed = transcript.get_challenge();

        Self {
            check_for_overflows,
            row_len,
            repetition_factor,
            num_column_opening,
            num_proximity_testing,
            perm_1_seed,
            perm_2_seed,
            phantom: PhantomData,
        }
    }

    /// Do the actual encoding, as per RAA spec
    fn encode_inner<In, Out>(&self, row: &[In]) -> Vec<Out>
    where
        Out: Zero + CheckedAdd + for<'a> AddAssign<&'a Out> + FromRef<In> + Clone,
    {
        debug_assert_eq!(
            row.len(),
            self.row_len,
            "Row length must match the code's row length"
        );
        let mut result: Vec<Out> = repeat(row, self.repetition_factor);
        shuffle_seeded(&mut result, self.perm_1_seed);
        if self.check_for_overflows {
            accumulate(&mut result);
        } else {
            accumulate_unchecked(&mut result);
        }
        shuffle_seeded(&mut result, self.perm_2_seed);
        if self.check_for_overflows {
            accumulate(&mut result);
        } else {
            accumulate_unchecked(&mut result);
        }
        debug_assert_eq!(result.len(), self.codeword_len());
        result
    }
}

impl<Eval, Cw, Comb> LinearCode<Eval, Cw, Comb> for RaaCode<Eval, Cw, Comb>
where
    Eval: Ring,
    Cw: Ring + FromRef<Eval>,
    Comb: Ring + FromRef<Comb>,
{
    type Inner = u64; // Doesn't really matter, we don't use it

    fn row_len(&self) -> usize {
        self.row_len
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn codeword_len(&self) -> usize {
        self.row_len * self.repetition_factor
    }

    fn num_column_opening(&self) -> usize {
        self.num_column_opening
    }

    fn num_proximity_testing(&self) -> usize {
        self.num_proximity_testing
    }

    fn encode(&self, row: &[Eval]) -> Vec<Cw> {
        self.encode_inner(row)
    }

    fn encode_wide(&self, row: &[Comb]) -> Vec<Comb> {
        self.encode_inner(row)
    }

    fn encode_f<F>(&self, row: &[F]) -> Vec<F>
    where
        F: PrimeField + FromRef<F> + FromRef<Self::Inner>,
    {
        self.encode_inner(row)
    }
}

/// Repeat the given slice N times, e.g `[1,2,3] => [1,2,3,1,2,3]`
#[allow(clippy::arithmetic_side_effects)]
fn repeat<In, Out: FromRef<In> + Clone>(input: &[In], repetition_factor: usize) -> Vec<Out> {
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
fn accumulate<I>(input: &mut [I])
where
    I: Zero + CheckedAdd + Clone,
{
    for i in 1..input.len() {
        input[i] = add!(input[i], &input[i - 1], "Accumulation overflow");
    }
}

#[allow(clippy::arithmetic_side_effects)]
fn accumulate_unchecked<I>(input: &mut [I])
where
    I: Zero + for<'a> AddAssign<&'a I> + Clone,
{
    for i in 1..input.len() {
        // This allows us to circumvent Rust bounds checking
        unsafe {
            let input_i: *mut I = input.get_unchecked_mut(i);
            let input_i: &mut I = &mut *input_i;
            // Note:
            // For Int ring, AddAssign here still results in a panic, but that's more
            // efficient than doing a checked_add and converting ConstCtOption
            // into Option and panicking later.
            *input_i += input.get_unchecked(i - 1);
        };
    }
}

#[cfg(test)]
mod tests {
    use crypto_primitives::crypto_bigint_int::Int;
    use num_traits::Zero;

    use super::*;
    use crate::{
        code::{DefaultLinearCodeSpec, LinearCode},
        pcs::test_utils::MockTranscript,
        utils::shuffle_seeded,
    };

    // Define common types for testing
    const INT_LIMBS: usize = 1;

    const N: usize = INT_LIMBS;
    const K: usize = INT_LIMBS * 4;
    const M: usize = INT_LIMBS * 8;

    fn test_raa<Eval, Cw, Comb, F>(poly_size: usize, f: F)
    where
        Eval: Ring,
        Cw: Ring + FromRef<Eval>,
        Comb: Ring + FromRef<Comb>,
        F: Fn(&RaaCode<Eval, Cw, Comb>),
    {
        for check_for_overflows in [true, false] {
            let mut transcript = MockTranscript::default();
            let code = RaaCode::<Eval, Cw, Comb>::new(
                &DefaultLinearCodeSpec,
                poly_size,
                check_for_overflows,
                &mut transcript,
            );
            f(&code)
        }
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
    fn encoding_preserves_linearity() {
        test_raa::<Int<N>, Int<K>, Int<M>, _>(16, |code| {
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
        })
    }

    /// Since our shuffle seeds are fixed, we can test the encoding
    /// against a known output.
    #[test]
    fn encoding_produces_predictable_results() {
        test_raa::<Int<N>, Int<K>, Int<M>, _>(16, |code| {
            let a: Vec<Int<N>> = (1..=4).map(Int::<N>::from).collect();

            let encode_a: Vec<Int<K>> = code.encode(&a);

            assert_eq!(
                encode_a,
                [0x11, 0x1C, 0x21, 0x28, 0x3C, 0x48, 0x49, 0x58].map(Int::<K>::from)
            );
        })
    }

    #[test]
    fn encoding_zero_vector_results_in_zero_codeword() {
        test_raa::<Int<N>, Int<K>, Int<M>, _>(16, |code| {
            let zero_vector: Vec<_> = vec![Int::<N>::zero(); code.row_len()];
            let encoded_vector: Vec<Int<K>> = code.encode(&zero_vector);

            let expected_codeword: Vec<Int<K>> = vec![Int::zero(); code.codeword_len()];

            assert_eq!(
                encoded_vector, expected_codeword,
                "Encoding a zero vector should result in a zero codeword"
            );
        })
    }

    #[test]
    #[should_panic]
    fn constructor_panics_on_insufficient_codeword_width() {
        const N: usize = 1;
        const K: usize = 1;

        let mut transcript = MockTranscript::default();
        let _code = RaaCode::<Int<N>, Int<K>, Int<M>>::new(
            &DefaultLinearCodeSpec,
            1 << 30,
            true,
            &mut transcript,
        );
    }

    #[test]
    #[should_panic(expected = "Row length must match the code's row length")]
    #[cfg(debug_assertions)]
    fn encode_panics_on_mismatched_row_length() {
        test_raa::<Int<N>, Int<K>, Int<M>, _>(16, |code| {
            let incorrect_row = vec![Int::<N>::from(1), Int::<N>::from(2), Int::<N>::from(3)];
            let _: Vec<Int<K>> = code.encode(&incorrect_row);
        })
    }
}
