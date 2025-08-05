use ark_std::{fmt::Debug, marker::PhantomData, ops::AddAssign, vec::Vec};
use num_traits::Zero;

use crate::{
    traits::{Field, FieldMap, Integer, Words, ZipTypes},
    code::{LinearCode, LinearCodeSpec},
    pcs::structs::ZipTranscript,
    utils::shuffle_seeded,
};

/// Implementation of a repeat-accumulate-accumulate (RAA) codes over the binary field,
/// as defined by the Blaze paper (https://eprint.iacr.org/2024/1609)
#[derive(Debug, Clone)]
pub struct RaaCode<ZT: ZipTypes> {
    row_len: usize,

    repetition_factor: usize,

    num_column_opening: usize,

    num_proximity_testing: usize,

    /// Randomness seed for the first permutation
    perm_1_seed: u64,

    /// Randomness seed for the second permutation
    perm_2_seed: u64,

    phantom: PhantomData<ZT>,
}

impl<ZT: ZipTypes> RaaCode<ZT> {
    pub fn new<S: LinearCodeSpec, T: ZipTranscript<ZT::L>>(
        spec: &S,
        poly_size: usize,
        transcript: &mut T,
    ) -> Self {
        // Taken from original Zip codes

        let num_vars = poly_size.ilog2() as usize;
        let row_len = ((1 << num_vars) as u64).isqrt().next_power_of_two() as usize;
        let repetition_factor = spec.repetition_factor();

        let num_column_opening = spec.num_column_opening();
        let log2_q = <ZT::N as Integer>::W::num_words();
        let n_0 = 20.min((1 << num_vars) - 1);
        let num_proximity_testing = spec.num_proximity_testing(log2_q, row_len, n_0);

        // Width of each entry in codeword vector, in bits.
        // For RAA it's initial_bits + 2*log(repetition_factor) + num_variables
        let codeword_width_bits = {
            let initial_bits = ZT::N::num_bits();
            let rep_factor_log: usize = repetition_factor
                .checked_next_power_of_two()
                .expect("Repetition factor is too large")
                .ilog2()
                .try_into()
                .expect("Repetition factor logarithm is too large");
            let num_vars_even = if num_vars % 2 == 0 {
                num_vars
            } else {
                num_vars + 1
            };
            initial_bits + num_vars_even + (2 * rep_factor_log)
        };
        assert!(
            ZT::K::num_bits() >= codeword_width_bits,
            "Cannot fit {codeword_width_bits}-bit wide codeword entries in {} bits integers",
            ZT::K::num_bits()
        );

        let perm_1_seed = transcript.get_u64();
        let perm_2_seed = transcript.get_u64();

        Self {
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
        Out: Zero + AddAssign<Out> + for<'a> From<&'a In> + Clone,
    {
        debug_assert_eq!(
            row.len(),
            self.row_len,
            "Row length must match the code's row length"
        );
        let mut result: Vec<Out> = repeat(row, self.repetition_factor);
        shuffle_seeded(&mut result, self.perm_1_seed);
        accumulate(&mut result);
        shuffle_seeded(&mut result, self.perm_2_seed);
        accumulate(&mut result);
        debug_assert_eq!(result.len(), self.codeword_len());
        result
    }
}

impl<ZT: ZipTypes> LinearCode<ZT> for RaaCode<ZT> {
    fn row_len(&self) -> usize {
        self.row_len
    }

    fn codeword_len(&self) -> usize {
        self.row_len * self.repetition_factor
    }

    fn num_column_opening(&self) -> usize {
        self.num_column_opening
    }

    fn num_proximity_testing(&self) -> usize {
        self.num_proximity_testing
    }

    fn encode_wide<In, Out>(&self, row: &[In]) -> Vec<Out>
    where
        In: Integer,
        Out: Integer + for<'a> From<&'a In> + for<'a> From<&'a ZT::L>,
    {
        self.encode_inner(row)
    }

    fn encode_f<F: Field>(&self, row: &[F], _field: F::R) -> Vec<F>
    where
        ZT::L: FieldMap<F, Output = F>,
    {
        self.encode_inner(row)
    }
}

/// Repeat the given slice N times, e.g `[1,2,3] => [1,2,3,1,2,3]`
fn repeat<In, Out: for<'a> From<&'a In> + Clone>(
    input: &[In],
    repetition_factor: usize,
) -> Vec<Out> {
    input
        .iter()
        .map(|i| Out::from(i))
        .cycle()
        .take(input.len() * repetition_factor)
        .collect()
}

/// Perform an operation equivalent to multiplying the slice in-place by the accumulation matrix
/// from the RAA code - a lower triangular matrix of the appropriate size, i.e. a matrix looking
/// like this:
///
/// ```text
/// 1 0 0 0
/// 1 1 0 0
/// 1 1 1 0
/// 1 1 1 1
/// ```
fn accumulate<I>(input: &mut [I])
where
    I: Zero + AddAssign<I> + Clone,
{
    for i in 1..input.len() {
        input[i] += input[i - 1].clone();
    }
}
