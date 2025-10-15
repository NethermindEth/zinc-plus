use crate::{
    pcs::structs::MulByScalar,
    traits::{ConstTranscribable, Named, Transcribable},
};
use crypto_bigint::{BitOps, BoxedUint, Word};
use crypto_primitives::{crypto_bigint_int::Int, crypto_bigint_uint::Uint};
use itertools::Itertools;
use num_traits::CheckedAdd;
use rand::{rngs::StdRng, seq::SliceRandom};
use rand_core::SeedableRng;
use std::iter::Iterator;

use crypto_primitives::boolean::Boolean;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(target_pointer_width = "64")]
const WORD_FACTOR: usize = 1;
#[cfg(target_pointer_width = "32")]
const WORD_FACTOR: usize = 2;

#[macro_export]
macro_rules! neg {
    ($a:expr) => {
        neg!($a, "Negation overflow")
    };
    ($a:expr, $msg:expr) => {
        $a.checked_neg().expect($msg)
    };
}

#[macro_export]
macro_rules! add {
    ($a:expr, $b:expr) => {
        add!($a, $b, "Addition overflow")
    };
    ($a:expr, $b:expr, $msg:expr) => {
        $a.checked_add($b).expect($msg)
    };
}

#[macro_export]
macro_rules! sub {
    ($a:expr, $b:expr) => {
        sub!($a, $b, "Subtraction overflow")
    };
    ($a:expr, $b:expr, $msg:expr) => {
        $a.checked_sub($b).expect($msg)
    };
}

#[macro_export]
macro_rules! mul {
    ($a:expr, $b:expr) => {
        mul!($a, $b, "Multiplication overflow")
    };
    ($a:expr, $b:expr, $msg:expr) => {
        $a.checked_mul($b).expect($msg)
    };
}

#[macro_export]
macro_rules! div {
    ($a:expr, $b:expr) => {
        div!($a, $b, "Division by zero")
    };
    ($a:expr, $b:expr, $msg:expr) => {
        $a.checked_div($b).expect($msg)
    };
}

#[macro_export]
macro_rules! rem {
    ($a:expr, $b:expr) => {
        rem!($a, $b, "Division by zero")
    };
    ($a:expr, $b:expr, $msg:expr) => {
        $a.checked_rem($b).expect($msg)
    };
}

#[macro_export]
macro_rules! ilog_round_up {
    ($a:expr, $tp: ty) => {{
        let res = if $a.is_power_of_two() {
            $a.ilog2()
        } else {
            add!($a.ilog2(), 1)
        };
        <$tp>::try_from(res).expect(concat!("ilog doesn't fit ", stringify!($tp)))
    }};
}

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

pub(crate) fn num_threads() -> usize {
    #[cfg(feature = "parallel")]
    return rayon::current_num_threads();

    #[cfg(not(feature = "parallel"))]
    return 1;
}

#[cfg(not(feature = "parallel"))]
pub(crate) fn parallelize_into_iter_map_collect<I, T, F, R, C>(iterable: I, f: F) -> C
where
    I: Send + IntoIterator<Item = T>,
    T: Send,
    R: Send,
    F: Fn(T) -> R + Send + Sync + Clone,
    C: FromIterator<R>,
{
    iterable.into_iter().map(f).collect()
}

#[cfg(feature = "parallel")]
pub(crate) fn parallelize_into_iter_map_collect<I, T, F, R, C>(iterable: I, f: F) -> C
where
    I: Send + IntoParallelIterator<Item = T>,
    T: Send,
    R: Send,
    F: Fn(T) -> R + Send + Sync + Clone,
    C: FromParallelIterator<R>,
{
    iterable.into_par_iter().map(f).collect()
}

pub(crate) fn parallelize_for_each<I, T, F>(iter: I, f: F)
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
        let num_threads = num_threads();
        let chunk_size = v.len().div_ceil(num_threads);
        if chunk_size < num_threads {
            f((v, 0));
        } else {
            parallelize_for_each(v.chunks_mut(chunk_size).zip((0..).step_by(chunk_size)), f);
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
pub(super) fn combine_rows<Coeff, El>(
    coeffs: &[Coeff],
    evaluations: &[El],
    row_len: usize,
    zero: El,
) -> Vec<El>
where
    Coeff: Send + Sync,
    El: Clone + CheckedAdd + for<'z> MulByScalar<&'z Coeff> + Send + Sync,
{
    let mut combined_row = vec![zero; row_len];
    parallelize(&mut combined_row, |(combined_row, offset)| {
        combined_row
            .iter_mut()
            .zip(offset..)
            .for_each(|(combined, column)| {
                coeffs
                    .iter()
                    .zip(evaluations.iter().skip(column).step_by(row_len))
                    .for_each(|(coeff, eval)| {
                        *combined = add!(
                            combined,
                            &eval
                                .mul_by_scalar(coeff)
                                .expect("Cannot multiply evaluation by coefficient")
                        );
                    });
            })
    });

    combined_row
}

/// Reorder the elements in slice using the given randomness seed
pub(super) fn shuffle_seeded<T>(slice: &mut [T], seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    slice.shuffle(&mut rng);
}

//
// Named implementations
//

macro_rules! impl_named_for_primitives {
    ($($type:ty),+) => {
        $(
            impl Named for $type {
                fn type_name() -> String {
                    stringify!($type).to_string()
                }
            }
        )+
    };
}

impl_named_for_primitives!(i8, i16, i32, i64, i128);
impl_named_for_primitives!(u8, u16, u32, u64, u128);

impl<const LIMBS: usize> Named for Int<LIMBS> {
    fn type_name() -> String {
        format!("Int<{}>", LIMBS)
    }
}

impl<const LIMBS: usize> Named for Uint<LIMBS> {
    fn type_name() -> String {
        format!("Uint<{}>", LIMBS)
    }
}

impl Named for Boolean {
    fn type_name() -> String {
        "b".to_owned()
    }
}

//
// Transcribable implementations
//

macro_rules! impl_transcribable_for_primitives {
    ($($type:ty),+) => {
        $(
            impl ConstTranscribable for $type {
                const NUM_BYTES: usize = std::mem::size_of::<$type>();

                fn read_transcription_bytes(bytes: &[u8]) -> Self {
                    Self::from_le_bytes(bytes.try_into().expect("Invalid byte slice length"))
                }

                fn write_transcription_bytes(&self, buf: &mut [u8]) {
                    assert_eq!(buf.len(), std::mem::size_of::<$type>());
                    buf.copy_from_slice(&self.to_le_bytes());
                }
            }
        )+
    };
}

impl_transcribable_for_primitives!(u8, u16, u32, u64, u128);
impl_transcribable_for_primitives!(i8, i16, i32, i64, i128);

impl ConstTranscribable for Boolean {
    const NUM_BYTES: usize = 1;
    const NUM_BITS: usize = 1;

    fn read_transcription_bytes(bytes: &[u8]) -> Self {
        (bytes[0] != 0).into()
    }

    fn write_transcription_bytes(&self, buf: &mut [u8]) {
        buf[0] = self.to_u8();
    }
}

impl<const LIMBS: usize> ConstTranscribable for Uint<LIMBS> {
    const NUM_BYTES: usize = 8 * LIMBS / WORD_FACTOR;

    fn read_transcription_bytes(bytes: &[u8]) -> Self {
        // crypto_bigint::Uint stores limbs in least-to-most significant order.
        // It matches little-endian order ef limbs encoding, so platform pointer width
        // does not matter.
        let (chunked, rem) = bytes.as_chunks::<{ 8 / WORD_FACTOR }>();
        assert!(rem.is_empty(), "Invalid byte slice length for Uint");
        let words = chunked
            .iter()
            .map(|chunk| Word::from_le_bytes(*chunk))
            .collect_array::<LIMBS>()
            .expect("Invalid length for Uint");
        Uint::<LIMBS>::from_words(words)
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn write_transcription_bytes(&self, buf: &mut [u8]) {
        // crypto_bigint::Uint stores limbs in least-to-most significant order.
        // It matches little-endian order ef limbs encoding, so platform pointer width
        // does not matter.
        assert_eq!(buf.len(), Self::NUM_BYTES, "Buffer size mismatch for Uint");
        const W_SIZE: usize = size_of::<Word>();
        for (i, w) in self.as_words().iter().enumerate() {
            // Performance: reuse buffer and help compiler optimize away materializing
            // vector
            buf[(i * W_SIZE)..(i * W_SIZE + W_SIZE)].copy_from_slice(w.to_le_bytes().as_ref());
        }
    }
}

impl<const LIMBS: usize> ConstTranscribable for Int<LIMBS> {
    const NUM_BYTES: usize = Uint::<LIMBS>::NUM_BYTES;

    fn read_transcription_bytes(bytes: &[u8]) -> Self {
        *<Uint<LIMBS> as ConstTranscribable>::read_transcription_bytes(bytes).as_int()
    }

    fn write_transcription_bytes(&self, buf: &mut [u8]) {
        ConstTranscribable::write_transcription_bytes(self.as_uint(), buf)
    }
}

impl Transcribable for BoxedUint {
    /// Up to 255 bytes - so up to 2040 bits - should be plenty.
    const LENGTH_NUM_BYTES: usize = 1;

    fn read_num_bytes(bytes: &[u8]) -> usize {
        assert_eq!(bytes.len(), Self::LENGTH_NUM_BYTES);
        usize::from(bytes[0])
    }

    fn read_transcription_bytes(bytes: &[u8]) -> Self {
        // crypto_bigint::BoxedUint stores limbs in least-to-most significant order.
        // It matches little-endian order ef limbs encoding, so platform pointer width
        // does not matter.
        let (chunked, rem) = bytes.as_chunks::<{ 8 / WORD_FACTOR }>();
        assert!(rem.is_empty(), "Invalid byte slice length for BoxedUint");
        let words = chunked
            .iter()
            .map(|chunk| Word::from_le_bytes(*chunk))
            .collect_vec();
        BoxedUint::from_words(words)
    }

    fn get_num_bytes(&self) -> usize {
        usize::from(u8::try_from(self.bytes_precision()).expect("BoxedUint size must fit into u8"))
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn write_transcription_bytes(&self, buf: &mut [u8]) {
        // crypto_bigint::BoxedUint stores limbs in least-to-most significant order.
        // It matches little-endian order ef limbs encoding, so platform pointer width
        // does not matter.
        assert_eq!(
            buf.len(),
            self.bytes_precision(),
            "Buffer size mismatch for BoxedUint"
        );
        const W_SIZE: usize = size_of::<Word>();
        for (i, w) in self.as_words().iter().enumerate() {
            // Performance: reuse buffer and help compiler optimize away materializing
            // vector
            buf[(i * W_SIZE)..(i * W_SIZE + W_SIZE)].copy_from_slice(w.to_le_bytes().as_ref());
        }
    }
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

        let result = combine_rows(&coeffs, &evaluations, row_len, 0);

        assert_eq!(result, vec![(3 + 2 * 5), (4 + 2 * 6)]);
    }

    #[test]
    fn test_second_combination() {
        let coeffs = vec![3, 4];
        let evaluations = vec![2, 4, 6, 8];
        let row_len = 2;

        let result = combine_rows(&coeffs, &evaluations, row_len, 0);

        assert_eq!(result, vec![(3 * 2 + 4 * 6), (3 * 4 + 4 * 8)]);
    }
    #[test]
    fn test_large_values() {
        let coeffs = vec![1000, -500];
        let evaluations = vec![2000, -3000, 4000, -5000];
        let row_len = 2;

        let result = combine_rows(&coeffs, &evaluations, row_len, 0);

        assert_eq!(
            result,
            vec![
                (1000 * 2000 + (-500) * 4000),
                (1000 * -3000 + (-500) * -5000)
            ]
        );
    }
}
