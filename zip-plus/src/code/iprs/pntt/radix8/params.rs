use ark_ff::{FftField, Field, FpConfig, PrimeField};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use itertools::Itertools;
use std::{array, marker::PhantomData};

/// The integer types of twiddles.
pub type PnttInt = i64;

/// Configuration of radix-8 pseudo NTT.
pub trait Config: Copy + Send + Sync {
    /// The field used to generate the twiddle factors
    /// and the base matrix for this pseudo NTT.
    type Field: FftField;

    /// The number of steps where NTT is performed
    /// recursivily.
    const DEPTH: usize;
    /// The number of columns of the base matrix.
    const BASE_LEN: usize;
    /// The number of rows of the base matrix, always a power of 2.
    const BASE_DIM: usize;
    /// log2 of the number of rows of the base matrix.
    const BASE_DIM_LOG2: u32 = Self::BASE_DIM.trailing_zeros();
    /// The mask to compute `i % Self::BASE_DIM`.
    const BASE_DIM_MASK: usize = Self::BASE_DIM - 1;
    /// The coefficients used to combine subresults.
    /// They are the 8-th roots of unity from the field `Self::Field`
    /// lifted to `Self::Int`.
    const TWIDDLES: [PnttInt; 8];

    /// The length of the pseudo NTT's input.
    const INPUT_LEN: usize = Self::BASE_LEN * (1 << (3 * Self::DEPTH));
    /// The length of the pseudo NTT's output.
    const OUTPUT_LEN: usize = Self::BASE_DIM * (1 << (3 * Self::DEPTH));

    /// A helper to get an integer representation that
    /// lies in the range `[-(p - 1)/2, (p - 1)/2]` from a field element.
    fn field_to_int_normalized(x: Self::Field) -> PnttInt;
}

// Precomputed parameters needed for
// pseudo NTT algorithm.
#[derive(Clone, Debug)]
pub struct Radix8PnttParams<C: Config> {
    /// The base matrix of the pseudo NTT.
    pub base_matrix: Vec<Vec<PnttInt>>, // TODO(Alex): Maybe use DenseRowMatrix for this?
    /// Precomputed twiddles for every stage that already contain the relevant
    /// root-of-unity factor. This lets the butterfly apply a single
    /// multiplication per term instead of two.
    pub butterfly_twiddles: Vec<Vec<[[PnttInt; 8]; 7]>>,
    _phantom: PhantomData<C>,
}

impl<C: Config> Default for Radix8PnttParams<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: Config> Radix8PnttParams<C> {
    /// Precompute pseudo NTT parameters.
    pub fn new() -> Self {
        let roots_of_unity = Self::precompute_roots_of_unity(C::OUTPUT_LEN);

        Self {
            base_matrix: Self::precompute_base_matrix(),
            butterfly_twiddles: Self::precompute_butterfly_twiddles(&roots_of_unity),
            _phantom: PhantomData,
        }
    }

    fn precompute_base_matrix() -> Vec<Vec<PnttInt>> {
        let domain = Radix2EvaluationDomain::<C::Field>::new(C::BASE_DIM)
            .expect("Failed to create NTT domain");

        let mut matrix = Vec::with_capacity(C::BASE_DIM);

        for root in domain.elements() {
            matrix.push(
                (0..C::BASE_LEN)
                    .map(move |i| C::field_to_int_normalized(root.pow([i as u64])))
                    .collect_vec(),
            )
        }

        matrix
    }

    fn precompute_roots_of_unity(n: usize) -> Vec<PnttInt> {
        let domain =
            Radix2EvaluationDomain::<C::Field>::new(n).expect("Failed to create NTT domain");

        domain
            .elements()
            .map(C::field_to_int_normalized)
            .collect_vec()
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn precompute_butterfly_twiddles(roots_of_unity: &[PnttInt]) -> Vec<Vec<[[PnttInt; 8]; 7]>> {
        let modulus = <C::Field as Field>::BasePrimeField::MODULUS.as_ref()[0];
        let modulus_i64: i64 = modulus
            .try_into()
            .expect("Field modulus should fit into i64 for pseudo NTT parameters");
        let modulus_u128 = u128::from(modulus);

        (0..C::DEPTH)
            .map(|k| {
                let sub_chunk_length = C::BASE_DIM * (1 << (3 * k));
                let curr_prim_root_power = 1 << (3 * (C::DEPTH - 1 - k));

                (0..sub_chunk_length)
                    .map(|i| {
                        array::from_fn(|j_minus_1| {
                            let root = roots_of_unity[curr_prim_root_power * i * (j_minus_1 + 1)];

                            array::from_fn(|twiddle_idx| {
                                mul_and_normalize_twiddle(
                                    C::TWIDDLES[twiddle_idx],
                                    root,
                                    modulus_i64,
                                    modulus,
                                    modulus_u128,
                                )
                            })
                        })
                    })
                    .collect()
            })
            .collect()
    }
}

/// Pseudo NTT configuration derived from
/// the field `Fp` for `p = 2^16 + 1`.
///
/// Supports `DEPTH` up to `3`.
#[derive(Clone, Copy)]
pub struct PnttConfigF2_16_1<const DEPTH: usize>;

mod fq {
    #![allow(non_local_definitions)]
    use ark_ff::{Fp64, MontBackend, MontConfig};
    #[derive(MontConfig)]
    #[modulus = "65537"]
    #[generator = "3"]
    pub struct FqConfig;

    pub type FqBackend = MontBackend<FqConfig, 1>;
    pub type Fq = Fp64<FqBackend>;

    pub const MODULUS: u64 = FqConfig::MODULUS.0[0];
}

impl<const DEPTH: usize> Config for PnttConfigF2_16_1<DEPTH> {
    type Field = fq::Fq;
    const BASE_LEN: usize = 32;
    const BASE_DIM: usize = 64;
    const DEPTH: usize = DEPTH;
    const TWIDDLES: [PnttInt; 8] = [1, 4096, -256, 16, -1, -4096, 256, -16];

    fn field_to_int_normalized(x: Self::Field) -> PnttInt {
        let big_int = fq::FqBackend::into_bigint(x);

        normalize_field_element(big_int.0[0], fq::MODULUS)
    }
}

/// Field normalization for at most 32-bit fields.
/// Might have unpleasant overflows if used for bigger fields.
#[allow(clippy::arithmetic_side_effects)]
#[allow(clippy::cast_possible_wrap)]
fn normalize_field_element(x: u64, p: u64) -> i64 {
    if x >= (p - 1) / 2 {
        x as i64 - p as i64
    } else {
        x as i64
    }
}

#[allow(clippy::arithmetic_side_effects)]
fn mul_and_normalize_twiddle(
    twiddle: PnttInt,
    root: PnttInt,
    modulus_i64: i64,
    modulus_u64: u64,
    modulus_u128: u128,
) -> PnttInt {
    let twiddle_mod = to_positive_mod_repr(twiddle, modulus_i64);
    let root_mod = to_positive_mod_repr(root, modulus_i64);
    let product = (twiddle_mod * root_mod) % modulus_u128;
    let product_u64: u64 = product
        .try_into()
        .expect("Product reduced modulo prime field fits into u64");

    normalize_field_element(product_u64, modulus_u64)
}

#[allow(clippy::arithmetic_side_effects)]
fn to_positive_mod_repr(value: PnttInt, modulus: i64) -> u128 {
    let mut repr = value % modulus;
    if repr < 0 {
        repr += modulus;
    }
    repr.try_into()
        .expect("Representation is guaranteed to be non-negative and fit into u128")
}

#[cfg(test)]
mod tests {
    use super::*;

    // Twiddles are indeed the 8th roots of unity.
    fn check_twiddles_generic<C: Config>() {
        let expected = Radix8PnttParams::<C>::precompute_roots_of_unity(8);

        let our = C::TWIDDLES.to_vec();

        assert_eq!(expected, our);
    }

    #[test]
    fn check_twiddles() {
        check_twiddles_generic::<PnttConfigF2_16_1<1>>();
    }
}
