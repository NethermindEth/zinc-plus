use crate::ZipError;
use ark_ff::{FftField, FpConfig};
use num_traits::Euclid;
use std::{fmt::Debug, marker::PhantomData};
use zinc_utils::{mul, sub};

/// The integer types of twiddles.
pub type PnttInt = i64;

/// Configuration of radix-8 pseudo NTT.
pub trait Config: Debug + Copy + Send + Sync {
    /// The field used to generate the twiddle factors
    /// and the base matrix for this pseudo NTT.
    type Field: FftField;
    const FIELD_MODULUS: u32;

    /// The coefficients used to combine subresults.
    /// They are the 8-th roots of unity from the field `Self::Field`
    /// lifted to `Self::Int`.
    const BASE_TWIDDLES: [PnttInt; 8];

    /// A helper to get an integer representation that
    /// lies in the range `[-(p - 1)/2, (p - 1)/2]` from a field element.
    // TODO(alex): If we need it somewhere else, it would make sense to make this a
    // property of the field.
    fn field_to_int_normalized(x: Self::Field) -> PnttInt;
}

// Precomputed parameters needed for
// pseudo NTT algorithm.
#[derive(Clone, Debug)]
pub struct Radix8PnttParams<C: Config> {
    /// The length of the pseudo NTT's input.
    pub row_len: usize,
    /// The length of the pseudo NTT's output.
    pub codeword_len: usize,
    /// The number of steps where NTT is performed recursively.
    pub depth: usize,
    /// The number of columns of the base matrix.
    pub base_len: usize,
    /// The number of rows of the base matrix, always a power of 2.
    pub base_dim: usize,
    /// log2 of the number of rows of the base matrix.
    pub base_dim_log2: u32,
    /// The mask to compute `i % base_dim`.
    pub base_dim_mask: usize,
    /// The base matrix of the pseudo NTT.
    pub base_matrix: Vec<Vec<PnttInt>>, // TODO(Alex): Maybe use DenseRowMatrix for this?
    /// Precomputed twiddles for every stage that already contain the relevant
    /// root-of-unity factor. This lets the butterfly apply a single
    /// multiplication per term instead of two.
    pub butterfly_twiddles: Vec<Vec<[[PnttInt; 8]; 7]>>,

    _phantom: PhantomData<C>,
}

impl<C: Config> Radix8PnttParams<C> {
    /// Precompute pseudo NTT parameters.
    pub fn new(row_len: usize, depth: usize, rep_factor: usize) -> Result<Self, ZipError> {
        let codeword_len = mul!(row_len, rep_factor);
        if codeword_len >= C::FIELD_MODULUS as usize {
            return Err(ZipError::InvalidPcsParam(
                "Codeword length is more than the number of elements in the field".to_owned(),
            ));
        }

        let coeff = 1_usize << mul!(3, depth);
        let (base_len, base_len_rem) = row_len.div_rem_euclid(&coeff);
        if base_len_rem != 0 {
            return Err(ZipError::InvalidPcsParam(format!(
                "Row length {row_len} must be a multiple of {coeff}"
            )));
        }

        let base_dim = mul!(base_len, rep_factor);
        if !base_dim.is_power_of_two() {
            return Err(ZipError::InvalidPcsParam(format!(
                "Base dimension {base_dim} must be a power of 2"
            )));
        }

        let base_dim_log2: u32 = base_dim.trailing_zeros();

        let base_dim_mask: usize = sub!(base_dim, 1);

        Ok(Self {
            row_len,
            codeword_len,
            depth,
            base_len,
            base_dim,
            base_dim_log2,
            base_dim_mask,
            base_matrix: precompute::precompute_base_matrix::<C>(base_dim, base_len),
            butterfly_twiddles: precompute::precompute_butterfly_twiddles::<C>(
                base_dim,
                codeword_len,
                depth,
            ),
            _phantom: PhantomData,
        })
    }
}

mod precompute {
    use super::{Config, PnttInt};
    use ark_ff::Field;
    use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
    use itertools::Itertools;
    use std::array;
    use zinc_utils::mul;

    #[allow(clippy::arithmetic_side_effects)]
    pub(super) fn precompute_butterfly_twiddles<C: Config>(
        base_dim: usize,
        output_len: usize,
        depth: usize,
    ) -> Vec<Vec<[[PnttInt; 8]; 7]>> {
        let roots_of_unity = precompute_roots_of_unity::<C>(output_len);

        (0..depth)
            .map(|k| {
                let sub_chunk_length = base_dim * (1 << (3 * k));
                let curr_prim_root_power = 1 << (3 * (depth - 1 - k));

                (0..sub_chunk_length)
                    .map(|i| {
                        array::from_fn(|j_minus_1| {
                            let root = roots_of_unity[curr_prim_root_power * i * (j_minus_1 + 1)];

                            array::from_fn(|twiddle_idx| {
                                mul_and_normalize_twiddle(
                                    C::BASE_TWIDDLES[twiddle_idx],
                                    root,
                                    C::FIELD_MODULUS,
                                )
                            })
                        })
                    })
                    .collect()
            })
            .collect()
    }

    pub(super) fn precompute_base_matrix<C: Config>(
        base_dim: usize,
        base_len: usize,
    ) -> Vec<Vec<PnttInt>> {
        let domain =
            Radix2EvaluationDomain::<C::Field>::new(base_dim).expect("Failed to create NTT domain");

        domain
            .elements()
            .map(|root| {
                (0..base_len)
                    .map(move |i| C::field_to_int_normalized(root.pow([i as u64])))
                    .collect_vec()
            })
            .collect()
    }

    #[allow(clippy::arithmetic_side_effects)]
    pub(super) fn precompute_roots_of_unity<C: Config>(n: usize) -> Vec<PnttInt> {
        let domain =
            Radix2EvaluationDomain::<C::Field>::new(n).expect("Failed to create NTT domain");

        domain
            .elements()
            .map(C::field_to_int_normalized)
            .collect_vec()
    }

    /// Field normalization for at most 32-bit fields.
    /// Might have unpleasant overflows if used for bigger fields.
    #[allow(clippy::arithmetic_side_effects, clippy::cast_possible_wrap)]
    pub(super) fn normalize_field_element(x: u64, p: u32) -> i64 {
        debug_assert!(x <= i64::MAX as u64);
        let x = x as i64;
        let p = i64::from(p);
        if x >= (p - 1) / 2 { x - p } else { x }
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn mul_and_normalize_twiddle(twiddle: PnttInt, root: PnttInt, modulus: u32) -> PnttInt {
        let twiddle_mod = to_positive_mod_repr(twiddle, modulus);
        let root_mod = to_positive_mod_repr(root, modulus);
        let product = mul!(twiddle_mod, root_mod) % u64::from(modulus);

        normalize_field_element(product, modulus)
    }

    #[allow(clippy::cast_sign_loss)]
    fn to_positive_mod_repr(value: PnttInt, modulus: u32) -> u64 {
        value.rem_euclid(i64::from(modulus)) as u64
    }
}

mod f65537 {
    #![allow(non_local_definitions)]
    use ark_ff::{Fp64, MontBackend, MontConfig};
    #[derive(MontConfig)]
    #[modulus = "65537"]
    #[generator = "3"]
    pub struct Config;

    pub type Backend = MontBackend<Config, 1>;
    pub type Field = Fp64<Backend>;

    #[allow(clippy::cast_possible_truncation)] // We know modulus is small enough.
    pub const MODULUS: u32 = Config::MODULUS.0[0] as u32;
}

/// Pseudo NTT configuration for F65537 (2^16 + 1).
#[derive(Debug, Clone, Copy)]
pub struct PnttConfigF65537;

impl Config for PnttConfigF65537 {
    type Field = f65537::Field;
    const FIELD_MODULUS: u32 = f65537::MODULUS;
    const BASE_TWIDDLES: [PnttInt; 8] = [1, 4096, -256, 16, -1, -4096, 256, -16];

    fn field_to_int_normalized(x: Self::Field) -> PnttInt {
        let big_int = f65537::Backend::into_bigint(x);
        precompute::normalize_field_element(big_int.0[0], Self::FIELD_MODULUS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Twiddles are indeed the 8th roots of unity.
    fn check_twiddles_generic<C: Config>() {
        let expected = precompute::precompute_roots_of_unity::<C>(8);
        let our = C::BASE_TWIDDLES.to_vec();
        assert_eq!(expected, our);
    }

    #[test]
    fn check_twiddles() {
        check_twiddles_generic::<PnttConfigF65537>();
    }
}
