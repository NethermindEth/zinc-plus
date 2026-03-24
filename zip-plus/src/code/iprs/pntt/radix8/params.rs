use ark_ff::{FftField, FpConfig};
use std::{fmt::Debug, marker::PhantomData};

/// The integer types of twiddles.
pub type PnttInt = i64;

/// Configuration of radix-8 pseudo NTT.
pub trait Config: Debug + Copy + Send + Sync {
    /// The field used to generate the twiddle factors
    /// and the base matrix for this pseudo NTT.
    type Field: FftField;
    const FIELD_MODULUS: u32;

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
    const BASE_TWIDDLES: [PnttInt; 8];

    /// The length of the pseudo NTT's input.
    const INPUT_LEN: usize = Self::BASE_LEN * (1 << (3 * Self::DEPTH));
    /// The length of the pseudo NTT's output.
    const OUTPUT_LEN: usize = {
        let value = Self::BASE_DIM * (1 << (3 * Self::DEPTH));
        assert!(
            value < Self::FIELD_MODULUS as usize,
            "Output length is more than the number of elements in the field"
        );
        value
    };

    /// A helper to get an integer representation that
    /// lies in the range `[-(p - 1)/2, (p - 1)/2]` from a field element.
    // TODO(alex): If we need it somewhere else, it would make sense to make this a
    // property of the field.
    fn field_to_int_normalized(x: Self::Field) -> PnttInt;
}

mod precompute {
    use super::{Config, PnttInt};
    use ark_ff::Field;
    use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
    use itertools::Itertools;
    use std::array;
    use zinc_utils::mul;

    #[allow(clippy::arithmetic_side_effects)]
    pub(super) fn precompute_butterfly_twiddles<C: Config>() -> Vec<Vec<[[PnttInt; 8]; 7]>> {
        let roots_of_unity = precompute_roots_of_unity::<C>(C::OUTPUT_LEN);

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

    pub(super) fn precompute_base_matrix<C: Config>() -> Vec<Vec<PnttInt>> {
        let domain = Radix2EvaluationDomain::<C::Field>::new(C::BASE_DIM)
            .expect("Failed to create NTT domain");

        domain
            .elements()
            .map(|root| {
                (0..C::BASE_LEN)
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
        Self {
            base_matrix: precompute::precompute_base_matrix::<C>(),
            butterfly_twiddles: precompute::precompute_butterfly_twiddles::<C>(),
            _phantom: PhantomData,
        }
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

macro_rules! define_configs {
    ($($(#[$outer:meta])* $name:ident(BASE_LEN=$base_len:literal, BASE_DIM=$base_dim:literal)),* $(,)?) => {
        $(
            $(#[$outer])*
            #[derive(Debug, Clone, Copy)]
            pub struct $name<const DEPTH: usize>;

            impl<const DEPTH: usize> Config for $name<DEPTH> {
                type Field = f65537::Field;
                const FIELD_MODULUS: u32 = f65537::MODULUS;
                const BASE_LEN: usize = $base_len;
                const BASE_DIM: usize = $base_dim;
                const DEPTH: usize = DEPTH;
                const BASE_TWIDDLES: [PnttInt; 8] = [1, 4096, -256, 16, -1, -4096, 256, -16];

                fn field_to_int_normalized(x: Self::Field) -> PnttInt {
                    let big_int = f65537::Backend::into_bigint(x);

                    precompute::normalize_field_element(big_int.0[0], Self::FIELD_MODULUS)
                }
            }
        )*
    };
}

define_configs! {
    //
    // Rate 1/2 configurations
    //

    /// Pseudo NTT configuration for F65537 (2^16 + 1) with BASE_LEN=32, BASE_DIM=64 (rate 1/2).
    ///
    /// Supports `DEPTH` up to `3`.
    PnttConfigF65537_32_64(BASE_LEN=32, BASE_DIM=64),

    //
    // Rate 1/4 configurations
    //

    /// Pseudo NTT configuration for F65537 (2^16 + 1) with BASE_LEN=1, BASE_DIM=4 (rate 1/4).
    /// NTT domain up to 2^16. Row lengths: 8 (D=1), 64 (D=2), 512 (D=3)..
    PnttConfigF65537_1_4(BASE_LEN=1, BASE_DIM=4),

    /// Pseudo NTT configuration for F65537 (2^16 + 1) with BASE_LEN=2, BASE_DIM=8 (rate 1/4).
    /// NTT domain up to 2^16. Row lengths: 16 (D=1), 128 (D=2), 1024 (D=3).
    PnttConfigF65537_2_8(BASE_LEN=2, BASE_DIM=8),

    /// Pseudo NTT configuration for F65537 (2^16 + 1) with BASE_LEN=4, BASE_DIM=16 (rate 1/4).
    /// NTT domain up to 2^16. Row lengths: 32 (D=1), 256 (D=2), 2048 (D=3).
    PnttConfigF65537_4_16(BASE_LEN=4, BASE_DIM=16),

    /// Pseudo NTT configuration for F65537 (2^16 + 1) with BASE_LEN=16, BASE_DIM=64 (rate 1/4).
    /// NTT domain up to 2^16, enabling row lengths up to 2^13.
    /// Row lengths: 128 (D=1), 1024 (D=2), 8192 (D=3).
    PnttConfigF65537_16_64(BASE_LEN=16, BASE_DIM=64),

    /// Pseudo NTT configuration for F65537 (2^16 + 1) with BASE_LEN=32, BASE_DIM=128 (rate 1/4).
    /// NTT domain up to 2^16, enabling row lengths up to 2^11.
    /// Row lengths: 256 (D=1), 2048 (D=2).
    PnttConfigF65537_32_128(BASE_LEN=32, BASE_DIM=128),

    /// Pseudo NTT configuration for F65537 (2^16 + 1) with BASE_LEN=64, BASE_DIM=256 (rate 1/4).
    /// NTT domain up to 2^16, enabling row lengths up to 2^12.
    /// Row lengths: 512 (D=1), 4096 (D=2).
    PnttConfigF65537_64_256(BASE_LEN=64, BASE_DIM=256),
}

#[cfg(test)]
mod tests {
    use super::*;

    // Twiddles are indeed the 8th roots of unity.
    fn check_twiddles_generic<C: Config>() {
        let expected = precompute::precompute_roots_of_unity::<C>(8);

        let our = C::BASE_TWIDDLES.to_vec();
        let ol = C::OUTPUT_LEN;
        let field_modulus = C::FIELD_MODULUS as usize;
        assert!(ol < field_modulus);

        assert_eq!(expected, our);
    }

    #[test]
    fn check_twiddles_f65537() {
        check_twiddles_generic::<PnttConfigF65537_32_64<1>>();
        check_twiddles_generic::<PnttConfigF65537_1_4<1>>();
        check_twiddles_generic::<PnttConfigF65537_2_8<1>>();
        check_twiddles_generic::<PnttConfigF65537_4_16<1>>();
        check_twiddles_generic::<PnttConfigF65537_16_64<1>>();
        check_twiddles_generic::<PnttConfigF65537_32_128<1>>();
        check_twiddles_generic::<PnttConfigF65537_64_256<1>>();
        check_twiddles_generic::<PnttConfigF65537_64_256<2>>();
    }
}
