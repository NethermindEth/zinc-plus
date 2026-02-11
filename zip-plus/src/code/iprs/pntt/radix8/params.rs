use ark_ff::{FftField, FpConfig};
use std::marker::PhantomData;

/// The integer types of twiddles.
pub type PnttInt = i64;

/// Configuration of radix-8 pseudo NTT.
pub trait Config: Copy + Send + Sync {
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
    const OUTPUT_LEN: usize = Self::BASE_DIM * (1 << (3 * Self::DEPTH));

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

/// Pseudo NTT configuration derived from
/// the field `Fp` for `p = 2^16 + 1`.
///
/// Supports `DEPTH` up to `3`.
#[derive(Clone, Copy)]
pub struct PnttConfigF2_16_1<const DEPTH: usize>;

/// Pseudo NTT configuration with BASE_LEN=16, BASE_DIM=32 (rate 1/2).
/// For row lengths: 128 (DEPTH=1), 1024 (DEPTH=2), 8192 (DEPTH=3), 65536 (DEPTH=4).
#[derive(Clone, Copy)]
pub struct PnttConfigF2_16B16<const DEPTH: usize>;

/// Pseudo NTT configuration with BASE_LEN=64, BASE_DIM=128 (rate 1/2).
/// For row lengths: 512 (DEPTH=1), 4096 (DEPTH=2), 32768 (DEPTH=3).
#[derive(Clone, Copy)]
pub struct PnttConfigF2_16B64<const DEPTH: usize>;

/// Pseudo NTT configuration for F167772161 (5 × 2^25 + 1) with BASE_LEN=32, BASE_DIM=64 (rate 1/2).
/// Supports NTT domains up to 2^25, enabling row lengths up to 2^20.
/// Row lengths: 256 (D=1), 2048 (D=2), 16384 (D=3), 131072 (D=4), 1048576 (D=5).
#[derive(Clone, Copy)]
pub struct PnttConfigF167772161<const DEPTH: usize>;

/// Pseudo NTT configuration for F167772161 with BASE_LEN=16, BASE_DIM=32 (rate 1/2).
/// Row lengths: 128 (D=1), 1024 (D=2), 8192 (D=3), 65536 (D=4), 524288 (D=5).
#[derive(Clone, Copy)]
pub struct PnttConfigF167772161B16<const DEPTH: usize>;

/// Pseudo NTT configuration for F167772161 with BASE_LEN=64, BASE_DIM=128 (rate 1/2).
/// Row lengths: 512 (D=1), 4096 (D=2), 32768 (D=3), 262144 (D=4).
#[derive(Clone, Copy)]
pub struct PnttConfigF167772161B64<const DEPTH: usize>;

mod fq {
    #![allow(non_local_definitions)]
    use ark_ff::{Fp64, MontBackend, MontConfig};
    #[derive(MontConfig)]
    #[modulus = "65537"]
    #[generator = "3"]
    pub struct FqConfig;

    pub type FqBackend = MontBackend<FqConfig, 1>;
    pub type Fq = Fp64<FqBackend>;

    #[allow(clippy::cast_possible_truncation)] // We know modulus is small enough.
    pub const MODULUS: u32 = FqConfig::MODULUS.0[0] as u32;
}

/// Field F167772161 = 5 × 2^25 + 1, supports NTT domains up to 2^25.
mod fq_large {
    #![allow(non_local_definitions)]
    use ark_ff::{Fp64, MontBackend, MontConfig};
    #[derive(MontConfig)]
    #[modulus = "167772161"]
    #[generator = "3"]
    pub struct FqLargeConfig;

    pub type FqLargeBackend = MontBackend<FqLargeConfig, 1>;
    pub type FqLarge = Fp64<FqLargeBackend>;

    #[allow(clippy::cast_possible_truncation)]
    pub const MODULUS: u32 = FqLargeConfig::MODULUS.0[0] as u32;
}

/// Field F1179649 = 9 × 2^17 + 1, supports NTT domains up to 2^17.
mod fq_mid {
    #![allow(non_local_definitions)]
    use ark_ff::{Fp64, MontBackend, MontConfig};
    #[derive(MontConfig)]
    #[modulus = "1179649"]
    #[generator = "19"]
    pub struct FqMidConfig;

    pub type FqMidBackend = MontBackend<FqMidConfig, 1>;
    pub type FqMid = Fp64<FqMidBackend>;

    #[allow(clippy::cast_possible_truncation)]
    pub const MODULUS: u32 = FqMidConfig::MODULUS.0[0] as u32;
}

/// Field F7340033 = 7 × 2^20 + 1, supports NTT domains up to 2^20.
mod fq_mid2 {
    #![allow(non_local_definitions)]
    use ark_ff::{Fp64, MontBackend, MontConfig};
    #[derive(MontConfig)]
    #[modulus = "7340033"]
    #[generator = "3"]
    pub struct FqMid2Config;

    pub type FqMid2Backend = MontBackend<FqMid2Config, 1>;
    pub type FqMid2 = Fp64<FqMid2Backend>;

    #[allow(clippy::cast_possible_truncation)]
    pub const MODULUS: u32 = FqMid2Config::MODULUS.0[0] as u32;
}

impl<const DEPTH: usize> Config for PnttConfigF2_16_1<DEPTH> {
    type Field = fq::Fq;
    const FIELD_MODULUS: u32 = fq::MODULUS;
    const BASE_LEN: usize = 32;
    const BASE_DIM: usize = 64;
    const DEPTH: usize = DEPTH;
    const BASE_TWIDDLES: [PnttInt; 8] = [1, 4096, -256, 16, -1, -4096, 256, -16];

    fn field_to_int_normalized(x: Self::Field) -> PnttInt {
        let big_int = fq::FqBackend::into_bigint(x);

        precompute::normalize_field_element(big_int.0[0], Self::FIELD_MODULUS)
    }
}

impl<const DEPTH: usize> Config for PnttConfigF2_16B16<DEPTH> {
    type Field = fq::Fq;
    const FIELD_MODULUS: u32 = fq::MODULUS;
    const BASE_LEN: usize = 16;
    const BASE_DIM: usize = 32;
    const DEPTH: usize = DEPTH;
    const BASE_TWIDDLES: [PnttInt; 8] = [1, 4096, -256, 16, -1, -4096, 256, -16];

    fn field_to_int_normalized(x: Self::Field) -> PnttInt {
        let big_int = fq::FqBackend::into_bigint(x);

        precompute::normalize_field_element(big_int.0[0], Self::FIELD_MODULUS)
    }
}

impl<const DEPTH: usize> Config for PnttConfigF2_16B64<DEPTH> {
    type Field = fq::Fq;
    const FIELD_MODULUS: u32 = fq::MODULUS;
    const BASE_LEN: usize = 64;
    const BASE_DIM: usize = 128;
    const DEPTH: usize = DEPTH;
    const BASE_TWIDDLES: [PnttInt; 8] = [1, 4096, -256, 16, -1, -4096, 256, -16];

    fn field_to_int_normalized(x: Self::Field) -> PnttInt {
        let big_int = fq::FqBackend::into_bigint(x);

        precompute::normalize_field_element(big_int.0[0], Self::FIELD_MODULUS)
    }
}

// F167772161 = 5 × 2^25 + 1 configurations
// 8th roots of unity for F167772161: [1, 71493608, 65249968, 30406922, -1, -71493608, -65249968, -30406922]
impl<const DEPTH: usize> Config for PnttConfigF167772161<DEPTH> {
    type Field = fq_large::FqLarge;
    const FIELD_MODULUS: u32 = fq_large::MODULUS;
    const BASE_LEN: usize = 32;
    const BASE_DIM: usize = 64;
    const DEPTH: usize = DEPTH;
    // 8th roots of unity in F167772161, centered around 0
    const BASE_TWIDDLES: [PnttInt; 8] = [1, 71493608, 65249968, 30406922, -1, -71493608, -65249968, -30406922];

    fn field_to_int_normalized(x: Self::Field) -> PnttInt {
        let big_int = fq_large::FqLargeBackend::into_bigint(x);

        precompute::normalize_field_element(big_int.0[0], Self::FIELD_MODULUS)
    }
}

impl<const DEPTH: usize> Config for PnttConfigF167772161B16<DEPTH> {
    type Field = fq_large::FqLarge;
    const FIELD_MODULUS: u32 = fq_large::MODULUS;
    const BASE_LEN: usize = 16;
    const BASE_DIM: usize = 32;
    const DEPTH: usize = DEPTH;
    const BASE_TWIDDLES: [PnttInt; 8] = [1, 71493608, 65249968, 30406922, -1, -71493608, -65249968, -30406922];

    fn field_to_int_normalized(x: Self::Field) -> PnttInt {
        let big_int = fq_large::FqLargeBackend::into_bigint(x);

        precompute::normalize_field_element(big_int.0[0], Self::FIELD_MODULUS)
    }
}

impl<const DEPTH: usize> Config for PnttConfigF167772161B64<DEPTH> {
    type Field = fq_large::FqLarge;
    const FIELD_MODULUS: u32 = fq_large::MODULUS;
    const BASE_LEN: usize = 64;
    const BASE_DIM: usize = 128;
    const DEPTH: usize = DEPTH;
    const BASE_TWIDDLES: [PnttInt; 8] = [1, 71493608, 65249968, 30406922, -1, -71493608, -65249968, -30406922];

    fn field_to_int_normalized(x: Self::Field) -> PnttInt {
        let big_int = fq_large::FqLargeBackend::into_bigint(x);

        precompute::normalize_field_element(big_int.0[0], Self::FIELD_MODULUS)
    }
}

/// Pseudo NTT configuration for F1179649 (9 × 2^17 + 1) with BASE_LEN=16, BASE_DIM=32 (rate 1/2).
/// Supports NTT domains up to 2^17, enabling row lengths up to 2^16.
/// Row lengths: 128 (D=1), 1024 (D=2), 8192 (D=3), 65536 (D=4).
#[derive(Clone, Copy)]
pub struct PnttConfigF1179649B16<const DEPTH: usize>;

impl<const DEPTH: usize> Config for PnttConfigF1179649B16<DEPTH> {
    type Field = fq_mid::FqMid;
    const FIELD_MODULUS: u32 = fq_mid::MODULUS;
    const BASE_LEN: usize = 16;
    const BASE_DIM: usize = 32;
    const DEPTH: usize = DEPTH;
    // 8th roots of unity in F1179649, centered around 0
    const BASE_TWIDDLES: [PnttInt; 8] = [1, 494273, 490629, -495809, -1, -494273, -490629, 495809];

    fn field_to_int_normalized(x: Self::Field) -> PnttInt {
        let big_int = fq_mid::FqMidBackend::into_bigint(x);

        precompute::normalize_field_element(big_int.0[0], Self::FIELD_MODULUS)
    }
}

/// Pseudo NTT configuration for F7340033 (7 × 2^20 + 1) with BASE_LEN=16, BASE_DIM=32 (rate 1/2).
/// Supports NTT domains up to 2^20, enabling row lengths up to 2^19.
/// Row lengths: 128 (D=1), 1024 (D=2), 8192 (D=3), 65536 (D=4), 524288 (D=5).
#[derive(Clone, Copy)]
pub struct PnttConfigF7340033B16<const DEPTH: usize>;

impl<const DEPTH: usize> Config for PnttConfigF7340033B16<DEPTH> {
    type Field = fq_mid2::FqMid2;
    const FIELD_MODULUS: u32 = fq_mid2::MODULUS;
    const BASE_LEN: usize = 16;
    const BASE_DIM: usize = 32;
    const DEPTH: usize = DEPTH;
    // 8th roots of unity in F7340033, centered around 0
    const BASE_TWIDDLES: [PnttInt; 8] = [1, 2001861, 2306278, -3413510, -1, -2001861, -2306278, 3413510];

    fn field_to_int_normalized(x: Self::Field) -> PnttInt {
        let big_int = fq_mid2::FqMid2Backend::into_bigint(x);

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
    fn check_twiddles_f65537() {
        check_twiddles_generic::<PnttConfigF2_16_1<1>>();
    }

    #[test]
    fn check_twiddles_f167772161() {
        check_twiddles_generic::<PnttConfigF167772161<1>>();
    }

    #[test]
    fn check_twiddles_f1179649() {
        check_twiddles_generic::<PnttConfigF1179649B16<1>>();
    }

    #[test]
    fn check_twiddles_f7340033() {
        check_twiddles_generic::<PnttConfigF7340033B16<1>>();
    }
}
