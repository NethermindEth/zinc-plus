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
    /// Precomputed octet-reversal permutation table.
    /// `oct_rev_table[chunk]` = `octet_reversal(chunk, DEPTH)` for
    /// `chunk` in `0 .. 8^DEPTH`.
    pub oct_rev_table: Vec<usize>,
    _phantom: PhantomData<C>,
}

impl<C: Config> Default for Radix8PnttParams<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: Config> Radix8PnttParams<C> {
    /// Precompute pseudo NTT parameters.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn new() -> Self {
        use super::octet_reversal::octet_reversal;
        let num_chunks = 1 << (3 * C::DEPTH);
        let oct_rev_table: Vec<usize> = (0..num_chunks)
            .map(|c| octet_reversal(c, C::DEPTH))
            .collect();
        Self {
            base_matrix: precompute::precompute_base_matrix::<C>(),
            butterfly_twiddles: precompute::precompute_butterfly_twiddles::<C>(),
            oct_rev_table,
            _phantom: PhantomData,
        }
    }
}

/// Pseudo NTT configuration derived from
/// the field `Fp` for `p = 2^16 + 1`.
///
/// Supports `DEPTH` up to `3`.
///
/// With `BASE_LEN = 32` and `BASE_DIM = 64`,
/// this yields rate $\frac{1}{2}$ codes.
#[derive(Clone, Copy)]
pub struct PnttConfigF2_16_1<const DEPTH: usize>;

/// Pseudo NTT configuration derived from
/// the field `Fp` for `p = 2^16 + 1`.
///
/// Supports `DEPTH` up to `3`.
///
/// With `BASE_LEN = 32` and `BASE_DIM = 128`,
/// this yields rate $\frac{1}{4}$ codes.
#[derive(Clone, Copy)]
pub struct PnttConfigF2_16_1_Rate1_4<const DEPTH: usize>;

/// Pseudo NTT configuration derived from
/// the field `Fp` for `p = 2^16 + 1`.
///
/// With `BASE_LEN` and `BASE_DIM = 2 * BASE_LEN`,
/// this yields rate $\frac{1}{2}$ codes.
#[derive(Clone, Copy)]
pub struct PnttConfigF2_16_1_Rate1_2_Base<const BASE_LEN: usize, const DEPTH: usize>;

/// Pseudo NTT configuration derived from
/// the field `Fp` for `p = 2^16 + 1`.
///
/// With `BASE_LEN` and `BASE_DIM = 4 * BASE_LEN`,
/// this yields rate $\frac{1}{4}$ codes.
#[derive(Clone, Copy)]
pub struct PnttConfigF2_16_1_Rate1_4_Base<const BASE_LEN: usize, const DEPTH: usize>;

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

/// Field Fp for p = 12289 = 3 * 2^12 + 1.
/// This is the smallest prime supporting 4096th roots of unity,
/// enabling depth-3 configurations with BASE_DIM up to 8.
mod fq_12289 {
    #![allow(non_local_definitions)]
    use ark_ff::{Fp64, MontBackend, MontConfig};
    #[derive(MontConfig)]
    #[modulus = "12289"]
    #[generator = "11"]
    pub struct FqConfig;

    pub type FqBackend = MontBackend<FqConfig, 1>;
    pub type Fq = Fp64<FqBackend>;

    #[allow(clippy::cast_possible_truncation)]
    pub const MODULUS: u32 = FqConfig::MODULUS.0[0] as u32;
}

/// Field Fp for p = 257 = 2^8 + 1 (Fermat prime).
/// Supports up to 256th roots of unity, enabling small depth-2
/// configurations with compact base matrices.
mod fq_257 {
    #![allow(non_local_definitions)]
    use ark_ff::{Fp64, MontBackend, MontConfig};
    #[derive(MontConfig)]
    #[modulus = "257"]
    #[generator = "3"]
    pub struct FqConfig;

    pub type FqBackend = MontBackend<FqConfig, 1>;
    pub type Fq = Fp64<FqBackend>;

    #[allow(clippy::cast_possible_truncation)]
    pub const MODULUS: u32 = FqConfig::MODULUS.0[0] as u32;
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

impl<const DEPTH: usize> Config for PnttConfigF2_16_1_Rate1_4<DEPTH> {
    type Field = fq::Fq;
    const FIELD_MODULUS: u32 = fq::MODULUS;
    const BASE_LEN: usize = 32;
    const BASE_DIM: usize = 128;
    const DEPTH: usize = DEPTH;
    const BASE_TWIDDLES: [PnttInt; 8] = [1, 4096, -256, 16, -1, -4096, 256, -16];

    fn field_to_int_normalized(x: Self::Field) -> PnttInt {
        let big_int = fq::FqBackend::into_bigint(x);

        precompute::normalize_field_element(big_int.0[0], Self::FIELD_MODULUS)
    }
}

impl<const BASE_LEN: usize, const DEPTH: usize> Config
    for PnttConfigF2_16_1_Rate1_2_Base<BASE_LEN, DEPTH>
{
    type Field = fq::Fq;
    const FIELD_MODULUS: u32 = fq::MODULUS;
    const BASE_LEN: usize = BASE_LEN;
    const BASE_DIM: usize = BASE_LEN * 2;
    const DEPTH: usize = DEPTH;
    const BASE_TWIDDLES: [PnttInt; 8] = [1, 4096, -256, 16, -1, -4096, 256, -16];

    fn field_to_int_normalized(x: Self::Field) -> PnttInt {
        let big_int = fq::FqBackend::into_bigint(x);

        precompute::normalize_field_element(big_int.0[0], Self::FIELD_MODULUS)
    }
}

impl<const BASE_LEN: usize, const DEPTH: usize> Config
    for PnttConfigF2_16_1_Rate1_4_Base<BASE_LEN, DEPTH>
{
    type Field = fq::Fq;
    const FIELD_MODULUS: u32 = fq::MODULUS;
    const BASE_LEN: usize = BASE_LEN;
    const BASE_DIM: usize = BASE_LEN * 4;
    const DEPTH: usize = DEPTH;
    const BASE_TWIDDLES: [PnttInt; 8] = [1, 4096, -256, 16, -1, -4096, 256, -16];

    fn field_to_int_normalized(x: Self::Field) -> PnttInt {
        let big_int = fq::FqBackend::into_bigint(x);

        precompute::normalize_field_element(big_int.0[0], Self::FIELD_MODULUS)
    }
}

/// Depth-2 configuration for message size $2^{11}$ with rate $\frac{1}{2}$.
pub type PnttConfigF2_16_1_Depth2_Rate1_2 = PnttConfigF2_16_1<2>;

/// Depth-2 configuration for message size $2^{11}$ with rate $\frac{1}{4}$.
pub type PnttConfigF2_16_1_Depth2_Rate1_4 = PnttConfigF2_16_1_Rate1_4<2>;

/// Depth-1 configuration for message size $2^{7}$ with rate $\frac{1}{2}$.
pub type PnttConfigF2_16_1_Base16_Depth1_Rate1_2 =
    PnttConfigF2_16_1_Rate1_2_Base<16, 1>;

/// Depth-1 configuration for message size $2^{8}$ with rate $\frac{1}{2}$.
pub type PnttConfigF2_16_1_Base32_Depth1_Rate1_2 =
    PnttConfigF2_16_1_Rate1_2_Base<32, 1>;

/// Depth-1 configuration for message size $2^{9}$ with rate $\frac{1}{2}$.
pub type PnttConfigF2_16_1_Base64_Depth1_Rate1_2 =
    PnttConfigF2_16_1_Rate1_2_Base<64, 1>;

/// Depth-1 configuration for message size $2^{7}$ with rate $\frac{1}{4}$.
pub type PnttConfigF2_16_1_Base16_Depth1_Rate1_4 =
    PnttConfigF2_16_1_Rate1_4_Base<16, 1>;

/// Depth-1 configuration for message size $2^{8}$ with rate $\frac{1}{4}$.
pub type PnttConfigF2_16_1_Base32_Depth1_Rate1_4 =
    PnttConfigF2_16_1_Rate1_4_Base<32, 1>;

/// Depth-1 configuration for message size $2^{9}$ with rate $\frac{1}{4}$.
pub type PnttConfigF2_16_1_Base64_Depth1_Rate1_4 =
    PnttConfigF2_16_1_Rate1_4_Base<64, 1>;

/// Depth-2 configuration for message size $2^{12}$ with rate $\frac{1}{2}$.
/// Uses the Fermat prime field $\mathbb{F}_{65537}$ where $65537 = 2^{16} + 1$.
///
/// Configuration:
/// - `BASE_LEN = 64`, `BASE_DIM = 128`
/// - `INPUT_LEN = 64 \cdot 8^2 = 4096 = 2^{12}`
/// - `OUTPUT_LEN = 128 \cdot 8^2 = 8192 = 2^{13}`
pub type PnttConfigF2_16_1_Base64_Depth2_Rate1_2 =
    PnttConfigF2_16_1_Rate1_2_Base<64, 2>;

/// Depth-2 configuration for message size $2^{13}$ with rate $\frac{1}{2}$.
/// Uses the Fermat prime field $\mathbb{F}_{65537}$ where $65537 = 2^{16} + 1$.
///
/// Configuration:
/// - `BASE_LEN = 128`, `BASE_DIM = 256`
/// - `INPUT_LEN = 128 \cdot 8^2 = 8192 = 2^{13}`
/// - `OUTPUT_LEN = 256 \cdot 8^2 = 16384 = 2^{14}`
pub type PnttConfigF2_16_1_Base128_Depth2_Rate1_2 =
    PnttConfigF2_16_1_Rate1_2_Base<128, 2>;

/// Depth-2 configuration for message size $2^{14}$ with rate $\frac{1}{2}$.
/// Uses the Fermat prime field $\mathbb{F}_{65537}$ where $65537 = 2^{16} + 1$.
///
/// Configuration:
/// - `BASE_LEN = 256`, `BASE_DIM = 512`
/// - `INPUT_LEN = 256 \cdot 8^2 = 16384 = 2^{14}`
/// - `OUTPUT_LEN = 512 \cdot 8^2 = 32768 = 2^{15}`
pub type PnttConfigF2_16_1_Base256_Depth2_Rate1_2 =
    PnttConfigF2_16_1_Rate1_2_Base<256, 2>;

/// Depth-2 configuration for message size $2^{15}$ with rate $\frac{1}{2}$.
/// Uses the Fermat prime field $\mathbb{F}_{65537}$ where $65537 = 2^{16} + 1$.
///
/// Configuration:
/// - `BASE_LEN = 512`, `BASE_DIM = 1024`
/// - `INPUT_LEN = 512 \cdot 8^2 = 32768 = 2^{15}`
/// - `OUTPUT_LEN = 1024 \cdot 8^2 = 65536 = 2^{16}`
pub type PnttConfigF2_16_1_Base512_Depth2_Rate1_2 =
    PnttConfigF2_16_1_Rate1_2_Base<512, 2>;

/// Depth-3 configuration for message size $2^{12}$ with rate $\frac{1}{2}$.
/// Uses the Fermat prime field $\mathbb{F}_{65537}$ where $65537 = 2^{16} + 1$.
/// 
/// Configuration:
/// - `BASE_LEN = 8`, `BASE_DIM = 16`
/// - `INPUT_LEN = 8 \cdot 8^3 = 4096 = 2^{12}`
/// - `OUTPUT_LEN = 16 \cdot 8^3 = 8192 = 2^{13}`
pub type PnttConfigF2_16_1_Base8_Depth3_Rate1_2 =
    PnttConfigF2_16_1_Rate1_2_Base<8, 3>;

/// Depth-3 configuration for message size $2^{12}$ with rate $\frac{1}{4}$.
/// Uses the Fermat prime field $\mathbb{F}_{65537}$ where $65537 = 2^{16} + 1$.
///
/// **Warning:** This configuration overflows `i64` codeword coefficients at
/// butterfly stage 2 (56-bit intermediate × 16-bit twiddle = 72 bits > 63).
/// Use [`PnttConfigF2_16_1_Base64_Depth2_Rate1_4`] instead for `i64` coefficients.
///
/// Configuration:
/// - `BASE_LEN = 8`, `BASE_DIM = 32`
/// - `INPUT_LEN = 8 \cdot 8^3 = 4096 = 2^{12}`
/// - `OUTPUT_LEN = 32 \cdot 8^3 = 16384 = 2^{14}`
pub type PnttConfigF2_16_1_Base8_Depth3_Rate1_4 =
    PnttConfigF2_16_1_Rate1_4_Base<8, 3>;

/// Depth-2 configuration for message size $2^{12}$ with rate $\frac{1}{4}$.
/// Uses the Fermat prime field $\mathbb{F}_{65537}$ where $65537 = 2^{16} + 1$.
///
/// This is the recommended depth-2 alternative to the depth-3 config which
/// overflows `i64`. The larger base matrix (256×64) trades a bigger base layer
/// for fewer butterfly stages, keeping intermediates within 59 bits.
///
/// Configuration:
/// - `BASE_LEN = 64`, `BASE_DIM = 256`
/// - `INPUT_LEN = 64 \cdot 8^2 = 4096 = 2^{12}`
/// - `OUTPUT_LEN = 256 \cdot 8^2 = 16384 = 2^{14}`
pub type PnttConfigF2_16_1_Base64_Depth2_Rate1_4 =
    PnttConfigF2_16_1_Rate1_4_Base<64, 2>;

/// Depth-2 configuration for message size $2^{13}$ with rate $\frac{1}{4}$.
/// Uses the Fermat prime field $\mathbb{F}_{65537}$ where $65537 = 2^{16} + 1$.
///
/// Configuration:
/// - `BASE_LEN = 128`, `BASE_DIM = 512`
/// - `INPUT_LEN = 128 \cdot 8^2 = 8192 = 2^{13}`
/// - `OUTPUT_LEN = 512 \cdot 8^2 = 32768 = 2^{15}`
pub type PnttConfigF2_16_1_Base128_Depth2_Rate1_4 =
    PnttConfigF2_16_1_Rate1_4_Base<128, 2>;

/// Depth-2 configuration for message size $2^{14}$ with rate $\frac{1}{4}$.
/// Uses the Fermat prime field $\mathbb{F}_{65537}$ where $65537 = 2^{16} + 1$.
///
/// Configuration:
/// - `BASE_LEN = 256`, `BASE_DIM = 1024`
/// - `INPUT_LEN = 256 \cdot 8^2 = 16384 = 2^{14}`
/// - `OUTPUT_LEN = 1024 \cdot 8^2 = 65536 = 2^{16}`
pub type PnttConfigF2_16_1_Base256_Depth2_Rate1_4 =
    PnttConfigF2_16_1_Rate1_4_Base<256, 2>;

/// Pseudo NTT configuration derived from the field Fp for p = 12289.
/// This is the smallest prime supporting 4096th roots of unity.
///
/// With `BASE_LEN = 4` and `BASE_DIM = 8`, this yields rate 1/2 codes.
/// At `DEPTH = 3`, this encodes messages of size 2^11.
#[derive(Clone, Copy)]
pub struct PnttConfigF12289_Rate1_2<const BASE_LEN: usize, const DEPTH: usize>;

impl<const BASE_LEN: usize, const DEPTH: usize> Config
    for PnttConfigF12289_Rate1_2<BASE_LEN, DEPTH>
{
    type Field = fq_12289::Fq;
    const FIELD_MODULUS: u32 = fq_12289::MODULUS;
    const BASE_LEN: usize = BASE_LEN;
    const BASE_DIM: usize = BASE_LEN * 2;
    const DEPTH: usize = DEPTH;
    // 8th roots of unity in F_12289 (normalized to [-p/2, p/2])
    const BASE_TWIDDLES: [PnttInt; 8] = [1, -4043, 1479, 5146, -1, 4043, -1479, -5146];

    fn field_to_int_normalized(x: Self::Field) -> PnttInt {
        let big_int = fq_12289::FqBackend::into_bigint(x);

        precompute::normalize_field_element(big_int.0[0], Self::FIELD_MODULUS)
    }
}

/// Depth-3 configuration for message size $2^{11}$ with rate $\frac{1}{2}$.
/// Uses the smallest possible base field: $\mathbb{F}_{12289}$ where $12289 = 3 \cdot 2^{12} + 1$.
pub type PnttConfigF12289_Depth3_Rate1_2 = PnttConfigF12289_Rate1_2<4, 3>;

/// Pseudo NTT configuration derived from the field Fp for p = 257 = 2^8 + 1.
/// This is a Fermat prime with multiplicative group of order 256.
///
/// With `BASE_LEN` and `BASE_DIM = 2 * BASE_LEN`, this yields rate 1/2 codes.
#[derive(Clone, Copy)]
pub struct PnttConfigF257_Rate1_2<const BASE_LEN: usize, const DEPTH: usize>;

impl<const BASE_LEN: usize, const DEPTH: usize> Config
    for PnttConfigF257_Rate1_2<BASE_LEN, DEPTH>
{
    type Field = fq_257::Fq;
    const FIELD_MODULUS: u32 = fq_257::MODULUS;
    const BASE_LEN: usize = BASE_LEN;
    const BASE_DIM: usize = BASE_LEN * 2;
    const DEPTH: usize = DEPTH;
    // 8th roots of unity in F_257 (normalized to [-128, 128])
    // ω = 64 is a primitive 8th root of unity (3^32 mod 257)
    const BASE_TWIDDLES: [PnttInt; 8] = [1, 64, -16, 4, -1, -64, 16, -4];

    fn field_to_int_normalized(x: Self::Field) -> PnttInt {
        let big_int = fq_257::FqBackend::into_bigint(x);

        precompute::normalize_field_element(big_int.0[0], Self::FIELD_MODULUS)
    }
}

/// Depth-2 configuration for message size $2^7$ with rate $\frac{1}{2}$.
/// Uses the Fermat prime field: $\mathbb{F}_{257}$ where $257 = 2^8 + 1$.
pub type PnttConfigF257_Depth2_Rate1_2 = PnttConfigF257_Rate1_2<2, 2>;

/// Depth-1 configuration for message size $2^7$ with rate $\frac{1}{2}$.
/// Uses the Fermat prime field: $\mathbb{F}_{257}$ where $257 = 2^8 + 1$.
/// Alternative to Depth2 with larger base matrix (32x16 vs 4x2).
pub type PnttConfigF257_Base16_Depth1_Rate1_2 = PnttConfigF257_Rate1_2<16, 1>;

/// Depth-1 configuration for message size $2^6$ with rate $\frac{1}{2}$.
/// Uses the Fermat prime field: $\mathbb{F}_{257}$ where $257 = 2^8 + 1$.
pub type PnttConfigF257_Base8_Depth1_Rate1_2 = PnttConfigF257_Rate1_2<8, 1>;

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
        check_twiddles_generic::<PnttConfigF2_16_1<1>>();
    }

    #[test]
    fn check_twiddles_f12289() {
        check_twiddles_generic::<PnttConfigF12289_Depth3_Rate1_2>();
    }

    #[test]
    fn check_f12289_depth3_config() {
        // Verify the configuration parameters
        type C = PnttConfigF12289_Depth3_Rate1_2;
        assert_eq!(C::BASE_LEN, 4);
        assert_eq!(C::BASE_DIM, 8);
        assert_eq!(C::DEPTH, 3);
        assert_eq!(C::INPUT_LEN, 2048); // 4 * 8^3 = 2^11
        assert_eq!(C::OUTPUT_LEN, 4096); // 8 * 8^3 = 2^12
        assert_eq!(C::FIELD_MODULUS, 12289);
        // Rate = INPUT_LEN / OUTPUT_LEN = 1/2
    }

    #[test]
    fn check_twiddles_f257() {
        check_twiddles_generic::<PnttConfigF257_Depth2_Rate1_2>();
    }

    #[test]
    fn check_f257_depth2_config() {
        // Verify the configuration parameters
        type C = PnttConfigF257_Depth2_Rate1_2;
        assert_eq!(C::BASE_LEN, 2);
        assert_eq!(C::BASE_DIM, 4);
        assert_eq!(C::DEPTH, 2);
        assert_eq!(C::INPUT_LEN, 128); // 2 * 8^2 = 2^7
        assert_eq!(C::OUTPUT_LEN, 256); // 4 * 8^2 = 2^8
        assert_eq!(C::FIELD_MODULUS, 257);
        // Rate = INPUT_LEN / OUTPUT_LEN = 1/2
    }

    #[test]
    fn check_f257_base16_depth1_config() {
        // Verify the configuration parameters for 2^7 message size with depth 1
        type C = PnttConfigF257_Base16_Depth1_Rate1_2;
        assert_eq!(C::BASE_LEN, 16);
        assert_eq!(C::BASE_DIM, 32);
        assert_eq!(C::DEPTH, 1);
        assert_eq!(C::INPUT_LEN, 128); // 16 * 8^1 = 2^7
        assert_eq!(C::OUTPUT_LEN, 256); // 32 * 8^1 = 2^8
        assert_eq!(C::FIELD_MODULUS, 257);
    }

    #[test]
    fn check_f257_base8_depth1_config() {
        // Verify the configuration parameters for 2^6 message size
        type C = PnttConfigF257_Base8_Depth1_Rate1_2;
        assert_eq!(C::BASE_LEN, 8);
        assert_eq!(C::BASE_DIM, 16);
        assert_eq!(C::DEPTH, 1);
        assert_eq!(C::INPUT_LEN, 64); // 8 * 8^1 = 2^6
        assert_eq!(C::OUTPUT_LEN, 128); // 16 * 8^1 = 2^7
        assert_eq!(C::FIELD_MODULUS, 257);
    }

    #[test]
    fn check_f65537_base64_depth2_rate1_2_config() {
        type C = PnttConfigF2_16_1_Base64_Depth2_Rate1_2;
        assert_eq!(C::BASE_LEN, 64);
        assert_eq!(C::BASE_DIM, 128);
        assert_eq!(C::DEPTH, 2);
        assert_eq!(C::INPUT_LEN, 4096); // 64 * 8^2 = 2^12
        assert_eq!(C::OUTPUT_LEN, 8192); // 128 * 8^2 = 2^13
        assert_eq!(C::FIELD_MODULUS, 65537);
    }

    #[test]
    fn check_f65537_base128_depth2_rate1_2_config() {
        type C = PnttConfigF2_16_1_Base128_Depth2_Rate1_2;
        assert_eq!(C::BASE_LEN, 128);
        assert_eq!(C::BASE_DIM, 256);
        assert_eq!(C::DEPTH, 2);
        assert_eq!(C::INPUT_LEN, 8192); // 128 * 8^2 = 2^13
        assert_eq!(C::OUTPUT_LEN, 16384); // 256 * 8^2 = 2^14
        assert_eq!(C::FIELD_MODULUS, 65537);
    }

    #[test]
    fn check_f65537_base256_depth2_rate1_2_config() {
        type C = PnttConfigF2_16_1_Base256_Depth2_Rate1_2;
        assert_eq!(C::BASE_LEN, 256);
        assert_eq!(C::BASE_DIM, 512);
        assert_eq!(C::DEPTH, 2);
        assert_eq!(C::INPUT_LEN, 16384); // 256 * 8^2 = 2^14
        assert_eq!(C::OUTPUT_LEN, 32768); // 512 * 8^2 = 2^15
        assert_eq!(C::FIELD_MODULUS, 65537);
    }

    #[test]
    fn check_f65537_base512_depth2_rate1_2_config() {
        type C = PnttConfigF2_16_1_Base512_Depth2_Rate1_2;
        assert_eq!(C::BASE_LEN, 512);
        assert_eq!(C::BASE_DIM, 1024);
        assert_eq!(C::DEPTH, 2);
        assert_eq!(C::INPUT_LEN, 32768); // 512 * 8^2 = 2^15
        assert_eq!(C::OUTPUT_LEN, 65536); // 1024 * 8^2 = 2^16
        assert_eq!(C::FIELD_MODULUS, 65537);
    }

    #[test]
    fn check_f65537_base8_depth3_config() {
        // Verify the configuration parameters for 2^12 message size
        type C = PnttConfigF2_16_1_Base8_Depth3_Rate1_2;
        assert_eq!(C::BASE_LEN, 8);
        assert_eq!(C::BASE_DIM, 16);
        assert_eq!(C::DEPTH, 3);
        assert_eq!(C::INPUT_LEN, 4096); // 8 * 8^3 = 2^12
        assert_eq!(C::OUTPUT_LEN, 8192); // 16 * 8^3 = 2^13
        assert_eq!(C::FIELD_MODULUS, 65537);
        // Rate = INPUT_LEN / OUTPUT_LEN = 1/2
    }

    #[test]
    fn check_f65537_base64_depth2_rate1_4_config() {
        // Verify the depth-2 rate 1/4 configuration for 2^12 message size
        type C = PnttConfigF2_16_1_Base64_Depth2_Rate1_4;
        assert_eq!(C::BASE_LEN, 64);
        assert_eq!(C::BASE_DIM, 256);
        assert_eq!(C::DEPTH, 2);
        assert_eq!(C::INPUT_LEN, 4096); // 64 * 8^2 = 2^12
        assert_eq!(C::OUTPUT_LEN, 16384); // 256 * 8^2 = 2^14
        assert_eq!(C::FIELD_MODULUS, 65537);
        // Rate = INPUT_LEN / OUTPUT_LEN = 1/4
    }

    #[test]
    fn check_twiddles_f65537_base64_depth2_rate1_4() {
        check_twiddles_generic::<PnttConfigF2_16_1_Base64_Depth2_Rate1_4>();
    }
}
