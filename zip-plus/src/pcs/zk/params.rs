//! Parameters for the zero-knowledge masking layer.
//!
//! The parameter chain follows the companion note (§3 and §6):
//!
//! ```text
//! B          = bit width of witness evaluations          (Zt::Eval)
//! B_rho      = bit width of combination challenges       (Zt::Chal, signed)
//! combo_bits = B + B_rho + log2(num_rows) + 1            (|sum_j rho_j w_j|)
//! blind_bits = combo_bits + lambda_zk                    (B_0, blinding box)
//! entry_bits = blind_bits + codeword_growth_bits         (|Enc(w_0)| dominates)
//! p_bits     = entry_bits + lambda_zk                    (mask modulus size)
//! ```
//!
//! The mask modulus `p` is the smallest (probable) prime with exactly
//! `p_bits + 1` bits; both parties derive it deterministically from the bit
//! budgets, so it is a public parameter and never enters the transcript.

use crate::{
    ZipError,
    code::LinearCode,
    pcs::structs::{ZipPlusParams, ZipTypes},
};
use crypto_primitives::crypto_bigint_uint::Uint;
use num_traits::ConstOne;
use zinc_poly::ConstCoeffBitWidth;
use zinc_primality::{MillerRabin, PrimalityTest};
use zinc_transcript::traits::ConstTranscribable;
use zinc_utils::{add, mul, sub};

/// `2^bits` as a `Uint<WL>` (requires `bits < 64 * WL`).
#[allow(clippy::arithmetic_side_effects)] // word/bit index arithmetic bounded by the assert
pub(crate) fn pow2<const WL: usize>(bits: u32) -> Uint<WL> {
    let bit_index = usize::try_from(bits).expect("bit count fits usize");
    assert!(bit_index < WL * 64, "pow2 exponent out of range");
    let mut words = [0 as crypto_bigint::Word; WL];
    words[bit_index / 64] = 1 << (bit_index % 64);
    Uint::from_words(words)
}

/// Public parameters of the modular-masking layer, generic over the limb
/// count `WL` of the wide integer ring in which masked codeword entries and
/// remainders live (`Int<WL>` / `Uint<WL>`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZkMaskParams<const WL: usize> {
    /// The mask modulus `p` (an odd probable prime).
    pub mask_modulus: Uint<WL>,
    /// Bit length of `p`.
    pub mask_modulus_bits: u32,
    /// Reed--Solomon mask dimension `D`; must be at least the total number of
    /// column openings across the lifetime of the commitment.
    pub mask_dim: usize,
    /// Statistical hiding parameter `lambda_zk` (per-symbol smudging
    /// distance `2^-lambda_zk`).
    pub lambda_zk: u32,
    /// Bit length `B_0` of the blinding-row box `[0, 2^B_0)`.
    pub blind_bits: u32,
}

impl<const WL: usize> ZkMaskParams<WL> {
    /// Derives the full parameter chain for a Zip+ instance with `num_rows`
    /// committed witness rows.
    ///
    /// `codeword_growth_bits` bounds the bit growth of the linear code: for
    /// every message row `m`, `|Enc(m)|_inf <= 2^codeword_growth_bits *
    /// |m|_inf`. This is code-specific (for IPRS it follows from the norm
    /// bound in the Zinc+ paper) and is taken as an explicit input here.
    ///
    /// # Errors
    /// Returns [`ZipError::InvalidPcsParam`] if the derived sizes do not fit
    /// the `WL`-limb rings with the headroom required by the remainder
    /// arithmetic, or if the blinding row does not fit `Zt::CombR`.
    pub fn derive<Zt: ZipTypes>(
        num_rows: usize,
        codeword_growth_bits: u32,
        lambda_zk: u32,
        mask_dim: usize,
    ) -> Result<Self, ZipError> {
        if num_rows == 0 || !num_rows.is_power_of_two() {
            return Err(ZipError::InvalidPcsParam(
                "num_rows must be a nonzero power of two".into(),
            ));
        }
        if mask_dim == 0 {
            return Err(ZipError::InvalidPcsParam("mask_dim must be > 0".into()));
        }

        let eval_bits = u32::try_from(Zt::Eval::COEFF_BIT_WIDTH)
            .map_err(|_| ZipError::InvalidPcsParam("Eval bit width overflow".into()))?;
        let chal_bits = u32::try_from(mul!(Zt::Chal::NUM_BYTES, 8usize))
            .map_err(|_| ZipError::InvalidPcsParam("Chal bit width overflow".into()))?;
        let log_rows = num_rows.ilog2();

        // |sum_j rho_j w_j|_inf < 2^(B + B_rho + log J + 1)  (signed challenges).
        let combo_bits = add!(add!(add!(eval_bits, chal_bits), log_rows), 1u32);
        let blind_bits = add!(combo_bits, lambda_zk);
        let entry_bits = add!(blind_bits, codeword_growth_bits);
        let p_bits = add!(entry_bits, lambda_zk);

        // The blinding row and the blinded combination w* must fit CombR
        // (signed): |w*|_inf < 2^(blind_bits + 1).
        let comb_r_bits = u32::try_from(mul!(Zt::CombR::NUM_BYTES, 8usize))
            .map_err(|_| ZipError::InvalidPcsParam("CombR bit width overflow".into()))?;
        if add!(blind_bits, 2u32) > sub!(comb_r_bits, 1u32) {
            return Err(ZipError::InvalidPcsParam(format!(
                "CombR too narrow for the blinding row: need {} bits, have {}",
                add!(blind_bits, 2u32),
                sub!(comb_r_bits, 1u32),
            )));
        }

        // Remainder headroom in Int<WL> (signed): |rem| <= (1 + sum |rho|)(p-1)
        // < 2^(p_bits + chal_bits + log J + 2).
        let wl_bits = u32::try_from(mul!(WL, 64usize))
            .map_err(|_| ZipError::InvalidPcsParam("WL overflow".into()))?;
        let rem_bits = add!(add!(add!(p_bits, chal_bits), log_rows), 2u32);
        if add!(rem_bits, 2u32) > sub!(wl_bits, 1u32) {
            return Err(ZipError::InvalidPcsParam(format!(
                "WL too narrow: remainder arithmetic needs {} bits, Int<WL> has {}",
                add!(rem_bits, 2u32),
                sub!(wl_bits, 1u32),
            )));
        }

        let mask_modulus = next_probable_prime_with_bits::<WL>(p_bits)?;

        Ok(Self {
            mask_modulus,
            mask_modulus_bits: mask_modulus.inner().bits(),
            mask_dim,
            lambda_zk,
            blind_bits,
        })
    }

    /// Like [`Self::derive`], but takes the codeword growth bound from the
    /// linear code itself ([`LinearCode::codeword_growth_bits`]).
    pub fn derive_for_code<Zt: ZipTypes, Lc: LinearCode<Zt>>(
        pp: &ZipPlusParams<Zt, Lc>,
        lambda_zk: u32,
        mask_dim: usize,
    ) -> Result<Self, ZipError> {
        let growth = pp.linear_code.codeword_growth_bits().ok_or_else(|| {
            ZipError::InvalidPcsParam(
                "linear code provides no growth bound; use derive() with an explicit bound"
                    .into(),
            )
        })?;
        Self::derive::<Zt>(pp.num_rows, growth, lambda_zk, mask_dim)
    }
}

/// Returns the smallest probable prime `>= 2^bits` (deterministic search, so
/// both parties derive the same modulus from the same bit budget).
fn next_probable_prime_with_bits<const WL: usize>(bits: u32) -> Result<Uint<WL>, ZipError> {
    let wl_bits = u32::try_from(mul!(WL, 64usize))
        .map_err(|_| ZipError::InvalidPcsParam("WL overflow".into()))?;
    // Leave two bits of headroom so candidate stepping cannot wrap.
    if add!(bits, 2u32) >= wl_bits {
        return Err(ZipError::InvalidPcsParam(format!(
            "mask modulus of {bits} bits does not fit Uint<{WL}>",
        )));
    }
    let one = Uint::<WL>::ONE;
    let two = add!(one, &one, "Uint overflow building 2");
    // candidate = 2^bits + 1 (odd).
    let mut candidate = add!(
        pow2::<WL>(bits),
        &one,
        "Uint overflow building first prime candidate"
    );
    // By Bertrand's postulate a prime exists below 2^(bits+1); the loop is
    // guaranteed to terminate well before the headroom is exhausted.
    while !MillerRabin::is_probably_prime(&candidate) {
        candidate = add!(candidate, &two, "Uint overflow searching for prime");
    }
    Ok(candidate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcs::test_utils::TestZipTypes;

    type Zt = TestZipTypes<1, 2, 4>;
    const WL: usize = 8;

    #[test]
    fn derives_prime_of_expected_size() {
        let params = ZkMaskParams::<WL>::derive::<Zt>(4, 80, 64, 147).expect("derive must work");
        // B = 64, B_rho = 64, log J = 2 => combo = 131, blind = 195,
        // entry = 275, p_bits = 339.
        assert_eq!(params.blind_bits, 195);
        assert_eq!(params.mask_modulus_bits, 340); // smallest prime >= 2^339
        assert!(MillerRabin::is_probably_prime(&params.mask_modulus));
        assert_eq!(params.mask_modulus.as_words()[0] & 1, 1, "p must be odd");
    }

    #[test]
    fn derivation_is_deterministic() {
        let a = ZkMaskParams::<WL>::derive::<Zt>(4, 80, 64, 147).expect("derive must work");
        let b = ZkMaskParams::<WL>::derive::<Zt>(4, 80, 64, 147).expect("derive must work");
        assert_eq!(a, b);
    }

    #[test]
    fn rejects_too_narrow_wl() {
        // WL = 6 gives 384-bit ints; the remainder needs ~407 bits here.
        let res = ZkMaskParams::<6>::derive::<Zt>(4, 80, 64, 147);
        assert!(res.is_err());
    }

    #[test]
    fn rejects_non_power_of_two_rows() {
        assert!(ZkMaskParams::<WL>::derive::<Zt>(3, 80, 64, 147).is_err());
    }

    #[test]
    #[allow(clippy::arithmetic_side_effects, clippy::cast_possible_wrap)]
    fn iprs_growth_bound_is_sound_empirically() {
        use crate::{
            code::{LinearCode, iprs::IprsCode},
            pcs::test_utils::{IPRS_DEPTH, IPRS_ROW_LEN, REP_FACTOR, TestIprsConfig},
        };
        use crypto_primitives::crypto_bigint_int::Int;
        use zinc_utils::CHECKED;

        type Lc = IprsCode<Zt, TestIprsConfig, REP_FACTOR, CHECKED>;
        let code: Lc = IprsCode::new(IPRS_ROW_LEN, IPRS_DEPTH).expect("iprs code");
        let growth = code.codeword_growth_bits().expect("IPRS provides a bound");

        // Encode an adversarially large row (alternating max-magnitude
        // entries) and check the measured growth respects the bound.
        let max_eval_bits = 40u32; // |entry| < 2^40, leaves headroom in Cw
        let entry = (1i64 << max_eval_bits) - 1;
        let row: Vec<Int<1>> = (0..IPRS_ROW_LEN)
            .map(|i| Int::from(if i % 2 == 0 { entry } else { -entry }))
            .collect();
        let cw = code.encode(&row);
        let measured = cw
            .iter()
            .map(|v| v.inner().abs().bits())
            .max()
            .expect("nonempty codeword");
        assert!(
            measured <= max_eval_bits + growth,
            "measured growth {} exceeds bound {}",
            measured - max_eval_bits,
            growth,
        );
    }
}
