//! Compile-time projecting prime for the `fixed-prime` branch.
//!
//! In Zinc+, the projecting prime `q` for the
//! `\phi_q : Z[X] -> F_q[X]` step (Step 1 of the protocol) is normally
//! drawn from the Fiat–Shamir transcript. On this branch we replace
//! that with the **secp256k1 base field prime**
//! `p = 2^256 − 2^32 − 977`
//!   `= 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F`.
//!
//! Soundness: a fixed projecting prime is in general NOT sound — a
//! constraint-grinding adversary can craft witnesses whose ideal-check
//! residues vanish mod a known `q`. For the targeted SHA+ECDSA proving
//! application this does not break soundness (the relevant constraints
//! are honest mod `p`). Do not reuse this branch for other applications
//! without re-doing the soundness analysis.

use ark_ec::AffineRepr;
use ark_ff::{BigInteger, PrimeField as ArkPrimeField};
use crypto_primitives::PrimeField;
use zinc_transcript::traits::ConstTranscribable;
use zinc_utils::from_ref::FromRef;

/// secp256k1 base field prime, little-endian byte order (32 bytes).
///
/// `Uint<LIMBS>::read_transcription_bytes_exact` interprets its input
/// as little-endian limb chunks (see `transcript::traits` impl), so we
/// store the prime in that same order.
pub const SECP256K1_P_LE_BYTES: [u8; 32] = [
    0x2F, 0xFC, 0xFF, 0xFF, 0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
];

/// Build `F::Config` from the secp256k1 base prime, replacing the
/// previous `Transcript::get_random_field_cfg` draw.
///
/// Trait bounds match the existing call-site bounds, so no changes to
/// the surrounding generic signatures are required.
///
/// Panics if `FMod` cannot hold a 256-bit value (its `NUM_BYTES` differs
/// from `SECP256K1_P_LE_BYTES.len()`).
pub fn secp256k1_field_cfg<F, FMod>() -> F::Config
where
    F: PrimeField,
    FMod: ConstTranscribable,
    F::Modulus: FromRef<FMod>,
{
    assert_eq!(
        FMod::NUM_BYTES,
        SECP256K1_P_LE_BYTES.len(),
        "Fmod must be exactly 256 bits to hold the secp256k1 base prime",
    );
    let prime = FMod::read_transcription_bytes_exact(&SECP256K1_P_LE_BYTES);
    F::make_cfg(&F::Modulus::from_ref(&prime)).expect("secp256k1 base field prime is prime")
}

/// Build `F::Config` from the scalar field of an arkworks curve.
///
/// This is the field configuration Hyrax must use: PCS scalar operations
/// happen in `C::ScalarField`, so the PIOP field modulus must match it.
pub fn field_cfg_from_curve_scalar<F, FMod, C>() -> F::Config
where
    F: PrimeField,
    FMod: ConstTranscribable,
    F::Modulus: FromRef<FMod>,
    C: AffineRepr,
{
    let prime = fmod_from_curve_scalar::<FMod, C>();
    F::make_cfg(&F::Modulus::from_ref(&prime))
        .expect("curve scalar modulus must define a valid prime field")
}

fn fmod_from_curve_scalar<FMod, C>() -> FMod
where
    FMod: ConstTranscribable,
    C: AffineRepr,
{
    let modulus_bytes = <C::ScalarField as ArkPrimeField>::MODULUS.to_bytes_le();
    assert!(
        modulus_bytes.len() <= FMod::NUM_BYTES,
        "curve scalar modulus does not fit in the protocol modulus type",
    );

    let mut bytes = vec![0u8; FMod::NUM_BYTES];
    bytes[..modulus_bytes.len()].copy_from_slice(&modulus_bytes);
    FMod::read_transcription_bytes_exact(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_primitives::{crypto_bigint_monty::MontyField, crypto_bigint_uint::Uint};
    use zinc_transcript::traits::GenTranscribable;

    /// `SECP256K1_P_LE_BYTES` decodes to the well-known secp256k1 base
    /// prime when read through the transcription convention used by
    /// `Uint<LIMBS>` (little-endian limb chunks).
    #[test]
    fn secp256k1_p_le_bytes_decode_to_prime() {
        let prime = Uint::<4>::read_transcription_bytes_exact(&SECP256K1_P_LE_BYTES);
        assert_eq!(
            prime.as_words(),
            &[
                0xFFFF_FFFE_FFFF_FC2F,
                0xFFFF_FFFF_FFFF_FFFF,
                0xFFFF_FFFF_FFFF_FFFF,
                0xFFFF_FFFF_FFFF_FFFF,
            ],
        );
    }

    /// Construction succeeds for the concrete `MontyField<4>` / `Uint<4>`
    /// combination used by the e2e bench.
    #[test]
    fn secp256k1_field_cfg_constructs() {
        let _cfg = secp256k1_field_cfg::<MontyField<4>, Uint<4>>();
    }

    #[test]
    fn curve_scalar_field_cfg_constructs() {
        let bn_cfg = field_cfg_from_curve_scalar::<MontyField<4>, Uint<4>, ark_bn254::G1Affine>();
        let secp_cfg =
            field_cfg_from_curve_scalar::<MontyField<4>, Uint<4>, ark_secp256k1::Affine>();

        let bn_modulus = MontyField::<4>::one_with_cfg(&bn_cfg).modulus();
        let secp_modulus = MontyField::<4>::one_with_cfg(&secp_cfg).modulus();

        assert_eq!(
            bn_modulus,
            fmod_from_curve_scalar::<Uint<4>, ark_bn254::G1Affine>(),
        );
        assert_eq!(
            secp_modulus,
            fmod_from_curve_scalar::<Uint<4>, ark_secp256k1::Affine>(),
        );
    }
}
