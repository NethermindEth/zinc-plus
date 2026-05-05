//! Compile-time projecting primes for the `fixed-prime` branch.
//!
//! In Zinc+, the projecting prime `q` for the
//! `\phi_q : Z[X] -> F_q[X]` step (Step 1 of the protocol) is normally
//! drawn from the Fiat–Shamir transcript. On this branch we replace
//! that with a **fixed** prime supplied by the caller.
//!
//! Two well-known secp256k1 primes are provided as constants:
//! - [`SECP256K1_P_LE_BYTES`] — the secp256k1 base field prime
//!   `p = 2^256 − 2^32 − 977`
//!     `= 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F`.
//! - [`SECP256K1_N_LE_BYTES`] — the secp256k1 curve order
//!   `n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141`.
//!
//! The generic [`field_cfg`] constructor accepts an arbitrary 256-bit
//! prime supplied as a little-endian byte slice; thin wrappers
//! [`secp256k1_base_field_cfg`] and [`secp256k1_scalar_field_cfg`] cover
//! the two secp256k1 primes.
//!
//! Soundness: a fixed projecting prime is in general NOT sound — a
//! constraint-grinding adversary can craft witnesses whose ideal-check
//! residues vanish mod a known `q`. For the targeted SHA+ECDSA proving
//! application this does not break soundness (the relevant constraints
//! are honest mod the chosen prime). Do not reuse this branch for other
//! applications without re-doing the soundness analysis.

use crypto_primitives::PrimeField;
use zinc_transcript::traits::ConstTranscribable;
use zinc_utils::from_ref::FromRef;

/// secp256k1 base field prime, little-endian byte order (32 bytes).
///
/// `Uint<LIMBS>::read_transcription_bytes_exact` interprets its input
/// as little-endian limb chunks (see `transcript::traits` impl), so we
/// store the prime in that same order.
pub const SECP256K1_P_LE_BYTES: [u8; 32] = [
    0x2F, 0xFC, 0xFF, 0xFF, 0xFE, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
];

/// secp256k1 curve order `n`, little-endian byte order (32 bytes).
///
/// `n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141`.
///
/// Same little-endian-limb-chunk convention as
/// [`SECP256K1_P_LE_BYTES`].
pub const SECP256K1_N_LE_BYTES: [u8; 32] = [
    0x41, 0x41, 0x36, 0xD0, 0x8C, 0x5E, 0xD2, 0xBF,
    0x3B, 0xA0, 0x48, 0xAF, 0xE6, 0xDC, 0xAE, 0xBA,
    0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
];

/// Build `F::Config` from an arbitrary 256-bit prime supplied as a
/// little-endian byte slice. Replaces the previous
/// `Transcript::get_random_field_cfg` draw.
///
/// Trait bounds match the existing call-site bounds, so no changes to
/// the surrounding generic signatures are required.
///
/// Panics if `FMod::NUM_BYTES != prime_le_bytes.len()` (i.e. the field
/// modulus type cannot hold the supplied prime), or if the supplied
/// bytes do not encode a prime accepted by `F::make_cfg`.
pub fn field_cfg<F, FMod>(prime_le_bytes: &[u8]) -> F::Config
where
    F: PrimeField,
    FMod: ConstTranscribable,
    F::Modulus: FromRef<FMod>,
{
    assert_eq!(
        FMod::NUM_BYTES,
        prime_le_bytes.len(),
        "Fmod byte width must match the supplied prime byte width",
    );
    let prime = FMod::read_transcription_bytes_exact(prime_le_bytes);
    F::make_cfg(&F::Modulus::from_ref(&prime))
        .expect("supplied bytes must encode a prime accepted by F::make_cfg")
}

/// Build `F::Config` for the secp256k1 base field prime `p`.
///
/// Thin wrapper around [`field_cfg`] that supplies
/// [`SECP256K1_P_LE_BYTES`].
pub fn secp256k1_base_field_cfg<F, FMod>() -> F::Config
where
    F: PrimeField,
    FMod: ConstTranscribable,
    F::Modulus: FromRef<FMod>,
{
    field_cfg::<F, FMod>(&SECP256K1_P_LE_BYTES)
}

/// Build `F::Config` for the secp256k1 scalar field prime `n` (the
/// curve order).
///
/// Thin wrapper around [`field_cfg`] that supplies
/// [`SECP256K1_N_LE_BYTES`].
pub fn secp256k1_scalar_field_cfg<F, FMod>() -> F::Config
where
    F: PrimeField,
    FMod: ConstTranscribable,
    F::Modulus: FromRef<FMod>,
{
    field_cfg::<F, FMod>(&SECP256K1_N_LE_BYTES)
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

    /// `SECP256K1_N_LE_BYTES` decodes to the well-known secp256k1
    /// curve order `n` under the same little-endian-limb-chunk
    /// convention.
    #[test]
    fn secp256k1_n_le_bytes_decode_to_curve_order() {
        let n = Uint::<4>::read_transcription_bytes_exact(&SECP256K1_N_LE_BYTES);
        assert_eq!(
            n.as_words(),
            &[
                0xBFD2_5E8C_D036_4141,
                0xBAAE_DCE6_AF48_A03B,
                0xFFFF_FFFF_FFFF_FFFE,
                0xFFFF_FFFF_FFFF_FFFF,
            ],
        );
    }

    /// Construction succeeds for the concrete `MontyField<4>` / `Uint<4>`
    /// combination used by the e2e bench.
    #[test]
    fn secp256k1_base_field_cfg_constructs() {
        let _cfg = secp256k1_base_field_cfg::<MontyField<4>, Uint<4>>();
    }

    /// Mirror of `secp256k1_base_field_cfg_constructs` for the scalar
    /// field config (curve order `n`).
    #[test]
    fn secp256k1_scalar_field_cfg_constructs() {
        let _cfg = secp256k1_scalar_field_cfg::<MontyField<4>, Uint<4>>();
    }
}
