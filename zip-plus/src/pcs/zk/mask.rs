//! Modular Reed--Solomon masks (paper §3).
//!
//! A mask is the canonical-representative vector `[G s]_p in [0, p)^n`, where
//! `G` is the Vandermonde generator of a Reed--Solomon code of dimension `D`
//! over `F_p` with evaluation points `alpha_l = l + 1` for column `l`, and
//! `s in [0, p)^D` is a uniform seed. Any `<= D` symbols are exactly uniform
//! (Lemma 2.2 of the note). All mod-`p` arithmetic runs in the
//! runtime-modulus Montgomery field [`MontyField`]; canonical lifts come out
//! via `retrieve()`.

use crate::ZipError;
use crypto_primitives::{
    PrimeField, crypto_bigint_monty::MontyField, crypto_bigint_uint::Uint,
};
use rand_core::CryptoRng;
use zinc_transcript::traits::{ConstTranscribable, GenTranscribable};
use zinc_utils::{add, mul, sub};

/// Per-row mask seeds: `seeds[i]` is the seed `s_i in [0, p)^D` of committed
/// row `i` (row 0 is the blinding row).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaskSeeds<const WL: usize> {
    pub seeds: Vec<Vec<Uint<WL>>>,
}

/// The mask-consistency argument attached to an opening (the "inner proof"
/// seam of the construction).
///
/// The final design discharges consistency with a zero-knowledge IOP for the
/// lift relation `R_lift` (paper §5: linear over `F_{p'}` plus range checks).
/// The `Transparent` variant is the v0 stub: it reveals the seeds and lets
/// the verifier recompute the mask symbols at the opened columns. It is
/// binding and complete with the final round structure, but does not hide
/// the opened columns.
// TODO(zk-inner): add a `LiftIop` variant implementing the F_{p'} linear
// ZK-IOP of paper §5 and make it the default; `Transparent` then remains as
// a test/debug mode.
#[derive(Clone, Debug)]
pub enum MaskConsistencyProof<const WL: usize> {
    Transparent {
        seeds: MaskSeeds<WL>,
        salt: [u8; 32],
    },
}

impl<const WL: usize> MaskSeeds<WL> {
    /// Samples `count` independent seeds of dimension `dim`, each entry
    /// uniform in the power-of-two window `[0, 2^(bits(p)-1)) ⊂ [0, p)`.
    ///
    /// The window misses only `[2^(bits(p)-1), p)` — a `(prime gap)/p`
    /// fraction, statistically negligible — and in exchange the seed range
    /// checks of the inner ZK-IOP become pure bit decompositions (cf. the
    /// note's §5 / power-of-two-modulus discussion).
    ///
    /// Mask randomness is *secret prover randomness*: it must come from an
    /// RNG, never from the public transcript.
    pub fn sample(
        rng: &mut impl CryptoRng,
        count: usize,
        dim: usize,
        modulus: &Uint<WL>,
    ) -> Self {
        let window_bits = sub!(modulus.inner().bits(), 1u32);
        let seeds = (0..count)
            .map(|_| (0..dim).map(|_| sample_window(rng, window_bits)).collect())
            .collect();
        Self { seeds }
    }

    /// Binding commitment to the seeds: `blake3(count || dim || entries || salt)`.
    /// Stubs the witness oracle of the future inner ZK-IOP.
    pub fn commitment(&self, salt: &[u8; 32]) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"zk-zip/seed-commitment/v0");
        hasher.update(&(self.seeds.len() as u64).to_le_bytes());
        let dim = self.seeds.first().map_or(0, Vec::len);
        hasher.update(&(dim as u64).to_le_bytes());
        let mut buf = vec![0u8; Uint::<WL>::NUM_BYTES];
        for seed in &self.seeds {
            for entry in seed {
                entry.write_transcription_bytes_exact(&mut buf);
                hasher.update(&buf);
            }
        }
        hasher.update(salt);
        *hasher.finalize().as_bytes()
    }

    /// Checks every seed entry is a canonical residue (`< p`). Load-bearing
    /// for soundness: the extracted mask must be a well-defined function of
    /// the committed seed on *all* coordinates (paper Thm 4.2).
    pub fn validate(&self, dim: usize, modulus: &Uint<WL>) -> Result<(), ZipError> {
        for seed in &self.seeds {
            if seed.len() != dim {
                return Err(ZipError::InvalidPcsOpen(
                    "mask seed has wrong dimension".into(),
                ));
            }
            if seed.iter().any(|entry| entry >= modulus) {
                return Err(ZipError::InvalidPcsOpen(
                    "mask seed entry is not a canonical residue".into(),
                ));
            }
        }
        Ok(())
    }
}

/// Uniform value in `[0, 2^bits)`.
#[allow(clippy::arithmetic_side_effects)] // byte/bit index arithmetic bounded by construction
fn sample_window<const WL: usize>(rng: &mut impl CryptoRng, bits: u32) -> Uint<WL> {
    debug_assert!(usize::try_from(bits).expect("bits fits usize") < WL * 64);
    let full_bytes = usize::try_from(bits.div_ceil(8)).expect("bit count fits usize");
    let top_mask: u8 = match bits % 8 {
        0 => 0xff,
        r => u8::MAX >> (8 - r),
    };
    let mut buf = vec![0u8; mul!(WL, 8usize)];
    rng.fill_bytes(&mut buf[..full_bytes]);
    if let Some(last) = buf[..full_bytes].last_mut() {
        *last &= top_mask;
    }
    Uint::new(crypto_bigint::Uint::from_le_slice(&buf))
}

/// Returns the field configuration (`MontyParams`) for the mask field `F_p`.
pub fn mask_field_cfg<const WL: usize>(
    modulus: &Uint<WL>,
) -> Result<<MontyField<WL> as PrimeField>::Config, ZipError> {
    MontyField::<WL>::make_cfg(modulus)
        .map_err(|e| ZipError::InvalidPcsParam(format!("mask modulus rejected by field: {e:?}")))
}

/// Canonical mask symbol `[<G_l, s>]_p` at column `l` (evaluation point
/// `alpha = l + 1`), by Horner's rule over `F_p`.
#[allow(clippy::arithmetic_side_effects)] // field ops are modular (infallible)
pub fn mask_symbol_at<const WL: usize>(
    seed: &[Uint<WL>],
    column: usize,
    cfg: &<MontyField<WL> as PrimeField>::Config,
) -> Uint<WL> {
    let column_u64 = u64::try_from(column).expect("column index fits u64");
    let alpha = MontyField::<WL>::new_with_cfg(Uint::from_u64(add!(column_u64, 1u64)), cfg);
    let mut acc = MontyField::<WL>::zero_with_cfg(cfg);
    for coeff in seed.iter().rev() {
        acc = acc * alpha.clone() + MontyField::<WL>::new_with_cfg(*coeff, cfg);
    }
    acc.retrieve()
}

/// The full canonical mask vector `[G s]_p in [0, p)^codeword_len`.
///
/// Direct Horner per position: `O(n D)` field multiplications. Fine for the
/// current scope; an FFT-friendly `p` (or the Galois-ring limb variant of
/// the note, Remark 5.7) brings this to `O(n log n)` when it matters.
pub fn mask_row<const WL: usize>(
    seed: &[Uint<WL>],
    codeword_len: usize,
    cfg: &<MontyField<WL> as PrimeField>::Config,
) -> Vec<Uint<WL>> {
    (0..codeword_len)
        .map(|column| mask_symbol_at(seed, column, cfg))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    const WL: usize = 8;

    fn test_modulus() -> Uint<WL> {
        // Any odd prime works for these tests; reuse the parameter derivation
        // search starting from 2^200.
        let mut candidate = add!(crate::pcs::zk::params::pow2::<WL>(200), &Uint::from_u64(1));
        while !<zinc_primality::MillerRabin as zinc_primality::PrimalityTest<Uint<WL>>>::is_probably_prime(
            &candidate,
        ) {
            candidate = add!(candidate, &Uint::from_u64(2));
        }
        candidate
    }

    #[test]
    fn seeds_are_canonical_and_deterministic_per_rng() {
        let p = test_modulus();
        let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF);
        let seeds = MaskSeeds::<WL>::sample(&mut rng, 3, 16, &p);
        assert_eq!(seeds.seeds.len(), 3);
        seeds.validate(16, &p).expect("sampled seeds must validate");

        let mut rng2 = StdRng::seed_from_u64(0xDEAD_BEEF);
        let seeds2 = MaskSeeds::<WL>::sample(&mut rng2, 3, 16, &p);
        assert_eq!(seeds, seeds2, "same RNG seed must give same mask seeds");
    }

    #[test]
    fn mask_row_matches_pointwise_evaluation() {
        let p = test_modulus();
        let cfg = mask_field_cfg(&p).expect("cfg");
        let mut rng = StdRng::seed_from_u64(7);
        let seeds = MaskSeeds::<WL>::sample(&mut rng, 1, 8, &p);
        let row = mask_row(&seeds.seeds[0], 32, &cfg);
        assert_eq!(row.len(), 32);
        for (column, symbol) in row.iter().enumerate() {
            assert_eq!(*symbol, mask_symbol_at(&seeds.seeds[0], column, &cfg));
            assert!(symbol < &p, "mask symbols must be canonical residues");
        }
    }

    #[test]
    fn distinct_seeds_give_distinct_masks() {
        let p = test_modulus();
        let cfg = mask_field_cfg(&p).expect("cfg");
        let mut rng = StdRng::seed_from_u64(8);
        let seeds = MaskSeeds::<WL>::sample(&mut rng, 2, 8, &p);
        let row0 = mask_row(&seeds.seeds[0], 16, &cfg);
        let row1 = mask_row(&seeds.seeds[1], 16, &cfg);
        assert_ne!(row0, row1);
    }

    #[test]
    fn seed_commitment_binds_seeds_and_salt() {
        let p = test_modulus();
        let mut rng = StdRng::seed_from_u64(9);
        let seeds = MaskSeeds::<WL>::sample(&mut rng, 2, 8, &p);
        let salt = [7u8; 32];
        let comm = seeds.commitment(&salt);
        assert_ne!(comm, seeds.commitment(&[8u8; 32]));
        let mut other = seeds.clone();
        other.seeds[0][0] = add!(other.seeds[0][0], &Uint::from_u64(1));
        assert_ne!(comm, other.commitment(&salt));
    }
}
