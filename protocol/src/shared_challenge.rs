//! Shared-challenge sampling across constraint families.
//!
//! With multiple field families active (one Q[X] family over a sampled prime
//! $q_0$, plus one F_q[X] family per declared prime $q_1, \ldots, q_n$),
//! every protocol-level challenge is sampled **once** as an integer in
//! $[0, q^*)$ — where $q^* := \min(q_0, q_1, \ldots, q_n)$ — and then lifted
//! into each family's field via [`F::new_with_cfg`]. Because each shared
//! integer is strictly less than every $q_i$, the per-family lift is a
//! type cast: all families see the same integer value, just typed in
//! different fields.
//!
//! The functions in this module are independent of the `MultiDegreeSumcheck`
//! / `IdealCheck` / `CombinedPolyResolver` / etc. interfaces: they produce
//! `Vec<F>` (one entry per family). The caller distributes the per-family
//! lifts to the appropriate sub-protocol invocations.

use crypto_primitives::PrimeField;
use zinc_transcript::traits::{ConstTranscribable, Transcript};

/// Compute the index of $q^* := \min$ in `prime_cfgs` (the family whose
/// modulus is smallest).
///
/// `prime_cfgs[0]` is $q_0$ ($Q[X]$ family); `prime_cfgs[1..=n]` are $q_i$
/// ($F_q[X]$ families).
///
/// # Panics
/// Panics if `prime_cfgs` is empty.
#[inline]
pub fn compute_q_star_idx<F>(prime_cfgs: &[F::Config]) -> usize
where
    F: PrimeField,
    F::Integer: Ord,
{
    prime_cfgs
        .iter()
        .enumerate()
        .min_by_key(|(_, cfg)| F::modulus(cfg))
        .map(|(i, _)| i)
        .expect("prime_cfgs must be non-empty")
}

/// Sample one shared integer challenge in $[0, q^*)$ from the transcript and
/// return it lifted into each per-family field.
///
/// The returned `Vec<F>` has length `prime_cfgs.len()`; entry $i$ is the
/// same underlying integer typed in $F_{q_i}$ via
/// [`F::new_with_cfg`]. Since the integer is `< q^* <= q_i`, no per-family
/// modular reduction occurs.
///
/// Internally this samples a wide [`F::Integer`] from the transcript and
/// reduces mod $q^*$ once. The reduction bias is negligible whenever
/// $q^* \ll \text{F::Integer}$ size (the typical case: a $\sim 2^{60}$-ish
/// prime versus a $\sim 2^{192}$-wide `F::Integer`).
#[inline]
pub fn sample_shared_field_challenge<F>(
    transcript: &mut impl Transcript,
    q_star_cfg: &F::Config,
    prime_cfgs: &[F::Config],
) -> Vec<F>
where
    F: PrimeField,
    F::Integer: ConstTranscribable,
{
    let v: F = transcript.get_field_challenge(q_star_cfg);
    let v_int = v.lift_to_integer();
    prime_cfgs
        .iter()
        .map(|cfg| F::from_with_cfg(v_int.clone(), cfg))
        .collect()
}

/// Sample `n` shared integer challenges in $[0, q^*)$ and return the result
/// as a per-family matrix: outer length `prime_cfgs.len()`, inner length
/// `n`. Entry `[i][k]` is the $k$-th shared integer typed in
/// $F_{q_i}$.
#[inline]
pub fn sample_shared_field_challenges<F>(
    transcript: &mut impl Transcript,
    n: usize,
    q_star_cfg: &F::Config,
    prime_cfgs: &[F::Config],
) -> Vec<Vec<F>>
where
    F: PrimeField,
    F::Integer: ConstTranscribable,
{
    // Sample shared integers first (one transcript draw per challenge),
    // then transpose into per-family vectors. This guarantees that each
    // family sees challenges in the same order they were squeezed.
    let shared: Vec<F::Integer> = (0..n)
        .map(|_| {
            transcript
                .get_field_challenge::<F>(q_star_cfg)
                .lift_to_integer()
        })
        .collect();

    prime_cfgs
        .iter()
        .map(|cfg| {
            shared
                .iter()
                .map(|v| F::from_with_cfg(v.clone(), cfg))
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_bigint::U64;
    use crypto_primitives::{
        Field, HasPrimeFieldConfig, crypto_bigint_monty::MontyField, crypto_bigint_uint::Uint,
    };
    use zinc_transcript::Blake3Transcript;
    use zinc_utils::from_ref::FromRef;

    const FIELD_LIMBS: usize = U64::LIMBS * 3;
    type F = MontyField<FIELD_LIMBS>;
    type FMod = Uint<FIELD_LIMBS>;

    fn cfg_from_u64(prime: u64) -> <F as HasPrimeFieldConfig>::Config {
        let fmod = FMod::from(prime);
        F::make_cfg(&FMod::from_ref(&fmod)).expect("prime")
    }

    #[test]
    fn compute_q_star_picks_minimum() {
        // q_0 (random sampled prime) > q_1 > q_2: expect q_star == q_2.
        let cfgs = vec![
            cfg_from_u64(0xFFFF_FFFF_FFFF_FFC5),
            cfg_from_u64(0xFFFF_FFFF_FFFF_FFAD),
            cfg_from_u64(0xFFFF_FFFF_FFFF_F9C5),
        ];
        let idx = compute_q_star_idx::<F>(&cfgs);
        assert_eq!(idx, 2);
        let q_star = F::modulus(&cfgs[idx]);
        let expected = FMod::from(0xFFFF_FFFF_FFFF_F9C5_u64);
        assert_eq!(q_star, expected);
    }

    #[test]
    fn shared_challenge_lifts_same_integer_to_each_family() {
        let cfgs = vec![
            cfg_from_u64(0xFFFF_FFFF_FFFF_FFC5),
            cfg_from_u64(0xFFFF_FFFF_FFFF_FFAD),
        ];
        let q_star = &cfgs[compute_q_star_idx::<F>(&cfgs)];
        let mut transcript = Blake3Transcript::new();

        let per_family = sample_shared_field_challenge::<F>(&mut transcript, q_star, &cfgs);
        assert_eq!(per_family.len(), cfgs.len());

        // Every family must hold the same underlying *natural integer*
        // (modular reduction is the identity because the shared integer is
        // strictly less than every q_i). Compare via `LiftToInteger` so the
        // Montgomery encoding (which differs per cfg) doesn't get in the way.
        let int_0: FMod = per_family[0].lift_to_integer();
        for (i, fe) in per_family.iter().enumerate().skip(1) {
            let int_i: FMod = fe.lift_to_integer();
            assert_eq!(
                int_i, int_0,
                "family {i}: shared challenge differs from family 0",
            );
        }

        // Sanity: the shared integer is < q_star.
        assert!(
            int_0 < F::modulus(q_star),
            "shared challenge must be < q_star"
        );
    }

    #[test]
    fn shared_challenges_batched_matches_repeated_singles() {
        let cfgs = vec![
            cfg_from_u64(0xFFFF_FFFF_FFFF_FFC5),
            cfg_from_u64(0xFFFF_FFFF_FFFF_FFAD),
        ];
        let q_star = &cfgs[compute_q_star_idx::<F>(&cfgs)];
        let n = 5;

        // Batched sampling.
        let mut t_batch = Blake3Transcript::new();
        let batched = sample_shared_field_challenges::<F>(&mut t_batch, n, q_star, &cfgs);
        assert_eq!(batched.len(), cfgs.len());
        for family in &batched {
            assert_eq!(family.len(), n);
        }

        // Same n single-element samples on a fresh transcript.
        let mut t_single = Blake3Transcript::new();
        let mut singles_per_family: Vec<Vec<F>> =
            cfgs.iter().map(|_| Vec::with_capacity(n)).collect();
        for _ in 0..n {
            let per_family = sample_shared_field_challenge::<F>(&mut t_single, q_star, &cfgs);
            for (i, fe) in per_family.into_iter().enumerate() {
                singles_per_family[i].push(fe);
            }
        }

        // Batched output must equal repeated singles, element-wise (compared
        // via natural integer form).
        for (i, (b, s)) in batched.iter().zip(&singles_per_family).enumerate() {
            for (k, (bk, sk)) in b.iter().zip(s).enumerate() {
                let bk_int: FMod = bk.lift_to_integer();
                let sk_int: FMod = sk.lift_to_integer();
                assert_eq!(
                    bk_int, sk_int,
                    "family {i}, idx {k}: batched != repeated singles",
                );
            }
        }
    }
}
