//! Shared-challenge sampling across constraint families.
//!
//! With multiple field families active (one Q[X] family over a sampled prime
//! $q_0$, plus one F_q[X] family per declared prime $q_1, \ldots, q_n$),
//! every protocol-level challenge is sampled **once** as an integer in
//! $[0, q^*)$ — where $q^* := \min(q_0, q_1, \ldots, q_n)$ — and then
//! projected into each family's field via the family's config. Because each
//! shared integer is strictly less than every $q_i$, the per-family
//! projection is a type cast: all families see the same integer value, just
//! typed in different fields.
//!
//! The functions in this module are independent of the `MultiDegreeSumcheck`
//! / `IdealCheck` / `CombinedPolyResolver` / etc. interfaces: they produce
//! `Vec<F>` (one entry per family). The caller distributes the per-family
//! lifts to the appropriate sub-protocol invocations.

use crypto_primitives::BaseFieldConfig;
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
pub fn compute_q_star_idx<C>(prime_cfgs: &[C]) -> usize
where
    C: BaseFieldConfig,
{
    prime_cfgs
        .iter()
        .enumerate()
        .min_by_key(|(_, cfg)| cfg.modulus())
        .map(|(i, _)| i)
        .expect("prime_cfgs must be non-empty")
}

/// Sample one shared integer challenge in $[0, q^*)$ from the transcript and
/// return it lifted into each per-family field.
///
/// The returned `Vec<C::Element>` has length `prime_cfgs.len()`; entry $i$
/// is the same underlying integer typed in $F_{q_i}$ via that family's
/// config. Since the integer is `< q^* <= q_i`, no per-family modular
/// reduction occurs.
///
/// Internally this samples a wide [`C::Integer`] from the transcript and
/// reduces mod $q^*$ once. The reduction bias is negligible whenever
/// $q^* \ll \text{C::Integer}$ size (the typical case: a $\sim 2^{60}$-ish
/// prime versus a $\sim 2^{192}$-wide `C::Integer`).
#[inline]
pub fn sample_shared_field_challenge<C>(
    transcript: &mut impl Transcript,
    q_star_cfg: &C,
    prime_cfgs: &[C],
) -> Vec<C::Element>
where
    C: BaseFieldConfig,
    C::Integer: ConstTranscribable,
{
    let v = transcript.get_field_challenge(q_star_cfg);
    let v_int = q_star_cfg.lift(&v);
    prime_cfgs.iter().map(|cfg| cfg.project(&v_int)).collect()
}

/// Sample `n` shared integer challenges in $[0, q^*)$ and return the result
/// as a per-family matrix: outer length `prime_cfgs.len()`, inner length
/// `n`. Entry `[i][k]` is the $k$-th shared integer typed in
/// $F_{q_i}$.
#[inline]
pub fn sample_shared_field_challenges<C>(
    transcript: &mut impl Transcript,
    n: usize,
    q_star_cfg: &C,
    prime_cfgs: &[C],
) -> Vec<Vec<C::Element>>
where
    C: BaseFieldConfig,
    C::Integer: ConstTranscribable,
{
    // Sample shared integers first (one transcript draw per challenge),
    // then transpose into per-family vectors. This guarantees that each
    // family sees challenges in the same order they were squeezed.
    let shared: Vec<C::Integer> = (0..n)
        .map(|_| {
            let v = transcript.get_field_challenge(q_star_cfg);
            q_star_cfg.lift(&v)
        })
        .collect();

    prime_cfgs
        .iter()
        .map(|cfg| shared.iter().map(|v| cfg.project(v)).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_primitives::{
        LiftElementWithConfig,
        crypto_bigint_monty::MontyField,
        crypto_bigint_uint::{U64, Uint},
    };
    use zinc_transcript::Blake3Transcript;

    const FIELD_LIMBS: usize = U64::LIMBS * 3;
    type Cfg = MontyField<FIELD_LIMBS>;
    type FMod = Uint<FIELD_LIMBS>;

    fn cfg_from_u64(prime: u64) -> Cfg {
        Cfg::new(&FMod::from(prime)).expect("prime")
    }

    #[test]
    fn compute_q_star_picks_minimum() {
        // q_0 (random sampled prime) > q_1 > q_2: expect q_star == q_2.
        let cfgs = vec![
            cfg_from_u64(0xFFFF_FFFF_FFFF_FFC5),
            cfg_from_u64(0xFFFF_FFFF_FFFF_FFAD),
            cfg_from_u64(0xFFFF_FFFF_FFFF_F9C5),
        ];
        let idx = compute_q_star_idx(&cfgs);
        assert_eq!(idx, 2);
        let q_star = cfgs[idx].modulus();
        let expected = FMod::from(0xFFFF_FFFF_FFFF_F9C5_u64);
        assert_eq!(q_star, expected);
    }

    #[test]
    fn shared_challenge_lifts_same_integer_to_each_family() {
        let cfgs = vec![
            cfg_from_u64(0xFFFF_FFFF_FFFF_FFC5),
            cfg_from_u64(0xFFFF_FFFF_FFFF_FFAD),
        ];
        let q_star = &cfgs[compute_q_star_idx(&cfgs)];
        let mut transcript = Blake3Transcript::new();

        let per_family = sample_shared_field_challenge(&mut transcript, q_star, &cfgs);
        assert_eq!(per_family.len(), cfgs.len());

        // Every family must hold the same underlying *natural integer*
        // (modular reduction is the identity because the shared integer is
        // strictly less than every q_i). Compare via lifting to the integer
        // form so the Montgomery encoding (which differs per cfg) doesn't
        // get in the way.
        let int_0: FMod = cfgs[0].lift(&per_family[0]);
        for (i, fe) in per_family.iter().enumerate().skip(1) {
            let int_i: FMod = cfgs[i].lift(fe);
            assert_eq!(
                int_i, int_0,
                "family {i}: shared challenge differs from family 0",
            );
        }

        // Sanity: the shared integer is < q_star.
        assert!(
            int_0 < q_star.modulus(),
            "shared challenge must be < q_star"
        );
    }

    #[test]
    fn shared_challenges_batched_matches_repeated_singles() {
        let cfgs = vec![
            cfg_from_u64(0xFFFF_FFFF_FFFF_FFC5),
            cfg_from_u64(0xFFFF_FFFF_FFFF_FFAD),
        ];
        let q_star = &cfgs[compute_q_star_idx(&cfgs)];
        let n = 5;

        // Batched sampling.
        let mut t_batch = Blake3Transcript::new();
        let batched = sample_shared_field_challenges(&mut t_batch, n, q_star, &cfgs);
        assert_eq!(batched.len(), cfgs.len());
        for family in &batched {
            assert_eq!(family.len(), n);
        }

        // Same n single-element samples on a fresh transcript.
        let mut t_single = Blake3Transcript::new();
        let mut singles_per_family: Vec<Vec<_>> =
            cfgs.iter().map(|_| Vec::with_capacity(n)).collect();
        for _ in 0..n {
            let per_family = sample_shared_field_challenge(&mut t_single, q_star, &cfgs);
            for (i, fe) in per_family.into_iter().enumerate() {
                singles_per_family[i].push(fe);
            }
        }

        // Batched output must equal repeated singles, element-wise (compared
        // via natural integer form).
        for (i, (b, s)) in batched.iter().zip(&singles_per_family).enumerate() {
            for (k, (bk, sk)) in b.iter().zip(s).enumerate() {
                let bk_int: FMod = cfgs[i].lift(bk);
                let sk_int: FMod = cfgs[i].lift(sk);
                assert_eq!(
                    bk_int, sk_int,
                    "family {i}, idx {k}: batched != repeated singles",
                );
            }
        }
    }
}
