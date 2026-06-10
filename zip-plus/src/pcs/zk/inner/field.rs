//! Inner-field tooling: FFT-friendly prime selection, modular
//! exponentiation, and a radix-2 NTT over the runtime-modulus
//! [`MontyField`].
//!
//! The inner prime has the form `p' = c * 2^tau + 1` with `tau` chosen large
//! enough for the Ligero domains of the next increment (message domain of
//! size `2^a` plus a disjoint coset codeword domain needs `tau >= a + 3`).
//! Both parties derive `p'` deterministically from the bit budget, so it is
//! a public parameter.

use crate::{ZipError, pcs::zk::params::pow2};
use crypto_primitives::{
    PrimeField, crypto_bigint_monty::MontyField, crypto_bigint_uint::Uint,
};
use zinc_primality::{MillerRabin, PrimalityTest};
use zinc_utils::{add, mul, sub};

/// Field configuration of the inner field `F_p'` with `IL` limbs.
pub type InnerCfg<const IL: usize> = <MontyField<IL> as PrimeField>::Config;

/// The inner prime together with its two-adic decomposition
/// `p' - 1 = cofactor * 2^two_adicity`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InnerFieldParams<const IL: usize> {
    pub modulus: Uint<IL>,
    pub cofactor: Uint<IL>,
    pub two_adicity: u32,
}

impl<const IL: usize> InnerFieldParams<IL> {
    /// Finds the smallest probable prime `p' >= 2^min_bits` of the form
    /// `c * 2^two_adicity + 1` (deterministic search).
    pub fn find(min_bits: u32, two_adicity: u32) -> Result<Self, ZipError> {
        let il_bits = u32::try_from(mul!(IL, 64usize))
            .map_err(|_| ZipError::InvalidPcsParam("IL overflow".into()))?;
        if add!(min_bits, 2u32) >= il_bits || two_adicity >= min_bits {
            return Err(ZipError::InvalidPcsParam(format!(
                "inner prime of {min_bits} bits with 2-adicity {two_adicity} \
                 does not fit Uint<{IL}>",
            )));
        }
        let one = Uint::<IL>::from_u64(1);
        let step = pow2::<IL>(two_adicity);
        // candidate_k = 2^min_bits + k * 2^two_adicity + 1  (== 1 mod 2^tau).
        let mut candidate = add!(pow2::<IL>(min_bits), &one, "prime search overflow");
        let mut steps: u64 = 0;
        while !MillerRabin::is_probably_prime(&candidate) {
            candidate = add!(candidate, &step, "prime search overflow");
            steps = add!(steps, 1u64, "prime search exhausted");
        }
        // cofactor = (candidate - 1) / 2^tau = 2^(min_bits - tau) + steps.
        let cofactor = add!(
            pow2::<IL>(sub!(min_bits, two_adicity)),
            &Uint::from_u64(steps),
            "cofactor overflow"
        );
        debug_assert_eq!(
            add!(mul!(cofactor, &step), &one),
            candidate,
            "two-adic decomposition mismatch"
        );
        Ok(Self {
            modulus: candidate,
            cofactor,
            two_adicity,
        })
    }

    /// Field configuration for `F_p'`.
    pub fn cfg(&self) -> Result<InnerCfg<IL>, ZipError> {
        MontyField::<IL>::make_cfg(&self.modulus).map_err(|e| {
            ZipError::InvalidPcsParam(format!("inner modulus rejected by field: {e:?}"))
        })
    }

    /// A quadratic non-residue `g` (so `g^((p'-1)/2) != 1`); used both as the
    /// generator of roots of unity and as the coset shift for disjoint
    /// Ligero domains.
    pub fn qnr(&self, cfg: &InnerCfg<IL>) -> Result<MontyField<IL>, ZipError> {
        // (p' - 1) / 2 = cofactor * 2^(tau - 1).
        let half_exp = mul!(
            self.cofactor,
            &pow2::<IL>(sub!(self.two_adicity, 1u32)),
            "exponent overflow"
        );
        let one = MontyField::<IL>::one_with_cfg(cfg);
        for g in 2u64..256 {
            let gf = MontyField::<IL>::new_with_cfg(Uint::from_u64(g), cfg);
            if pow(&gf, &half_exp) != one {
                return Ok(gf);
            }
        }
        Err(ZipError::InvalidPcsParam(
            "no quadratic non-residue below 256 (astronomically unlikely)".into(),
        ))
    }

    /// A root of unity of order exactly `2^order_log2`
    /// (requires `order_log2 <= two_adicity`).
    pub fn root_of_unity(
        &self,
        order_log2: u32,
        cfg: &InnerCfg<IL>,
    ) -> Result<MontyField<IL>, ZipError> {
        if order_log2 > self.two_adicity {
            return Err(ZipError::InvalidPcsParam(format!(
                "requested 2^{order_log2} root of unity exceeds 2-adicity {}",
                self.two_adicity,
            )));
        }
        let g = self.qnr(cfg)?;
        // omega = g^(cofactor * 2^(tau - order_log2)) has order exactly
        // 2^order_log2 since g^((p'-1)/2) != 1.
        let exp = mul!(
            self.cofactor,
            &pow2::<IL>(sub!(self.two_adicity, order_log2)),
            "exponent overflow"
        );
        Ok(pow(&g, &exp))
    }

    /// Multiplicative inverse via Fermat (`x^(p'-2)`).
    pub fn inverse(
        &self,
        value: &MontyField<IL>,
    ) -> Result<MontyField<IL>, ZipError> {
        let two = Uint::<IL>::from_u64(2);
        let exp = sub!(self.modulus, &two, "modulus too small");
        Ok(pow(value, &exp))
    }
}

/// `base^exp` by square-and-multiply (MSB first).
pub fn pow<const IL: usize>(base: &MontyField<IL>, exp: &Uint<IL>) -> MontyField<IL> {
    let cfg = *base.cfg();
    let mut acc = MontyField::<IL>::one_with_cfg(&cfg);
    let bits = exp.inner().bits();
    #[allow(clippy::arithmetic_side_effects)] // bit index arithmetic, bounded
    for k in (0..bits).rev() {
        acc = acc.clone() * acc;
        let word = exp.as_words()[(k / 64) as usize];
        if (word >> (k % 64)) & 1 == 1 {
            acc *= base.clone();
        }
    }
    acc
}

/// In-place radix-2 NTT: on return, `values[i] = poly(omega^i)` where `poly`
/// has the input as coefficients and `omega` has order exactly `values.len()`
/// (a power of two).
#[allow(clippy::arithmetic_side_effects)] // butterfly index arithmetic, bounded
pub fn ntt_in_place<const IL: usize>(values: &mut [MontyField<IL>], omega: &MontyField<IL>) {
    let n = values.len();
    assert!(n.is_power_of_two(), "NTT size must be a power of two");
    if n == 1 {
        return;
    }
    let log_n = n.ilog2();

    // Bit-reversal permutation.
    for i in 0..n {
        let j = (i.reverse_bits()) >> (usize::BITS - log_n);
        if i < j {
            values.swap(i, j);
        }
    }

    // Cooley-Tukey butterflies (decimation in time, natural-order output).
    let mut len = 2usize;
    while len <= n {
        let w_len = pow(omega, &Uint::from_u64((n / len) as u64));
        for chunk in values.chunks_mut(len) {
            let mut w = MontyField::<IL>::one_with_cfg(omega.cfg());
            let half = len / 2;
            for j in 0..half {
                let t = w.clone() * chunk[j + half].clone();
                let u = chunk[j].clone();
                chunk[j] = u.clone() + t.clone();
                chunk[j + half] = u - t;
                w *= w_len.clone();
            }
        }
        len <<= 1;
    }
}

/// Inverse NTT (interpolation): undoes [`ntt_in_place`].
#[allow(clippy::arithmetic_side_effects)]
pub fn intt_in_place<const IL: usize>(
    values: &mut [MontyField<IL>],
    omega: &MontyField<IL>,
    field: &InnerFieldParams<IL>,
) -> Result<(), ZipError> {
    let n = values.len();
    let omega_inv = field.inverse(omega)?;
    ntt_in_place(values, &omega_inv);
    let cfg = *omega.cfg();
    let n_f = MontyField::<IL>::new_with_cfg(Uint::from_u64(n as u64), &cfg);
    let n_inv = field.inverse(&n_f)?;
    for value in values.iter_mut() {
        *value = value.clone() * n_inv.clone();
    }
    Ok(())
}

/// Evaluates the polynomial with coefficients `coeffs` on the coset
/// `shift * <omega>` of size `domain_size` (a power of two; `omega` of that
/// exact order): returns `out[i] = poly(shift * omega^i)`.
#[allow(clippy::arithmetic_side_effects)]
pub fn coset_evaluate<const IL: usize>(
    coeffs: &[MontyField<IL>],
    shift: &MontyField<IL>,
    domain_size: usize,
    omega: &MontyField<IL>,
) -> Vec<MontyField<IL>> {
    assert!(domain_size.is_power_of_two());
    assert!(coeffs.len() <= domain_size, "polynomial too long for domain");
    let cfg = *shift.cfg();
    let zero = MontyField::<IL>::zero_with_cfg(&cfg);
    let mut out = vec![zero; domain_size];
    // Scale coefficient i by shift^i, then evaluate on <omega> via NTT.
    let mut power = MontyField::<IL>::one_with_cfg(&cfg);
    for (slot, coeff) in out.iter_mut().zip(coeffs) {
        *slot = coeff.clone() * power.clone();
        power *= shift.clone();
    }
    ntt_in_place(&mut out, omega);
    out
}

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects)]
mod tests {
    use super::*;

    const IL: usize = 4; // 256-bit test field

    fn test_field() -> InnerFieldParams<IL> {
        InnerFieldParams::<IL>::find(180, 20).expect("inner prime")
    }

    fn fe(v: u64, cfg: &InnerCfg<IL>) -> MontyField<IL> {
        MontyField::<IL>::new_with_cfg(Uint::from_u64(v), cfg)
    }

    /// Horner evaluation, the reference for the NTT tests.
    fn horner(coeffs: &[MontyField<IL>], x: &MontyField<IL>) -> MontyField<IL> {
        let cfg = *x.cfg();
        let mut acc = MontyField::<IL>::zero_with_cfg(&cfg);
        for c in coeffs.iter().rev() {
            acc = acc * x.clone() + c.clone();
        }
        acc
    }

    #[test]
    fn finds_prime_with_required_two_adicity() {
        let field = test_field();
        assert!(field.modulus.inner().bits() >= 180);
        assert!(MillerRabin::is_probably_prime(&field.modulus));
        // p' - 1 = cofactor * 2^20 exactly.
        let one = Uint::<IL>::from_u64(1);
        assert_eq!(
            add!(mul!(field.cofactor, &pow2::<IL>(20)), &one),
            field.modulus
        );
    }

    #[test]
    fn root_of_unity_has_exact_order() {
        let field = test_field();
        let cfg = field.cfg().expect("cfg");
        let omega = field.root_of_unity(3, &cfg).expect("omega");
        let one = MontyField::<IL>::one_with_cfg(&cfg);
        assert_eq!(pow(&omega, &Uint::from_u64(8)), one, "omega^8 must be 1");
        assert_ne!(pow(&omega, &Uint::from_u64(4)), one, "omega^4 must not be 1");
    }

    #[test]
    fn ntt_matches_horner_and_roundtrips() {
        let field = test_field();
        let cfg = field.cfg().expect("cfg");
        let omega = field.root_of_unity(3, &cfg).expect("omega");
        let coeffs: Vec<_> = (1..=8u64).map(|v| fe(v * v + 3, &cfg)).collect();

        let mut values = coeffs.clone();
        ntt_in_place(&mut values, &omega);
        let mut x = MontyField::<IL>::one_with_cfg(&cfg);
        for value in &values {
            assert_eq!(*value, horner(&coeffs, &x), "NTT must match evaluation");
            x *= omega.clone();
        }

        intt_in_place(&mut values, &omega, &field).expect("intt");
        assert_eq!(values, coeffs, "INTT must invert NTT");
    }

    #[test]
    fn coset_evaluation_matches_horner_and_misses_subgroup() {
        let field = test_field();
        let cfg = field.cfg().expect("cfg");
        let omega = field.root_of_unity(4, &cfg).expect("omega"); // order 16
        let shift = field.qnr(&cfg).expect("qnr");
        let coeffs: Vec<_> = (0..5u64).map(|v| fe(7 * v + 1, &cfg)).collect();

        let evals = coset_evaluate(&coeffs, &shift, 16, &omega);
        let mut point = shift.clone();
        for value in &evals {
            assert_eq!(*value, horner(&coeffs, &point));
            point *= omega.clone();
        }

        // Coset disjointness: shift * omega^i never lands in the subgroup
        // generated by omega (shift is a QNR, the subgroup has 2-power order
        // below the 2-adicity).
        let one = MontyField::<IL>::one_with_cfg(&cfg);
        let mut point = shift.clone();
        for _ in 0..16 {
            assert_ne!(pow(&point, &Uint::from_u64(16)), one, "coset hits subgroup");
            point *= omega.clone();
        }
    }
}
