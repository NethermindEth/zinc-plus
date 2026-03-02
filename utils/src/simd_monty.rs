//! Specialized Montgomery arithmetic for 128-bit primes in 192-bit containers.
//!
//! When the PIOP field uses a 128-bit prime stored in 3×64-bit limbs, the
//! standard CIOS Montgomery multiplication wastes work on the zero top limb.
//! This module provides routines that exploit the known-zero structure,
//! reducing the number of 64×64→128 multiplications from 18 to 10 per
//! Montgomery multiply (~44% reduction, ~1.8× speedup on the multiply core).
//!
//! On aarch64, the `u128` arithmetic compiles to optimal `MUL`+`UMULH` pairs,
//! and the Apple M-series out-of-order engine interleaves independent
//! multiplications automatically when processing batches in `fix_variables`.

/// Compute `−m₀⁻¹ mod 2⁶⁴` via Newton's method.
///
/// Used by Montgomery reduction: `u = t₀ · (−m⁻¹) mod 2⁶⁴`.
#[inline(always)]
#[allow(clippy::arithmetic_side_effects)]
pub(crate) const fn compute_mod_neg_inv(m0: u64) -> u64 {
    let mut inv: u64 = 1;
    let mut i = 0;
    // 6 Newton iterations for full 64-bit convergence
    while i < 6 {
        inv = inv.wrapping_mul(2u64.wrapping_sub(m0.wrapping_mul(inv)));
        i += 1;
    }
    inv.wrapping_neg()
}

/// `a·b + c + d` → `(lo, hi)` where `wide = a*b + c + d`.
#[inline(always)]
#[allow(clippy::arithmetic_side_effects, clippy::cast_possible_truncation)]
const fn mac(a: u64, b: u64, c: u64, d: u64) -> (u64, u64) {
    let wide = (a as u128) * (b as u128) + (c as u128) + (d as u128);
    (wide as u64, (wide >> 64) as u64)
}

/// Montgomery multiplication for 128-bit primes in 3×64-bit containers.
///
/// Computes `a · b · R⁻¹ mod m` where `R = 2¹⁹²`.
///
/// # Preconditions
///
/// - `modulus[2] == 0` (the prime fits in 128 bits)
/// - Inputs are in Montgomery form and reduced (`< modulus`)
///
/// Performs **10** full 64×64→128 multiplications (plus 3 single-word
/// `wrapping_mul` for the reduction factors), compared to the generic
/// CIOS's **18** multiplications for 3-limb operands.
#[inline(always)]
#[allow(clippy::arithmetic_side_effects, clippy::cast_possible_truncation)]
pub fn monty_mul_128in192(a: &[u64; 3], b: &[u64; 3], modulus: &[u64; 3]) -> [u64; 3] {
    debug_assert_eq!(modulus[2], 0, "modulus must fit in 128 bits");

    let (a0, a1) = (a[0], a[1]);
    let (b0, b1) = (b[0], b[1]);
    let (m0, m1) = (modulus[0], modulus[1]);
    let ninv = compute_mod_neg_inv(m0);

    let mut t0: u64;
    let mut t1: u64;
    let mut t2: u64;
    let mut acc_hi: u64;

    // ── Iteration 0: process a[0] ───────────────────────────────────
    //
    // Step 1: acc = a₀·b  (2 muls; b[2]=0 → only carry propagation)
    let (lo, hi) = mac(a0, b0, 0, 0);
    t0 = lo;
    let (lo, hi) = mac(a0, b1, 0, hi);
    t1 = lo;
    t2 = hi; // carry from a0*b1

    // Step 2: Montgomery reduction  (2 muls; m[2]=0)
    let u = t0.wrapping_mul(ninv);
    let (_zero, carry) = mac(u, m0, t0, 0); // low word is zero by construction
    let (lo, carry) = mac(u, m1, t1, carry);
    t0 = lo; // shift: t[1]→t[0]
    let w = (t2 as u128) + (carry as u128);
    t1 = w as u64; // shift: t[2]→t[1]
    acc_hi = (w >> 64) as u64;
    t2 = 0;

    // ── Iteration 1: process a[1] ───────────────────────────────────
    //
    // Step 1: acc += a₁·b  (2 muls)
    let (lo, carry) = mac(a1, b0, t0, 0);
    t0 = lo;
    let (lo, carry) = mac(a1, b1, t1, carry);
    t1 = lo;
    // b[2]=0: propagate carry into t2 and acc_hi
    let w = (t2 as u128) + (carry as u128);
    t2 = w as u64;
    let hi = (w >> 64) as u64;
    let (new_acc, mc) = acc_hi.overflowing_add(hi);
    acc_hi = new_acc;

    // Step 2: Montgomery reduction  (2 muls)
    let u = t0.wrapping_mul(ninv);
    let (_zero, carry) = mac(u, m0, t0, 0);
    let (lo, carry) = mac(u, m1, t1, carry);
    t0 = lo;
    let w = (t2 as u128) + (carry as u128);
    t1 = w as u64;
    let hi = (w >> 64) as u64;
    let (sum, c) = acc_hi.overflowing_add(hi);
    t2 = sum;
    acc_hi = (mc as u64) + (c as u64);

    // ── Iteration 2: a[2]=0 → Step 1 is no-op, only reduction ──────
    //
    // Step 2: Montgomery reduction  (2 muls)
    let u = t0.wrapping_mul(ninv);
    let (_zero, carry) = mac(u, m0, t0, 0);
    let (lo, carry) = mac(u, m1, t1, carry);
    t0 = lo;
    let w = (t2 as u128) + (carry as u128);
    t1 = w as u64;
    let hi = (w >> 64) as u64;
    let (sum, c) = acc_hi.overflowing_add(hi);
    t2 = sum;
    acc_hi = c as u64;

    // ── Conditional subtraction ─────────────────────────────────────
    conditional_sub([t0, t1, t2], modulus, acc_hi)
}

/// If `carry ≠ 0` or `result ≥ modulus`, subtract modulus. Constant-time.
#[inline(always)]
#[allow(clippy::arithmetic_side_effects)]
fn conditional_sub(result: [u64; 3], modulus: &[u64; 3], carry: u64) -> [u64; 3] {
    // Compute result − modulus
    let (d0, b0) = result[0].overflowing_sub(modulus[0]);
    let (d1, b1) = result[1].overflowing_sub(modulus[1]);
    let (d1, b2) = d1.overflowing_sub(b0 as u64);
    let borrow_01 = (b1 as u64) | (b2 as u64);
    let (d2, b3) = result[2].overflowing_sub(modulus[2]);
    let (d2, b4) = d2.overflowing_sub(borrow_01);
    let borrow = (b3 as u64) | (b4 as u64);

    // Use diff if: carry ≠ 0 (overflow) OR borrow = 0 (result ≥ modulus)
    let use_diff = (carry != 0) | (borrow == 0);
    let mask = 0u64.wrapping_sub(use_diff as u64);

    [
        (d0 & mask) | (result[0] & !mask),
        (d1 & mask) | (result[1] & !mask),
        (d2 & mask) | (result[2] & !mask),
    ]
}

/// Modular addition: `(a + b) mod m` for values < m < 2¹²⁸.
///
/// All inputs must be reduced (< modulus). Result is reduced.
#[inline(always)]
#[allow(clippy::arithmetic_side_effects)]
pub fn add_mod_128in192(a: &[u64; 3], b: &[u64; 3], modulus: &[u64; 3]) -> [u64; 3] {
    // a + b  (two-limb add with carry)
    let (s0, c0) = a[0].overflowing_add(b[0]);
    let (s1, c1) = a[1].overflowing_add(b[1]);
    let (s1, c2) = s1.overflowing_add(c0 as u64);
    let s2 = (c1 as u64) | (c2 as u64); // carry into limb 2 (≤ 1)

    // (s − modulus) for conditional subtraction
    let (d0, b0) = s0.overflowing_sub(modulus[0]);
    let (d1, b1) = s1.overflowing_sub(modulus[1]);
    let (d1, b2) = d1.overflowing_sub(b0 as u64);
    let borrow = (b1 as u64) | (b2 as u64);

    // Subtract if s ≥ modulus: either s2>0 (s ≥ 2¹²⁸ > m) or no borrow
    let need_sub = (s2 != 0) | (borrow == 0);
    let mask = 0u64.wrapping_sub(need_sub as u64);

    [
        (d0 & mask) | (s0 & !mask),
        (d1 & mask) | (s1 & !mask),
        0, // result always < modulus < 2¹²⁸
    ]
}

/// Modular subtraction: `(a − b) mod m` for values < m < 2¹²⁸.
///
/// All inputs must be reduced (< modulus). Result is reduced.
#[inline(always)]
#[allow(clippy::arithmetic_side_effects)]
pub fn sub_mod_128in192(a: &[u64; 3], b: &[u64; 3], modulus: &[u64; 3]) -> [u64; 3] {
    // a − b  (wrapping)
    let (d0, b0) = a[0].overflowing_sub(b[0]);
    let (d1, b1) = a[1].overflowing_sub(b[1]);
    let (d1, b2) = d1.overflowing_sub(b0 as u64);
    let underflow = b1 | b2; // true ⇒ a < b, need to add modulus

    // Conditionally add modulus
    let mask = 0u64.wrapping_sub(underflow as u64);
    let (r0, c0) = d0.overflowing_add(modulus[0] & mask);
    let r1 = d1.wrapping_add(modulus[1] & mask).wrapping_add(c0 as u64);

    [r0, r1, 0]
}

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects, clippy::cast_possible_truncation)]
mod tests {
    use super::*;

    fn u128_to_words(v: u128) -> [u64; 3] {
        [v as u64, (v >> 64) as u64, 0]
    }

    /// Test that `compute_mod_neg_inv` satisfies m₀ · (−inv) ≡ −1 (mod 2⁶⁴).
    #[test]
    fn test_mod_neg_inv() {
        for m0 in [1u64, 3, 0xFFFFFFFF, 0xE1B1BD1E39D54B33, 0xFFFFFFFFFFFFFF61] {
            let ninv = compute_mod_neg_inv(m0);
            let product = m0.wrapping_mul(ninv);
            // mod_neg_inv = −m⁻¹ mod 2⁶⁴, so m0 * ninv ≡ −1 (mod 2⁶⁴)
            // i.e. m0 * ninv + 1 ≡ 0 (mod 2⁶⁴)
            assert_eq!(product.wrapping_add(1), 0, "mod_neg_inv failed for m0={m0:#x}");
        }
    }

    /// Verify our specialized monty_mul against crypto-bigint's MontyForm.
    #[test]
    fn test_monty_mul_matches_reference() {
        use crypto_bigint::{
            Odd,
            modular::{MontyForm, MontyParams},
        };

        // 128-bit prime: p = 2¹²⁸ − 159
        let p_u128: u128 = u128::MAX - 158;
        let p_words = u128_to_words(p_u128);
        let p = crypto_bigint::Uint::<3>::from_words(p_words);
        let params = MontyParams::<3>::new(Odd::new(p).unwrap());

        let test_pairs: &[(u128, u128)] = &[
            (0, 0),
            (1, 1),
            (2, 3),
            (0, 12345),
            (p_u128 - 1, p_u128 - 1),
            (p_u128 - 1, 1),
            (0x123456789ABCDEF0, 0xFEDCBA9876543210),
            (p_u128 / 2, p_u128 / 3),
            (p_u128 - 2, p_u128 - 3),
        ];

        for &(a_val, b_val) in test_pairs {
            let a = crypto_bigint::Uint::<3>::from_words(u128_to_words(a_val));
            let b = crypto_bigint::Uint::<3>::from_words(u128_to_words(b_val));

            // Reference: crypto-bigint MontyForm
            let a_monty = MontyForm::new(&a, params);
            let b_monty = MontyForm::new(&b, params);
            let ref_result = MontyForm::mul(&a_monty, &b_monty);
            let expected = *ref_result.as_montgomery().as_words();

            // Our implementation operates on Montgomery representations
            let a_mont_words = *a_monty.as_montgomery().as_words();
            let b_mont_words = *b_monty.as_montgomery().as_words();

            let actual = monty_mul_128in192(&a_mont_words, &b_mont_words, &p_words);
            assert_eq!(actual, expected, "mismatch for a={a_val}, b={b_val}");
        }
    }

    /// Verify modular addition.
    #[test]
    fn test_add_mod() {
        let m_u128: u128 = u128::MAX - 158; // 2^128 - 159
        let m = u128_to_words(m_u128);
        let pm1 = u128_to_words(m_u128 - 1);

        // 0 + 0 = 0
        assert_eq!(add_mod_128in192(&[0, 0, 0], &[0, 0, 0], &m), [0, 0, 0]);

        // 1 + 1 = 2
        assert_eq!(add_mod_128in192(&[1, 0, 0], &[1, 0, 0], &m), [2, 0, 0]);

        // (p−1) + 1 = 0
        assert_eq!(add_mod_128in192(&pm1, &[1, 0, 0], &m), [0, 0, 0]);

        // (p−1) + (p−1) = p − 2
        let expected = u128_to_words(m_u128 - 2);
        assert_eq!(add_mod_128in192(&pm1, &pm1, &m), expected);
    }

    /// Verify modular subtraction.
    #[test]
    fn test_sub_mod() {
        let m_u128: u128 = u128::MAX - 158; // 2^128 - 159
        let m = u128_to_words(m_u128);
        let pm1 = u128_to_words(m_u128 - 1);

        // 0 − 0 = 0
        assert_eq!(sub_mod_128in192(&[0, 0, 0], &[0, 0, 0], &m), [0, 0, 0]);

        // 5 − 3 = 2
        assert_eq!(sub_mod_128in192(&[5, 0, 0], &[3, 0, 0], &m), [2, 0, 0]);

        // 0 − 1 = p − 1
        assert_eq!(sub_mod_128in192(&[0, 0, 0], &[1, 0, 0], &m), pm1);

        // 1 − (p−1) = 2
        assert_eq!(sub_mod_128in192(&[1, 0, 0], &pm1, &m), [2, 0, 0]);
    }

    /// Verify monty_mul with the actual Zinc+ PIOP prime.
    #[test]
    fn test_monty_mul_zinc_prime() {
        use crypto_bigint::{
            Odd,
            modular::{MontyForm, MontyParams},
        };

        // Zinc+ PIOP prime: 0x00860995AE68FC80E1B1BD1E39D54B33
        let m = [0xE1B1BD1E39D54B33u64, 0x00860995AE68FC80, 0];
        let p = crypto_bigint::Uint::<3>::from_words(m);
        let params = MontyParams::<3>::new(Odd::new(p).unwrap());

        for (a_val, b_val) in [(42u128, 99u128), (1 << 100, 1 << 90), (1, 0)] {
            let a = crypto_bigint::Uint::<3>::from_words(u128_to_words(a_val));
            let b = crypto_bigint::Uint::<3>::from_words(u128_to_words(b_val));
            let a_m = MontyForm::new(&a, params);
            let b_m = MontyForm::new(&b, params);
            let expected = *MontyForm::mul(&a_m, &b_m).as_montgomery().as_words();

            let actual = monty_mul_128in192(
                a_m.as_montgomery().as_words(),
                b_m.as_montgomery().as_words(),
                &m,
            );
            assert_eq!(actual, expected);
        }
    }
}
