/// Optimized Montgomery multiplication.
///
/// Uses CIOS (Coarsely Integrated Operand Scanning) method which is more
/// cache-friendly than the FIOS method used in crypto-bigint.
/// Based on the implementation described in the Section 5 of the paper
/// "Analyzing and comparing Montgomery multiplication algorithms" with some
/// slight tweaks: https://www.microsoft.com/en-us/research/wp-content/uploads/1996/01/j37acmon.pdf
pub mod mul {
    use crypto_bigint::{Uint, WideWord, Word};
    use num_traits::ConstZero;

    const LOG2_WORD_BITS: u32 = Word::BITS.trailing_zeros();

    /// Compute modulus^-1 mod 2^word_bits (the negative of it, for Montgomery
    /// reduction). Uses Newton's method: x_{n+1} = x_n * (2 - m * x_n)
    #[inline(always)]
    #[allow(clippy::arithmetic_side_effects)]
    const fn compute_mod_neg_inv(m0: Word) -> Word {
        const TWO: Word = 2;

        let mut inv: Word = 1;
        let mut i = 0;
        // Newton iterations - converges in log2(word_bits) = 6 iterations for 64-bit
        while i < LOG2_WORD_BITS {
            inv = inv.wrapping_mul(TWO.wrapping_sub(m0.wrapping_mul(inv)));
            i += 1;
        }
        inv.wrapping_neg()
    }

    /// Montgomery multiplication: compute `a * b * R^-1 mod m` more efficiently
    /// than crypto-bigint does it.
    ///
    /// Uses CIOS (Coarsely Integrated Operand Scanning) method.
    #[inline(always)]
    pub fn monty_mul<const LIMBS: usize>(
        a: &Uint<LIMBS>,
        b: &Uint<LIMBS>,
        modulus: &Uint<LIMBS>,
    ) -> Uint<LIMBS> {
        let a_words = a.as_words();
        let b_words = b.as_words();
        let mod_words = modulus.as_words();
        let mod_neg_inv = compute_mod_neg_inv(mod_words[0]);

        let mut result = [0; LIMBS];
        let carry =
            montgomery_mul_cios::<LIMBS>(a_words, b_words, mod_words, mod_neg_inv, &mut result);

        // Conditional subtraction: subtract modulus if carry != 0 OR result >= modulus
        // First compute result - modulus
        let mut diff: [Word; _] = [0; LIMBS];
        let mut borrow: Word = 0;
        for i in 0..LIMBS {
            let (d, b1) = result[i].overflowing_sub(mod_words[i]);
            let (d, b2) = d.overflowing_sub(borrow);
            diff[i] = d;
            borrow = Word::from(b1) | Word::from(b2);
        }

        // Use diff if: carry != 0 (overflow) OR borrow == 0 (result >= modulus)
        // i.e., use result only if: carry == 0 AND borrow != 0
        let use_diff = (carry != 0) | (borrow == 0);
        let mask = Word::ZERO.wrapping_sub(Word::from(use_diff));
        for i in 0..LIMBS {
            result[i] = (diff[i] & mask) | (result[i] & !mask);
        }

        Uint::from_words(result)
    }

    /// CIOS (Coarsely Integrated Operand Scanning) Montgomery multiplication.
    ///
    /// For each limb of a, multiply by all of b, add to accumulator,
    /// then do Montgomery reduction step. Returns (result, carry) where
    /// carry indicates if an additional modulus subtraction is needed.
    ///
    /// Expects `out` to be initialized to zero.
    #[inline(always)]
    #[allow(clippy::arithmetic_side_effects)]
    fn montgomery_mul_cios<const LIMBS: usize>(
        a: &[Word; LIMBS],
        b: &[Word; LIMBS],
        modulus: &[Word; LIMBS],
        mod_neg_inv: Word,
        out: &mut [Word; LIMBS],
    ) -> Word {
        let mut acc_hi: Word = 0;

        for &a_i in a {
            // Step 1: acc += a_i * b
            let mut carry = 0;
            for j in 0..LIMBS {
                let (lo, hi) = mul_add_carry(a_i, b[j], out[j], carry);
                out[j] = lo;
                carry = hi;
            }
            let (new_acc_hi, meta_carry) = acc_hi.overflowing_add(carry);
            acc_hi = new_acc_hi;

            // Step 2: Montgomery reduction
            // u = acc[0] * mod_neg_inv mod 2^word_bits
            let u = out[0].wrapping_mul(mod_neg_inv);

            // acc += u * modulus, then shift right by one limb
            let (_, hi) = mul_add_carry(u, modulus[0], out[0], 0);
            carry = hi;

            for j in 1..LIMBS {
                let (lo, hi) = mul_add_carry(u, modulus[j], out[j], carry);
                out[j - 1] = lo;
                carry = hi;
            }

            let (sum, c) = acc_hi.overflowing_add(carry);
            out[LIMBS - 1] = sum;
            acc_hi = Word::from(meta_carry) + Word::from(c);
        }

        // Return carry - if non-zero, result >= 2^(word_bits*LIMBS) and needs reduction
        acc_hi
    }

    /// Compute a * b + c + d, returning (lo, hi) of WideWord result.
    #[inline(always)]
    #[allow(clippy::cast_possible_truncation, clippy::arithmetic_side_effects)]
    fn mul_add_carry(a: Word, b: Word, c: Word, d: Word) -> (Word, Word) {
        let wide = WideWord::from(a) * WideWord::from(b) + WideWord::from(c) + WideWord::from(d);
        (wide as Word, (wide >> Word::BITS) as Word)
    }
}
