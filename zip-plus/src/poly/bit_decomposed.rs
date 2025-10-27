use crate::{
    add, mul,
    pcs::structs::{MulByScalar, ProjectableToField},
    poly::{ConstCoeffBitWidth, EvaluatablePolynomial, EvaluationError, dense::DensePolynomial},
    traits::{ConstTranscribable, FromRef, Named},
};
use crypto_primitives::{FromWithConfig, PrimeField, Semiring, crypto_bigint_int::Int, Ring};
use num_traits::{CheckedAdd, CheckedMul, CheckedNeg, CheckedSub, One, Zero};
use rand::{distr::StandardUniform, prelude::*};
use std::{
    fmt::Display,
    hash::Hash,
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

/// * `BITS` is the maximum bit depth required, INCLUDING a sign bit.
/// * `DEGREE` cannot be larger than 63
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BitDecomposedPolynomial<const BITS: usize, const DEGREE: usize> {
    /// `slices[i]` holds the i-th bit (LSB at i=0) of all DEGREE+1 coefficients
    /// packed into a u64.
    /// The representation is Two's Complement. slices[K-1] is the sign bit slice.
    slices: [u64; BITS],
}

impl<const BITS: usize, const DEGREE: usize> BitDecomposedPolynomial<BITS, DEGREE> {
    /// Extract the value of a specific coefficient at the given index as a
    /// signed integer. This reconstructs the coefficient from its binary
    /// representation across slices.
    #[inline]
    fn extract_coefficient(&self, index: usize) -> i64 {
        debug_assert!(index <= DEGREE, "Coefficient index out of bounds");

        // 1. Reconstruct the K-bit value (stored temporarily as u64).
        let mut value: u64 = 0;

        // The compiler will unroll this loop as K is constant.
        // We iterate through the slices (bit positions).
        for k in 0..BITS {
            // Extract the bit corresponding to the 'index'-th coefficient from the k-th slice.
            let bit = (self.slices[k] >> index) & 1;
            // Place it into the k-th position of the reconstructed value.
            value |= bit << k;
        }

        // 2. Sign-extend the K-bit value to i64.
        sign_extend::<BITS>(value)
    }

    /// Helper function to efficiently invert the remaining slices once the carry is zero.
    #[inline(always)]
    fn invert_remaining_slices(input: &Self, output: &mut Self, start_index: usize) {
        let coefficient_mask: u64 = get_coefficient_mask(DEGREE);
        for j in start_index..BITS {
            // Invert and mask to ensure unused upper bits remain zero.
            output.slices[j] = (!input.slices[j]) & coefficient_mask;
        }
    }
}

impl<const BITS: usize, const DEGREE: usize> Neg for BitDecomposedPolynomial<BITS, DEGREE> {
    type Output = Self;

    /// Computes the Two's Complement negation (-self).
    /// Implements the (~self) + 1 algorithm efficiently in O(K).
    #[inline]
    fn neg(self) -> Self::Output {
        let mut result = Self::default();

        let coefficient_mask: u64 = get_coefficient_mask(DEGREE);

        // Initialize the carry to '1' for all coefficients simultaneously.
        // This implements the "+ 1" part of the algorithm.
        let mut carry: u64 = coefficient_mask;

        // The compiler will unroll this loop as K is constant.
        for i in 0..BITS {
            // 1. Invert the bits (the "~self" part).
            // We must apply the COEFFICIENT_MASK here. If NUM_COEFFS < 64, the inversion (!)
            // would set the unused upper bits to 1. The mask ensures they are treated as 0
            // for the arithmetic operation.
            let a_inv = (!self.slices[i]) & coefficient_mask;

            // 2. Specialized Ripple-Carry (Increment Logic):

            // Calculate Sum: S = A_inv XOR C_in
            result.slices[i] = a_inv ^ carry;

            // Calculate Carry-out: C_out = A_inv AND C_in
            // The carry propagates upwards only if the inverted value was 1.
            carry = a_inv & carry;

            // Optimization: If the carry has resolved to 0, the "+1" is complete.
            // The remaining slices just need to be inverted.
            if carry == 0 {
                Self::invert_remaining_slices(&self, &mut result, i + 1);
                break;
            }
        }

        // Note: This implementation follows standard Rust wrapping behavior.
        // Negating the minimum representable K-bit value results in itself.
        result
    }
}

impl<const BITS: usize, const DEGREE: usize> Add for BitDecomposedPolynomial<BITS, DEGREE> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects, clippy::op_ref)]
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        self + &rhs
    }
}

impl<'a, const BITS: usize, const DEGREE: usize> Add<&'a Self>
    for BitDecomposedPolynomial<BITS, DEGREE>
{
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn add(mut self, rhs: &'a Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<const BITS: usize, const DEGREE: usize> Sub for BitDecomposedPolynomial<BITS, DEGREE> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects, clippy::op_ref)]
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        self - &rhs
    }
}

impl<'a, const BITS: usize, const DEGREE: usize> Sub<&'a Self>
    for BitDecomposedPolynomial<BITS, DEGREE>
{
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn sub(mut self, rhs: &'a Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl<const BITS: usize, const DEGREE: usize> Mul for BitDecomposedPolynomial<BITS, DEGREE> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects, clippy::op_ref)]
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        self * &rhs
    }
}

impl<'a, const BITS: usize, const DEGREE: usize> Mul<&'a Self>
    for BitDecomposedPolynomial<BITS, DEGREE>
{
    type Output = Self;

    fn mul(self, _rhs: &'a Self) -> Self::Output {
        unimplemented!("Polynomial multiplication is not implemented")
    }
}

impl<const BITS: usize, const DEGREE: usize> AddAssign for BitDecomposedPolynomial<BITS, DEGREE> {
    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self += &rhs;
    }
}

impl<const BITS: usize, const DEGREE: usize> AddAssign<&Self>
    for BitDecomposedPolynomial<BITS, DEGREE>
{
    /// Efficient Polynomial Addition (Parallel Ripple-Carry Adder)
    /// This implementation works correctly for
    /// both Unsigned and Two's Complement Signed integers.
    #[allow(clippy::arithmetic_side_effects)]
    // #[inline(always)] // Critical for performance in tight loops // FIXME
    fn add_assign(&mut self, rhs: &Self) {
        let mut carry: u64 = 0;

        // The compiler will unroll this loop as K is constant.
        for i in 0..BITS {
            let a = self.slices[i];
            let b = rhs.slices[i];

            // Calculate Sum bit: S = A XOR B XOR C_in
            let a_xor_b = a ^ b;
            self.slices[i] = a_xor_b ^ carry;

            // Calculate Carry_out (Optimized Majority function):
            // C_out = (A & B) | (C_in & (A XOR B))
            // This correctly propagates the carry through the sign bit.
            carry = (a & b) | (carry & a_xor_b);
        }

        // Note: In Two's Complement, we rely on K being large enough to hold the result range.
        // We do not check the final carry bit for overflow detection.

        // In debug builds, check if K was large enough.
        // debug_assert!(
        //     carry == 0,
        //     "Coefficient overflow detected! K must be increased."
        // );
    }
}

impl<const BITS: usize, const DEGREE: usize> SubAssign for BitDecomposedPolynomial<BITS, DEGREE> {
    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self -= &rhs;
    }
}

impl<'a, const BITS: usize, const DEGREE: usize> SubAssign<&'a Self>
    for BitDecomposedPolynomial<BITS, DEGREE>
{
    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn sub_assign(&mut self, rhs: &'a Self) {
        // Not very efficient, but we don't use subtraction
        *self += -*rhs;
    }
}

impl<const BITS: usize, const DEGREE: usize> MulAssign for BitDecomposedPolynomial<BITS, DEGREE> {
    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self *= &rhs;
    }
}

impl<'a, const BITS: usize, const DEGREE: usize> MulAssign<&'a Self>
    for BitDecomposedPolynomial<BITS, DEGREE>
{
    fn mul_assign(&mut self, _rhs: &'a Self) {
        unimplemented!("Polynomial multiplication is not implemented")
    }
}

impl<const BITS: usize, const DEGREE: usize> Default for BitDecomposedPolynomial<BITS, DEGREE> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<const BITS: usize, const DEGREE: usize> Display for BitDecomposedPolynomial<BITS, DEGREE> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        // Extract and display the first coefficient
        let coeff_0 = self.extract_coefficient(0);
        write!(f, "{}", coeff_0)?;
        // Display remaining coefficients
        for i in 1..=DEGREE {
            let coeff = self.extract_coefficient(i);
            write!(f, ", {}", coeff)?;
        }
        write!(f, "]")?;
        Ok(())
    }
}

impl<const BITS: usize, const DEGREE: usize> Hash for BitDecomposedPolynomial<BITS, DEGREE> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for slice in self.slices.iter() {
            slice.hash(state);
        }
    }
}

impl<const BITS: usize, const DEGREE: usize> Zero for BitDecomposedPolynomial<BITS, DEGREE> {
    fn zero() -> Self {
        Self { slices: [0; BITS] }
    }

    fn is_zero(&self) -> bool {
        self.slices.iter().all(|&s| s == 0)
    }
}

impl<const BITS: usize, const DEGREE: usize> One for BitDecomposedPolynomial<BITS, DEGREE> {
    fn one() -> Self {
        let mut result = Self::zero();
        // Set the constant term (coefficient 0) to 1
        // Bit 0 of coefficient 0 is set, which is bit 0 of slice 0
        result.slices[0] = 1;
        result
    }
}

impl<const BITS: usize, const DEGREE: usize> CheckedNeg for BitDecomposedPolynomial<BITS, DEGREE> {
    fn checked_neg(&self) -> Option<Self> {
        let mut result = Self::default();

        let coefficient_mask: u64 = get_coefficient_mask(DEGREE);

        // Initialize carry to '1' for all coefficients (the "+ 1" part).
        let mut carry: u64 = coefficient_mask;

        // The compiler will unroll this loop.
        for i in 0..BITS {
            // Invert the input slice (~A) and mask to keep unused bits zero.
            let a_inv = (!self.slices[i]) & coefficient_mask;

            // Specialized Ripple-Carry (Increment Logic):
            // Sum: S = A_inv XOR C_in
            result.slices[i] = a_inv ^ carry;

            // Carry-out: C_out = A_inv AND C_in
            carry = a_inv & carry;

            // Optimization: If carry is 0, the "+1" is complete.
            if carry == 0 {
                Self::invert_remaining_slices(self, &mut result, i + 1);
                break;
            }
        }

        // 2. Check for overflow (O(1)).
        // Overflow occurs iff (Input is Negative) AND (Result is Negative).

        let sign_index = BITS - 1;
        let input_sign = self.slices[sign_index];
        let result_sign = result.slices[sign_index];

        // Create a mask where '1' indicates the corresponding coefficient overflowed.
        // Since input slices are correctly masked, we don't need to mask here again.
        let overflow_mask = input_sign & result_sign;

        if overflow_mask != 0 {
            None
        } else {
            Some(result)
        }
    }
}

impl<const BITS: usize, const DEGREE: usize> CheckedAdd for BitDecomposedPolynomial<BITS, DEGREE> {
    fn checked_add(&self, other: &Self) -> Option<Self> {
        // Ensure we have at least 1 bit (the sign bit) to perform meaningful overflow checks.
        if BITS == 0 {
            return Some(Self::zero());
        }

        // Initialize result. We will compute self + other.
        let mut result = *self;
        let mut carry: u64 = 0;

        // 1. Perform the Ripple-Carry Addition (O(K)).
        // The logic is identical for signed and unsigned arithmetic.
        // The compiler will unroll this loop as K is constant.
        for i in 0..BITS {
            // Read the i-th slice of A (from self, via result) and B (from other).
            let a = result.slices[i];
            let b = other.slices[i];

            // Calculate Sum: S = A XOR B XOR C_in
            let a_xor_b = a ^ b;
            result.slices[i] = a_xor_b ^ carry;

            // Calculate Carry_out (Majority function): C_out = (A&B) | (C_in & (A^B))
            carry = (a & b) | (carry & a_xor_b);
        }

        // 2. Detect Two's Complement Overflow (O(1)).
        // Note: We ignore the final 'carry' value here.

        // The index of the slice containing the sign bits.
        let sign_index = BITS - 1;

        // Extract the sign slices of the inputs (A and B) and the sum (S).
        // We must use the original inputs ('self' and 'other') to compare against the result.
        let a_sign = self.slices[sign_index];
        let b_sign = other.slices[sign_index];
        let s_sign = result.slices[sign_index];

        // Calculate the overflow mask using the optimized rule:
        // Overflow = (A_sign XOR S_sign) & (B_sign XOR S_sign)
        // This generates a mask where '1' indicates the corresponding coefficient overflowed.
        let overflow_mask = (a_sign ^ s_sign) & (b_sign ^ s_sign);

        // If any bit in the overflow mask is set, the operation failed.
        if overflow_mask != 0 {
            return None;
        }

        Some(result)
    }
}

impl<const BITS: usize, const DEGREE: usize> CheckedSub for BitDecomposedPolynomial<BITS, DEGREE> {
    fn checked_sub(&self, other: &Self) -> Option<Self> {
        // Not very efficient, but we don't use subtraction
        let neg_other = other.checked_neg()?;
        self.checked_add(&neg_other)
    }
}

impl<const BITS: usize, const DEGREE: usize> CheckedMul for BitDecomposedPolynomial<BITS, DEGREE> {
    fn checked_mul(&self, _other: &Self) -> Option<Self> {
        unimplemented!("Polynomial multiplication is not implemented")
    }
}

impl<const BITS: usize, const DEGREE: usize> Sum for BitDecomposedPolynomial<BITS, DEGREE> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| {
            acc.checked_add(&x).expect("Overflow in sum")
        })
    }
}

impl<'a, const BITS: usize, const DEGREE: usize> Sum<&'a Self>
    for BitDecomposedPolynomial<BITS, DEGREE>
{
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| {
            acc.checked_add(x).expect("Overflow in sum")
        })
    }
}

impl<const BITS: usize, const DEGREE: usize> Product for BitDecomposedPolynomial<BITS, DEGREE> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| {
            acc.checked_mul(&x).expect("Overflow in product")
        })
    }
}

impl<'a, const BITS: usize, const DEGREE: usize> Product<&'a Self>
    for BitDecomposedPolynomial<BITS, DEGREE>
{
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| {
            acc.checked_mul(x).expect("Overflow in product")
        })
    }
}

//
// BitDecomposedPolynomial => DensePolynomial<Int<LIMBS>, DEGREE>
//

impl<const LIMBS: usize, const BITS: usize, const DEGREE: usize>
    From<BitDecomposedPolynomial<BITS, DEGREE>> for DensePolynomial<Int<LIMBS>, DEGREE>
{
    fn from(bit_poly: BitDecomposedPolynomial<BITS, DEGREE>) -> Self {
        (&bit_poly).into()
    }
}

impl<const LIMBS: usize, const BITS: usize, const DEGREE: usize>
    From<&BitDecomposedPolynomial<BITS, DEGREE>> for DensePolynomial<Int<LIMBS>, DEGREE>
{
    fn from(bit_poly: &BitDecomposedPolynomial<BITS, DEGREE>) -> Self {
        DensePolynomial::from_ref(bit_poly)
    }
}

impl<const BITS: usize, const DEGREE: usize, const LIMBS: usize>
    FromRef<BitDecomposedPolynomial<BITS, DEGREE>> for DensePolynomial<Int<LIMBS>, DEGREE>
{
    #[allow(clippy::arithmetic_side_effects)]
    fn from_ref(bit_poly: &BitDecomposedPolynomial<BITS, DEGREE>) -> Self {
        let int_bits =
            usize::try_from(Int::<LIMBS>::BITS).expect("Int<LIMBS>::BITS must fit in usize");
        assert!(
            BITS <= int_bits,
            "Cannot convert BitDecomposedPolynomial with BITS > Int<LIMBS>::BITS"
        );
        // TODO: Make more efficient
        let i64_poly = DensePolynomial::<i64, DEGREE>::from(bit_poly);
        let coeffs = i64_poly
            .to_coeffs()
            .iter()
            .map(|&coeff| Int::<LIMBS>::from(&coeff))
            .collect::<Vec<_>>();
        DensePolynomial::new(&coeffs)
    }
}

//
// BitDecomposedPolynomial <=> DensePolynomial<i64, DEGREE>
//

impl<const BITS: usize, const DEGREE: usize> From<BitDecomposedPolynomial<BITS, DEGREE>>
    for DensePolynomial<i64, DEGREE>
{
    fn from(bit_poly: BitDecomposedPolynomial<BITS, DEGREE>) -> Self {
        (&bit_poly).into()
    }
}

impl<const BITS: usize, const DEGREE: usize> From<&BitDecomposedPolynomial<BITS, DEGREE>>
    for DensePolynomial<i64, DEGREE>
{
    #[allow(clippy::arithmetic_side_effects)]
    fn from(bit_poly: &BitDecomposedPolynomial<BITS, DEGREE>) -> Self {
        // let int_bits =
        //     usize::try_from(Int::<LIMBS>::BITS).expect("Int<LIMBS>::BITS must fit in usize");
        // assert!(
        //     BITS <= int_bits,
        //     "Cannot convert BitDecomposedPolynomial with BITS > Int<LIMBS>::BITS"
        // );
        let mut coeffs = Vec::with_capacity(DEGREE + 1);

        // Iterate over all DEGREE + 1 coefficients.
        for i in 0..=DEGREE {
            let mut reconstructed_val: u64 = 0;

            // Reconstruct the coefficient from the slices.
            for k in 0..BITS {
                // Extract the i-th bit from the k-th slice.
                let bit = (bit_poly.slices[k] >> i) & 1;
                // Place the bit into the coefficient value.
                reconstructed_val |= bit << k;
            }

            // Interpret the K-bit value as Two's Complement (Sign Extension).
            coeffs.push(sign_extend::<BITS>(reconstructed_val));
        }

        DensePolynomial::new(&coeffs)
    }
}

impl<const BITS: usize, const DEGREE: usize> From<DensePolynomial<i64, DEGREE>>
    for BitDecomposedPolynomial<BITS, DEGREE>
{
    fn from(dense: DensePolynomial<i64, DEGREE>) -> Self {
        (&dense).into()
    }
}

impl<const BITS: usize, const DEGREE: usize> From<&DensePolynomial<i64, DEGREE>>
    for BitDecomposedPolynomial<BITS, DEGREE>
{
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn from(dense: &DensePolynomial<i64, DEGREE>) -> Self {
        let mut result = BitDecomposedPolynomial::zero();

        // Iterate over the bit depth K.
        for k in 0..BITS {
            let mut slice_val: u64 = 0;
            // Iterate over the 32 coefficients.
            for (i, &coeff_val) in dense.to_coeffs().iter().enumerate() {
                // Extract the k-th bit. Casting i64 to u64 preserves the Two's Complement pattern.
                let unsigned_coeff_val = coeff_val as u64;

                // Optional: Validate input range in debug builds if K < 64.
                if BITS < 64 && k == 0 {
                     // Check if the value fits within the K-bit signed range.
                     // The upper 64-K bits must match the sign bit (K-1).
                     let shift = 64 - BITS;
                     let sign_extended = ((unsigned_coeff_val << shift) as i64 >> shift) as u64;
                     debug_assert_eq!(unsigned_coeff_val, sign_extended,
                                      "Input coefficient magnitude too large for K-bit Two's Complement");
                }

                let bit = (unsigned_coeff_val >> k) & 1;

                // Place the bit into the i-th position of the slice.
                slice_val |= bit << i;
            }
            result.slices[k] = slice_val;
        }
        result
    }
}

//
// BitDecomposedPolynomial <=> BitDecomposedPolynomial
//

impl<const BITS: usize, const BITS_2: usize, const DEGREE: usize>
    FromRef<BitDecomposedPolynomial<BITS_2, DEGREE>> for BitDecomposedPolynomial<BITS, DEGREE>
{
    fn from_ref(bit_poly: &BitDecomposedPolynomial<BITS_2, DEGREE>) -> Self {
        // Ensure that BITS is greater than or equal to BITS_2
        assert!(BITS >= BITS_2, "Cannot convert to smaller bit depth");

        let mut result = Self::zero();
        
        // Copy the lower slices
        for i in 0..BITS_2 {
            result.slices[i] = bit_poly.slices[i];
        }
        
        // Perform sign extension for the remaining slices
        // The sign bit is at index BITS_2 - 1
        let sign_slice = bit_poly.slices[BITS_2 - 1];
        
        // For each coefficient, if its sign bit is set, we need to set all higher bits
        for i in BITS_2..BITS {
            result.slices[i] = sign_slice;
        }
        
        result
    }
}

//
// Ring
//

impl<const BITS: usize, const DEGREE: usize> Semiring for BitDecomposedPolynomial<BITS, DEGREE> {}

impl<const BITS: usize, const DEGREE: usize> Ring for BitDecomposedPolynomial<BITS, DEGREE> {}

//
// RNG
//

impl<const BITS: usize, const DEGREE: usize> Distribution<BitDecomposedPolynomial<BITS, DEGREE>>
    for StandardUniform
{
    #[allow(clippy::arithmetic_side_effects)]
    fn sample<Gen: Rng + ?Sized>(&self, rng: &mut Gen) -> BitDecomposedPolynomial<BITS, DEGREE> {
        // For DEGREE=31, this evaluates to 0xFFFFFFFF.
        // For DEGREE=63, this evaluates to u64::MAX.
        let coefficient_mask: u64 = get_coefficient_mask(DEGREE);

        let mut slices = [0u64; BITS];

        // Iterate over all K slices (the bit depth).
        // By generating random bits for all K slices (including the sign slice),
        // we achieve a uniform distribution across the signed range.
        // The compiler will unroll this loop as K is constant.
        for slice in slices.iter_mut() {
            // 1. Generate a full random u64.
            // rng.gen() is the idiomatic way to generate standard types.
            let random_val: u64 = rng.random();

            // 2. Apply the mask. This ensures that bits outside the coefficient packing
            //    (e.g., bits 32 through 63 if NUM_COEFFS=32) are zeroed out.
            *slice = random_val & coefficient_mask;
        }

        BitDecomposedPolynomial { slices }
    }
}

//
// Zip-specific traits
//

impl<const BITS: usize, const DEGREE: usize> Named for BitDecomposedPolynomial<BITS, DEGREE> {
    fn type_name() -> String {
        format!("BPoly<{BITS}b, {DEGREE}>")
    }
}

impl<const BITS: usize, const DEGREE: usize> ConstCoeffBitWidth
    for BitDecomposedPolynomial<BITS, DEGREE>
{
    const COEFF_BIT_WIDTH: usize = BITS;
}

impl<const BITS: usize, const DEGREE: usize> ConstTranscribable
    for BitDecomposedPolynomial<BITS, DEGREE>
{
    const NUM_BYTES: usize = Self::NUM_BITS.div_ceil(8);
    const NUM_BITS: usize = u64::BITS as usize * BITS;

    fn read_transcription_bytes(bytes: &[u8]) -> Self {
        assert_eq!(
            bytes.len(),
            Self::NUM_BYTES,
            "Buffer size mismatch for transcription"
        );

        let mut slices = [0u64; BITS];
        for (i, slice) in slices.iter_mut().enumerate() {
            let start = mul!(i, 8);
            let end = add!(start, 8);
            *slice = u64::read_transcription_bytes(&bytes[start..end]);
        }

        BitDecomposedPolynomial { slices }
    }

    fn write_transcription_bytes(&self, buf: &mut [u8]) {
        assert_eq!(
            buf.len(),
            Self::NUM_BYTES,
            "Buffer size mismatch for transcription"
        );
        for (i, &slice) in self.slices.iter().enumerate() {
            let start = mul!(i, 8);
            let end = add!(start, 8);
            slice.write_transcription_bytes(&mut buf[start..end]);
        }
    }
}

impl<F, const BITS: usize, const DEGREE: usize> ProjectableToField<F>
    for BitDecomposedPolynomial<BITS, DEGREE>
where
    F: PrimeField + for<'a> FromWithConfig<&'a i64> + for<'a> MulByScalar<&'a F> + 'static,
{
    fn prepare_projection(sampled_value: &F) -> impl Fn(&Self) -> F + 'static {
        // TODO: Make more efficient
        let dense_proj = DensePolynomial::<i64, DEGREE>::prepare_projection(sampled_value);
        move |poly: &BitDecomposedPolynomial<BITS, DEGREE>| dense_proj(&poly.into())
    }
}

impl<const BITS: usize, const DEGREE: usize> EvaluatablePolynomial<i128, Int<5>>
    for BitDecomposedPolynomial<BITS, DEGREE>
{
    fn evaluate_at_point(&self, _point: &[i128]) -> Result<Int<5>, EvaluationError> {
        todo!()
    }
}

//
// Helpers
//

/// Helper to correctly sign-extend a K-bit value stored in a u64 to an i64.
#[inline(always)]
fn sign_extend<const BITS: usize>(val: u64) -> i64 {
    if BITS == 64 {
        // If K is the full width, a simple cast interprets the Two's Complement correctly.
        val as i64
    } else {
        // If K < 64, we must manually sign-extend.
        let shift = 64 - BITS;
        // 1. Left shift to move the sign bit (at K-1) to the MSB (63).
        // 2. Cast to i64.
        // 3. Arithmetic right shift (>>) fills the upper bits with the sign bit.
        ((val as i64) << shift) >> shift
    }
}

/// We use u128 arithmetic to robustly handle the calculation even if DEGREE=63.
/// Calculation: (2^(DEGREE + 1)) - 1.
const fn get_coefficient_mask(degree: usize) -> u64 {
    ((1_u128 << (degree + 1)) - 1) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::{One, Zero};

    const TEST_BITS: usize = 28;
    const TEST_DEGREE: usize = 9;

    /// 2^27 - 1 = 134217727
    const MAX_VAL: i64 = (1_i64 << (TEST_BITS - 1)) - 1;

    /// -2^27 = -134217728
    const MIN_VAL: i64 = -(1i64 << (TEST_BITS - 1));

    type TestPoly = BitDecomposedPolynomial<TEST_BITS, TEST_DEGREE>;

    /// Helper to create a bit decomposed polynomial from u32 coefficients
    fn make_poly(coeffs: &[i64]) -> TestPoly {
        DensePolynomial::<_, TEST_DEGREE>::new(coeffs).into()
    }

    #[test]
    fn test_zero_polynomial() {
        let zero = TestPoly::zero();
        assert!(zero.is_zero());

        // All coefficients should be zero
        for i in 0..=TEST_DEGREE {
            assert_eq!(zero.extract_coefficient(i), 0);
        }
    }

    #[test]
    fn test_one_polynomial() {
        let one = TestPoly::one();
        assert!(!one.is_zero());

        // Only coefficient 0 should be 1
        assert_eq!(one.extract_coefficient(0), 1);
        for i in 1..=TEST_DEGREE {
            assert_eq!(one.extract_coefficient(i), 0);
        }
    }

    #[test]
    fn test_addition_basic() {
        // 1 + 2x + 3x^2
        let poly1 = make_poly(&[1, 2, 3]);
        // 4 + 5x + 6x^2
        let poly2 = make_poly(&[4, 5, 6]);

        let sum = poly1 + poly2;

        // Should be 5 + 7x + 9x^2
        assert_eq!(sum.extract_coefficient(0), 5);
        assert_eq!(sum.extract_coefficient(1), 7);
        assert_eq!(sum.extract_coefficient(2), 9);
    }

    #[test]
    fn test_addition_with_carry() {
        // Test addition that requires bit carries across coefficient boundaries
        // Use values that test carry propagation without overflowing K bits

        // Create polynomials where lower bits will produce carries
        // Example: coefficient has pattern that will generate carries between bit
        // positions
        let poly1 = make_poly(&[1000000, 1]); // 1000000 + x
        let poly2 = make_poly(&[2000000, 0]); // 2000000

        let sum = poly1 + poly2;

        // 1000000 + 2000000 = 3000000
        // This tests the ripple-carry adder working correctly
        assert_eq!(sum.extract_coefficient(0), 3000000);
        assert_eq!(sum.extract_coefficient(1), 1);

        // Test another carry scenario: adding 1 to a value with all lower bits set
        let poly3 = make_poly(&[255, 0]); // 255
        let poly4 = make_poly(&[1, 0]); // 1

        let sum2 = poly3 + poly4;
        assert_eq!(sum2.extract_coefficient(0), 256); // 256
    }

    #[test]
    fn test_addition_zero_identity() {
        let poly = make_poly(&[1, 2, 3, 4, 5]);
        let zero = TestPoly::zero();

        let sum1 = poly + zero;
        let sum2 = zero + poly;

        assert_eq!(sum1, poly);
        assert_eq!(sum2, poly);
    }

    #[test]
    fn test_addition_commutative() {
        let poly1 = make_poly(&[1, 2, 3]);
        let poly2 = make_poly(&[4, 5, 6]);

        assert_eq!(poly1 + poly2, poly2 + poly1);
    }

    #[test]
    fn test_checked_add_success() {
        let poly1 = make_poly(&[1, 2, 3]);
        let poly2 = make_poly(&[4, 5, 6]);

        let sum = poly1.checked_add(&poly2);
        assert!(sum.is_some());

        let sum = sum.unwrap();
        assert_eq!(sum.extract_coefficient(0), 5);
        assert_eq!(sum.extract_coefficient(1), 7);
        assert_eq!(sum.extract_coefficient(2), 9);
    }

    #[test]
    fn test_checked_add_overflow() {
        // Create two polynomials that will overflow when added
        let poly1 = make_poly(&[MAX_VAL; 10]);
        let poly2 = make_poly(&[1; 10]);

        let sum = poly1.checked_add(&poly2);
        // Should return None due to overflow
        assert!(sum.is_none());
    }

    #[test]
    fn test_sum_iterator() {
        let polys = vec![
            make_poly(&[1, 0, 0]),
            make_poly(&[0, 2, 0]),
            make_poly(&[0, 0, 3]),
        ];

        let sum: TestPoly = polys.into_iter().sum();

        assert_eq!(sum.extract_coefficient(0), 1);
        assert_eq!(sum.extract_coefficient(1), 2);
        assert_eq!(sum.extract_coefficient(2), 3);
    }

    #[test]
    fn test_sum_iterator_refs() {
        let polys = vec![
            make_poly(&[1, 0, 0]),
            make_poly(&[0, 2, 0]),
            make_poly(&[0, 0, 3]),
        ];

        let sum: TestPoly = polys.iter().sum();

        assert_eq!(sum.extract_coefficient(0), 1);
        assert_eq!(sum.extract_coefficient(1), 2);
        assert_eq!(sum.extract_coefficient(2), 3);
    }

    #[test]
    fn test_display() {
        let poly = make_poly(&[1, 2, 3]);
        let s = format!("{}", poly);

        // Should start with "[1, 2, 3" and have 10 total coefficients
        assert!(s.starts_with("[1, 2, 3"));
        assert!(s.ends_with("]"));
    }

    #[test]
    fn test_equality() {
        let poly1 = make_poly(&[1, 2, 3, 4, 5]);
        let poly2 = make_poly(&[1, 2, 3, 4, 5]);
        let poly3 = make_poly(&[1, 2, 3, 4, 6]);

        assert_eq!(poly1, poly2);
        assert_ne!(poly1, poly3);
    }

    #[test]
    fn test_hash_consistency() {
        use std::{
            collections::hash_map::DefaultHasher,
            hash::{Hash, Hasher},
        };

        let poly1 = make_poly(&[1, 2, 3]);
        let poly2 = make_poly(&[1, 2, 3]);

        let mut hasher1 = DefaultHasher::new();
        poly1.hash(&mut hasher1);
        let hash1 = hasher1.finish();

        let mut hasher2 = DefaultHasher::new();
        poly2.hash(&mut hasher2);
        let hash2 = hasher2.finish();

        // Equal values should have equal hashes
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_clone_and_copy() {
        let poly = make_poly(&[1, 2, 3]);
        // Test that Copy trait works
        let copied = poly;
        // Can still use poly after copy
        assert_eq!(poly, copied);

        // Test Debug derive
        let debug_str = format!("{:?}", poly);
        assert!(debug_str.contains("BitDecomposedPolynomial"));
    }

    #[test]
    fn test_default() {
        let poly = TestPoly::default();
        assert!(poly.is_zero());
    }

    #[test]
    fn test_extract_coefficient_signed() {
        // Test positive values
        let poly = make_poly(&[100, 200, 300]);
        assert_eq!(poly.extract_coefficient(0), 100);
        assert_eq!(poly.extract_coefficient(1), 200);
        assert_eq!(poly.extract_coefficient(2), 300);

        // Test negative values
        let poly2 = make_poly(&[-100, -200, -300]);
        assert_eq!(poly2.extract_coefficient(0), -100);
        assert_eq!(poly2.extract_coefficient(1), -200);
        assert_eq!(poly2.extract_coefficient(2), -300);

        // Test value that is negative in two's complement (28-bit MIN)
        let poly3 = make_poly(&[MIN_VAL]);
        let signed = poly3.extract_coefficient(0);
        assert_eq!(signed, MIN_VAL);
    }

    #[test]
    fn test_conversion_to_dense() {
        let bit_poly = make_poly(&[1, 2, 3, 4, 5]);
        let dense: DensePolynomial<i64, TEST_DEGREE> = bit_poly.into();

        let coeffs = dense.to_coeffs();
        assert_eq!(coeffs[0], 1);
        assert_eq!(coeffs[1], 2);
        assert_eq!(coeffs[2], 3);
        assert_eq!(coeffs[3], 4);
        assert_eq!(coeffs[4], 5);
    }

    #[test]
    fn test_conversion_from_dense() {
        let coeffs = vec![1, 2, 3, 4, 5];
        let dense = DensePolynomial::<i64, TEST_DEGREE>::new(coeffs);

        let bit_poly: TestPoly = (&dense).into();

        assert_eq!(bit_poly.extract_coefficient(0), 1);
        assert_eq!(bit_poly.extract_coefficient(1), 2);
        assert_eq!(bit_poly.extract_coefficient(2), 3);
        assert_eq!(bit_poly.extract_coefficient(3), 4);
        assert_eq!(bit_poly.extract_coefficient(4), 5);
    }

    #[test]
    fn test_conversion_roundtrip() {
        let original = make_poly(&[1, 2, 3, 4, 5, 6, 7, 8]);

        // Convert to dense and back
        let dense: DensePolynomial<i64, TEST_DEGREE> = original.into();
        let roundtrip: TestPoly = (&dense).into();

        assert_eq!(original, roundtrip);
    }

    #[test]
    fn test_conversion_zero() {
        let zero = TestPoly::zero();
        let dense: DensePolynomial<i64, TEST_DEGREE> = zero.into();
        let back: TestPoly = (&dense).into();

        assert!(back.is_zero());
    }

    #[test]
    fn test_conversion_one() {
        let one = TestPoly::one();
        let dense: DensePolynomial<i64, TEST_DEGREE> = one.into();
        let back: TestPoly = (&dense).into();

        assert_eq!(back, one);
    }

    #[test]
    fn test_conversion_max_values() {
        // Test with maximum and minimum signed 28-bit values
        let poly = make_poly(&[MAX_VAL, MIN_VAL, MAX_VAL]);
        let dense: DensePolynomial<i64, TEST_DEGREE> = poly.into();
        let back: TestPoly = (&dense).into();

        assert_eq!(back, poly);
    }

    #[test]
    fn test_add_assign() {
        let mut poly1 = make_poly(&[1, 2, 3]);
        let poly2 = make_poly(&[4, 5, 6]);

        poly1 += poly2;

        assert_eq!(poly1.extract_coefficient(0), 5);
        assert_eq!(poly1.extract_coefficient(1), 7);
        assert_eq!(poly1.extract_coefficient(2), 9);
    }

    #[test]
    fn test_add_assign_ref() {
        let mut poly1 = make_poly(&[1, 2, 3]);
        let poly2 = make_poly(&[4, 5, 6]);

        poly1 += &poly2;

        assert_eq!(poly1.extract_coefficient(0), 5);
        assert_eq!(poly1.extract_coefficient(1), 7);
        assert_eq!(poly1.extract_coefficient(2), 9);
    }

    #[test]
    fn test_large_coefficients() {
        // Test with coefficients near the 28-bit signed limit
        // For 28-bit signed integers, the range is -2^27 to 2^27-1
        let large_positive = MAX_VAL;
        let large_negative = MIN_VAL;
        let poly = make_poly(&[large_positive, large_negative, large_positive]);

        assert_eq!(poly.extract_coefficient(0), large_positive);
        assert_eq!(poly.extract_coefficient(1), large_negative);
        assert_eq!(poly.extract_coefficient(2), large_positive);
    }

    #[test]
    fn test_all_coefficients_set() {
        // Test setting all coefficients
        let mut coeffs = Vec::new();
        #[allow(clippy::cast_possible_truncation)]
        for i in 0..=TEST_DEGREE {
            coeffs.push(i as i64);
        }

        let poly = make_poly(&coeffs);

        #[allow(clippy::cast_possible_truncation)]
        for i in 0..=TEST_DEGREE {
            assert_eq!(poly.extract_coefficient(i), i as i64);
        }
    }

    #[test]
    fn test_sparse_polynomial() {
        // Only a few coefficients set
        let mut coeffs = vec![0; TEST_DEGREE + 1];
        coeffs[0] = 1; // Constant term
        coeffs[3] = 100; // x^3 term
        coeffs[6] = 200; // x^6 term
        coeffs[9] = 999; // x^9 term (highest degree)
        let poly = make_poly(&coeffs);

        assert_eq!(poly.extract_coefficient(0), 1);
        assert_eq!(poly.extract_coefficient(3), 100);
        assert_eq!(poly.extract_coefficient(6), 200);
        assert_eq!(poly.extract_coefficient(9), 999);

        // Check that other coefficients are zero
        assert_eq!(poly.extract_coefficient(2), 0);
        assert_eq!(poly.extract_coefficient(5), 0);
    }

    #[test]
    fn test_neg_positive_values() {
        // Test negation of positive values
        let poly = make_poly(&[100, 200, 300]);
        let negated = -poly;

        assert_eq!(negated.extract_coefficient(0), -100);
        assert_eq!(negated.extract_coefficient(1), -200);
        assert_eq!(negated.extract_coefficient(2), -300);
    }

    #[test]
    fn test_neg_negative_values() {
        // Test negation of negative values
        let poly = make_poly(&[-100, -200, -300]);
        let negated = -poly;

        assert_eq!(negated.extract_coefficient(0), 100);
        assert_eq!(negated.extract_coefficient(1), 200);
        assert_eq!(negated.extract_coefficient(2), 300);
    }

    #[test]
    fn test_neg_zero() {
        // Negation of zero should be zero
        let zero = TestPoly::zero();
        let negated = -zero;

        assert!(negated.is_zero());
    }

    #[test]
    fn test_neg_double_negation() {
        // Double negation should return the original value
        let poly = make_poly(&[1, 2, 3, 4, 5]);
        let double_neg = -(-poly);

        assert_eq!(double_neg, poly);
    }

    #[test]
    fn test_neg_extreme_values() {
        // Test negation with extreme values
        // Negate 28-bit MAX should give -(2^27 - 1) = -134217727
        let poly_max = make_poly(&[MAX_VAL]);
        let neg_max = -poly_max;
        assert_eq!(neg_max.extract_coefficient(0), -MAX_VAL);

        // Negate 28-bit MIN (-2^27) should give -(-2^27) = 2^27
        // But 2^27 overflows 28-bit range, so this wraps to MIN in two's complement
        // This is expected behavior for two's complement negation
        let poly_min = make_poly(&[MIN_VAL]);
        let neg_min = -poly_min;
        // -MIN overflows and wraps to MIN in 28-bit two's complement
        assert_eq!(neg_min.extract_coefficient(0), MIN_VAL);
    }

    #[test]
    fn test_neg_mixed_values() {
        // Test negation with mixed positive and negative values
        let poly = make_poly(&[100, -50, 0, 200, -300]);
        let negated = -poly;

        assert_eq!(negated.extract_coefficient(0), -100);
        assert_eq!(negated.extract_coefficient(1), 50);
        assert_eq!(negated.extract_coefficient(2), 0);
        assert_eq!(negated.extract_coefficient(3), -200);
        assert_eq!(negated.extract_coefficient(4), 300);
    }

    #[test]
    fn test_neg_addition_identity() {
        // Test that poly + (-poly) = 0
        let poly = make_poly(&[1, 2, 3, 4, 5]);
        let sum = poly + (-poly);

        assert!(sum.is_zero());
    }

    /// Equivalence tests: ensure BitDecomposedPolynomial behaves exactly like DensePolynomial
    mod equivalence_with_dense {
        use super::*;

        type DensePoly = DensePolynomial<i64, TEST_DEGREE>;

        /// Helper to create both types from the same coefficients
        fn make_both(coeffs: &[i64]) -> (TestPoly, DensePoly) {
            let bit_poly = make_poly(coeffs);
            let dense_poly = DensePoly::new(coeffs);
            (bit_poly, dense_poly)
        }

        /// Verify conversion round-trips correctly
        fn assert_conversion_equivalence(bit_poly: &TestPoly, dense_poly: &DensePoly) {
            let bit_to_dense: DensePoly = bit_poly.into();
            let dense_to_bit: TestPoly = dense_poly.into();
            
            assert_eq!(bit_to_dense.to_coeffs(), dense_poly.to_coeffs(), 
                "BitPoly->Dense conversion mismatch");
            assert_eq!(dense_to_bit, *bit_poly, 
                "Dense->BitPoly conversion mismatch");
        }

        #[test]
        fn test_zero_equivalence() {
            let bit_zero = TestPoly::zero();
            let dense_zero = DensePoly::zero();

            assert_eq!(bit_zero.is_zero(), dense_zero.is_zero());
            assert_conversion_equivalence(&bit_zero, &dense_zero);
        }

        #[test]
        fn test_one_equivalence() {
            let bit_one = TestPoly::one();
            let dense_one = DensePoly::one();

            assert_conversion_equivalence(&bit_one, &dense_one);
        }

        #[test]
        fn test_addition_equivalence() {
            let coeffs1 = vec![1, 2, 3, 4, 5];
            let coeffs2 = vec![10, 20, 30, 40, 50];

            let (bit1, dense1) = make_both(&coeffs1);
            let (bit2, dense2) = make_both(&coeffs2);

            let bit_sum = bit1 + bit2;
            let dense_sum = dense1 + dense2;

            assert_conversion_equivalence(&bit_sum, &dense_sum);
        }

        #[test]
        fn test_addition_with_zero_equivalence() {
            let coeffs = vec![1, 2, 3];
            let (bit_poly, dense_poly) = make_both(&coeffs);

            let bit_zero = TestPoly::zero();
            let dense_zero = DensePoly::zero();

            let bit_result = bit_poly + bit_zero;
            let dense_result = dense_poly + dense_zero;

            assert_conversion_equivalence(&bit_result, &dense_result);
        }

        #[test]
        fn test_subtraction_equivalence() {
            let coeffs1 = vec![100, 200, 300];
            let coeffs2 = vec![50, 75, 100];

            let (bit1, dense1) = make_both(&coeffs1);
            let (bit2, dense2) = make_both(&coeffs2);

            let bit_diff = bit1 - bit2;
            let dense_diff = dense1 - dense2;

            assert_conversion_equivalence(&bit_diff, &dense_diff);
        }

        #[test]
        fn test_negation_equivalence() {
            let coeffs = vec![100, -50, 0, 200, -300];
            let (bit_poly, dense_poly) = make_both(&coeffs);

            let bit_neg = -bit_poly;
            let dense_neg = -dense_poly;

            assert_conversion_equivalence(&bit_neg, &dense_neg);
        }

        #[test]
        fn test_add_assign_equivalence() {
            let coeffs1 = vec![1, 2, 3];
            let coeffs2 = vec![4, 5, 6];

            let (mut bit1, mut dense1) = make_both(&coeffs1);
            let (bit2, dense2) = make_both(&coeffs2);

            bit1 += bit2;
            dense1 += dense2;

            assert_conversion_equivalence(&bit1, &dense1);
        }

        #[test]
        fn test_add_assign_ref_equivalence() {
            let coeffs1 = vec![1, 2, 3];
            let coeffs2 = vec![4, 5, 6];

            let (mut bit1, mut dense1) = make_both(&coeffs1);
            let (bit2, dense2) = make_both(&coeffs2);

            bit1 += &bit2;
            dense1 += &dense2;

            assert_conversion_equivalence(&bit1, &dense1);
        }

        #[test]
        fn test_sub_assign_equivalence() {
            let coeffs1 = vec![100, 200, 300];
            let coeffs2 = vec![10, 20, 30];

            let (mut bit1, mut dense1) = make_both(&coeffs1);
            let (bit2, dense2) = make_both(&coeffs2);

            bit1 -= bit2;
            dense1 -= dense2;

            assert_conversion_equivalence(&bit1, &dense1);
        }

        #[test]
        fn test_checked_add_success_equivalence() {
            let coeffs1 = vec![100, 200, 300];
            let coeffs2 = vec![10, 20, 30];

            let (bit1, dense1) = make_both(&coeffs1);
            let (bit2, dense2) = make_both(&coeffs2);

            let bit_result = bit1.checked_add(&bit2);
            let dense_result = dense1.checked_add(&dense2);

            assert!(bit_result.is_some());
            assert!(dense_result.is_some());

            assert_conversion_equivalence(&bit_result.unwrap(), &dense_result.unwrap());
        }

        #[test]
        fn test_checked_add_overflow_equivalence() {
            let max_val = (1i64 << 27) - 1;  // 28-bit max
            let coeffs1 = vec![max_val; 10];
            let coeffs2 = vec![1; 10];

            let (bit1, dense1) = make_both(&coeffs1);
            let (bit2, dense2) = make_both(&coeffs2);

            let bit_result = bit1.checked_add(&bit2);
            let dense_result = dense1.checked_add(&dense2);

            // BitDecomposedPolynomial should return None (28-bit overflow)
            assert!(bit_result.is_none(), "BitPoly should overflow at 28-bit boundary");
            
            // DensePolynomial<i64> will succeed because it has 64-bit range
            // This is expected behavior - they have different bit widths
            // We verify the conversion works for values within 28-bit range
            if let Some(dense_sum) = dense_result {
                // Verify the dense result is out of 28-bit range
                let coeffs = dense_sum.to_coeffs();
                let expected = max_val + 1;
                assert!(coeffs[0] == expected && expected > max_val);
            }
        }

        #[test]
        fn test_checked_sub_success_equivalence() {
            let coeffs1 = vec![100, 200, 300];
            let coeffs2 = vec![10, 20, 30];

            let (bit1, dense1) = make_both(&coeffs1);
            let (bit2, dense2) = make_both(&coeffs2);

            let bit_result = bit1.checked_sub(&bit2);
            let dense_result = dense1.checked_sub(&dense2);

            assert!(bit_result.is_some());
            assert!(dense_result.is_some());

            assert_conversion_equivalence(&bit_result.unwrap(), &dense_result.unwrap());
        }

        #[test]
        fn test_checked_neg_equivalence() {
            let coeffs = vec![100, -50, 0, 200];
            let (bit_poly, dense_poly) = make_both(&coeffs);

            let bit_result = bit_poly.checked_neg();
            let dense_result = dense_poly.checked_neg();

            assert!(bit_result.is_some());
            assert!(dense_result.is_some());

            assert_conversion_equivalence(&bit_result.unwrap(), &dense_result.unwrap());
        }

        #[test]
        fn test_display_equivalence() {
            let coeffs = vec![1, 2, 3, 4, 5];
            let (bit_poly, dense_poly) = make_both(&coeffs);

            let bit_str = format!("{}", bit_poly);
            let dense_str = format!("{}", dense_poly);

            // Both should produce the same string representation
            assert_eq!(bit_str, dense_str);
        }

        #[test]
        fn test_hash_equivalence() {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let coeffs = vec![1, 2, 3, 4, 5];
            let (bit_poly, dense_poly) = make_both(&coeffs);

            // Convert to same type for hash comparison
            let bit_as_dense: DensePoly = (&bit_poly).into();
            let dense_clone = dense_poly.clone();

            let mut hasher1 = DefaultHasher::new();
            bit_as_dense.hash(&mut hasher1);
            let hash1 = hasher1.finish();

            let mut hasher2 = DefaultHasher::new();
            dense_clone.hash(&mut hasher2);
            let hash2 = hasher2.finish();

            assert_eq!(hash1, hash2);
        }

        #[test]
        fn test_extreme_values_equivalence() {
            let max_val = (1i64 << 27) - 1;  // 28-bit max: 134217727
            let near_min = -(1i64 << 27) + 1; // 28-bit min + 1: -134217727 (can be negated safely)

            let coeffs = vec![max_val, near_min, 0, max_val];
            let (bit_poly, dense_poly) = make_both(&coeffs);

            assert_conversion_equivalence(&bit_poly, &dense_poly);

            // Test negation of extreme values
            // Note: we use near_min instead of min because -min_val overflows in 28-bit
            let bit_neg = -bit_poly;
            let dense_neg = -dense_poly;

            assert_conversion_equivalence(&bit_neg, &dense_neg);
            
            // Verify the negated values are correct
            let bit_neg_dense: DensePoly = (&bit_neg).into();
            let coeffs_neg = bit_neg_dense.to_coeffs();
            assert_eq!(coeffs_neg[0], -max_val);
            assert_eq!(coeffs_neg[1], -near_min);
            assert_eq!(coeffs_neg[2], 0);
            assert_eq!(coeffs_neg[3], -max_val);
        }

        #[test]
        fn test_mixed_operations_equivalence() {
            // Complex sequence: (a + b) - c + (-d)
            let (bit_a, dense_a) = make_both(&[10, 20, 30]);
            let (bit_b, dense_b) = make_both(&[5, 10, 15]);
            let (bit_c, dense_c) = make_both(&[3, 6, 9]);
            let (bit_d, dense_d) = make_both(&[1, 2, 3]);

            let bit_result = (bit_a + bit_b) - bit_c + (-bit_d);
            let dense_result = (dense_a + dense_b) - dense_c + (-dense_d);

            assert_conversion_equivalence(&bit_result, &dense_result);
        }

        #[test]
        fn test_default_equivalence() {
            let bit_default = TestPoly::default();
            let dense_default = DensePoly::default();

            assert_eq!(bit_default.is_zero(), dense_default.is_zero());
            assert_conversion_equivalence(&bit_default, &dense_default);
        }

        #[test]
        fn test_commutative_addition_equivalence() {
            let coeffs1 = vec![1, 2, 3];
            let coeffs2 = vec![4, 5, 6];

            let (bit1, dense1) = make_both(&coeffs1);
            let (bit2, dense2) = make_both(&coeffs2);

            let bit_sum1 = bit1 + bit2;
            let bit_sum2 = bit2 + bit1;
            let dense_sum1 = dense1.clone() + dense2.clone();
            let dense_sum2 = dense2 + dense1;

            assert_eq!(bit_sum1, bit_sum2);
            assert_eq!(dense_sum1.to_coeffs(), dense_sum2.to_coeffs());
            assert_conversion_equivalence(&bit_sum1, &dense_sum1);
        }

        #[test]
        fn test_associative_addition_equivalence() {
            let (bit_a, dense_a) = make_both(&[1, 2]);
            let (bit_b, dense_b) = make_both(&[3, 4]);
            let (bit_c, dense_c) = make_both(&[5, 6]);

            let bit_left = (bit_a + bit_b) + bit_c;
            let bit_right = bit_a + (bit_b + bit_c);
            let dense_left = (dense_a.clone() + dense_b.clone()) + dense_c.clone();
            let dense_right = dense_a + (dense_b + dense_c);

            assert_eq!(bit_left, bit_right);
            assert_eq!(dense_left.to_coeffs(), dense_right.to_coeffs());
            assert_conversion_equivalence(&bit_left, &dense_left);
        }

        #[test]
        fn test_double_negation_equivalence() {
            let coeffs = vec![1, -2, 3, -4, 5];
            let (bit_poly, dense_poly) = make_both(&coeffs);

            let bit_double_neg = -(-bit_poly);
            let dense_double_neg = -(-dense_poly);

            assert_eq!(bit_double_neg, bit_poly);
            assert_eq!(dense_double_neg.to_coeffs(), dense_poly.to_coeffs());
            assert_conversion_equivalence(&bit_double_neg, &dense_double_neg);
        }

        #[test]
        fn test_additive_inverse_equivalence() {
            let coeffs = vec![10, 20, 30];
            let (bit_poly, dense_poly) = make_both(&coeffs);

            let bit_sum = bit_poly + (-bit_poly);
            let dense_sum = dense_poly.clone() + (-dense_poly);

            assert!(bit_sum.is_zero());
            assert!(dense_sum.is_zero());
            assert_conversion_equivalence(&bit_sum, &dense_sum);
        }
    }

    #[test]
    fn test_from_ref_bit_width_expansion() {
        // Test expanding from 4-bit to 16-bit representation
        // With 4 bits, the signed range is -8 to 7
        type SmallPoly = BitDecomposedPolynomial<4, 5>;
        type LargePoly = BitDecomposedPolynomial<16, 5>;

        // Create a polynomial with both positive and negative coefficients
        // Coefficients: [5, -3, 0, 7, -8, 2]
        let coeffs = vec![5_i64, -3, 0, 7, -8, 2];
        let small_poly: SmallPoly = DensePolynomial::<_, 5>::new(&coeffs).into();

        // Verify the small polynomial has correct coefficients
        assert_eq!(small_poly.extract_coefficient(0), 5);
        assert_eq!(small_poly.extract_coefficient(1), -3);
        assert_eq!(small_poly.extract_coefficient(2), 0);
        assert_eq!(small_poly.extract_coefficient(3), 7);
        assert_eq!(small_poly.extract_coefficient(4), -8);
        assert_eq!(small_poly.extract_coefficient(5), 2);

        // Expand to larger bit width using from_ref
        let large_poly = LargePoly::from_ref(&small_poly);

        // Verify all coefficients are preserved after expansion
        assert_eq!(large_poly.extract_coefficient(0), 5);
        assert_eq!(large_poly.extract_coefficient(1), -3);
        assert_eq!(large_poly.extract_coefficient(2), 0);
        assert_eq!(large_poly.extract_coefficient(3), 7);
        assert_eq!(large_poly.extract_coefficient(4), -8);
        assert_eq!(large_poly.extract_coefficient(5), 2);

        // Verify sign extension worked correctly
        // Convert both to dense polynomials and compare
        let small_dense = DensePolynomial::<i64, 5>::from(&small_poly);
        let large_dense = DensePolynomial::<i64, 5>::from(&large_poly);
        assert_eq!(small_dense.to_coeffs(), large_dense.to_coeffs());
    }
}
