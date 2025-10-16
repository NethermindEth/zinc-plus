use crate::{
    add, mul,
    pcs::structs::{MulByScalar, ProjectableToField},
    poly::{ConstCoeffBitWidth, EvaluatablePolynomial, EvaluationError, dense::DensePolynomial},
    traits::{ConstTranscribable, FromRef, Named},
};
use crypto_primitives::{
    FromWithConfig, PrimeField, Semiring, boolean::Boolean, crypto_bigint_int::Int,
};
use num_traits::{CheckedAdd, CheckedMul, CheckedSub, One, Zero};
use rand::{distr::StandardUniform, prelude::*};
use std::{
    array,
    fmt::Display,
    hash::Hash,
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

/// * `BITS` is the maximum bit depth required
/// * `DEGREE` cannot be larger than 63
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BitDecomposedPolynomial<const BITS: usize, const DEGREE: usize> {
    /// `slices[i]` holds the i-th bit (LSB at i=0) of all DEGREE+1 coefficients
    /// packed into a u64.
    slices: [u64; BITS],
}

impl<const BITS: usize, const DEGREE: usize> BitDecomposedPolynomial<BITS, DEGREE> {
    /// Extract the value of a specific coefficient at the given index as a
    /// signed integer. This reconstructs the coefficient from its binary
    /// representation across slices.
    #[inline]
    fn extract_coefficient(&self, index: usize) -> i64 {
        debug_assert!(index <= DEGREE, "Coefficient index out of bounds");

        // Build the value as unsigned first to avoid cast warnings
        let mut value: u64 = 0;
        for (bit_pos, &slice) in self.slices.iter().enumerate() {
            // Extract the bit at position `index` from this slice
            let bit = (slice >> index) & 1;
            value |= bit << bit_pos;
        }

        // Truncate to K bits and sign-extend from K-bit two's complement to i64
        // Since K=32, we interpret the lower 32 bits as i32 and extend to i64
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let value_i32 = value as u32 as i32;
        i64::from(value_i32)
    }

    /// Extract the value of a specific coefficient at the given index as an
    /// unsigned integer. This is the natural representation stored in
    /// bit decomposed format.
    #[inline]
    fn extract_coefficient_unsigned(&self, index: usize) -> u32 {
        debug_assert!(index <= DEGREE, "Coefficient index out of bounds");

        let mut value: u64 = 0;
        for (bit_pos, &slice) in self.slices.iter().enumerate() {
            // Extract the bit at position `index` from this slice
            let bit = (slice >> index) & 1;
            value |= bit << bit_pos;
        }

        // Truncate to K bits (K=32, so this fits exactly in u32)
        #[allow(clippy::cast_possible_truncation)]
        let value_u32 = value as u32;
        value_u32
    }

    /// Set a specific coefficient at the given index from an unsigned integer
    /// value. Returns None if the value doesn't fit in K bits.
    #[inline]
    fn set_coefficient_unsigned(&mut self, index: usize, value: u32) {
        debug_assert!(index <= DEGREE, "Coefficient index out of bounds");

        // Distribute the bits of the value across the slices
        for (bit_pos, slice) in self.slices.iter_mut().enumerate() {
            // Extract the bit at position `bit_pos` from the value
            let bit = u64::from((value >> bit_pos) & 1);
            // Clear the bit at position `index` in this slice, then set it
            *slice = (*slice & !(1u64 << index)) | (bit << index);
        }
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
    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn add_assign(&mut self, rhs: &Self) {
        let mut carry: u64 = 0;

        // The compiler will unroll this loop as BITS is constant.
        for i in 0..BITS {
            let a = self.slices[i];
            let b = rhs.slices[i];

            // Calculate Sum bit: S = A XOR B XOR C_in
            let a_xor_b = a ^ b;
            self.slices[i] = a_xor_b ^ carry;

            // Calculate Carry_out (Optimized Majority function):
            // C_out = (A & B) | (C_in & (A XOR B))
            carry = (a & b) | (carry & a_xor_b);
        }

        // In debug builds, check if K was large enough.
        debug_assert!(
            carry == 0,
            "Coefficient overflow detected! K must be increased."
        );
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
    /// Efficient Polynomial Subtraction (Parallel Ripple-Borrow Subtractor)
    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn sub_assign(&mut self, _rhs: &'a Self) {
        unimplemented!("Polynomial subtraction is not implemented");
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

impl<const BITS: usize, const DEGREE: usize> CheckedAdd for BitDecomposedPolynomial<BITS, DEGREE> {
    fn checked_add(&self, other: &Self) -> Option<Self> {
        let mut result = *self;
        let mut carry: u64 = 0;

        for i in 0..BITS {
            let a = result.slices[i];
            let b = other.slices[i];

            let a_xor_b = a ^ b;
            result.slices[i] = a_xor_b ^ carry;
            carry = (a & b) | (carry & a_xor_b);
        }

        // Check for overflow
        if carry != 0 {
            return None;
        }

        Some(result)
    }
}

impl<const BITS: usize, const DEGREE: usize> CheckedSub for BitDecomposedPolynomial<BITS, DEGREE> {
    fn checked_sub(&self, _other: &Self) -> Option<Self> {
        unimplemented!("Polynomial subtraction is not implemented")
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
        DensePolynomial {
            coeff_0: Int::from(bit_poly.extract_coefficient_unsigned(0)),
            coeffs: array::from_fn::<_, DEGREE, _>(|i| {
                Int::from(bit_poly.extract_coefficient_unsigned(i + 1))
            }),
        }
    }
}

//
// BitDecomposedPolynomial => DensePolynomial<Boolean, DEGREE>
//

impl<const BITS: usize, const DEGREE: usize> From<BitDecomposedPolynomial<BITS, DEGREE>>
    for DensePolynomial<Boolean, DEGREE>
{
    fn from(bit_poly: BitDecomposedPolynomial<BITS, DEGREE>) -> Self {
        (&bit_poly).into()
    }
}

impl<const BITS: usize, const DEGREE: usize> From<&BitDecomposedPolynomial<BITS, DEGREE>>
    for DensePolynomial<Boolean, DEGREE>
{
    fn from(bit_poly: &BitDecomposedPolynomial<BITS, DEGREE>) -> Self {
        DensePolynomial::from_ref(bit_poly)
    }
}

impl<const BITS: usize, const DEGREE: usize> FromRef<BitDecomposedPolynomial<BITS, DEGREE>>
    for DensePolynomial<Boolean, DEGREE>
{
    fn from_ref(bit_poly: &BitDecomposedPolynomial<BITS, DEGREE>) -> Self {
        let c = bit_poly.extract_coefficient_unsigned(0);
        assert!(
            c <= 1,
            "Cannot convert BitDecomposedPolynomial with coeff_0 > 1 to Boolean"
        );
        DensePolynomial {
            coeff_0: Boolean::new(c != 0),
            coeffs: [Boolean::FALSE; DEGREE],
        }
    }
}

impl<const BITS: usize, const DEGREE: usize> From<DensePolynomial<Boolean, DEGREE>>
    for BitDecomposedPolynomial<BITS, DEGREE>
{
    fn from(dense: DensePolynomial<Boolean, DEGREE>) -> Self {
        BitDecomposedPolynomial::from(&dense)
    }
}

impl<const BITS: usize, const DEGREE: usize> From<&DensePolynomial<Boolean, DEGREE>>
    for BitDecomposedPolynomial<BITS, DEGREE>
{
    fn from(dense: &DensePolynomial<Boolean, DEGREE>) -> Self {
        let mut result = BitDecomposedPolynomial::zero();
        result.set_coefficient_unsigned(0, dense.coeff_0.inner().into());
        result
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
        let int_bits = usize::try_from(i64::BITS).expect("Int<LIMBS>::BITS must fit in usize");
        assert!(
            BITS <= int_bits,
            "Cannot convert BitDecomposedPolynomial with BITS > Int<LIMBS>::BITS"
        );
        DensePolynomial {
            coeff_0: i64::from(bit_poly.extract_coefficient_unsigned(0)),
            coeffs: array::from_fn::<_, DEGREE, _>(|i| {
                i64::from(bit_poly.extract_coefficient_unsigned(i + 1))
            }),
        }
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

        // Extract all coefficients and set them
        let coeffs = dense.to_coeffs();
        for (i, &coeff) in coeffs.iter().enumerate() {
            assert!(coeff >= 0);
            result.set_coefficient_unsigned(i, coeff as u32);
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
        for i in 0..BITS_2 {
            result.slices[i] = bit_poly.slices[i];
        }
        result
    }
}

//
// Ring
//

impl<const BITS: usize, const DEGREE: usize> Semiring for BitDecomposedPolynomial<BITS, DEGREE> {}

//
// RNG
//

impl<const BITS: usize, const DEGREE: usize> Distribution<BitDecomposedPolynomial<BITS, DEGREE>>
    for StandardUniform
{
    #[allow(clippy::arithmetic_side_effects)]
    fn sample<Gen: Rng + ?Sized>(&self, rng: &mut Gen) -> BitDecomposedPolynomial<BITS, DEGREE> {
        let mut slices = [0u64; BITS];

        // Sample each slice independently
        for slice in slices.iter_mut() {
            *slice = rng.random::<u64>() & ((1_u64 << DEGREE) - 1); // Ensure we only use DEGREE bits
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

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::{One, Zero};

    const TEST_BITS: usize = 32;
    const TEST_DEGREE: usize = 31;
    type TestPoly = BitDecomposedPolynomial<TEST_BITS, TEST_DEGREE>;

    /// Helper to create a bit decomposed polynomial from u32 coefficients
    fn make_poly(coeffs: &[u32]) -> TestPoly {
        let mut poly = TestPoly::zero();
        for (i, &coeff) in coeffs.iter().enumerate() {
            if i <= TEST_DEGREE {
                poly.set_coefficient_unsigned(i, coeff);
            }
        }
        poly
    }

    #[test]
    fn test_zero_polynomial() {
        let zero = TestPoly::zero();
        assert!(zero.is_zero());

        // All coefficients should be zero
        for i in 0..=TEST_DEGREE {
            assert_eq!(zero.extract_coefficient_unsigned(i), 0);
        }
    }

    #[test]
    fn test_one_polynomial() {
        let one = TestPoly::one();
        assert!(!one.is_zero());

        // Only coefficient 0 should be 1
        assert_eq!(one.extract_coefficient_unsigned(0), 1);
        for i in 1..=TEST_DEGREE {
            assert_eq!(one.extract_coefficient_unsigned(i), 0);
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
        assert_eq!(sum.extract_coefficient_unsigned(0), 5);
        assert_eq!(sum.extract_coefficient_unsigned(1), 7);
        assert_eq!(sum.extract_coefficient_unsigned(2), 9);
    }

    #[test]
    fn test_addition_with_carry() {
        // Test addition that requires bit carries across coefficient boundaries
        // Use values that test carry propagation without overflowing K bits

        // Create polynomials where lower bits will produce carries
        // Example: coefficient has pattern that will generate carries between bit
        // positions
        let poly1 = make_poly(&[0x7FFFFFFF, 0x1]); // (2^31 - 1) + x
        let poly2 = make_poly(&[0x7FFFFFFF, 0x0]); // (2^31 - 1)

        let sum = poly1 + poly2;

        // 0x7FFFFFFF + 0x7FFFFFFF = 0xFFFFFFFE (no overflow within 32 bits)
        // This tests the ripple-carry adder working correctly
        assert_eq!(sum.extract_coefficient_unsigned(0), 0xFFFFFFFE);
        assert_eq!(sum.extract_coefficient_unsigned(1), 1);

        // Test another carry scenario: adding 1 to a value with all lower bits set
        let poly3 = make_poly(&[0xFF, 0x0]); // 255
        let poly4 = make_poly(&[0x01, 0x0]); // 1

        let sum2 = poly3 + poly4;
        assert_eq!(sum2.extract_coefficient_unsigned(0), 0x100); // 256
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
        assert_eq!(sum.extract_coefficient_unsigned(0), 5);
        assert_eq!(sum.extract_coefficient_unsigned(1), 7);
        assert_eq!(sum.extract_coefficient_unsigned(2), 9);
    }

    #[test]
    fn test_checked_add_overflow() {
        // Create two polynomials that will overflow when added
        let poly1 = make_poly(&[0xFFFFFFFF; 32]);
        let poly2 = make_poly(&[0x1; 32]);

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

        assert_eq!(sum.extract_coefficient_unsigned(0), 1);
        assert_eq!(sum.extract_coefficient_unsigned(1), 2);
        assert_eq!(sum.extract_coefficient_unsigned(2), 3);
    }

    #[test]
    fn test_sum_iterator_refs() {
        let polys = vec![
            make_poly(&[1, 0, 0]),
            make_poly(&[0, 2, 0]),
            make_poly(&[0, 0, 3]),
        ];

        let sum: TestPoly = polys.iter().sum();

        assert_eq!(sum.extract_coefficient_unsigned(0), 1);
        assert_eq!(sum.extract_coefficient_unsigned(1), 2);
        assert_eq!(sum.extract_coefficient_unsigned(2), 3);
    }

    #[test]
    fn test_display() {
        let poly = make_poly(&[1, 2, 3]);
        let s = format!("{}", poly);

        // Should start with "[1, 2, 3" and have 32 total coefficients
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

        // Test value that would be negative in two's complement
        let poly2 = make_poly(&[0x80000000]); // MSB set
        let signed = poly2.extract_coefficient(0);
        // Should be interpreted as negative in i32 two's complement
        assert!(signed < 0);
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

        assert_eq!(bit_poly.extract_coefficient_unsigned(0), 1);
        assert_eq!(bit_poly.extract_coefficient_unsigned(1), 2);
        assert_eq!(bit_poly.extract_coefficient_unsigned(2), 3);
        assert_eq!(bit_poly.extract_coefficient_unsigned(3), 4);
        assert_eq!(bit_poly.extract_coefficient_unsigned(4), 5);
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
        // Test with maximum u32 values
        let poly = make_poly(&[0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF]);
        let dense: DensePolynomial<i64, TEST_DEGREE> = poly.into();
        let back: TestPoly = (&dense).into();

        assert_eq!(back, poly);
    }

    #[test]
    fn test_add_assign() {
        let mut poly1 = make_poly(&[1, 2, 3]);
        let poly2 = make_poly(&[4, 5, 6]);

        poly1 += poly2;

        assert_eq!(poly1.extract_coefficient_unsigned(0), 5);
        assert_eq!(poly1.extract_coefficient_unsigned(1), 7);
        assert_eq!(poly1.extract_coefficient_unsigned(2), 9);
    }

    #[test]
    fn test_add_assign_ref() {
        let mut poly1 = make_poly(&[1, 2, 3]);
        let poly2 = make_poly(&[4, 5, 6]);

        poly1 += &poly2;

        assert_eq!(poly1.extract_coefficient_unsigned(0), 5);
        assert_eq!(poly1.extract_coefficient_unsigned(1), 7);
        assert_eq!(poly1.extract_coefficient_unsigned(2), 9);
    }

    #[test]
    fn test_large_coefficients() {
        // Test with coefficients near the K-bit limit
        let large = 0xFFFFFF00u32;
        let poly = make_poly(&[large, large, large]);

        assert_eq!(poly.extract_coefficient_unsigned(0), large);
        assert_eq!(poly.extract_coefficient_unsigned(1), large);
        assert_eq!(poly.extract_coefficient_unsigned(2), large);
    }

    #[test]
    fn test_all_coefficients_set() {
        // Test setting all 32 coefficients
        let mut coeffs = Vec::new();
        #[allow(clippy::cast_possible_truncation)]
        for i in 0..=TEST_DEGREE {
            coeffs.push(i as u32);
        }

        let poly = make_poly(&coeffs);

        #[allow(clippy::cast_possible_truncation)]
        for i in 0..=TEST_DEGREE {
            assert_eq!(poly.extract_coefficient_unsigned(i), i as u32);
        }
    }

    #[test]
    fn test_sparse_polynomial() {
        // Only a few coefficients set
        let mut poly = TestPoly::zero();
        poly.set_coefficient_unsigned(0, 1);
        poly.set_coefficient_unsigned(10, 100);
        poly.set_coefficient_unsigned(20, 200);
        poly.set_coefficient_unsigned(31, 999);

        assert_eq!(poly.extract_coefficient_unsigned(0), 1);
        assert_eq!(poly.extract_coefficient_unsigned(10), 100);
        assert_eq!(poly.extract_coefficient_unsigned(20), 200);
        assert_eq!(poly.extract_coefficient_unsigned(31), 999);

        // Check that other coefficients are zero
        assert_eq!(poly.extract_coefficient_unsigned(5), 0);
        assert_eq!(poly.extract_coefficient_unsigned(15), 0);
    }
}
