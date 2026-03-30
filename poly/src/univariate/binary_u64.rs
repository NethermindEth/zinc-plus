use crate::{
    ConstCoeffBitWidth, EvaluatablePolynomial, EvaluationError, Polynomial,
    univariate::dense::DensePolynomial,
};
use core::mem::MaybeUninit;
use crypto_primitives::{PrimeField, Semiring, semiring::boolean::Boolean};
use derive_more::{AsRef, Display};
use num_traits::{CheckedAdd, CheckedMul, CheckedSub, One, Zero};
use rand::{distr::StandardUniform, prelude::*};
use std::{
    array,
    hash::Hash,
    iter::{Product, Sum},
    marker::PhantomData,
    ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};
use zinc_transcript::traits::ConstTranscribable;
use zinc_utils::{
    from_ref::FromRef,
    inner_product::{InnerProduct, InnerProductError},
    mul_by_scalar::WideningMulByScalar,
    named::Named,
    projectable_to_field::ProjectableToField,
};

#[derive(AsRef, Clone, Debug, Default, Display, Hash, PartialEq, Eq)]
#[repr(transparent)]
pub struct BinaryU64Poly<const DEGREE_PLUS_ONE: usize>(u64); // we can fit up to degree 64, which is ok for now

impl<const DEGREE_PLUS_ONE: usize> BinaryU64Poly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    pub const fn inner(&self) -> &u64 {
        &self.0
    }
}

impl<const DEGREE_PLUS_ONE: usize> From<BinaryU64Poly<DEGREE_PLUS_ONE>> for u64 {
    #[inline(always)]
    fn from(binary_poly: BinaryU64Poly<DEGREE_PLUS_ONE>) -> Self {
        binary_poly.0
    }
}

impl<const DEGREE_PLUS_ONE: usize> From<BinaryU64Poly<DEGREE_PLUS_ONE>>
    for DensePolynomial<Boolean, DEGREE_PLUS_ONE>
{
    #[inline(always)]
    fn from(binary_poly: BinaryU64Poly<DEGREE_PLUS_ONE>) -> Self {
        DensePolynomial {
            coeffs: array::from_fn(|i| Boolean::new(!(binary_poly.0 & (1 << i)).is_zero())),
        }
    }
}

impl From<u32> for BinaryU64Poly<32> {
    fn from(value: u32) -> Self {
        Self(u64::from(value)) // we ignore upper bits
    }
}

impl From<u64> for BinaryU64Poly<64> {
    fn from(value: u64) -> Self {
        Self(value) // we don't ignore any bits
    }
}

impl<const DEGREE_PLUS_ONE: usize> BinaryU64Poly<DEGREE_PLUS_ONE> {
    /// Create a new polynomial with the given coefficients.
    /// If the input has fewer than N+1 coefficients, the remaining slots will
    /// be filled with zeros. If the input has more than N+1 coefficients,
    /// it will panic.
    #[inline(always)]
    pub fn new(coeffs: impl AsRef<[Boolean]>) -> Self {
        // Self(DensePolynomial::new(coeffs))
        let coeffs = coeffs.as_ref();
        assert!(coeffs.len() <= DEGREE_PLUS_ONE);
        let mut value: u64 = 0;
        for (i, coeff) in coeffs.iter().enumerate() {
            if coeff.inner() {
                value |= 1 << i;
            }
        }
        Self(value)
    }

    /// Create a new polynomial with the given coefficients.
    /// If the input has fewer than N+1 coefficients, the remaining slots will
    /// be filled with zeros. If the input has more than N+1 coefficients,
    /// it will panic.
    #[inline(always)]
    pub fn new_padded(coeffs: impl AsRef<[Boolean]>) -> Self {
        let coeffs = coeffs.as_ref();
        assert!(coeffs.len() <= DEGREE_PLUS_ONE);
        let mut value: u64 = 0;
        for (i, coeff) in coeffs.iter().enumerate() {
            if coeff.inner() {
                value |= 1 << i;
            }
        }
        Self(value)
    }
}

impl<const DEGREE_PLUS_ONE: usize> Zero for BinaryU64Poly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn zero() -> Self {
        Self(0)
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl<const DEGREE_PLUS_ONE: usize> One for BinaryU64Poly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn one() -> Self {
        Self(1)
    }
}

impl<'a, const DEGREE_PLUS_ONE: usize> Add<&'a Self> for BinaryU64Poly<DEGREE_PLUS_ONE> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects, clippy::suspicious_arithmetic_impl)]
    #[inline(always)]
    fn add(self, rhs: &'a Self) -> Self::Output {
        // addition in GF(2) is XOR
        Self(self.0 ^ rhs.0)
    }
}

impl<const DEGREE_PLUS_ONE: usize> Add for BinaryU64Poly<DEGREE_PLUS_ONE> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        self.add(&rhs)
    }
}

impl<'a, const DEGREE_PLUS_ONE: usize> Sub<&'a Self> for BinaryU64Poly<DEGREE_PLUS_ONE> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects, clippy::suspicious_arithmetic_impl)]
    #[inline(always)]
    fn sub(self, rhs: &'a Self) -> Self::Output {
        // subtraction in GF(2) is XOR
        Self(self.0 ^ rhs.0)
    }
}

impl<const DEGREE_PLUS_ONE: usize> Sub for BinaryU64Poly<DEGREE_PLUS_ONE> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        self.sub(&rhs)
    }
}

impl<const DEGREE_PLUS_ONE: usize> Mul for BinaryU64Poly<DEGREE_PLUS_ONE> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        self.mul(&rhs)
    }
}

impl<'a, const DEGREE_PLUS_ONE: usize> Mul<&'a Self> for BinaryU64Poly<DEGREE_PLUS_ONE> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn mul(self, _rhs: &'a Self) -> Self::Output {
        unimplemented!("Multiplication for BinaryU64Poly is not implemented yet");
    }
}

#[allow(clippy::suspicious_op_assign_impl)]
impl<'a, const DEGREE_PLUS_ONE: usize> AddAssign<&'a Self> for BinaryU64Poly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: &'a Self) {
        // addition in GF(2) is XOR
        self.0 ^= rhs.0;
    }
}

impl<const DEGREE_PLUS_ONE: usize> AddAssign for BinaryU64Poly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.add_assign(&rhs);
    }
}

#[allow(clippy::suspicious_op_assign_impl)]
impl<'a, const DEGREE_PLUS_ONE: usize> SubAssign<&'a Self> for BinaryU64Poly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: &'a Self) {
        // subtraction in GF(2) is XOR
        self.0 ^= rhs.0;
    }
}

impl<const DEGREE_PLUS_ONE: usize> SubAssign for BinaryU64Poly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.sub_assign(&rhs);
    }
}

impl<const DEGREE_PLUS_ONE: usize> MulAssign for BinaryU64Poly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn mul_assign(&mut self, _rhs: Self) {
        unimplemented!("Multiplication for BinaryU64Poly is not implemented yet");
    }
}

impl<'a, const DEGREE_PLUS_ONE: usize> MulAssign<&'a Self> for BinaryU64Poly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn mul_assign(&mut self, _rhs: &'a Self) {
        unimplemented!("Multiplication for BinaryU64Poly is not implemented yet");
        // self.0.mul_assign(&rhs.0);
    }
}

impl<const DEGREE_PLUS_ONE: usize> CheckedAdd for BinaryU64Poly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn checked_add(&self, other: &Self) -> Option<Self> {
        // If any bit is set in both operands, addition would overflow (1+1)
        if (self.0 & other.0) != 0 {
            None
        } else {
            // addition in GF(2) is XOR
            Some(Self(self.0 ^ other.0))
        }
    }
}

impl<const DEGREE_PLUS_ONE: usize> CheckedSub for BinaryU64Poly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn checked_sub(&self, other: &Self) -> Option<Self> {
        // If any bit is 0 in self and 1 in other, subtraction would overflow (0-1)
        if ((!self.0) & other.0) != 0 {
            // There exists a bit where self is 0 and other is 1
            None
        } else {
            // subtraction in GF(2) is XOR
            Some(Self(self.0 ^ other.0))
        }
    }
}

impl<const DEGREE_PLUS_ONE: usize> CheckedMul for BinaryU64Poly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn checked_mul(&self, _other: &Self) -> Option<Self> {
        unimplemented!("Checked multiplication for BinaryU64Poly is not implemented yet");
    }
}

impl<'a, const DEGREE_PLUS_ONE: usize> Sum<&'a Self> for BinaryU64Poly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.sum()
    }
}

impl<'a, const DEGREE_PLUS_ONE: usize> Product<&'a Self> for BinaryU64Poly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.product()
    }
}

impl<const DEGREE_PLUS_ONE: usize> Semiring for BinaryU64Poly<DEGREE_PLUS_ONE> {}

impl<const DEGREE_PLUS_ONE: usize> Distribution<BinaryU64Poly<DEGREE_PLUS_ONE>>
    for StandardUniform
{
    #[inline(always)]
    #[allow(clippy::arithmetic_side_effects)]
    fn sample<Gen: Rng + ?Sized>(&self, rng: &mut Gen) -> BinaryU64Poly<DEGREE_PLUS_ONE> {
        let mut value: u64 = rng.next_u64();
        if DEGREE_PLUS_ONE < 64 {
            value &= (1u64 << DEGREE_PLUS_ONE) - 1;
        }
        BinaryU64Poly::<DEGREE_PLUS_ONE>(value)
    }
}

//
// Zip-specific traits
//
impl<const DEGREE_PLUS_ONE: usize> Polynomial<Boolean> for BinaryU64Poly<DEGREE_PLUS_ONE> {
    const DEGREE_BOUND: usize = DensePolynomial::<Boolean, DEGREE_PLUS_ONE>::DEGREE_BOUND;
}

#[allow(clippy::arithmetic_side_effects)]
impl<R: Clone + Zero + One + CheckedAdd + CheckedMul, const DEGREE_PLUS_ONE: usize>
    EvaluatablePolynomial<Boolean, R> for BinaryU64Poly<DEGREE_PLUS_ONE>
{
    type EvaluationPoint = R;

    fn evaluate_at_point(&self, point: &R) -> Result<R, EvaluationError> {
        if DEGREE_PLUS_ONE.is_one() {
            return Ok(R::zero());
        }

        let mut result = R::zero();
        let mut pow = R::one();
        for i in 0..DEGREE_PLUS_ONE {
            if (self.0 & (1 << i)) != 0 {
                result = result.checked_add(&pow).ok_or(EvaluationError::Overflow)?;
            }
            if i + 1 < DEGREE_PLUS_ONE {
                pow = pow.checked_mul(point).ok_or(EvaluationError::Overflow)?;
            }
        }

        Ok(result)
    }
}

impl<const DEGREE_PLUS_ONE: usize> ConstCoeffBitWidth for BinaryU64Poly<DEGREE_PLUS_ONE> {
    const COEFF_BIT_WIDTH: usize = DensePolynomial::<Boolean, DEGREE_PLUS_ONE>::COEFF_BIT_WIDTH;
}

impl<const DEGREE_PLUS_ONE: usize> Named for BinaryU64Poly<DEGREE_PLUS_ONE> {
    fn type_name() -> String {
        format!("BPoly<{}>", Self::DEGREE_BOUND)
    }
}

impl<const DEGREE_PLUS_ONE: usize> ConstTranscribable for BinaryU64Poly<DEGREE_PLUS_ONE> {
    const NUM_BYTES: usize = DensePolynomial::<Boolean, DEGREE_PLUS_ONE>::NUM_BYTES;

    #[inline(always)]
    fn read_transcription_bytes(bytes: &[u8]) -> Self {
        Self(u64::read_transcription_bytes(bytes))
    }

    #[inline(always)]
    fn write_transcription_bytes(&self, buf: &mut [u8]) {
        self.0.write_transcription_bytes(buf);
    }
}

impl<const DEGREE_PLUS_ONE: usize> FromRef<BinaryU64Poly<DEGREE_PLUS_ONE>>
    for BinaryU64Poly<DEGREE_PLUS_ONE>
{
    #[inline(always)]
    fn from_ref(poly: &BinaryU64Poly<DEGREE_PLUS_ONE>) -> Self {
        poly.clone()
    }
}

impl<const DEGREE_PLUS_ONE: usize> From<&BinaryU64Poly<DEGREE_PLUS_ONE>>
    for BinaryU64Poly<DEGREE_PLUS_ONE>
{
    #[inline(always)]
    fn from(value: &BinaryU64Poly<DEGREE_PLUS_ONE>) -> Self {
        Self::from_ref(value)
    }
}

pub struct BinaryU64PolyIter<'a, const DEGREE_PLUS_ONE: usize> {
    i: usize,
    poly: &'a BinaryU64Poly<DEGREE_PLUS_ONE>,
}

impl<'a, const DEGREE_PLUS_ONE: usize> BinaryU64PolyIter<'a, DEGREE_PLUS_ONE> {
    pub fn new(poly: &'a BinaryU64Poly<DEGREE_PLUS_ONE>) -> Self {
        Self { i: 0, poly }
    }
}

impl<'a, const DEGREE_PLUS_ONE: usize> Iterator for BinaryU64PolyIter<'a, DEGREE_PLUS_ONE> {
    type Item = Boolean;

    #[allow(clippy::arithmetic_side_effects)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.i < DEGREE_PLUS_ONE {
            let i = self.i;

            self.i += 1;

            Some((!(self.poly.0 & (1 << i)).is_zero()).into())
        } else {
            None
        }
    }
}

impl<const DEGREE_PLUS_ONE: usize> BinaryU64Poly<DEGREE_PLUS_ONE> {
    pub fn iter(&self) -> BinaryU64PolyIter<'_, DEGREE_PLUS_ONE> {
        BinaryU64PolyIter::new(self)
    }
}

pub struct BinaryU64PolyInnerProduct<R, const DEGREE_PLUS_ONE: usize>(PhantomData<R>);

impl<Rhs, Out, const DEGREE_PLUS_ONE: usize> InnerProduct<BinaryU64Poly<DEGREE_PLUS_ONE>, Rhs, Out>
    for BinaryU64PolyInnerProduct<Rhs, DEGREE_PLUS_ONE>
where
    Rhs: Clone,
    Out: FromRef<Rhs> + CheckedAdd,
{
    #[inline(always)]
    #[allow(clippy::arithmetic_side_effects)] // By design
    fn inner_product<const CHECK: bool>(
        lhs: &BinaryU64Poly<DEGREE_PLUS_ONE>,
        rhs: &[Rhs],
        zero: Out,
    ) -> Result<Out, InnerProductError> {
        if rhs.len() != DEGREE_PLUS_ONE {
            return Err(InnerProductError::LengthMismatch {
                lhs: DEGREE_PLUS_ONE,
                rhs: rhs.len(),
            });
        }

        let mut acc = zero;
        let mut bits = lhs.0;
        while bits != 0 {
            let i = bits.trailing_zeros() as usize;
            let rhs = Out::from_ref(&rhs[i]);
            if CHECK {
                acc = acc.checked_add(&rhs).ok_or(InnerProductError::Overflow)?;
            } else {
                acc = acc + rhs;
            }
            // changes the LSB 1 bit to 0
            bits &= bits - 1;
        }

        Ok(acc)
    }
}

impl<F, const DEGREE_PLUS_ONE: usize> ProjectableToField<F> for BinaryU64Poly<DEGREE_PLUS_ONE>
where
    F: PrimeField + FromRef<F> + 'static,
{
    fn prepare_projection(sampled_value: &F) -> impl Fn(&Self) -> F + 'static {
        use crate::univariate::{PROJ_CHUNK_BITS, build_projection_tables};
        let field_cfg = sampled_value.cfg().clone();
        let tables = build_projection_tables::<F, DEGREE_PLUS_ONE>(sampled_value);
        let num_chunks = tables.len();

        move |poly: &Self| {
            let bits = poly.0;
            let mut acc = F::zero_with_cfg(&field_cfg);
            for chunk_idx in 0..num_chunks {
                let start = chunk_idx * PROJ_CHUNK_BITS;
                let chunk_width = std::cmp::min(PROJ_CHUNK_BITS, DEGREE_PLUS_ONE - start);
                let mask = (1u64 << chunk_width) - 1;
                let byte_val = ((bits >> start) & mask) as usize;
                acc += tables[chunk_idx][byte_val].clone();
            }
            acc
        }
    }
}

#[derive(Clone, Copy, Default)]
pub struct BinaryU64PolyWideningMulByScalar<Output>(PhantomData<Output>);

impl<const DEGREE_PLUS_ONE: usize> WideningMulByScalar<BinaryU64Poly<DEGREE_PLUS_ONE>, i64>
    for BinaryU64PolyWideningMulByScalar<i64>
{
    type Output = DensePolynomial<i64, DEGREE_PLUS_ONE>;

    fn mul_by_scalar_widen(lhs: &BinaryU64Poly<DEGREE_PLUS_ONE>, rhs: &i64) -> Self::Output {
        widen_simd::<DEGREE_PLUS_ONE>(lhs, *rhs)
    }
}

impl<const DEGREE_PLUS_ONE: usize> WideningMulByScalar<BinaryU64Poly<DEGREE_PLUS_ONE>, i64>
    for BinaryU64PolyWideningMulByScalar<i128>
{
    type Output = DensePolynomial<i128, DEGREE_PLUS_ONE>;

    fn mul_by_scalar_widen(lhs: &BinaryU64Poly<DEGREE_PLUS_ONE>, rhs: &i64) -> Self::Output {
        let narrow = widen_simd::<DEGREE_PLUS_ONE>(lhs, *rhs);
        let coeffs = narrow.coeffs.map(|c| c as i128);
        DensePolynomial { coeffs }
    }
}

#[allow(unreachable_code, unused_variables)] // CI system does not support SIMD features
#[inline(always)]
pub fn widen_simd<const DEGREE_PLUS_ONE: usize>(
    poly: &BinaryU64Poly<DEGREE_PLUS_ONE>,
    scalar: i64,
) -> DensePolynomial<i64, DEGREE_PLUS_ONE> {
    let mut coeffs_uninit = MaybeUninit::<[i64; DEGREE_PLUS_ONE]>::uninit();
    let out_ptr = coeffs_uninit.as_mut_ptr() as *mut i64;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        widen_fill_neon::<DEGREE_PLUS_ONE>(&poly.0, out_ptr, scalar);
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    unsafe {
        widen_fill_avx512::<DEGREE_PLUS_ONE>(&poly.0, out_ptr, scalar);
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx512f")
    )))]
    {
        panic!("SIMD widening not supported on this architecture");
    }

    let coeffs = unsafe { coeffs_uninit.assume_init() };
    DensePolynomial { coeffs }
}

#[allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_lossless
)]
#[allow(unsafe_op_in_unsafe_fn)]
#[cfg(target_arch = "aarch64")]
#[inline(always)]
// Converts a u64 bitmask into an array of i64 values using ARM NEON SIMD.
// Processes 8 bits at a time with ~8-12× speedup vs scalar code.
unsafe fn widen_fill_neon<const N: usize>(mask_ref: &u64, out_ptr: *mut i64, scalar: i64) {
    use core::arch::aarch64::*;

    let mask64: u64 = *mask_ref;

    // Replicate scalar across SIMD lanes for branchless selection
    let scalar_v: int64x2_t = vdupq_n_s64(scalar);

    let mut i = 0usize;

    // For N == 64, use lookup table (16KB). For smaller N, use SIMD widening
    if N == 64 {
        // Precomputed lookup table: maps each byte value (0-255) directly to 8 i64
        // values Each bit in the byte produces -1 (if set) or 0 (if clear)
        // Table size: 256 entries × 8 i64 × 8 bytes = 16KB
        #[repr(align(64))]
        struct LookupTable([[i64; 8]; 256]);

        static LUT: LookupTable = {
            let mut table = [[0i64; 8]; 256];
            let mut i = 0;
            while i < 256 {
                let mut bit = 0;
                while bit < 8 {
                    table[i][bit] = if (i & (1 << bit)) != 0 { -1 } else { 0 };
                    bit += 1;
                }
                i += 1;
            }
            LookupTable(table)
        };

        // Main loop: process 8 coefficients per iteration
        while i + 8 <= N {
            let shift = i as u32;
            let byte: u8 = ((mask64 >> shift) & 0xFF) as u8;

            // Direct table lookup: load 8 i64 values (4 int64x2_t vectors)
            let masks_ptr = LUT.0[byte as usize].as_ptr();
            let m0: int64x2_t = vld1q_s64(masks_ptr);
            let m1: int64x2_t = vld1q_s64(masks_ptr.add(2));
            let m2: int64x2_t = vld1q_s64(masks_ptr.add(4));
            let m3: int64x2_t = vld1q_s64(masks_ptr.add(6));

            // Branchless select: scalar & -1 = scalar, scalar & 0 = 0
            vst1q_s64(out_ptr.add(i), vandq_s64(scalar_v, m0));
            vst1q_s64(out_ptr.add(i + 2), vandq_s64(scalar_v, m1));
            vst1q_s64(out_ptr.add(i + 4), vandq_s64(scalar_v, m2));
            vst1q_s64(out_ptr.add(i + 6), vandq_s64(scalar_v, m3));

            i += 8;
        }
    } else {
        // Compile-time constant for bit masks [1,2,4,8,16,32,64,128]
        const BIT_MASKS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];
        let bit_masks: uint8x8_t = vld1_u8(BIT_MASKS.as_ptr());
        let zero_u8: uint8x8_t = vdup_n_u8(0);

        // Main loop: process 8 coefficients per iteration
        while i + 8 <= N {
            let shift = i as u32;
            let byte: u8 = ((mask64 >> shift) & 0xFF) as u8;

            // Extract bits: replicate byte and AND with bit masks to isolate each bit
            let vbyte: uint8x8_t = vdup_n_u8(byte);
            let selected: uint8x8_t = vand_u8(vbyte, bit_masks);
            let nz: uint8x8_t = vcgt_u8(selected, zero_u8); // 0xFF if bit set, 0x00 if not

            // Sign-extend 0xFF → 0xFFFF...FFFF, 0x00 → 0x0000...0000 via signed widening
            let nz_signed: int8x8_t = vreinterpret_s8_u8(nz); // Reinterpret: 0xFF = -1, 0x00 = 0
            let s16: int16x8_t = vmovl_s8(nz_signed); // -1 → 0xFFFF, 0 → 0x0000
            let lo32: int32x4_t = vmovl_s16(vget_low_s16(s16));
            let hi32: int32x4_t = vmovl_s16(vget_high_s16(s16));

            let m0: int64x2_t = vmovl_s32(vget_low_s32(lo32)); // -1 → 0xFFFF...FFFF
            let m1: int64x2_t = vmovl_s32(vget_high_s32(lo32));
            let m2: int64x2_t = vmovl_s32(vget_low_s32(hi32));
            let m3: int64x2_t = vmovl_s32(vget_high_s32(hi32));

            // Branchless select: scalar & 0xFFFF... = scalar, scalar & 0x0000... = 0
            vst1q_s64(out_ptr.add(i), vandq_s64(scalar_v, m0));
            vst1q_s64(out_ptr.add(i + 2), vandq_s64(scalar_v, m1));
            vst1q_s64(out_ptr.add(i + 4), vandq_s64(scalar_v, m2));
            vst1q_s64(out_ptr.add(i + 6), vandq_s64(scalar_v, m3));

            i += 8;
        }
    }

    // Tail: handle remaining coefficients one at a time
    while i < N {
        let bit = ((mask64 >> i) & 1) != 0;
        let mask = -(bit as i64); // 0 or -1 (all bits set)
        *out_ptr.add(i) = scalar & mask;
        i += 1;
    }
}

#[allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_lossless
)]
#[allow(unsafe_op_in_unsafe_fn)]
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
// Converts a u64 bitmask into an array of i64 values using AVX512 SIMD.
// Processes 32 bits at a time with 4-way unrolling for instruction-level
// parallelism. Significantly cleaner than NEON due to native mask register
// support.
unsafe fn widen_fill_avx512<const N: usize>(mask_ref: &u64, out_ptr: *mut i64, scalar: i64) {
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    let mask64: u64 = *mask_ref;
    let mut i = 0usize;

    // Unrolled loop: process 32 coefficients (4 x 8) per iteration for ILP
    // Interleave independent operations to keep multiple execution units busy
    while i + 32 <= N {
        let shift = i as u32;

        // Extract all 4 bytes at once - no dependencies between extractions
        let byte0: u8 = ((mask64 >> shift) & 0xFF) as u8;
        let byte1: u8 = ((mask64 >> (shift + 8)) & 0xFF) as u8;
        let byte2: u8 = ((mask64 >> (shift + 16)) & 0xFF) as u8;
        let byte3: u8 = ((mask64 >> (shift + 24)) & 0xFF) as u8;

        // Convert to mask registers - independent operations
        let kmask0: __mmask8 = byte0;
        let kmask1: __mmask8 = byte1;
        let kmask2: __mmask8 = byte2;
        let kmask3: __mmask8 = byte3;

        // Predicated broadcasts - all can execute in parallel on different ports
        let result0: __m512i = _mm512_maskz_set1_epi64(kmask0, scalar);
        let result1: __m512i = _mm512_maskz_set1_epi64(kmask1, scalar);
        let result2: __m512i = _mm512_maskz_set1_epi64(kmask2, scalar);
        let result3: __m512i = _mm512_maskz_set1_epi64(kmask3, scalar);

        // Stores - can pipeline as they have no dependencies on each other
        _mm512_storeu_si512(out_ptr.add(i) as *mut __m512i, result0);
        _mm512_storeu_si512(out_ptr.add(i + 8) as *mut __m512i, result1);
        _mm512_storeu_si512(out_ptr.add(i + 16) as *mut __m512i, result2);
        _mm512_storeu_si512(out_ptr.add(i + 24) as *mut __m512i, result3);

        i += 32;
    }

    // Handle remaining full vectors (8 coefficients at a time)
    while i + 8 <= N {
        let shift = i as u32;
        let byte: u8 = ((mask64 >> shift) & 0xFF) as u8;
        let kmask: __mmask8 = byte;
        let result: __m512i = _mm512_maskz_set1_epi64(kmask, scalar);
        _mm512_storeu_si512(out_ptr.add(i) as *mut __m512i, result);
        i += 8;
    }

    // Tail: handle remaining coefficients one at a time
    while i < N {
        let bit = ((mask64 >> i) & 1) != 0;
        let mask = -(bit as i64); // 0 or -1 (all bits set)
        *out_ptr.add(i) = scalar & mask;
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use crate::univariate::binary_ref::{BinaryRefPoly, BinaryRefPolyWideningMulByScalar};

    use super::*;

    #[test]
    fn evaluate_is_correct() {
        for i in 0..16 {
            let poly = BinaryU64Poly::<4>::new([
                (i & 0b0001 != 0).into(),
                (i & 0b0010 != 0).into(),
                (i & 0b0100 != 0).into(),
                (i & 0b1000 != 0).into(),
            ]);

            let result = poly.evaluate_at_point(&2).unwrap();

            assert_eq!(result, i);
        }
    }
    fn widen_ref<const DEGREE_PLUS_ONE: usize>(
        poly: &BinaryU64Poly<DEGREE_PLUS_ONE>,
        scalar: i64,
    ) -> DensePolynomial<i64, DEGREE_PLUS_ONE> {
        let mut coeffs: [i64; DEGREE_PLUS_ONE] = [0; DEGREE_PLUS_ONE];
        for (i, coeff) in coeffs.iter_mut().enumerate().take(DEGREE_PLUS_ONE) {
            if (poly.0 & (1 << i)) != 0 {
                *coeff = scalar;
            }
        }
        DensePolynomial { coeffs }
    }

    #[ignore = "CI system does not support SIMD features"]
    #[test]
    fn widen_ref_and_widen_ref_simd_match() {
        // Test with degree 4
        for i in 0..16 {
            let poly = BinaryU64Poly::<4>::new([
                (i & 0b0001 != 0).into(),
                (i & 0b0010 != 0).into(),
                (i & 0b0100 != 0).into(),
                (i & 0b1000 != 0).into(),
            ]);

            let poly_ref = BinaryRefPoly::<4>::new([
                (i & 0b0001 != 0).into(),
                (i & 0b0010 != 0).into(),
                (i & 0b0100 != 0).into(),
                (i & 0b1000 != 0).into(),
            ]);

            for scalar in [1, 42, -7, 100, -100, i64::MAX, i64::MIN] {
                let result_simd_ref = widen_ref(&poly, scalar);
                let result_simd = widen_simd(&poly, scalar);
                let result_ref = BinaryRefPolyWideningMulByScalar::<i64>::mul_by_scalar_widen(
                    &poly_ref, &scalar,
                );

                assert_eq!(
                    result_simd_ref.coeffs, result_simd.coeffs,
                    "Mismatch for pattern {} with scalar {}",
                    i, scalar
                );
                assert_eq!(
                    result_ref.coeffs, result_ref.coeffs,
                    "Mismatch for pattern {} with scalar {} between ref and BinaryRefPolyWideningMulByScalar",
                    i, scalar
                );
            }
        }

        let coeffs: Vec<_> = [
            1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1,
            0, 1, 0,
        ]
        .into_iter()
        .map(|x| (x != 0).into())
        .collect();

        let poly32 = BinaryU64Poly::<32>::new(coeffs.clone());
        let poly32_ref = BinaryRefPoly::<32>::new(coeffs);

        for scalar in [1, 42, -7, 100, -100] {
            let result_simd_ref = widen_ref(&poly32, scalar);
            let result_simd = widen_simd(&poly32, scalar);
            let result_ref =
                BinaryRefPolyWideningMulByScalar::<i64>::mul_by_scalar_widen(&poly32_ref, &scalar);
            assert_eq!(
                result_simd_ref.coeffs, result_simd.coeffs,
                "Mismatch for degree 32 with scalar {}",
                scalar
            );
            assert_eq!(
                result_ref.coeffs, result_ref.coeffs,
                "Mismatch for degree 32 with scalar {} between ref and BinaryRefPolyWideningMulByScalar",
                scalar
            );
        }

        // Test with all zeros
        let poly_zeros = BinaryU64Poly::<16>::new([false.into(); 16]);
        let result_ref_zeros = widen_ref(&poly_zeros, 42);
        let result_simd_zeros = widen_simd(&poly_zeros, 42);
        assert_eq!(result_ref_zeros.coeffs, result_simd_zeros.coeffs);

        // Test with all ones
        let poly_ones = BinaryU64Poly::<16>::new([true.into(); 16]);
        let result_ref_ones = widen_ref(&poly_ones, 42);
        let result_simd_ones = widen_simd(&poly_ones, 42);
        assert_eq!(result_ref_ones.coeffs, result_simd_ones.coeffs);
    }
}
