#[allow(unused)]
use ark_ff::ark_ff_macros::unroll_for_loops;
use ark_ff::const_for;
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use ark_std::ops::{Index, IndexMut, Range, RangeTo};
use ark_std::{
    borrow::Borrow,
    // convert::TryFrom,
    fmt::{Debug, Display, UpperHex},
    io::{Read, Write},
    ops::{
        BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, ShlAssign, Shr,
        ShrAssign,
    },
    rand::{
        distributions::{Distribution, Standard},
        Rng,
    },
    str::FromStr,
    vec::Vec,
    Zero,
};
use num_bigint::BigUint;
use zeroize::Zeroize;

use crate::{
    adc,
    const_helpers::SerBuffer,
    field::{config, Int},
    traits::{BigInteger, FromBytes, Integer, Uinteger},
};

#[macro_use]
pub mod arithmetic;
mod bits;
#[derive(Copy, Clone, PartialEq, Eq, Hash, Zeroize)]
pub struct BigInt<const N: usize>([u64; N]);

impl<const N: usize> From<[u64; N]> for BigInt<N> {
    #[inline]
    fn from(value: [u64; N]) -> Self {
        Self(value)
    }
}

impl<const N: usize> From<SerBuffer<N>> for BigInt<N> {
    #[inline]
    fn from(value: SerBuffer<N>) -> Self {
        let mut self_integer = BigInt::from(0u64);
        self_integer
            .0
            .iter_mut()
            .zip(value.buffers)
            .for_each(|(other, this)| *other = u64::from_le_bytes(this));
        self_integer
    }
}

impl<const N: usize> Default for BigInt<N> {
    fn default() -> Self {
        Self([0u64; N])
    }
}

impl<const N: usize> CanonicalSerialize for BigInt<N> {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.0.serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.0.serialized_size(compress)
    }
}

impl<const N: usize> Valid for BigInt<N> {
    fn check(&self) -> Result<(), SerializationError> {
        self.0.check()
    }
}

impl<const N: usize> CanonicalDeserialize for BigInt<N> {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        Ok(BigInt(<[u64; N]>::deserialize_with_mode(
            reader, compress, validate,
        )?))
    }
}

/// Construct a [`struct@BigInt<N>`] element from a literal string.
///
/// # Panics
///
/// If the integer represented by the string cannot fit in the number
/// of limbs of the `BigInt`, this macro results in a
/// * compile-time error if used in a const context
/// * run-time error otherwise.
///
/// # Usage
/// ```rust
/// # use ark_ff::BigInt;
/// const ONE: BigInt<6> = BigInt!("1");
///
/// fn check_correctness() {
///     assert_eq!(ONE, BigInt::from(1u8));
/// }
/// ```
#[macro_export]
macro_rules! BigInt {
    ($c0:expr) => {{
        let (is_positive, limbs) = $crate::ark_ff_macros::to_sign_and_limbs!($c0);
        assert!(is_positive);
        let mut integer = $crate::BigInt::zero();
        assert!(integer.0.len() >= limbs.len());
        $crate::const_for!((i in 0..(limbs.len())) {
            integer.0[i] = limbs[i];
        });
        integer
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! const_modulo {
    ($a:expr, $divisor:expr) => {{
        // Stupid slow base-2 long division taken from
        // https://en.wikipedia.org/wiki/Division_algorithm
        assert!(!$divisor.const_is_zero());
        let mut remainder = BigInt::new([0u64; N]);
        let mut i = ($a.num_bits() - 1) as isize;
        let mut carry;
        while i >= 0 {
            (remainder, carry) = remainder.const_mul2_with_carry();
            remainder.0[0] |= $a.get_bit(i as usize) as u64;
            if remainder.const_geq($divisor) || carry {
                let (r, borrow) = remainder.const_sub_with_borrow($divisor);
                remainder = r;
                assert!(borrow == carry);
            }
            i -= 1;
        }
        remainder
    }};
}

impl<const N: usize> BigInt<N> {
    pub const fn new(value: [u64; N]) -> Self {
        Self(value)
    }

    pub const fn zero() -> Self {
        Self([0u64; N])
    }

    pub const fn one() -> Self {
        let mut one = Self::zero();
        one.0[0] = 1;
        one
    }

    #[doc(hidden)]
    pub const fn const_is_even(&self) -> bool {
        self.0[0].is_multiple_of(2)
    }

    #[doc(hidden)]
    pub const fn const_is_odd(&self) -> bool {
        !self.const_is_even()
    }

    #[doc(hidden)]
    pub const fn mod_4(&self) -> u8 {
        // To compute n % 4, we need to simply look at the
        // 2 least significant bits of n, and check their value mod 4.
        (((self.0[0] << 62) >> 62) % 4) as u8
    }

    /// Compute a right shift of `self`
    /// This is equivalent to a (saturating) division by 2.
    #[doc(hidden)]
    pub const fn const_shr(&self) -> Self {
        let mut result = *self;
        let mut t = 0;
        crate::const_for!((i in 0..N) {
            let a = result.0[N - i - 1];
            let t2 = a << 63;
            result.0[N - i - 1] >>= 1;
            result.0[N - i - 1] |= t;
            t = t2;
        });
        result
    }

    pub(crate) const fn const_geq(&self, other: &Self) -> bool {
        const_for!((i in 0..N) {
            let a = self.0[N - i - 1];
            let b = other.0[N - i - 1];
            if a < b {
                return false;
            } else if a > b {
                return true;
            }
        });
        true
    }

    /// Compute the largest integer `s` such that `self = 2**s * t + 1` for odd `t`.
    #[doc(hidden)]
    pub const fn two_adic_valuation(mut self) -> u32 {
        assert!(self.const_is_odd());
        let mut two_adicity = 0;
        // Since `self` is odd, we can always subtract one
        // without a borrow
        self.0[0] -= 1;
        while self.const_is_even() {
            self = self.const_shr();
            two_adicity += 1;
        }
        two_adicity
    }

    /// Compute the smallest odd integer `t` such that `self = 2**s * t + 1` for some
    /// integer `s = self.two_adic_valuation()`.
    #[doc(hidden)]
    pub const fn two_adic_coefficient(mut self) -> Self {
        assert!(self.const_is_odd());
        // Since `self` is odd, we can always subtract one
        // without a borrow
        self.0[0] -= 1;
        while self.const_is_even() {
            self = self.const_shr();
        }
        assert!(self.const_is_odd());
        self
    }

    /// Divide `self` by 2, rounding down if necessary.
    /// That is, if `self.is_odd()`, compute `(self - 1)/2`.
    /// Else, compute `self/2`.
    #[doc(hidden)]
    pub const fn divide_by_2_round_down(mut self) -> Self {
        if self.const_is_odd() {
            self.0[0] -= 1;
        }
        self.const_shr()
    }

    /// Find the number of bits in the binary decomposition of `self`.
    #[doc(hidden)]
    pub const fn const_num_bits(self) -> u32 {
        ((N - 1) * 64) as u32 + (64 - self.0[N - 1].leading_zeros())
    }

    #[inline]
    pub(crate) const fn const_sub_with_borrow(mut self, other: &Self) -> (Self, bool) {
        let mut borrow = 0;

        const_for!((i in 0..N) {
            self.0[i] = sbb!(self.0[i], other.0[i], &mut borrow);
        });

        (self, borrow != 0)
    }

    pub(crate) const fn const_mul2_with_carry(mut self) -> (Self, bool) {
        let mut last = 0;
        crate::const_for!((i in 0..N) {
            let a = self.0[i];
            let tmp = a >> 63;
            self.0[i] <<= 1;
            self.0[i] |= last;
            last = tmp;
        });
        (self, last != 0)
    }

    pub(crate) const fn const_is_zero(&self) -> bool {
        let mut is_zero = true;
        crate::const_for!((i in 0..N) {
            is_zero &= self.0[i] == 0;
        });
        is_zero
    }

    /// Computes the Montgomery R constant modulo `self`.
    #[doc(hidden)]
    pub const fn montgomery_r(&self) -> Self {
        let two_pow_n_times_64 = crate::const_helpers::RBuffer([0u64; N], 1);
        const_modulo!(two_pow_n_times_64, self)
    }

    /// Computes the Montgomery R2 constant modulo `self`.
    #[doc(hidden)]
    pub const fn montgomery_r2(&self) -> Self {
        let two_pow_n_times_64_square = crate::const_helpers::R2Buffer([0u64; N], [0u64; N], 1);
        const_modulo!(two_pow_n_times_64_square, self)
    }

    pub const fn has_spare_bit(&self) -> bool {
        self.0[N - 1] >> 63 == 0
    }

    pub const fn first(&self) -> u64 {
        self.0[0]
    }
}

impl<const N: usize> BigInt<N> {
    #[unroll_for_loops(6)]
    #[inline]
    pub fn add_with_carry(&mut self, other: &Self) -> bool {
        let mut carry = 0;

        for i in 0..N {
            carry = arithmetic::adc_for_add_with_carry(&mut self.0[i], other.0[i], carry);
        }

        carry != 0
    }

    #[unroll_for_loops(6)]
    #[inline]
    pub fn sub_with_borrow(&mut self, other: &Self) -> bool {
        let mut borrow = 0;

        for i in 0..N {
            borrow = arithmetic::sbb_for_sub_with_borrow(&mut self.0[i], other.0[i], borrow);
        }

        borrow != 0
    }

    #[inline]
    #[allow(unused)]
    pub fn mul2(&mut self) -> bool {
        #[cfg(all(target_arch = "x86_64", feature = "asm"))]
        {
            let mut carry = 0;

            for i in 0..N {
                unsafe {
                    use core::arch::x86_64::_addcarry_u64;
                    carry = _addcarry_u64(carry, self.0[i], self.0[i], &mut self.0[i])
                };
            }

            carry != 0
        }

        #[cfg(not(all(target_arch = "x86_64", feature = "asm")))]
        {
            let mut last = 0;
            for i in 0..N {
                let a = &mut self.0[i];
                let tmp = *a >> 63;
                *a <<= 1;
                *a |= last;
                last = tmp;
            }
            last != 0
        }
    }

    #[inline]
    pub fn muln(&mut self, mut n: u32) {
        if n >= (64 * N) as u32 {
            *self = Self::from(0u64);
            return;
        }

        while n >= 64 {
            let mut t = 0;
            for i in 0..N {
                core::mem::swap(&mut t, &mut self.0[i]);
            }
            n -= 64;
        }

        if n > 0 {
            let mut t = 0;
            #[allow(unused)]
            for i in 0..N {
                let a = &mut self.0[i];
                let t2 = *a >> (64 - n);
                *a <<= n;
                *a |= t;
                t = t2;
            }
        }
    }

    #[inline]
    pub fn mul(&self, other: &Self) -> (Self, Self) {
        if self.is_zero() || other.is_zero() {
            let zero = Self::zero();
            return (zero, zero);
        }

        let mut r = crate::const_helpers::MulBuffer::zeroed();

        let mut carry = 0;

        for i in 0..N {
            for j in 0..N {
                r[i + j] = mac_with_carry!(r[i + j], self.0[i], other.0[j], &mut carry);
            }
            r.b1[i] = carry;
            carry = 0;
        }

        (Self(r.b0), Self(r.b1))
    }

    #[inline]
    pub fn mul_low(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }

        let mut res = Self::zero();
        let mut carry = 0;

        for i in 0..N {
            for j in 0..(N - i) {
                res.0[i + j] = mac_with_carry!(res.0[i + j], self.0[i], other.0[j], &mut carry);
            }
            carry = 0;
        }

        res
    }

    #[inline]
    pub fn mul_high(&self, other: &Self) -> Self {
        self.mul(other).1
    }

    #[inline]
    pub fn mul_naive(&self, other: &Self) -> (Self, Self) {
        let (mut lo, mut hi) = (Self::zero(), Self::zero());
        crate::const_for!((i in 0..N) {
            let mut carry = 0;
            crate::const_for!((j in 0..N) {
                let k = i + j;
                if k >= N {
                    hi.0[k - N] = mac_with_carry!(hi.0[k - N], self.0[i], other.0[j], &mut carry);
                } else {
                    lo.0[k] = mac_with_carry!(lo.0[k], self.0[i], other.0[j], &mut carry);
                }
            });
            hi.0[i] = carry;
        });

        (lo, hi)
    }

    #[inline]
    pub fn div2(&mut self) {
        let mut t = 0;
        for a in self.0.iter_mut().rev() {
            let t2 = *a << 63;
            *a >>= 1;
            *a |= t;
            t = t2;
        }
    }

    #[inline]
    pub fn divn(&mut self, mut n: u32) {
        if n >= (64 * N) as u32 {
            *self = Self::from(0u64);
            return;
        }

        while n >= 64 {
            let mut t = 0;
            for i in 0..N {
                core::mem::swap(&mut t, &mut self.0[N - i - 1]);
            }
            n -= 64;
        }

        if n > 0 {
            let mut t = 0;
            #[allow(unused)]
            for i in 0..N {
                let a = &mut self.0[N - i - 1];
                let t2 = *a << (64 - n);
                *a >>= n;
                *a |= t;
                t = t2;
            }
        }
    }

    #[inline]
    pub fn is_odd(&self) -> bool {
        self.0[0] & 1 == 1
    }

    #[inline]
    pub fn is_even(&self) -> bool {
        !self.is_odd()
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.0.iter().all(Zero::is_zero)
    }

    #[inline]
    pub fn get_bit(&self, i: usize) -> bool {
        if i >= 64 * N {
            false
        } else {
            let limb = i / 64;
            let bit = i - (64 * limb);
            (self.0[limb] & (1 << bit)) != 0
        }
    }

    #[inline]
    pub fn montgomery_reduction(
        &mut self,
        lo: &mut Self,
        hi: &mut Self,
        modulus: &Self,
        inv: u64,
    ) -> bool {
        let mut carry2 = 0;
        crate::const_for!((i in 0..N) {
            let tmp = lo.0[i].wrapping_mul(inv);
            let mut carry;
            mac!(lo.0[i], tmp, modulus.0[0], &mut carry);
            crate::const_for!((j in 1..N) {
                let k = i + j;
                if k >= N {
                    hi.0[k - N] = mac_with_carry!(hi.0[k - N], tmp, modulus.0[j], &mut carry);
                }  else {
                    lo.0[k] = mac_with_carry!(lo.0[k], tmp, modulus.0[j], &mut carry);
                }
            });
            hi.0[i] = adc!(hi.0[i], carry, &mut carry2);
        });

        crate::const_for!((i in 0..N) {
            self.0[i] = hi.0[i];
        });

        carry2 != 0
    }

    #[inline]
    pub fn demontgomery(&self, modulus: &Self, inv: u64) -> Self {
        let mut r = self.0;
        // Montgomery Reduction
        for i in 0..N {
            let k = r[i].wrapping_mul(inv);
            let mut carry = 0;

            config::mac_with_carry(r[i], k, modulus.0[0], &mut carry);
            for j in 1..N {
                r[(j + i) % N] =
                    config::mac_with_carry(r[(j + i) % N], k, modulus.0[j], &mut carry);
            }
            r[i % N] = carry;
        }

        BigInt::new(r)
    }

    #[inline]
    pub fn last_mut(&mut self) -> &mut u64 {
        &mut self.0[N - 1]
    }
}

impl<const N: usize> BigInteger for BigInt<N> {
    type W = Words<N>;
    fn to_words(&self) -> Words<N> {
        Words(self.0)
    }

    fn one() -> Self {
        Self::one()
    }

    #[inline]
    fn from_bits_be(bits: &[bool]) -> Self {
        let mut bits = bits.to_vec();
        bits.reverse();
        Self::from_bits_le(&bits)
    }

    fn from_bits_le(bits: &[bool]) -> Self {
        let mut res = Self::zero();
        for (bits64, res_i) in bits.chunks(64).zip(&mut res.0) {
            for (i, bit) in bits64.iter().enumerate() {
                *res_i |= (*bit as u64) << i;
            }
        }
        res
    }

    #[inline]
    fn num_bits(&self) -> u32 {
        let mut ret = N as u32 * 64;
        for i in self.0.iter().rev() {
            let leading = i.leading_zeros();
            ret -= leading;
            if leading != 64 {
                break;
            }
        }

        ret
    }

    fn new(words: Words<N>) -> Self {
        Self(words.0)
    }

    #[inline]
    fn to_bytes_be(self) -> Vec<u8> {
        let mut le_bytes = self.to_bytes_le();
        le_bytes.reverse();
        le_bytes
    }

    #[inline]
    fn to_bytes_le(self) -> Vec<u8> {
        self.0.iter().flat_map(|&limb| limb.to_le_bytes()).collect()
    }
}

impl<const N: usize> UpperHex for BigInt<N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:016X}", BigUint::from(*self))
    }
}

impl<const N: usize> Debug for BigInt<N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", BigUint::from(*self))
    }
}

impl<const N: usize> Display for BigInt<N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", BigUint::from(*self))
    }
}

impl<const N: usize> Ord for BigInt<N> {
    #[inline]
    #[cfg_attr(target_arch = "x86_64", unroll_for_loops(12))]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        use core::cmp::Ordering;
        #[cfg(target_arch = "x86_64")]
        for i in 0..N {
            let a = &self.0[N - i - 1];
            let b = &other.0[N - i - 1];
            match a.cmp(b) {
                Ordering::Equal => {}
                order => return order,
            };
        }
        #[cfg(not(target_arch = "x86_64"))]
        for (a, b) in self.0.iter().rev().zip(other.0.iter().rev()) {
            if let order @ (Ordering::Less | Ordering::Greater) = a.cmp(b) {
                return order;
            }
        }
        Ordering::Equal
    }
}

impl<const N: usize> PartialOrd for BigInt<N> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<const N: usize> Distribution<BigInt<N>> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> BigInt<N> {
        BigInt::from([(); N].map(|_| rng.r#gen()))
    }
}

impl<const N: usize> AsMut<[u64]> for BigInt<N> {
    #[inline]
    fn as_mut(&mut self) -> &mut [u64] {
        &mut self.0
    }
}

impl<const N: usize> AsRef<[u64]> for BigInt<N> {
    #[inline]
    fn as_ref(&self) -> &[u64] {
        &self.0
    }
}

macro_rules! impl_from_uint {
    ($type:ty) => {
        impl<const N: usize> From<$type> for BigInt<N> {
            #[inline]
            fn from(val: $type) -> BigInt<N> {
                let mut repr = Self::default();
                repr.0[0] = val.into();
                repr
            }
        }
    };
}

impl_from_uint!(u64);
impl_from_uint!(u32);
impl_from_uint!(u16);
impl_from_uint!(u8);

impl<const N: usize> From<u128> for BigInt<N> {
    #[inline]
    fn from(val: u128) -> BigInt<N> {
        if N < 2 {
            panic!("Integer is 128 bits but field is 64 bits");
        }

        let mut repr = Self::default();
        repr.0[0] = val as u64;
        repr.0[1] = (val >> 64) as u64;
        repr
    }
}

impl<const N: usize> TryFrom<BigUint> for BigInt<N> {
    type Error = ();

    /// Returns `Err(())` if the bit size of `val` is more than `N * 64`.
    #[inline]
    fn try_from(val: num_bigint::BigUint) -> Result<BigInt<N>, Self::Error> {
        let bytes = val.to_bytes_le();

        if bytes.len() > N * 8 {
            Err(())
        } else {
            let mut limbs = [0u64; N];

            bytes.chunks(8).enumerate().for_each(|(i, chunk)| {
                let mut chunk_padded = [0u8; 8];
                chunk_padded[..chunk.len()].copy_from_slice(chunk);
                limbs[i] = u64::from_le_bytes(chunk_padded)
            });

            Ok(Self(limbs))
        }
    }
}

impl<const N: usize> FromStr for BigInt<N> {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let biguint = BigUint::from_str(s).unwrap();
        Self::try_from(biguint)
    }
}

impl<const N: usize> From<BigInt<N>> for BigUint {
    #[inline]
    fn from(val: BigInt<N>) -> num_bigint::BigUint {
        BigUint::from_bytes_le(&val.to_bytes_le())
    }
}

impl<const N: usize> From<BigInt<N>> for num_bigint::BigInt {
    #[inline]
    fn from(val: BigInt<N>) -> num_bigint::BigInt {
        use num_bigint::Sign;
        let sign = if val.is_zero() {
            Sign::NoSign
        } else {
            Sign::Plus
        };
        num_bigint::BigInt::from_bytes_le(sign, &val.to_bytes_le())
    }
}

// Only returns the absolute value for the integer
impl<const M: usize, const N: usize> From<Int<M>> for BigInt<N> {
    fn from(value: Int<M>) -> Self {
        Self::from(&value)
    }
}
impl<const M: usize, const N: usize> From<&Int<M>> for BigInt<N> {
    fn from(value: &Int<M>) -> Self {
        let abs = value.abs();
        let words = abs.to_words();
        let min_width = M.min(N);

        let mut result = [0u64; N];
        result[..min_width].copy_from_slice(&words[..min_width]);

        BigInt(result)
    }
}

impl<const M: usize, const N: usize> From<BigInt<N>> for Int<M> {
    fn from(value: BigInt<N>) -> Int<M> {
        (&value).into()
    }
}

impl<const M: usize, const N: usize> From<&BigInt<N>> for Int<M> {
    fn from(value: &BigInt<N>) -> Int<M> {
        let words = value.to_words();
        let min_width = M.min(N);

        let mut result = [0u64; M];
        result[..min_width].copy_from_slice(&words[..min_width]);

        Int::from(result)
    }
}

impl<B: Borrow<Self>, const N: usize> BitXorAssign<B> for BigInt<N> {
    fn bitxor_assign(&mut self, rhs: B) {
        (0..N).for_each(|i| self.0[i] ^= rhs.borrow().0[i])
    }
}

impl<B: Borrow<Self>, const N: usize> BitXor<B> for BigInt<N> {
    type Output = Self;

    fn bitxor(mut self, rhs: B) -> Self::Output {
        self ^= rhs;
        self
    }
}

impl<B: Borrow<Self>, const N: usize> BitAndAssign<B> for BigInt<N> {
    fn bitand_assign(&mut self, rhs: B) {
        (0..N).for_each(|i| self.0[i] &= rhs.borrow().0[i])
    }
}

impl<B: Borrow<Self>, const N: usize> BitAnd<B> for BigInt<N> {
    type Output = Self;

    fn bitand(mut self, rhs: B) -> Self::Output {
        self &= rhs;
        self
    }
}

impl<B: Borrow<Self>, const N: usize> BitOrAssign<B> for BigInt<N> {
    fn bitor_assign(&mut self, rhs: B) {
        (0..N).for_each(|i| self.0[i] |= rhs.borrow().0[i])
    }
}

impl<B: Borrow<Self>, const N: usize> BitOr<B> for BigInt<N> {
    type Output = Self;

    fn bitor(mut self, rhs: B) -> Self::Output {
        self |= rhs;
        self
    }
}

impl<const N: usize> ShrAssign<u32> for BigInt<N> {
    /// Computes the bitwise shift right operation in place.
    ///
    /// Differently from the built-in numeric types (u8, u32, u64, etc.) this
    /// operation does *not* return an underflow error if the number of bits
    /// shifted is larger than N * 64. Instead the result will be saturated to
    /// zero.
    fn shr_assign(&mut self, mut rhs: u32) {
        if rhs >= (64 * N) as u32 {
            *self = Self::from(0u64);
            return;
        }

        while rhs >= 64 {
            let mut t = 0;
            for limb in self.0.iter_mut().rev() {
                core::mem::swap(&mut t, limb);
            }
            rhs -= 64;
        }

        if rhs > 0 {
            let mut t = 0;
            for a in self.0.iter_mut().rev() {
                let t2 = *a << (64 - rhs);
                *a >>= rhs;
                *a |= t;
                t = t2;
            }
        }
    }
}

impl<const N: usize> ShrAssign<usize> for BigInt<N> {
    /// Computes the bitwise shift right operation in place.
    ///
    /// Differently from the built-in numeric types (u8, u32, u64, etc.) this
    /// operation does *not* return an underflow error if the number of bits
    /// shifted is larger than N * 64. Instead the result will be saturated to
    /// zero.
    fn shr_assign(&mut self, mut rhs: usize) {
        if rhs >= (64 * N) {
            *self = Self::from(0u64);
            return;
        }

        while rhs >= 64 {
            let mut t = 0;
            for limb in self.0.iter_mut().rev() {
                core::mem::swap(&mut t, limb);
            }
            rhs -= 64;
        }

        if rhs > 0 {
            let mut t = 0;
            for a in self.0.iter_mut().rev() {
                let t2 = *a << (64 - rhs);
                *a >>= rhs;
                *a |= t;
                t = t2;
            }
        }
    }
}

impl<const N: usize> Shr<u32> for BigInt<N> {
    type Output = Self;

    /// Computes bitwise shift right operation.
    ///
    /// Differently from the built-in numeric types (u8, u32, u64, etc.) this
    /// operation does *not* return an underflow error if the number of bits
    /// shifted is larger than N * 64. Instead the result will be saturated to
    /// zero.
    fn shr(mut self, rhs: u32) -> Self::Output {
        self >>= rhs;
        self
    }
}

impl<const N: usize> ShlAssign<u32> for BigInt<N> {
    /// Computes the bitwise shift left operation in place.
    ///
    /// Differently from the built-in numeric types (u8, u32, u64, etc.) this
    /// operation does *not* return an overflow error if the number of bits
    /// shifted is larger than N * 64. Instead, the overflow will be chopped
    /// off.
    fn shl_assign(&mut self, mut rhs: u32) {
        if rhs >= (64 * N) as u32 {
            *self = Self::from(0u64);
            return;
        }

        while rhs >= 64 {
            let mut t = 0;
            for i in 0..N {
                core::mem::swap(&mut t, &mut self.0[i]);
            }
            rhs -= 64;
        }

        if rhs > 0 {
            let mut t = 0;
            #[allow(unused)]
            for i in 0..N {
                let a = &mut self.0[i];
                let t2 = *a >> (64 - rhs);
                *a <<= rhs;
                *a |= t;
                t = t2;
            }
        }
    }
}

impl<const N: usize> ShlAssign<usize> for BigInt<N> {
    /// Computes the bitwise shift left operation in place.
    ///
    /// Differently from the built-in numeric types (u8, u32, u64, etc.) this
    /// operation does *not* return an overflow error if the number of bits
    /// shifted is larger than N * 64. Instead, the overflow will be chopped
    /// off.
    fn shl_assign(&mut self, mut rhs: usize) {
        if rhs >= (64 * N) {
            *self = Self::from(0u64);
            return;
        }

        while rhs >= 64 {
            let mut t = 0;
            for i in 0..N {
                core::mem::swap(&mut t, &mut self.0[i]);
            }
            rhs -= 64;
        }

        if rhs > 0 {
            let mut t = 0;
            #[allow(unused)]
            for i in 0..N {
                let a = &mut self.0[i];
                let t2 = *a >> (64 - rhs);
                *a <<= rhs;
                *a |= t;
                t = t2;
            }
        }
    }
}

impl<const N: usize> Shl<u32> for BigInt<N> {
    type Output = Self;

    /// Computes the bitwise shift left operation in place.
    ///
    /// Differently from the built-in numeric types (u8, u32, u64, etc.) this
    /// operation does *not* return an overflow error if the number of bits
    /// shifted is larger than N * 64. Instead, the overflow will be chopped
    /// off.
    fn shl(mut self, rhs: u32) -> Self::Output {
        self <<= rhs;
        self
    }
}

impl<const N: usize> Not for BigInt<N> {
    type Output = Self;

    fn not(self) -> Self::Output {
        let mut result = Self::zero();
        for i in 0..N {
            result.0[i] = !self.0[i];
        }
        result
    }
}

impl<const N: usize> FromBytes for BigInt<N> {
    fn from_bytes_le(bytes: &[u8]) -> Option<Self> {
        const LIMB_SIZE: usize = size_of::<u64>();
        if bytes.len() > N * LIMB_SIZE {
            return None;
        }

        let mut limbs = [0u64; N];

        // Process byte chunks, handling cases where chunk < 8 bytes
        for (i, chunk) in bytes.chunks(LIMB_SIZE).enumerate() {
            let mut padded_chunk = [0u8; LIMB_SIZE];

            // Copy bytes aligning to the least significant bytes
            padded_chunk[..chunk.len()].copy_from_slice(chunk);

            limbs[i] = u64::from_le_bytes(padded_chunk);
        }

        Some(Self(limbs))
    }

    fn from_bytes_be(bytes: &[u8]) -> Option<Self> {
        const LIMB_SIZE: usize = size_of::<u64>();
        if bytes.len() > N * LIMB_SIZE {
            return None;
        }

        let mut limbs = [0u64; N];

        // Process byte chunks, handling cases where chunk < 8 bytes
        for (i, chunk) in bytes.chunks(LIMB_SIZE).rev().enumerate() {
            let mut padded_chunk = [0u8; LIMB_SIZE];

            // Copy bytes aligning to the most significant bytes
            let start_idx = LIMB_SIZE.saturating_sub(chunk.len());

            padded_chunk[start_idx..].copy_from_slice(chunk);

            limbs[i] = u64::from_be_bytes(padded_chunk);
        }

        Some(Self(limbs))
    }
}

/// Compute the signed modulo operation on a u64 representation, returning the result.
/// If n % modulus > modulus / 2, return modulus - n
/// # Example
/// ```
/// use ark_ff::signed_mod_reduction;
/// let res = signed_mod_reduction(6u64, 8u64);
/// assert_eq!(res, -2i64);
/// ```
pub fn signed_mod_reduction(n: u64, modulus: u64) -> i64 {
    let t = (n % modulus) as i64;
    if t as u64 >= (modulus / 2) {
        t - (modulus as i64)
    } else {
        t
    }
}

pub type BigInteger64 = BigInt<1>;
pub type BigInteger128 = BigInt<2>;
pub type BigInteger256 = BigInt<4>;
pub type BigInteger320 = BigInt<5>;
pub type BigInteger384 = BigInt<6>;
pub type BigInteger448 = BigInt<7>;
pub type BigInteger768 = BigInt<12>;
pub type BigInteger832 = BigInt<13>;

#[derive(Copy, Clone)]
pub struct Words<const N: usize>(pub(crate) [u64; N]);

impl<const N: usize> Default for Words<N> {
    fn default() -> Self {
        Self([0u64; N])
    }
}

impl<const N: usize> crate::traits::Words for Words<N> {
    type Word = u64;

    fn num_words() -> usize {
        N
    }
}

impl<const N: usize> Index<usize> for Words<N> {
    type Output = u64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const N: usize> IndexMut<usize> for Words<N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<const N: usize> Index<Range<usize>> for Words<N> {
    type Output = [u64];

    fn index(&self, index: Range<usize>) -> &Self::Output {
        &self.0[index]
    }
}

impl<const N: usize> IndexMut<Range<usize>> for Words<N> {
    fn index_mut(&mut self, index: Range<usize>) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<const N: usize> Index<RangeTo<usize>> for Words<N> {
    type Output = [u64];

    fn index(&self, index: RangeTo<usize>) -> &Self::Output {
        &self.0[index]
    }
}

impl<const N: usize> IndexMut<RangeTo<usize>> for Words<N> {
    fn index_mut(&mut self, index: RangeTo<usize>) -> &mut Self::Output {
        &mut self.0[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_from_bytes_le_valid() {
        let bytes = [0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01];
        let bigint = BigInteger64::from_bytes_le(&bytes).unwrap();

        // Same as BE but reversed
        let expected = BigInteger64::from(0x0123456789ABCDEFu64);
        assert_eq!(bigint, expected);
    }

    #[test]
    fn converts_from_bytes_be_valid() {
        let bytes = [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF];
        let bigint = BigInteger64::from_bytes_be(&bytes).unwrap();

        let expected = BigInteger64::from(0x0123456789ABCDEFu64);
        assert_eq!(bigint, expected);
    }

    #[test]
    fn converts_from_bytes_le_single_byte() {
        let bytes = [0xAB]; // Only 1 byte
        let bigint = BigInteger64::from_bytes_le(&bytes).unwrap();
        let expected = BigInteger64::from(0xABu64);
        assert_eq!(bigint, expected);
    }

    #[test]
    fn converts_from_bytes_be_single_byte() {
        let bytes = [0xAB]; // Only 1 byte
        let bigint = BigInteger64::from_bytes_be(&bytes).unwrap();
        let expected = BigInteger64::from(0xABu64);
        assert_eq!(bigint, expected);
    }

    #[test]
    fn converts_from_bytes_le_partial_limb() {
        let bytes = [0x12, 0x34, 0x56]; // Only 3 bytes
        let bigint = BigInteger64::from_bytes_le(&bytes).unwrap();
        let expected = BigInteger64::from(0x563412u64);
        assert_eq!(bigint, expected);
    }

    #[test]
    fn converts_from_bytes_be_partial_limb() {
        let bytes = [0x12, 0x34, 0x56]; // Only 3 bytes
        let bigint = BigInteger64::from_bytes_be(&bytes).unwrap();
        let expected = BigInteger64::from(0x123456u64);
        assert_eq!(bigint, expected);
    }

    #[test]
    fn converts_from_bytes_le_zero() {
        let bytes = [0x00; 8];
        let bigint = BigInteger64::from_bytes_le(&bytes).unwrap();
        assert_eq!(bigint, BigInteger64::zero());
    }

    #[test]
    fn converts_from_bytes_be_zero() {
        let bytes = [0x00; 8];
        let bigint = BigInteger64::from_bytes_be(&bytes).unwrap();
        assert_eq!(bigint, BigInteger64::zero());
    }

    #[test]
    fn converts_from_bytes_le_max_value() {
        let bytes = [0xFF; 8];
        let bigint = BigInteger64::from_bytes_le(&bytes).unwrap();
        let expected = BigInteger64::from(u64::MAX);
        assert_eq!(bigint, expected);
    }

    #[test]
    fn converts_from_bytes_be_max_value() {
        let bytes = [0xFF; 8];
        let bigint = BigInteger64::from_bytes_be(&bytes).unwrap();
        let expected = BigInteger64::from(u64::MAX);
        assert_eq!(bigint, expected);
    }

    #[test]
    fn converts_from_bytes_le_vs_be() {
        let bytes = [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0];
        let bigint_be = BigInteger64::from_bytes_be(&bytes);
        let mut bytes_reversed = bytes;
        bytes_reversed.reverse();
        let bigint_le = BigInteger64::from_bytes_le(&bytes_reversed);

        assert_eq!(bigint_be, bigint_le);
    }

    #[test]
    fn converts_from_bytes_le_with_leading_zeros() {
        let bytes = [0x00, 0x00, 0x00, 0x00, 0x01, 0x23, 0x45, 0x67];
        let bigint = BigInteger64::from_bytes_le(&bytes).unwrap();
        let expected = BigInteger64::from(0x6745230100000000u64);
        assert_eq!(bigint, expected);
    }

    #[test]
    fn converts_from_bytes_be_with_leading_zeros() {
        let bytes = [0x00, 0x00, 0x00, 0x00, 0x01, 0x23, 0x45, 0x67];
        let bigint = BigInteger64::from_bytes_be(&bytes).unwrap();
        let expected = BigInteger64::from(0x1234567u64);
        assert_eq!(bigint, expected);
    }

    #[test]
    fn converts_bigint256_from_bytes_le_valid() {
        let bytes = [
            0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, // LSB
            0x11, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0x21, 0x23, 0x45, 0x67, 0x89, 0xAB,
            0xCD, 0xEF, 0x31, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, // MSB
        ];

        let bigint = BigInteger256::from_bytes_le(&bytes).unwrap();

        let expected = BigInt([
            0xEFCDAB8967452301,
            0xEFCDAB8967452311,
            0xEFCDAB8967452321,
            0xEFCDAB8967452331,
        ]);

        assert_eq!(bigint, expected);
    }

    #[test]
    fn converts_bigint256_from_bytes_be_valid() {
        let bytes = [
            0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, // MSB
            0x11, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, //
            0x21, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, //
            0x31, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, // LSB
        ];

        let bigint = BigInteger256::from_bytes_be(&bytes).unwrap();

        let expected = BigInt([
            0x3123456789ABCDEF,
            0x2123456789ABCDEF,
            0x1123456789ABCDEF,
            0x0123456789ABCDEF,
        ]);

        assert_eq!(bigint, expected);
    }
}
