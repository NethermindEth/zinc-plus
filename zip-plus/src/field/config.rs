use crate::field::BigInt;
use num_traits::Zero;

#[macro_export]
macro_rules! mac_with_carry {
    ($a:expr, $b:expr, $c:expr, &mut $carry:expr$(,)?) => {{
        let tmp = ($a as u128) + widening_mul($b, $c) + ($carry as u128);
        $carry = (tmp >> 64) as u64;
        tmp as u64
    }};
}

#[macro_export]
macro_rules! adc {
    ($a:expr, $b:expr, &mut $carry:expr$(,)?) => {{
        let tmp = ($a as u128) + ($b as u128) + ($carry as u128);
        $carry = (tmp >> 64) as u64;
        tmp as u64
    }};
}
pub fn mac_with_carry(a: u64, b: u64, c: u64, carry: &mut u64) -> u64 {
    let tmp = (a as u128) + widening_mul(b, c) + (*carry as u128);
    *carry = (tmp >> 64) as u64;
    tmp as u64
}

pub trait FieldConfigBase<T> {
    /// The modulus of the field.
    fn modulus() -> T;

    /// Let `M` be the power of 2^64 nearest to `Self::MODULUS_BITS`. Then
    /// `R = M % Self::MODULUS`.
    fn r() -> T;

    /// `R^2 = R * R mod MODULUS`
    fn r_squared() -> T;

    /// Does the modulus have a spare unused bit
    ///
    /// This condition applies if
    /// (a) `Self::MODULUS[N-1] >> 63 == 0`
    fn modulus_has_spare_bit() -> bool;

    /// INV = -MODULUS^{-1} mod 2^64
    fn inv() -> u64;
}

pub trait ConstFieldConfigBase1<T> {
    /// The modulus of the field.
    const MODULUS: T;
}

pub trait ConstFieldConfigBase2<T> {
    /// Let `M` be the power of 2^64 nearest to `Self::MODULUS_BITS`. Then
    /// `R = M % Self::MODULUS`.
    const R: T;

    /// `R^2 = R * R mod MODULUS`
    const R_SQUARED: T;

    /// Does the modulus have a spare unused bit
    ///
    /// This condition applies if
    /// (a) `Self::MODULUS[N-1] >> 63 == 0`
    const MODULUS_HAS_SPARE_BIT: bool;

    /// INV = -MODULUS^{-1} mod 2^64
    const INV: u64;
}

impl<T, C> FieldConfigBase<T> for C
where
    C: ConstFieldConfigBase1<T> + ConstFieldConfigBase2<T>,
{
    #[inline(always)]
    fn modulus() -> T {
        C::MODULUS
    }

    #[inline(always)]
    fn r() -> T {
        C::R
    }

    #[inline(always)]
    fn r_squared() -> T {
        C::R_SQUARED
    }

    #[inline(always)]
    fn modulus_has_spare_bit() -> bool {
        C::MODULUS_HAS_SPARE_BIT
    }

    #[inline(always)]
    fn inv() -> u64 {
        C::INV
    }
}

impl<const N: usize, C> ConstFieldConfigBase2<BigInt<N>> for C
where
    C: ConstFieldConfigBase1<BigInt<N>>,
{
    const INV: u64 = inv(C::MODULUS);
    const MODULUS_HAS_SPARE_BIT: bool = C::MODULUS.has_spare_bit();
    const R: BigInt<N> = C::MODULUS.montgomery_r();
    const R_SQUARED: BigInt<N> = C::MODULUS.montgomery_r2();
}

pub trait FieldConfigOps<T> {
    fn add_assign(a: &mut T, b: &T);

    fn sub_assign(a: &mut T, b: &T);

    fn reduce_modulus(a: &mut T, carry: bool);

    fn inverse(a: &T) -> Option<T>;

    fn mul_assign(a: &mut T, b: &T);
}

impl<const N: usize, T> FieldConfigOps<BigInt<N>> for T
where
    T: FieldConfigBase<BigInt<N>>,
{
    fn add_assign(a: &mut BigInt<N>, b: &BigInt<N>) {
        // This cannot exceed the backing capacity.
        let c = a.add_with_carry(b);
        // However, it may need to be reduced
        Self::reduce_modulus(a, c);
    }

    fn sub_assign(a: &mut BigInt<N>, b: &BigInt<N>) {
        // If `other` is larger than `self`, add the modulus to self first.
        if b > a {
            // Add modulus first so the subtraction doesn't underflow; carry is intentionally
            // ignored (we operate modulo 2^(64*N) here and the subsequent subtraction pulls it back).
            a.add_with_carry(&Self::modulus());
        }
        a.sub_with_borrow(b);
    }

    fn reduce_modulus(a: &mut BigInt<{ N }>, carry: bool) {
        if Self::modulus_has_spare_bit() {
            if *a >= Self::modulus() {
                a.sub_with_borrow(&Self::modulus());
            }
        } else if carry || *a >= Self::modulus() {
            a.sub_with_borrow(&Self::modulus());
        }
    }

    fn inverse(a: &BigInt<N>) -> Option<BigInt<N>> {
        if a.is_zero() {
            return None;
        }

        // Guajardo Kumar Paar Pelzl
        // Efficient Software-Implementation of Finite Fields with Applications to
        // Cryptography
        // Algorithm 16 (BEA for Inversion in Fp)

        let one = BigInt::one();

        let mut u = *a;
        let mut v = Self::modulus();
        let mut b = Self::r_squared(); // Avoids unnecessary reduction step.
        let mut c = BigInt::zero();

        while u != one && v != one {
            while u.is_even() {
                u.div2();

                if b.is_even() {
                    b.div2();
                } else {
                    let carry = b.add_with_carry(&Self::modulus());
                    b.div2();
                    if !Self::modulus_has_spare_bit() && carry {
                        *b.last_mut() |= 1 << 63;
                    }
                }
            }

            while v.is_even() {
                v.div2();

                if c.is_even() {
                    c.div2();
                } else {
                    let carry = c.add_with_carry(&Self::modulus());
                    c.div2();

                    if !Self::modulus_has_spare_bit() && carry {
                        *c.last_mut() |= 1 << 63;
                    }
                }
            }

            if v < u {
                u.sub_with_borrow(&v);

                if c > b {
                    b.add_with_carry(&Self::modulus());
                }
                b.sub_with_borrow(&c);
            } else {
                v.sub_with_borrow(&u);

                if b > c {
                    c.add_with_carry(&Self::modulus());
                }

                c.sub_with_borrow(&b);
            }
        }

        if u == one { Some(b) } else { Some(c) }
    }

    #[inline(always)]
    fn mul_assign(a: &mut BigInt<N>, b: &BigInt<N>) {
        let (mut lo, mut hi) = a.mul_naive(b);

        // Montgomery reduction
        let carry = a.montgomery_reduction(&mut lo, &mut hi, &Self::modulus(), Self::inv());

        Self::reduce_modulus(a, carry);
    }
}

pub trait FieldConfig<T>: FieldConfigBase<T> + FieldConfigOps<T> + Clone {}

impl<T, C> FieldConfig<T> for C where C: FieldConfigBase<T> + FieldConfigOps<T> + Clone {}

#[macro_export]
macro_rules! define_field_config {
    ($name:ident, $modulus:expr) => {
        #[allow(dead_code)]
        #[derive(Clone, Debug)]
        struct $name<const N: usize>;

        impl<const N: usize> $crate::field::config::ConstFieldConfigBase1<$crate::field::BigInt<N>>
            for $name<N>
        {
            const MODULUS: $crate::field::BigInt<N> = $crate::BigInt!($modulus);
        }
    };
}

/// Compute -M^{-1} mod 2^64.
pub const fn inv<const N: usize>(modulus: BigInt<N>) -> u64 {
    // We compute this as follows.
    // First, MODULUS mod 2^64 is just the lower 64 bits of MODULUS.
    // Hence MODULUS mod 2^64 = MODULUS.0[0] mod 2^64.
    //
    // Next, computing the inverse mod 2^64 involves exponentiating by
    // the multiplicative group order, which is euler_totient(2^64) - 1.
    // Now, euler_totient(2^64) = 1 << 63, and so
    // euler_totient(2^64) - 1 = (1 << 63) - 1 = 1111111... (63 digits).
    // We compute this powering via standard square and multiply.
    let mut inv = 1u64;
    crate::const_for!((_i in 0..63) {
        // Square
        inv = inv.wrapping_mul(inv);
        // Multiply
        inv = inv.wrapping_mul(modulus.first());
    });
    inv.wrapping_neg()
}

fn widening_mul(a: u64, b: u64) -> u128 {
    a as u128 * b as u128
}

#[cfg(test)]
mod tests {
    use super::{ConstFieldConfigBase1, FieldConfigOps};
    use crate::{
        big_int,
        field::{BigInt, BigInteger128},
    };
    use num_traits::ConstZero;

    //BIGINTS ARE LITTLE ENDIAN!!
    #[test]
    fn test_addition() {
        #[derive(Clone, Debug)]
        struct Fc;

        impl ConstFieldConfigBase1<BigInt<2>> for Fc {
            const MODULUS: BigInteger128 =
                { BigInteger128::new([9307119299070690521, 9320126393725433252]) };
        }

        let mut a = BigInteger128::new([2, 0]);
        let b = BigInteger128::new([2, 0]);
        Fc::add_assign(&mut a, &b);
        assert_eq!(a, BigInteger128::new([4, 0]));
    }

    #[test]
    fn test_subtraction() {
        #[derive(Clone, Debug)]
        struct Fc;

        impl ConstFieldConfigBase1<BigInt<2>> for Fc {
            const MODULUS: BigInteger128 =
                { BigInteger128::new([9307119299070690521, 9320126393725433252]) };
        }

        let mut a = BigInteger128::new([2, 0]);
        let b = BigInteger128::new([2, 0]);
        Fc::sub_assign(&mut a, &b);
        assert_eq!(a, BigInteger128::ZERO);
    }

    #[test]
    fn test_multiplication() {
        define_field_config!(Fc, "695962179703626800597079116051991347");

        let mut a = big_int!(423024736033, 4);
        let b = big_int!(246308734);
        Fc::mul_assign(&mut a, &b);

        assert_eq!(big_int!(504579159360957705315139767875358506), a);
    }
}
