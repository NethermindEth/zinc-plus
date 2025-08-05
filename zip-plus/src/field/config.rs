use crate::{
    field::BigInt,
    traits::{Config, ConfigReference},
};

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

#[derive(Clone, Copy, Default)]
pub struct FieldConfig<const N: usize> {
    /// The modulus of the field.
    modulus: BigInt<N>,

    /// Let `M` be the power of 2^64 nearest to `Self::MODULUS_BITS`. Then
    /// `R = M % Self::MODULUS`.
    r: BigInt<N>,

    /// R2 = R^2 % Self::MODULUS
    r2: BigInt<N>,

    /// INV = -MODULUS^{-1} mod 2^64
    inv: u64,

    /// Does the modulus have a spare unused bit
    ///
    /// This condition applies if
    /// (a) `Self::MODULUS[N-1] >> 63 == 0`
    #[doc(hidden)]
    modulus_has_spare_bit: bool,
}

impl<const N: usize> FieldConfig<N> {
    pub fn add_assign(&self, a: &mut BigInt<N>, b: &BigInt<N>) {
        // This cannot exceed the backing capacity.
        let c = a.add_with_carry(b);
        // However, it may need to be reduced
        self.reduce_modulus(a, c);
    }

    pub fn sub_assign(&self, a: &mut BigInt<N>, b: &BigInt<N>) {
        // If `other` is larger than `self`, add the modulus to self first.
        if b > a {
            a.add_with_carry(&self.modulus);
        }
        a.sub_with_borrow(b);
    }

    fn reduce_modulus(&self, a: &mut BigInt<{ N }>, carry: bool) {
        if self.modulus_has_spare_bit {
            if *a >= self.modulus {
                a.sub_with_borrow(&self.modulus);
            }
        } else if carry || *a >= self.modulus {
            a.sub_with_borrow(&self.modulus);
        }
    }

    pub fn inverse(&self, a: &BigInt<N>) -> Option<BigInt<N>> {
        if a.is_zero() {
            return None;
        }

        // Guajardo Kumar Paar Pelzl
        // Efficient Software-Implementation of Finite Fields with Applications to
        // Cryptography
        // Algorithm 16 (BEA for Inversion in Fp)

        let one = BigInt::one();

        let mut u = *a;
        let mut v = self.modulus;
        let mut b = self.r2; // Avoids unnecessary reduction step.
        let mut c = BigInt::zero();

        while u != one && v != one {
            while u.is_even() {
                u.div2();

                if b.is_even() {
                    b.div2();
                } else {
                    let carry = b.add_with_carry(&self.modulus);
                    b.div2();
                    if !self.modulus_has_spare_bit && carry {
                        *b.last_mut() |= 1 << 63;
                    }
                }
            }

            while v.is_even() {
                v.div2();

                if c.is_even() {
                    c.div2();
                } else {
                    let carry = c.add_with_carry(&self.modulus);
                    c.div2();

                    if !self.modulus_has_spare_bit && carry {
                        *c.last_mut() |= 1 << 63;
                    }
                }
            }

            if v < u {
                u.sub_with_borrow(&v);

                if c > b {
                    b.add_with_carry(&self.modulus);
                }
                b.sub_with_borrow(&c);
            } else {
                v.sub_with_borrow(&u);

                if b > c {
                    c.add_with_carry(&self.modulus);
                }

                c.sub_with_borrow(&b);
            }
        }

        if u == one {
            Some(b)
        } else {
            Some(c)
        }
    }

    #[inline]
    pub fn r(&self) -> &BigInt<N> {
        &self.r
    }

    #[inline]
    pub fn inv(&self) -> u64 {
        self.inv
    }
}

impl<const N: usize> Config for FieldConfig<N> {
    type B = BigInt<N>;
    fn modulus(&self) -> &BigInt<N> {
        &self.modulus
    }

    fn mul_assign(&self, a: &mut BigInt<N>, b: &BigInt<N>) {
        let (mut lo, mut hi) = a.mul_naive(b);

        // Montgomery reduction
        let carry = a.montgomery_reduction(&mut lo, &mut hi, &self.modulus, self.inv);

        self.reduce_modulus(a, carry);
    }

    fn r2(&self) -> &BigInt<N> {
        &self.r2
    }

    fn new(modulus: BigInt<N>) -> Self {
        let modulus_has_spare_bit = modulus.has_spare_bit();
        Self {
            modulus,
            r: modulus.montgomery_r(),
            r2: modulus.montgomery_r2(),
            inv: inv(modulus),

            modulus_has_spare_bit,
        }
    }
}

impl<const N: usize> ark_std::fmt::Debug for FieldConfig<N> {
    fn fmt(&self, f: &mut ark_std::fmt::Formatter<'_>) -> ark_std::fmt::Result {
        write!(f, " Z_{}", self.modulus,)
    }
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

impl<const N: usize> PartialEq for FieldConfig<N> {
    fn eq(&self, other: &Self) -> bool {
        self.modulus == other.modulus
    }
}

impl<const N: usize> Eq for FieldConfig<N> {}

#[allow(dead_code)]
#[derive(Debug)]
pub struct DebugFieldConfig {
    /// The modulus of the field.
    modulus: num_bigint::BigInt,

    /// Let `M` be the power of 2^64 nearest to `Self::MODULUS_BITS`. Then
    /// `R = M % Self::MODULUS`.
    r: num_bigint::BigInt,

    /// R2 = R^2 % Self::MODULUS
    r2: num_bigint::BigInt,

    /// INV = -MODULUS^{-1} mod 2^64
    inv: u64,

    /// Does the modulus have a spare unused bit
    ///
    /// This condition applies if
    /// (a) `Self::MODULUS[N-1] >> 63 == 0`
    #[doc(hidden)]
    modulus_has_spare_bit: bool,
}

impl<const N: usize> From<FieldConfig<N>> for DebugFieldConfig {
    fn from(value: FieldConfig<N>) -> Self {
        Self {
            modulus: value.modulus.into(),
            r: value.r.into(),
            r2: value.r2.into(),
            inv: value.inv,
            modulus_has_spare_bit: value.modulus_has_spare_bit,
        }
    }
}

/// A wrapper around an optional reference to a `FieldConfig`.
///
/// This struct is used to represent a pointer to a `FieldConfig` instance,
/// allowing for safe handling of nullable references. It provides methods
/// to access the underlying reference or raw pointer.
///
/// # Type Parameters
/// - `'cfg`: The lifetime of the `FieldConfig` reference.
/// - `N`: The size of the `FieldConfig` (e.g., the number of limbs in the `BigInt` modulus).
#[derive(Debug, Copy, Clone, Eq)]
pub struct ConfigRef<'cfg, const N: usize>(Option<&'cfg FieldConfig<N>>);

impl<'cfg, const N: usize> ConfigReference for ConfigRef<'cfg, N> {
    type C = FieldConfig<N>;
    fn reference(&self) -> Option<&'cfg FieldConfig<N>> {
        self.0
    }

    unsafe fn new(config_ptr: *mut FieldConfig<N>) -> Self {
        Self(Option::from(config_ptr.as_ref().unwrap()))
    }

    fn pointer(&self) -> Option<*mut FieldConfig<N>> {
        self.0.map(|p| p as *const _ as *mut _)
    }

    const NONE: Self = Self(None);
}

impl<const N: usize> PartialEq for ConfigRef<'_, N> {
    fn eq(&self, other: &Self) -> bool {
        self.reference() == other.reference()
    }
}

impl<'cfg, const N: usize> From<&'cfg FieldConfig<N>> for ConfigRef<'cfg, N> {
    fn from(value: &'cfg FieldConfig<N>) -> Self {
        Self(Some(value))
    }
}

unsafe impl<const N: usize> Sync for ConfigRef<'_, N> {}
unsafe impl<const N: usize> Send for ConfigRef<'_, N> {}

#[cfg(test)]
mod tests {
    use super::FieldConfig;
    use crate::{big_int, field::BigInteger128, field_config, traits::Config};
    //BIGINTS ARE LITTLE ENDIAN!!
    #[test]
    fn test_addition() {
        let field = FieldConfig::new(BigInteger128::new([
            9307119299070690521,
            9320126393725433252,
        ]));
        let mut a = BigInteger128::new([2, 0]);
        let b = BigInteger128::new([2, 0]);
        field.add_assign(&mut a, &b);
        assert_eq!(a, BigInteger128::new([4, 0]));
    }

    #[test]
    fn test_subtraction() {
        let field = FieldConfig::new(BigInteger128::new([
            9307119299070690521,
            9320126393725433252,
        ]));
        let mut a = BigInteger128::new([2, 0]);
        let b = BigInteger128::new([2, 0]);
        field.sub_assign(&mut a, &b);
        assert_eq!(a, BigInteger128::zero());
    }

    #[test]
    fn test_multiplication() {
        let field = field_config!(695962179703626800597079116051991347);
        let mut a = big_int!(423024736033, 4);
        let b = big_int!(246308734);
        field.mul_assign(&mut a, &b);

        assert_eq!(big_int!(504579159360957705315139767875358506), a);
    }
}
