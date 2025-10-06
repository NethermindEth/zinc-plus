//! Contains helper utility functions and macros not visible to the outside
//! world.

#[macro_export]
macro_rules! impl_pow_via_repeated_squaring {
    () => {
        fn pow(self, rhs: u32) -> Self::Output {
            // Implement exponentiation using repeated squaring
            if rhs == 0 {
                return Self::one();
            }

            let mut base = self;
            let mut result = Self::one();
            let mut exp = rhs;

            while exp > 0 {
                if exp & 1 == 1 {
                    result = result
                        .checked_mul(&base)
                        .expect("overflow in exponentiation");
                }
                exp >>= 1;
                if exp > 0 {
                    base = base.checked_mul(&base).expect("overflow in exponentiation");
                }
            }

            result
        }
    };
}
