/// A macro to construct a `BigInt<N>` from a numeric literal or string literal
/// at compile time.
///
/// ## Variants
///
/// ### 1. `big_int!(value)`
/// - Parses a numeric literal using the crate's default `BigInt` type.
/// - Requires the user to have `use crate::field::BigInt` in scope with a fixed
///   `N`.
///
/// ### 2. `big_int!(value, N)`
/// - Parses a numeric literal into a specific `BigInt<N>`.
/// - Uses `.unwrap()` for simplicity (intended for test or infallible usage).
///
/// ### 3. `big_int!("value", N, "error message")`
/// - Parses a **string literal** into `BigInt<N>`.
/// - Uses `.expect("error message")` to allow user-defined error context.
///
/// ## Example
/// ```rust
/// use zip_plus::big_int;
/// let a = big_int!(123, 1);
/// let b = big_int!(4567890123456789, 4, "failed to parse b");
/// ```
#[macro_export]
macro_rules! big_int {
    ($v:literal) => {
        (|| {
            use ark_std::str::FromStr;
            use $crate::field::BigInt;
            BigInt::from_str(stringify!($v)).unwrap()
        })()
    };

    ($v:literal, $n:expr) => {
        (|| {
            use ark_std::str::FromStr;
            use $crate::field::BigInt;
            BigInt::<$n>::from_str(stringify!($v)).unwrap()
        })()
    };

    ($v:literal, $n:expr, $msg:literal) => {
        (|| {
            use ark_std::str::FromStr;
            use $crate::field::BigInt;
            BigInt::<$n>::from_str(stringify!($v)).expect($msg)
        })()
    };
}

/// Constructs a `RandomField` element from a literal value.
///
/// This macro provides two modes:
///
/// ## Variants
///
/// ### 1. `random_field!(value, config)`
/// Converts a numeric literal into a `RandomField` element using a provided
/// field configuration.
///
/// - This leverages the `FieldMap` trait’s `.map_to_field()` method.
/// - The `config` argument must be a valid configuration object of type `F::R`.
///
/// ```rust
/// use zip_plus::{big_int, define_field_config, field::RandomField, random_field};
///
/// define_field_config!(Fc, 19);
///
/// let x: RandomField<1, Fc<1>> = random_field!(123);
/// ```
///
/// ## Notes
/// - In both cases, the macro uses a closure internally to maintain hygiene and
///   isolate `use` statements.
/// - The first form (`random_field!(value)`) uses `$crate::field::{BigInt,
///   RandomField}`.
/// - The second form (`random_field!(value, config)`) uses
///   `$crate::traits::FieldMap`.
#[macro_export]
macro_rules! random_field {
    ($v:literal) => {
        (|| {
            use $crate::traits::FieldMap;
            $v.map_to_field()
        })()
    };
}
