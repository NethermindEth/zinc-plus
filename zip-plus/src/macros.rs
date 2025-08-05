/// A macro to construct a `BigInt<N>` from a numeric literal or string literal at compile time.
///
/// ## Variants
///
/// ### 1. `big_int!(value)`
/// - Parses a numeric literal using the crate's default `BigInt` type.
/// - Requires the user to have `use crate::field::BigInt` in scope with a fixed `N`.
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
/// use zinc::big_int;
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

/// Constructs a `FieldConfig<N>` from a numeric literal.
///
/// This macro provides a shorthand for creating field configuration constants from integer literals,
/// using the `big_int!` macro internally to produce the modulus.
///
/// ## Variants
///
/// ### 1. `field_config!(value)`
/// - Creates a `FieldConfig` using the default `BigInt` type in scope.
/// - Assumes `BigInt` is imported or aliased appropriately.
/// - Useful when the generic parameter `N` is inferred or fixed elsewhere.
///
/// ```rust
/// use zinc::field::{ConfigRef, RandomField};
/// use zinc::{field_config, random_field};
/// let config = field_config!(19);
/// let config_ref = ConfigRef::from(&config);
/// let f: RandomField<1> = random_field!(1u32, config_ref);
/// ```
///
/// ### 2. `field_config!(value, N)`
/// - Creates a `FieldConfig<N>` by explicitly specifying the const generic `N`.
/// - Uses `big_int!(value, N)` internally to construct the modulus.
///
/// ```rust
/// use zinc::field_config;
/// let config = field_config!(123456789, 3);
/// ```
///
/// ## Notes
/// - The macro expands into a scoped closure to maintain hygiene and allow internal imports.
/// - Internally uses:
///   - `$crate::big_int!` to construct the modulus.
///   - `$crate::field::FieldConfig` for the config struct.
///   - `$crate::traits::Config` for trait bound resolution.
///
/// ## See also
/// - [`big_int!`] — constructs `BigInt<N>` values from literals
/// - [`random_field!`] — constructs field elements using configs
///
#[macro_export]
macro_rules! field_config {
    ($v:literal) => {
        (|| {
            use $crate::{big_int, field::FieldConfig, traits::Config};
            FieldConfig::new(big_int!($v))
        })()
    };
    ($v:literal, $n:expr) => {
        (|| {
            use $crate::{big_int, field::FieldConfig, traits::Config};
            FieldConfig::new(big_int!($v, $n))
        })()
    };
}

/// Constructs a `RandomField` element from a literal value.
///
/// This macro provides two modes:
///
/// ## Variants
///
/// ### 1. `random_field!(value)`
/// Constructs a `RandomField::Raw` element directly from an integer literal.
///
/// - The input is parsed into a `BigInt` using `BigInt::from`.
/// - Requires no configuration or context.
/// - Useful for testing or constructing raw field elements.
///
/// ```rust
/// use zinc::field::RandomField;
/// use zinc::random_field;
/// let x: RandomField<1> = random_field!(42u32);
/// ```
///
/// ### 2. `random_field!(value, config)`
/// Converts a numeric literal into a `RandomField` element using a provided field configuration.
///
/// - This leverages the `FieldMap` trait’s `.map_to_field()` method.
/// - The `config` argument must be a valid configuration object of type `F::R`.
///
/// ```rust
/// use zinc::field::{ConfigRef, RandomField};
/// use zinc::{big_int, field_config, random_field};
///
/// let config = field_config!(19);
/// let config_ref = ConfigRef::from(&config);
/// let x: RandomField<1> = random_field!(123, config_ref);
/// ```
///
/// ## Notes
/// - In both cases, the macro uses a closure internally to maintain hygiene and isolate `use` statements.
/// - The first form (`random_field!(value)`) uses `$crate::field::{BigInt, RandomField}`.
/// - The second form (`random_field!(value, config)`) uses `$crate::traits::FieldMap`.
///
#[macro_export]
macro_rules! random_field {
    ($v:literal) => {
        (|| {
            use $crate::field::{BigInt, RandomField};
            RandomField::Raw {
                value: BigInt::from($v),
            }
        })()
    };

    ($v:literal, $config:expr) => {
        (|| {
            use $crate::traits::FieldMap;
            $v.map_to_field($config)
        })()
    };

    ($v:literal, $n:literal) => {
        (|| {
            use $crate::{
                big_int,
                field::{BigInt, RandomField},
            };
            RandomField::Raw {
                value: big_int!($v),
            }
        })()
    };

    ($v:expr, $n:literal, $config:expr) => {
        (|| {
            use $crate::{big_int, field::BigInt, random_field, traits::FieldMap};
            <BigInt<$n> as FieldMap<RandomField<$n>>>::map_to_field(&big_int!($v), $config)
        })()
    };
}
