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
    ($v:literal) => {{
        use ark_std::str::FromStr;
        $crate::field::BigInt::from_str(stringify!($v)).unwrap()
    }};

    ($v:literal, $n:expr) => {{
        use ark_std::str::FromStr;
        $crate::field::BigInt::<$n>::from_str(stringify!($v)).unwrap()
    }};

    ($v:literal, $n:expr, $msg:literal) => {{
        use ark_std::str::FromStr;
        $crate::field::BigInt::<$n>::from_str(stringify!($v)).expect($msg)
    }};
}
