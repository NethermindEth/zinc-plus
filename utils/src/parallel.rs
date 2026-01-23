//! Parallel iteration macros that conditionally use rayon when the "parallel"
//! feature is enabled.
//!
//! These macros provide a convenient way to switch between parallel and
//! sequential iteration without changing the calling code - just enable/disable
//! the "parallel" cargo feature.

/// Conditionally iterate over a slice in parallel.
///
/// When the "parallel" feature is enabled, uses `par_iter()` from rayon.
/// Otherwise, uses standard `iter()`.
#[macro_export]
macro_rules! cfg_iter {
    ($e:expr, $min_len:expr) => {{
        #[cfg(feature = "parallel")]
        let result = $e.par_iter().with_min_len($min_len);

        #[cfg(not(feature = "parallel"))]
        let result = $e.iter();

        result
    }};
    ($e:expr) => {{
        #[cfg(feature = "parallel")]
        let result = $e.par_iter();

        #[cfg(not(feature = "parallel"))]
        let result = $e.iter();

        result
    }};
}

/// Conditionally iterate mutably over a slice in parallel.
///
/// When the "parallel" feature is enabled, uses `par_iter_mut()` from rayon.
/// Otherwise, uses standard `iter_mut()`.
#[macro_export]
macro_rules! cfg_iter_mut {
    ($e:expr, $min_len:expr) => {{
        #[cfg(feature = "parallel")]
        let result = $e.par_iter_mut().with_min_len($min_len);

        #[cfg(not(feature = "parallel"))]
        let result = $e.iter_mut();

        result
    }};
    ($e:expr) => {{
        #[cfg(feature = "parallel")]
        let result = $e.par_iter_mut();

        #[cfg(not(feature = "parallel"))]
        let result = $e.iter_mut();

        result
    }};
}

/// Conditionally consume and iterate over a collection in parallel.
///
/// When the "parallel" feature is enabled, uses `into_par_iter()` from rayon.
/// Otherwise, uses standard `into_iter()`.
#[macro_export]
macro_rules! cfg_into_iter {
    ($e:expr, $min_len:expr) => {{
        #[cfg(feature = "parallel")]
        let result = $e.into_par_iter().with_min_len($min_len);

        #[cfg(not(feature = "parallel"))]
        let result = $e.into_iter();

        result
    }};
    ($e:expr) => {{
        #[cfg(feature = "parallel")]
        let result = $e.into_par_iter();

        #[cfg(not(feature = "parallel"))]
        let result = $e.into_iter();

        result
    }};
}

/// Conditionally iterate over chunks of a slice in parallel.
///
/// When the "parallel" feature is enabled, uses `par_chunks()` from rayon.
/// Otherwise, uses standard `chunks()`.
#[macro_export]
macro_rules! cfg_chunks {
    ($e:expr, $size:expr) => {{
        #[cfg(feature = "parallel")]
        let result = $e.par_chunks($size);

        #[cfg(not(feature = "parallel"))]
        let result = $e.chunks($size);

        result
    }};
}

/// Conditionally iterate mutably over chunks of a slice in parallel.
///
/// When the "parallel" feature is enabled, uses `par_chunks_mut()` from rayon.
/// Otherwise, uses standard `chunks_mut()`.
#[macro_export]
macro_rules! cfg_chunks_mut {
    ($e:expr, $size:expr) => {{
        #[cfg(feature = "parallel")]
        let result = $e.par_chunks_mut($size);

        #[cfg(not(feature = "parallel"))]
        let result = $e.chunks_mut($size);

        result
    }};
}
