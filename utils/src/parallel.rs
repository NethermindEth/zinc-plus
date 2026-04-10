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

/// Conditionally extend a collection from an iterator.
///
/// When the "parallel" feature is enabled, uses `par_extend()` from rayon.
/// Otherwise, uses standard `extend()`.
#[macro_export]
macro_rules! cfg_extend {
    ($e:expr, $i:expr) => {{
        #[cfg(feature = "parallel")]
        let result = $e.par_extend($i);

        #[cfg(not(feature = "parallel"))]
        let result = $e.extend($i);

        result
    }};
}

/// Compute chunk size for parallel/sequential dispatch.
///
/// When the "parallel" feature is enabled and `$len >= $threshold`,
/// returns `$len / num_threads`. Otherwise returns `$len` (one chunk =
/// sequential).
///
/// ```ignore
/// let chunk_size = cfg_chunk_size!(n, 4096);
/// cfg_chunks_mut!(result, chunk_size).zip(cfg_chunks!(values, chunk_size)).for_each(|..| ..);
/// ```
#[macro_export]
macro_rules! cfg_chunk_size {
    ($len:expr, $threshold:expr) => {{
        #[cfg(feature = "parallel")]
        let _cs = if $len >= $threshold {
            $len.div_ceil(rayon::current_num_threads())
        } else {
            $len
        };

        #[cfg(not(feature = "parallel"))]
        let _cs = $len;

        _cs
    }};
}

/// Conditionally fork expressions into parallel tasks using `rayon::join`.
///
/// When the "parallel" feature is enabled, uses right-leaning nested
/// `rayon::join` calls to execute all expressions concurrently. Otherwise,
/// evaluates them sequentially.
///
/// Returns a right-nested pair tree: 2 items → `(A, B)`,
/// 3 items → `(A, (B, C))`, 4 items → `(A, (B, (C, D)))`, etc.
/// The shape is identical in both parallel and sequential mode.
///
/// (Unfortunately, we can't flatten the result into a tuple of all items using
/// just `macro_rules`)
#[macro_export]
macro_rules! cfg_join {
    ($a:expr, $b:expr $(,)?) => {{
        #[cfg(feature = "parallel")]
        let result = rayon::join(|| $a, || $b);

        #[cfg(not(feature = "parallel"))]
        let result = ($a, $b);

        result
    }};
    ($first:expr, $($rest:expr),+ $(,)?) => {{
        #[cfg(feature = "parallel")]
        let result = rayon::join(
            || $first,
            || cfg_join!($($rest),+),
        );

        #[cfg(not(feature = "parallel"))]
        let result = ($first, cfg_join!($($rest),+));

        result
    }};
}
