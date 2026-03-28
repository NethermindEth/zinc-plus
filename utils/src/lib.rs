pub mod field;
pub mod from_ref;
pub mod inner_product;
pub mod inner_transparent_field;
pub mod mul_by_scalar;
pub mod named;
pub mod ops_macros;
pub mod parallel;
pub mod projectable_to_field;

use crypto_primitives::Semiring;

// Can't use enums in const generics in stable Rust yet, so we use constants
// instead.
pub const CHECKED: bool = true;
pub const UNCHECKED: bool = false;

/// Returns ceil(log2(x)).
/// Copied from ark-std.
#[inline(always)]
#[allow(clippy::arithmetic_side_effects)]
pub const fn log2(x: usize) -> u32 {
    if x == 0 {
        0
    } else if x.is_power_of_two() {
        1usize.leading_zeros() - x.leading_zeros()
    } else {
        0usize.leading_zeros() - x.leading_zeros()
    }
}

#[allow(clippy::arithmetic_side_effects)]
pub fn powers<R: Semiring>(x: R, one: R, num_pows: usize) -> Vec<R> {
    if num_pows == 0 {
        return Vec::new();
    }

    let mut pows = Vec::with_capacity(num_pows);

    pows.push(one);

    if num_pows == 1 {
        return pows;
    }

    let mut curr_pow = x.clone();

    for _ in 1..num_pows {
        pows.push(curr_pow.clone());
        curr_pow *= &x;
    }

    pows
}

/// Formats a number with spaces as thousands separators, e.g. 1234567 becomes
/// "1 234 567".
#[allow(clippy::unwrap_used)]
pub fn fmt_thousands(n: usize) -> String {
    let s = n.to_string();
    s.as_bytes()
        .rchunks(3)
        .rev()
        .map(|c| std::str::from_utf8(c).unwrap())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Prints proof size to stderr in a consistent format across all benchmarks.
pub fn eprint_proof_size(label: impl std::fmt::Display, size_bytes: usize) {
    eprintln!(
        "    Proof size ({label}): {} bytes ({} KiB)",
        fmt_thousands(size_bytes),
        size_bytes.div_ceil(1024),
    );
}
