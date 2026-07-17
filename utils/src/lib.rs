pub mod field;
pub mod from_ref;
pub mod inner_product;
pub mod mul_by_scalar;
pub mod named;
pub mod ops_macros;
pub mod parallel;
pub mod projectable_to_field;

use crypto_primitives::SemiringConfig;

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

/// Powers `[1, x, x^2, ..., x^(num_pows-1)]` computed via the config.
pub fn powers<S: SemiringConfig>(cfg: &S, x: &S::Element, num_pows: usize) -> Vec<S::Element> {
    if num_pows == 0 {
        return Vec::new();
    }

    let mut pows = Vec::with_capacity(num_pows);
    pows.push(cfg.one());

    let mut curr_pow = x.clone();
    for _ in 1..num_pows {
        pows.push(curr_pow.clone());
        cfg.mul_assign(&mut curr_pow, x);
    }

    pows
}
