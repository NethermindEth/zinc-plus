pub mod field;
pub mod from_ref;
pub mod inner_product;
pub mod inner_transparent_field;
pub mod mul_by_scalar;
pub mod named;
pub mod ops_macros;
pub mod parallel;
pub mod projectable_to_field;

// Can't use enums in consts in stable Rust yet, so we use consts instead.
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
