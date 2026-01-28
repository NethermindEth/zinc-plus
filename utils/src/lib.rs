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
