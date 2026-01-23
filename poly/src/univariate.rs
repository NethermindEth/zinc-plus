pub mod binary_ref;
pub mod binary_u64;

pub mod dense;
pub mod nat_evaluation;

pub mod binary {
    pub use crate::univariate::binary_u64::{
        BinaryU64Poly as BinaryPoly, BinaryU64PolyInnerProduct as BinaryPolyInnerProduct,
        BinaryU64PolyWideningMulByScalar as BinaryPolyWideningMulByScalar,
    };
    // pub use crate::univariate::binary_ref::{
    //     BinaryRefPoly as BinaryPoly,
    //     BinaryRefPolyInnerProduct as BinaryPolyInnerProduct,
    //     BinaryRefPolyWideningMulByScalar as BinaryPolyWideningMulByScalar,
    // };
}
