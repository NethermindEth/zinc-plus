//! NeutronNova small-value accumulation helpers.
//!
//! This module is intentionally standalone for now: it provides the linear
//! row-space accumulator and SumFold prefix-table primitives without changing
//! protocol proof objects or verifier flow.

pub mod accumulator;
pub mod linear_cpr;
pub mod sumfold;

pub use accumulator::{
    AccumulatorError, RowWeights, SmallValueBitAccumulator, accumulate_binary_column_projected,
};
pub use linear_cpr::{
    CoeffClass, LinearBinarySource, LinearCprAccumulatorError, LinearCprScalarWeights,
    LinearCprWeights, LinearFamilySpec, LinearTermSpec, SumFoldEqWeights,
    build_linear_cpr_prefix_table, build_sumfold_eq_weights,
};
pub use sumfold::{LinearInstanceClaims, LinearPrefixTable, SumFoldError};
