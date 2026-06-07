//! NeutronNova small-value accumulation helpers.
//!
//! This module is intentionally standalone for now: it provides the linear
//! row-space accumulator and SumFold prefix-table primitives without changing
//! protocol proof objects or verifier flow.

pub mod accumulator;
pub mod booleanity;
pub mod linear_cpr;
pub mod projection_sha;
pub mod sumfold;

pub use accumulator::{
    AccumulatorError, RowWeights, SmallValueBitAccumulator, accumulate_binary_column_projected,
};
pub use booleanity::{
    BooleanityAccumulatorError, BooleanityPrefixTable, BooleanityScalarWeights, BooleanityWeights,
    ExtendedPrefixPoint, build_booleanity_prefix_table, extended_point_from_index,
    extended_point_index, ternary_domain_size,
};
pub use linear_cpr::{
    CoeffClass, LinearBinarySource, LinearCprAccumulatorError, LinearCprScalarWeights,
    LinearCprWeights, LinearFamilySpec, LinearTermSpec, SumFoldEqWeights,
    build_linear_cpr_hybrid_sumcheck_group, build_linear_cpr_prefix_bound_tail_table,
    build_linear_cpr_prefix_table, build_sumfold_eq_weights,
};
pub use projection_sha::{
    FoldedCommitments, FoldedShaAccumulator, FoldedShaWitness, FreshShaIdealCache,
    NUM_NONZERO_SHA_FAMILIES, NUM_SHA_RESIDUAL_FAMILIES, ProjectedShaPublic, ProjectedShaTrace,
    SHA_ROW_COUNT, SHA_ROW_VARS, SHA_WORD_BITS, ShaBitSliceColumns, ShaBooleanitySource, ShaIntCol,
    ShaIntColumns, ShaProductionIdeal, ShaProjectionError, ShaPublicCol, ShaPublicColumns,
    ShaResidualFamily, ShaScalarizedRows, ShaSumFoldOutput, ShaWordCol, VirtualChMajValues,
    build_dense_sha_sumfold_group, build_expression_folded_row_sumcheck_group,
    build_folded_row_sumcheck_group, build_fresh_sha_ideal_cache,
    build_production_sha_sumfold_group, build_production_sha_sumfold_group_owned,
    build_sha_ideal_values_at_point, check_fresh_sha_ideal_cache, check_sha_ideal_values,
    evaluate_fresh_sha_targets, expression_folded_row_sum, finalize_sha_sumfold,
    fold_projected_sha_traces, folded_row_integrand_sum, folded_row_integrand_values,
    production_sha_booleanity_sources, production_sha_nonzero_families,
    production_sha_nonzero_ideals, reconstruct_virtual_ch_maj_at_row, scalarize_trace_words,
    sha_int_at_point, sha_linear_residual_row_value, sha_linear_residual_sum, sha_public_at_point,
    sha_scalarized_word_at_point, sha_word_bits_at_point, validate_fresh_sha_ideal_polys_canonical,
    verify_folded_row_sumcheck_claim, verify_folded_scalarization_links,
    verify_folded_scalarization_links_at_point, verify_folded_shifted_scalarization_link_at_point,
    verify_fresh_sha_ideal_polys,
};
pub use sumfold::{LinearInstanceClaims, LinearPrefixTable, SumFoldError};
