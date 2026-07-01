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
    FoldedCommitments, FreshIdealEvaluationCache, InstanceFoldClaim, LinearResidualCoeffTable,
    MleColumn, MleTable, NUM_NONZERO_SHA_FAMILIES, NUM_SHA_RESIDUAL_FAMILIES,
    PreparedCanonicalShaBooleanityBasis, PreparedShaBooleanityBasis,
    PreparedShaQuadraticPrefixArtifacts, PreparedShaResidualBasis, PreparedShaResidualFamily,
    PreparedShaResidualRow, PreparedShaResidualTerm, PreparedShaSmallResidualBasis,
    PreparedShaSmallResidualFamily, PreparedShaSmallResidualRow, PreparedShaSmallResidualTerm,
    PreparedShaSumFoldBasis, ProjectedPublic, ProjectedTrace, ProjectionFoldAccumulator,
    ProjectionFoldWitness, SHA_ROW_COUNT, SHA_ROW_VARS, SHA_WORD_BITS, ShaAggregateIdealWeightPlan,
    ShaBinaryFoldField, ShaBooleanitySource, ShaIntCol, ShaLinearAccumulatorField,
    ShaLinearResidualWeightPlan, ShaProductionIdeal, ShaProjectionError, ShaPublicCol,
    ShaPublicWordCol, ShaResidualFamily, ShaSmallFieldDecode, ShaSuffixScannerField, ShaWordCol,
    VirtualChMajValues, beta_aggregate_nonzero_ideal_polys,
    beta_aggregate_nonzero_ideal_polys_direct_with_weights,
    beta_aggregate_nonzero_ideal_polys_with_weights, bit_slice_index, build_booleanity_weights,
    build_dense_sha_sumfold_group, build_dense_sha_sumfold_group_with_weights,
    build_expression_folded_row_sumcheck_group,
    build_expression_folded_row_sumcheck_group_with_row_weights, build_folded_row_sumcheck_group,
    build_fresh_sha_ideal_cache, build_linear_residual_coeff_tables,
    build_linear_residual_coeff_tables_with_row_weights, build_production_sha_sumfold_group,
    build_production_sha_sumfold_group_from_prefix_accumulators,
    build_production_sha_sumfold_group_from_prefix_accumulators_with_initial_claim,
    build_production_sha_sumfold_group_from_prepared_prefix_accumulators_with_initial_claim,
    build_production_sha_sumfold_group_from_prepared_prefix_artifacts_with_initial_claim,
    build_production_sha_sumfold_group_owned, build_production_sha_sumfold_group_with_linear_cache,
    build_production_sha_sumfold_group_with_linear_cache_and_weights,
    build_sha_ideal_values_at_point, build_sha_lambda_powers, build_sha_residual_eval_powers,
    build_sha_sumfold_linear_accumulator, build_sha_sumfold_linear_accumulator_direct_with_weights,
    build_sha_sumfold_linear_accumulator_from_bases,
    build_sha_sumfold_linear_accumulator_from_small_bases,
    build_sha_sumfold_quadratic_prefix_accumulator,
    build_sha_sumfold_quadratic_prefix_accumulator_from_bases,
    build_sha_sumfold_quadratic_prefix_artifacts_from_bases, check_fresh_sha_ideal_cache,
    check_sha_ideal_values, derive_instance_fold_claim, evaluate_fresh_sha_targets,
    expression_folded_row_sum, expression_folded_row_sum_with_row_weights,
    expression_folded_row_sum_with_vectors, fold_projected_traces, folded_row_integrand_sum,
    folded_row_integrand_values, folded_row_integrand_values_with_row_weights,
    folded_row_integrand_values_with_vectors, is_production_sha_booleanity_sources,
    prepare_sha_sumfold_basis, prepare_sha_sumfold_basis_production_fast,
    production_sha_booleanity_sources, production_sha_nonzero_families,
    production_sha_nonzero_ideals, reconstruct_virtual_ch_maj_at_row, scalarize_bit_slices,
    sha_int_at_point, sha_int_at_point_with_weights, sha_int_at_point_with_weights_unchecked,
    sha_linear_residual_row_value, sha_linear_residual_sum, sha_public_at_point,
    sha_public_at_point_with_weights, sha_scalarized_word_at_point,
    sha_scalarized_word_at_point_with_weights, sha_word_bits_at_point,
    sha_word_bits_at_point_with_weights,
    sha_word_bits_at_point_with_weights_inner_product_unchecked,
    sha_word_bits_at_point_with_weights_unchecked, validate_fresh_sha_ideal_polys_canonical,
    validate_projected_trace, verify_folded_row_sumcheck_claim, verify_folded_scalarization_links,
    verify_folded_scalarization_links_at_point, verify_folded_shifted_scalarization_link_at_point,
    verify_fresh_sha_ideal_polys,
};
pub use sumfold::{LinearInstanceClaims, LinearPrefixTable, SumFoldError};
