//! Proof types for the logup-GKR subprotocol.

use crate::sumcheck::SumcheckProof;

/// Per-layer round proof: four "tail evaluations" of the child layer
/// at the sumcheck's output point plus the layer's sumcheck proof.
///
/// When `num_vars_parent == 0` (the layer just below the root), the
/// sumcheck is trivial and `sumcheck_proof` is `None`; the verifier
/// checks the fold identity directly from the four tail values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogupGkrRoundProof<F> {
    /// `N_child(x*, 0)`.
    pub numerator_0: F,
    /// `N_child(x*, 1)`.
    pub numerator_1: F,
    /// `D_child(x*, 0)`.
    pub denominator_0: F,
    /// `D_child(x*, 1)`.
    pub denominator_1: F,
    /// Inner sumcheck proof for the layer, or `None` if the layer has
    /// 0 parent variables (first descent from the scalar root).
    pub sumcheck_proof: Option<SumcheckProof<F>>,
}

/// Top-level logup-GKR proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogupGkrProof<F> {
    /// Root numerator `N_0` (scalar, = `sum_x N_leaves(x) * prod_layers`).
    pub root_numerator: F,
    /// Root denominator `D_0` (scalar).
    pub root_denominator: F,
    /// One round proof per non-trivial layer, from top (just below the
    /// root) to bottom (just above the leaves). Length =
    /// `num_leaf_vars`.
    pub round_proofs: Vec<LogupGkrRoundProof<F>>,
}
