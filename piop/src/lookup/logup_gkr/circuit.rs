//! Grand-sum GKR circuit construction.
//!
//! Given leaf MLEs `N, D` with `n` variables, iteratively folds pairs
//! along the highest variable:
//!
//! ```text
//! parent.n[i] = child.n_0[i] * child.d_1[i] + child.n_1[i] * child.d_0[i]
//! parent.d[i] = child.d_0[i] * child.d_1[i]
//! ```
//!
//! where `child.n_b[i] = child.n[i + b * 2^{n-1}]` (the last variable
//! indexes the MSB of the child's evaluation vector). Produces `n+1`
//! layers: layer `0` is the root (scalar), layer `n` is the leaves.

use crypto_primitives::PrimeField;
use std::marker::PhantomData;
use zinc_poly::mle::DenseMultilinearExtension;
use zinc_utils::inner_transparent_field::InnerTransparentField;

/// One layer of the grand-sum circuit, storing both the numerator and
/// denominator MLEs at that layer.
#[derive(Clone, Debug)]
pub struct GrandSumLayer<T> {
    pub num_vars: usize,
    pub numerator: DenseMultilinearExtension<T>,
    pub denominator: DenseMultilinearExtension<T>,
}

/// All layers of the grand-sum circuit, from root (`layers[0]`, 0 vars)
/// to leaves (`layers[n]`, `n` vars). Stores inner-field evaluations
/// (`F::Inner`) for direct consumption by the sumcheck machinery.
#[derive(Clone, Debug)]
pub struct GrandSumCircuit<F: PrimeField> {
    pub layers: Vec<GrandSumLayer<F::Inner>>,
    _phantom: PhantomData<F>,
}

impl<F: PrimeField> GrandSumCircuit<F> {
    pub fn num_leaf_vars(&self) -> usize {
        self.layers.len().saturating_sub(1)
    }

    pub fn root(&self) -> &GrandSumLayer<F::Inner> {
        &self.layers[0]
    }

    pub fn leaves(&self) -> &GrandSumLayer<F::Inner> {
        self.layers.last().expect("at least one layer")
    }
}

impl<F> GrandSumCircuit<F>
where
    F: PrimeField + InnerTransparentField,
{
    /// Build all layers of the grand-sum circuit bottom-up from the
    /// leaf MLEs. Both input MLEs must share `num_vars`.
    pub fn build(
        leaf_numerator: DenseMultilinearExtension<F::Inner>,
        leaf_denominator: DenseMultilinearExtension<F::Inner>,
        cfg: &F::Config,
    ) -> Self {
        assert_eq!(
            leaf_numerator.num_vars, leaf_denominator.num_vars,
            "N and D must share num_vars"
        );
        let n_leaves = leaf_numerator.num_vars;

        let mut layers = Vec::with_capacity(n_leaves + 1);
        // Push leaves first; we'll reverse at the end so layers[0] = root.
        layers.push(GrandSumLayer {
            num_vars: n_leaves,
            numerator: leaf_numerator,
            denominator: leaf_denominator,
        });

        for k in 0..n_leaves {
            let child = &layers[k];
            let parent = fold_one_layer::<F>(child, cfg);
            layers.push(parent);
        }

        // layers is [leaves, layer_{n-1}, ..., root]. Reverse so [root, ..., leaves].
        layers.reverse();

        Self {
            layers,
            _phantom: PhantomData,
        }
    }
}

/// Fold a child layer into its parent: splits by the highest variable
/// and applies the grand-sum fold.
#[allow(clippy::arithmetic_side_effects)]
fn fold_one_layer<F>(
    child: &GrandSumLayer<F::Inner>,
    cfg: &F::Config,
) -> GrandSumLayer<F::Inner>
where
    F: PrimeField + InnerTransparentField,
{
    assert!(child.num_vars >= 1, "cannot fold a 0-var layer");
    let nvp = child.num_vars - 1;
    let half = 1usize << nvp;

    let (n_lo, n_hi) = child.numerator.evaluations.split_at(half);
    let (d_lo, d_hi) = child.denominator.evaluations.split_at(half);

    let mut parent_n = Vec::with_capacity(half);
    let mut parent_d = Vec::with_capacity(half);
    for i in 0..half {
        // Lift each inner value to F, do arithmetic in F, then lower back.
        let n0 = F::new_unchecked_with_cfg(n_lo[i].clone(), cfg);
        let n1 = F::new_unchecked_with_cfg(n_hi[i].clone(), cfg);
        let d0 = F::new_unchecked_with_cfg(d_lo[i].clone(), cfg);
        let d1 = F::new_unchecked_with_cfg(d_hi[i].clone(), cfg);

        let n_parent = n0.clone() * &d1 + n1 * &d0;
        let d_parent = d0 * &d1;
        parent_n.push(n_parent.into_inner());
        parent_d.push(d_parent.into_inner());
    }

    let zero_inner = F::zero_with_cfg(cfg).into_inner();
    GrandSumLayer {
        num_vars: nvp,
        numerator: DenseMultilinearExtension::from_evaluations_vec(
            nvp,
            parent_n,
            zero_inner.clone(),
        ),
        denominator: DenseMultilinearExtension::from_evaluations_vec(nvp, parent_d, zero_inner),
    }
}

/// Split an MLE by its highest variable: returns `(P_0, P_1)` with
/// `P_b(x) = P(x, b)`. Each has `num_vars - 1` variables.
#[allow(clippy::arithmetic_side_effects)]
pub fn split_last_variable<T: Clone>(
    mle: &DenseMultilinearExtension<T>,
    zero: &T,
) -> (DenseMultilinearExtension<T>, DenseMultilinearExtension<T>) {
    assert!(mle.num_vars >= 1, "cannot split a 0-var MLE");
    let half = 1usize << (mle.num_vars - 1);
    let nvp = mle.num_vars - 1;
    let first = mle.evaluations[..half].to_vec();
    let second = mle.evaluations[half..].to_vec();
    (
        DenseMultilinearExtension::from_evaluations_vec(nvp, first, zero.clone()),
        DenseMultilinearExtension::from_evaluations_vec(nvp, second, zero.clone()),
    )
}
