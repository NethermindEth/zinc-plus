//! Per-constraint column-use tracker.
//!
//! Drives the dual-prime per-branch projection optimisation: by symbolically
//! evaluating `U::constrain_general` with a `Vec<bool>`-backed semiring whose
//! arithmetic ops union column-use masks, we recover for each constraint the
//! exact set of column indices it touches.
//!
//! Combined with the per-constraint [`crate::ConstraintRing`] tags from
//! [`crate::ideal_collector::IdealCollector`], this lets each branch
//! (`Z` / `Fp`) project only the columns its constraints actually reference
//! — saving the per-cell `Z[X] -> F_q[X]` cost on untouched columns.

use std::{
    fmt::{Debug, Display},
    hash::{Hash, Hasher},
    ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

use crypto_primitives::Semiring;
use num_traits::{CheckedAdd, CheckedMul, CheckedSub};
use zinc_utils::from_ref::FromRef;

use crate::{
    ConstraintBuilder, ConstraintRing, TraceRow, Uair,
    constraint_counter::count_constraints,
    ideal::ImpossibleIdeal,
    ideal_collector::collect_ideals,
};

/// Symbolic semiring carrying a column-use mask.
///
/// Cell at column index `c` starts as a singleton mask with bit `c` set;
/// arithmetic ops `+`, `-`, `*` union the operand masks, faithfully
/// approximating which columns flow into the resulting expression.
#[derive(Clone, Debug, Default)]
pub struct ColumnUseSemiring {
    pub mask: Vec<bool>,
}

impl ColumnUseSemiring {
    fn empty(num_cols: usize) -> Self {
        Self {
            mask: vec![false; num_cols],
        }
    }

    fn singleton(num_cols: usize, col: usize) -> Self {
        let mut mask = vec![false; num_cols];
        mask[col] = true;
        Self { mask }
    }

    fn union_in_place(&mut self, rhs: &Self) {
        debug_assert_eq!(self.mask.len(), rhs.mask.len());
        for (a, b) in self.mask.iter_mut().zip(rhs.mask.iter()) {
            *a = *a || *b;
        }
    }

    fn union(mut self, rhs: &Self) -> Self {
        self.union_in_place(rhs);
        self
    }
}

impl PartialEq for ColumnUseSemiring {
    fn eq(&self, other: &Self) -> bool {
        self.mask == other.mask
    }
}
impl Eq for ColumnUseSemiring {}

impl Hash for ColumnUseSemiring {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.mask.hash(state);
    }
}

impl Display for ColumnUseSemiring {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

macro_rules! impl_binary_op {
    ($trait:ident, $op:ident) => {
        impl $trait<&ColumnUseSemiring> for ColumnUseSemiring {
            type Output = Self;
            fn $op(self, rhs: &ColumnUseSemiring) -> Self::Output {
                self.union(rhs)
            }
        }
        impl $trait<ColumnUseSemiring> for ColumnUseSemiring {
            type Output = Self;
            fn $op(self, rhs: ColumnUseSemiring) -> Self::Output {
                self.union(&rhs)
            }
        }
    };
}

impl_binary_op!(Add, add);
impl_binary_op!(Sub, sub);
impl_binary_op!(Mul, mul);

macro_rules! impl_op_assign {
    ($trait:ident, $op:ident) => {
        impl $trait<&ColumnUseSemiring> for ColumnUseSemiring {
            fn $op(&mut self, rhs: &ColumnUseSemiring) {
                self.union_in_place(rhs);
            }
        }
        impl $trait<ColumnUseSemiring> for ColumnUseSemiring {
            fn $op(&mut self, rhs: ColumnUseSemiring) {
                self.union_in_place(&rhs);
            }
        }
    };
}

impl_op_assign!(AddAssign, add_assign);
impl_op_assign!(SubAssign, sub_assign);
impl_op_assign!(MulAssign, mul_assign);

impl CheckedAdd for ColumnUseSemiring {
    fn checked_add(&self, rhs: &Self) -> Option<Self> {
        Some(self.clone().union(rhs))
    }
}
impl CheckedSub for ColumnUseSemiring {
    fn checked_sub(&self, rhs: &Self) -> Option<Self> {
        Some(self.clone().union(rhs))
    }
}
impl CheckedMul for ColumnUseSemiring {
    fn checked_mul(&self, rhs: &Self) -> Option<Self> {
        Some(self.clone().union(rhs))
    }
}

impl Semiring for ColumnUseSemiring {}

/// `ConstraintBuilder` that records, per constraint, the union of column
/// indices touched by the constraint expression and the constraint's
/// dual-prime branch tag (`Z` or `Fp`).
pub struct ColumnUseTracker {
    pub uses: Vec<ColumnUseSemiring>,
    pub tags: Vec<ConstraintRing>,
    num_cols: usize,
}

impl ColumnUseTracker {
    fn new(num_cols: usize, num_constraints: usize) -> Self {
        Self {
            uses: Vec::with_capacity(num_constraints),
            tags: Vec::with_capacity(num_constraints),
            num_cols,
        }
    }
}

impl ConstraintBuilder for ColumnUseTracker {
    type Expr = ColumnUseSemiring;
    type Ideal = ImpossibleIdeal;

    fn assert_in_ideal(&mut self, expr: Self::Expr, _ideal: &Self::Ideal) {
        debug_assert_eq!(expr.mask.len(), self.num_cols);
        self.uses.push(expr);
        self.tags.push(ConstraintRing::Z);
    }

    fn assert_in_ideal_typed(
        &mut self,
        expr: Self::Expr,
        _ideal: &Self::Ideal,
        ring: ConstraintRing,
    ) {
        debug_assert_eq!(expr.mask.len(), self.num_cols);
        self.uses.push(expr);
        self.tags.push(ring);
    }

    fn assert_zero(&mut self, expr: Self::Expr) {
        debug_assert_eq!(expr.mask.len(), self.num_cols);
        self.uses.push(expr);
        self.tags.push(ConstraintRing::Z);
    }
}

/// Compute, for each `ConstraintRing` branch, the union of column indices
/// touched by any constraint tagged for that branch. `assert_zero`
/// constraints are tagged `Z` by convention; a column referenced by an
/// `assert_zero` is therefore included in the `Z`-branch mask. Length of
/// each returned mask matches `total_cols.cols()` from the UAIR signature.
pub fn compute_branch_column_masks<U: Uair>() -> BranchColumnMasks {
    let sig = U::signature();
    let total_cols = sig.total_cols().cols();
    let down_layout = sig.down_cols().as_column_layout();
    let bit_op_count = sig.bit_op_down_count();
    let num_constraints = count_constraints::<U>();

    let mut up_cells: Vec<ColumnUseSemiring> = (0..total_cols)
        .map(|c| ColumnUseSemiring::singleton(total_cols, c))
        .collect();
    // `down` cells map back to their source column indices via the
    // signature's shift specs — bit-op virtuals are handled by appending
    // their source column at the right slot. The number of down cells is
    // total_down_cols + bit_op_count.
    let down_cols_total = sig.down_cols().cols();
    let mut down_cells: Vec<ColumnUseSemiring> = sig
        .shifts()
        .iter()
        .map(|spec| ColumnUseSemiring::singleton(total_cols, spec.source_col()))
        .collect();
    debug_assert_eq!(down_cells.len(), down_cols_total);
    for spec in sig.bit_op_specs() {
        down_cells.push(ColumnUseSemiring::singleton(total_cols, spec.source_col()));
    }

    let up_row = TraceRow::from_slice_with_layout(&up_cells, sig.total_cols().as_column_layout());
    let down_row =
        TraceRow::from_slice_with_layout_and_bit_op(&down_cells, down_layout, bit_op_count);

    let mut tracker = ColumnUseTracker::new(total_cols, num_constraints);
    U::constrain_general(
        &mut tracker,
        up_row,
        down_row,
        |_| ColumnUseSemiring::empty(total_cols),
        |x, _| Some(x.clone()),
        ImpossibleIdeal::from_ref,
    );

    let _ = (&mut up_cells,); // keep up_cells alive across U::constrain_general call

    let mut z_mask = vec![false; total_cols];
    let mut fp_mask = vec![false; total_cols];
    for (use_set, tag) in tracker.uses.iter().zip(tracker.tags.iter()) {
        let target = match tag {
            ConstraintRing::Z => &mut z_mask,
            ConstraintRing::Fp => &mut fp_mask,
        };
        for (t, u) in target.iter_mut().zip(use_set.mask.iter()) {
            *t = *t || *u;
        }
    }

    // Soundness sanity: every column referenced by any constraint must
    // also be present in `IdealCollector::collect_ideals` ordering — we
    // depend on the same iteration order to align tags and uses.
    debug_assert_eq!(
        tracker.tags.len(),
        collect_ideals::<U>(num_constraints).tags.len()
    );

    BranchColumnMasks { z_mask, fp_mask }
}

/// Per-branch column-projection masks. `mask[col] == true` means the
/// branch must project that column for its ideal check; otherwise the
/// projection helper writes the zero polynomial in that slot.
#[derive(Clone, Debug)]
pub struct BranchColumnMasks {
    pub z_mask: Vec<bool>,
    pub fp_mask: Vec<bool>,
}

impl BranchColumnMasks {
    pub fn for_branch(&self, ring: ConstraintRing) -> &[bool] {
        match ring {
            ConstraintRing::Z => &self.z_mask,
            ConstraintRing::Fp => &self.fp_mask,
        }
    }
}

/// Helper: a fully-masked-in column list (every column projected).
/// Useful for callers that want the unmasked code path without an
/// `Option<&[bool]>` parameter.
pub fn full_mask<U: Uair>() -> Vec<bool> {
    let total_cols = U::signature().total_cols().cols();
    vec![true; total_cols]
}
