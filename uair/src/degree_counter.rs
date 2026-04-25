use std::{
    fmt::{Debug, Display},
    ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

use crypto_primitives::Semiring;
use num_traits::{CheckedAdd, CheckedMul, CheckedSub};

use crate::{
    ConstraintBuilder, TraceRow, Uair, constraint_counter::count_constraints,
    ideal::ImpossibleIdeal, ideal_collector::collect_ideals,
};

/// Compute the maximum number of multiplicands
/// in products of witness elements in the UAIR `U`.
pub fn count_max_degree<U: Uair>() -> usize {
    count_constraint_degrees::<U>()
        .into_iter()
        .max()
        .unwrap_or(0)
}

/// Like [`count_max_degree`], but excludes constraints asserted via
/// `assert_zero` (equivalently: constraints whose ideal is the zero
/// ideal). For an honest prover, such constraints are identically zero on
/// the hypercube, so their contribution to the combined polynomial is
/// zero and their degree does not constrain the downstream sumcheck.
///
/// Used to drive the MLE-first vs combined path selection and the
/// `fq_sumcheck` degree parameter when zero-ideal constraints can be
/// skipped.
pub fn count_effective_max_degree<U: Uair>() -> usize {
    let degrees = count_constraint_degrees::<U>();
    let ideals = collect_ideals::<U>(count_constraints::<U>()).ideals;
    debug_assert_eq!(
        degrees.len(),
        ideals.len(),
        "degree collector and ideal collector should visit constraints in the same order"
    );
    degrees
        .into_iter()
        .zip(ideals.iter())
        .filter_map(|(deg, ideal)| (!ideal.is_zero_ideal()).then_some(deg))
        .max()
        .unwrap_or(0)
}

/// Per-constraint mask of "linear" constraints — those with degree at most
/// 1 in the trace MLEs. Used by the hybrid ideal-check dispatch in
/// `protocol::prove` to route linear constraints through the MLE-first lane
/// while non-linear constraints stay on the combined-polynomial lane.
///
/// Mask order matches the constraint order produced by
/// [`count_constraint_degrees`] and the ideal collector — both walk the
/// UAIR's `constrain_general` in declaration order.
pub fn linear_constraint_mask<U: Uair>() -> Vec<bool> {
    count_constraint_degrees::<U>()
        .into_iter()
        .map(|d| d <= 1)
        .collect()
}

/// Compute the degree of each individual constraint in the UAIR `U`.
/// Returns a `Vec<usize>` where the i-th element is the degree
/// of the i-th constraint.
pub fn count_constraint_degrees<U: Uair>() -> Vec<usize> {
    let mut dc = ConstraintDegreeCollector::default();

    let sig = U::signature();
    let (up_dummy, down_dummy) = sig.dummy_rows(DegreeCountingSemiring::var());
    let up_row = TraceRow::from_slice_with_layout(&up_dummy, sig.total_cols().as_column_layout());
    let down_row =
        TraceRow::from_slice_with_layout(&down_dummy, sig.down_cols().as_column_layout());

    U::constrain_general(
        &mut dc,
        up_row,
        down_row,
        |_| DegreeCountingSemiring::scalar(),
        |x, _| Some(*x),
        |_| ImpossibleIdeal,
    );

    dc.degrees
}

/// Collects the degree of each constraint in a UAIR by implementing the
/// `ConstraintBuilder` trait.
#[derive(Debug, Default)]
pub(crate) struct ConstraintDegreeCollector {
    degrees: Vec<usize>,
}

impl ConstraintBuilder for ConstraintDegreeCollector {
    type Expr = DegreeCountingSemiring;
    type Ideal = ImpossibleIdeal;

    fn assert_in_ideal(&mut self, expr: Self::Expr, _ideal: &Self::Ideal) {
        self.degrees.push(expr.0);
    }

    fn assert_zero(&mut self, expr: Self::Expr) {
        self.degrees.push(expr.0);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct DegreeCountingSemiring(usize);

impl DegreeCountingSemiring {
    pub fn var() -> Self {
        DegreeCountingSemiring(1)
    }

    pub fn scalar() -> Self {
        DegreeCountingSemiring(0)
    }
}

impl Display for DegreeCountingSemiring {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self, f)
    }
}

macro_rules! impl_binary_additive_op {
    ($trait:ident, $op:ident) => {
        impl $trait<&DegreeCountingSemiring> for DegreeCountingSemiring {
            type Output = Self;

            #[inline(always)]
            fn $op(self, rhs: &DegreeCountingSemiring) -> Self::Output {
                DegreeCountingSemiring(std::cmp::max(self.0, rhs.0))
            }
        }

        impl $trait<DegreeCountingSemiring> for DegreeCountingSemiring {
            type Output = Self;

            #[inline(always)]
            fn $op(self, rhs: DegreeCountingSemiring) -> Self::Output {
                self.$op(&rhs)
            }
        }
    };
}

impl_binary_additive_op!(Add, add);
impl_binary_additive_op!(Sub, sub);

impl Mul<&Self> for DegreeCountingSemiring {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects, clippy::suspicious_arithmetic_impl)]
    #[inline(always)]
    fn mul(self, rhs: &Self) -> Self::Output {
        DegreeCountingSemiring(self.0 + rhs.0)
    }
}

impl Mul<Self> for DegreeCountingSemiring {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        self.mul(&rhs)
    }
}

macro_rules! impl_additive_op_assign {
    ($trait:ident, $op:ident) => {
        impl $trait<&DegreeCountingSemiring> for DegreeCountingSemiring {
            #[inline(always)]
            fn $op(&mut self, rhs: &DegreeCountingSemiring) {
                self.0 = std::cmp::max(self.0, rhs.0);
            }
        }

        impl $trait<DegreeCountingSemiring> for DegreeCountingSemiring {
            #[inline(always)]
            fn $op(&mut self, rhs: DegreeCountingSemiring) {
                self.$op(&rhs);
            }
        }
    };
}

impl_additive_op_assign!(AddAssign, add_assign);
impl_additive_op_assign!(SubAssign, sub_assign);

impl MulAssign<&Self> for DegreeCountingSemiring {
    #[allow(clippy::arithmetic_side_effects, clippy::suspicious_op_assign_impl)]
    #[inline(always)]
    fn mul_assign(&mut self, rhs: &Self) {
        self.0 += rhs.0;
    }
}

impl MulAssign<Self> for DegreeCountingSemiring {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        self.add_assign(&rhs);
    }
}

macro_rules! impl_checked_additive_op {
    ($trait:ident, $op:ident) => {
        impl $trait for DegreeCountingSemiring {
            #[inline(always)]
            fn $op(&self, rhs: &DegreeCountingSemiring) -> Option<Self::Output> {
                Some(DegreeCountingSemiring(std::cmp::max(self.0, rhs.0)))
            }
        }
    };
}

impl_checked_additive_op!(CheckedAdd, checked_add);
impl_checked_additive_op!(CheckedSub, checked_sub);

impl CheckedMul for DegreeCountingSemiring {
    #[inline(always)]
    fn checked_mul(&self, rhs: &Self) -> Option<Self> {
        Some(DegreeCountingSemiring(self.0.checked_add(rhs.0)?))
    }
}

impl Semiring for DegreeCountingSemiring {}
