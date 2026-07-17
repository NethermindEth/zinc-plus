use crate::{
    ConstraintBuilder, TraceRow, Uair,
    dummy_semiring::{DUMMY_SEMIRING_CONFIG, DummySemiring},
    ideal::ImpossibleIdeal,
};
use zinc_utils::{add, from_ref::FromRef};

/// Per-family breakdown of constraint counts. Returned by
/// [`count_constraints_split`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ConstraintCount {
    /// Number of constraints emitted via
    /// [`crate::ConstraintBuilder::assert_in_ideal`] or
    /// [`crate::ConstraintBuilder::assert_zero`] (the $Q[X]$ family).
    pub q: usize,
    /// Number of constraints emitted via
    /// [`crate::ConstraintBuilder::assert_in_fq_ideal`] (the
    /// $F_{q_i}[X]$ family, aggregated for each prime).
    pub fq: Vec<usize>,
}

impl ConstraintCount {
    pub fn for_prime(&self, idx: usize) -> usize {
        self.fq.get(idx).cloned().unwrap_or(0)
    }

    /// Get the total number of polynomial constraints, summed across
    /// the $Q[X]$ family and all $F_{q_i}[X]$ families.
    pub fn total(&self) -> usize {
        add!(self.q, self.fq.iter().sum())
    }
}

/// Get the per-family constraint counts in a `Uair`.
pub fn count_constraints<U: Uair>() -> ConstraintCount {
    let mut cc = ConstraintCounter::default();

    let sig = U::signature();
    let (up_dummy, down_dummy) = sig.dummy_rows(DummySemiring);
    let up_row = TraceRow::from_slice_with_layout(&up_dummy, sig.total_cols().as_column_layout());
    let down_row =
        TraceRow::from_slice_with_layout(&down_dummy, sig.down_cols().as_column_layout());

    U::constrain_general(
        &mut cc,
        &DUMMY_SEMIRING_CONFIG,
        up_row,
        down_row,
        |_| DummySemiring,
        |_, _| Some(DummySemiring),
        ImpossibleIdeal::from_ref,
        ImpossibleIdeal::from_ref,
    );

    ConstraintCount { q: cc.q, fq: cc.fq }
}

#[derive(Clone, Debug, Default)]
struct ConstraintCounter {
    q: usize,
    fq: Vec<usize>,
}

impl ConstraintBuilder for ConstraintCounter {
    type Expr = DummySemiring;
    type Ideal = ImpossibleIdeal;
    type FqIdeal = ImpossibleIdeal;

    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn assert_in_ideal(&mut self, _expr: Self::Expr, _ideal_generator: &Self::Ideal) {
        self.q += 1;
    }

    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn assert_zero(&mut self, _expr: Self::Expr) {
        self.q += 1;
    }

    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn assert_in_fq_ideal(&mut self, prime_idx: usize, _expr: Self::Expr, _ideal: &Self::FqIdeal) {
        if self.fq.len() <= prime_idx {
            self.fq.resize(prime_idx + 1, 0);
        }
        self.fq[prime_idx] += 1;
    }
}
