use crate::{
    ConstraintBuilder, TraceRow, Uair, dummy_semiring::DummySemiring, ideal::ImpossibleIdeal,
};
use zinc_utils::add;

/// Get the total number of polynomial constraints in a `Uair`, summed across
/// the $\mathbb{Q}[X]$ family and all $\mathbb{F}_{q_i}[X]$ families.
///
/// For dispatch-aware counts use [`count_constraints_q`] /
/// [`count_constraints_fq`] / [`count_constraints`].
pub fn count_constraints_total<U: Uair>() -> usize {
    let split = count_constraints::<U>();
    add!(split.q, split.fq.into_iter().sum())
}

/// Per-family breakdown of constraint counts. Returned by
/// [`count_constraints_split`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ConstraintCount {
    /// Number of constraints emitted via
    /// [`crate::ConstraintBuilder::assert_in_ideal`] or
    /// [`crate::ConstraintBuilder::assert_zero`] (the $\mathbb{Q}[X]$ family).
    pub q: usize,
    /// Number of constraints emitted via
    /// [`crate::ConstraintBuilder::assert_in_fq_ideal`] (the
    /// $\mathbb{F}_{q_i}[X]$ family, aggregated for each prime).
    pub fq: Vec<usize>,
}

impl ConstraintCount {
    pub fn for_prime(&self, idx: usize) -> usize {
        self.fq.get(idx).cloned().unwrap_or(0)
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

    U::constrain(&mut cc, up_row, down_row);

    ConstraintCount { q: cc.q, fq: cc.fq }
}

/// Get the number of $\mathbb{Q}[X]$ constraints in a `Uair`.
pub fn count_constraints_q<U: Uair>() -> usize {
    count_constraints::<U>().q
}

/// Get the number of $\mathbb{F}_{q_i}[X]$ constraints in a `Uair`, summed
/// across all declared primes.
pub fn count_constraints_fq<U: Uair>() -> Vec<usize> {
    count_constraints::<U>().fq
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ConstraintCounter {
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
