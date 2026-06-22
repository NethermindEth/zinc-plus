use std::fmt::{Display, Formatter};
use zinc_utils::{add, from_ref::FromRef};

use crate::{
    ConstraintBuilder, TraceRow, Uair,
    dummy_semiring::DummySemiring,
    ideal::{Ideal, IdealCheck, IdealCheckError},
};

/// A `ConstraintBuilder` that collects ideals used in a `Uair`.
///
/// Both $Q[X]$-only-ideals (from [`ConstraintBuilder::assert_in_ideal`] /
/// [`ConstraintBuilder::assert_zero`]) and new $F_{q_i}[X]$-ideals (from
/// [`ConstraintBuilder::assert_in_fq_ideal`]) are kept in separate vectors, so
/// downstream consumers (the PIOP layer) can dispatch them independently. The
/// $F_q[X]$ ideals are wrapped in [`IdealOrZero`] to reuse the unconditional
/// [`IdealCheck<DummySemiring>`] proxy impl used during the collection;
/// the `Zero` variant is never produced by the collection (there is no
/// `assert_fq_zero`) but downstream consumers may construct it.
pub struct IdealCollector<I: Ideal, IFq: Ideal> {
    pub ideals: Vec<IdealOrZero<I>>,
    /// $F_{q_i}[X]$-ideals indexed by their `prime_idx` into the
    /// owning UAIR's [`crate::UairSignature::primes`] tuple. Empty for
    /// UAIRs with $Q[X]$-only constraints.
    pub fq_ideals: Vec<Vec<IdealOrZero<IFq>>>,
}

impl<I: Ideal, IFq: Ideal> IdealCollector<I, IFq> {
    /// Create a new ideal collector
    /// and hint the number of constraints
    /// a target UAIR might have.
    pub fn new(num_constraints: usize) -> Self {
        Self {
            ideals: Vec::with_capacity(num_constraints),
            fq_ideals: Vec::new(),
        }
    }
}

/// Given a `Uair` and a hint of how many constraints
/// it is going to have, creates an `IdealCollector`
/// object and collects ideals from the `Uair`.
pub fn collect_ideals<U: Uair>(num_constraints: usize) -> IdealCollector<U::Ideal, U::FqIdeal> {
    let mut ideal_collector = IdealCollector::new(num_constraints);

    let sig = U::signature();
    let (up_dummy, down_dummy) = sig.dummy_rows(DummySemiring);
    let up_row = TraceRow::from_slice_with_layout(&up_dummy, sig.total_cols().as_column_layout());
    let down_row =
        TraceRow::from_slice_with_layout(&down_dummy, sig.down_cols().as_column_layout());
    U::constrain(&mut ideal_collector, up_row, down_row);

    ideal_collector
}

impl<I, IFq> ConstraintBuilder for IdealCollector<I, IFq>
where
    I: Ideal,
    IFq: Ideal,
{
    type Expr = DummySemiring;
    type Ideal = IdealOrZero<I>;
    type FqIdeal = IdealOrZero<IFq>;

    fn assert_in_ideal(&mut self, _expr: Self::Expr, ideal: &Self::Ideal) {
        self.ideals.push(ideal.clone());
    }

    fn assert_zero(&mut self, _expr: Self::Expr) {
        self.ideals.push(IdealOrZero::zero());
    }

    fn assert_in_fq_ideal(&mut self, prime_idx: usize, _expr: Self::Expr, ideal: &Self::FqIdeal) {
        if self.fq_ideals.len() <= prime_idx {
            self.fq_ideals.resize(add!(prime_idx, 1), Vec::new());
        }
        self.fq_ideals[prime_idx].push(ideal.clone());
    }
}

/// Either a non-trivial ideal (from `assert_in_ideal`) or the zero ideal
/// (from `assert_zero`).
#[derive(Clone, Copy, Debug)]
pub enum IdealOrZero<I: Ideal> {
    NonZero(I),
    Zero,
}

impl<I: Ideal> IdealOrZero<I> {
    pub fn zero() -> Self {
        IdealOrZero::Zero
    }

    /// Returns `true` if this is the zero ideal
    /// (i.e., the ideal used by `assert_zero` constraints).
    pub fn is_zero_ideal(&self) -> bool {
        matches!(self, IdealOrZero::Zero)
    }

    pub fn map<I2: Ideal>(&self, f: impl FnOnce(&I) -> I2) -> IdealOrZero<I2> {
        match self {
            IdealOrZero::NonZero(ideal) => IdealOrZero::NonZero(f(ideal)),
            IdealOrZero::Zero => IdealOrZero::Zero,
        }
    }
}

impl<I: Ideal> Display for IdealOrZero<I> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "IdealOrZero<")?;
        match self {
            Self::Zero => write!(f, "Zero")?,
            Self::NonZero(ideal) => write!(f, "{}", ideal)?,
        }
        write!(f, ">")?;
        Ok(())
    }
}

impl<I: Ideal> Ideal for IdealOrZero<I> {}

impl<I: Ideal> FromRef<IdealOrZero<I>> for IdealOrZero<I> {
    fn from_ref(value: &IdealOrZero<I>) -> Self {
        value.clone()
    }
}

impl<I: Ideal> FromRef<I> for IdealOrZero<I> {
    fn from_ref(value: &I) -> Self {
        IdealOrZero::NonZero(value.clone())
    }
}

impl<I: Ideal> IdealCheck<DummySemiring> for IdealOrZero<I> {
    fn contains(&self, _value: &DummySemiring) -> Result<bool, IdealCheckError> {
        // Do nothing.
        Ok(true)
    }
}
