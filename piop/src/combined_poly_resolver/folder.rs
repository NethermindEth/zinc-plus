use crypto_primitives::PrimeField;
use zinc_uair::{ConstraintBuilder, ConstraintRing, ideal::ImpossibleIdeal};

/// There are several situations where we need to
/// compute an RLC `u_0 + \alpha * u_1 + ... + \alpha ^ k * u_k`,
/// where `u_0,...,u_k` are field evaluations of
/// the constraint polynomials of a UAIR on certain values:
/// $$
/// u_0 = f_0(r_0,...,r_n)
/// ...
/// u_k = f_k(r_0,...,r_n)
/// $$
/// This situation happens twice: in the combined poly resolver
/// prover when we instantiate the sumcheck and batch together
/// all the evaluation claims for the combined polynomial MLEs;
/// and, secondly, in the combined poly resolver verifier where
/// check correctness of the resulting sumcheck claim.
///
/// This constraint builder handles those situations.
/// It's `Expr` associated type is the field `F`, so once
/// an `assert_*` method is called it adds it to the RLC
/// with the next power of the challenge `\alpha`.
pub struct ConstraintFolder<'a, F: PrimeField> {
    /// A reference to precomputed powers of the challenge.
    challenge_powers: &'a [F],
    /// Index of the current constraint,
    /// and therefore the current power of the challenge.
    current_constraint: usize,
    /// The RLC computed so far.
    pub folded_constraints: F,
    /// Optional tag filter: when `Some(t)`, only constraints
    /// asserted via `assert_in_ideal_typed(_, _, t)` contribute to the
    /// fold; off-tag constraints (and `assert_zero` constraints) are
    /// skipped while still advancing the alpha-power index. Used by the
    /// dual-prime F_p branch CPR sumcheck so the consistency check
    /// matches the F_p branch IC's tag-filtered absorbed values.
    tag_filter: Option<ConstraintRing>,
}

impl<'a, F: PrimeField> ConstraintFolder<'a, F> {
    pub fn new(challenge_powers: &'a [F], zero: &F) -> Self {
        Self {
            challenge_powers,
            current_constraint: 0,
            folded_constraints: zero.clone(),
            tag_filter: None,
        }
    }

    pub fn new_with_tag_filter(
        challenge_powers: &'a [F],
        zero: &F,
        tag_filter: ConstraintRing,
    ) -> Self {
        Self {
            challenge_powers,
            current_constraint: 0,
            folded_constraints: zero.clone(),
            tag_filter: Some(tag_filter),
        }
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn fold_constraint(&mut self, expr: F) {
        self.folded_constraints += expr * &self.challenge_powers[self.current_constraint];
        self.current_constraint += 1;
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn skip_constraint(&mut self) {
        self.current_constraint += 1;
    }
}

impl<'a, F: PrimeField> ConstraintBuilder for ConstraintFolder<'a, F> {
    type Expr = F;

    type Ideal = ImpossibleIdeal;

    #[inline(always)]
    fn assert_in_ideal(&mut self, expr: Self::Expr, _ideal: &Self::Ideal) {
        // No tag attached: when a filter is active treat as off-tag (skip).
        if self.tag_filter.is_some() {
            self.skip_constraint();
        } else {
            self.fold_constraint(expr);
        }
    }

    #[inline(always)]
    fn assert_in_ideal_typed(
        &mut self,
        expr: Self::Expr,
        _ideal: &Self::Ideal,
        ring: ConstraintRing,
    ) {
        match self.tag_filter {
            Some(t) if t != ring => self.skip_constraint(),
            _ => self.fold_constraint(expr),
        }
    }

    /// `assert_zero` constraints contribute to the folded RLC like any
    /// other constraint. For an honest prover they evaluate to zero on
    /// every row of the hypercube, so their fold-sum is 0 — but the
    /// polynomial expression itself can have per-variable degree > 1
    /// (e.g. `b·(b-1)·s_accum`), so the sumcheck protocol must run at
    /// `count_max_degree::<U>() + 2`, NOT `count_effective_max_degree`.
    /// A previous version of this method was a no-op coupled with
    /// `count_effective_max_degree` for the protocol degree; that
    /// combination silently dropped the binding between assert_zero
    /// constraints and the witness, breaking soundness for every UAIR
    /// with assert_zero constraints.
    ///
    /// Under a tag filter, `assert_zero` slots are skipped: their
    /// fold-contribution is zero anyway for an honest prover.
    #[inline(always)]
    fn assert_zero(&mut self, expr: Self::Expr) {
        if self.tag_filter.is_some() {
            self.skip_constraint();
        } else {
            self.fold_constraint(expr);
        }
    }
}
