use crypto_primitives::SemiringConfig;
use zinc_uair::{ConstraintBuilder, ideal::ImpossibleIdeal};
use zinc_utils::add;

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
pub struct ConstraintFolder<'a, C: SemiringConfig> {
    /// Family index selecting which family of constraints to fold.
    ///
    /// - `0` -> $Q[X]$ constraints (from `assert_in_ideal` / assert_zero`).
    /// - `i >= 1` -> $F_{q_{i-1}}[X]$ (from `assert_in_fq_ideal`).
    ///
    /// All unrelated constraints are skipped.
    family_idx: usize,
    /// A reference to precomputed powers of the challenge.
    challenge_powers: &'a [C::Element],
    /// Index of the current constraint,
    /// and therefore the current power of the challenge.
    current_constraint: usize,
    /// The RLC computed so far.
    pub folded_constraints: C::Element,
    /// The field config providing operations on the folded elements.
    cfg: &'a C,
}

impl<'a, C: SemiringConfig> ConstraintFolder<'a, C> {
    /// Build a folder for a given `family_idx` (`0` = Q[X], `i >= 1` =
    /// declared prime `i - 1`).
    pub fn new(family_idx: usize, challenge_powers: &'a [C::Element], cfg: &'a C) -> Self {
        Self {
            family_idx,
            challenge_powers,
            current_constraint: 0,
            folded_constraints: cfg.zero(),
            cfg,
        }
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn fold_constraint(&mut self, expr: C::Element) {
        let term = self
            .cfg
            .mul(&expr, &self.challenge_powers[self.current_constraint]);
        self.cfg.add_assign(&mut self.folded_constraints, &term);
        self.current_constraint += 1;
    }
}

impl<'a, C: SemiringConfig> ConstraintBuilder for ConstraintFolder<'a, C> {
    type Expr = C::Element;

    type Ideal = ImpossibleIdeal;
    type FqIdeal = ImpossibleIdeal;

    #[inline(always)]
    fn assert_in_ideal(&mut self, expr: Self::Expr, _ideal: &Self::Ideal) {
        // Q[X] family -> index 0.
        if self.family_idx == 0 {
            self.fold_constraint(expr);
        }
    }

    #[inline(always)]
    fn assert_zero(&mut self, expr: Self::Expr) {
        // Q[X] family -> index 0.
        if self.family_idx == 0 {
            self.fold_constraint(expr);
        }
    }

    #[inline(always)]
    fn assert_in_fq_ideal(&mut self, prime_idx: usize, expr: Self::Expr, _ideal: &Self::FqIdeal) {
        // F_{q_{prime_idx}}[X] family -> index (prime_idx + 1).
        if self.family_idx == add!(prime_idx, 1) {
            self.fold_constraint(expr);
        }
    }
}
