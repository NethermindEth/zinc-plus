//! Data types for the shift sumcheck protocol.

use crypto_primitives::PrimeField;

/// A single shift-evaluation claim.
///
/// Represents: `MLE[shift_c(v)](r) = claimed_eval`, where `v` is a
/// committed column (identified by `source_col`) and `c` is the shift.
#[derive(Clone, Debug)]
pub struct ShiftClaim<F: PrimeField> {
    /// Index of the source column in the flattened trace.
    pub source_col: usize,
    /// The shift amount (> 0).
    pub shift_amount: usize,
    /// The evaluation point `r ∈ F^m` at which the shifted MLE was
    /// claimed to evaluate. This is typically the CPR sumcheck point.
    pub eval_point: Vec<F>,
    /// The claimed evaluation value.
    pub claimed_eval: F,
}

/// Round polynomial for the shift sumcheck: degree-2, represented by
/// evaluations at X = 0, 1, 2.
#[derive(Clone, Debug)]
pub struct ShiftRoundPoly<F: PrimeField> {
    pub evals: [F; 3],
}

impl<F: PrimeField> ShiftRoundPoly<F> {
    /// Evaluate the round polynomial at `x` using degree-2 Lagrange
    /// interpolation over nodes {0, 1, 2}.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn evaluate(&self, x: &F) -> F {
        let one = F::one_with_cfg(&x.cfg().clone());
        let two = one.clone() + &one;
        let two_inv = one.clone() / &two;

        let xm1 = x.clone() - &one;
        let xm2 = x.clone() - &two;

        // l_0(x) = (x−1)(x−2)/2
        // l_1(x) = −x(x−2)
        // l_2(x) = x(x−1)/2
        self.evals[0].clone() * &xm1 * &xm2 * &two_inv
            - &(self.evals[1].clone() * x * &xm2)
            + &(self.evals[2].clone() * x * &xm1 * &two_inv)
    }
}

/// Proof for the batched shift sumcheck.
#[derive(Clone, Debug)]
pub struct ShiftSumcheckProof<F: PrimeField> {
    /// One round polynomial per variable (m rounds total).
    pub rounds: Vec<ShiftRoundPoly<F>>,
}

/// Output of the shift sumcheck prover.
pub struct ShiftSumcheckProverOutput<F: PrimeField> {
    /// The proof (round polynomials).
    pub proof: ShiftSumcheckProof<F>,
    /// The random challenge point `s ∈ F^m` produced during the protocol.
    pub challenge_point: Vec<F>,
    /// The final reduced claim value.
    pub final_claim: F,
    /// Per-claim folded shift-table values: `h_i(s) = S_{c_i}(s, r_i)`.
    pub h_finals: Vec<F>,
    /// Per-claim folded witness values: `MLE[v_i](s)`.
    pub v_finals: Vec<F>,
}
