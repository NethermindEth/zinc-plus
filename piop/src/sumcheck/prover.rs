//! Prover

use std::slice;

use crypto_primitives::{FromWithConfig, PrimeField};
#[cfg(feature = "parallel")]
use rayon::iter::*;
use zinc_poly::mle::{DenseMultilinearExtension, MultilinearExtensionWithConfig};
use zinc_transcript::{delegate_transcribable, traits::ConstTranscribable};
use zinc_utils::{cfg_into_iter, cfg_iter_mut, inner_transparent_field::InnerTransparentField};

/// Evaluation of a polynomial on natural points without the constant term.
#[repr(transparent)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NatEvaluatedPolyWithoutConstant<F> {
    /// Evaluations at 1, 2, ... (P(0) is omitted).
    pub tail_evaluations: Vec<F>,
}

impl<F> NatEvaluatedPolyWithoutConstant<F> {
    pub fn new(tail_evaluations: Vec<F>) -> Self {
        Self { tail_evaluations }
    }
}

impl<F> std::ops::Deref for NatEvaluatedPolyWithoutConstant<F> {
    type Target = [F];

    fn deref(&self) -> &Self::Target {
        &self.tail_evaluations
    }
}

impl<F> std::ops::DerefMut for NatEvaluatedPolyWithoutConstant<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.tail_evaluations
    }
}

delegate_transcribable!(NatEvaluatedPolyWithoutConstant<F> { tail_evaluations: Vec<F> }
    where F: PrimeField, F::Inner: ConstTranscribable, F::Modulus: ConstTranscribable);

#[repr(transparent)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProverMsg<F>(pub NatEvaluatedPolyWithoutConstant<F>);

delegate_transcribable!(ProverMsg<F>(NatEvaluatedPolyWithoutConstant<F>)
    where F: PrimeField, F::Inner: ConstTranscribable, F::Modulus: ConstTranscribable);

/// Optional per-round polynomial evaluator that replaces the generic
/// per-point value-array gather in [`ProverState::prove_round`]. When a
/// state carries one, `prove_round` calls it to compute the round
/// polynomial evaluations `[M(0), M(1), …, M(degree)]` directly from the
/// (already-folded) `mles` — instead of rebuilding a `polys.len()`-element
/// value array per hypercube point and invoking the generic `comb_fn`
/// `degree + 1` times.
///
/// The fold (`fix_variables`) and message formation are unchanged, and
/// because field arithmetic is exact the summation order is irrelevant — so
/// the emitted proof is **byte-identical** to the generic path. This is a
/// pure access-pattern / arithmetic-fusion win for combs whose structure
/// lets the prover stream a few MLEs at a time (e.g. the Hadamard
/// discharge's per-`(relation, bit)` triples) rather than scatter-gather all
/// `polys.len()` of them per hypercube point. Round 1 is unaffected:
/// fast-path groups emit it via [`super::multi_degree::Round1FastPath`], so
/// this evaluator only runs for rounds ≥ 2 in such groups.
pub trait RoundPolyEvaluator<F: PrimeField>: Send + Sync {
    /// Return `[M(0), M(1), …, M(degree)]` (length `degree + 1`) for the
    /// round-`round` polynomial over `mles` (each holding
    /// `2^(num_vars − round + 1)` evaluations, folded for the prior rounds).
    /// Same boundary convention as the generic path: each MLE is linear in
    /// the round variable, so its value at boundary point `F::from(x)` is
    /// `mle[2·pt] + F::from(x)·(mle[2·pt+1] − mle[2·pt])`.
    fn round_evals(
        &self,
        mles: &[DenseMultilinearExtension<F::Inner>],
        round: usize,
        num_vars: usize,
        degree: usize,
        config: &F::Config,
    ) -> Vec<F>;
}

/// Sumcheck Prover State.
pub struct ProverState<F: PrimeField> {
    /// Sampled randomness given by the verifier.
    pub randomness: Vec<F>,
    /// Stores the list of multilinear extensions
    /// the sumcheck polynomial is comprised of.
    pub mles: Vec<DenseMultilinearExtension<F::Inner>>,
    /// Number of variables.
    pub num_vars: usize,
    /// Max degree.
    pub max_degree: usize,
    /// The current round number.
    pub round: usize,
    /// Claimed sum for the first round polynomial.
    pub asserted_sum: Option<F>,
    /// When `true`, the next `prove_round` invocation pushes the verifier
    /// challenge into `randomness` but skips the `fix_variables` fold of
    /// `mles`. Used by round-1 fast paths that pre-fold the MLEs as part
    /// of their setup, so the standard prover must not fold them a second
    /// time. The flag is reset to `false` after the skipped fold.
    pub skip_next_fold: bool,
    /// Optional fused round-polynomial evaluator (see [`RoundPolyEvaluator`]).
    /// When `Some`, `prove_round` uses it instead of the generic per-point
    /// gather to compute the round message — byte-identical, faster access
    /// pattern. `None` ⇒ the standard path.
    pub round_evaluator: Option<Box<dyn RoundPolyEvaluator<F>>>,
}

impl<F: PrimeField> ProverState<F> {
    /// Initialize the prover to argue for the sum of products of
    /// MLE's in {0,1}^`num_vars`.
    pub fn new(
        mles: Vec<DenseMultilinearExtension<F::Inner>>,
        nvars: usize,
        degree: usize,
    ) -> Self {
        Self {
            randomness: Vec::with_capacity(nvars),
            mles,
            num_vars: nvars,
            max_degree: degree,
            round: 0,
            asserted_sum: None,
            skip_next_fold: false,
            round_evaluator: None,
        }
    }
}

impl<F> ProverState<F>
where
    F: InnerTransparentField + FromWithConfig<u64>,
{
    /// Receive message from verifier, generate prover message, and proceed to
    /// next round.
    ///
    /// Adapted Jolt's sumcheck implementation.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn prove_round(
        &mut self,
        v_msg: &Option<F>,
        comb_fn: impl Fn(&[F]) -> F + Send + Sync,
        config: &F::Config,
    ) -> ProverMsg<F> {
        if let Some(msg) = v_msg {
            if self.round == 0 {
                panic!("first round should be prover first.");
            }
            self.randomness.push(msg.clone());

            if self.skip_next_fold {
                // Round-1 fast path already produced pre-folded `mles`.
                // Consume the flag and skip the fold; the just-pushed
                // randomness still slots into `randomness[round - 1]`,
                // matching the layout the next round expects.
                self.skip_next_fold = false;
            } else {
                // fix the next variable at the verifier randomness for this round
                let i = self.round;
                let r = self.randomness[i - 1].clone();

                cfg_iter_mut!(self.mles).for_each(|multiplicand| {
                    multiplicand.fix_variables_with_config(slice::from_ref(&r), config);
                });
            }
        } else if self.round > 0 {
            panic!("verifier message is empty");
        }

        self.round += 1;

        if self.round > self.num_vars {
            panic!("Prover is not active");
        }

        let i = self.round;
        let nv = self.num_vars;
        let degree = self.max_degree;

        // Fused fast path: a group may supply a `RoundPolyEvaluator` that
        // computes the round polynomial evaluations directly, bypassing the
        // generic per-point 1..=polys.len() value-array gather below.
        // Byte-identical (exact-field, order-independent sum); see the trait.
        let evaluations: Vec<F> = match self.round_evaluator.as_ref() {
            Some(ev) => ev.round_evals(&self.mles, i, nv, degree, config),
            None => {

        let polys = &self.mles;

        struct Scratch<R> {
            evals: Vec<R>,
            steps: Vec<R>,
            vals0: Vec<R>,
            vals1: Vec<R>,
            vals: Vec<R>,
            levals: Vec<R>,
        }
        let zero = F::zero_with_cfg(config);
        let zero_vec_deg = vec![zero.clone(); degree + 1];
        let zero_vec_poly = vec![zero.clone(); polys.len()];
        let scratch = || Scratch {
            evals: zero_vec_deg.clone(),
            steps: zero_vec_poly.clone(),
            vals0: zero_vec_poly.clone(),
            vals1: zero_vec_poly.clone(),
            vals: zero_vec_poly.clone(),
            levals: zero_vec_deg.clone(),
        };

        #[cfg(not(feature = "parallel"))]
        let zeros = scratch();
        #[cfg(feature = "parallel")]
        let zeros = scratch;

        // Precompute the boundary points `F::from(k)` for k = 2..=degree.
        // The existing code used integer-step extrapolation
        // (`vals[i] += step` per advancement), which is fast but
        // implicitly assumes `F::from(k)` is the integer `k` —
        // i.e. that there's a ring homomorphism `Z → F`. That holds
        // for prime fields (`From<u64>` is the canonical
        // reduction-mod-p) but **fails for binary extension fields**:
        // in characteristic 2 the only such map factors through F_2,
        // so `F::from(2)` is either 0 (collapsing boundary points and
        // breaking Lagrange via zero denominators) or — under our
        // bit-pattern convention — equals the field element `X` (not
        // the integer 2), so integer-step extrapolation no longer
        // hits `F::from(2)`.
        //
        // Fix: extrapolate field-honestly via
        // `M(F::from(k)) = M(0) + F::from(k) · slope`
        // where `slope = M(1) − M(0)`. For prime fields this gives
        // the exact same values the integer-step path produced
        // (since `F::from(k) = k`); for char-2 fields it's
        // correct by construction. Performance: one field
        // multiplication per (point, MLE) instead of one addition,
        // i.e. roughly `degree − 1` extra field-mul per hypercube
        // slot. For sumcheck-shaped `degree ≤ 5` workloads this is
        // a negligible constant.
        let boundary_points_above_one: Vec<F> = (2..=degree)
            .map(|k| F::from_with_cfg(k as u64, config))
            .collect();

        let summer = cfg_into_iter!(0..1 << (nv - i)).fold(zeros, |mut s, b| {
            let index = b << 1;

            s.vals0
                .iter_mut()
                .zip(polys.iter())
                .for_each(|(v0, poly)| *v0.inner_mut() = poly[index].clone());
            s.levals[0] = comb_fn(&s.vals0);

            if degree > 0 {
                s.vals1
                    .iter_mut()
                    .zip(polys.iter())
                    .for_each(|(v1, poly)| *v1.inner_mut() = poly[index + 1].clone());
                s.levals[1] = comb_fn(&s.vals1);

                // `steps[i] = M_i(1) − M_i(0)` — the linear-MLE slope.
                for (i, (v1, v0)) in s.vals1.iter().zip(s.vals0.iter()).enumerate() {
                    s.steps[i] = v1.clone() - v0.clone();
                }

                // For each boundary point `point = F::from(k)` with
                // k = 2..=degree, evaluate the round polynomial
                // contribution as `comb_fn(M_0(point), …, M_k(point))`
                // where `M_i(point) = M_i(0) + point · slope_i`.
                for (kth_eval, point) in s
                    .levals
                    .iter_mut()
                    .take(degree + 1)
                    .skip(2)
                    .zip(boundary_points_above_one.iter())
                {
                    for poly_i in 0..polys.len() {
                        // s.vals[poly_i] = M_i(0) + point * step_i
                        s.vals[poly_i] = s.vals0[poly_i].clone();
                        s.vals[poly_i] += point.clone() * &s.steps[poly_i];
                    }
                    *kth_eval = comb_fn(&s.vals);
                }
            }

            // TODO(Alex): It seems that the only thing
            //             we pass around meaningfully is evals,
            //             so this loop could be reworked to map/reduce - maybe even without
            //             #[cfg(feature = "parallel")]. Would help to get benchmarks up and
            //             running first though.
            s.evals
                .iter_mut()
                .zip(s.levals.iter())
                .for_each(|(e, l)| *e += l);

            s
        });

        // Rayon's fold outputs an iter which still needs to be summed over
        #[cfg(feature = "parallel")]
        let evaluations = summer.map(|s| s.evals).reduce(
            || vec![zero.clone(); degree + 1],
            |mut evaluations, evals| {
                evaluations
                    .iter_mut()
                    .zip(evals)
                    .for_each(|(e, l)| *e += &l);
                evaluations
            },
        );

        #[cfg(not(feature = "parallel"))]
        let evaluations = summer.evals;

                evaluations
            }
        };

        // Record the claimed sum once during the first round.
        if self.round == 1 {
            let p0 = evaluations
                .first()
                .expect("evaluations should always contain the constant term");
            let sum = if degree > 0 {
                p0.clone()
                    + evaluations
                        .get(1)
                        .expect("degree > 0 implies evaluation at 1 is present")
            } else {
                p0.clone()
            };
            self.asserted_sum = Some(sum);
        }

        // Strip the constant term before sending, without re-allocating all elements.
        let mut tail = evaluations;
        tail.remove(0); // leaves P(0) behind; tail holds P(1..)

        ProverMsg(NatEvaluatedPolyWithoutConstant::new(tail))
    }
}
