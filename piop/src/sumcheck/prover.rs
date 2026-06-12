//! Prover

use std::slice;

use crypto_primitives::{FromPrimitiveWithConfig, PrimeField};
#[cfg(feature = "parallel")]
use rayon::iter::*;
use zinc_poly::{
    EvaluatablePolynomial,
    mle::{DenseMultilinearExtension, MultilinearExtensionWithConfig},
    univariate::nat_evaluation::NatEvaluatedPoly,
};
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
        }
    }
}

impl<F> ProverState<F>
where
    F: InnerTransparentField,
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

        let polys = &self.mles;

        struct Scratch<R> {
            evals: Vec<R>,
            steps: Vec<R>,
            vals0: Vec<R>,
            vals: Vec<R>,
            levals: Vec<R>,
        }
        let zero = F::zero_with_cfg(config);
        let zero_vec_deg = vec![zero.clone(); degree + 1];
        let scratch = || Scratch {
            evals: zero_vec_deg.clone(),
            steps: Vec::with_capacity(polys.len()),
            vals0: Vec::with_capacity(polys.len()),
            vals: Vec::with_capacity(polys.len()),
            levals: Vec::with_capacity(degree + 1),
        };

        #[cfg(not(feature = "parallel"))]
        let zeros = scratch();
        #[cfg(feature = "parallel")]
        let zeros = scratch;

        let summer = cfg_into_iter!(0..1 << (nv - i)).fold(zeros, |mut s, b| {
            let index = b << 1;

            // TODO(Alex): Once you have benches set,
            //             could please try getting rid of vals0 and vals1 fields in the
            // structs, replacing them with
            //
            //             ```rust
            //             let vals0: Vec<_> = polys.iter().map(|poly|
            // poly[index].clone()).collect();             let vals1: Vec<_> =
            // polys.iter().map(|poly| poly[index + 1].clone()).collect();
            //             ```
            //             My bet is that it won't affect running time, but better safe than
            // sorry.

            s.vals0.clear();
            s.vals0.extend(
                polys
                    .iter()
                    .map(|poly| F::new_unchecked_with_cfg(poly[index].clone(), config)),
            );
            s.levals.clear();
            s.levals.push(comb_fn(&s.vals0));

            if degree > 0 {
                s.vals.clear();
                s.vals.extend(
                    polys
                        .iter()
                        .map(|poly| F::new_unchecked_with_cfg(poly[index + 1].clone(), config)),
                );
                s.levals.push(comb_fn(&s.vals));

                s.steps.clear();
                s.steps.extend(
                    s.vals
                        .iter()
                        .zip(s.vals0.iter())
                        .map(|(v1, v0)| v1.clone() - v0.clone()),
                );

                for _ in 2..=degree {
                    for (value, step) in s.vals.iter_mut().zip(s.steps.iter()) {
                        *value += step;
                    }
                    s.levals.push(comb_fn(&s.vals));
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

pub(crate) struct EqualityFactorizedRoundOutput<F> {
    pub asserted_sum: Option<F>,
    pub tail_evaluations: Vec<F>,
}

pub(crate) struct EqualityFactorizedProver<F: PrimeField, C> {
    beta: Vec<F>,
    h_degree: usize,
    h_poly: Vec<DenseMultilinearExtension<F::Inner>>,
    h_comb_fn: C,
    round: usize,
    asserted_sum: Option<F>,
    eq_prefix_scale: Option<F>,
    suffix_eq_weights: Option<Vec<F::Inner>>,
    product_minus_fast: bool,
}

impl<F, C> EqualityFactorizedProver<F, C>
where
    F: FromPrimitiveWithConfig + InnerTransparentField,
    C: Fn(&[F]) -> F + Send + Sync,
{
    pub fn new(
        beta: Vec<F>,
        h_degree: usize,
        h_poly: Vec<DenseMultilinearExtension<F::Inner>>,
        h_comb_fn: C,
    ) -> Self {
        Self {
            beta,
            h_degree,
            h_poly,
            h_comb_fn,
            round: 0,
            asserted_sum: None,
            eq_prefix_scale: None,
            suffix_eq_weights: None,
            product_minus_fast: false,
        }
    }

    pub fn new_product_minus(
        beta: Vec<F>,
        h_poly: Vec<DenseMultilinearExtension<F::Inner>>,
        h_comb_fn: C,
    ) -> Self {
        Self {
            beta,
            h_degree: 2,
            h_poly,
            h_comb_fn,
            round: 0,
            asserted_sum: None,
            eq_prefix_scale: None,
            suffix_eq_weights: None,
            product_minus_fast: true,
        }
    }

    pub fn prefix_len(&self) -> usize {
        self.beta.len()
    }

    pub fn verifier_degree(&self) -> usize {
        self.h_degree + 1
    }

    fn eq_prefix_scale(&self, config: &F::Config) -> F {
        self.eq_prefix_scale
            .clone()
            .unwrap_or_else(|| F::one_with_cfg(config))
    }

    fn eq_line_at(beta: &F, t: &F, config: &F::Config) -> F {
        let one = F::one_with_cfg(config);
        let two_beta = beta.clone() + beta;
        (one.clone() - beta) + t.clone() * (two_beta - one)
    }

    fn bind_round(&mut self, round: usize, challenge: &F, config: &F::Config) {
        for mle in &mut self.h_poly {
            mle.fix_variables_with_config(slice::from_ref(challenge), config);
        }
        let eq_at_challenge = Self::eq_line_at(&self.beta[round], challenge, config);
        self.eq_prefix_scale = Some(self.eq_prefix_scale(config) * eq_at_challenge);
    }

    fn build_next_suffix_weights(
        weights: &[F::Inner],
        beta: &F,
        config: &F::Config,
    ) -> Vec<F::Inner> {
        let zero = F::zero_with_cfg(config).inner().clone();
        let mut next = vec![zero; weights.len() << 1];
        cfg_iter_mut!(next).enumerate().for_each(|(i, val)| {
            let base = F::new_unchecked_with_cfg(weights[i >> 1].clone(), config);
            let with_bit_set = beta.clone() * &base;
            *val = if (i & 1) == 0 {
                (base - with_bit_set).into_inner()
            } else {
                with_bit_set.into_inner()
            };
        });
        next
    }

    fn ensure_suffix_eq_weights(&mut self, config: &F::Config) {
        if self.suffix_eq_weights.is_some() {
            return;
        }

        let mut weights = vec![F::one_with_cfg(config).inner().clone()];

        for beta in self
            .beta
            .iter()
            .rev()
            .take(self.beta.len().saturating_sub(1))
        {
            weights = Self::build_next_suffix_weights(&weights, beta, config);
        }

        self.suffix_eq_weights = Some(weights);
    }

    fn advance_suffix_eq_weights(&mut self, config: &F::Config) {
        let Some(weights) = self.suffix_eq_weights.as_mut() else {
            return;
        };
        if weights.len() <= 1 {
            return;
        }

        let next_len = weights.len() >> 1;
        for idx in 0..next_len {
            weights[idx] = F::add_inner(&weights[idx << 1], &weights[(idx << 1) + 1], config);
        }
        weights.truncate(next_len);
    }

    fn h_round_evaluations(&self, config: &F::Config) -> Vec<F> {
        let suffix_weights = &self
            .suffix_eq_weights
            .as_ref()
            .expect("suffix equality weights should be initialized");

        if self.product_minus_fast {
            debug_assert_eq!(self.h_degree, 2);
            debug_assert_eq!(self.h_poly.len(), 3);
            return self.h_round_evaluations_product_minus(suffix_weights, config);
        }

        if self.h_degree == 2 && self.h_poly.len() == 3 {
            return self.h_round_evaluations_quadratic_three(suffix_weights, config);
        }

        struct Scratch<R> {
            h_evals: Vec<R>,
            vals0: Vec<R>,
            vals: Vec<R>,
            steps: Vec<R>,
        }

        let zero = F::zero_with_cfg(config);
        let zero_evals = vec![zero.clone(); self.h_degree + 1];
        let scratch = || Scratch {
            h_evals: zero_evals.clone(),
            vals0: Vec::with_capacity(self.h_poly.len()),
            vals: Vec::with_capacity(self.h_poly.len()),
            steps: Vec::with_capacity(self.h_poly.len()),
        };

        #[cfg(not(feature = "parallel"))]
        let zeros = scratch();
        #[cfg(feature = "parallel")]
        let zeros = scratch;

        let h_degree = self.h_degree;
        let h_poly = &self.h_poly;
        let h_comb_fn = &self.h_comb_fn;

        let summer = cfg_into_iter!(0..suffix_weights.len()).fold(zeros, |mut s, suffix_idx| {
            let suffix_weight = &suffix_weights[suffix_idx];
            let index = suffix_idx << 1;

            s.vals0.clear();
            s.vals0.extend(
                h_poly
                    .iter()
                    .map(|poly| F::new_unchecked_with_cfg(poly[index].clone(), config)),
            );

            let mut weighted = h_comb_fn(&s.vals0);
            weighted.mul_assign_by_inner(suffix_weight);
            s.h_evals[0] += weighted;
            if h_degree == 0 {
                return s;
            }

            s.vals.clear();
            s.vals.extend(
                h_poly
                    .iter()
                    .map(|poly| F::new_unchecked_with_cfg(poly[index + 1].clone(), config)),
            );

            let mut weighted = h_comb_fn(&s.vals);
            weighted.mul_assign_by_inner(suffix_weight);
            s.h_evals[1] += weighted;

            s.steps.clear();
            s.steps.extend(
                s.vals
                    .iter()
                    .zip(s.vals0.iter())
                    .map(|(v1, v0)| v1.clone() - v0.clone()),
            );

            for out in s.h_evals.iter_mut().take(h_degree + 1).skip(2) {
                for (value, step) in s.vals.iter_mut().zip(s.steps.iter()) {
                    *value += step;
                }
                let mut weighted = h_comb_fn(&s.vals);
                weighted.mul_assign_by_inner(suffix_weight);
                *out += weighted;
            }

            s
        });

        #[cfg(feature = "parallel")]
        let h_evals = summer.map(|s| s.h_evals).reduce(
            || vec![zero.clone(); self.h_degree + 1],
            |mut acc, evals| {
                acc.iter_mut().zip(evals).for_each(|(out, val)| *out += val);
                acc
            },
        );

        #[cfg(not(feature = "parallel"))]
        let h_evals = summer.h_evals;

        h_evals
    }

    fn h_round_evaluations_quadratic_three(
        &self,
        suffix_weights: &[F::Inner],
        config: &F::Config,
    ) -> Vec<F> {
        let zero = F::zero_with_cfg(config);
        let zero_evals = || [zero.clone(), zero.clone(), zero.clone()];
        #[cfg(not(feature = "parallel"))]
        let zeros = zero_evals();
        #[cfg(feature = "parallel")]
        let zeros = zero_evals;
        let h_poly = &self.h_poly;
        let h_comb_fn = &self.h_comb_fn;

        let summer =
            cfg_into_iter!(0..suffix_weights.len()).fold(zeros, |mut evals, suffix_idx| {
                let suffix_weight = &suffix_weights[suffix_idx];
                let index = suffix_idx << 1;

                let a0 = F::new_unchecked_with_cfg(h_poly[0][index].clone(), config);
                let b0 = F::new_unchecked_with_cfg(h_poly[1][index].clone(), config);
                let c0 = F::new_unchecked_with_cfg(h_poly[2][index].clone(), config);
                let vals0 = [a0.clone(), b0.clone(), c0.clone()];
                let mut weighted = h_comb_fn(&vals0);
                weighted.mul_assign_by_inner(suffix_weight);
                evals[0] += weighted;

                let a1 = F::new_unchecked_with_cfg(h_poly[0][index + 1].clone(), config);
                let b1 = F::new_unchecked_with_cfg(h_poly[1][index + 1].clone(), config);
                let c1 = F::new_unchecked_with_cfg(h_poly[2][index + 1].clone(), config);
                let vals1 = [a1.clone(), b1.clone(), c1.clone()];
                let mut weighted = h_comb_fn(&vals1);
                weighted.mul_assign_by_inner(suffix_weight);
                evals[1] += weighted;

                let vals2 = [
                    a1.clone() + (a1 - a0),
                    b1.clone() + (b1 - b0),
                    c1.clone() + (c1 - c0),
                ];
                let mut weighted = h_comb_fn(&vals2);
                weighted.mul_assign_by_inner(suffix_weight);
                evals[2] += weighted;

                evals
            });

        #[cfg(feature = "parallel")]
        let h_evals = summer.reduce(zeros, |mut acc, evals| {
            acc.iter_mut().zip(evals).for_each(|(out, val)| *out += val);
            acc
        });

        #[cfg(not(feature = "parallel"))]
        let h_evals = summer;

        h_evals.into_iter().collect()
    }

    fn h_round_evaluations_product_minus(
        &self,
        suffix_weights: &[F::Inner],
        config: &F::Config,
    ) -> Vec<F> {
        let zero = F::zero_with_cfg(config);
        let zeros = || [zero.clone(), zero.clone(), zero.clone()];
        #[cfg(not(feature = "parallel"))]
        let zeros = zeros();
        #[cfg(feature = "parallel")]
        let zeros = zeros;
        let h_poly = &self.h_poly;

        let summer =
            cfg_into_iter!(0..suffix_weights.len()).fold(zeros, |mut evals, suffix_idx| {
                let suffix_weight = &suffix_weights[suffix_idx];
                let index = suffix_idx << 1;

                let a0 = F::new_unchecked_with_cfg(h_poly[0][index].clone(), config);
                let b0 = F::new_unchecked_with_cfg(h_poly[1][index].clone(), config);
                let c0 = F::new_unchecked_with_cfg(h_poly[2][index].clone(), config);
                let mut weighted = a0.clone() * &b0 - &c0;
                weighted.mul_assign_by_inner(suffix_weight);
                evals[0] += weighted;

                let a1 = F::new_unchecked_with_cfg(h_poly[0][index + 1].clone(), config);
                let b1 = F::new_unchecked_with_cfg(h_poly[1][index + 1].clone(), config);
                let c1 = F::new_unchecked_with_cfg(h_poly[2][index + 1].clone(), config);
                let mut weighted = a1.clone() * &b1 - &c1;
                weighted.mul_assign_by_inner(suffix_weight);
                evals[1] += weighted;

                let a2 = a1.clone() + (a1 - a0);
                let b2 = b1.clone() + (b1 - b0);
                let c2 = c1.clone() + (c1 - c0);
                let mut weighted = a2 * &b2 - &c2;
                weighted.mul_assign_by_inner(suffix_weight);
                evals[2] += weighted;

                evals
            });

        #[cfg(feature = "parallel")]
        let h_evals = summer.reduce(zeros, |mut acc, evals| {
            acc.iter_mut().zip(evals).for_each(|(out, val)| *out += val);
            acc
        });

        #[cfg(not(feature = "parallel"))]
        let h_evals = summer;

        h_evals.into_iter().collect()
    }

    fn extrapolate_h_extra(&self, h_evals: &[F], config: &F::Config) -> F {
        match self.h_degree {
            0 => h_evals[0].clone(),
            1 => {
                let two = F::from_with_cfg(2, config);
                two * &h_evals[1] - &h_evals[0]
            }
            2 => {
                let three = F::from_with_cfg(3, config);
                h_evals[0].clone() - three.clone() * &h_evals[1] + three * &h_evals[2]
            }
            _ => {
                let extrapolation_point = F::from_with_cfg((self.h_degree + 1) as u64, config);
                NatEvaluatedPoly::new(h_evals.to_vec())
                    .evaluate_at_point(&extrapolation_point)
                    .expect("natural evaluations should interpolate")
            }
        }
    }

    fn round_evaluations(&mut self, config: &F::Config) -> Vec<F> {
        self.ensure_suffix_eq_weights(config);
        let mut h_evals = self.h_round_evaluations(config);
        let h_extra = self.extrapolate_h_extra(&h_evals, config);
        h_evals.push(h_extra);

        let beta = &self.beta[self.round];
        let eq_prefix = self.eq_prefix_scale(config);
        let eq_factors = (0..h_evals.len())
            .map(|t| {
                let t = F::from_with_cfg(t as u64, config);
                eq_prefix.clone() * Self::eq_line_at(beta, &t, config)
            })
            .collect::<Vec<_>>();
        h_evals
            .into_iter()
            .zip(eq_factors)
            .map(|(h_at_t, eq_factor)| eq_factor * h_at_t)
            .collect()
    }

    pub fn prove_round(
        &mut self,
        verifier_msg: &Option<F>,
        config: &F::Config,
    ) -> EqualityFactorizedRoundOutput<F> {
        if let Some(r) = verifier_msg {
            if self.round == 0 {
                panic!("first round should be prover first.");
            }
            self.bind_round(self.round - 1, r, config);
        } else if self.round > 0 {
            panic!("verifier message is empty");
        }

        if self.round >= self.beta.len() {
            panic!("Prover is not active");
        }

        let evaluations = self.round_evaluations(config);
        let asserted_sum = if self.round == 0 {
            Some(evaluations[0].clone() + &evaluations[1])
        } else {
            None
        };
        if let Some(sum) = asserted_sum.clone() {
            self.asserted_sum = Some(sum);
        }
        self.round += 1;
        if self.round < self.beta.len() {
            self.advance_suffix_eq_weights(config);
        }

        EqualityFactorizedRoundOutput {
            asserted_sum,
            tail_evaluations: evaluations[1..].to_vec(),
        }
    }

    pub fn finish(
        mut self,
        prefix_challenges: &[F],
        config: &F::Config,
    ) -> DenseMultilinearExtension<F::Inner> {
        debug_assert_eq!(prefix_challenges.len(), self.beta.len());
        if let Some(last_challenge) = prefix_challenges.last() {
            self.bind_round(self.beta.len() - 1, last_challenge, config);
        }

        let vals = self
            .h_poly
            .iter()
            .map(|poly| {
                debug_assert_eq!(poly.evaluations.len(), 1);
                F::new_unchecked_with_cfg(poly.evaluations[0].clone(), config)
            })
            .collect::<Vec<_>>();
        let terminal = self.eq_prefix_scale(config) * (self.h_comb_fn)(&vals);
        DenseMultilinearExtension::from_evaluations_vec(
            0,
            vec![terminal.inner().clone()],
            F::zero_with_cfg(config).inner().clone(),
        )
    }

    pub fn into_terminal_state(
        self,
        challenges: Vec<F>,
        nvars: usize,
        config: &F::Config,
    ) -> ProverState<F> {
        let degree = self.verifier_degree();
        let asserted_sum = self
            .asserted_sum
            .clone()
            .expect("asserted sum should be recorded after the first prover round");
        let terminal = self.finish(&challenges, config);
        ProverState {
            randomness: challenges,
            mles: vec![terminal],
            num_vars: nvars,
            max_degree: degree,
            round: nvars,
            asserted_sum: Some(asserted_sum),
            skip_next_fold: false,
        }
    }
}
