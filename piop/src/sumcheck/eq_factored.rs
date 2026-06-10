//! **Eq-factored sumcheck prover** for combs of the shape
//! `Σ_x Σ_t eq(x; q_t) · Σ_i L_{t,i}(x)·R_{t,i}(x)` — an eq-weighted sum of
//! inner products of multilinear pairs, the shape shared by the GKR
//! product-tree layers (`one group, one pair: eq·L·R`) and the F_2
//! lookup-adder binding (`one group, one (mask, Q) pair per relation`).
//!
//! The eq factors are **never materialised as sumcheck multiplicands and
//! never folded**. Round `j` (fixing variable `j−1`, the LOW bit — the same
//! order as the generic prover) writes each group's contribution as
//!
//! ```text
//!   M_j(c) = Σ_t A_{t,j} · eq1(c; q_t[j−1]) · H_{t,j}(c)
//!   A_{t,j}    = Π_{i<j−1} eq1(ρ_{i+1}; q_t[i])          (prefix scalar)
//!   H_{t,j}(c) = Σ_b V_{t,j}[b] · Σ_i L_i(c,b)·R_i(c,b)  (degree 2 in c)
//!   V_{t,j}[b] = Π_{i≥j} eq1(b_{i−j}; q_t[i])            (suffix tensor)
//! ```
//!
//! Suffix tensors are precomputed back-to-front (total `O(2^k)` per group —
//! the cost of ONE eq build, no divisions: the generic field bounds have no
//! inverse) and read densely; only the `L`/`R` buffers fold. Each `H_t` is
//! quadratic, so its value at the fourth Lagrange node is free in
//! characteristic 2: the nodes `{0, 1, X, X+1}` (`F::from(0..=3)` under the
//! bit-pattern convention) form an affine 2-flat, over which every
//! polynomial of degree ≤ 2 sums to zero ⇒ `H(X+1) = H(0) + H(1) + H(X)`.
//! Char 2 is detected exactly at runtime (`1+1 == 0`); any other field
//! accumulates `H(c3)` in the same pass, keeping the driver field-generic.
//!
//! **Byte-identical** to [`MLSumcheck::prove_as_subprotocol`] over the
//! materialised `[eq_1, …, eq_T, all L's, all R's]` with the degree-3 comb
//! `Σ_t eq_t·Σ_i L_i·R_i`: the same round polynomials evaluated at the same
//! nodes, the same transcript ops (the `nvars`/`degree` header, the `P(1..)`
//! tail absorb, the post-draw challenge re-absorb), the same proof layout —
//! the generic [`MLSumcheck::verify_as_subprotocol`] verifies it unchanged,
//! so every caller's existing test doubles as an equivalence check.

use crypto_primitives::FromPrimitiveWithConfig;
use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_utils::{cfg_chunks_mut, cfg_into_iter, cfg_iter, inner_transparent_field::InnerTransparentField};

use super::prover::{NatEvaluatedPolyWithoutConstant, ProverMsg};
use super::SumcheckProof;

/// One eq-weighted group: contributes `eq(x; q)·Σ_i pairs[i].0(x)·pairs[i].1(x)`
/// to the proven sum. All groups (and all pair vectors) must share the same
/// number of variables `k = q.len()`, with `2^k`-length pair vectors.
pub struct EqInnerGroup<F> {
    /// The eq point of this group.
    pub q: Vec<F>,
    /// The `(L_i, R_i)` multilinear pairs (evaluation vectors, consumed —
    /// they become the fold buffers).
    pub pairs: Vec<(Vec<F>, Vec<F>)>,
}

/// Per-group suffix tensors `V_j` (`j = 1..=k`), built back-to-front:
/// `V_k = [1]`, `V_j[2b' | b0] = eq1(b0; q[j])·V_{j+1}[b']`.
#[allow(clippy::arithmetic_side_effects)]
fn suffix_tensors<F>(q: &[F], field_cfg: &F::Config) -> Vec<Vec<F>>
where
    F: InnerTransparentField + Send + Sync,
    F::Config: Sync,
{
    let k = q.len();
    let one = F::one_with_cfg(field_cfg);
    let zero = F::zero_with_cfg(field_cfg);
    let mut suffix = Vec::with_capacity(k);
    let mut cur = vec![one.clone()];
    suffix.push(cur.clone());
    for j in (1..k).rev() {
        let e1 = q[j].clone();
        let e0 = one.clone() - &e1;
        let mut next = vec![zero.clone(); cur.len() * 2];
        cfg_chunks_mut!(next, 2).zip(cfg_iter!(cur)).for_each(|(pair, v)| {
            pair[0] = v.clone() * &e0;
            pair[1] = v.clone() * &e1;
        });
        suffix.push(next.clone());
        cur = next;
    }
    suffix.reverse(); // suffix[j−1] = V_j, length 2^{k−j}
    suffix
}

/// Prove `Σ_x Σ_t eq(x; q_t)·Σ_i L_{t,i}(x)·R_{t,i}(x)` (see the module
/// doc). Returns `(proof, point, final_evals)` where `final_evals[t][i] =
/// (L_{t,i}(point), R_{t,i}(point))` — the per-pair multilinear evaluations
/// at the sumcheck point, interpolated from the fold buffers exactly as the
/// generic prover's final-state interpolation does.
#[allow(clippy::arithmetic_side_effects, clippy::type_complexity)]
pub fn prove_eq_inner_sumcheck<F>(
    transcript: &mut impl Transcript,
    groups: Vec<EqInnerGroup<F>>,
    field_cfg: &F::Config,
) -> (SumcheckProof<F>, Vec<F>, Vec<Vec<(F, F)>>)
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
    F::Modulus: ConstTranscribable,
    F::Config: Sync,
{
    let k = groups.first().map_or(0, |g| g.q.len());
    debug_assert!(k >= 1, "eq-factored sumcheck needs ≥ 1 variable");
    debug_assert!(groups.iter().all(|g| {
        g.q.len() == k && g.pairs.iter().all(|(l, r)| l.len() == 1 << k && r.len() == 1 << k)
    }));
    let one = F::one_with_cfg(field_cfg);
    let zero = F::zero_with_cfg(field_cfg);
    // The generic path's boundary nodes: F::from(2) = X, F::from(3) = X+1
    // (bit-pattern convention; for prime fields these are the integers).
    let c2 = F::from_with_cfg(2u64, field_cfg);
    let c3 = F::from_with_cfg(3u64, field_cfg);
    // See the module doc: in char 2 the fourth node is the affine-flat sum.
    let char2 = one.clone() + &one == zero && c3 == c2.clone() + &one;

    let suffix: Vec<Vec<Vec<F>>> = {
        let _g = zinc_utils::prof::scope("eqf:suffix");
        groups.iter().map(|g| suffix_tensors(&g.q, field_cfg)).collect()
    };
    let qs: Vec<Vec<F>> = groups.iter().map(|g| g.q.clone()).collect();
    // Per-group fold buffers.
    let mut bufs: Vec<Vec<(Vec<F>, Vec<F>)>> =
        groups.into_iter().map(|g| g.pairs).collect();
    let num_groups = qs.len();

    let _g = zinc_utils::prof::scope("eqf:rounds");
    let mut buf = vec![0u8; F::Inner::NUM_BYTES];
    // Header — mirror `prove_as_subprotocol`.
    transcript.absorb_random_field(&F::from_with_cfg(k as u64, field_cfg), &mut buf);
    transcript.absorb_random_field(&F::from_with_cfg(3u64, field_cfg), &mut buf);

    let mut a_scalars = vec![one.clone(); num_groups];
    let mut randomness: Vec<F> = Vec::with_capacity(k);
    let mut messages: Vec<ProverMsg<F>> = Vec::with_capacity(k);
    let mut claimed_sum = zero.clone();

    for j in 1..=k {
        let half = bufs
            .iter()
            .find_map(|g| g.first())
            .map_or(0, |(l, _)| l.len() >> 1);

        // Per group: H(0), H(1), H(c2) [, H(c3)] in one parallel pass over b
        // (scalar accumulators only — no per-point allocation).
        let h_acc = || vec![(zero.clone(), zero.clone(), zero.clone(), zero.clone()); num_groups];
        let body = |mut acc: Vec<(F, F, F, F)>, b: usize| {
            for (t, group_bufs) in bufs.iter().enumerate() {
                let mut i0 = zero.clone();
                let mut i1 = zero.clone();
                let mut i2 = zero.clone();
                let mut i3 = zero.clone();
                for (l, r) in group_bufs {
                    let (l0, l1) = (&l[b << 1], &l[(b << 1) | 1]);
                    let (r0, r1) = (&r[b << 1], &r[(b << 1) | 1]);
                    i0 += l0.clone() * r0;
                    i1 += l1.clone() * r1;
                    let dl = l1.clone() - l0;
                    let dr = r1.clone() - r0;
                    let l2 = l0.clone() + &(c2.clone() * &dl);
                    let r2 = r0.clone() + &(c2.clone() * &dr);
                    i2 += l2 * &r2;
                    if !char2 {
                        let l3 = l0.clone() + &(c3.clone() * &dl);
                        let r3 = r0.clone() + &(c3.clone() * &dr);
                        i3 += l3 * &r3;
                    }
                }
                let w = &suffix[t][j - 1][b];
                let a = &mut acc[t];
                a.0 += w.clone() * &i0;
                a.1 += w.clone() * &i1;
                a.2 += w.clone() * &i2;
                if !char2 {
                    a.3 += w.clone() * &i3;
                }
            }
            acc
        };
        #[cfg(feature = "parallel")]
        let hs = cfg_into_iter!(0..half).fold(h_acc, body).reduce(h_acc, |mut a, b| {
            for (x, y) in a.iter_mut().zip(b) {
                x.0 += &y.0;
                x.1 += &y.1;
                x.2 += &y.2;
                x.3 += &y.3;
            }
            a
        });
        #[cfg(not(feature = "parallel"))]
        let hs = (0..half).fold(h_acc(), body);

        // M(c) = Σ_t A_t · eq1(c; q_t[j−1]) · H_t(c) at the four nodes.
        let mut m = (zero.clone(), zero.clone(), zero.clone(), zero.clone());
        for (t, h) in hs.into_iter().enumerate() {
            let qj = &qs[t][j - 1];
            let e0 = one.clone() - qj;
            let e1 = qj.clone();
            let eq1_at =
                |c: &F| -> F { e0.clone() * &(one.clone() - c) + &(e1.clone() * c) };
            let h3 = if char2 { h.0.clone() + &h.1 + &h.2 } else { h.3 };
            let a = &a_scalars[t];
            m.0 += a.clone() * &(e0.clone() * &h.0);
            m.1 += a.clone() * &(e1.clone() * &h.1);
            m.2 += a.clone() * &(eq1_at(&c2) * &h.2);
            m.3 += a.clone() * &(eq1_at(&c3) * &h3);
        }

        if j == 1 {
            claimed_sum = m.0.clone() + &m.1;
        }
        let tail = vec![m.1, m.2, m.3];
        transcript.absorb_random_field_slice(&tail, &mut buf);
        messages.push(ProverMsg(NatEvaluatedPolyWithoutConstant::new(tail)));

        let rho: F = transcript.get_field_challenge(field_cfg);
        transcript.absorb_random_field(&rho, &mut buf);

        // A_{t,j+1} = A_{t,j} · eq1(ρ_j; q_t[j−1]); fold all L, R at ρ_j.
        for (t, a) in a_scalars.iter_mut().enumerate() {
            let qj = &qs[t][j - 1];
            let e = (one.clone() - qj) * &(one.clone() - &rho) + &(qj.clone() * &rho);
            *a = a.clone() * &e;
        }
        if j < k {
            for group_bufs in bufs.iter_mut() {
                for (l, r) in group_bufs.iter_mut() {
                    let fold = |v: &Vec<F>| -> Vec<F> {
                        cfg_into_iter!(0..half)
                            .map(|b| {
                                let v0 = &v[b << 1];
                                let v1 = &v[(b << 1) | 1];
                                v0.clone() + &(rho.clone() * &(v1.clone() - v0))
                            })
                            .collect()
                    };
                    *l = fold(l);
                    *r = fold(r);
                }
            }
            randomness.push(rho);
        } else {
            // Final interpolation of every pair at ρ_k.
            let interp = |v: &Vec<F>| -> F {
                v[0].clone() + &(rho.clone() * &(v[1].clone() - &v[0]))
            };
            let final_evals: Vec<Vec<(F, F)>> = bufs
                .iter()
                .map(|group_bufs| {
                    group_bufs.iter().map(|(l, r)| (interp(l), interp(r))).collect()
                })
                .collect();
            randomness.push(rho);
            return (SumcheckProof { messages, claimed_sum }, randomness, final_evals);
        }
    }
    unreachable!("the final round returns")
}
