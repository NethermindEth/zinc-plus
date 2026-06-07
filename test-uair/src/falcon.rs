//! Batched Falcon (FN-DSA / FIPS 206) signature-verification UAIR.
//!
//! Verifies **2^10 signatures at once** under a **shared public key `h`** with
//! **per-signature messages** (so `h` is one shared public polynomial and the
//! hash-point `c` is per-signature). `HashToPoint` and `Decompress` are out of
//! scope: `c` is given as public input and `s_2` as a witness. Design note:
//! `documentation/falcon-arithmetization-design.md`.
//!
//! ## Layout (one signature per row)
//!
//! The trace has `2^10` rows, one signature per row. Each ring element of
//! `Z_q[X]/(X^n+1)` (degree `< n = 512`) is stored as `L = n/32 = 16` **limb
//! cells**, each a degree-`<32` `arbitrary_poly` cell holding 32 consecutive
//! integer coefficients. Keeping cells at the native `W = 32` degree (rather
//! than one giant `D = n` cell) means:
//!
//! - the **norm** reads the coefficients straight off the limb cells via the
//!   coefficient-slice machinery (the integer-coefficient analogue of
//!   booleanity's bit-slice extraction) — **no second representation**; and
//! - the **ring equation** reconstructs the full degree-`<n` polynomial from
//!   its limbs, `P = Σ_{m<L} X^{32 m} · limb_m`, for a single per-row ideal
//!   check.
//!
//! Per signature there are 5 logical polynomials — `h` (shared public), `c`
//! (per-sig public), `s_1`, `s_2`, `u` (witness) — each `L` limbs, so
//! `5·L = 80` `arbitrary_poly` columns, plus one `int` column `slack`.
//!
//! ## Constraints
//!
//! - **Ring equation (per row = per signature)** — Option B (`Z[X]` + quotient):
//!   `s_1 + s_2·h − c − q·u ∈ (X^n+1)`, one `assert_in_ideal` against
//!   `RotationIdeal::<R, N>::new(−1)`. One constraint verifies all 2^10 sigs.
//!   This is [`FalconBatchUair::constrain_general`] below.
//!
//! - **Per-signature norm bound** — `Σ_i s_1[i]² + Σ_i s_2[i]² + slack = ⌊β²⌋`,
//!   `slack ≥ 0`, *for each signature*. This is a booleanity-adapted **zerocheck
//!   over the signature rows**: `Σ_j eq(r,j)·( Σ_slices slice(j)² + slack(j) −
//!   ⌊β²⌋ ) = 0`, where the `slice`s are the coefficient-slices of the `s_1, s_2`
//!   limb cells. The per-row `eq`-zerocheck enforces the bound per signature
//!   (exactly how booleanity enforces `v²−v=0` per row). The per-row combiner is
//!   [`falcon_norm_comb_fn`]. This group is protocol-level (a new sumcheck group
//!   adapting `piop/src/lookup/booleanity.rs`) — not a per-row UAIR constraint —
//!   so it is sketched here and wired in a follow-up (design note §"norm group").
//!
//! ## Open points (see design note)
//!
//! - **Limb count.** `L = N/32`. For Falcon-512 (`n = 512`) that is `16`. (A
//!   request for "5 limbs" would be `n = 160`, a reduced ring — set `L` to
//!   change it; everything is parameterized by `N`, `W`, `L`.)
//! - **Centering / `slack ≥ 0` range checks** (each `s_1[i] ∈ (−q/2, q/2]`) are a
//!   separate bit-decomposition + booleanity pass, not yet wired here.
//! - **Per-row scalar build.** `constrain_general` builds the `X^{32 m}` scalars
//!   inline; these are constants and should be hoisted once the group is wired
//!   (perf, not correctness).

use core::marker::PhantomData;

use crypto_primitives::{ConstSemiring, Semiring};
use rand::RngCore;
use zinc_poly::{mle::DenseMultilinearExtension, univariate::dense::DensePolynomial};
use zinc_uair::{
    ConstraintBuilder, NormSpec, PublicColumnLayout, TotalColumnLayout, TraceRow, Uair,
    UairSignature, UairTrace, ideal::rotation::RotationIdeal,
};

use crate::GenerateRandomTrace;

// ---------------------------------------------------------------------------
// Parameters (Falcon-512; one signature per row).
// ---------------------------------------------------------------------------

/// Limb (chunk) degree bound — the native cell width.
pub const W: usize = 32;
/// Limbs per ring element, `L = N / W`. Falcon-512 ⇒ `16`.
pub const L: usize = 16;
/// Ring degree `n = W · L` (`512` for Falcon-512).
pub const N: usize = W * L;
/// Number of signatures verified at once is `2^SIGS_LOG`.
pub const SIGS_LOG: usize = 10;

/// Falcon modulus `q = 12289`.
pub const Q_FALCON: i32 = 12289;
/// Centered half-modulus `(q−1)/2 = 6144`: `s_1[i] ∈ [−6144, 6144]`.
pub const Q_HALF: i32 = (Q_FALCON - 1) / 2;
/// Acceptance bound `⌊β²⌋`. Falcon-512: `34034726`; Falcon-1024: `70265242`.
pub const BETA_SQ_FALCON512: i32 = 34_034_726;

/// Flat `arbitrary_poly` column bases (each logical polynomial spans `L` limbs).
pub mod cols {
    use super::L;
    /// Shared public key `h` (public; identical on every row).
    pub const H_BASE: usize = 0;
    /// Per-signature hash point `c` (public; varies per row).
    pub const C_BASE: usize = L;
    /// Signature short vector half `s_1` (witness).
    pub const S1_BASE: usize = 2 * L;
    /// Signature short vector half `s_2` (witness).
    pub const S2_BASE: usize = 3 * L;
    /// Mod-`q` / negacyclic quotient `u` (witness; Option B).
    pub const U_BASE: usize = 4 * L;
    /// Total `arbitrary_poly` columns: `h, c, s_1, s_2, u`, each `L` limbs.
    pub const NUM_ARB: usize = 5 * L;
    /// Public `arbitrary_poly` columns: `h, c`.
    pub const NUM_ARB_PUB: usize = 2 * L;

    /// Per-signature squared-norm slack (witness int column).
    pub const SLACK: usize = 0;
    pub const NUM_INT: usize = 1;
    pub const NUM_INT_PUB: usize = 0;
}

// ===========================================================================
// FalconBatchUair — the per-row (per-signature) ring equation.
// ===========================================================================

/// Batched Falcon verification: the per-signature ring equation
/// `s_1 + s_2·h − c − q·u ∈ (X^n+1)`, reconstructed from limb cells.
#[derive(Clone, Debug)]
pub struct FalconBatchUair<R>(PhantomData<R>);

impl<R> Uair for FalconBatchUair<R>
where
    R: ConstSemiring + From<i32> + 'static,
{
    type Ideal = RotationIdeal<R, N>;
    /// Scalars must hold `X^{W·(L−1)}` (degree `W·(L−1) = 480 < N`) for limb
    /// reconstruction, so the scalar degree bound is `N`.
    type Scalar = DensePolynomial<R, N>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(0, cols::NUM_ARB, cols::NUM_INT);
        let public = PublicColumnLayout::new(0, cols::NUM_ARB_PUB, cols::NUM_INT_PUB);
        // No shifts: the ring equation is row-local (one signature per row).
        // The per-signature norm zerocheck consumes the s_1 then s_2 limb
        // columns' coefficient-slices, plus the `slack` int column.
        let mut coeff_slice_arb_cols: Vec<usize> = (cols::S1_BASE..cols::S1_BASE + L).collect();
        coeff_slice_arb_cols.extend(cols::S2_BASE..cols::S2_BASE + L);
        let norm_spec = NormSpec {
            coeff_slice_arb_cols,
            slack_int_col: cols::SLACK,
            beta_sq: BETA_SQ_FALCON512 as i64,
        };
        UairSignature::new(total, public, vec![], vec![], vec![]).with_norm_spec(norm_spec)
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        _down: TraceRow<B::Expr>,
        _from_ref: FromR,
        mbs: MulByScalar,
        ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&Self::Scalar) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &Self::Scalar) -> Option<B::Expr>,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
    {
        let arb = up.arbitrary_poly;

        // Reconstruct a degree-<N polynomial from its L limbs:
        //   P(X) = Σ_{m<L} X^{W·m} · limb_m(X),   deg(limb_m) < W.
        let reconstruct = |base: usize| -> B::Expr {
            let mut acc = arb[base].clone(); // m = 0: X^0 = 1
            for m in 1..L {
                let mut coeffs: [R; N] = core::array::from_fn(|_| R::ZERO);
                coeffs[W * m] = R::ONE; // X^{W·m}
                let x_pow = DensePolynomial::<R, N>::new(coeffs);
                acc = acc + &mbs(&arb[base + m], &x_pow).expect("X^{Wm}·limb overflow");
            }
            acc
        };

        let h = reconstruct(cols::H_BASE);
        let c = reconstruct(cols::C_BASE);
        let s1 = reconstruct(cols::S1_BASE);
        let s2 = reconstruct(cols::S2_BASE);
        let u = reconstruct(cols::U_BASE);

        // q·u  (Option B: explicit mod-q quotient).
        let q_scalar = {
            let mut cf: [R; N] = core::array::from_fn(|_| R::ZERO);
            cf[0] = R::from(Q_FALCON);
            DensePolynomial::<R, N>::new(cf)
        };
        let q_u = mbs(&u, &q_scalar).expect("q·u overflow");

        // residual = s_1 + s_2·h − c − q·u  ∈  (X^n + 1)
        let s2_h = s2 * &h;
        let residual = s1 + &s2_h - &c - &q_u;

        let negacyclic = ideal_from_ref(&RotationIdeal::<R, N>::new(R::from(-1)));
        b.assert_in_ideal(residual, &negacyclic);
    }
}

// ---------------------------------------------------------------------------
// Witness generation (integer arithmetic; not generic over R).
// ---------------------------------------------------------------------------

/// `a · b mod (X^N + 1)` — negacyclic convolution over the integers.
fn negacyclic_mul(a: &[i64], b: &[i64]) -> Vec<i64> {
    let mut out = vec![0i64; N];
    for i in 0..N {
        let ai = a[i];
        if ai == 0 {
            continue;
        }
        for (j, &bj) in b.iter().enumerate() {
            let prod = ai * bj;
            let k = i + j;
            if k < N {
                out[k] += prod;
            } else {
                out[k - N] -= prod; // X^N ≡ −1
            }
        }
    }
    out
}

/// Sample an integer uniformly in `[−bound, bound]`.
fn sample_centered(rng: &mut (impl RngCore + ?Sized), bound: i64) -> i64 {
    (rng.next_u32() % (2 * bound as u32 + 1)) as i64 - bound
}

/// Generate a valid short `(s_1, s_2)` under public key `h`, the matching hash
/// point `c = s_1 + s_2·h mod q ∈ [0, q)`, the mod-`q`/negacyclic quotient `u`
/// (so `s_1[k] + (s_2·h mod X^N+1)[k] = c[k] + q·u[k]`, hence the residual
/// `s_1 + s_2·h − c − q·u` is exactly `0 mod (X^N+1)`), and `slack = ⌊β²⌋ −
/// ‖(s_1, s_2)‖²`. Returns length-`N` coefficient vectors.
fn gen_falcon_signature(
    h: &[i64],
    q: i64,
    beta_sq: i64,
    bound: i64,
    rng: &mut (impl RngCore + ?Sized),
) -> (Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>, i64) {
    let s1: Vec<i64> = (0..N).map(|_| sample_centered(rng, bound)).collect();
    let s2: Vec<i64> = (0..N).map(|_| sample_centered(rng, bound)).collect();
    let p = negacyclic_mul(&s2, h);
    let mut c = vec![0i64; N];
    let mut u = vec![0i64; N];
    for k in 0..N {
        let val = s1[k] + p[k];
        c[k] = val.rem_euclid(q);
        u[k] = val.div_euclid(q);
    }
    let norm: i64 = (0..N).map(|k| s1[k] * s1[k] + s2[k] * s2[k]).sum();
    let slack = beta_sq - norm;
    (s1, s2, c, u, slack)
}

impl<R> GenerateRandomTrace<W> for FalconBatchUair<R>
where
    R: ConstSemiring + From<i32> + 'static,
{
    type PolyCoeff = R;
    type Int = R;

    /// Generate a batch of valid Falcon signatures under a freshly sampled
    /// shared public key `h` (one signature per row). HashToPoint/Decompress
    /// are out of scope, so the hash point `c` is *chosen* as the point the
    /// generated short `(s_1, s_2)` verifies against. Coefficients are stored
    /// in their `L`-limb decomposition (`L = N/W` cells of degree `< W`).
    fn generate_random_trace<Rng: RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, R, R, W> {
        let n_rows = 1usize << num_vars;
        let q = Q_FALCON as i64;
        let beta_sq = BETA_SQ_FALCON512 as i64;
        // |coeff| ≤ COEFF_BOUND keeps ‖(s_1,s_2)‖² ≤ 2·N·bound² < ⌊β²⌋, so
        // `slack ≥ 0`. (Real Falcon norms sit just under ⌊β²⌋; this is a
        // valid, comfortably-short synthetic instance.)
        const COEFF_BOUND: i64 = 120;

        // Shared public key h ∈ [0, q), broadcast to every row.
        let h: Vec<i64> = (0..N)
            .map(|_| (rng.next_u32() % Q_FALCON as u32) as i64)
            .collect();

        let mut arb_cols: Vec<Vec<DensePolynomial<R, W>>> = (0..cols::NUM_ARB)
            .map(|_| Vec::with_capacity(n_rows))
            .collect();
        let mut slack_col: Vec<R> = Vec::with_capacity(n_rows);

        for _ in 0..n_rows {
            let (s1, s2, c, u, slack) =
                gen_falcon_signature(&h, q, beta_sq, COEFF_BOUND, rng);
            for (base, poly) in [
                (cols::H_BASE, &h),
                (cols::C_BASE, &c),
                (cols::S1_BASE, &s1),
                (cols::S2_BASE, &s2),
                (cols::U_BASE, &u),
            ] {
                for m in 0..L {
                    // All stored coefficients fit i32 (h, c < q; s_1, s_2
                    // small; |u| ≲ N·bound; slack ≤ ⌊β²⌋). m·W+p < N always.
                    let coeffs: [R; W] =
                        core::array::from_fn(|p| R::from(poly[m * W + p] as i32));
                    arb_cols[base + m].push(DensePolynomial::new(coeffs));
                }
            }
            slack_col.push(R::from(slack as i32));
        }

        let arbitrary_poly: Vec<DenseMultilinearExtension<DensePolynomial<R, W>>> =
            arb_cols
                .into_iter()
                .map(|cells| cells.into_iter().collect())
                .collect();
        let int: Vec<DenseMultilinearExtension<R>> =
            vec![slack_col.into_iter().collect()];
        UairTrace {
            arbitrary_poly: arbitrary_poly.into(),
            int: int.into(),
            ..Default::default()
        }
    }
}

// ===========================================================================
// Per-signature squared-norm bound — booleanity-adapted zerocheck combiner.
// ===========================================================================

/// Per-row (per-signature) combiner for the squared-norm zerocheck.
///
/// The bound `Σ_i s_1[i]² + Σ_i s_2[i]² + slack = ⌊β²⌋` is enforced *per
/// signature* by a zerocheck over the signature rows, adapting
/// `piop/src/lookup/booleanity.rs`:
///
/// ```text
///   Σ_j eq(r, j) · ( Σ_slices slice(j)² + slack(j) − ⌊β²⌋ ) = 0.
/// ```
///
/// `vals = [slice_0, …, slice_{2N−1}, slack]` are the coefficient-slices of the
/// `s_1` and `s_2` limb cells (extracted like booleanity's bit-slices) plus the
/// per-row `slack`. This returns `Σ slice² + slack`; the group multiplies by
/// `eq(r, j)` and subtracts the public `⌊β²⌋`. Degree 2 (the squares); the
/// per-row `eq`-zerocheck makes the bound hold for every signature
/// independently. `slack ≥ 0` is a separate range check.
pub fn falcon_norm_comb_fn<F: Semiring>(vals: &[F]) -> F {
    let (slices, slack) = vals.split_at(vals.len() - 1);
    let mut acc = slack[0].clone();
    for s in slices {
        acc = acc + &(s.clone() * s);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_primitives::crypto_bigint_int::Int;
    use rand::{RngCore, SeedableRng, rngs::StdRng};
    use zinc_uair::constraint_counter::count_constraints;

    use crate::GenerateRandomTrace;

    const LIMBS: usize = 4;

    #[test]
    fn falcon_batch_ring_eq_is_one_ideal_constraint() {
        // The per-signature ring equation is a single ideal-membership
        // constraint, applied to every row (signature).
        assert_eq!(count_constraints::<FalconBatchUair<Int<LIMBS>>>(), 1);
    }

    #[test]
    fn falcon_norm_comb_fn_squares_and_sums() {
        // slices = [2, 3], slack = 4  ⇒  2² + 3² + 4 = 17.
        let vals = [
            Int::<LIMBS>::from_i8(2),
            Int::<LIMBS>::from_i8(3),
            Int::<LIMBS>::from_i8(4),
        ];
        assert_eq!(falcon_norm_comb_fn(&vals), Int::<LIMBS>::from_i8(17));
    }

    #[test]
    fn negacyclic_mul_wraps_with_sign() {
        // X^{N-1} · X = X^N ≡ −1 (mod X^N + 1): a constant −1.
        let mut a = vec![0i64; N];
        let mut b = vec![0i64; N];
        a[N - 1] = 1;
        b[1] = 1;
        let p = negacyclic_mul(&a, &b);
        assert_eq!(p[0], -1);
        assert!(p[1..].iter().all(|&x| x == 0));
    }

    #[test]
    fn falcon_witness_satisfies_ring_eq_and_norm() {
        let mut rng = StdRng::seed_from_u64(1);
        let q = Q_FALCON as i64;
        let beta_sq = BETA_SQ_FALCON512 as i64;
        let h: Vec<i64> = (0..N)
            .map(|_| (rng.next_u32() % Q_FALCON as u32) as i64)
            .collect();
        for _ in 0..4 {
            let (s1, s2, c, u, slack) = gen_falcon_signature(&h, q, beta_sq, 120, &mut rng);
            // Reduced ring residual s_1 + (s_2·h mod X^N+1) − c − q·u must be 0.
            let p = negacyclic_mul(&s2, &h);
            for k in 0..N {
                assert_eq!(s1[k] + p[k] - c[k] - q * u[k], 0, "ring eq at coeff {k}");
                assert!((0..q).contains(&c[k]), "c[{k}] out of [0, q)");
            }
            // Norm bound: Σ s_1² + Σ s_2² + slack = ⌊β²⌋, slack ≥ 0.
            let norm: i64 = (0..N).map(|k| s1[k] * s1[k] + s2[k] * s2[k]).sum();
            assert_eq!(norm + slack, beta_sq);
            assert!(slack >= 0);
        }
    }

    #[test]
    fn generate_random_trace_has_expected_shape() {
        let mut rng = StdRng::seed_from_u64(7);
        // 2^2 = 4 signatures.
        let trace = FalconBatchUair::<Int<LIMBS>>::generate_random_trace(2, &mut rng);
        assert_eq!(trace.arbitrary_poly.len(), cols::NUM_ARB);
        assert_eq!(trace.int.len(), 1);
        assert_eq!(trace.arbitrary_poly[0].evaluations.len(), 4);
    }
}
