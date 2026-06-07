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
    ConstraintBuilder, PublicColumnLayout, TotalColumnLayout, TraceRow, Uair, UairSignature,
    UairTrace, ideal::rotation::RotationIdeal,
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
        // No shifts: the ring equation is row-local (one signature per row). The
        // per-signature norm zerocheck consumes the s_1/s_2 limb columns through
        // a separate group (follow-up), not through this signature.
        UairSignature::new(total, public, vec![], vec![], vec![])
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

impl<R> GenerateRandomTrace<W> for FalconBatchUair<R>
where
    R: ConstSemiring + From<i32> + 'static,
{
    type PolyCoeff = R;
    type Int = R;

    /// Trivial placeholder witness: all-zero limbs, for which the residual is
    /// exactly `0 ∈ (X^n+1)`. A real generator fills `h, c` (public), the
    /// decompressed `s_2`, the recomputed `s_1`, the quotient `u`, and `slack`
    /// per signature — future work.
    fn generate_random_trace<Rng: RngCore + ?Sized>(
        num_vars: usize,
        _rng: &mut Rng,
    ) -> UairTrace<'static, R, R, W> {
        let n_rows = 1usize << num_vars;
        let zero_cell = || -> DensePolynomial<R, W> {
            let cf: [R; W] = core::array::from_fn(|_| R::ZERO);
            DensePolynomial::new(cf)
        };
        let zero_col = || -> DenseMultilinearExtension<DensePolynomial<R, W>> {
            (0..n_rows).map(|_| zero_cell()).collect()
        };
        let arb: Vec<_> = (0..cols::NUM_ARB).map(|_| zero_col()).collect();
        let slack: DenseMultilinearExtension<R> = (0..n_rows).map(|_| R::ZERO).collect();
        UairTrace {
            arbitrary_poly: arb.into(),
            int: vec![slack].into(),
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
    use zinc_uair::constraint_counter::count_constraints;

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
}
