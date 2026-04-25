//! ECDSA Jacobian point-doubling UAIR slice (F_p / EC operations — first
//! increment).
//!
//! Spec source: `arithmetization_standalone/hybrid_arithmetics/ecdsa/ecdsa_intro.tex`
//! eqs. 124–127.
//!
//! ## What this slice covers
//!
//! Per row, an **independent** Jacobian doubling on the secp256k1 curve:
//! given input `(X, Y, Z) ∈ F_p^3`, produce `(X_mid, Y_mid, Z_mid)` with
//!
//!     S      = Y²                                  (mod p)
//!     Z_mid  = 2·Y·Z                               (mod p)
//!     X_mid  = 9·X⁴ − 8·X·S                        (mod p)
//!     Y_mid  = 3·X²·(4·X·S − X_mid) − 8·S²         (mod p)
//!
//! No row-to-row chaining, no Shamir loop, no signature check — those are
//! the next increments.
//!
//! ## Direct F_p constraints (no quotient witnesses)
//!
//! The `fixed-prime` branch hardcodes the projecting prime to the
//! secp256k1 base-field prime `p`, so the verifier checks every
//! constraint mod `p`. Each F_p identity becomes a direct constraint:
//!
//! ```text
//!     S − Y² = 0   (mod p, i.e. mod the proving field)
//! ```
//!
//! No quotient witness columns are needed: the verifier's mod-p check is
//! exactly the F_p semantics we want.
//!
//! ## Soundness caveats
//!
//! - The output `(X_mid, Y_mid, Z_mid)` is not range-checked into
//!   `[0, p)` — a malicious prover could store any equivalent
//!   representative. Closing this gap requires range-lookup integration.
//! - Curve-equation membership (`Y² = X³ + 7·Z⁶` in Jacobian form) is not
//!   enforced. The doubling formulas are valid polynomial identities in
//!   F_p regardless of curve membership; binding inputs to the curve is a
//!   separate constraint group, deferred until the full Shamir loop.
//!
//! ## Required Int width
//!
//! Trace cells store F_p elements in `[0, p)` (so `< 2^256`). Constraint
//! expressions evaluated by the prover are reduced mod `p` (the proving
//! field) so wide intermediate Z values never appear in protocol
//! arithmetic. We use `Int<5>` (320 bits, signed): four limbs hold the
//! F_p value, and the top limb's sign bit accommodates any signed
//! intermediates.
//!
//! ## Out of scope
//!
//! - Full Shamir loop (this slice has no row-to-row chaining or shifts).
//! - Jacobian addition, Shamir addend selection, affine conversion via
//!   Z⁻¹, signature modular check.
//! - Composition with `Sha256CompressionSliceUair`.
//! - Public-key validity (Q on curve, non-identity).
//! - End-to-end protocol test.

use core::marker::PhantomData;

use crypto_bigint::{CheckedSub, NonZero, Uint as CbUint};
use crypto_primitives::{ConstSemiring, crypto_bigint_int::Int};
use rand::RngCore;
use zinc_poly::{mle::DenseMultilinearExtension, univariate::dense::DensePolynomial};
use zinc_uair::{
    ConstraintBuilder, PublicColumnLayout, ShiftSpec, TotalColumnLayout, TraceRow, Uair,
    UairSignature, UairTrace,
    ideal::ImpossibleIdeal,
};

use crate::GenerateRandomTrace;

// ---------------------------------------------------------------------------
// Secp256k1 base-field prime `p` and Int width.
// ---------------------------------------------------------------------------

/// Number of 64-bit limbs used by the int columns of this slice.
///
/// `Int<5>` = 320 bits, signed. Trace cells hold F_p elements (`< p < 2^256`)
/// so 4 unsigned limbs would suffice for the *value*; we use 5 so that
/// (a) the unsigned arithmetic helpers' `widening_mul → Uint<10>` pair
/// works (`ConcatMixed` is impl'd for the (5, 10) widths), and (b) the
/// top bit can serve as a sign for any signed intermediates.
pub const EC_FP_INT_LIMBS: usize = 5;

/// secp256k1 base-field prime `p = 2^256 − 2^32 − 977`, as a 5-limb
/// `Uint` (top limb zero-padded).
const SECP256K1_P_HEX: &str = concat!(
    "0000000000000000",
    "FFFFFFFFFFFFFFFF",
    "FFFFFFFFFFFFFFFF",
    "FFFFFFFFFFFFFFFF",
    "FFFFFFFEFFFFFC2F",
);

/// `p` as a `crypto_bigint::Uint<5>`. Arithmetic helpers operate at this
/// width.
pub const SECP256K1_P_UINT: CbUint<EC_FP_INT_LIMBS> = CbUint::from_be_hex(SECP256K1_P_HEX);

/// Trait knob: a `ConstSemiring` whose representation can hold a
/// secp256k1 base-field element. The `From<u32>` bound lets the UAIR
/// build small constant scalars (3, 8, 9, 12) for `mul_by_scalar`.
/// Implemented for `Int<EC_FP_INT_LIMBS>`.
pub trait EcdsaFpRing: ConstSemiring + From<u32> + 'static {}

impl EcdsaFpRing for Int<EC_FP_INT_LIMBS> {}

// ---------------------------------------------------------------------------
// Column layout.
// ---------------------------------------------------------------------------

pub mod cols {
    /// Public column: row activator (1 on every row that runs the doubling
    /// formula; 0 on padding).
    pub const S_ACTIVE: usize = 0;
    pub const NUM_INT_PUB: usize = 1;

    // Witness: input (X, Y, Z), scratch (S), output (X_mid, Y_mid, Z_mid).
    pub const W_X: usize = 1;
    pub const W_Y: usize = 2;
    pub const W_Z: usize = 3;
    pub const W_S: usize = 4;
    pub const W_X_MID: usize = 5;
    pub const W_Y_MID: usize = 6;
    pub const W_Z_MID: usize = 7;

    pub const NUM_INT: usize = 8;
}

/// The doubling UAIR. One independent Jacobian doubling per row.
#[derive(Clone, Debug)]
pub struct JacobianDoublingUair<R>(PhantomData<R>);

impl<R> Uair for JacobianDoublingUair<R>
where
    R: EcdsaFpRing,
{
    type Ideal = ImpossibleIdeal;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(0, 0, cols::NUM_INT);
        let public = PublicColumnLayout::new(0, 0, cols::NUM_INT_PUB);
        // No shifts — every row is independent.
        let shifts: Vec<ShiftSpec> = vec![];
        UairSignature::new(total, public, shifts, vec![])
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        _down: TraceRow<B::Expr>,
        _from_ref: FromR,
        mbs: MulByScalar,
        _ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&Self::Scalar) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &Self::Scalar) -> Option<B::Expr>,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
    {
        let int = up.int;
        let s_active = &int[cols::S_ACTIVE];
        let x = &int[cols::W_X];
        let y = &int[cols::W_Y];
        let z = &int[cols::W_Z];
        let s_w = &int[cols::W_S];
        let x_mid = &int[cols::W_X_MID];
        let y_mid = &int[cols::W_Y_MID];
        let z_mid = &int[cols::W_Z_MID];

        // Small-integer scalar constants. `mul_by_scalar` against a
        // degree-0 scalar polynomial costs one polynomial scaling vs.
        // the (k − 1) polynomial additions a `k·t` repeated-add chain
        // would need.
        let two_scalar = const_scalar::<R>(R::from(2_u32));
        let three_scalar = const_scalar::<R>(R::from(3_u32));
        let eight_scalar = const_scalar::<R>(R::from(8_u32));
        let nine_scalar = const_scalar::<R>(R::from(9_u32));
        let twelve_scalar = const_scalar::<R>(R::from(12_u32));

        // -----------------------------------------------------------
        // (1)  S − Y² = 0  (mod p),  gated by s_active.
        // -----------------------------------------------------------
        let y_sq = y.clone() * y;
        let c1_inner = s_w.clone() - &y_sq;
        b.assert_zero(s_active.clone() * &c1_inner);

        // -----------------------------------------------------------
        // (2)  Z_mid − 2·Y·Z = 0  (mod p)
        // -----------------------------------------------------------
        let yz = y.clone() * z;
        let two_yz = mbs(&yz, &two_scalar).expect("2·Y·Z overflow");
        let c2_inner = z_mid.clone() - &two_yz;
        b.assert_zero(s_active.clone() * &c2_inner);

        // -----------------------------------------------------------
        // (3)  X_mid − (9·X⁴ − 8·X·S) = 0  (mod p)
        //      Expanded as `X_mid − 9·X⁴ + 8·X·S = 0`.
        // -----------------------------------------------------------
        let x_sq = x.clone() * x;
        let x_pow4 = x_sq.clone() * &x_sq;
        let nine_x4 = mbs(&x_pow4, &nine_scalar).expect("9·X⁴ overflow");
        let xs = x.clone() * s_w;
        let eight_xs = mbs(&xs, &eight_scalar).expect("8·X·S overflow");
        let c3_inner = x_mid.clone() - &nine_x4 + &eight_xs;
        b.assert_zero(s_active.clone() * &c3_inner);

        // -----------------------------------------------------------
        // (4)  Y_mid − (3·X²·(4·X·S − X_mid) − 8·S²) = 0  (mod p)
        //      Expanded as
        //        Y_mid − 12·X³·S + 3·X²·X_mid + 8·S² = 0.
        // -----------------------------------------------------------
        let x_sq_x_s = x_sq.clone() * &xs; // X²·X·S = X³·S
        let twelve_x3s = mbs(&x_sq_x_s, &twelve_scalar).expect("12·X³·S overflow");
        let x_sq_xmid = x_sq.clone() * x_mid;
        let three_xsq_xmid =
            mbs(&x_sq_xmid, &three_scalar).expect("3·X²·X_mid overflow");
        let s_sq = s_w.clone() * s_w;
        let eight_s_sq = mbs(&s_sq, &eight_scalar).expect("8·S² overflow");
        let c4_inner = y_mid.clone() - &twelve_x3s + &three_xsq_xmid + &eight_s_sq;
        b.assert_zero(s_active.clone() * &c4_inner);
    }
}

// ---------------------------------------------------------------------------
// Witness generator.
// ---------------------------------------------------------------------------

impl<R> GenerateRandomTrace<32> for JacobianDoublingUair<R>
where
    R: EcdsaFpRing + From<Int<EC_FP_INT_LIMBS>>,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_random_trace<Rng: RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, R, R, 32> {
        let n_rows = 1usize << num_vars;

        let zero_r = R::ZERO;
        let one_r = R::ONE;
        let mk_col = || vec![zero_r.clone(); n_rows];

        let mut s_active_col: Vec<R> = mk_col();
        let mut x_col: Vec<R> = mk_col();
        let mut y_col: Vec<R> = mk_col();
        let mut z_col: Vec<R> = mk_col();
        let mut s_col: Vec<R> = mk_col();
        let mut x_mid_col: Vec<R> = mk_col();
        let mut y_mid_col: Vec<R> = mk_col();
        let mut z_mid_col: Vec<R> = mk_col();

        for v in s_active_col.iter_mut() {
            *v = one_r.clone();
        }

        for row in 0..n_rows {
            let x_in = rand_fp(rng);
            let y_in = rand_fp(rng);
            let z_in = rand_fp(rng);

            let DoublingWitness {
                s,
                x_mid,
                y_mid,
                z_mid,
            } = compute_doubling_witness(&x_in, &y_in, &z_in);

            x_col[row] = R::from(uint_to_int(x_in));
            y_col[row] = R::from(uint_to_int(y_in));
            z_col[row] = R::from(uint_to_int(z_in));
            s_col[row] = R::from(s);
            x_mid_col[row] = R::from(x_mid);
            y_mid_col[row] = R::from(y_mid);
            z_mid_col[row] = R::from(z_mid);
        }

        let to_mle = |col: Vec<R>| -> DenseMultilinearExtension<R> { col.into_iter().collect() };

        let int = vec![
            to_mle(s_active_col),
            to_mle(x_col),
            to_mle(y_col),
            to_mle(z_col),
            to_mle(s_col),
            to_mle(x_mid_col),
            to_mle(y_mid_col),
            to_mle(z_mid_col),
        ];

        UairTrace {
            int: int.into(),
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Scalar helpers.
// ---------------------------------------------------------------------------

/// Build a constant-polynomial (degree 0) `c` as a `DensePolynomial<R, 32>`.
fn const_scalar<R: ConstSemiring>(c: R) -> DensePolynomial<R, 32> {
    let mut coeffs = [R::ZERO; 32];
    coeffs[0] = c;
    DensePolynomial::<R, 32>::new(coeffs)
}

// ---------------------------------------------------------------------------
// F_p arithmetic helpers (secp256k1 base field).
// ---------------------------------------------------------------------------

/// Sample a uniformly-random base-field element `[0, p)`.
fn rand_fp<Rng: RngCore + ?Sized>(rng: &mut Rng) -> CbUint<EC_FP_INT_LIMBS> {
    let p_nz = NonZero::new(SECP256K1_P_UINT).expect("p is nonzero");
    let mut limbs = [0u64; EC_FP_INT_LIMBS];
    for limb in &mut limbs {
        *limb = rng.next_u64();
    }
    limbs[EC_FP_INT_LIMBS - 1] = 0;
    let raw = CbUint::<EC_FP_INT_LIMBS>::from_words(limbs);
    raw.rem_vartime(&p_nz)
}

/// `(a · b) mod p`.
fn mul_mod_p(
    a: &CbUint<EC_FP_INT_LIMBS>,
    b: &CbUint<EC_FP_INT_LIMBS>,
) -> CbUint<EC_FP_INT_LIMBS> {
    let wide: CbUint<{ EC_FP_INT_LIMBS * 2 }> = a.widening_mul(b).into();
    let p_wide: CbUint<{ EC_FP_INT_LIMBS * 2 }> = SECP256K1_P_UINT.resize();
    let p_wide_nz = NonZero::new(p_wide).expect("p is nonzero");
    let (_, rem) = wide.div_rem_vartime(&p_wide_nz);
    rem.resize()
}

/// `(a · k) mod p` for small integer `k` via repeated addition.
fn small_mul_mod_p(a: &CbUint<EC_FP_INT_LIMBS>, k: u32) -> CbUint<EC_FP_INT_LIMBS> {
    let p_nz = NonZero::new(SECP256K1_P_UINT).expect("p is nonzero");
    let mut acc = CbUint::<EC_FP_INT_LIMBS>::ZERO;
    for _ in 0..k {
        acc = acc.wrapping_add(a);
        if p_geq(&acc) {
            acc = acc.rem_vartime(&p_nz);
        }
    }
    acc
}

#[inline]
fn p_geq(a: &CbUint<EC_FP_INT_LIMBS>) -> bool {
    a.checked_sub(&SECP256K1_P_UINT).is_some().into()
}

/// `(a − b) mod p`, allowing `a < b`.
fn sub_mod_p(
    a: &CbUint<EC_FP_INT_LIMBS>,
    b: &CbUint<EC_FP_INT_LIMBS>,
) -> CbUint<EC_FP_INT_LIMBS> {
    let p_nz = NonZero::new(SECP256K1_P_UINT).expect("p is nonzero");
    if a.checked_sub(b).is_some().into() {
        a.wrapping_sub(b).rem_vartime(&p_nz)
    } else {
        let a_plus_p = a.wrapping_add(&SECP256K1_P_UINT);
        a_plus_p.wrapping_sub(b).rem_vartime(&p_nz)
    }
}

struct DoublingWitness {
    s: Int<EC_FP_INT_LIMBS>,
    x_mid: Int<EC_FP_INT_LIMBS>,
    y_mid: Int<EC_FP_INT_LIMBS>,
    z_mid: Int<EC_FP_INT_LIMBS>,
}

/// Compute the mod-p doubling outputs for a single row from inputs
/// `(X, Y, Z)`. All arithmetic is in `Uint<5>` mod p; outputs are then
/// widened to `Int<22>` for trace storage.
fn compute_doubling_witness(
    x: &CbUint<EC_FP_INT_LIMBS>,
    y: &CbUint<EC_FP_INT_LIMBS>,
    z: &CbUint<EC_FP_INT_LIMBS>,
) -> DoublingWitness {
    let s_u = mul_mod_p(y, y); // S = Y² mod p
    let x_sq_u = mul_mod_p(x, x);
    let x_quad_u = mul_mod_p(&x_sq_u, &x_sq_u);
    let xs_u = mul_mod_p(x, &s_u);
    let nine_xq_u = small_mul_mod_p(&x_quad_u, 9);
    let eight_xs_u = small_mul_mod_p(&xs_u, 8);
    let x_mid_u = sub_mod_p(&nine_xq_u, &eight_xs_u);

    let yz_u = mul_mod_p(y, z);
    let z_mid_u = small_mul_mod_p(&yz_u, 2);

    let four_xs_u = small_mul_mod_p(&xs_u, 4);
    let four_xs_minus_xmid_u = sub_mod_p(&four_xs_u, &x_mid_u);
    let three_x_sq_u = small_mul_mod_p(&x_sq_u, 3);
    let big_term_u = mul_mod_p(&three_x_sq_u, &four_xs_minus_xmid_u);
    let s_sq_u = mul_mod_p(&s_u, &s_u);
    let eight_s_sq_u = small_mul_mod_p(&s_sq_u, 8);
    let y_mid_u = sub_mod_p(&big_term_u, &eight_s_sq_u);

    DoublingWitness {
        s: uint_to_int(s_u),
        x_mid: uint_to_int(x_mid_u),
        y_mid: uint_to_int(y_mid_u),
        z_mid: uint_to_int(z_mid_u),
    }
}

// ---------------------------------------------------------------------------
// Uint ↔ Int bridge for trace construction.
// ---------------------------------------------------------------------------

/// Reinterpret a non-negative `Uint<5>` as `Int<5>`. Honest values are
/// always in `[0, p)` with `p < 2^256`, so the top bit is zero and the
/// reinterpretation is the identity as an integer.
fn uint_to_int(u: CbUint<EC_FP_INT_LIMBS>) -> Int<EC_FP_INT_LIMBS> {
    debug_assert!(
        u.bits() <= 64 * EC_FP_INT_LIMBS as u32 - 1,
        "uint top bit must be 0 to reinterpret as signed"
    );
    Int::new(*u.as_int())
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_bigint::ConstOne;
    use rand::rng;
    use zinc_uair::{
        constraint_counter::count_constraints,
        degree_counter::{count_constraint_degrees, count_max_degree},
    };

    /// Sanity: the UAIR has the expected number of constraints (4) at the
    /// expected degrees.
    #[test]
    fn doubling_constraint_shape() {
        type U = JacobianDoublingUair<Int<EC_FP_INT_LIMBS>>;
        assert_eq!(count_constraints::<U>(), 4);
        let degrees = count_constraint_degrees::<U>();
        // Each constraint is `s_active * inner`; degree = 1 + inner degree.
        //   C1: s_active * (S − Y²)        → 1 + 2 = 3
        //   C2: s_active * (Z_mid − 2YZ)   → 1 + 2 = 3
        //   C3: s_active * (X_mid − 9X⁴…)  → 1 + 4 = 5
        //   C4: s_active * (Y_mid − 12X³S…) → 1 + 4 = 5
        assert_eq!(degrees, vec![3, 3, 5, 5]);
        assert_eq!(count_max_degree::<U>(), 5);
    }

    /// The witness generator produces a trace where every constraint
    /// vanishes mod p per row. Each step is computed in `Uint<5>` mod-p
    /// arithmetic, matching what the prover's F_p projection produces.
    #[test]
    fn witness_satisfies_constraints_mod_p() {
        let num_vars = 2;
        let mut r = rng();
        let trace = <JacobianDoublingUair<Int<EC_FP_INT_LIMBS>> as GenerateRandomTrace<32>>::
            generate_random_trace(num_vars, &mut r);
        let n_rows = 1 << num_vars;
        assert_eq!(trace.int.len(), cols::NUM_INT);

        // Honest trace cells are non-negative (F_p elements in [0, p)),
        // so the Int → Uint reinterpret is the identity as an integer.
        let int_to_uint = |v: &Int<EC_FP_INT_LIMBS>| -> CbUint<EC_FP_INT_LIMBS> {
            *v.inner().as_uint()
        };

        for row in 0..n_rows {
            let s_active = trace.int[cols::S_ACTIVE][row].clone();
            assert_eq!(s_active, Int::ONE);

            let read = |c: usize| int_to_uint(&trace.int[c][row]);
            let x = read(cols::W_X);
            let y = read(cols::W_Y);
            let z = read(cols::W_Z);
            let s = read(cols::W_S);
            let x_mid = read(cols::W_X_MID);
            let y_mid = read(cols::W_Y_MID);
            let z_mid = read(cols::W_Z_MID);

            // C1: S = Y² (mod p)
            assert_eq!(s, mul_mod_p(&y, &y), "C1 (S = Y²) at row {row}");

            // C2: Z_mid = 2·Y·Z (mod p)
            let yz = mul_mod_p(&y, &z);
            assert_eq!(
                z_mid,
                small_mul_mod_p(&yz, 2),
                "C2 (Z_mid = 2YZ) at row {row}",
            );

            // C3: X_mid = 9·X⁴ − 8·X·S (mod p)
            let x_sq = mul_mod_p(&x, &x);
            let x_quad = mul_mod_p(&x_sq, &x_sq);
            let nine_x4 = small_mul_mod_p(&x_quad, 9);
            let xs = mul_mod_p(&x, &s);
            let eight_xs = small_mul_mod_p(&xs, 8);
            assert_eq!(
                x_mid,
                sub_mod_p(&nine_x4, &eight_xs),
                "C3 (X_mid) at row {row}",
            );

            // C4: Y_mid = 3·X²·(4·X·S − X_mid) − 8·S² (mod p)
            let four_xs = small_mul_mod_p(&xs, 4);
            let four_xs_minus_xmid = sub_mod_p(&four_xs, &x_mid);
            let three_xsq = small_mul_mod_p(&x_sq, 3);
            let big_term = mul_mod_p(&three_xsq, &four_xs_minus_xmid);
            let s_sq = mul_mod_p(&s, &s);
            let eight_s_sq = small_mul_mod_p(&s_sq, 8);
            assert_eq!(
                y_mid,
                sub_mod_p(&big_term, &eight_s_sq),
                "C4 (Y_mid) at row {row}",
            );
        }
    }
}
