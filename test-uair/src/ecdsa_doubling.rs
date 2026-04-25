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
//! the next increments. This slice exists to land the F_p quotient-witness
//! machinery and validate the column / constraint layout that the rest of
//! the EC ops (addition, Shamir selectors, affine conversion) will reuse.
//!
//! ## F_p → Q[X] quotient-witness lift
//!
//! The spec places the constraints in `F_p[X]`. The zinc-plus protocol has
//! a single `Int` / `Fmod` parameterization, so we re-express each `F_p`
//! identity as an integer identity with an explicit **quotient witness**:
//!
//! ```text
//!     S − Y²                  ≡ 0  (mod p)        [spec, F_p]
//!     S − Y² + q_S · p         = 0   (in Z)        [Q[X]]
//! ```
//!
//! and similarly for the other three constraints. The prover supplies the
//! quotient (`q_S`, `q_Z`, `q_X`, `q_Y`) per row; the verifier runs the
//! standard `assert_zero` Q[X] check. Quotient ranges are NOT enforced
//! (lookup PIOP stubbed — same stance as the existing scalar slice).
//!
//! ## Soundness caveats
//!
//! - Quotient ranges aren't bounded. A malicious prover could supply
//!   absurdly-large `q` values; for honest provers the quotients are
//!   bounded by `9·X⁴/p ≈ 2^771` (`q_X`) and `3·X²·(4XS)/p ≈ 2^1028`
//!   (`q_Y`). Closing the gap requires range-lookup integration.
//! - The output `(X_mid, Y_mid, Z_mid)` is not range-checked into
//!   `[0, p)` — a malicious prover could store any equivalent
//!   representative, then later `q_S` etc. absorb the difference. Same
//!   "free representative" stance as `x̂` in the scalar slice.
//! - Curve-equation membership (`Y² = X³ + 7·Z⁶` in Jacobian form) is not
//!   enforced. The doubling formulas are valid polynomial identities in
//!   F_p regardless of curve membership; binding inputs to the curve is a
//!   separate constraint group, deferred until the full Shamir loop.
//!
//! ## Required Int width
//!
//! The maximum constraint-LHS magnitude is in the `Y_mid` constraint:
//!
//! - `3·X²·(4·X·S)` is roughly `2^1027` for `X, Y < 2^256`.
//! - The lifted form adds `q_Y · p` of similar magnitude.
//!
//! So we need `Int<N>` with `64·N − 1 ≥ 1028`, i.e. `N ≥ 17`. We pick
//! `N = 22` (1408 bits) for safety margin; the corresponding random field
//! `MontyField<N'>` for the proof needs `N' ≥ 17` (a 1088+ bit prime). End-
//! to-end protocol wiring at this width is **not done in this slice** —
//! see the trailing TODO.
//!
//! ## Out of scope
//!
//! - Full Shamir loop (this slice has no row-to-row chaining or shifts).
//! - Jacobian addition, Shamir addend selection, affine conversion via
//!   Z⁻¹, signature modular check.
//! - Composition with the existing `EcdsaScalarSliceUair` (scalar
//!   accumulation, inverse) and `Sha256CompressionSliceUair`.
//! - Public-key validity (Q on curve, non-identity).
//! - End-to-end protocol test. The current test in this module only
//!   verifies that an honest witness satisfies every constraint as an
//!   integer equation. Wiring a real `do_test` requires a wider
//!   `ZincTypes` than the existing `EcdsaZincTypes` (which is `Int<5>` /
//!   `MontyField<10>`); both Int and MontyField widths roughly need to
//!   triple to handle this slice's constraint magnitudes.

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
/// `Int<22>` = 1408 bits, large enough to hold the `Y_mid` constraint
/// LHS (`≤ 2^1027`) and its quotient witness (`≤ 2^772`) with a safety
/// margin.
pub const EC_FP_INT_LIMBS: usize = 22;

/// Width of the base-field arithmetic helpers. We use `Uint<5>` for
/// modular multiplication / reduction because `crypto_bigint`'s
/// `widening_mul` requires `ConcatMixed`, which is implemented for small
/// limb counts (5, 10) but not arbitrary widths like 22. Values from the
/// arithmetic side are widened to `Int<EC_FP_INT_LIMBS>` when stored in
/// the trace.
pub const FP_ARITH_LIMBS: usize = 5;

/// secp256k1 base-field prime `p = 2^256 − 2^32 − 977`, as a 5-limb
/// `Uint` (top limb zero-padded to align with `Int<5>` widths used
/// elsewhere). Hex (limb_4 … limb_0, big-endian):
/// `0x0000000000000000 FFFFFFFFFFFFFFFF FFFFFFFFFFFFFFFF FFFFFFFFFFFFFFFF FFFFFFFEFFFFFC2F`
const SECP256K1_P_HEX_5: &str = concat!(
    "0000000000000000", // limb 4 (zero pad)
    "FFFFFFFFFFFFFFFF", // limb 3
    "FFFFFFFFFFFFFFFF", // limb 2
    "FFFFFFFFFFFFFFFF", // limb 1
    "FFFFFFFEFFFFFC2F", // limb 0  ←  -2^32 - 977 (mod 2^64)
);

/// `p` as a `crypto_bigint::Uint<5>`. Arithmetic helpers operate at this
/// width.
pub const SECP256K1_P_UINT: CbUint<FP_ARITH_LIMBS> = CbUint::from_be_hex(SECP256K1_P_HEX_5);

/// secp256k1 base-field prime `p` as a `crypto_primitives::Int<22>` —
/// the width used in the trace columns. Constructed by zero-padding the
/// 5-limb representation up to 22 limbs.
pub const SECP256K1_P: Int<EC_FP_INT_LIMBS> = {
    // Build the 22-limb hex string at compile time by zero-padding 17
    // limbs (= 272 hex chars) on top of the 5-limb hex.
    const HEX_22: &str = concat!(
        // 17 zero limbs (272 hex chars):
        "00000000000000000000000000000000",
        "00000000000000000000000000000000",
        "00000000000000000000000000000000",
        "00000000000000000000000000000000",
        "00000000000000000000000000000000",
        "00000000000000000000000000000000",
        "00000000000000000000000000000000",
        "00000000000000000000000000000000",
        "0000000000000000",
        // 5 active limbs (80 hex chars):
        "0000000000000000", // limb 4 (zero, top of the 5-limb representation)
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFEFFFFFC2F",
    );
    Int::new(crypto_bigint::Int::from_be_hex(HEX_22))
};

/// Trait knob: a `ConstSemiring` exposing secp256k1's base-field prime.
/// Implemented only for `Int<EC_FP_INT_LIMBS>`.
pub trait EcdsaFpRing: ConstSemiring + 'static {
    fn secp256k1_p() -> Self;
}

impl EcdsaFpRing for Int<EC_FP_INT_LIMBS> {
    fn secp256k1_p() -> Self {
        SECP256K1_P
    }
}

// ---------------------------------------------------------------------------
// Column layout.
// ---------------------------------------------------------------------------

pub mod cols {
    /// Public column: row activator (1 on every row that runs the doubling
    /// formula; 0 on padding). This slice's `signature()` declares no
    /// shifts and no row-to-row dependencies, so the activator is just a
    /// per-row "this row is meaningful" flag. With shifts added later
    /// (Shamir loop) it'll become a proper boundary selector.
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

    // Quotient witnesses, one per F_p constraint:
    pub const W_Q_S: usize = 8; //  S − Y²            + q_S·p = 0
    pub const W_Q_Z_MID: usize = 9; //  Z_mid − 2YZ       + q_Z·p = 0
    pub const W_Q_X_MID: usize = 10; //  X_mid − (9X⁴−8XS) + q_X·p = 0
    pub const W_Q_Y_MID: usize = 11; //  Y_mid − (...)     + q_Y·p = 0

    pub const NUM_INT: usize = 12;
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
        let q_s = &int[cols::W_Q_S];
        let q_z = &int[cols::W_Q_Z_MID];
        let q_x = &int[cols::W_Q_X_MID];
        let q_y = &int[cols::W_Q_Y_MID];

        // Constant scalar `p` (the secp256k1 base-field prime), as a
        // degree-0 polynomial. Used via `mbs(q, &p_scalar)` to get `q · p`.
        let p_scalar = const_scalar::<R>(R::secp256k1_p());

        // -----------------------------------------------------------
        // (1)  S − Y² + q_S · p = 0,  gated by s_active.
        // -----------------------------------------------------------
        let y_sq = y.clone() * y; // Y²
        let q_s_p = mbs(q_s, &p_scalar).expect("q_S · p overflow");
        let c1_inner = s_w.clone() - &y_sq + &q_s_p;
        b.assert_zero(s_active.clone() * &c1_inner);

        // -----------------------------------------------------------
        // (2)  Z_mid − 2·Y·Z + q_Z · p = 0
        //      Written as `Z_mid − Y·Z − Y·Z + q_Z·p` so we don't need
        //      a "2·" scalar; matches the style used in the scalar slice.
        // -----------------------------------------------------------
        let yz = y.clone() * z;
        let q_z_p = mbs(q_z, &p_scalar).expect("q_Z · p overflow");
        let c2_inner = z_mid.clone() - &yz - &yz + &q_z_p;
        b.assert_zero(s_active.clone() * &c2_inner);

        // -----------------------------------------------------------
        // (3)  X_mid − (9·X⁴ − 8·X·S) + q_X · p = 0
        //      Expanded as
        //        X_mid − 9·(X²)² + 8·X·S + q_X·p = 0.
        //      Powers of small integer constants are implemented via
        //      repeated addition (e.g. `9·t = t+t+t+t+t+t+t+t+t`) to
        //      avoid relying on `From<i32>` on R.
        // -----------------------------------------------------------
        let x_sq = x.clone() * x; // X²
        let x_pow4 = x_sq.clone() * &x_sq; // X⁴
        // 9·X⁴
        let mut nine_x4 = x_pow4.clone();
        for _ in 0..8 {
            nine_x4 = nine_x4 + &x_pow4;
        }
        // 8·X·S
        let xs = x.clone() * s_w;
        let mut eight_xs = xs.clone();
        for _ in 0..7 {
            eight_xs = eight_xs + &xs;
        }
        let q_x_p = mbs(q_x, &p_scalar).expect("q_X · p overflow");
        let c3_inner = x_mid.clone() - &nine_x4 + &eight_xs + &q_x_p;
        b.assert_zero(s_active.clone() * &c3_inner);

        // -----------------------------------------------------------
        // (4)  Y_mid − (3·X²·(4·X·S − X_mid) − 8·S²) + q_Y · p = 0
        //      Expanded as
        //        Y_mid − 3·X²·4·X·S + 3·X²·X_mid + 8·S² + q_Y·p = 0
        //      i.e.
        //        Y_mid − 12·(X²·X·S) + 3·(X²·X_mid) + 8·S² + q_Y·p = 0
        // -----------------------------------------------------------
        let x_sq_x_s = x_sq.clone() * &xs; // X²·X·S = X³·S
        // 12·X²·X·S
        let mut twelve_term = x_sq_x_s.clone();
        for _ in 0..11 {
            twelve_term = twelve_term + &x_sq_x_s;
        }
        let x_sq_xmid = x_sq.clone() * x_mid; // X²·X_mid
        // 3·X²·X_mid
        let mut three_xsq_xmid = x_sq_xmid.clone();
        for _ in 0..2 {
            three_xsq_xmid = three_xsq_xmid + &x_sq_xmid;
        }
        let s_sq = s_w.clone() * s_w; // S²
        // 8·S²
        let mut eight_s_sq = s_sq.clone();
        for _ in 0..7 {
            eight_s_sq = eight_s_sq + &s_sq;
        }
        let q_y_p = mbs(q_y, &p_scalar).expect("q_Y · p overflow");
        let c4_inner = y_mid.clone() - &twelve_term + &three_xsq_xmid + &eight_s_sq + &q_y_p;
        b.assert_zero(s_active.clone() * &c4_inner);
    }
}

/// Build a constant-polynomial (degree 0) `c` as a `DensePolynomial<R, 32>`.
/// Same helper as the other ECDSA / SHA slices; declared here to keep the
/// module self-contained.
fn const_scalar<R: ConstSemiring>(c: R) -> DensePolynomial<R, 32> {
    let mut coeffs = [R::ZERO; 32];
    coeffs[0] = c;
    DensePolynomial::<R, 32>::new(coeffs)
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
        let mut q_s_col: Vec<R> = mk_col();
        let mut q_z_col: Vec<R> = mk_col();
        let mut q_x_col: Vec<R> = mk_col();
        let mut q_y_col: Vec<R> = mk_col();

        // Mark every row active.
        for v in s_active_col.iter_mut() {
            *v = one_r.clone();
        }

        for row in 0..n_rows {
            // Sample `(X, Y, Z) ∈ F_p^3` uniformly.
            let x_in = rand_fp(rng);
            let y_in = rand_fp(rng);
            let z_in = rand_fp(rng);

            let DoublingWitness {
                s,
                x_mid,
                y_mid,
                z_mid,
                q_s,
                q_z,
                q_x,
                q_y,
            } = compute_doubling_witness(&x_in, &y_in, &z_in);

            x_col[row] = R::from(uint_to_int_22(x_in));
            y_col[row] = R::from(uint_to_int_22(y_in));
            z_col[row] = R::from(uint_to_int_22(z_in));
            s_col[row] = R::from(s);
            x_mid_col[row] = R::from(x_mid);
            y_mid_col[row] = R::from(y_mid);
            z_mid_col[row] = R::from(z_mid);
            q_s_col[row] = R::from(q_s);
            q_z_col[row] = R::from(q_z);
            q_x_col[row] = R::from(q_x);
            q_y_col[row] = R::from(q_y);
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
            to_mle(q_s_col),
            to_mle(q_z_col),
            to_mle(q_x_col),
            to_mle(q_y_col),
        ];

        UairTrace {
            int: int.into(),
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// F_p arithmetic helpers (secp256k1 base field).
// ---------------------------------------------------------------------------

/// Sample a uniformly-random base-field element `[0, p)`.
fn rand_fp<Rng: RngCore + ?Sized>(rng: &mut Rng) -> CbUint<FP_ARITH_LIMBS> {
    let p_nz = NonZero::new(SECP256K1_P_UINT).expect("p is nonzero");
    // Sample 320 uniform bits, then reduce mod p (top limb zeroed so the
    // raw value is < 2^256, giving bias-free reduction since `p ≈ 2^256`).
    let mut limbs = [0u64; FP_ARITH_LIMBS];
    for limb in &mut limbs {
        *limb = rng.next_u64();
    }
    limbs[FP_ARITH_LIMBS - 1] = 0;
    let raw = CbUint::<FP_ARITH_LIMBS>::from_words(limbs);
    raw.rem_vartime(&p_nz)
}

/// `(a · b) mod p`.
fn mul_mod_p(
    a: &CbUint<FP_ARITH_LIMBS>,
    b: &CbUint<FP_ARITH_LIMBS>,
) -> CbUint<FP_ARITH_LIMBS> {
    // Widen to 2× for the raw product, then reduce mod p.
    let wide: CbUint<{ FP_ARITH_LIMBS * 2 }> = a.widening_mul(b).into();
    let p_wide: CbUint<{ FP_ARITH_LIMBS * 2 }> = SECP256K1_P_UINT.resize();
    let p_wide_nz = NonZero::new(p_wide).expect("p is nonzero");
    let (_, rem) = wide.div_rem_vartime(&p_wide_nz);
    rem.resize()
}


/// All quantities the doubling witness gen produces per row, expressed
/// as `Int<22>` (the trace cell type). Quotient witnesses can be
/// negative; output values (`s`, `x_mid`, `y_mid`, `z_mid`) are
/// non-negative and `< p`.
struct DoublingWitness {
    s: Int<EC_FP_INT_LIMBS>,
    x_mid: Int<EC_FP_INT_LIMBS>,
    y_mid: Int<EC_FP_INT_LIMBS>,
    z_mid: Int<EC_FP_INT_LIMBS>,
    q_s: Int<EC_FP_INT_LIMBS>,
    q_z: Int<EC_FP_INT_LIMBS>,
    q_x: Int<EC_FP_INT_LIMBS>,
    q_y: Int<EC_FP_INT_LIMBS>,
}

/// Exact division in `Int<22>`. Panics if the division leaves a non-zero
/// remainder (we use this only on values that are known to be exact
/// multiples of `p`).
fn int22_div_exact(num: &Int<EC_FP_INT_LIMBS>, denom: &Int<EC_FP_INT_LIMBS>) -> Int<EC_FP_INT_LIMBS> {
    use crypto_bigint::NonZero as CbNonZero;
    let n_inner: crypto_bigint::Int<EC_FP_INT_LIMBS> = *num.inner();
    let d_inner: crypto_bigint::Int<EC_FP_INT_LIMBS> = *denom.inner();
    let d_nz = CbNonZero::new(d_inner).expect("zero divisor");
    // `checked_div_rem_vartime` returns `(ConstCtOption<quotient>, remainder)`.
    let (q_opt, r_inner) = n_inner.checked_div_rem_vartime(&d_nz);
    let q_inner: crypto_bigint::Int<EC_FP_INT_LIMBS> = Option::from(q_opt)
        .expect("checked_div_rem_vartime: division failed");
    debug_assert!(
        r_inner == crypto_bigint::Int::<EC_FP_INT_LIMBS>::ZERO,
        "int22_div_exact: division left a nonzero remainder"
    );
    Int::new(q_inner)
}

/// Compute the doubling witness for a single row from inputs `(X, Y, Z)`.
///
/// Strategy: compute the mod-p output values (`s`, `x_mid`, `y_mid`,
/// `z_mid`) using the `Uint<5>` helpers, then compute each quotient
/// witness in **`Int<22>` arithmetic over the full Z magnitudes**. The
/// constraint is `output − formula(X, Y, Z) + q · p = 0` in Z, so the
/// quotient is `q = (formula(X, Y, Z) − output) / p`.
fn compute_doubling_witness(
    x: &CbUint<FP_ARITH_LIMBS>,
    y: &CbUint<FP_ARITH_LIMBS>,
    z: &CbUint<FP_ARITH_LIMBS>,
) -> DoublingWitness {
    // ---------- Mod-p output values via Uint<5> arithmetic. ----------
    let s_u = mul_mod_p(y, y); // S = Y² mod p
    let x_sq_u = mul_mod_p(x, x);
    let x_quad_u = mul_mod_p(&x_sq_u, &x_sq_u);
    let xs_u = mul_mod_p(x, &s_u);
    let nine_xq_u = small_mul_mod_p(&x_quad_u, 9);
    let eight_xs_u = small_mul_mod_p(&xs_u, 8);
    let x_mid_u = sub_mod_p(&nine_xq_u, &eight_xs_u); // X_mid = 9X⁴ − 8XS mod p

    let yz_u = mul_mod_p(y, z);
    let z_mid_u = small_mul_mod_p(&yz_u, 2); // Z_mid = 2YZ mod p

    let four_xs_u = small_mul_mod_p(&xs_u, 4);
    let four_xs_minus_xmid_u = sub_mod_p(&four_xs_u, &x_mid_u);
    let three_x_sq_u = small_mul_mod_p(&x_sq_u, 3);
    let big_term_u = mul_mod_p(&three_x_sq_u, &four_xs_minus_xmid_u);
    let s_sq_u = mul_mod_p(&s_u, &s_u);
    let eight_s_sq_u = small_mul_mod_p(&s_sq_u, 8);
    let y_mid_u = sub_mod_p(&big_term_u, &eight_s_sq_u); // Y_mid mod p

    // ---------- Lift everything to Int<22>. ----------
    let x_i = uint_to_int_22(*x);
    let y_i = uint_to_int_22(*y);
    let z_i = uint_to_int_22(*z);
    let s_i = uint_to_int_22(s_u);
    let x_mid_i = uint_to_int_22(x_mid_u);
    let y_mid_i = uint_to_int_22(y_mid_u);
    let z_mid_i = uint_to_int_22(z_mid_u);
    let p_i = SECP256K1_P;

    // ---------- Quotient witnesses via Int<22> Z arithmetic. ----------
    // q_S · p = Y² − S            (in Z, exact since Y² ≡ S mod p).
    let target_s = y_i.clone() * &y_i - &s_i;
    let q_s = int22_div_exact(&target_s, &p_i);

    // q_Z · p = 2YZ − Z_mid       (in Z).
    let yz_i = y_i.clone() * &z_i;
    let target_z = yz_i.clone() + &yz_i - &z_mid_i;
    let q_z = int22_div_exact(&target_z, &p_i);

    // q_X · p = 9X⁴ − 8XS − X_mid (in Z).
    let x_sq_i = x_i.clone() * &x_i;
    let x_quad_i = x_sq_i.clone() * &x_sq_i;
    let mut nine_x_quad_i = x_quad_i.clone();
    for _ in 0..8 {
        nine_x_quad_i = nine_x_quad_i + &x_quad_i;
    }
    let xs_i = x_i.clone() * &s_i;
    let mut eight_xs_i = xs_i.clone();
    for _ in 0..7 {
        eight_xs_i = eight_xs_i + &xs_i;
    }
    let target_x = nine_x_quad_i - &eight_xs_i - &x_mid_i;
    let q_x = int22_div_exact(&target_x, &p_i);

    // q_Y · p = 12·X³·S − 3·X²·X_mid − 8·S² − Y_mid    (in Z).
    let x_cube_s_i = x_sq_i.clone() * &xs_i; // X²·(X·S) = X³·S
    let mut twelve_x3s_i = x_cube_s_i.clone();
    for _ in 0..11 {
        twelve_x3s_i = twelve_x3s_i + &x_cube_s_i;
    }
    let xsq_xmid_i = x_sq_i.clone() * &x_mid_i;
    let mut three_xsq_xmid_i = xsq_xmid_i.clone();
    for _ in 0..2 {
        three_xsq_xmid_i = three_xsq_xmid_i + &xsq_xmid_i;
    }
    let s_sq_i = s_i.clone() * &s_i;
    let mut eight_s_sq_i = s_sq_i.clone();
    for _ in 0..7 {
        eight_s_sq_i = eight_s_sq_i + &s_sq_i;
    }
    let target_y = twelve_x3s_i - &three_xsq_xmid_i - &eight_s_sq_i - &y_mid_i;
    let q_y = int22_div_exact(&target_y, &p_i);

    DoublingWitness {
        s: s_i,
        x_mid: x_mid_i,
        y_mid: y_mid_i,
        z_mid: z_mid_i,
        q_s,
        q_z,
        q_x,
        q_y,
    }
}

/// `(a · k) mod p` for small integer `k`. Implemented by repeated addition
/// to avoid widening past 2× the input width.
fn small_mul_mod_p(a: &CbUint<FP_ARITH_LIMBS>, k: u32) -> CbUint<FP_ARITH_LIMBS> {
    let p_nz = NonZero::new(SECP256K1_P_UINT).expect("p is nonzero");
    let mut acc = CbUint::<FP_ARITH_LIMBS>::ZERO;
    for _ in 0..k {
        acc = acc.wrapping_add(a);
        // Reduce eagerly to keep the running sum < 2·p.
        if p_geq(&acc) {
            acc = acc.rem_vartime(&p_nz);
        }
    }
    acc
}

#[inline]
fn p_geq(a: &CbUint<FP_ARITH_LIMBS>) -> bool {
    // a ≥ p iff a − p does not underflow. Use checked_sub.
    a.checked_sub(&SECP256K1_P_UINT).is_some().into()
}

/// `(a − b) mod p`, allowing `a < b`.
fn sub_mod_p(
    a: &CbUint<FP_ARITH_LIMBS>,
    b: &CbUint<FP_ARITH_LIMBS>,
) -> CbUint<FP_ARITH_LIMBS> {
    let p_nz = NonZero::new(SECP256K1_P_UINT).expect("p is nonzero");
    if a.checked_sub(b).is_some().into() {
        a.wrapping_sub(b).rem_vartime(&p_nz)
    } else {
        // a < b; result = (a + p) − b mod p.
        let a_plus_p = a.wrapping_add(&SECP256K1_P_UINT);
        a_plus_p.wrapping_sub(b).rem_vartime(&p_nz)
    }
}

// ---------------------------------------------------------------------------
// Int<22> bridge for trace construction.
// ---------------------------------------------------------------------------

/// Widen a non-negative `Uint<5>` to `Int<22>` via zero-padding.
fn uint_to_int_22(u: CbUint<FP_ARITH_LIMBS>) -> Int<EC_FP_INT_LIMBS> {
    let widened: CbUint<EC_FP_INT_LIMBS> = u.resize();
    debug_assert!(
        widened.bits() <= 64 * EC_FP_INT_LIMBS as u32 - 1,
        "uint top bit must be 0 to reinterpret as signed"
    );
    Int::new(*widened.as_int())
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_bigint::{ConstOne, ConstZero};
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
        // Degrees: each constraint is `s_active * inner`, so add 1 to the
        // raw inner degree. Inner degrees:
        //   C1: s_active * (S − Y² + q·p)            → 1·(2)        = 3
        //   C2: s_active * (Z_mid − Y·Z − Y·Z + ...) → 1·(2)        = 3
        //   C3: s_active * (X_mid − 9X⁴ + 8XS + ...) → 1·(4)        = 5
        //   C4: s_active * (Y_mid − 12X³S + ...)     → 1·(4)        = 5
        // (max-degree counter takes max across monomials in the constraint
        //  expression's polynomial in trace MLEs.)
        assert_eq!(degrees, vec![3, 3, 5, 5]);
        assert_eq!(count_max_degree::<U>(), 5);
    }

    /// The witness generator produces a trace where every constraint
    /// evaluates to zero per row when interpreted as integer arithmetic.
    /// This catches bugs in the doubling formulas / quotient-witness
    /// computation without needing the full PIOP wiring.
    #[test]
    fn witness_satisfies_constraints_in_z() {
        // Tiny num_vars so the test is fast (4 rows).
        let num_vars = 2;
        let mut r = rng();
        let trace = <JacobianDoublingUair<Int<EC_FP_INT_LIMBS>> as GenerateRandomTrace<32>>::
            generate_random_trace(num_vars, &mut r);
        let n_rows = 1 << num_vars;
        assert_eq!(trace.int.len(), cols::NUM_INT);

        for row in 0..n_rows {
            // Read columns at this row.
            let read = |c: usize| trace.int[c][row].clone();
            let s_active = read(cols::S_ACTIVE);
            assert_eq!(s_active, Int::ONE);

            let x = read(cols::W_X);
            let y = read(cols::W_Y);
            let z = read(cols::W_Z);
            let s = read(cols::W_S);
            let x_mid = read(cols::W_X_MID);
            let y_mid = read(cols::W_Y_MID);
            let z_mid = read(cols::W_Z_MID);
            let q_s = read(cols::W_Q_S);
            let q_z = read(cols::W_Q_Z_MID);
            let q_x = read(cols::W_Q_X_MID);
            let q_y = read(cols::W_Q_Y_MID);

            let p = SECP256K1_P;

            // C1: S − Y² + q_S · p == 0
            // Computed using Int<22> arithmetic (which is mod 2^1408 — we
            // pick widths so honest values don't overflow).
            let y_sq = y.clone() * &y;
            let qs_p = q_s.clone() * &p;
            assert_eq!(
                s.clone() - &y_sq + &qs_p,
                Int::ZERO,
                "C1 (S = Y² mod p) failed at row {row}"
            );

            // C2: Z_mid − 2YZ + q_Z · p == 0
            let yz = y.clone() * &z;
            let two_yz = yz.clone() + &yz;
            let qz_p = q_z.clone() * &p;
            assert_eq!(
                z_mid.clone() - &two_yz + &qz_p,
                Int::ZERO,
                "C2 (Z_mid = 2YZ mod p) failed at row {row}"
            );

            // C3: X_mid − 9X⁴ + 8XS + q_X · p == 0
            let x_sq = x.clone() * &x;
            let x_quad = x_sq.clone() * &x_sq;
            let mut nine_x4 = x_quad.clone();
            for _ in 0..8 {
                nine_x4 = nine_x4 + &x_quad;
            }
            let xs = x.clone() * &s;
            let mut eight_xs = xs.clone();
            for _ in 0..7 {
                eight_xs = eight_xs + &xs;
            }
            let qx_p = q_x.clone() * &p;
            assert_eq!(
                x_mid.clone() - &nine_x4 + &eight_xs + &qx_p,
                Int::ZERO,
                "C3 (X_mid) failed at row {row}"
            );

            // C4: Y_mid − 12·X³·S + 3·X²·X_mid + 8·S² + q_Y · p == 0
            let x3s = x_sq.clone() * &xs; // X²·(X·S) = X³·S
            let mut twelve = x3s.clone();
            for _ in 0..11 {
                twelve = twelve + &x3s;
            }
            let xsq_xmid = x_sq.clone() * &x_mid;
            let mut three_xsq_xmid = xsq_xmid.clone();
            for _ in 0..2 {
                three_xsq_xmid = three_xsq_xmid + &xsq_xmid;
            }
            let s_sq = s.clone() * &s;
            let mut eight_s_sq = s_sq.clone();
            for _ in 0..7 {
                eight_s_sq = eight_s_sq + &s_sq;
            }
            let qy_p = q_y.clone() * &p;
            assert_eq!(
                y_mid.clone() - &twelve + &three_xsq_xmid + &eight_s_sq + &qy_p,
                Int::ZERO,
                "C4 (Y_mid) failed at row {row}"
            );
        }
    }
}
