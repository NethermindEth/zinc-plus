//! ECDSA Shamir scalar-multiplication UAIR (F_p / EC ops — composed
//! UAIR).
//!
//! Per row, computes one Shamir step `R_{t+1} = 2·R_t + addend_t`
//! where the verifier-supplied addend is one of `{O, G, Q, G+Q}` chosen
//! by the `(b_1[t], b_2[t])` bit pair. Implements:
//!
//! - **Row chaining**: `R_{t+1}.X1 ← R_t.X_out`, etc., via shifts of
//!   the `W_X1 / W_Y1 / W_Z1` columns.
//! - **Init boundary**: row 0's `(X1, Y1, Z1) = (PA_R_INIT_X, _Y, _Z)`,
//!   the verifier-supplied starting point.
//! - **Final boundary**: at the final row, Jacobian → affine
//!   conversion, exposing `R_x_aff` for the verifier's
//!   off-protocol `R_x mod n == r` check.
//! - **Conditional add** via `S_ADD` selector: the addition formula
//!   always runs and produces `(X_add, Y_add, Z_add)`, but the row
//!   output `(X_out, Y_out, Z_out)` selects between the
//!   doubled-only `(X_pa, Y_pa, Z_pa)` and the doubled-then-added
//!   value based on `S_ADD`.
//!
//! ## What's deferred
//!
//! - **Identity-aware initial step.** Starting from the Jacobian
//!   identity `O = (1, 1, 0)` breaks the mixed addition formulas
//!   (Z1=0 makes A=B=0, leading to `(0, 0, 0)` output for `O + G`
//!   instead of `G`). For now the verifier supplies a non-identity
//!   `R_init` (e.g., a precomputed Shamir-friendly offset point).
//!   Adding unified addition formulas to handle the identity input
//!   is a follow-up.
//! - **Bit columns and addend coordinates as derived publics.**
//!   This slice expects the verifier to supply both `(B_1, B_2)`
//!   bits and the corresponding `(PA_X_ADDEND, PA_Y_ADDEND)` per
//!   row directly. No in-circuit constraint binds the addend to
//!   the bits — that's a verifier-side check (and it's trivial:
//!   the verifier knows both because both derive from `u_1, u_2`).
//!
//! ## Constraint shape
//!
//! 28 constraints, max degree 5 (inherited from the doubling
//! sub-UAIR's `Y_mid` constraint).

use core::marker::PhantomData;

use crypto_bigint::{NonZero, Odd, Uint as CbUint};
use crypto_primitives::{ConstSemiring, crypto_bigint_int::Int};
use rand::RngCore;
use zinc_poly::{mle::DenseMultilinearExtension, univariate::dense::DensePolynomial};
use zinc_uair::{
    ConstraintBuilder, PublicColumnLayout, ShiftSpec, TotalColumnLayout, TraceRow, Uair,
    UairSignature, UairTrace,
    ideal::ImpossibleIdeal,
};

use crate::GenerateRandomTrace;
use crate::ecdsa_doubling::{EC_FP_INT_LIMBS, EcdsaFpRing, SECP256K1_P_UINT};

/// Number of Shamir doubling+add rounds. With `num_vars >= 9`,
/// trace rows = 512, so 256 active rounds + 1 final row + 255 padding
/// fits.
pub const NUM_SHAMIR_ROUNDS: usize = 256;

/// The trace row at which the affine-conversion / final-output
/// constraints apply (one past the last active doubling round).
pub const FINAL_ROW: usize = NUM_SHAMIR_ROUNDS;

// ---------------------------------------------------------------------------
// Column layout.
// ---------------------------------------------------------------------------

pub mod cols {
    // === Public columns (verifier-supplied) ===

    /// `1` at row 0; `0` elsewhere. Drives the initial-state binding.
    pub const S_INIT: usize = 0;
    /// `1` for `t ∈ 0..NUM_SHAMIR_ROUNDS`; `0` elsewhere. Gates every
    /// per-step constraint (doubling, addition, output selection,
    /// row chaining).
    pub const S_ACTIVE: usize = 1;
    /// `1` at `FINAL_ROW`; `0` elsewhere. Drives the affine-conversion
    /// and output-exposure constraints.
    pub const S_FINAL: usize = 2;
    /// `1` if the row's bit pair is non-zero (i.e., addition should
    /// take effect); `0` otherwise. Verifier supplies this since
    /// `(b_1, b_2)` are known to the verifier.
    pub const S_ADD: usize = 3;
    /// X-coordinate of the affine addend at this row. Verifier picks
    /// one of `{G_x, Q_x, (G+Q)_x}` based on `(b_1, b_2)`. When
    /// `S_ADD = 0`, this is unused (we recommend setting it to 0).
    pub const PA_X_ADDEND: usize = 4;
    /// Y-coordinate of the affine addend.
    pub const PA_Y_ADDEND: usize = 5;
    /// Initial Jacobian point coordinates (boundary input at row 0).
    pub const PA_R_INIT_X: usize = 6;
    pub const PA_R_INIT_Y: usize = 7;
    pub const PA_R_INIT_Z: usize = 8;
    pub const NUM_INT_PUB: usize = 9;

    // === Witness columns ===

    // Per-row Jacobian input `R_t` (chained from previous row's output).
    pub const W_X1: usize = 9;
    pub const W_Y1: usize = 10;
    pub const W_Z1: usize = 11;

    // Doubling intermediates (Y1², 9X1⁴-8X1S etc.) and outputs.
    pub const W_S: usize = 12; // = Y1²
    pub const W_X_PA: usize = 13; // = 9·X1⁴ - 8·X1·S    (= "X_pre_add")
    pub const W_Y_PA: usize = 14; // = 3·X1²·(4·X1·S - X_pa) - 8·S²
    pub const W_Z_PA: usize = 15; // = 2·Y1·Z1

    // Addition intermediates (using doubled point + addend).
    pub const W_Z_PA_SQ: usize = 16; // Z_pa²
    pub const W_Z_PA_CUBE: usize = 17; // Z_pa³
    pub const W_C: usize = 18; // = X_addend·Z_pa_sq − X_pa  (= H)
    pub const W_D: usize = 19; // = Y_addend·Z_pa_cube − Y_pa (= r)
    pub const W_E: usize = 20; // = C²
    pub const W_F: usize = 21; // = C·E
    pub const W_G: usize = 22; // = X_pa·E

    // Addition output (= 2R + addend, before s_add selection).
    pub const W_X_ADD: usize = 23;
    pub const W_Y_ADD: usize = 24;
    pub const W_Z_ADD: usize = 25;

    // Per-row final output (selected: doubled or added).
    pub const W_X_OUT: usize = 26;
    pub const W_Y_OUT: usize = 27;
    pub const W_Z_OUT: usize = 28;

    // Final-row affine conversion (only meaningful at FINAL_ROW).
    pub const W_Z_INV: usize = 29;
    pub const W_Z_INV_SQ: usize = 30;
    pub const W_Z_INV_CUBE: usize = 31;
    pub const W_X_AFF: usize = 32;
    pub const W_Y_AFF: usize = 33;

    pub const NUM_INT: usize = 34;

    // Flat indices for shift specs (no bin/poly columns, so flat = int).
    pub const FLAT_W_X1: usize = W_X1;
    pub const FLAT_W_Y1: usize = W_Y1;
    pub const FLAT_W_Z1: usize = W_Z1;
}

// ---------------------------------------------------------------------------
// The UAIR.
// ---------------------------------------------------------------------------

/// Shamir scalar-multiplication UAIR. See module docs for scope.
#[derive(Clone, Debug)]
pub struct EcdsaUair<R>(PhantomData<R>);

impl<R> Uair for EcdsaUair<R>
where
    R: EcdsaFpRing,
{
    type Ideal = ImpossibleIdeal;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(0, 0, cols::NUM_INT);
        let public = PublicColumnLayout::new(0, 0, cols::NUM_INT_PUB);
        // Shift X1, Y1, Z1 by 1 so we can constrain `down.X1[t] = up.X_out[t]`.
        let shifts: Vec<ShiftSpec> = vec![
            ShiftSpec::new(cols::FLAT_W_X1, 1),
            ShiftSpec::new(cols::FLAT_W_Y1, 1),
            ShiftSpec::new(cols::FLAT_W_Z1, 1),
        ];
        UairSignature::new(total, public, shifts, vec![])
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        down: TraceRow<B::Expr>,
        from_ref: FromR,
        mbs: MulByScalar,
        _ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&Self::Scalar) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &Self::Scalar) -> Option<B::Expr>,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
    {
        let int = up.int;
        let s_init = &int[cols::S_INIT];
        let s_active = &int[cols::S_ACTIVE];
        let s_final = &int[cols::S_FINAL];
        let s_add = &int[cols::S_ADD];
        let pa_x_addend = &int[cols::PA_X_ADDEND];
        let pa_y_addend = &int[cols::PA_Y_ADDEND];
        let pa_r_init_x = &int[cols::PA_R_INIT_X];
        let pa_r_init_y = &int[cols::PA_R_INIT_Y];
        let pa_r_init_z = &int[cols::PA_R_INIT_Z];
        let x1 = &int[cols::W_X1];
        let y1 = &int[cols::W_Y1];
        let z1 = &int[cols::W_Z1];
        let s_w = &int[cols::W_S];
        let x_pa = &int[cols::W_X_PA];
        let y_pa = &int[cols::W_Y_PA];
        let z_pa = &int[cols::W_Z_PA];
        let z_pa_sq = &int[cols::W_Z_PA_SQ];
        let z_pa_cube = &int[cols::W_Z_PA_CUBE];
        let c = &int[cols::W_C];
        let d = &int[cols::W_D];
        let e = &int[cols::W_E];
        let f = &int[cols::W_F];
        let g = &int[cols::W_G];
        let x_add = &int[cols::W_X_ADD];
        let y_add = &int[cols::W_Y_ADD];
        let z_add = &int[cols::W_Z_ADD];
        let x_out = &int[cols::W_X_OUT];
        let y_out = &int[cols::W_Y_OUT];
        let z_out = &int[cols::W_Z_OUT];
        let z_inv = &int[cols::W_Z_INV];
        let z_inv_sq = &int[cols::W_Z_INV_SQ];
        let z_inv_cube = &int[cols::W_Z_INV_CUBE];
        let x_aff = &int[cols::W_X_AFF];
        let y_aff = &int[cols::W_Y_AFF];

        // down.int[i] in source-col-ascending order: X1, Y1, Z1.
        let down_x1 = &down.int[0];
        let down_y1 = &down.int[1];
        let down_z1 = &down.int[2];

        let two_scalar = const_scalar::<R>(R::from(2_u32));
        let three_scalar = const_scalar::<R>(R::from(3_u32));
        let eight_scalar = const_scalar::<R>(R::from(8_u32));
        let nine_scalar = const_scalar::<R>(R::from(9_u32));
        let twelve_scalar = const_scalar::<R>(R::from(12_u32));
        let one_expr = from_ref(&const_scalar::<R>(R::ONE));

        // ===================================================================
        // Doubling block (4 constraints, max degree 5; copied from
        // ecdsa_doubling slice but operating on (X1,Y1,Z1) → (X_pa,Y_pa,Z_pa)).
        // ===================================================================

        // C-D1: S − Y1² = 0
        let d1_inner = s_w.clone() - &(y1.clone() * y1);
        b.assert_zero(s_active.clone() * &d1_inner);

        // C-D2: Z_pa − 2·Y1·Z1 = 0
        let yz = y1.clone() * z1;
        let two_yz = mbs(&yz, &two_scalar).expect("2·Y1·Z1 overflow");
        let d2_inner = z_pa.clone() - &two_yz;
        b.assert_zero(s_active.clone() * &d2_inner);

        // C-D3: X_pa − 9·X1⁴ + 8·X1·S = 0
        let x_sq = x1.clone() * x1;
        let x_pow4 = x_sq.clone() * &x_sq;
        let nine_x4 = mbs(&x_pow4, &nine_scalar).expect("9·X1⁴ overflow");
        let xs = x1.clone() * s_w;
        let eight_xs = mbs(&xs, &eight_scalar).expect("8·X1·S overflow");
        let d3_inner = x_pa.clone() - &nine_x4 + &eight_xs;
        b.assert_zero(s_active.clone() * &d3_inner);

        // C-D4: Y_pa − 12·X1³·S + 3·X1²·X_pa + 8·S² = 0
        let x3s = x_sq.clone() * &xs;
        let twelve_x3s = mbs(&x3s, &twelve_scalar).expect("12·X1³·S overflow");
        let x_sq_x_pa = x_sq.clone() * x_pa;
        let three_x2_xpa =
            mbs(&x_sq_x_pa, &three_scalar).expect("3·X1²·X_pa overflow");
        let s_sq = s_w.clone() * s_w;
        let eight_s_sq = mbs(&s_sq, &eight_scalar).expect("8·S² overflow");
        let d4_inner = y_pa.clone() - &twelve_x3s + &three_x2_xpa + &eight_s_sq;
        b.assert_zero(s_active.clone() * &d4_inner);

        // ===================================================================
        // Addition block (10 constraints, max degree 3; mirrors the
        // ecdsa_addition slice but using (X_pa,Y_pa,Z_pa) as the Jacobian
        // input and (PA_X_ADDEND, PA_Y_ADDEND) as the affine addend).
        // ===================================================================

        // C-A1: Z_pa_sq − Z_pa·Z_pa = 0
        let a1_inner = z_pa_sq.clone() - &(z_pa.clone() * z_pa);
        b.assert_zero(s_active.clone() * &a1_inner);

        // C-A2: Z_pa_cube − Z_pa·Z_pa_sq = 0
        let a2_inner = z_pa_cube.clone() - &(z_pa.clone() * z_pa_sq);
        b.assert_zero(s_active.clone() * &a2_inner);

        // C-A3: C − X_addend·Z_pa_sq + X_pa = 0
        let a3_inner = c.clone() + x_pa - &(pa_x_addend.clone() * z_pa_sq);
        b.assert_zero(s_active.clone() * &a3_inner);

        // C-A4: D − Y_addend·Z_pa_cube + Y_pa = 0
        let a4_inner = d.clone() + y_pa - &(pa_y_addend.clone() * z_pa_cube);
        b.assert_zero(s_active.clone() * &a4_inner);

        // C-A5: E − C·C = 0
        let a5_inner = e.clone() - &(c.clone() * c);
        b.assert_zero(s_active.clone() * &a5_inner);

        // C-A6: F − C·E = 0
        let a6_inner = f.clone() - &(c.clone() * e);
        b.assert_zero(s_active.clone() * &a6_inner);

        // C-A7: G − X_pa·E = 0
        let a7_inner = g.clone() - &(x_pa.clone() * e);
        b.assert_zero(s_active.clone() * &a7_inner);

        // C-A8: X_add − D² + F + 2·G = 0
        let d_sq = d.clone() * d;
        let two_g = mbs(g, &two_scalar).expect("2·G overflow");
        let a8_inner = x_add.clone() - &d_sq + f + &two_g;
        b.assert_zero(s_active.clone() * &a8_inner);

        // C-A9: Y_add − D·(G − X_add) + Y_pa·F = 0
        let g_minus_xadd = g.clone() - x_add;
        let d_times = d.clone() * &g_minus_xadd;
        let y_pa_f = y_pa.clone() * f;
        let a9_inner = y_add.clone() - &d_times + &y_pa_f;
        b.assert_zero(s_active.clone() * &a9_inner);

        // C-A10: Z_add − Z_pa·C = 0
        let a10_inner = z_add.clone() - &(z_pa.clone() * c);
        b.assert_zero(s_active.clone() * &a10_inner);

        // ===================================================================
        // Output selection (3 constraints, degree 3).
        //   X_out = s_add ? X_add : X_pa
        //   ⟺ X_out − X_pa − s_add·(X_add − X_pa) = 0
        // ===================================================================

        let x_diff = x_add.clone() - x_pa;
        let s_add_xdiff = s_add.clone() * &x_diff;
        let o1_inner = x_out.clone() - x_pa - &s_add_xdiff;
        b.assert_zero(s_active.clone() * &o1_inner);

        let y_diff = y_add.clone() - y_pa;
        let s_add_ydiff = s_add.clone() * &y_diff;
        let o2_inner = y_out.clone() - y_pa - &s_add_ydiff;
        b.assert_zero(s_active.clone() * &o2_inner);

        let z_diff = z_add.clone() - z_pa;
        let s_add_zdiff = s_add.clone() * &z_diff;
        let o3_inner = z_out.clone() - z_pa - &s_add_zdiff;
        b.assert_zero(s_active.clone() * &o3_inner);

        // ===================================================================
        // Init boundary: at row 0, R = (PA_R_INIT_X, PA_R_INIT_Y, PA_R_INIT_Z).
        // ===================================================================

        b.assert_zero(s_init.clone() * &(x1.clone() - pa_r_init_x));
        b.assert_zero(s_init.clone() * &(y1.clone() - pa_r_init_y));
        b.assert_zero(s_init.clone() * &(z1.clone() - pa_r_init_z));

        // ===================================================================
        // Row chaining: down.X1[t] = up.X_out[t] for active rows.
        // ===================================================================

        b.assert_zero(s_active.clone() * &(down_x1.clone() - x_out));
        b.assert_zero(s_active.clone() * &(down_y1.clone() - y_out));
        b.assert_zero(s_active.clone() * &(down_z1.clone() - z_out));

        // ===================================================================
        // Final-row affine conversion (5 constraints, gated by s_final).
        // Uses the FINAL_ROW's (X1, Y1, Z1) (which equal R_NUM_SHAMIR_ROUNDS
        // via the chaining at the last active row).
        // ===================================================================

        // F1: Z1·Z_inv − 1 = 0
        let f1_inner = z1.clone() * z_inv - &one_expr;
        b.assert_zero(s_final.clone() * &f1_inner);

        // F2: Z_inv_sq − Z_inv² = 0
        let f2_inner = z_inv_sq.clone() - &(z_inv.clone() * z_inv);
        b.assert_zero(s_final.clone() * &f2_inner);

        // F3: Z_inv_cube − Z_inv·Z_inv_sq = 0
        let f3_inner = z_inv_cube.clone() - &(z_inv.clone() * z_inv_sq);
        b.assert_zero(s_final.clone() * &f3_inner);

        // F4: X_aff − X1·Z_inv_sq = 0
        let f4_inner = x_aff.clone() - &(x1.clone() * z_inv_sq);
        b.assert_zero(s_final.clone() * &f4_inner);

        // F5: Y_aff − Y1·Z_inv_cube = 0
        let f5_inner = y_aff.clone() - &(y1.clone() * z_inv_cube);
        b.assert_zero(s_final.clone() * &f5_inner);
    }
}

/// Build a constant-polynomial (degree 0) `c` as a `DensePolynomial<R, 32>`.
fn const_scalar<R: ConstSemiring>(c: R) -> DensePolynomial<R, 32> {
    let mut coeffs = [R::ZERO; 32];
    coeffs[0] = c;
    DensePolynomial::<R, 32>::new(coeffs)
}

// ---------------------------------------------------------------------------
// F_p arithmetic helpers.
// ---------------------------------------------------------------------------

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

fn rand_nonzero_fp<Rng: RngCore + ?Sized>(rng: &mut Rng) -> CbUint<EC_FP_INT_LIMBS> {
    use crypto_bigint::Zero as _;
    loop {
        let candidate = rand_fp(rng);
        if !bool::from(candidate.is_zero()) {
            return candidate;
        }
    }
}

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
    use crypto_bigint::CheckedSub;
    a.checked_sub(&SECP256K1_P_UINT).is_some().into()
}

fn sub_mod_p(
    a: &CbUint<EC_FP_INT_LIMBS>,
    b: &CbUint<EC_FP_INT_LIMBS>,
) -> CbUint<EC_FP_INT_LIMBS> {
    use crypto_bigint::CheckedSub;
    let p_nz = NonZero::new(SECP256K1_P_UINT).expect("p is nonzero");
    if a.checked_sub(b).is_some().into() {
        a.wrapping_sub(b).rem_vartime(&p_nz)
    } else {
        let a_plus_p = a.wrapping_add(&SECP256K1_P_UINT);
        a_plus_p.wrapping_sub(b).rem_vartime(&p_nz)
    }
}

fn inv_mod_p(a: &CbUint<EC_FP_INT_LIMBS>) -> CbUint<EC_FP_INT_LIMBS> {
    let p_odd = Odd::new(SECP256K1_P_UINT).expect("p is odd");
    a.invert_odd_mod(&p_odd).expect("a has no inverse mod p")
}

fn uint_to_int(u: CbUint<EC_FP_INT_LIMBS>) -> Int<EC_FP_INT_LIMBS> {
    debug_assert!(
        u.bits() <= 64 * EC_FP_INT_LIMBS as u32 - 1,
        "uint top bit must be 0 to reinterpret as signed"
    );
    Int::new(*u.as_int())
}

// ---------------------------------------------------------------------------
// Reference per-step computation (for witness gen and tests).
// ---------------------------------------------------------------------------

/// One Shamir step: doubled-then-conditionally-added Jacobian point.
/// All the per-row F_p scratch values are returned.
struct StepValues {
    s: CbUint<EC_FP_INT_LIMBS>,
    x_pa: CbUint<EC_FP_INT_LIMBS>,
    y_pa: CbUint<EC_FP_INT_LIMBS>,
    z_pa: CbUint<EC_FP_INT_LIMBS>,
    z_pa_sq: CbUint<EC_FP_INT_LIMBS>,
    z_pa_cube: CbUint<EC_FP_INT_LIMBS>,
    c: CbUint<EC_FP_INT_LIMBS>,
    d: CbUint<EC_FP_INT_LIMBS>,
    e: CbUint<EC_FP_INT_LIMBS>,
    f: CbUint<EC_FP_INT_LIMBS>,
    g: CbUint<EC_FP_INT_LIMBS>,
    x_add: CbUint<EC_FP_INT_LIMBS>,
    y_add: CbUint<EC_FP_INT_LIMBS>,
    z_add: CbUint<EC_FP_INT_LIMBS>,
    x_out: CbUint<EC_FP_INT_LIMBS>,
    y_out: CbUint<EC_FP_INT_LIMBS>,
    z_out: CbUint<EC_FP_INT_LIMBS>,
}

fn compute_step(
    x1: &CbUint<EC_FP_INT_LIMBS>,
    y1: &CbUint<EC_FP_INT_LIMBS>,
    z1: &CbUint<EC_FP_INT_LIMBS>,
    pa_x: &CbUint<EC_FP_INT_LIMBS>,
    pa_y: &CbUint<EC_FP_INT_LIMBS>,
    s_add_bit: bool,
) -> StepValues {
    // --- Doubling ---
    let s = mul_mod_p(y1, y1);
    let x_sq = mul_mod_p(x1, x1);
    let x_quad = mul_mod_p(&x_sq, &x_sq);
    let xs = mul_mod_p(x1, &s);
    let nine_xq = small_mul_mod_p(&x_quad, 9);
    let eight_xs = small_mul_mod_p(&xs, 8);
    let x_pa = sub_mod_p(&nine_xq, &eight_xs);

    let yz = mul_mod_p(y1, z1);
    let z_pa = small_mul_mod_p(&yz, 2);

    let four_xs = small_mul_mod_p(&xs, 4);
    let four_xs_minus_xpa = sub_mod_p(&four_xs, &x_pa);
    let three_xsq = small_mul_mod_p(&x_sq, 3);
    let big_term = mul_mod_p(&three_xsq, &four_xs_minus_xpa);
    let s_sq = mul_mod_p(&s, &s);
    let eight_s_sq = small_mul_mod_p(&s_sq, 8);
    let y_pa = sub_mod_p(&big_term, &eight_s_sq);

    // --- Addition (always computed; output selection happens after) ---
    let z_pa_sq = mul_mod_p(&z_pa, &z_pa);
    let z_pa_cube = mul_mod_p(&z_pa, &z_pa_sq);
    let a_val = mul_mod_p(pa_x, &z_pa_sq);
    let b_val = mul_mod_p(pa_y, &z_pa_cube);
    let c = sub_mod_p(&a_val, &x_pa);
    let d = sub_mod_p(&b_val, &y_pa);
    let e = mul_mod_p(&c, &c);
    let f = mul_mod_p(&c, &e);
    let g = mul_mod_p(&x_pa, &e);

    let d_sq = mul_mod_p(&d, &d);
    let two_g = small_mul_mod_p(&g, 2);
    let x_add = sub_mod_p(&sub_mod_p(&d_sq, &f), &two_g);

    let g_minus_x_add = sub_mod_p(&g, &x_add);
    let d_times = mul_mod_p(&d, &g_minus_x_add);
    let y_pa_f = mul_mod_p(&y_pa, &f);
    let y_add = sub_mod_p(&d_times, &y_pa_f);

    let z_add = mul_mod_p(&z_pa, &c);

    // --- Output selection ---
    let (x_out, y_out, z_out) = if s_add_bit {
        (x_add.clone(), y_add.clone(), z_add.clone())
    } else {
        (x_pa.clone(), y_pa.clone(), z_pa.clone())
    };

    StepValues {
        s,
        x_pa,
        y_pa,
        z_pa,
        z_pa_sq,
        z_pa_cube,
        c,
        d,
        e,
        f,
        g,
        x_add,
        y_add,
        z_add,
        x_out,
        y_out,
        z_out,
    }
}

// ---------------------------------------------------------------------------
// Witness generator.
// ---------------------------------------------------------------------------

impl<R> GenerateRandomTrace<32> for EcdsaUair<R>
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
        assert!(
            n_rows > FINAL_ROW,
            "Shamir UAIR needs > {FINAL_ROW} rows; got {n_rows}",
        );

        // Pick a non-identity initial point and a non-identity addend.
        // (The mixed addition formulas don't handle identity inputs.)
        let r_init_x = rand_fp(rng);
        let r_init_y = rand_fp(rng);
        let r_init_z = rand_nonzero_fp(rng);
        let pa_x = rand_fp(rng);
        let pa_y = rand_fp(rng);

        // Sample all bits as 1 (s_add = 1 every row) to keep the test
        // simple and exercise the addition path. (s_add = 0 is also
        // covered by the constraint structure, just not by this trace.)

        // Build the per-row state by simulating the Shamir loop.
        let mut x1_seq: Vec<CbUint<EC_FP_INT_LIMBS>> = Vec::with_capacity(n_rows);
        let mut y1_seq: Vec<CbUint<EC_FP_INT_LIMBS>> = Vec::with_capacity(n_rows);
        let mut z1_seq: Vec<CbUint<EC_FP_INT_LIMBS>> = Vec::with_capacity(n_rows);
        x1_seq.push(r_init_x.clone());
        y1_seq.push(r_init_y.clone());
        z1_seq.push(r_init_z.clone());

        let mut steps: Vec<StepValues> = Vec::with_capacity(NUM_SHAMIR_ROUNDS);
        for t in 0..NUM_SHAMIR_ROUNDS {
            let step = compute_step(&x1_seq[t], &y1_seq[t], &z1_seq[t], &pa_x, &pa_y, true);
            x1_seq.push(step.x_out.clone());
            y1_seq.push(step.y_out.clone());
            z1_seq.push(step.z_out.clone());
            steps.push(step);
        }

        // Pad the chained-input sequence to n_rows.
        let zero_uint = CbUint::<EC_FP_INT_LIMBS>::ZERO;
        while x1_seq.len() < n_rows {
            x1_seq.push(zero_uint.clone());
            y1_seq.push(zero_uint.clone());
            z1_seq.push(zero_uint.clone());
        }

        // Affine conversion at FINAL_ROW.
        let z_final = z1_seq[FINAL_ROW].clone();
        let z_inv_v = inv_mod_p(&z_final);
        let z_inv_sq_v = mul_mod_p(&z_inv_v, &z_inv_v);
        let z_inv_cube_v = mul_mod_p(&z_inv_v, &z_inv_sq_v);
        let x_aff_v = mul_mod_p(&x1_seq[FINAL_ROW], &z_inv_sq_v);
        let y_aff_v = mul_mod_p(&y1_seq[FINAL_ROW], &z_inv_cube_v);

        // ---- Populate columns. ----
        let zero_r = || R::ZERO;
        let mk_col = || vec![zero_r(); n_rows];

        let mut s_init_col: Vec<R> = mk_col();
        let mut s_active_col: Vec<R> = mk_col();
        let mut s_final_col: Vec<R> = mk_col();
        let mut s_add_col: Vec<R> = mk_col();
        let mut pa_x_addend_col: Vec<R> = mk_col();
        let mut pa_y_addend_col: Vec<R> = mk_col();
        let mut pa_r_init_x_col: Vec<R> = mk_col();
        let mut pa_r_init_y_col: Vec<R> = mk_col();
        let mut pa_r_init_z_col: Vec<R> = mk_col();
        let mut x1_col: Vec<R> = mk_col();
        let mut y1_col: Vec<R> = mk_col();
        let mut z1_col: Vec<R> = mk_col();
        let mut s_col: Vec<R> = mk_col();
        let mut x_pa_col: Vec<R> = mk_col();
        let mut y_pa_col: Vec<R> = mk_col();
        let mut z_pa_col: Vec<R> = mk_col();
        let mut z_pa_sq_col: Vec<R> = mk_col();
        let mut z_pa_cube_col: Vec<R> = mk_col();
        let mut c_col: Vec<R> = mk_col();
        let mut d_col: Vec<R> = mk_col();
        let mut e_col: Vec<R> = mk_col();
        let mut f_col: Vec<R> = mk_col();
        let mut g_col: Vec<R> = mk_col();
        let mut x_add_col: Vec<R> = mk_col();
        let mut y_add_col: Vec<R> = mk_col();
        let mut z_add_col: Vec<R> = mk_col();
        let mut x_out_col: Vec<R> = mk_col();
        let mut y_out_col: Vec<R> = mk_col();
        let mut z_out_col: Vec<R> = mk_col();
        let mut z_inv_col: Vec<R> = mk_col();
        let mut z_inv_sq_col: Vec<R> = mk_col();
        let mut z_inv_cube_col: Vec<R> = mk_col();
        let mut x_aff_col: Vec<R> = mk_col();
        let mut y_aff_col: Vec<R> = mk_col();

        // Selectors.
        s_init_col[0] = R::ONE;
        for t in 0..NUM_SHAMIR_ROUNDS {
            s_active_col[t] = R::ONE;
            s_add_col[t] = R::ONE;
            pa_x_addend_col[t] = R::from(uint_to_int(pa_x.clone()));
            pa_y_addend_col[t] = R::from(uint_to_int(pa_y.clone()));
        }
        s_final_col[FINAL_ROW] = R::ONE;

        // PA_R_INIT is constant per row 0 (but harmless to fill all rows
        // — we only constrain it at row 0 via S_INIT).
        pa_r_init_x_col[0] = R::from(uint_to_int(r_init_x));
        pa_r_init_y_col[0] = R::from(uint_to_int(r_init_y));
        pa_r_init_z_col[0] = R::from(uint_to_int(r_init_z));

        // Chained input columns.
        for t in 0..n_rows {
            x1_col[t] = R::from(uint_to_int(x1_seq[t].clone()));
            y1_col[t] = R::from(uint_to_int(y1_seq[t].clone()));
            z1_col[t] = R::from(uint_to_int(z1_seq[t].clone()));
        }

        // Per-step intermediate / output columns (rows 0..NUM_SHAMIR_ROUNDS).
        for (t, step) in steps.iter().enumerate() {
            s_col[t] = R::from(uint_to_int(step.s.clone()));
            x_pa_col[t] = R::from(uint_to_int(step.x_pa.clone()));
            y_pa_col[t] = R::from(uint_to_int(step.y_pa.clone()));
            z_pa_col[t] = R::from(uint_to_int(step.z_pa.clone()));
            z_pa_sq_col[t] = R::from(uint_to_int(step.z_pa_sq.clone()));
            z_pa_cube_col[t] = R::from(uint_to_int(step.z_pa_cube.clone()));
            c_col[t] = R::from(uint_to_int(step.c.clone()));
            d_col[t] = R::from(uint_to_int(step.d.clone()));
            e_col[t] = R::from(uint_to_int(step.e.clone()));
            f_col[t] = R::from(uint_to_int(step.f.clone()));
            g_col[t] = R::from(uint_to_int(step.g.clone()));
            x_add_col[t] = R::from(uint_to_int(step.x_add.clone()));
            y_add_col[t] = R::from(uint_to_int(step.y_add.clone()));
            z_add_col[t] = R::from(uint_to_int(step.z_add.clone()));
            x_out_col[t] = R::from(uint_to_int(step.x_out.clone()));
            y_out_col[t] = R::from(uint_to_int(step.y_out.clone()));
            z_out_col[t] = R::from(uint_to_int(step.z_out.clone()));
        }

        // Final-row affine cells.
        z_inv_col[FINAL_ROW] = R::from(uint_to_int(z_inv_v));
        z_inv_sq_col[FINAL_ROW] = R::from(uint_to_int(z_inv_sq_v));
        z_inv_cube_col[FINAL_ROW] = R::from(uint_to_int(z_inv_cube_v));
        x_aff_col[FINAL_ROW] = R::from(uint_to_int(x_aff_v));
        y_aff_col[FINAL_ROW] = R::from(uint_to_int(y_aff_v));

        let to_mle = |col: Vec<R>| -> DenseMultilinearExtension<R> { col.into_iter().collect() };

        let int = vec![
            to_mle(s_init_col),
            to_mle(s_active_col),
            to_mle(s_final_col),
            to_mle(s_add_col),
            to_mle(pa_x_addend_col),
            to_mle(pa_y_addend_col),
            to_mle(pa_r_init_x_col),
            to_mle(pa_r_init_y_col),
            to_mle(pa_r_init_z_col),
            to_mle(x1_col),
            to_mle(y1_col),
            to_mle(z1_col),
            to_mle(s_col),
            to_mle(x_pa_col),
            to_mle(y_pa_col),
            to_mle(z_pa_col),
            to_mle(z_pa_sq_col),
            to_mle(z_pa_cube_col),
            to_mle(c_col),
            to_mle(d_col),
            to_mle(e_col),
            to_mle(f_col),
            to_mle(g_col),
            to_mle(x_add_col),
            to_mle(y_add_col),
            to_mle(z_add_col),
            to_mle(x_out_col),
            to_mle(y_out_col),
            to_mle(z_out_col),
            to_mle(z_inv_col),
            to_mle(z_inv_sq_col),
            to_mle(z_inv_cube_col),
            to_mle(x_aff_col),
            to_mle(y_aff_col),
        ];

        UairTrace {
            int: int.into(),
            ..Default::default()
        }
    }
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

    /// Sanity: 28 constraints. Max degree is 5 (inherited from
    /// the doubling slice's `Y_pa` constraint).
    #[test]
    fn shamir_constraint_shape() {
        type U = EcdsaUair<Int<EC_FP_INT_LIMBS>>;
        assert_eq!(count_constraints::<U>(), 28);
        assert_eq!(count_max_degree::<U>(), 5);
        // Spot-check: doubling block contributes [3, 3, 5, 5];
        // addition block all 3s; output selection 3s; init 2s;
        // chaining 2s; final 3s.
        let degrees = count_constraint_degrees::<U>();
        assert!(degrees.iter().any(|&d| d == 5), "expected at least one deg-5");
        assert!(degrees.iter().filter(|&&d| d == 2).count() >= 6, "init+chain ≥ 6 deg-2");
    }

    /// Witness gen produces a trace where every constraint vanishes
    /// mod p. Exercises the active-row block (doubling+addition+
    /// selection+chaining) and the final-row affine conversion.
    #[test]
    fn witness_satisfies_constraints_mod_p() {
        let num_vars = 9; // 512 rows ≥ FINAL_ROW + 1 = 257
        let mut r = rng();
        let trace = <EcdsaUair<Int<EC_FP_INT_LIMBS>> as GenerateRandomTrace<32>>::
            generate_random_trace(num_vars, &mut r);
        let n_rows = 1 << num_vars;
        assert_eq!(trace.int.len(), cols::NUM_INT);

        let int_to_uint = |v: &Int<EC_FP_INT_LIMBS>| -> CbUint<EC_FP_INT_LIMBS> {
            *v.inner().as_uint()
        };
        let read_uint = |c: usize, t: usize| int_to_uint(&trace.int[c][t]);
        let one_uint: CbUint<EC_FP_INT_LIMBS> = CbUint::ONE;
        let zero_uint: CbUint<EC_FP_INT_LIMBS> = CbUint::ZERO;

        for t in 0..n_rows {
            let s_active_int = trace.int[cols::S_ACTIVE][t].clone();
            let s_init_int = trace.int[cols::S_INIT][t].clone();
            let s_final_int = trace.int[cols::S_FINAL][t].clone();
            let active = s_active_int == Int::ONE;
            let init = s_init_int == Int::ONE;
            let final_row = s_final_int == Int::ONE;

            // Active rows: doubling + addition + selection + chaining.
            if active {
                let s_add_int = trace.int[cols::S_ADD][t].clone();
                let s_add_bit = s_add_int == Int::ONE;

                let pa_x = read_uint(cols::PA_X_ADDEND, t);
                let pa_y = read_uint(cols::PA_Y_ADDEND, t);
                let x1 = read_uint(cols::W_X1, t);
                let y1 = read_uint(cols::W_Y1, t);
                let z1 = read_uint(cols::W_Z1, t);

                let expected = compute_step(&x1, &y1, &z1, &pa_x, &pa_y, s_add_bit);

                assert_eq!(read_uint(cols::W_S, t), expected.s, "S at row {t}");
                assert_eq!(read_uint(cols::W_X_PA, t), expected.x_pa, "X_pa at row {t}");
                assert_eq!(read_uint(cols::W_Y_PA, t), expected.y_pa, "Y_pa at row {t}");
                assert_eq!(read_uint(cols::W_Z_PA, t), expected.z_pa, "Z_pa at row {t}");
                assert_eq!(read_uint(cols::W_Z_PA_SQ, t), expected.z_pa_sq, "Z_pa² at row {t}");
                assert_eq!(read_uint(cols::W_Z_PA_CUBE, t), expected.z_pa_cube, "Z_pa³ at row {t}");
                assert_eq!(read_uint(cols::W_C, t), expected.c, "C at row {t}");
                assert_eq!(read_uint(cols::W_D, t), expected.d, "D at row {t}");
                assert_eq!(read_uint(cols::W_E, t), expected.e, "E at row {t}");
                assert_eq!(read_uint(cols::W_F, t), expected.f, "F at row {t}");
                assert_eq!(read_uint(cols::W_G, t), expected.g, "G at row {t}");
                assert_eq!(read_uint(cols::W_X_ADD, t), expected.x_add, "X_add at row {t}");
                assert_eq!(read_uint(cols::W_Y_ADD, t), expected.y_add, "Y_add at row {t}");
                assert_eq!(read_uint(cols::W_Z_ADD, t), expected.z_add, "Z_add at row {t}");
                assert_eq!(read_uint(cols::W_X_OUT, t), expected.x_out, "X_out at row {t}");
                assert_eq!(read_uint(cols::W_Y_OUT, t), expected.y_out, "Y_out at row {t}");
                assert_eq!(read_uint(cols::W_Z_OUT, t), expected.z_out, "Z_out at row {t}");

                // Chaining: row t+1 input = row t output.
                if t + 1 < n_rows {
                    assert_eq!(read_uint(cols::W_X1, t + 1), expected.x_out, "chain X at {t}");
                    assert_eq!(read_uint(cols::W_Y1, t + 1), expected.y_out, "chain Y at {t}");
                    assert_eq!(read_uint(cols::W_Z1, t + 1), expected.z_out, "chain Z at {t}");
                }
            }

            // Init boundary: row 0 input = PA_R_INIT.
            if init {
                assert_eq!(
                    read_uint(cols::W_X1, t),
                    read_uint(cols::PA_R_INIT_X, t),
                    "init X at {t}",
                );
                assert_eq!(
                    read_uint(cols::W_Y1, t),
                    read_uint(cols::PA_R_INIT_Y, t),
                    "init Y at {t}",
                );
                assert_eq!(
                    read_uint(cols::W_Z1, t),
                    read_uint(cols::PA_R_INIT_Z, t),
                    "init Z at {t}",
                );
            }

            // Final boundary: affine conversion of (X1, Y1, Z1).
            if final_row {
                let x1 = read_uint(cols::W_X1, t);
                let y1 = read_uint(cols::W_Y1, t);
                let z1 = read_uint(cols::W_Z1, t);
                let z_inv = read_uint(cols::W_Z_INV, t);
                let z_inv_sq = read_uint(cols::W_Z_INV_SQ, t);
                let z_inv_cube = read_uint(cols::W_Z_INV_CUBE, t);
                let x_aff = read_uint(cols::W_X_AFF, t);
                let y_aff = read_uint(cols::W_Y_AFF, t);

                assert_eq!(mul_mod_p(&z1, &z_inv), one_uint, "F1 (Z·Z_inv=1) at {t}");
                assert_eq!(z_inv_sq, mul_mod_p(&z_inv, &z_inv), "F2 (Z_inv²) at {t}");
                assert_eq!(
                    z_inv_cube,
                    mul_mod_p(&z_inv, &z_inv_sq),
                    "F3 (Z_inv³) at {t}",
                );
                assert_eq!(x_aff, mul_mod_p(&x1, &z_inv_sq), "F4 (X_aff) at {t}");
                assert_eq!(y_aff, mul_mod_p(&y1, &z_inv_cube), "F5 (Y_aff) at {t}");
            }

            // Padding rows: chained inputs are zero (uninitialized but
            // unconstrained — sanity check only).
            if !active && !final_row {
                assert_eq!(read_uint(cols::W_X1, t), zero_uint, "pad X1 at {t}");
            }
        }
    }
}
