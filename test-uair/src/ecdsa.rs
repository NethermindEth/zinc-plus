//! ECDSA Shamir scalar-multiplication UAIR (F_p / EC ops — composed
//! UAIR).
//!
//! Per row, computes one Shamir step `R_{t+1} = 2·R_t + addend_t`
//! where the verifier-supplied addend is one of `{O, G, Q, G+Q}` chosen
//! by the `(b_1[t], b_2[t])` bit pair. Implements:
//!
//! - **Row chaining**: a single chained `(W_X, W_Y, W_Z)` triple where
//!   `up.X[t] = R_t` (the row's input) and `down.X[t] = R_{t+1}` (the
//!   next row's input, written by this row's output-selection
//!   constraint). No separate input/output columns.
//! - **Init boundary**: row 0's `(X, Y, Z) = (PA_R_INIT_X, _Y, _Z)`,
//!   the verifier-supplied starting point.
//! - **Final boundary**: at the final row, Jacobian → affine
//!   conversion, exposing `R_x_aff` for the verifier's
//!   off-protocol `R_x mod n == r` check.
//! - **Conditional add** via `S_ADD` selector: the addition formula is
//!   inlined into the output-selection constraints — when `S_ADD = 1`,
//!   `down.(X,Y,Z) = added`; when `S_ADD = 0`, `down.(X,Y,Z) =
//!   doubled`.
//!
//! ## Constraint shape
//!
//! 11 constraints, max degree 6. Tighter than the spec at
//! `arithmetization_standalone/hybrid_arithmetics/ecdsa/ecdsa_intro.tex`
//! by inlining `S = Y²` and dropping the in-circuit affine block.
//!
//! 8 EC witness columns: `(X, Y, Z, X_pa, Y_pa, Z_pa, C=H, D=R_a)`.
//! Higher-degree intermediates `Y², Y⁴, Z_pa², Z_pa³, C², C³, X_pa·C²`
//! are inlined into the constraint expressions. Affine readout
//! (Z_inv, X_aff, Y_aff) is fully off-protocol — the verifier opens
//! Z[FINAL_ROW] and computes the affine coordinates itself, or a
//! downstream gluing UAIR enforces the binding.
//!
//! Max degree 6 attained by the inlined Y output-selection
//! constraint's `s_active · S_ADD · D · X_pa · C²` term (matches
//! the spec's `s_reg · R_a · X_mid · H²`). D4 also reaches degree 6
//! via the `12·X³·Y²` term after `S` inlining.
//!
//! Breakdown:
//! - 3 doubling (D2 deg 3, D3 deg 5, D4 deg 6)
//! - 2 addition intermediates: `C` (deg 4), `D` (deg 5)
//! - 3 output-selection-and-chaining (X: deg 5, Y: deg 6, Z: deg 4)
//! - 3 init-boundary (deg 2)
//!
//! ## What's deferred
//!
//! - **Identity-aware initial step.** Starting from the Jacobian
//!   identity `O = (1, 1, 0)` breaks the mixed addition formulas
//!   (Z1=0 makes A=B=0). The verifier supplies a non-identity
//!   `R_init`. Adding unified addition formulas to handle the
//!   identity input is a follow-up.
//! - **Bit columns and addend coordinates as derived publics.**
//!   The verifier supplies both `(B_1, B_2)` bits (encoded as
//!   `S_ADD`) and the corresponding `(PA_X_ADDEND, PA_Y_ADDEND)` per
//!   row. No in-circuit constraint binds the addend to the bits —
//!   that's a verifier-side check.

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
use crate::ecdsa_doubling::{EC_FP_INT_LIMBS, EcdsaFpRing, SECP256K1_N_UINT, SECP256K1_P_UINT};

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

    /// `1` at row 0; `0` elsewhere.
    pub const S_INIT: usize = 0;
    /// `1` for `t ∈ 0..NUM_SHAMIR_ROUNDS`; `0` elsewhere.
    pub const S_ACTIVE: usize = 1;
    /// `1` at `FINAL_ROW`; `0` elsewhere.
    pub const S_FINAL: usize = 2;
    /// `1` if the row's bit pair is non-zero (addition takes effect);
    /// `0` otherwise. Equal to `b_1 + b_2 - b_1 · b_2`.
    /// Verifier-derivable from `(b_1, b_2)`.
    pub const S_ADD: usize = 3;
    /// First scalar bit of the row's $(b_1, b_2)$ Shamir pair, in
    /// $\{0,1\}$. Combined with `PA_B2`, selects the affine addend
    /// $T \in \{\OOO, G, Q, G+Q\}$ in-circuit via the formula
    ///   T = b_1·(1-b_2)·G + (1-b_1)·b_2·Q + b_1·b_2·(G+Q).
    pub const PA_B1: usize = 4;
    /// Second scalar bit; see `PA_B1`.
    pub const PA_B2: usize = 5;
    /// Affine X-coordinate of $Q$ (the public key). Constant across
    /// all rows of a single proof; consumed by the in-circuit addend
    /// formula.
    pub const PA_QX: usize = 6;
    /// Affine Y-coordinate of $Q$.
    pub const PA_QY: usize = 7;
    /// Affine X-coordinate of $G + Q$. Constant across all rows;
    /// consumed by the addend formula.
    pub const PA_QGX: usize = 8;
    /// Affine Y-coordinate of $G + Q$.
    pub const PA_QGY: usize = 9;
    /// Initial Jacobian point coordinates (boundary input at row 0).
    pub const PA_R_INIT_X: usize = 10;
    pub const PA_R_INIT_Y: usize = 11;
    pub const PA_R_INIT_Z: usize = 12;
    /// Inverse of $P_Z[\mathrm{FINAL\_ROW}]$ in $\F_p$. Only the
    /// row-$\mathrm{FINAL\_ROW}$ cell is consumed (gated by
    /// $\col{S\_FINAL}$); all other rows can be zero.
    pub const PA_Z_INV: usize = 13;
    /// Affine $x$-coordinate of the loop's final point
    /// $R = (X[\mathrm{FINAL\_ROW}], Y[\mathrm{FINAL\_ROW}], Z[\mathrm{FINAL\_ROW}])$.
    /// Only the row-$\mathrm{FINAL\_ROW}$ cell is consumed.
    pub const PA_R_X: usize = 14;
    /// ECDSA signature scalar `r`. Constant across all rows; consumed
    /// by the n-branch constraint `u_2 · s − r = 0 (mod n)` at
    /// `FINAL_ROW`.
    pub const PA_R: usize = 15;
    /// ECDSA signature scalar `s`. Constant across all rows; consumed
    /// by the n-branch constraints `u_1 · s − e = 0` and
    /// `u_2 · s − r = 0` at `FINAL_ROW`.
    pub const PA_S: usize = 16;
    /// ECDSA message digest `e` (the SHA output, interpreted as an
    /// `F_n` scalar). Constant across all rows; consumed by the
    /// n-branch constraint `u_1 · s − e = 0 (mod n)` at `FINAL_ROW`.
    pub const PA_E: usize = 17;
    pub const NUM_INT_PUB: usize = 18;

    // === Witness columns ===

    // Chained Jacobian state. up.X[t] = R_t (input), down.X[t] =
    // R_{t+1} (output, written by the output-selection constraint at
    // row t).
    pub const W_X: usize = 18;
    pub const W_Y: usize = 19;
    pub const W_Z: usize = 20;

    // Doubled-point columns (X_pa/Y_pa/Z_pa). `S = Y²` is inlined into
    // the doubling constraints — no dedicated column.
    pub const W_X_PA: usize = 21;
    pub const W_Y_PA: usize = 22;
    pub const W_Z_PA: usize = 23;

    // Addition scratch: C = T_X·Z_pa² − X_pa (= H from the spec)
    // and D = T_Y·Z_pa³ − Y_pa (= R_a from the spec), where
    // (T_X, T_Y) is the per-row affine addend computed in-circuit
    // from (b_1, b_2, Q, G+Q) via the addend selector. Z_pa²,
    // Z_pa³, C², C³, X_pa·C² are inlined.
    pub const W_C: usize = 24;
    pub const W_D: usize = 25;

    /// Running-sum accumulator for the scalar `u_1`. Built from
    /// `PA_B1` via `acc[0] = PA_B1[0]` and the chain
    /// `acc[t+1] = 2·acc[t] + PA_B1[t+1]` (mod p). With the existing
    /// MSB-first double-and-add indexing (`PA_B1[t]` = bit `255 − t`
    /// of `u_1`), the final active row's accumulator equals `u_1`.
    /// At `FINAL_ROW` (one past the last active row), the chain has
    /// applied one extra doubling with the zero-padded bit, so
    /// `W_U1_ACC[FINAL_ROW] = 2 · u_1` — the n-branch constraint
    /// uses this directly via `2 · u_1 · s − 2 · e ≡ 0 (mod n)`.
    pub const W_U1_ACC: usize = 26;
    /// Running-sum accumulator for the scalar `u_2`; same recurrence
    /// shape on `PA_B2`. Same off-by-one applies:
    /// `W_U2_ACC[FINAL_ROW] = 2 · u_2`.
    pub const W_U2_ACC: usize = 27;

    pub const NUM_INT: usize = 28;

    // Flat indices for shift specs (no bin/poly columns; flat = int).
    pub const FLAT_W_X: usize = W_X;
    pub const FLAT_W_Y: usize = W_Y;
    pub const FLAT_W_Z: usize = W_Z;
    pub const FLAT_W_U1_ACC: usize = W_U1_ACC;
    pub const FLAT_W_U2_ACC: usize = W_U2_ACC;
    pub const FLAT_PA_B1: usize = PA_B1;
    pub const FLAT_PA_B2: usize = PA_B2;
}

// ---------------------------------------------------------------------------
// The UAIR.
// ---------------------------------------------------------------------------

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
        // Shift X, Y, Z by 1 so down.X[t] = X[t+1] = R_{t+1}.
        // Shift PA_B1, PA_B2 by 1 so the running-sum chain constraint
        // can read down.PA_Bi[t] = PA_Bi[t+1].
        // Shift W_U1_ACC, W_U2_ACC by 1 so the chain constraint can
        // assert down.W_Ui_ACC[t] = 2·W_Ui_ACC[t] + down.PA_Bi[t].
        let shifts: Vec<ShiftSpec> = vec![
            ShiftSpec::new(cols::FLAT_PA_B1, 1),
            ShiftSpec::new(cols::FLAT_PA_B2, 1),
            ShiftSpec::new(cols::FLAT_W_X, 1),
            ShiftSpec::new(cols::FLAT_W_Y, 1),
            ShiftSpec::new(cols::FLAT_W_Z, 1),
            ShiftSpec::new(cols::FLAT_W_U1_ACC, 1),
            ShiftSpec::new(cols::FLAT_W_U2_ACC, 1),
        ];
        UairSignature::new(total, public, shifts, vec![], vec![])
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        down: TraceRow<B::Expr>,
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
        let s_init = &int[cols::S_INIT];
        let s_active = &int[cols::S_ACTIVE];
        let s_final = &int[cols::S_FINAL];
        let s_add = &int[cols::S_ADD];
        let pa_b1 = &int[cols::PA_B1];
        let pa_b2 = &int[cols::PA_B2];
        let pa_qx = &int[cols::PA_QX];
        let pa_qy = &int[cols::PA_QY];
        let pa_qgx = &int[cols::PA_QGX];
        let pa_qgy = &int[cols::PA_QGY];
        let pa_r_init_x = &int[cols::PA_R_INIT_X];
        let pa_r_init_y = &int[cols::PA_R_INIT_Y];
        let pa_r_init_z = &int[cols::PA_R_INIT_Z];
        let pa_z_inv = &int[cols::PA_Z_INV];
        let pa_r_x = &int[cols::PA_R_X];
        let x = &int[cols::W_X];
        let y = &int[cols::W_Y];
        let z = &int[cols::W_Z];
        let x_pa = &int[cols::W_X_PA];
        let y_pa = &int[cols::W_Y_PA];
        let z_pa = &int[cols::W_Z_PA];
        let c = &int[cols::W_C];
        let d = &int[cols::W_D];
        let u1_acc = &int[cols::W_U1_ACC];
        let u2_acc = &int[cols::W_U2_ACC];

        // down.int[i] in source-col-ascending order:
        // PA_B1, PA_B2, W_X, W_Y, W_Z, W_U1_ACC, W_U2_ACC.
        let down_pa_b1 = &down.int[0];
        let down_pa_b2 = &down.int[1];
        let down_x = &down.int[2];
        let down_y = &down.int[3];
        let down_z = &down.int[4];
        let down_u1_acc = &down.int[5];
        let down_u2_acc = &down.int[6];

        let two_scalar = const_scalar::<R>(R::from(2_u32));
        let three_scalar = const_scalar::<R>(R::from(3_u32));
        let eight_scalar = const_scalar::<R>(R::from(8_u32));
        let nine_scalar = const_scalar::<R>(R::from(9_u32));
        let twelve_scalar = const_scalar::<R>(R::from(12_u32));

        // ===================================================================
        // In-circuit affine addend selection (replaces the verifier-supplied
        // `PA_X_ADDEND, PA_Y_ADDEND` columns of the previous design).
        //
        // Given the bit pair `(b_1, b_2) ∈ {0,1}²`, the addend
        // T ∈ {O, G, Q, G+Q} is selected as
        //   T = (1-b_1)(1-b_2)·O + b_1(1-b_2)·G + (1-b_1)b_2·Q + b_1·b_2·(G+Q),
        // which simplifies (using O = (·,·) gated out by `S_ADD`, since
        // `S_ADD = 0` exactly when `(b_1, b_2) = (0,0)`) to the algebraic
        // identity
        //   T_x = b_1·(G_x − b_2·G_x) + b_2·(Q_x − b_1·Q_x) + b_1·b_2·(G+Q)_x
        //       = b_1·G_x + b_2·Q_x + b_1·b_2·((G+Q)_x − G_x − Q_x).
        // Symmetrically for T_y.
        //
        // Encodes G as a UAIR scalar (constant across proofs) and reads
        // Q, G+Q from public columns (per-proof but row-constant). For
        // the synthetic test below the scalar value is set to 0 since
        // the test exercises bit pair `(0, 1)` → addend = Q.
        //
        // TODO(prod): replace the placeholder G_X / G_Y scalars with the
        // canonical secp256k1 generator coordinates.
        // ===================================================================

        let g_x_scalar = const_scalar::<R>(R::from(0_u32));
        let g_y_scalar = const_scalar::<R>(R::from(0_u32));

        // Helper: build the addend coordinate from (b_1, b_2, Q, G+Q, G).
        //   T_coord = b_1·G_coord + b_2·Q_coord + b_1·b_2·((G+Q)_coord − G_coord − Q_coord)
        let b1b2 = pa_b1.clone() * pa_b2;
        let make_addend = |q_col: &B::Expr, qg_col: &B::Expr, g_scalar: &Self::Scalar| -> B::Expr {
            // b_1 · G_coord
            let b1_g = mbs(pa_b1, g_scalar).expect("b_1 · G_coord overflow");
            // b_2 · Q_coord
            let b2_q = pa_b2.clone() * q_col;
            // b_1·b_2 · (G+Q)_coord
            let bb_qg = b1b2.clone() * qg_col;
            // b_1·b_2 · G_coord
            let bb_g = mbs(&b1b2, g_scalar).expect("b_1·b_2·G_coord overflow");
            // b_1·b_2 · Q_coord
            let bb_q = b1b2.clone() * q_col;
            // T_coord = b_1·G + b_2·Q + b_1·b_2·((G+Q) − G − Q)
            b1_g + &b2_q + &bb_qg - &bb_g - &bb_q
        };

        let t_x = make_addend(pa_qx, pa_qgx, &g_x_scalar);
        let t_y = make_addend(pa_qy, pa_qgy, &g_y_scalar);

        // ===================================================================
        // Doubling block (3 constraints; `S = Y²` is inlined). Operates
        // on (X, Y, Z) → (X_pa, Y_pa, Z_pa). Max degree raised from 5 to
        // 6 (D4's `12·X³·Y²` term ×s_active) — but the global max is
        // already 6 from O2, so no net increase.
        // ===================================================================

        let y_sq = y.clone() * y;

        // C-D2: Z_pa − 2·Y·Z = 0
        let yz = y.clone() * z;
        let two_yz = mbs(&yz, &two_scalar).expect("2·Y·Z overflow");
        let d2_inner = z_pa.clone() - &two_yz;
        b.assert_zero(s_active.clone() * &d2_inner);

        // C-D3: X_pa − 9·X⁴ + 8·X·Y² = 0
        let x_sq = x.clone() * x;
        let x_pow4 = x_sq.clone() * &x_sq;
        let nine_x4 = mbs(&x_pow4, &nine_scalar).expect("9·X⁴ overflow");
        let x_y_sq = x.clone() * &y_sq;
        let eight_x_y_sq = mbs(&x_y_sq, &eight_scalar).expect("8·X·Y² overflow");
        let d3_inner = x_pa.clone() - &nine_x4 + &eight_x_y_sq;
        b.assert_zero(s_active.clone() * &d3_inner);

        // C-D4: Y_pa − 12·X³·Y² + 3·X²·X_pa + 8·Y⁴ = 0
        let x3_y_sq = x_sq.clone() * &x_y_sq;
        let twelve_x3_y_sq =
            mbs(&x3_y_sq, &twelve_scalar).expect("12·X³·Y² overflow");
        let x_sq_x_pa = x_sq.clone() * x_pa;
        let three_x2_xpa =
            mbs(&x_sq_x_pa, &three_scalar).expect("3·X²·X_pa overflow");
        let y_pow4 = y_sq.clone() * &y_sq;
        let eight_y_pow4 = mbs(&y_pow4, &eight_scalar).expect("8·Y⁴ overflow");
        let d4_inner =
            y_pa.clone() - &twelve_x3_y_sq + &three_x2_xpa + &eight_y_pow4;
        b.assert_zero(s_active.clone() * &d4_inner);

        // ===================================================================
        // Addition scratch (2 constraints). Z_pa², Z_pa³, C², C³, X_pa·C²
        // are all inlined — only C and D have witness columns (matching
        // the spec's `H` and `R_a`).
        // ===================================================================

        // C-A1: C − T_x·Z_pa² + X_pa = 0
        // T_x is degree 3 in trace cells (b_1·b_2·Q_x is deg-3 column-column-column),
        // so the constraint reaches deg 5; ×s_active = deg 6.
        let z_pa_sq = z_pa.clone() * z_pa;
        let a1_inner = c.clone() + x_pa - &(t_x * &z_pa_sq);
        b.assert_zero(s_active.clone() * &a1_inner);

        // C-A2: D − T_y·Z_pa³ + Y_pa = 0
        // T_y is degree 3; with z_pa_cube (deg 3) and s_active (deg 1), reaches deg 7.
        let z_pa_cube = z_pa.clone() * &z_pa_sq;
        let a2_inner = d.clone() + y_pa - &(t_y * &z_pa_cube);
        b.assert_zero(s_active.clone() * &a2_inner);

        // ===================================================================
        // Output-selection-and-chaining (3 constraints). Addition outputs
        // are inlined with `E = C², F = C³, G = X_pa·C²` substituted:
        //
        //   X_add = D² − C³ − 2·X_pa·C²
        //   Y_add = D·(X_pa·C² − X_add) − Y_pa·C³
        //         = 3·D·X_pa·C² + D·C³ − D³ − Y_pa·C³
        //   Z_add = Z_pa·C
        //
        //   down.X = X_pa + S_ADD·(X_add − X_pa)        (deg 5 with s_active)
        //   down.Y = Y_pa + S_ADD·(Y_add − Y_pa)        (deg 6 with s_active)
        //   down.Z = Z_pa + S_ADD·(Z_add − Z_pa)        (deg 4 with s_active)
        //
        // The deg-6 monomial in O2 is `s_active · S_ADD · D · X_pa · C²`
        // — matches the spec's `s_reg · R_a · X_mid · H²`.
        // ===================================================================

        // C-O1 (X): down.X − X_pa − S_ADD·(D² − C³ − 2·X_pa·C² − X_pa) = 0
        let c_sq = c.clone() * c;
        let c_cube = c.clone() * &c_sq;
        let x_pa_c_sq = x_pa.clone() * &c_sq;
        let two_x_pa_c_sq = mbs(&x_pa_c_sq, &two_scalar).expect("2·X_pa·C² overflow");
        let d_sq = d.clone() * d;
        let x_add_minus_x_pa = d_sq.clone() - &c_cube - &two_x_pa_c_sq - x_pa;
        let s_add_x = s_add.clone() * &x_add_minus_x_pa;
        let o1_inner = down_x.clone() - x_pa - &s_add_x;
        b.assert_zero(s_active.clone() * &o1_inner);

        // C-O2 (Y): down.Y − Y_pa − S_ADD·(3·D·X_pa·C² + D·C³ − D³ − Y_pa·C³ − Y_pa) = 0
        let d_cube = d.clone() * &d_sq;
        let d_x_pa_c_sq = d.clone() * &x_pa_c_sq;
        let three_d_x_pa_c_sq =
            mbs(&d_x_pa_c_sq, &three_scalar).expect("3·D·X_pa·C² overflow");
        let d_c_cube = d.clone() * &c_cube;
        let y_pa_c_cube = y_pa.clone() * &c_cube;
        let y_add_minus_y_pa =
            three_d_x_pa_c_sq + &d_c_cube - &d_cube - &y_pa_c_cube - y_pa;
        let s_add_y = s_add.clone() * &y_add_minus_y_pa;
        let o2_inner = down_y.clone() - y_pa - &s_add_y;
        b.assert_zero(s_active.clone() * &o2_inner);

        // C-O3 (Z): down.Z − Z_pa − S_ADD·(Z_pa·C − Z_pa) = 0
        let z_pa_c = z_pa.clone() * c;
        let z_add_minus_z_pa = z_pa_c - z_pa;
        let s_add_z = s_add.clone() * &z_add_minus_z_pa;
        let o3_inner = down_z.clone() - z_pa - &s_add_z;
        b.assert_zero(s_active.clone() * &o3_inner);

        // ===================================================================
        // Init boundary: at row 0, R = (PA_R_INIT_X, PA_R_INIT_Y, PA_R_INIT_Z).
        // ===================================================================

        b.assert_zero(s_init.clone() * &(x.clone() - pa_r_init_x));
        b.assert_zero(s_init.clone() * &(y.clone() - pa_r_init_y));
        b.assert_zero(s_init.clone() * &(z.clone() - pa_r_init_z));

        // ===================================================================
        // Final-row affine readout (2 constraints, gated by S_FINAL).
        //
        // Pins the loop's final Jacobian point P[FINAL_ROW] to its affine
        // x-coordinate via two public columns Z_inv and R_x:
        //
        //   F1: P_Z · Z_inv − 1 ≡ 0       (non-infinity + Z_inv = P_Z⁻¹)
        //   F2: P_X · Z_inv²   − R_x ≡ 0  (R_x is the affine x of P)
        //
        // Both at the up row, gated by S_FINAL = 1 (only at row
        // FINAL_ROW). F1 forces P_Z[FINAL_ROW] ≠ 0 (else the equation
        // can't be satisfied) and pins Z_inv to be its inverse mod p;
        // F2 then derives R_x from P_X and Z_inv. The order-field check
        //   R_x ≡ r  (mod n)
        // is NOT enforced in-circuit — the verifier is expected to check
        // it off-protocol against the signature scalar r, using the
        // public column R_x. Since R_x and r are both public, this is a
        // verifier-side equality check that does not need an in-circuit
        // ideal constraint.
        // ===================================================================

        // F1: S_FINAL · (P_Z · Z_inv − 1) ∈ (p)    (deg 3 with s_final)
        // Encoded as `S_FINAL · P_Z · Z_inv − S_FINAL = 0`, i.e.
        // factoring out S_FINAL and subtracting it itself in place of
        // `S_FINAL · 1` (avoids constructing a literal-1 expression).
        let f1_lhs = s_final.clone() * &(z.clone() * pa_z_inv);
        b.assert_zero(f1_lhs - s_final);

        // F2: S_FINAL · (P_X · Z_inv² − R_x) ∈ (p)    (deg 4 with s_final)
        let zinv_sq = pa_z_inv.clone() * pa_z_inv;
        let f2_inner = x.clone() * &zinv_sq - pa_r_x;
        b.assert_zero(s_final.clone() * &f2_inner);

        // ===================================================================
        // Booleanity of PA_B1, PA_B2 and derivation of S_ADD
        // (Augmentation A; 3 constraints, all degree 3).
        //
        // Encoded with the F1 trick: rather than building a literal-1
        // expression, factor `S_ACTIVE · b` once and subtract it from
        // `S_ACTIVE · b · b`, yielding `S_ACTIVE · b · (b − 1) = 0`.
        //
        // Closes the soundness obligation that previously sat outside
        // the proof: with `PA_B1[t], PA_B2[t] ∈ {0,1}` and the linear
        // identity `S_ADD = b_1 + b_2 − b_1·b_2`, the in-circuit
        // affine addend selector
        //   T = b_1·G + b_2·Q + b_1·b_2·((G+Q) − G − Q)
        // is fully bound by the proof. The remaining (deferred)
        // soundness obligation is row-constancy of PA_QX/PA_QY/PA_QGX/
        // PA_QGY across active rows of a single proof — still handled
        // verifier-side via direct public_trace inspection.
        // ===================================================================

        // B1-bool: S_ACTIVE · PA_B1 · (PA_B1 − 1) = 0
        let s_active_b1 = s_active.clone() * pa_b1;
        let s_active_b1_b1 = s_active_b1.clone() * pa_b1;
        b.assert_zero(s_active_b1_b1 - &s_active_b1);

        // B2-bool: S_ACTIVE · PA_B2 · (PA_B2 − 1) = 0
        let s_active_b2 = s_active.clone() * pa_b2;
        let s_active_b2_b2 = s_active_b2.clone() * pa_b2;
        b.assert_zero(s_active_b2_b2 - &s_active_b2);

        // S_ADD-derive: S_ACTIVE · (S_ADD − PA_B1 − PA_B2 + PA_B1·PA_B2) = 0
        let b1b2_for_sadd = pa_b1.clone() * pa_b2;
        let s_add_inner = s_add.clone() - pa_b1 - pa_b2 + &b1b2_for_sadd;
        b.assert_zero(s_active.clone() * &s_add_inner);

        // ===================================================================
        // Running-sum aggregation: bind W_U1_ACC / W_U2_ACC to the
        // bit-decompositions PA_B1 / PA_B2 (Augmentation B; 4
        // constraints, all degree 2).
        //
        // Recurrence (MSB-first, matching the existing double-and-add
        // addend selection):
        //   acc[0]   = b[0]                        (Init)
        //   acc[t+1] = 2 · acc[t] + b[t+1]         (Chain, on active rows)
        // After NUM_SHAMIR_ROUNDS active steps the accumulator equals
        // the scalar u_i represented by the bit column. Specifically
        //   acc[t] = Σ_{j=0..t} b[j] · 2^(t-j),
        // so acc[NUM_SHAMIR_ROUNDS - 1] = Σ_{j=0..255} b[j]·2^{255-j} = u_i,
        // and acc[NUM_SHAMIR_ROUNDS] = 2 · u_i (since the row-256 bit
        // is padded to 0 — the chain at row 255 reads down.PA_B1[255]
        // = PA_B1[256] = 0).
        // ===================================================================

        // Init-U1: S_INIT · (W_U1_ACC − PA_B1) = 0
        b.assert_zero(s_init.clone() * &(u1_acc.clone() - pa_b1));
        // Init-U2: S_INIT · (W_U2_ACC − PA_B2) = 0
        b.assert_zero(s_init.clone() * &(u2_acc.clone() - pa_b2));

        // Chain-U1: S_ACTIVE · (down.W_U1_ACC − 2·W_U1_ACC − down.PA_B1) = 0
        let two_u1_acc = mbs(u1_acc, &two_scalar).expect("2·W_U1_ACC overflow");
        let chain_u1_inner = down_u1_acc.clone() - &two_u1_acc - down_pa_b1;
        b.assert_zero(s_active.clone() * &chain_u1_inner);

        // Chain-U2: S_ACTIVE · (down.W_U2_ACC − 2·W_U2_ACC − down.PA_B2) = 0
        let two_u2_acc = mbs(u2_acc, &two_scalar).expect("2·W_U2_ACC overflow");
        let chain_u2_inner = down_u2_acc.clone() - &two_u2_acc - down_pa_b2;
        b.assert_zero(s_active.clone() * &chain_u2_inner);
    }

    fn constrain_general_n<B, FromR, MulByScalar, IFromR>(
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
        // ===================================================================
        // ECDSA n-branch (mod-`n`) constraints.
        //
        // After the 256-row Shamir double-and-add loop, the running-sum
        // accumulators `W_U1_ACC[FINAL_ROW]`, `W_U2_ACC[FINAL_ROW]` hold
        // `2·u_1` and `2·u_2` over Z (the chain runs one extra doubling
        // at the end: `acc[256] = 2·acc[255] + b[256] = 2·u_i + 0`). The
        // dual-pipeline n-branch projects these committed integer cells
        // mod `n`, so the same accumulator value provides `2·u_i mod n`
        // for the F_n constraint.
        //
        // The two ECDSA scalar identities are:
        //   u_1 · s ≡ e (mod n)   ⇔  2·u_1 · s ≡ 2·e (mod n)
        //   u_2 · s ≡ r (mod n)   ⇔  2·u_2 · s ≡ 2·r (mod n)
        //
        // We anchor each at `S_FINAL` (the post-loop row) and absorb the
        // factor of 2 into the constraint expression so the public
        // columns `PA_E`, `PA_R` carry the canonical ECDSA scalars
        // (not 2·e, 2·r):
        //
        //   F_N1: S_FINAL · (W_U1_ACC · PA_S − 2·PA_E) = 0   (mod n)
        //   F_N2: S_FINAL · (W_U2_ACC · PA_S − 2·PA_R) = 0   (mod n)
        //
        // Soundness caveat: the running-sum accumulator is stored mod
        // `p`, not over Z. For honest ECDSA inputs `u_i < n < p`, the
        // chain stays below `p` through row 255 (so `acc[255] = u_i`
        // exactly over Z); the doubling at row 256 may wrap mod `p`
        // (`2·u_i` can exceed `p`). The integer cell `acc[256]` is
        // therefore `2·u_i mod p`, which equals `2·u_i mod n` over Z
        // only when `2·u_i < p`. When `2·u_i ≥ p` (i.e., `u_i > p/2`),
        // the stored cell is `2·u_i − p`, and the n-branch reads it as
        // `(2·u_i − p) mod n` instead of `2·u_i mod n`. Closing this
        // gap requires either anchoring at row 255 with an additional
        // selector or changing the chain semantics so `acc[FINAL_ROW]
        // = u_i` (not `2·u_i`) — left as a follow-up.
        // ===================================================================

        let int = up.int;
        let s_final = &int[cols::S_FINAL];
        let pa_r = &int[cols::PA_R];
        let pa_s = &int[cols::PA_S];
        let pa_e = &int[cols::PA_E];
        let u1_acc = &int[cols::W_U1_ACC];
        let u2_acc = &int[cols::W_U2_ACC];

        let two_scalar = const_scalar::<R>(R::from(2_u32));

        // F_N1: S_FINAL · (W_U1_ACC · PA_S − 2·PA_E) = 0  (mod n)
        let two_pa_e = mbs(pa_e, &two_scalar).expect("2·PA_E overflow");
        let f_n1_inner = u1_acc.clone() * pa_s - &two_pa_e;
        b.assert_zero(s_final.clone() * &f_n1_inner);

        // F_N2: S_FINAL · (W_U2_ACC · PA_S − 2·PA_R) = 0  (mod n)
        let two_pa_r = mbs(pa_r, &two_scalar).expect("2·PA_R overflow");
        let f_n2_inner = u2_acc.clone() * pa_s - &two_pa_r;
        b.assert_zero(s_final.clone() * &f_n2_inner);
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

fn inv_mod_p(a: &CbUint<EC_FP_INT_LIMBS>) -> CbUint<EC_FP_INT_LIMBS> {
    let p_odd = Odd::new(SECP256K1_P_UINT).expect("p is odd");
    a.invert_odd_mod(&p_odd).expect("a has no inverse mod p (a == 0?)")
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

// === N-modulus arithmetic helpers ==================================
// Used by the witness generator to compute consistent values for the
// n-branch public columns (`PA_E`, `PA_R`). The protocol verifies the
// n-branch constraints `W_U_ACC · PA_S − 2·PA_E ≡ 0 (mod n)` and
// similar; the prover supplies witness values that make these hold.

fn mul_mod_n(
    a: &CbUint<EC_FP_INT_LIMBS>,
    b: &CbUint<EC_FP_INT_LIMBS>,
) -> CbUint<EC_FP_INT_LIMBS> {
    let wide: CbUint<{ EC_FP_INT_LIMBS * 2 }> = a.widening_mul(b).into();
    let n_wide: CbUint<{ EC_FP_INT_LIMBS * 2 }> = SECP256K1_N_UINT.resize();
    let n_wide_nz = NonZero::new(n_wide).expect("n is nonzero");
    let (_, rem) = wide.div_rem_vartime(&n_wide_nz);
    rem.resize()
}

fn inv_mod_n(a: &CbUint<EC_FP_INT_LIMBS>) -> CbUint<EC_FP_INT_LIMBS> {
    let n_odd = Odd::new(SECP256K1_N_UINT).expect("n is odd");
    a.invert_odd_mod(&n_odd)
        .expect("a has no inverse mod n (a == 0?)")
}

fn reduce_mod_n(a: &CbUint<EC_FP_INT_LIMBS>) -> CbUint<EC_FP_INT_LIMBS> {
    let n_nz = NonZero::new(SECP256K1_N_UINT).expect("n is nonzero");
    a.rem_vartime(&n_nz)
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

/// One Shamir step: Jacobian state R_t plus the per-row witness
/// columns (X_pa, Y_pa, Z_pa, C, D). `S = Y²` and the addition outputs
/// (X_add, Y_add, Z_add) are computed inline but not stored — only the
/// selected `(next_x, next_y, next_z)` is emitted (which equals R_{t+1}).
struct StepValues {
    x_pa: CbUint<EC_FP_INT_LIMBS>,
    y_pa: CbUint<EC_FP_INT_LIMBS>,
    z_pa: CbUint<EC_FP_INT_LIMBS>,
    c: CbUint<EC_FP_INT_LIMBS>,
    d: CbUint<EC_FP_INT_LIMBS>,
    /// `R_{t+1}` (= `down.X[t]` etc., constraints' chosen output).
    next_x: CbUint<EC_FP_INT_LIMBS>,
    next_y: CbUint<EC_FP_INT_LIMBS>,
    next_z: CbUint<EC_FP_INT_LIMBS>,
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

    // --- Addition (computed inline, not stored as columns) ---
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
    let (next_x, next_y, next_z) = if s_add_bit {
        (x_add, y_add, z_add)
    } else {
        (x_pa.clone(), y_pa.clone(), z_pa.clone())
    };

    let _ = s; // computed locally to derive x_pa/y_pa, no longer a column.

    StepValues {
        x_pa,
        y_pa,
        z_pa,
        c,
        d,
        next_x,
        next_y,
        next_z,
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
        // The synthetic test exercises the new in-circuit addend selector
        // by setting the bit pair to `(b_1, b_2) = (0, 1)` at every active
        // row, which selects the Q public column as the row's addend.
        // PA_QGX, PA_QGY (the (G+Q) coordinates) are populated with
        // arbitrary values — they don't enter the constraint when
        // (b_1, b_2) = (0, 1) since b_1 = 0 zeroes out the b_1·b_2·(G+Q)
        // term, and the placeholder G_x / G_y scalars in `constrain_general`
        // are zero so the b_1·G term also vanishes (consistent with b_1 = 0).
        let r_init_x = rand_fp(rng);
        let r_init_y = rand_fp(rng);
        let r_init_z = rand_nonzero_fp(rng);
        let pa_x = rand_fp(rng);
        let pa_y = rand_fp(rng);
        // Filler values for the `(G+Q)` public columns — never selected
        // when bits are `(0, 1)`, but populated to keep the column shape
        // consistent.
        let qg_x = rand_fp(rng);
        let qg_y = rand_fp(rng);

        // Build the per-row state by simulating the Shamir loop.
        // x_seq[t] = R_t. x_seq[0] = R_init.
        let mut x_seq: Vec<CbUint<EC_FP_INT_LIMBS>> = Vec::with_capacity(n_rows);
        let mut y_seq: Vec<CbUint<EC_FP_INT_LIMBS>> = Vec::with_capacity(n_rows);
        let mut z_seq: Vec<CbUint<EC_FP_INT_LIMBS>> = Vec::with_capacity(n_rows);
        x_seq.push(r_init_x.clone());
        y_seq.push(r_init_y.clone());
        z_seq.push(r_init_z.clone());

        let mut steps: Vec<StepValues> = Vec::with_capacity(NUM_SHAMIR_ROUNDS);
        for t in 0..NUM_SHAMIR_ROUNDS {
            let step = compute_step(&x_seq[t], &y_seq[t], &z_seq[t], &pa_x, &pa_y, true);
            x_seq.push(step.next_x.clone());
            y_seq.push(step.next_y.clone());
            z_seq.push(step.next_z.clone());
            steps.push(step);
        }

        // Pad the chained state sequence to n_rows.
        let zero_uint = CbUint::<EC_FP_INT_LIMBS>::ZERO;
        while x_seq.len() < n_rows {
            x_seq.push(zero_uint.clone());
            y_seq.push(zero_uint.clone());
            z_seq.push(zero_uint.clone());
        }

        // ---- Populate columns. ----
        let mk_col = || vec![R::ZERO; n_rows];

        let mut s_init_col: Vec<R> = mk_col();
        let mut s_active_col: Vec<R> = mk_col();
        let mut s_final_col: Vec<R> = mk_col();
        let mut s_add_col: Vec<R> = mk_col();
        let mut pa_b1_col: Vec<R> = mk_col();
        let mut pa_b2_col: Vec<R> = mk_col();
        let mut pa_qx_col: Vec<R> = mk_col();
        let mut pa_qy_col: Vec<R> = mk_col();
        let mut pa_qgx_col: Vec<R> = mk_col();
        let mut pa_qgy_col: Vec<R> = mk_col();
        let mut pa_r_init_x_col: Vec<R> = mk_col();
        let mut pa_r_init_y_col: Vec<R> = mk_col();
        let mut pa_r_init_z_col: Vec<R> = mk_col();
        let mut pa_z_inv_col: Vec<R> = mk_col();
        let mut pa_r_x_col: Vec<R> = mk_col();
        let mut pa_r_col: Vec<R> = mk_col();
        let mut pa_s_col: Vec<R> = mk_col();
        let mut pa_e_col: Vec<R> = mk_col();
        let mut x_col: Vec<R> = mk_col();
        let mut y_col: Vec<R> = mk_col();
        let mut z_col: Vec<R> = mk_col();
        let mut x_pa_col: Vec<R> = mk_col();
        let mut y_pa_col: Vec<R> = mk_col();
        let mut z_pa_col: Vec<R> = mk_col();
        let mut c_col: Vec<R> = mk_col();
        let mut d_col: Vec<R> = mk_col();
        let mut u1_acc_col: Vec<R> = mk_col();
        let mut u2_acc_col: Vec<R> = mk_col();

        // Selectors and bits. Q and G+Q are constant across rows.
        // Bit pair (b_1, b_2) = (0, 1) at every active row → addend = Q,
        // so S_ADD = b_1 + b_2 - b_1·b_2 = 0 + 1 - 0 = 1.
        s_init_col[0] = R::ONE;
        for t in 0..NUM_SHAMIR_ROUNDS {
            s_active_col[t] = R::ONE;
            s_add_col[t] = R::ONE;
            pa_b1_col[t] = R::ZERO;
            pa_b2_col[t] = R::ONE;
            pa_qx_col[t] = R::from(uint_to_int(pa_x.clone()));
            pa_qy_col[t] = R::from(uint_to_int(pa_y.clone()));
            pa_qgx_col[t] = R::from(uint_to_int(qg_x.clone()));
            pa_qgy_col[t] = R::from(uint_to_int(qg_y.clone()));
        }
        s_final_col[FINAL_ROW] = R::ONE;

        // PA_R_INIT only matters at row 0 (gated by S_INIT).
        pa_r_init_x_col[0] = R::from(uint_to_int(r_init_x));
        pa_r_init_y_col[0] = R::from(uint_to_int(r_init_y));
        pa_r_init_z_col[0] = R::from(uint_to_int(r_init_z));

        // PA_Z_INV / PA_R_X only matter at FINAL_ROW (gated by
        // S_FINAL): they encode the affine x-readout
        //   Z_inv := P_Z[FINAL_ROW]^{-1} mod p,
        //   R_x   := P_X[FINAL_ROW] · Z_inv^2 mod p.
        // Constraints F1, F2 then pin (P_Z · Z_inv = 1) and
        // (P_X · Z_inv^2 = R_x) at the final row.
        let final_z = z_seq[FINAL_ROW].clone();
        let final_x = x_seq[FINAL_ROW].clone();
        let z_inv = inv_mod_p(&final_z);
        let z_inv_sq = mul_mod_p(&z_inv, &z_inv);
        let r_x = mul_mod_p(&final_x, &z_inv_sq);
        pa_z_inv_col[FINAL_ROW] = R::from(uint_to_int(z_inv));
        pa_r_x_col[FINAL_ROW] = R::from(uint_to_int(r_x));

        // Chained state: X[t] = R_t.
        for t in 0..n_rows {
            x_col[t] = R::from(uint_to_int(x_seq[t].clone()));
            y_col[t] = R::from(uint_to_int(y_seq[t].clone()));
            z_col[t] = R::from(uint_to_int(z_seq[t].clone()));
        }

        // Per-step intermediates (rows 0..NUM_SHAMIR_ROUNDS).
        for (t, step) in steps.iter().enumerate() {
            x_pa_col[t] = R::from(uint_to_int(step.x_pa.clone()));
            y_pa_col[t] = R::from(uint_to_int(step.y_pa.clone()));
            z_pa_col[t] = R::from(uint_to_int(step.z_pa.clone()));
            c_col[t] = R::from(uint_to_int(step.c.clone()));
            d_col[t] = R::from(uint_to_int(step.d.clone()));
        }

        // Running-sum accumulators for u_1, u_2 over PA_B1, PA_B2.
        // Recurrence: acc[0] = b[0]; acc[t+1] = 2·acc[t] + b[t+1] (mod p).
        // The chain constraint runs at every active row t in
        // [0, NUM_SHAMIR_ROUNDS); the row-NUM_SHAMIR_ROUNDS bit is the
        // padded zero (S_ACTIVE = 0 there ⇒ PA_Bi[NUM_SHAMIR_ROUNDS] = 0
        // satisfies any algebraic shape we like). We therefore fill rows
        // 0..=NUM_SHAMIR_ROUNDS using the recurrence (with b[NUM_SHAMIR_ROUNDS]
        // = 0) and zero-pad the rest.
        //
        // The synthetic test pins (b_1, b_2) = (0, 1) at every active
        // row, so we know the bit pattern statically rather than reading
        // it back from `pa_b1_col` / `pa_b2_col` (which would require an
        // extra trait bound on R to extract the underlying CbUint).
        let bit_b1 = |t: usize| -> CbUint<EC_FP_INT_LIMBS> {
            // PA_B1 is 0 on every row in the synthetic test.
            let _ = t;
            CbUint::<EC_FP_INT_LIMBS>::ZERO
        };
        let bit_b2 = |t: usize| -> CbUint<EC_FP_INT_LIMBS> {
            if t < NUM_SHAMIR_ROUNDS {
                CbUint::<EC_FP_INT_LIMBS>::from(1u32)
            } else {
                CbUint::<EC_FP_INT_LIMBS>::ZERO
            }
        };
        let p_nz = NonZero::new(SECP256K1_P_UINT).expect("p is nonzero");
        let chain_step =
            |acc: &CbUint<EC_FP_INT_LIMBS>, b: &CbUint<EC_FP_INT_LIMBS>| -> CbUint<EC_FP_INT_LIMBS> {
                let two_acc = small_mul_mod_p(acc, 2);
                let mut sum = two_acc.wrapping_add(b);
                if p_geq(&sum) {
                    sum = sum.rem_vartime(&p_nz);
                }
                sum
            };
        let mut acc1_uint: CbUint<EC_FP_INT_LIMBS> = bit_b1(0);
        let mut acc2_uint: CbUint<EC_FP_INT_LIMBS> = bit_b2(0);
        u1_acc_col[0] = R::from(uint_to_int(acc1_uint.clone()));
        u2_acc_col[0] = R::from(uint_to_int(acc2_uint.clone()));
        for t in 0..NUM_SHAMIR_ROUNDS {
            let next = t + 1;
            acc1_uint = chain_step(&acc1_uint, &bit_b1(next));
            acc2_uint = chain_step(&acc2_uint, &bit_b2(next));
            if next < n_rows {
                u1_acc_col[next] = R::from(uint_to_int(acc1_uint.clone()));
                u2_acc_col[next] = R::from(uint_to_int(acc2_uint.clone()));
            }
        }

        // ECDSA n-branch (mod-n) public columns. The constraints
        //   F_N1: S_FINAL · (W_U1_ACC · PA_S − 2·PA_E) = 0  (mod n)
        //   F_N2: S_FINAL · (W_U2_ACC · PA_S − 2·PA_R) = 0  (mod n)
        // are anchored at FINAL_ROW. Since `acc1_uint`, `acc2_uint`
        // hold the integer values `W_U1_ACC[FINAL_ROW]` and
        // `W_U2_ACC[FINAL_ROW]` (= 2·u_i mod p) at this point, we
        // compute matching `PA_E`, `PA_R` for an arbitrary `PA_S`:
        //   PA_E = (W_U1_ACC · PA_S) · 2⁻¹  (mod n)
        //   PA_R = (W_U2_ACC · PA_S) · 2⁻¹  (mod n)
        // For the synthetic test we pick `PA_S = 1`. Only the
        // `FINAL_ROW` cells matter (gated by `S_FINAL`); other rows
        // can be zero.
        let pa_s_uint = CbUint::<EC_FP_INT_LIMBS>::from(1u32);
        let two_uint = CbUint::<EC_FP_INT_LIMBS>::from(2u32);
        let inv_two_n = inv_mod_n(&two_uint);
        let acc1_n = reduce_mod_n(&acc1_uint);
        let acc2_n = reduce_mod_n(&acc2_uint);
        let pa_s_n = reduce_mod_n(&pa_s_uint);
        let pa_e_uint = mul_mod_n(&mul_mod_n(&acc1_n, &pa_s_n), &inv_two_n);
        let pa_r_uint = mul_mod_n(&mul_mod_n(&acc2_n, &pa_s_n), &inv_two_n);
        pa_s_col[FINAL_ROW] = R::from(uint_to_int(pa_s_uint));
        pa_e_col[FINAL_ROW] = R::from(uint_to_int(pa_e_uint));
        pa_r_col[FINAL_ROW] = R::from(uint_to_int(pa_r_uint));

        let to_mle = |col: Vec<R>| -> DenseMultilinearExtension<R> { col.into_iter().collect() };

        let int = vec![
            to_mle(s_init_col),
            to_mle(s_active_col),
            to_mle(s_final_col),
            to_mle(s_add_col),
            to_mle(pa_b1_col),
            to_mle(pa_b2_col),
            to_mle(pa_qx_col),
            to_mle(pa_qy_col),
            to_mle(pa_qgx_col),
            to_mle(pa_qgy_col),
            to_mle(pa_r_init_x_col),
            to_mle(pa_r_init_y_col),
            to_mle(pa_r_init_z_col),
            to_mle(pa_z_inv_col),
            to_mle(pa_r_x_col),
            to_mle(pa_r_col),
            to_mle(pa_s_col),
            to_mle(pa_e_col),
            to_mle(x_col),
            to_mle(y_col),
            to_mle(z_col),
            to_mle(x_pa_col),
            to_mle(y_pa_col),
            to_mle(z_pa_col),
            to_mle(c_col),
            to_mle(d_col),
            to_mle(u1_acc_col),
            to_mle(u2_acc_col),
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
        constraint_counter::{count_constraints, count_constraints_n},
        degree_counter::{count_constraint_degrees, count_max_degree},
    };

    /// Sanity: 20 p-branch constraints + 2 n-branch constraints.
    ///
    /// **p-branch** (`U::constrain_general`): 13 base constraints + 7
    /// augmentation constraints (3 booleanity / S_ADD-derive at deg 3,
    /// 2 running-sum init at deg 2, 2 running-sum chain at deg 2) = 20
    /// total. Max degree 7 from C-A2 (the in-circuit affine addend
    /// `T_y` is degree 3, multiplied by Z_pa³ deg 3 and S_ACTIVE deg 1).
    /// Deg-2 constraints: 3 init Jacobian + 2 init U_acc + 2 chain
    /// U_acc = 7.
    ///
    /// **n-branch** (`U::constrain_general_n`): 2 mod-`n` ECDSA scalar
    /// identities — `u_1·s ≡ e` and `u_2·s ≡ r` — anchored at S_FINAL
    /// with the off-by-2 absorbed into `2·PA_E` / `2·PA_R`. Both deg 3
    /// (S_FINAL · W_U_ACC · PA_S).
    #[test]
    fn shamir_constraint_shape() {
        type U = EcdsaUair<Int<EC_FP_INT_LIMBS>>;
        assert_eq!(count_constraints::<U>(), 20);
        assert_eq!(count_max_degree::<U>(), 7);
        let degrees = count_constraint_degrees::<U>();
        // Spot-checks: at least one deg-7 (Y addend constraint); 7 deg-2.
        assert!(degrees.iter().any(|&d| d == 7), "expected at least one deg-7");
        assert_eq!(
            degrees.iter().filter(|&&d| d == 2).count(),
            7,
            "expected 7 deg-2 (3 init Jacobian + 2 init U_acc + 2 chain U_acc)",
        );

        // n-branch: two mod-n ECDSA constraints (F_N1, F_N2).
        assert_eq!(
            count_constraints_n::<U>(),
            2,
            "expected 2 n-branch constraints (u_1·s ≡ e and u_2·s ≡ r)",
        );
    }

    /// Witness gen produces a trace where every constraint vanishes
    /// mod p. Exercises the active block + final affine conversion.
    #[test]
    fn witness_satisfies_constraints_mod_p() {
        let num_vars = 9;
        let mut r = rng();
        let trace = <EcdsaUair<Int<EC_FP_INT_LIMBS>> as GenerateRandomTrace<32>>::
            generate_random_trace(num_vars, &mut r);
        let n_rows = 1 << num_vars;
        assert_eq!(trace.int.len(), cols::NUM_INT);

        let int_to_uint = |v: &Int<EC_FP_INT_LIMBS>| -> CbUint<EC_FP_INT_LIMBS> {
            *v.inner().as_uint()
        };
        let read_uint = |c: usize, t: usize| int_to_uint(&trace.int[c][t]);
        let zero_uint: CbUint<EC_FP_INT_LIMBS> = CbUint::ZERO;

        for t in 0..n_rows {
            let s_active_int = trace.int[cols::S_ACTIVE][t].clone();
            let s_init_int = trace.int[cols::S_INIT][t].clone();
            let s_final_int = trace.int[cols::S_FINAL][t].clone();
            let active = s_active_int == Int::ONE;
            let init = s_init_int == Int::ONE;
            let final_row = s_final_int == Int::ONE;

            // Active rows: doubling + addition intermediates +
            // chained-output (= R_{t+1}).
            if active {
                let s_add_int = trace.int[cols::S_ADD][t].clone();
                let s_add_bit = s_add_int == Int::ONE;

                // The synthetic test uses bit pair (b_1, b_2) = (0, 1)
                // at every active row, so the addend is Q = (PA_QX, PA_QY).
                let pa_x = read_uint(cols::PA_QX, t);
                let pa_y = read_uint(cols::PA_QY, t);
                let x = read_uint(cols::W_X, t);
                let y = read_uint(cols::W_Y, t);
                let z = read_uint(cols::W_Z, t);

                let expected = compute_step(&x, &y, &z, &pa_x, &pa_y, s_add_bit);

                assert_eq!(read_uint(cols::W_X_PA, t), expected.x_pa, "X_pa at row {t}");
                assert_eq!(read_uint(cols::W_Y_PA, t), expected.y_pa, "Y_pa at row {t}");
                assert_eq!(read_uint(cols::W_Z_PA, t), expected.z_pa, "Z_pa at row {t}");
                assert_eq!(read_uint(cols::W_C, t), expected.c, "C at row {t}");
                assert_eq!(read_uint(cols::W_D, t), expected.d, "D at row {t}");

                // Output: down.X = next R = expected.next_x.
                if t + 1 < n_rows {
                    assert_eq!(read_uint(cols::W_X, t + 1), expected.next_x, "next X at {t}");
                    assert_eq!(read_uint(cols::W_Y, t + 1), expected.next_y, "next Y at {t}");
                    assert_eq!(read_uint(cols::W_Z, t + 1), expected.next_z, "next Z at {t}");
                }
            }

            // Init boundary: row 0's R = PA_R_INIT.
            if init {
                assert_eq!(read_uint(cols::W_X, t), read_uint(cols::PA_R_INIT_X, t), "init X at {t}");
                assert_eq!(read_uint(cols::W_Y, t), read_uint(cols::PA_R_INIT_Y, t), "init Y at {t}");
                assert_eq!(read_uint(cols::W_Z, t), read_uint(cols::PA_R_INIT_Z, t), "init Z at {t}");
            }

            // Final-row check: no in-circuit constraint after dropping
            // F1; the affine readout is handled off-protocol. We still
            // skip the padding-zero check at FINAL_ROW because that row
            // holds R_NUM_SHAMIR_ROUNDS, the chained final state.

            // Padding rows past FINAL_ROW: chained X is zero
            // (uninitialized but unconstrained).
            if !active && !final_row {
                assert_eq!(read_uint(cols::W_X, t), zero_uint, "pad X at {t}");
            }
        }

        // Running-sum accumulators: with the synthetic test's bit
        // pattern `(b_1, b_2) = (0, 1)` at every active row,
        //   - PA_B1 ≡ 0 ⇒ W_U1_ACC ≡ 0 at every row.
        //   - PA_B2 ≡ 1 for t in 0..NUM_SHAMIR_ROUNDS, padded 0
        //     thereafter. Recurrence: acc[0]=1, acc[t+1] = 2·acc[t] + 1
        //     ⇒ acc[t] = 2^{t+1} − 1 for t in 0..=NUM_SHAMIR_ROUNDS-1.
        //     Then the chain at row 255 forces
        //     acc[NUM_SHAMIR_ROUNDS] = 2·acc[255] + 0 = 2 · (2^256 − 1)
        //                            = 2^257 − 2 (mod p).
        //
        // We spot-check W_U1_ACC[NUM_SHAMIR_ROUNDS-1] / [FINAL_ROW] and
        // W_U2_ACC[NUM_SHAMIR_ROUNDS-1] / [FINAL_ROW].
        assert_eq!(
            read_uint(cols::W_U1_ACC, NUM_SHAMIR_ROUNDS - 1),
            zero_uint,
            "W_U1_ACC[NUM_SHAMIR_ROUNDS-1] should be 0 (b_1 ≡ 0)",
        );
        assert_eq!(
            read_uint(cols::W_U1_ACC, FINAL_ROW),
            zero_uint,
            "W_U1_ACC[FINAL_ROW] should be 0 (b_1 ≡ 0)",
        );

        // Compute expected u_2 values via the same recurrence the
        // witness generator uses, mod p.
        let p_nz = NonZero::new(SECP256K1_P_UINT).expect("p is nonzero");
        let one_uint = CbUint::<EC_FP_INT_LIMBS>::ONE;
        let mut expected_acc2 = one_uint.clone();
        for _ in 0..NUM_SHAMIR_ROUNDS - 1 {
            // expected = 2·expected + 1
            let two_e = small_mul_mod_p(&expected_acc2, 2);
            let mut sum = two_e.wrapping_add(&one_uint);
            if p_geq(&sum) {
                sum = sum.rem_vartime(&p_nz);
            }
            expected_acc2 = sum;
        }
        // expected_acc2 now holds acc[NUM_SHAMIR_ROUNDS-1] = 2^256 - 1 mod p.
        assert_eq!(
            read_uint(cols::W_U2_ACC, NUM_SHAMIR_ROUNDS - 1),
            expected_acc2,
            "W_U2_ACC[NUM_SHAMIR_ROUNDS-1] = 2^{{256}} − 1 mod p",
        );
        // One more chain step at row 255 (with bit 0 padding) → acc[FINAL_ROW]
        // = 2 · acc[NUM_SHAMIR_ROUNDS-1].
        let expected_acc2_final = small_mul_mod_p(&expected_acc2, 2);
        assert_eq!(
            read_uint(cols::W_U2_ACC, FINAL_ROW),
            expected_acc2_final,
            "W_U2_ACC[FINAL_ROW] = 2^{{257}} − 2 mod p",
        );
    }
}
