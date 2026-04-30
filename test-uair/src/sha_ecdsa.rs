//! Combined SHA-256 + ECDSA UAIR (side-by-side merge).
//!
//! Runs both `Sha256CompressionSliceUair` and `EcdsaUair` inside a
//! single UAIR on disjoint columns of the same trace. One proof
//! attests to both the SHA-256 compression round-trip **and** the
//! ECDSA Shamir scalar multiplication (doubling + addition + Jacobian
//! → affine output).
//!
//! ## What this is (and isn't)
//!
//! Structural side-by-side merge: both sub-UAIRs' constraints live on
//! disjoint column ranges of one trace. There is **no** in-circuit
//! cross-binding constraint between SHA's digest output and ECDSA's
//! addend / R_init inputs — the binding is implicit in the verifier's
//! choice of public columns: the verifier reads SHA's `pa_e` digest
//! out-of-band, computes `u_1 = e · s⁻¹ mod n`, derives the bit
//! pattern, and writes the corresponding `(PA_X_ADDEND, PA_Y_ADDEND)`
//! per row. The merged proof attests internal consistency of each
//! half; the *cross-half consistency* is enforced by the verifier
//! supplying coherent public inputs.
//!
//! ## Column layout
//!
//! Flat trace = `binary_poly || arbitrary_poly || int` (no
//! arbitrary_poly columns).
//!
//! **binary_poly section** (23 cols, mirrors SHA standalone):
//! - `[0..6]` public: `PA_A`, `PA_E`, `PA_OV_SIG0/SIG1/LSIG0/LSIG1`
//!   (rotation-ideal overflow witnesses, made public)
//! - `[6..23]` witness: SHA's remaining witness bit-polys, including
//!   the Ch operand split (`u_ef`, `u_{¬e,g}`) that replaces the old
//!   `Ch` column and the three Table 9 affine-combination
//!   materializations (`B_1`, `B_2`, `B_3`).
//!
//! **int section** (27 cols total, 16 pubs + 11 witness):
//! - `[0..6]` public: SHA pubs (S_INIT, S_FINAL, PA_K, PA_C_C7/8/9)
//! - `[6..15]` public: ECDSA pubs (S_INIT, S_ACTIVE, S_FINAL, S_ADD,
//!   PA_X_ADDEND, PA_Y_ADDEND, PA_R_INIT_X/Y/Z)
//! - `[15]` public: SHA `S_B_ACTIVE` selector for the Table 9
//!   materialization constraints
//! - `[16..19]` witness: SHA `mu_W, mu_a, mu_e`
//! - `[19..27]` witness: ECDSA chained Jacobian state + doubled point
//!   + addition scratch (8 cols; `S = Y²` and `Z_inv` are inlined or
//!   handled off-protocol)
//!
//! Both halves' shifts and lookup specs are unioned. Lookup groups
//! come from the SHA half only (ECDSA has no range-checked carries
//! in the no-quotient F_p formulation).
//!
//! ## Selectors and trace length
//!
//! Both halves use row-0 init and end-of-trace final selectors on
//! disjoint columns. Trace length is bounded by ECDSA: needs >
//! 256 (`FINAL_ROW = NUM_SHAMIR_ROUNDS = 256`), so `num_vars >= 9`.
//! SHA needs >= 16 rows; satisfied.
//!
//! ## Quotient-witness convention
//!
//! ECDSA F_p constraints are direct (no quotients) — the proving
//! field is the secp256k1 base prime. SHA uses public **compensator**
//! columns (`PA_C_C7/8/9`) plus integer-carry witness columns
//! (`mu_W/mu_a/mu_e`) range-checked via lookup. SHA's quotient-like
//! columns are **public** (compensators) or **lookup-checked**
//! (carries); ECDSA has no quotient columns at all.

use core::marker::PhantomData;

use crypto_primitives::ConstSemiring;
use rand::RngCore;
use zinc_poly::{
    mle::DenseMultilinearExtension,
    univariate::dense::DensePolynomial,
};
use zinc_uair::{
    BitOp, BitOpSpec, ConstraintBuilder, LookupColumnSpec, PublicColumnLayout,
    ShiftSpec, TotalColumnLayout, TraceRow, Uair, UairSignature, UairTrace,
    ideal::rotation::RotationIdeal,
};

use crate::{
    GenerateRandomTrace,
    ecdsa::{self, FINAL_ROW as ECDSA_FINAL_ROW, NUM_SHAMIR_ROUNDS},
    ecdsa_doubling::{EC_FP_INT_LIMBS, EcdsaFpRing},
    sha256::{self, Sha256CompressionSliceUair, Sha256Ideal},
};

use crypto_primitives::crypto_bigint_int::Int;

// Re-export for convenience.
pub use crate::ecdsa::FINAL_ROW;

// ---------------------------------------------------------------------------
// Column layout for the merged trace.
// ---------------------------------------------------------------------------

pub mod cols {
    // ===== binary_poly (mirrors sha256.rs: 6 pub + 17 witness) =====
    // OV cols are public — overflow witnesses for the rotation-ideal
    // constraints (C1, C2, C4, C6) are verifier-derivable.
    pub const PA_A: usize = 0;
    pub const PA_E: usize = 1;
    pub const PA_OV_SIG0: usize = 2;
    pub const PA_OV_SIG1: usize = 3;
    pub const PA_OV_LSIG0: usize = 4;
    pub const PA_OV_LSIG1: usize = 5;
    pub const W_A: usize = 6;
    pub const W_SIG0: usize = 7;
    pub const W_E: usize = 8;
    pub const W_SIG1: usize = 9;
    pub const W_W: usize = 10;
    pub const W_LSIG0: usize = 11;
    pub const W_LSIG1: usize = 12;
    // Ch is split into two AND-operand bit-polys (see sha256.rs doc).
    pub const W_U_EF: usize = 13;
    pub const W_U_NEG_E_G: usize = 14;
    pub const W_MAJ: usize = 15;
    // Table 9 affine-combination materializations.
    pub const W_B1: usize = 16;
    pub const W_B2: usize = 17;
    pub const W_B3: usize = 18;
    pub const NUM_BIN: usize = 19;
    pub const NUM_BIN_PUB: usize = 6;

    // ===== int section =====
    // SHA publics (0..8) — see sha256.rs cols module for chained-
    // compression layout details. S_INIT_PREFIX/S_FEEDFORWARD replace
    // the old S_INIT/S_FINAL pair. PA_C_FF_A/E are the feed-forward
    // compensators (added so C12/C13 stay degree-1 in the trace MLEs,
    // preserving MLE-first eligibility — see sha256.rs cols doc).
    pub const SHA_S_INIT_PREFIX: usize = 0;
    pub const SHA_S_FEEDFORWARD: usize = 1;
    pub const SHA_PA_K: usize = 2;
    pub const SHA_PA_C_C7: usize = 3;
    pub const SHA_PA_C_C8: usize = 4;
    pub const SHA_PA_C_C9: usize = 5;
    pub const SHA_PA_C_FF_A: usize = 6;
    pub const SHA_PA_C_FF_E: usize = 7;
    // ECDSA publics (8..17)
    pub const ECDSA_S_INIT: usize = 8;
    pub const ECDSA_S_ACTIVE: usize = 9;
    pub const ECDSA_S_FINAL: usize = 10;
    pub const ECDSA_S_ADD: usize = 11;
    pub const ECDSA_PA_X_ADDEND: usize = 12;
    pub const ECDSA_PA_Y_ADDEND: usize = 13;
    pub const ECDSA_PA_R_INIT_X: usize = 14;
    pub const ECDSA_PA_R_INIT_Y: usize = 15;
    pub const ECDSA_PA_R_INIT_Z: usize = 16;
    // SHA selector for the Table 9 materialization constraints.
    pub const SHA_S_B_ACTIVE: usize = 17;
    pub const NUM_INT_PUB: usize = 18;

    // SHA witnesses (18..21) — carry-range columns.
    pub const SHA_W_MU_W: usize = 18;
    pub const SHA_W_MU_A: usize = 19;
    pub const SHA_W_MU_E: usize = 20;

    // ECDSA witnesses (21..29): chained Jacobian state + doubled
    // point + addition scratch. 8 cols; `S = Y²` and `Z_inv` are
    // inlined / off-protocol respectively.
    pub const ECDSA_W_X: usize = 21;
    pub const ECDSA_W_Y: usize = 22;
    pub const ECDSA_W_Z: usize = 23;
    pub const ECDSA_W_X_PA: usize = 24;
    pub const ECDSA_W_Y_PA: usize = 25;
    pub const ECDSA_W_Z_PA: usize = 26;
    pub const ECDSA_W_C: usize = 27;
    pub const ECDSA_W_D: usize = 28;

    // SHA feed-forward carry witnesses (mu_junction_a, mu_junction_e),
    // appended after the ECDSA witnesses. Each is in {0, 1} on junction
    // rows and 0 elsewhere.
    pub const SHA_W_MU_JUNCTION_A: usize = 29;
    pub const SHA_W_MU_JUNCTION_E: usize = 30;

    pub const NUM_INT: usize = 31;

    // Flat indices (binary_poly || arbitrary_poly || int).
    pub const FLAT_W_A: usize = W_A;
    pub const FLAT_W_SIG0: usize = W_SIG0;
    pub const FLAT_W_E: usize = W_E;
    pub const FLAT_W_SIG1: usize = W_SIG1;
    pub const FLAT_W_W: usize = W_W;
    pub const FLAT_W_LSIG0: usize = W_LSIG0;
    pub const FLAT_W_LSIG1: usize = W_LSIG1;
    pub const FLAT_W_U_EF: usize = W_U_EF;
    pub const FLAT_W_U_NEG_E_G: usize = W_U_NEG_E_G;
    pub const FLAT_W_MAJ: usize = W_MAJ;
    pub const FLAT_W_B1: usize = W_B1;
    pub const FLAT_W_B2: usize = W_B2;
    pub const FLAT_W_B3: usize = W_B3;
    pub const FLAT_SHA_PA_K: usize = NUM_BIN + SHA_PA_K;
    pub const FLAT_SHA_W_MU_W: usize = NUM_BIN + SHA_W_MU_W;
    pub const FLAT_SHA_W_MU_A: usize = NUM_BIN + SHA_W_MU_A;
    pub const FLAT_SHA_W_MU_E: usize = NUM_BIN + SHA_W_MU_E;
    pub const FLAT_ECDSA_W_X: usize = NUM_BIN + ECDSA_W_X;
    pub const FLAT_ECDSA_W_Y: usize = NUM_BIN + ECDSA_W_Y;
    pub const FLAT_ECDSA_W_Z: usize = NUM_BIN + ECDSA_W_Z;
}

// ---------------------------------------------------------------------------
// The merged UAIR.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ShaEcdsaUair<R>(PhantomData<R>);

impl<R> Uair for ShaEcdsaUair<R>
where
    R: EcdsaFpRing + From<u32>,
{
    type Ideal = Sha256Ideal<R>;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(cols::NUM_BIN, 0, cols::NUM_INT);
        let public = PublicColumnLayout::new(cols::NUM_BIN_PUB, 0, cols::NUM_INT_PUB);

        // Shifts: union of SHA's and ECDSA's (sorted by source_col by
        // UairSignature::new; insertion order breaks ties — list within
        // a source_col in shift_amount order to mirror sha256.rs).
        let shifts: Vec<ShiftSpec> = vec![
            // === SHA binary_poly shifts ===
            ShiftSpec::new(cols::FLAT_W_A, 1),
            ShiftSpec::new(cols::FLAT_W_A, 2),
            ShiftSpec::new(cols::FLAT_W_A, 4),
            ShiftSpec::new(cols::FLAT_W_SIG0, 3),
            ShiftSpec::new(cols::FLAT_W_E, 1),
            ShiftSpec::new(cols::FLAT_W_E, 2),
            ShiftSpec::new(cols::FLAT_W_E, 4),
            ShiftSpec::new(cols::FLAT_W_SIG1, 3),
            ShiftSpec::new(cols::FLAT_W_W, 3),
            ShiftSpec::new(cols::FLAT_W_W, 9),
            ShiftSpec::new(cols::FLAT_W_W, 16),
            ShiftSpec::new(cols::FLAT_W_LSIG0, 1),
            ShiftSpec::new(cols::FLAT_W_LSIG1, 14),
            ShiftSpec::new(cols::FLAT_W_U_EF, 2),
            ShiftSpec::new(cols::FLAT_W_U_EF, 3),
            ShiftSpec::new(cols::FLAT_W_U_NEG_E_G, 2),
            ShiftSpec::new(cols::FLAT_W_U_NEG_E_G, 3),
            ShiftSpec::new(cols::FLAT_W_MAJ, 2),
            ShiftSpec::new(cols::FLAT_W_MAJ, 3),
            ShiftSpec::new(cols::FLAT_W_B1, 2),
            ShiftSpec::new(cols::FLAT_W_B2, 2),
            ShiftSpec::new(cols::FLAT_W_B3, 2),
            // === SHA int shifts (PA_K, mu_W, mu_a, mu_e) ===
            ShiftSpec::new(cols::FLAT_SHA_PA_K, 3),
            ShiftSpec::new(cols::FLAT_SHA_W_MU_W, 16),
            ShiftSpec::new(cols::FLAT_SHA_W_MU_A, 3),
            ShiftSpec::new(cols::FLAT_SHA_W_MU_E, 3),
            // === ECDSA int shifts (X, Y, Z by 1 each: down.X[t] = R_{t+1}) ===
            ShiftSpec::new(cols::FLAT_ECDSA_W_X, 1),
            ShiftSpec::new(cols::FLAT_ECDSA_W_Y, 1),
            ShiftSpec::new(cols::FLAT_ECDSA_W_Z, 1),
        ];

        // No explicit lookup specs — booleanity covers all witness
        // binary_poly columns for free. See sha256.rs for the
        // discussion.
        let lookup_specs: Vec<LookupColumnSpec> = Vec::new();
        // Bit-op virtual columns over W for σ_0/σ_1. Mirrors the SHA
        // standalone UAIR — see `sha256::Sha256CompressionSliceUair::signature`
        // for the full mapping (`Rot(c)` ≡ `ROTR^{32-c}` ≡ multiplication
        // by `X^c mod (X^32 − 1)`). All six specs target FLAT_W_W.
        let bit_op_specs: Vec<BitOpSpec> = vec![
            BitOpSpec::new(cols::FLAT_W_W, BitOp::Rot(25)),    // σ_0: ROTR^7
            BitOpSpec::new(cols::FLAT_W_W, BitOp::Rot(14)),    // σ_0: ROTR^18
            BitOpSpec::new(cols::FLAT_W_W, BitOp::ShiftR(3)),  // σ_0: SHR^3
            BitOpSpec::new(cols::FLAT_W_W, BitOp::Rot(15)),    // σ_1: ROTR^17
            BitOpSpec::new(cols::FLAT_W_W, BitOp::Rot(13)),    // σ_1: ROTR^19
            BitOpSpec::new(cols::FLAT_W_W, BitOp::ShiftR(10)), // σ_1: SHR^10
        ];

        UairSignature::new(total, public, shifts, lookup_specs, bit_op_specs)
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        down: TraceRow<B::Expr>,
        from_ref: FromR,
        mbs: MulByScalar,
        ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&Self::Scalar) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &Self::Scalar) -> Option<B::Expr>,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
    {
        // ===================================================================
        // SHA-256 half — mirrors sha256.rs's constrain_general,
        // referencing merged column indices.
        // ===================================================================
        let bp = up.binary_poly;
        let int = up.int;

        let pa_a = &bp[cols::PA_A];
        let pa_e = &bp[cols::PA_E];
        let pa_ov_sig0 = &bp[cols::PA_OV_SIG0];
        let pa_ov_sig1 = &bp[cols::PA_OV_SIG1];
        let pa_ov_lsig0 = &bp[cols::PA_OV_LSIG0];
        let pa_ov_lsig1 = &bp[cols::PA_OV_LSIG1];
        let w_a = &bp[cols::W_A];
        let w_sig0 = &bp[cols::W_SIG0];
        let w_e = &bp[cols::W_E];
        let w_sig1 = &bp[cols::W_SIG1];
        let w_big_w = &bp[cols::W_W];
        let w_lsig0 = &bp[cols::W_LSIG0];
        let w_lsig1 = &bp[cols::W_LSIG1];

        let sha_s_init_prefix = &int[cols::SHA_S_INIT_PREFIX];
        // sha_s_feedforward is retained in the public layout for the
        // future verifier-side check that pa_c_ff_{a,e} are zero on
        // junction rows; the constraint expression doesn't reference it.
        let _sha_s_feedforward = &int[cols::SHA_S_FEEDFORWARD];
        let pa_c_c7 = &int[cols::SHA_PA_C_C7];
        let pa_c_c8 = &int[cols::SHA_PA_C_C8];
        let pa_c_c9 = &int[cols::SHA_PA_C_C9];
        let pa_c_ff_a = &int[cols::SHA_PA_C_FF_A];
        let pa_c_ff_e = &int[cols::SHA_PA_C_FF_E];
        let sha_s_b_active = &int[cols::SHA_S_B_ACTIVE];
        let sha_w_mu_junction_a = &int[cols::SHA_W_MU_JUNCTION_A];
        let sha_w_mu_junction_e = &int[cols::SHA_W_MU_JUNCTION_E];

        // SHA `down` slots (in source-col-ascending order — see signature()).
        // bin slots (22 SHA shifts, then 0 ECDSA shifts on bin):
        let down_w_a_sh1 = &down.binary_poly[0];
        let down_w_a_sh2 = &down.binary_poly[1];
        let down_w_a_sh4 = &down.binary_poly[2];
        let down_w_sig0_sh3 = &down.binary_poly[3];
        let down_w_e_sh1 = &down.binary_poly[4];
        let down_w_e_sh2 = &down.binary_poly[5];
        let down_w_e_sh4 = &down.binary_poly[6];
        let down_w_sig1_sh3 = &down.binary_poly[7];
        let down_w_w_sh3 = &down.binary_poly[8];
        let down_w_w_sh9 = &down.binary_poly[9];
        let down_w_w_sh16 = &down.binary_poly[10];
        let down_w_lsig0_sh1 = &down.binary_poly[11];
        let down_w_lsig1_sh14 = &down.binary_poly[12];
        let down_w_u_ef_sh2 = &down.binary_poly[13];
        let down_w_u_ef_sh3 = &down.binary_poly[14];
        let down_w_u_neg_e_g_sh2 = &down.binary_poly[15];
        let down_w_u_neg_e_g_sh3 = &down.binary_poly[16];
        let down_w_maj_sh2 = &down.binary_poly[17];
        let down_w_maj_sh3 = &down.binary_poly[18];
        let down_w_b1_sh2 = &down.binary_poly[19];
        let down_w_b2_sh2 = &down.binary_poly[20];
        let down_w_b3_sh2 = &down.binary_poly[21];
        // int slots: SHA shifts come first (4), then ECDSA (3).
        let down_pa_k_sh3 = &down.int[0];
        let down_w_mu_w_sh16 = &down.int[1];
        let down_w_mu_a_sh3 = &down.int[2];
        let down_w_mu_e_sh3 = &down.int[3];
        let down_ecdsa_x_sh1 = &down.int[4];
        let down_ecdsa_y_sh1 = &down.int[5];
        let down_ecdsa_z_sh1 = &down.int[6];

        // Bit-op virtual columns over W (sorted by `(source_col,
        // op_kind, c)` inside `UairSignature::new`). Six slots, all
        // with `source_col = W_W`. See `sha256.rs` for the mapping
        // between Rot/ShiftR amounts and SHA-256 ROTR/SHR.
        let down_w_rot13 = &down.bit_op[0]; // σ_1: ROTR^19
        let down_w_rot14 = &down.bit_op[1]; // σ_0: ROTR^18
        let down_w_rot15 = &down.bit_op[2]; // σ_1: ROTR^17
        let down_w_rot25 = &down.bit_op[3]; // σ_0: ROTR^7
        let down_w_shr3 = &down.bit_op[4]; //  σ_0: SHR^3
        let down_w_shr10 = &down.bit_op[5]; // σ_1: SHR^10

        let ideal_rot_xw1 = ideal_from_ref(&Sha256Ideal::<R>::RotXw1);
        let ideal_rot_x2 = ideal_from_ref(&Sha256Ideal::<R>::RotX2(RotationIdeal::new(
            R::ONE + R::ONE,
        )));

        let rho_sig0 = rho_poly::<R>(&[10, 19, 30]);
        let rho_sig1 = rho_poly::<R>(&[7, 21, 26]);
        let two_scalar_sha = const_scalar::<R>(R::ONE + R::ONE);
        let two_times_x31 = {
            let mut coeffs = [R::ZERO; 32];
            coeffs[31] = R::ONE + R::ONE;
            DensePolynomial::<R, 32>::new(coeffs)
        };

        // C1: Sigma_0 rotation
        b.assert_in_ideal(
            mbs(w_a, &rho_sig0).expect("a · rho_sig0 overflow") - w_sig0
                - &mbs(pa_ov_sig0, &two_scalar_sha).expect("2 · ov_sig0 overflow"),
            &ideal_rot_xw1,
        );

        // C2: Sigma_1 rotation
        b.assert_in_ideal(
            mbs(w_e, &rho_sig1).expect("e · rho_sig1 overflow") - w_sig1
                - &mbs(pa_ov_sig1, &two_scalar_sha).expect("2 · ov_sig1 overflow"),
            &ideal_rot_xw1,
        );

        // C4 (was σ_0 (X^32 − 1) ideal-lift): row-local Q[X] equality
        // with bit-XOR overflow correction. See sha256.rs for the
        // derivation. `pa_ov_lsig0` retained — only the modular lift
        // and the C3/C5 right-shift decompositions go away.
        //   ROT^25(W) + ROT^14(W) + SHIFTR^3(W) − lsig0 − 2 · pa_ov_lsig0 == 0
        b.assert_zero(
            down_w_rot25.clone() + down_w_rot14 + down_w_shr3 - w_lsig0
                - &mbs(pa_ov_lsig0, &two_scalar_sha).expect("2 · ov_lsig0 overflow"),
        );

        // C6 (was σ_1 (X^32 − 1) ideal-lift): σ_1 analogue of C4.
        //   ROT^15(W) + ROT^13(W) + SHIFTR^10(W) − lsig1 − 2 · pa_ov_lsig1 == 0
        b.assert_zero(
            down_w_rot15.clone() + down_w_rot13 + down_w_shr10 - w_lsig1
                - &mbs(pa_ov_lsig1, &two_scalar_sha).expect("2 · ov_lsig1 overflow"),
        );

        // C7: Message-schedule modular sum.
        let two_x31_mu_w =
            mbs(down_w_mu_w_sh16, &two_times_x31).expect("2·X^31 · mu_W overflow");
        let sched_inner = down_w_w_sh16.clone()
            - w_big_w
            - down_w_lsig0_sh1
            - down_w_w_sh9
            - down_w_lsig1_sh14
            + &two_x31_mu_w;
        b.assert_in_ideal(sched_inner + pa_c_c7, &ideal_rot_x2);

        // C8: Register-update for `a`. Ch[t] = u_ef[t] + u_{¬e,g}[t].
        let two_x31_mu_a =
            mbs(down_w_mu_a_sh3, &two_times_x31).expect("2·X^31 · mu_a overflow");
        let a_update_inner = down_w_a_sh4.clone()
            - w_e
            - down_w_sig1_sh3
            - down_w_u_ef_sh3
            - down_w_u_neg_e_g_sh3
            - down_pa_k_sh3
            - down_w_w_sh3
            - down_w_sig0_sh3
            - down_w_maj_sh3
            + &two_x31_mu_a;
        b.assert_in_ideal(a_update_inner + pa_c_c8, &ideal_rot_x2);

        // C9: Register-update for `e`. Ch[t] = u_ef[t] + u_{¬e,g}[t].
        let two_x31_mu_e =
            mbs(down_w_mu_e_sh3, &two_times_x31).expect("2·X^31 · mu_e overflow");
        let e_update_inner = down_w_e_sh4.clone()
            - w_a
            - w_e
            - down_w_sig1_sh3
            - down_w_u_ef_sh3
            - down_w_u_neg_e_g_sh3
            - down_pa_k_sh3
            - down_w_w_sh3
            + &two_x31_mu_e;
        b.assert_in_ideal(e_update_inner + pa_c_c9, &ideal_rot_x2);

        // C13–C15: Table 9 affine-combination materializations
        // (anchored at k = t − 2, gated by sha_s_b_active). See sha256.rs
        // for the full derivation.
        let two_scalar_poly = const_scalar::<R>(R::ONE + R::ONE);
        let one_32_poly = {
            let mut coeffs = [R::ZERO; 32];
            for c in coeffs.iter_mut() {
                *c = R::ONE;
            }
            DensePolynomial::<R, 32>::new(coeffs)
        };

        // C13: B_1[t] = e[t] + e[t-1] − 2·u_ef[t]  (Ch eq 62)
        let two_u_ef_sh2 = mbs(down_w_u_ef_sh2, &two_scalar_poly)
            .expect("2 · u_ef overflow");
        let b1_inner = down_w_b1_sh2.clone()
            - down_w_e_sh2
            - down_w_e_sh1
            + &two_u_ef_sh2;
        b.assert_zero(sha_s_b_active.clone() * &b1_inner);

        // C14: B_2[t] = (1₃₂ − e[t]) + e[t-2] − 2·u_{¬e,g}[t]  (Ch eq 63)
        let two_u_neg_e_g_sh2 = mbs(down_w_u_neg_e_g_sh2, &two_scalar_poly)
            .expect("2 · u_{¬e,g} overflow");
        let b2_inner = down_w_b2_sh2.clone()
            - &from_ref(&one_32_poly)
            + down_w_e_sh2
            - w_e
            + &two_u_neg_e_g_sh2;
        b.assert_zero(sha_s_b_active.clone() * &b2_inner);

        // C15: B_3[t] = a[t] + a[t-1] + a[t-2] − 2·Maj[t]  (Maj eq 64)
        let two_maj_sh2 = mbs(down_w_maj_sh2, &two_scalar_poly)
            .expect("2 · Maj overflow");
        let b3_inner = down_w_b3_sh2.clone()
            - down_w_a_sh2
            - down_w_a_sh1
            - w_a
            + &two_maj_sh2;
        b.assert_zero(sha_s_b_active.clone() * &b3_inner);

        // C10/C11: per-compression init-prefix pinning. See sha256.rs
        // for the chained-compression layout. Subsumes the old per-trace
        // init/final boundary constraints — every compression's init
        // prefix and the final H_N output block are pinned by the same
        // s_init_prefix selector.
        b.assert_zero(sha_s_init_prefix.clone() * &(w_a.clone() - pa_a));
        b.assert_zero(sha_s_init_prefix.clone() * &(w_e.clone() - pa_e));

        // C12/C13: SHA-256 feed-forward addition at each junction
        // window. Mirrors the standalone SHA UAIR — uses the public
        // compensator pattern (pa_c_ff_{a,e}) instead of a multiplicative
        // selector to keep the constraint at degree 1 in the trace MLEs
        // (so the merged UAIR can stay MLE-first eligible). References:
        //   up.w_a            = internal_final_i, j-th component
        //   up.pa_a           = H_i, j-th component (junction copy)
        //   down.w_a^↓4       = w_a[k+4] = H_{i+1}, j-th component (pinned by C10)
        //   up.sha_w_mu_junction_a = carry ∈ {0, 1}
        let two_x31_mu_ff_a = mbs(sha_w_mu_junction_a, &two_times_x31)
            .expect("2·X^31 · mu_junction_a overflow");
        let ff_a_inner = down_w_a_sh4.clone()
            - w_a
            - pa_a
            + &two_x31_mu_ff_a;
        b.assert_in_ideal(ff_a_inner + pa_c_ff_a, &ideal_rot_x2);

        let two_x31_mu_ff_e = mbs(sha_w_mu_junction_e, &two_times_x31)
            .expect("2·X^31 · mu_junction_e overflow");
        let ff_e_inner = down_w_e_sh4.clone()
            - w_e
            - pa_e
            + &two_x31_mu_ff_e;
        b.assert_in_ideal(ff_e_inner + pa_c_ff_e, &ideal_rot_x2);

        // ===================================================================
        // ECDSA half — verbatim from ecdsa.rs's constrain_general,
        // referencing merged column indices.
        // ===================================================================
        let e_s_init = &int[cols::ECDSA_S_INIT];
        let e_s_active = &int[cols::ECDSA_S_ACTIVE];
        let e_s_final = &int[cols::ECDSA_S_FINAL];
        let e_s_add = &int[cols::ECDSA_S_ADD];
        let e_pa_x_addend = &int[cols::ECDSA_PA_X_ADDEND];
        let e_pa_y_addend = &int[cols::ECDSA_PA_Y_ADDEND];
        let e_pa_r_init_x = &int[cols::ECDSA_PA_R_INIT_X];
        let e_pa_r_init_y = &int[cols::ECDSA_PA_R_INIT_Y];
        let e_pa_r_init_z = &int[cols::ECDSA_PA_R_INIT_Z];
        let e_x = &int[cols::ECDSA_W_X];
        let e_y = &int[cols::ECDSA_W_Y];
        let e_z = &int[cols::ECDSA_W_Z];
        let e_x_pa = &int[cols::ECDSA_W_X_PA];
        let e_y_pa = &int[cols::ECDSA_W_Y_PA];
        let e_z_pa = &int[cols::ECDSA_W_Z_PA];
        let e_c = &int[cols::ECDSA_W_C];
        let e_d = &int[cols::ECDSA_W_D];

        let two_scalar = const_scalar::<R>(R::from(2_u32));
        let three_scalar = const_scalar::<R>(R::from(3_u32));
        let eight_scalar = const_scalar::<R>(R::from(8_u32));
        let nine_scalar = const_scalar::<R>(R::from(9_u32));
        let twelve_scalar = const_scalar::<R>(R::from(12_u32));

        // === Doubling block (3 constraints; `S = Y²` inlined) ===
        let e_y_sq = e_y.clone() * e_y;

        let yz = e_y.clone() * e_z;
        let two_yz = mbs(&yz, &two_scalar).expect("2·Y·Z overflow");
        let d2_inner = e_z_pa.clone() - &two_yz;
        b.assert_zero(e_s_active.clone() * &d2_inner);

        let x_sq = e_x.clone() * e_x;
        let x_pow4 = x_sq.clone() * &x_sq;
        let nine_x4 = mbs(&x_pow4, &nine_scalar).expect("9·X⁴ overflow");
        let x_y_sq = e_x.clone() * &e_y_sq;
        let eight_x_y_sq = mbs(&x_y_sq, &eight_scalar).expect("8·X·Y² overflow");
        let d3_inner = e_x_pa.clone() - &nine_x4 + &eight_x_y_sq;
        b.assert_zero(e_s_active.clone() * &d3_inner);

        let x3_y_sq = x_sq.clone() * &x_y_sq;
        let twelve_x3_y_sq =
            mbs(&x3_y_sq, &twelve_scalar).expect("12·X³·Y² overflow");
        let x_sq_x_pa = x_sq.clone() * e_x_pa;
        let three_x2_xpa =
            mbs(&x_sq_x_pa, &three_scalar).expect("3·X²·X_pa overflow");
        let y_pow4 = e_y_sq.clone() * &e_y_sq;
        let eight_y_pow4 = mbs(&y_pow4, &eight_scalar).expect("8·Y⁴ overflow");
        let d4_inner =
            e_y_pa.clone() - &twelve_x3_y_sq + &three_x2_xpa + &eight_y_pow4;
        b.assert_zero(e_s_active.clone() * &d4_inner);

        // === Addition scratch (2 constraints; Z_pa², Z_pa³ inlined) ===
        let e_z_pa_sq = e_z_pa.clone() * e_z_pa;
        let a1_inner = e_c.clone() + e_x_pa - &(e_pa_x_addend.clone() * &e_z_pa_sq);
        b.assert_zero(e_s_active.clone() * &a1_inner);

        let e_z_pa_cube = e_z_pa.clone() * &e_z_pa_sq;
        let a2_inner = e_d.clone() + e_y_pa - &(e_pa_y_addend.clone() * &e_z_pa_cube);
        b.assert_zero(e_s_active.clone() * &a2_inner);

        // === Output-selection-and-chaining (3 constraints, with
        //    E=C², F=C³, G=X_pa·C² inlined; Y attains deg 6) ===

        // X: down.X − X_pa − S_ADD·(D² − C³ − 2·X_pa·C² − X_pa) = 0
        let e_c_sq = e_c.clone() * e_c;
        let e_c_cube = e_c.clone() * &e_c_sq;
        let e_x_pa_c_sq = e_x_pa.clone() * &e_c_sq;
        let two_x_pa_c_sq = mbs(&e_x_pa_c_sq, &two_scalar).expect("2·X_pa·C² overflow");
        let d_sq = e_d.clone() * e_d;
        let x_add_minus_x_pa = d_sq.clone() - &e_c_cube - &two_x_pa_c_sq - e_x_pa;
        let s_add_x = e_s_add.clone() * &x_add_minus_x_pa;
        let o1_inner = down_ecdsa_x_sh1.clone() - e_x_pa - &s_add_x;
        b.assert_zero(e_s_active.clone() * &o1_inner);

        // Y: down.Y − Y_pa − S_ADD·(3·D·X_pa·C² + D·C³ − D³ − Y_pa·C³ − Y_pa) = 0
        let d_cube = e_d.clone() * &d_sq;
        let d_x_pa_c_sq = e_d.clone() * &e_x_pa_c_sq;
        let three_d_x_pa_c_sq =
            mbs(&d_x_pa_c_sq, &three_scalar).expect("3·D·X_pa·C² overflow");
        let d_c_cube = e_d.clone() * &e_c_cube;
        let y_pa_c_cube = e_y_pa.clone() * &e_c_cube;
        let y_add_minus_y_pa =
            three_d_x_pa_c_sq + &d_c_cube - &d_cube - &y_pa_c_cube - e_y_pa;
        let s_add_y = e_s_add.clone() * &y_add_minus_y_pa;
        let o2_inner = down_ecdsa_y_sh1.clone() - e_y_pa - &s_add_y;
        b.assert_zero(e_s_active.clone() * &o2_inner);

        // Z: down.Z − Z_pa − S_ADD·(Z_pa·C − Z_pa) = 0
        let z_pa_c = e_z_pa.clone() * e_c;
        let z_add_minus_z_pa = z_pa_c - e_z_pa;
        let s_add_z = e_s_add.clone() * &z_add_minus_z_pa;
        let o3_inner = down_ecdsa_z_sh1.clone() - e_z_pa - &s_add_z;
        b.assert_zero(e_s_active.clone() * &o3_inner);

        // === Init boundary (3 constraints) ===
        b.assert_zero(e_s_init.clone() * &(e_x.clone() - e_pa_r_init_x));
        b.assert_zero(e_s_init.clone() * &(e_y.clone() - e_pa_r_init_y));
        b.assert_zero(e_s_init.clone() * &(e_z.clone() - e_pa_r_init_z));

        // No final-row affine conversion in-circuit. Z_inv / X_aff /
        // Y_aff are reconstructed off-protocol from the opened
        // Z[FINAL_ROW]. `e_s_final` is no longer used on the ECDSA
        // side but is kept in the public layout for downstream gluing
        // UAIRs that compose with this one.
        let _ = e_s_final;
    }
}

// ---------------------------------------------------------------------------
// Helpers (rho/monomial/const-scalar) — duplicated from sha256.rs since
// those are private to that module.
// ---------------------------------------------------------------------------

fn rho_poly<R: ConstSemiring>(positions: &[usize]) -> DensePolynomial<R, 32> {
    let mut coeffs = [R::ZERO; 32];
    for &p in positions {
        debug_assert!(p < 32);
        coeffs[p] = R::ONE;
    }
    DensePolynomial::<R, 32>::new(coeffs)
}

fn const_scalar<R: ConstSemiring>(c: R) -> DensePolynomial<R, 32> {
    let mut coeffs = [R::ZERO; 32];
    coeffs[0] = c;
    DensePolynomial::<R, 32>::new(coeffs)
}

// ---------------------------------------------------------------------------
// GenerateRandomTrace — call both sub-UAIRs' generators, splice the int
// sections together at the merged column positions.
// ---------------------------------------------------------------------------

impl<R> GenerateRandomTrace<32> for ShaEcdsaUair<R>
where
    R: EcdsaFpRing + From<u32> + From<Int<EC_FP_INT_LIMBS>>,
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
            "ShaEcdsa UAIR needs > {FINAL_ROW} rows; got {n_rows}",
        );

        let sha_trace = <Sha256CompressionSliceUair<R> as GenerateRandomTrace<32>>::
            generate_random_trace(num_vars, rng);
        let ecdsa_trace = <super::EcdsaUair<R> as GenerateRandomTrace<32>>::
            generate_random_trace(num_vars, rng);

        // Sanity: column counts match the standalone UAIRs.
        assert_eq!(sha_trace.binary_poly.len(), sha256::cols::NUM_BIN);
        assert_eq!(sha_trace.int.len(), sha256::cols::NUM_INT);
        assert_eq!(ecdsa_trace.int.len(), ecdsa::cols::NUM_INT);

        // Binary_poly: copy SHA's directly (ECDSA contributes nothing).
        let binary_poly: Vec<DenseMultilinearExtension<_>> =
            sha_trace.binary_poly.into_owned();

        // Int section: merge per the layout in `cols`.
        // SHA standalone int layout (14 cols):
        //   0..8   pubs (S_INIT_PREFIX, S_FEEDFORWARD, PA_K,
        //                PA_C_C7/8/9, PA_C_FF_A, PA_C_FF_E)
        //   8      pub  (S_B_ACTIVE)
        //   9..12  witnesses (mu_W, mu_a, mu_e)
        //   12..14 witnesses (mu_junction_a, mu_junction_e) — chained-comp
        //          feed-forward carries.
        // ECDSA standalone int layout (17 cols):
        //   0..9   pubs
        //   9..17  witnesses (8 EC cols)
        let mut int: Vec<DenseMultilinearExtension<R>> = Vec::with_capacity(cols::NUM_INT);
        let sha_ints = sha_trace.int.into_owned();
        let ecdsa_ints = ecdsa_trace.int.into_owned();

        // [0..8] SHA pubs (sha[0..8])
        int.extend(sha_ints[0..8].iter().cloned());
        // [8..17] ECDSA pubs (ecdsa[0..9])
        int.extend(ecdsa_ints[0..9].iter().cloned());
        // [17] SHA S_B_ACTIVE pub (sha[8])
        int.push(sha_ints[8].clone());
        // [18..21] SHA carry witnesses (sha[9..12] = mu_W, mu_a, mu_e)
        int.extend(sha_ints[9..12].iter().cloned());
        // [21..29] ECDSA witnesses (ecdsa[9..17], 8 cols)
        int.extend(ecdsa_ints[9..17].iter().cloned());
        // [29..31] SHA junction-carry witnesses (sha[12..14]). Appended
        // after ECDSA to preserve all existing ECDSA_W_* indices.
        int.extend(sha_ints[12..14].iter().cloned());

        debug_assert_eq!(int.len(), cols::NUM_INT);

        UairTrace {
            binary_poly: binary_poly.into(),
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
    use rand::rng;
    use zinc_uair::{
        constraint_counter::count_constraints,
        degree_counter::{count_constraint_degrees, count_max_degree},
    };

    /// Sanity: 13 SHA + 11 ECDSA = 24 constraints (after the σ_0/σ_1
    /// bit-op virtual-column rewrite removed C3 and C5, and the chained-
    /// compressions rewrite split the 3 boundary constraints
    /// (init-a, final-a, final-e) into 4 (init-prefix-a, init-prefix-e,
    /// feed-forward-a, feed-forward-e)). Max degree 6 from the ECDSA Y
    /// output-selection constraint (and from D4's `12·X³·Y²` term).
    #[test]
    fn sha_ecdsa_constraint_shape() {
        type U = ShaEcdsaUair<Int<EC_FP_INT_LIMBS>>;
        assert_eq!(count_constraints::<U>(), 24);
        assert_eq!(count_max_degree::<U>(), 6);
        let degrees = count_constraint_degrees::<U>();
        // Spot checks: at least one deg-6 (ECDSA Y output sel / D4),
        // some deg-2 (boundaries + chaining + B_i materializations),
        // some deg-1 (SHA C1, C2, C4, C6 — including the new row-local
        // σ_0/σ_1 equalities).
        assert!(degrees.iter().any(|&d| d == 6), "expected deg-6 from ECDSA");
        assert!(degrees.iter().filter(|&&d| d == 2).count() >= 6, "expected ≥6 deg-2");
    }

    /// The merged trace builder produces a trace with the right column
    /// shape (we don't re-run the full mod-p witness check here — the
    /// sub-UAIRs already test their halves individually).
    #[test]
    fn merged_trace_shape() {
        let num_vars = 9;
        let mut r = rng();
        let trace = <ShaEcdsaUair<Int<EC_FP_INT_LIMBS>> as GenerateRandomTrace<32>>::
            generate_random_trace(num_vars, &mut r);

        assert_eq!(trace.binary_poly.len(), cols::NUM_BIN);
        assert_eq!(trace.int.len(), cols::NUM_INT);
        for col in trace.binary_poly.iter() {
            assert_eq!(col.len(), 1 << num_vars);
        }
        for col in trace.int.iter() {
            assert_eq!(col.len(), 1 << num_vars);
        }
    }

    /// Re-export sanity: NUM_SHAMIR_ROUNDS, FINAL_ROW are accessible
    /// through this module (matching `crate::ecdsa`).
    #[test]
    fn re_exports() {
        let _ = NUM_SHAMIR_ROUNDS;
        let _ = FINAL_ROW;
        let _ = ECDSA_FINAL_ROW;
    }
}
