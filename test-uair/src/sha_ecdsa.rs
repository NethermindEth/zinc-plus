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
//! **binary_poly section** (19 cols, unchanged from SHA standalone):
//! - `[0..2]` public: `PA_A`, `PA_E`
//! - `[2..19]` witness: SHA's witness bit-polys
//!
//! **int section** (45 cols total, 15 pubs + 28 witness + 2
//!   multiplicities):
//! - `[0..6]` public: SHA pubs (S_INIT, S_FINAL, PA_K, PA_C_C7/8/9)
//! - `[6..15]` public: ECDSA pubs (S_INIT, S_ACTIVE, S_FINAL, S_ADD,
//!   PA_X_ADDEND, PA_Y_ADDEND, PA_R_INIT_X/Y/Z)
//! - `[15..18]` witness: SHA `mu_W, mu_a, mu_e`
//! - `[18..43]` witness: ECDSA chained-input + intermediates +
//!   outputs + affine cells (25 cols)
//! - `[43..45]` witness: SHA lookup multiplicities (`M_W2, M_W3`) —
//!   per-protocol convention, multiplicity columns are the last N
//!   ints (one per lookup group).
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
    ConstraintBuilder, LookupColumnSpec, LookupTableType, PublicColumnLayout, ShiftSpec,
    TotalColumnLayout, TraceRow, Uair, UairSignature, UairTrace,
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
    // ===== binary_poly (unchanged from SHA standalone) =====
    pub const PA_A: usize = 0;
    pub const PA_E: usize = 1;
    pub const W_A: usize = 2;
    pub const W_SIG0: usize = 3;
    pub const W_OV_SIG0: usize = 4;
    pub const W_E: usize = 5;
    pub const W_SIG1: usize = 6;
    pub const W_OV_SIG1: usize = 7;
    pub const W_W: usize = 8;
    pub const W_LSIG0: usize = 9;
    pub const W_S0: usize = 10;
    pub const W_T0: usize = 11;
    pub const W_OV_LSIG0: usize = 12;
    pub const W_LSIG1: usize = 13;
    pub const W_S1: usize = 14;
    pub const W_T1: usize = 15;
    pub const W_OV_LSIG1: usize = 16;
    pub const W_CH: usize = 17;
    pub const W_MAJ: usize = 18;
    pub const NUM_BIN: usize = 19;
    pub const NUM_BIN_PUB: usize = 2;

    // ===== int section =====
    // SHA publics (0..6)
    pub const SHA_S_INIT: usize = 0;
    pub const SHA_S_FINAL: usize = 1;
    pub const SHA_PA_K: usize = 2;
    pub const SHA_PA_C_C7: usize = 3;
    pub const SHA_PA_C_C8: usize = 4;
    pub const SHA_PA_C_C9: usize = 5;
    // ECDSA publics (6..15)
    pub const ECDSA_S_INIT: usize = 6;
    pub const ECDSA_S_ACTIVE: usize = 7;
    pub const ECDSA_S_FINAL: usize = 8;
    pub const ECDSA_S_ADD: usize = 9;
    pub const ECDSA_PA_X_ADDEND: usize = 10;
    pub const ECDSA_PA_Y_ADDEND: usize = 11;
    pub const ECDSA_PA_R_INIT_X: usize = 12;
    pub const ECDSA_PA_R_INIT_Y: usize = 13;
    pub const ECDSA_PA_R_INIT_Z: usize = 14;
    pub const NUM_INT_PUB: usize = 15;

    // SHA witnesses (15..18) — carry-range columns.
    pub const SHA_W_MU_W: usize = 15;
    pub const SHA_W_MU_A: usize = 16;
    pub const SHA_W_MU_E: usize = 17;

    // ECDSA witnesses (18..43): chained input + doubling/addition
    // intermediates + outputs + affine cells.
    pub const ECDSA_W_X1: usize = 18;
    pub const ECDSA_W_Y1: usize = 19;
    pub const ECDSA_W_Z1: usize = 20;
    pub const ECDSA_W_S: usize = 21;
    pub const ECDSA_W_X_PA: usize = 22;
    pub const ECDSA_W_Y_PA: usize = 23;
    pub const ECDSA_W_Z_PA: usize = 24;
    pub const ECDSA_W_Z_PA_SQ: usize = 25;
    pub const ECDSA_W_Z_PA_CUBE: usize = 26;
    pub const ECDSA_W_C: usize = 27;
    pub const ECDSA_W_D: usize = 28;
    pub const ECDSA_W_E: usize = 29;
    pub const ECDSA_W_F: usize = 30;
    pub const ECDSA_W_G: usize = 31;
    pub const ECDSA_W_X_ADD: usize = 32;
    pub const ECDSA_W_Y_ADD: usize = 33;
    pub const ECDSA_W_Z_ADD: usize = 34;
    pub const ECDSA_W_X_OUT: usize = 35;
    pub const ECDSA_W_Y_OUT: usize = 36;
    pub const ECDSA_W_Z_OUT: usize = 37;
    pub const ECDSA_W_Z_INV: usize = 38;
    pub const ECDSA_W_Z_INV_SQ: usize = 39;
    pub const ECDSA_W_Z_INV_CUBE: usize = 40;
    pub const ECDSA_W_X_AFF: usize = 41;
    pub const ECDSA_W_Y_AFF: usize = 42;

    // SHA multiplicities (43..45) — MUST be the last N int cols.
    pub const SHA_W_M_W2: usize = 43;
    pub const SHA_W_M_W3: usize = 44;

    pub const NUM_INT: usize = 45;

    // Flat indices (binary_poly || arbitrary_poly || int).
    pub const FLAT_W_A: usize = W_A;
    pub const FLAT_W_SIG0: usize = W_SIG0;
    pub const FLAT_W_E: usize = W_E;
    pub const FLAT_W_SIG1: usize = W_SIG1;
    pub const FLAT_W_W: usize = W_W;
    pub const FLAT_W_LSIG0: usize = W_LSIG0;
    pub const FLAT_W_LSIG1: usize = W_LSIG1;
    pub const FLAT_W_CH: usize = W_CH;
    pub const FLAT_W_MAJ: usize = W_MAJ;
    pub const FLAT_SHA_PA_K: usize = NUM_BIN + SHA_PA_K;
    pub const FLAT_SHA_W_MU_W: usize = NUM_BIN + SHA_W_MU_W;
    pub const FLAT_SHA_W_MU_A: usize = NUM_BIN + SHA_W_MU_A;
    pub const FLAT_SHA_W_MU_E: usize = NUM_BIN + SHA_W_MU_E;
    pub const FLAT_ECDSA_W_X1: usize = NUM_BIN + ECDSA_W_X1;
    pub const FLAT_ECDSA_W_Y1: usize = NUM_BIN + ECDSA_W_Y1;
    pub const FLAT_ECDSA_W_Z1: usize = NUM_BIN + ECDSA_W_Z1;
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
        // UairSignature::new; insertion order breaks ties).
        let shifts: Vec<ShiftSpec> = vec![
            // === SHA binary_poly shifts ===
            ShiftSpec::new(cols::FLAT_W_A, 4),
            ShiftSpec::new(cols::FLAT_W_SIG0, 3),
            ShiftSpec::new(cols::FLAT_W_E, 4),
            ShiftSpec::new(cols::FLAT_W_SIG1, 3),
            ShiftSpec::new(cols::FLAT_W_W, 3),
            ShiftSpec::new(cols::FLAT_W_W, 9),
            ShiftSpec::new(cols::FLAT_W_W, 16),
            ShiftSpec::new(cols::FLAT_W_LSIG0, 1),
            ShiftSpec::new(cols::FLAT_W_LSIG1, 14),
            ShiftSpec::new(cols::FLAT_W_CH, 3),
            ShiftSpec::new(cols::FLAT_W_MAJ, 3),
            // === SHA int shifts (PA_K, mu_W, mu_a, mu_e) ===
            ShiftSpec::new(cols::FLAT_SHA_PA_K, 3),
            ShiftSpec::new(cols::FLAT_SHA_W_MU_W, 16),
            ShiftSpec::new(cols::FLAT_SHA_W_MU_A, 3),
            ShiftSpec::new(cols::FLAT_SHA_W_MU_E, 3),
            // === ECDSA int shifts (X1, Y1, Z1 by 1 each for chaining) ===
            ShiftSpec::new(cols::FLAT_ECDSA_W_X1, 1),
            ShiftSpec::new(cols::FLAT_ECDSA_W_Y1, 1),
            ShiftSpec::new(cols::FLAT_ECDSA_W_Z1, 1),
        ];

        // Lookup specs: SHA's three carry range checks. (ECDSA has none
        // in the no-quotient F_p formulation.)
        let lookup_specs = vec![
            LookupColumnSpec {
                column_index: cols::FLAT_SHA_W_MU_W,
                table_type: LookupTableType::Word {
                    width: 2,
                    chunk_width: None,
                },
            },
            LookupColumnSpec {
                column_index: cols::FLAT_SHA_W_MU_A,
                table_type: LookupTableType::Word {
                    width: 3,
                    chunk_width: None,
                },
            },
            LookupColumnSpec {
                column_index: cols::FLAT_SHA_W_MU_E,
                table_type: LookupTableType::Word {
                    width: 3,
                    chunk_width: None,
                },
            },
        ];

        UairSignature::new(total, public, shifts, lookup_specs)
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
        // SHA-256 half — verbatim from sha256.rs's constrain_general,
        // referencing merged column indices.
        // ===================================================================
        let bp = up.binary_poly;
        let int = up.int;

        let pa_a = &bp[cols::PA_A];
        let pa_e = &bp[cols::PA_E];
        let w_a = &bp[cols::W_A];
        let w_sig0 = &bp[cols::W_SIG0];
        let w_ov_sig0 = &bp[cols::W_OV_SIG0];
        let w_e = &bp[cols::W_E];
        let w_sig1 = &bp[cols::W_SIG1];
        let w_ov_sig1 = &bp[cols::W_OV_SIG1];
        let w_big_w = &bp[cols::W_W];
        let w_lsig0 = &bp[cols::W_LSIG0];
        let w_s0 = &bp[cols::W_S0];
        let w_t0 = &bp[cols::W_T0];
        let w_ov_lsig0 = &bp[cols::W_OV_LSIG0];
        let w_lsig1 = &bp[cols::W_LSIG1];
        let w_s1 = &bp[cols::W_S1];
        let w_t1 = &bp[cols::W_T1];
        let w_ov_lsig1 = &bp[cols::W_OV_LSIG1];

        let sha_s_init = &int[cols::SHA_S_INIT];
        let sha_s_final = &int[cols::SHA_S_FINAL];
        let pa_c_c7 = &int[cols::SHA_PA_C_C7];
        let pa_c_c8 = &int[cols::SHA_PA_C_C8];
        let pa_c_c9 = &int[cols::SHA_PA_C_C9];

        // SHA `down` slots (in source-col-ascending order — see signature()).
        // bin slots:
        let down_w_a_sh4 = &down.binary_poly[0];
        let down_w_sig0_sh3 = &down.binary_poly[1];
        let down_w_e_sh4 = &down.binary_poly[2];
        let down_w_sig1_sh3 = &down.binary_poly[3];
        let down_w_w_sh3 = &down.binary_poly[4];
        let down_w_w_sh9 = &down.binary_poly[5];
        let down_w_w_sh16 = &down.binary_poly[6];
        let down_w_lsig0_sh1 = &down.binary_poly[7];
        let down_w_lsig1_sh14 = &down.binary_poly[8];
        let down_w_ch_sh3 = &down.binary_poly[9];
        let down_w_maj_sh3 = &down.binary_poly[10];
        // int slots: SHA shifts come first (4), then ECDSA (3).
        let down_pa_k_sh3 = &down.int[0];
        let down_w_mu_w_sh16 = &down.int[1];
        let down_w_mu_a_sh3 = &down.int[2];
        let down_w_mu_e_sh3 = &down.int[3];
        let down_ecdsa_x1_sh1 = &down.int[4];
        let down_ecdsa_y1_sh1 = &down.int[5];
        let down_ecdsa_z1_sh1 = &down.int[6];

        let ideal_rot_xw1 = ideal_from_ref(&Sha256Ideal::<R>::RotXw1);
        let ideal_rot_x2 = ideal_from_ref(&Sha256Ideal::<R>::RotX2(RotationIdeal::new(
            R::ONE + R::ONE,
        )));

        let rho_sig0 = rho_poly::<R>(&[10, 19, 30]);
        let rho_sig1 = rho_poly::<R>(&[7, 21, 26]);
        let rho_lsig0 = rho_poly::<R>(&[14, 25]);
        let rho_lsig1 = rho_poly::<R>(&[13, 15]);
        let two_scalar_sha = const_scalar::<R>(R::ONE + R::ONE);
        let x_pow_3 = mono_x_pow::<R>(3);
        let x_pow_10 = mono_x_pow::<R>(10);
        let two_times_x31 = {
            let mut coeffs = [R::ZERO; 32];
            coeffs[31] = R::ONE + R::ONE;
            DensePolynomial::<R, 32>::new(coeffs)
        };

        // C1: Sigma_0 rotation
        b.assert_in_ideal(
            mbs(w_a, &rho_sig0).expect("a · rho_sig0 overflow") - w_sig0
                - &mbs(w_ov_sig0, &two_scalar_sha).expect("2 · ov_sig0 overflow"),
            &ideal_rot_xw1,
        );

        // C2: Sigma_1 rotation
        b.assert_in_ideal(
            mbs(w_e, &rho_sig1).expect("e · rho_sig1 overflow") - w_sig1
                - &mbs(w_ov_sig1, &two_scalar_sha).expect("2 · ov_sig1 overflow"),
            &ideal_rot_xw1,
        );

        // C3: sigma_0 right-shift decomposition
        b.assert_zero(
            w_big_w.clone() - w_t0 - &mbs(w_s0, &x_pow_3).expect("X^3 · S_0 overflow"),
        );

        // C4: sigma_0 rotation
        b.assert_in_ideal(
            mbs(w_big_w, &rho_lsig0).expect("W · rho_lsig0 overflow") + w_s0 - w_lsig0
                - &mbs(w_ov_lsig0, &two_scalar_sha).expect("2 · ov_lsig0 overflow"),
            &ideal_rot_xw1,
        );

        // C5: sigma_1 right-shift decomposition
        b.assert_zero(
            w_big_w.clone() - w_t1 - &mbs(w_s1, &x_pow_10).expect("X^10 · S_1 overflow"),
        );

        // C6: sigma_1 rotation
        b.assert_in_ideal(
            mbs(w_big_w, &rho_lsig1).expect("W · rho_lsig1 overflow") + w_s1 - w_lsig1
                - &mbs(w_ov_lsig1, &two_scalar_sha).expect("2 · ov_lsig1 overflow"),
            &ideal_rot_xw1,
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

        // C8: Register-update for `a`.
        let two_x31_mu_a =
            mbs(down_w_mu_a_sh3, &two_times_x31).expect("2·X^31 · mu_a overflow");
        let a_update_inner = down_w_a_sh4.clone()
            - w_e
            - down_w_sig1_sh3
            - down_w_ch_sh3
            - down_pa_k_sh3
            - down_w_w_sh3
            - down_w_sig0_sh3
            - down_w_maj_sh3
            + &two_x31_mu_a;
        b.assert_in_ideal(a_update_inner + pa_c_c8, &ideal_rot_x2);

        // C9: Register-update for `e`.
        let two_x31_mu_e =
            mbs(down_w_mu_e_sh3, &two_times_x31).expect("2·X^31 · mu_e overflow");
        let e_update_inner = down_w_e_sh4.clone()
            - w_a
            - w_e
            - down_w_sig1_sh3
            - down_w_ch_sh3
            - down_pa_k_sh3
            - down_w_w_sh3
            + &two_x31_mu_e;
        b.assert_in_ideal(e_update_inner + pa_c_c9, &ideal_rot_x2);

        // C10: Init boundary on a.
        b.assert_zero(sha_s_init.clone() * &(w_a.clone() - pa_a));
        // C11: Final boundary on a-family.
        b.assert_zero(sha_s_final.clone() * &(w_a.clone() - pa_a));
        // C12: Final boundary on e-family.
        b.assert_zero(sha_s_final.clone() * &(w_e.clone() - pa_e));

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
        let e_x1 = &int[cols::ECDSA_W_X1];
        let e_y1 = &int[cols::ECDSA_W_Y1];
        let e_z1 = &int[cols::ECDSA_W_Z1];
        let e_s_w = &int[cols::ECDSA_W_S];
        let e_x_pa = &int[cols::ECDSA_W_X_PA];
        let e_y_pa = &int[cols::ECDSA_W_Y_PA];
        let e_z_pa = &int[cols::ECDSA_W_Z_PA];
        let e_z_pa_sq = &int[cols::ECDSA_W_Z_PA_SQ];
        let e_z_pa_cube = &int[cols::ECDSA_W_Z_PA_CUBE];
        let e_c = &int[cols::ECDSA_W_C];
        let e_d = &int[cols::ECDSA_W_D];
        let e_e = &int[cols::ECDSA_W_E];
        let e_f = &int[cols::ECDSA_W_F];
        let e_g = &int[cols::ECDSA_W_G];
        let e_x_add = &int[cols::ECDSA_W_X_ADD];
        let e_y_add = &int[cols::ECDSA_W_Y_ADD];
        let e_z_add = &int[cols::ECDSA_W_Z_ADD];
        let e_x_out = &int[cols::ECDSA_W_X_OUT];
        let e_y_out = &int[cols::ECDSA_W_Y_OUT];
        let e_z_out = &int[cols::ECDSA_W_Z_OUT];
        let e_z_inv = &int[cols::ECDSA_W_Z_INV];
        let e_z_inv_sq = &int[cols::ECDSA_W_Z_INV_SQ];
        let e_z_inv_cube = &int[cols::ECDSA_W_Z_INV_CUBE];
        let e_x_aff = &int[cols::ECDSA_W_X_AFF];
        let e_y_aff = &int[cols::ECDSA_W_Y_AFF];

        let two_scalar = const_scalar::<R>(R::from(2_u32));
        let three_scalar = const_scalar::<R>(R::from(3_u32));
        let eight_scalar = const_scalar::<R>(R::from(8_u32));
        let nine_scalar = const_scalar::<R>(R::from(9_u32));
        let twelve_scalar = const_scalar::<R>(R::from(12_u32));
        let one_expr = from_ref(&const_scalar::<R>(R::ONE));

        // === Doubling block (4 constraints, max degree 5) ===
        let d1_inner = e_s_w.clone() - &(e_y1.clone() * e_y1);
        b.assert_zero(e_s_active.clone() * &d1_inner);

        let yz = e_y1.clone() * e_z1;
        let two_yz = mbs(&yz, &two_scalar).expect("2·Y1·Z1 overflow");
        let d2_inner = e_z_pa.clone() - &two_yz;
        b.assert_zero(e_s_active.clone() * &d2_inner);

        let x_sq = e_x1.clone() * e_x1;
        let x_pow4 = x_sq.clone() * &x_sq;
        let nine_x4 = mbs(&x_pow4, &nine_scalar).expect("9·X1⁴ overflow");
        let xs = e_x1.clone() * e_s_w;
        let eight_xs = mbs(&xs, &eight_scalar).expect("8·X1·S overflow");
        let d3_inner = e_x_pa.clone() - &nine_x4 + &eight_xs;
        b.assert_zero(e_s_active.clone() * &d3_inner);

        let x3s = x_sq.clone() * &xs;
        let twelve_x3s = mbs(&x3s, &twelve_scalar).expect("12·X1³·S overflow");
        let x_sq_x_pa = x_sq.clone() * e_x_pa;
        let three_x2_xpa =
            mbs(&x_sq_x_pa, &three_scalar).expect("3·X1²·X_pa overflow");
        let s_sq = e_s_w.clone() * e_s_w;
        let eight_s_sq = mbs(&s_sq, &eight_scalar).expect("8·S² overflow");
        let d4_inner = e_y_pa.clone() - &twelve_x3s + &three_x2_xpa + &eight_s_sq;
        b.assert_zero(e_s_active.clone() * &d4_inner);

        // === Addition block (10 constraints, max degree 3) ===
        let a1_inner = e_z_pa_sq.clone() - &(e_z_pa.clone() * e_z_pa);
        b.assert_zero(e_s_active.clone() * &a1_inner);

        let a2_inner = e_z_pa_cube.clone() - &(e_z_pa.clone() * e_z_pa_sq);
        b.assert_zero(e_s_active.clone() * &a2_inner);

        let a3_inner = e_c.clone() + e_x_pa - &(e_pa_x_addend.clone() * e_z_pa_sq);
        b.assert_zero(e_s_active.clone() * &a3_inner);

        let a4_inner = e_d.clone() + e_y_pa - &(e_pa_y_addend.clone() * e_z_pa_cube);
        b.assert_zero(e_s_active.clone() * &a4_inner);

        let a5_inner = e_e.clone() - &(e_c.clone() * e_c);
        b.assert_zero(e_s_active.clone() * &a5_inner);

        let a6_inner = e_f.clone() - &(e_c.clone() * e_e);
        b.assert_zero(e_s_active.clone() * &a6_inner);

        let a7_inner = e_g.clone() - &(e_x_pa.clone() * e_e);
        b.assert_zero(e_s_active.clone() * &a7_inner);

        let d_sq = e_d.clone() * e_d;
        let two_g = mbs(e_g, &two_scalar).expect("2·G overflow");
        let a8_inner = e_x_add.clone() - &d_sq + e_f + &two_g;
        b.assert_zero(e_s_active.clone() * &a8_inner);

        let g_minus_xadd = e_g.clone() - e_x_add;
        let d_times = e_d.clone() * &g_minus_xadd;
        let y_pa_f = e_y_pa.clone() * e_f;
        let a9_inner = e_y_add.clone() - &d_times + &y_pa_f;
        b.assert_zero(e_s_active.clone() * &a9_inner);

        let a10_inner = e_z_add.clone() - &(e_z_pa.clone() * e_c);
        b.assert_zero(e_s_active.clone() * &a10_inner);

        // === Output selection (3 constraints) ===
        let x_diff = e_x_add.clone() - e_x_pa;
        let s_add_xdiff = e_s_add.clone() * &x_diff;
        let o1_inner = e_x_out.clone() - e_x_pa - &s_add_xdiff;
        b.assert_zero(e_s_active.clone() * &o1_inner);

        let y_diff = e_y_add.clone() - e_y_pa;
        let s_add_ydiff = e_s_add.clone() * &y_diff;
        let o2_inner = e_y_out.clone() - e_y_pa - &s_add_ydiff;
        b.assert_zero(e_s_active.clone() * &o2_inner);

        let z_diff = e_z_add.clone() - e_z_pa;
        let s_add_zdiff = e_s_add.clone() * &z_diff;
        let o3_inner = e_z_out.clone() - e_z_pa - &s_add_zdiff;
        b.assert_zero(e_s_active.clone() * &o3_inner);

        // === Init boundary (3 constraints) ===
        b.assert_zero(e_s_init.clone() * &(e_x1.clone() - e_pa_r_init_x));
        b.assert_zero(e_s_init.clone() * &(e_y1.clone() - e_pa_r_init_y));
        b.assert_zero(e_s_init.clone() * &(e_z1.clone() - e_pa_r_init_z));

        // === Row chaining (3 constraints) ===
        b.assert_zero(e_s_active.clone() * &(down_ecdsa_x1_sh1.clone() - e_x_out));
        b.assert_zero(e_s_active.clone() * &(down_ecdsa_y1_sh1.clone() - e_y_out));
        b.assert_zero(e_s_active.clone() * &(down_ecdsa_z1_sh1.clone() - e_z_out));

        // === Final-row affine conversion (5 constraints) ===
        let f1_inner = e_z1.clone() * e_z_inv - &one_expr;
        b.assert_zero(e_s_final.clone() * &f1_inner);

        let f2_inner = e_z_inv_sq.clone() - &(e_z_inv.clone() * e_z_inv);
        b.assert_zero(e_s_final.clone() * &f2_inner);

        let f3_inner = e_z_inv_cube.clone() - &(e_z_inv.clone() * e_z_inv_sq);
        b.assert_zero(e_s_final.clone() * &f3_inner);

        let f4_inner = e_x_aff.clone() - &(e_x1.clone() * e_z_inv_sq);
        b.assert_zero(e_s_final.clone() * &f4_inner);

        let f5_inner = e_y_aff.clone() - &(e_y1.clone() * e_z_inv_cube);
        b.assert_zero(e_s_final.clone() * &f5_inner);
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

fn mono_x_pow<R: ConstSemiring>(k: usize) -> DensePolynomial<R, 32> {
    let mut coeffs = [R::ZERO; 32];
    coeffs[k] = R::ONE;
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
        // SHA standalone int layout (11 cols):
        //   0..6   pubs (S_INIT, S_FINAL, PA_K, PA_C_C7/8/9)
        //   6..9   witnesses (mu_W, mu_a, mu_e)
        //   9..11  multiplicities (M_W2, M_W3)
        // ECDSA standalone int layout (34 cols):
        //   0..9   pubs
        //   9..34  witnesses
        let mut int: Vec<DenseMultilinearExtension<R>> = Vec::with_capacity(cols::NUM_INT);
        let sha_ints = sha_trace.int.into_owned();
        let ecdsa_ints = ecdsa_trace.int.into_owned();

        // [0..6] SHA pubs (sha[0..6])
        int.extend(sha_ints[0..6].iter().cloned());
        // [6..15] ECDSA pubs (ecdsa[0..9])
        int.extend(ecdsa_ints[0..9].iter().cloned());
        // [15..18] SHA witnesses (sha[6..9])
        int.extend(sha_ints[6..9].iter().cloned());
        // [18..43] ECDSA witnesses (ecdsa[9..34])
        int.extend(ecdsa_ints[9..34].iter().cloned());
        // [43..45] SHA multiplicities (sha[9..11])
        int.extend(sha_ints[9..11].iter().cloned());

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

    /// Sanity: 12 SHA + 28 ECDSA = 40 constraints. Max degree from
    /// either half (5 from ECDSA's doubling block).
    #[test]
    fn sha_ecdsa_constraint_shape() {
        type U = ShaEcdsaUair<Int<EC_FP_INT_LIMBS>>;
        assert_eq!(count_constraints::<U>(), 40);
        assert_eq!(count_max_degree::<U>(), 5);
        let degrees = count_constraint_degrees::<U>();
        // Spot checks: at least one deg-5 (doubling Y_pa), some deg-2
        // (boundaries + chaining), some deg-1 (SHA C1-C6 ideal checks).
        assert!(degrees.iter().any(|&d| d == 5), "expected deg-5 from doubling");
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
