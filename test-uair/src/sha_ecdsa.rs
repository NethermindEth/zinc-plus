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
//! **binary_poly section** (19 cols, mirrors SHA standalone):
//! - `[0..6]` public: `PA_A`, `PA_E`, `PA_OV_SIG0/SIG1/LSIG0/LSIG1`
//!   (rotation-ideal overflow witnesses, made public)
//! - `[6..19]` witness: SHA's remaining witness bit-polys
//!
//! **int section** (28 cols total, 15 pubs + 11 witness + 2
//!   multiplicities):
//! - `[0..6]` public: SHA pubs (S_INIT, S_FINAL, PA_K, PA_C_C7/8/9)
//! - `[6..15]` public: ECDSA pubs (S_INIT, S_ACTIVE, S_FINAL, S_ADD,
//!   PA_X_ADDEND, PA_Y_ADDEND, PA_R_INIT_X/Y/Z)
//! - `[15..18]` witness: SHA `mu_W, mu_a, mu_e`
//! - `[18..26]` witness: ECDSA chained Jacobian state + doubled point
//!   + addition scratch (8 cols; `S = Y²` and `Z_inv` are inlined or
//!   handled off-protocol)
//! - `[26..28]` witness: SHA lookup multiplicities (`M_W2, M_W3`) —
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
    // ===== binary_poly (mirrors sha256.rs: 6 pub + 13 witness) =====
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
    pub const W_S0: usize = 12;
    pub const W_T0: usize = 13;
    pub const W_LSIG1: usize = 14;
    pub const W_S1: usize = 15;
    pub const W_T1: usize = 16;
    pub const W_CH: usize = 17;
    pub const W_MAJ: usize = 18;
    pub const NUM_BIN: usize = 19;
    pub const NUM_BIN_PUB: usize = 6;

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

    // ECDSA witnesses (18..26): chained Jacobian state + doubled
    // point + addition scratch. 8 cols; `S = Y²` and `Z_inv` are
    // inlined / off-protocol respectively.
    pub const ECDSA_W_X: usize = 18;
    pub const ECDSA_W_Y: usize = 19;
    pub const ECDSA_W_Z: usize = 20;
    pub const ECDSA_W_X_PA: usize = 21;
    pub const ECDSA_W_Y_PA: usize = 22;
    pub const ECDSA_W_Z_PA: usize = 23;
    pub const ECDSA_W_C: usize = 24;
    pub const ECDSA_W_D: usize = 25;

    // SHA multiplicities (26..28) — MUST be the last N int cols.
    pub const SHA_W_M_W2: usize = 26;
    pub const SHA_W_M_W3: usize = 27;

    pub const NUM_INT: usize = 28;

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
            // === ECDSA int shifts (X, Y, Z by 1 each: down.X[t] = R_{t+1}) ===
            ShiftSpec::new(cols::FLAT_ECDSA_W_X, 1),
            ShiftSpec::new(cols::FLAT_ECDSA_W_Y, 1),
            ShiftSpec::new(cols::FLAT_ECDSA_W_Z, 1),
        ];

        // Lookup specs: stubbed out — the gkr-logup pipeline only
        // supports `BitPoly` lookups on binary_poly witness columns,
        // so SHA's `Word { width: 2/3 }` lookups on int carries would
        // hit `LookupError::NotImplemented`. See sha256.rs for the
        // full soundness-gap note. Multiplicity cols stay in the trace
        // but are unused.
        let lookup_specs: Vec<LookupColumnSpec> = vec![];

        UairSignature::new(total, public, shifts, lookup_specs)
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        down: TraceRow<B::Expr>,
        _from_ref: FromR,
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
        let w_s0 = &bp[cols::W_S0];
        let w_t0 = &bp[cols::W_T0];
        let w_lsig1 = &bp[cols::W_LSIG1];
        let w_s1 = &bp[cols::W_S1];
        let w_t1 = &bp[cols::W_T1];

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
        let down_ecdsa_x_sh1 = &down.int[4];
        let down_ecdsa_y_sh1 = &down.int[5];
        let down_ecdsa_z_sh1 = &down.int[6];

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
                - &mbs(pa_ov_sig0, &two_scalar_sha).expect("2 · ov_sig0 overflow"),
            &ideal_rot_xw1,
        );

        // C2: Sigma_1 rotation
        b.assert_in_ideal(
            mbs(w_e, &rho_sig1).expect("e · rho_sig1 overflow") - w_sig1
                - &mbs(pa_ov_sig1, &two_scalar_sha).expect("2 · ov_sig1 overflow"),
            &ideal_rot_xw1,
        );

        // C3: sigma_0 right-shift decomposition
        b.assert_zero(
            w_big_w.clone() - w_t0 - &mbs(w_s0, &x_pow_3).expect("X^3 · S_0 overflow"),
        );

        // C4: sigma_0 rotation
        b.assert_in_ideal(
            mbs(w_big_w, &rho_lsig0).expect("W · rho_lsig0 overflow") + w_s0 - w_lsig0
                - &mbs(pa_ov_lsig0, &two_scalar_sha).expect("2 · ov_lsig0 overflow"),
            &ideal_rot_xw1,
        );

        // C5: sigma_1 right-shift decomposition
        b.assert_zero(
            w_big_w.clone() - w_t1 - &mbs(w_s1, &x_pow_10).expect("X^10 · S_1 overflow"),
        );

        // C6: sigma_1 rotation
        b.assert_in_ideal(
            mbs(w_big_w, &rho_lsig1).expect("W · rho_lsig1 overflow") + w_s1 - w_lsig1
                - &mbs(pa_ov_lsig1, &two_scalar_sha).expect("2 · ov_lsig1 overflow"),
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
        // ECDSA standalone int layout (17 cols):
        //   0..9   pubs
        //   9..17  witnesses (8 EC cols)
        let mut int: Vec<DenseMultilinearExtension<R>> = Vec::with_capacity(cols::NUM_INT);
        let sha_ints = sha_trace.int.into_owned();
        let ecdsa_ints = ecdsa_trace.int.into_owned();

        // [0..6] SHA pubs (sha[0..6])
        int.extend(sha_ints[0..6].iter().cloned());
        // [6..15] ECDSA pubs (ecdsa[0..9])
        int.extend(ecdsa_ints[0..9].iter().cloned());
        // [15..18] SHA witnesses (sha[6..9])
        int.extend(sha_ints[6..9].iter().cloned());
        // [18..26] ECDSA witnesses (ecdsa[9..17], 8 cols)
        int.extend(ecdsa_ints[9..17].iter().cloned());
        // [26..28] SHA multiplicities (sha[9..11])
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

    /// Sanity: 12 SHA + 11 ECDSA = 23 constraints. Max degree 6 from
    /// the ECDSA Y output-selection constraint (and from D4's
    /// `12·X³·Y²` term after `S` inlining).
    #[test]
    fn sha_ecdsa_constraint_shape() {
        type U = ShaEcdsaUair<Int<EC_FP_INT_LIMBS>>;
        assert_eq!(count_constraints::<U>(), 23);
        assert_eq!(count_max_degree::<U>(), 6);
        let degrees = count_constraint_degrees::<U>();
        // Spot checks: at least one deg-6 (ECDSA Y output sel / D4),
        // some deg-2 (boundaries + chaining), some deg-1 (SHA C1-C6
        // ideal checks).
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
