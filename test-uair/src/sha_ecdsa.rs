//! Combined SHA-256 + ECDSA UAIR (side-by-side merge).
//!
//! Runs both `Sha256CompressionSliceUair` and `EcdsaScalarSliceUair` inside a
//! single UAIR on disjoint columns of the same trace. One proof attests to
//! both the SHA-256 compression/message-schedule round-trip **and** the
//! ECDSA scalar-side arithmetic (bit accumulation, inverse, signature
//! modular check).
//!
//! ## What this is (and isn't)
//!
//! This is the **structural** merge: the two existing UAIRs coexist without
//! any cross-binding between their outputs. In particular, the SHA digest
//! output (final `(h, g, f, e)` on the bit-poly side) is **not** bound to
//! the ECDSA message hash `pa_e` (on the int side). So the merged proof
//! says "some SHA compression happened, and some valid ECDSA scalar-side
//! computation happened" but not "the ECDSA input hash equals the SHA
//! output." Closing that gap is the next increment.
//!
//! ## Column layout
//!
//! Flat trace is (binary_poly || arbitrary_poly || int):
//! - binary_poly: 19 columns, all from SHA — same indices as
//!   `Sha256CompressionSliceUair`.
//! - arbitrary_poly: none.
//! - int: 27 columns total, publics first then witnesses:
//!     - [0..6)   SHA public int (S_INIT, S_ACTIVE, S_SCHED_ANCH, S_UPD_ANCH,
//!                S_FINAL, PA_K)
//!     - [6..12)  ECDSA public int (S_INIT, S_ACCUM, S_FINAL, PA_E, PA_R, PA_S)
//!     - [12..17) SHA witness int (MU_W, MU_A, MU_E, M_W2, M_W3)
//!     - [17..27) ECDSA witness int (B1, B2, U1, U2, W_INV, XHAT, K, Q_U1,
//!                                    Q_U2, Q_SW)
//!
//! Public column order is SHA-first then ECDSA-first so each slice's public
//! inputs live contiguously; the protocol only cares about the (pub_count,
//! total_count) split per section.
//!
//! ## Shifts and lookup specs
//!
//! Shifts are the union of both slices' shifts. Lookup specs are SHA's three
//! range checks on `mu_W`, `mu_a`, `mu_e`; ECDSA has none. Flat indices
//! account for the new int-section offsets.
//!
//! ## Ideals
//!
//! Reuses `Sha256Ideal<R>` as the `Uair::Ideal`. ECDSA never emits any ideal
//! membership (it uses `assert_zero` only), so the SHA ideal set is a
//! superset of what's needed.
//!
//! ## Selector compatibility
//!
//! Both slices use row-0 init, row-specific update, and end-of-trace final
//! selectors. Because they live in separate selector columns, no conflict.
//! The trace length constraint is the max of both slices' requirements:
//! SHA needs `n ≥ 16`; ECDSA needs `n ≥ 257`. Combined: `n ≥ 257` (so
//! `num_vars ≥ 9`).

use core::marker::PhantomData;

use crypto_bigint::{NonZero, Odd, Uint as CbUint};
use crypto_primitives::{ConstSemiring, crypto_bigint_int::Int};
use num_traits::Zero;
use rand::RngCore;
use zinc_poly::{
    mle::DenseMultilinearExtension,
    univariate::{binary::BinaryPoly, dense::DensePolynomial},
};
use zinc_uair::{
    AffineExpr, ConstraintBuilder, LookupColumnSpec, LookupTableType, PublicColumnLayout, ShiftSpec,
    TotalColumnLayout, TraceRow, Uair, UairSignature, UairTrace,
    ideal::rotation::RotationIdeal,
};

use crate::{
    GenerateRandomTrace,
    ecdsa::{ECDSA_INT_LIMBS, EcdsaScalarRing, SECP256K1_N_UINT},
    sha256::Sha256Ideal,
};

// ---------------------------------------------------------------------------
// Column indices within the merged flat trace.
// ---------------------------------------------------------------------------

pub mod cols {
    // ===== Binary-poly section (from SHA; unchanged indices) ==============
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

    // ===== Int section =====================================================
    // Order constraint from the protocol: the last `num_lookup_groups`
    // columns of the entire flat trace are the multiplicity columns (one
    // per group), per the convention documented in
    // `protocol::prover::step4b_lookup`. We have 2 groups → int[25] and
    // int[26] must be the multiplicity columns.
    //
    // SHA publics (0..6):
    pub const SHA_S_INIT: usize = 0;
    pub const SHA_S_ACTIVE: usize = 1;
    pub const SHA_S_SCHED_ANCH: usize = 2;
    pub const SHA_S_UPD_ANCH: usize = 3;
    pub const SHA_S_FINAL: usize = 4;
    pub const SHA_PA_K: usize = 5;
    // ECDSA publics (6..12):
    pub const ECDSA_S_INIT: usize = 6;
    pub const ECDSA_S_ACCUM: usize = 7;
    pub const ECDSA_S_FINAL: usize = 8;
    pub const ECDSA_PA_E: usize = 9;
    pub const ECDSA_PA_R: usize = 10;
    pub const ECDSA_PA_S: usize = 11;
    pub const NUM_INT_PUB: usize = 12;

    // SHA witnesses (12..15): carry-range values.
    pub const SHA_W_MU_W: usize = 12;
    pub const SHA_W_MU_A: usize = 13;
    pub const SHA_W_MU_E: usize = 14;
    // ECDSA witnesses (15..25):
    pub const ECDSA_W_B1: usize = 15;
    pub const ECDSA_W_B2: usize = 16;
    pub const ECDSA_W_U1: usize = 17;
    pub const ECDSA_W_U2: usize = 18;
    pub const ECDSA_W_W_INV: usize = 19;
    pub const ECDSA_W_XHAT: usize = 20;
    pub const ECDSA_W_K: usize = 21;
    pub const ECDSA_W_Q_U1: usize = 22;
    pub const ECDSA_W_Q_U2: usize = 23;
    pub const ECDSA_W_Q_SW: usize = 24;
    // Multiplicity columns (25..27) — MUST be the last two int cols.
    pub const SHA_W_M_W2: usize = 25; // group 0 (Word{width:2}, for mu_W)
    pub const SHA_W_M_W3: usize = 26; // group 1 (Word{width:3}, for mu_a/mu_e)
    pub const NUM_INT: usize = 27;

    // ===== Flat indices ====================================================
    // Binary-poly flats coincide with the bp-section indices (bp section
    // starts at flat 0).
    pub const FLAT_W_A: usize = W_A;
    pub const FLAT_W_SIG0: usize = W_SIG0;
    pub const FLAT_W_E: usize = W_E;
    pub const FLAT_W_SIG1: usize = W_SIG1;
    pub const FLAT_W_W: usize = W_W;
    pub const FLAT_W_LSIG0: usize = W_LSIG0;
    pub const FLAT_W_LSIG1: usize = W_LSIG1;
    pub const FLAT_W_CH: usize = W_CH;
    pub const FLAT_W_MAJ: usize = W_MAJ;

    // Int flats are offset by NUM_BIN (the bp section size).
    pub const FLAT_SHA_PA_K: usize = NUM_BIN + SHA_PA_K;
    pub const FLAT_SHA_W_MU_W: usize = NUM_BIN + SHA_W_MU_W;
    pub const FLAT_SHA_W_MU_A: usize = NUM_BIN + SHA_W_MU_A;
    pub const FLAT_SHA_W_MU_E: usize = NUM_BIN + SHA_W_MU_E;
    pub const FLAT_ECDSA_W_U1: usize = NUM_BIN + ECDSA_W_U1;
    pub const FLAT_ECDSA_W_U2: usize = NUM_BIN + ECDSA_W_U2;
}

/// The ECDSA final row (also where ECDSA's sig-modular and inverse
/// constraints apply). Mirrors `ecdsa::FINAL_ROW`.
pub const ECDSA_FINAL_ROW: usize = 256;

// ---------------------------------------------------------------------------
// The merged UAIR.
// ---------------------------------------------------------------------------

/// Combined SHA-256 + ECDSA UAIR (side-by-side merge). See module docs.
#[derive(Clone, Debug)]
pub struct ShaEcdsaUair<R>(PhantomData<R>);

impl<R> Uair for ShaEcdsaUair<R>
where
    R: EcdsaScalarRing + From<u32>,
{
    type Ideal = Sha256Ideal<R>;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(cols::NUM_BIN, 0, cols::NUM_INT);
        let public = PublicColumnLayout::new(cols::NUM_BIN_PUB, 0, cols::NUM_INT_PUB);
        // Shifts — union of SHA's and ECDSA's. UairSignature::new stably sorts
        // by source_col, so the explicit order here is for readability only.
        let shifts: Vec<ShiftSpec> = vec![
            // --- SHA binary_poly shifts (source_col ascending) ---
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
            // --- SHA int shifts (source_col ascending: PA_K, MU_W, MU_A, MU_E) ---
            ShiftSpec::new(cols::FLAT_SHA_PA_K, 3),
            ShiftSpec::new(cols::FLAT_SHA_W_MU_W, 16),
            ShiftSpec::new(cols::FLAT_SHA_W_MU_A, 3),
            ShiftSpec::new(cols::FLAT_SHA_W_MU_E, 3),
            // --- ECDSA int shifts ---
            ShiftSpec::new(cols::FLAT_ECDSA_W_U1, 1),
            ShiftSpec::new(cols::FLAT_ECDSA_W_U2, 1),
        ];
        // Lookup specs — just SHA's three range checks.
        let lookup_specs = vec![
            LookupColumnSpec {
                expression: AffineExpr::single(cols::FLAT_SHA_W_MU_W),
                table_type: LookupTableType::Word {
                    width: 2,
                    chunk_width: None,
                },
            },
            LookupColumnSpec {
                expression: AffineExpr::single(cols::FLAT_SHA_W_MU_A),
                table_type: LookupTableType::Word {
                    width: 3,
                    chunk_width: None,
                },
            },
            LookupColumnSpec {
                expression: AffineExpr::single(cols::FLAT_SHA_W_MU_E),
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
        _from_ref: FromR,
        mbs: MulByScalar,
        ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&Self::Scalar) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &Self::Scalar) -> Option<B::Expr>,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
    {
        // =====================================================================
        // SHA-256 section.
        // =====================================================================
        let bp = up.binary_poly;
        let ints = up.int;

        // Binary-poly columns (public + witness).
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

        // SHA selectors.
        let sha_s_init = &ints[cols::SHA_S_INIT];
        let sha_s_active = &ints[cols::SHA_S_ACTIVE];
        let sha_s_sched_anch = &ints[cols::SHA_S_SCHED_ANCH];
        let sha_s_upd_anch = &ints[cols::SHA_S_UPD_ANCH];
        let sha_s_final = &ints[cols::SHA_S_FINAL];

        // Down slots for SHA — order follows the stable sort-by-source_col.
        // bp slots (11 shifts in source_col order):
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
        // int slots (SHA 4, then ECDSA 2 — in source_col order):
        let down_pa_k_sh3 = &down.int[0];
        let down_w_mu_w_sh16 = &down.int[1];
        let down_w_mu_a_sh3 = &down.int[2];
        let down_w_mu_e_sh3 = &down.int[3];
        let down_ecdsa_u_1_sh1 = &down.int[4];
        let down_ecdsa_u_2_sh1 = &down.int[5];

        // SHA ideals and scalars.
        let ideal_rot_xw1 = ideal_from_ref(&Sha256Ideal::<R>::RotXw1);
        let ideal_rot_x2 = ideal_from_ref(&Sha256Ideal::<R>::RotX2(RotationIdeal::new(
            R::ONE + R::ONE,
        )));

        let rho_sig0 = rho_poly::<R>(&[10, 19, 30]);
        let rho_sig1 = rho_poly::<R>(&[7, 21, 26]);
        let rho_lsig0 = rho_poly::<R>(&[14, 25]);
        let rho_lsig1 = rho_poly::<R>(&[13, 15]);
        let two_scalar = const_scalar::<R>(R::ONE + R::ONE);
        let x_pow_3 = mono_x_pow::<R>(3);
        let x_pow_10 = mono_x_pow::<R>(10);
        let two_times_x31 = {
            let mut coeffs = [R::ZERO; 32];
            coeffs[31] = R::ONE + R::ONE;
            DensePolynomial::<R, 32>::new(coeffs)
        };

        // C1: Sigma_0 rotation (Q[X]-lifted).
        b.assert_in_ideal(
            sha_s_active.clone()
                * &(mbs(w_a, &rho_sig0).expect("a · rho_sig0 overflow") - w_sig0
                    - &mbs(w_ov_sig0, &two_scalar).expect("2 · ov_sig0 overflow")),
            &ideal_rot_xw1,
        );
        // C2: Sigma_1 rotation.
        b.assert_in_ideal(
            sha_s_active.clone()
                * &(mbs(w_e, &rho_sig1).expect("e · rho_sig1 overflow") - w_sig1
                    - &mbs(w_ov_sig1, &two_scalar).expect("2 · ov_sig1 overflow")),
            &ideal_rot_xw1,
        );
        // C3: sigma_0 right-shift decomposition (exact).
        b.assert_zero(
            sha_s_active.clone()
                * &(w_big_w.clone() - w_t0
                    - &mbs(w_s0, &x_pow_3).expect("X^3 · S_0 overflow")),
        );
        // C4: sigma_0 rotation.
        b.assert_in_ideal(
            sha_s_active.clone()
                * &(mbs(w_big_w, &rho_lsig0).expect("W · rho_lsig0 overflow") + w_s0
                    - w_lsig0
                    - &mbs(w_ov_lsig0, &two_scalar).expect("2 · ov_lsig0 overflow")),
            &ideal_rot_xw1,
        );
        // C5: sigma_1 right-shift decomposition.
        b.assert_zero(
            sha_s_active.clone()
                * &(w_big_w.clone() - w_t1
                    - &mbs(w_s1, &x_pow_10).expect("X^10 · S_1 overflow")),
        );
        // C6: sigma_1 rotation.
        b.assert_in_ideal(
            sha_s_active.clone()
                * &(mbs(w_big_w, &rho_lsig1).expect("W · rho_lsig1 overflow") + w_s1
                    - w_lsig1
                    - &mbs(w_ov_lsig1, &two_scalar).expect("2 · ov_lsig1 overflow")),
            &ideal_rot_xw1,
        );

        // C7: Message-schedule modular sum (anchor k = t − 16).
        let two_x31_mu_w =
            mbs(down_w_mu_w_sh16, &two_times_x31).expect("2·X^31 · mu_W overflow");
        let sched_inner = down_w_w_sh16.clone()
            - w_big_w
            - down_w_lsig0_sh1
            - down_w_w_sh9
            - down_w_lsig1_sh14
            + &two_x31_mu_w;
        b.assert_in_ideal(sha_s_sched_anch.clone() * &sched_inner, &ideal_rot_x2);

        // C8: Register-update for `a` (anchor k = t − 3).
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
        b.assert_in_ideal(sha_s_upd_anch.clone() * &a_update_inner, &ideal_rot_x2);

        // C9: Register-update for `e` (anchor k = t − 3).
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
        b.assert_in_ideal(sha_s_upd_anch.clone() * &e_update_inner, &ideal_rot_x2);

        // C10: Init boundary — a_hat[0] = pa_a[0].
        b.assert_zero(sha_s_init.clone() * &(w_a.clone() - pa_a));
        // C11: Final boundary (a-family).
        b.assert_zero(sha_s_final.clone() * &(w_a.clone() - pa_a));
        // C12: Final boundary (e-family).
        b.assert_zero(sha_s_final.clone() * &(w_e.clone() - pa_e));

        // =====================================================================
        // ECDSA section.
        // =====================================================================
        let e_s_init = &ints[cols::ECDSA_S_INIT];
        let e_s_accum = &ints[cols::ECDSA_S_ACCUM];
        let e_s_final = &ints[cols::ECDSA_S_FINAL];
        let e_pa_r = &ints[cols::ECDSA_PA_R];
        let e_pa_s = &ints[cols::ECDSA_PA_S];
        let e_b_1 = &ints[cols::ECDSA_W_B1];
        let e_b_2 = &ints[cols::ECDSA_W_B2];
        let e_u_1 = &ints[cols::ECDSA_W_U1];
        let e_u_2 = &ints[cols::ECDSA_W_U2];
        let e_w_inv = &ints[cols::ECDSA_W_W_INV];
        let e_x_hat = &ints[cols::ECDSA_W_XHAT];
        let e_k = &ints[cols::ECDSA_W_K];
        let e_q_u1 = &ints[cols::ECDSA_W_Q_U1];
        let e_q_u2 = &ints[cols::ECDSA_W_Q_U2];
        let e_q_sw = &ints[cols::ECDSA_W_Q_SW];

        let n_scalar = const_scalar::<R>(R::secp256k1_n());

        // E1: Bit range — b_i · (b_i − 1) == 0 (gated by s_accum).
        let b1_sq_minus_b1 = e_b_1.clone() * e_b_1 - e_b_1;
        b.assert_zero(e_s_accum.clone() * &b1_sq_minus_b1);
        let b2_sq_minus_b2 = e_b_2.clone() * e_b_2 - e_b_2;
        b.assert_zero(e_s_accum.clone() * &b2_sq_minus_b2);

        // E2: Scalar bit accumulation (mod n).
        let q_u1_times_n = mbs(e_q_u1, &n_scalar).expect("q_U1 · n overflow");
        let accum1_inner =
            e_u_1.clone() + e_u_1 + e_b_1 - down_ecdsa_u_1_sh1 - &q_u1_times_n;
        b.assert_zero(e_s_accum.clone() * &accum1_inner);

        let q_u2_times_n = mbs(e_q_u2, &n_scalar).expect("q_U2 · n overflow");
        let accum2_inner =
            e_u_2.clone() + e_u_2 + e_b_2 - down_ecdsa_u_2_sh1 - &q_u2_times_n;
        b.assert_zero(e_s_accum.clone() * &accum2_inner);

        // E3: Init boundary — U_1[0] = U_2[0] = 0.
        b.assert_zero(e_s_init.clone() * e_u_1);
        b.assert_zero(e_s_init.clone() * e_u_2);

        // E4: Scalar inverse at ECDSA_FINAL_ROW.
        let s_times_w = e_pa_s.clone() * e_w_inv;
        let s_final_sw = e_s_final.clone() * &s_times_w;
        let q_sw_times_n = mbs(e_q_sw, &n_scalar).expect("q_sw · n overflow");
        let s_final_q_sw_n = e_s_final.clone() * &q_sw_times_n;
        let inv_expr = s_final_sw - e_s_final - &s_final_q_sw_n;
        b.assert_zero(inv_expr);

        // E5: Signature modular check at ECDSA_FINAL_ROW.
        let k_sq_minus_k = e_k.clone() * e_k - e_k;
        b.assert_zero(e_s_final.clone() * &k_sq_minus_k);
        let k_times_n = mbs(e_k, &n_scalar).expect("k · n overflow");
        let sig_inner = e_x_hat.clone() - e_pa_r - &k_times_n;
        b.assert_zero(e_s_final.clone() * &sig_inner);
    }
}

// ---------------------------------------------------------------------------
// Scalar helpers (shared with the two sub-UAIRs but reimplemented here to
// avoid inter-module pub-vs-private friction).
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
// SHA reference helpers (for trace generation only).
// ---------------------------------------------------------------------------

#[inline]
fn rotr(x: u32, n: u32) -> u32 {
    x.rotate_right(n)
}
#[inline]
fn big_sigma0(x: u32) -> u32 {
    rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)
}
#[inline]
fn big_sigma1(x: u32) -> u32 {
    rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)
}
#[inline]
fn small_sigma0(x: u32) -> u32 {
    rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3)
}
#[inline]
fn small_sigma1(x: u32) -> u32 {
    rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10)
}
#[inline]
fn ch(x: u32, y: u32, z: u32) -> u32 {
    (x & y) ^ ((!x) & z)
}
#[inline]
fn maj(x: u32, y: u32, z: u32) -> u32 {
    (x & y) ^ (x & z) ^ (y & z)
}

fn rotation_overflow(
    input_bits: u32,
    rho_positions: &[usize],
    s0_bits: u32,
    out_bits: u32,
) -> u32 {
    let mut prod = [0u32; 64];
    for i in 0..32 {
        if (input_bits >> i) & 1 == 1 {
            for &p in rho_positions {
                prod[i + p] += 1;
            }
        }
    }
    let mut reduced = [0u32; 32];
    reduced.copy_from_slice(&prod[..32]);
    for k in 32..64 {
        reduced[k - 32] += prod[k];
    }
    for k in 0..32 {
        reduced[k] += (s0_bits >> k) & 1;
    }
    let mut overflow: u32 = 0;
    for k in 0..32 {
        let out_k = (out_bits >> k) & 1;
        let ov_k = (reduced[k] - out_k) / 2;
        overflow |= ov_k << k;
    }
    overflow
}
fn sigma0_overflow(a: u32, s: u32) -> u32 {
    rotation_overflow(a, &[10, 19, 30], 0, s)
}
fn sigma1_overflow(e: u32, s: u32) -> u32 {
    rotation_overflow(e, &[7, 21, 26], 0, s)
}
fn lsig0_overflow(w: u32, l: u32) -> u32 {
    rotation_overflow(w, &[14, 25], w >> 3, l)
}
fn lsig1_overflow(w: u32, l: u32) -> u32 {
    rotation_overflow(w, &[13, 15], w >> 10, l)
}

// ---------------------------------------------------------------------------
// ECDSA reference helpers.
// ---------------------------------------------------------------------------

fn ecdsa_rand_scalar<Rng: RngCore + ?Sized>(rng: &mut Rng) -> CbUint<ECDSA_INT_LIMBS> {
    let n_nz = NonZero::new(SECP256K1_N_UINT).expect("n is nonzero");
    loop {
        let mut limbs = [0u64; ECDSA_INT_LIMBS];
        for limb in &mut limbs {
            *limb = rng.next_u64();
        }
        limbs[ECDSA_INT_LIMBS - 1] = 0;
        let raw = CbUint::<ECDSA_INT_LIMBS>::from_words(limbs);
        let reduced = raw.rem_vartime(&n_nz);
        if !bool::from(reduced.is_zero()) {
            return reduced;
        }
    }
}
fn ecdsa_mul_mod_n(
    a: &CbUint<ECDSA_INT_LIMBS>,
    b: &CbUint<ECDSA_INT_LIMBS>,
) -> CbUint<ECDSA_INT_LIMBS> {
    let wide: CbUint<{ ECDSA_INT_LIMBS * 2 }> = a.widening_mul(b).into();
    let n_wide: CbUint<{ ECDSA_INT_LIMBS * 2 }> = SECP256K1_N_UINT.resize();
    let n_wide_nz = NonZero::new(n_wide).expect("n nonzero");
    let (_, rem) = wide.div_rem_vartime(&n_wide_nz);
    rem.resize()
}
fn ecdsa_inv_mod_n(a: &CbUint<ECDSA_INT_LIMBS>) -> CbUint<ECDSA_INT_LIMBS> {
    let n_odd = Odd::new(SECP256K1_N_UINT).expect("n is odd");
    a.invert_odd_mod(&n_odd).expect("a has no inverse mod n")
}
fn ecdsa_uint_to_int(u: CbUint<ECDSA_INT_LIMBS>) -> Int<ECDSA_INT_LIMBS> {
    debug_assert!(u.bits() <= 256);
    Int::new(*u.as_int())
}
fn ecdsa_u32_to_int(q: u32) -> Int<ECDSA_INT_LIMBS> {
    Int::<ECDSA_INT_LIMBS>::from(q)
}
fn ecdsa_extract_bits_be(u: &CbUint<ECDSA_INT_LIMBS>) -> Vec<u32> {
    let words = u.to_words();
    debug_assert_eq!(words[ECDSA_INT_LIMBS - 1], 0);
    let mut bits = Vec::with_capacity(256);
    for limb_idx in (0..4).rev() {
        let limb = words[limb_idx];
        for bit_idx in (0..64).rev() {
            bits.push(((limb >> bit_idx) & 1) as u32);
        }
    }
    bits
}

// ---------------------------------------------------------------------------
// GenerateRandomTrace.
// ---------------------------------------------------------------------------

impl<R> GenerateRandomTrace<32> for ShaEcdsaUair<R>
where
    R: EcdsaScalarRing + From<u32> + From<Int<ECDSA_INT_LIMBS>>,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_random_trace<Rng: RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, R, R, 32> {
        let n = 1usize << num_vars;
        assert!(
            n > ECDSA_FINAL_ROW,
            "ShaEcdsa UAIR needs > {ECDSA_FINAL_ROW} rows; got {n}"
        );

        // =====================================================================
        // SHA section — same logic as sha256::generate_random_trace.
        // =====================================================================
        let k_vals: Vec<u32> = (0..n).map(|_| rng.next_u32()).collect();

        let mut w_vals: Vec<u32> = (0..16).map(|_| rng.next_u32()).collect();
        let mut mu_w_vals: Vec<u32> = vec![0u32; 16];
        for t in 16..n {
            let sum_u64: u64 = (w_vals[t - 16] as u64)
                + (small_sigma0(w_vals[t - 15]) as u64)
                + (w_vals[t - 7] as u64)
                + (small_sigma1(w_vals[t - 2]) as u64);
            w_vals.push(sum_u64 as u32);
            mu_w_vals.push((sum_u64 >> 32) as u32);
        }

        let mut a_vals: Vec<u32> = Vec::with_capacity(n);
        let mut e_vals: Vec<u32> = Vec::with_capacity(n);
        for _ in 0..4 {
            a_vals.push(rng.next_u32());
            e_vals.push(rng.next_u32());
        }
        let mut mu_a_vals: Vec<u32> = vec![0u32; 3];
        let mut mu_e_vals: Vec<u32> = vec![0u32; 3];
        for t in 3..(n - 1) {
            let sig0_a_t = big_sigma0(a_vals[t]);
            let sig1_e_t = big_sigma1(e_vals[t]);
            let ch_t = ch(e_vals[t], e_vals[t - 1], e_vals[t - 2]);
            let maj_t = maj(a_vals[t], a_vals[t - 1], a_vals[t - 2]);
            let h_t = e_vals[t - 3];
            let d_t = a_vals[t - 3];
            let w_t = w_vals[t];
            let k_t = k_vals[t];
            let t1: u64 =
                (h_t as u64) + (sig1_e_t as u64) + (ch_t as u64) + (k_t as u64) + (w_t as u64);
            let t2: u64 = (sig0_a_t as u64) + (maj_t as u64);
            let a_sum: u64 = t1 + t2;
            let e_sum: u64 = (d_t as u64) + t1;
            a_vals.push(a_sum as u32);
            e_vals.push(e_sum as u32);
            mu_a_vals.push((a_sum >> 32) as u32);
            mu_e_vals.push((e_sum >> 32) as u32);
        }
        while mu_a_vals.len() < n {
            mu_a_vals.push(0);
        }
        while mu_e_vals.len() < n {
            mu_e_vals.push(0);
        }

        let ch_vals: Vec<u32> = (0..n)
            .map(|t| {
                if t >= 2 {
                    ch(e_vals[t], e_vals[t - 1], e_vals[t - 2])
                } else {
                    0
                }
            })
            .collect();
        let maj_vals: Vec<u32> = (0..n)
            .map(|t| {
                if t >= 2 {
                    maj(a_vals[t], a_vals[t - 1], a_vals[t - 2])
                } else {
                    0
                }
            })
            .collect();

        let sig0_vals: Vec<u32> = a_vals.iter().copied().map(big_sigma0).collect();
        let sig1_vals: Vec<u32> = e_vals.iter().copied().map(big_sigma1).collect();
        let lsig0_vals: Vec<u32> = w_vals.iter().copied().map(small_sigma0).collect();
        let lsig1_vals: Vec<u32> = w_vals.iter().copied().map(small_sigma1).collect();

        let ov_sig0_vals: Vec<u32> = a_vals
            .iter()
            .zip(&sig0_vals)
            .map(|(&a, &s)| sigma0_overflow(a, s))
            .collect();
        let ov_sig1_vals: Vec<u32> = e_vals
            .iter()
            .zip(&sig1_vals)
            .map(|(&e, &s)| sigma1_overflow(e, s))
            .collect();
        let ov_lsig0_vals: Vec<u32> = w_vals
            .iter()
            .zip(&lsig0_vals)
            .map(|(&w, &l)| lsig0_overflow(w, l))
            .collect();
        let ov_lsig1_vals: Vec<u32> = w_vals
            .iter()
            .zip(&lsig1_vals)
            .map(|(&w, &l)| lsig1_overflow(w, l))
            .collect();

        let s0_vals: Vec<u32> = w_vals.iter().map(|&w| w >> 3).collect();
        let t0_vals: Vec<u32> = w_vals.iter().map(|&w| w & 0b111).collect();
        let s1_vals: Vec<u32> = w_vals.iter().map(|&w| w >> 10).collect();
        let t1_vals: Vec<u32> = w_vals.iter().map(|&w| w & 0x3FF).collect();

        let mut pa_a_col: Vec<BinaryPoly<32>> =
            (0..n).map(|_| BinaryPoly::<32>::zero()).collect();
        pa_a_col[0] = BinaryPoly::<32>::from(a_vals[0]);
        for i in (n - 4)..n {
            pa_a_col[i] = BinaryPoly::<32>::from(a_vals[i]);
        }
        let mut pa_e_col: Vec<BinaryPoly<32>> =
            (0..n).map(|_| BinaryPoly::<32>::zero()).collect();
        for i in (n - 4)..n {
            pa_e_col[i] = BinaryPoly::<32>::from(e_vals[i]);
        }

        let to_bits = |v: &[u32]| -> Vec<BinaryPoly<32>> {
            v.iter().copied().map(BinaryPoly::<32>::from).collect()
        };
        let to_bin_mle = |col: Vec<BinaryPoly<32>>| -> DenseMultilinearExtension<BinaryPoly<32>> {
            col.into_iter().collect()
        };

        let binary_poly = vec![
            to_bin_mle(pa_a_col),
            to_bin_mle(pa_e_col),
            to_bin_mle(to_bits(&a_vals)),
            to_bin_mle(to_bits(&sig0_vals)),
            to_bin_mle(to_bits(&ov_sig0_vals)),
            to_bin_mle(to_bits(&e_vals)),
            to_bin_mle(to_bits(&sig1_vals)),
            to_bin_mle(to_bits(&ov_sig1_vals)),
            to_bin_mle(to_bits(&w_vals)),
            to_bin_mle(to_bits(&lsig0_vals)),
            to_bin_mle(to_bits(&s0_vals)),
            to_bin_mle(to_bits(&t0_vals)),
            to_bin_mle(to_bits(&ov_lsig0_vals)),
            to_bin_mle(to_bits(&lsig1_vals)),
            to_bin_mle(to_bits(&s1_vals)),
            to_bin_mle(to_bits(&t1_vals)),
            to_bin_mle(to_bits(&ov_lsig1_vals)),
            to_bin_mle(to_bits(&ch_vals)),
            to_bin_mle(to_bits(&maj_vals)),
        ];

        // SHA selectors.
        let mut sha_s_init_col: Vec<R> = (0..n).map(|_| R::ZERO).collect();
        sha_s_init_col[0] = R::ONE;
        let sha_s_active_col: Vec<R> = (0..n).map(|_| R::ONE).collect();
        let sched_anch_last = n - 17;
        let sha_s_sched_anch_col: Vec<R> = (0..n)
            .map(|i| if i <= sched_anch_last { R::ONE } else { R::ZERO })
            .collect();
        let upd_anch_last = n - 5;
        let sha_s_upd_anch_col: Vec<R> = (0..n)
            .map(|i| if i <= upd_anch_last { R::ONE } else { R::ZERO })
            .collect();
        let sha_s_final_col: Vec<R> = (0..n)
            .map(|i| if i + 4 >= n { R::ONE } else { R::ZERO })
            .collect();
        let sha_pa_k_col: Vec<R> = k_vals.iter().copied().map(R::from).collect();
        let sha_mu_w_col: Vec<R> = mu_w_vals.iter().copied().map(R::from).collect();
        let sha_mu_a_col: Vec<R> = mu_a_vals.iter().copied().map(R::from).collect();
        let sha_mu_e_col: Vec<R> = mu_e_vals.iter().copied().map(R::from).collect();

        // SHA multiplicity columns.
        let mut m_w2_raw = vec![0u32; n];
        for &v in &mu_w_vals {
            debug_assert!(v < 4);
            m_w2_raw[v as usize] += 1;
        }
        let mut m_w3_raw = vec![0u32; n];
        for &v in mu_a_vals.iter().chain(mu_e_vals.iter()) {
            debug_assert!(v < 8);
            m_w3_raw[v as usize] += 1;
        }
        let sha_m_w2_col: Vec<R> = m_w2_raw.into_iter().map(R::from).collect();
        let sha_m_w3_col: Vec<R> = m_w3_raw.into_iter().map(R::from).collect();

        // =====================================================================
        // ECDSA section — same logic as ecdsa::generate_random_trace.
        // =====================================================================
        let r_sig = ecdsa_rand_scalar(rng);
        let s_sig = ecdsa_rand_scalar(rng);
        let e_sig = ecdsa_rand_scalar(rng);
        let w_inv = ecdsa_inv_mod_n(&s_sig);
        let u_1 = ecdsa_mul_mod_n(&e_sig, &w_inv);
        let u_2 = ecdsa_mul_mod_n(&r_sig, &w_inv);

        let u_1_bits = ecdsa_extract_bits_be(&u_1);
        let u_2_bits = ecdsa_extract_bits_be(&u_2);

        let mut u1_seq: Vec<CbUint<ECDSA_INT_LIMBS>> = Vec::with_capacity(257);
        let mut u2_seq: Vec<CbUint<ECDSA_INT_LIMBS>> = Vec::with_capacity(257);
        u1_seq.push(CbUint::<ECDSA_INT_LIMBS>::ZERO);
        u2_seq.push(CbUint::<ECDSA_INT_LIMBS>::ZERO);
        let mut q_u1_seq: Vec<u32> = Vec::with_capacity(256);
        let mut q_u2_seq: Vec<u32> = Vec::with_capacity(256);
        let n_wide: CbUint<{ ECDSA_INT_LIMBS * 2 }> = SECP256K1_N_UINT.resize();
        for t in 0..256 {
            let step = |prev: &CbUint<ECDSA_INT_LIMBS>, bit: u32|
             -> (CbUint<ECDSA_INT_LIMBS>, u32) {
                let prev_wide: CbUint<{ ECDSA_INT_LIMBS * 2 }> = prev.resize();
                let two_prev = prev_wide.wrapping_shl(1);
                let bit_wide = CbUint::<{ ECDSA_INT_LIMBS * 2 }>::from_u32(bit);
                let sum = two_prev.wrapping_add(&bit_wide);
                let n_nz = NonZero::new(n_wide).expect("n nonzero");
                let (q, rem) = sum.div_rem_vartime(&n_nz);
                let q_u32 = q.to_words()[0] as u32;
                (rem.resize(), q_u32)
            };
            let (u1_next, q_u1_t) = step(&u1_seq[t], u_1_bits[t]);
            let (u2_next, q_u2_t) = step(&u2_seq[t], u_2_bits[t]);
            u1_seq.push(u1_next);
            u2_seq.push(u2_next);
            q_u1_seq.push(q_u1_t);
            q_u2_seq.push(q_u2_t);
        }

        let sw_wide: CbUint<{ ECDSA_INT_LIMBS * 2 }> = s_sig.widening_mul(&w_inv).into();
        let sw_minus_one = sw_wide.wrapping_sub(&CbUint::<{ ECDSA_INT_LIMBS * 2 }>::ONE);
        let n_nz_wide = NonZero::new(n_wide).expect("n nonzero");
        let (q_sw_wide, _) = sw_minus_one.div_rem_vartime(&n_nz_wide);
        let q_sw_uint: CbUint<ECDSA_INT_LIMBS> = q_sw_wide.resize();

        let x_hat = r_sig;
        let k_val: u32 = 0;

        let zero_r = || R::ZERO;
        let mk_col = |_: ()| -> Vec<R> { vec![zero_r(); n] };
        let mut e_s_init_col = mk_col(());
        let mut e_s_accum_col = mk_col(());
        let mut e_s_final_col = mk_col(());
        let mut e_pa_e_col = mk_col(());
        let mut e_pa_r_col = mk_col(());
        let mut e_pa_s_col = mk_col(());
        let mut e_b1_col = mk_col(());
        let mut e_b2_col = mk_col(());
        let mut e_u1_col = mk_col(());
        let mut e_u2_col = mk_col(());
        let mut e_w_col = mk_col(());
        let mut e_xhat_col = mk_col(());
        let mut e_k_col = mk_col(());
        let mut e_q_u1_col = mk_col(());
        let mut e_q_u2_col = mk_col(());
        let mut e_q_sw_col = mk_col(());

        let one_r = R::ONE;
        e_s_init_col[0] = one_r.clone();
        for t in 0..=255 {
            e_s_accum_col[t] = one_r.clone();
        }
        e_s_final_col[ECDSA_FINAL_ROW] = one_r.clone();
        e_pa_e_col[ECDSA_FINAL_ROW] = R::from(ecdsa_uint_to_int(e_sig));
        e_pa_r_col[ECDSA_FINAL_ROW] = R::from(ecdsa_uint_to_int(r_sig));
        e_pa_s_col[ECDSA_FINAL_ROW] = R::from(ecdsa_uint_to_int(s_sig));
        for t in 0..256 {
            e_b1_col[t] = R::from(ecdsa_u32_to_int(u_1_bits[t]));
            e_b2_col[t] = R::from(ecdsa_u32_to_int(u_2_bits[t]));
        }
        for t in 0..=256 {
            e_u1_col[t] = R::from(ecdsa_uint_to_int(u1_seq[t]));
            e_u2_col[t] = R::from(ecdsa_uint_to_int(u2_seq[t]));
        }
        for t in 0..256 {
            e_q_u1_col[t] = R::from(ecdsa_u32_to_int(q_u1_seq[t]));
            e_q_u2_col[t] = R::from(ecdsa_u32_to_int(q_u2_seq[t]));
        }
        e_w_col[ECDSA_FINAL_ROW] = R::from(ecdsa_uint_to_int(w_inv));
        e_xhat_col[ECDSA_FINAL_ROW] = R::from(ecdsa_uint_to_int(x_hat));
        e_k_col[ECDSA_FINAL_ROW] = R::from(ecdsa_u32_to_int(k_val));
        e_q_sw_col[ECDSA_FINAL_ROW] = R::from(ecdsa_uint_to_int(q_sw_uint));

        // =====================================================================
        // Assemble the int section. Order must match `cols::*` exactly.
        // =====================================================================
        let to_int_mle =
            |col: Vec<R>| -> DenseMultilinearExtension<R> { col.into_iter().collect() };

        let int = vec![
            // SHA publics (0..6):
            to_int_mle(sha_s_init_col),
            to_int_mle(sha_s_active_col),
            to_int_mle(sha_s_sched_anch_col),
            to_int_mle(sha_s_upd_anch_col),
            to_int_mle(sha_s_final_col),
            to_int_mle(sha_pa_k_col),
            // ECDSA publics (6..12):
            to_int_mle(e_s_init_col),
            to_int_mle(e_s_accum_col),
            to_int_mle(e_s_final_col),
            to_int_mle(e_pa_e_col),
            to_int_mle(e_pa_r_col),
            to_int_mle(e_pa_s_col),
            // SHA witnesses (12..15):
            to_int_mle(sha_mu_w_col),
            to_int_mle(sha_mu_a_col),
            to_int_mle(sha_mu_e_col),
            // ECDSA witnesses (15..25):
            to_int_mle(e_b1_col),
            to_int_mle(e_b2_col),
            to_int_mle(e_u1_col),
            to_int_mle(e_u2_col),
            to_int_mle(e_w_col),
            to_int_mle(e_xhat_col),
            to_int_mle(e_k_col),
            to_int_mle(e_q_u1_col),
            to_int_mle(e_q_u2_col),
            to_int_mle(e_q_sw_col),
            // Multiplicity cols (25..27) — last N int cols by convention:
            to_int_mle(sha_m_w2_col),
            to_int_mle(sha_m_w3_col),
        ];

        UairTrace {
            binary_poly: binary_poly.into(),
            int: int.into(),
            ..Default::default()
        }
    }
}
