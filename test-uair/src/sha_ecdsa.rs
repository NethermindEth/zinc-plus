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
    PublicStructureError, ShiftSpec, ShiftedBitSliceSpec, TotalColumnLayout, TraceRow, Uair,
    UairSignature, UairTrace, VirtualBinaryPolySource, VirtualBinaryPolySpec,
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
    // ===== binary_poly (mirrors sha256.rs: 8 pub + 10 witness) =====
    // OV cols are public — overflow witnesses for the rotation-ideal
    // constraints (C1, C2, C4, C6) are verifier-derivable. PA_R_*_CORR
    // are public boundary correctors for the Ch (63) / Maj (64) virtual
    // binary_poly residuals (see sha256.rs doc).
    pub const PA_A: usize = 0;
    pub const PA_E: usize = 1;
    pub const PA_OV_SIG0: usize = 2;
    pub const PA_OV_SIG1: usize = 3;
    pub const PA_OV_LSIG0: usize = 4;
    pub const PA_OV_LSIG1: usize = 5;
    pub const PA_R_CH2_CORR: usize = 6;
    pub const PA_R_MAJ_CORR: usize = 7;
    // Public message-block words (Table 9 row 77). See sha256.rs cols
    // doc for the layout.
    pub const PA_M: usize = 8;
    pub const W_A: usize = 9;
    pub const W_SIG0: usize = 10;
    pub const W_E: usize = 11;
    pub const W_SIG1: usize = 12;
    pub const W_W: usize = 13;
    pub const W_LSIG0: usize = 14;
    pub const W_LSIG1: usize = 15;
    // Ch is split into two AND-operand bit-polys (see sha256.rs doc).
    pub const W_U_EF: usize = 16;
    pub const W_U_NEG_E_G: usize = 17;
    pub const W_MAJ: usize = 18;
    // Packed integer-carry witness column. Replaces the 5 prior int
    // carry columns. See sha256.rs cols doc for bit layout +
    // soundness argument.
    pub const W_MU_PACKED: usize = 19;
    // The Table 9 affine combinations B_1 / B_2 / B_3 are now declared
    // as packed virtual binary_poly columns in `signature()` via
    // `with_virtual_binary_poly_cols` — no committed columns.
    pub const NUM_BIN: usize = 20;
    pub const NUM_BIN_PUB: usize = 9;

    // ===== int section =====
    // SHA publics (0..9) — see sha256.rs cols module for chained-
    // compression layout details. S_INIT_PREFIX/S_FEEDFORWARD replace
    // the old S_INIT/S_FINAL pair. PA_C_FF_A/E are the feed-forward
    // compensators (added so C12/C13 stay degree-1 in the trace MLEs,
    // preserving MLE-first eligibility — see sha256.rs cols doc).
    // S_MSG_INIT gates the C16 message-init pinning constraint.
    //
    // The two SHA compensator-zero selector columns
    // (SHA_S_ACTIVE_SCHED/SHA_S_ACTIVE_UPD) are gone — compensator-zero
    // is enforced by direct verifier inspection of public_trace via
    // `verify_public_structure` (see sha256.rs doc + the SHA UAIR doc).
    pub const SHA_S_INIT_PREFIX: usize = 0;
    pub const SHA_S_FEEDFORWARD: usize = 1;
    pub const SHA_S_MSG_INIT: usize = 2;
    pub const SHA_PA_K: usize = 3;
    pub const SHA_PA_C_C7: usize = 4;
    pub const SHA_PA_C_C8: usize = 5;
    pub const SHA_PA_C_C9: usize = 6;
    pub const SHA_PA_C_FF_A: usize = 7;
    pub const SHA_PA_C_FF_E: usize = 8;
    // ECDSA publics (9..18)
    pub const ECDSA_S_INIT: usize = 9;
    pub const ECDSA_S_ACTIVE: usize = 10;
    pub const ECDSA_S_FINAL: usize = 11;
    pub const ECDSA_S_ADD: usize = 12;
    pub const ECDSA_PA_X_ADDEND: usize = 13;
    pub const ECDSA_PA_Y_ADDEND: usize = 14;
    pub const ECDSA_PA_R_INIT_X: usize = 15;
    pub const ECDSA_PA_R_INIT_Y: usize = 16;
    pub const ECDSA_PA_R_INIT_Z: usize = 17;
    // SHA_S_B_ACTIVE is gone — the Table 9 materialization constraints
    // (C13–C15) are dropped, replaced by virtual binary_poly residuals
    // pinned via the booleanity sumcheck (see sha256.rs doc).
    pub const NUM_INT_PUB: usize = 18;

    // The 5 prior SHA int carry columns (W_MU_W/A/E/JUNCTION_A/E) are
    // gone — replaced by W_MU_PACKED (binary_poly index 19), with
    // booleanity providing free range-checks. See sha256.rs cols doc.

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

    pub const NUM_INT: usize = 26;

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
    pub const FLAT_SHA_PA_K: usize = NUM_BIN + SHA_PA_K;
    pub const FLAT_W_MU_PACKED: usize = W_MU_PACKED;
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
            // === SHA int shifts: only PA_K survives. The 3 mu_*
            //     shift specs are gone alongside the dropped int carry
            //     columns — carries now packed in W_MU_PACKED, accessed
            //     via BitOp::ShiftR virtuals declared below.
            ShiftSpec::new(cols::FLAT_SHA_PA_K, 3),
            // === ECDSA int shifts (X, Y, Z by 1 each: down.X[t] = R_{t+1}) ===
            ShiftSpec::new(cols::FLAT_ECDSA_W_X, 1),
            ShiftSpec::new(cols::FLAT_ECDSA_W_Y, 1),
            ShiftSpec::new(cols::FLAT_ECDSA_W_Z, 1),
        ];

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
            // Bit-op virtuals over W_MU_PACKED for extracting the 5
            // chained-comp carries. See sha256.rs cols doc.
            BitOpSpec::new(cols::FLAT_W_MU_PACKED, BitOp::ShiftR(2)),
            BitOpSpec::new(cols::FLAT_W_MU_PACKED, BitOp::ShiftR(5)),
            BitOpSpec::new(cols::FLAT_W_MU_PACKED, BitOp::ShiftR(8)),
            BitOpSpec::new(cols::FLAT_W_MU_PACKED, BitOp::ShiftR(9)),
            BitOpSpec::new(cols::FLAT_W_MU_PACKED, BitOp::ShiftR(10)),
        ];

        // Witness-relative col indices (post-public) for virtual specs.
        const W_A_WIT_IDX: usize = cols::W_A - cols::NUM_BIN_PUB; // 0
        const W_E_WIT_IDX: usize = cols::W_E - cols::NUM_BIN_PUB; // 2
        const W_U_EF_WIT_IDX: usize = cols::W_U_EF - cols::NUM_BIN_PUB; // 7
        const W_U_NEG_E_G_WIT_IDX: usize = cols::W_U_NEG_E_G - cols::NUM_BIN_PUB; // 8
        const W_MAJ_WIT_IDX: usize = cols::W_MAJ - cols::NUM_BIN_PUB; // 9
        // Order = the spec_idx that VirtualBinaryPolySource uses (must
        // match the ShiftSpec ordering above; UairSignature::new sorts
        // shifts by source_col, then shift_amount).
        const SBS_W_A_SH1: usize = 0;
        const SBS_W_A_SH2: usize = 1;
        const SBS_W_E_SH1: usize = 2;
        const SBS_W_E_SH2: usize = 3;
        const SBS_W_U_EF_SH2: usize = 4;
        const SBS_W_U_NEG_E_G_SH2: usize = 5;
        const SBS_W_MAJ_SH2: usize = 6;
        let shifted_bit_slice_specs = vec![
            ShiftedBitSliceSpec::new(W_A_WIT_IDX, 1),
            ShiftedBitSliceSpec::new(W_A_WIT_IDX, 2),
            ShiftedBitSliceSpec::new(W_E_WIT_IDX, 1),
            ShiftedBitSliceSpec::new(W_E_WIT_IDX, 2),
            ShiftedBitSliceSpec::new(W_U_EF_WIT_IDX, 2),
            ShiftedBitSliceSpec::new(W_U_NEG_E_G_WIT_IDX, 2),
            ShiftedBitSliceSpec::new(W_MAJ_WIT_IDX, 2),
        ];
        // Virtual binary_poly cols — mirrors sha256.rs (Ch eq 62/63 and
        // Maj eq 64, all anchored at k = t-2). See sha256.rs's signature
        // for the residual definitions and the alt-complement form.
        let virtual_binary_poly_cols = vec![
            VirtualBinaryPolySpec {
                terms: vec![
                    (
                        1,
                        VirtualBinaryPolySource::ShiftedWitnessCol {
                            shifted_spec_idx: SBS_W_E_SH2,
                        },
                    ),
                    (
                        1,
                        VirtualBinaryPolySource::ShiftedWitnessCol {
                            shifted_spec_idx: SBS_W_E_SH1,
                        },
                    ),
                    (
                        -2,
                        VirtualBinaryPolySource::ShiftedWitnessCol {
                            shifted_spec_idx: SBS_W_U_EF_SH2,
                        },
                    ),
                ],
            },
            VirtualBinaryPolySpec {
                terms: vec![
                    (
                        1,
                        VirtualBinaryPolySource::ShiftedWitnessCol {
                            shifted_spec_idx: SBS_W_E_SH2,
                        },
                    ),
                    (
                        -1,
                        VirtualBinaryPolySource::SelfWitnessCol {
                            witness_col_idx: W_E_WIT_IDX,
                        },
                    ),
                    (
                        2,
                        VirtualBinaryPolySource::ShiftedWitnessCol {
                            shifted_spec_idx: SBS_W_U_NEG_E_G_SH2,
                        },
                    ),
                    (
                        2,
                        VirtualBinaryPolySource::PublicCol {
                            public_col_idx: cols::PA_R_CH2_CORR,
                        },
                    ),
                ],
            },
            VirtualBinaryPolySpec {
                terms: vec![
                    (
                        1,
                        VirtualBinaryPolySource::SelfWitnessCol {
                            witness_col_idx: W_A_WIT_IDX,
                        },
                    ),
                    (
                        1,
                        VirtualBinaryPolySource::ShiftedWitnessCol {
                            shifted_spec_idx: SBS_W_A_SH1,
                        },
                    ),
                    (
                        1,
                        VirtualBinaryPolySource::ShiftedWitnessCol {
                            shifted_spec_idx: SBS_W_A_SH2,
                        },
                    ),
                    (
                        -2,
                        VirtualBinaryPolySource::ShiftedWitnessCol {
                            shifted_spec_idx: SBS_W_MAJ_SH2,
                        },
                    ),
                    (
                        -2,
                        VirtualBinaryPolySource::PublicCol {
                            public_col_idx: cols::PA_R_MAJ_CORR,
                        },
                    ),
                ],
            },
        ];
        UairSignature::new(total, public, shifts, lookup_specs, bit_op_specs)
            .with_shifted_bit_slice_specs(shifted_bit_slice_specs)
            .with_virtual_binary_poly_cols(virtual_binary_poly_cols)
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
        let pa_m = &bp[cols::PA_M];
        let w_a = &bp[cols::W_A];
        let w_sig0 = &bp[cols::W_SIG0];
        let w_e = &bp[cols::W_E];
        let w_sig1 = &bp[cols::W_SIG1];
        let w_big_w = &bp[cols::W_W];
        let w_lsig0 = &bp[cols::W_LSIG0];
        let w_lsig1 = &bp[cols::W_LSIG1];

        let sha_s_init_prefix = &int[cols::SHA_S_INIT_PREFIX];
        // SHA_S_FEEDFORWARD is no longer multiplied into any in-circuit
        // constraint (FF compensator-zero pins moved to verifier-side
        // `verify_public_structure`); it remains as a public selector
        // documenting the junction-window row pattern.
        let _sha_s_feedforward = &int[cols::SHA_S_FEEDFORWARD];
        let sha_s_msg_init = &int[cols::SHA_S_MSG_INIT];
        let pa_c_c7 = &int[cols::SHA_PA_C_C7];
        let pa_c_c8 = &int[cols::SHA_PA_C_C8];
        let pa_c_c9 = &int[cols::SHA_PA_C_C9];
        let pa_c_ff_a = &int[cols::SHA_PA_C_FF_A];
        let pa_c_ff_e = &int[cols::SHA_PA_C_FF_E];
        // The 5 prior int carry columns are gone — replaced by
        // W_MU_PACKED (binary_poly), accessed below via `up.bp[W_MU_PACKED]`
        // and the BitOp::ShiftR virtuals.
        let w_mu_packed = &bp[cols::W_MU_PACKED];

        // SHA `down` slots (in source-col-ascending order — see signature()).
        // bin slots (19 SHA shifts, then 0 ECDSA shifts on bin). The
        // sh1/sh2 entries on a/e/u_ef/u_¬e_g/Maj are kept in the shift
        // list to feed the booleanity batch's shifted-bit-slice
        // consistency check (declared via `with_shifted_bit_slice_specs`)
        // — they're not consumed by `constrain_general` directly.
        let _down_w_a_sh1 = &down.binary_poly[0];
        let _down_w_a_sh2 = &down.binary_poly[1];
        let down_w_a_sh4 = &down.binary_poly[2];
        let down_w_sig0_sh3 = &down.binary_poly[3];
        let _down_w_e_sh1 = &down.binary_poly[4];
        let _down_w_e_sh2 = &down.binary_poly[5];
        let down_w_e_sh4 = &down.binary_poly[6];
        let down_w_sig1_sh3 = &down.binary_poly[7];
        let down_w_w_sh3 = &down.binary_poly[8];
        let down_w_w_sh9 = &down.binary_poly[9];
        let down_w_w_sh16 = &down.binary_poly[10];
        let down_w_lsig0_sh1 = &down.binary_poly[11];
        let down_w_lsig1_sh14 = &down.binary_poly[12];
        let _down_w_u_ef_sh2 = &down.binary_poly[13];
        let down_w_u_ef_sh3 = &down.binary_poly[14];
        let _down_w_u_neg_e_g_sh2 = &down.binary_poly[15];
        let down_w_u_neg_e_g_sh3 = &down.binary_poly[16];
        let _down_w_maj_sh2 = &down.binary_poly[17];
        let down_w_maj_sh3 = &down.binary_poly[18];
        // int slots: SHA pa_K_sh3 (slot 0), then ECDSA X/Y/Z sh1 (1, 2, 3).
        // The 3 prior SHA mu_* int shifts are gone alongside the
        // dropped int carry columns.
        let down_pa_k_sh3 = &down.int[0];
        let down_ecdsa_x_sh1 = &down.int[1];
        let down_ecdsa_y_sh1 = &down.int[2];
        let down_ecdsa_z_sh1 = &down.int[3];

        // Bit-op virtual columns. With FLAT_W_W < FLAT_W_MU_PACKED,
        // W's 6 bit-ops occupy slots 0-5, then W_MU_PACKED's 5
        // bit-ops occupy slots 6-10.
        let down_w_rot13 = &down.bit_op[0]; // σ_1: ROTR^19
        let down_w_rot14 = &down.bit_op[1]; // σ_0: ROTR^18
        let down_w_rot15 = &down.bit_op[2]; // σ_1: ROTR^17
        let down_w_rot25 = &down.bit_op[3]; // σ_0: ROTR^7
        let down_w_shr3 = &down.bit_op[4]; //  σ_0: SHR^3
        let down_w_shr10 = &down.bit_op[5]; // σ_1: SHR^10
        // Bit-extraction shifts on W_MU_PACKED.
        let down_w_mu_packed_shr2 = &down.bit_op[6];
        let down_w_mu_packed_shr5 = &down.bit_op[7];
        let down_w_mu_packed_shr8 = &down.bit_op[8];
        let down_w_mu_packed_shr9 = &down.bit_op[9];
        let down_w_mu_packed_shr10 = &down.bit_op[10];

        let ideal_rot_xw1 = ideal_from_ref(&Sha256Ideal::<R>::RotXw1);
        let ideal_rot_x2 = ideal_from_ref(&Sha256Ideal::<R>::RotX2(RotationIdeal::new(
            R::ONE + R::ONE,
        )));

        let rho_sig0 = rho_poly::<R>(&[10, 19, 30]);
        let rho_sig1 = rho_poly::<R>(&[7, 21, 26]);
        let two_scalar_sha = const_scalar::<R>(R::ONE + R::ONE);
        // Carry-extraction multipliers (mirror sha256.rs). Each
        // contribution `2^32 · mu_X` is built from
        // `2^32 · ShiftR(k_low) − 2^{32+w} · ShiftR(k_low+w)`.
        let const_2_to_32 = const_scalar::<R>(pow_two::<R>(32));
        let const_2_to_33 = const_scalar::<R>(pow_two::<R>(33));
        let const_2_to_34 = const_scalar::<R>(pow_two::<R>(34));
        let const_2_to_35 = const_scalar::<R>(pow_two::<R>(35));

        let mu_w_contrib = mbs(w_mu_packed, &const_2_to_32)
            .expect("2^32 · w_mu_packed overflow")
            - &mbs(down_w_mu_packed_shr2, &const_2_to_34)
                .expect("2^34 · ShiftR(2)(w_mu_packed) overflow");
        let mu_a_contrib = mbs(down_w_mu_packed_shr2, &const_2_to_32)
            .expect("2^32 · ShiftR(2)(w_mu_packed) overflow")
            - &mbs(down_w_mu_packed_shr5, &const_2_to_35)
                .expect("2^35 · ShiftR(5)(w_mu_packed) overflow");
        let mu_e_contrib = mbs(down_w_mu_packed_shr5, &const_2_to_32)
            .expect("2^32 · ShiftR(5)(w_mu_packed) overflow")
            - &mbs(down_w_mu_packed_shr8, &const_2_to_35)
                .expect("2^35 · ShiftR(8)(w_mu_packed) overflow");
        let mu_ff_a_contrib = mbs(down_w_mu_packed_shr8, &const_2_to_32)
            .expect("2^32 · ShiftR(8)(w_mu_packed) overflow")
            - &mbs(down_w_mu_packed_shr9, &const_2_to_33)
                .expect("2^33 · ShiftR(9)(w_mu_packed) overflow");
        let mu_ff_e_contrib = mbs(down_w_mu_packed_shr9, &const_2_to_32)
            .expect("2^32 · ShiftR(9)(w_mu_packed) overflow")
            - &mbs(down_w_mu_packed_shr10, &const_2_to_33)
                .expect("2^33 · ShiftR(10)(w_mu_packed) overflow");

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

        // C7: Message-schedule modular sum. mu_W from up.w_mu_packed
        // bits 0-1 via mu_w_contrib (chained-comp re-anchoring stores
        // each carry at its constraint's anchor row).
        let sched_inner = down_w_w_sh16.clone()
            - w_big_w
            - down_w_lsig0_sh1
            - down_w_w_sh9
            - down_w_lsig1_sh14
            + &mu_w_contrib;
        b.assert_in_ideal(sched_inner + pa_c_c7, &ideal_rot_x2);

        // C8: Register-update for `a`. mu_a from bits 2-4 of W_MU_PACKED.
        let a_update_inner = down_w_a_sh4.clone()
            - w_e
            - down_w_sig1_sh3
            - down_w_u_ef_sh3
            - down_w_u_neg_e_g_sh3
            - down_pa_k_sh3
            - down_w_w_sh3
            - down_w_sig0_sh3
            - down_w_maj_sh3
            + &mu_a_contrib;
        b.assert_in_ideal(a_update_inner + pa_c_c8, &ideal_rot_x2);

        // C9: Register-update for `e`. mu_e from bits 5-7 of W_MU_PACKED.
        let e_update_inner = down_w_e_sh4.clone()
            - w_a
            - w_e
            - down_w_sig1_sh3
            - down_w_u_ef_sh3
            - down_w_u_neg_e_g_sh3
            - down_pa_k_sh3
            - down_w_w_sh3
            + &mu_e_contrib;
        b.assert_in_ideal(e_update_inner + pa_c_c9, &ideal_rot_x2);

        // C13–C15 (B_1/B_2/B_3 materializations) are gone — the
        // residuals are now packed virtual binary_poly columns,
        // declared in `signature()` via `with_virtual_binary_poly_cols`
        // and pinned by the booleanity sumcheck. See sha256.rs's
        // `signature()` for the residual definitions.

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
        // mu_ff_a / mu_ff_e from bits 8 / 9 of W_MU_PACKED.
        let ff_a_inner = down_w_a_sh4.clone()
            - w_a
            - pa_a
            + &mu_ff_a_contrib;
        b.assert_in_ideal(ff_a_inner + pa_c_ff_a, &ideal_rot_x2);

        let ff_e_inner = down_w_e_sh4.clone()
            - w_e
            - pa_e
            + &mu_ff_e_contrib;
        b.assert_in_ideal(ff_e_inner + pa_c_ff_e, &ideal_rot_x2);

        // C16: message init (Table 9 row 77). Pin w_W to public message
        // words pa_m at the 16 message-block-seed rows of every
        // compression. Mirrors the standalone SHA UAIR.
        b.assert_zero(sha_s_msg_init.clone() * &(w_big_w.clone() - pa_m));

        // Compensator-zero pinning (formerly C17–C21) is now enforced
        // by direct verifier inspection of public_trace via
        // `Uair::verify_public_structure`, not as in-circuit
        // constraints. The pa_c_* binders are retained because they
        // appear inside C7/C8/C9 (W) and C12/C13 (FF) above.
        let _ = (pa_c_c7, pa_c_c8, pa_c_c9, pa_c_ff_a, pa_c_ff_e);

        // C17 (renumbered from C22): high-bits-zero pin on W_MU_PACKED. Forces positions
        // 10..31 of w_mu_packed to be 0 at every row. Combined with
        // booleanity, confines mu_X to declared bit widths. See
        // sha256.rs cols doc for the soundness argument.
        b.assert_zero(down_w_mu_packed_shr10.clone());

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

    /// Verify the SHA-side public-column structural properties
    /// (compensator-zero on active rows, tail-corrector-zero on
    /// inner rows). Mirrors `Sha256CompressionSliceUair::verify_public_structure`
    /// with merged-layout column indices. The ECDSA half has no
    /// compensator/corrector pattern that needs verifier-side
    /// inspection.
    fn verify_public_structure<RT, IntT, const D: usize>(
        public_trace: &UairTrace<'_, RT, IntT, D>,
        num_vars: usize,
    ) -> Result<(), PublicStructureError>
    where
        RT: Clone,
        IntT: Clone + num_traits::Zero,
    {
        let n = 1usize << num_vars;
        debug_assert_eq!(public_trace.int.len(), cols::NUM_INT_PUB);
        debug_assert!(public_trace.binary_poly.len() >= cols::NUM_BIN_PUB);

        let pa_c_c7 = &public_trace.int[cols::SHA_PA_C_C7].evaluations;
        let pa_c_c8 = &public_trace.int[cols::SHA_PA_C_C8].evaluations;
        let pa_c_c9 = &public_trace.int[cols::SHA_PA_C_C9].evaluations;
        let pa_c_ff_a = &public_trace.int[cols::SHA_PA_C_FF_A].evaluations;
        let pa_c_ff_e = &public_trace.int[cols::SHA_PA_C_FF_E].evaluations;
        let pa_r_ch2_corr = &public_trace.binary_poly[cols::PA_R_CH2_CORR].evaluations;
        let pa_r_maj_corr = &public_trace.binary_poly[cols::PA_R_MAJ_CORR].evaluations;

        for i in 0..sha256::cols::NUM_COMPRESSIONS {
            let start = i * sha256::cols::ROWS_PER_COMP;

            let sched_end = start + (sha256::cols::ROUNDS_PER_COMP - 16);
            for k in start..sched_end.min(n) {
                if !pa_c_c7[k].is_zero() {
                    return Err(PublicStructureError::NonZeroOnRequiredZeroRow {
                        column: "PA_C_C7",
                        row: k,
                    });
                }
            }

            let upd_end = start + sha256::cols::ROUNDS_PER_COMP;
            for k in start..upd_end.min(n) {
                if !pa_c_c8[k].is_zero() {
                    return Err(PublicStructureError::NonZeroOnRequiredZeroRow {
                        column: "PA_C_C8",
                        row: k,
                    });
                }
                if !pa_c_c9[k].is_zero() {
                    return Err(PublicStructureError::NonZeroOnRequiredZeroRow {
                        column: "PA_C_C9",
                        row: k,
                    });
                }
            }

            let junc_start = start + sha256::cols::ROUNDS_PER_COMP;
            let junc_end = start + sha256::cols::ROWS_PER_COMP;
            for k in junc_start.min(n)..junc_end.min(n) {
                if !pa_c_ff_a[k].is_zero() {
                    return Err(PublicStructureError::NonZeroOnRequiredZeroRow {
                        column: "PA_C_FF_A",
                        row: k,
                    });
                }
                if !pa_c_ff_e[k].is_zero() {
                    return Err(PublicStructureError::NonZeroOnRequiredZeroRow {
                        column: "PA_C_FF_E",
                        row: k,
                    });
                }
            }
        }

        let inner_end = n.saturating_sub(2);
        for k in 0..inner_end {
            if !pa_r_ch2_corr[k].iter().all(|c| !c.into_inner()) {
                return Err(PublicStructureError::NonZeroOnRequiredZeroRow {
                    column: "PA_R_CH2_CORR",
                    row: k,
                });
            }
            if !pa_r_maj_corr[k].iter().all(|c| !c.into_inner()) {
                return Err(PublicStructureError::NonZeroOnRequiredZeroRow {
                    column: "PA_R_MAJ_CORR",
                    row: k,
                });
            }
        }

        Ok(())
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

/// Compute `2^k` as an `R` value via repeated doubling. Mirrors the
/// helper in sha256.rs (private there).
fn pow_two<R: ConstSemiring>(k: u32) -> R {
    let mut result = R::ONE;
    for _ in 0..k {
        let copy = result.clone();
        result += &copy;
    }
    result
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
        // SHA standalone int layout (9 cols, all public — the 5 prior
        // witness int carry columns are gone, packed into W_MU_PACKED;
        // the 2 prior compensator-zero selector columns are gone,
        // replaced by direct verifier inspection of public_trace):
        //   0..9   pubs (S_INIT_PREFIX, S_FEEDFORWARD, S_MSG_INIT, PA_K,
        //                PA_C_C7/8/9, PA_C_FF_A/E)
        // ECDSA standalone int layout (17 cols):
        //   0..9   pubs
        //   9..17  witnesses (8 EC cols)
        let mut int: Vec<DenseMultilinearExtension<R>> = Vec::with_capacity(cols::NUM_INT);
        let sha_ints = sha_trace.int.into_owned();
        let ecdsa_ints = ecdsa_trace.int.into_owned();

        // [0..9] SHA pubs (sha[0..9])
        int.extend(sha_ints[0..9].iter().cloned());
        // [9..18] ECDSA pubs (ecdsa[0..9])
        int.extend(ecdsa_ints[0..9].iter().cloned());
        // [18..26] ECDSA witnesses (ecdsa[9..17], 8 cols)
        int.extend(ecdsa_ints[9..17].iter().cloned());

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

    /// Sanity: 13 SHA + 11 ECDSA = 24 constraints. The 5 SHA
    /// compensator-zero pins (formerly C17–C21) moved to verifier-side
    /// `verify_public_structure`, dropping the in-circuit count from
    /// 29 to 24. Max degree 6 from the ECDSA Y output-selection (and
    /// D4's `12·X³·Y²` term).
    #[test]
    fn sha_ecdsa_constraint_shape() {
        type U = ShaEcdsaUair<Int<EC_FP_INT_LIMBS>>;
        assert_eq!(count_constraints::<U>(), 24);
        assert_eq!(count_max_degree::<U>(), 6);
        let degrees = count_constraint_degrees::<U>();
        // Spot checks: at least one deg-6 (ECDSA Y output sel / D4),
        // some deg-2 (boundaries + chaining), some deg-1 (SHA C1, C2,
        // C4, C6 — including the new row-local σ_0/σ_1 equalities).
        assert!(degrees.iter().any(|&d| d == 6), "expected deg-6 from ECDSA");
        assert!(degrees.iter().filter(|&&d| d == 2).count() >= 3, "expected ≥3 deg-2");
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
